"""Shared ONNX policy, observation history and operator sequencing; no DDS backend."""
from __future__ import annotations
import json
import struct
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Explicit SDK PR-mode indices, including unlocked waist and seven-DoF arms.
SDK_JOINTS = [f'{side}_{joint}_joint' for side in ('left', 'right')
              for joint in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
SDK_JOINTS += ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
SDK_JOINTS += [f'{side}_{joint}_joint' for side in ('left', 'right')
               for joint in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                             'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
START, SELECT, A, B = 2, 3, 8, 9
STATE_TIMEOUT = .1

class Stop(RuntimeError):
    pass

def gravity_from_quat(quat):
    quat = np.asarray(quat, dtype=np.float64)
    if not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-6:
        raise Stop('invalid IMU quaternion')
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array([-2 * (x*z-w*y), -2 * (y*z+w*x), -(1-2*(x*x+y*y))])

def joystick_command(remote, scale=(1., 1., 1.), deadzone=.01):
    """Official RL Lab axis directions, Unitree 40-byte wireless payload.

    Float offsets: lx=4, rx=8, ry=12, L2=16, ly=20 (little endian).
    Deadzone suppresses small readings; remaining values are not rescaled.
    """
    raw = bytes(remote)
    if len(raw) != 40:
        raise Stop('invalid wireless remote payload')
    lx = struct.unpack_from('<f', raw, 4)[0]
    rx = struct.unpack_from('<f', raw, 8)[0]
    ly = struct.unpack_from('<f', raw, 20)[0]
    axes = np.array([ly, -lx, -rx], dtype=np.float64)
    if not np.isfinite(axes).all():
        raise Stop('non-finite joystick axis')
    axes = np.clip(axes, -1., 1.)
    axes[np.abs(axes) < deadzone] = 0.
    # Current G1 AMP Flat command ranges; independent of the official policy's ranges.
    return np.clip(axes * scale, [-1.5, -1., -1.57], [3., 1., 1.57]).astype(np.float32)

class PolicyController:
    def __init__(self, policy, max_tilt_deg=None, velocity_scale=(1., 1., 1.), deadzone=.01):
        policy = Path(policy)
        meta = json.loads(policy.with_suffix('.json').read_text())
        if meta['schema_version'] != 1 or not meta['normalization_in_onnx']:
            raise ValueError('Unsupported policy metadata')
        if len(meta['joint_names']) != 29 or set(meta['joint_names']) != set(SDK_JOINTS):
            raise ValueError('Requires G1 29-DoF with unlocked waist')
        self.motor_ids = [SDK_JOINTS.index(n) for n in meta['joint_names']]
        self.default_q = np.array(meta['default_q'], dtype=np.float64)
        self.scale = np.array(meta['action_scale'], dtype=np.float64)
        self.kp = np.array([a['stiffness'] for a in meta['motors']])
        self.kd = np.array([a['damping'] for a in meta['motors']])
        self.dt = meta['physics_options']['timestep'] * meta['decimation']
        if abs(self.dt - .02) > 1e-9:
            raise ValueError('Expected 50 Hz policy')
        for value in (self.default_q, self.scale, self.kp, self.kd):
            if value.shape != (29,) or not np.isfinite(value).all():
                raise ValueError('Invalid deployment parameters')
        options = ort.SessionOptions(); options.intra_op_num_threads = 1
        self.policy = ort.InferenceSession(str(policy), sess_options=options, providers=['CPUExecutionProvider'])
        if self.policy.get_inputs()[0].shape != [1,384] or self.policy.get_outputs()[0].shape != [1,29]:
            raise ValueError('Expected observation [1,384] -> action [1,29]')
        self.velocity_scale = np.asarray(velocity_scale, dtype=np.float64)
        if (self.velocity_scale.shape != (3,) or not np.isfinite(self.velocity_scale).all()
                or np.any(self.velocity_scale < 0)):
            raise ValueError('Velocity scale must contain three finite nonnegative values')
        if not np.isfinite(deadzone) or not 0 <= deadzone < 1:
            raise ValueError('Deadzone must be in [0,1)')
        self.deadzone = deadzone
        self.command = np.zeros(3, dtype=np.float32)
        self.max_tilt_deg = max_tilt_deg
        self.low_state = None
        self.buttons = 0
        self._init_transport()
        self.reset()

    def motor_state(self, msg):
        q = np.array([msg.motor_state[i].q for i in self.motor_ids])
        dq = np.array([msg.motor_state[i].dq for i in self.motor_ids])
        return q, dq

    def reset(self):
        self.history = None
        self.a_prev = np.zeros(29, dtype=np.float32)
        self.last_observation = None

    def send(self, q, kp, kd):
        msg = self.snapshot()
        return self.write(q, kp, kd, msg.mode_machine)

    def step(self):
        msg = self.snapshot()
        self.command = joystick_command(msg.wireless_remote, self.velocity_scale, self.deadzone)
        q, dq = self.motor_state(msg)
        gravity = gravity_from_quat(msg.imu_state.quaternion)
        if self.max_tilt_deg is not None and -gravity[2] < np.cos(np.deg2rad(self.max_tilt_deg)):
            raise Stop('tilt limit')
        frame = np.concatenate((msg.imu_state.gyroscope, gravity, self.command,
                                q - self.default_q, dq, self.a_prev)).astype(np.float32)
        if not np.isfinite(frame).all():
            raise Stop('non-finite observation')
        self.history = (np.repeat(frame[None], 4, axis=0) if self.history is None
                        else np.concatenate((self.history[1:], frame[None])))
        self.last_observation = self.history.reshape(1,384)
        action = self.policy.run(['action'], {'observation': self.last_observation})[0][0]
        if not np.isfinite(action).all():
            raise Stop('non-finite policy action')
        # Hardware receives the desired joint position, not the simulator's
        # equivalent position signal used to emulate its torque-speed model.
        cmd = self.send(self.default_q + self.scale * action, self.kp, self.kd)
        self.a_prev = action.copy()
        return cmd

    def wait_button(self, button, q, kp, kd):
        while not self.pressed(button):
            self.send(q, kp, kd)
            time.sleep(self.dt)

    def move_to_start(self, seconds=2.):
        q0, _ = self.motor_state(self.snapshot())
        count = round(seconds / self.dt)
        for i in range(1, count+1):
            self.send(q0 + (self.default_q-q0)*i/count, self.kp, self.kd)
            time.sleep(self.dt)

    def run(self):
        self.reset()
        deadline = time.monotonic()
        while True:
            self.step()
            deadline += self.dt
            now = time.monotonic()
            if now - deadline > STATE_TIMEOUT:
                raise Stop('control loop overrun > 100 ms')
            time.sleep(max(0., deadline-now))

    def _init_transport(self):
        raise NotImplementedError

    def snapshot(self):
        raise NotImplementedError

    def pressed(self, button):
        raise NotImplementedError

    def write(self, q, kp, kd, mode_machine):
        raise NotImplementedError
