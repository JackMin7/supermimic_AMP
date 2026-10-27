"""Standalone G1 MuJoCo sim2sim: NumPy + MuJoCo + ONNX Runtime only."""
import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from types import SimpleNamespace



class Simulator:
    def __init__(self, policy, command):
        policy = Path(policy).resolve()
        metadata = json.loads(policy.with_suffix('.json').read_text())
        if metadata['schema_version'] != 1:
            raise ValueError('Unsupported deployment schema')
        with (policy.parent / metadata['scene']).open('rb') as scene_file:
            spec = mujoco.MjSpec.from_zip(scene_file)
        self.model = spec.compile()
        for name, value in metadata['physics_options'].items():
            setattr(self.model.opt, name, value)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.decimation = metadata['decimation']
        self.dt = self.model.opt.timestep * self.decimation
        self.names = metadata['joint_names']
        self.joints = np.array([self.model.joint(name).id for name in self.names])
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        self.default_q = np.array(metadata['default_q'])
        self.scale = np.array(metadata['action_scale'])
        self.actuator_ids = np.array([next(i for i in range(self.model.nu)
                                          if self.model.actuator_trnid[i, 0] == j) for j in self.joints])
        self.motor_cfg = [SimpleNamespace(**a) for a in metadata['motors']]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(policy), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        if self.session.get_inputs()[0].shape != [1, 384] or self.session.get_outputs()[0].shape != [1, 29]:
            raise ValueError('Expected ONNX observation [1,384] -> action [1,29]')
        self.command = np.asarray(command, dtype=np.float32)
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.history = None
        self.pelvis = self.model.body('pelvis').id
        self.torso = self.model.body('torso_link').id
        mujoco.mj_forward(self.model, self.data)

    def observation(self):
        rotation = self.data.xmat[self.pelvis].reshape(3, 3)
        frame = np.concatenate((self.data.sensor('imu_ang_vel').data,
                                rotation.T @ np.array([0., 0., -1.]), self.command,
                                self.data.qpos[self.qadr] - self.default_q,
                                self.data.qvel[self.vadr], self.previous_action)).astype(np.float32)
        if self.history is None:
            self.history = np.repeat(frame[None], 4, axis=0)
        else:
            self.history = np.concatenate((self.history[1:], frame[None]), axis=0)
        return self.history.reshape(1, 384)

    def control(self, action):
        """Match UnitreeActuator.compute, including implicit position actuation."""
        q, dq = self.data.qpos[self.qadr], self.data.qvel[self.vadr]
        target = self.default_q + self.scale * action
        for k, a in enumerate(self.motor_cfg):
            effort = a.stiffness * (target[k] - q[k]) - a.damping * dq[k]
            limit = a.Y1 if dq[k] * effort > 0 else (a.Y1 if a.Y2 is None else a.Y2)
            if a.effort_limit is not None:
                limit = min(limit, a.effort_limit)
            if abs(dq[k]) >= a.X1:
                limit *= max(0., 1 - (abs(dq[k]) - a.X1) / max(a.X2 - a.X1, 1e-6))
            effort = np.clip(effort, -limit, limit)
            effort -= a.Fs * np.tanh(dq[k] / a.Va) + a.Fd * dq[k]
            self.data.ctrl[self.actuator_ids[k]] = q[k] + (effort + a.damping * dq[k]) / max(a.stiffness, 1e-6)

    def step(self):
        obs = self.observation()
        action = self.session.run(['action'], {'observation': obs})[0][0]
        if not np.isfinite(action).all():
            raise FloatingPointError('Non-finite policy action')
        for _ in range(self.decimation):
            self.control(action)
            mujoco.mj_step(self.model, self.data)
        # Refresh derived positions and sensors at the integrated state.
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise FloatingPointError('Non-finite simulation state')
        self.previous_action = action.copy()
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['Unitree-G1-AMP-Flat'])
    parser.add_argument('--policy', type=Path, required=True, help='ONNX policy with adjacent JSON and scene ZIP')
    parser.add_argument('--viewer', choices=['native', 'none'], default='native')
    parser.add_argument('--duration', type=float, default=60.)
    parser.add_argument('--command', type=float, nargs=3, default=[0.5, 0., 0.], metavar=('VX', 'VY', 'WZ'))
    parser.add_argument('--output', type=Path, help='Write measured rollout summary JSON')
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('--duration must be positive')
    sim = Simulator(args.policy, args.command)
    viewer = None
    if args.viewer == 'native':
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(sim.model, sim.data)
        viewer.cam.distance = 3.
        viewer.cam.elevation = -15
    start = time.perf_counter()
    heights, errors, peaks = [], [], []
    try:
        for _ in range(round(args.duration / sim.dt)):
            if viewer is not None and not viewer.is_running():
                break
            tick = time.perf_counter()
            action = sim.step()
            heights.append(float(sim.data.xpos[sim.pelvis, 2]))
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(sim.model, sim.data, mujoco.mjtObj.mjOBJ_BODY, sim.torso, velocity, 0)
            rotation = sim.data.xmat[sim.torso].reshape(3, 3)
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            c, s = np.cos(yaw), np.sin(yaw)
            local_xy = np.array([[c, s], [-s, c]]) @ velocity[3:5]
            errors.append(float(np.sum((local_xy - sim.command[:2]) ** 2)))
            peaks.append(float(np.max(np.abs(action))))
            if viewer is not None:
                viewer.cam.lookat[:] = sim.data.xpos[sim.pelvis]
                viewer.sync()
                time.sleep(max(0., sim.dt - (time.perf_counter() - tick)))
    finally:
        if viewer is not None:
            viewer.close()
    result = dict(policy=str(args.policy.resolve()), command=args.command,
                  simulated_seconds=len(heights) * sim.dt, wall_seconds=time.perf_counter() - start,
                  min_pelvis_height=min(heights, default=None), final_pelvis_height=heights[-1] if heights else None,
                  below_05m_fraction=float(np.mean(np.array(heights) < .5)) if heights else None,
                  velocity_xy_rmse=float(np.sqrt(np.mean(errors))) if errors else None,
                  action_abs_max=max(peaks, default=None), physics='native_mujoco_cpu',
                  initialization='nominal_pose', randomization=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
