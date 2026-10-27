"""Cross-check standalone contracts against training implementations."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from scripts.export_sim2sim import export
from rsl_rl.modules.normalizer import EmpiricalNormalization
import numpy as np
import torch
from mjlab.utils.buffers.circular_buffer import CircularBuffer
from mjlab.utils.lab_api.math import quat_apply_inverse
from src.assets.robots.unitree_g1.unitree_actuators import UnitreeActuator
from supermimic.deployment.mujoco_runner import Simulator


class Sim2SimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        actor = SimpleNamespace(actor=torch.nn.Sequential(torch.nn.Linear(384, 29)))
        normalizer = EmpiricalNormalization([384]).eval()
        with patch('scripts.export_sim2sim.load_policy', return_value=(actor, normalizer)):
            cls.policy_path = export(Path(cls.tmp.name) / 'fixture.pt', cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.sim = Simulator(self.policy_path, [.5, -.2, .3])

    def test_observation_matches_training_history(self):
        import mujoco
        s = self.sim
        history = CircularBuffer(4, 1, 'cpu')
        rng = np.random.default_rng(42)
        for _ in range(7):
            quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
            s.data.qpos[3:7] = quat
            s.data.qpos[s.qadr] += rng.normal(size=29) * .01
            s.data.qvel[:] = rng.normal(size=s.model.nv)
            s.previous_action = rng.normal(size=29).astype(np.float32)
            mujoco.mj_forward(s.model, s.data)
            gravity = quat_apply_inverse(torch.tensor(quat[None]), torch.tensor([[0., 0., -1.]], dtype=torch.float64))[0].numpy()
            frame = np.concatenate((s.data.qvel[3:6], gravity, s.command,
                                    s.data.qpos[s.qadr] - s.default_q,
                                    s.data.qvel[s.vadr], s.previous_action)).astype(np.float32)
            history.append(torch.tensor(frame[None]))
            np.testing.assert_allclose(s.observation(), history.buffer.numpy().reshape(1, -1), atol=1e-6)

    def test_motor_matches_training_at_high_speed(self):
        s = self.sim
        rng = np.random.default_rng(12)
        for _ in range(20):
            s.data.qvel[s.vadr] = rng.uniform(-50, 50, 29)
            action = rng.uniform(-30, 30, 29)
            s.control(action)
            for k, cfg in enumerate(s.motor_cfg):
                # Execute the actual training actuator implementation on the same inputs.
                a = object.__new__(UnitreeActuator)
                a.cfg = cfg
                tensor = lambda x: torch.tensor([x], dtype=torch.float64)
                a._joint_vel = tensor(0.)
                for attr, value in [('effort_y1', cfg.Y1), ('effort_y2', cfg.Y1 if cfg.Y2 is None else cfg.Y2),
                                    ('velocity_x1', cfg.X1), ('velocity_x2', cfg.X2),
                                    ('friction_static', cfg.Fs), ('friction_dynamic', cfg.Fd), ('activation_vel', cfg.Va)]:
                    setattr(a, '_' + attr, tensor(value))
                cmd = SimpleNamespace(pos=tensor(s.data.qpos[s.qadr[k]]), vel=tensor(s.data.qvel[s.vadr[k]]),
                                      position_target=tensor(s.default_q[k] + s.scale[k] * action[k]),
                                      velocity_target=tensor(0.), effort_target=tensor(0.))
                expected = a.compute(cmd).item()
                self.assertAlmostEqual(s.data.ctrl[s.actuator_ids[k]], expected, places=9)


if __name__ == '__main__':
    unittest.main()
