"""CPU invariants for sequence boundaries, SO(3), gradient flow and serialization."""
import io
import unittest
import torch
from supermimic.core.replay import SequenceReplay
from supermimic.core.state import delta, integrate, features
from supermimic.core.world_model import WorldModel


def state(batch, joints=2):
    s = torch.randn(batch, 13+2*joints) * 0.1
    s[:, 3:7] = torch.tensor([1., 0., 0., 0.])
    return s


class ReplayTests(unittest.TestCase):
    def test_wrap_reset_invalid_and_holdout(self):
        replay = SequenceReplay(5, 2)
        for t in range(8):
            x = torch.full((2, 17), float(t))
            replay.add_values(dict(state=x, next_state=x+1, action=torch.zeros(2, 2), context=torch.zeros(2, 1),
                                   terminated=torch.tensor([t == 4, False]), truncated=torch.tensor([False, t == 6]),
                                   episode_id=torch.tensor([int(t > 4), int(t > 6)]),
                                   dynamics_valid=torch.tensor([t != 6, True])))
            x.fill_(-100)  # The buffer must own snapshots.
        ages, envs = replay.valid_starts(2, [0, 1])
        windows = replay.gather(ages, envs, 2)
        pairs = {(int(w[0, 0]), int(e)) for w, e in zip(windows['state'], envs)}
        self.assertEqual(pairs, {(3, 0), (3, 1), (4, 1), (5, 1)})
        heldout = replay.sample(50, 2, [1])
        self.assertTrue((heldout['_env_ids'] == 1).all())
        self.assertTrue((heldout['state'][:, 1, 0] - heldout['state'][:, 0, 0] == 1).all())
        with self.assertRaises(ValueError):
            replay.sample(1, 5, [0])
        restored = SequenceReplay.from_state_dict(replay.state_dict())
        a, e = restored.valid_starts(2, [0, 1])
        self.assertTrue(torch.equal(ages, a) and torch.equal(envs, e))

    def test_manual_reset_without_done_still_breaks_windows(self):
        replay = SequenceReplay(4, 1)
        for t in range(4):
            replay.add_values(dict(state=torch.full((1, 17), float(t)), next_state=torch.zeros(1, 17),
                                   action=torch.zeros(1, 2), context=torch.zeros(1, 1),
                                   terminated=torch.zeros(1, dtype=torch.bool), truncated=torch.zeros(1, dtype=torch.bool),
                                   episode_id=torch.tensor([int(t >= 2)]), dynamics_valid=torch.ones(1, dtype=torch.bool)))
        ages, _ = replay.valid_starts(2, [0])
        self.assertEqual(ages.tolist(), [0, 2])

    def test_shape_change_does_not_advance_ring(self):
        replay = SequenceReplay(3, 2)
        values = dict(state=torch.zeros(2, 17), next_state=torch.zeros(2, 17), action=torch.zeros(2, 2),
                      context=torch.zeros(2, 1), terminated=torch.zeros(2, dtype=torch.bool),
                      truncated=torch.zeros(2, dtype=torch.bool), episode_id=torch.zeros(2, dtype=torch.long),
                      dynamics_valid=torch.ones(2, dtype=torch.bool))
        replay.add_values(values)
        values['action'] = torch.zeros(2, 3)
        with self.assertRaises(ValueError): replay.add_values(values)
        self.assertEqual(replay.total_steps, 1)


class DynamicsTests(unittest.TestCase):
    def test_increment_roundtrip_and_quaternion_sign(self):
        torch.manual_seed(4)
        x = state(16)
        d = torch.randn(16, 16) * 0.03
        y = integrate(x, d)
        self.assertTrue(torch.allclose(delta(x, y), d, atol=1e-5))
        flipped = y.clone(); flipped[:, 3:7] *= -1
        self.assertTrue(torch.allclose(features(y), features(flipped), atol=1e-6))
        self.assertTrue(torch.allclose(delta(y, flipped), torch.zeros_like(d), atol=1e-5))

    def test_backward_through_frozen_model_and_reload(self):
        torch.manual_seed(9)
        model = WorldModel(2, 3, (32, 32))
        x = state(8); a = torch.randn(8, 2); c = torch.randn(8, 3)
        model.fit_stats(x, integrate(x, torch.randn(8, 16)*0.01), a, c)
        # Start with nonzero dynamics weights to test actual action-to-state gradients.
        torch.nn.init.normal_(model.dynamics.weight, std=0.03)
        for p in model.parameters(): p.requires_grad_(False)
        a.requires_grad_(True)
        y, _ = model(x, a, c)
        y, _ = model(y, a, c)
        loss = y.square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        stream = io.BytesIO(); torch.save(model.state_dict(), stream); stream.seek(0)
        restored = WorldModel(**model.architecture())
        restored.load_state_dict(torch.load(stream, weights_only=True))
        self.assertTrue(torch.equal(model(x, a, c)[0], restored(x, a, c)[0]))
        self.assertTrue(torch.allclose(y[:, 3:7].norm(dim=-1), torch.ones(8), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
