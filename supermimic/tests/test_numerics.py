import unittest
import torch
from supermimic.core.numerics import guarded_action,valid_actions,valid_mean


class NumericsTests(unittest.TestCase):
    def test_real_and_imagined_contract_and_gradients(self):
        action=torch.tensor([[99.,-99.],[100.,0.],[float('nan'),0.]],requires_grad=True)
        safe,valid=guarded_action(action)
        self.assertEqual(valid.tolist(),[True,False,False])
        self.assertTrue(torch.equal(valid,valid_actions(action)))
        torch.testing.assert_close(safe[0],action[0])
        self.assertTrue(torch.isfinite(safe).all())
        # Rejected trajectories contribute neither actor gradients nor fake zero targets.
        valid_mean(safe.square().sum(-1),valid).backward()
        torch.testing.assert_close(action.grad,torch.tensor([[198.,-198.],[0.,0.],[0.,0.]]))
