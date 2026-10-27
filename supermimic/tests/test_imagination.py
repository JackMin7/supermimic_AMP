import unittest
import torch
from torch import nn
from supermimic.core.imagination import freeze_parameters, gae_targets, imagine, ValueModel, update_critic


class ToyDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.1))
        self.register_buffer('input_mean', torch.zeros(3))
        self.register_buffer('input_std', torch.ones(3))

    def inputs(self, state, action, context): return torch.cat((state, action, context), -1)
    def forward(self, state, action, context): return state + self.gain * action, torch.zeros_like(action)


class ToyHeads:
    dt = 0.02
    def reward(self, state, action, previous, command): return -self.dt * (state-command).square().sum(-1)
    def observation(self, state, action, command): return torch.cat((state, action, command), -1)
    def continuation(self, state): return torch.ones(state.shape[0])


class ImaginationTests(unittest.TestCase):
    def test_invalid_rollout_is_excluded_without_clipping_valid_targets(self):
        torch.manual_seed(1)
        value=ValueModel(1)
        rollout={'critic_inputs':torch.zeros(3,2,1),'rewards':torch.tensor([[2.,-10000.],[2.,-10000.]]),
                 'continuations':torch.ones(2,2),'valid_trajectories':torch.tensor([True,False])}
        reference=ValueModel(1);reference.load_state_dict(value.state_dict())
        clean={k:v[:,:1] for k,v in rollout.items() if k!='valid_trajectories'}
        opt=torch.optim.Adam(value.parameters());refopt=torch.optim.Adam(reference.parameters())
        loss,diagnostics=update_critic(value,opt,rollout,epochs=1,minibatches=1,return_metrics=True)
        expected=update_critic(reference,refopt,clean,epochs=1,minibatches=1)
        self.assertAlmostEqual(loss.item(),expected,places=6)
        self.assertGreater(diagnostics['critic_target_abs_max'].item(),2)
        for actual,wanted in zip(value.parameters(),reference.parameters()):torch.testing.assert_close(actual,wanted)
        rollout['valid_trajectories'].zero_()
        with self.assertRaisesRegex(FloatingPointError,'No valid'):
            update_critic(value,opt,rollout)

    def test_critic_diagnostics_expose_rare_reward_outlier(self):
        critic = nn.Linear(1, 1, bias=False)
        # Squeeze the scalar value dimension as ValueModel does.
        class ScalarValue(nn.Module):
            def __init__(self):
                super().__init__(); self.net = critic
            def forward(self, x): return self.net(x).squeeze(-1)
        value = ScalarValue()
        with torch.no_grad(): critic.weight.zero_()
        rollout = {'critic_inputs': torch.zeros(3, 4, 1),
                   'rewards': torch.zeros(2, 4), 'continuations': torch.ones(2, 4)}
        rollout['rewards'][1, 0] = -10000
        loss, metrics = update_critic(value, torch.optim.SGD(value.parameters(), lr=0),
                                      rollout, epochs=1, minibatches=1, return_metrics=True)
        self.assertEqual(metrics['imagined_reward_abs_max'].item(), 10000)
        self.assertEqual(metrics['critic_target_abs_max'].item(), 10000)
        self.assertEqual(metrics['critic_value_abs_max'].item(), 0)
        self.assertGreater(loss.item(), 1e6)
        self.assertFalse(rollout['critic_targets'].requires_grad)

    def test_gae_terminal_and_bootstrap(self):
        rewards = torch.tensor([[1.], [2.]])
        values = torch.tensor([[100.], [200.], [3.]])
        continuing = torch.ones(2, 1)
        targets = gae_targets(rewards, continuing, values, gamma=0.5, lam=1)
        self.assertTrue(torch.equal(targets, torch.tensor([[2.75], [3.5]])))
        continuing[0] = 0
        targets = gae_targets(rewards, continuing, values, gamma=0.5, lam=1)
        self.assertEqual(targets[0].item(), 1)
        self.assertFalse(targets.requires_grad)

    def test_bootstrap_ablation_changes_only_terminal_value(self):
        wm=ToyDynamics(); actor=nn.Linear(12,1); critic=ValueModel(4)
        starts={'state':torch.zeros(2,1,1),'context':torch.zeros(2,1,1),
                'task/previous_action':torch.zeros(2,1,1),'task/command':torch.ones(2,1,1),
                'task/observation_history':torch.zeros(2,1,4,3)}
        with torch.no_grad():
            critic.net[-1].bias.fill_(3)
            full=imagine(wm,actor,nn.Identity(),critic,ToyHeads(),starts,horizon=4)
            short=imagine(wm,actor,nn.Identity(),critic,ToyHeads(),starts,horizon=4,bootstrap_weight=0)
        self.assertTrue(torch.equal(full['task_return'],short['task_return']))
        self.assertAlmostEqual((full['objective']-short['objective']).item(),3*0.99**4,places=5)
        self.assertTrue(torch.equal(short['objective'],short['task_return']))

    def test_bptt_updates_actor_only_and_matches_finite_difference(self):
        torch.manual_seed(2)
        wm = ToyDynamics(); actor = nn.Linear(12, 1)
        critic = ValueModel(4)
        starts = {'state':torch.zeros(4, 1, 1), 'context':torch.zeros(4, 1, 1),
                  'task/previous_action':torch.zeros(4, 1, 1),
                  'task/command':torch.ones(4, 1, 1),
                  'task/observation_history':torch.zeros(4, 1, 4, 3)}
        optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
        original_wm = wm.gain.detach().clone()
        actor_before = actor.bias.detach().clone()
        with freeze_parameters(wm, critic):
            rollout = imagine(wm, actor, nn.Identity(), critic, ToyHeads(), starts, horizon=4)
            derivative = torch.autograd.grad(rollout['objective'], actor.bias, retain_graph=True)[0].item()
            with torch.no_grad():
                actor.bias.add_(0.001)
                plus = imagine(wm, actor, nn.Identity(), critic, ToyHeads(), starts, horizon=4)['objective'].item()
                actor.bias.sub_(0.002)
                minus = imagine(wm, actor, nn.Identity(), critic, ToyHeads(), starts, horizon=4)['objective'].item()
                actor.bias.copy_(actor_before)
            self.assertAlmostEqual(derivative, (plus-minus)/0.002, places=4)
            optimizer.zero_grad(); (-rollout['objective']).backward(); optimizer.step()
            self.assertIsNone(wm.gain.grad)
            self.assertTrue(all(p.grad is None for p in critic.parameters()))
        self.assertTrue(wm.gain.requires_grad)
        self.assertTrue(torch.equal(wm.gain, original_wm))
        self.assertFalse(torch.equal(actor.bias, actor_before))
        actor_after = actor.bias.detach().clone()
        loss = update_critic(critic, torch.optim.Adam(critic.parameters()), rollout, epochs=1)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(torch.equal(actor.bias, actor_after))


if __name__=='__main__': unittest.main()
