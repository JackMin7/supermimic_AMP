"""Collect from the CURRENT actor, recording numerical resets and policy version."""
import torch
from supermimic.adapters.mjlab import PhysicsCollector
from .state import pack
from .numerics import ACTION_ABS_LIMIT, valid_actions


class OnlineCollection:
    def __init__(self, env, policy, normalizer, replay, action_noise=0.1):
        self.env, self.policy, self.normalizer, self.replay = env, policy, normalizer, replay
        self.collector = PhysicsCollector(env)
        self.action_noise = action_noise
        self.guard_events = []
        self.total_steps = 0
        # A restarted simulator must NEVER join the last episode in saved replay.
        if replay.size:
            rows = replay.rows(torch.arange(replay.size, device=replay.device))
            self.collector.episode_ids.fill_(int(replay.storage['episode_id'][rows].max()) + 1)

    def reset_bad(self, ids, kind, version, details=None):
        self.guard_events.append({'collection_step': self.total_steps, 'policy_version': version,
                                  'kind': kind, 'env_ids': ids.tolist(), **(details or {})})
        self.env.reset(env_ids=ids)
        self.env.termination_manager._delay_counters[ids] = 0
        self.collector.episode_ids[ids] += 1

    @torch.no_grad()
    def collect(self, count, policy_version):
        self.policy.eval(); self.normalizer.eval()
        for _ in range(count):
            for attempt in range(4):
                action = self.policy.act_inference(self.normalizer(self.env.obs_buf['actor']))
                finite = torch.isfinite(action).all(-1)
                excessive = finite & (action.abs().amax(-1) >= ACTION_ABS_LIMIT)
                good = valid_actions(action)
                if good.all(): break
                self.reset_bad((~good).nonzero().flatten(), 'action', policy_version,
                               {'nonfinite_env_ids': (~finite).nonzero().flatten().tolist(),
                                'excessive_finite_env_ids': excessive.nonzero().flatten().tolist()})
            else:
                raise FloatingPointError(f'Current actor remained invalid after four resets: {self.guard_events[-4:]}')
            action = action + self.action_noise * torch.randn_like(action)
            transition = self.collector.step(action)
            valid = torch.isfinite(pack(transition.state)).all(-1) & torch.isfinite(pack(transition.next_state)).all(-1)
            for state in (transition.state, transition.next_state):
                for velocity in (state.root_lin_vel_w, state.root_ang_vel_w, state.joint_vel):
                    valid &= velocity.abs().amax(-1) < 1000
            for value in transition.labels.values():
                valid &= torch.isfinite(value).reshape(self.env.num_envs, -1).all(-1)
            version = torch.full_like(transition.episode_id, policy_version)
            transition.task['policy_version'] = version
            transition.next_task['policy_version'] = version.clone()
            self.replay.add(transition, dynamics_valid=valid)
            if not valid.all(): self.reset_bad((~valid).nonzero().flatten(), 'physics', policy_version)
            self.total_steps += 1
