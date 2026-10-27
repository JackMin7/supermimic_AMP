"""Short-horizon analytic policy gradients and detached GAE critic targets."""
from contextlib import contextmanager
import torch
from torch import nn


@contextmanager
def freeze_parameters(*modules):
    original = [(p, p.requires_grad) for m in modules for p in m.parameters()]
    try:
        for p, _ in original: p.requires_grad_(False)
        yield
    finally:
        for p, value in original: p.requires_grad_(value)


class ValueModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def critic_input(wm, state, previous, context, command):
    base = (wm.inputs(state, previous, context) - wm.input_mean) / wm.input_std
    return torch.cat((base, command), -1)


def imagine(wm, actor, normalizer, critic, heads, starts, horizon=8, gamma=0.99, teacher=None, anchor_weight=0.0, bootstrap_weight=1.0):
    state = starts['state'][:, 0]
    history = starts['task/observation_history'][:, 0]
    previous = starts['task/previous_action'][:, 0]
    context = starts['context'][:, 0]
    command = starts['task/command'][:, 0]
    alive = torch.ones(state.shape[0], device=state.device)
    objective = torch.zeros_like(alive)
    raw_return = torch.zeros_like(alive)
    xs, rewards, continuations, anchor_costs = [], [], [], []
    for t in range(horizon):
        xs.append(critic_input(wm, state, previous, context, command))
        obs = normalizer(history.flatten(1))
        action = actor(obs)
        # Detach only the regularization target; never detach imagined state or actor history.
        anchor = torch.zeros_like(alive)
        if teacher is not None and anchor_weight:
            with torch.no_grad(): reference = teacher(obs)
            anchor = (action - reference).square().mean(-1)
        next_state, _ = wm(state, action, context)
        reward = heads.reward(next_state, action, previous, command)
        continuation = heads.continuation(next_state)
        objective = objective + gamma**t * alive * (reward - heads.dt * anchor_weight * anchor)
        raw_return = raw_return + gamma**t * alive * reward
        alive = alive * continuation
        rewards.append(reward); continuations.append(continuation); anchor_costs.append(anchor)
        history = torch.cat((history[:, 1:], heads.observation(next_state, action, command)[:, None]), dim=1)
        state, previous = next_state, action
    xs.append(critic_input(wm, state, previous, context, command))
    bootstrap = gamma**horizon * alive * critic(xs[-1])
    return {'objective': (objective + bootstrap_weight * bootstrap).mean(), 'task_return': raw_return.mean(),
            'anchor_mse': torch.stack(anchor_costs).mean(),
            'critic_inputs': torch.stack(xs), 'rewards': torch.stack(rewards),
            'continuations': torch.stack(continuations), 'bootstrap': bootstrap.mean()}


@torch.no_grad()
def gae_targets(rewards, continuations, values, gamma=0.99, lam=0.95):
    if values.shape[0] != rewards.shape[0] + 1:
        raise ValueError('Need H+1 value predictions for H transitions')
    result = values[-1]
    targets = []
    for t in reversed(range(rewards.shape[0])):
        result = rewards[t] + gamma * continuations[t] * ((1-lam) * values[t+1] + lam * result)
        targets.append(result)
    return torch.stack(targets[::-1])


def update_critic(critic, optimizer, rollout, epochs=2, minibatches=4, return_tensor=False, gamma=0.99, lam=0.95, return_metrics=False):
    x = rollout['critic_inputs'].detach()
    valid = rollout.get('valid_trajectories', torch.ones(x.shape[1], dtype=torch.bool, device=x.device))
    if not valid.any():
        raise FloatingPointError('No valid imagined trajectories for critic update')
    with torch.no_grad():
        values = critic(x)
        targets = gae_targets(rollout['rewards'].detach(), rollout['continuations'].detach(), values, gamma=gamma, lam=lam)
        if return_metrics:
            rollout['critic_targets'] = targets
            rollout['critic_values_before'] = values
            residual = (values[:-1] - targets).square()
            alive = torch.cat((torch.ones_like(rollout['continuations'][:1]),
                               rollout['continuations'].detach().cumprod(0)[:-1]), 0)
            diagnostics = {
                'critic_target_abs_max': targets[:,valid].abs().amax(),
                'critic_value_abs_max': values[:,valid].abs().amax(),
                'critic_target_std': targets[:,valid].std(unbiased=False),
                'critic_input_abs_max': x[:,valid].abs().amax(),
                'imagined_reward_abs_max': rollout['rewards'].detach()[:,valid].abs().amax(),
                'critic_mse_before': residual[:,valid].mean(),
                'critic_alive_weighted_mse': (residual[:,valid] * alive[:,valid]).sum() / alive[:,valid].sum().clamp_min(1),
            }
    x, targets = x[:-1,valid].flatten(0, 1), targets[:,valid].flatten()
    losses = []
    for _ in range(epochs):
        for indices in torch.randperm(len(x), device=x.device).tensor_split(minibatches):
            if not len(indices): continue
            loss = (critic(x[indices]) - targets[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
    result = torch.stack(losses).mean()
    if return_metrics:
        return result, diagnostics
    return result if return_tensor else float(result)
