"""Local-increment differentiable dynamics with supervised contact logits.

Architecture and local state increment formulation follow SuperMimic and the MIT
supermimic_vibed reproduction; see ../licenses/supermimic_vibed.MIT.
This implementation has dynamic joint/context dimensions and explicit frozen stats.
"""
import torch
from torch import nn
from .state import features, delta, integrate


class WorldModel(nn.Module):
    def __init__(self, joints, context_dim, hidden=(2048, 2048, 2048), contact_dim=3):
        super().__init__()
        self.joints, self.context_dim, self.contact_dim = joints, context_dim, contact_dim
        self.hidden = tuple(hidden)
        self.state_dim = 13 + 2 * joints
        self.increment_dim = 12 + 2 * joints
        inp = 13 + 2 * joints + joints + context_dim
        layers = []
        width = inp
        for h in hidden:
            layers.extend((nn.Linear(width, h), nn.LayerNorm(h), nn.SiLU()))
            width = h
        self.trunk = nn.Sequential(*layers)
        self.dynamics = nn.Linear(width, self.increment_dim)
        self.contact = nn.Linear(width, contact_dim)
        nn.init.zeros_(self.dynamics.weight); nn.init.zeros_(self.dynamics.bias)
        self.register_buffer('input_mean', torch.zeros(inp))
        self.register_buffer('input_std', torch.ones(inp))
        self.register_buffer('delta_mean', torch.zeros(self.increment_dim))
        self.register_buffer('delta_std', torch.ones(self.increment_dim))
        self.register_buffer('stats_fitted', torch.tensor(False))

    @staticmethod
    def inputs(state, action, context):
        return torch.cat((features(state), action, context), -1)

    @torch.no_grad()
    def fit_stats(self, state, next_state, action, context):
        x = self.inputs(state, action, context).reshape(-1, self.input_mean.numel())
        y = delta(state, next_state).reshape(-1, self.increment_dim)
        self.input_mean.copy_(x.mean(0))
        self.input_std.copy_(x.std(0, unbiased=False).clamp_min(0.01))
        floor = y.new_tensor([0.003]*3 + [0.005]*3 + [0.02]*3 + [0.05]*3 +
                             [0.003]*self.joints + [0.05]*self.joints)
        self.delta_mean.copy_(y.mean(0))
        self.delta_std.copy_(torch.maximum(y.std(0, unbiased=False), floor))
        self.stats_fitted.fill_(True)

    def forward(self, state, action, context):
        x = (self.inputs(state, action, context) - self.input_mean) / self.input_std
        dtype = getattr(self, 'mlp_dtype', None)
        if dtype is None:
            hidden = self.trunk(x)
            increments, contacts = self.dynamics(hidden), self.contact(hidden)
        else:
            with torch.autocast('cuda', dtype=dtype):
                hidden = self.trunk(x)
                increments, contacts = self.dynamics(hidden), self.contact(hidden)
        return integrate(state, increments.to(state.dtype) * self.delta_std + self.delta_mean), contacts.to(state.dtype)

    def architecture(self):
        return dict(joints=self.joints, context_dim=self.context_dim, hidden=self.hidden, contact_dim=self.contact_dim)


def contact_targets(window):
    feet = window['labels/feet_in_contact'].float()
    collision = (window['labels/self_collision_raw'] > 0).float().unsqueeze(-1)
    return torch.cat((feet, collision), -1)


def rollout_loss(model, window, contact_weight=0.1):
    state = window['state'][:, 0]
    targets = contact_targets(window)
    state_losses, contact_losses = [], []
    for t in range(window['action'].shape[1]):
        state, logits = model(state, window['action'][:, t], window['context'][:, t])
        error = delta(window['next_state'][:, t], state) / model.delta_std
        state_losses.append(error.square().mean())
        contact_losses.append(nn.functional.binary_cross_entropy_with_logits(logits, targets[:, t]))
    physics = torch.stack(state_losses).mean()
    contacts = torch.stack(contact_losses).mean()
    return physics + contact_weight * contacts, physics.detach(), contacts.detach()
