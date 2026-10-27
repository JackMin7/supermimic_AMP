"""Shared numerical action contract for real collection and learned rollouts."""
import torch

ACTION_ABS_LIMIT = 100.0


def valid_actions(action):
    return torch.isfinite(action).all(-1) & (action.abs().amax(-1) < ACTION_ABS_LIMIT)


def guarded_action(action):
    """Keep valid actions untouched; substitute only to evaluate a rejected rollout.

    Callers MUST exclude every trajectory with an invalid action from learning.
    The zero substitute is not a simulated action target or an action clamp.
    """
    valid = valid_actions(action)
    return torch.where(valid[..., None], action, torch.zeros_like(action)), valid


def valid_mean(values, valid):
    return torch.where(valid, values, torch.zeros_like(values)).sum() / valid.sum().clamp_min(1)
