"""Open-loop held-out errors; no predicted-state teacher forcing."""
import torch
from supermimic.core.state import delta, constant_velocity
from supermimic.core.rotations import quat_to_mat
from supermimic.core.world_model import contact_targets

DYNAMICS_KEYS = ['state', 'next_state', 'action', 'context', 'labels/feet_in_contact', 'labels/self_collision_raw']


def state_category(state):
    upright = (state[:, 2] >= 0.5) & (quat_to_mat(state[:, 3:7])[:, 2, 2] >= 0.34202014)
    moving = state[:, 7:9].norm(dim=-1) >= 0.2
    return torch.where(~upright, 2, torch.where(moving, 1, 0))


def physical_errors(prediction, target):
    error = delta(target, prediction)
    joints = (error.shape[-1] - 12) // 2
    sections = error.split((3, 3, 3, 3, joints, joints), -1)
    names = ['root_position_rmse_m', 'root_orientation_rmse_rad', 'root_linear_velocity_rmse_m_s',
             'root_angular_velocity_rmse_rad_s', 'joint_position_rmse_rad', 'joint_velocity_rmse_rad_s']
    return {name: value.square().mean().sqrt().item() for name, value in zip(names, sections)}


@torch.no_grad()
def evaluate(model, replay, env_ids, dt=0.02, batch_size=128, seed=1234):
    generator = torch.Generator(device=replay.device).manual_seed(seed)
    was_training = model.training
    model.eval()
    results = {}
    try:
        for horizon in (1, 8, 32):
            ages, envs = replay.valid_starts(horizon, env_ids)
            starts = replay.storage['state'][replay.rows(ages), envs]
            categories = state_category(starts)
            groups = {}
            for name, category in [('all', None), ('standing', 0), ('moving', 1), ('fallen', 2)]:
                indices = torch.arange(len(ages), device=replay.device) if category is None else (categories == category).nonzero().flatten()
                if not len(indices):
                    groups[name] = {'available_windows': 0, 'status': 'no_data'}
                    continue
                # Fixed sample without replacement, so small categories aren't inflated.
                selected = indices[torch.randperm(len(indices), device=replay.device, generator=generator)[:batch_size]]
                w = replay.gather(ages[selected], envs[selected], horizon, DYNAMICS_KEYS)
                x = w['state'][:, 0]
                prediction, velocity = x, x
                for t in range(horizon):
                    prediction, logits = model(prediction, w['action'][:, t], w['context'][:, t])
                    velocity = constant_velocity(velocity, dt)
                target = w['next_state'][:, -1]
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError(f'Non-finite prediction at horizon {horizon}')
                groups[name] = {'available_windows': len(indices), 'sampled_windows': len(selected),
                                'world_model': physical_errors(prediction, target),
                                'zero_increment': physical_errors(x, target),
                                'constant_velocity': physical_errors(velocity, target),
                                'quaternion_norm_max_error': (prediction[:, 3:7].norm(dim=-1)-1).abs().max().item(),
                                'contact_bce': torch.nn.functional.binary_cross_entropy_with_logits(logits, contact_targets(w)[:, -1]).item(),
                                'contact_accuracy': ((logits > 0) == (contact_targets(w)[:, -1] > 0.5)).float().mean().item()}
            results[str(horizon)] = groups
    finally:
        model.train(was_training)
    return results


def main():
    import argparse
    import json
    from pathlib import Path
    from supermimic.core.replay import SequenceReplay
    from supermimic.core.world_model import WorldModel
    parser = argparse.ArgumentParser(description='Re-evaluate a P2 checkpoint on its held-out replay environments')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    replay = SequenceReplay.from_state_dict(torch.load(args.replay, map_location=args.device, weights_only=False), args.device)
    model = WorldModel(**saved['architecture']).to(args.device)
    model.load_state_dict(saved['model'])
    ids = saved['metadata']['validation_env_ids']
    if set(ids) & set(saved['metadata']['train_env_ids']):
        raise ValueError('Training and validation environment IDs overlap')
    if replay.num_envs != saved['metadata']['num_envs']:
        raise ValueError('Replay environment schema mismatch')
    result = evaluate(model, replay, ids, saved['metadata']['control_dt'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f'Wrote independent checkpoint evaluation: {args.output}')


if __name__ == '__main__':
    main()
