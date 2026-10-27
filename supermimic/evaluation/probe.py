"""Small GPU integration probe for P1, not a policy performance evaluation."""
import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from supermimic.adapters.mjlab import CaptureEnv, PhysicsCollector, read_physics
from supermimic.contracts import PhysicsState
from supermimic.mdp.kinematics import RobotKinematics
from supermimic.mdp.observations import actor_frame, terminal_actor_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
    cfg = g1_amp_flat_env_cfg()
    cfg.scene.num_envs = 4
    cfg.seed = 42
    cfg.events.pop('push_robot', None)
    cfg.episode_length_s = 0.06  # Exercise real timeout/reset boundaries quickly.
    env = CaptureEnv(cfg=cfg, device=args.device)
    try:
        env.reset(seed=42)
        collector = PhysicsCollector(env)
        robot = env.scene['robot']
        state = read_physics(env)
        root_error = (state.root_pos + env.scene.env_origins - robot.data.root_link_pos_w).abs().max().item()
        velocity_error = (state.root_ang_vel_w - robot.data.root_link_ang_vel_w).abs().max().item()
        assert root_error < 1e-5, root_error
        assert velocity_error < 1e-4, velocity_error
        kin = RobotKinematics(env)
        fk_errors = dict(position=0.0, rotation=0.0, linear_velocity=0.0, angular_velocity=0.0)
        def check_fk():
            current = read_physics(env)
            p, q, v, w = kin(current)
            targets = (robot.data.body_link_pos_w - env.scene.env_origins[:, None],
                       robot.data.body_link_quat_w, robot.data.body_link_lin_vel_w,
                       robot.data.body_link_ang_vel_w)
            for key, actual, target in zip(fk_errors, (p, q, v, w), targets):
                error = (actual - target).abs()
                if key == 'rotation':
                    error = torch.minimum(error.amax(-1), (actual + target).abs().amax(-1))
                fk_errors[key] = max(fk_errors[key], error.max().item())
                assert fk_errors[key] < 2e-4, (key, fk_errors[key])
        check_fk()
        # Compare clean analytic frame with actual configured term functions.
        frame = actor_frame(env, state, noise=False)
        terms = env.observation_manager._group_obs_term_cfgs['actor']
        reference = torch.cat([t.func(env, **t.params) for t in terms], -1)
        assert torch.allclose(frame, reference, atol=1e-5), (frame-reference).abs().max()
        rng_before = torch.cuda.get_rng_state() if str(env.device).startswith('cuda') else torch.get_rng_state()
        obs_before = env.obs_buf['actor'].clone()
        terminal_actor_history(env, state)
        rng_after = torch.cuda.get_rng_state() if str(env.device).startswith('cuda') else torch.get_rng_state()
        assert torch.equal(rng_before, rng_after), 'Terminal instrumentation changed RNG'
        assert torch.equal(obs_before, env.obs_buf['actor']), 'Terminal instrumentation changed live history'
        differentiable = read_physics(env)
        differentiable.joint_pos.requires_grad_(True)
        fk_pos = kin(differentiable)[0]
        gradient = torch.autograd.grad(fk_pos.square().sum(), differentiable.joint_pos)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        direction = torch.randn_like(gradient)
        epsilon = 1e-3
        center = differentiable.joint_pos.detach().clone()
        differentiable.joint_pos = center + epsilon * direction
        plus = kin(differentiable)[0].square().sum()
        differentiable.joint_pos = center - epsilon * direction
        minus = kin(differentiable)[0].square().sum()
        finite_difference = (plus - minus) / (2 * epsilon)
        analytic = (gradient * direction).sum()
        assert torch.allclose(finite_difference, analytic, atol=0.03, rtol=0.01), (finite_difference, analytic)
        initial = state.joint_pos.clone()
        transitions = []
        for step in range(8):
            tr = collector.step(torch.full((env.num_envs, env.action_manager.total_action_dim), 0.01 * step, device=env.device))
            for f in fields(PhysicsState):
                assert torch.isfinite(getattr(tr.next_state, f.name)).all(), f.name
            assert torch.isfinite(tr.context).all()
            if env.terminal_snapshot is not None:
                ids, captured, _ = env.terminal_snapshot
                for f in fields(PhysicsState):
                    assert torch.equal(getattr(tr.next_state, f.name)[ids], getattr(captured, f.name)[ids])
                assert torch.all(tr.next_task['episode_step'][ids] > 0)
                assert torch.all(env.episode_length_buf[ids] == 0)
                assert torch.equal(tr.next_task['observation_history'][ids, :-1],
                                   tr.task['observation_history'][ids, 1:])
                assert torch.equal(tr.next_task['observation_history'][ids, -1, 67:], tr.action[ids])
                # Contact labels must match the exact target reward formula.
            labels = tr.labels
            collision_count = (labels['self_collision/force_history'].norm(dim=-1) > 10).any(dim=1).sum(-1)
            assert torch.equal(collision_count.float(), labels['self_collision_raw'])
            active = (tr.task['command'][:, :2].norm(dim=-1) + tr.task['command'][:, 2].abs()) > 0.1
            slip = (labels['feet_velocity_xy'].square().sum(-1) * labels['feet_in_contact']).sum(-1) * active
            assert torch.allclose(slip, labels['foot_slip_raw'])
            check_fk()
            if transitions:
                ongoing = ~(transitions[-1].terminated | transitions[-1].truncated)
                for f in fields(PhysicsState):
                    assert torch.equal(getattr(tr.state, f.name)[ongoing],
                                       getattr(transitions[-1].next_state, f.name)[ongoing])
            transitions.append(tr)
        assert torch.equal(state.joint_pos, initial), 'State aliases live simulation'
        timeouts = sum(int(t.truncated.sum()) for t in transitions)
        assert timeouts > 0, 'Probe did not exercise truncation'
        assert collector.episode_ids.sum() > 0
        # Force an actual terminal (not a timeout), without changing the source task.
        env.termination_manager._delay_env_mask[0] = False
        env.episode_length_buf[0] = 0
        free_q = robot.indexing.free_joint_q_adr.long()
        env.sim.data.qpos[0, free_q[2]] = env.scene.env_origins[0, 2] + 0.1
        env.sim.forward()
        terminal = collector.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
        assert terminal.terminated[0] and not terminal.truncated[0]
        assert terminal.next_state.root_pos[0, 2] < 0.5
        assert not torch.equal(terminal.next_state.joint_pos[0], read_physics(env).joint_pos[0])
        check_fk()
        result = {'status': 'passed', 'root_position_error': root_error,
                  'root_angular_velocity_error': velocity_error, 'timeouts': timeouts,
                  'fk_max_errors': fk_errors, 'fk_gradient_finite_nonzero': True, 'fk_finite_difference_passed': True,
                  'forced_terminal_capture_passed': True,
                  'num_envs': env.num_envs, 'control_dt': env.step_dt,
                  'joint_names': list(robot.joint_names), 'body_names': list(robot.body_names),
                  'context_schema': collector.context_schema,
                  'context_dim': transitions[0].context.shape[-1],
                  'actor_observation_shape': list(env.obs_buf['actor'].shape),
                  'terminal_history_and_contact_labels_passed': True,
                  'terminal_capture_preserves_rng_and_live_history': True,
                  'contact_label_shapes': {k: list(v.shape) for k, v in terminal.labels.items()},
                  'limitations': ['Terminal frame uses post-integration physics and ending command; contact labels use reward-time sensor data',
                                  'Push events disabled; no policy evaluation performed']}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
