"""Compare the P3 four-term Torch reward to native managers at a matched instant."""
import argparse
import json
from pathlib import Path
import torch
from supermimic.adapters.mjlab import CaptureEnv, read_physics
from supermimic.core.state import pack
from supermimic.mdp.velocity import VelocityHeads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
    cfg = g1_amp_flat_env_cfg()
    cfg.scene.num_envs = 4; cfg.events.pop('push_robot', None)
    cfg.observations['actor'].enable_corruption = False
    env = CaptureEnv(cfg=cfg, device='cuda:0')
    try:
        env.reset(seed=2027)
        heads = VelocityHeads(env)
        # Disable recovery masking for this upright-task term parity probe.
        env.termination_manager._delay_env_mask.zero_()
        max_error = 0.
        for t in range(5):
            if t:
                env.step(torch.randn(4, 29, device=env.device)*0.05)
            # Compare the SAME post-forward state, not stale reward-time body transforms.
            env.sim.forward(); env.sim.sense()
            state = pack(read_physics(env))
            actual = heads.reward(state, env.action_manager.action, env.action_manager.prev_action,
                                  env.command_manager.get_command('twist'))
            expected = torch.zeros(4, device=env.device)
            for name in ('track_anchor_linear_velocity', 'track_anchor_angular_velocity', 'action_rate_l2', 'joint_pos_limits'):
                term = env.reward_manager.get_term_cfg(name)
                expected += env.step_dt * term.weight * term.func(env, **term.params)
            max_error = max(max_error, (actual-expected).abs().max().item())
        assert max_error < 1e-5, max_error
        result = {'status':'passed','four_term_reward_max_abs_error':max_error,
                  'comparison_time':'matched post-forward; not stale native reward buffer', 'states':20}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2))
        print(result)
    finally: env.close()


if __name__=='__main__': main()
