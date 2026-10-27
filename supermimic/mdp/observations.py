"""Post-integration Flat actor frame, without mutating manager history/sensors.

Terminal commands remain those of the ending transition (before command resampling).
Noise is sampled with restored RNG state so instrumentation cannot affect live policy.
"""
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.noise.noise_cfg import NoiseCfg


def actor_frame(env, state, *, noise=True):
    manager = env.observation_manager
    names = manager._group_obs_term_names['actor']
    expected = ['base_ang_vel', 'projected_gravity', 'command', 'joint_pos', 'joint_vel', 'actions']
    if names != expected:
        raise ValueError(f'Only Flat actor schema supported, got {names}')
    robot = env.scene['robot']
    # The configured IMU is aligned with the pelvis in the current target asset.
    model = env.sim.mj_model
    import mujoco
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'robot/imu_in_pelvis')
    if site < 0 or not (model.site_quat[site] == [1, 0, 0, 0]).all():
        raise ValueError('Expected pelvis-aligned IMU site')
    gravity = torch.zeros_like(state.root_pos)
    gravity[:, 2] = -1
    cfgs = manager._group_obs_term_cfgs['actor']
    joint_q = state.joint_pos
    if cfgs[3].params.get('biased', False):
        joint_q = joint_q + robot.data.encoder_bias
    values = [quat_apply_inverse(state.root_quat_wxyz, state.root_ang_vel_w),
              quat_apply_inverse(state.root_quat_wxyz, gravity),
              env.command_manager.get_command('twist').clone(),
              joint_q - robot.data.default_joint_pos,
              state.joint_vel - robot.data.default_joint_vel,
              env.action_manager.action.clone()]
    result = []
    for value, cfg in zip(values, cfgs, strict=True):
        if cfg.delay_max_lag:
            raise ValueError('Observation delay requires an explicit state contract')
        if noise and cfg.noise is not None:
            if not isinstance(cfg.noise, NoiseCfg):
                raise ValueError('Stateful observation noise is not supported')
            value = cfg.noise.apply(value)
        if cfg.clip is not None:
            value = value.clamp(*cfg.clip)
        if cfg.scale is not None:
            value = value * cfg.scale
        result.append(value)
    return torch.cat(result, -1)


def terminal_actor_history(env, state):
    cfg = env.cfg.observations['actor']
    if cfg.history_ordering != 'time' or cfg.history_length != 4:
        raise ValueError('Expected 4-frame time-major actor history')
    previous = env.obs_buf['actor'].reshape(env.num_envs, cfg.history_length, -1)
    devices = [torch.device(env.device).index or 0] if str(env.device).startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        frame = actor_frame(env, state)
    return torch.cat((previous[:, 1:], frame[:, None]), dim=1).clone()
