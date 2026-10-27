# PPO 固定场景基线

Checkpoint：`logs/rsl_rl/g1_amp_locomotion/2026-10-24_13-24-33/model_66600.pt`。种子：[42, 43, 44]。

速度场景：15 次，20 秒存活率 100%。
恢复场景：24 次，成功率 100%；成功判定平均耗时 1.647 秒。

恢复判定：5 秒内，连续 1 秒高度 ≥ 0.5 m、倾角 ≤ 70°；耗时包含稳定判定的 1 秒。

| 指令 vx / vy / wz | XY 速度 RMSE（m/s） | yaw rate RMSE（rad/s） | 存活 |
|---|---:|---:|---:|
| [0.0, 0.0, 0.0] | 0.011 | 0.033 | 3/3 |
| [0.5, 0.0, 0.0] | 0.262 | 0.167 | 3/3 |
| [1.5, 0.0, 0.0] | 0.315 | 0.362 | 3/3 |
| [0.0, 0.5, 0.0] | 0.491 | 0.054 | 3/3 |
| [0.0, 0.0, 0.75] | 0.018 | 0.712 | 3/3 |

表中误差先按 episode 求 RMSE，再对三个种子取均值。速度使用 torso heading 坐标，yaw rate 使用 torso 世界坐标 z 分量。

当前策略的侧向移动与原地转向跟踪较弱。存活率与恢复成功率不能替代指令跟踪质量。

恢复帧：`[38, 213, 574, 1003, 1426, 1941, 2302, 2478]`，来源：`src/assets/motions/g1/amp/Recovery/fallAndGetUp1_subject1.npz`。

实测 rollout 总耗时：95.25 秒，不含环境构建与模型加载；这是评估耗时，不是训练 GPU 时间。

范围：关闭观测噪声与推扰，保留启动物理随机化。固定 8 帧来自同一个恢复动作，不代表任意倒地恢复；没有地形、推扰或训练速度对比。

复现：

```bash
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.evaluation.ppo --output /tmp/ppo_fixed_scenarios.json
```

完整逐场景指标、动作变化、接触足滑指标见同名 JSON，协议在 `.protocol.json` 中，均保留 checkpoint 和数据哈希。
