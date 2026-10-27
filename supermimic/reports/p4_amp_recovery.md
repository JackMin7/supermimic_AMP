# P4：可微 AMP 与恢复在线闭环

本轮完整工程可比验收：通过。仅代表从 PPO/P3 初始化的小规模诊断，不代表从零训练、统计等效或论文加速复现。

## 实现与验证

- AMP/任务奖励与原接口逐项数值对齐，输入及端到端 actor 梯度通过有限差分；见 p4_parity.json、p4_gradient.json。
- 保留原 PPO 路径，新增可微 AMP 特征与奖励；195 维特征、0.1 style 系数及 0.75 task 混合权重。
- 判别器仅使用真实策略/专家 transition，沿用原 PPO least-squares、原始专家梯度惩罚和 normalizer 更新语义；想象阶段冻结参数与统计。
- 40% 起点来自恢复环境；加入延迟计数、恢复清零、timeout、奖励模式与 heading。critic 显式接收任务状态。
- 控制步加速度、接触概率、自碰撞计数与软终止属于代理；真实评估采用原生任务奖励、真实接触和硬终止。

采集 16,896 条 transition，actor/critic 各 100 次交替更新，WM 200 次，判别器 100 次。训练耗时 251.0 秒；数值保护事件 0 次。

网络更新/统计冻结检查：`{"actor_changed": true, "wm_changed": true, "discriminator_changed": true, "amp_statistics_updated": true, "actor_normalizer_frozen": true, "wm_statistics_frozen": true, "recovery_starts_used": true, "recovery_reward_used": true}`。

## 真实仿真

两套策略用原 PPO 的同一冻结判别器与 AMP normalizer 评分，避免移动奖励导致虚假收益。最终 checkpoint 在测试前固定；初态和物理上下文 SHA256 配对。

验收：P3 四项回报下降 ≤2%，XY/yaw RMSE 各增加 ≤5%，存活和恢复成功率不下降；新增完整 AMP 混合回报在速度和恢复场景分别下降 ≤2%。

### fixed：通过

| 指标 | 原 PPO | P4 |
|---|---:|---:|
| mean_task_return | 38.231413 | 38.214392 |
| velocity_survival_rate | 1.000000 | 1.000000 |
| xy_rmse | 0.212326 | 0.216957 |
| yaw_rmse | 0.261129 | 0.245445 |
| recovery_success_rate | 0.958333 | 1.000000 |
| velocity_full_return | 56.594046 | 56.548429 |
| velocity_style_reward | 0.085747 | 0.085526 |
| recovery_full_return | 12.026504 | 12.380330 |
| recovery_style_reward | 0.074623 | 0.075399 |

检查：`{"finite_metrics": true, "return_within_margin": true, "xy_error_within_margin": true, "yaw_error_within_margin": true, "survival_preserved": true, "recovery_preserved": true, "full_velocity_return": true, "full_recovery_return": true}`。

### push：通过

| 指标 | 原 PPO | P4 |
|---|---:|---:|
| mean_task_return | 38.191795 | 38.179736 |
| velocity_survival_rate | 1.000000 | 1.000000 |
| xy_rmse | 0.230371 | 0.232068 |
| yaw_rmse | 0.283765 | 0.277813 |
| recovery_success_rate | 1.000000 | 1.000000 |
| velocity_full_return | 56.478475 | 56.400782 |
| velocity_style_reward | 0.085006 | 0.084960 |
| recovery_full_return | 12.228609 | 12.242171 |
| recovery_style_reward | 0.074670 | 0.075010 |

检查：`{"finite_metrics": true, "return_within_margin": true, "xy_error_within_margin": true, "yaw_error_within_margin": true, "survival_preserved": true, "recovery_preserved": true, "full_velocity_return": true, "full_recovery_return": true}`。

| 种子 | 推扰步 | 单种子检查 | 未通过项 |
|---:|---:|---|---|
| 82 | None | 通过 |  |
| 83 | None | 通过 |  |
| 84 | None | 通过 |  |
| 82 | 250 | 通过 |  |

推扰仅对速度场景施加：第 5 秒根速度增加 (0.6, 0.3) m/s。固定恢复场景观察前 5 秒，要求连续直立至少 1 秒。

## 限制与后续

训练保留 startup DR，关闭随机 interval 推扰；只有一个评估种子的确定性推扰对照。尚需完整随机推扰训练、多训练种子、接触/加速度代理消融与较长训练预算，才能判断完整任务稳健性和学习效率。

专家邻帧必须来自同一 clip 且采样间隔匹配 50 Hz；本实现排除原 loader 的末帧重复 pair。延迟计数严格匹配现有实现，包括 timeout 当步参与计数但不延迟 timeout。

保存 actor、两种 normalizer、WM、critic、判别器、全部优化器、专家游标、RNG 与 replay；尚无模拟器精确续训 CLI。

产物：`logs/supermimic/p4_amp_recovery_20261026/actor_final.pt`、`logs/supermimic/p4_amp_recovery_20261026/replay.pt`。

```bash
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.train_p4 --output logs/supermimic/p4_repeat --actor-lr 1e-07 --seed 2030 --test-seeds 82 83 84
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m unittest discover -s supermimic/tests -v
```

验证产物：[奖励与特征对齐](p4_parity.json)、[端到端梯度检查](p4_gradient.json)。14 项测试全部通过，覆盖恢复边界、专家 clip 边界、完整恢复回报验收和非有限指标。

恢复成功率的 23/24 → 24/24 是本组有限初态上的观察，不将其归因于某一新增损失；本轮没有进行“同一 P3 起点但禁用 AMP/恢复目标”的配对训练消融。
