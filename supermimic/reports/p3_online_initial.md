# P3 在线交替更新验证

固定最终 checkpoint 的 3 种子配对评估：未通过 PPO 工程可比验收。平均任务回报变化 -0.3879%。

验收标准在运行前写入 protocol.json：回报下降不超过 2%；XY/yaw RMSE 各增加不超过 5%；存活率和固定恢复成功率不下降。无需超越 PPO。

| 指标 | 原 PPO | 在线 BPTT |
|---|---:|---:|
| mean_task_return | 38.180952 | 38.032854 |
| velocity_survival_rate | 1.000000 | 1.000000 |
| xy_rmse | 0.216888 | 0.230705 |
| yaw_rmse | 0.256686 | 0.259827 |
| recovery_success_rate | 1.000000 | 0.958333 |

| 测试种子 | 回报变化 | 单种子可比检查 |
|---:|---:|---|
| 62 | -0.3601% | 通过 |
| 63 | -0.2699% | 未通过 |
| 64 | -0.5336% | 未通过 |

非有限指标（JSON 中存为 null，验收失败）：`[{"path": "/tests/2/online/trial/episodes/8/p3_task_return", "value": "-inf"}, {"path": "/tests/2/online/trial/episodes/8/action_delta_squared_sum_mean", "value": "inf"}]`。


## 在线训练与验证

64 个环境，48/16 训练/验证划分。采集 16,896 条当前策略 transition；100 次 H=8 actor/critic 交替更新和 200 次 WM 更新。训练与 WM 验证耗时 74.6 秒，不含环境初始化和真实配对评估。

梯度隔离审计 11 次；actor 和 WM 确实更新，PPO normalizer 与 WM 归一化统计逐项保持不变。数值保护重置事件 0 次。

最终 checkpoint 在真实测试前固定，通过原生 PPO 加载接口重载并逐项核对参数。每组基线/候选的初态和物理上下文哈希完全一致。

| 固定 holdout 预测误差 | 刷新前 | 刷新后 |
|---|---:|---:|
| 1 步 root_position_rmse_m | 0.003878 | 0.003857 |
| 1 步 joint_position_rmse_rad | 0.017750 | 0.016917 |
| 8 步 root_position_rmse_m | 0.024450 | 0.024095 |
| 8 步 joint_position_rmse_rad | 0.077997 | 0.073515 |
| 32 步 root_position_rmse_m | 0.075542 | 0.082418 |
| 32 步 joint_position_rmse_rad | 0.095094 | 0.083301 |

长时域根位置误差没有同步改善；保持短时域训练，不能仅依据训练 loss 扩展 horizon。

## 范围与下一阶段

这是从已有 PPO/P3 策略微调的单训练种子速度任务消融。三组评估种子不是三次独立训练；工程容差通过不等于统计等效、完整 AMP 复现或从零学习能力。恢复仅作固定场景回归测试，本轮未训练 AMP/恢复目标，也未评估推扰。

下一阶段接入可微 AMP reward、真实 transition 判别器更新、恢复计数和奖励模式，再用完整任务评估。checkpoint 保存网络、优化器、RNG，replay 另存；当前无模拟器精确续训 CLI。

实验目录：`logs/supermimic/p3_online_20261026`；策略：`logs/supermimic/p3_online_20261026/actor_final.pt`。

复现：

```bash
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.train_online --output logs/supermimic/p3_online_new --actor-lr 1e-06 --seed 2028 --updates 100 --test-seeds 62 63 64
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m unittest discover -s supermimic/tests -v
```
