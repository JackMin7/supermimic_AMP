# P2 世界模型诊断结果

已完成真实数据采集、8 步自回归训练与独立环境验证；本轮不更新 actor，也不是完整在线 WMRL。

- 设备：RTX 4090；64 个环境 × 512 步 = 32,768 条 transition。
- 训练/验证按环境 ID 固定为 48/16；归一化仅使用训练环境有效数据。
- 数据源：冻结 PPO + 标准差 0.1 的动作噪声；保留启动 DR，关闭推扰，episode 上限 5 秒。
- 网络：3 × 512，动态关节/上下文维度，局部状态增量 + 3 维接触 logits。论文默认 3 × 2048 尚未运行。
- 训练：500 次 Adam 更新，batch 128，8 步窗口，学习率 2e-4，梯度范数裁剪 1。
- 采集耗时 14.56 秒，WM 更新耗时 45.52 秒；不是策略训练速度对比。

## 验证集总体误差

| 时域 | 根位置 WM / 恒速（m） | 关节位置 WM / 恒速（rad） | 关节速度 WM / 恒速（rad/s） |
|---|---:|---:|---:|
| 1 | 0.0032 / 0.0007 | 0.0139 / 0.0142 | 0.6110 / 0.8808 |
| 8 | 0.0243 / 0.0291 | 0.1125 / 0.3376 | 2.4539 / 4.0244 |
| 32 | 0.0908 / 0.1824 | 0.1192 / 1.2535 | 1.1400 / 2.1465 |

误差是按样本及坐标/关节维度计算的 RMSE；root orientation 是旋转向量各分量 RMSE。每个时域/类别最多抽取 128 个不重复窗口。不同长度窗口的可用起点不同，因此不能把表中各行当成同一组轨迹的误差曲线。JSON 同时提供状态不变基线、standing/moving/fallen 分类、窗口数量和接触 BCE。

## 结论与限制

- 8/32 步总体误差优于恒速度外推；单步根位置仍不如恒速度基线。站立类别部分指标也未全面胜出。
- 32 步倒地起点的根位置 RMSE 为 0.219 m，关节位置 RMSE 为 0.300 rad。不能据此认定长时域恢复 BPTT 已可靠。
- 采集时 1 个 PPO 动作源超过 abs(action)<100 的数值保护阈值，在执行前被重置；本轮无 physics_guard 拒绝。该 reset 已写入 episode_id，Replay 不跨此边界。
- 首次无保护采集在 step 112 发生 PPO 非有限动作；物理状态速度检查未捕获该动作历史发散。因此最终诊断增加动作源保护，未将大动作裁剪后继续冒充正常样本。
- 物理/接触非有限值或绝对速度≥1000 的 transition 另有 dynamics_valid 标记，禁止进入训练统计/序列；阈值是数值保护，不是任务终止标准。
- checkpoint 重载输出误差为 0；从保存的 Replay 和 checkpoint 独立复算，所有 1/8/32 步指标完全一致。

下一步可以开展短时域 BPTT 验证；长时域恢复能力仍需真实仿真反馈和更多数据。

## 复现

```bash
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.train_world_model --output logs/supermimic/p2_new_run --save-replay
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.evaluation.world_model --checkpoint logs/supermimic/p2_diagnostic_20261026/world_model.pt --replay logs/supermimic/p2_diagnostic_20261026/replay.pt --output /tmp/p2_validation.json
/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m unittest discover -s supermimic/tests -v
```

训练日志、完整 500 次 loss 和 checkpoint 位于 `logs/supermimic/p2_diagnostic_20261026/`；精简结果位于 `supermimic/reports/p2_world_model.json`。
