# P4 性能优化与长训入口

使用原入口 `scripts/train.py`，默认 WMRL，无需指定 backend。`train_p4` 保留为原始诊断路径，不能仅延长它的更新次数来获得原项目的吞吐。

实测与限制见 [性能报告](reports/performance.md)。512 环境 / batch 512 已完成 100 次更新测试，14,000 次预计总计约 45–70 分钟（含首次编译和最终评估，未代跑完整长训）。

一键启动已测试的扩大配置：
```bash
cd /home/ubuntu/Sia/AMP_mjlab
bash supermimic/run_fast_14000.sh
```

真实回归的 82、84 和 82 推扰配对通过；83 的原 PPO 基线出现非有限动作，记录为无效，尚不宣称整组回归通过。

## 改动

- 想象时域先展开，再将 H+1 个状态的 FK 合并计算；同一结果供速度、足滑和 AMP 使用。原路径 H=8 需要 32 次 FK 调用，新路径只有一次更大的批量调用。
- 编译世界模型损失、想象单步及批量奖励；AMP normalizer 的运行统计留在 GPU，保存时导出为原生格式。
- 默认 FP32，原生策略推理和真实评估保持最高精度。TF32/BF16 是实验选项，尚不作为默认推荐。
- 日志按间隔追加 JSONL；参数隔离审计按间隔执行。环境数、actor/WM batch、horizon、UTD 可配置。
- 周期性原子保存 `last.pt`（包含 replay）和 `model_<update>.pt`（原始模块入口为 `actor_<update>.pt`）；最终 `actor_final.pt` 可由原 PPO 加载接口读取。

编译首次运行需要额外时间，GPU 利用率可能较低；编译缓存可供后续相同图形/形状复用。不要把首次编译时间当作稳定训练吞吐。

## 14,000 次更新

沿用原 AMP 入口，默认选择 WMRL 的优化 P4 训练器。PPO 训练入口已移除；`--backend supermimic` 仅作为 `wmrl` 的兼容别名。

```bash
cd /home/ubuntu/Sia/AMP_mjlab
python scripts/train.py Unitree-G1-AMP-Flat \
  --env.scene.num-envs=512 \
  --agent.max-iterations=14000 \
  --agent.seed=2031 \
  --agent.run-name=supermimic_14000 \
  --agent.save-interval=1000
```

默认 FP32、开启编译、batch/WM batch 512、H=8。日志和 TensorBoard 位于 `logs/rsl_rl/g1_supermimic_locomotion/<时间戳>_<run_name>/`，周期模型为 `model_<iteration>.pt`，完整续训文件为 `last.pt`。可加 `--supermimic.evaluate=True --supermimic.test-seeds 92 93 94` 在结束后评估。

环境参数仍使用 `--env.*`，通用训练参数仍使用 `--agent.*`。`num-steps-per-env` 表示每次更新的真实采集步数（默认 2）；学习率控制 actor，epoch/minibatch 控制 critic，gamma/lam 控制回报。SuperMimic 专用选项位于 `--supermimic.*`，可通过 `--help` 查看。选择该后端时默认关闭周期推扰和 curriculum，episode 长度为 5 秒；显式环境参数会覆盖默认值。目前仅支持 Flat、单 GPU、TensorBoard；未支持的 PPO 参数覆盖会报错。

512 环境已测试，不能按环境数线性外推吞吐；H=32 属于另一次算法实验。原始模块入口 `python -m supermimic.train_fast` 仍保留用于诊断。

## 续训

沿用原 AMP 的续训参数，重复原运行的环境和算法选项。`max-iterations` 是最终总更新次数：

```bash
python scripts/train.py Unitree-G1-AMP-Flat \
  --env.scene.num-envs=512 --agent.seed=2031 \
  --agent.max-iterations=14000 \
  --agent.resume=True \
  --agent.load-run=原运行目录名 \
  --agent.load-checkpoint=last.pt
```

续训恢复网络、归一化、全部优化器、replay、专家游标、采集比例累加器和随机状态。仿真重新初始化并采集至少 32 步，新 episode ID 大于旧 replay 的最大 ID；不会跨新旧仿真边界采样。**这不是模拟器内部状态的逐位续训。** 环境及训练参数冲突会报错。

## 独立测试与评估

```bash
$PYTHON_BIN -m unittest discover -s supermimic/tests -v
$PYTHON_BIN -m supermimic.performance.verify --compile --output /tmp/fast_parity.json
$PYTHON_BIN -m supermimic.performance.benchmark --engine fast --compile --output /tmp/fast_benchmark.json
$PYTHON_BIN -m supermimic.evaluation.p4 \
  --candidate logs/supermimic/运行目录/actor_final.pt \
  --output /tmp/p4_fast_evaluation.json --seeds 92 93 94
```

基准使用同步后的分段计时，剔除预热轮。TensorBoard 的首个区间含编译，后续 `Train/updates_per_second` 可用于该机器的实际长训估算。

## 范围

此版本仍是从已有 PPO/P3 初始化的 P4 训练；不是从零训练，也不是原项目 LAFAN1 tracking 任务。训练保留 startup DR，随机 interval 推扰关闭，真实评估含确定性推扰。14,000 次完整长训尚未代跑，其收敛与最终性能由实际训练结果判断。
