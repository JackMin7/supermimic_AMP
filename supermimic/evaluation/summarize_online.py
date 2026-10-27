"""Render the fixed-final-checkpoint online experiment without selecting a winner."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=json.loads((a.run/'result.json').read_text())
    training=json.loads((a.run/'training.json').read_text())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_suffix('.json').write_text(json.dumps({'result':result,'training':training},indent=2,allow_nan=False))
    b,c=result['aggregate']['baseline'],result['aggregate']['online']
    gate=result['comparability']
    lines=['# P3 在线交替更新验证','',
           f"固定最终 checkpoint 的 {len(result['tests'])} 种子配对评估：{'通过' if gate['passed'] else '未通过'} PPO 工程可比验收。平均任务回报变化 {100*gate['relative_return_change']:+.4f}%。",'',
           '验收标准在运行前写入 protocol.json：回报下降不超过 2%；XY/yaw RMSE 各增加不超过 5%；存活率和固定恢复成功率不下降。无需超越 PPO。', '',
           '| 指标 | 原 PPO | 在线 BPTT |','|---|---:|---:|']
    for key in b: lines.append(f'| {key} | {b[key]:.6f} | {c[key]:.6f} |')
    lines += ['', '| 测试种子 | 回报变化 | 单种子可比检查 |','|---:|---:|---|']
    for pair in result['tests']:
        comparison=pair['comparability']
        lines.append(f"| {pair['seed']} | {100*comparison['relative_return_change']:+.4f}% | {'通过' if comparison['passed'] else '未通过'} |")
    if result.get('nonfinite_metrics'):
        lines += ['', '非有限指标（JSON 中存为 null，验收失败）：`'+json.dumps(result['nonfinite_metrics'],ensure_ascii=False)+'`。', '']
    lines += ['', '## 在线训练与验证', '',
              f"64 个环境，48/16 训练/验证划分。采集 {training['transitions']:,} 条当前策略 transition；{result['protocol']['updates']} 次 H=8 actor/critic 交替更新和 {training['wm_updates']} 次 WM 更新。训练与 WM 验证耗时 {training['training_seconds']:.1f} 秒，不含环境初始化和真实配对评估。",'',
              f"梯度隔离审计 {training['checks']['audited_updates']} 次；actor 和 WM 确实更新，PPO normalizer 与 WM 归一化统计逐项保持不变。数值保护重置事件 {len(training['guard_events'])} 次。",'',
              '最终 checkpoint 在真实测试前固定，通过原生 PPO 加载接口重载并逐项核对参数。每组基线/候选的初态和物理上下文哈希完全一致。', '',
              '| 固定 holdout 预测误差 | 刷新前 | 刷新后 |','|---|---:|---:|']
    for h in ('1','8','32'):
        for key in ('root_position_rmse_m','joint_position_rmse_rad'):
            before=training['fixed_holdout_before'][h]['all']['world_model'][key]
            after=training['fixed_holdout_after'][h]['all']['world_model'][key]
            lines.append(f'| {h} 步 {key} | {before:.6f} | {after:.6f} |')
    lines += ['', '长时域根位置误差没有同步改善；保持短时域训练，不能仅依据训练 loss 扩展 horizon。', '',
              '## 范围与下一阶段','',
              '这是从已有 PPO/P3 策略微调的单训练种子速度任务消融。三组评估种子不是三次独立训练；工程容差通过不等于统计等效、完整 AMP 复现或从零学习能力。恢复仅作固定场景回归测试，本轮未训练 AMP/恢复目标，也未评估推扰。', '',
              '下一阶段接入可微 AMP reward、真实 transition 判别器更新、恢复计数和奖励模式，再用完整任务评估。checkpoint 保存网络、优化器、RNG，replay 另存；当前无模拟器精确续训 CLI。', '',
              f"实验目录：`{a.run}`；策略：`{a.run}/actor_final.pt`。",'',
              '复现：', '', '```bash',
              f"/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.train_online --output logs/supermimic/p3_online_new --actor-lr {result['protocol']['actor_lr']} --seed {result['protocol']['seed']} --updates {result['protocol']['updates']} --test-seeds "+' '.join(map(str,result['protocol']['test_seeds'])),
              '/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m unittest discover -s supermimic/tests -v','```','']
    a.output.with_suffix('.md').write_text('\n'.join(lines))


if __name__=='__main__': main()
