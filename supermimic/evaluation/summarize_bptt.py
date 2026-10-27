"""Summarize all P3 attempts, including failed gates and non-selected candidates."""
import argparse
import json
from pathlib import Path
from supermimic.evaluation.acceptance import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    experiments=[]
    for path in args.results:
        r=json.loads(path.read_text())
        if r['status']!='completed': raise ValueError(f'Incomplete run: {path}')
        experiments.append({'source':str(path), **r, 'retrospective_comparability':compare(r['aggregate']['baseline'],r['aggregate']['bptt'])})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps({'experiments':experiments},indent=2,allow_nan=False))
    conclusion = '部分变体达到预设真实收益门槛，具体适用范围见下表。' if any(r['real_benefit_gate_passed'] for r in experiments) else '**历史结论：工程链路通过，两种变体未通过当时要求的收益提升门槛。**'
    lines=['# P3：短时域 BPTT 与真实仿真对照','',conclusion,'',
           '用户随后将目标调整为媲美 PPO：回报下降 ≤2%，XY/yaw RMSE 各增加 ≤5%，存活与恢复成功率不下降。以下旧门槛保留；新标准回溯结果：'+str([r['retrospective_comparability']['passed'] for r in experiments])+'。该回溯判断不是预注册试验。','',
           '保留原 PPO actor、384 维输入和归一化，使用 P2 的 3×512 冻结世界模型。训练起点仅来自 P2 训练环境，排除恢复子集、heading 控制和即将重采样的命令。', '',
           '训练目标为已有 torso 线/角速度跟踪、动作平滑和 soft joint limits 四项奖励；不含 AMP、接触奖励或恢复目标。真实终止额外计入 -200×dt，想象使用软存活权重。结果不是完整 AMP-WMRL 的性能结论。','',
           '## 真实仿真结果','',
           '| 变体 | 测试种子 | 选中更新 | 原 PPO 回报 | BPTT 回报 | 相对变化 | 改善种子 | 验收 |',
           '|---|---|---:|---:|---:|---:|---:|---|']
    for r in experiments:
        weight=r['protocol'].get('bootstrap_weight',1.0)
        b,a=r['aggregate']['baseline'],r['aggregate']['bptt']
        lines.append(f"| bootstrap={weight:g} | {r['protocol']['test_seeds']} | {r['selected_update']} | {b['mean_task_return']:.5f} | {a['mean_task_return']:.5f} | {100*r['relative_return_gain']:+.3f}% | {r['winning_test_seeds']}/{len(r['tests'])} | {'通过' if r['real_benefit_gate_passed'] else '未通过'} |")
    lines += ['', '预先设定的验收线：平均回报提升 >0.5%，至少 2/3 测试种子改善，速度场景存活率不降低，XY/yaw 平均 RMSE 各不超过基线的 105%。恢复成功率另列为回归观察，不作为已完成恢复训练的证据。','',
              '每个候选只用开发种子 14201 筛选，选定后才运行测试种子。baseline/BPTT 的初始状态和物理随机化上下文以 SHA256 校验配对。两个实验使用不同测试种子，不能将两列绝对回报直接当成受控的跨变体比较。','']
    for r in experiments:
        weight=r['protocol'].get('bootstrap_weight',1.0)
        b,a=r['aggregate']['baseline'],r['aggregate']['bptt']
        lines += [f'## bootstrap={weight:g}', '',
                  f"XY RMSE：{b['xy_rmse']:.5f} → {a['xy_rmse']:.5f} m/s；yaw RMSE：{b['yaw_rmse']:.5f} → {a['yaw_rmse']:.5f} rad/s。",
                  f"速度存活率：{b['velocity_survival_rate']:.0%} → {a['velocity_survival_rate']:.0%}；固定恢复初态成功率：{b['recovery_success_rate']:.0%} → {a['recovery_success_rate']:.0%}。", '',
                  '| 开发候选更新 | 任务回报 | 存活率 |', '|---:|---:|---:|']
        for update, entry in r['development'].items():
            summary=entry['summary']
            lines.append(f"| {update} | {summary['mean_task_return']:.5f} | {summary['velocity_survival_rate']:.0%} |")
        lines += ['', f"完整记录：`{r['source']}`。", '']
    lines += ['## 工程验证', '',
              '- 真实网络的 actor 梯度通过有限差分方向检查；WM 参数/统计在训练期间逐项不变。',
              '- GAE terminal/bootstrap、actor 与 critic 梯度隔离、Replay 边界与状态积分测试通过。',
              '- 四项可微奖励在相同 post-forward 时刻与原生环境逐项合成结果对齐，误差见 p3_heads_probe.json。', '',
              '本轮为单一训练种子的离线微调实验；多个评估种子不是多个独立训练复现。不得把想象目标上升当成真实收益，也不得把未通过门槛的 checkpoint 替换现有 PPO。','',
              '## 复现', '', '```bash',
              '/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.train_bptt --output logs/supermimic/p3_new_run',
              '/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.evaluation.bptt --candidate logs/supermimic/p3_bptt_20261026/actor_10.pt --output /tmp/p3_recheck.json',
              '/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m unittest discover -s supermimic/tests -v',
              '```','']
    args.output.with_suffix('.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
