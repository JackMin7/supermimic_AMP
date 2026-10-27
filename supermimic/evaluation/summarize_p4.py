"""Publish all P4 evaluation settings and approximation limits."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.run/'result.json').read_text());t=json.loads((a.run/'training.json').read_text())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_suffix('.json').write_text(json.dumps({'result':r,'training':t},indent=2,allow_nan=False))
    lines=['# P4：可微 AMP 与恢复在线闭环','',
           f"本轮完整工程可比验收：{'通过' if r['passed'] else '未通过'}。仅代表从 PPO/P3 初始化的小规模诊断，不代表从零训练、统计等效或论文加速复现。",'',
           '## 实现与验证','',
           '- AMP/任务奖励与原接口逐项数值对齐，输入及端到端 actor 梯度通过有限差分；见 p4_parity.json、p4_gradient.json。',
           '- 保留原 PPO 路径，新增可微 AMP 特征与奖励；195 维特征、0.1 style 系数及 0.75 task 混合权重。',
           '- 判别器仅使用真实策略/专家 transition，沿用原 PPO least-squares、原始专家梯度惩罚和 normalizer 更新语义；想象阶段冻结参数与统计。',
           '- 40% 起点来自恢复环境；加入延迟计数、恢复清零、timeout、奖励模式与 heading。critic 显式接收任务状态。',
           '- 控制步加速度、接触概率、自碰撞计数与软终止属于代理；真实评估采用原生任务奖励、真实接触和硬终止。', '',
           f"采集 {t['transitions']:,} 条 transition，actor/critic 各 {r['protocol']['updates']} 次交替更新，WM {2*r['protocol']['updates']} 次，判别器 {r['protocol']['updates']} 次。训练耗时 {t['training_seconds']:.1f} 秒；数值保护事件 {len(t['guard_events'])} 次。",'',
           '网络更新/统计冻结检查：`'+json.dumps(t['checks'],ensure_ascii=False)+'`。', '',
           '## 真实仿真','',
           '两套策略用原 PPO 的同一冻结判别器与 AMP normalizer 评分，避免移动奖励导致虚假收益。最终 checkpoint 在测试前固定；初态和物理上下文 SHA256 配对。', '',
           '验收：P3 四项回报下降 ≤2%，XY/yaw RMSE 各增加 ≤5%，存活和恢复成功率不下降；新增完整 AMP 混合回报在速度和恢复场景分别下降 ≤2%。', '']
    for name,g in r['groups'].items():
        b,c=g['aggregate']['baseline'],g['aggregate']['p4']
        lines += [f"### {name}：{'通过' if g['comparability']['passed'] else '未通过'}",'',
                  '| 指标 | 原 PPO | P4 |','|---|---:|---:|']
        for key in b:
            fmt=lambda value: 'nonfinite' if value is None else f'{value:.6f}'
            lines.append(f'| {key} | {fmt(b[key])} | {fmt(c[key])} |')
        lines += ['', '检查：`'+json.dumps(g['comparability']['checks'],ensure_ascii=False)+'`。','']
    lines += ['| 种子 | 推扰步 | 单种子检查 | 未通过项 |','|---:|---:|---|---|']
    for p in r['tests']:
        gate=p['comparability'];failed=', '.join(k for k,v in gate['checks'].items() if not v)
        lines.append(f"| {p['seed']} | {p['push_step']} | {'通过' if gate['passed'] else '未通过'} | {failed} |")
    lines += ['', '推扰仅对速度场景施加：第 5 秒根速度增加 (0.6, 0.3) m/s。固定恢复场景观察前 5 秒，要求连续直立至少 1 秒。', '',
              '## 限制与后续', '',
              '训练保留 startup DR，关闭随机 interval 推扰；只有一个评估种子的确定性推扰对照。尚需完整随机推扰训练、多训练种子、接触/加速度代理消融与较长训练预算，才能判断完整任务稳健性和学习效率。', '',
              '专家邻帧必须来自同一 clip 且采样间隔匹配 50 Hz；本实现排除原 loader 的末帧重复 pair。延迟计数严格匹配现有实现，包括 timeout 当步参与计数但不延迟 timeout。', '',
              '保存 actor、两种 normalizer、WM、critic、判别器、全部优化器、专家游标、RNG 与 replay；尚无模拟器精确续训 CLI。', '',
              f"产物：`{a.run}/actor_final.pt`、`{a.run}/replay.pt`。", '',
              '```bash',f"/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m supermimic.train_p4 --output logs/supermimic/p4_repeat --actor-lr {r['protocol']['actor_lr']} --seed {r['protocol']['seed']} --test-seeds "+' '.join(map(str,r['protocol']['test_seeds'])),
              '/home/ubuntu/USE/Miniconda/envs/mjlab/bin/python -m unittest discover -s supermimic/tests -v','```','']
    if r['nonfinite_metrics']:lines += ['非有限指标：`'+json.dumps(r['nonfinite_metrics'])+'`。']
    a.output.with_suffix('.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
