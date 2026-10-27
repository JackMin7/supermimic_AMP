"""Record native MuJoCo states/actions for SDK2 replay (run in mjlab)."""
import argparse
import hashlib
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sim2sim import Simulator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('deploy/wmrl/trace.npz'))
    p.add_argument('--seconds', type=float, default=20.)
    p.add_argument('--command', nargs=3, type=float, default=[1.5,0.,0.])
    a = p.parse_args()
    if a.seconds <= 0: p.error('--seconds must be positive')
    sim = Simulator(a.policy, a.command)
    trace = {k: [] for k in ['q','dq','quat','gyro','observation','action','target']}
    for _ in range(max(1, round(a.seconds/sim.dt))):
        trace['q'].append(sim.data.qpos[sim.qadr].copy())
        trace['dq'].append(sim.data.qvel[sim.vadr].copy())
        trace['quat'].append(sim.data.xquat[sim.pelvis].copy())
        trace['gyro'].append(sim.data.sensor('imu_ang_vel').data.copy())
        action = sim.step()
        trace['observation'].append(sim.history.reshape(384).copy())
        trace['action'].append(action.copy())
        trace['target'].append(sim.default_q + sim.scale * action)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, **{k: np.asarray(v) for k,v in trace.items()}, command=a.command,
             joint_names=sim.names, dt=sim.dt,
             policy_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest(),
             metadata_sha256=hashlib.sha256(a.policy.with_suffix('.json').read_bytes()).hexdigest())
    print(f'Recorded {len(trace["action"])} steps -> {a.output}')


if __name__ == '__main__': main()
