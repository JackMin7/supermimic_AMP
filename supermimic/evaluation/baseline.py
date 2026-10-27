"""Freeze dependency/code/data provenance; does not claim a scored baseline."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    checkpoints = list((root / 'logs').glob('**/model_*.pt'))
    # Selection is reproducible, not a claim of best performance.
    latest = max(checkpoints, key=lambda p: (p.parent.name, int(p.stem.split('_')[-1]))) if checkpoints else None
    versions = {}
    for name in ('mjlab', 'mujoco', 'mujoco-warp', 'warp-lang', 'torch', 'rsl-rl-lib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    files = list((root / 'src/tasks/amp_loco').rglob('*.py'))
    files += list((root / 'src/assets/robots/unitree_g1').rglob('*.xml'))
    files += list((root / 'src/assets/robots/unitree_g1').glob('*.py'))
    files += list((root / 'src/assets/motions/g1/amp').rglob('*.npz'))
    files += list((root / 'rsl_rl').rglob('*.py'))
    report = {
        'status': 'provenance_frozen_evaluation_pending',
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
        'versions': versions,
        'checkpoint_selection': 'latest lexicographic run directory, then largest iteration; not best score',
        'checkpoint': None if latest is None else {'path': str(latest.relative_to(root)), 'sha256': sha256(latest)},
        'files_sha256': {str(p.relative_to(root)): sha256(p) for p in sorted(files)},
        'evaluation_protocol': {
            'seeds': [42, 43, 44], 'control_dt': 0.02,
            'velocity_commands': [[0, 0, 0], [0.5, 0, 0], [1.5, 0, 0], [0, 0.5, 0], [0, 0, 0.75]],
            'velocity_episode_seconds': 20,
            'recovery_initial_states': 'fixed frame indices from Recovery, to be materialized before scoring',
            'recovery_success_candidate': 'root height >= 0.5 m and tilt <= 70 degrees continuously for 1 second within 5 seconds',
            'push_scenarios': 'pending deterministic event schedule; do not report robustness until fixed',
            'metrics': ['velocity_rmse', 'survival_seconds', 'recovery_success', 'recovery_time', 'foot_slip', 'action_rate', 'gpu_hours'],
            'scores': None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f'Wrote provenance (evaluation pending): {args.output}')


if __name__ == '__main__':
    main()
