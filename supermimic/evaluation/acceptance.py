"""Engineering comparability margins requested by the user; not a statistical equivalence test."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class ComparabilityMargins:
    return_drop_fraction: float = 0.02
    error_increase_fraction: float = 0.05
    success_drop_fraction: float = 0.0


def compare(baseline, candidate, margins=ComparabilityMargins()):
    keys = ('mean_task_return', 'xy_rmse', 'yaw_rmse', 'velocity_survival_rate', 'recovery_success_rate')
    finite = all(math.isfinite(float(m[k])) for m in (baseline, candidate) for k in keys)
    checks = {'finite_metrics': finite,
              'return_within_margin': candidate['mean_task_return'] >= baseline['mean_task_return'] - margins.return_drop_fraction * abs(baseline['mean_task_return']),
              'xy_error_within_margin': candidate['xy_rmse'] <= (1 + margins.error_increase_fraction) * baseline['xy_rmse'],
              'yaw_error_within_margin': candidate['yaw_rmse'] <= (1 + margins.error_increase_fraction) * baseline['yaw_rmse'],
              'survival_preserved': candidate['velocity_survival_rate'] >= baseline['velocity_survival_rate'] - margins.success_drop_fraction,
              'recovery_preserved': candidate['recovery_success_rate'] >= baseline['recovery_success_rate'] - margins.success_drop_fraction}
    return {'passed': all(checks.values()), 'checks': checks, 'margins': asdict(margins),
            'relative_return_change': (candidate['mean_task_return']-baseline['mean_task_return']) / max(abs(baseline['mean_task_return']), 1e-12)}


def json_safe_metrics(value, path=''):
    """Keep failures reportable: nonfinite scalars become null with explicit paths."""
    failures = []
    if isinstance(value, float) and not math.isfinite(value):
        return None, [{'path':path,'value':str(value)}]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result[key], bad = json_safe_metrics(item, f'{path}/{key}')
            failures.extend(bad)
        return result, failures
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            clean, bad = json_safe_metrics(item, f'{path}/{index}')
            result.append(clean); failures.extend(bad)
        return result, failures
    return value, failures
