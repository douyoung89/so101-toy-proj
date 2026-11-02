"""Online policy evaluation: fresh MuJoCo frames and OCR after each move."""
import argparse
import copy
from collections import defaultdict
import importlib.util
from pathlib import Path

from common import DEFAULT_DATA, read, rows, write, metrics, sha
from collect_partial import run_episode


def evaluate(dataset, policy_path=None, development=False):
    config = read(dataset/'dataset_config.json')
    if config['development_only'] and not development:
        raise ValueError('These labels were used in prior development. Use --development for pilot comparison; final evaluation requires an untouched dataset.')
    policy_factory = None
    if policy_path:
        spec = importlib.util.spec_from_file_location('partial_user_policy', policy_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        policy_factory = module.make_policy
    plans = [p for p in rows(dataset/'rollout_plan.jsonl') if p['strategy'] == 'shortest'
             and read(dataset/'scenes'/p['scene_id']/'scene.json')['split'] == 'test']
    results = defaultdict(list)
    if not plans:
        raise ValueError('No test starts available; sample a complete dataset first')
    for plan in plans:
        directory = dataset/'scenes'/plan['scene_id']
        for strategy in ['fixed_sweep', 'left_first', 'right_first', 'random'] + (['policy'] if policy_factory else []):
            ep = {k: plan[k] for k in ['scene_id', 'seed', 'start_deg']}
            ep.update(strategy=strategy, episode_id=f'eval_{plan["scene_id"]}_{strategy}')
            # Same start/config/seed for each comparator; fresh simulator and memory.
            policy = None
            if strategy == 'policy':
                selected_policy = policy_factory()

                def policy(state):
                    # Stored episode paths are scene-relative; external policies need usable files.
                    resolved = copy.deepcopy(state)
                    for observation in resolved['observations']:
                        for key in ['image', 'policy_image']:
                            resolved_path = (directory/observation[key]).resolve()
                            if not resolved_path.is_file():
                                raise FileNotFoundError(resolved_path)
                            observation[key] = str(resolved_path)
                    return selected_policy(resolved)
            if strategy == 'fixed_sweep':
                # Sweep toward the closer boundary first, then scan in the other direction.
                safe = read(directory/'scene.json')['safe_angles_deg']
                ep['strategy'] = 'left_first' if plan['start_deg']-min(safe) <= max(safe)-plan['start_deg'] else 'right_first'
            result = run_episode(directory, config, ep, policy=policy,
                                 destination=dataset/'evaluation'/f'{ep["episode_id"]}.json')
            result['strategy'] = strategy
            write(dataset/'evaluation'/f'{ep["episode_id"]}.json', result)
            results[strategy].append(result)
    report = {'development_only': config['development_only'], 'test_starts': len(plans),
              'policy_sha256': sha(policy_path) if policy_path else None,
              'evaluation_code_sha256': sha(Path(__file__)),
              'online_fresh_frames': True, 'comparisons': {s: metrics(es) for s, es in results.items()},
              'technical_failures': {s: sum(bool(e['technical_errors']) for e in es) for s, es in results.items()}}
    write(dataset/'evaluation/report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--policy', type=Path, help='Python module exporting make_policy() -> callable(input) -> action')
    p.add_argument('--development', action='store_true')
    a = p.parse_args()
    print(evaluate(a.data, a.policy, a.development))
