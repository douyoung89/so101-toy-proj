"""Select starts with approximate 20/40/40 difficulty and balanced directions."""
import argparse
from collections import Counter
from pathlib import Path
import random

from common import DEFAULT_DATA, read, rows, jsonl, write, scene_dirs


def bucket(n):
    return '1' if n == 1 else '2-3' if n <= 3 else '4-6' if n <= 6 else '7-8'


def sample(dataset, seed=20261029):
    rng = random.Random(seed)
    counts, directions = Counter(), Counter()
    plans = []
    targets = {'1': .2, '2-3': .4, '4-6': .4}
    directories = scene_dirs(dataset)
    rng.shuffle(directories)
    for directory in directories:
        check = read(directory/'check.json')
        if not check['accepted']:
            continue
        candidates = check['eligible_starts'][:]
        rng.shuffle(candidates)
        selected = max(candidates, key=lambda s: (
            targets.get(bucket(s['shortest_moves']), 0)*(len(plans)/4+1)-counts[bucket(s['shortest_moves'])],
            -directions[s['first_direction']]))
        counts[bucket(selected['shortest_moves'])] += 1
        directions[selected['first_direction']] += 1
        for strategy in ['left_first', 'right_first', 'random', 'shortest']:
            plans.append({'scene_id': directory.name, 'episode_id': f'episode_{len(plans)+1:06d}',
                          'seed': rng.randrange(2**31), 'strategy': strategy, **selected})
    jsonl(dataset/'rollout_plan.jsonl', plans)
    write(dataset/'sampling_report.json', {'seed': seed, 'scenes': len(plans)//4,
          'difficulty_counts': dict(counts), 'direction_counts': dict(directions), 'targets': targets,
          'note': 'Targets are pilot goals; availability limits the realized distribution. Paths are privileged supervision only.'})
    return plans


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--seed', type=int, default=20261029)
    a = p.parse_args()
    sample(a.data, a.seed)
