"""Seeded, resumable 23-view scene collection. Run init, then collect."""
import argparse
import random
from pathlib import Path

from common import DEFAULT_DATA, ROOT, read, rows, write, jsonl, sha, FIELDS


def initialize(dataset, count=15, seed=20261029):
    from simulator import baseline, labels
    import importlib.metadata
    dataset = Path(dataset)
    pool = labels()
    rng = random.Random(seed)
    rng.shuffle(pool)
    # Keep pre-existing label splits; balance pilot coverage across all three splits.
    pools = {s: [l for l in pool if l['split'] == s] for s in ['train', 'val', 'test']}
    order = ['train']*7 + ['val']*2 + ['test']
    plan = []
    for i in range(count):
        split = order[i % 10]
        label = pools[split][(i//10*order.count(split)+order[:i % 10].count(split)) % len(pools[split])]
        plan.append({'scene_id': f'scene_{i+1:06d}', 'label_id': label['label_id'],
                     'split': split, 'seed': seed+i*1009, 'label': label})
    config = {'version': 'partial_v1', 'seed': seed, 'planned_scenes': count,
              'angles_deg': list(range(-70, 41, 5)), 'start_angles_deg': list(range(-30, 26, 5)),
              'actions': ['LEFT_5', 'RIGHT_5', 'STOP'], 'max_moves': 8,
              'ocr_size': [960, 720], 'policy_size': [224, 224],
              'thresholds': read(ROOT/'assets/reference/thresholds.json')['thresholds'],
              'difficulty_targets': {'1': .2, '2-3': .4, '4-6': .4},
              'baseline': baseline(), 'image_reuse': False,
              'development_only': True,
              'evaluation_note': 'Existing labels were used in prior experiments; final untouched labels must be supplied separately.',
              'sources': {str(p.relative_to(ROOT)): sha(p) for p in [
                  ROOT/'data/labels/labels.csv', ROOT/'assets/reference/thresholds.json',
                  *[ROOT/'support'/n for n in ['stage2.py', 'stage3.py', 'stage5.py', 'wrist_sweep.py']],
                  *sorted((ROOT/'data/easyocr_models').glob('*.pth')),
                  *sorted((ROOT/'external/SO-ARM100/Simulation/SO101/assets').glob('*'))] if p.is_file()},
              'packages': {p: importlib.metadata.version(p) for p in ['mujoco', 'numpy', 'Pillow', 'torch', 'easyocr', 'tqdm']},
              'code_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}}
    if (dataset/'dataset_config.json').exists():
        if read(dataset/'dataset_config.json') != config or rows(dataset/'manifest.jsonl') != plan:
            raise ValueError('Existing config/source/plan differs; use a new dataset directory')
        return config
    write(dataset/'dataset_config.json', config)
    jsonl(dataset/'manifest.jsonl', plan)
    write(dataset/'splits.json', {s: sorted({p['label_id'] for p in plan if p['split'] == s}) for s in pools})
    return config


def collect(dataset, retries=20):
    from simulator import Simulator, create_scene, sample_settings
    from check_partial_scene import check_scene
    from tqdm import tqdm
    config = read(dataset/'dataset_config.json')
    for name, digest in {**config['sources'], **config['baseline']['source_sha256']}.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Shared asset changed: {name}; use a new dataset')
    for name, digest in config['code_sha256'].items():
        if sha(Path(__file__).parent/name) != digest:
            raise ValueError(f'Code changed after initialization: {name}; use a new dataset')
    plan = rows(dataset/'manifest.jsonl')
    progress = tqdm(plan, desc='Scenes', unit='scene')
    rejected = skipped = 0
    for item in progress:
        directory = dataset/'scenes'/item['scene_id']
        if (directory/'complete.json').exists():
            completion = read(directory/'complete.json')
            if not all((directory/p).exists() and sha(directory/p) == h for p, h in completion['hashes'].items()):
                raise ValueError(f'Corrupt completed scene: {directory}')
            skipped += 1
            progress.set_postfix(rejected=rejected, skipped=skipped)
            continue
        attempt_log = rows(directory/'attempts.jsonl') if (directory/'attempts.jsonl').exists() else []
        start_attempt = len(attempt_log)
        for attempt in range(start_attempt, retries):
            seed = item['seed']+attempt*1000003
            settings = sample_settings(seed)
            create_scene(directory/'simulation', item['label'], settings)
            scene = {k: item[k] for k in ['scene_id', 'label_id', 'split', 'seed']}
            scene.update(attempt=attempt, accepted_seed=seed, safe_angles_deg=[], image_reuse_verified=False)
            write(directory/'scene.json', scene)
            write(directory/'truth.json', {**{f: int(item['label'][f]) for f in FIELDS},
                                          'label': item['label'], 'settings': settings})
            sim = Simulator(directory/'simulation', settings, item['label'])
            observations = []
            try:
                sim.move(0)
                for index, angle in enumerate(tqdm(config['angles_deg'], desc=item['scene_id'], leave=False, unit='view')):
                    motion = sim.move(angle)
                    record = sim.capture(directory/'views', f'view_{index:02d}', config['thresholds'])
                    record.update(view_id=f'view_{index:02d}', command_relative_deg=angle, motion=motion,
                                  physical_exposure=sim.exposure(), invalid_reasons=motion['invalid_reasons'])
                    record['image'] = 'views/'+record['image']
                    record['policy_image'] = 'views/'+record['policy_image']
                    observations.append(record)
                    jsonl(directory/'view_ocr.jsonl', observations)
            finally:
                sim.close()
            report = check_scene(directory, config)
            write(directory/'check.json', report)
            attempt_log.append({'attempt': attempt, 'seed': seed, 'accepted': report['accepted'],
                                'reasons': report['reasons']})
            jsonl(directory/'attempts.jsonl', attempt_log)
            if report['accepted']:
                (directory/'exhausted.json').unlink(missing_ok=True)
                scene['safe_angles_deg'] = report['safe_angles_deg']
                write(directory/'scene.json', scene)
                files = [p for p in directory.rglob('*') if p.is_file() and p.name not in ['complete.json', 'attempts.jsonl']]
                write(directory/'complete.json', {'hashes': {str(p.relative_to(directory)): sha(p) for p in files},
                                                  'config_sha256': sha(dataset/'dataset_config.json')})
                break
            # Retain the rejected candidate, including its original images and XML.
            import shutil
            archive = dataset/'rejected'/item['scene_id']/f'attempt_{attempt:03d}'
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(directory, archive, dirs_exist_ok=True)
            rejected += 1
            progress.set_postfix(rejected=rejected, retry=attempt+1, skipped=skipped)
        else:
            write(directory/'exhausted.json', {'attempts': len(attempt_log), 'reason': 'retry_limit'})
            progress.set_postfix(rejected=rejected, exhausted=item['scene_id'], skipped=skipped)
    write(dataset/'collection_status.json', {
        'planned_scenes': len(plan), 'completed_scenes': sum((dataset/'scenes'/p['scene_id']/'complete.json').exists() for p in plan),
        'exhausted_scenes': [p['scene_id'] for p in plan if (dataset/'scenes'/p['scene_id']/'exhausted.json').exists()
                             and not (dataset/'scenes'/p['scene_id']/'complete.json').exists()]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'collect'])
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--scenes', type=int, default=15)
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--retries', type=int, default=20)
    args = parser.parse_args()
    if args.command == 'init':
        if args.scenes <= 0:
            parser.error('--scenes must be positive')
        initialize(args.data, args.scenes, args.seed)
    else:
        collect(args.data, args.retries)


if __name__ == '__main__':
    main()
