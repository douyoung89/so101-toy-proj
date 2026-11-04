"""Import local raw episodes once; identical PNG files share one data/images path."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import read_json, read_jsonl


def import_data(source):
    destination = ROOT / 'data'
    images = destination / 'images'
    images.mkdir(parents=True, exist_ok=True)
    image_paths = {}  # Exact PNG bytes -> shared path; no approximate image matching.
    original_count = original_bytes = 0

    def store_image(path):
        nonlocal original_count, original_bytes
        content = path.read_bytes()
        original_count += 1
        original_bytes += len(content)
        if content not in image_paths:
            name = f'frame_{len(image_paths) + 1:06d}.png'
            (images / name).write_bytes(content)
            image_paths[content] = f'data/images/{name}'
        return image_paths[content]

    starts = {p['scene_id']: p['start_deg'] for p in read_jsonl(source / 'rollout_plan.jsonl')}
    scenes, counts = [], {split: {'scenes': 0, 'episodes': 0, 'observations': 0} for split in ('train', 'val', 'test')}
    for directory in sorted((source / 'scenes').iterdir()):
        if not (directory / 'complete.json').exists():
            continue
        original = read_json(directory / 'scene.json')
        truth = read_json(directory / 'truth.json')
        split = original['split']
        (destination / split).mkdir(exist_ok=True)
        views = []
        for row in read_jsonl(directory / 'view_ocr.jsonl'):
            views.append({'angle': row['command_relative_deg'], 'readings': row['readings'],
                          'ocr': row['ocr'], 'image': store_image(directory / row['image']),
                          'invalid_reasons': row['invalid_reasons']})
        scenes.append({'scene_id': original['scene_id'], 'split': split, 'label_id': original['label_id'],
                       'start_deg': starts[original['scene_id']], 'safe_angles': original['safe_angles_deg'],
                       'truth': {f: truth[f] for f in ('protein_g', 'sugar_g')},
                       'label_image': Path(truth['label']['image']).name,
                       'settings': truth['settings'], 'views': views})
        counts[split]['scenes'] += 1
        for path in sorted((directory / 'episodes').glob('*.json')):
            old = read_json(path)
            observations = []
            for i, row in enumerate(old['observations']):
                action = old['actions'][i]['action'] if i < len(old['actions']) else None
                if i == len(old['observations']) - 1 and old['terminal_action'] is not None:
                    action = old['terminal_action']['action']
                observations.append({'step': i, 'image': store_image(directory / row['image']),
                                     'wrist_deg': row['command_relative_deg'],
                                     'relative_actual_deg': row['relative_actual_deg'],
                                     'joint_state': row['joint_state'], 'ocr': row['ocr'],
                                     'readings': row['readings'], 'memory': row['memory'],
                                     'arrival': row['arrival'], 'action': action})
            episode = {k: old[k] for k in ('episode_id', 'scene_id', 'strategy', 'seed', 'start_deg',
                                           'stop_reason', 'success', 'wrong_stop', 'technical_errors')}
            episode.update(label_id=original['label_id'], observations=observations)
            (destination / split / f'{episode["episode_id"]}.json').write_text(json.dumps(episode, indent=2) + '\n')
            counts[split]['episodes'] += 1
            counts[split]['observations'] += len(observations)
    (destination / 'scenes.json').write_text(json.dumps(scenes, indent=2) + '\n')
    summary = {'source': str(source.resolve()), 'splits': counts,
               'referenced_images': original_count, 'unique_images': len(image_paths),
               'image_bytes_before_sharing': original_bytes,
               'image_bytes_after_sharing': sum(len(content) for content in image_paths),
               'image_policy': 'Original-resolution PNGs. Exact duplicates share paths. 224x224 derivatives are omitted.'}
    (destination / 'source.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Path to an original local collection')
    args = parser.parse_args()
    configured = read_json(ROOT / 'config.json').get('source_data')
    source = args.source or (Path(configured) if configured else None)
    if source is None:
        parser.error('Pass --source /path/to/original/data; the published dataset is already included')
    import_data(source)
