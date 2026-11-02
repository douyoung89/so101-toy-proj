"""Export causal histories and executed action labels, split by label identity."""
import argparse
from pathlib import Path

from common import DEFAULT_DATA, read, write, jsonl, scene_dirs, policy_input, sha
from audit_partial import audit


def build(dataset):
    report = audit(dataset)
    if report['split_errors']:
        raise ValueError('Label leakage between splits')
    if not report['complete_collection']:
        raise ValueError('Scene collection is incomplete')
    if not report['complete_episode_collection']:
        raise ValueError('Episode collection is missing or incomplete')
    excluded = set(report['excluded_episodes'])
    output = {s: [] for s in ['train', 'val', 'test']}
    config = read(dataset/'dataset_config.json')
    for directory in scene_dirs(dataset):
        scene = read(directory/'scene.json')
        for path in sorted((directory/'episodes').glob('*.json')):
            if str(path.relative_to(dataset)) in excluded:
                continue
            e = read(path)
            for i, obs in enumerate(e['observations']):
                action = e['actions'][i]['action'] if i < len(e['actions']) else (
                    e['terminal_action']['action'] if e['terminal_action'] and e['terminal_action']['action'] == 'STOP' else None)
                # Budget exhaustion is not a STOP choice and supplies no fictitious label.
                if action is None:
                    continue
                state = policy_input(e['observations'][:i+1], e['actions'][:i], config['max_moves'])
                for o in state['observations']:
                    for key in ['image', 'policy_image']:
                        o[key] = str((directory/o[key]).relative_to(dataset))
                output[scene['split']].append({'input': state, 'target': action,
                    'metadata': {'scene_id': scene['scene_id'], 'label_id': scene['label_id'],
                                 'episode_id': e['episode_id'], 'strategy': e['strategy']}})
    for split, records in output.items():
        jsonl(dataset/'policy'/f'{split}.jsonl', records)
    write(dataset/'policy/dataset_card.json', {'counts': {s: len(v) for s, v in output.items()},
          'config_sha256': sha(dataset/'dataset_config.json'),
          'semantics': 'Behavior cloning of recorded actions; shortest strategy uses privileged planning only. Failures retained.',
          'loader_contract': 'Feed only input to model. metadata is for grouping/audit, target for supervision.',
          'development_only': config['development_only']})
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    a = p.parse_args()
    build(a.data)
