"""Audit integrity, causal replay, splits, feasibility and retained failures."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from common import (DEFAULT_DATA, FIELDS, ACTIONS, read, rows, write, sha,
                    scene_dirs, complete, correct, metrics)
from ocr_memory import InformationMemory, parse, accepted


def audit_episode(directory, episode, config):
    errors = []
    truth = read(directory/'truth.json')
    obs, actions = episode['observations'], episode['actions']
    if not obs or len(obs) != len(actions)+1 or len(actions) > config['max_moves']:
        return ['invalid_observation_action_counts']
    memory = InformationMemory()
    for i, o in enumerate(obs):
        if o['sequence_index'] != i or o['observation_id'] != f'obs_{i:02d}':
            errors.append('observation_order')
        for key in ['image', 'policy_image']:
            path = directory/o[key]
            if not path.exists():
                errors.append('missing_'+key)
            else:
                if sha(path) != o.get(key+'_sha256'):
                    errors.append(key+'_hash')
                from PIL import Image
                with Image.open(path) as im:
                    if list(im.size) != config['ocr_size' if key == 'image' else 'policy_size']:
                        errors.append('image_size')
        readings = accepted(parse(o['ocr']), config['thresholds'])
        if readings != o['readings']:
            errors.append('ocr_reading_mismatch')
        replay = memory.observe(o['observation_id'], i, readings)
        if replay != o['memory']:
            errors.append('noncausal_or_incorrect_memory')
        if o['arrival']['invalid_reasons']:
            errors.extend(o['arrival']['invalid_reasons'])
        if i < len(actions):
            a = actions[i]
            if a['from_observation'] != o['observation_id'] or a['to_observation'] != obs[i+1]['observation_id']:
                errors.append('action_observation_link')
            if a['action'] not in ['LEFT_5', 'RIGHT_5']:
                errors.append('invalid_motion_action')
                continue
            expected = o['command_relative_deg']+ACTIONS[a['action']]
            motion = a['motion']
            if expected != obs[i+1]['command_relative_deg'] or motion['command_relative_deg'] != expected:
                errors.append('command_angle_link')
            if motion != obs[i+1]['arrival'] or motion['before'] != o['joint_state'] or motion['after'] != obs[i+1]['joint_state']:
                errors.append('joint_state_link')
            if abs(motion['actual_relative_deg']-expected) > .2:
                errors.append('arrival_angle_error')
            if expected not in read(directory/'scene.json')['safe_angles_deg']:
                errors.append('unsafe_action')
    if correct(obs[-1]['memory'], truth) != episode['success']:
        errors.append('success_annotation')
    if episode['config_sha256'] != sha(directory.parent.parent/'dataset_config.json'):
        errors.append('episode_config_hash')
    expected_wrong = bool(episode['terminal_action'] and episode['terminal_action']['action'] == 'STOP' and not episode['success'])
    if expected_wrong != episode['wrong_stop']:
        errors.append('wrong_stop_annotation')
    return sorted(set(errors))


def audit(dataset, reproduce=False):
    config = read(dataset/'dataset_config.json')
    splits = read(dataset/'splits.json')
    split_errors = [label for s, labels in splits.items() for t, other in splits.items()
                    if s < t for label in set(labels) & set(other)]
    errors, episodes, excluded, feasibility = {}, [], [], []
    labels_seen = defaultdict(set)
    for directory in scene_dirs(dataset):
        scene = read(directory/'scene.json')
        labels_seen[scene['label_id']].add(scene['split'])
        issues = []
        if scene['label_id'] not in splits.get(scene['split'], []):
            issues.append('split_membership')
        completion = read(directory/'complete.json')
        if completion['config_sha256'] != sha(dataset/'dataset_config.json'):
            issues.append('config_hash')
        for p, h in completion['hashes'].items():
            if not (directory/p).exists() or sha(directory/p) != h:
                issues.append('file_hash:'+p)
        if reproduce:
            from simulator import Simulator
            import numpy as np
            truth = read(directory/'truth.json')
            sim = Simulator(directory/'simulation', truth['settings'], truth['label'], render=False)
            try:
                sim.move(0)
                for view in rows(directory/'view_ocr.jsonl'):
                    motion = sim.move(view['command_relative_deg'])
                    if motion['invalid_reasons'] or not np.allclose(motion['after']['qpos_rad'], view['joint_state']['qpos_rad'], atol=1e-6):
                        issues.append('scene_physics_reproduction')
            finally:
                sim.close()
        if issues:
            errors[directory.name] = issues
        check = read(directory/'check.json')
        feasibility.extend(check['eligible_starts'])
        for path in sorted((directory/'episodes').glob('*.json')):
            e = read(path)
            problems = audit_episode(directory, e, config)
            if problems or issues:
                excluded.append(str(path.relative_to(dataset)))
                errors[str(path.relative_to(dataset))] = problems + issues
            else:
                episodes.append(e)  # OCR mistakes and exploration failures are intentionally retained.
    split_errors += [label for label, ss in labels_seen.items() if len(ss) > 1]
    plan = rows(dataset/'rollout_plan.jsonl') if (dataset/'rollout_plan.jsonl').exists() else []
    missing_episodes = [p['episode_id'] for p in plan if not (
        dataset/'scenes'/p['scene_id']/'episodes'/f'{p["episode_id"]}.json').exists()]
    report = {'scenes': len(scene_dirs(dataset)), 'planned_scenes': config['planned_scenes'],
              'complete_collection': len(scene_dirs(dataset)) == config['planned_scenes'],
              'planned_episodes': len(plan), 'collected_episodes': len(episodes)+len(excluded),
              'missing_episodes': missing_episodes,
              'complete_episode_collection': bool(plan) and not missing_episodes,
              'split_errors': sorted(set(split_errors)), 'technical_errors': errors,
              'excluded_episodes': excluded, 'retained': metrics(episodes),
              'ocr_misreads': sum(len(e['posthoc']['ocr_misreads']) for e in episodes),
              'shortest_moves_histogram': dict(Counter(s['shortest_moves'] for s in feasibility)),
              'first_direction_counts': dict(Counter(s['first_direction'] for s in feasibility)),
              'solvable_scene_rate': sum(bool(read(d/'check.json')['eligible_starts']) for d in scene_dirs(dataset))/len(scene_dirs(dataset)) if scene_dirs(dataset) else None,
              'candidate_acceptance': candidate_acceptance(dataset),
              'physics_reproduction_checked': reproduce, 'image_reproduction_checked': False,
              'note': 'Physics replay does not certify pixel identity; no cross-episode image reuse is enabled.'}
    write(dataset/'audit.json', report)
    return report


def candidate_acceptance(dataset):
    attempts = [r for p in (dataset/'scenes').glob('*/attempts.jsonl') for r in rows(p)]
    return {'candidates': len(attempts), 'accepted': sum(r['accepted'] for r in attempts),
            'rate': sum(r['accepted'] for r in attempts)/len(attempts) if attempts else None,
            'rejection_reasons': dict(Counter(reason for r in attempts if not r['accepted'] for reason in r['reasons']))}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--reproduce', action='store_true')
    a = p.parse_args()
    r = audit(a.data, a.reproduce)
    print(r)
    if r['technical_errors'] or r['split_errors'] or not r['complete_collection']:
        raise SystemExit(1)
