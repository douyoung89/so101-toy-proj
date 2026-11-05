"""Evaluate classification and closed-loop reading; save flat CSV result tables."""
import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path

import torch

from dataset import (ROOT, ACTIONS, DELTAS, ANGLES, MAX_MOVES, REFERENCE_RAD, DirectionDataset,
                     legal_actions, memory_complete, memory_correct, read_json)
from model import load_model, predict_action
from simulation.memory import InformationMemory

POLICIES = ('mlp', 'left_first', 'right_first', 'random', 'fixed_sweep')


def rule_action(method, angle, actions, rng, start):
    allowed = [a for a, valid in zip(ACTIONS, legal_actions(angle)) if valid]
    if method == 'random':
        return rng.choice(allowed)
    first = 'RIGHT_5' if method == 'right_first' else 'LEFT_5'
    if method == 'fixed_sweep':
        first = 'LEFT_5' if start - min(ANGLES) <= max(ANGLES) - start else 'RIGHT_5'
    direction = actions[-1] if actions else first
    return direction if direction in allowed else allowed[0]


class ReplayEnvironment:
    """Optional fast check using recorded OCR, not a new camera/physics evaluation."""
    def __init__(self, scene):
        self.table = {v['angle']: v['readings'] for v in scene['views']}
        self.angle = 0

    def state(self):
        return {'qpos_rad': [0, 0, 0, 0, REFERENCE_RAD + math.radians(self.angle), 0]}

    def move(self, angle, on_frame=None):
        before = self.state()
        self.angle = angle
        return {'before': before, 'after': self.state(), 'invalid_reasons': []}

    def capture(self):
        return {'joint_state': self.state(), 'relative_actual_deg': float(self.angle),
                'readings': copy.deepcopy(self.table[self.angle]), 'ocr': []}

    def close(self):
        pass


def run_episode(scene, method, model, mode, seed, on_frame=None):
    """
    scene (dict): Settings and data for the scene to run.
    method (str): Policy used to choose the next action. (mlp)
    model (torch.nn.Module | None): Trained model used by the MLP policy.
    mode (str): Evaluation environment, either 'live' or 'replay'.
    seed (int): Random seed for reproducible action selection.
    on_frame (callable | None): Optional function called with rendered frames.
    """
    if mode == 'live': # using mujoco simulation 
        from simulation.environment import RobotEnvironment
        environment = RobotEnvironment(scene)
    else: # replay 
        environment = ReplayEnvironment(scene)
    
    memory = InformationMemory()
    observations, actions, errors = [], [], []
    angle = scene['start_deg']
    rng = random.Random(seed)
    rotation = 0.0
    reason = 'budget_exhausted'
    try:
        arrival = environment.move(angle, on_frame)
        
        for step in range(MAX_MOVES + 1):
            observation = environment.capture()
            observation.update(step=step, wrist_deg=angle, action=None)
            observation['memory'] = memory.observe(f'obs_{step}', step, observation['readings']) # cumulate memory, observation['readings'] = current observation 
            observations.append(observation)
            errors.extend(arrival['invalid_reasons'])
            if errors:
                reason = 'technical_error'
                break
            if memory_complete(observation['memory']):
                # if status is confirmed, then stop the action 
                observation['action'] = 'STOP'
                reason = 'memory_complete'
                break
            if len(actions) == MAX_MOVES:
                break
            
            # method 
            if method == 'mlp':
                action = predict_action(model, observations, actions)
            else:
                action = rule_action(method, angle, actions, rng, scene['start_deg'])
            
            target = angle + DELTAS[ACTIONS.index(action)]
            observation['action'] = action
            if target not in scene['safe_angles']:
                reason = 'unsafe_action_rejected'
                break
            
            arrival = environment.move(target, on_frame)
            rotation += abs(arrival['after']['qpos_rad'][4] - arrival['before']['qpos_rad'][4]) * 180 / math.pi
            actions.append(action)
            angle = target
    finally:
        environment.close()
    success = memory_correct(observations[-1]['memory'], scene['truth'])
    return {'mode': mode, 'split': scene['split'], 'scene_id': scene['scene_id'], 'policy': method,
            'success': success, 'wrong_stop': reason == 'memory_complete' and not success,
            'moves': len(actions), 'rotation_deg': rotation, 'stop_reason': reason,
            'technical_errors': sorted(set(errors)), 'observations': observations}


def write_csv(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


@torch.inference_mode()
def classify(model, data_dir):
    results = []
    for split in ('train', 'val', 'test'):
        data = DirectionDataset(data_dir / split / 'samples.jsonl')
        scores = model(data.x, data.legal)
        prediction = scores.argmax(dim=-1)
        accepted = data.target[torch.arange(len(data)), prediction] > 0
        loss = -(data.target * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
        results.append({'split': split, 'samples': len(data), 'accuracy': float(accepted.float().mean()),
                        'loss': float(loss), 'left_targets': int((data.target[:, 0] > 0).sum()),
                        'right_targets': int((data.target[:, 1] > 0).sum()),
                        'always_right_accuracy': float((data.target[:, 1] > 0).float().mean())})
    return results


def summarize(episodes):
    results = []
    for split in ('val', 'test'):
        for policy in POLICIES:
            group = [e for e in episodes if e['split'] == split and e['policy'] == policy]
            n = len(group)
            if not n:
                continue
            results.append({'mode': group[0]['mode'], 'split': split, 'policy': policy, 'episodes': n,
                            'successes': sum(e['success'] for e in group),
                            'success_rate': sum(e['success'] for e in group) / n,
                            'wrong_stops': sum(e['wrong_stop'] for e in group),
                            'mean_moves': sum(e['moves'] for e in group) / n,
                            'mean_rotation_deg': sum(e['rotation_deg'] for e in group) / n,
                            'budget_exhausted': sum(e['stop_reason'] == 'budget_exhausted' for e in group),
                            'technical_errors': sum(bool(e['technical_errors']) for e in group)})
    return results


def write_report(results_dir, classification, metrics, config):
    training = read_json(results_dir / 'training_summary.json')
    text = ['# SO-101 local experiment', '',
            f"Seed: {config['seed']}; selected epoch: {training['selected_epoch']}; parameters: {training['parameters']}.", '',
            'Existing local raw observations and episodes were reorganized; teacher labels were regenerated.',
            'The model was initialized and trained again. Test data was not used by train.py.',
            'The split remains development-only: 12 train labels, 2 validation labels, 1 test label.', '',
            '| Split | Samples | Direction accuracy | Always-right accuracy |', '|---|---:|---:|---:|']
    for row in classification:
        text.append(f"| {row['split']} | {row['samples']} | {row['accuracy']:.1%} | {row['always_right_accuracy']:.1%} |")
    text += ['', '| Mode | Split | Policy | Success | Mean moves | Wrong STOP | Technical errors |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in metrics:
        text.append(f"| {row['mode']} | {row['split']} | {row['policy']} | {row['successes']}/{row['episodes']} | {row['mean_moves']:.2f} | {row['wrong_stops']} | {row['technical_errors']} |")
    text += ['', 'live = fresh MuJoCo movement, rendering and OCR after each action.',
             'replay = recorded OCR table transitions; these are not fresh physical observations.',
             'All cases, including failures and technical errors, remain in the CSV denominator.',
             'Rotation excludes the move to the initial pose. STOP is a memory rule, not an MLP output.',
             'Validation/test teacher labels favor RIGHT_5; check the always-right column before interpreting accuracy.',
             'One test scene does not establish generalization or superiority over simple rules.', '']
    (results_dir / 'report.md').write_text('\n'.join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('live', 'replay'), default='live')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    config = read_json(ROOT / 'config.json')
    torch.set_num_threads(1)
    model = load_model(args.results_dir / 'best_model.pt')
    classification = classify(model, args.data_dir)
    write_csv(args.results_dir / 'classification.csv', classification)
    episodes = []
    scenes = [s for s in read_json(args.data_dir / 'scenes.json') if s['split'] in ('val', 'test')]
    with (args.results_dir / 'evaluation.jsonl').open('w') as stream:
        for index, scene in enumerate(scenes):
            for policy in POLICIES:
                episode = run_episode(scene, policy, model, args.mode, config['seed'] + index)
                episodes.append(episode)
                stream.write(json.dumps(episode) + '\n')
                stream.flush()
                print(f"{args.mode} {scene['split']} {scene['scene_id']} {policy}: success={episode['success']} moves={episode['moves']} reason={episode['stop_reason']}", flush=True)
    rows = [{k: v for k, v in e.items() if k != 'observations'} for e in episodes]
    write_csv(args.results_dir / 'episodes.csv', rows)
    metrics = summarize(episodes)
    write_csv(args.results_dir / 'metrics.csv', metrics)
    write_report(args.results_dir, classification, metrics, config)
    print(f'Saved CSV tables and report in {args.results_dir}', flush=True)
    if any(e['technical_errors'] or e['stop_reason'] == 'unsafe_action_rejected' for e in episodes):
        raise SystemExit('Evaluation saved with technical failures; inspect episodes.csv')


if __name__ == '__main__':
    main()
