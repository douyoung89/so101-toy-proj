"""Live sequential capture -> OCR -> memory -> action -> MuJoCo move."""
import argparse
from pathlib import Path
import random

from common import (DEFAULT_DATA, ACTIONS, read, rows, write, sha, complete, correct,
                    policy_input, FIELDS)
from ocr_memory import InformationMemory


def choose(strategy, angle, step, safe, rng, route=None, history=None):
    options = [a for a in ['LEFT_5', 'RIGHT_5'] if angle+ACTIONS[a] in safe]
    if not options:
        return 'STOP'
    if strategy == 'random':
        return rng.choice(options)
    if strategy == 'shortest' and route and step+1 < len(route):
        delta = route[step+1]-angle
        if delta in [-5, 5]:
            return 'LEFT_5' if delta < 0 else 'RIGHT_5'
    # Maintain direction until a boundary then reverse; never alternate at the boundary.
    history = history or []
    desired = history[-1]['action'] if history else ('RIGHT_5' if strategy == 'right_first' else 'LEFT_5')
    return desired if desired in options else options[0]


def run_episode(directory, config, plan, policy=None, destination=None):
    from simulator import Simulator
    truth = read(directory/'truth.json')
    scene = read(directory/'scene.json')
    episode_id = plan['episode_id']
    target = destination or directory/'episodes'/f'{episode_id}.json'
    memory = InformationMemory()
    rng = random.Random(plan['seed'])
    observations, actions = [], []
    sim = Simulator(directory/'simulation', truth['settings'], truth['label'])
    angle = plan['start_deg']
    stop_reason = 'budget_exhausted'
    technical = []
    terminal_action = None
    try:
        arrival = sim.move(angle)
        technical.extend(arrival['invalid_reasons'])
        for step in range(config['max_moves']+1):
            # Each acquisition is unique, including revisits. Do not duplicate cached evidence.
            obs = sim.capture(directory/'views', f'{episode_id}_obs_{step:02d}', config['thresholds'])
            obs.update(observation_id=f'obs_{step:02d}', sequence_index=step,
                       image='views/'+obs['image'], policy_image='views/'+obs['policy_image'],
                       command_relative_deg=angle, arrival=arrival)
            obs['memory'] = memory.observe(obs['observation_id'], step, obs['readings'])
            observations.append(obs)
            if technical:
                stop_reason = 'technical_failure'
                break
            state = policy_input(observations, actions, config['max_moves'])
            if policy is not None:
                action = policy(state)
                if action not in ACTIONS:
                    raise ValueError(f'Invalid policy action: {action}')
            elif complete(obs['memory']):
                action = 'STOP'
            elif step == config['max_moves']:
                break
            else:
                action = choose(plan['strategy'], angle, step, scene['safe_angles_deg'], rng,
                                plan.get('oracle_path_deg'), actions)
            if action == 'STOP':
                terminal_action = {'action': 'STOP', 'at_observation': obs['observation_id']}
                stop_reason = 'policy_stop' if policy is not None else 'memory_complete' if complete(obs['memory']) else 'no_safe_action'
                break
            if step == config['max_moves']:
                break
            commanded = angle+ACTIONS[action]
            if commanded not in scene['safe_angles_deg']:
                stop_reason = 'unsafe_action_rejected'
                terminal_action = {'action': action, 'at_observation': obs['observation_id'], 'executed': False}
                break
            arrival = sim.move(commanded)
            actions.append({'action': action, 'from_observation': obs['observation_id'],
                            'to_observation': f'obs_{step+1:02d}', 'motion': arrival})
            technical.extend(arrival['invalid_reasons'])
            angle = commanded
    finally:
        sim.close()
    final = observations[-1]['memory']
    success = correct(final, truth)
    episode = {'episode_id': episode_id, 'scene_id': scene['scene_id'],
               'strategy': plan['strategy'], 'seed': plan['seed'], 'start_deg': plan['start_deg'],
               'observations': observations, 'actions': actions, 'terminal_action': terminal_action,
               'stop_reason': stop_reason, 'moves': len(actions),
               'rotation_deg': sum(abs(a['motion']['after']['qpos_rad'][4]-a['motion']['before']['qpos_rad'][4])*180/3.141592653589793 for a in actions),
               'commanded_rotation_deg': 5*len(actions), 'success': success,
               'wrong_stop': bool(terminal_action and terminal_action['action'] == 'STOP' and not success),
               'technical_errors': sorted(set(technical)),
               'posthoc': {'ocr_misreads': [
                   {'observation_id': o['observation_id'], 'field': f, 'read': o['readings'][f]['value'], 'truth': truth[f]}
                   for o in observations for f in FIELDS
                   if o['readings'][f]['value'] is not None and o['readings'][f]['value'] != truth[f]]},
               'image_reuse': False, 'config_sha256': sha(directory.parent.parent/'dataset_config.json')}
    write(target, episode)
    return episode


def collect(dataset):
    from tqdm import tqdm
    config = read(dataset/'dataset_config.json')
    for plan in tqdm(rows(dataset/'rollout_plan.jsonl'), desc='Episodes', unit='episode'):
        directory = dataset/'scenes'/plan['scene_id']
        target = directory/'episodes'/f'{plan["episode_id"]}.json'
        if target.exists():
            old = read(target)
            if old['config_sha256'] != sha(dataset/'dataset_config.json'):
                raise ValueError(f'Config changed: {target}')
            if all((directory/o['image']).exists() for o in old['observations']):
                continue
        run_episode(directory, config, plan)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    a = p.parse_args()
    collect(a.data)
