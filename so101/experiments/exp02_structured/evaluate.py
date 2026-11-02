"""Frozen-model direction classification and live MuJoCo/OCR comparisons."""
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch

from core import (HERE, FIELDS, MAX_MOVES, ANGLES, DIRECTIONS, DELTAS, causal_state, grid_angle)
from model import DirectionActor, load_model
from train import classification
from common import read, rows, write, sha, complete, correct, metrics
from ocr_memory import InformationMemory
from prepare import verify_preserved


def rule_direction(method, angle, actions, rng, start):
    legal = [DIRECTIONS[i] for i, d in enumerate(DELTAS) if angle+d in ANGLES]
    if not legal:
        raise ValueError('No legal movement')
    if method == 'random':
        return rng.choice(legal)
    first = 'RIGHT_5' if method == 'right_first' else 'LEFT_5'
    if method == 'fixed_sweep':
        first = 'LEFT_5' if start-min(ANGLES) <= max(ANGLES)-start else 'RIGHT_5'
    direction = actions[-1]['action'] if actions else first
    return direction if direction in legal else legal[0]


def live_episode(dataset, output, case, method, checkpoint):
    """Simulator sees truth for rendering; actor only receives causal_state."""
    from simulator import Simulator
    config = read(dataset/'dataset_config.json')
    directory = dataset/'scenes'/case['scene_id']
    scene, truth = read(directory/'scene.json'), read(directory/'truth.json')
    observations, actions, errors = [], [], []
    memory = InformationMemory()
    rng = random.Random(case['seed'])
    actor = DirectionActor(checkpoint) if method == 'mlp' else None
    sim = Simulator(directory/'simulation', truth['settings'], truth['label'])
    angle = case['start_deg']
    termination, stop, budget = None, False, False
    try:
        arrival = sim.move(angle)
        for step in range(MAX_MOVES+1):
            o = sim.capture(output/'frames', f'obs_{step:02d}', config['thresholds'])
            o.update(observation_id=f'obs_{step:02d}', sequence_index=step,
                     command_relative_deg=angle, arrival=arrival)
            o['image'] = 'frames/'+o['image']
            o['policy_image'] = 'frames/'+o['policy_image']
            o['memory'] = memory.observe(o['observation_id'], step, o['readings'])
            observations.append(o)
            errors.extend(arrival['invalid_reasons'])
            if errors:
                termination = 'technical_error'
                break
            # The MLP has no STOP logit and never decides whether the budget was exhausted.
            if complete(o['memory']):
                termination, stop = 'memory_complete', True
                break
            if len(actions) == MAX_MOVES:
                termination, budget = 'budget_exhausted', True
                break
            state = causal_state(observations, actions)
            if method == 'mlp':
                action, diagnostic = actor.direction(state)
            else:
                action = rule_direction(method, angle, actions, rng, case['start_deg'])
                diagnostic = {}
            if action not in DIRECTIONS:
                raise ValueError('A movement policy returned a non-direction')
            command = angle+DELTAS[DIRECTIONS.index(action)]
            target_rad = sim.reference+math.radians(command)
            j, a = sim.model.joint('wrist_roll').id, sim.aids[4]
            low = max(sim.model.jnt_range[j,0], sim.model.actuator_ctrlrange[a,0])
            high = min(sim.model.jnt_range[j,1], sim.model.actuator_ctrlrange[a,1])
            if command not in scene['safe_angles_deg'] or not low <= target_rad <= high:
                termination = 'unsafe_direction_rejected'
                write(output/'rejected_action.json', {'action':action, 'command_deg':command, 'state':state})
                break  # No physical movement and no fictitious STOP.
            arrival = sim.move(command)
            actions.append({'action':action, 'from_observation':o['observation_id'],
                            'to_observation':f'obs_{step+1:02d}', 'decision_state':state,
                            'diagnostic':diagnostic, 'motion':arrival})
            angle = command
    finally:
        sim.close()
    final = observations[-1]['memory']
    success = correct(final, truth)
    episode = {'scene_id':case['scene_id'], 'split':case['split'], 'label_id':scene['label_id'],
               'start_deg':case['start_deg'], 'seed':case['seed'], 'method':method,
               'checkpoint_sha256':sha(checkpoint), 'observations':observations, 'actions':actions,
               'stop_reason':termination, 'stop_executed':stop, 'budget_exhausted':budget,
               'moves':len(actions), 'rotation_deg':sum(abs(a['motion']['after']['qpos_rad'][4]-
                 a['motion']['before']['qpos_rad'][4])*180/math.pi for a in actions),
               'success':success, 'wrong_stop':stop and not success, 'technical_errors':sorted(set(errors)),
               'posthoc':{'truth':{f:truth[f] for f in FIELDS}, 'ocr_misreads':[
                 {'step':i, 'field':f, 'read':o['readings'][f]['value'], 'truth':truth[f]}
                 for i,o in enumerate(observations) for f in FIELDS
                 if o['readings'][f]['value'] is not None and o['readings'][f]['value'] != truth[f]]}}
    # Post-hoc truth is attached only after every policy decision has been made.
    write(output/'episode.json', episode)
    return episode


def aggregate(episodes):
    valid = [e for e in episodes if not e['technical_errors']]
    return {'all_cases':metrics(episodes), 'valid_cases':metrics(valid),
            'technical_errors':sum(bool(e['technical_errors']) for e in episodes),
            'termination_counts':dict(Counter(e['stop_reason'] for e in episodes)),
            'ocr_misreads':sum(len(e['posthoc']['ocr_misreads']) for e in episodes),
            'failures':[{'scene_id':e['scene_id'], 'reason':e['stop_reason'], 'moves':e['moves'],
                         'wrong_stop':e['wrong_stop'], 'posthoc':e['posthoc']} for e in episodes if not e['success']]}


def evaluate(run):
    torch.set_num_threads(1)
    verify_preserved(run)
    experiment, selection = read(run/'experiment.json'), read(run/'selection.json')
    checkpoint = run/'best.pt'
    if sha(checkpoint) != selection['checkpoint_sha256'] or sha(run/'experiment.json') != selection['experiment_sha256']:
        raise ValueError('Selection/checkpoint changed after lock')
    dataset = Path(experiment['dataset'])
    source_hashes = {p.name:sha(p) for p in HERE.glob('*.py')}
    if source_hashes != read(run/'implementation_sha256.json'):
        raise ValueError('Implementation changed since preparation; use a new run before looking at test')
    plan_path = run/'evaluation_plan.json'
    cases = []
    for p in rows(dataset/'rollout_plan.jsonl'):
        if p['strategy'] != 'shortest':
            continue
        split = read(dataset/'scenes'/p['scene_id']/'scene.json')['split']
        if split in ['val','test']:
            cases.append({k:p[k] for k in ['scene_id','start_deg','seed']} | {'split':split})
    plan = {'cases':cases, 'methods':experiment['live_evaluation']['methods'],
            'checkpoint_sha256':sha(checkpoint), 'implementation_sha256':source_hashes}
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError('Frozen evaluation plan changed')
    write(plan_path, plan)
    result = {'development_only':True, 'selection_locked_before_live_evaluation':True,
              'test_labels':read(dataset/'splits.json')['test'], 'checkpoint_sha256':sha(checkpoint),
              'offline':{}, 'live':{}}
    for split in ['val','test']:
        if split == 'test':
            receipt = {'checkpoint_sha256':sha(checkpoint), 'selection_sha256':sha(run/'selection.json'),
                       'evaluation_plan_sha256':sha(plan_path)}
            gate = run/'test_evaluation_started.json'
            if gate.exists() and read(gate) != receipt:
                raise ValueError('Test was already started with a different selection')
            if not gate.exists():
                write(gate, receipt)
        offline = run/'evaluation'/f'{split}_classification.json'
        if not offline.exists():
            if sha(run/'data'/f'{split}.jsonl') != experiment['training_data_sha256'][split]:
                raise ValueError('Prepared split changed')
            write(offline, classification(load_model(checkpoint), rows(run/'data'/f'{split}.jsonl')))
        result['offline'][split] = read(offline)
        grouped = defaultdict(list)
        for case in [c for c in cases if c['split'] == split]:
            for method in plan['methods']:
                output = run/'evaluation'/split/case['scene_id']/method
                if (output/'episode.json').exists():
                    episode = read(output/'episode.json')  # Resume never repeats a completed test case.
                    if episode['checkpoint_sha256'] != sha(checkpoint):
                        raise ValueError('Completed case uses another checkpoint')
                else:
                    episode = live_episode(dataset, output, case, method, checkpoint)
                grouped[method].append(episode)
                print(f'{split} {case["scene_id"]} {method}: success={episode["success"]} moves={episode["moves"]} {episode["stop_reason"]}', flush=True)
        result['live'][split] = {k:aggregate(v) for k,v in grouped.items()}
        write(run/'evaluation_report.json', result)
    write(run/'preservation_after_evaluation.json', verify_preserved(run))
    write(run/'test_evaluation_complete.json', {'checkpoint_sha256':sha(checkpoint),
                                               'report_sha256':sha(run/'evaluation_report.json')})
    return result
