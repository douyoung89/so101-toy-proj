"""Causal numeric features and privileged offline direction supervision."""
import math
from collections import deque
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SO101 = HERE.parents[1]
sys.path.insert(0, str(SO101))
from common import FIELDS, complete, correct
from ocr_memory import InformationMemory

ANGLES = tuple(range(-70, 41, 5))
MAX_MOVES = 8
REFERENCE_RAD = .75  # Fixed wrist calibration, not per-scene hidden geometry.
DIRECTIONS = ('LEFT_5', 'RIGHT_5')
DELTAS = (-5, 5)
STATUSES = ('unknown', 'confirmed', 'conflict')


def feature_names():
    names = ['current_relative_deg/70', 'remaining_moves/8']
    for f in FIELDS:
        names += [f'{f}:{s}' for s in STATUSES] + [f'{f}:confidence', f'{f}:pending_count/2']
    names += [f'visits_at_{a}/9' for a in ANGLES]
    names += [f'visited_{i}:{k}' for i in range(9) for k in ['relative_deg/70', 'present']]
    names += [f'past_action_{i}:{k}' for i in range(8) for k in ['left', 'right', 'present']]
    return names


def causal_state(observations, actions):
    """Allowlist only. Never inspect current target, raw text, truth, images or strategy."""
    if not observations or len(observations) != len(actions)+1 or len(actions) > MAX_MOVES:
        raise ValueError('History must contain t+1 observations and t past actions')
    history = [a if isinstance(a, str) else a['action'] for a in actions]
    if any(a not in DIRECTIONS for a in history):
        raise ValueError('History contains a non-movement action')
    current = observations[-1]
    return {'current_wrist_rad': float(current['joint_state']['qpos_rad'][4]),
            'memory_fields': {f: {k: current['memory']['fields'][f][k]
                                 for k in ['status', 'confidence', 'pending_count']} for f in FIELDS},
            'visited_angles_deg': [float(o['relative_actual_deg']) for o in observations],
            'past_actions': history, 'remaining_moves': MAX_MOVES-len(history)}


def features(state):
    visited = [round(a, 3) for a in state['visited_angles_deg']]
    past = state['past_actions']
    if not 1 <= len(visited) <= 9 or len(past)+1 != len(visited):
        raise ValueError('Invalid causal history lengths')
    if state['remaining_moves'] != MAX_MOVES-len(past):
        raise ValueError('Budget/history mismatch')
    relative = round(math.degrees(state['current_wrist_rad']-REFERENCE_RAD), 3)
    if abs(relative-visited[-1]) > .01:
        raise ValueError('Wrist and observation angle disagree')
    out = [relative/70, state['remaining_moves']/8]
    for f in FIELDS:
        m = state['memory_fields'][f]
        if m['status'] not in STATUSES:
            raise ValueError('Unknown memory status')
        out.extend(float(m['status'] == s) for s in STATUSES)
        out += [round(float(m['confidence'] or 0), 4), min(m['pending_count'], 2)/2]
    # Occupancy counts and chronological history both derive only from past visits.
    for a in ANGLES:
        out.append(sum(abs(v-a) < .2 for v in visited)/9)
    for i in range(9):
        out.extend([visited[i]/70, 1.] if i < len(visited) else [0., 0.])
    for i in range(8):
        out.extend([float(past[i] == DIRECTIONS[0]), float(past[i] == DIRECTIONS[1]), 1.]
                   if i < len(past) else [0., 0., 0.])
    if len(out) != len(feature_names()) or not all(math.isfinite(v) for v in out):
        raise ValueError('Invalid features')
    return out


def grid_angle(relative):
    nearest = min(ANGLES, key=lambda a: abs(a-relative))
    if abs(nearest-relative) > .2:
        raise ValueError('Wrist did not reach a known 5 degree grid point')
    return nearest


def legal_mask(relative):
    a = grid_angle(relative)
    return [a+d in ANGLES for d in DELTAS]


def compact(memory):
    return tuple((memory['fields'][f]['value'], memory['fields'][f]['status'],
                  memory['fields'][f]['pending'], memory['fields'][f]['pending_count']) for f in FIELDS)


def transition(key, readings):
    """Execute the existing conflict/confirmation rule, with a fresh observation ID."""
    mem = InformationMemory()
    for f, (value, status, pending, count) in zip(FIELDS, key):
        mem.fields[f].update(value=value, status=status, pending=pending, pending_count=count)
    return compact(mem.observe('new_teacher_observation', 0, readings))


def terminal(key, truth):
    full = all(value is not None and status == 'confirmed' for value, status, _, _ in key)
    if not full:
        return 'incomplete'
    return 'correct' if all(key[i][0] == truth[f] for i, f in enumerate(FIELDS)) else 'wrong'


def shortest_after_first(table, current, initial, truth, budget, first):
    next_angle = current+DELTAS[first]
    if budget <= 0 or next_angle not in table:
        return None
    next_key = transition(initial, table[next_angle]['readings'])
    queue = deque([(next_angle, next_key, 1)])
    seen = set()
    while queue:
        angle, key, moves = queue.popleft()
        status = terminal(key, truth)
        if status == 'correct':
            return moves
        # STOP gate is mandatory: no hypothetical recovery after a wrong complete memory.
        if status == 'wrong' or moves == budget or (angle, key) in seen:
            continue
        seen.add((angle, key))
        for delta in DELTAS:
            target = angle+delta
            if target in table:
                queue.append((target, transition(key, table[target]['readings']), moves+1))
    return None


def direction_teacher(views, current, memory, remaining, truth):
    key = compact(memory)
    status = terminal(key, truth)
    if status != 'incomplete':
        return {'status': 'complete_'+status}
    if remaining <= 0:
        return {'status': 'budget_exhausted'}
    table = {v['command_relative_deg']: v for v in views if not v['invalid_reasons']}
    current = grid_angle(current)
    distances = [shortest_after_first(table, current, key, truth, remaining, a) for a in range(2)]
    possible = [d for d in distances if d is not None]
    partial_wrong = any(value is not None and status == 'confirmed' and value != truth[f]
                        for f, (value, status, _, _) in zip(FIELDS, key))
    if not possible:
        return {'status': 'unsolvable', 'distances': distances, 'partial_wrong': partial_wrong}
    best = min(possible)
    optimal = [d == best for d in distances]
    target = [float(v)/sum(optimal) for v in optimal]
    return {'status': 'tie' if sum(optimal) == 2 else 'unique', 'target': target,
            'optimal': optimal, 'distances': distances, 'shortest_moves': best,
            'partial_wrong': partial_wrong}
