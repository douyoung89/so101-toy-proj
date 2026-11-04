"""Build supervised direction examples from the imported OCR tables and episodes."""
from collections import Counter, deque
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import FIELDS, DELTAS, MAX_MOVES, encode_history, legal_actions, memory_complete, read_json
from simulation.memory import InformationMemory


def memory_key(memory):
    return tuple((memory['fields'][f]['value'], memory['fields'][f]['status'],
                  memory['fields'][f]['pending'], memory['fields'][f]['pending_count']) for f in FIELDS)


def next_memory(key, readings):
    memory = InformationMemory()
    for field, (value, status, pending, count) in zip(FIELDS, key):
        memory.fields[field].update(value=value, status=status, pending=pending, pending_count=count)
    return memory_key(memory.observe('teacher_step', 0, readings))


def status(key, truth):
    if not all(value is not None and state == 'confirmed' for value, state, _, _ in key):
        return 'incomplete'
    return 'correct' if all(key[i][0] == truth[f] for i, f in enumerate(FIELDS)) else 'wrong'


def distance_after_first(table, angle, memory, truth, budget, delta):
    """BFS counts actual memory updates, including OCR conflicts and reconfirmations."""
    if angle + delta not in table:
        return None
    first = next_memory(memory, table[angle + delta])
    queue = deque([(angle + delta, first, 1)])
    visited = set()
    while queue:
        current, key, moves = queue.popleft()
        outcome = status(key, truth)
        if outcome == 'correct':
            return moves
        if outcome == 'wrong' or moves == budget or (current, key) in visited:
            continue
        visited.add((current, key))
        for step in DELTAS:
            if current + step in table:
                updated = next_memory(key, table[current + step])
                queue.append((current + step, updated, moves + 1))
    return None


def teacher_target(scene, observation, remaining):
    key = memory_key(observation['memory'])
    outcome = status(key, scene['truth'])
    if outcome != 'incomplete':
        return None, 'complete_' + outcome
    if remaining == 0:
        return None, 'budget_exhausted'
    table = {v['angle']: v['readings'] for v in scene['views'] if not v['invalid_reasons']}
    angle = observation['wrist_deg']
    distances = [distance_after_first(table, angle, key, scene['truth'], remaining, delta) for delta in DELTAS]
    possible = [d for d in distances if d is not None]
    if not possible:
        return None, 'unsolvable'
    best = min(possible)
    optimal = [d == best for d in distances]
    return [float(v) / sum(optimal) for v in optimal], 'tie' if sum(optimal) == 2 else 'unique'


def prepare():
    scenes = {s['scene_id']: s for s in read_json(ROOT / 'data/scenes.json')}
    label_sets = {split: {s['label_id'] for s in scenes.values() if s['split'] == split}
                  for split in ('train', 'val', 'test')}
    if any(label_sets[a] & label_sets[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('A label appears in more than one split')
    summary = {}
    for split in ('train', 'val', 'test'):
        records, seen, counts, excluded = [], {}, Counter(), []
        for path in sorted((ROOT / 'data' / split).glob('episode_*.json')):
            episode = read_json(path)
            scene = scenes[episode['scene_id']]
            if episode['technical_errors']:
                excluded.append({'episode': episode['episode_id'], 'reason': 'technical_error'})
                continue
            observations, actions = [], []
            memory = InformationMemory()
            table = {v['angle']: v['readings'] for v in scene['views']}
            for i, original in enumerate(episode['observations']):
                current = dict(original)
                current['memory'] = memory.observe(f'obs_{i}', i, current['readings'])
                observations.append(current)
                # Keep the teacher's table-transition assumption explicit on train/val only.
                if split in ('train', 'val'):
                    for field in FIELDS:
                        a, b = current['readings'][field], table[current['wrist_deg']][field]
                        if (a['value'], a['ambiguous']) != (b['value'], b['ambiguous']):
                            raise ValueError('Table/live readings disagree: this BFS teacher needs a different dataset')
                target, reason = teacher_target(scene, current, MAX_MOVES - len(actions))
                counts[reason] += 1
                if target is None:
                    excluded.append({'episode': episode['episode_id'], 'step': i, 'reason': reason})
                else:
                    x = encode_history(observations, actions)
                    key = (scene['scene_id'], tuple(x), tuple(target))
                    source = {'episode': episode['episode_id'], 'step': i}
                    if key in seen:
                        records[seen[key]]['sources'].append(source)
                        counts['merged_duplicates'] += 1
                    else:
                        seen[key] = len(records)
                        records.append({'x': x, 'target': target, 'legal': legal_actions(current['wrist_deg']),
                                        'scene_id': scene['scene_id'], 'label_id': scene['label_id'], 'sources': [source]})
                if current['action'] in ('LEFT_5', 'RIGHT_5'):
                    actions.append(current['action'])
        (ROOT / 'data' / split / 'samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        summary[split] = {'samples': len(records), 'labels': len(label_sets[split]),
                          'states': dict(counts), 'excluded': excluded}
        print(f'{split}: {len(records)} direction samples, {len(label_sets[split])} labels', flush=True)
    (ROOT / 'data/summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    prepare()
