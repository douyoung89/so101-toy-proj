"""Offline feasibility checks use truth only for review and route supervision."""
import argparse
from collections import deque
from pathlib import Path

from common import DEFAULT_DATA, FIELDS, read, rows, write, scene_dirs, correct, complete
from ocr_memory import InformationMemory


def shortest_path(views, start, truth, budget=8):
    """BFS over actual OCR-memory states, including conflicts and confirmations."""
    import copy
    table = {v['command_relative_deg']: v for v in views if not v['invalid_reasons']}
    if start not in table:
        return None
    mem = InformationMemory()
    snapshot = mem.observe('obs_0', 0, table[start]['readings'])
    queue = deque([(start, mem, [start], snapshot)])
    seen = set()
    while queue:
        angle, memory, path, snapshot = queue.popleft()
        # Match the collector: a wrong but complete memory STOPs immediately.
        if complete(snapshot):
            if correct(snapshot, truth):
                return path
            continue
        if len(path)-1 == budget:
            continue
        key = (angle, tuple((memory.fields[f]['value'], memory.fields[f]['status'],
                            memory.fields[f]['pending'], memory.fields[f]['pending_count']) for f in FIELDS))
        if key in seen:
            continue
        seen.add(key)
        for nxt in [angle-5, angle+5]:
            if nxt not in table:
                continue
            cloned = copy.deepcopy(memory)
            state = cloned.observe(f'obs_{len(path)}', len(path), table[nxt]['readings'])
            queue.append((nxt, cloned, path+[nxt], state))
    return None


def check_scene(directory, config):
    views = rows(directory/'view_ocr.jsonl')
    truth = read(directory/'truth.json')
    reasons = []
    if [v['command_relative_deg'] for v in views] != config['angles_deg']:
        reasons.append('incomplete_or_duplicate_angle_table')
    safe = [v['command_relative_deg'] for v in views if not v['invalid_reasons']]
    # Every sweep view must be valid before it becomes a navigable scene.
    if len(safe) != len(config['angles_deg']):
        reasons.append('unsafe_view_or_sweep_transition')
    if any(not (directory/v[key]).exists() for v in views for key in ['image', 'policy_image']):
        reasons.append('missing_image')
    starts = []
    for start in config['start_angles_deg']:
        view = next((v for v in views if v['command_relative_deg'] == start), None)
        if view is None or view['invalid_reasons']:
            continue
        initial = InformationMemory().observe('initial', 0, view['readings'])
        if complete(initial):
            continue
        path = shortest_path(views, start, truth, config['max_moves'])
        if path is not None and len(path) > 1:
            starts.append({'start_deg': start, 'shortest_moves': len(path)-1, 'oracle_path_deg': path,
                           'first_direction': 'left' if path[1] < start else 'right'})
    if not starts:
        reasons.append('no_partial_start_solvable_within_budget')
    return {'accepted': not reasons, 'reasons': reasons, 'safe_angles_deg': safe,
            'eligible_starts': starts,
            'ocr_field_available': {f: any(v['readings'][f]['value'] == truth[f] and not v['invalid_reasons'] for v in views) for f in FIELDS},
            'physically_exposed_views': {f: sum(v['physical_exposure'][f]['all_samples_exposed'] for v in views) for f in FIELDS},
            'physical_exposure_note': 'Ray-sampled rectangle visibility is a geometric approximation, separately recorded from OCR.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    a = p.parse_args()
    config = read(a.data/'dataset_config.json')
    for directory in scene_dirs(a.data):
        write(directory/'review.json', check_scene(directory, config))


if __name__ == '__main__':
    main()
