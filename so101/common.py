"""Shared storage and causal episode semantics; no simulator dependency."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DATA = HERE / 'data/partial_v1'
FIELDS = ('protein_g', 'sugar_g')
ACTIONS = {'LEFT_5': -5, 'RIGHT_5': 5, 'STOP': 0}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete(memory):
    return all(memory['fields'][f]['value'] is not None and
               memory['fields'][f]['status'] == 'confirmed' for f in FIELDS)


def correct(memory, truth):
    return complete(memory) and all(memory['fields'][f]['value'] == truth[f] for f in FIELDS)


def scene_dirs(dataset):
    return sorted(p.parent for p in (Path(dataset) / 'scenes').glob('*/complete.json'))


def policy_input(observations, actions, budget=8):
    """Explicit allowlist: history through t only, without identifiers encoding truth."""
    return {'observations': [{k: o[k] for k in (
                'image', 'policy_image', 'joint_state', 'relative_actual_deg',
                'ocr', 'readings', 'memory')} for o in observations],
            'current_wrist_rad': observations[-1]['joint_state']['qpos_rad'][4],
            'action_history': [a['action'] for a in actions],
            'remaining_moves': budget - len(actions)}


def metrics(episodes):
    n = len(episodes)
    return {'episodes': n, 'exact_acquisition_rate': sum(e['success'] for e in episodes)/n if n else None,
            'wrong_stop_rate': sum(e['wrong_stop'] for e in episodes)/n if n else None,
            'mean_moves': sum(e['moves'] for e in episodes)/n if n else None,
            'mean_rotation_deg': sum(e['rotation_deg'] for e in episodes)/n if n else None}
