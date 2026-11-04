"""Load direction examples and turn observation history into 77 model inputs."""
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
FIELDS = ('protein_g', 'sugar_g')
ACTIONS = ('LEFT_5', 'RIGHT_5')
DELTAS = (-5, 5)
ANGLES = tuple(range(-70, 41, 5))
MAX_MOVES = 8
REFERENCE_RAD = 0.75
INPUT_DIM = 77


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def memory_complete(memory):
    return all(memory['fields'][field]['status'] == 'confirmed' and
               memory['fields'][field]['value'] is not None for field in FIELDS)


def memory_correct(memory, truth):
    return memory_complete(memory) and all(memory['fields'][f]['value'] == truth[f] for f in FIELDS)


def legal_actions(angle):
    return [angle + delta in ANGLES for delta in DELTAS]


def encode_history(observations, actions):
    """Only information available now: no images, OCR values, truth or future views."""
    if len(observations) != len(actions) + 1 or len(actions) > MAX_MOVES:
        raise ValueError('Expected t+1 observations and t past actions')
    current = observations[-1]
    visited = [round(o['relative_actual_deg'], 3) for o in observations]
    relative = round(math.degrees(current['joint_state']['qpos_rad'][4] - REFERENCE_RAD), 3)
    features = [relative / 70, (MAX_MOVES - len(actions)) / MAX_MOVES]
    for field in FIELDS:
        memory = current['memory']['fields'][field]
        features.extend(float(memory['status'] == status) for status in ('unknown', 'confirmed', 'conflict'))
        features.extend([round(float(memory['confidence'] or 0), 4), min(memory['pending_count'], 2) / 2])
    features.extend(sum(abs(v - angle) < 0.2 for v in visited) / 9 for angle in ANGLES)
    for i in range(9):
        features.extend([visited[i] / 70, 1.0] if i < len(visited) else [0.0, 0.0])
    for i in range(MAX_MOVES):
        features.extend([float(actions[i] == a) for a in ACTIONS] + [1.0]
                        if i < len(actions) else [0.0, 0.0, 0.0])
    if len(features) != INPUT_DIM:
        raise ValueError('Feature vector must have 77 values')
    return features


class DirectionDataset(Dataset):
    """One item is (77 features, 2 teacher probabilities, 2 legal-action flags)."""
    def __init__(self, path):
        self.records = read_jsonl(path)
        if not self.records:
            raise ValueError(f'No training examples in {path}')
        self.x = torch.tensor([r['x'] for r in self.records], dtype=torch.float32)
        self.target = torch.tensor([r['target'] for r in self.records], dtype=torch.float32)
        self.legal = torch.tensor([r['legal'] for r in self.records], dtype=torch.bool)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.x[index], self.target[index], self.legal[index]
