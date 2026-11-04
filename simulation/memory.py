"""Conservative English nutrition parsing and per-object observation memory."""
import re
from copy import deepcopy

FIELDS = {'protein_g': ('PROTEIN', 40), 'sugar_g': ('SUGAR', 20)}
PATTERN = re.compile(r'(PROTEIN|SUGAR)\s*:?\s*(\d{1,2})\s*G?', re.I)


def parse(detections):
    """Join only spatially aligned detections, never unrelated text rows."""
    lines = []
    for detection in sorted(detections, key=lambda d: sum(p[1] for p in d['bbox'])/4):
        box = detection['bbox']
        x = min(p[0] for p in box)
        y = sum(p[1] for p in box)/4
        height = max(p[1] for p in box)-min(p[1] for p in box)
        line = next((line for line in lines if abs(line['y']-y) < .45*min(line['height'], height)), None)
        if line is None:
            line = {'y': y, 'height': height, 'parts': []}
            lines.append(line)
        line['parts'].append((x, detection))
    candidates = {field: [] for field in FIELDS}
    for line in lines:
        parts = [d for _, d in sorted(line['parts'], key=lambda p: p[0])]
        # Try each detection as well as a spatially joined row.
        groups = [[part] for part in parts]
        if len(parts) > 1:
            groups.append(parts)
        for group in groups:
            text = ' '.join(d['text'] for d in group).strip()
            match = PATTERN.fullmatch(text)
            if not match:
                continue
            key, number = match.group(1).upper(), int(match.group(2))
            field = next(f for f, (name, _) in FIELDS.items() if name == key)
            if not 0 <= number <= FIELDS[field][1]:
                continue
            candidate = {'value': number, 'confidence': min(float(d['confidence']) for d in group),
                         'text': text, 'boxes': [d['bbox'] for d in group]}
            if candidate not in candidates[field]:
                candidates[field].append(candidate)
    return candidates


def accepted(candidates, thresholds):
    result = {}
    for field in FIELDS:
        eligible = [c for c in candidates[field] if c['confidence'] >= thresholds[field]]
        values = {c['value'] for c in eligible}
        result[field] = {'value': next(iter(values)) if len(values) == 1 else None,
                         'confidence': max((c['confidence'] for c in eligible), default=None),
                         'ambiguous': len(values) > 1, 'candidates': eligible}
    return result


class InformationMemory:
    def __init__(self, protein_min=20, sugar_max=5):
        self.protein_min = protein_min
        self.sugar_max = sugar_max
        self.fields = {f: {'value': None, 'confidence': None, 'status': 'unknown',
                           'last_observation': None, 'evidence': [], 'pending': None,
                           'pending_count': 0} for f in FIELDS}
        self.seen = set()

    def observe(self, observation_id, sequence_index, readings):
        if observation_id in self.seen:
            raise ValueError('Duplicate observation cannot count as new evidence')
        self.seen.add(observation_id)
        for field, reading in readings.items():
            state = self.fields[field]
            evidence = {'observation_id': observation_id, 'sequence_index': sequence_index,
                        'value': reading['value'], 'confidence': reading['confidence'],
                        'ambiguous': reading['ambiguous']}
            state['evidence'].append(evidence)
            value = reading['value']
            if reading['ambiguous']:
                state.update(value=None, confidence=None, status='conflict', pending=None, pending_count=0,
                             last_observation=deepcopy(evidence))
            elif value is None:
                continue  # Missing text must not erase previous evidence or imply zero.
            elif state['status'] == 'conflict':
                count = state['pending_count']+1 if state['pending'] == value else 1
                state.update(pending=value, pending_count=count, last_observation=deepcopy(evidence))
                if count >= 2:
                    state.update(value=value, confidence=reading['confidence'], status='confirmed',
                                 pending=None, pending_count=0)
            elif state['value'] is not None and state['value'] != value:
                # Two later matching observations are required after detecting a conflict.
                state.update(value=None, confidence=None, status='conflict', pending=None, pending_count=0,
                             last_observation=deepcopy(evidence))
            else:
                state.update(value=value, confidence=reading['confidence'], status='confirmed',
                             last_observation=deepcopy(evidence))
        return self.snapshot()

    def snapshot(self):
        protein, sugar = (self.fields[f]['value'] for f in FIELDS)
        if (protein is not None and protein < self.protein_min) or (sugar is not None and sugar > self.sugar_max):
            decision = 'fail'
        elif protein is not None and sugar is not None:
            decision = 'pass'
        else:
            decision = 'need_more'
        return {'decision': decision, 'fields': deepcopy(self.fields)}
