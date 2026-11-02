"""Read-only audit of pilot files and relabeling from every recorded causal state."""
from collections import Counter, defaultdict
import importlib.metadata
from pathlib import Path
import platform
import shutil

from core import SO101, HERE, FIELDS, MAX_MOVES, DIRECTIONS, causal_state, features, feature_names, direction_teacher
from common import read, rows, write, jsonl, sha, scene_dirs
from audit_partial import audit_episode
from ocr_memory import InformationMemory, parse, accepted


def preserved_files(dataset):
    # Exclude caches, not original data. All partial_v1 files (including PNGs) are hashed.
    paths = [p for p in dataset.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [SO101/n for n in ['README.md', 'DESIGN.md', 'PILOT_REPORT.md', 'common.py',
                              'ocr_memory.py', 'simulator.py', 'audit_partial.py', 'collect_partial.py']]
    paths += [SO101.parent/'support'/n for n in ['stage2.py', 'stage3.py', 'stage5.py', 'wrist_sweep.py']]
    return {str(p.resolve()): sha(p) for p in sorted(set(paths))}


def verify_preserved(run):
    baseline = read(run/'source_inventory.json')
    changes = [p for p, h in baseline.items() if not Path(p).is_file() or sha(p) != h]
    dataset = Path(read(run/'experiment.json')['dataset'])
    additions = sorted(set(preserved_files(dataset))-set(baseline))
    if changes or additions:
        raise ValueError(f'Original files changed: {changes}; unexpected additions: {additions}')
    return {'checked_files': len(baseline), 'modified_files': [], 'added_files': []}


def reading_key(o):
    return tuple((o['readings'][f]['value'], o['readings'][f]['ambiguous']) for f in FIELDS)


def prepare(dataset, run):
    dataset, run = dataset.resolve(), run.resolve()
    if run.exists() and any(run.iterdir()):
        raise ValueError('Use an empty run directory; existing experiments are preserved')
    run.mkdir(parents=True, exist_ok=True)
    write(run/'source_inventory.json', preserved_files(dataset))
    config, splits = read(dataset/'dataset_config.json'), read(dataset/'splits.json')
    if config['max_moves'] != MAX_MOVES or config['angles_deg'] != list(range(-70,41,5)):
        raise ValueError('This experiment expects the audited partial_v1 grid and budget')
    for a in splits:
        for b in splits:
            if a < b and set(splits[a]) & set(splits[b]):
                raise ValueError('Label identity crosses splits')
    inventory = {'scenes': Counter(), 'episodes': Counter(), 'views': Counter(),
                 'original_policy_rows': {s: len(rows(dataset/'policy'/f'{s}.jsonl')) for s in splits},
                 'label_ids': splits, 'technical_errors': {}, 'table_agreement_train_val': Counter()}
    loaded = []
    for d in scene_dirs(dataset):
        scene = read(d/'scene.json')
        split = scene['split']
        if scene['label_id'] not in splits[split]:
            raise ValueError('Scene split disagrees with label split')
        completion = read(d/'complete.json')
        if completion['config_sha256'] != sha(dataset/'dataset_config.json'):
            raise ValueError(f'Wrong config hash: {d}')
        for p, h in completion['hashes'].items():
            if not (d/p).exists() or sha(d/p) != h:
                raise ValueError(f'Corrupt scene file: {d/p}')
        table = rows(d/'view_ocr.jsonl')
        if len(table) != 23 or any(v['invalid_reasons'] for v in table):
            raise ValueError(f'Incomplete/unsafe table: {d}')
        for v in table:
            if accepted(parse(v['ocr']), config['thresholds']) != v['readings']:
                raise ValueError('Observation table OCR parse mismatch')
        inventory['scenes'][split] += 1
        inventory['views'][split] += len(table)
        lookup = {v['command_relative_deg']: v for v in table}
        for p in sorted((d/'episodes').glob('*.json')):
            episode = read(p)
            errors = audit_episode(d, episode, config)  # This helper does not write to pilot files.
            if errors:
                inventory['technical_errors'][str(p)] = errors
                continue
            inventory['episodes'][split] += 1
            loaded.append((scene, table, read(d/'truth.json'), episode))
            if split in ['train', 'val']:
                for o in episode['observations']:
                    agreement = inventory['table_agreement_train_val']
                    agreement['observations'] += 1
                    agreement['matching'] += int(reading_key(o) == reading_key(lookup[o['command_relative_deg']]))
    agreement = inventory['table_agreement_train_val']
    # Predeclared reliability gate uses train/val only. Fail closed rather than invent labels.
    if not agreement['observations'] or agreement['matching'] != agreement['observations']:
        write(run/'inventory.json', inventory)
        raise ValueError('Table and live accepted readings differ on train/val; use successful-shortest fallback, not this relabeler')
    records = {s: [] for s in splits}
    excluded, details = {s: [] for s in splits}, {s: Counter() for s in splits}
    dedup = {s: {} for s in splits}
    for scene, table, truth, episode in loaded:
        split = scene['split']
        obs, actions = episode['observations'], episode['actions']
        for i, o in enumerate(obs):
            teacher = direction_teacher(table, o['relative_actual_deg'], o['memory'], MAX_MOVES-i, truth)
            details[split][teacher['status']] += 1
            details[split]['partial_wrong_memory'] += int(teacher.get('partial_wrong', False))
            ref = {'scene_id': scene['scene_id'], 'episode_id': episode['episode_id'], 'step': i,
                   'label_id': scene['label_id']}
            if teacher['status'] not in ['unique', 'tie']:
                excluded[split].append({**ref, **teacher})
                continue
            x = features(causal_state(obs[:i+1], actions[:i]))
            key = (scene['scene_id'], tuple(x), tuple(teacher['target']))
            if key in dedup[split]:
                records[split][dedup[split][key]]['source_refs'].append(ref)
                details[split]['duplicate_states_collapsed'] += 1
                continue
            dedup[split][key] = len(records[split])
            records[split].append({'x': x, 'target': teacher['target'],
                 'legal': [o['command_relative_deg']-5 in config['angles_deg'], o['command_relative_deg']+5 in config['angles_deg']],
                 'metadata': {'scene_id': scene['scene_id'], 'label_id': scene['label_id']},
                 'source_refs': [ref], 'teacher': teacher})
    for split in splits:
        jsonl(run/'data'/f'{split}.jsonl', records[split])
        jsonl(run/'data'/f'{split}_excluded.jsonl', excluded[split])
        write(run/'data'/f'{split}_summary.json', {**details[split], 'retained_unique_states': len(records[split])})
    if not records['train'] or not records['val']:
        raise ValueError('Insufficient train/val states with trustworthy labels')
    # Feature aliasing is expected under partial observability; report conflicting labels explicitly.
    aliases = {}
    for split, recs in records.items():
        groups = defaultdict(set)
        for r in recs:
            groups[tuple(r['x'])].add(tuple(r['target']))
        aliases[split] = {'unique_feature_vectors': len(groups),
                          'feature_vectors_with_conflicting_targets': sum(len(v)>1 for v in groups.values())}
    write(run/'inventory.json', inventory)
    write(run/'feature_schema.json', {'names': feature_names(), 'dimension': len(feature_names()),
          'numeric_values_used': False, 'images_used': False, 'angle_rounding_deg': .001,
          'confidence_rounding': 4, 'feature_aliasing': aliases,
          'allowed_information': 'current wrist; current memory status/confidence/pending count; past actual visits/actions; remaining budget'})
    write(run/'experiment.json', {'dataset': str(dataset), 'development_only': True, 'seed': 20261102,
          'architecture': {'linear_layers': 2, 'hidden': 32, 'activation': 'ReLU', 'outputs': list(DIRECTIONS)},
          'training': {'epochs': 400, 'patience': 60, 'lr': .003, 'weight_decay': .001,
                       'batch': 'full', 'device': 'cpu', 'selection': 'minimum val soft cross entropy; earliest epoch on tie'},
          'teacher': {'mode': 'current-memory budgeted BFS using table accepted OCR',
              'success': 'both values confirmed and equal to offline truth',
              'ties': 'equal probability soft targets, either shortest direction counts correct',
              'wrong_ocr': 'retain actual memory; simulate conflict repair; wrong complete memory STOPs and is excluded from direction training',
              'unsolvable': 'retain in excluded log; no direction invented',
              'dedup': 'same scene, quantized causal feature vector and teacher target collapsed; all source refs retained',
              'reliability_gate': '100 percent accepted value/ambiguity agreement on train/val visits; pixel invariance not assumed'},
          'live_evaluation': {'splits': ['val','test'], 'methods': ['mlp','left_first','right_first','random','fixed_sweep'],
                              'starts': 'one existing sampled start per scene; common start and seed across methods'},
          'software': {p: importlib.metadata.version(p) for p in ['torch','numpy','mujoco','Pillow','easyocr']},
          'platform': platform.platform(), 'machine': platform.machine(),
          'training_data_sha256': {s: sha(run/'data'/f'{s}.jsonl') for s in splits}})
    snapshot = run/'source_snapshot'
    snapshot.mkdir()
    for p in HERE.glob('*.py'):
        shutil.copy2(p, snapshot/p.name)
    write(run/'implementation_sha256.json', {p.name: sha(p) for p in HERE.glob('*.py')})
    write(run/'preservation_before_training.json', verify_preserved(run))
    return {'inventory': inventory, 'train': details['train'], 'val': details['val']}
