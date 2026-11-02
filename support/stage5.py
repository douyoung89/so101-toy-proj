"""Collect complete, causal six-view robot episodes with separate supervision."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import time
import warnings
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

import stage2
from ocr_memory import InformationMemory, accepted, parse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/stage5'
SEED = 20261030
ROI = [290, 180, 670, 540]  # Fixed image coordinates; no model/label/answer lookup.
READER = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def plan():
    rng = random.Random(SEED)
    poses = json.loads((ROOT/'poses.json').read_text())['poses']
    environments = []
    for i in range(10):
        order = [p['name'] for p in poses]
        rng.shuffle(order)
        environments.append({'environment_id': i, 'rng_seed': SEED, 'rng_draw_block': i,
                             'mount_rotation_xy_rad': [rng.uniform(-.10, .10), rng.uniform(-.10, .10)],
                             'camera_position': (stage2.CAMERA+np.array([rng.uniform(-.012, .012) for _ in range(3)])).tolist(),
                             'light_diffuse': [rng.uniform(.65, .85)]*3,
                             'headlight_ambient': [rng.uniform(.23, .33)]*3,
                             'light_position': [rng.uniform(-.2, .2), rng.uniform(-.2, .2), rng.uniform(3.3, 3.7)],
                             'object_rgb': [rng.uniform(.48, .62), rng.uniform(.38, .52), rng.uniform(.24, .36)],
                             'condition': {'protein_min': [15, 20, 25, 30, 20][i % 5],
                                           'sugar_max': [3, 5, 8, 5, 10][i % 5]},
                             'pose_order': order})
    with (ROOT/'data/labels/labels.csv').open() as f:
        labels = list(csv.DictReader(f))
    episodes = []
    # Each of the same ten environments is used for EVERY label: nuisance and answers are independent.
    for label in labels:
        for env in environments:
            episodes.append({'episode_id': f'episode_{len(episodes):04d}', 'label_id': label['label_id'],
                             'split': label['split'], 'label': label, 'environment': env})
    return {'seed': SEED, 'observations_per_episode': 6, 'environments': environments, 'episodes': episodes,
            'sources': {p: sha(ROOT/p) for p in ['stage5.py', 'stage2.py', 'ocr_memory.py']},
            'model_commit': subprocess.check_output(['git', '-C', str(ROOT/'external/SO-ARM100'), 'rev-parse', 'HEAD'], text=True).strip(),
            'packages': {p: importlib.metadata.version(p) for p in ['mujoco', 'easyocr', 'torch', 'Pillow']},
            'base_scene_sha256': sha(stage2.BUILD/'scene.xml'), 'base_robot_sha256': sha(stage2.BUILD/'robot.xml'),
            'physics': {'timestep_s': .002, 'interpolation_s': 1.5, 'settle_error_rad': .035,
                        'settle_velocity_rad_s': .05, 'stable_duration_s': .3, 'timeout_s': 7.,
                        'penetration_tolerance_m': .0005, 'attachment_contact_exclusions': ['gripper', 'moving_jaw_so101_v1']},
            'thresholds_sha256': sha(ROOT/'outputs/stage4/thresholds.json'),
            'labels_sha256': sha(ROOT/'data/labels/labels.csv'), 'poses_sha256': sha(ROOT/'poses.json'),
            'ocr': {'engine': 'EasyOCR en CPU', 'fixed_pixel_roi_xyxy': ROI, 'coordinate_output': 'original 960x720',
                    'thresholds': json.loads((ROOT/'outputs/stage4/thresholds.json').read_text())['thresholds'],
                    'note': 'Stage4 thresholds frozen; ROI preprocessing and domain variation can change recognition accuracy.'}}


def ensure_plan():
    stage2.build()
    payload = plan()
    target = DATA/'plan.json'
    if target.exists() and json.loads(target.read_text()) != payload:
        raise ValueError('Existing collection plan differs; preserve the experiment and use a new output directory.')
    write(target, payload)
    return payload


def episode_path(ep):
    return DATA/'episodes'/ep['split']/ep['episode_id']


def scene(ep, directory):
    env = ep['environment']
    robot = ET.parse(stage2.BUILD/'robot.xml')
    body = robot.find(".//body[@name='label_box']")
    base = np.fromstring(body.get('xyaxes'), sep=' ').reshape(2, 3).T
    rotation = np.column_stack((base, np.cross(base[:, 0], base[:, 1])))
    x, y = env['mount_rotation_xy_rad']
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rotation = rotation @ rx @ ry
    body.set('xyaxes', stage2.numbers(rotation[:, :2].T.ravel()))
    directory.mkdir(parents=True, exist_ok=True)
    robot.write(directory/'robot.xml')
    tree = ET.parse(stage2.BUILD/'scene.xml')
    tree.find('include').set('file', 'robot.xml')
    texture = tree.find("asset/texture[@name='label_texture']")
    texture.set('filefront', str(ROOT/'data/labels'/ep['label']['image']))
    texture.set('rgb1', stage2.numbers(env['object_rgb']))
    camera = tree.find("worldbody/camera[@name='label_camera']")
    # Keep the same neutral payload point as stage 2; camera is fixed throughout an episode.
    model = mujoco.MjModel.from_xml_path(str(stage2.BUILD/'scene.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    center = data.xpos[model.body('label_box').id]
    camera.set('pos', stage2.numbers(env['camera_position']))
    camera.set('xyaxes', stage2.numbers(stage2.axes(env['camera_position'], center)[:, :2].T.ravel()))
    camera.set('fovy', '20')
    tree.find('worldbody/light').set('pos', stage2.numbers(env['light_position']))
    tree.find('worldbody/light').set('diffuse', stage2.numbers(env['light_diffuse']))
    tree.find('visual/headlight').set('ambient', stage2.numbers(env['headlight_ambient']))
    tree.write(directory/'scene.xml')
    return mujoco.MjModel.from_xml_path(str(directory/'scene.xml'))


def recognize(image):
    global READER
    if READER is None:
        warnings.filterwarnings('ignore', category=UserWarning, message="'pin_memory'.*")
        warnings.filterwarnings('ignore', category=UserWarning, message='torch.quantize_per_tensor.*')
        import torch
        import easyocr
        torch.set_num_threads(1)
        READER = easyocr.Reader(['en'], gpu=False, download_enabled=False, verbose=False,
                                model_storage_directory=str(ROOT/'data/easyocr_models'),
                                user_network_directory=str(ROOT/'data/easyocr_models/user_network'))
    left, top, right, bottom = ROI
    results = READER.readtext(image[top:bottom, left:right], detail=1, paragraph=False, workers=0,
                              batch_size=1, canvas_size=1280, mag_ratio=1.)
    return [{'text': text, 'confidence': float(confidence),
             'bbox': [[float(x)+left, float(y)+top] for x, y in box]} for box, text, confidence in results]


def joint_state(model, data):
    qadr, dadr, aids = stage2.ids(model)
    return {'qpos_rad': data.qpos[qadr].tolist(), 'qvel_rad_s': data.qvel[dadr].tolist(),
            'actuator_targets_rad': data.ctrl[aids].tolist(), 'simulation_time_s': float(data.time)}


def supervision(ep, observation, raw):
    protein, sugar = int(ep['label']['protein_g']), int(ep['label']['sugar_g'])
    fields = observation['memory']['fields']
    known_p, known_s = fields['protein_g']['value'], fields['sugar_g']['value']
    condition = ep['environment']['condition']
    truth = protein >= condition['protein_min'] and sugar <= condition['sugar_max']
    proven_pass = known_p == protein and known_s == sugar and truth
    proven_fail = ((known_p == protein and protein < condition['protein_min']) or
                   (known_s == sugar and sugar > condition['sugar_max']))
    return {'observation_id': observation['observation_id'], 'protein_g': protein, 'sugar_g': sugar,
            'truth_meets_condition': truth, 'ocr_candidate_correctness': {
                f: [c['value'] == (protein if f == 'protein_g' else sugar) for c in cs] for f, cs in parse(raw).items()},
            'reference_stop_action': 'stop_pass' if proven_pass else 'stop_fail' if proven_fail else 'continue',
            'reference_definition': 'current memory contains correct sufficient evidence, checked with ground truth; no future images used'}


def collect_episode(ep):
    directory = episode_path(ep)
    if (directory/'complete.json').exists():
        return ep['episode_id'], 'cached'
    model = scene(ep, directory/'evaluation')
    data = mujoco.MjData(model)
    _, _, aids = stage2.ids(model)
    data.ctrl[aids] = 0.
    mujoco.mj_forward(model, data)
    payload = json.loads((DATA/'plan.json').read_text())
    thresholds = payload['ocr']['thresholds']
    condition = ep['environment']['condition']
    memory = InformationMemory(**condition)
    poses = {p['name']: p for p in json.loads((ROOT/'poses.json').read_text())['poses']}
    observations, transitions, targets, visited = [], [], [], []
    setup = None
    with mujoco.Renderer(model, height=stage2.HEIGHT, width=stage2.WIDTH) as renderer:
        for index, pose_id in enumerate(ep['environment']['pose_order']):
            before = joint_state(model, data)
            goal = np.array([poses[pose_id]['joints'][n] for n in stage2.JOINTS])
            result = stage2.move(model, data, goal)
            # stage2's visibility helper assumes the unperturbed camera; do not log it here.
            result.pop('visibility', None)
            after = joint_state(model, data)
            transition = {'from_observation_id': f'obs_{index-1:02d}' if index else None,
                          'to_observation_id': f'obs_{index:02d}',
                          'action': {'type': 'move_to_pose', 'pose_id': pose_id,
                                     'target_joints_rad': goal.tolist(), 'interpolation_seconds': 1.5},
                          'before': before, 'after': after, 'settling': result}
            if index:
                transitions.append(transition)
            else:
                setup = transition
            renderer.update_scene(data, camera='label_camera')
            rgb = renderer.render().copy()
            observation_id = f'obs_{index:02d}'
            rgb_path = directory/'policy/rgb'/f'{observation_id}.png'
            policy_path = directory/'policy/rgb224'/f'{observation_id}.png'
            rgb_path.parent.mkdir(parents=True, exist_ok=True)
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.fromarray(rgb)
            image.save(rgb_path)
            image.resize((224, 224)).save(policy_path)
            raw = recognize(rgb)
            readings = accepted(parse(raw), thresholds)
            visited.append(pose_id)
            state = memory.observe(observation_id, index, readings)
            observation = {'observation_id': observation_id, 'sequence_index': index,
                           'rgb': f'rgb/{observation_id}.png', 'policy_rgb': f'rgb224/{observation_id}.png',
                           'joint_order': stage2.JOINTS, 'joint_state': after, 'pose_id': pose_id,
                           'visited_pose_ids': visited.copy(), 'condition': condition,
                           'ocr': raw, 'readings': readings, 'memory': state}
            observations.append(observation)
            target = supervision(ep, observation, raw)
            box_id = model.body('label_box').id
            target['label_body_world_rotation'] = data.xmat[box_id].reshape(3, 3).tolist()
            target['label_face_world_normal'] = data.xmat[box_id].reshape(3, 3)[:, 2].tolist()
            if index == 5 and target['reference_stop_action'] == 'continue':
                target['reference_stop_action'] = 'stop_unknown'
            targets.append(target)
            # Intentionally no break on pass/fail: collect every prepared pose.
    jsonl(directory/'policy/observations.jsonl', observations)
    jsonl(directory/'supervision/transitions.jsonl', transitions)
    jsonl(directory/'supervision/targets.jsonl', targets)
    write(directory/'evaluation/config.json', {'episode': ep, 'setup_motion': setup,
                                              'ocr': payload['ocr'], 'joint_order': stage2.JOINTS,
                                              'termination_reason': 'all_prepared_poses_visited'})
    replay_html(directory, observations, transitions)
    validate_episode(directory)
    files = list((directory/'policy').rglob('*.png')) + [directory/'policy/observations.jsonl',
            directory/'supervision/transitions.jsonl', directory/'supervision/targets.jsonl', directory/'evaluation/config.json']
    write(directory/'complete.json', {'episode_id': ep['episode_id'], 'observations': len(observations),
                                      'plan_sha256': sha(DATA/'plan.json'),
                                      'hashes': {str(p.relative_to(directory)): sha(p) for p in files}})
    return ep['episode_id'], 'collected'


def load_policy_step(directory, index):
    """Only current/past information. No transitions, generation configs or targets."""
    records = read_jsonl(Path(directory)/'policy/observations.jsonl')
    return records[index]


def validate_episode(directory):
    observations = read_jsonl(directory/'policy/observations.jsonl')
    transitions = read_jsonl(directory/'supervision/transitions.jsonl')
    targets = read_jsonl(directory/'supervision/targets.jsonl')
    assert len(observations) == 6 and len(transitions) == 5 and len(targets) == 6
    assert len({o['pose_id'] for o in observations}) == 6
    forbidden = {'protein_g_truth', 'sugar_g_truth', 'truth_meets_condition', 'label_id', 'label',
                 'mount_rotation_xy_rad', 'object_rgb', 'camera_position', 'future', 'next_observation', 'reference_stop_action'}
    def check_keys(value):
        if isinstance(value, dict):
            assert not (set(value) & forbidden)
            for child in value.values():
                check_keys(child)
        elif isinstance(value, list):
            for child in value:
                check_keys(child)
    for i, o in enumerate(observations):
        check_keys(o)
        assert o['sequence_index'] == i and o['observation_id'] == f'obs_{i:02d}'
        assert o['visited_pose_ids'] == [r['pose_id'] for r in observations[:i+1]]
        for field in o['memory']['fields'].values():
            assert all(e['sequence_index'] <= i for e in field['evidence'])
        for key, size in [('rgb', (960, 720)), ('policy_rgb', (224, 224))]:
            with Image.open(directory/'policy'/o[key]) as image:
                assert image.size == size
        if i:
            t = transitions[i-1]
            assert t['from_observation_id'] == observations[i-1]['observation_id']
            assert t['to_observation_id'] == o['observation_id']
            assert t['before'] == observations[i-1]['joint_state']
            assert t['after'] == o['joint_state']
            assert t['action']['pose_id'] == o['pose_id']
            assert t['after']['simulation_time_s'] > t['before']['simulation_time_s']
            assert max(abs(a-b) for a, b in zip(t['action']['target_joints_rad'], t['after']['qpos_rad'])) < .035
    return len(observations)


def replay_html(directory, observations, transitions):
    # Offline replay only: embedded complete timeline is NEVER used by load_policy_step.
    template = '''<!doctype html><meta charset="utf-8"><title>Episode replay</title>
<style>body{font:16px system-ui;margin:24px;background:#eef2f7}img{width:min(70vw,960px)}pre{white-space:pre-wrap;background:white;padding:16px}button{padding:10px;margin:5px}</style>
<h1>관측 에피소드 재생</h1><p>안정화된 관측과 그 사이의 실제 제어 기록입니다.</p>
<button onclick="show(i-1)">이전</button><button onclick="playing=!playing">재생 / 정지</button><button onclick="show(i+1)">다음</button><span id="title"></span><br>
<img id="frame"><pre id="state"></pre><script>
const observations=OBSERVATIONS, transitions=TRANSITIONS; let i=0,playing=false;
function show(n){i=Math.max(0,Math.min(observations.length-1,n));const o=observations[i];
document.getElementById('title').textContent=`${i+1}/${observations.length} · ${o.pose_id} · ${o.memory.decision}`;
document.getElementById('frame').src='policy/'+o.rgb;
document.getElementById('state').textContent=JSON.stringify({condition:o.condition,visited:o.visited_pose_ids,previous_motion:i?transitions[i-1]:null,ocr:o.ocr,memory:o.memory},null,2);}
setInterval(()=>{if(playing){if(i===observations.length-1)playing=false;else show(i+1)}},1500);show(0);
</script>'''
    (directory/'replay.html').write_text(template.replace('OBSERVATIONS', json.dumps(observations)).replace('TRANSITIONS', json.dumps(transitions)))


def collect(limit, workers):
    payload = ensure_plan()
    # First ten are spread across labels, environment variants and splits.
    pilot_indices = [i*10+i for i in range(7)] + [700+7, 850+8, 990+9]
    selected = [payload['episodes'][i] for i in pilot_indices] if limit == 10 else payload['episodes'][:limit]
    errors = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(collect_episode, ep): ep for ep in selected}
        for n, future in enumerate(as_completed(futures), 1):
            ep = futures[future]
            try:
                _, status = future.result()
            except Exception as error:
                status = 'FAILED'
                errors.append({'episode_id': ep['episode_id'], 'error': repr(error)})
            if n % 10 == 0 or status == 'FAILED':
                print(f'{n}/{len(selected)} episodes, {len(errors)} failures, {time.monotonic()-start:.0f}s', flush=True)
    write(DATA/'last_collection.json', {'requested': len(selected), 'errors': errors,
                                       'seconds': time.monotonic()-start})
    if errors:
        raise RuntimeError(f'{len(errors)} episodes failed; inspect last_collection.json and rerun to resume')
    audit()


def audit():
    payload = json.loads((DATA/'plan.json').read_text())
    completed, counts, observations = [], {}, 0
    nuisance_labels = {i: set() for i in range(10)}
    condition_counts = {i: {'pass': 0, 'fail': 0} for i in range(10)}
    continued_after_decision = 0
    for ep in payload['episodes']:
        directory = episode_path(ep)
        complete = directory/'complete.json'
        if not complete.exists():
            continue
        marker = json.loads(complete.read_text())
        assert marker['plan_sha256'] == sha(DATA/'plan.json')
        for name, expected in marker['hashes'].items():
            assert sha(directory/name) == expected
        observations += validate_episode(directory)
        records = read_jsonl(directory/'policy/observations.jsonl')
        continued_after_decision += int(any(o['memory']['decision'] != 'need_more' for o in records[:-1]))
        nuisance_labels[ep['environment']['environment_id']].add(ep['label_id'])
        c = ep['environment']['condition']
        truth = int(ep['label']['protein_g']) >= c['protein_min'] and int(ep['label']['sugar_g']) <= c['sugar_max']
        condition_counts[ep['environment']['environment_id']]['pass' if truth else 'fail'] += 1
        completed.append({'episode_id': ep['episode_id'], 'label_id': ep['label_id'], 'split': ep['split'],
                          'environment_id': ep['environment']['environment_id'], 'path': str(directory.relative_to(DATA))})
        counts[ep['split']] = counts.get(ep['split'], 0)+1
    assert len({(e['label_id'], e['environment_id']) for e in completed}) == len(completed)
    ids = {s: {e['label_id'] for e in completed if e['split'] == s} for s in ['train', 'val', 'test']}
    assert not (ids['train'] & ids['val'] or ids['train'] & ids['test'] or ids['val'] & ids['test'])
    write(DATA/'index.json', completed)
    write(DATA/'audit.json', {'planned': len(payload['episodes']), 'completed': len(completed),
                             'observations': observations, 'split_episode_counts': counts,
                             'split_label_overlap': 0, 'continued_after_decision': continued_after_decision,
                             'labels_per_environment': {i: len(v) for i, v in nuisance_labels.items()},
                             'condition_counts': condition_counts,
                             'checks': ['hashes', 'transition state alignment', 'no repeated poses',
                             'causal memory', 'policy/evaluation key separation', 'image sizes', 'label-level split inheritance']})
    print(f'Audit: {len(completed)} episodes, {observations} observations, split overlap 0', flush=True)


def video(episode_id):
    import imageio.v2 as imageio
    payload = json.loads((DATA/'plan.json').read_text())
    ep = next(ep for ep in payload['episodes'] if ep['episode_id'] == episode_id)
    directory = episode_path(ep)
    model = mujoco.MjModel.from_xml_path(str(directory/'evaluation/scene.xml'))
    data = mujoco.MjData(model)
    poses = {p['name']: p for p in json.loads((ROOT/'poses.json').read_text())['poses']}
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        with imageio.get_writer(str(directory/'motion.mp4'), fps=30, macro_block_size=2) as writer:
            def frame(d):
                renderer.update_scene(d, camera='label_camera')
                writer.append_data(renderer.render().copy())
            for pose in ep['environment']['pose_order']:
                goal = np.array([poses[pose]['joints'][n] for n in stage2.JOINTS])
                stage2.move(model, data, goal, frame)
                for _ in range(15):
                    frame(data)
    print(directory/'motion.mp4')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('plan')
    c = sub.add_parser('collect')
    c.add_argument('--limit', type=int, default=10)
    c.add_argument('--workers', type=int, default=2)
    sub.add_parser('audit')
    v = sub.add_parser('video')
    v.add_argument('--episode', default='episode_0000')
    args = parser.parse_args()
    if args.command == 'plan':
        ensure_plan()
    elif args.command == 'collect':
        if not 1 <= args.limit <= 1000 or not 1 <= args.workers <= 8:
            parser.error('limit must be 1..1000 and workers 1..8')
        collect(args.limit, args.workers)
    elif args.command == 'audit':
        audit()
    else:
        video(args.episode)


if __name__ == '__main__':
    main()
