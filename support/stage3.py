"""Deterministic synthetic nutrition labels, label-level splits, and robot previews."""
import argparse
import csv
import hashlib
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import stage2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/labels'
OUT = ROOT / 'outputs/stage3'
SEED = 20261029
FONT = Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')


def plan():
    """Separate random streams: answers/splits never determine appearance parameters."""
    values = random.Random(SEED)
    style = random.Random(SEED + 1)
    positive = [(p, s) for p in range(41) for s in range(21) if p >= 20 and s <= 5]
    negative = [(p, s) for p in range(41) for s in range(21) if not (p >= 20 and s <= 5)]
    values.shuffle(positive)
    values.shuffle(negative)
    rows = []
    for split, yes, no in [('train', 35, 35), ('val', 7, 8), ('test', 8, 7)]:
        batch = [(*positive.pop(), 1) for _ in range(yes)] + [(*negative.pop(), 0) for _ in range(no)]
        if split == 'train':
            # Ten simple pilot labels are balanced, and remain unchanged on expansion.
            pilot = batch[:5] + batch[35:40]
            rest = batch[5:35] + batch[40:]
            values.shuffle(pilot)
            values.shuffle(rest)
            batch = pilot + rest
        else:
            values.shuffle(batch)
        for protein, sugar, meets in batch:
            index = len(rows) + 1
            simple = index <= 10
            rows.append(dict(label_id=f'label_{index:03d}', protein_g=protein, sugar_g=sugar,
                             meets_condition=meets, split=split, image=f'images/label_{index:03d}.png',
                             font_size=58 if simple else style.randint(52, 61),
                             line_spacing=120 if simple else style.randint(108, 132),
                             offset_x=0 if simple else style.randint(-10, 10),
                             offset_y=0 if simple else style.randint(-12, 12),
                             alignment='center' if simple else style.choice(['center', 'left']),
                             background_rgb='255 255 255', object_rgb='0.55 0.45 0.3'))
    return rows


def draw_label(row, path):
    if not FONT.exists():
        raise FileNotFoundError(f'Set FONT to an installed bold TrueType font: {FONT}')
    font = ImageFont.truetype(str(FONT), int(row['font_size']))
    image = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(image)
    lines = ['NUTRITION', f'PROTEIN {row["protein_g"]} g', f'SUGAR {row["sugar_g"]} g']
    widths = [draw.textlength(line, font=font) for line in lines]
    left = (512 - max(widths)) / 2 + int(row['offset_x'])
    for i, line in enumerate(lines):
        x = 256 + int(row['offset_x']) if row['alignment'] == 'center' else left
        y = 256 + (i-1)*int(row['line_spacing']) + int(row['offset_y'])
        anchor = 'mm' if row['alignment'] == 'center' else 'lm'
        bounds = draw.textbbox((x, y), line, font=font, anchor=anchor)
        if min(bounds[:2]) < 16 or max(bounds[2:]) > 496:
            raise ValueError(f'Text clipped: {row["label_id"]}, {line}, {bounds}')
        draw.text((x, y), line, fill='black', font=font, anchor=anchor)
    image.save(path)


def generate(count):
    rows = plan()[:count]
    (DATA / 'images').mkdir(parents=True, exist_ok=True)
    # Do not silently downgrade a completed dataset to a pilot manifest.
    if count == 10 and (DATA / 'labels.csv').exists():
        with (DATA / 'labels.csv').open() as f:
            if len(list(csv.DictReader(f))) > 10:
                raise ValueError('Full dataset exists; use generate --count 100.')
    for row in rows:
        path = DATA / row['image']
        draw_label(row, path)
        row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (DATA / 'labels.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for split in ['train', 'val', 'test']:
        (DATA / f'{split}_ids.txt').write_text(''.join(r['label_id']+'\n' for r in rows if r['split'] == split))
    (DATA / 'config.json').write_text(json.dumps({
        'seed': SEED, 'condition': 'protein_g >= 20 and sugar_g <= 5',
        'protein_range_g': [0, 40], 'sugar_range_g': [0, 20],
        'font': str(FONT), 'image_size': [512, 512],
        'appearance': 'Independent RNG for layout; black text, white background and constant brown object for every answer.',
        'split_unit': 'label_id; every derived view/lighting variant must inherit its label split',
        'camera_fovy': 20, 'pilot_count': 10,
    }, indent=2)+'\n')
    validate()


def load_rows():
    with (DATA / 'labels.csv').open() as f:
        return list(csv.DictReader(f))


def validate():
    rows = load_rows()
    assert len(rows) in (10, 100)
    assert len({r['label_id'] for r in rows}) == len(rows)
    assert len({(r['protein_g'], r['sugar_g']) for r in rows}) == len(rows)
    assert len({r['sha256'] for r in rows}) == len(rows)
    for r in rows:
        p, s = int(r['protein_g']), int(r['sugar_g'])
        assert 0 <= p <= 40 and 0 <= s <= 20
        assert int(r['meets_condition']) == int(p >= 20 and s <= 5)
        path = DATA / r['image']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r['sha256']
        with Image.open(path) as im:
            assert im.size == (512, 512) and im.mode == 'RGB'
    ids = {split: set((DATA / f'{split}_ids.txt').read_text().splitlines()) for split in ['train', 'val', 'test']}
    assert not (ids['train'] & ids['val'] or ids['train'] & ids['test'] or ids['val'] & ids['test'])
    for split in ids:
        assert ids[split] == {r['label_id'] for r in rows if r['split'] == split}
    counts = {s: dict(Counter(r['meets_condition'] for r in rows if r['split'] == s)) for s in ids}
    if len(rows) == 100:
        assert [len(ids[s]) for s in ['train', 'val', 'test']] == [70, 15, 15]
        assert sum(int(r['meets_condition']) for r in rows) == 50
        assert all(abs(int(c.get('1', 0))-int(c.get('0', 0))) <= 1 for c in counts.values())
    observations = OUT / 'observations.csv'
    observed = 0
    if observations.exists():
        by_id = {r['label_id']: r for r in rows}
        with observations.open() as f:
            for observation in csv.DictReader(f):
                label = by_id[observation['label_id']]
                assert all(observation[key] == label[key] for key in ['split', 'protein_g', 'sugar_g'])
                assert (ROOT / observation['image']).exists()
                observed += 1
    report = {'labels': len(rows), 'splits': {s: len(v) for s, v in ids.items()},
              'condition_counts': counts, 'overlapping_ids': 0, 'unique_answer_pairs': len(rows),
              'observations_checked': observed,
              'checks': 'IDs, split inheritance lists, answer range/condition, image hashes, image size, uniqueness, observation split/answers'}
    (DATA / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def preview(limit, pose_names):
    """Compile a separate scene per label; never overwrite the stage-2 label/scene."""
    stage2.build()
    OUT.mkdir(parents=True, exist_ok=True)
    poses = json.loads((ROOT / 'poses.json').read_text())['poses']
    selected = [p for p in poses if p['name'] in pose_names]
    if len(selected) != len(set(pose_names)):
        raise ValueError('Unknown pose requested')
    rows = load_rows()[:limit]
    manifest = []
    for row in rows:
        tree = ET.parse(stage2.BUILD / 'scene.xml')
        tree.find('include').set('file', str(stage2.BUILD / 'robot.xml'))
        tree.find("asset/texture[@name='label_texture']").set('filefront', str(DATA / row['image']))
        tree.find("worldbody/camera[@name='label_camera']").set('fovy', '20')
        scene_path = OUT / 'preview_scene.xml'
        tree.write(scene_path)
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        qadr, _, aids = stage2.ids(model)
        data.ctrl[aids] = data.qpos[qadr]
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=stage2.HEIGHT, width=stage2.WIDTH) as renderer:
            for pose in selected:
                goal = np.array([pose['joints'][n] for n in stage2.JOINTS])
                result = stage2.move(model, data, goal)
                renderer.update_scene(data, camera='label_camera')
                target = OUT / row['split'] / row['label_id'] / f'{pose["name"]}.png'
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(renderer.render()).save(target)
                manifest.append(dict(label_id=row['label_id'], split=row['split'], pose=pose['name'],
                                     protein_g=row['protein_g'], sugar_g=row['sugar_g'],
                                     image=str(target.relative_to(ROOT)),
                                     max_error_rad=result['max_error_rad']))
    with (OUT / 'observations.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    # Camera crops are enlarged for review; original full frames remain available.
    tiles = []
    front = [r for r in manifest if r['pose'] == 'P5']
    sampled = [front[i] for i in np.linspace(0, len(front)-1, min(10, len(front))).astype(int)] if front else []
    for r in sampled:
        with Image.open(ROOT / r['image']) as image:
            crop = image.crop((350, 220, 610, 500)).resize((390, 420))
        tile = Image.new('RGB', (390, 455), 'white')
        tile.paste(crop, (0, 35))
        ImageDraw.Draw(tile).text((8, 8), r['label_id'], fill='black', font=ImageFont.load_default(size=20))
        tiles.append(tile)
    if tiles:
        sheet = Image.new('RGB', (390*5, 455*((len(tiles)+4)//5)), 'white')
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i%5)*390, (i//5)*455))
        sheet.save(OUT / 'camera_overview.jpg')
    print(f'Saved {len(manifest)} observations; split inherited from labels.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--count', type=int, choices=[10, 100], default=100)
    view = sub.add_parser('preview')
    view.add_argument('--limit', type=int, default=10)
    view.add_argument('--poses', nargs='+', default=['P0', 'P3', 'P5'])
    sub.add_parser('check')
    args = parser.parse_args()
    if args.command == 'generate':
        generate(args.count)
    elif args.command == 'preview':
        if args.limit <= 0:
            parser.error('--limit must be positive')
        preview(args.limit, args.poses)
    else:
        validate()


if __name__ == '__main__':
    main()
