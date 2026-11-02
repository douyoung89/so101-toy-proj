"""Build, validate and render the fixed-camera label experiment (no OCR)."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'external/SO-ARM100/Simulation/SO101'
BUILD = ROOT / 'stage2_assets'
OUT = ROOT / 'outputs/stage2'
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
CAMERA = np.array([0.73, -0.48, 0.46])
WIDTH, HEIGHT = 960, 720


def axes(eye, target):
    z = np.asarray(eye) - np.asarray(target)
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def numbers(values):
    return ' '.join(f'{v:.10g}' for v in values)


def build():
    BUILD.mkdir(exist_ok=True)
    label = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(label)
    draw.rectangle((18, 18, 494, 494), outline='black', width=12)
    font_path = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
    font = ImageFont.truetype(font_path, 90) if Path(font_path).exists() else ImageFont.load_default(size=90)
    draw.text((256, 256), 'BOX-01', fill='black', font=font, anchor='mm')
    draw.text((256, 370), 'UP', fill='black', font=font, anchor='mm')
    label.save(BUILD / 'label.png')

    # Establish the label's mounting orientation once at the neutral configuration.
    original = mujoco.MjModel.from_xml_path(str(SOURCE / 'scene.xml'))
    data = mujoco.MjData(original)
    mujoco.mj_forward(original, data)
    gid = original.body('gripper').id
    mount = np.array([-0.0079, 0., -0.115])
    rotation = data.xmat[gid].reshape(3, 3)
    center = data.xpos[gid] + rotation @ mount
    local_axes = rotation.T @ axes(CAMERA, center)

    robot = ET.parse(SOURCE / 'so101_new_calib.xml')
    robot.find('compiler').set('meshdir', str(SOURCE / 'assets'))
    # Give every geometry a stable name for contact reports.
    for i, geom in enumerate(robot.findall('.//worldbody//geom')):
        if geom.get('type') == 'mesh' and not geom.get('name'):
            geom.set('name', f'robot_geom_{i}')
    gripper = robot.find(".//body[@name='gripper']")
    box = ET.SubElement(gripper, 'body', name='label_box', pos=numbers(mount),
                        xyaxes=numbers(local_axes[:, :2].T.ravel()))
    ET.SubElement(box, 'geom', name='label_box_geom', type='box', size='0.025 0.025 0.025',
                  mass='0.03', material='label_material', contype='1', conaffinity='1')
    # This is a rigid mounted payload. Its attachment contacts are intentionally excluded.
    contact = ET.SubElement(robot.getroot(), 'contact')
    for body in ['gripper', 'moving_jaw_so101_v1']:
        ET.SubElement(contact, 'exclude', body1='label_box', body2=body)
    robot.write(BUILD / 'robot.xml')

    scene = ET.parse(SOURCE / 'scene.xml')
    scene.find('include').set('file', 'robot.xml')
    asset = scene.find('asset')
    ET.SubElement(asset, 'texture', name='label_texture', type='cube', filefront='label.png', rgb1='0.55 0.45 0.3')
    ET.SubElement(asset, 'material', name='label_material', texture='label_texture', rgba='1 1 1 1', specular='0')
    asset.find("material[@name='groundplane']").set('reflectance', '0')
    target = center.copy()
    cam_axes = axes(CAMERA, target)
    ET.SubElement(scene.find('worldbody'), 'camera', name='label_camera', pos=numbers(CAMERA),
                  xyaxes=numbers(cam_axes[:, :2].T.ravel()), fovy='35')
    ET.SubElement(scene.getroot(), 'option', timestep='0.002', integrator='implicitfast')
    scene.find('visual').append(ET.Element('quality', shadowsize='2048'))
    scene.find('visual/global').set('offwidth', str(WIDTH))
    scene.find('visual/global').set('offheight', str(HEIGHT))
    scene.write(BUILD / 'scene.xml')
    return mujoco.MjModel.from_xml_path(str(BUILD / 'scene.xml'))


def ids(model):
    joints = [model.joint(n).id for n in JOINTS]
    return (model.jnt_qposadr[joints], model.jnt_dofadr[joints],
            np.array([model.actuator(n).id for n in JOINTS]))


def contacts(model, data):
    bad = []
    for c in data.contact:
        if c.dist >= -0.0005:
            continue
        bodies = {model.body(model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)}
        if bodies == {'world', 'base'}:
            continue
        bad.append({'geoms': [model.geom(g).name for g in (c.geom1, c.geom2)], 'depth_m': float(-c.dist)})
    return bad


def visibility(model, data):
    """Geometric screening only; final readability requires rendered images."""
    bid = model.body('label_box').id
    rot = data.xmat[bid].reshape(3, 3)
    face = data.xpos[bid] + rot[:, 2] * .025
    direction = CAMERA - face
    cosine = float(rot[:, 2] @ direction / np.linalg.norm(direction))
    cid = model.camera('label_camera').id
    cam = data.cam_xmat[cid].reshape(3, 3)
    points = [face + rot @ np.array([x, y, 0.]) for x in [-.025, .025] for y in [-.025, .025]]
    pixels = []
    focal = HEIGHT / (2 * np.tan(np.deg2rad(model.cam_fovy[cid]) / 2))
    for p in points:
        local = cam.T @ (p - CAMERA)
        if local[2] >= 0:
            return {'cosine': cosine, 'in_frame': False, 'width_px': 0., 'clear_rays': 0}
        pixels.append([WIDTH/2 + focal*local[0]/-local[2], HEIGHT/2-focal*local[1]/-local[2]])
    pixels = np.asarray(pixels)
    clear = 0
    for p in [face, *[face + .9*(p-face) for p in points]]:
        ray = p - CAMERA
        ray /= np.linalg.norm(ray)
        hit = np.array([-1], dtype=np.int32)
        mujoco.mj_ray(model, data, CAMERA, ray, None, 1, -1, hit)
        clear += int(hit[0] == model.geom('label_box_geom').id)
    return {'cosine': cosine, 'in_frame': bool(np.all(pixels > [15, 15]) and np.all(pixels < [WIDTH-15, HEIGHT-15])),
            'width_px': float(np.ptp(pixels[:, 0])), 'clear_rays': clear}


def move(model, data, goal, record=None):
    qadr, dadr, aids = ids(model)
    start = data.ctrl[aids].copy()
    elapsed, stable, worst = 0., 0., 0.
    next_frame = data.time
    while elapsed < 7.:
        t = min(elapsed / 1.5, 1.)
        blend = t*t*(3-2*t)
        data.ctrl[aids] = start + blend*(goal-start)
        mujoco.mj_step(model, data)
        elapsed += model.opt.timestep
        bad = contacts(model, data)
        if bad:
            raise RuntimeError(f'Collision during motion: {bad[:3]}')
        error = float(np.max(np.abs(data.qpos[qadr] - goal)))
        speed = float(np.max(np.abs(data.qvel[dadr])))
        worst = max(worst, speed)
        stable = stable + model.opt.timestep if t == 1 and error < .035 and speed < .05 else 0.
        if record is not None and data.time >= next_frame:
            record(data)
            next_frame += 1/30
        if stable >= .3:
            mujoco.mj_forward(model, data)
            return {'settle_seconds': elapsed, 'max_error_rad': error, 'max_speed_rad_s': worst,
                    'visibility': visibility(model, data)}
    raise RuntimeError(f'Failed to settle: error={error:.4f} rad, speed={speed:.4f} rad/s')


def prepare(model):
    qadr, _, aids = ids(model)
    ranges = np.array([model.joint(n).range for n in JOINTS])
    controls = model.actuator_ctrlrange[aids]
    data = mujoco.MjData(model)
    candidates = []
    # Keep the arm near a known elevated configuration; vary wrist roll first.
    for roll in np.linspace(-2.4, 2.4, 49):
        q = np.array([0., 0., 0., 0., roll, 0.])
        if np.any(q < np.maximum(ranges[:, 0], controls[:, 0])) or np.any(q > np.minimum(ranges[:, 1], controls[:, 1])):
            continue
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        v = visibility(model, data)
        if not contacts(model, data) and v['in_frame']:
            candidates.append((q.copy(), v))
    if len(candidates) < 6:
        raise RuntimeError('Insufficient safe candidates; adjust camera/mount.')
    # Six distinct orientations, from back-facing to front-facing.
    candidates.sort(key=lambda pair: pair[1]['cosine'])
    chosen = [candidates[i] for i in np.linspace(0, len(candidates)-1, 6).astype(int)]
    if chosen[0][1]['cosine'] > -.1 or chosen[-1][1]['cosine'] < .8:
        raise RuntimeError('Candidates do not span hidden and front-facing label states.')
    payload = {'units': 'radian', 'joint_order': JOINTS, 'poses': []}
    for i, (q, v) in enumerate(chosen):
        payload['poses'].append({'name': f'P{i}', 'joints': dict(zip(JOINTS, q.tolist())), 'geometric_preview': v})
    # Validate the actual sequential trajectory before publishing poses.json.
    report = execute(model, payload)
    if report[0]['visibility']['cosine'] >= 0 or report[-1]['visibility']['cosine'] < .8:
        raise RuntimeError('Settled poses do not span hidden and readable orientations.')
    if report[-1]['visibility']['clear_rays'] != 5 or report[-1]['visibility']['width_px'] < 80:
        raise RuntimeError('Front-facing label is occluded or too small.')
    (ROOT / 'poses.json').write_text(json.dumps(payload, indent=2) + '\n')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Prepared poses.json and outputs/stage2/validation.json')


def execute(model, payload, render=False):
    qadr, _, aids = ids(model)
    data = mujoco.MjData(model)
    # Initialize once at the model reference pose; all subsequent motion uses actuators.
    data.ctrl[aids] = data.qpos[qadr]
    mujoco.mj_forward(model, data)
    report = []
    renderer = writer = None
    if render:
        import imageio.v2 as imageio
        OUT.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        writer = imageio.get_writer(str(OUT / 'motion.mp4'), fps=30, macro_block_size=2)
    def frame(d):
        renderer.update_scene(d, camera='label_camera')
        return renderer.render().copy()
    try:
        for pose in payload['poses']:
            goal = np.array([pose['joints'][n] for n in JOINTS])
            for j, n in enumerate(JOINTS):
                low = max(model.joint(n).range[0], model.actuator(n).ctrlrange[0])
                high = min(model.joint(n).range[1], model.actuator(n).ctrlrange[1])
                if not np.isfinite(goal[j]) or not low <= goal[j] <= high:
                    raise ValueError(f'{pose["name"]}: {n} outside joint/control limits')
            result = move(model, data, goal, (lambda d: writer.append_data(frame(d))) if render else None)
            result.update(name=pose['name'], actual_joints=dict(zip(JOINTS, data.qpos[qadr].tolist())))
            report.append(result)
            if render:
                image = frame(data)
                Image.fromarray(image).save(OUT / f'{pose["name"]}.png')
                for _ in range(15):
                    writer.append_data(image)
            print(pose['name'], result['visibility'])
        if render:
            sheet = Image.new('RGB', (WIDTH*3, (HEIGHT+40)*2), 'white')
            draw = ImageDraw.Draw(sheet)
            for i, pose in enumerate(payload['poses']):
                x, y = (i % 3)*WIDTH, (i // 3)*(HEIGHT+40)
                with Image.open(OUT / f'{pose["name"]}.png') as image:
                    sheet.paste(image, (x, y+40))
                draw.text((x+12, y+10), pose['name'], fill='black', font=ImageFont.load_default(size=24))
            sheet.save(OUT / 'overview.jpg')
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check', 'render'])
    args = parser.parse_args()
    model = build()
    if args.command == 'prepare':
        prepare(model)
    else:
        payload = json.loads((ROOT / 'poses.json').read_text())
        report = execute(model, payload, render=args.command == 'render')
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f'{args.command}_report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
