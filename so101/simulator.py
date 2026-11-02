"""MuJoCo adapter. Existing v2_005 XML is the immutable geometry reference."""
import csv
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import ROOT, FIELDS, read, sha

# Reuse existing contact checks, projection and cached EasyOCR weights.
sys.path.append(str(ROOT / 'support'))
import stage2
import stage3
import stage5
import wrist_sweep

BASE = ROOT / 'assets/reference'


def labels():
    with (ROOT / 'data/labels/labels.csv').open() as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def baseline():
    robot = ET.parse(BASE / 'robot.xml')
    cover = robot.find(".//geom[@name='partial_cover']")
    body = robot.find(".//body[@name='shutter']")
    eq = robot.find("equality/joint[@joint1='label_shutter']")
    position = np.fromstring(body.get('pos'), sep=' ')
    size = np.fromstring(cover.get('size'), sep=' ')
    coefficients = np.fromstring(eq.get('polycoef'), sep=' ')
    ep = read(BASE / 'config.json')['episode']
    reference = ep['design']['reference_joints_rad']
    center = position[1] + coefficients[0] + coefficients[1]*reference[4]
    inner = []
    for name, sign in [('bottom', 1), ('top', -1)]:
        wall = robot.find(f".//geom[@name='hood_{name}']")
        inner.append(float(wall.get('pos').split()[1]) + sign*float(wall.get('size').split()[1]))
    return {'body_position_m': position.tolist(), 'half_size_m': size.tolist(),
            'reference_center_y_m': float(center), 'slope_m_per_rad': float(coefficients[1]),
            'original_slide_limited': False, 'hood_inner_y_m': inner,
            'reference_joints_rad': reference,
            'row_spacing_m': .05*int(ep['label']['line_spacing'])/512,
            'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                              [BASE/'scene.xml', BASE/'robot.xml', BASE/'config.json']}}


def sample_settings(seed):
    rng = np.random.default_rng(seed)
    base = baseline()
    size = np.array(base['half_size_m'])
    size[0] *= rng.uniform(.9, 1.1)
    size[1] *= rng.uniform(.8, 1.2)
    center = base['reference_center_y_m'] + rng.uniform(-.25, .25)*base['row_spacing_m']
    # v2 has NO slide limits. Define new limits from hood inner faces with clearance.
    clearance = .0003
    safe = [base['hood_inner_y_m'][0] + size[1] + clearance,
            base['hood_inner_y_m'][1] - size[1] - clearance]
    span = base['hood_inner_y_m'][1]-base['hood_inner_y_m'][0]-2*base['half_size_m'][1]
    nominal = [base['hood_inner_y_m'][0]+base['half_size_m'][1]+clearance,
               base['hood_inner_y_m'][1]-base['half_size_m'][1]-clearance]
    limits = [max(safe[0], nominal[0]+rng.uniform(-.2, .2)*span),
              min(safe[1], nominal[1]+rng.uniform(-.2, .2)*span)]
    if not limits[0] < center < limits[1]:
        raise ValueError('Sampled shutter limits do not contain reference center')
    return {'center_y_m': float(center), 'half_size_m': size.tolist(),
            'slope_m_per_rad': float(rng.choice([-1, 1])*abs(base['slope_m_per_rad'])*rng.uniform(.5, 1.5)),
            'center_limits_y_m': limits,
            'coupling': 'clipped center target updated before every physics step; soft equality servo'}


def create_scene(path, label, settings):
    path.mkdir(parents=True, exist_ok=True)
    robot = ET.parse(BASE/'robot.xml')
    robot.find('compiler').set('meshdir', str(ROOT/'external/SO-ARM100/Simulation/SO101/assets'))
    robot.find(".//geom[@name='partial_cover']").set('size', stage2.numbers(settings['half_size_m']))
    eq = robot.find("equality/joint[@joint1='label_shutter']")
    # A fixed target equality driven per step implements saturation without opposing constraints.
    eq.set('polycoef', '0 0 0 0 0')
    robot.write(path/'robot.xml')
    scene = ET.parse(BASE/'scene.xml')
    scene.find("asset/texture[@name='label_texture']").set('filefront', str(ROOT/'data/labels'/label['image']))
    scene.write(path/'scene.xml')


class Simulator:
    def __init__(self, path, settings, label, render=True):
        self.model = mujoco.MjModel.from_xml_path(str(path/'scene.xml'))
        self.data = mujoco.MjData(self.model)
        self.settings, self.label = settings, label
        self.goal = np.array(baseline()['reference_joints_rad'], dtype=float)
        self.reference = float(self.goal[4])
        self.qadr, self.dadr, self.aids = stage2.ids(self.model)
        self.slide = self.model.jnt_qposadr[self.model.joint('label_shutter').id]
        self.equality = next(
            i for i in range(self.model.neq) if self.model.eq_obj1id[i] == self.model.joint('label_shutter').id)
        self.pairs, self.fixture, self.payload = wrist_sweep.fixture_pairs(self.model)
        # Initial setup is not an episode action; start at the known reference configuration.
        self.data.qpos[self.qadr] = self.goal
        self.data.ctrl[self.aids] = self.goal
        self.couple()
        self.data.qpos[self.slide] = self.model.eq_data[self.equality, 0]
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=720, width=960) if render else None

    def couple(self):
        s = self.settings
        center = np.clip(s['center_y_m'] + s['slope_m_per_rad']*(self.data.qpos[self.qadr[4]]-self.reference),
                         *s['center_limits_y_m'])
        self.model.eq_data[self.equality, :5] = [center-baseline_body_y(), 0, 0, 0, 0]

    def move(self, relative_deg):
        target = self.goal.copy()
        target[4] = self.reference + np.deg2rad(relative_deg)
        low = max(self.model.jnt_range[self.model.joint('wrist_roll').id, 0], self.model.actuator_ctrlrange[self.aids[4], 0])
        high = min(self.model.jnt_range[self.model.joint('wrist_roll').id, 1], self.model.actuator_ctrlrange[self.aids[4], 1])
        if not low <= target[4] <= high:
            raise ValueError('Mechanical wrist limit exceeded')
        before = stage5.joint_state(self.model, self.data)
        start = self.data.ctrl[self.aids].copy()
        elapsed = stable = 0.
        reasons = set()
        steps = 0
        while elapsed < 7:
            t = min(elapsed/1.5, 1.)
            self.data.ctrl[self.aids] = start+t*t*(3-2*t)*(target-start)
            self.couple()
            mujoco.mj_step(self.model, self.data)
            elapsed += self.model.opt.timestep
            steps += 1
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError('Nonfinite physics')
            if steps % 5 == 0:
                mujoco.mj_forward(self.model, self.data)
                reasons.update(self.geometry_reasons())
            error = float(np.max(np.abs(self.data.qpos[self.qadr]-target)))
            speed = float(np.max(np.abs(self.data.qvel[self.dadr])))
            slide_error = abs(self.data.qpos[self.slide]-self.model.eq_data[self.equality, 0])
            stable = stable+self.model.opt.timestep if t == 1 and error < .003 and speed < .01 and slide_error < .0001 else 0.
            if stable >= .3:
                break
        if stable < .3:
            reasons.add('move_failed')
        mujoco.mj_forward(self.model, self.data)
        return {'command_relative_deg': relative_deg, 'command_wrist_rad': float(target[4]),
                'before': before, 'after': stage5.joint_state(self.model, self.data),
                'actual_relative_deg': float(np.rad2deg(self.data.qpos[self.qadr[4]]-self.reference)),
                'elapsed_s': elapsed, 'max_error_rad': error, 'final_speed_rad_s': speed,
                'invalid_reasons': sorted(reasons | set(self.geometry_reasons()))}

    def geometry_reasons(self):
        g = wrist_sweep.camera_geometry(self.model, self.data, self.fixture, self.payload)
        reasons = [k for k in ['box_in_frame', 'label_in_frame', 'fixture_in_frame'] if not g[k]]
        if g['label_facing_cosine'] <= .15:
            reasons.append('label_back_or_edge')
        if wrist_sweep.geometric_penetrations(self.model, self.data, self.pairs):
            reasons.append('geometry_overlap')
        if stage2.contacts(self.model, self.data):
            reasons.append('enabled_contact')
        return reasons

    def capture(self, directory, stem, thresholds):
        from ocr_memory import parse, accepted
        if self.renderer is None:
            raise RuntimeError('Rendering disabled')
        self.renderer.update_scene(self.data, camera='label_camera')
        image = self.renderer.render().copy()
        raw = stage5.recognize(image)
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(directory/f'{stem}.png')
        Image.fromarray(image).resize((224, 224)).save(directory/f'{stem}_224.png')
        return {'image': str((directory/f'{stem}.png').name), 'policy_image': f'{stem}_224.png',
                'image_sha256': sha(directory/f'{stem}.png'),
                'policy_image_sha256': sha(directory/f'{stem}_224.png'),
                'joint_state': stage5.joint_state(self.model, self.data),
                'relative_actual_deg': float(np.rad2deg(self.data.qpos[self.qadr[4]]-self.reference)),
                'ocr': raw, 'readings': accepted(parse(raw), thresholds)}

    def exposure(self):
        """Sample text rectangles with rays; geometric proxy, not a legibility assertion."""
        font = ImageFont.truetype(str(stage3.FONT), int(self.label['font_size']))
        draw = ImageDraw.Draw(Image.new('RGB', (512, 512)))
        texts = ['NUTRITION', f'PROTEIN {self.label["protein_g"]} g', f'SUGAR {self.label["sugar_g"]} g']
        left = (512-max(draw.textlength(t, font=font) for t in texts))/2+int(self.label['offset_x'])
        center_align = self.label['alignment'] == 'center'
        box = self.model.geom('label_box_geom').id
        rot = self.data.geom_xmat[box].reshape(3, 3)
        camera = self.data.cam_xpos[self.model.camera('label_camera').id]
        result = {}
        for i, field in enumerate(FIELDS, 1):
            bounds = draw.textbbox((256+int(self.label['offset_x']) if center_align else left,
                                   256+(i-1)*int(self.label['line_spacing'])+int(self.label['offset_y'])),
                                  texts[i], font=font, anchor='mm' if center_align else 'lm')
            hits = 0
            for x in np.linspace(bounds[0], bounds[2], 17):
                for y in np.linspace(bounds[1], bounds[3], 5):
                    point = self.data.geom_xpos[box]+rot@np.array([.05*(x/512-.5), .05*(.5-y/512), .025])
                    ray = point-camera
                    ray /= np.linalg.norm(ray)
                    geom = np.array([-1], dtype=np.int32)
                    mujoco.mj_ray(self.model, self.data, camera, ray, None, 1, -1, geom)
                    hits += int(geom[0] == box)
            result[field] = {'sampled_visible_fraction': hits/85, 'all_samples_exposed': hits == 85,
                             'method': '17x5 text bounding rectangle rays; approximate geometry only'}
        return result

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def baseline_body_y():
    return baseline()['body_position_m'][1]
