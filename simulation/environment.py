"""MuJoCo wrist control -> camera image -> EasyOCR, with no legacy project imports."""
import json
from pathlib import Path
import tempfile
import warnings
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from dataset import ROOT, ANGLES
from simulation import geometry
from simulation.memory import accepted, parse

ASSETS = ROOT / 'simulation/assets'
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
_READER = None


def recognize(image):
    global _READER
    if _READER is None:
        import easyocr
        import torch
        torch.set_num_threads(1)
        warnings.filterwarnings('ignore', message=".*pin_memory.*")
        warnings.filterwarnings('ignore', message='torch.quantize_per_tensor.*')
        _READER = easyocr.Reader(['en'], gpu=False, verbose=False, download_enabled=False,
                                 model_storage_directory=str(ASSETS / 'ocr_models'),
                                 user_network_directory=str(ASSETS / 'ocr_models/user_network'))
    # A fixed crop shared by all labels; return boxes in full-image coordinates.
    left, top, right, bottom = 290, 180, 670, 540
    output = _READER.readtext(image[top:bottom, left:right], detail=1, paragraph=False,
                              workers=0, batch_size=1, canvas_size=1280, mag_ratio=1.0)
    return [{'text': text, 'confidence': float(confidence),
             'bbox': [[float(x) + left, float(y) + top] for x, y in box]}
            for box, text, confidence in output]


class RobotEnvironment:
    def __init__(self, scene, render=True):
        self.scene = scene
        self.constants = json.loads((ASSETS / 'settings.json').read_text())
        self.settings = scene['settings']
        self.temporary = tempfile.TemporaryDirectory(prefix='so101-scene-')
        directory = Path(self.temporary.name)
        robot = ET.parse(ASSETS / 'robot.xml')
        robot.find('compiler').set('meshdir', str((ASSETS / 'meshes').resolve()))
        robot.find(".//geom[@name='partial_cover']").set('size', ' '.join(map(str, self.settings['half_size_m'])))
        robot.find("equality/joint[@joint1='label_shutter']").set('polycoef', '0 0 0 0 0')
        robot.write(directory / 'robot.xml')
        camera_scene = ET.parse(ASSETS / 'scene.xml')
        camera_scene.find("asset/texture[@name='label_texture']").set(
            'filefront', str((ASSETS / 'labels' / scene['label_image']).resolve()))
        camera_scene.write(directory / 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(str(directory / 'scene.xml'))
        self.data = mujoco.MjData(self.model)
        joint_ids = [self.model.joint(name).id for name in JOINTS]
        self.qpos_ids = self.model.jnt_qposadr[joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[joint_ids]
        self.actuator_ids = np.array([self.model.actuator(name).id for name in JOINTS])
        self.slide = self.model.jnt_qposadr[self.model.joint('label_shutter').id]
        self.equality = next(i for i in range(self.model.neq)
                             if self.model.eq_obj1id[i] == self.model.joint('label_shutter').id)
        self.goal = np.array(self.constants['reference_joints_rad'])
        self.reference = float(self.goal[4])
        self.pairs, self.fixture, self.payload = geometry.fixture_pairs(self.model)
        self.data.qpos[self.qpos_ids] = self.goal
        self.data.ctrl[self.actuator_ids] = self.goal
        self.couple()
        self.data.qpos[self.slide] = self.model.eq_data[self.equality, 0]
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=720, width=960) if render else None
        self.last_image = None

    def couple(self):
        """A clipped shutter target follows the wrist before every physics step."""
        settings = self.settings
        wrist = self.data.qpos[self.qpos_ids[4]]
        center = np.clip(settings['center_y_m'] + settings['slope_m_per_rad'] * (wrist - self.reference),
                         *settings['center_limits_y_m'])
        self.model.eq_data[self.equality, :5] = [center - self.constants['shutter_body_y'], 0, 0, 0, 0]

    def state(self):
        return {'qpos_rad': self.data.qpos[self.qpos_ids].tolist(),
                'qvel_rad_s': self.data.qvel[self.qvel_ids].tolist(),
                'actuator_targets_rad': self.data.ctrl[self.actuator_ids].tolist(),
                'simulation_time_s': float(self.data.time)}

    def safety_errors(self):
        view = geometry.camera_geometry(self.model, self.data, self.fixture, self.payload)
        errors = [name for name in ('box_in_frame', 'label_in_frame', 'fixture_in_frame') if not view[name]]
        if view['label_facing_cosine'] <= 0.15:
            errors.append('label_back_or_edge')
        if geometry.geometric_penetrations(self.model, self.data, self.pairs):
            errors.append('geometry_overlap')
        if geometry.contacts(self.model, self.data):
            errors.append('enabled_contact')
        return errors

    def move(self, angle, on_frame=None):
        if angle not in ANGLES:
            raise ValueError(f'Angle outside the 5-degree action grid: {angle}')
        target = self.goal.copy()
        target[4] = self.reference + np.deg2rad(angle)
        joint, actuator = self.model.joint('wrist_roll').id, self.actuator_ids[4]
        lower = max(self.model.jnt_range[joint, 0], self.model.actuator_ctrlrange[actuator, 0])
        upper = min(self.model.jnt_range[joint, 1], self.model.actuator_ctrlrange[actuator, 1])
        if not lower <= target[4] <= upper:
            raise ValueError('Mechanical wrist limit exceeded')
        before = self.state()
        start = self.data.ctrl[self.actuator_ids].copy()
        elapsed, stable, steps = 0.0, 0.0, 0
        errors = set()
        while elapsed < 7:
            progress = min(elapsed / 1.5, 1.0)
            blend = progress * progress * (3 - 2 * progress)
            self.data.ctrl[self.actuator_ids] = start + blend * (target - start)
            self.couple()
            mujoco.mj_step(self.model, self.data)
            elapsed += self.model.opt.timestep
            steps += 1
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError('Non-finite physics state')
            if steps % 5 == 0:
                mujoco.mj_forward(self.model, self.data)
                errors.update(self.safety_errors())
            if on_frame is not None and steps % 50 == 0:
                self.renderer.update_scene(self.data, camera='label_camera')
                on_frame(self.renderer.render().copy())
            error = float(np.max(np.abs(self.data.qpos[self.qpos_ids] - target)))
            speed = float(np.max(np.abs(self.data.qvel[self.qvel_ids])))
            slide_error = abs(self.data.qpos[self.slide] - self.model.eq_data[self.equality, 0])
            stable = stable + self.model.opt.timestep if progress == 1 and error < 0.003 and speed < 0.01 and slide_error < 0.0001 else 0.0
            if stable >= 0.3:
                break
        if stable < 0.3:
            errors.add('move_failed')
        mujoco.mj_forward(self.model, self.data)
        errors.update(self.safety_errors())
        return {'before': before, 'after': self.state(), 'invalid_reasons': sorted(errors)}

    def capture(self):
        if self.renderer is None:
            raise RuntimeError('Camera rendering is disabled')
        self.renderer.update_scene(self.data, camera='label_camera')
        self.last_image = self.renderer.render().copy()
        raw = recognize(self.last_image)
        return {'joint_state': self.state(),
                'relative_actual_deg': float(np.rad2deg(self.data.qpos[self.qpos_ids[4]] - self.reference)),
                'ocr': raw, 'readings': accepted(parse(raw), self.constants['thresholds'])}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.temporary.cleanup()
