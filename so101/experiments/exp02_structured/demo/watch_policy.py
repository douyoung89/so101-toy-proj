"""Run best.pt closed-loop with fresh OCR; record continuous movement and optionally show a viewer."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import math
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT))

from core import causal_state, DIRECTIONS, DELTAS, FIELDS, MAX_MOVES
from model import DirectionActor
from common import read, rows, write, sha, complete, correct
from ocr_memory import InformationMemory
from prepare import verify_preserved
from simulator import Simulator

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


@contextmanager
def observe_steps(model, callback):
    """Single-threaded observation hook; original mj_step and physics parameters are unchanged."""
    original = mujoco.mj_step

    def step(m, d, *args, **kwargs):
        result = original(m, d, *args, **kwargs)
        if m is model:
            callback(d)
        return result

    mujoco.mj_step = step
    try:
        yield
    finally:
        mujoco.mj_step = original


class Display:
    def __init__(self, sim, output, fps, viewer=None, realtime=False):
        self.sim, self.output, self.fps, self.viewer = sim, output, fps, viewer
        self.realtime = realtime
        self.writer = imageio.get_writer(str(output/'policy_motion.mp4'), fps=fps, macro_block_size=2)
        self.frames = 0
        self.phase, self.action, self.moves = 'POSITIONING', '-', 0
        self.memory = InformationMemory().snapshot()
        self.next_frame = 0.
        font_path = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
        self.font = ImageFont.truetype(str(font_path), 21) if font_path.exists() else ImageFont.load_default(size=21)
        self.small = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default(size=17)
        self.overview = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.overview)
        self.overview.lookat[:] = [.05, 0, .14]
        self.overview.distance = .8
        self.overview.azimuth = 135
        self.overview.elevation = -20

    def frame(self):
        sim = self.sim
        sim.renderer.update_scene(sim.data, camera='label_camera')
        label = sim.renderer.render().copy()
        sim.renderer.update_scene(sim.data, camera=self.overview)
        wide = Image.fromarray(sim.renderer.render().copy()).resize((320,240))
        canvas = Image.new('RGB',(1280,720),(20,25,34))
        canvas.paste(Image.fromarray(label),(0,0))
        canvas.paste(wide,(960,0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0,0,960,42),fill=(20,25,34))
        draw.text((15,9),'EXP 2 | trained MLP | fresh MuJoCo + OCR',font=self.font,fill='white')
        angle = math.degrees(float(sim.data.qpos[sim.qadr[4]])-sim.reference)
        lines = [f'Phase: {self.phase}',f'Model: {self.action}',f'Wrist: {angle:+.2f} deg',
                 f'Moves: {self.moves} / {MAX_MOVES}','', 'OCR memory:']
        for field in FIELDS:
            m = self.memory['fields'][field]
            value = '?' if m['value'] is None else f'{m["value"]} g'
            lines.extend([f'{field}: {value}',f'  {m["status"]}'])
        lines += ['', 'Direction: MLP', 'STOP: memory rule', 'Images: display/OCR only', 'VAL scene demo']
        for i,line in enumerate(lines):
            draw.text((974,255+i*26),line,font=self.small,fill='white')
        self.writer.append_data(np.asarray(canvas))
        self.frames += 1
        canvas.save(self.output/'last_frame.png')
        if self.frames == 1:
            canvas.save(self.output/'first_frame.png')
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt('Viewer closed')
            self.viewer.sync()

    def moved(self, d):
        if float(d.time)+1e-9 < self.next_frame:
            return
        self.frame()
        self.next_frame = float(d.time)+1/self.fps
        if self.realtime:
            delay = self.wall_start+(float(d.time)-self.sim_start)-time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def move(self, angle):
        self.wall_start, self.sim_start = time.perf_counter(), float(self.sim.data.time)
        self.next_frame = self.sim_start
        with observe_steps(self.sim.model,self.moved):
            motion = self.sim.move(angle)
        self.frame()
        return motion

    def hold(self, seconds=1):
        for _ in range(round(seconds*self.fps)):
            self.frame()
            if self.realtime:
                time.sleep(1/self.fps)

    def close(self):
        self.writer.close()


def demo(args):
    torch.set_num_threads(1)
    run = args.run.resolve()
    dataset = Path(read(run/'experiment.json')['dataset'])
    selection = read(run/'selection.json')
    checkpoint = run/'best.pt'
    if sha(checkpoint) != selection['checkpoint_sha256']:
        raise ValueError('Checkpoint does not match the locked selection')
    preserved = verify_preserved(run)
    directory = dataset/'scenes'/args.scene
    scene, truth = read(directory/'scene.json'), read(directory/'truth.json')
    if scene['split'] == 'test':
        raise ValueError('Demo uses train/val scenes; do not repeat the locked test evaluation')
    plan = next(p for p in rows(dataset/'rollout_plan.jsonl')
                if p['scene_id']==args.scene and p['strategy']=='shortest')
    angle = plan['start_deg'] if args.start is None else args.start
    if angle not in scene['safe_angles_deg']:
        raise ValueError('Unsafe initial angle')
    config = read(dataset/'dataset_config.json')
    output = args.output or HERE/'outputs'/f'{args.scene}_{datetime.now(ZoneInfo("Asia/Seoul")):%Y%m%d_%H%M%S}'
    output = output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    actor = DirectionActor(checkpoint)
    sim = Simulator(directory/'simulation',truth['settings'],truth['label'])
    viewer = None
    display = None
    observations, actions = [], []
    memory = InformationMemory()
    termination = 'budget_exhausted'
    try:
        if args.viewer:
            import mujoco.viewer as mj_viewer
            viewer = mj_viewer.launch_passive(sim.model,sim.data,show_left_ui=False,show_right_ui=False)
            viewer.cam.lookat[:] = [.05,0,.14]
            viewer.cam.distance = .8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
        display = Display(sim,output,args.fps,viewer,args.viewer)
        arrival = display.move(angle)
        for step in range(MAX_MOVES+1):
            display.phase, display.action = 'OBSERVE + OCR','-'
            obs = sim.capture(output/'observations',f'obs_{step:02d}',config['thresholds'])
            obs['image'] = 'observations/'+obs['image']
            obs['policy_image'] = 'observations/'+obs['policy_image']
            obs.update(observation_id=f'obs_{step:02d}',sequence_index=step,
                       command_relative_deg=angle,arrival=arrival)
            obs['memory'] = memory.observe(obs['observation_id'],step,obs['readings'])
            observations.append(obs)
            display.memory, display.moves = obs['memory'],len(actions)
            print(f'obs {step}: angle={angle:+} memory='+str({f:obs['memory']['fields'][f]['value'] for f in FIELDS}),flush=True)
            if arrival['invalid_reasons']:
                termination,display.phase = 'technical_error','TECHNICAL ERROR'
                break
            if complete(obs['memory']):
                termination,display.phase,display.action = 'memory_complete','MEMORY COMPLETE','STOP (rule)'
                display.hold(2)
                break
            if step == MAX_MOVES:
                display.phase = 'BUDGET EXHAUSTED'
                display.hold(2)
                break
            state = causal_state(observations,actions)
            action,diagnostic = actor.direction(state)
            command = angle+DELTAS[DIRECTIONS.index(action)]
            j,a = sim.model.joint('wrist_roll').id,sim.aids[4]
            target = sim.reference+math.radians(command)
            low = max(sim.model.jnt_range[j,0],sim.model.actuator_ctrlrange[a,0])
            high = min(sim.model.jnt_range[j,1],sim.model.actuator_ctrlrange[a,1])
            if command not in scene['safe_angles_deg'] or not low <= target <= high:
                termination,display.phase = 'unsafe_direction_rejected','SAFETY REJECTION'
                break
            display.phase,display.action = 'MODEL DECISION',action
            print(f'  MLP -> {action} ({command:+} deg)',flush=True)
            display.hold(1)
            display.phase = 'MOVING'
            arrival = display.move(command)
            actions.append({'action':action,'decision_state':state,'diagnostic':diagnostic,'motion':arrival})
            angle = command
        result = {'purpose':'visual demonstration, not another held-out test evaluation',
                  'scene_id':args.scene,'split':scene['split'],'start_deg':observations[0]['command_relative_deg'],
                  'checkpoint_sha256':sha(checkpoint),'observations':observations,'actions':actions,
                  'stop_reason':termination,'moves':len(actions),'success':correct(observations[-1]['memory'],truth),
                  'wrong_stop':termination=='memory_complete' and not correct(observations[-1]['memory'],truth),
                  'video_frames':display.frames,'fps':args.fps,'video_seconds':display.frames/args.fps,
                  'original_preservation':preserved}
        write(output/'demo.json',result)
        if viewer is not None and args.hold:
            print('Demo finished. Close the MuJoCo window to exit.',flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(.05)
    finally:
        if display is not None:
            display.close()
        if viewer is not None:
            viewer.close()
        sim.close()
    write(output/'preservation.json',verify_preserved(run))
    print(output/'policy_motion.mp4',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=EXPERIMENT/'runs/pilot_20261102')
    p.add_argument('--scene',default='scene_000008')
    p.add_argument('--start',type=int)
    p.add_argument('--output',type=Path)
    p.add_argument('--fps',type=int,default=20)
    p.add_argument('--viewer',action='store_true',help='macOS: launch this script using mjpython')
    p.add_argument('--hold',action='store_true',help='keep the viewer open after the rollout')
    args = p.parse_args()
    if not 1 <= args.fps <= 60:
        p.error('--fps must be 1..60')
    if args.hold and not args.viewer:
        p.error('--hold requires --viewer')
    demo(args)


if __name__=='__main__':
    main()
