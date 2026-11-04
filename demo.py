"""Record the newly trained model on a validation scene as results/demo.gif."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import imageio.v2 as imageio
import imageio_ffmpeg
import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import ROOT, ACTIONS, DELTAS, MAX_MOVES, memory_complete, memory_correct, read_json
from model import load_model, predict_action
from simulation.environment import RobotEnvironment
from simulation.memory import InformationMemory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='scene_000008')
    args = parser.parse_args()
    scene = next(s for s in read_json(ROOT / 'data/scenes.json') if s['scene_id'] == args.scene)
    if scene['split'] == 'test':
        raise ValueError('Use a train or validation scene for demonstration')
    torch.set_num_threads(1)
    model = load_model(ROOT / 'results/best_model.pt')
    memory = InformationMemory()
    observations, actions = [], []
    status = {'phase': 'POSITIONING', 'memory': memory.snapshot()}
    environment = RobotEnvironment(scene)
    angle = scene['start_deg']
    output = ROOT / 'results/demo.gif'
    font_path = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default(size=20)
    try:
        with tempfile.TemporaryDirectory(prefix='so101-demo-') as directory:
            video = Path(directory) / 'demo.mp4'
            with imageio.get_writer(video, fps=10, codec='libx264', macro_block_size=1) as writer:
                def record(image):
                    frame = Image.fromarray(image).resize((800, 600))
                    draw = ImageDraw.Draw(frame)
                    actual = math.degrees(environment.state()['qpos_rad'][4] - environment.reference)
                    values = status['memory']['fields']
                    draw.rectangle((0, 0, 800, 78), fill='#10151e')
                    draw.text((12, 5), 'Fresh trained MLP | VAL | ' + status['phase'], font=font, fill='white')
                    draw.text((12, 30), f'Wrist: {actual:+.1f} deg   Moves: {len(actions)}/8', font=font, fill='white')
                    text = '   '.join(f'{name}: {values[name]["value"] if values[name]["value"] is not None else "?"}'
                                      for name in ('protein_g', 'sugar_g'))
                    draw.text((12, 54), text, font=font, fill='white')
                    writer.append_data(np.asarray(frame))

                arrival = environment.move(angle, record)
                reason = 'budget_exhausted'
                for step in range(MAX_MOVES + 1):
                    observation = environment.capture()
                    observation.update(step=step, wrist_deg=angle, action=None)
                    observation['memory'] = memory.observe(f'obs_{step}', step, observation['readings'])
                    observations.append(observation)
                    status.update(phase='OBSERVE + OCR', memory=observation['memory'])
                    for _ in range(10):
                        record(environment.last_image)
                    if arrival['invalid_reasons']:
                        raise RuntimeError(f'Unsafe motion: {arrival["invalid_reasons"]}')
                    if memory_complete(observation['memory']):
                        observation['action'] = 'STOP'
                        status['phase'] = 'STOP (memory rule)'
                        reason = 'memory_complete'
                        break
                    if step == MAX_MOVES:
                        status['phase'] = 'BUDGET EXHAUSTED'
                        break
                    action = predict_action(model, observations, actions)
                    observation['action'] = action
                    target = angle + DELTAS[ACTIONS.index(action)]
                    if target not in scene['safe_angles']:
                        raise RuntimeError('Unsafe direction rejected')
                    status['phase'] = 'MODEL: ' + action
                    for _ in range(5):
                        record(environment.last_image)
                    arrival = environment.move(target, record)
                    actions.append(action)
                    angle = target
                for _ in range(20):
                    record(environment.last_image)
            filters = ('fps=10,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];'
                       '[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle')
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
                            '-i', str(video), '-filter_complex', filters, '-loop', '0', str(output)], check=True)
    finally:
        environment.close()
    result = {'scene_id': scene['scene_id'], 'split': scene['split'], 'moves': len(actions),
              'success': memory_correct(observations[-1]['memory'], scene['truth']), 'stop_reason': reason,
              'actions': actions, 'observations': observations, 'gif': str(output.relative_to(ROOT))}
    (ROOT / 'results/demo.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Demo: {output}; moves={len(actions)}; success={result["success"]}', flush=True)


if __name__ == '__main__':
    main()
