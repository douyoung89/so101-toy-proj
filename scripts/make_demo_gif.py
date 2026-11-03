"""Render the README GIF from the original demo (requires requirements-sim.txt)."""
from pathlib import Path
import subprocess
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
        '-i', str(ROOT/'docs/media/policy-motion.mp4'),
        '-filter_complex',
        'fps=10,scale=960:-1:flags=lanczos,split[a][b];'
        '[a]palettegen=max_colors=128:stats_mode=diff[p];'
        '[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle',
        '-loop', '0', str(ROOT/'docs/media/policy-motion.gif'),
    ], check=True)
