"""Keep tracked files below 5 MiB each and the checkout payload below 20 MiB."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z']).decode().split('\0')
sizes = [(name, (ROOT/name).stat().st_size) for name in paths if name]
large = [(name, size) for name, size in sizes if size > 5*1024**2]
total = sum(size for _, size in sizes)
print(f'{len(sizes)} tracked files, {total/1024**2:.2f} MiB; largest: {max(sizes, key=lambda item:item[1])}')
if large or total > 20*1024**2:
    raise SystemExit(f'Size budget exceeded: individual files={large}, total={total}')
