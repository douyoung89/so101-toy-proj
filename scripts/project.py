"""Portable entry points for the compact publication bundle (Python 3.12)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'so101/experiments/exp02_structured'
sys.path.insert(0, str(EXP))
import core  # Establish the original experiment's import path.
from common import read, rows, write, sha
from prepare import preserved_files


def verify(splits=('train', 'val', 'test')):
    manifest = read(ROOT/'assets/publication-manifest.json')
    for name, expected in manifest['sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Published file changed: {name}')
    from model import load_model
    from train import classification
    source = ROOT/'results/exp02'
    if sha(source/'best.pt') != read(source/'selection.json')['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from original selection')
    model = load_model(source/'best.pt')
    print(json.dumps({s: classification(model, rows(source/'data'/f'{s}.jsonl'))
                      for s in splits}, indent=2))
    print(f"Verified {len(manifest['sha256'])} published files; classification only, no new online test.")


def offline_train(output):
    verify(splits=('train', 'val'))
    output = output.resolve()
    if output.exists():
        raise ValueError('Use a new --output directory')
    source = ROOT/'results/exp02'
    experiment = read(source/'experiment.json')
    experiment['dataset'] = str(source/'data')
    experiment['publication_mode'] = 'retrain prepared structured features; no image audit or relabeling'
    for split in ['train','val']:
        target = output/'data'/f'{split}.jsonl'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/'data'/f'{split}.jsonl', target)
    write(output/'experiment.json', experiment)
    write(output/'source_inventory.json', preserved_files(source/'data'))
    from train import train
    print(json.dumps(train(output), indent=2))


def demo(output, viewer, hold):
    verify()
    output = output.resolve()
    if output.exists():
        raise ValueError('Use a new --output directory')
    if not (ROOT/'external/SO-ARM100/Simulation/SO101/assets').is_dir():
        raise ValueError('Run scripts/setup_assets.py first')
    from simulator import create_scene
    dataset, run = output/'dataset', output/'run'
    shutil.copytree(ROOT/'assets/demo', dataset)
    scene = dataset/'scenes/scene_000008'
    truth = read(scene/'truth.json')
    create_scene(scene/'simulation', truth['label'], truth['settings'])
    run.mkdir(parents=True)
    source = ROOT/'results/exp02'
    for name in ['best.pt','selection.json']:
        shutil.copy2(source/name, run/name)
    write(run/'experiment.json', {'dataset': str(dataset), 'development_only': True})
    write(run/'source_inventory.json', preserved_files(dataset))
    sys.path.insert(0, str(EXP/'demo'))
    from watch_policy import demo as watch
    watch(SimpleNamespace(run=run, scene='scene_000008', start=None,
                          output=output/'video', fps=20, viewer=viewer, hold=hold))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['verify','train','demo'])
    p.add_argument('--output', type=Path)
    p.add_argument('--viewer', action='store_true')
    p.add_argument('--hold', action='store_true')
    args = p.parse_args()
    if args.hold and not args.viewer:
        p.error('--hold requires --viewer')
    if args.command == 'verify':
        verify()
    elif args.command == 'train':
        offline_train(args.output or ROOT/'runtime/offline-training')
    else:
        demo(args.output or ROOT/'runtime/demo', args.viewer, args.hold)

if __name__ == '__main__':
    main()
