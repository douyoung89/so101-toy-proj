"""Reproducible experiment 2. All outputs live in a separate run directory."""
import argparse
from pathlib import Path

from core import HERE, SO101


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare','train','evaluate','report','integrity'])
    p.add_argument('--run', type=Path, default=HERE/'runs/pilot_20261102')
    p.add_argument('--data', type=Path, default=SO101/'data/partial_v1')
    args = p.parse_args()
    run = args.run.resolve()
    if args.stage == 'prepare':
        from prepare import prepare
        print(prepare(args.data,run))
    elif args.stage == 'train':
        from train import train
        print(train(run))
    elif args.stage == 'evaluate':
        from evaluate import evaluate
        evaluate(run)
    elif args.stage == 'report':
        from report import report
        print(report(run))
    else:
        from prepare import verify_preserved
        print(verify_preserved(run))


if __name__ == '__main__':
    main()
