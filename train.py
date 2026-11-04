"""Train on train data, select the best epoch using validation loss, save one model."""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ROOT, DirectionDataset, read_json
from model import DirectionModel


def soft_cross_entropy(scores, target):
    return -(target * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()


@torch.inference_mode()
def validation_metrics(model, dataset):
    scores = model(dataset.x, dataset.legal)
    selected_targets = dataset.target[torch.arange(len(dataset)), scores.argmax(dim=-1)]
    return float(soft_cross_entropy(scores, dataset.target)), float((selected_targets > 0).float().mean())


def train(data_dir, results_dir, config):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results_dir.mkdir(parents=True, exist_ok=True)
    train_data = DirectionDataset(data_dir / 'train/samples.jsonl')
    val_data = DirectionDataset(data_dir / 'val/samples.jsonl')
    # The toy dataset is small, so each batch contains all training examples.
    loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    model = DirectionModel(config['hidden_dim'])
    with torch.no_grad():
        model.mean.copy_(train_data.x.mean(dim=0))
        std = train_data.x.std(dim=0, unbiased=False)
        model.scale.copy_(torch.where(std < 1e-6, torch.ones_like(std), std))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    best_loss, best_epoch, best_weights, stale = float('inf'), 0, None, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        for x, target, legal in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = soft_cross_entropy(model(x, legal), target)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss, val_accuracy = validation_metrics(model, val_data)
        history.append({'epoch': epoch, 'train_loss': float(loss.detach()),
                        'val_loss': val_loss, 'val_accuracy': val_accuracy})
        if val_loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 1 or epoch % 50 == 0:
            print(f'epoch {epoch:3d}: train_loss={loss.item():.5f} val_loss={val_loss:.5f} val_accuracy={val_accuracy:.1%}', flush=True)
        if stale >= config['patience']:
            break
    # This function never reads the test split.
    torch.save({'model_state_dict': best_weights, 'hidden_dim': config['hidden_dim'],
                'epoch': best_epoch, 'val_loss': best_loss, 'config': config}, results_dir / 'best_model.pt')
    with (results_dir / 'training.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(history)
    summary = {'seed': config['seed'], 'train_samples': len(train_data), 'val_samples': len(val_data),
               'selected_epoch': best_epoch, 'epochs_run': len(history), 'best_val_loss': best_loss,
               'parameters': sum(p.numel() for p in model.parameters()),
               'training_seconds': time.perf_counter() - started, 'selection': 'lowest validation loss'}
    (results_dir / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Saved {results_dir / "best_model.pt"} (epoch {best_epoch})', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    args = parser.parse_args()
    train(args.data_dir, args.results_dir, read_json(args.config))


if __name__ == '__main__':
    main()
