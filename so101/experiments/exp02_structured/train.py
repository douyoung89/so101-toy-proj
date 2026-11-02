"""Small deterministic CPU training; model selection reads train and val only."""
import copy
import time
import random

import numpy as np
import torch

from core import feature_names
from model import DirectionMLP
from common import read, rows, write, jsonl, sha
from prepare import verify_preserved


def tensors(records):
    if not records:
        raise ValueError('No supervised direction states')
    return (torch.tensor([r['x'] for r in records], dtype=torch.float32),
            torch.tensor([r['target'] for r in records], dtype=torch.float32),
            torch.tensor([r['legal'] for r in records], dtype=torch.bool))


def soft_ce(logits, target):
    return -(target*torch.log_softmax(logits, dim=-1)).sum(-1).mean()


@torch.inference_mode()
def classification(model, records):
    x, y, legal = tensors(records)
    logits = model(x, legal)
    prediction = logits.argmax(-1)
    optimal = y > 0
    accepted = optimal[torch.arange(len(y)), prediction]
    unique = optimal.sum(-1) == 1
    confusion = [[0,0],[0,0]]
    for target, pred in zip(y[unique].argmax(-1).tolist(), prediction[unique].tolist()):
        confusion[target][pred] += 1
    return {'states': len(records), 'soft_ce': float(soft_ce(logits, y)),
            'optimal_direction_accuracy': float(accepted.float().mean()),
            'unique_direction_states': int(unique.sum()), 'tie_states': int((~unique).sum()),
            'unique_direction_accuracy': float(accepted[unique].float().mean()) if unique.any() else None,
            'confusion_true_rows_pred_columns_left_right': confusion,
            'prediction_counts': {'LEFT_5': int((prediction==0).sum()), 'RIGHT_5': int((prediction==1).sum())}}


def train(run):
    if (run/'test_evaluation_started.json').exists() or (run/'selection.json').exists():
        raise ValueError('Selection already locked; create a new run instead of retraining')
    verify_preserved(run)
    experiment = read(run/'experiment.json')
    # Test data and test summaries are deliberately not opened here.
    train_rows, val_rows = rows(run/'data/train.jsonl'), rows(run/'data/val.jsonl')
    for split in ['train','val']:
        if sha(run/'data'/f'{split}.jsonl') != experiment['training_data_sha256'][split]:
            raise ValueError('Prepared data changed')
    seed, settings = experiment['seed'], experiment['training']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = DirectionMLP(experiment['architecture']['hidden'])
    x, y, legal = tensors(train_rows)
    with torch.no_grad():
        model.mean.copy_(x.mean(0))
        std = x.std(0, unbiased=False)
        model.scale.copy_(torch.where(std < 1e-6, torch.ones_like(std), std))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    best_loss, best_state, best_epoch, stale, log = float('inf'), None, 0, 0, []
    started = time.perf_counter()
    for epoch in range(1, settings['epochs']+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = soft_ce(model(x, legal), y)
        loss.backward()
        optimizer.step()
        model.eval()
        validation = classification(model, val_rows)
        log.append({'epoch':epoch, 'train_ce':float(loss.detach()), 'val':validation})
        if validation['soft_ce'] < best_loss-1e-8:
            best_loss, best_epoch, stale = validation['soft_ce'], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= settings['patience']:
            break
    model.load_state_dict(best_state)
    torch.save({'state_dict':model.state_dict(), 'hidden':experiment['architecture']['hidden'],
                'feature_names':feature_names(), 'selected_epoch':best_epoch}, run/'best.pt')
    jsonl(run/'training_history.jsonl', log)
    selection = {'seed':seed, 'selected_epoch':best_epoch, 'epochs_run':len(log),
                 'parameters':sum(p.numel() for p in model.parameters()),
                 'train':classification(model,train_rows), 'val':classification(model,val_rows),
                 'elapsed_seconds':time.perf_counter()-started,
                 'selection_criterion':settings['selection'], 'test_read_during_selection':False,
                 'checkpoint_sha256':sha(run/'best.pt'), 'experiment_sha256':sha(run/'experiment.json'),
                 'data_sha256':experiment['training_data_sha256']}
    write(run/'selection.json', selection)
    return selection
