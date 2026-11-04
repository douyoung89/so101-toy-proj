# SO-101 local experiment

Seed: 20261104; selected epoch: 400; parameters: 2562.

Existing local raw observations and episodes were reorganized; teacher labels were regenerated.
The model was initialized and trained again. Test data was not used by train.py.
The split remains development-only: 12 train labels, 2 validation labels, 1 test label.

| Split | Samples | Direction accuracy | Always-right accuracy |
|---|---:|---:|---:|
| train | 81 | 100.0% | 40.7% |
| val | 14 | 100.0% | 100.0% |
| test | 5 | 100.0% | 100.0% |

| Mode | Split | Policy | Success | Mean moves | Wrong STOP | Technical errors |
|---|---|---|---:|---:|---:|---:|
| live | val | mlp | 2/2 | 3.50 | 0 | 0 |
| live | val | left_first | 0/2 | 8.00 | 0 | 0 |
| live | val | right_first | 2/2 | 3.50 | 0 | 0 |
| live | val | random | 0/2 | 8.00 | 0 | 0 |
| live | val | fixed_sweep | 0/2 | 8.00 | 0 | 0 |
| live | test | mlp | 1/1 | 3.00 | 0 | 0 |
| live | test | left_first | 0/1 | 8.00 | 0 | 0 |
| live | test | right_first | 1/1 | 3.00 | 0 | 0 |
| live | test | random | 0/1 | 8.00 | 0 | 0 |
| live | test | fixed_sweep | 1/1 | 3.00 | 0 | 0 |

live = fresh MuJoCo movement, rendering and OCR after each action.
replay = recorded OCR table transitions; these are not fresh physical observations.
All cases, including failures and technical errors, remain in the CSV denominator.
Rotation excludes the move to the initial pose. STOP is a memory rule, not an MLP output.
Validation/test teacher labels favor RIGHT_5; check the always-right column before interpreting accuracy.
One test scene does not establish generalization or superiority over simple rules.
