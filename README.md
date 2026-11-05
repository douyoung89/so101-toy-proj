# SO-101: Learning a Wrist Direction Policy

A robot rotates its wrist in 5° steps to read partially occluded nutrition labels. OCR readings accumulate in memory, and a small MLP chooses the next movement direction. A separate memory rule issues STOP once both values are confirmed.

![MuJoCo demonstration using the trained model](results/demo.gif)

## Running the experiment

The experiment was run with **Python 3.12.0 in the `so101` Conda environment**. Training data and original observation images are included in the repository.

```bash
conda activate so101
./run_experiment.sh
```

The script runs the following steps:

1. Extract observation images, robot meshes, and synthetic labels, and download OCR weights if needed.
2. Build direction targets and 77-dimensional model inputs from recorded episodes and observation tables.
3. Train a model from scratch and save the checkpoint with the lowest validation loss.
4. Evaluate the model and rule-based baselines on validation and test scenes using MuJoCo movement, fresh camera frames, and fresh OCR.
5. Generate a validation-scene demonstration GIF using the new checkpoint.

Each run overwrites files in `results/`. Files are not protected by hash checks or write locks. Run the script in a regular macOS terminal with graphics access. An `invalid CoreGraphics connection` error indicates a graphics-access problem in the execution environment.

To create the environment for the first time:

```bash
conda create -n so101 python=3.12
conda activate so101
python -m pip install -r requirements.txt
```

The script uses `python` from the active environment. To specify an interpreter, use `PYTHON=/path/to/python ./run_experiment.sh`. The initial OCR weight download requires internet access.

For a faster evaluation using recorded observations:

```bash
python evaluate.py --mode replay
```

This command overwrites the CSV evaluation results. Replay uses stored OCR observations rather than fresh camera frames and must be distinguished from the default `live` evaluation. Run `./run_experiment.sh` again to generate new live results.

## Project structure

```text
so101-toy-proj/
  dataset.py                 # Data loading and 77-dimensional input encoding
  model.py                   # 77 -> 32 -> 2 MLP and action selection
  train.py                   # Training, validation selection, best_model.pt
  evaluate.py                # Classification and online evaluation, CSV output
  demo.py                    # Demonstration GIF from the trained checkpoint
  config.json                # Seed, learning rate, epochs, and other settings
  requirements.txt           # pip freeze from the experiment environment
  run_experiment.sh          # Full experiment pipeline
  data/
    images/                  # Original PNG archives, extracted here at runtime
    train/episode_*.json      # Training episodes
    val/episode_*.json        # Validation episodes
    test/episode_*.json       # Test episodes
    train/samples.jsonl       # Encoded training inputs and teacher targets
    val/samples.jsonl
    test/samples.jsonl
    scenes.json              # Scene settings, ground truth, angle-wise OCR tables
    source.json              # Source data locations and image-sharing statistics
    summary.json             # Sample counts and exclusion reasons
  simulation/
    environment.py           # Robot movement -> rendering -> OCR
    memory.py                # OCR parsing, accumulated memory, conflict resolution
    geometry.py              # Frame bounds, penetration, and contact checks
    assets/                  # Reference XML, labels, meshes, and OCR weights
  scripts/
    import_data.py           # Import another source dataset into this layout
    setup_assets.py          # Extract meshes and labels, prepare OCR weights
    prepare_data.py          # Generate teacher targets using BFS
  results/                   # Result files stored in one directory
  tests/test_pipeline.py     # Causal inputs, memory, teacher, and stopping tests
```

A suggested reading order is **`evaluate.py:run_episode()` → `dataset.py:encode_history()` → `model.py` → `train.py` → `scripts/prepare_data.py`**. See `simulation/environment.py` for movement and rendering details.

## Dataset

The dataset comes from an earlier local SO-101 collection experiment. The recorded observation tables and collection episodes are reused. This pipeline organizes the source data, rebuilds teacher targets, trains a new model, and runs fresh online evaluations and demonstrations.

The source contains 15 scenes, 345 angle observations, and 60 episodes. Scenes are split by label into 12 training, 2 validation, and 1 test scene, with 48/8/4 episodes respectively. Relabeling and merging duplicate states produces 81/14/5 model input samples.

An episode is an **ordered sequence of observations taken at different wrist angles in the same scene**. The `observations` array in files such as `data/train/episode_000005.json` records that sequence.

```json
{
  "episode_id": "example",
  "scene_id": "scene_000001",
  "observations": [
    {"step": 0, "image": "data/images/frame_000001.png", "wrist_deg": 0, "action": "RIGHT_5"},
    {"step": 1, "image": "data/images/frame_000002.png", "wrist_deg": 5, "action": "STOP"}
  ]
}
```

This JSON illustrates the structure. Actual files also contain OCR readings, accumulated memory, joint states, and movement information on arrival. Paths are relative to the project root. Each `action` is selected after its observation; termination without an action, such as budget exhaustion, is recorded as `null`.

The 718 image references share 356 original PNG files, reducing image storage from about 90.6 MB to 44.6 MB. Only byte-identical files are shared. **Observation steps and IDs remain separate**, so sharing an image path does not merge time steps or memory histories. Original images are preserved; reproducible 224×224 derivatives are omitted.

Ground truth and future observation tables in `data/scenes.json` are used for teacher generation and final evaluation. Model inputs produced by `dataset.py` exclude images, numeric OCR readings, ground truth, future observations, scene IDs, and collection strategies.

## Training and results

The random seed and training settings are defined in `config.json`. The model learns to predict LEFT_5 or RIGHT_5 from a 77-dimensional state. STOP decisions and joint position control are handled separately. Normalization statistics are computed on the training split, and the best epoch is selected by validation loss. `train.py` does not read the test split.

| Output | Contents |
|---|---|
| `results/best_model.pt` | Selected model weights, normalization statistics, and training settings |
| `results/training.csv` | Per-epoch training/validation loss and validation accuracy |
| `results/training_summary.json` | Selected epoch, parameter count, and training time |
| `results/classification.csv` | Train/validation/test direction classification and an always-right baseline |
| `results/metrics.csv` | Per-policy validation/test success, movement, incorrect STOP decisions, and technical errors |
| `results/episodes.csv` | Individual evaluation episode results |
| `results/evaluation.jsonl` | Full observation and action logs from fresh online evaluation |
| `results/report.md` | Result tables and interpretation limits |
| `results/demo.gif`, `results/demo.json` | Validation demonstration and detailed logs |

These labels were previously used during development, and the test split contains only one scene. All validation and test teacher directions are RIGHT_5. A classification accuracy of 100% or a test success of 1/1 does not establish generalization to new labels or superiority over the rule-based baselines. See the [experiment report](results/report.md) for detailed results.

## Environment and assets

`requirements.txt` contains 46 packages exported with `python -m pip freeze` from the `so101` environment used for the experiment. It can be installed in a matching Python 3.12 environment. The experiment used the existing environment; installation into a separate clean environment has not been verified.

The repository includes reference XML files, the robot meshes used in the experiment in `meshes.zip`, and 15 synthetic labels in `labels.zip`. The run script extracts them automatically. EasyOCR weights, approximately 94 MB, are downloaded on first use and are not tracked in Git. No Python code from the earlier project or machine-specific paths are required.

All 356 original PNGs referenced by episodes are included in `data/images/images_01.zip` through `images_08.zip`. The archives are split to fit GitHub API upload limits. On first run, they are extracted into `data/images/`. See the [image documentation](data/images/README.md). Shared source images occupy about 44.6 MB, and the published files total about 58 MB.

The included dataset does not require an import step. To import another local collection, use `python scripts/import_data.py --source /path/to/original/data`. This command rebuilds the data files.

The robot model comes from TheRobotStudio/SO-ARM100. The reference XML has been modified to add a label, hood, occluder, and camera. Its Apache 2.0 license is included in `simulation/assets/ROBOT_LICENSE.txt`.

To run the tests:

```bash
python -m unittest discover -s tests -v
```
