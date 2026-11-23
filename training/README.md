# Training TactusNet

This directory holds the scripts and trainers used to train the TactusNet heads on TACTUS. The V-JEPA 2.1 encoder stays frozen: its tokens are computed once and cached, and every head trains on the cached tokens.

No trained weights are included. Result numbers are not reported here.

## Layout

| Directory | Variable | Contents |
| --- | --- | --- |
| [`pipeline/`](pipeline/README.md) | `$X` | Orchestration: data download, crop boxes, clip export, token caching, one script per experiment, evaluation and comparison. Also the benchmark split in `splits/gold_v2/`. |
| `unitap/` | `$U` | Spatial Head code (working name: UniTAP). `tools/` holds dataset builders, the pixel refiner, inference and evaluation. `third_party/t2b/kpt_combo/` is the trainer. `experiments/runs/{E007k,E009,C006,C004}/run.sh` are the recipes. `third_party/t2b/vjepa2/` is vendored V-JEPA 2 source. |
| `repro/` | | Semantic and Temporal Head: trainer in `action_range/code/kpt_combo/`, launchers `classification/train_stage{1,2}.sh` and `action_range/train_best.sh`. |

The whole directory is the code bundle, `$BUNDLE`.

Two things are kept as they were used for the reported results:

- **Two-directory setup.** The scripts were written for one persistent directory (`GV`: raw data, weights, token shards, saved state, results) and one fast local disk (`WORK`: everything being built or trained on). Steps work in `WORK` and copy what must survive to `GV`. Both can be directories on the same machine.
- **Two trainer copies.** `unitap/third_party/t2b/kpt_combo` trains the Spatial Head. `repro/action_range/code/kpt_combo` trains the Semantic and Temporal Heads. They are two snapshots of the same trainer and are not merged.

Naming: the code calls the paper's action window the clip or `affordance_range`, and the paper's interaction window the `action_range` (see [`data/README.md`](../data/README.md)). `case32/v1` in paths is the name of the exported dataset root, not a subset.

## Prerequisites

| Item | Source |
| --- | --- |
| Data | [hding2455/surgical-action-affordance](https://huggingface.co/datasets/hding2455/surgical-action-affordance) on Hugging Face. `pipeline/fill_volume.py` downloads it. |
| Encoder weights | `vjepa2_1_vitl_dist_vitG_384.pt`, the V-JEPA 2.1 ViT-L 384 checkpoint of the official V-JEPA 2 release. `fill_volume.py` downloads it and checks its SHA-256. Expected at `$GV/weights/`. |
| Python packages | [`pipeline/requirements.txt`](pipeline/requirements.txt). `run_all.sh` installs it unless `SKIP_PIP=1`. PyTorch with CUDA is not in the list and must already be installed. `fill_volume.py` also imports `requests`. |
| Tools | `bash`, `tar`, `curl`, `nvidia-smi`. |

Data used by the benchmark:

| Set | Videos | Clips |
| --- | ---: | ---: |
| Gold train | 79 | 12,372 |
| Gold val | 11 | 1,604 |
| Gold test | 11 | 1,523 |
| Silver (SurgVU) | 40 | 7,327 annotations in `index_silver.json` |

`pipeline/splits/gold_v2/` assigns whole videos to the three Gold partitions. The export step writes the final per-clip lists and a report of any annotation it could not export.

## Environment variables

`pipeline/env.sh` is sourced by every script.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GV` | `/workspace` | Persistent directory: `raw/`, `weights/`, `code.tar`, `export/`, `tokens/`, `state/`, `results/`. |
| `WORK` | `/work` | Fast local disk for all work in progress. |
| `BUNDLE` | `$WORK/code` | Unpacked code bundle (this directory). |
| `U` | `$BUNDLE/unitap` | Spatial Head code. Derived, not overridable. |
| `KC_RNG` | `$BUNDLE/repro/action_range/code` | Trainer of the Semantic and Temporal Heads. Derived. |
| `X` | `$BUNDLE/pipeline` | Pipeline scripts. Derived. |
| `HF_RAW` | `$WORK/hf` | Unpacked dataset: `<video>/{clips,annotation}` and the index files. Derived. |
| `EXPORT` | `$WORK/export` | Per-clip arrays shared by all experiments. Derived. |
| `VJEPA_CKPT` | `$GV/weights/vjepa2_1_vitl_dist_vitG_384.pt` | Encoder weights. |
| `PY` | `python` | Python interpreter. |
| `N_GPU` | `2` | Number of GPUs to use. |
| `CACHE_BATCH` | `4` | Batch size of token caching. |
| `WORKERS` | `16` | CPU workers for export and dataset building. |

`env.sh` also sets `PYTHONUNBUFFERED=1` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Switches read by single scripts:

| Variable | Script | Effect |
| --- | --- | --- |
| `PHASE=cpu` | `run_all.sh` | Run the CPU phase (default is the GPU phase). |
| `STOP_BEFORE_PACK=1` | `run_all.sh` | CPU phase: stop after the crop sheets are written. |
| `STOP_AFTER_PILOT=1` | `run_all.sh` | GPU phase: stop after the pilot. |
| `SKIP_PIP=1` | `run_all.sh` | Do not install `requirements.txt`. |
| `VOL_JOBS` | `run_all.sh`, `volume_shards.py` | Parallel reads and writes on `GV` (default 6). |
| `KB` | `run_exp.sh`, `pilot.sh` | Batch size of the Spatial Head trainers. |
| `SMOKE=1` | `run_exp.sh` | One epoch per step at a small batch (`SBATCH`, default 8). Only shows that the code runs. |
| `HF_TOKEN` | `fill_volume.py` | Optional Hugging Face token for a higher rate limit. |
| `DL` | `fill_volume.py` | Download scratch directory (default `/root/dl`). |

## Steps

Set the two directories first. All commands run from the repository root.

```bash
export GV=/path/to/persistent WORK=/path/to/fast_disk
```

**1. Pack the code bundle.** `run_all.sh` unpacks `$GV/code.tar` into `$WORK/code` when that directory does not exist, and `fill_volume.py` refuses to finish without the archive. It must contain `pipeline/`, `unitap/` and `repro/` at its top level:

```bash
mkdir -p "$GV" && tar -cf "$GV/code.tar" -C training pipeline unitap repro
```

After a change to the code, pack again and delete `$WORK/code`.

**2. Download data and weights.** No GPU needed. Re-run after any interruption.

```bash
pip install huggingface_hub requests
python training/pipeline/fill_volume.py
```

It writes one tar per video and the index files to `$GV/raw/`, the encoder weights to `$GV/weights/`, and finally `$GV/raw/READY`.

**3. CPU phase: crop boxes and clip export.**

```bash
PHASE=cpu bash training/pipeline/run_all.sh
```

It unpacks the raw data to `$HF_RAW`, computes one crop box per video, exports every Gold and Silver clip as 16 sampled frames with labels to `$EXPORT`, writes crop check sheets to `$GV/results/crop/`, and packs the export to `$GV/export/`.

**4. GPU phase: tokens, pilot, experiments A and B.**

```bash
bash training/pipeline/run_all.sh
```

It restores the export, caches V-JEPA 2.1 tokens for all clips, runs a short pilot to check GPU memory, runs `run_exp.sh A` and `run_exp.sh B`, and writes `$GV/results/compare.md`.

**5. Single experiments.** After the tokens exist, any experiment can be run or resumed on its own:

```bash
bash "$WORK/code/pipeline/run_exp.sh" A    # Gold train
bash "$WORK/code/pipeline/run_exp.sh" B    # Gold train + Silver
bash "$WORK/code/pipeline/run_exp.sh" C    # Silver only
```

Val and test are the same Gold clips in all three. `run_all.sh` runs A and B only.

Every step leaves a marker file, so a re-run continues where it stopped. On one machine, run steps 3 and 4 one after the other with the same `GV` and `WORK`.

## Steps of one experiment

`run_exp.sh E` builds `$WORK/exp/E/` as a complete copy of the Spatial Head code with its data links, then runs the steps below. Paths are relative to `$WORK/exp/E/`. Recipe paths are relative to `training/`.

| Step | Head | Recipe | Checkpoint written |
| --- | --- | --- | --- |
| `cls1` | Semantic Head, stage 1 | `repro/classification/train_stage1.sh` | `experiments/runs/cls1/checkpoints/best.pt` |
| `cls2` | Semantic Head, stage 2 (final) | `repro/classification/train_stage2.sh`, initialised from `cls1` | `experiments/runs/cls2/checkpoints/best.pt` |
| `interval` | Temporal Head | `repro/action_range/train_best.sh` | `experiments/runs/interval/checkpoints/best.pt` |
| `stage1` | Spatial Head, stage 1: tool points on the full frame | `unitap/experiments/runs/E007k/run.sh` | `experiments/runs/E007k/out/checkpoints/best.pt` |
| `stage2` | Spatial Head, stage 2: tool points on the zoomed crop | `unitap/experiments/runs/E009/run.sh` | `experiments/runs/E009/out/checkpoints/best.pt` |
| `contact` | Spatial Head, contact decoder on the crop | `unitap/experiments/runs/C006/run.sh` | `experiments/runs/C006/out/checkpoints/best.pt` |
| `refiner` | Spatial Head, pixel refiner for tool points | `unitap/tools/pixel_refiner.py train` | `experiments/runs/E008b/pixref/best.pt` |

Steps that build data or evaluate:

| Step | What it does | Output |
| --- | --- | --- |
| `s1_after` | Dumps stage-1 predictions on val and test, builds the refiner's error bank from the val dump. | `experiments/analysis/pred_E007k_best_{val,test}/`, `experiments/analysis/error_bank_E007j_val_384px.npy` |
| `zoom_build` | Builds the zoomed dataset: one square box per clip around the tool trajectory. Train boxes come from the labels with jitter, val and test boxes from the stage-1 predictions. | `data/case32/v1_zoom/` |
| `zoom_tok` | Caches V-JEPA 2.1 tokens of the zoomed clips and packs them to `$GV/tokens/zoom_E/`. | `data/case32/v1_zoom/vjepa_token_cache/` |
| `labels` | Builds both contact-target roots (see [CONTACT_SUPERVISION.md](CONTACT_SUPERVISION.md)). | `data/case32/v1_zoom_ct/`, `data/case32/v1_zoom_hs/` |
| `evaluate` | Scores every head on val and test and runs the full Spatial Head on test without cached tokens. | `results/` |

Order: `stage1` runs in parallel with `cls1 → cls2 → interval`. Then `zoom_build`, `zoom_tok`, `labels`. Then `stage2` in parallel with `contact`. Then `refiner` and `evaluate`.

Checkpoint selection: the Temporal Head keeps the epoch with the highest val tIoU. The pixel refiner keeps the epoch with the lowest gated tool error on the val dump. All other heads keep the epoch with the lowest val loss.

After each step, `persist.py` saves markers, best checkpoints, prediction dumps, results and logs to `$GV/state/E/state.tar`. At the end, `results/` and `logs/` are copied to `$GV/results/E/`.

The Spatial Head recipes can also be run one at a time from inside an experiment directory:

```bash
source "$WORK/code/pipeline/env.sh" && cd "$WORK/exp/A"
EXTRA_ARGS="--num-workers 8 --batch-size 24" bash "$X/recipe.sh" C006 0    # recipe id, GPU index
```

`recipe.sh` copies the recipe, sets its paths to the current directory, and runs it. Flags in `EXTRA_ARGS` override the recipe's own.

## Hardware notes

These are the statements made in the scripts.

- The Spatial Head recipes use batch 32, which needs about 28 GB of GPU memory. Measured: batch 8 uses 9 GB, batch 16 uses 15.5 GB.
- On a card with 30 GB or less, `run_exp.sh` uses batch 24 for the Spatial Head trainers. The learning rate stays as in the recipe. Set `KB` to choose another batch.
- `pilot.sh` runs the stage-1 recipe on every GPU at the same time, for up to 10 minutes at the real batch size. It fails if a trainer dies, for example from running out of memory, and prints the peak GPU memory.
- With `N_GPU=2`, an experiment runs two steps in parallel. With `N_GPU=4` or more, it uses four GPUs: Spatial Head on 0 and 3, Semantic Head on 1, Temporal Head then refiner on 2. With one GPU the steps run in sequence.
- Token caching runs one process per GPU at batch `CACHE_BATCH`.
- The encoder weights file is about 5.2 GB. Tokens are stored as fp16 arrays of shape `[8, 24, 24, 1024]`, about 9.4 MB per clip, once for the full frames and once per experiment for the zoomed crops. Plan for several hundred GB on `WORK`.

## Related documents

- [`pipeline/README.md`](pipeline/README.md): the pipeline scripts in detail, cropping, and what differs from the recipes.
- [`CONTACT_SUPERVISION.md`](CONTACT_SUPERVISION.md): the two ways to supervise the contact decoder.
- [`../evaluation/README.md`](../evaluation/README.md): the benchmark metrics and scoring script.
- [`../tactusnet/README.md`](../tactusnet/README.md): the architecture package.
