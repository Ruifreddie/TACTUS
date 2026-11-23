# Fine-tuning the encoder

TactusNet keeps its V-JEPA 2.1 encoder frozen. The code here fine-tunes that encoder inside TactusNet and answers two questions:

- Does fine-tuning help over the frozen encoder?
- Does learning all heads on one shared fine-tuned encoder pull a head below what it reaches with its own fine-tuned encoder?

## Rows

Every row starts from the trained heads of the Gold-only TactusNet run (experiment A in the pipeline, `training/pipeline`). This is LP-FT: linear-probe-then-fine-tune (Kumar et al., ICLR 2022). Every row trains 10 more epochs with the same clips, batches, schedule, code and selection rule. Two rows differ in one thing.

| Row | Encoder | Which losses reach the encoder | Selection |
| --- | --- | --- | --- |
| `tactusnet` | frozen | none (experiment A itself, epoch 0, scored through the same path) | A's own |
| `control` | frozen | none; heads train 10 more epochs | each head group at its best epoch |
| `joint` | 1 trainable, shared | all heads | one epoch for the whole model |
| `separate` | 3 trainable, one per head group | Semantic / Temporal / Spatial each alone | each at its best epoch |
| `joint_each` (supplement) | the joint run | all heads | each head group at its own best epoch |

- `control` vs `joint`: does unfreezing help? `joint` vs `separate`: does sharing pull a head down? `joint_each` vs `separate` separates the shared gradient from the one-epoch constraint.
- The rows are built from five training runs (`--run`): `control`, `joint`, and `sem`, `tmp`, `spa` (the three runs of the `separate` row).
- Heads: Semantic (`sem`, cls2 checkpoint, tool + action), Temporal (`tmp`, interval), Spatial (`s1` stage 1 on the full frame, `zm` zoom and `ct` contact on the zoom crop). Track Propagation has no trainable weights. The pixel refiner reads native crops, not tokens, so it stays A's.
- Verdict per metric: paired bootstrap over test videos (`baselines.common.score`); worse or better if the 95% CI of the difference excludes 0.

## Recipe (the same in every row)

- **Batch and clip stream.** 32 clips per optimizer step, 10 epochs over the 12,372 Gold training clips (387 steps per epoch). `--micro` sets the clips per forward and affects memory only; `run.sh` uses 32, one forward per step. The clip stream is drawn like the trainers' `WeightedRandomSampler` (pair weights, power 0.5, cap 10) from the seed and the epoch only (`data.epoch_plan`), so all runs see the same clips in the same batches.
- **Schedule.** One schedule for every weight: VideoMAE's `cosine_scheduler` shape scaled to 10 epochs (warmup 5/35 of the steps from 5e-4 x peak, cosine to 5e-4 x peak).
- **Encoder (when trainable).** VideoMAE ViT-L K400 fine-tuning (`encoder.py`): LR 2e-3 per 256 clips (2.5e-4 at 32), layer decay 0.75, drop path 0.1, AdamW (0.9, 0.999), weight decay 0.05 except 1-D parameters, biases and embeddings, no gradient clipping, EMA 0.999. Activation checkpointing saves memory and does not change values.
- **Heads.** Their recipe's AdamW, weight decay and clipping. The Semantic and Temporal Heads halve their LR for batch 32 (recipe batch 64 -> 2.5e-5). The point decoders keep 1e-4. EMA 0.999 for every head, also for the Temporal Head, whose recipe has none. The recipes' epoch-based schedules (token masking, stage curriculum) continue from the checkpoint's epoch.
- **Encoder gradient.** Sum over heads of `d loss_h / d tokens x 1 / (L0_h x groups x heads in the group)`. `L0_h` is each head's training loss at the start (`l0.json`, measured once and shared by all runs), so every head group counts equally.
- **Semantic Head.** One loss for both branches, tool CE 1.0 + action CE 1.0. The original recipes train the tool branch first and freeze it while the action branch trains.
- **Precision.** bf16 autocast in training. Validation and test tokens are plain fp32 with PyTorch's own TF32 defaults, as TactusNet's token converter ran (torch 2.8: cuDNN convolutions in TF32, matmuls not). `FT_FP32=strict` turns TF32 off (`data.fp32_mode`).
- **Validation.** After every epoch: each head's own `evaluate()` and criterion (`loss_total`, or `range_tiou` for the Temporal Head) at its recipe's batch size, with EMA weights. Multi-head runs choose one epoch by the mean over head groups of each head's criterion relative to epoch 0.
- **Guard.** Before any step, A's heads must reproduce A's recorded validation on the restored data (`epoch0_check` in `train.py`: 1% on the criterion; Semantic Head: accuracies within 0.5 points). Otherwise the run stops, unless `--no-epoch0-check` is given.

## Run

### Requirements

- A finished experiment A in the `training/pipeline/make_exp.py` layout (`--exp`): the best checkpoints of `cls2`, `interval`, `E007k`, `E009` and `C006`, and the zoom roots `data/case32/v1_zoom`, `v1_zoom_ct` and `v1_zoom_hs`.
- The export with `video_16_npy` for the Gold clips, and the raw clips.
- The V-JEPA 2.1 weight file (`VJEPA_CKPT` or `--vjepa-ckpt`).
- Environment variables: `KC_RNG` (`training/repro/action_range/code`, or `--kc-rng`), `BL_EXPORT` and `BL_HF_RAW` for evaluation and `collect` (or `--export`, `--hf-raw`), and the repository root on `PYTHONPATH`. The Spatial Head code (working name UniTAP) is read from `<exp>/third_party`, which `make_exp.py` copies from `training/unitap`.

### Commands

```bash
# once, before the runs: the starting loss of every head (shared by all runs)
python -m baselines.ft.train --measure-l0 --exp $EXP_A --out $FT/l0.json

# one run: control | joint | sem | tmp | spa
python -m baselines.ft.train --run joint --exp $EXP_A --l0 $FT/l0.json --out $FT/runs/joint [--micro 8]

# score one epoch snapshot; --epoch N | select:<combined|semantic|temporal|spatial>
python -m baselines.ft.evaluate all --exp $EXP_A --run-dir $FT/runs/joint --epoch select:combined --out $FT/eval/joint

# rows, paired bootstraps and verdicts
python -m baselines.ft.collect --ft $FT [--out $FT/results]
```

- A run writes `run.json`, `history.jsonl`, `selection.json`, `epochs/eNN/<head>.pt`, `epochs/eNN/encoder.pt` (not for `control`), `state_last.pt` and `log.txt` into `--out`. It resumes from `state_last.pt`.
- `collect` expects the layout that `run.sh` produces: `$FT/runs/<run>` and `$FT/eval/<run>/eNN`.
- `--pilot-steps N` stops a run after N steps of epoch 1 and writes `pilot.json` (seconds per step, memory).

A small test of the code path, with real flags and tiny sizes:

```bash
python -m baselines.ft.train --measure-l0 --exp $EXP_A --out $FT/l0.json --batch 4 --micro 2 --l0-micro 4 --workers 2
python -m baselines.ft.train --run joint --exp $EXP_A --l0 $FT/l0.json --out $FT/runs/joint \
       --epochs 2 --batch 4 --micro 1 --limit-clips 8 --workers 2 --val-batch 1
python -m baselines.ft.evaluate all --exp $EXP_A --run-dir $FT/runs/joint --epoch 2 --out $FT/eval/joint/e02
```

### All runs with `run.sh`

`run.sh` runs the whole experiment on one multi-GPU machine, one run per GPU:

```bash
bash run.sh prep_cpu        # optional, CPU only: decode the protocol frames once and pack them to $GV/ft/frames
bash run.sh prep            # code, export, raw clips, V-JEPA weight, frames, experiment A, zoom labels, frame check
bash run.sh pilot [GPU]     # L0, then 20 steps of the joint run after the epoch-0 check: $FT/pilot/pilot.json
nohup bash run.sh all > $WORK/ft_run.log 2>&1 &     # prep + L0 + every run + evaluation + collect
bash run.sh status
```

- Finished steps are skipped on a re-run (markers in `$FT/queue`). Training resumes from `state_last.pt`.
- Results: `$FT/results/ft_results_test.{json,md}`, copied to `$GV/results/ft`. The selected epoch snapshots are copied to `$GV/state/ft`.
- Settings: `FT_RUNS` (default `joint spa sem tmp control`), `FT_EPOCHS` (10), `FT_MICRO` (32), `FT_WORKERS` (8), `FT_VAL_BATCH` (4), `FT_EXTRA` (extra `train.py` flags for every run), `FRAMES_FROM` (`decode`, `export` or `shards`).
- `check_frames` (part of `prep`) encodes a few clips with the frozen encoder and compares the tokens with experiment A's token cache.

Limitation: `run.sh` assumes a two-directory layout, `GV` for persistent storage (default `/workspace-global`) and `WORK` for a fast local disk (default `/work`). Its `setup` step unpacks the code from two archives on `$GV` (`baselines_code.tgz`, `code.tar`) into fixed paths under `$WORK`. It also reads `results/baselines/export_gt.tgz`, `raw/*.tar`, `weights/vjepa2_1_vitl_dist_vitG_384.pt`, `state/A/state.tar` and `tokens/full` from `$GV`. The script that packs the code archives is not in this repository, so on another setup call the Python modules directly as shown above.

## Files

| File | What |
| --- | --- |
| `snapshots.py` | Both trainer snapshots in one process (module isolation) |
| `heads.py` | One head = its trainer's model, data, loss step, `evaluate()` and checkpoint format |
| `encoder.py` | Trainable V-JEPA 2.1 through TactusNet's wrapper, VideoMAE parameter groups and schedule, EMA |
| `data.py` | The shared clip stream, full frames and on-the-fly zoom crops, token cache writer |
| `train.py` | The runs, L0, validation, selection, snapshots, resume, pilot |
| `evaluate.py` | fp32 tokens, `baselines.ablate.heads`, the online Spatial Head pipeline, `baselines.common.score` |
| `collect.py` | Rows, paired bootstraps, verdicts |
| `frames.py` | Exact `video_16_npy` decoded from the raw clips |
| `export_frames.py` | `video_16_npy` taken from experiment A's full export |
| `check_frames.py` | Frozen-encoder tokens of the frames against experiment A's token cache |
| `run.sh` | All steps on one multi-GPU machine |
