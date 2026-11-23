# Swapping the backbone

This directory retrains TactusNet's heads on the tokens of another frozen video or image encoder. Use it to test your own encoder under the TactusNet protocol.

## What the swap does

1. `tokens.py` runs a frozen encoder once over every Gold clip and stores one token file per clip.
2. `run.sh` lays out an experiment directory whose token cache points at those files, then runs the unchanged TactusNet training recipes on it.
3. `heads.py` and `convert.py` turn the trained heads' outputs into standard prediction files, which `baselines.common.score` scores like every other row.

| Retrained on the new tokens | Recipe that `run.sh` calls |
| --- | --- |
| Semantic Head (two stages, `cls1` then `cls2`) | `$BUNDLE/repro/classification/train_stage1.sh`, `train_stage2.sh` |
| Temporal Head | `$BUNDLE/repro/action_range/train_best.sh` |
| Stage-1 tool decoder of the Spatial Head (full frame) | `$U/experiments/runs/E007k/run.sh` |

What stays fixed:

- The data: train on Gold train, select on Gold val, predict Gold test. `training/pipeline/make_exp.py` writes the split lists.
- The recipes, their flags and the trainer code. `pyshim.sh` and `launch.py` stand in for the recipes' Python, so no recipe file is edited.
- The frame preprocessing (see the token contract). Only the encoder and its input size change.
- The encoder is frozen. No trainer ever runs it: every token comes from the cache, and a cache miss is an error (`launch.py`, `ABLATE_STUB_ENCODER`).

The zoom decoder, the contact decoder and the pixel refiner of the Spatial Head are not retrained by the swap jobs.

## Token contract

`tokens.py extract` writes `<out>/all/<sid>.npy`, one file per clip. `tokens.py check` and the trainers rely on this layout.

| Property | Requirement |
| --- | --- |
| Shape | `[8, grid, grid, dim]`: time, then rows, then columns. The grid is square. |
| Dtype | `float16`, all values finite |
| Time steps | 8. Step `k` covers protocol frames `2k` and `2k+1` of the 16 protocol frames. |
| `dim` | 1024 or 768. These are the two widths that the stub encoders and `heads.py` map (`vjepa2_1_vit_large_384`, `vjepa2_1_vit_base_384`). |
| `grid` | Read from the tokens by the heads. The registered encoders use 14, 16 and 24. |

Input and preprocessing (`backbones.Encoder._prep`):

- Input: the 16 protocol frames of a clip, `uint8` RGB `[16, H, W, 3]`, content crop at native resolution (`ex.frames(sid)` in `baselines/common/data.py`).
- Divide by 255, bilinear resize to `size x size` (`align_corners=False`, no antialias, float32). The frame is squashed, not cropped.
- Normalize with the ImageNet mean and std, unless the spec gives its own `mean` and `std`.
- The encoder then receives `[B, 16, 3, size, size]` and must return `[B, 8, grid, grid, dim]`.

An image model yields 16 maps. Average frames `2k` and `2k+1` so that it hands over the same 8 time steps as a video model with tubelet size 2. The registered per-frame encoders do this in `Encoder._forward`.

## Registered encoders

`SPECS` in `backbones.py`:

| Name | Model | Input size | Token grid (`8 x grid x grid x dim`) |
| --- | --- | ---: | --- |
| `vjepa21_l384` | V-JEPA 2.1 ViT-L/16, distilled from ViT-G (TactusNet's encoder) | 384 | 8 x 24 x 24 x 1024 |
| `vjepa21_l256` | The same model run at 256 px | 256 | 8 x 16 x 16 x 1024 |
| `vjepa2_l256` | V-JEPA 2 ViT-L/16 (`facebook/vjepa2-vitl-fpc64-256`) | 256 | 8 x 16 x 16 x 1024 |
| `dinov3_l16_384` | DINOv3 ViT-L/16, LVD-1689M, per frame | 384 | 8 x 24 x 24 x 1024 |
| `dinov2_l14_336` | DINOv2 ViT-L/14, LVD-142M, per frame | 336 | 8 x 24 x 24 x 1024 |
| `lemonfm_384` | LemonFM ConvNeXt-L (`visurg/LemonFM`), stage-3 map, per frame | 384 | 8 x 24 x 24 x 768 |
| `videomae_b224` | VideoMAE ViT-B/16 (`MCG-NJU/videomae-base`) | 224 | 8 x 14 x 14 x 768 |
| `videomae_l224` | VideoMAE ViT-L/16 (`MCG-NJU/videomae-large`) | 224 | 8 x 14 x 14 x 1024 |
| `endofm_b224` | Endo-FM ViT-B/16 TimeSformer, own mean 0.45 and std 0.225 | 224 | 8 x 14 x 14 x 768 |

`run.sh` runs `vjepa2_l256 videomae_l224 endofm_b224` by default (`ABL_BACKBONES`).

### Weights and licences

The `license` field of `SPECS` records the following. Check the upstream terms before you use or redistribute any weight.

| Name | Weights | `license` in `SPECS` |
| --- | --- | --- |
| `vjepa21_l384`, `vjepa21_l256` | Local file `$VJEPA_CKPT` (`vjepa2_1_vitl_dist_vitG_384.pt`) | not recorded |
| `vjepa2_l256` | Hugging Face, pinned revision | `mit` |
| `dinov3_l16_384` | Hugging Face (`timm/vit_large_patch16_dinov3.lvd1689m`), pinned revision | `DINOv3 License` |
| `dinov2_l14_336` | Hugging Face (`timm/vit_large_patch14_dinov2.lvd142m`), pinned revision | `apache-2.0` |
| `lemonfm_384` | Hugging Face, pinned revision, file `lemonfm.pth` | `apache-2.0` |
| `videomae_b224`, `videomae_l224` | Hugging Face, pinned revision | `cc-by-nc-4.0` |
| `endofm_b224` | `$BL_WEIGHTS/cls/endofm/endo_fm_teacher_backbone.pth`, fetched and extracted by `baselines.cls.weights.endofm_backbone` | `Apache-2.0 (code); weights: Endo-FM release` |

Hugging Face files are downloaded into `$HF_HOME`.

## Add your own encoder

### 1. Register it in `SPECS`

Add one entry to `SPECS` in `backbones.py`. The key is the name you pass to `--backbone`.

| Field | Needed | Meaning |
| --- | --- | --- |
| `kind` | yes | Which builder and forward branch to use (next step). |
| `size` | yes | Input side in pixels. Frames are squashed to `size x size`. |
| `grid` | yes | Side of the token grid. |
| `dim` | yes | Token width, 1024 or 768. |
| `stub` | yes | `vjepa2_1_vit_large_384` for `dim=1024`, `vjepa2_1_vit_base_384` for `dim=768`. |
| `per_frame` | by convention | `True` for an image model whose frame pairs are averaged. No code reads it. |
| `desc` | by convention | One line, copied into the manifest. |
| `mean`, `std` | optional | Normalization, if not ImageNet's. |
| `license` | optional | Licence of the weights, copied into the manifest. |
| `repo`, `revision` | per kind | Hugging Face repo and pinned revision (`vjepa2_hf`, `videomae_hf`, `timm_vit`, `lemonfm`). |
| `model` | `timm_vit` | timm model name. |
| `file` | `lemonfm` | Weight file name in the repo. |
| `arch` | `vjepa_repo` | Architecture name for TactusNet's V-JEPA wrapper. |
| `weights` | optional | Where the weight file lives, for the manifest. |

What `stub` is for: the trainers still build an encoder object, because the head's input layers take their width from it. With `ABLATE_STUB_ENCODER=<arch>`, `launch.py` makes the trainer build a randomly initialized `<arch>` without loading any weight. It is never run. It only sets the token width of the heads (1024 or 768).

`run.sh` does not read `stub` from `SPECS`. Its `_stub` function repeats the mapping by name, so add a 768-wide encoder to the `case` list in `_stub` as well.

### 2. Pick a `kind`, or implement a new one

Existing kinds. Reuse one if your model loads the same way:

| `kind` | Loads | Forward |
| --- | --- | --- |
| `vjepa_repo` | TactusNet's own wrapper (`$U/third_party/t2b/kpt_combo/vjepa_wrapper.py`) with `$VJEPA_CKPT` | Native `[B, 8, g, g, D]` |
| `vjepa2_hf` | `transformers.VJEPA2Model`, encoder only | `last_hidden_state` reshaped |
| `videomae_hf` | `transformers.VideoMAEModel` | `last_hidden_state` reshaped |
| `timm_vit` | `timm.create_model` with the pinned `model.safetensors` | Per frame, prefix tokens dropped, frame pairs averaged |
| `lemonfm` | torchvision ConvNeXt-L with `lemonfm.pth` | Per frame, stage-3 map, frame pairs averaged |
| `endofm` | The vendored Endo-FM TimeSformer of `baselines/cls` | 16 frames, CLS dropped, frame pairs averaged |

For a new kind, edit three places:

1. `Encoder.__init__`: add a branch that calls your builder. The builder must set `self.net` to a `torch.nn.Module`. `__init__` then moves it to the device, sets `eval()` and freezes every parameter.
2. `Encoder._forward`: add a branch that maps `x` `[B, 16, 3, size, size]` to `tok` `[B, 8, grid, grid, dim]`. The method raises if the shape differs.
3. `cmd_prefetch` in `tokens.py`: add your kind to the `files` dict (the files to download from `repo`, or `[]`). `prep_ablate` calls `prefetch` for every name in `ABL_BACKBONES` and fails on an unknown kind.

`Encoder.encode` handles autocast, the finite check and the cast to `float16`.

### 3. Check the registration

```bash
python -m baselines.ablate.tests.test_ablate
```

The first check verifies that every `stub` matches its `dim`. It needs no GPU and no data. The last check of this script reads `table_ablation.tex`, which is not in this directory, so the script stops there.

## Run it

All commands run from the repository root with the repository on `PYTHONPATH` (`run.sh` sets this itself).

### Environment variables

`run.sh` reuses the names of `training/pipeline/env.sh` and of `baselines/common/config.py`.

| Variable | Default in `run.sh` | Meaning |
| --- | --- | --- |
| `GV` | `/workspace` | Persistent storage: weights, saved experiment state, copies of the results |
| `WORK` | `/work` | Fast local disk: export, tokens, experiment directories |
| `BUNDLE` | `$WORK/code` | The training code. In this repository: `training/` |
| `U` | `$BUNDLE/unitap` | Spatial Head code (working name UniTAP); needs `third_party/t2b/vjepa2` and `third_party/t2b/kpt_combo` |
| `X` | `$BUNDLE/pipeline` | Pipeline scripts (`make_exp.py`, `persist.py`, `shards.py`) |
| `KC_RNG` | `$BUNDLE/repro/action_range/code` | Trainer code of the Semantic and Temporal Heads |
| `EXPORT` | `$WORK/export` | Per-clip export (`meta.json`, `labels.json`, `all_<split>.txt`, arrays) |
| `HF_RAW` | `$WORK/hf` | Raw clips in the dataset's layout |
| `VJEPA_CKPT` | `$WORK/weights/vjepa2_1_vitl_dist_vitG_384.pt` | V-JEPA 2.1 weight file |
| `PY` | `python` | Interpreter of the recipes |
| `BL_EXPORT`, `BL_HF_RAW` | `$EXPORT`, `$HF_RAW` | What `baselines.common.data.get_export` reads |
| `BL_TOKENS` | `$WORK/tok_full/all` | TactusNet's own V-JEPA 2.1 tokens, `<sid>.npy` |
| `BL_OUT` | `$WORK/baselines` | Outputs of the baselines |
| `BL_WEIGHTS` | `$GV/weights/baselines` | Downloaded weights |
| `BL_PY` | `$PY` | Interpreter of the `baselines` modules |
| `HF_HOME` | `$BL_WEIGHTS/hf` | Hugging Face cache |
| `AB` | `$WORK/ablate` | Tokens (`$AB/tok/<name>`), experiment dirs (`$AB/exp/<name>`), step markers |
| `AO` | `$BL_OUT/ablate` | Prediction files, logs, queue state |
| `EXP_A`, `EXP_B` | `$WORK/exp/A`, `$WORK/exp/B` | Restored TactusNet experiments (Gold only, Gold + Silver) |
| `ABL_BACKBONES` | `vjepa2_l256 videomae_l224 endofm_b224` | Encoders to run, space-separated |
| `ABL_S1_SCHEDULE` | `full` | Stage-1 schedule: `full` (the recipe's 80 epochs) or `short` (20 epochs) |
| `ABL_GPUS` | all GPUs | GPU indices used as queue lanes |
| `TOK_BATCH`, `TOK_WORKERS` | `2`, `6` | Extraction batch size and loader workers |
| `SKIP_RESTORE` | unset | Set it to skip restoring experiments A and B in `prep_ablate` |
| `SMOKE` | `0` | `1`: 1 epoch at batch `$SBATCH` (default 2), to test the code path |

Limitation: the scripts assume a two-directory layout, `GV` for persistent storage and `WORK` for a fast local disk; on one machine, point both at local directories and set `BUNDLE` to this repository's `training/` directory.

### Token commands

`tokens.py` has four subcommands. The flags below are the ones the parser defines.

```bash
# download the pinned weights of the listed encoders (CPU, network)
python -m baselines.ablate.tokens prefetch --backbones "dinov3_l16_384 lemonfm_384"

# encode every clip of the splits (GPU); finished files are skipped, so a killed run resumes
python -m baselines.ablate.tokens extract --backbone dinov3_l16_384 --out $WORK/ablate/tok/dinov3_l16_384 \
       [--splits train,val,test] [--shard 0/1] [--batch 2] [--workers 6] [--limit N] \
       [--amp none|bf16|fp16] [--tf32] [--device cuda] [--export DIR] [--hf-raw DIR]

# every clip has a token file of the expected shape and dtype (CPU); writes <out>/check.json
python -m baselines.ablate.tokens check --backbone dinov3_l16_384 --out $WORK/ablate/tok/dinov3_l16_384 \
       [--splits train,val,test] [--full]

# re-encode a few test clips with V-JEPA 2.1 at 384 px and compare with TactusNet's own cache (GPU)
python -m baselines.ablate.tokens verify --out $BL_OUT/ablate/verify [--n 3] [--sids a,b] [--tok-root DIR]
```

- `extract` writes `<out>/manifest.shard<i>of<n>.json` with the spec, the clip count and the seconds per clip. `--shard i/n` splits the clips over several GPUs.
- `check` loads every 50th file to test for non-finite values; `--full` loads all. It exits with an error if a file is missing, has the wrong shape or dtype, or is not finite. Run it before any training.
- `verify` checks the preprocessing path of this code, not your encoder. It needs `$VJEPA_CKPT` and TactusNet's own token cache (`$BL_TOKENS` or `--tok-root`).

### Steps with `run.sh`

`source baselines/ablate/run.sh` defines the step functions and runs nothing. `bash baselines/ablate/run.sh <step> [args]` runs one step.

```bash
export GV=... WORK=... BUNDLE=$PWD/training VJEPA_CKPT=...
export ABL_BACKBONES="myenc"          # your SPECS name; set it before sourcing
source baselines/ablate/run.sh

SKIP_RESTORE=1 prep_ablate            # weights (prefetch), experiment dirs, classification recipes
ablate_tokens   myenc 0               # extract on GPU 0 for train,val,test, then check
ablate_cls      myenc 0               # Semantic Head: cls1, cls2, then predictions on test and val
ablate_interval myenc 0               # Temporal Head, then predictions on test and val
ablate_stage1   myenc 0               # stage-1 tool decoder, test dump, prediction file
```

- The second argument of each GPU step is the GPU index (default `0`). `ablate_tokens` takes an optional third argument, the shard (`0/1` by default); the check runs only for `0/1`.
- `ablate_expdir <name>` (called by `prep_ablate`) runs `make_exp.py` with the experiment name `abl_<name>`. The result, `$AB/exp/<name>`, is a full experiment directory whose `data/case32/v1/vjepa_token_cache/<split>/<sid>.npy` are links to `$AB/tok/<name>/all/<sid>.npy`.
- Without `SKIP_RESTORE`, `prep_ablate` first restores experiments A and B from `$GV/state/A` and `$GV/state/B` and unpacks TactusNet's tokens from `$GV/tokens/full`. Those inputs come from a finished run of `training/pipeline`.
- `ablate_queue` runs every job of `ablate_jobs` in parallel, one lane per GPU, in dependency order. Its list also holds `ours_heads` and `cascade`, which need the restored experiment A. `ablate_status` prints the state of each job.
- Each step writes a marker and is skipped when re-run. Logs of the training runs are in `$AB/exp/<name>/experiments/runs/`.

Outputs, under `$AO`:

| File | Content |
| --- | --- |
| `enc/<name>/cls/pred_{test,val}.npz` | Pair logits of the Semantic Head (`.heads.npz` also keeps tool and action logits) |
| `enc/<name>/when/pred_{test,val}.npz` | Interval start and end of the Temporal Head |
| `enc/<name>/stage1/pred_test.npz` | Tool points of the stage-1 decoder |

### Score

Each prediction file is scored like every baseline:

```bash
python -m baselines.common.score --pred $AO/enc/myenc/cls/pred_test.npz --out $AO/enc/myenc/cls/metrics_test.json
python -m baselines.common.score --pred $AO/enc/myenc/when/pred_test.npz --compare $AO/enc/ours/when/pred_test.npz
```

`--compare` runs a paired bootstrap over the test videos against a second file. `heads.py` also writes `<out>.check.json` next to each prediction file, with the trainer's own metrics and the unbootstrapped `score.py` numbers.

`ablate_collect` scores all rows at once into `$AO/results.json` and fills a LaTeX table. It needs the template `baselines/ablate/table_ablation.tex`, which is not part of this directory.

## Files

| File | What |
| --- | --- |
| `backbones.py` | `SPECS` and the `Encoder` class: build, preprocess, forward |
| `tokens.py` | `prefetch`, `extract`, `check`, `verify` |
| `launch.py`, `pyshim.sh` | Run a trainer with the stub encoder, the strict cache and, for the condition ablation, a constant condition |
| `heads.py` | Per-clip predictions of the Semantic and Temporal Heads from a checkpoint |
| `convert.py` | Tool-point dumps to prediction files; stage 1 plus pixel refiner without the zoom stage |
| `cascade.py` | The Spatial Head with the pair predicted by the Semantic Head |
| `collect.py` | Scores every row and fills the table template |
| `run.sh` | Step functions and the GPU job queue |
| `tests/test_ablate.py` | CPU checks |

`cascade.py`, `convert.py refine` and the `nocond`, `scalar` and `noquality` variants of `ablate_interval` and `ablate_stage1` belong to other ablations of TactusNet. They operate on the restored experiment A, not on a swapped encoder.
