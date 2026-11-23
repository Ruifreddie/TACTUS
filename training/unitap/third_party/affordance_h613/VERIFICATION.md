# Packaging Verification

- Complete training_core.py is byte-identical to the local original affordance trainer.
- All 504 historical arguments are present; only four runtime paths are sanitized.
- H613 checkpoint loads strictly in the inference API, including token_mask_value.
- Extracted model and original model give bit-identical synthetic quad predictions on CPU.
- Synthetic batch inference, RGB normalization and input rejection checks pass.
- Synthetic training runs backward, AdamW, EMA and saves a selected checkpoint.
- The saved synthetic checkpoint passes eval-only loading and metric computation.
- No real frames, targets, IDs, splits, caches or checkpoints are in the archive.

Tests used Python 3.11 / PyTorch 2.10.0+cu128 on CPU. CUDA was unavailable.
Full nine-epoch training and pretrained V-JEPA video encoding were not rerun.
See SHA256SUMS for the packaged file hashes and PROVENANCE.json for source hashes.
