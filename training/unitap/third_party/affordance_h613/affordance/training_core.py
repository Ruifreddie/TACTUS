#!/usr/bin/env python3
"""Train a lightweight H161 Stage1 ROI head from frozen V-JEPA2.1 tokens.

This is intentionally separate from the old H-series contact-mask trainer.  The
target is the operation-support ROI:

    B_A = bbox(polygon(4 VIS affordance/contact points))

The script is designed for fast proxy iteration:
  - reads selected H161 splits and V-JEPA token cache;
  - optionally uses 16-frame action-range/frame-role features;
  - optionally uses 16-frame tool masks when a full mask root is available;
  - reports per-frame bbox IoU and target-box coverage.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


def parse_float_csv(value: str | None) -> list[float]:
    if value is None:
        return []
    vals: list[float] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def parse_int_csv(value: str | None) -> list[int]:
    if value is None:
        return []
    vals: list[int] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return vals


def token_affinity_variant_count(extra_top_fracs: str | None, extra_size_scales: str | None) -> int:
    return 1 + max(len(parse_float_csv(extra_top_fracs)), len(parse_float_csv(extra_size_scales)))


def parse_template_sizes_px(value: str | None, mask_res: int = 64) -> list[tuple[float, float]]:
    if value is None:
        return []
    sizes: list[tuple[float, float]] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"template size must be WxH pixels, got {part!r}")
        a, b = part.split("x", 1)
        sizes.append((float(a) / float(mask_res), float(b) / float(mask_res)))
    return sizes


def topology_contact_template_count(sizes_px: str | None, offsets_px: str | None, mask_res: int = 64) -> int:
    sizes = parse_template_sizes_px(sizes_px, mask_res=mask_res)
    offsets = parse_float_csv(offsets_px)
    if not sizes or not offsets:
        return 0
    # Hook: hook-tip templates at all offsets plus a perpendicular zero-offset
    # family. Non-hook tools: tip-pair, shaft, and two endpoint families.
    hook_count = 2 * len(sizes) * (len(offsets) + 1)
    nonhook_count = 2 * len(sizes) * (2 * len(offsets) + 6)
    return max(hook_count, nonhook_count)


def parse_combo_index_allowlist(value: str | None, count: int, num_combos: int = 17) -> torch.Tensor | None:
    if value is None or not str(value).strip() or int(count) <= 0:
        return None
    raw = str(value).strip()
    path = Path(raw)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(raw)
    mask = torch.zeros((int(num_combos), int(count)), dtype=torch.bool)

    def _indices(obj: Any) -> list[int]:
        if isinstance(obj, dict) and "indices" in obj:
            obj = obj["indices"]
        return [int(x) for x in obj]

    if isinstance(payload, dict):
        default_indices = payload.get("default")
        if default_indices is not None:
            idx = [i for i in _indices(default_indices) if 0 <= i < int(count)]
            if idx:
                mask[:, idx] = True
        combo_payload = payload.get("combo_indices", payload)
        if isinstance(combo_payload, dict):
            for combo_key, combo_indices in combo_payload.items():
                if str(combo_key) in {"default", "combo_indices", "meta"}:
                    continue
                combo = int(combo_key)
                if not 0 <= combo < int(num_combos):
                    continue
                idx = [i for i in _indices(combo_indices) if 0 <= i < int(count)]
                if idx:
                    mask[combo, :] = False
                    mask[combo, idx] = True
    elif isinstance(payload, list):
        idx = [i for i in _indices(payload) if 0 <= i < int(count)]
        if idx:
            mask[:, idx] = True
    else:
        raise ValueError(f"Unsupported combo allowlist payload: {type(payload).__name__}")

    empty = ~mask.any(dim=1)
    if bool(empty.any()):
        mask[empty, :] = True
    return mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--video-root", type=Path, default=None)
    p.add_argument("--tool-mask-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--bbox-pad-ratio", type=float, default=0.05)
    p.add_argument("--context-aux-pad-ratio", type=float, default=0.45)
    p.add_argument("--min-box-size", type=float, default=0.03)
    p.add_argument("--l1-weight", type=float, default=2.0)
    p.add_argument("--giou-weight", type=float, default=2.0)
    p.add_argument("--ciou-weight", type=float, default=0.0)
    p.add_argument("--center-ce-weight", type=float, default=0.25)
    p.add_argument("--temporal-smooth-weight", type=float, default=0.02)
    p.add_argument("--active-loss-weight", type=float, default=1.5)
    p.add_argument("--inactive-loss-weight", type=float, default=0.75)
    p.add_argument("--hard-box-focal-weight", type=float, default=0.0)
    p.add_argument("--hard-box-focal-threshold", type=float, default=0.60)
    p.add_argument("--hard-box-focal-gamma", type=float, default=1.0)
    p.add_argument("--hard-box-focal-max-factor", type=float, default=2.5)
    p.add_argument("--small-box-loss-weight", type=float, default=0.0)
    p.add_argument("--small-box-ref-area", type=float, default=0.015)
    p.add_argument("--small-box-max-factor", type=float, default=3.0)
    p.add_argument("--use-role-features", action="store_true")
    p.add_argument("--use-role-loss-weighting", action="store_true")
    p.add_argument("--active-prop-boxes", action="store_true")
    p.add_argument("--active-prop-quads", action="store_true")
    p.add_argument("--use-tool-mask", action="store_true")
    p.add_argument("--use-rgb-sidecar", action="store_true")
    p.add_argument("--rgb-sidecar-size", type=int, default=96)
    p.add_argument(
        "--tool-mask-geom-only",
        action="store_true",
        help="For quad_temporal only: use tool-mask moment geometry but do not concatenate the raw mask as a spatial channel.",
    )
    p.add_argument("--use-tool-keypoints", action="store_true")
    p.add_argument("--tool-keypoint-sigma-px", type=float, default=1.5)
    p.add_argument("--tool-keypoint-jitter-std", type=float, default=0.0)
    p.add_argument("--tool-keypoint-dropout", type=float, default=0.0)
    p.add_argument("--tool-keypoint-frame-dropout", type=float, default=0.0)
    p.add_argument("--use-tool-trajectory-bias", action="store_true")
    p.add_argument("--use-tool-trajectory-attn", action="store_true")
    p.add_argument(
        "--use-tool-corner-query-bias",
        action="store_true",
        help=(
            "For point/proposal temporal quad refiners: turn the 16-frame tool "
            "keypoint trajectory into four ordered corner-query biases.  This "
            "preserves corner identity instead of collapsing the region into a "
            "single hotspot prior."
        ),
    )
    p.add_argument(
        "--stage1-drop-tool-keypoints",
        action="store_true",
        help=(
            "For quad_local_point_temporal: do not expose tool keypoints to the "
            "coarse Stage1 predictor; keep them only for the Stage2 refiner."
        ),
    )
    p.add_argument(
        "--detach-stage1-refiner",
        action="store_true",
        help=(
            "For quad_local_point_temporal: detach Stage1 references before "
            "Stage2, forcing Stage1 to learn from its own coarse losses while "
            "Stage2 learns an explicit refinement contract."
        ),
    )
    p.add_argument(
        "--model-mode",
        choices=[
            "condlite",
            "corner_heatmap",
            "support_heatmap",
            "support_moment",
            "edge_dist",
            "box_dist",
            "anchor_residual",
            "tool_mask_anchor_residual",
            "temporal_residual",
            "tool_mask_geom_temporal_residual",
            "combo_expert_temporal_residual",
            "rgb_sidecar_temporal_residual",
            "temporal_transformer_residual",
            "local_refine_residual",
            "local_refine_gated",
            "clip_prior_temporal",
            "quad_temporal",
            "query_temporal",
            "query_temporal_dab",
            "dab_refine_temporal",
            "point_token_temporal",
            "point_token_local_temporal",
            "point_token_local_rect_temporal",
            "proposal_point_temporal",
            "trajectory_prior_temporal",
            "trajectory_prior_point_temporal",
            "quad_local_point_temporal",
            "traj_contact_point_temporal",
            "context_refine",
        ],
        default="condlite",
    )
    p.add_argument("--context-loss-weight", type=float, default=0.0)
    p.add_argument("--anchor-loss-weight", type=float, default=0.5)
    p.add_argument("--coarse-loss-weight", type=float, default=0.5)
    p.add_argument("--area-weight", type=float, default=0.0)
    p.add_argument("--size-weight", type=float, default=0.0)
    p.add_argument("--aspect-weight", type=float, default=0.0)
    p.add_argument("--nwd-weight", type=float, default=0.0)
    p.add_argument("--nwd-c-px", type=float, default=12.8)
    p.add_argument("--nwd-image-size", type=float, default=384.0)
    p.add_argument("--clip-prior-loss-weight", type=float, default=0.0)
    p.add_argument("--clip-prior-output-mix", type=float, default=0.35)
    p.add_argument("--residual-scale", type=float, default=0.75)
    p.add_argument("--point-loss-weight", type=float, default=0.0)
    p.add_argument("--point-shape-loss-weight", type=float, default=0.0)
    p.add_argument("--point-temporal-loss-weight", type=float, default=0.0)
    p.add_argument("--quad-axis-loss-weight", type=float, default=0.0)
    p.add_argument("--quad-axis-length-weight", type=float, default=0.25)
    p.add_argument(
        "--quad-center-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary SmoothL1 loss on the free-quad center, measured directly from the ordered 4 points.",
    )
    p.add_argument(
        "--quad-log-area-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary SmoothL1 loss on log polygon area for the ordered free quad.",
    )
    p.add_argument(
        "--quad-over-area-loss-weight",
        type=float,
        default=0.0,
        help="One-sided auxiliary loss that penalizes predicted polygon area larger than GT area.",
    )
    p.add_argument(
        "--quad-soft-mask-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Differentiable soft-rasterized polygon IoU loss on the final free quad. "
            "This targets mask overlap without requiring an external mask head."
        ),
    )
    p.add_argument("--quad-soft-mask-grid-size", type=int, default=64)
    p.add_argument("--quad-soft-mask-edge-sharpness", type=float, default=128.0)
    p.add_argument(
        "--quad-topology-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary free-quad topology loss.  It keeps the predicted ordered "
            "four-point polygon convex and orientation-consistent with the GT "
            "without converting the target into a rotated rectangle."
        ),
    )
    p.add_argument("--quad-topology-margin", type=float, default=5e-4)
    p.add_argument(
        "--quad-kfiou-loss-weight",
        type=float,
        default=0.0,
        help=(
            "MMRotate KFIoU-style auxiliary geometry loss.  The model still "
            "outputs a free 4-point quad; this loss only regularizes the "
            "oriented rectangle implied by the ordered quad axes."
        ),
    )
    p.add_argument("--quad-kfiou-beta", type=float, default=1.0 / 9.0)
    p.add_argument("--quad-kfiou-fun", choices=["none", "ln", "exp"], default="none")
    p.add_argument("--quad-iou-grid-size", type=int, default=64)
    p.add_argument("--box-mask-loss-weight", type=float, default=0.0)
    p.add_argument("--box-mask-pos-weight", type=float, default=4.0)
    p.add_argument("--edge-ce-weight", type=float, default=0.0)
    p.add_argument("--dfl-weight", type=float, default=0.0)
    p.add_argument("--box-dist-bins", type=int, default=64)
    p.add_argument("--corner-heatmap-weight", type=float, default=0.0)
    p.add_argument("--corner-heatmap-sigma-px", type=float, default=1.25)
    p.add_argument(
        "--corner-heatmap-proposal",
        action="store_true",
        help=(
            "For proposal_point_temporal: decode the ordered four-corner "
            "heatmap field into an explicit quad candidate and append it to "
            "the proposal bank.  This follows CornerNet-style keypoint fields "
            "as proposals, not just auxiliary supervision."
        ),
    )
    p.add_argument("--corner-heatmap-proposal-temperature", type=float, default=0.35)
    p.add_argument(
        "--corner-heatmap-proposal-detach",
        action="store_true",
        help="Detach the corner-field decoded proposal before proposal-bank losses/selectors consume it.",
    )
    p.add_argument(
        "--proposal-source-calibration",
        action="store_true",
        help=(
            "Add a learnable source-specific logit bias over proposal sources. "
            "This is a small Mask-Scoring/Varifocal-style calibration layer "
            "for mixed candidate banks."
        ),
    )
    p.add_argument("--proposal-source-calibration-scale", type=float, default=1.0)
    p.add_argument(
        "--proposal-handoff-source-calibration",
        action="store_true",
        help=(
            "For proposal_point_temporal refine-all runs: after Stage1 and "
            "refined candidates are recollected into one bank, add a learnable "
            "source-specific logit calibration before final handoff selection."
        ),
    )
    p.add_argument("--proposal-handoff-source-calibration-scale", type=float, default=1.0)
    p.add_argument(
        "--proposal-handoff-source4-init",
        type=float,
        default=0.0,
        help=(
            "Optional initial logit bias for source 4 in the final handoff "
            "source calibration.  Kept at 0 for confirmatory training unless "
            "explicitly running an ablation."
        ),
    )
    p.add_argument(
        "--proposal-handoff-source-router",
        action="store_true",
        help=(
            "For proposal_point_temporal refine-all runs: after Stage1 and "
            "refined candidates are recollected into one bank, add a "
            "frame- and candidate-conditioned source router before final "
            "handoff selection.  Unlike source calibration, this is not a "
            "fixed global source bias."
        ),
    )
    p.add_argument("--proposal-handoff-source-router-scale", type=float, default=1.0)
    p.add_argument(
        "--proposal-handoff-source-router-candidate-select-scale",
        type=float,
        default=0.0,
        help=(
            "Add the source-router per-candidate logits to final handoff "
            "selection.  This is intended to be paired with the candidate "
            "quality loss so the router can rank candidates inside the chosen "
            "source, not only choose the source group."
        ),
    )
    p.add_argument(
        "--proposal-handoff-source-router-candidate-quality-weight",
        type=float,
        default=0.0,
        help=(
            "GFL/QFL-style IoU-aware quality loss for source-router candidate "
            "logits.  It directly supervises the router candidate head with "
            "candidate quad IoU so source-local selection is not inherited "
            "only from the global proposal logits."
        ),
    )
    p.add_argument(
        "--proposal-handoff-source-representatives-only",
        action="store_true",
        help=(
            "For proposal_point_temporal refine-all runs: at eval/final handoff, "
            "keep only the current best-logit representative from each source "
            "group before final candidate selection.  This tests whether a "
            "source-level router is failing by lifting bad candidates inside "
            "an otherwise useful source."
        ),
    )
    p.add_argument(
        "--proposal-handoff-source5-topk",
        type=int,
        default=0,
        help=(
            "For proposal_point_temporal refine-all runs: before final handoff "
            "selection, keep only the top-K source5/token-affinity candidates "
            "by current handoff logit and leave other sources unchanged. This "
            "tests whether source5 is useful but over-dispersed under soft "
            "candidate averaging."
        ),
    )
    p.add_argument(
        "--proposal-handoff-source5-readout",
        action="store_true",
        help=(
            "For proposal_point_temporal refine-all runs: add a source5-only "
            "candidate-local V-JEPA readout score before final handoff.  This "
            "targets token-affinity candidates whose oracle IoU is high but "
            "whose current cross-source selector identity is poor."
        ),
    )
    p.add_argument("--proposal-handoff-source5-readout-scale", type=float, default=1.0)
    p.add_argument("--proposal-handoff-source5-readout-grid-size", type=int, default=3)
    p.add_argument(
        "--proposal-handoff-source5-readout-sources",
        default="5",
        help=(
            "Comma-separated proposal source ids that receive the source5 local "
            "readout bonus. Default keeps historical source5-only behavior; H505 "
            "uses 5,7 to rerank token-affinity and topology-template candidates."
        ),
    )
    p.add_argument(
        "--proposal-handoff-source5-readout-detach",
        action="store_true",
        help="Detach the source readout feature/geometry inputs before the scoring head.",
    )
    p.add_argument(
        "--proposal-handoff-source5-readout-varifocal-weight",
        type=float,
        default=0.0,
        help=(
            "Train the source-local handoff readout logits with continuous "
            "quad-IoU Varifocal/QFL targets inside the configured readout "
            "sources. This targets H517's failure where the readout selected "
            "source5 more often but did not rank source5 candidates by IoU."
        ),
    )
    p.add_argument(
        "--proposal-handoff-source5-readout-soft-rank-weight",
        type=float,
        default=0.0,
        help=(
            "KL loss from source-local quad-IoU soft labels to the handoff "
            "readout logits. It teaches the readout the whole within-source "
            "quality distribution instead of only the top candidate."
        ),
    )
    p.add_argument("--proposal-handoff-source5-readout-loss-topk", type=int, default=3)
    p.add_argument("--proposal-handoff-source5-readout-soft-rank-tau", type=float, default=0.08)
    p.add_argument(
        "--proposal-handoff-mask-quality",
        action="store_true",
        help=(
            "For proposal_point_temporal refine-all runs: after the final "
            "handoff bank is assembled, predict a candidate-local aligned-grid "
            "mask and IoU-quality residual for selected source ids. This is a "
            "Mask-Scoring/Deformable-DETR-style evidence head for source5/7 "
            "candidates, not a global source bias."
        ),
    )
    p.add_argument("--proposal-handoff-mask-quality-scale", type=float, default=0.50)
    p.add_argument("--proposal-handoff-mask-grid-size", type=int, default=7)
    p.add_argument("--proposal-handoff-mask-expand-scale", type=float, default=1.25)
    p.add_argument(
        "--proposal-handoff-mask-refine-scale",
        type=float,
        default=0.0,
        help=(
            "If positive, decode the handoff point-mask logits into a "
            "candidate-aligned moment quad and blend selected-source proposal "
            "geometry toward that quad before final handoff. This tests a "
            "PointRend-style geometry path instead of using the mask head only "
            "as a scalar quality score."
        ),
    )
    p.add_argument("--proposal-handoff-mask-refine-size-scale", type=float, default=2.0)
    p.add_argument("--proposal-handoff-mask-refine-min-size-px", type=float, default=2.0)
    p.add_argument("--proposal-handoff-mask-refine-max-ratio", type=float, default=1.05)
    p.add_argument(
        "--proposal-handoff-mask-quality-sources",
        default="5,7",
        help="Comma-separated source ids whose logits receive the handoff mask-quality residual.",
    )
    p.add_argument(
        "--proposal-handoff-mask-detach",
        action="store_true",
        help="Detach tokens/geometry/readout before the handoff mask-quality head.",
    )
    p.add_argument(
        "--proposal-handoff-mask-score-detach",
        action="store_true",
        help="Detach the handoff mask-quality logit before adding it to proposal logits.",
    )
    p.add_argument(
        "--proposal-handoff-rank-adaptive-scorer",
        action="store_true",
        help=(
            "For proposal_point_temporal refine-all runs: add a Rank-DETR-style "
            "rank-position-aware residual scorer to the final handoff bank. "
            "Candidates are ranked by detached current score, receive a learned "
            "rank embedding plus source/geometry/logit context, and then get a "
            "zero-initialized score residual before final selection."
        ),
    )
    p.add_argument("--proposal-handoff-rank-adaptive-scale", type=float, default=0.45)
    p.add_argument("--proposal-handoff-rank-adaptive-max-rank", type=int, default=512)
    p.add_argument(
        "--proposal-handoff-rank-adaptive-detach",
        action="store_true",
        help="Detach candidate token/geometry/logit features before the rank-adaptive scorer.",
    )
    p.add_argument(
        "--proposal-handoff-sparse-set-scorer",
        action="store_true",
        help=(
            "For proposal_point_temporal refine-all runs: add a Sparse-RCNN-style "
            "candidate-set self-attention residual scorer before final handoff. "
            "Unlike source bias or independent readout heads, each candidate score "
            "can compare against the rest of the per-frame proposal bank."
        ),
    )
    p.add_argument("--proposal-handoff-sparse-set-scorer-scale", type=float, default=0.35)
    p.add_argument(
        "--proposal-handoff-sparse-set-scorer-sources",
        default="0,5,7",
        help=(
            "Comma-separated source ids whose logits receive the sparse-set "
            "residual. The attention still sees the full bank."
        ),
    )
    p.add_argument(
        "--proposal-handoff-sparse-set-scorer-detach",
        action="store_true",
        help="Detach tokens/geometry/logits before the sparse-set scorer.",
    )
    p.add_argument(
        "--proposal-handoff-source5-refiner",
        action="store_true",
        help=(
            "For proposal_point_temporal refine-all runs: consume the source5 "
            "candidate subset as a small set, predict one refined source5 "
            "candidate, append it to the proposal bank, and let the normal "
            "handoff choose it. This targets H468/H490 where source5 contains "
            "oracle candidates but scalar source5 scoring does not identify "
            "the right one."
        ),
    )
    p.add_argument("--proposal-handoff-source5-refiner-topk", type=int, default=12)
    p.add_argument("--proposal-handoff-source5-refiner-temperature", type=float, default=0.75)
    p.add_argument("--proposal-handoff-source5-refiner-scale", type=float, default=0.35)
    p.add_argument("--proposal-handoff-source5-refiner-logit-scale", type=float, default=0.65)
    p.add_argument("--proposal-handoff-source5-refiner-init-logit-bias", type=float, default=0.0)
    p.add_argument(
        "--proposal-handoff-source5-refiner-detach",
        action="store_true",
        help="Detach source5 candidate tokens/geometry/logits before the set refiner.",
    )
    p.add_argument(
        "--proposal-handoff-source5-refiner-loss-weight",
        type=float,
        default=0.0,
        help="Direct quad supervision for the appended source5-set refined candidate.",
    )
    p.add_argument(
        "--proposal-handoff-source5-refiner-rank-weight",
        type=float,
        default=0.0,
        help=(
            "Within-source5 CE loss that trains the source5-set refiner scores "
            "to choose the highest-IoU source5 candidate before the compact "
            "source6 candidate competes in the final handoff."
        ),
    )
    p.add_argument(
        "--proposal-handoff-source5-refiner-axis-weight",
        type=float,
        default=0.25,
        help="Axis-alignment term weight inside the source5-set refiner aux loss.",
    )
    p.add_argument(
        "--proposal-handoff-source5-refiner-quality-weight",
        type=float,
        default=0.0,
        help=(
            "IoU-aware quality loss for the appended source5-set candidate's "
            "own logit. This is a dedicated VFNet/GFL-style score target so "
            "the source5 refiner does not rely only on diluted proposal-bank "
            "classification losses."
        ),
    )
    p.add_argument(
        "--proposal-handoff-source5-refiner-min-source5-iou",
        type=float,
        default=0.0,
        help=(
            "Optional gate for the source5-set refiner aux loss. If positive, "
            "only frames whose existing source5 oracle quad IoU exceeds this "
            "threshold supervise the new refiner candidate."
        ),
    )
    p.add_argument(
        "--proposal-handoff-source5-refiner-source-id",
        type=int,
        default=5,
        help=(
            "Candidate source consumed by the source5-set refiner implementation. "
            "Default keeps the historical source5 behavior; setting this to 7 "
            "reuses the same compact set-refiner for topology-template candidates."
        ),
    )
    p.add_argument(
        "--proposal-handoff-source5-refiner-output-source-id",
        type=int,
        default=6,
        help=(
            "Source id assigned to the appended compact refiner candidate. "
            "Default 6 preserves prior source5-refiner audits; H500 sets 7 so "
            "the compact topology candidate remains in the source7 family."
        ),
    )
    p.add_argument(
        "--proposal-handoff-topk-query-refiner",
        action="store_true",
        help=(
            "For proposal_point_temporal refine-all runs: take top-K candidates "
            "from configured source groups, treat them as a Sparse-RCNN/DAB-style "
            "query set with candidate-local V-JEPA readout, refine every query "
            "individually, and append the refined candidates back to the handoff "
            "bank. This differs from the source5 set refiner because it does not "
            "pool the candidate family into one average quad."
        ),
    )
    p.add_argument(
        "--proposal-handoff-topk-query-sources",
        default="0,5",
        help="Comma-separated source ids consumed by --proposal-handoff-topk-query-refiner.",
    )
    p.add_argument("--proposal-handoff-topk-query-topk-per-source", type=int, default=12)
    p.add_argument("--proposal-handoff-topk-query-grid-size", type=int, default=3)
    p.add_argument("--proposal-handoff-topk-query-scale", type=float, default=0.30)
    p.add_argument("--proposal-handoff-topk-query-logit-scale", type=float, default=0.75)
    p.add_argument("--proposal-handoff-topk-query-output-source-id", type=int, default=6)
    p.add_argument(
        "--proposal-handoff-topk-query-replace-logits",
        action="store_true",
        help=(
            "Use the top-k query quality/rank head as the appended candidate logit "
            "instead of adding a delta to the original proposal logit. This tests "
            "whether source handoff fails because original candidate scores are "
            "miscalibrated against localization quality."
        ),
    )
    p.add_argument(
        "--proposal-handoff-topk-query-detach",
        action="store_true",
        help="Detach top-k query tokens/geometry/logits before the handoff top-k query refiner.",
    )
    p.add_argument(
        "--proposal-handoff-topk-query-loss-weight",
        type=float,
        default=0.0,
        help="Direct quad supervision for the individually refined top-k query candidates.",
    )
    p.add_argument(
        "--proposal-handoff-topk-query-quality-weight",
        type=float,
        default=0.0,
        help="IoU-aware Varifocal/QFL loss for refined top-k query logits.",
    )
    p.add_argument(
        "--proposal-handoff-topk-query-rank-weight",
        type=float,
        default=0.0,
        help="CE rank loss selecting the best refined top-k query candidate per frame.",
    )
    p.add_argument(
        "--proposal-handoff-topk-query-soft-rank-weight",
        type=float,
        default=0.0,
        help=(
            "Listwise soft-rank CE over the top-k query candidates, using a "
            "softmax over candidate IoUs as the target distribution."
        ),
    )
    p.add_argument(
        "--proposal-handoff-topk-query-soft-rank-tau",
        type=float,
        default=0.05,
        help="Temperature for the IoU soft target used by --proposal-handoff-topk-query-soft-rank-weight.",
    )
    p.add_argument(
        "--proposal-handoff-topk-query-min-base-iou",
        type=float,
        default=0.0,
        help="Optional training gate: ignore top-k query candidates whose base IoU is below this value.",
    )
    p.add_argument(
        "--proposal-handoff-router-gated-source",
        action="store_true",
        help=(
            "For proposal_point_temporal refine-all runs with a source router: "
            "at eval/final handoff, choose the source by router logits first "
            "and mask final candidate selection to candidates from that source."
        ),
    )
    p.add_argument(
        "--proposal-handoff-router-normalized-select",
        action="store_true",
        help=(
            "For proposal_point_temporal refine-all runs with a source router: "
            "at eval/final handoff, replace raw cross-source candidate logits "
            "with source-normalized within-source logits plus router source logits."
        ),
    )
    p.add_argument("--proposal-handoff-router-normalized-source-scale", type=float, default=1.0)
    p.add_argument("--proposal-handoff-router-normalized-within-scale", type=float, default=1.0)
    p.add_argument(
        "--token-affinity-proposal",
        action="store_true",
        help=(
            "For proposal_point_temporal: append TokenCut/CutLER-style visual "
            "proposal candidates decoded from frozen V-JEPA token affinity. "
            "This adds proposal-source 5 and keeps the strict no-trajectory "
            "input contract."
        ),
    )
    p.add_argument(
        "--token-affinity-proposal-count",
        type=int,
        default=1,
        help="Number of token-affinity proposals appended per frame.",
    )
    p.add_argument(
        "--token-affinity-proposal-temperature",
        type=float,
        default=0.08,
        help="Softmax temperature used for token-affinity seed/region weights.",
    )
    p.add_argument(
        "--token-affinity-proposal-top-frac",
        type=float,
        default=0.055,
        help="Fraction of the 24x24 token grid kept for the affinity region moment.",
    )
    p.add_argument(
        "--token-affinity-proposal-size-scale",
        type=float,
        default=3.6,
        help="Scale mapping affinity-region covariance to oriented proposal side lengths.",
    )
    p.add_argument(
        "--token-affinity-proposal-extra-top-fracs",
        type=str,
        default="",
        help=(
            "Comma-separated extra token fractions for multi-scale token-affinity "
            "proposal variants. Each entry reuses the same seeds as "
            "--token-affinity-proposal-top-frac."
        ),
    )
    p.add_argument(
        "--token-affinity-proposal-extra-size-scales",
        type=str,
        default="",
        help=(
            "Comma-separated extra size scales for multi-scale token-affinity "
            "proposal variants. If the count differs from extra-top-fracs, "
            "missing entries fall back to the base value."
        ),
    )
    p.add_argument(
        "--token-affinity-proposal-detach",
        action="store_true",
        help=(
            "Detach the token-affinity proposal geometry. This matches the "
            "external frozen-token proposal role of TokenCut/CutLER and avoids "
            "letting this branch destabilize the learned proposal bank."
        ),
    )
    p.add_argument(
        "--topology-contact-template-proposal",
        action="store_true",
        help=(
            "For proposal_point_temporal: append deterministic, tool-topology "
            "contact-region templates from tool keypoints as proposal-source 7. "
            "This targets the H497 finding that hook/grasper hard cases miss "
            "explicit contact-shape candidates even when token proposals have "
            "good visual regions."
        ),
    )
    p.add_argument(
        "--topology-contact-template-sizes-px",
        type=str,
        default="4x3,6x4,8x4,10x5",
        help="Comma-separated oriented template sizes in a 64x64 mask grid.",
    )
    p.add_argument(
        "--topology-contact-template-offsets-px",
        type=str,
        default="-2,0,2",
        help="Comma-separated offsets along the contact axis in a 64x64 mask grid.",
    )
    p.add_argument(
        "--topology-contact-template-logit-bias",
        type=float,
        default=0.0,
        help="Initial logit bias added to valid topology-template source7 candidates.",
    )
    p.add_argument(
        "--topology-contact-template-index-embedding",
        action="store_true",
        help="Add a learned template-index embedding to topology-template source7 candidate tokens.",
    )
    p.add_argument(
        "--topology-contact-template-combo-bias",
        action="store_true",
        help=(
            "Add a learned combo-conditioned bias over topology-template "
            "source7 candidate indices."
        ),
    )
    p.add_argument("--topology-contact-template-combo-bias-scale", type=float, default=1.0)
    p.add_argument(
        "--topology-contact-template-combo-allowlist",
        type=str,
        default="",
        help=(
            "Optional JSON string or JSON file path mapping combo ids to valid "
            "source7 topology-template indices. Invalid template logits are "
            "masked out, enabling train-set anchor/template pruning."
        ),
    )
    p.add_argument("--support-heatmap-weight", type=float, default=0.0)
    p.add_argument("--support-heatmap-pos-weight", type=float, default=4.0)
    p.add_argument("--support-heatmap-gamma", type=float, default=0.5)
    p.add_argument(
        "--support-heatmap-target",
        choices=("aabb", "quad"),
        default="aabb",
        help="Target used for support heatmap supervision. quad keeps oriented-region structure.",
    )
    p.add_argument("--decoder-aux-loss-weight", type=float, default=0.0)
    p.add_argument("--decoder-query-noise-scale", type=float, default=0.0)
    p.add_argument(
        "--dn-refiner-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only DN-style auxiliary loss: feed noisy GT quad queries "
            "through the point refiner and reconstruct the tight oriented quad."
        ),
    )
    p.add_argument(
        "--dn-refiner-noise-scale",
        type=float,
        default=0.0,
        help="Relative point noise scale for the DN-style noisy GT quad refiner branch.",
    )
    p.add_argument(
        "--dn-refiner-groups",
        type=int,
        default=1,
        help=(
            "Number of repeated noisy GT query groups for the DN refiner branch. "
            "This is a DINO-style grouped denoising variant; groups are "
            "training-only and do not change inference."
        ),
    )
    p.add_argument(
        "--dn-refiner-noise-min-scale",
        type=float,
        default=1.0,
        help=(
            "Minimum multiplier for grouped DN noise. If groups > 1, noise "
            "scales are linearly spaced from this value to 1.0, then multiplied "
            "by --dn-refiner-noise-scale."
        ),
    )
    p.add_argument(
        "--proposal-oracle-refiner-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only auxiliary loss: select the best current proposal by "
            "GT quad IoU, feed that proposal through the shared point refiner, "
            "and reconstruct the tight oriented quad. This trains the refiner "
            "on the actual proposal-error distribution without changing inference."
        ),
    )
    p.add_argument(
        "--proposal-oracle-refiner-noise-scale",
        type=float,
        default=0.0,
        help="Relative point noise added around the oracle selected proposal before refiner training.",
    )
    p.add_argument(
        "--stage2-bridge-noise-mode",
        choices=["local_only", "replace"],
        default="local_only",
        help=(
            "For quad_local_point_temporal: local_only keeps the clean stage1 "
            "reference as the main stage2 query and perturbs local sampling; "
            "replace perturbs the actual stage1-to-stage2 reference."
        ),
    )
    p.add_argument("--decoder-layers", type=int, default=3)
    p.add_argument(
        "--proposal-count",
        type=int,
        default=6,
        help="Number of visual proposal candidates for proposal/rerank heads.",
    )
    p.add_argument(
        "--proposal-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary min-over-candidates proposal localization loss.",
    )
    p.add_argument(
        "--proposal-rank-weight",
        type=float,
        default=0.0,
        help="Cross-entropy loss that trains proposal quality logits to select the best candidate.",
    )
    p.add_argument(
        "--proposal-quality-weight",
        type=float,
        default=0.0,
        help="Regression loss that calibrates proposal quality logits to candidate quad IoU.",
    )
    p.add_argument(
        "--proposal-varifocal-weight",
        type=float,
        default=0.0,
        help="IoU-aware varifocal/quality-focal loss for proposal logits.",
    )
    p.add_argument(
        "--proposal-geom-quality-weight",
        type=float,
        default=0.0,
        help=(
            "Additional varifocal loss for proposal logits using a detached "
            "center/area/contact-geometry quality target. This targets the "
            "observed gap where the best candidate is near GT but logits "
            "prefer over-large or off-center proposals."
        ),
    )
    p.add_argument("--proposal-geom-center-tau", type=float, default=2.0)
    p.add_argument("--proposal-geom-area-tau", type=float, default=0.75)
    p.add_argument(
        "--proposal-center-area-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary top-k proposal loss on candidate center and log-area calibration.",
    )
    p.add_argument("--proposal-center-area-topk", type=int, default=2)
    p.add_argument("--proposal-center-area-over-weight", type=float, default=0.5)
    p.add_argument(
        "--proposal-source-balanced-varifocal-weight",
        type=float,
        default=0.0,
        help=(
            "Source-balanced VFL over proposal candidates.  For each proposal "
            "source group, only that source's best-IoU candidate is kept as a "
            "positive, plus the global oracle candidate.  This is an "
            "ATSS/PAA-style assignment diagnostic for source dominance."
        ),
    )
    p.add_argument(
        "--proposal-source-balanced-task-aligned-weight",
        type=float,
        default=0.0,
        help=(
            "TOOD-style task-aligned VFL applied per proposal source.  Each "
            "source gets its own top-K positives by score^alpha * IoU^beta, "
            "so low-scored but high-oracle sources can learn within-source "
            "quality ranking before global source competition."
        ),
    )
    p.add_argument(
        "--proposal-source-balanced-task-aligned-topk",
        type=int,
        default=2,
        help="Number of task-aligned positives kept per proposal source.",
    )
    p.add_argument(
        "--proposal-source-group-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Cross-entropy over proposal source groups.  The target source is "
            "the source id of the candidate with highest true quad IoU.  This "
            "directly penalizes source4/corner over-selection when another "
            "source contains the oracle candidate."
        ),
    )
    p.add_argument(
        "--proposal-source-local-rank-weight",
        type=float,
        default=0.0,
        help=(
            "Cross-entropy over candidates inside each proposal source group. "
            "For every present source, the target is that source's highest "
            "quad-IoU candidate.  This attacks route-correct/candidate-wrong "
            "failures directly."
        ),
    )
    p.add_argument(
        "--proposal-handoff-source-router-ce-weight",
        type=float,
        default=0.0,
        help=(
            "CE supervision for the final handoff source router.  The target "
            "is the source id of the highest quad-IoU candidate in the final "
            "Stage1+refined candidate bank."
        ),
    )
    p.add_argument(
        "--proposal-pcl-cluster-ce-weight",
        type=float,
        default=0.0,
        help=(
            "C-MIL/PCL-style cluster-owner cross entropy over proposal clusters. "
            "The top-IoU candidates define local clusters, candidates are "
            "assigned by proposal overlap, and the oracle cluster is trained "
            "to own the aggregate cluster logit. This is softer than exact "
            "top1 CE but more direct than per-candidate cluster BCE."
        ),
    )
    p.add_argument(
        "--proposal-oicr-refine-head",
        action="store_true",
        help=(
            "Add a PCL/OICR-style separate proposal refinement selector head. "
            "The head predicts a residual delta on top of the base proposal "
            "logits from candidate tokens, geometry, and detached base score."
        ),
    )
    p.add_argument(
        "--proposal-oicr-refine-use-for-selection",
        action="store_true",
        help=(
            "Use the OICR refinement selector logits for proposal handoff. "
            "The base proposal logits are kept in proposal_base_logits for "
            "diagnostics, while proposal_logits becomes the refinement score."
        ),
    )
    p.add_argument(
        "--proposal-oicr-refine-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Cluster-owner CE loss on proposal_oicr_logits, using the same "
            "top-IoU cluster target as --proposal-pcl-cluster-ce-weight. "
            "This follows PCL/OICR's separate refine_score supervision rather "
            "than applying the CE directly to the base MIL/proposal logits."
        ),
    )
    p.add_argument(
        "--proposal-handoff-oicr-refine-times",
        type=int,
        default=0,
        help=(
            "Number of OICR-style refinement selector heads applied to the "
            "final handoff proposal bank.  This mirrors REFINE_TIMES in "
            "PCL/OICR, but operates after proposal recollection so the "
            "supervised logits line up with the final candidate set."
        ),
    )
    p.add_argument(
        "--proposal-handoff-oicr-refine-scale",
        type=float,
        default=1.0,
        help="Residual logit scale for each handoff OICR refinement head.",
    )
    p.add_argument(
        "--proposal-handoff-oicr-refine-aggregate",
        type=str,
        default="last",
        choices=("last", "mean", "mean_base"),
        help=(
            "How handoff OICR refinement logits are used for final selection. "
            "'last' keeps the H506 behavior. 'mean' follows PCL test-time "
            "averaging over refinement streams. 'mean_base' also includes the "
            "base selector logits as conservative Boosted-OICR-style score "
            "fusion for residual refine heads."
        ),
    )
    p.add_argument(
        "--proposal-handoff-oicr-refine-detach",
        action="store_true",
        help=(
            "Detach candidate tokens/geometry and previous logits before the "
            "handoff OICR refinement heads, keeping gradients local to the "
            "selector heads."
        ),
    )
    p.add_argument(
        "--proposal-handoff-oicr-refine-use-for-selection",
        action="store_true",
        help=(
            "Use the last handoff OICR refinement logits as final proposal "
            "selection logits."
        ),
    )
    p.add_argument(
        "--proposal-source0-multimatch-weight",
        type=float,
        default=0.0,
        help=(
            "SPE-style multi-match supervision for the base visual proposal "
            "source.  The top-K source0 candidates by true quad IoU are kept "
            "as positives, so several visual proposals can learn the GT "
            "rather than letting the selector over-rely on source4."
        ),
    )
    p.add_argument(
        "--proposal-source0-multimatch-topk",
        type=int,
        default=3,
        help="Number of source0 visual proposals to supervise as positives.",
    )
    p.add_argument(
        "--proposal-source5-multimatch-weight",
        type=float,
        default=0.0,
        help=(
            "SPE-style multi-match supervision for token-affinity/source5 "
            "proposals. The top-K source5 candidates by true quad IoU are "
            "kept as positives so source5 can compete when it contains the "
            "oracle candidate."
        ),
    )
    p.add_argument(
        "--proposal-source5-multimatch-topk",
        type=int,
        default=2,
        help="Number of source5 token-affinity proposals to supervise as positives.",
    )
    p.add_argument(
        "--proposal-source4-demote-weight",
        type=float,
        default=0.0,
        help=(
            "Source-aware hard-negative loss for source4/corner candidates. "
            "Only high-logit source4 candidates whose quad IoU is below the "
            "frame oracle by a margin are demoted against the best non-source4 "
            "positive candidate. This is an asymmetric Soft-NMS/DeepBox-style "
            "proposal reranking term, not a blanket source deletion."
        ),
    )
    p.add_argument("--proposal-source4-demote-topk", type=int, default=12)
    p.add_argument("--proposal-source4-demote-margin", type=float, default=0.18)
    p.add_argument("--proposal-source4-demote-iou-gap", type=float, default=0.08)
    p.add_argument(
        "--proposal-source4-demote-positive-sources",
        type=str,
        default="0,5,7,8",
        help="Comma-separated source ids that may serve as positives for source4 demotion.",
    )
    p.add_argument(
        "--proposal-source5-soft-owner-weight",
        type=float,
        default=0.0,
        help=(
            "Source5-local KL loss: train the source5 logits to match a soft "
            "IoU ownership distribution within the source5 bag, preserving the "
            "soft route instead of forcing a brittle source5 top1 winner."
        ),
    )
    p.add_argument("--proposal-source5-soft-owner-topk", type=int, default=32)
    p.add_argument("--proposal-source5-soft-owner-tau", type=float, default=0.08)
    p.add_argument(
        "--proposal-source5-mass-preserve-weight",
        type=float,
        default=0.0,
        help=(
            "Weak BCE on the total source5 soft-selection mass. The target is "
            "derived from source5 best-IoU versus non-source5 best-IoU and "
            "clamped, so source5 is preserved only when its candidate bag has "
            "evidence."
        ),
    )
    p.add_argument("--proposal-source5-mass-preserve-tau", type=float, default=0.07)
    p.add_argument("--proposal-source5-mass-preserve-min", type=float, default=0.15)
    p.add_argument("--proposal-source5-mass-preserve-max", type=float, default=0.75)
    p.add_argument(
        "--proposal-source7-multimatch-weight",
        type=float,
        default=0.0,
        help=(
            "SPE-style multi-match supervision for topology-template/source7 "
            "proposals. The top-K source7 candidates by true quad IoU are "
            "kept as positives so deterministic contact templates can compete "
            "when they contain the oracle candidate."
        ),
    )
    p.add_argument(
        "--proposal-source7-multimatch-topk",
        type=int,
        default=3,
        help="Number of source7 topology-template proposals to supervise as positives.",
    )
    p.add_argument("--proposal-varifocal-alpha", type=float, default=0.75)
    p.add_argument("--proposal-varifocal-gamma", type=float, default=2.0)
    p.add_argument(
        "--proposal-quality-target-mode",
        "--proposal-vfl-positive-mode",
        dest="proposal_quality_target_mode",
        choices=(
            "all",
            "threshold",
            "threshold_or_oracle",
            "oracle",
            "adaptive_meanstd",
            "adaptive_meanstd_or_oracle",
            "adaptive_topk",
            "adaptive_topk_or_oracle",
        ),
        default="all",
        help=(
            "How candidate quad-IoU is converted into selector quality targets. "
            "'all' preserves the historical continuous target.  Threshold modes "
            "train low-IoU overlap candidates as negatives, following the "
            "positive/negative split used by VarifocalNet/GFL-style scorers. "
            "Adaptive modes select positives from the per-frame candidate IoU "
            "distribution, following ATSS/PAA-style sample assignment."
        ),
    )
    p.add_argument(
        "--proposal-quality-positive-threshold",
        "--proposal-vfl-positive-threshold",
        dest="proposal_quality_positive_threshold",
        type=float,
        default=0.0,
        help=(
            "IoU threshold for positive candidate quality targets when "
            "--proposal-quality-target-mode is threshold/threshold_or_oracle."
        ),
    )
    p.add_argument(
        "--proposal-quality-adaptive-std-scale",
        type=float,
        default=1.0,
        help=(
            "Std multiplier for adaptive_meanstd quality targets.  Positives are "
            "candidate_iou >= mean(candidate_iou) + scale * std(candidate_iou)."
        ),
    )
    p.add_argument(
        "--proposal-quality-adaptive-topk",
        type=int,
        default=2,
        help="Number of positives for adaptive_topk quality targets.",
    )
    p.add_argument(
        "--proposal-active-tip-selector",
        action="store_true",
        help=(
            "Add an explicit action-range/tool-tip-trajectory quality scorer "
            "to proposal logits.  It does not create new proposals; it teaches "
            "the selector to score existing ROI candidates using current tip, "
            "active-range tip path, tool axis, and tip motion geometry."
        ),
    )
    p.add_argument(
        "--proposal-active-tip-selector-scale",
        type=float,
        default=1.0,
        help="Scale for the active-tip trajectory selector logit delta.",
    )
    p.add_argument(
        "--proposal-active-tip-selector-aux-only",
        action="store_true",
        help=(
            "Train the active-tip trajectory selector as an independent "
            "candidate-quality head instead of adding it to proposal logits "
            "during training."
        ),
    )
    p.add_argument(
        "--proposal-active-tip-selector-loss-weight",
        type=float,
        default=0.0,
        help="Varifocal-style candidate-quad-IoU loss for the active-tip selector head.",
    )
    p.add_argument(
        "--proposal-active-tip-selector-eval-fuse-scale",
        type=float,
        default=0.0,
        help=(
            "When --proposal-active-tip-selector-aux-only is set, add this "
            "scale times the active-tip quality logit to proposal logits only "
            "during eval.  This implements post-hoc rescoring."
        ),
    )
    p.add_argument(
        "--proposal-active-tip-feature-fusion",
        action="store_true",
        help=(
            "Fuse the active-tip/action-range candidate geometry features into "
            "proposal candidate tokens before the quality head. This makes the "
            "trajectory condition modify ROI features instead of only adding a "
            "post-hoc selector logit."
        ),
    )
    p.add_argument(
        "--proposal-active-tip-feature-scale",
        type=float,
        default=0.35,
        help="Residual scale for --proposal-active-tip-feature-fusion.",
    )
    p.add_argument(
        "--proposal-pairwise-rank-weight",
        type=float,
        default=0.0,
        help=(
            "Pairwise margin ranking loss over proposal logits.  If candidate i "
            "has clearly higher target IoU than candidate j, logit_i should be "
            "larger than logit_j."
        ),
    )
    p.add_argument(
        "--proposal-pairwise-rank-margin",
        type=float,
        default=0.20,
        help="Logit margin for --proposal-pairwise-rank-weight.",
    )
    p.add_argument(
        "--proposal-pairwise-iou-gap",
        type=float,
        default=0.03,
        help="Minimum IoU gap required to form a pairwise proposal ranking pair.",
    )
    p.add_argument(
        "--proposal-hardneg-rank-weight",
        type=float,
        default=0.0,
        help=(
            "Hard-negative proposal ranking loss.  For each frame, rank the "
            "oracle-IoU candidate above the currently highest-logit lower-IoU "
            "negatives.  This follows proposal-ranking hard pair mining and "
            "targets source-prior failures without turning all candidates into "
            "a dense oracle CE problem."
        ),
    )
    p.add_argument("--proposal-hardneg-rank-topk", type=int, default=4)
    p.add_argument("--proposal-hardneg-rank-margin", type=float, default=0.24)
    p.add_argument("--proposal-hardneg-rank-iou-gap", type=float, default=0.02)
    p.add_argument(
        "--proposal-tool-hardneg-weight",
        type=float,
        default=0.0,
        help=(
            "Tool-body hard-negative ranking loss.  Low-IoU candidates whose "
            "centers lie close to the tool keypoint polyline are mined as "
            "negatives and ranked below the oracle-IoU candidate."
        ),
    )
    p.add_argument("--proposal-tool-hardneg-topk", type=int, default=4)
    p.add_argument("--proposal-tool-hardneg-margin", type=float, default=0.22)
    p.add_argument("--proposal-tool-hardneg-iou-gap", type=float, default=0.04)
    p.add_argument("--proposal-tool-hardneg-sigma", type=float, default=0.045)
    p.add_argument("--proposal-tool-hardneg-min-score", type=float, default=0.18)
    p.add_argument(
        "--proposal-soft-rank-weight",
        type=float,
        default=0.0,
        help=(
            "KL distillation from candidate quad-IoU soft labels to proposal "
            "selection logits.  This teaches the selector the whole proposal "
            "quality distribution instead of only argmax or pairwise order."
        ),
    )
    p.add_argument(
        "--proposal-soft-rank-tau",
        type=float,
        default=0.10,
        help="Temperature for candidate quad-IoU soft labels in --proposal-soft-rank-weight.",
    )
    p.add_argument(
        "--proposal-owner-band-weight",
        type=float,
        default=0.0,
        help=(
            "Listwise owner-bag selector loss.  Only candidates within an IoU "
            "band of the per-frame oracle candidate are positives; all lower "
            "quality candidates receive zero target mass.  This is a narrower "
            "PCL/VFNet-style quality owner target than --proposal-soft-rank, "
            "which still assigns non-trivial mass to medium/ring candidates."
        ),
    )
    p.add_argument(
        "--proposal-owner-band-gap",
        type=float,
        default=0.04,
        help="Candidate is positive when IoU >= oracle_iou - gap.",
    )
    p.add_argument(
        "--proposal-owner-band-min-iou",
        type=float,
        default=0.05,
        help="Minimum IoU for owner-band positives; the oracle candidate is always kept.",
    )
    p.add_argument(
        "--proposal-owner-band-tau",
        type=float,
        default=0.05,
        help="Temperature for distributing probability mass inside the owner band.",
    )
    p.add_argument(
        "--proposal-owner-band-topk",
        type=int,
        default=0,
        help="If >0, intersect the owner band with the top-K candidates by true IoU.",
    )
    p.add_argument(
        "--proposal-owner-band-source-balanced",
        action="store_true",
        help=(
            "Normalize owner-band target mass per source id before merging. "
            "This prevents a source with many near-duplicate positives from "
            "dominating the listwise target."
        ),
    )
    p.add_argument(
        "--proposal-neuralsort-top1-weight",
        type=float,
        default=0.0,
        help=(
            "NeuralSort-style listwise top-1 selector loss.  It builds a "
            "differentiable permutation matrix from proposal logits and "
            "directly supervises the first sorted row to choose the highest "
            "quad-IoU candidate."
        ),
    )
    p.add_argument(
        "--proposal-neuralsort-tau",
        type=float,
        default=0.35,
        help="Temperature for --proposal-neuralsort-top1-weight.",
    )
    p.add_argument(
        "--proposal-ranksort-weight",
        type=float,
        default=0.0,
        help=(
            "RankSortLoss-style proposal selector loss adapted from the "
            "official RankSortLoss implementation.  It directly supervises "
            "proposal logits to rank candidates by detached quad IoU."
        ),
    )
    p.add_argument("--proposal-ranksort-delta", type=float, default=0.50)
    p.add_argument(
        "--proposal-ranksort-positive-threshold",
        type=float,
        default=0.05,
        help=(
            "Candidate quad-IoU threshold used to mark positives for the "
            "RankSort-style selector loss; the best candidate is always kept "
            "positive to avoid empty-positive frames."
        ),
    )
    p.add_argument(
        "--proposal-task-aligned-weight",
        type=float,
        default=0.0,
        help=(
            "TOOD-style task-aligned selector loss.  Candidate positives are "
            "chosen by detached score^alpha * quad_iou^beta and trained with "
            "IoU-aware soft targets.  This targets the oracle-gap failure mode "
            "where good proposals exist but selector logits do not rank them."
        ),
    )
    p.add_argument("--proposal-task-aligned-topk", type=int, default=3)
    p.add_argument("--proposal-task-aligned-alpha", type=float, default=1.0)
    p.add_argument("--proposal-task-aligned-beta", type=float, default=6.0)
    p.add_argument(
        "--proposal-c2f-assign-weight",
        type=float,
        default=0.0,
        help=(
            "DCFL-style coarse-to-fine candidate assignment for proposal logits. "
            "First take the top-K candidates by detached quad IoU, then select "
            "top-Q positives by IoU plus detached selector score, and supervise "
            "them with continuous IoU targets."
        ),
    )
    p.add_argument("--proposal-c2f-topk", type=int, default=5)
    p.add_argument("--proposal-c2f-topq", type=int, default=2)
    p.add_argument("--proposal-c2f-score-weight", type=float, default=0.25)
    p.add_argument(
        "--proposal-pcl-cluster-weight",
        type=float,
        default=0.0,
        help=(
            "PCL/OICR-style cluster mining loss for proposal selector logits. "
            "High-IoU candidates become cluster centers; overlapping proposals "
            "receive soft localization-quality targets, far-away proposals are "
            "negatives, and ambiguous mid-overlap proposals are ignored."
        ),
    )
    p.add_argument(
        "--proposal-pcl-cluster-topk",
        type=int,
        default=2,
        help="Number of detached high-IoU candidate centers used by --proposal-pcl-cluster-weight.",
    )
    p.add_argument(
        "--proposal-pcl-cluster-iou-threshold",
        type=float,
        default=0.30,
        help="AABB IoU threshold to assign a proposal to a PCL-style positive candidate cluster.",
    )
    p.add_argument(
        "--proposal-pcl-cluster-bg-threshold",
        type=float,
        default=0.08,
        help=(
            "AABB IoU threshold below which proposals are treated as clean "
            "background for --proposal-pcl-cluster-weight; proposals between "
            "bg and fg thresholds are ignored."
        ),
    )
    p.add_argument(
        "--proposal-handoff-oracle-distill-weight",
        type=float,
        default=0.0,
        help=(
            "Train-time only handoff distillation from the final selected quad "
            "toward the detached oracle proposal candidate. This directly "
            "targets the oracle-gap failure where the bank has a good candidate "
            "but the soft final handoff averages or selects the wrong geometry."
        ),
    )
    p.add_argument(
        "--proposal-handoff-oracle-distill-axis-weight",
        type=float,
        default=0.25,
        help="Axis-alignment term mixed into --proposal-handoff-oracle-distill-weight.",
    )
    p.add_argument(
        "--proposal-handoff-detach-selection-geometry",
        action="store_true",
        help=(
            "Detach candidate quads only inside the final refined-proposal "
            "handoff weighted sum. Final quad losses still train selector "
            "weights, but they cannot rewrite proposal/refiner geometry into "
            "a soft averaged ring."
        ),
    )
    p.add_argument(
        "--proposal-quad-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary min-over-candidates proposal loss on ordered quad corners.",
    )
    p.add_argument(
        "--stage1-full-proposal-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary min-over-candidates localization loss on the original "
            "Stage1 full proposal bank when later refine-all/top-k paths "
            "replace proposal_quads with refined candidates."
        ),
    )
    p.add_argument(
        "--stage1-full-proposal-quad-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary ordered-quad loss on the original Stage1 full proposal bank.",
    )
    p.add_argument(
        "--stage1-full-proposal-quality-weight",
        type=float,
        default=0.0,
        help="SmoothL1 quality calibration loss for original Stage1 full proposal logits.",
    )
    p.add_argument(
        "--stage1-full-proposal-varifocal-weight",
        type=float,
        default=0.0,
        help="Varifocal/GFL-style IoU-aware quality loss for original Stage1 full proposal logits.",
    )
    p.add_argument(
        "--proposal-selection-mode",
        choices=["train_soft_eval_hard", "soft", "hard"],
        default="train_soft_eval_hard",
        help="How proposal quality logits are converted into the selected Stage1 proposal.",
    )
    p.add_argument(
        "--proposal-selection-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for proposal selection when soft selection is enabled.",
    )
    p.add_argument(
        "--proposal-selection-topk",
        type=int,
        default=0,
        help=(
            "If >0, soft proposal selection keeps only the top-k proposal logits "
            "before softmax.  This tests sparse soft selection between hard "
            "argmax and averaging all candidates."
        ),
    )
    p.add_argument(
        "--proposal-selection-straight-through",
        action="store_true",
        help=(
            "For soft proposal selection, use a straight-through argmax: the "
            "forward selected quad is one real candidate, while gradients flow "
            "as if the softmax weights were used. This tests whether soft "
            "candidate averaging is the handoff bottleneck."
        ),
    )
    p.add_argument(
        "--proposal-selection-train-logit-noise-std",
        type=float,
        default=0.0,
        help=(
            "Training-only Gaussian noise added to proposal selector logits "
            "before selected-quad construction.  Auxiliary quality/rank losses "
            "still see clean logits."
        ),
    )
    p.add_argument(
        "--proposal-selection-train-drop-prob",
        type=float,
        default=0.0,
        help=(
            "Training-only random proposal dropout before selector softmax. "
            "At least one candidate is kept per frame; eval uses the clean bank."
        ),
    )
    p.add_argument(
        "--proposal-geom-temporal-fusion",
        action="store_true",
        help=(
            "QTNet-style proposal temporal fusion: before final proposal "
            "selection, each candidate reads adjacent-frame candidate tokens "
            "matched by proposal geometry/source rather than by fixed candidate "
            "index."
        ),
    )
    p.add_argument(
        "--proposal-geom-temporal-fusion-scale",
        type=float,
        default=0.30,
        help="Residual scale for --proposal-geom-temporal-fusion token/logit updates.",
    )
    p.add_argument(
        "--proposal-geom-temporal-center-scale",
        type=float,
        default=0.18,
        help="Normalized center-distance scale used in geometry-matched temporal proposal attention.",
    )
    p.add_argument(
        "--proposal-geom-temporal-source-penalty",
        type=float,
        default=4.0,
        help="Soft attention penalty when adjacent-frame proposals come from different proposal sources.",
    )
    p.add_argument(
        "--proposal-clip-context-fusion",
        action="store_true",
        help=(
            "SELSA/TubeR-style proposal context: before final proposal "
            "selection, flatten all 16-frame candidates into one sequence so "
            "proposal selectors can attend to the full clip candidate bank."
        ),
    )
    p.add_argument(
        "--proposal-clip-context-scale",
        type=float,
        default=0.22,
        help="Residual scale for --proposal-clip-context-fusion token/logit updates.",
    )
    p.add_argument(
        "--proposal-clip-context-dropout",
        type=float,
        default=0.10,
        help="Dropout used inside the full-clip proposal context attention.",
    )
    p.add_argument(
        "--proposal-frame-anchor-memory",
        action="store_true",
        help=(
            "RMem-style restricted proposal memory: learn a small set of "
            "essential frames from the 16-frame proposal bank, aggregate their "
            "frame-level proposal evidence, and broadcast that memory back to "
            "candidate tokens/logits before final handoff."
        ),
    )
    p.add_argument(
        "--proposal-frame-anchor-scale",
        type=float,
        default=0.20,
        help="Residual scale for --proposal-frame-anchor-memory token/logit updates.",
    )
    p.add_argument(
        "--proposal-frame-anchor-topk",
        type=int,
        default=2,
        help="Number of high-anchor frames kept in the soft frame memory. 0 keeps all frames.",
    )
    p.add_argument(
        "--proposal-frame-anchor-temperature",
        type=float,
        default=0.12,
        help="Softmax temperature for selecting essential frames.",
    )
    p.add_argument(
        "--proposal-frame-anchor-candidate-temperature",
        type=float,
        default=0.45,
        help="Detached candidate-softmax temperature used to summarize each frame before anchor selection.",
    )
    p.add_argument(
        "--proposal-frame-anchor-loss-weight",
        type=float,
        default=0.0,
        help="KL loss weight that trains frame-anchor logits toward per-frame proposal-oracle IoU.",
    )
    p.add_argument(
        "--proposal-frame-anchor-loss-tau",
        type=float,
        default=0.05,
        help="Temperature for converting per-frame proposal-oracle IoU into frame-anchor targets.",
    )
    p.add_argument(
        "--proposal-final-fusion",
        action="store_true",
        help=(
            "For proposal_point_temporal: keep the soft proposal weights but "
            "replace raw weighted-average quad output with a learned residual "
            "geometry head over weighted candidate tokens/geometry. This tests "
            "a Sparse-RCNN/DAB-style candidate handoff without hard argmax."
        ),
    )
    p.add_argument(
        "--proposal-final-fusion-scale",
        type=float,
        default=0.18,
        help="Residual scale for --proposal-final-fusion relative to candidate side length.",
    )
    p.add_argument(
        "--proposal-quality-readout",
        action="store_true",
        help=(
            "Rescore each proposal with local V-JEPA-derived features sampled "
            "at the candidate center and corners.  This targets selector/logit "
            "misalignment with candidate quad IoU."
        ),
    )
    p.add_argument(
        "--proposal-quality-readout-detach",
        action="store_true",
        help=(
            "For proposal_point_temporal with --proposal-quality-readout: stop "
            "quality-readout gradients from flowing back into proposal geometry "
            "and shared V-JEPA features. The readout still trains and is used "
            "for proposal selection at inference."
        ),
    )
    p.add_argument(
        "--proposal-quality-readout-replace-logits",
        action="store_true",
        help=(
            "For proposal_point_temporal with --proposal-quality-readout: use "
            "the local readout quality as the selector logit instead of adding "
            "it as a residual to the base proposal logit.  This tests whether "
            "the old proposal-logit bias is preventing recovery of high-IoU "
            "candidate oracles."
        ),
    )
    p.add_argument(
        "--proposal-quality-readout-traj-context",
        action="store_true",
        help=(
            "For proposal_point_temporal with --proposal-quality-readout and "
            "tool keypoints: append a learned tool-trajectory/frame-role "
            "context to each candidate quality readout.  This follows "
            "trajectory-conditioned hotspot decoders while keeping the readout "
            "separate from candidate geometry when detach is enabled."
        ),
    )
    p.add_argument(
        "--proposal-quality-readout-points",
        choices=["center_corners", "center_corners_edges", "aligned_grid_3x3", "aligned_grid_5x5"],
        default="center_corners",
        help=(
            "Spatial samples used by the proposal-quality readout. "
            "center_corners uses 5 points: candidate center and four corners. "
            "center_corners_edges uses 9 points by adding four edge midpoints, "
            "which gives the selector more tight-quad shape evidence without "
            "changing candidate geometry. aligned_grid_* samples a regular "
            "grid inside each candidate quad, following the oriented feature "
            "alignment idea used by S2A-Net AlignConv."
        ),
    )
    p.add_argument(
        "--proposal-quality-readout-pool",
        choices=["mean", "attention"],
        default="mean",
        help=(
            "How to pool multiple candidate-local readout points before the "
            "proposal quality head. mean is the historical H321/H325 path. "
            "attention learns candidate-conditioned point weights, following "
            "the Cascade-DETR / Deformable-DETR idea that the candidate "
            "reference should guide local evidence aggregation."
        ),
    )
    p.add_argument(
        "--proposal-support-overlap-score",
        action="store_true",
        help=(
            "Rescore proposal candidates by their normalized overlap with the "
            "predicted support heatmap. This turns the H435 contact-support "
            "auxiliary signal into a candidate-selection signal without using "
            "GT or eval-only source biases."
        ),
    )
    p.add_argument("--proposal-support-overlap-scale", type=float, default=0.0)
    p.add_argument("--proposal-support-overlap-grid-size", type=int, default=24)
    p.add_argument(
        "--proposal-support-overlap-detach",
        action="store_true",
        help="Detach the support heatmap before computing support-overlap proposal scores.",
    )
    p.add_argument(
        "--proposal-support-geom-score",
        action="store_true",
        help=(
            "Rescore candidates by center/area agreement with the predicted "
            "support heatmap.  This is a deployable analogue of the GT-geometry "
            "audit score: candidate center should be near support centroid and "
            "candidate area should match support mass."
        ),
    )
    p.add_argument("--proposal-support-geom-scale", type=float, default=0.0)
    p.add_argument("--proposal-support-geom-grid-size", type=int, default=24)
    p.add_argument("--proposal-support-geom-center-tau", type=float, default=0.07)
    p.add_argument("--proposal-support-geom-area-tau", type=float, default=0.75)
    p.add_argument(
        "--proposal-support-geom-detach",
        action="store_true",
        help="Detach the support heatmap before computing support-geometry proposal scores.",
    )
    p.add_argument(
        "--proposal-support-moment-proposal",
        action="store_true",
        help=(
            "Append one proposal decoded from the predicted support heatmap "
            "peak. The candidate is built from top-token support moments, so "
            "it tests contact-support candidate generation rather than another "
            "post-hoc scalar rescore."
        ),
    )
    p.add_argument("--proposal-support-moment-top-frac", type=float, default=0.035)
    p.add_argument("--proposal-support-moment-size-scale", type=float, default=1.8)
    p.add_argument("--proposal-support-moment-temperature", type=float, default=0.16)
    p.add_argument(
        "--proposal-support-moment-detach",
        action="store_true",
        help="Detach the support logits before decoding the support-moment proposal.",
    )
    p.add_argument(
        "--proposal-decoupled-selector",
        action="store_true",
        help=(
            "Add a separate proposal selector head over candidate token, quad "
            "geometry, source id, and detached base logit. This targets the "
            "failure where useful ROI candidates exist but the coupled selector "
            "keeps a default/source prior."
        ),
    )
    p.add_argument(
        "--proposal-decoupled-selector-detach",
        action="store_true",
        help=(
            "Detach candidate token/geometry/base-logit inputs before the "
            "decoupled selector head, so selector-ranking losses do not rewrite "
            "the proposal generator that produced the candidate bank."
        ),
    )
    p.add_argument(
        "--proposal-decoupled-selector-replace-logits",
        action="store_true",
        help=(
            "Use the decoupled selector score as proposal_logits for handoff "
            "and selector losses. Without this flag it is only a residual delta "
            "on the existing proposal logits."
        ),
    )
    p.add_argument(
        "--proposal-decoupled-selector-scale",
        type=float,
        default=1.0,
        help="Scale applied to the decoupled selector logit delta.",
    )
    p.add_argument(
        "--proposal-decoupled-selector-base-scale",
        type=float,
        default=0.0,
        help=(
            "When replacing logits, add this scale times the detached old "
            "proposal logit to the decoupled selector. 0.0 makes the new head "
            "fully own ranking; small positive values preserve a weak prior."
        ),
    )
    p.add_argument(
        "--proposal-decoupled-selector-readout",
        action="store_true",
        help=(
            "Append a candidate-local V-JEPA readout vector to the decoupled "
            "selector input.  This keeps proposal generation detached while "
            "letting the selector see local evidence around each candidate "
            "instead of relying only on source/logit/quad geometry."
        ),
    )
    p.add_argument(
        "--refined-quality-readout",
        action="store_true",
        help=(
            "For proposal_point_temporal with --refine-all-proposals: rescore "
            "post-refined candidates by sampling local V-JEPA features at each "
            "refined candidate geometry.  This is the H305 scorer/handoff path."
        ),
    )
    p.add_argument(
        "--refined-quality-readout-detach",
        action="store_true",
        help=(
            "Stop post-refined local-readout scorer gradients from flowing back "
            "into refined geometry and shared token features. The scorer still "
            "trains and is used for hard candidate selection at eval."
        ),
    )
    p.add_argument(
        "--refined-quality-readout-replace-logits",
        action="store_true",
        help=(
            "Use the post-refined local-readout quality as the refined selector "
            "logit instead of adding it as a residual to grouped-query logits."
        ),
    )
    p.add_argument(
        "--refined-quality-readout-points",
        choices=["center_corners", "center_corners_edges", "aligned_grid_3x3", "aligned_grid_5x5"],
        default="aligned_grid_3x3",
        help=(
            "Spatial samples used by the post-refined candidate scorer. "
            "aligned_grid_3x3 is the default H305 setting because it forces the "
            "quality head to look inside the refined oriented region, not only "
            "at point-query features."
        ),
    )
    p.add_argument(
        "--refiner-aligned-context",
        action="store_true",
        help=(
            "For proposal_point_temporal with --refine-all-proposals: inject "
            "candidate-aligned V-JEPA grid features into the point refiner "
            "queries before the decoder.  This is an S2A/Deformable-style "
            "feature-alignment path, not a late scalar scorer."
        ),
    )
    p.add_argument(
        "--refiner-aligned-context-points",
        choices=["aligned_grid_3x3", "aligned_grid_5x5"],
        default="aligned_grid_3x3",
        help="Candidate-aligned grid used by --refiner-aligned-context.",
    )
    p.add_argument(
        "--refiner-aligned-context-scale",
        type=float,
        default=0.35,
        help="Residual scale for candidate-aligned context injected into refiner queries.",
    )
    p.add_argument(
        "--local-grid-refine",
        action="store_true",
        help=(
            "For quad_local/proposal_point_temporal: replace the final direct "
            "point output with a Grid-R-CNN/SABL-style local corner distribution "
            "head.  A rotated local grid is built around the current quad and "
            "each ordered corner is decoded by softargmax over that grid."
        ),
    )
    p.add_argument("--local-grid-refine-grid-size", type=int, default=5)
    p.add_argument("--local-grid-refine-scale", type=float, default=1.35)
    p.add_argument("--local-grid-refine-temperature", type=float, default=0.20)
    p.add_argument("--local-grid-refine-prior-scale", type=float, default=6.0)
    p.add_argument("--local-grid-refine-prior-sigma", type=float, default=0.28)
    p.add_argument(
        "--local-grid-refine-detach-base",
        action="store_true",
        help="Detach the base quad used to build the local-grid search region.",
    )
    p.add_argument(
        "--disable-final-local-grid-refine",
        action="store_true",
        help=(
            "When local-grid refinement is enabled for refine-all proposal "
            "runs, keep the candidate-grid refinement but skip the final "
            "selected-quad grid snap."
        ),
    )
    p.add_argument(
        "--local-grid-corner-ce-weight",
        type=float,
        default=0.0,
        help="Auxiliary CE target for the nearest local-grid point per ordered GT corner.",
    )
    p.add_argument(
        "--candidate-local-grid-corner-ce-weight",
        type=float,
        default=0.0,
        help=(
            "For refine-all local-grid runs, apply the local-grid CE only to "
            "the best pre-grid candidate in each frame. This trains the "
            "candidate-bank refiner without forcing poor candidates to chase GT."
        ),
    )
    p.add_argument(
        "--local-grid-confidence-score",
        action="store_true",
        help=(
            "For refine-all local-grid runs, add candidate-local-grid peak "
            "margin as a correspondence-style selector score before final "
            "proposal selection."
        ),
    )
    p.add_argument("--local-grid-confidence-scale", type=float, default=0.35)
    p.add_argument(
        "--local-grid-confidence-detach",
        action="store_true",
        help="Detach local-grid confidence before adding it to proposal logits.",
    )
    p.add_argument(
        "--candidate-mask-quality",
        action="store_true",
        help=(
            "For refine-all proposal_point_temporal runs: predict a "
            "candidate-conditioned local contact mask on an aligned grid and "
            "a mask-quality logit for each refined candidate.  This follows "
            "Mask R-CNN/Mask Scoring-style candidate-local evidence instead "
            "of another global scalar selector."
        ),
    )
    p.add_argument("--candidate-mask-grid-size", type=int, default=7)
    p.add_argument("--candidate-mask-loss-weight", type=float, default=0.0)
    p.add_argument("--candidate-mask-quality-weight", type=float, default=0.0)
    p.add_argument("--candidate-mask-score-scale", type=float, default=0.0)
    p.add_argument(
        "--candidate-mask-score-detach",
        action="store_true",
        help="Detach the candidate mask-quality logit before adding it to proposal logits.",
    )
    p.add_argument("--candidate-mask-positive-topk", type=int, default=3)
    p.add_argument("--candidate-mask-pos-weight", type=float, default=2.0)
    p.add_argument("--proposal-handoff-mask-loss-weight", type=float, default=0.0)
    p.add_argument("--proposal-handoff-mask-quality-weight", type=float, default=0.0)
    p.add_argument(
        "--proposal-handoff-mask-boundary-weight",
        type=float,
        default=0.0,
        help=(
            "BoundaryLoss-style auxiliary for proposal handoff masks.  Sampled "
            "point probabilities are multiplied by signed distance to the GT "
            "quad boundary, encouraging compact boundary-aware local masks for "
            "tiny contact regions."
        ),
    )
    p.add_argument(
        "--proposal-handoff-mask-boundary-scale-px",
        type=float,
        default=8.0,
        help="Clamp/normalize signed boundary distances in 64-grid pixels for the handoff mask boundary loss.",
    )
    p.add_argument("--proposal-handoff-mask-positive-topk", type=int, default=6)
    p.add_argument("--proposal-handoff-mask-pos-weight", type=float, default=3.0)
    p.add_argument(
        "--proposal-pointset-refine",
        action="store_true",
        help=(
            "For proposal_point_temporal: refine every candidate quad as a "
            "4-point set using local V-JEPA token readout at its corners. "
            "This follows the OrientedRepPoints idea that the region should be "
            "represented as adaptive points rather than only as cxcywh/angle."
        ),
    )
    p.add_argument(
        "--proposal-pointset-refine-scale",
        type=float,
        default=0.18,
        help="Scale for the point-set residual relative to candidate side length.",
    )
    p.add_argument(
        "--proposal-pointset-refine-grad-mul",
        type=float,
        default=0.25,
        help=(
            "Gradient multiplier for the candidate points used by the point-set "
            "readout, mirroring RepPoints' init-point gradient throttling."
        ),
    )
    p.add_argument(
        "--proposal-pointset-quality-scale",
        type=float,
        default=1.0,
        help=(
            "Scale for the point-set local-readout quality-logit correction. "
            "Set to 0 to test candidate-as-reference geometric refinement "
            "without giving the new local scorer ownership of selection."
        ),
    )
    p.add_argument(
        "--proposal-pointset-refine-update-token",
        action="store_true",
        help=(
            "Also update candidate tokens with the averaged local point-set "
            "readout before selector scoring."
        ),
    )
    p.add_argument(
        "--proposal-set-refiner",
        action="store_true",
        help=(
            "For proposal_point_temporal: predict the final contact quad from "
            "the whole proposal set instead of relying on the hard top1 "
            "selector.  This is a Sparse-RCNN/DETR-style set consumption path; "
            "proposal logits are still exported for diagnostics."
        ),
    )
    p.add_argument(
        "--proposal-set-refiner-topk",
        type=int,
        default=0,
        help=(
            "Optional number of highest-logit candidates kept by "
            "--proposal-set-refiner. 0 keeps all candidates, which is safest "
            "when oracle ranks are often outside top3/top5."
        ),
    )
    p.add_argument(
        "--proposal-set-refiner-scale",
        type=float,
        default=0.22,
        help="Residual scale for --proposal-set-refiner relative to candidate-set side length.",
    )
    p.add_argument(
        "--proposal-set-refiner-temperature",
        type=float,
        default=0.90,
        help="Softmax temperature for the proposal-set refiner's initial logit prior.",
    )
    p.add_argument(
        "--proposal-set-refiner-detach",
        action="store_true",
        help=(
            "Detach candidate tokens/geometry/logits before the set refiner. "
            "This lets the final refiner learn to consume the proposal bank "
            "without letting final loss collapse proposal generation."
        ),
    )
    p.add_argument(
        "--proposal-set-refiner-rank-weight",
        type=float,
        default=0.0,
        help=(
            "CE supervision for the proposal-set refiner's own scores.  The "
            "target is the highest true quad-IoU candidate, but inference still "
            "uses only the learned set-refiner scores."
        ),
    )
    p.add_argument(
        "--proposal-set-refiner-soft-rank-weight",
        type=float,
        default=0.0,
        help=(
            "KL supervision for set-refiner scores against a softmax over true "
            "candidate quad IoUs.  This is a softer assignment alternative to "
            "--proposal-set-refiner-rank-weight."
        ),
    )
    p.add_argument(
        "--proposal-set-refiner-soft-rank-tau",
        type=float,
        default=0.07,
        help="Temperature for --proposal-set-refiner-soft-rank-weight targets.",
    )
    p.add_argument(
        "--proposal-handoff-final-fusion",
        action="store_true",
        help=(
            "For proposal_point_temporal refine-all: after the final candidate "
            "handoff over refined+stage1 candidates, replace the raw soft "
            "weighted quad with a zero-initialized residual geometry head. "
            "This targets the H461/H466 candidate-to-final bottleneck."
        ),
    )
    p.add_argument(
        "--proposal-handoff-final-fusion-scale",
        type=float,
        default=0.18,
        help="Residual scale for --proposal-handoff-final-fusion relative to handoff quad side length.",
    )
    p.add_argument(
        "--proposal-local-corr-score",
        action="store_true",
        help=(
            "For proposal_point_temporal: add a tracking-style local "
            "correspondence score before final handoff by comparing "
            "candidate-aligned V-JEPA token grids with tool-keypoint token "
            "readouts."
        ),
    )
    p.add_argument("--proposal-local-corr-scale", type=float, default=0.45)
    p.add_argument("--proposal-local-corr-grid-size", type=int, default=3)
    p.add_argument("--proposal-local-corr-temperature", type=float, default=0.08)
    p.add_argument(
        "--proposal-local-corr-detach",
        action="store_true",
        help="Detach local-correlation features before the scoring head.",
    )
    p.add_argument(
        "--proposal-trajectory-rerank",
        action="store_true",
        help=(
            "For proposal_point_temporal: let every proposal token cross-attend "
            "to the full 16-frame tool-keypoint/action-role trajectory before "
            "adding a learned proposal-logit delta.  This keeps the visual "
            "candidate pool intact while making trajectory a selector context."
        ),
    )
    p.add_argument(
        "--proposal-trajectory-rerank-scale",
        type=float,
        default=0.45,
        help="Scale applied to the learned trajectory-conditioned proposal-logit delta.",
    )
    p.add_argument(
        "--proposal-trajectory-refine",
        action="store_true",
        help=(
            "For proposal_point_temporal: let proposal tokens cross-attend to "
            "the full 16-frame tool trajectory and predict a zero-init oriented "
            "geometry residual for every proposal.  Unlike rerank, this updates "
            "candidate geometry while preserving the full candidate pool."
        ),
    )
    p.add_argument(
        "--proposal-trajectory-refine-scale",
        type=float,
        default=0.30,
        help="Scale applied to the learned trajectory-conditioned proposal geometry residual.",
    )
    p.add_argument(
        "--condition-prior-proposal",
        action="store_true",
        help=(
            "For proposal_point_temporal: append one explicit proposal generated "
            "from tool-keypoint trajectory, action-range role, and ids."
        ),
    )
    p.add_argument(
        "--use-h259-prior-proposal",
        action="store_true",
        help=(
            "For proposal_point_temporal: append a stronger mandatory proposal "
            "from the H259 trajectory-prior architecture, then let the V-JEPA "
            "proposal/refiner path score or refine it. This is a stricter test "
            "than the lightweight condition-prior proposal."
        ),
    )
    p.add_argument(
        "--condition-prior-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary loss on the explicit condition-prior proposal quad.",
    )
    p.add_argument(
        "--trajectory-query-proposal-count",
        type=int,
        default=0,
        help=(
            "For proposal_point_temporal: append trajectory-conditioned proposal "
            "queries that cross-attend to V-JEPA spatial tokens before predicting "
            "oriented quad candidates."
        ),
    )
    p.add_argument(
        "--trajectory-query-output-mode",
        choices=["oriented_rect", "direct_quad"],
        default="oriented_rect",
        help=(
            "How trajectory-query proposals are parameterized. oriented_rect "
            "predicts center/size/axis; direct_quad predicts four ordered corners "
            "as residuals around a tool-trajectory anchor quad."
        ),
    )
    p.add_argument(
        "--trajectory-query-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary min-over-candidates loss on trajectory-query proposal quads.",
    )
    p.add_argument(
        "--use-trajectory-readout-bias",
        action="store_true",
        help=(
            "For proposal_point_temporal: use tool-trajectory anchor corners to "
            "read local V-JEPA features and bias visual proposal tokens, without "
            "appending trajectory-generated quads as final candidates."
        ),
    )
    p.add_argument(
        "--use-condition-prior-readout-bias",
        action="store_true",
        help=(
            "For proposal_point_temporal: first predict a learned condition-only "
            "quad prior from tool-keypoint trajectory, action-range/frame-role, "
            "and IDs, then use that quad as the V-JEPA readout anchor. This tests "
            "whether trajectory can be forced to become the localization prior "
            "instead of a weak side condition."
        ),
    )
    p.add_argument("--condition-prior-readout-scale", type=float, default=0.45)
    p.add_argument(
        "--trajectory-anchor-logit-bias-scale",
        type=float,
        default=0.0,
        help="If >0, add a deployable proposal-logit prior favoring candidates near the current tool-keypoint trajectory anchor.",
    )
    p.add_argument(
        "--trajectory-anchor-logit-bias-radius",
        type=float,
        default=0.22,
        help="Normalized image radius for the trajectory-anchor proposal-logit prior.",
    )
    p.add_argument(
        "--trajectory-anchor-axis-bias-weight",
        type=float,
        default=0.0,
        help="Optional axis-alignment term for the trajectory-anchor proposal-logit prior.",
    )
    p.add_argument(
        "--refine-all-proposals",
        action="store_true",
        help=(
            "For proposal_point_temporal: refine every proposal candidate with "
            "the point-temporal decoder before final proposal selection."
        ),
    )
    p.add_argument(
        "--refine-proposal-topk",
        type=int,
        default=0,
        help=(
            "For --refine-all-proposals: if >0, refine only the top-k proposals "
            "by Stage1 proposal logits.  This is a Sparse-RCNN/DAB-style "
            "candidate-to-final refinement path that avoids H243's all-proposal "
            "noise amplification."
        ),
    )
    p.add_argument(
        "--refine-recollect-stage1",
        action="store_true",
        help=(
            "For --refine-all-proposals: select from the union of recalled "
            "Stage1 proposals and refined proposals. This keeps earlier good "
            "queries selectable instead of letting the final refinement stage "
            "overwrite the candidate bank."
        ),
    )
    p.add_argument(
        "--refine-recollect-refined-base-only",
        action="store_true",
        help=(
            "For --refine-recollect-stage1: append only refined source0/base "
            "proposals after the recalled Stage1 bank. This avoids duplicating "
            "high-bias auxiliary sources such as source4 corner proposals."
        ),
    )
    p.add_argument(
        "--decoder-cascade-pad-ratios",
        type=str,
        default="",
        help="Comma-separated pad ratios for intermediate decoder layers, e.g. '0.35,0.15'. Final layer remains tight.",
    )
    p.add_argument("--moment-size-scale", type=float, default=4.8)
    p.add_argument("--token-noise-std", type=float, default=0.0)
    p.add_argument("--token-dropout-prob", type=float, default=0.0)
    p.add_argument(
        "--token-dropout-mode",
        choices=["zero", "mask_token"],
        default="zero",
        help="Replacement used by token/frame/block dropout. mask_token adds a learned vector to the ROI head.",
    )
    p.add_argument("--token-block-dropout-prob", type=float, default=0.0)
    p.add_argument("--token-block-size", type=int, default=5)
    p.add_argument("--token-frame-dropout-prob", type=float, default=0.0)
    p.add_argument(
        "--force-zero-tokens",
        action="store_true",
        help="Diagnostic mode: replace V-JEPA token cache with zeros during train/eval.",
    )
    p.add_argument("--condition-dropout", type=float, default=0.0)
    p.add_argument("--condition-id-dropout", type=float, default=0.0)
    p.add_argument(
        "--condition-id-drop-mode",
        choices=["random", "null"],
        default="random",
        help="random replaces categorical ids with another real class; null maps them to learned null embeddings.",
    )
    p.add_argument("--tool-mask-dropout", type=float, default=0.0)
    p.add_argument("--tool-mask-noise-std", type=float, default=0.0)
    p.add_argument("--tool-mask-morph-prob", type=float, default=0.0)
    p.add_argument("--tool-mask-morph-kernel", type=int, default=3)
    p.add_argument("--tool-mask-block-dropout-prob", type=float, default=0.0)
    p.add_argument("--tool-mask-block-size", type=int, default=7)
    p.add_argument(
        "--clean-pass-loss-weight",
        type=float,
        default=0.0,
        help="If >0, add a supervised loss on an uncorrupted condition pass.",
    )
    p.add_argument(
        "--clean-corrupt-consistency-weight",
        type=float,
        default=0.0,
        help="If >0, align corrupted-condition boxes to clean-condition boxes.",
    )
    p.add_argument(
        "--regularization-warmup-epochs",
        type=float,
        default=0.0,
        help="If >0, linearly ramp stochastic condition/token regularization to its configured value over this many epochs.",
    )
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="If >0, maintain an exponential moving average of model weights and select checkpoints by EMA validation.",
    )
    p.add_argument("--only-cached", action="store_true", default=True)
    p.add_argument("--allow-missing-tool-mask", action="store_true", default=True)
    p.add_argument(
        "--strict-tool-mask",
        action="store_true",
        help="Fail instead of substituting zero masks when --use-tool-mask is enabled and a mask file is missing.",
    )
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument(
        "--selection-metric",
        choices=["box_iou_mean", "quad_iou_mean"],
        default="box_iou_mean",
        help="Validation metric used for best checkpoint selection.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=161)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--amp", action="store_true")
    p.add_argument(
        "--eval-only-checkpoint",
        type=Path,
        default=None,
        help="Load this checkpoint, evaluate, write eval_condition_ablation.json, and exit.",
    )
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize model weights from this checkpoint before training with a fresh optimizer.",
    )
    p.add_argument(
        "--teacher-ensemble-checkpoints",
        type=str,
        default="",
        help=(
            "Comma-separated frozen checkpoints used only during training for "
            "selected-output distillation.  The teacher uses the same input contract "
            "as the student and is never optimized."
        ),
    )
    p.add_argument(
        "--teacher-ensemble-mode",
        choices=[
            "selected_mean",
            "selected_median",
            "selected_mean_median_blend",
            "selected_consensus_weighted_mean",
            "selected_mean_gate_first",
            "selected_mean_gate_median",
            "proposal_soft",
            "proposal_hard",
        ],
        default="selected_mean_gate_median",
        help="How to combine --teacher-ensemble-checkpoints during training distillation.",
    )
    p.add_argument(
        "--teacher-ensemble-temperature",
        type=float,
        default=0.45,
        help="Teacher ensemble temperature, matching --eval-ensemble-temperature semantics.",
    )
    p.add_argument(
        "--teacher-ensemble-consensus-threshold",
        type=float,
        default=0.026,
        help="Teacher consensus threshold for selected_mean_gate_* modes.",
    )
    p.add_argument(
        "--teacher-ensemble-blend-alpha",
        type=float,
        default=0.50,
        help="Teacher alpha for selected_mean_median_blend.",
    )
    p.add_argument(
        "--teacher-quad-distill-weight",
        type=float,
        default=0.0,
        help="SmoothL1 distillation weight from frozen teacher quad_xy to student quad_xy.",
    )
    p.add_argument(
        "--teacher-box-distill-weight",
        type=float,
        default=0.0,
        help="SmoothL1 distillation weight from frozen teacher boxes_cxcywh to student boxes_cxcywh.",
    )
    p.add_argument(
        "--teacher-cache-root",
        type=Path,
        default=None,
        help="Optional root containing split/sid.npz files with cached teacher quad_xy and boxes_cxcywh.",
    )
    p.add_argument(
        "--write-teacher-cache-root",
        type=Path,
        default=None,
        help=(
            "Generate cached teacher predictions under this root and exit. "
            "Requires --teacher-ensemble-checkpoints."
        ),
    )
    p.add_argument(
        "--write-teacher-cache-splits",
        type=str,
        default="train,val",
        help="Comma-separated dataset splits to write when --write-teacher-cache-root is set.",
    )
    p.add_argument(
        "--write-teacher-cache-proposals",
        action="store_true",
        help="Also save proposal_quads/logits/source_ids/selection_weights for failure analysis.",
    )
    p.add_argument(
        "--eval-ensemble-checkpoints",
        type=str,
        default="",
        help=(
            "Comma-separated additional checkpoints for eval-only strict checkpoint ensemble. "
            "The --eval-only-checkpoint path is used as the first member."
        ),
    )
    p.add_argument(
        "--eval-ensemble-mode",
        choices=[
            "selected_mean",
            "selected_median",
            "selected_mean_median_blend",
            "selected_consensus_weighted_mean",
            "selected_mean_gate_first",
            "selected_mean_gate_median",
            "proposal_soft",
            "proposal_hard",
        ],
        default="selected_mean",
        help="How to combine eval-only checkpoint ensemble outputs.",
    )
    p.add_argument(
        "--eval-ensemble-temperature",
        type=float,
        default=0.45,
        help=(
            "Softmax temperature for eval-only proposal_soft checkpoint ensemble. "
            "For selected_consensus_weighted_mean, this is the normalized corner-distance temperature."
        ),
    )
    p.add_argument(
        "--eval-ensemble-consensus-threshold",
        type=float,
        default=0.02,
        help=(
            "Normalized mean corner-disagreement threshold for selected_mean_gate_* "
            "eval-only ensemble modes.  Frames below the threshold use selected_mean; "
            "frames above it use the mode fallback."
        ),
    )
    p.add_argument(
        "--eval-ensemble-blend-alpha",
        type=float,
        default=0.50,
        help=(
            "For selected_mean_median_blend, output alpha * selected_mean + "
            "(1 - alpha) * selected_median."
        ),
    )
    p.add_argument(
        "--proposal-eval-source4-logit-bias",
        type=float,
        default=0.0,
        help=(
            "Eval-only diagnostic: add this constant to proposal logits whose "
            "source id is 4 before recomputing proposal selection.  Negative "
            "values test whether the corner-field/source4 shortcut is directly "
            "blocking candidate consumption.  Training is unaffected."
        ),
    )
    p.add_argument(
        "--eval-quad-voting-mode",
        choices=["none", "cluster_mean", "cluster_geomedian"],
        default="none",
        help=(
            "Eval-only H334 diagnostic: fuse proposal quads by a WBF-style "
            "candidate cluster instead of using the model's selected quad."
        ),
    )
    p.add_argument(
        "--eval-quad-voting-topk",
        type=int,
        default=0,
        help="Eval-only quad voting: keep top-k proposal logits before clustering. 0 means all candidates.",
    )
    p.add_argument(
        "--eval-quad-voting-temperature",
        type=float,
        default=0.45,
        help="Eval-only quad voting: softmax temperature for proposal logits.",
    )
    p.add_argument(
        "--eval-quad-voting-iou-threshold",
        type=float,
        default=0.20,
        help="Eval-only quad voting: AABB IoU threshold for joining the seed candidate cluster.",
    )
    p.add_argument(
        "--eval-quad-voting-score-mode",
        choices=["sum", "max", "seed"],
        default="sum",
        help="Eval-only quad voting: cluster score used to choose the final cluster.",
    )
    p.add_argument(
        "--eval-quad-voting-max-geomedian-iters",
        type=int,
        default=8,
        help="Eval-only quad voting: Weiszfeld iterations for cluster_geomedian.",
    )
    p.add_argument(
        "--eval-center-identity-shift",
        action="store_true",
        help=(
            "Eval-only H558 diagnostic: keep the selected final quad shape but "
            "shift its center toward a proposal-bank center computed from "
            "source-filtered proposal weights/logits."
        ),
    )
    p.add_argument(
        "--eval-center-identity-mode",
        choices=[
            "top1",
            "weight_all",
            "logit_all",
            "weight_no4",
            "logit_no4",
            "weight_s5",
            "logit_s5",
            "weight_s0s5",
            "logit_s0s5",
        ],
        default="weight_no4",
        help="Eval-only H558: proposal center source/scoring mode.",
    )
    p.add_argument("--eval-center-identity-topk", type=int, default=13)
    p.add_argument("--eval-center-identity-alpha", type=float, default=0.50)
    p.add_argument("--eval-center-identity-temperature", type=float, default=1.00)
    p.add_argument(
        "--eval-tubelet-dp",
        action="store_true",
        help=(
            "Eval-only H336 diagnostic: replace per-frame selected quads with "
            "a Viterbi path through proposal candidates over the 16-frame clip."
        ),
    )
    p.add_argument(
        "--eval-tubelet-dp-topk",
        type=int,
        default=0,
        help="Eval-only tubelet DP: keep top-k proposal logits per frame. 0 means all candidates.",
    )
    p.add_argument(
        "--eval-tubelet-dp-node-weight",
        type=float,
        default=0.25,
        help="Eval-only tubelet DP: weight for per-frame proposal log-softmax node score.",
    )
    p.add_argument(
        "--eval-tubelet-dp-edge-weight",
        type=float,
        default=1.0,
        help="Eval-only tubelet DP: global multiplier for adjacent-frame transition scores.",
    )
    p.add_argument(
        "--eval-tubelet-dp-edge-iou-weight",
        type=float,
        default=1.0,
        help="Eval-only tubelet DP: component weight for adjacent candidate AABB IoU.",
    )
    p.add_argument(
        "--eval-tubelet-dp-center-weight",
        type=float,
        default=0.5,
        help="Eval-only tubelet DP: component weight for adjacent center smoothness.",
    )
    p.add_argument(
        "--eval-tubelet-dp-corner-weight",
        type=float,
        default=0.5,
        help="Eval-only tubelet DP: component weight for adjacent ordered-corner smoothness.",
    )
    p.add_argument(
        "--eval-tubelet-dp-area-weight",
        type=float,
        default=0.25,
        help="Eval-only tubelet DP: component weight for adjacent AABB-area smoothness.",
    )
    p.add_argument(
        "--eval-tubelet-dp-aspect-weight",
        type=float,
        default=0.25,
        help="Eval-only tubelet DP: component weight for adjacent AABB-aspect smoothness.",
    )
    p.add_argument(
        "--eval-tubelet-dp-center-scale",
        type=float,
        default=0.18,
        help="Eval-only tubelet DP: normalized center-distance scale for smoothness.",
    )
    p.add_argument(
        "--eval-tubelet-dp-corner-scale",
        type=float,
        default=0.18,
        help="Eval-only tubelet DP: normalized ordered-corner-distance scale for smoothness.",
    )
    p.add_argument(
        "--eval-token-corner-track",
        action="store_true",
        help=(
            "Eval-only H337 diagnostic: propagate the selected quad corners in "
            "V-JEPA token space, then optionally snap each frame back to the "
            "nearest proposal candidate."
        ),
    )
    p.add_argument(
        "--eval-token-corner-track-mode",
        choices=["candidate_snap", "tracked_quad"],
        default="candidate_snap",
        help="Eval-only token corner tracking: use tracked corners directly or snap to proposal candidates.",
    )
    p.add_argument(
        "--eval-token-corner-track-seed",
        choices=["max_logit", "max_margin", "middle"],
        default="max_margin",
        help="Eval-only token corner tracking: frame used to initialize the four visual point tracks.",
    )
    p.add_argument(
        "--eval-token-corner-track-radius",
        type=float,
        default=0.22,
        help="Eval-only token corner tracking: normalized local search radius around previous point.",
    )
    p.add_argument(
        "--eval-token-corner-track-temperature",
        type=float,
        default=0.05,
        help="Eval-only token corner tracking: softargmax temperature over token cosine scores.",
    )
    p.add_argument(
        "--eval-token-corner-track-softargmax",
        action="store_true",
        help="Eval-only token corner tracking: use softargmax instead of hard nearest token.",
    )
    p.add_argument(
        "--eval-token-corner-track-logit-weight",
        type=float,
        default=0.20,
        help="Eval-only token corner tracking candidate-snap score weight for proposal log-softmax.",
    )
    p.add_argument(
        "--eval-token-corner-track-dist-weight",
        type=float,
        default=1.0,
        help="Eval-only token corner tracking candidate-snap score weight for corner distance.",
    )
    p.add_argument(
        "--eval-token-corner-track-dist-scale",
        type=float,
        default=0.12,
        help="Eval-only token corner tracking candidate-snap normalized corner-distance scale.",
    )
    p.add_argument(
        "--eval-candidate-memory",
        action="store_true",
        help=(
            "Eval-only H338 diagnostic: select proposal candidates by "
            "candidate-local V-JEPA memory compatibility across frames."
        ),
    )
    p.add_argument(
        "--eval-candidate-memory-seed",
        choices=["max_logit", "max_margin", "middle", "active", "active_mid", "active_center"],
        default="max_margin",
        help=(
            "Eval-only candidate memory: ranking used to choose memory frames. "
            "'active', 'active_mid', and 'active_center' use SurgAct action_range/active_16 "
            "as structural memory anchors."
        ),
    )
    p.add_argument(
        "--eval-candidate-memory-frames",
        type=int,
        default=3,
        help="Eval-only candidate memory: number of essential frames used as memory.",
    )
    p.add_argument(
        "--eval-candidate-memory-source",
        choices=["selected", "top_logit"],
        default="selected",
        help="Eval-only candidate memory: how to choose memory candidate descriptors on seed frames.",
    )
    p.add_argument(
        "--eval-candidate-memory-weight",
        type=float,
        default=1.0,
        help="Eval-only candidate memory: weight for candidate-memory cosine compatibility.",
    )
    p.add_argument(
        "--eval-candidate-memory-logit-weight",
        type=float,
        default=0.25,
        help="Eval-only candidate memory: weight for proposal log-softmax.",
    )
    p.add_argument(
        "--eval-condition-ablations",
        action="store_true",
        help="During eval-only, also evaluate VAL after removing/shuffling individual conditioning streams.",
    )
    p.add_argument(
        "--condition-ablation-variants",
        type=str,
        default=(
            "clean,zero_tool_kpt_all,zero_tool_kpt_heatmaps,zero_tool_kpt_vec,"
            "reverse_tool_kpt_time,zero_role,null_ids,zero_tool_role,"
            "zero_tool_role_ids,zero_tokens"
        ),
        help="Comma-separated ablation variants for --eval-condition-ablations.",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_split(data_root: Path, split: str) -> list[str]:
    path = data_root / f"{split}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def xyxy_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = box.unbind(dim=-1)
    return torch.stack([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (x1 - x0), (y1 - y0)], dim=-1)


def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(dim=-1)
    return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=-1).clamp(0.0, 1.0)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(float(eps), 1.0 - float(eps))
    return torch.log(x / (1.0 - x))


def xyxy_to_cxcywh_torch(box: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = box.unbind(dim=-1)
    return torch.stack([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (x1 - x0), (y1 - y0)], dim=-1).clamp(0.0, 1.0)


def quad_to_xyxy(quad: torch.Tensor, min_size: float = 0.02) -> torch.Tensor:
    x0 = quad[..., 0].min(dim=-1).values
    y0 = quad[..., 1].min(dim=-1).values
    x1 = quad[..., 0].max(dim=-1).values
    y1 = quad[..., 1].max(dim=-1).values
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    w = (x1 - x0).clamp_min(float(min_size))
    h = (y1 - y0).clamp_min(float(min_size))
    return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=-1).clamp(0.0, 1.0)


def roll_quad_start_min_sum(quad: torch.Tensor) -> torch.Tensor:
    """Rotate cyclic corner order so the first point is roughly top-left."""
    if quad.shape[-2:] != (4, 2):
        return quad
    start = quad.sum(dim=-1).argmin(dim=-1)
    base = torch.arange(4, device=quad.device, dtype=start.dtype)
    order = (start.unsqueeze(-1) + base.view(*([1] * start.ndim), 4)) % 4
    return quad.gather(-2, order.unsqueeze(-1).expand(*order.shape, 2))


def oriented_rect_to_quad(center: torch.Tensor, size: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Convert center/size/axis to a cyclic ordered normalized quad."""
    axis = F.normalize(axis, dim=-1, eps=1e-6)
    perp = torch.stack([-axis[..., 1], axis[..., 0]], dim=-1)
    half_u = axis * size[..., :1] * 0.5
    half_v = perp * size[..., 1:2] * 0.5
    quad = torch.stack(
        [
            center - half_u - half_v,
            center + half_u - half_v,
            center + half_u + half_v,
            center - half_u + half_v,
        ],
        dim=-2,
    )
    return roll_quad_start_min_sum(quad).clamp(0.0, 1.0)


def box_area_xyxy(box: torch.Tensor) -> torch.Tensor:
    wh = (box[..., 2:] - box[..., :2]).clamp(min=0.0)
    return wh[..., 0] * wh[..., 1]


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tl = torch.maximum(a[..., :2], b[..., :2])
    br = torch.minimum(a[..., 2:], b[..., 2:])
    inter = box_area_xyxy(torch.cat([tl, br], dim=-1))
    union = box_area_xyxy(a) + box_area_xyxy(b) - inter
    return inter / union.clamp_min(1e-6)


def giou_loss_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tl = torch.maximum(a[..., :2], b[..., :2])
    br = torch.minimum(a[..., 2:], b[..., 2:])
    inter = box_area_xyxy(torch.cat([tl, br], dim=-1))
    area_a = box_area_xyxy(a)
    area_b = box_area_xyxy(b)
    union = area_a + area_b - inter
    iou = inter / union.clamp_min(1e-6)
    c_tl = torch.minimum(a[..., :2], b[..., :2])
    c_br = torch.maximum(a[..., 2:], b[..., 2:])
    c_area = box_area_xyxy(torch.cat([c_tl, c_br], dim=-1)).clamp_min(1e-6)
    giou = iou - (c_area - union) / c_area
    return 1.0 - giou


def ciou_loss_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    iou = box_iou_xyxy(a, b)
    ac = (a[..., :2] + a[..., 2:]) * 0.5
    bc = (b[..., :2] + b[..., 2:]) * 0.5
    center_dist = ((ac - bc) ** 2).sum(dim=-1)
    c_tl = torch.minimum(a[..., :2], b[..., :2])
    c_br = torch.maximum(a[..., 2:], b[..., 2:])
    c_diag = ((c_br - c_tl) ** 2).sum(dim=-1).clamp_min(1e-6)
    aw = (a[..., 2] - a[..., 0]).clamp_min(1e-6)
    ah = (a[..., 3] - a[..., 1]).clamp_min(1e-6)
    bw = (b[..., 2] - b[..., 0]).clamp_min(1e-6)
    bh = (b[..., 3] - b[..., 1]).clamp_min(1e-6)
    v = (4.0 / (math.pi**2)) * (torch.atan(bw / bh) - torch.atan(aw / ah)).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v).clamp_min(1e-6)
    ciou = iou - center_dist / c_diag - alpha * v
    return 1.0 - ciou


def make_boxes_from_points(points: np.ndarray, pad_ratio: float, min_size: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    x0 = pts[..., 0].min(axis=-1)
    y0 = pts[..., 1].min(axis=-1)
    x1 = pts[..., 0].max(axis=-1)
    y1 = pts[..., 1].max(axis=-1)
    w = np.maximum(x1 - x0, min_size)
    h = np.maximum(y1 - y0, min_size)
    pad = np.maximum(w, h) * float(pad_ratio)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    w = np.maximum(w + 2.0 * pad, min_size)
    h = np.maximum(h + 2.0 * pad, min_size)
    out = np.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], axis=-1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def expand_xyxy_torch(box: torch.Tensor, pad_ratio: float, min_size: float = 0.0) -> torch.Tensor:
    x0, y0, x1, y1 = box.unbind(dim=-1)
    w = (x1 - x0).clamp_min(float(min_size))
    h = (y1 - y0).clamp_min(float(min_size))
    pad = torch.maximum(w, h) * float(pad_ratio)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    w = (w + 2.0 * pad).clamp_min(float(min_size))
    h = (h + 2.0 * pad).clamp_min(float(min_size))
    return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=-1).clamp(0.0, 1.0)


def parse_float_list(text: str) -> list[float]:
    if not str(text).strip():
        return []
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def order_quad_clockwise_np(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    flat = pts.reshape(-1, 4, 2)
    out = np.empty_like(flat)
    for i, q in enumerate(flat):
        c = q.mean(axis=0)
        angles = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
        ordered = q[np.argsort(angles)]
        start = np.argmin(ordered[:, 0] + ordered[:, 1])
        out[i] = np.roll(ordered, -int(start), axis=0)
    return out.reshape(pts.shape).astype(np.float32)


def pair_pool_16_to_8(x: np.ndarray, mode: str = "mean") -> np.ndarray:
    arr = np.asarray(x)
    if arr.shape[0] != 16:
        raise ValueError(f"expected 16 time steps, got shape={arr.shape}")
    arr = arr.reshape(8, 2, *arr.shape[1:])
    if mode == "max":
        return arr.max(axis=1)
    return arr.mean(axis=1)


def load_tool_mask_16(root: Path | None, sid: str) -> np.ndarray | None:
    if root is None:
        return None
    npy = root / f"{sid}.npy"
    if npy.is_file():
        return np.load(npy)
    npz = root / f"{sid}.npz"
    if npz.is_file():
        data = np.load(npz)
        key = "mask" if "mask" in data else ("masks" if "masks" in data else data.files[0])
        return data[key]
    d = root / sid
    if d.is_dir():
        files = sorted([*d.glob("*.png"), *d.glob("*.jpg"), *d.glob("*.jpeg")])
        if len(files) >= 16:
            masks = []
            for fp in files[:16]:
                img = Image.open(fp).convert("L")
                masks.append((np.asarray(img) > 127).astype(np.float32))
            return np.stack(masks, axis=0)
    return None


def downsample_mask_16_to_8_24(mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.zeros((8, 1, 24, 24), dtype=np.float32)
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[:, 0] if arr.shape[1] == 1 else arr[..., 0]
    if arr.shape[0] != 16:
        raise ValueError(f"tool mask must have 16 frames, got shape={arr.shape}")
    out = []
    for t in range(16):
        img = Image.fromarray((arr[t] > 0.5).astype(np.uint8) * 255)
        img = img.resize((24, 24), Image.BILINEAR)
        out.append(np.asarray(img, dtype=np.float32) / 255.0)
    out16 = np.stack(out, axis=0)
    out8 = pair_pool_16_to_8(out16, mode="max")
    return out8[:, None, :, :].astype(np.float32)


def regularization_scale(args: argparse.Namespace, epoch: int) -> float:
    warmup = float(args.regularization_warmup_epochs)
    if warmup <= 0.0:
        return 1.0
    return float(min(max(float(epoch) / warmup, 0.0), 1.0))


def apply_token_regularization(
    tokens: torch.Tensor,
    args: argparse.Namespace,
    scale: float = 1.0,
    mask_token: torch.Tensor | None = None,
) -> torch.Tensor:
    """Training-only corruption for frozen V-JEPA cache tokens.

    The cache tensor is shaped [B, T=8, H=24, W=24, C]. Gaussian noise jitters
    values, while token/frame/block dropout forces the ROI head not to depend on
    a small set of memorized token locations.
    """
    out = tokens
    scale = float(scale)
    mode = str(getattr(args, "token_dropout_mode", "zero"))
    use_mask_token = mode == "mask_token"
    if use_mask_token and mask_token is None:
        raise ValueError("--token-dropout-mode mask_token requires model.token_mask_value")

    def replace_with_mask_token(x: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
        token = mask_token.to(device=x.device, dtype=x.dtype).view(1, 1, 1, 1, x.shape[-1])
        while drop.ndim < x.ndim:
            drop = drop.unsqueeze(-1)
        return torch.where(drop.to(torch.bool), token, x)

    noise_std = float(args.token_noise_std) * scale
    if noise_std > 0.0:
        out = out + torch.randn_like(out) * noise_std
    p_token = min(max(float(args.token_dropout_prob) * scale, 0.0), 0.95)
    if p_token > 0.0:
        drop = torch.rand(out.shape[:-1], device=out.device) < p_token
        if use_mask_token:
            out = replace_with_mask_token(out, drop)
        else:
            keep = (~drop).to(out.dtype)
            out = out * keep[..., None] / max(1.0 - p_token, 1e-6)
    p_frame = min(max(float(args.token_frame_dropout_prob) * scale, 0.0), 0.95)
    if p_frame > 0.0:
        drop_t = torch.rand(out.shape[0], out.shape[1], 1, 1, device=out.device) < p_frame
        if use_mask_token:
            out = replace_with_mask_token(out, drop_t)
        else:
            keep_t = (~drop_t).to(out.dtype).unsqueeze(-1)
            out = out * keep_t / max(1.0 - p_frame, 1e-6)
    p_block = min(max(float(args.token_block_dropout_prob) * scale, 0.0), 0.95)
    block = max(int(args.token_block_size), 1)
    if p_block > 0.0 and block > 1:
        b, t, h, w, _ = out.shape
        drop_block = torch.zeros((b, t, h, w), device=out.device, dtype=torch.bool)
        half = block // 2
        for bi in range(b):
            for ti in range(t):
                if torch.rand((), device=out.device) > p_block:
                    continue
                cy = int(torch.randint(0, h, (), device=out.device).item())
                cx = int(torch.randint(0, w, (), device=out.device).item())
                y0 = max(cy - half, 0)
                y1 = min(y0 + block, h)
                x0 = max(cx - half, 0)
                x1 = min(x0 + block, w)
                drop_block[bi, ti, y0:y1, x0:x1] = True
        if use_mask_token:
            out = replace_with_mask_token(out, drop_block)
        else:
            out = out * (~drop_block).to(out.dtype)[..., None]
    return out


def apply_tool_mask_regularization(
    tool_mask: torch.Tensor,
    args: argparse.Namespace,
    scale: float = 1.0,
) -> torch.Tensor:
    """Training-only corruption for the optional tool-mask condition.

    The SAM-derived tool mask is a condition, not a target.  Full modality
    dropout tests whether the ROI head can fall back to V-JEPA tokens; boundary
    morphology and block dropout simulate common over/under-segmentation errors.
    """
    out = tool_mask
    scale = float(scale)
    p_mask = min(max(float(args.tool_mask_dropout) * scale, 0.0), 0.95)
    if p_mask > 0.0:
        keep_mask = (torch.rand(out.shape[0], 1, 1, 1, 1, device=out.device) > p_mask).to(out.dtype)
        out = out * keep_mask

    p_morph = min(max(float(args.tool_mask_morph_prob) * scale, 0.0), 0.95)
    k = max(int(args.tool_mask_morph_kernel), 1)
    if p_morph > 0.0 and k > 1:
        if k % 2 == 0:
            k += 1
        b, t, c, h, w = out.shape
        flat = out.reshape(b * t * c, 1, h, w)
        dilated = F.max_pool2d(flat, kernel_size=k, stride=1, padding=k // 2)
        eroded = 1.0 - F.max_pool2d(1.0 - flat, kernel_size=k, stride=1, padding=k // 2)
        morph = torch.where(
            (torch.rand(b * t * c, 1, 1, 1, device=out.device) < 0.5),
            dilated,
            eroded,
        ).reshape_as(out)
        apply = torch.rand(b, t, 1, 1, 1, device=out.device) < p_morph
        out = torch.where(apply, morph, out)

    p_block = min(max(float(args.tool_mask_block_dropout_prob) * scale, 0.0), 0.95)
    block = max(int(args.tool_mask_block_size), 1)
    if p_block > 0.0 and block > 1:
        b, t, _, h, w = out.shape
        keep = torch.ones((b, t, h, w), device=out.device, dtype=out.dtype)
        half = block // 2
        for bi in range(b):
            for ti in range(t):
                if torch.rand((), device=out.device) > p_block:
                    continue
                cy = int(torch.randint(0, h, (), device=out.device).item())
                cx = int(torch.randint(0, w, (), device=out.device).item())
                y0 = max(cy - half, 0)
                y1 = min(y0 + block, h)
                x0 = max(cx - half, 0)
                x1 = min(x0 + block, w)
                keep[bi, ti, y0:y1, x0:x1] = 0.0
        out = out * keep[:, :, None]

    mask_noise_std = float(args.tool_mask_noise_std) * scale
    if mask_noise_std > 0.0:
        out = (out + torch.randn_like(out) * mask_noise_std).clamp(0.0, 1.0)
    return out


def render_tool_keypoint_heatmaps_8_from_vec16(tool_kpt_vec_16: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Render 16-frame tool-keypoint vectors back into paired 8-frame heatmaps."""
    if tool_kpt_vec_16.ndim != 3 or tool_kpt_vec_16.shape[1:] != (16, 9):
        raise ValueError(f"expected tool_kpt_vec_16 shape [B,16,9], got {tuple(tool_kpt_vec_16.shape)}")
    b = tool_kpt_vec_16.shape[0]
    device = tool_kpt_vec_16.device
    dtype = tool_kpt_vec_16.dtype
    xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).clamp(0.0, 1.0)
    vis = tool_kpt_vec_16[..., 6:9].clamp(0.0, 1.0)
    yy, xx = torch.meshgrid(
        torch.arange(24, device=device, dtype=dtype) + 0.5,
        torch.arange(24, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    gx = xx.view(1, 1, 1, 24, 24)
    gy = yy.view(1, 1, 1, 24, 24)
    px = (xy[..., 0] * 24.0).view(b, 16, 3, 1, 1)
    py = (xy[..., 1] * 24.0).view(b, 16, 3, 1, 1)
    sigma2 = max(float(sigma_px), 1e-3) ** 2
    hm16 = torch.exp(-0.5 * ((gx - px) ** 2 + (gy - py) ** 2) / sigma2)
    hm16 = hm16 * (vis > 0.0).to(dtype).view(b, 16, 3, 1, 1)
    return hm16.reshape(b, 8, 2, 3, 24, 24).amax(dim=2)


def apply_tool_keypoint_regularization(
    tool_kpt_heatmaps_8: torch.Tensor,
    tool_kpt_vec_16: torch.Tensor,
    args: argparse.Namespace,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training-only corruption for tool keypoints.

    Keypoints are cleaner than SAM tool masks but still annotation-like
    conditions.  Small coordinate jitter and sparse dropout make the ROI head
    depend on stable tool trajectory geometry instead of memorizing exact points.
    """
    scale = float(scale)
    jitter_std = float(args.tool_keypoint_jitter_std) * scale
    p_point = min(max(float(args.tool_keypoint_dropout) * scale, 0.0), 0.95)
    p_frame = min(max(float(args.tool_keypoint_frame_dropout) * scale, 0.0), 0.95)
    if jitter_std <= 0.0 and p_point <= 0.0 and p_frame <= 0.0:
        return tool_kpt_heatmaps_8, tool_kpt_vec_16

    if tool_kpt_vec_16.ndim != 3 or tool_kpt_vec_16.shape[1:] != (16, 9):
        raise ValueError(f"expected tool_kpt_vec_16 shape [B,16,9], got {tuple(tool_kpt_vec_16.shape)}")
    b = tool_kpt_vec_16.shape[0]
    xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).clone()
    vis = tool_kpt_vec_16[..., 6:9].clamp(0.0, 1.0).clone()
    if p_frame > 0.0:
        keep_frame = (torch.rand(b, 16, 1, device=vis.device) > p_frame).to(vis.dtype)
        vis = vis * keep_frame
    if p_point > 0.0:
        keep_point = (torch.rand(b, 16, 3, device=vis.device) > p_point).to(vis.dtype)
        vis = vis * keep_point
    if jitter_std > 0.0:
        valid = (vis > 0.0).to(xy.dtype).unsqueeze(-1)
        xy = (xy + torch.randn_like(xy) * jitter_std * valid).clamp(0.0, 1.0)
    xy = xy * vis.unsqueeze(-1)
    vec = torch.cat([xy.reshape(b, 16, 6), vis], dim=-1)
    heatmaps = render_tool_keypoint_heatmaps_8_from_vec16(vec, sigma_px=float(args.tool_keypoint_sigma_px))
    return heatmaps.to(dtype=tool_kpt_heatmaps_8.dtype), vec.to(dtype=tool_kpt_vec_16.dtype)


def corrupt_condition_ids(
    tool_id: torch.Tensor,
    act_id: torch.Tensor,
    combo_id: torch.Tensor,
    p_drop: float,
    mode: str = "random",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrupt categorical conditions during training to reduce combo-prior reliance."""
    if p_drop <= 0.0:
        return tool_id, act_id, combo_id
    b = tool_id.shape[0]
    drop = torch.rand(b, device=tool_id.device) < float(p_drop)
    if not bool(drop.any()):
        return tool_id, act_id, combo_id
    if str(mode) == "null":
        tool_rand = torch.full_like(tool_id, 8)
        act_rand = torch.full_like(act_id, 8)
        combo_rand = torch.full_like(combo_id, 16)
    else:
        tool_rand = torch.randint(0, 8, tool_id.shape, device=tool_id.device, dtype=tool_id.dtype)
        act_rand = torch.randint(0, 8, act_id.shape, device=act_id.device, dtype=act_id.dtype)
        combo_rand = torch.randint(0, 16, combo_id.shape, device=combo_id.device, dtype=combo_id.dtype)
    return (
        torch.where(drop, tool_rand, tool_id),
        torch.where(drop, act_rand, act_id),
        torch.where(drop, combo_rand, combo_id),
    )


def load_rgb_sidecar_16(root: Path | None, sid: str, size: int) -> np.ndarray:
    """Load materialized cropped video frames as RGB float [16,3,S,S].

    `materialize_ct_video16_from_meta.py` stores BGR uint8 clips because the old
    SurgKpt data loader used OpenCV ordering.  This helper flips to RGB and
    normalizes to [0,1] for the lightweight localization sidecar.
    """

    size = int(size)
    if root is None:
        return np.zeros((16, 3, size, size), dtype=np.float32)
    path = root / f"{sid}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing RGB sidecar clip for sid={sid}: {path}")
    arr = np.load(path)
    if arr.shape[0] != 16 or arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"RGB sidecar must be [16,H,W,3], got sid={sid} shape={arr.shape}")
    if arr.shape[1] != size or arr.shape[2] != size:
        frames = []
        for t in range(16):
            img = Image.fromarray(arr[t].astype(np.uint8))
            frames.append(np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.uint8))
        arr = np.stack(frames, axis=0)
    arr = arr[..., ::-1].astype(np.float32) / 255.0
    return np.transpose(arr, (0, 3, 1, 2)).astype(np.float32)


def render_tool_keypoints_16_to_8_24(
    xy16: np.ndarray | None,
    vis16: np.ndarray | None,
    sigma_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    if xy16 is None or vis16 is None:
        return (
            np.zeros((8, 3, 24, 24), dtype=np.float32),
            np.zeros((16, 9), dtype=np.float32),
        )
    xy = np.asarray(xy16, dtype=np.float32)
    vis = np.asarray(vis16, dtype=np.float32)
    if xy.shape != (16, 3, 2):
        raise ValueError(f"tool keypoints must have shape (16,3,2), got {xy.shape}")
    if vis.shape != (16, 3):
        raise ValueError(f"tool keypoint vis must have shape (16,3), got {vis.shape}")

    xy_visible = xy * vis[..., None]
    vec16 = np.concatenate([xy_visible.reshape(16, 6), vis], axis=1).astype(np.float32)

    yy, xx = np.mgrid[0:24, 0:24].astype(np.float32)
    xx = xx + 0.5
    yy = yy + 0.5
    sigma2 = max(float(sigma_px), 1e-3) ** 2
    hms = np.zeros((16, 3, 24, 24), dtype=np.float32)
    for t in range(16):
        for k in range(3):
            if vis[t, k] <= 0.0:
                continue
            cx = xy[t, k, 0] * 24.0
            cy = xy[t, k, 1] * 24.0
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            hms[t, k] = np.exp(-0.5 * d2 / sigma2).astype(np.float32)
    hms8 = pair_pool_16_to_8(hms, mode="max")
    return hms8.astype(np.float32), vec16


@dataclass(frozen=True)
class SampleIndex:
    sid: str
    split: str


class H161RoiDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        cache_root: Path,
        split: str,
        *,
        video_root: Path | None,
        tool_mask_root: Path | None,
        only_cached: bool,
        max_samples: int,
        bbox_pad_ratio: float,
        context_aux_pad_ratio: float,
        min_box_size: float,
        use_role_features: bool,
        use_rgb_sidecar: bool,
        rgb_sidecar_size: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        tool_keypoint_sigma_px: float,
        allow_missing_tool_mask: bool,
        teacher_cache_root: Path | None = None,
    ) -> None:
        self.data_root = data_root
        self.cache_root = cache_root
        self.split = split
        self.video_root = video_root
        self.tool_mask_root = tool_mask_root
        self.bbox_pad_ratio = float(bbox_pad_ratio)
        self.context_aux_pad_ratio = float(context_aux_pad_ratio)
        self.min_box_size = float(min_box_size)
        self.use_role_features = bool(use_role_features)
        self.use_rgb_sidecar = bool(use_rgb_sidecar)
        self.rgb_sidecar_size = int(rgb_sidecar_size)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.tool_keypoint_sigma_px = float(tool_keypoint_sigma_px)
        self.allow_missing_tool_mask = bool(allow_missing_tool_mask)
        self.teacher_cache_root = teacher_cache_root
        meta = load_json(data_root / "meta.json")
        self.meta_by_sid = {str(row["sid"]): row for row in meta}
        sids = read_split(data_root, split)
        if only_cached:
            sids = [sid for sid in sids if (cache_root / split / f"{sid}.npy").is_file()]
        if max_samples > 0:
            sids = sids[: int(max_samples)]
        self.samples = [SampleIndex(sid=sid, split=split) for sid in sids]
        if not self.samples:
            raise RuntimeError(f"no samples for split={split}; cache_root={cache_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        sid = sample.sid
        meta = self.meta_by_sid[sid]
        tokens = np.load(self.cache_root / self.split / f"{sid}.npy", mmap_mode="r").astype(np.float32)
        points = np.load(self.data_root / "gt_ct_xy" / f"{sid}.npy").astype(np.float32)
        quad_xy = order_quad_clockwise_np(points)
        boxes_xyxy = make_boxes_from_points(points, self.bbox_pad_ratio, self.min_box_size)
        context_boxes_xyxy = make_boxes_from_points(points, self.context_aux_pad_ratio, self.min_box_size)
        role_path = self.data_root / "action_range_16" / f"{sid}.npz"
        if role_path.is_file():
            role_npz = np.load(role_path)
            role_features = role_npz["features"].astype(np.float32)
            active = role_npz["inside_action_range"].astype(np.float32)
        else:
            role_features = np.zeros((16, 4), dtype=np.float32)
            active = np.zeros((16,), dtype=np.float32)
        if not self.use_role_features:
            role_features = np.zeros_like(role_features)
        rgb_16 = load_rgb_sidecar_16(self.video_root, sid, size=self.rgb_sidecar_size) if self.use_rgb_sidecar else np.zeros(
            (16, 3, self.rgb_sidecar_size, self.rgb_sidecar_size),
            dtype=np.float32,
        )
        tool_mask = None
        if self.use_tool_mask:
            tool_mask = load_tool_mask_16(self.tool_mask_root, sid)
            if tool_mask is None and not self.allow_missing_tool_mask:
                raise FileNotFoundError(f"missing tool mask for sid={sid} under {self.tool_mask_root}")
        tool_mask_8 = downsample_mask_16_to_8_24(tool_mask)
        tool_xy = None
        tool_vis = None
        if self.use_tool_keypoints:
            tool_xy_path = self.data_root / "gt_tool_xy" / f"{sid}.npy"
            tool_vis_path = self.data_root / "gt_tool_vis" / f"{sid}.npy"
            if not tool_xy_path.is_file() or not tool_vis_path.is_file():
                raise FileNotFoundError(
                    f"missing gt_tool_xy/gt_tool_vis for sid={sid}; run build_h161_tool_keypoints_from_annotations.py"
                )
            tool_xy = np.load(tool_xy_path).astype(np.float32)
            tool_vis = np.load(tool_vis_path).astype(np.float32)
        tool_kpt_heatmaps_8, tool_kpt_vec_16 = render_tool_keypoints_16_to_8_24(
            tool_xy,
            tool_vis,
            sigma_px=self.tool_keypoint_sigma_px,
        )
        teacher_quad_xy = np.zeros((16, 4, 2), dtype=np.float32)
        teacher_boxes_cxcywh = np.zeros((16, 4), dtype=np.float32)
        teacher_distill_mask = np.zeros((16,), dtype=np.float32)
        if self.teacher_cache_root is not None:
            teacher_path = self.teacher_cache_root / self.split / f"{sid}.npz"
            if teacher_path.is_file():
                teacher_npz = np.load(teacher_path)
                teacher_quad_xy = teacher_npz["quad_xy"].astype(np.float32)
                teacher_boxes_cxcywh = teacher_npz["boxes_cxcywh"].astype(np.float32)
                teacher_distill_mask = np.ones((16,), dtype=np.float32)
        return {
            "sid": sid,
            "tokens": torch.from_numpy(np.asarray(tokens)),
            "boxes_xyxy": torch.from_numpy(boxes_xyxy),
            "boxes_cxcywh": torch.from_numpy(np.asarray(xyxy_to_cxcywh(torch.from_numpy(boxes_xyxy)))),
            "context_boxes_xyxy": torch.from_numpy(context_boxes_xyxy),
            "context_boxes_cxcywh": torch.from_numpy(
                np.asarray(xyxy_to_cxcywh(torch.from_numpy(context_boxes_xyxy)))
            ),
            "quad_xy": torch.from_numpy(quad_xy),
            "role_features_16": torch.from_numpy(role_features),
            "active_16": torch.from_numpy(active),
            "rgb_16": torch.from_numpy(rgb_16),
            "tool_mask_8": torch.from_numpy(tool_mask_8),
            "tool_kpt_heatmaps_8": torch.from_numpy(tool_kpt_heatmaps_8),
            "tool_kpt_vec_16": torch.from_numpy(tool_kpt_vec_16),
            "teacher_quad_xy": torch.from_numpy(teacher_quad_xy),
            "teacher_boxes_cxcywh": torch.from_numpy(teacher_boxes_cxcywh),
            "teacher_distill_mask": torch.from_numpy(teacher_distill_mask),
            "tool_id": torch.tensor(int(meta.get("tool_id", 0)), dtype=torch.long),
            "act_id": torch.tensor(int(meta.get("act_id", 0)), dtype=torch.long),
            "combo_id": torch.tensor(int(meta.get("combo_id", 0)), dtype=torch.long),
        }


class H161RoiCondLite(nn.Module):
    def __init__(self, hidden_dim: int, use_tool_mask: bool, use_tool_keypoints: bool) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (1 if use_tool_mask else 0) + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        self.region_head = nn.Conv2d(c, 1, 1)
        self.size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.register_buffer("grid_xy", self._make_grid(24, 24), persistent=False)

    @staticmethod
    def _make_grid(h: int, w: int) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(0.5 / h, 1.0 - 0.5 / h, h),
            torch.linspace(0.5 / w, 1.0 - 0.5 / w, w),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=0)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # tokens: B,T=8,H=24,W=24,C=1024
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias = self.role_mlp(role8).view(b, t, -1, 1, 1)
        if self.use_tool_keypoints:
            kpt8 = tool_kpt_vec_16.reshape(b, 8, 2, 9).mean(dim=2)
            role_bias = role_bias + self.tool_kpt_mlp(kpt8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias) + beta
        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_mask:
            feats.append(tool_mask_8.to(device=x.device, dtype=x.dtype))
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = torch.cat(feats, dim=2).reshape(b * t, -1, h, w)
        y = self.spatial(y)
        center_logits_8 = self.center_head(y).view(b, t, h * w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled = (y.flatten(2) * prob.reshape(b * t, 1, h * w)).sum(dim=-1).view(b, t, -1)
        size8 = torch.sigmoid(self.size_head(pooled)).clamp(0.02, 0.95)
        box8_cxcywh = torch.cat([centers8, size8], dim=-1)
        box16_cxcywh = F.interpolate(box8_cxcywh.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        box16_cxcywh = box16_cxcywh.clamp(0.0, 1.0)
        return {
            "boxes_cxcywh": box16_cxcywh,
            "boxes_xyxy": cxcywh_to_xyxy(box16_cxcywh),
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "region_logits_8": self.region_head(y).view(b, t, h, w),
        }


class H161RoiEdgeDist(nn.Module):
    """Predict tight box sides as spatial edge distributions.

    This is a LocNet/GFL-style alternative to direct center-size regression:
    the head scores possible left/right/top/bottom sides over the token grid,
    then turns the distributions into a box.  It keeps the same V-JEPA cache
    and condition inputs as CondLite, so it is a drop-in Stage1 experiment.
    """

    def __init__(self, hidden_dim: int, use_tool_mask: bool, use_tool_keypoints: bool) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (1 if use_tool_mask else 0) + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        self.region_head = nn.Conv2d(c, 1, 1)
        self.x0_head = nn.Conv1d(c, 1, 1)
        self.x1_head = nn.Conv1d(c, 1, 1)
        self.y0_head = nn.Conv1d(c, 1, 1)
        self.y1_head = nn.Conv1d(c, 1, 1)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def _edge_expectation(self, logits: torch.Tensor, n: int) -> torch.Tensor:
        grid = torch.linspace(0.5 / n, 1.0 - 0.5 / n, n, device=logits.device, dtype=logits.dtype)
        return (F.softmax(logits, dim=-1) * grid).sum(dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias = self.role_mlp(role8).view(b, t, -1, 1, 1)
        if self.use_tool_keypoints:
            kpt8 = tool_kpt_vec_16.reshape(b, 8, 2, 9).mean(dim=2)
            role_bias = role_bias + self.tool_kpt_mlp(kpt8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias) + beta
        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_mask:
            feats.append(tool_mask_8.to(device=x.device, dtype=x.dtype))
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t, -1, h, w))

        x_seq = y.mean(dim=2)
        y_seq = y.mean(dim=3)
        x0_logits = self.x0_head(x_seq).squeeze(1)
        x1_logits = self.x1_head(x_seq).squeeze(1)
        y0_logits = self.y0_head(y_seq).squeeze(1)
        y1_logits = self.y1_head(y_seq).squeeze(1)
        x0_raw = self._edge_expectation(x0_logits, w).view(b, t)
        x1_raw = self._edge_expectation(x1_logits, w).view(b, t)
        y0_raw = self._edge_expectation(y0_logits, h).view(b, t)
        y1_raw = self._edge_expectation(y1_logits, h).view(b, t)
        x0 = torch.minimum(x0_raw, x1_raw)
        x1 = torch.maximum(x0_raw, x1_raw)
        y0 = torch.minimum(y0_raw, y1_raw)
        y1 = torch.maximum(y0_raw, y1_raw)
        box8_xyxy = torch.stack([x0, y0, x1, y1], dim=-1)
        box8_cxcywh = xyxy_to_cxcywh_torch(box8_xyxy)
        box8_cxcywh = torch.cat(
            [box8_cxcywh[..., :2], box8_cxcywh[..., 2:].clamp(0.02, 0.95)],
            dim=-1,
        )
        box16_cxcywh = F.interpolate(box8_cxcywh.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        box16_cxcywh = box16_cxcywh.clamp(0.0, 1.0)
        return {
            "boxes_cxcywh": box16_cxcywh,
            "boxes_xyxy": cxcywh_to_xyxy(box16_cxcywh),
            "center_logits_8": self.center_head(y).view(b, t, h, w),
            "region_logits_8": self.region_head(y).view(b, t, h, w),
            "edge_logits_8": {
                "x0": x0_logits.view(b, t, w),
                "x1": x1_logits.view(b, t, w),
                "y0": y0_logits.view(b, t, h),
                "y1": y1_logits.view(b, t, h),
            },
        }


class H161RoiBoxDist(nn.Module):
    """Predict continuous side distributions for each of the 16 target frames."""

    def __init__(self, hidden_dim: int, use_tool_mask: bool, use_tool_keypoints: bool, n_bins: int) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.n_bins = int(n_bins)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (1 if use_tool_mask else 0) + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        seq_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=4,
            dim_feedforward=4 * c,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(seq_layer, num_layers=2)
        self.seq_in = nn.Sequential(
            nn.LayerNorm(2 * c + 6),
            nn.Linear(2 * c + 6, c),
            nn.SiLU(),
        )
        self.side_head = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 4 * self.n_bins),
        )
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)
        self.register_buffer("dist_bins", torch.linspace(0.0, 1.0, self.n_bins), persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t, -1, 1, 1)
        if self.use_tool_keypoints:
            kpt8 = tool_kpt_vec_16.reshape(b, 8, 2, 9).mean(dim=2)
            role_bias8 = role_bias8 + self.tool_kpt_mlp(kpt8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_mask:
            feats.append(tool_mask_8.to(device=x.device, dtype=x.dtype))
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t, -1, h, w))

        center_logits_8 = self.center_head(y).view(b, t, h * w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled8 = (y.flatten(2) * prob.reshape(b * t, 1, h * w)).sum(dim=-1).view(b, t, -1)
        visual16 = F.interpolate(pooled8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        center16 = F.interpolate(centers8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        role16 = self.role_mlp(role_features_16)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        seq = self.seq_in(torch.cat([visual16, role16 + cond16, center16.detach(), role_features_16], dim=-1))
        seq = self.temporal(seq)

        side_logits = self.side_head(seq).view(b, 16, 4, self.n_bins)
        bins = self.dist_bins.to(device=side_logits.device, dtype=side_logits.dtype)
        raw = (F.softmax(side_logits, dim=-1) * bins).sum(dim=-1)
        x0 = torch.minimum(raw[..., 0], raw[..., 2])
        x1 = torch.maximum(raw[..., 0], raw[..., 2])
        y0 = torch.minimum(raw[..., 1], raw[..., 3])
        y1 = torch.maximum(raw[..., 1], raw[..., 3])
        box_xyxy = torch.stack([x0, y0, x1, y1], dim=-1).clamp(0.0, 1.0)
        box_xyxy = quad_to_xyxy(
            torch.stack(
                [
                    box_xyxy[..., [0, 1]],
                    box_xyxy[..., [2, 1]],
                    box_xyxy[..., [2, 3]],
                    box_xyxy[..., [0, 3]],
                ],
                dim=-2,
            ),
            min_size=0.02,
        )
        return {
            "boxes_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "boxes_xyxy": box_xyxy,
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "box_dist_logits_16": side_logits,
        }


class H162CornerHeatmapRegion(nn.Module):
    """Predict the four annotated operation-region points as heatmaps.

    The terminal metric is still tight bbox IoU, but the supervised object keeps
    the native four-point annotation structure.  This follows the detector
    pattern of using keypoint heatmaps to form boxes instead of regressing box
    sides as the only target.
    """

    def __init__(self, hidden_dim: int, use_tool_mask: bool, use_tool_keypoints: bool) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (1 if use_tool_mask else 0) + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.corner_head = nn.Conv2d(c, 4, 1)
        self.center_head = nn.Conv2d(c, 1, 1)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias = self.role_mlp(role8).view(b, t, -1, 1, 1)
        if self.use_tool_keypoints:
            kpt8 = tool_kpt_vec_16.reshape(b, 8, 2, 9).mean(dim=2)
            role_bias = role_bias + self.tool_kpt_mlp(kpt8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias) + beta

        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_mask:
            feats.append(tool_mask_8.to(device=x.device, dtype=x.dtype))
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t, -1, h, w))

        corner_logits = self.corner_head(y).view(b, t, 4, h, w)
        prob = F.softmax(corner_logits.flatten(-2), dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        quad8 = torch.matmul(prob, grid_flat)
        quad16 = F.interpolate(
            quad8.reshape(b, t, 8).transpose(1, 2),
            size=16,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2).reshape(b, 16, 4, 2)
        quad16 = quad16.clamp(0.0, 1.0)
        box_xyxy = quad_to_xyxy(quad16, min_size=0.02)
        return {
            "boxes_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "boxes_xyxy": box_xyxy,
            "center_logits_8": self.center_head(y).view(b, t, h, w),
            "corner_logits_8": corner_logits,
            "quad_xy": quad16,
        }


class H162SupportHeatmapRegion(nn.Module):
    """Predict an operation-support heatmap and derive a tight ROI.

    This treats the four-point annotation as a region/hotspot target rather
    than four independent landmarks.  The heatmap is the primary spatial
    representation; bbox losses remain the terminal metric bridge.
    """

    def __init__(self, hidden_dim: int, use_tool_mask: bool, use_tool_keypoints: bool) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (1 if use_tool_mask else 0) + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.support_head = nn.Conv2d(c, 1, 1)
        self.size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias = self.role_mlp(role8).view(b, t, -1, 1, 1)
        if self.use_tool_keypoints:
            kpt8 = tool_kpt_vec_16.reshape(b, 8, 2, 9).mean(dim=2)
            role_bias = role_bias + self.tool_kpt_mlp(kpt8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias) + beta

        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_mask:
            feats.append(tool_mask_8.to(device=x.device, dtype=x.dtype))
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t, -1, h, w))

        support_logits_8 = self.support_head(y).view(b, t, h, w)
        prob = F.softmax(support_logits_8.flatten(2), dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled8 = (y.flatten(2) * prob.reshape(b * t, 1, h * w)).sum(dim=-1).view(b, t, -1)
        size8 = torch.sigmoid(self.size_head(pooled8)).clamp(0.02, 0.95)
        box8_cxcywh = torch.cat([centers8, size8], dim=-1)
        box16_cxcywh = F.interpolate(
            box8_cxcywh.transpose(1, 2),
            size=16,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2).clamp(0.0, 1.0)
        return {
            "boxes_cxcywh": box16_cxcywh,
            "boxes_xyxy": cxcywh_to_xyxy(box16_cxcywh),
            "center_logits_8": support_logits_8,
            "support_logits_8": support_logits_8,
        }


class H162SupportMomentRegion(nn.Module):
    """Derive the ROI directly from support-heatmap moments.

    Unlike support_heatmap, there is no independent size MLP.  The predicted
    region extent must be encoded in the heatmap itself.
    """

    def __init__(self, hidden_dim: int, use_tool_mask: bool, use_tool_keypoints: bool, moment_size_scale: float) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.moment_size_scale = float(moment_size_scale)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (1 if use_tool_mask else 0) + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.support_head = nn.Conv2d(c, 1, 1)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias = self.role_mlp(role8).view(b, t, -1, 1, 1)
        if self.use_tool_keypoints:
            kpt8 = tool_kpt_vec_16.reshape(b, 8, 2, 9).mean(dim=2)
            role_bias = role_bias + self.tool_kpt_mlp(kpt8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias) + beta

        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_mask:
            feats.append(tool_mask_8.to(device=x.device, dtype=x.dtype))
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t, -1, h, w))

        support_logits_8 = self.support_head(y).view(b, t, h, w)
        heat = torch.sigmoid(support_logits_8).clamp_min(1e-5)
        mass = heat.sum(dim=(-1, -2)).clamp_min(1e-5)
        gx = self.grid_xy[0].to(device=heat.device, dtype=heat.dtype).view(1, 1, h, w)
        gy = self.grid_xy[1].to(device=heat.device, dtype=heat.dtype).view(1, 1, h, w)
        cx = (heat * gx).sum(dim=(-1, -2)) / mass
        cy = (heat * gy).sum(dim=(-1, -2)) / mass
        vx = (heat * (gx - cx[..., None, None]).pow(2)).sum(dim=(-1, -2)) / mass
        vy = (heat * (gy - cy[..., None, None]).pow(2)).sum(dim=(-1, -2)) / mass
        size = torch.stack([vx.sqrt(), vy.sqrt()], dim=-1) * float(self.moment_size_scale)
        box8_cxcywh = torch.cat([torch.stack([cx, cy], dim=-1), size.clamp(0.02, 0.95)], dim=-1)
        box16_cxcywh = F.interpolate(
            box8_cxcywh.transpose(1, 2),
            size=16,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2).clamp(0.0, 1.0)
        return {
            "boxes_cxcywh": box16_cxcywh,
            "boxes_xyxy": cxcywh_to_xyxy(box16_cxcywh),
            "center_logits_8": support_logits_8,
            "support_logits_8": support_logits_8,
        }


class H162ToolMaskAnchorResidual(nn.Module):
    """Use tool-mask geometry as a proposal, then refine to a tight ROI.

    Raw mask concatenation can behave like an unstructured shortcut.  This
    mode first compresses the 8-frame tool masks into explicit geometry,
    predicts a 16-frame anchor region, and then lets V-JEPA features refine
    that anchor with a temporal residual head.
    """

    def __init__(self, hidden_dim: int, residual_scale: float) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_geom_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.anchor_head = nn.Sequential(
            nn.LayerNorm(3 * c),
            nn.Linear(3 * c, c),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(c, 4),
        )
        in_ch = c + 2 + 1 + 2
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        seq_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=4,
            dim_feedforward=4 * c,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(seq_layer, num_layers=2)
        self.seq_in = nn.Sequential(
            nn.LayerNorm(3 * c + 4),
            nn.Linear(3 * c + 4, c),
            nn.SiLU(),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 4),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def _tool_mask_geom8(self, tool_mask_8: torch.Tensor) -> torch.Tensor:
        mask = tool_mask_8.squeeze(2)
        b, t, h, w = mask.shape
        grid = self.grid_xy.to(device=mask.device, dtype=mask.dtype)
        gx = grid[0].view(1, 1, h, w)
        gy = grid[1].view(1, 1, h, w)
        mass_raw = mask.sum(dim=(-1, -2))
        mass = mass_raw.clamp_min(1e-6)
        empty = (mass_raw <= 0.0).to(mask.dtype)
        cx = (mask * gx).sum(dim=(-1, -2)) / mass
        cy = (mask * gy).sum(dim=(-1, -2)) / mass
        area = mass_raw / float(h * w)
        sx = torch.sqrt(((gx - cx[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        sy = torch.sqrt(((gy - cy[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        fg = mask > 0.05
        one = torch.ones_like(mask)
        zero = torch.zeros_like(mask)
        x0 = torch.where(fg, gx.expand(b, t, h, w), one).amin(dim=(-1, -2))
        y0 = torch.where(fg, gy.expand(b, t, h, w), one).amin(dim=(-1, -2))
        x1 = torch.where(fg, gx.expand(b, t, h, w), zero).amax(dim=(-1, -2))
        y1 = torch.where(fg, gy.expand(b, t, h, w), zero).amax(dim=(-1, -2))
        cx = cx * (1.0 - empty) + 0.5 * empty
        cy = cy * (1.0 - empty) + 0.5 * empty
        x0 = x0 * (1.0 - empty)
        y0 = y0 * (1.0 - empty)
        x1 = x1 * (1.0 - empty) + empty
        y1 = y1 * (1.0 - empty) + empty
        return torch.stack([cx, cy, area, sx, sy, x0, y0, x1, y1], dim=-1)

    def _anchor_maps(self, anchor8_cxcywh: torch.Tensor, h: int, w: int, dtype: torch.dtype) -> torch.Tensor:
        grid = self.grid_xy.to(device=anchor8_cxcywh.device, dtype=dtype)
        gx = grid[0].view(1, 1, h, w)
        gy = grid[1].view(1, 1, h, w)
        cx, cy, bw, bh = anchor8_cxcywh.unbind(dim=-1)
        x0 = (cx - 0.5 * bw).clamp(0.0, 1.0).view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1)
        y0 = (cy - 0.5 * bh).clamp(0.0, 1.0).view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1)
        x1 = (cx + 0.5 * bw).clamp(0.0, 1.0).view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1)
        y1 = (cy + 0.5 * bh).clamp(0.0, 1.0).view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1)
        tau = 1.0 / 24.0
        box_map = torch.sigmoid((gx - x0) / tau) * torch.sigmoid((x1 - gx) / tau)
        box_map = box_map * torch.sigmoid((gy - y0) / tau) * torch.sigmoid((y1 - gy) / tau)
        sx = (bw.view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1) * 0.35).clamp_min(0.03)
        sy = (bh.view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1) * 0.35).clamp_min(0.03)
        center_map = torch.exp(
            -0.5
            * (
                ((gx - cx.view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1)) / sx) ** 2
                + ((gy - cy.view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1)) / sy) ** 2
            )
        )
        return torch.stack([box_map, center_map], dim=2)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del tool_kpt_heatmaps_8, tool_kpt_vec_16
        b, t, h, w, _ = tokens.shape
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        role16 = self.role_mlp(role_features_16)

        geom8 = self._tool_mask_geom8(tool_mask_8.to(device=tokens.device, dtype=tokens.dtype))
        geom16 = F.interpolate(geom8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        geom_feat16 = self.tool_geom_mlp(geom16)

        anchor_in = torch.cat([role16, geom_feat16, cond16], dim=-1)
        raw_anchor = self.anchor_head(anchor_in)
        anchor_center = torch.sigmoid(raw_anchor[..., :2])
        anchor_size = (0.02 + 0.93 * torch.sigmoid(raw_anchor[..., 2:])).clamp(0.02, 0.95)
        anchor16 = torch.cat([anchor_center, anchor_size], dim=-1)
        anchor8 = anchor16.reshape(b, 8, 2, 4).mean(dim=2)

        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        anchor_maps = self._anchor_maps(anchor8, h, w, x.dtype)
        feats = torch.cat(
            [
                x,
                grid,
                tool_mask_8.to(device=x.device, dtype=x.dtype),
                anchor_maps.to(device=x.device, dtype=x.dtype),
            ],
            dim=2,
        )
        y = self.spatial(feats.reshape(b * t, -1, h, w))
        center_logits_8 = self.center_head(y).view(b, t, h * w)
        local_w = (anchor_maps[:, :, 0] + anchor_maps[:, :, 1]).reshape(b * t, 1, h * w).clamp_min(1e-6)
        pooled_local = (y.flatten(2) * local_w).sum(dim=-1) / local_w.sum(dim=-1)
        prob = F.softmax(center_logits_8, dim=-1)
        pooled_center = (y.flatten(2) * prob.reshape(b * t, 1, h * w)).sum(dim=-1)
        pooled8 = 0.5 * (pooled_local + pooled_center)
        visual16 = F.interpolate(pooled8.view(b, t, -1).transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)

        seq = self.seq_in(torch.cat([visual16, role16 + cond16, geom_feat16, anchor16.detach()], dim=-1))
        seq = self.temporal(seq)
        delta = self.delta_head(seq)
        dc = torch.tanh(delta[..., :2]) * self.residual_scale * anchor16[..., 2:].detach()
        dwh = torch.tanh(delta[..., 2:]) * 0.6
        center = (anchor16[..., :2] + dc).clamp(0.0, 1.0)
        size = (anchor16[..., 2:] * torch.exp(dwh)).clamp(0.02, 0.95)
        pred = torch.cat([center, size], dim=-1)
        return {
            "boxes_cxcywh": pred,
            "boxes_xyxy": cxcywh_to_xyxy(pred),
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "anchor_cxcywh": anchor16,
            "anchor_xyxy": cxcywh_to_xyxy(anchor16),
        }


class H161RoiAnchorResidual(nn.Module):
    def __init__(self, hidden_dim: int, residual_scale: float) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.anchor_head = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 4),
        )
        in_ch = c + 2 + 3 + 2
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 4),
        )
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def make_anchor_maps(self, anchor8_cxcywh: torch.Tensor, h: int, w: int, dtype: torch.dtype) -> torch.Tensor:
        grid = self.grid_xy.to(device=anchor8_cxcywh.device, dtype=dtype)
        gx = grid[0].view(1, 1, h, w)
        gy = grid[1].view(1, 1, h, w)
        cx, cy, bw, bh = anchor8_cxcywh.unbind(dim=-1)
        x0 = (cx - 0.5 * bw).clamp(0.0, 1.0).view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1)
        y0 = (cy - 0.5 * bh).clamp(0.0, 1.0).view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1)
        x1 = (cx + 0.5 * bw).clamp(0.0, 1.0).view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1)
        y1 = (cy + 0.5 * bh).clamp(0.0, 1.0).view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1)
        tau = 1.0 / 24.0
        box_map = torch.sigmoid((gx - x0) / tau) * torch.sigmoid((x1 - gx) / tau)
        box_map = box_map * torch.sigmoid((gy - y0) / tau) * torch.sigmoid((y1 - gy) / tau)
        sx = (bw.view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1) * 0.35).clamp_min(0.03)
        sy = (bh.view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1) * 0.35).clamp_min(0.03)
        center_map = torch.exp(
            -0.5
            * (
                ((gx - cx.view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1)) / sx) ** 2
                + ((gy - cy.view(anchor8_cxcywh.shape[0], anchor8_cxcywh.shape[1], 1, 1)) / sy) ** 2
            )
        )
        return torch.stack([box_map, center_map], dim=2)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del tool_mask_8
        b, t, h, w, _ = tokens.shape
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        anchor_feat16 = self.kpt_mlp(tool_kpt_vec_16) + self.role_mlp(role_features_16) + cond16
        raw_anchor = self.anchor_head(anchor_feat16)
        anchor_center = torch.sigmoid(raw_anchor[..., :2])
        anchor_size = (0.02 + 0.93 * torch.sigmoid(raw_anchor[..., 2:])).clamp(0.02, 0.95)
        anchor16 = torch.cat([anchor_center, anchor_size], dim=-1)
        anchor8 = anchor16.reshape(b, 8, 2, 4).mean(dim=2)

        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        anchor_maps = self.make_anchor_maps(anchor8, h, w, x.dtype)
        feats = torch.cat(
            [
                x,
                grid,
                tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype),
                anchor_maps.to(device=x.device, dtype=x.dtype),
            ],
            dim=2,
        )
        y = self.spatial(feats.reshape(b * t, -1, h, w))
        center_logits_8 = self.center_head(y).view(b, t, h * w)
        local_w = anchor_maps[:, :, 1].reshape(b * t, 1, h * w).clamp_min(1e-6)
        pooled_local = (y.flatten(2) * local_w).sum(dim=-1) / local_w.sum(dim=-1)
        pooled_global = y.flatten(2).mean(dim=-1)
        pooled = (pooled_local + pooled_global).view(b, t, -1)
        delta8 = self.delta_head(pooled)
        delta16 = F.interpolate(delta8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        dc = torch.tanh(delta16[..., :2]) * self.residual_scale * anchor16[..., 2:].detach()
        dwh = torch.tanh(delta16[..., 2:]) * 0.7
        pred_center = (anchor16[..., :2] + dc).clamp(0.0, 1.0)
        pred_size = (anchor16[..., 2:] * torch.exp(dwh)).clamp(0.02, 0.95)
        pred = torch.cat([pred_center, pred_size], dim=-1)
        return {
            "boxes_cxcywh": pred,
            "boxes_xyxy": cxcywh_to_xyxy(pred),
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "anchor_cxcywh": anchor16,
            "anchor_xyxy": cxcywh_to_xyxy(anchor16),
        }


class H161RoiTemporalResidual(nn.Module):
    def __init__(self, hidden_dim: int, use_tool_keypoints: bool, residual_scale: float) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.residual_scale = float(residual_scale)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        self.corner_heatmap_head = nn.Conv2d(c, 4, 1)
        self.support_head = nn.Conv2d(c, 1, 1)
        self.size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.residual_head = nn.Sequential(
            nn.LayerNorm(3 * c + 4),
            nn.Linear(3 * c + 4, c),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 4),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del tool_mask_8
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta
        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t, -1, h, w))
        center_logits_8 = self.center_head(y).view(b, t, h * w)
        corner_logits_8 = self.corner_heatmap_head(y).view(b, t, 4, h, w)
        support_logits_8 = self.support_head(y).view(b, t, h, w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled8 = (y.flatten(2) * prob.reshape(b * t, 1, h * w)).sum(dim=-1).view(b, t, -1)
        size8 = torch.sigmoid(self.size_head(pooled8)).clamp(0.02, 0.95)
        coarse8 = torch.cat([centers8, size8], dim=-1)
        coarse16 = F.interpolate(coarse8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        visual16 = F.interpolate(pooled8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        role16 = self.role_mlp(role_features_16)
        if self.use_tool_keypoints:
            kpt16 = self.tool_kpt_mlp(tool_kpt_vec_16)
        else:
            kpt16 = torch.zeros_like(role16)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        residual_in = torch.cat([visual16, role16 + cond16, kpt16, coarse16.detach()], dim=-1)
        delta = self.residual_head(residual_in)
        dc = torch.tanh(delta[..., :2]) * self.residual_scale * coarse16[..., 2:].detach()
        dwh = torch.tanh(delta[..., 2:]) * 0.6
        pred_center = (coarse16[..., :2] + dc).clamp(0.0, 1.0)
        pred_size = (coarse16[..., 2:] * torch.exp(dwh)).clamp(0.02, 0.95)
        pred = torch.cat([pred_center, pred_size], dim=-1)
        return {
            "boxes_cxcywh": pred,
            "boxes_xyxy": cxcywh_to_xyxy(pred),
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "coarse_cxcywh": coarse16,
            "coarse_xyxy": cxcywh_to_xyxy(coarse16),
        }


class H170ToolMaskGeomTemporalResidual(nn.Module):
    """Matched temporal residual head using tool-mask geometry instead of keypoints."""

    def __init__(self, hidden_dim: int, residual_scale: float) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + 1
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        self.corner_heatmap_head = nn.Conv2d(c, 4, 1)
        self.support_head = nn.Conv2d(c, 1, 1)
        self.size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.residual_head = nn.Sequential(
            nn.LayerNorm(3 * c + 4),
            nn.Linear(3 * c + 4, c),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 4),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def _tool_mask_geom8(self, tool_mask_8: torch.Tensor) -> torch.Tensor:
        mask = tool_mask_8.squeeze(2)
        b, t, h, w = mask.shape
        grid = self.grid_xy.to(device=mask.device, dtype=mask.dtype)
        gx = grid[0].view(1, 1, h, w)
        gy = grid[1].view(1, 1, h, w)
        mass_raw = mask.sum(dim=(-1, -2))
        empty = (mass_raw <= 0.0).to(mask.dtype)
        mass = mass_raw.clamp_min(1e-6)
        cx = (mask * gx).sum(dim=(-1, -2)) / mass
        cy = (mask * gy).sum(dim=(-1, -2)) / mass
        area = mass_raw / float(h * w)
        sx = torch.sqrt(((gx - cx[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        sy = torch.sqrt(((gy - cy[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        cx = cx * (1.0 - empty) + 0.5 * empty
        cy = cy * (1.0 - empty) + 0.5 * empty
        return torch.stack([cx, cy, area, sx, sy], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del tool_kpt_heatmaps_8, tool_kpt_vec_16
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta
        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        tool_mask = tool_mask_8.to(device=x.device, dtype=x.dtype)
        y = self.spatial(torch.cat([x, grid, tool_mask], dim=2).reshape(b * t, -1, h, w))
        center_logits_8 = self.center_head(y).view(b, t, h * w)
        corner_logits_8 = self.corner_heatmap_head(y).view(b, t, 4, h, w)
        support_logits_8 = self.support_head(y).view(b, t, h, w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled8 = (y.flatten(2) * prob.reshape(b * t, 1, h * w)).sum(dim=-1).view(b, t, -1)
        size8 = torch.sigmoid(self.size_head(pooled8)).clamp(0.02, 0.95)
        coarse8 = torch.cat([centers8, size8], dim=-1)
        coarse16 = F.interpolate(coarse8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        visual16 = F.interpolate(pooled8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        role16 = self.role_mlp(role_features_16)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        mask_geom8 = self._tool_mask_geom8(tool_mask)
        mask_geom16 = F.interpolate(mask_geom8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        mask16 = self.tool_mask_geom_mlp(mask_geom16)
        residual_in = torch.cat([visual16, role16 + cond16, mask16, coarse16.detach()], dim=-1)
        delta = self.residual_head(residual_in)
        dc = torch.tanh(delta[..., :2]) * self.residual_scale * coarse16[..., 2:].detach()
        dwh = torch.tanh(delta[..., 2:]) * 0.6
        pred_center = (coarse16[..., :2] + dc).clamp(0.0, 1.0)
        pred_size = (coarse16[..., 2:] * torch.exp(dwh)).clamp(0.02, 0.95)
        pred = torch.cat([pred_center, pred_size], dim=-1)
        return {
            "boxes_cxcywh": pred,
            "boxes_xyxy": cxcywh_to_xyxy(pred),
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "coarse_cxcywh": coarse16,
            "coarse_xyxy": cxcywh_to_xyxy(coarse16),
        }


class H166ComboExpertTemporalResidual(nn.Module):
    """Temporal ROI head with combo-specific localization experts.

    S2R/S2S/S2T showed that stronger temporal or local refiners can fit the
    train split but do not improve locked VAL beyond the coarse head.  This
    head keeps the same V-JEPA-compatible input contract and the same tight-box
    target, but stops forcing all tool/action combos through one final
    localization head.  The shared trunk learns visual evidence; combo_id
    selects the center, size, and residual expert.
    """

    def __init__(self, hidden_dim: int, use_tool_keypoints: bool, residual_scale: float) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.num_combo_experts = 16
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.residual_scale = float(residual_scale)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(self.num_combo_experts, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, self.num_combo_experts, 1)
        self.size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, self.num_combo_experts * 2))
        self.residual_head = nn.Sequential(
            nn.LayerNorm(3 * c + 4),
            nn.Linear(3 * c + 4, c),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, self.num_combo_experts * 4),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def _select_combo_center_logits(
        self, center_logits_all: torch.Tensor, combo_id: torch.Tensor, b: int, t: int, h: int, w: int
    ) -> torch.Tensor:
        logits = center_logits_all.view(b, t, self.num_combo_experts, h * w)
        combo = combo_id.clamp(0, self.num_combo_experts - 1).view(b, 1, 1, 1)
        combo = combo.expand(-1, t, 1, h * w)
        return logits.gather(2, combo).squeeze(2)

    def _select_combo_size(self, size_all: torch.Tensor, combo_id: torch.Tensor) -> torch.Tensor:
        b, t, _ = size_all.shape
        size_all = size_all.view(b, t, self.num_combo_experts, 2)
        combo = combo_id.clamp(0, self.num_combo_experts - 1).view(b, 1, 1, 1)
        combo = combo.expand(-1, t, 1, 2)
        return size_all.gather(2, combo).squeeze(2)

    def _select_combo_delta(self, delta_all: torch.Tensor, combo_id: torch.Tensor) -> torch.Tensor:
        b, t, _ = delta_all.shape
        delta_all = delta_all.view(b, t, self.num_combo_experts, 4)
        combo = combo_id.clamp(0, self.num_combo_experts - 1).view(b, 1, 1, 1)
        combo = combo.expand(-1, t, 1, 4)
        return delta_all.gather(2, combo).squeeze(2)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del tool_mask_8
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id.clamp(0, self.num_combo_experts - 1))
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t, -1, h, w))

        center_logits_all = self.center_head(y)
        center_logits_8 = self._select_combo_center_logits(center_logits_all, combo_id, b, t, h, w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled8 = (y.flatten(2) * prob.reshape(b * t, 1, h * w)).sum(dim=-1).view(b, t, -1)
        size8 = torch.sigmoid(self._select_combo_size(self.size_head(pooled8), combo_id)).clamp(0.02, 0.95)
        coarse8 = torch.cat([centers8, size8], dim=-1)

        coarse16 = F.interpolate(coarse8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        visual16 = F.interpolate(pooled8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        role16 = self.role_mlp(role_features_16)
        if self.use_tool_keypoints:
            kpt16 = self.tool_kpt_mlp(tool_kpt_vec_16)
        else:
            kpt16 = torch.zeros_like(role16)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        residual_in = torch.cat([visual16, role16 + cond16, kpt16, coarse16.detach()], dim=-1)
        delta = self._select_combo_delta(self.residual_head(residual_in), combo_id)
        dc = torch.tanh(delta[..., :2]) * self.residual_scale * coarse16[..., 2:].detach()
        dwh = torch.tanh(delta[..., 2:]) * 0.50
        pred_center = (coarse16[..., :2] + dc).clamp(0.0, 1.0)
        pred_size = (coarse16[..., 2:] * torch.exp(dwh)).clamp(0.02, 0.95)
        pred = torch.cat([pred_center, pred_size], dim=-1)
        return {
            "boxes_cxcywh": pred,
            "boxes_xyxy": cxcywh_to_xyxy(pred),
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "coarse_cxcywh": coarse16,
            "coarse_xyxy": cxcywh_to_xyxy(coarse16),
        }


def make_rgb_sidecar_encoder(hidden_dim: int) -> nn.Sequential:
    c = int(hidden_dim)
    rgb_c1 = max(32, c // 4)
    rgb_c2 = max(64, c // 2)
    return nn.Sequential(
        nn.Conv2d(3, rgb_c1, 3, stride=2, padding=1),
        nn.GroupNorm(4, rgb_c1),
        nn.SiLU(),
        nn.Conv2d(rgb_c1, rgb_c2, 3, stride=2, padding=1),
        nn.GroupNorm(8, rgb_c2),
        nn.SiLU(),
        nn.Conv2d(rgb_c2, c, 3, padding=1),
        nn.GroupNorm(8, c),
        nn.SiLU(),
    )


def encode_rgb_sidecar_8(
    rgb_encoder: nn.Module,
    rgb_16: torch.Tensor | None,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if rgb_16 is None:
        raise ValueError("RGB sidecar fusion is enabled but rgb_16 was not provided")
    b = rgb_16.shape[0]
    rgb = rgb_16.to(dtype=dtype).reshape(b, 8, 2, 3, rgb_16.shape[-2], rgb_16.shape[-1]).mean(dim=2)
    y = rgb_encoder(rgb.reshape(b * 8, 3, rgb.shape[-2], rgb.shape[-1]))
    if y.shape[-2:] != (24, 24):
        y = F.interpolate(y, size=(24, 24), mode="bilinear", align_corners=False)
    return y.view(b, 8, -1, 24, 24)


class H167RgbSidecarTemporalResidual(nn.Module):
    """Temporal ROI head that fuses V-JEPA tokens with a tiny RGB localization path.

    V-JEPA cache remains the main semantic representation.  The RGB sidecar is a
    deployment-available low-level signal, encoded from 96x96 cropped frames and
    fused at the same 24x24 spatial grid as the frozen tokens.
    """

    def __init__(self, hidden_dim: int, use_tool_keypoints: bool, residual_scale: float) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.residual_scale = float(residual_scale)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        self.rgb_encoder = make_rgb_sidecar_encoder(c)
        in_ch = (2 * c) + 2 + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        self.size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.residual_head = nn.Sequential(
            nn.LayerNorm(3 * c + 4),
            nn.Linear(3 * c + 4, c),
            nn.SiLU(),
            nn.Dropout(0.12),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 4),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def _encode_rgb8(self, rgb_16: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return encode_rgb_sidecar_8(self.rgb_encoder, rgb_16, dtype=dtype)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
        rgb_16: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del tool_mask_8
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        rgb8 = self._encode_rgb8(rgb_16, dtype=x.dtype)
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, rgb8, grid]
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t, -1, h, w))
        center_logits_8 = self.center_head(y).view(b, t, h * w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled8 = (y.flatten(2) * prob.reshape(b * t, 1, h * w)).sum(dim=-1).view(b, t, -1)
        size8 = torch.sigmoid(self.size_head(pooled8)).clamp(0.02, 0.95)
        coarse8 = torch.cat([centers8, size8], dim=-1)
        coarse16 = F.interpolate(coarse8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        visual16 = F.interpolate(pooled8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        role16 = self.role_mlp(role_features_16)
        if self.use_tool_keypoints:
            kpt16 = self.tool_kpt_mlp(tool_kpt_vec_16)
        else:
            kpt16 = torch.zeros_like(role16)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        residual_in = torch.cat([visual16, role16 + cond16, kpt16, coarse16.detach()], dim=-1)
        delta = self.residual_head(residual_in)
        dc = torch.tanh(delta[..., :2]) * self.residual_scale * coarse16[..., 2:].detach()
        dwh = torch.tanh(delta[..., 2:]) * 0.50
        pred_center = (coarse16[..., :2] + dc).clamp(0.0, 1.0)
        pred_size = (coarse16[..., 2:] * torch.exp(dwh)).clamp(0.02, 0.95)
        pred = torch.cat([pred_center, pred_size], dim=-1)
        return {
            "boxes_cxcywh": pred,
            "boxes_xyxy": cxcywh_to_xyxy(pred),
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "coarse_cxcywh": coarse16,
            "coarse_xyxy": cxcywh_to_xyxy(coarse16),
        }


class H164TemporalTransformerResidual(nn.Module):
    """Temporal residual ROI head with explicit 16-frame self-attention.

    H161RoiTemporalResidual only interpolates 8 cache steps to 16 frames and then
    applies an independent residual MLP per frame.  This head keeps the same
    coarse spatial path, but lets the 16 frame-level features communicate before
    predicting the final tight box residual.
    """

    def __init__(self, hidden_dim: int, use_tool_keypoints: bool, residual_scale: float) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.residual_scale = float(residual_scale)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        self.size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.seq_in = nn.Sequential(
            nn.LayerNorm(3 * c + 4),
            nn.Linear(3 * c + 4, c),
            nn.SiLU(),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=4,
            dim_feedforward=4 * c,
            dropout=0.12,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(2 * c + 4),
            nn.Linear(2 * c + 4, c),
            nn.SiLU(),
            nn.Dropout(0.12),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 4),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.pos_emb = nn.Parameter(torch.zeros(1, 16, c))
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del tool_mask_8
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t, -1, h, w))
        center_logits_8 = self.center_head(y).view(b, t, h * w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled8 = (y.flatten(2) * prob.reshape(b * t, 1, h * w)).sum(dim=-1).view(b, t, -1)
        size8 = torch.sigmoid(self.size_head(pooled8)).clamp(0.02, 0.95)
        coarse8 = torch.cat([centers8, size8], dim=-1)

        coarse16 = F.interpolate(coarse8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        visual16 = F.interpolate(pooled8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        role16 = self.role_mlp(role_features_16)
        if self.use_tool_keypoints:
            kpt16 = self.tool_kpt_mlp(tool_kpt_vec_16)
        else:
            kpt16 = torch.zeros_like(role16)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        seq = self.seq_in(torch.cat([visual16, role16 + cond16, kpt16, coarse16.detach()], dim=-1))
        seq = self.temporal(seq + self.pos_emb.to(device=seq.device, dtype=seq.dtype))

        delta = self.residual_head(torch.cat([seq, visual16, coarse16.detach()], dim=-1))
        dc = torch.tanh(delta[..., :2]) * self.residual_scale * coarse16[..., 2:].detach()
        dwh = torch.tanh(delta[..., 2:]) * 0.50
        pred_center = (coarse16[..., :2] + dc).clamp(0.0, 1.0)
        pred_size = (coarse16[..., 2:] * torch.exp(dwh)).clamp(0.02, 0.95)
        pred = torch.cat([pred_center, pred_size], dim=-1)
        return {
            "boxes_cxcywh": pred,
            "boxes_xyxy": cxcywh_to_xyxy(pred),
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "coarse_cxcywh": coarse16,
            "coarse_xyxy": cxcywh_to_xyxy(coarse16),
        }


class H165LocalRefineResidual(nn.Module):
    """Predicted-proposal local feature refinement for tight ROI boxes.

    This keeps the S2H coarse path but changes the residual contract: instead
    of asking a global pooled vector to guess tight sides, it samples decoder
    features around the model's own predicted coarse box and regresses bounded
    coarse-to-tight corrections.  No GT ROI enters the forward pass.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_tool_keypoints: bool,
        residual_scale: float,
        use_gates: bool = False,
    ) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.use_gates = bool(use_gates)
        self.residual_scale = float(residual_scale)
        self.context_scale = 1.45
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        self.size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))

        refine_dim = 6 * c + 4
        out_dim = 5 if self.use_gates else 4
        self.refine1 = nn.Sequential(
            nn.LayerNorm(refine_dim),
            nn.Linear(refine_dim, 2 * c),
            nn.SiLU(),
            nn.Dropout(0.12),
            nn.Linear(2 * c, c),
            nn.SiLU(),
            nn.Linear(c, out_dim),
        )
        self.refine2 = nn.Sequential(
            nn.LayerNorm(refine_dim),
            nn.Linear(refine_dim, 2 * c),
            nn.SiLU(),
            nn.Dropout(0.12),
            nn.Linear(2 * c, c),
            nn.SiLU(),
            nn.Linear(c, out_dim),
        )
        nn.init.zeros_(self.refine1[-1].weight)
        nn.init.zeros_(self.refine1[-1].bias)
        nn.init.zeros_(self.refine2[-1].weight)
        nn.init.zeros_(self.refine2[-1].bias)
        if self.use_gates:
            nn.init.constant_(self.refine1[-1].bias[4], -2.5)
            nn.init.constant_(self.refine2[-1].bias[4], -2.5)

        offsets = torch.tensor(
            [
                [-0.50, -0.50],
                [0.00, -0.50],
                [0.50, -0.50],
                [-0.50, 0.00],
                [0.00, 0.00],
                [0.50, 0.00],
                [-0.50, 0.50],
                [0.00, 0.50],
                [0.50, 0.50],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("sample_offsets", offsets, persistent=False)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    @staticmethod
    def _interp_8_to_16(feat8: torch.Tensor) -> torch.Tensor:
        # feat8: [B,8,C,H,W] -> [B,16,C,H,W]
        b, _, c, h, w = feat8.shape
        feat = feat8.permute(0, 2, 1, 3, 4).contiguous()
        feat = F.interpolate(feat, size=(16, h, w), mode="trilinear", align_corners=True)
        return feat.permute(0, 2, 1, 3, 4).contiguous()

    def _sample_box_features(self, feat16: torch.Tensor, boxes16: torch.Tensor) -> torch.Tensor:
        # feat16: [B,16,C,H,W], boxes16: [B,16,4] cxcywh normalized.
        b, t, c, h, w = feat16.shape
        offsets = self.sample_offsets.to(device=feat16.device, dtype=feat16.dtype)
        cx, cy, bw, bh = boxes16.unbind(dim=-1)
        sx = (bw * self.context_scale).clamp(1.0 / float(w), 1.0)
        sy = (bh * self.context_scale).clamp(1.0 / float(h), 1.0)
        px = cx[..., None] + offsets[:, 0].view(1, 1, -1) * sx[..., None]
        py = cy[..., None] + offsets[:, 1].view(1, 1, -1) * sy[..., None]
        grid = torch.stack([px * 2.0 - 1.0, py * 2.0 - 1.0], dim=-1).clamp(-1.15, 1.15)
        sampled = F.grid_sample(
            feat16.reshape(b * t, c, h, w),
            grid.reshape(b * t, -1, 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, t, -1, c)
        local_mean = sampled.mean(dim=2)
        local_max = sampled.max(dim=2).values
        local_center = sampled[:, :, 4]
        return torch.cat([local_mean, local_max, local_center], dim=-1)

    def _apply_delta(self, boxes16: torch.Tensor, delta: torch.Tensor, wh_scale: float) -> torch.Tensor:
        dc = torch.tanh(delta[..., :2]) * self.residual_scale * boxes16[..., 2:].detach()
        dwh = torch.tanh(delta[..., 2:4]) * wh_scale
        center = (boxes16[..., :2] + dc).clamp(0.0, 1.0)
        size = (boxes16[..., 2:] * torch.exp(dwh)).clamp(0.02, 0.95)
        candidate = torch.cat([center, size], dim=-1)
        if not self.use_gates:
            return candidate
        gate = 0.55 * torch.sigmoid(delta[..., 4:5])
        mixed = (1.0 - gate) * boxes16 + gate * candidate
        center = mixed[..., :2].clamp(0.0, 1.0)
        size = mixed[..., 2:].clamp(0.02, 0.95)
        return torch.cat([center, size], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del tool_mask_8
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t, -1, h, w))

        center_logits_8 = self.center_head(y).view(b, t, h * w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled8 = (y.flatten(2) * prob.reshape(b * t, 1, h * w)).sum(dim=-1).view(b, t, -1)
        size8 = torch.sigmoid(self.size_head(pooled8)).clamp(0.02, 0.95)
        coarse8 = torch.cat([centers8, size8], dim=-1)

        coarse16 = F.interpolate(coarse8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        visual16 = F.interpolate(pooled8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        feat16 = self._interp_8_to_16(y.view(b, t, -1, h, w))
        role16 = self.role_mlp(role_features_16)
        if self.use_tool_keypoints:
            kpt16 = self.tool_kpt_mlp(tool_kpt_vec_16)
        else:
            kpt16 = torch.zeros_like(role16)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        role_cond = role16 + cond16

        local1 = self._sample_box_features(feat16, coarse16.detach())
        delta1 = self.refine1(torch.cat([local1, visual16, role_cond, kpt16, coarse16.detach()], dim=-1))
        pred1 = self._apply_delta(coarse16, delta1, wh_scale=0.45)

        local2 = self._sample_box_features(feat16, pred1.detach())
        delta2 = self.refine2(torch.cat([local2, visual16, role_cond, kpt16, pred1.detach()], dim=-1))
        pred2 = self._apply_delta(pred1, delta2, wh_scale=0.30)

        return {
            "boxes_cxcywh": pred2,
            "boxes_xyxy": cxcywh_to_xyxy(pred2),
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "coarse_cxcywh": coarse16,
            "coarse_xyxy": cxcywh_to_xyxy(coarse16),
            "anchor_cxcywh": pred1,
            "anchor_xyxy": cxcywh_to_xyxy(pred1),
        }


class H163ClipPriorTemporalResidual(nn.Module):
    """Predict a clip-level operation-region prior, then per-frame residuals.

    The goal is to regularize the 16-frame ROI trajectory instead of letting the
    head solve every frame as an independent tight box.  The clip prior is a
    shared latent box, while frame-level residuals model camera/tissue motion.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_tool_keypoints: bool,
        residual_scale: float,
        clip_prior_output_mix: float,
    ) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.residual_scale = float(residual_scale)
        self.clip_prior_output_mix = float(clip_prior_output_mix)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        self.size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        seq_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=4,
            dim_feedforward=4 * c,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(seq_layer, num_layers=2)
        self.seq_in = nn.Sequential(
            nn.LayerNorm(3 * c + 4),
            nn.Linear(3 * c + 4, c),
            nn.SiLU(),
        )
        self.prior_weight_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, 1))
        self.prior_head = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 4),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(2 * c + 8),
            nn.Linear(2 * c + 8, c),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 4),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        self.pos_emb = nn.Parameter(torch.zeros(1, 16, c))
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del tool_mask_8
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t, -1, h, w))
        center_logits_8 = self.center_head(y).view(b, t, h * w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled8 = (y.flatten(2) * prob.reshape(b * t, 1, h * w)).sum(dim=-1).view(b, t, -1)
        size8 = torch.sigmoid(self.size_head(pooled8)).clamp(0.02, 0.95)
        coarse8 = torch.cat([centers8, size8], dim=-1)

        visual16 = F.interpolate(pooled8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        coarse16 = F.interpolate(coarse8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        role16 = self.role_mlp(role_features_16)
        if self.use_tool_keypoints:
            kpt16 = self.tool_kpt_mlp(tool_kpt_vec_16)
        else:
            kpt16 = torch.zeros_like(role16)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        seq = self.seq_in(torch.cat([visual16, role16 + cond16, kpt16, coarse16.detach()], dim=-1))
        seq = self.temporal(seq + self.pos_emb.to(device=seq.device, dtype=seq.dtype))

        prior_logits = self.prior_weight_head(seq).squeeze(-1)
        active_hint = role_features_16[..., 0].to(device=seq.device, dtype=seq.dtype)
        prior_w = F.softmax(prior_logits + 0.35 * active_hint, dim=1)
        prior_feat = (seq * prior_w[..., None]).sum(dim=1)
        raw_prior = self.prior_head(prior_feat)
        prior_center = torch.sigmoid(raw_prior[..., :2])
        prior_size = (0.02 + 0.93 * torch.sigmoid(raw_prior[..., 2:])).clamp(0.02, 0.95)
        prior = torch.cat([prior_center, prior_size], dim=-1)
        prior16 = prior[:, None, :].expand(-1, 16, -1)

        delta_in = torch.cat([seq, visual16, coarse16.detach(), prior16.detach()], dim=-1)
        delta = self.delta_head(delta_in)
        # Keep the strong per-frame coarse localization path, but bias it with a
        # clip-level prior so the final trajectory cannot drift independently in
        # every frame.
        prior_mix = float(np.clip(self.clip_prior_output_mix, 0.0, 1.0))
        base_center = (1.0 - prior_mix) * coarse16[..., :2] + prior_mix * prior16[..., :2]
        base_size = ((1.0 - prior_mix) * coarse16[..., 2:] + prior_mix * prior16[..., 2:]).clamp(0.02, 0.95)
        dc = torch.tanh(delta[..., :2]) * self.residual_scale * base_size.detach()
        dwh = torch.tanh(delta[..., 2:]) * 0.45
        center = (base_center + dc).clamp(0.0, 1.0)
        size = (base_size * torch.exp(dwh)).clamp(0.02, 0.95)
        pred = torch.cat([center, size], dim=-1)
        return {
            "boxes_cxcywh": pred,
            "boxes_xyxy": cxcywh_to_xyxy(pred),
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "coarse_cxcywh": coarse16,
            "coarse_xyxy": cxcywh_to_xyxy(coarse16),
            "clip_prior_cxcywh": prior,
            "clip_prior_xyxy": cxcywh_to_xyxy(prior[:, None, :]).squeeze(1),
            "anchor_cxcywh": prior16,
            "anchor_xyxy": cxcywh_to_xyxy(prior16),
        }


class H161RoiContextRefine(nn.Module):
    """Predict a coarse context box, then refine it to a tight final bbox."""

    def __init__(self, hidden_dim: int, use_tool_mask: bool, use_tool_keypoints: bool) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (1 if use_tool_mask else 0) + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.context_center_head = nn.Conv2d(c, 1, 1)
        self.tight_center_head = nn.Conv2d(c, 1, 1)
        self.context_size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.refine_head = nn.Sequential(
            nn.LayerNorm(4 * c + 4),
            nn.Linear(4 * c + 4, c),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 4),
        )
        nn.init.zeros_(self.refine_head[-1].weight)
        nn.init.zeros_(self.refine_head[-1].bias)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def _soft_box_pool(self, y: torch.Tensor, box_cxcywh: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # y: (B*T,C,H,W), box_cxcywh: (B*T,4)
        xyxy = cxcywh_to_xyxy(box_cxcywh)
        grid = self.grid_xy.to(device=y.device, dtype=y.dtype)
        gx = grid[0].view(1, h, w)
        gy = grid[1].view(1, h, w)
        x0, y0, x1, y1 = xyxy.unbind(dim=-1)
        sharp = 60.0
        mask = (
            torch.sigmoid(sharp * (gx - x0[:, None, None]))
            * torch.sigmoid(sharp * (x1[:, None, None] - gx))
            * torch.sigmoid(sharp * (gy - y0[:, None, None]))
            * torch.sigmoid(sharp * (y1[:, None, None] - gy))
        )
        weighted = y * mask[:, None]
        return weighted.flatten(2).sum(dim=-1) / mask.flatten(1).sum(dim=-1).clamp_min(1e-6)[:, None]

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias = self.role_mlp(role8).view(b, t, -1, 1, 1)
        if self.use_tool_keypoints:
            kpt8 = tool_kpt_vec_16.reshape(b, 8, 2, 9).mean(dim=2)
            role_bias = role_bias + self.tool_kpt_mlp(kpt8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias) + beta
        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_mask:
            feats.append(tool_mask_8.to(device=x.device, dtype=x.dtype))
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t, -1, h, w))

        context_logits = self.context_center_head(y).view(b, t, h * w)
        context_prob = F.softmax(context_logits, dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        context_centers8 = torch.matmul(context_prob, grid_flat)
        context_center_pool = (y.flatten(2) * context_prob.reshape(b * t, 1, h * w)).sum(dim=-1).view(b, t, -1)
        context_size8 = torch.sigmoid(self.context_size_head(context_center_pool)).clamp(0.02, 0.95)
        context8 = torch.cat([context_centers8, context_size8], dim=-1)

        context_pool8 = self._soft_box_pool(y, context8.reshape(b * t, 4), h, w).view(b, t, -1)
        context16 = F.interpolate(context8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        context_pool16 = F.interpolate(context_pool8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        center_pool16 = F.interpolate(context_center_pool.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        role16 = self.role_mlp(role_features_16)
        if self.use_tool_keypoints:
            kpt16 = self.tool_kpt_mlp(tool_kpt_vec_16)
        else:
            kpt16 = torch.zeros_like(role16)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        refine_in = torch.cat([context_pool16, center_pool16, role16 + cond16, kpt16, context16.detach()], dim=-1)
        delta = self.refine_head(refine_in)
        pred_center = (context16[..., :2] + torch.tanh(delta[..., :2]) * 0.5 * context16[..., 2:].detach()).clamp(0.0, 1.0)
        pred_size = (context16[..., 2:] * torch.sigmoid(delta[..., 2:])).clamp(0.005, 0.95)
        pred = torch.cat([pred_center, pred_size], dim=-1)
        return {
            "boxes_cxcywh": pred,
            "boxes_xyxy": cxcywh_to_xyxy(pred),
            "center_logits_8": self.tight_center_head(y).view(b, t, h, w),
            "context_cxcywh": context16,
            "context_xyxy": cxcywh_to_xyxy(context16),
            "context_center_logits_8": context_logits.view(b, t, h, w),
        }


class H161QuadTemporalRegion(nn.Module):
    """Predict an ordered 4-point operation region, then derive the Stage1 ROI.

    This keeps the output compatible with the bbox metric while giving the head
    an explicit intra-frame region structure and a 16-frame temporal sequence.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        residual_scale: float,
        tool_mask_spatial: bool = True,
    ) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_mask_spatial = bool(use_tool_mask) and bool(tool_mask_spatial)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.residual_scale = float(residual_scale)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (1 if self.use_tool_mask_spatial else 0) + (3 if use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        self.corner_heatmap_head = nn.Conv2d(c, 4, 1)
        self.support_head = nn.Conv2d(c, 1, 1)
        self.size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        seq_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=4,
            dim_feedforward=4 * c,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(seq_layer, num_layers=2)
        seq_in_dim = 3 * c + 4 + (c if self.use_tool_mask else 0)
        self.seq_in = nn.Sequential(
            nn.LayerNorm(seq_in_dim),
            nn.Linear(seq_in_dim, c),
            nn.SiLU(),
        )
        self.box_delta_head = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 4),
        )
        self.point_offset_head = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, 8),
        )
        nn.init.zeros_(self.box_delta_head[-1].weight)
        nn.init.zeros_(self.box_delta_head[-1].bias)
        nn.init.zeros_(self.point_offset_head[-1].weight)
        nn.init.zeros_(self.point_offset_head[-1].bias)
        base = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            dtype=torch.float32,
        )
        self.register_buffer("base_corners", base, persistent=False)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def _tool_mask_geom8(self, tool_mask_8: torch.Tensor) -> torch.Tensor:
        # Return [cx, cy, area, sx, sy] per 8 pooled frame.
        mask = tool_mask_8.squeeze(2)
        b, t, h, w = mask.shape
        grid = self.grid_xy.to(device=mask.device, dtype=mask.dtype)
        gx = grid[0].view(1, 1, h, w)
        gy = grid[1].view(1, 1, h, w)
        mass = mask.sum(dim=(-1, -2)).clamp_min(1e-6)
        cx = (mask * gx).sum(dim=(-1, -2)) / mass
        cy = (mask * gy).sum(dim=(-1, -2)) / mass
        area = mass / float(h * w)
        sx = torch.sqrt(((gx - cx[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        sy = torch.sqrt(((gy - cy[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        empty = (mask.sum(dim=(-1, -2)) <= 0.0).to(mask.dtype)
        cx = cx * (1.0 - empty) + 0.5 * empty
        cy = cy * (1.0 - empty) + 0.5 * empty
        return torch.stack([cx, cy, area, sx, sy], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t, h, w, _ = tokens.shape
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=x.device, dtype=x.dtype).view(1, 1, 2, h, w).expand(b, t, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_mask_spatial:
            feats.append(tool_mask_8.to(device=x.device, dtype=x.dtype))
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=x.device, dtype=x.dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t, -1, h, w))
        center_logits_8 = self.center_head(y).view(b, t, h * w)
        corner_logits_8 = self.corner_heatmap_head(y).view(b, t, 4, h, w)
        support_logits_8 = self.support_head(y).view(b, t, h, w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=x.device, dtype=x.dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled8 = (y.flatten(2) * prob.reshape(b * t, 1, h * w)).sum(dim=-1).view(b, t, -1)
        size8 = torch.sigmoid(self.size_head(pooled8)).clamp(0.03, 0.95)
        coarse8 = torch.cat([centers8, size8], dim=-1)

        visual16 = F.interpolate(pooled8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        coarse16 = F.interpolate(coarse8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        role16 = self.role_mlp(role_features_16)
        if self.use_tool_keypoints:
            kpt16 = self.tool_kpt_mlp(tool_kpt_vec_16)
        else:
            kpt16 = torch.zeros_like(role16)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        seq_parts = [visual16, role16 + cond16, kpt16]
        if self.use_tool_mask:
            mask_geom8 = self._tool_mask_geom8(tool_mask_8)
            mask_geom16 = F.interpolate(mask_geom8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
            seq_parts.append(self.tool_mask_geom_mlp(mask_geom16))
        seq_parts.append(coarse16.detach())
        seq = self.seq_in(torch.cat(seq_parts, dim=-1))
        seq = self.temporal(seq)

        box_delta = self.box_delta_head(seq)
        dc = torch.tanh(box_delta[..., :2]) * self.residual_scale * coarse16[..., 2:].detach()
        dwh = torch.tanh(box_delta[..., 2:]) * 0.5
        center = (coarse16[..., :2] + dc).clamp(0.0, 1.0)
        size = (coarse16[..., 2:] * torch.exp(dwh)).clamp(0.03, 0.95)

        corner_offsets = 0.35 * torch.tanh(self.point_offset_head(seq).view(b, 16, 4, 2))
        base = self.base_corners.to(device=seq.device, dtype=seq.dtype).view(1, 1, 4, 2)
        quad = center[:, :, None, :] + (base + corner_offsets) * size[:, :, None, :]
        quad = quad.clamp(0.0, 1.0)
        box_xyxy = quad_to_xyxy(quad, min_size=0.02)
        return {
            "boxes_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "boxes_xyxy": box_xyxy,
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "corner_logits_8": corner_logits_8,
            "support_logits_8": support_logits_8,
            "coarse_cxcywh": coarse16,
            "coarse_xyxy": cxcywh_to_xyxy(coarse16),
            "quad_xy": quad,
        }


class H180TokenQueryTemporalRegion(nn.Module):
    """Token-native query decoder for tight operation-region ROI.

    This follows the DETR/DAB-DETR/TokenPose pattern in a compact form:
    learned region/corner queries attend directly to V-JEPA patch tokens, then
    each query is smoothed across the 16-frame clip before producing a tight
    bbox and ordered quad.  The tool mask is a spatial condition on tokens, not
    a GT-derived ROI proposal.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        residual_scale: float,
        dab_refine: bool = False,
    ) -> None:
        super().__init__()
        if use_tool_keypoints:
            raise ValueError("H180TokenQueryTemporalRegion is the mask-only query path; do not enable keypoints.")
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.residual_scale = float(residual_scale)
        self.dab_refine = bool(dab_refine)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        self.grid_mlp = nn.Sequential(nn.Linear(2, c), nn.SiLU(), nn.Linear(c, c))
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_pixel_mlp = nn.Sequential(nn.Linear(1, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_norm = nn.LayerNorm(c)
        self.memory_norm = nn.LayerNorm(c)
        self.query_embed = nn.Parameter(torch.zeros(1, 5, c))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=c,
            nhead=4,
            dim_feedforward=4 * c,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.frame_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=4,
            dim_feedforward=4 * c,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=2)
        self.time_pos = nn.Parameter(torch.zeros(1, 16, c))
        self.center_query_norm = nn.LayerNorm(c)
        self.box_ref_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.size_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.box_delta_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 4))
        self.corner_offset_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.zeros_(self.size_head[-1].weight)
        nn.init.constant_(self.size_head[-1].bias, -1.8)
        nn.init.zeros_(self.box_delta_head[-1].weight)
        nn.init.zeros_(self.box_delta_head[-1].bias)
        nn.init.zeros_(self.corner_offset_head[-1].weight)
        nn.init.zeros_(self.corner_offset_head[-1].bias)
        base = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            dtype=torch.float32,
        )
        self.register_buffer("base_corners", base, persistent=False)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def _tool_mask_geom8(self, tool_mask_8: torch.Tensor) -> torch.Tensor:
        mask = tool_mask_8.squeeze(2)
        b, t, h, w = mask.shape
        grid = self.grid_xy.to(device=mask.device, dtype=mask.dtype)
        gx = grid[0].view(1, 1, h, w)
        gy = grid[1].view(1, 1, h, w)
        mass_raw = mask.sum(dim=(-1, -2))
        mass = mass_raw.clamp_min(1e-6)
        cx = (mask * gx).sum(dim=(-1, -2)) / mass
        cy = (mask * gy).sum(dim=(-1, -2)) / mass
        area = mass / float(h * w)
        sx = torch.sqrt(((gx - cx[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        sy = torch.sqrt(((gy - cy[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        empty = (mass_raw <= 0.0).to(mask.dtype)
        cx = cx * (1.0 - empty) + 0.5 * empty
        cy = cy * (1.0 - empty) + 0.5 * empty
        return torch.stack([cx, cy, area, sx, sy], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del tool_kpt_heatmaps_8, tool_kpt_vec_16
        b, t, h, w, _ = tokens.shape
        n = h * w
        device = tokens.device
        dtype = tokens.dtype

        mem = self.token_proj(self.token_norm(tokens)).reshape(b, t, n, -1)
        grid_flat = self.grid_xy.to(device=device, dtype=dtype).view(2, n).transpose(0, 1)
        mem = mem + self.grid_mlp(grid_flat).view(1, 1, n, -1)

        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias = self.role_mlp(role8).view(b, t, 1, -1)
        cond = self.cond_norm(self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id))
        mem = mem + role_bias + cond.view(b, 1, 1, -1)

        query_bias = role_bias.squeeze(2) + cond.view(b, 1, -1)
        if self.use_tool_mask:
            mask = tool_mask_8.to(device=device, dtype=dtype)
            mem = mem + self.tool_mask_pixel_mlp(mask.flatten(3).transpose(2, 3))
            mask_geom = self._tool_mask_geom8(mask)
            query_bias = query_bias + self.tool_mask_geom_mlp(mask_geom)

        mem = self.memory_norm(mem)
        mem_bt = mem.reshape(b * t, n, -1)
        queries = self.query_embed.to(device=device, dtype=dtype) + query_bias.reshape(b * t, 1, -1)
        q8 = self.frame_decoder(queries, mem_bt).reshape(b, t, 5, -1)

        region8 = q8[:, :, 0]
        center_logits_8 = (
            mem * self.center_query_norm(region8).view(b, t, 1, -1)
        ).sum(dim=-1) / math.sqrt(region8.shape[-1])
        prob = F.softmax(center_logits_8, dim=-1)
        centers8 = torch.matmul(prob, grid_flat)
        size8 = (0.03 + 0.92 * torch.sigmoid(self.size_head(region8))).clamp(0.03, 0.95)
        coarse8 = torch.cat([centers8, size8], dim=-1)
        coarse16 = F.interpolate(coarse8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)

        q16 = F.interpolate(
            q8.permute(0, 2, 3, 1).reshape(b * 5, -1, t),
            size=16,
            mode="linear",
            align_corners=True,
        ).reshape(b, 5, -1, 16).permute(0, 1, 3, 2)
        if self.dab_refine:
            q16 = q16 + self.box_ref_mlp(coarse16).unsqueeze(1)
        q_tem = q16.reshape(b * 5, 16, -1) + self.time_pos.to(device=device, dtype=dtype)
        q_tem = self.temporal(q_tem).reshape(b, 5, 16, -1).permute(0, 2, 1, 3)

        region16 = q_tem[:, :, 0]
        corner16 = q_tem[:, :, 1:5]
        delta = self.box_delta_head(region16)
        if self.dab_refine:
            refined = torch.sigmoid(inverse_sigmoid(coarse16, eps=1e-3) + self.residual_scale * torch.tanh(delta))
            center = refined[..., :2].clamp(0.0, 1.0)
            size = refined[..., 2:].clamp(0.03, 0.95)
        else:
            center = (coarse16[..., :2] + torch.tanh(delta[..., :2]) * self.residual_scale * coarse16[..., 2:].detach()).clamp(0.0, 1.0)
            size = (coarse16[..., 2:] * torch.exp(0.5 * torch.tanh(delta[..., 2:]))).clamp(0.03, 0.95)
        corner_offsets = 0.35 * torch.tanh(self.corner_offset_head(corner16))
        base = self.base_corners.to(device=device, dtype=dtype).view(1, 1, 4, 2)
        quad = (center[:, :, None, :] + (base + corner_offsets) * size[:, :, None, :]).clamp(0.0, 1.0)
        box_xyxy = quad_to_xyxy(quad, min_size=0.02)
        return {
            "boxes_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "boxes_xyxy": box_xyxy,
            "center_logits_8": center_logits_8.view(b, t, h, w),
            "coarse_cxcywh": coarse16,
            "coarse_xyxy": cxcywh_to_xyxy(coarse16),
            "quad_xy": quad,
        }


class H185DabRefineTemporalRegion(nn.Module):
    """DAB-style single-region decoder for tight operation ROI.

    H182 only added one box embedding before one residual update.  This class is
    closer to DAB/DINO's useful part for our one-box-per-frame problem: keep an
    explicit cxcywh reference box, let a temporal/cross-attention decoder refine
    it layer by layer, and supervise every intermediate box.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        residual_scale: float,
        decoder_layers: int,
        query_noise_scale: float,
    ) -> None:
        super().__init__()
        if use_tool_keypoints:
            raise ValueError("H185DabRefineTemporalRegion uses tool mask only; do not enable keypoints.")
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.residual_scale = float(residual_scale)
        self.query_noise_scale = float(query_noise_scale)
        self.decoder_layers = max(int(decoder_layers), 1)

        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (1 if self.use_tool_mask else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        self.size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))

        seq_in_dim = 3 * c + 4 + (c if self.use_tool_mask else 0)
        self.seq_in = nn.Sequential(
            nn.LayerNorm(seq_in_dim),
            nn.Linear(seq_in_dim, c),
            nn.SiLU(),
        )
        self.time_pos = nn.Parameter(torch.zeros(1, 16, c))
        self.ref_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.query_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            for _ in range(self.decoder_layers)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.memory_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            for _ in range(self.decoder_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, 4 * c),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(4 * c, c),
            )
            for _ in range(self.decoder_layers)
        ])
        self.bbox_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 4),
            )
            for _ in range(self.decoder_layers)
        ])
        for head in self.bbox_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def _tool_mask_geom8(self, tool_mask_8: torch.Tensor) -> torch.Tensor:
        mask = tool_mask_8.squeeze(2)
        b, t, h, w = mask.shape
        grid = self.grid_xy.to(device=mask.device, dtype=mask.dtype)
        gx = grid[0].view(1, 1, h, w)
        gy = grid[1].view(1, 1, h, w)
        mass_raw = mask.sum(dim=(-1, -2))
        mass = mass_raw.clamp_min(1e-6)
        cx = (mask * gx).sum(dim=(-1, -2)) / mass
        cy = (mask * gy).sum(dim=(-1, -2)) / mass
        area = mass / float(h * w)
        sx = torch.sqrt(((gx - cx[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        sy = torch.sqrt(((gy - cy[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        empty = (mass_raw <= 0.0).to(mask.dtype)
        cx = cx * (1.0 - empty) + 0.5 * empty
        cy = cy * (1.0 - empty) + 0.5 * empty
        return torch.stack([cx, cy, area, sx, sy], dim=-1)

    def _maybe_perturb_ref(self, ref: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.query_noise_scale <= 0.0:
            return ref
        noise = (torch.rand_like(ref) * 2.0 - 1.0) * self.query_noise_scale
        # Perturb centers additively and sizes multiplicatively.  This mirrors
        # DINO's training-only reference perturbation without using GT at test.
        center = (ref[..., :2] + noise[..., :2] * ref[..., 2:].detach()).clamp(1e-4, 1.0 - 1e-4)
        size = (ref[..., 2:] * torch.exp(noise[..., 2:])).clamp(0.02, 0.95)
        return torch.cat([center, size], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del tool_kpt_heatmaps_8, tool_kpt_vec_16
        b, t8, h, w, _ = tokens.shape
        device = tokens.device
        dtype = tokens.dtype

        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t8, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=device, dtype=dtype).view(1, 1, 2, h, w).expand(b, t8, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_mask:
            feats.append(tool_mask_8.to(device=device, dtype=dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t8, -1, h, w))

        center_logits_8 = self.center_head(y).view(b, t8, h * w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=device, dtype=dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled8 = (y.flatten(2) * prob.reshape(b * t8, 1, h * w)).sum(dim=-1).view(b, t8, -1)
        size8 = torch.sigmoid(self.size_head(pooled8)).clamp(0.03, 0.95)
        coarse8 = torch.cat([centers8, size8], dim=-1)
        coarse16 = F.interpolate(coarse8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        visual16 = F.interpolate(pooled8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)

        mem8 = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
        mem16 = F.interpolate(mem8, size=(16, h, w), mode="trilinear", align_corners=True)
        mem16 = mem16.permute(0, 2, 3, 4, 1).reshape(b, 16, h * w, -1)

        role16 = self.role_mlp(role_features_16)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        seq_parts = [visual16, role16 + cond16, self.ref_mlp(coarse16.detach()), coarse16.detach()]
        if self.use_tool_mask:
            mask_geom8 = self._tool_mask_geom8(tool_mask_8.to(device=device, dtype=dtype))
            mask_geom16 = F.interpolate(mask_geom8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
            seq_parts.insert(3, self.tool_mask_geom_mlp(mask_geom16))
        q = self.seq_in(torch.cat(seq_parts, dim=-1)) + self.time_pos.to(device=device, dtype=dtype)
        ref = coarse16.clamp(1e-4, 1.0 - 1e-4)

        layer_boxes = []
        layer_xyxy = []
        mem_bt = mem16.reshape(b * 16, h * w, -1)
        for i in range(self.decoder_layers):
            q_norm = self.query_norms[i](q)
            self_out, _ = self.self_attn[i](q_norm, q_norm, q_norm, need_weights=False)
            q = q + self_out

            ref_for_layer = self._maybe_perturb_ref(ref) if i > 0 else ref
            q_ref = q + self.ref_mlp(ref_for_layer)
            cross_q = self.cross_norms[i](q_ref).reshape(b * 16, 1, -1)
            mem_norm = self.memory_norms[i](mem_bt)
            cross_out, _ = self.cross_attn[i](cross_q, mem_norm, mem_norm, need_weights=False)
            q = q + cross_out.reshape(b, 16, -1)
            q = q + self.ffns[i](self.ffn_norms[i](q))

            delta = self.bbox_heads[i](q)
            updated = torch.sigmoid(inverse_sigmoid(ref_for_layer, eps=1e-4) + self.residual_scale * torch.tanh(delta))
            center = updated[..., :2].clamp(0.0, 1.0)
            size = updated[..., 2:].clamp(0.02, 0.95)
            ref = torch.cat([center, size], dim=-1)
            layer_boxes.append(ref)
            layer_xyxy.append(cxcywh_to_xyxy(ref))
            if i + 1 < self.decoder_layers:
                ref = ref.detach()

        boxes_cxcywh = layer_boxes[-1]
        boxes_xyxy = layer_xyxy[-1]
        return {
            "boxes_cxcywh": boxes_cxcywh,
            "boxes_xyxy": boxes_xyxy,
            "center_logits_8": center_logits_8.view(b, t8, h, w),
            "coarse_cxcywh": coarse16,
            "coarse_xyxy": cxcywh_to_xyxy(coarse16),
            "decoder_layer_cxcywh": torch.stack(layer_boxes, dim=0),
            "decoder_layer_xyxy": torch.stack(layer_xyxy, dim=0),
        }


class H187PointTokenTemporalRegion(nn.Module):
    """Explicit four-point spatial-temporal decoder for tight ROI.

    The target box is defined by four VIS points, so this head keeps those four
    points as first-class tokens.  Each decoder layer performs:
      1. same-frame attention among the four points,
      2. same-point temporal attention across 16 frames,
      3. cross-attention from the point tokens to V-JEPA spatial memory,
      4. point-coordinate refinement.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        residual_scale: float,
        decoder_layers: int,
        query_noise_scale: float = 0.0,
        use_local_point_features: bool = False,
        constrain_oriented_rect: bool = False,
    ) -> None:
        super().__init__()
        if use_tool_keypoints:
            raise ValueError("H187PointTokenTemporalRegion uses tool mask only; do not enable keypoints.")
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.residual_scale = float(residual_scale)
        self.decoder_layers = max(int(decoder_layers), 1)
        self.query_noise_scale = float(query_noise_scale)
        self.use_local_point_features = bool(use_local_point_features)
        self.constrain_oriented_rect = bool(constrain_oriented_rect)

        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (1 if self.use_tool_mask else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, 1, 1)
        self.size_head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        if self.constrain_oriented_rect:
            nn.init.zeros_(self.size_head[-1].weight)
            nn.init.constant_(self.size_head[-1].bias, -2.1)
        self.point_index = nn.Parameter(torch.zeros(1, 1, 4, c))
        self.time_pos = nn.Parameter(torch.zeros(1, 16, 1, c))
        self.point_ref_mlp = nn.Sequential(nn.Linear(2, c), nn.SiLU(), nn.Linear(c, c))
        self.local_point_proj = nn.Sequential(
            nn.LayerNorm(3 * c),
            nn.Linear(3 * c, c),
            nn.SiLU(),
            nn.Linear(c, c),
        )
        self.query_norm = nn.LayerNorm(c)
        self.frame_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.frame_attn = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            for _ in range(self.decoder_layers)
        ])
        self.temporal_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.temporal_attn = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            for _ in range(self.decoder_layers)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.memory_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            for _ in range(self.decoder_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Dropout(0.10), nn.Linear(4 * c, c))
            for _ in range(self.decoder_layers)
        ])
        self.point_delta_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
            for _ in range(self.decoder_layers)
        ])
        self.rect_delta_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 6))
            for _ in range(self.decoder_layers)
        ])
        for head in self.point_delta_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        for head in self.rect_delta_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.trunc_normal_(self.point_index, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        base = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            dtype=torch.float32,
        )
        sample_offsets = torch.tensor(
            [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        self.register_buffer("base_corners", base, persistent=False)
        self.register_buffer("point_sample_offsets", sample_offsets, persistent=False)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)
        self.point_sample_radius = 1.25 / 24.0

    def _tool_mask_geom8(self, tool_mask_8: torch.Tensor) -> torch.Tensor:
        mask = tool_mask_8.squeeze(2)
        b, t, h, w = mask.shape
        grid = self.grid_xy.to(device=mask.device, dtype=mask.dtype)
        gx = grid[0].view(1, 1, h, w)
        gy = grid[1].view(1, 1, h, w)
        mass_raw = mask.sum(dim=(-1, -2))
        mass = mass_raw.clamp_min(1e-6)
        cx = (mask * gx).sum(dim=(-1, -2)) / mass
        cy = (mask * gy).sum(dim=(-1, -2)) / mass
        area = mass / float(h * w)
        sx = torch.sqrt(((gx - cx[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        sy = torch.sqrt(((gy - cy[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        empty = (mass_raw <= 0.0).to(mask.dtype)
        cx = cx * (1.0 - empty) + 0.5 * empty
        cy = cy * (1.0 - empty) + 0.5 * empty
        return torch.stack([cx, cy, area, sx, sy], dim=-1)

    def _sample_point_features(self, feat16: torch.Tensor, ref_xy: torch.Tensor) -> torch.Tensor:
        # feat16: [B,16,C,H,W], ref_xy: [B,16,4,2] normalized to [0,1].
        b, t, c, h, w = feat16.shape
        p = ref_xy.shape[2]
        offsets = (
            self.point_sample_offsets.to(device=feat16.device, dtype=feat16.dtype)
            * float(self.point_sample_radius)
        )
        pts = ref_xy[:, :, :, None, :] + offsets.view(1, 1, 1, -1, 2)
        grid = (pts.reshape(b * t, p * offsets.shape[0], 1, 2) * 2.0 - 1.0).clamp(-1.15, 1.15)
        sampled = F.grid_sample(
            feat16.reshape(b * t, c, h, w).float(),
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).to(feat16.dtype)
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, t, p, offsets.shape[0], c)
        center = sampled[:, :, :, 0]
        mean = sampled.mean(dim=3)
        maxv = sampled.max(dim=3).values
        return torch.cat([center, mean, maxv], dim=-1)

    def _maybe_perturb_points(self, ref_xy: torch.Tensor, step_scale: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.query_noise_scale <= 0.0:
            return ref_xy
        noise = (torch.rand_like(ref_xy) * 2.0 - 1.0) * self.query_noise_scale
        return (ref_xy + noise * step_scale.detach()).clamp(0.0, 1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del tool_kpt_heatmaps_8, tool_kpt_vec_16
        b, t8, h, w, _ = tokens.shape
        device = tokens.device
        dtype = tokens.dtype

        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t8, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=device, dtype=dtype).view(1, 1, 2, h, w).expand(b, t8, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_mask:
            feats.append(tool_mask_8.to(device=device, dtype=dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t8, -1, h, w))

        center_logits_8 = self.center_head(y).view(b, t8, h * w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=device, dtype=dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        pooled8 = (y.flatten(2) * prob.reshape(b * t8, 1, h * w)).sum(dim=-1).view(b, t8, -1)
        size8 = torch.sigmoid(self.size_head(pooled8)).clamp(0.03, 0.95)
        coarse8 = torch.cat([centers8, size8], dim=-1)
        coarse16 = F.interpolate(coarse8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        visual16 = F.interpolate(pooled8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)

        mem8 = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
        mem16_chw = F.interpolate(mem8, size=(16, h, w), mode="trilinear", align_corners=True)
        feat16_map = mem16_chw.permute(0, 2, 1, 3, 4).contiguous()
        mem16 = mem16_chw.permute(0, 2, 3, 4, 1).reshape(b, 16, h * w, -1)

        role16 = self.role_mlp(role_features_16)
        cond16 = z[:, None, :].expand(-1, 16, -1)
        query_base = visual16 + role16 + cond16
        if self.use_tool_mask:
            mask_geom8 = self._tool_mask_geom8(tool_mask_8.to(device=device, dtype=dtype))
            mask_geom16 = F.interpolate(mask_geom8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
            query_base = query_base + self.tool_mask_geom_mlp(mask_geom16)

        base = self.base_corners.to(device=device, dtype=dtype).view(1, 1, 4, 2)
        rect_center = coarse16[:, :, :2].clamp(0.0, 1.0)
        rect_size = coarse16[:, :, 2:].clamp(0.03, 0.95)
        rect_axis = torch.zeros(b, 16, 2, device=device, dtype=dtype)
        rect_axis[..., 0] = 1.0
        if self.constrain_oriented_rect:
            ref = oriented_rect_to_quad(rect_center, rect_size.detach(), rect_axis)
        else:
            ref = (coarse16[:, :, None, :2] + base * coarse16[:, :, None, 2:].detach()).clamp(0.0, 1.0)
        q = (
            query_base[:, :, None, :]
            + self.point_index.to(device=device, dtype=dtype)
            + self.time_pos.to(device=device, dtype=dtype)
            + self.point_ref_mlp(ref)
        )
        q = self.query_norm(q)

        layer_quads = []
        layer_boxes = []
        layer_xyxy = []
        mem_bt = mem16.reshape(b * 16, h * w, -1)
        for i in range(self.decoder_layers):
            step_scale = (self.residual_scale * rect_size[:, :, None, :].detach()).clamp(0.01, 0.50)
            ref_for_layer = self._maybe_perturb_points(ref, step_scale)
            if self.use_local_point_features:
                q = q + self.local_point_proj(self._sample_point_features(feat16_map, ref_for_layer))
            q_frame = q.reshape(b * 16, 4, -1)
            frame_out, _ = self.frame_attn[i](
                self.frame_norms[i](q_frame),
                self.frame_norms[i](q_frame),
                self.frame_norms[i](q_frame),
                need_weights=False,
            )
            q = (q_frame + frame_out).reshape(b, 16, 4, -1)

            q_tem = q.permute(0, 2, 1, 3).reshape(b * 4, 16, -1)
            tem_out, _ = self.temporal_attn[i](
                self.temporal_norms[i](q_tem),
                self.temporal_norms[i](q_tem),
                self.temporal_norms[i](q_tem),
                need_weights=False,
            )
            q = (q_tem + tem_out).reshape(b, 4, 16, -1).permute(0, 2, 1, 3)

            q_ref = q + self.point_ref_mlp(ref)
            cross_q = self.cross_norms[i](q_ref).reshape(b * 16, 4, -1)
            mem_norm = self.memory_norms[i](mem_bt)
            cross_out, _ = self.cross_attn[i](cross_q, mem_norm, mem_norm, need_weights=False)
            q = q + cross_out.reshape(b, 16, 4, -1)
            q = q + self.ffns[i](self.ffn_norms[i](q))

            if self.constrain_oriented_rect:
                rect_delta = torch.tanh(self.rect_delta_heads[i](q.mean(dim=2)))
                rect_center = (
                    rect_center + rect_delta[..., :2] * self.residual_scale * rect_size.detach()
                ).clamp(0.0, 1.0)
                rect_size = (rect_size * torch.exp(0.45 * rect_delta[..., 2:4])).clamp(0.02, 0.95)
                rect_axis = F.normalize(rect_axis + 0.35 * rect_delta[..., 4:6], dim=-1, eps=1e-6)
                ref_new = oriented_rect_to_quad(rect_center, rect_size, rect_axis)
            else:
                delta = self.point_delta_heads[i](q)
                ref_new = (ref + torch.tanh(delta) * step_scale).clamp(0.0, 1.0)
            layer_quads.append(ref_new)
            box_xyxy = quad_to_xyxy(ref_new, min_size=0.02)
            layer_xyxy.append(box_xyxy)
            layer_boxes.append(xyxy_to_cxcywh_torch(box_xyxy))
            ref = ref_new.detach() if i + 1 < self.decoder_layers else ref_new
            if self.constrain_oriented_rect and i + 1 < self.decoder_layers:
                rect_center = rect_center.detach()
                rect_size = rect_size.detach()
                rect_axis = rect_axis.detach()

        quad = layer_quads[-1]
        box_xyxy = layer_xyxy[-1]
        return {
            "boxes_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "boxes_xyxy": box_xyxy,
            "center_logits_8": center_logits_8.view(b, t8, h, w),
            "coarse_cxcywh": coarse16,
            "coarse_xyxy": cxcywh_to_xyxy(coarse16),
            "decoder_layer_cxcywh": torch.stack(layer_boxes, dim=0),
            "decoder_layer_xyxy": torch.stack(layer_xyxy, dim=0),
            "quad_xy": quad,
        }


class H191QuadLocalPointTemporalRegion(nn.Module):
    """H174-style quad stage followed by noisy local point refinement.

    Stage 1 predicts a structured quad/box with the existing quad-temporal head.
    Stage 2 treats the four predicted vertices as point queries, samples local
    V-JEPA/tool-mask features around each vertex, and refines them with
    same-frame and same-point temporal attention.  Training-time reference noise
    sits between the two stages so the refiner cannot learn a trivial copy.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        residual_scale: float,
        decoder_layers: int,
        query_noise_scale: float = 0.0,
        dn_query_noise_scale: float = 0.0,
        dn_query_groups: int = 1,
        dn_query_noise_min_scale: float = 1.0,
        use_proposal_oracle_refiner: bool = False,
        proposal_oracle_refiner_noise_scale: float = 0.0,
        stage2_bridge_noise_mode: str = "local_only",
        use_tool_trajectory_bias: bool = False,
        use_tool_trajectory_attn: bool = False,
        use_tool_corner_query_bias: bool = False,
        stage1_drop_tool_keypoints: bool = False,
        detach_stage1_refiner: bool = False,
        local_grid_refine: bool = False,
        local_grid_refine_grid_size: int = 5,
        local_grid_refine_scale: float = 1.35,
        local_grid_refine_temperature: float = 0.20,
        local_grid_refine_prior_scale: float = 6.0,
        local_grid_refine_prior_sigma: float = 0.28,
        local_grid_refine_detach_base: bool = False,
        local_grid_final_refine: bool = True,
        local_grid_confidence_score: bool = False,
        local_grid_confidence_scale: float = 0.35,
        local_grid_confidence_detach: bool = False,
        candidate_mask_quality: bool = False,
        candidate_mask_grid_size: int = 7,
        candidate_mask_score_scale: float = 0.0,
        candidate_mask_score_detach: bool = False,
        use_rgb_sidecar: bool = False,
    ) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.use_rgb_sidecar = bool(use_rgb_sidecar)
        self.use_tool_trajectory_bias = bool(use_tool_trajectory_bias) and self.use_tool_keypoints
        self.use_tool_trajectory_attn = bool(use_tool_trajectory_attn) and self.use_tool_keypoints
        self.use_tool_corner_query_bias = bool(use_tool_corner_query_bias) and self.use_tool_keypoints
        self.stage1_drop_tool_keypoints = bool(stage1_drop_tool_keypoints)
        self.detach_stage1_refiner = bool(detach_stage1_refiner)
        self.local_grid_refine = bool(local_grid_refine)
        self.local_grid_refine_grid_size = max(int(local_grid_refine_grid_size), 3)
        if self.local_grid_refine_grid_size % 2 == 0:
            self.local_grid_refine_grid_size += 1
        self.local_grid_refine_scale = max(float(local_grid_refine_scale), 0.25)
        self.local_grid_refine_temperature = max(float(local_grid_refine_temperature), 1e-3)
        self.local_grid_refine_prior_scale = float(local_grid_refine_prior_scale)
        self.local_grid_refine_prior_sigma = max(float(local_grid_refine_prior_sigma), 1e-3)
        self.local_grid_refine_detach_base = bool(local_grid_refine_detach_base)
        self.local_grid_final_refine = bool(local_grid_final_refine)
        stage1_use_tool_keypoints = self.use_tool_keypoints and not self.stage1_drop_tool_keypoints
        self.stage1 = H161QuadTemporalRegion(
            hidden_dim=c,
            use_tool_mask=self.use_tool_mask,
            use_tool_keypoints=stage1_use_tool_keypoints,
            residual_scale=float(residual_scale),
            tool_mask_spatial=self.use_tool_mask,
        )
        self.residual_scale = float(residual_scale)
        self.decoder_layers = max(int(decoder_layers), 1)
        self.query_noise_scale = float(query_noise_scale)
        self.dn_query_noise_scale = float(dn_query_noise_scale)
        self.dn_query_groups = max(int(dn_query_groups), 1)
        self.dn_query_noise_min_scale = float(dn_query_noise_min_scale)
        self.use_proposal_oracle_refiner = bool(use_proposal_oracle_refiner)
        self.proposal_oracle_refiner_noise_scale = float(proposal_oracle_refiner_noise_scale)
        if stage2_bridge_noise_mode not in {"local_only", "replace"}:
            raise ValueError(f"unknown stage2_bridge_noise_mode={stage2_bridge_noise_mode!r}")
        self.stage2_bridge_noise_mode = str(stage2_bridge_noise_mode)

        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_traj_mlp = nn.Sequential(nn.LayerNorm(16 * 9), nn.Linear(16 * 9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_traj_token_mlp = nn.Sequential(nn.LayerNorm(13), nn.Linear(13, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_corner_ref_mlp = nn.Sequential(nn.Linear(2, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.rgb_encoder = make_rgb_sidecar_encoder(c) if self.use_rgb_sidecar else None
        spatial_in_ch = (
            c
            + (c if self.use_rgb_sidecar else 0)
            + 2
            + (1 if self.use_tool_mask else 0)
            + (3 if self.use_tool_keypoints else 0)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(spatial_in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )

        self.point_index = nn.Parameter(torch.zeros(1, 1, 4, c))
        self.time_pos = nn.Parameter(torch.zeros(1, 16, 1, c))
        self.tool_time_pos = nn.Parameter(torch.zeros(1, 16, c))
        self.point_ref_mlp = nn.Sequential(nn.Linear(2, c), nn.SiLU(), nn.Linear(c, c))
        self.query_norm = nn.LayerNorm(c)
        self.local_point_proj = nn.Sequential(
            nn.LayerNorm(3 * c),
            nn.Linear(3 * c, c),
            nn.SiLU(),
            nn.Linear(c, c),
        )
        if self.local_grid_refine:
            self.local_grid_point_proj = nn.Sequential(
                nn.LayerNorm(3 * c + 2),
                nn.Linear(3 * c + 2, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.local_grid_corner_embed = nn.Parameter(torch.zeros(1, 1, 4, c))
            self.local_grid_corner_head = nn.Sequential(
                nn.LayerNorm(2 * c + 2),
                nn.Linear(2 * c + 2, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.trunc_normal_(self.local_grid_corner_embed, std=0.02)
            nn.init.zeros_(self.local_grid_corner_head[-1].weight)
            nn.init.zeros_(self.local_grid_corner_head[-1].bias)
        self.frame_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.frame_attn = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            for _ in range(self.decoder_layers)
        ])
        self.temporal_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.temporal_attn = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            for _ in range(self.decoder_layers)
        ])
        self.tool_query_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.tool_memory_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.tool_traj_attn = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            for _ in range(self.decoder_layers)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.memory_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            for _ in range(self.decoder_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(self.decoder_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Dropout(0.10), nn.Linear(4 * c, c))
            for _ in range(self.decoder_layers)
        ])
        self.point_delta_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
            for _ in range(self.decoder_layers)
        ])
        for head in self.point_delta_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.trunc_normal_(self.point_index, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.tool_time_pos, std=0.02)
        sample_offsets = torch.tensor(
            [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        self.register_buffer("point_sample_offsets", sample_offsets, persistent=False)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)
        self.point_sample_radius = 1.25 / 24.0

    def _tool_anchor_quad(self, tool_kpt_vec_16: torch.Tensor) -> torch.Tensor:
        b = tool_kpt_vec_16.shape[0]
        device = tool_kpt_vec_16.device
        dtype = tool_kpt_vec_16.dtype
        kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
        kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        vis_sum = kpt_vis.sum(dim=2).clamp_min(1e-4)
        base_center = (kpt_xy * kpt_vis).sum(dim=2) / vis_sum
        base_axis = kpt_xy[:, :, 0] - kpt_xy[:, :, -1]
        fallback = torch.zeros_like(base_axis)
        fallback[..., 0] = 1.0
        axis_norm = base_axis.norm(dim=-1, keepdim=True)
        base_axis = torch.where(axis_norm > 1e-4, base_axis / axis_norm.clamp_min(1e-4), fallback)
        kpt_dist = (kpt_xy - base_center[:, :, None, :]).norm(dim=-1)
        tool_extent = (kpt_dist * kpt_vis.squeeze(-1)).sum(dim=2) / vis_sum.squeeze(-1)
        base_size = torch.stack(
            [
                (0.75 * tool_extent + 0.060).clamp(0.045, 0.24),
                (0.45 * tool_extent + 0.045).clamp(0.035, 0.18),
            ],
            dim=-1,
        )
        return oriented_rect_to_quad(base_center, base_size, base_axis).clamp(0.0, 1.0)

    def _tool_mask_geom8(self, tool_mask_8: torch.Tensor) -> torch.Tensor:
        mask = tool_mask_8.squeeze(2)
        b, t, h, w = mask.shape
        grid = self.grid_xy.to(device=mask.device, dtype=mask.dtype)
        gx = grid[0].view(1, 1, h, w)
        gy = grid[1].view(1, 1, h, w)
        mass_raw = mask.sum(dim=(-1, -2))
        mass = mass_raw.clamp_min(1e-6)
        cx = (mask * gx).sum(dim=(-1, -2)) / mass
        cy = (mask * gy).sum(dim=(-1, -2)) / mass
        area = mass / float(h * w)
        sx = torch.sqrt(((gx - cx[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        sy = torch.sqrt(((gy - cy[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        empty = (mass_raw <= 0.0).to(mask.dtype)
        cx = cx * (1.0 - empty) + 0.5 * empty
        cy = cy * (1.0 - empty) + 0.5 * empty
        return torch.stack([cx, cy, area, sx, sy], dim=-1)

    def _sample_point_features(self, feat16: torch.Tensor, ref_xy: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = feat16.shape
        p = ref_xy.shape[2]
        offsets = (
            self.point_sample_offsets.to(device=feat16.device, dtype=feat16.dtype)
            * float(self.point_sample_radius)
        )
        pts = ref_xy[:, :, :, None, :] + offsets.view(1, 1, 1, -1, 2)
        grid = (pts.reshape(b * t, p * offsets.shape[0], 1, 2) * 2.0 - 1.0).clamp(-1.15, 1.15)
        sampled = F.grid_sample(
            feat16.reshape(b * t, c, h, w).float(),
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).to(feat16.dtype)
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, t, p, offsets.shape[0], c)
        center = sampled[:, :, :, 0]
        mean = sampled.mean(dim=3)
        maxv = sampled.max(dim=3).values
        return torch.cat([center, mean, maxv], dim=-1)

    def _local_grid_points(
        self,
        quad: torch.Tensor,
        grid_size: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a rotated local search grid around an ordered quad."""
        orig_shape = quad.shape[:-2]
        q = quad.reshape(-1, 4, 2)
        center = q.mean(dim=-2)
        dirs, lens = quad_axes(q)
        coords = torch.linspace(
            -0.5 * float(scale),
            0.5 * float(scale),
            int(grid_size),
            device=quad.device,
            dtype=quad.dtype,
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        rel = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        pts = (
            center[:, None, :]
            + dirs[:, None, 0, :] * (rel[None, :, 0:1] * lens[:, None, 0:1])
            + dirs[:, None, 1, :] * (rel[None, :, 1:2] * lens[:, None, 1:2])
        )
        pts = pts.clamp(0.0, 1.0)
        return pts.reshape(*orig_shape, grid_size * grid_size, 2), rel.reshape(1, 1, grid_size * grid_size, 2)

    def _local_grid_refine_quad(
        self,
        feat16_map: torch.Tensor,
        base_quad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t = base_quad.shape[:2]
        grid_size = int(self.local_grid_refine_grid_size)
        search_quad = base_quad.detach() if self.local_grid_refine_detach_base else base_quad
        grid_pts, rel = self._local_grid_points(
            search_quad,
            grid_size=grid_size,
            scale=float(self.local_grid_refine_scale),
        )
        p = grid_pts.shape[2]
        point_feat = self._sample_point_features(feat16_map, grid_pts)
        rel_feat = rel.to(device=point_feat.device, dtype=point_feat.dtype).expand(b, t, -1, -1)
        grid_feat = self.local_grid_point_proj(torch.cat([point_feat, rel_feat], dim=-1))
        corner_feat = self.local_grid_corner_embed.to(device=grid_feat.device, dtype=grid_feat.dtype)
        corner_feat = corner_feat.expand(b, t, -1, -1)
        corner_grid_feat = grid_feat[:, :, None, :, :].expand(-1, -1, 4, -1, -1)
        corner_token = corner_feat[:, :, :, None, :].expand(-1, -1, -1, p, -1)
        grid_xy = grid_pts[:, :, None, :, :].expand(-1, -1, 4, -1, -1)
        logits = self.local_grid_corner_head(
            torch.cat([corner_grid_feat, corner_token, grid_xy], dim=-1)
        ).squeeze(-1)
        if self.local_grid_refine_prior_scale != 0.0:
            prior_quad = search_quad.to(device=grid_pts.device, dtype=grid_pts.dtype)
            _dirs, lens = quad_axes(prior_quad.reshape(-1, 4, 2))
            side = lens.mean(dim=-1).reshape(b, t, 1, 1).clamp_min(1e-4)
            prior_dist2 = (grid_xy - prior_quad[:, :, :, None, :]).square().sum(dim=-1)
            sigma = float(self.local_grid_refine_prior_sigma) * side
            corner_prior = -prior_dist2 / (2.0 * sigma.square().clamp_min(1e-8))
            logits = logits + float(self.local_grid_refine_prior_scale) * corner_prior
        probs = F.softmax(logits / float(self.local_grid_refine_temperature), dim=-1)
        quad = (probs[..., None] * grid_xy).sum(dim=-2).clamp(0.0, 1.0)
        return quad, logits, grid_pts

    def _maybe_perturb_points(self, ref_xy: torch.Tensor, step_scale: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.query_noise_scale <= 0.0:
            return ref_xy
        noise = (torch.rand_like(ref_xy) * 2.0 - 1.0) * self.query_noise_scale
        return (ref_xy + noise * step_scale.detach()).clamp(0.0, 1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
        dn_quad_xy: torch.Tensor | None = None,
        rgb_16: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        stage1_kwargs: dict[str, torch.Tensor] = {
            "tokens": tokens,
            "role_features_16": role_features_16,
            "tool_mask_8": tool_mask_8,
            "tool_kpt_heatmaps_8": tool_kpt_heatmaps_8,
            "tool_kpt_vec_16": tool_kpt_vec_16,
            "tool_id": tool_id,
            "act_id": act_id,
            "combo_id": combo_id,
        }
        if getattr(self.stage1, "use_rgb_sidecar", False):
            stage1_kwargs["rgb_16"] = rgb_16
        stage1 = self.stage1(**stage1_kwargs)
        b, t8, h, w, _ = tokens.shape
        device = tokens.device
        dtype = tokens.dtype

        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t8, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=device, dtype=dtype).view(1, 1, 2, h, w).expand(b, t8, -1, -1, -1)
        spatial_parts = [x]
        if self.use_rgb_sidecar:
            if self.rgb_encoder is None:
                raise RuntimeError("RGB sidecar is enabled without an encoder")
            spatial_parts.append(encode_rgb_sidecar_8(self.rgb_encoder, rgb_16, dtype=dtype))
        spatial_parts.append(grid)
        if self.use_tool_mask:
            spatial_parts.append(tool_mask_8.to(device=device, dtype=dtype))
        if self.use_tool_keypoints:
            spatial_parts.append(tool_kpt_heatmaps_8.to(device=device, dtype=dtype))
        y = self.spatial(torch.cat(spatial_parts, dim=2).reshape(b * t8, -1, h, w))
        mem8 = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
        mem16_chw = F.interpolate(mem8, size=(16, h, w), mode="trilinear", align_corners=True)
        feat16_map = mem16_chw.permute(0, 2, 1, 3, 4).contiguous()
        mem16 = mem16_chw.permute(0, 2, 3, 4, 1).reshape(b, 16, h * w, -1)

        query_base = (
            self.role_mlp(role_features_16)
            + z[:, None, :].expand(-1, 16, -1)
        )
        if self.use_tool_mask:
            mask_geom8 = self._tool_mask_geom8(tool_mask_8.to(device=device, dtype=dtype))
            mask_geom16 = F.interpolate(mask_geom8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
            query_base = query_base + self.tool_mask_geom_mlp(mask_geom16)
        if self.use_tool_keypoints:
            query_base = query_base + self.tool_kpt_mlp(tool_kpt_vec_16.to(device=device, dtype=dtype))
            if self.use_tool_trajectory_bias:
                traj_bias = self.tool_traj_mlp(tool_kpt_vec_16.to(device=device, dtype=dtype).reshape(b, 16 * 9))
                query_base = query_base + traj_bias[:, None, :]
        tool_memory = None
        if self.use_tool_trajectory_attn:
            tool_memory = self.tool_traj_token_mlp(
                torch.cat(
                    [
                        tool_kpt_vec_16.to(device=device, dtype=dtype),
                        role_features_16.to(device=device, dtype=dtype),
                    ],
                    dim=-1,
                )
            )
            tool_memory = tool_memory + z[:, None, :] + self.tool_time_pos.to(device=device, dtype=dtype)
        corner_query_bias = 0.0
        if self.use_tool_corner_query_bias:
            tool_anchor_quad = self._tool_anchor_quad(tool_kpt_vec_16.to(device=device, dtype=dtype))
            corner_query_bias = self.tool_corner_ref_mlp(tool_anchor_quad)
        corner_query_bias_all = (
            corner_query_bias[:, :, None, :, :] if isinstance(corner_query_bias, torch.Tensor) else 0.0
        )

        ref_clean = stage1["quad_xy"].clamp(0.0, 1.0)
        initial_box_xyxy = stage1["boxes_xyxy"]
        if self.detach_stage1_refiner:
            ref_clean = ref_clean.detach()
            initial_box_xyxy = initial_box_xyxy.detach()
        initial_wh = xyxy_to_cxcywh_torch(initial_box_xyxy)[..., 2:].detach()
        step_scale = (self.residual_scale * initial_wh[:, :, None, :]).clamp(0.01, 0.50)

        if self.stage2_bridge_noise_mode == "replace":
            # H193/H194: corrupt the actual stage1-to-stage2 bridge.  This is
            # kept as an option because it is useful diagnostically, but the
            # default follows H192: clean main query plus noisy local sampling.
            ref = self._maybe_perturb_points(ref_clean, step_scale)
        else:
            ref = ref_clean
        q = (
            query_base[:, :, None, :]
            + self.point_index.to(device=device, dtype=dtype)
            + self.time_pos.to(device=device, dtype=dtype)
            + self.point_ref_mlp(ref)
            + corner_query_bias
        )
        q = self.query_norm(q)

        layer_boxes = []
        layer_xyxy = []
        layer_quads = []
        mem_bt = mem16.reshape(b * 16, h * w, -1)
        for i in range(self.decoder_layers):
            sample_ref = ref
            if self.stage2_bridge_noise_mode == "local_only":
                sample_ref = self._maybe_perturb_points(ref, step_scale)
            q = q + self.local_point_proj(self._sample_point_features(feat16_map, sample_ref))

            q_frame = q.reshape(b * 16, 4, -1)
            frame_out, _ = self.frame_attn[i](
                self.frame_norms[i](q_frame),
                self.frame_norms[i](q_frame),
                self.frame_norms[i](q_frame),
                need_weights=False,
            )
            q = (q_frame + frame_out).reshape(b, 16, 4, -1)

            q_tem = q.permute(0, 2, 1, 3).reshape(b * 4, 16, -1)
            tem_out, _ = self.temporal_attn[i](
                self.temporal_norms[i](q_tem),
                self.temporal_norms[i](q_tem),
                self.temporal_norms[i](q_tem),
                need_weights=False,
            )
            q = (q_tem + tem_out).reshape(b, 4, 16, -1).permute(0, 2, 1, 3)

            if tool_memory is not None:
                q_tool = q.reshape(b, 16 * 4, -1)
                tool_out, _ = self.tool_traj_attn[i](
                    self.tool_query_norms[i](q_tool),
                    self.tool_memory_norms[i](tool_memory),
                    self.tool_memory_norms[i](tool_memory),
                    need_weights=False,
                )
                q = (q_tool + tool_out).reshape(b, 16, 4, -1)

            q_ref = q + self.point_ref_mlp(ref)
            cross_q = self.cross_norms[i](q_ref).reshape(b * 16, 4, -1)
            mem_norm = self.memory_norms[i](mem_bt)
            cross_out, _ = self.cross_attn[i](cross_q, mem_norm, mem_norm, need_weights=False)
            q = q + cross_out.reshape(b, 16, 4, -1)
            q = q + self.ffns[i](self.ffn_norms[i](q))

            delta = self.point_delta_heads[i](q)
            ref_new = (ref + torch.tanh(delta) * step_scale).clamp(0.0, 1.0)
            layer_quads.append(ref_new)
            box_xyxy = quad_to_xyxy(ref_new, min_size=0.02)
            layer_xyxy.append(box_xyxy)
            layer_boxes.append(xyxy_to_cxcywh_torch(box_xyxy))
            ref = ref_new.detach() if i + 1 < self.decoder_layers else ref_new

        final_quad = layer_quads[-1]
        final_xyxy = layer_xyxy[-1]
        local_grid_corner_logits = None
        local_grid_points = None
        if self.local_grid_refine and self.local_grid_final_refine:
            final_quad, local_grid_corner_logits, local_grid_points = self._local_grid_refine_quad(
                feat16_map,
                final_quad,
            )
            final_xyxy = quad_to_xyxy(final_quad, min_size=0.02)
        out = dict(stage1)
        # Keep optional Stage1 OICR logits only when they still line up with
        # the proposal bank used by proposal losses. This is the normal
        # non-recollect path; stale tensors are dropped defensively.
        if (
            "proposal_base_logits" in out
            and "proposal_logits" in out
            and out["proposal_base_logits"].shape != out["proposal_logits"].shape
        ):
            out.pop("proposal_base_logits", None)
        if (
            "proposal_oicr_logits" in out
            and "proposal_logits" in out
            and out["proposal_oicr_logits"].shape != out["proposal_logits"].shape
        ):
            out.pop("proposal_oicr_logits", None)
        out["boxes_xyxy"] = final_xyxy
        out["boxes_cxcywh"] = xyxy_to_cxcywh_torch(final_xyxy)
        out["coarse_xyxy"] = initial_box_xyxy
        out["coarse_cxcywh"] = xyxy_to_cxcywh_torch(initial_box_xyxy)
        out["decoder_layer_xyxy"] = torch.stack(layer_xyxy, dim=0)
        out["decoder_layer_cxcywh"] = torch.stack(layer_boxes, dim=0)
        out["quad_xy"] = final_quad
        if local_grid_corner_logits is not None and local_grid_points is not None:
            out["local_grid_corner_logits"] = local_grid_corner_logits
            out["local_grid_points"] = local_grid_points
            out["local_grid_base_quad"] = layer_quads[-1]
        if dn_quad_xy is not None and self.training and self.dn_query_noise_scale > 0.0:
            gt_ref = dn_quad_xy.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            gt_xyxy = quad_to_xyxy(gt_ref, min_size=0.02)
            gt_wh = xyxy_to_cxcywh_torch(gt_xyxy)[..., 2:].detach()
            dn_groups = max(int(self.dn_query_groups), 1)
            dn_b = b * dn_groups
            if dn_groups > 1:
                gt_ref_dn = (
                    gt_ref[:, None]
                    .expand(b, dn_groups, 16, 4, 2)
                    .reshape(dn_b, 16, 4, 2)
                    .contiguous()
                )
                gt_wh_dn = (
                    gt_wh[:, None]
                    .expand(b, dn_groups, 16, 2)
                    .reshape(dn_b, 16, 2)
                    .contiguous()
                )
                noise_scales = torch.linspace(
                    float(self.dn_query_noise_min_scale),
                    1.0,
                    steps=dn_groups,
                    device=device,
                    dtype=dtype,
                ).clamp_min(0.0)
                noise_scales = noise_scales.view(1, dn_groups, 1, 1, 1).expand(b, dn_groups, 16, 4, 2)
                noise_scales = noise_scales.reshape(dn_b, 16, 4, 2)
                query_base_dn = (
                    query_base[:, None]
                    .expand(b, dn_groups, 16, -1)
                    .reshape(dn_b, 16, -1)
                    .contiguous()
                )
                feat16_map_dn = (
                    feat16_map[:, None]
                    .expand(b, dn_groups, 16, -1, h, w)
                    .reshape(dn_b, 16, -1, h, w)
                    .contiguous()
                )
                mem_bt_dn = (
                    mem16[:, None]
                    .expand(b, dn_groups, 16, h * w, -1)
                    .reshape(dn_b * 16, h * w, -1)
                    .contiguous()
                )
                if tool_memory is not None:
                    tool_memory_dn = (
                        tool_memory[:, None]
                        .expand(b, dn_groups, 16, -1)
                        .reshape(dn_b, 16, -1)
                        .contiguous()
                    )
                else:
                    tool_memory_dn = None
                if isinstance(corner_query_bias, torch.Tensor):
                    corner_query_bias_dn = (
                        corner_query_bias[:, None]
                        .expand(b, dn_groups, 16, 4, -1)
                        .reshape(dn_b, 16, 4, -1)
                        .contiguous()
                    )
                else:
                    corner_query_bias_dn = 0.0
            else:
                gt_ref_dn = gt_ref
                gt_wh_dn = gt_wh
                noise_scales = 1.0
                query_base_dn = query_base
                feat16_map_dn = feat16_map
                mem_bt_dn = mem_bt
                tool_memory_dn = tool_memory
                corner_query_bias_dn = corner_query_bias
            dn_step_scale = (self.residual_scale * gt_wh_dn[:, :, None, :]).clamp(0.01, 0.50)
            dn_noise = (
                (torch.rand_like(gt_ref_dn) * 2.0 - 1.0)
                * float(self.dn_query_noise_scale)
                * noise_scales
            )
            dn_ref = (gt_ref_dn + dn_noise * dn_step_scale.detach()).clamp(0.0, 1.0)
            dn_q = (
                query_base_dn[:, :, None, :]
                + self.point_index.to(device=device, dtype=dtype)
                + self.time_pos.to(device=device, dtype=dtype)
                + self.point_ref_mlp(dn_ref)
                + corner_query_bias_dn
            )
            dn_q = self.query_norm(dn_q)
            dn_layer_boxes = []
            dn_layer_xyxy = []
            dn_layer_quads = []
            for i in range(self.decoder_layers):
                dn_q = dn_q + self.local_point_proj(self._sample_point_features(feat16_map_dn, dn_ref))

                dn_q_frame = dn_q.reshape(dn_b * 16, 4, -1)
                dn_frame_out, _ = self.frame_attn[i](
                    self.frame_norms[i](dn_q_frame),
                    self.frame_norms[i](dn_q_frame),
                    self.frame_norms[i](dn_q_frame),
                    need_weights=False,
                )
                dn_q = (dn_q_frame + dn_frame_out).reshape(dn_b, 16, 4, -1)

                dn_q_tem = dn_q.permute(0, 2, 1, 3).reshape(dn_b * 4, 16, -1)
                dn_tem_out, _ = self.temporal_attn[i](
                    self.temporal_norms[i](dn_q_tem),
                    self.temporal_norms[i](dn_q_tem),
                    self.temporal_norms[i](dn_q_tem),
                    need_weights=False,
                )
                dn_q = (dn_q_tem + dn_tem_out).reshape(dn_b, 4, 16, -1).permute(0, 2, 1, 3)

                if tool_memory_dn is not None:
                    dn_q_tool = dn_q.reshape(dn_b, 16 * 4, -1)
                    dn_tool_out, _ = self.tool_traj_attn[i](
                        self.tool_query_norms[i](dn_q_tool),
                        self.tool_memory_norms[i](tool_memory_dn),
                        self.tool_memory_norms[i](tool_memory_dn),
                        need_weights=False,
                    )
                    dn_q = (dn_q_tool + dn_tool_out).reshape(dn_b, 16, 4, -1)

                dn_q_ref = dn_q + self.point_ref_mlp(dn_ref)
                dn_cross_q = self.cross_norms[i](dn_q_ref).reshape(dn_b * 16, 4, -1)
                mem_norm = self.memory_norms[i](mem_bt_dn)
                dn_cross_out, _ = self.cross_attn[i](dn_cross_q, mem_norm, mem_norm, need_weights=False)
                dn_q = dn_q + dn_cross_out.reshape(dn_b, 16, 4, -1)
                dn_q = dn_q + self.ffns[i](self.ffn_norms[i](dn_q))

                dn_delta = self.point_delta_heads[i](dn_q)
                dn_ref_new = (dn_ref + torch.tanh(dn_delta) * dn_step_scale).clamp(0.0, 1.0)
                dn_layer_quads.append(dn_ref_new)
                dn_xyxy = quad_to_xyxy(dn_ref_new, min_size=0.02)
                dn_layer_xyxy.append(dn_xyxy)
                dn_layer_boxes.append(xyxy_to_cxcywh_torch(dn_xyxy))
                dn_ref = dn_ref_new.detach() if i + 1 < self.decoder_layers else dn_ref_new
            out["dn_refiner_quad"] = dn_layer_quads[-1]
            out["dn_refiner_xyxy"] = dn_layer_xyxy[-1]
            out["dn_refiner_cxcywh"] = dn_layer_boxes[-1]
            out["dn_refiner_layer_xyxy"] = torch.stack(dn_layer_xyxy, dim=0)
            out["dn_refiner_layer_cxcywh"] = torch.stack(dn_layer_boxes, dim=0)
            out["dn_refiner_group_count"] = torch.tensor(dn_groups, device=device)
        if (
            self.use_proposal_oracle_refiner
            and dn_quad_xy is not None
            and self.training
            and "proposal_quads" in stage1
        ):
            gt_ref = dn_quad_xy.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            proposal_quads = stage1["proposal_quads"].to(device=device, dtype=dtype).detach().clamp(0.0, 1.0)
            with torch.no_grad():
                gt_prop = gt_ref[:, :, None, :, :].expand_as(proposal_quads)
                prop_quad_iou = quad_iou_approx(
                    proposal_quads.reshape(-1, 4, 2).float(),
                    gt_prop.reshape(-1, 4, 2).float(),
                    grid_size=24,
                ).reshape(proposal_quads.shape[:-2])
                best_idx = prop_quad_iou.argmax(dim=-1)
            gather_idx = best_idx[:, :, None, None, None].expand(-1, -1, 1, 4, 2)
            oracle_ref = proposal_quads.gather(2, gather_idx).squeeze(2).clamp(0.0, 1.0)
            oracle_start = oracle_ref.detach()
            oracle_xyxy = quad_to_xyxy(oracle_ref, min_size=0.02)
            oracle_wh = xyxy_to_cxcywh_torch(oracle_xyxy)[..., 2:].detach()
            oracle_step_scale = (self.residual_scale * oracle_wh[:, :, None, :]).clamp(0.01, 0.50)
            if self.proposal_oracle_refiner_noise_scale > 0.0:
                oracle_noise = (
                    (torch.rand_like(oracle_ref) * 2.0 - 1.0)
                    * float(self.proposal_oracle_refiner_noise_scale)
                )
                oracle_ref = (oracle_ref + oracle_noise * oracle_step_scale.detach()).clamp(0.0, 1.0)

            oracle_q = (
                query_base[:, :, None, :]
                + self.point_index.to(device=device, dtype=dtype)
                + self.time_pos.to(device=device, dtype=dtype)
                + self.point_ref_mlp(oracle_ref)
                + corner_query_bias
            )
            oracle_q = self.query_norm(oracle_q)
            oracle_layer_boxes = []
            oracle_layer_xyxy = []
            oracle_layer_quads = []
            for i in range(self.decoder_layers):
                oracle_q = oracle_q + self.local_point_proj(self._sample_point_features(feat16_map, oracle_ref))

                oracle_q_frame = oracle_q.reshape(b * 16, 4, -1)
                oracle_frame_out, _ = self.frame_attn[i](
                    self.frame_norms[i](oracle_q_frame),
                    self.frame_norms[i](oracle_q_frame),
                    self.frame_norms[i](oracle_q_frame),
                    need_weights=False,
                )
                oracle_q = (oracle_q_frame + oracle_frame_out).reshape(b, 16, 4, -1)

                oracle_q_tem = oracle_q.permute(0, 2, 1, 3).reshape(b * 4, 16, -1)
                oracle_tem_out, _ = self.temporal_attn[i](
                    self.temporal_norms[i](oracle_q_tem),
                    self.temporal_norms[i](oracle_q_tem),
                    self.temporal_norms[i](oracle_q_tem),
                    need_weights=False,
                )
                oracle_q = (oracle_q_tem + oracle_tem_out).reshape(b, 4, 16, -1).permute(0, 2, 1, 3)

                if tool_memory is not None:
                    oracle_q_tool = oracle_q.reshape(b, 16 * 4, -1)
                    oracle_tool_out, _ = self.tool_traj_attn[i](
                        self.tool_query_norms[i](oracle_q_tool),
                        self.tool_memory_norms[i](tool_memory),
                        self.tool_memory_norms[i](tool_memory),
                        need_weights=False,
                    )
                    oracle_q = (oracle_q_tool + oracle_tool_out).reshape(b, 16, 4, -1)

                oracle_q_ref = oracle_q + self.point_ref_mlp(oracle_ref)
                oracle_cross_q = self.cross_norms[i](oracle_q_ref).reshape(b * 16, 4, -1)
                mem_norm = self.memory_norms[i](mem_bt)
                oracle_cross_out, _ = self.cross_attn[i](oracle_cross_q, mem_norm, mem_norm, need_weights=False)
                oracle_q = oracle_q + oracle_cross_out.reshape(b, 16, 4, -1)
                oracle_q = oracle_q + self.ffns[i](self.ffn_norms[i](oracle_q))

                oracle_delta = self.point_delta_heads[i](oracle_q)
                oracle_ref_new = (oracle_ref + torch.tanh(oracle_delta) * oracle_step_scale).clamp(0.0, 1.0)
                oracle_layer_quads.append(oracle_ref_new)
                oracle_ref_xyxy = quad_to_xyxy(oracle_ref_new, min_size=0.02)
                oracle_layer_xyxy.append(oracle_ref_xyxy)
                oracle_layer_boxes.append(xyxy_to_cxcywh_torch(oracle_ref_xyxy))
                oracle_ref = oracle_ref_new.detach() if i + 1 < self.decoder_layers else oracle_ref_new
            out["proposal_oracle_refiner_start_quad"] = oracle_start
            out["proposal_oracle_refiner_quad"] = oracle_layer_quads[-1]
            out["proposal_oracle_refiner_xyxy"] = oracle_layer_xyxy[-1]
            out["proposal_oracle_refiner_cxcywh"] = oracle_layer_boxes[-1]
            out["proposal_oracle_refiner_layer_xyxy"] = torch.stack(oracle_layer_xyxy, dim=0)
            out["proposal_oracle_refiner_layer_cxcywh"] = torch.stack(oracle_layer_boxes, dim=0)
        return out


class H210ProposalOrientedStage1(nn.Module):
    """H151-style proposal/rerank reader for the H161 proxy.

    It generates a small set of spatial proposals from V-JEPA tokens, lets
    candidates interact within each frame and along their own 16-frame temporal
    tracks, then selects a differentiable proposal.  The selected proposal is an
    oriented rectangle so the next point-token stage starts from a structured
    tight-region hypothesis rather than from a single direct quad regression.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        residual_scale: float,
        proposal_count: int,
        proposal_selection_mode: str = "train_soft_eval_hard",
        proposal_selection_temperature: float = 1.0,
        proposal_selection_topk: int = 0,
        proposal_selection_straight_through: bool = False,
        proposal_selection_train_logit_noise_std: float = 0.0,
        proposal_selection_train_drop_prob: float = 0.0,
        proposal_geom_temporal_fusion: bool = False,
        proposal_geom_temporal_fusion_scale: float = 0.30,
        proposal_geom_temporal_center_scale: float = 0.18,
        proposal_geom_temporal_source_penalty: float = 4.0,
        proposal_clip_context_fusion: bool = False,
        proposal_clip_context_scale: float = 0.22,
        proposal_clip_context_dropout: float = 0.10,
        proposal_frame_anchor_memory: bool = False,
        proposal_frame_anchor_scale: float = 0.20,
        proposal_frame_anchor_topk: int = 2,
        proposal_frame_anchor_temperature: float = 0.12,
        proposal_frame_anchor_candidate_temperature: float = 0.45,
        proposal_final_fusion: bool = False,
        proposal_final_fusion_scale: float = 0.18,
        proposal_quality_readout: bool = False,
        proposal_quality_readout_detach: bool = False,
        proposal_quality_readout_replace_logits: bool = False,
        proposal_quality_readout_traj_context: bool = False,
        proposal_quality_readout_points: str = "center_corners",
        proposal_quality_readout_pool: str = "mean",
        proposal_support_overlap_score: bool = False,
        proposal_support_overlap_scale: float = 0.0,
        proposal_support_overlap_grid_size: int = 24,
        proposal_support_overlap_detach: bool = False,
        proposal_support_geom_score: bool = False,
        proposal_support_geom_scale: float = 0.0,
        proposal_support_geom_grid_size: int = 24,
        proposal_support_geom_center_tau: float = 0.07,
        proposal_support_geom_area_tau: float = 0.75,
        proposal_support_geom_detach: bool = False,
        proposal_support_moment_proposal: bool = False,
        proposal_support_moment_top_frac: float = 0.035,
        proposal_support_moment_size_scale: float = 1.8,
        proposal_support_moment_temperature: float = 0.16,
        proposal_support_moment_detach: bool = False,
        proposal_decoupled_selector: bool = False,
        proposal_decoupled_selector_detach: bool = False,
        proposal_decoupled_selector_replace_logits: bool = False,
        proposal_decoupled_selector_scale: float = 1.0,
        proposal_decoupled_selector_base_scale: float = 0.0,
        proposal_decoupled_selector_readout: bool = False,
        proposal_oicr_refine_head: bool = False,
        proposal_oicr_refine_use_for_selection: bool = False,
        proposal_pointset_refine: bool = False,
        proposal_pointset_refine_scale: float = 0.18,
        proposal_pointset_refine_grad_mul: float = 0.25,
        proposal_pointset_quality_scale: float = 1.0,
        proposal_pointset_refine_update_token: bool = False,
        proposal_set_refiner: bool = False,
        proposal_set_refiner_topk: int = 0,
        proposal_set_refiner_scale: float = 0.22,
        proposal_set_refiner_temperature: float = 0.90,
        proposal_set_refiner_detach: bool = False,
        proposal_local_corr_score: bool = False,
        proposal_local_corr_scale: float = 0.45,
        proposal_local_corr_grid_size: int = 3,
        proposal_local_corr_temperature: float = 0.08,
        proposal_local_corr_detach: bool = False,
        proposal_active_tip_selector: bool = False,
        proposal_active_tip_selector_scale: float = 1.0,
        proposal_active_tip_selector_aux_only: bool = False,
        proposal_active_tip_selector_eval_fuse_scale: float = 0.0,
        proposal_active_tip_feature_fusion: bool = False,
        proposal_active_tip_feature_scale: float = 0.35,
        proposal_trajectory_rerank: bool = False,
        proposal_trajectory_rerank_scale: float = 0.45,
        proposal_trajectory_refine: bool = False,
        proposal_trajectory_refine_scale: float = 0.30,
        condition_prior_proposal: bool = False,
        h259_prior_proposal: bool = False,
        trajectory_query_proposal_count: int = 0,
        trajectory_query_output_mode: str = "oriented_rect",
        trajectory_readout_bias: bool = False,
        condition_prior_readout_bias: bool = False,
        condition_prior_readout_scale: float = 0.45,
        trajectory_anchor_logit_bias_scale: float = 0.0,
        trajectory_anchor_logit_bias_radius: float = 0.22,
        trajectory_anchor_axis_bias_weight: float = 0.0,
        corner_heatmap_proposal: bool = False,
        corner_heatmap_proposal_temperature: float = 0.35,
        corner_heatmap_proposal_detach: bool = False,
        proposal_source_calibration: bool = False,
        proposal_source_calibration_scale: float = 1.0,
        proposal_handoff_source_calibration: bool = False,
        proposal_handoff_source_calibration_scale: float = 1.0,
        proposal_handoff_source4_init: float = 0.0,
        proposal_handoff_source_router: bool = False,
        proposal_handoff_source_router_scale: float = 1.0,
        proposal_handoff_source_router_candidate_select_scale: float = 0.0,
        proposal_handoff_source5_readout: bool = False,
        proposal_handoff_source5_readout_scale: float = 1.0,
        proposal_handoff_source5_readout_grid_size: int = 3,
        proposal_handoff_source5_readout_sources: str = "5",
        proposal_handoff_source5_readout_detach: bool = False,
        proposal_handoff_mask_quality: bool = False,
        proposal_handoff_mask_quality_scale: float = 0.50,
        proposal_handoff_mask_grid_size: int = 7,
        proposal_handoff_mask_expand_scale: float = 1.25,
        proposal_handoff_mask_refine_scale: float = 0.0,
        proposal_handoff_mask_refine_size_scale: float = 2.0,
        proposal_handoff_mask_refine_min_size_px: float = 2.0,
        proposal_handoff_mask_refine_max_ratio: float = 1.05,
        proposal_handoff_mask_quality_sources: str = "5,7",
        proposal_handoff_mask_detach: bool = False,
        proposal_handoff_mask_score_detach: bool = False,
        proposal_handoff_rank_adaptive_scorer: bool = False,
        proposal_handoff_rank_adaptive_scale: float = 0.45,
        proposal_handoff_rank_adaptive_max_rank: int = 512,
        proposal_handoff_rank_adaptive_detach: bool = False,
        proposal_handoff_sparse_set_scorer: bool = False,
        proposal_handoff_sparse_set_scorer_scale: float = 0.35,
        proposal_handoff_sparse_set_scorer_sources: str = "0,5,7",
        proposal_handoff_sparse_set_scorer_detach: bool = False,
        proposal_handoff_source5_refiner: bool = False,
        proposal_handoff_source5_refiner_topk: int = 12,
        proposal_handoff_source5_refiner_temperature: float = 0.75,
        proposal_handoff_source5_refiner_scale: float = 0.35,
        proposal_handoff_source5_refiner_logit_scale: float = 0.65,
        proposal_handoff_source5_refiner_init_logit_bias: float = 0.0,
        proposal_handoff_source5_refiner_detach: bool = False,
        proposal_handoff_source5_refiner_source_id: int = 5,
        proposal_handoff_source5_refiner_output_source_id: int = 6,
        proposal_handoff_topk_query_refiner: bool = False,
        proposal_handoff_topk_query_sources: str = "0,5",
        proposal_handoff_topk_query_topk_per_source: int = 12,
        proposal_handoff_topk_query_grid_size: int = 3,
        proposal_handoff_topk_query_scale: float = 0.30,
        proposal_handoff_topk_query_logit_scale: float = 0.75,
        proposal_handoff_topk_query_output_source_id: int = 6,
        proposal_handoff_topk_query_replace_logits: bool = False,
        proposal_handoff_topk_query_detach: bool = False,
        proposal_handoff_final_fusion: bool = False,
        proposal_handoff_final_fusion_scale: float = 0.18,
        proposal_handoff_detach_selection_geometry: bool = False,
        proposal_handoff_source_representatives_only: bool = False,
        proposal_handoff_source5_topk: int = 0,
        proposal_handoff_router_gated_source: bool = False,
        proposal_handoff_router_normalized_select: bool = False,
        proposal_handoff_router_normalized_source_scale: float = 1.0,
        proposal_handoff_router_normalized_within_scale: float = 1.0,
        token_affinity_proposal: bool = False,
        token_affinity_proposal_count: int = 1,
        token_affinity_proposal_temperature: float = 0.08,
        token_affinity_proposal_top_frac: float = 0.055,
        token_affinity_proposal_size_scale: float = 3.6,
        token_affinity_proposal_extra_top_fracs: str = "",
        token_affinity_proposal_extra_size_scales: str = "",
        token_affinity_proposal_detach: bool = True,
        topology_contact_template_proposal: bool = False,
        topology_contact_template_sizes_px: str = "4x3,6x4,8x4,10x5",
        topology_contact_template_offsets_px: str = "-2,0,2",
        topology_contact_template_logit_bias: float = 0.0,
        topology_contact_template_index_embedding: bool = False,
        topology_contact_template_combo_bias: bool = False,
        topology_contact_template_combo_bias_scale: float = 1.0,
        topology_contact_template_combo_allowlist: str = "",
        local_grid_refine: bool = False,
        local_grid_refine_grid_size: int = 5,
        local_grid_refine_scale: float = 1.35,
        local_grid_refine_temperature: float = 0.20,
        local_grid_refine_prior_scale: float = 6.0,
        local_grid_refine_prior_sigma: float = 0.28,
        local_grid_refine_detach_base: bool = False,
        local_grid_final_refine: bool = True,
        local_grid_confidence_score: bool = False,
        local_grid_confidence_scale: float = 0.35,
        local_grid_confidence_detach: bool = False,
        candidate_mask_quality: bool = False,
        candidate_mask_grid_size: int = 7,
        candidate_mask_score_scale: float = 0.0,
        candidate_mask_score_detach: bool = False,
        use_rgb_sidecar: bool = False,
    ) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.use_rgb_sidecar = bool(use_rgb_sidecar)
        self.condition_prior_proposal = bool(condition_prior_proposal)
        self.h259_prior_proposal = bool(h259_prior_proposal) and self.use_tool_keypoints
        self.residual_scale = float(residual_scale)
        self.proposal_count = max(int(proposal_count), 2)
        self.proposal_selection_mode = str(proposal_selection_mode)
        self.proposal_selection_temperature = max(float(proposal_selection_temperature), 1e-3)
        self.proposal_selection_topk = max(int(proposal_selection_topk), 0)
        self.proposal_selection_straight_through = bool(proposal_selection_straight_through)
        self.proposal_selection_train_logit_noise_std = max(
            float(proposal_selection_train_logit_noise_std),
            0.0,
        )
        self.proposal_selection_train_drop_prob = float(
            np.clip(proposal_selection_train_drop_prob, 0.0, 0.95)
        )
        self.proposal_geom_temporal_fusion = bool(proposal_geom_temporal_fusion)
        self.proposal_geom_temporal_fusion_scale = max(float(proposal_geom_temporal_fusion_scale), 0.0)
        self.proposal_geom_temporal_center_scale = max(float(proposal_geom_temporal_center_scale), 1e-3)
        self.proposal_geom_temporal_source_penalty = max(float(proposal_geom_temporal_source_penalty), 0.0)
        self.proposal_clip_context_fusion = bool(proposal_clip_context_fusion)
        self.proposal_clip_context_scale = max(float(proposal_clip_context_scale), 0.0)
        self.proposal_clip_context_dropout = float(np.clip(proposal_clip_context_dropout, 0.0, 0.5))
        self.proposal_frame_anchor_memory = bool(proposal_frame_anchor_memory)
        self.proposal_frame_anchor_scale = max(float(proposal_frame_anchor_scale), 0.0)
        self.proposal_frame_anchor_topk = max(int(proposal_frame_anchor_topk), 0)
        self.proposal_frame_anchor_temperature = max(float(proposal_frame_anchor_temperature), 1e-3)
        self.proposal_frame_anchor_candidate_temperature = max(
            float(proposal_frame_anchor_candidate_temperature),
            1e-3,
        )
        self.proposal_final_fusion = bool(proposal_final_fusion)
        self.proposal_final_fusion_scale = max(float(proposal_final_fusion_scale), 0.0)
        self.proposal_quality_readout = bool(proposal_quality_readout)
        self.proposal_quality_readout_detach = bool(proposal_quality_readout_detach)
        self.proposal_quality_readout_replace_logits = bool(proposal_quality_readout_replace_logits)
        self.proposal_quality_readout_traj_context = (
            bool(proposal_quality_readout_traj_context) and self.use_tool_keypoints
        )
        if proposal_quality_readout_pool not in {"mean", "attention"}:
            proposal_quality_readout_pool = "mean"
        self.proposal_quality_readout_pool = str(proposal_quality_readout_pool)
        if proposal_quality_readout_points not in {
            "center_corners",
            "center_corners_edges",
            "aligned_grid_3x3",
            "aligned_grid_5x5",
        }:
            proposal_quality_readout_points = "center_corners"
        self.proposal_quality_readout_points = str(proposal_quality_readout_points)
        self.proposal_support_overlap_score = bool(proposal_support_overlap_score)
        self.proposal_support_overlap_scale = float(proposal_support_overlap_scale)
        self.proposal_support_overlap_grid_size = max(int(proposal_support_overlap_grid_size), 8)
        self.proposal_support_overlap_detach = bool(proposal_support_overlap_detach)
        self.proposal_support_geom_score = bool(proposal_support_geom_score)
        self.proposal_support_geom_scale = float(proposal_support_geom_scale)
        self.proposal_support_geom_grid_size = max(int(proposal_support_geom_grid_size), 8)
        self.proposal_support_geom_center_tau = max(float(proposal_support_geom_center_tau), 1e-4)
        self.proposal_support_geom_area_tau = max(float(proposal_support_geom_area_tau), 1e-4)
        self.proposal_support_geom_detach = bool(proposal_support_geom_detach)
        self.proposal_support_moment_proposal = bool(proposal_support_moment_proposal)
        self.proposal_support_moment_top_frac = float(np.clip(proposal_support_moment_top_frac, 0.01, 0.35))
        self.proposal_support_moment_size_scale = max(float(proposal_support_moment_size_scale), 0.25)
        self.proposal_support_moment_temperature = max(float(proposal_support_moment_temperature), 1e-3)
        self.proposal_support_moment_detach = bool(proposal_support_moment_detach)
        self.proposal_decoupled_selector = bool(proposal_decoupled_selector)
        self.proposal_decoupled_selector_detach = bool(proposal_decoupled_selector_detach)
        self.proposal_decoupled_selector_replace_logits = bool(
            proposal_decoupled_selector_replace_logits
        )
        self.proposal_decoupled_selector_scale = float(proposal_decoupled_selector_scale)
        self.proposal_decoupled_selector_base_scale = float(proposal_decoupled_selector_base_scale)
        self.proposal_decoupled_selector_readout = (
            bool(proposal_decoupled_selector_readout) and self.proposal_decoupled_selector
        )
        self.proposal_oicr_refine_head_enabled = bool(proposal_oicr_refine_head)
        self.proposal_oicr_refine_use_for_selection = bool(proposal_oicr_refine_use_for_selection)
        self.proposal_pointset_refine = bool(proposal_pointset_refine)
        self.proposal_pointset_refine_scale = float(proposal_pointset_refine_scale)
        self.proposal_pointset_refine_grad_mul = float(np.clip(proposal_pointset_refine_grad_mul, 0.0, 1.0))
        self.proposal_pointset_quality_scale = float(proposal_pointset_quality_scale)
        self.proposal_pointset_refine_update_token = bool(proposal_pointset_refine_update_token)
        self.proposal_set_refiner = bool(proposal_set_refiner)
        self.proposal_set_refiner_topk = max(int(proposal_set_refiner_topk), 0)
        self.proposal_set_refiner_scale = max(float(proposal_set_refiner_scale), 0.0)
        self.proposal_set_refiner_temperature = max(float(proposal_set_refiner_temperature), 1e-3)
        self.proposal_set_refiner_detach = bool(proposal_set_refiner_detach)
        self.proposal_local_corr_score = bool(proposal_local_corr_score)
        self.proposal_active_tip_selector = bool(proposal_active_tip_selector) and self.use_tool_keypoints
        self.proposal_active_tip_selector_scale = float(proposal_active_tip_selector_scale)
        self.proposal_active_tip_selector_aux_only = bool(proposal_active_tip_selector_aux_only)
        self.proposal_active_tip_selector_eval_fuse_scale = float(proposal_active_tip_selector_eval_fuse_scale)
        self.proposal_active_tip_feature_fusion = (
            bool(proposal_active_tip_feature_fusion) and self.use_tool_keypoints
        )
        self.proposal_active_tip_feature_scale = float(proposal_active_tip_feature_scale)
        self.proposal_trajectory_rerank = bool(proposal_trajectory_rerank) and self.use_tool_keypoints
        self.proposal_trajectory_rerank_scale = float(proposal_trajectory_rerank_scale)
        self.proposal_trajectory_refine = bool(proposal_trajectory_refine) and self.use_tool_keypoints
        self.proposal_trajectory_refine_scale = float(proposal_trajectory_refine_scale)
        self.trajectory_query_proposal_count = (
            max(int(trajectory_query_proposal_count), 0) if self.use_tool_keypoints else 0
        )
        self.trajectory_query_output_mode = str(trajectory_query_output_mode)
        if self.trajectory_query_output_mode not in {"oriented_rect", "direct_quad"}:
            raise ValueError(f"Unsupported trajectory_query_output_mode={self.trajectory_query_output_mode!r}")
        self.trajectory_readout_bias = bool(trajectory_readout_bias) and self.use_tool_keypoints
        self.condition_prior_readout_bias = bool(condition_prior_readout_bias)
        self.condition_prior_readout_scale = float(condition_prior_readout_scale)
        self.condition_prior_enabled = self.condition_prior_proposal or self.condition_prior_readout_bias
        self.trajectory_anchor_logit_bias_scale = (
            float(trajectory_anchor_logit_bias_scale) if self.use_tool_keypoints else 0.0
        )
        self.trajectory_anchor_logit_bias_radius = max(float(trajectory_anchor_logit_bias_radius), 1e-3)
        self.trajectory_anchor_axis_bias_weight = float(trajectory_anchor_axis_bias_weight)
        self.corner_heatmap_proposal = bool(corner_heatmap_proposal)
        self.corner_heatmap_proposal_temperature = max(float(corner_heatmap_proposal_temperature), 1e-3)
        self.corner_heatmap_proposal_detach = bool(corner_heatmap_proposal_detach)
        self.proposal_source_calibration = bool(proposal_source_calibration)
        self.proposal_source_calibration_scale = float(proposal_source_calibration_scale)
        self.proposal_handoff_source_router = bool(proposal_handoff_source_router)
        self.proposal_handoff_source_router_scale = float(proposal_handoff_source_router_scale)
        self.proposal_handoff_source_router_candidate_select_scale = float(
            proposal_handoff_source_router_candidate_select_scale
        )
        self.proposal_handoff_router_normalized_select = bool(
            proposal_handoff_router_normalized_select
        )
        self.proposal_handoff_router_normalized_source_scale = float(
            proposal_handoff_router_normalized_source_scale
        )
        self.proposal_handoff_router_normalized_within_scale = float(
            proposal_handoff_router_normalized_within_scale
        )
        self.token_affinity_proposal = bool(token_affinity_proposal)
        self.token_affinity_proposal_count = max(int(token_affinity_proposal_count), 1)
        self.token_affinity_proposal_temperature = max(float(token_affinity_proposal_temperature), 1e-3)
        self.token_affinity_proposal_top_frac = float(np.clip(token_affinity_proposal_top_frac, 0.01, 0.35))
        self.token_affinity_proposal_size_scale = max(float(token_affinity_proposal_size_scale), 0.25)
        extra_top_fracs = parse_float_csv(token_affinity_proposal_extra_top_fracs)
        extra_size_scales = parse_float_csv(token_affinity_proposal_extra_size_scales)
        variant_n = max(len(extra_top_fracs), len(extra_size_scales))
        self.token_affinity_proposal_variants = [
            (self.token_affinity_proposal_top_frac, self.token_affinity_proposal_size_scale)
        ]
        for i in range(variant_n):
            top_frac = extra_top_fracs[i] if i < len(extra_top_fracs) else self.token_affinity_proposal_top_frac
            size_scale = extra_size_scales[i] if i < len(extra_size_scales) else self.token_affinity_proposal_size_scale
            self.token_affinity_proposal_variants.append(
                (
                    float(np.clip(top_frac, 0.01, 0.35)),
                    max(float(size_scale), 0.25),
                )
            )
        self.token_affinity_proposal_detach = bool(token_affinity_proposal_detach)
        self.topology_contact_template_proposal = bool(topology_contact_template_proposal) and self.use_tool_keypoints
        self.topology_contact_template_sizes = parse_template_sizes_px(
            topology_contact_template_sizes_px,
            mask_res=64,
        )
        self.topology_contact_template_offsets = [
            float(v) / 64.0 for v in parse_float_csv(topology_contact_template_offsets_px)
        ]
        if not self.topology_contact_template_sizes:
            self.topology_contact_template_sizes = parse_template_sizes_px("4x3,6x4,8x4,10x5", mask_res=64)
        if not self.topology_contact_template_offsets:
            self.topology_contact_template_offsets = [-2.0 / 64.0, 0.0, 2.0 / 64.0]
        self.topology_contact_template_count = topology_contact_template_count(
            ",".join(f"{int(round(w * 64))}x{int(round(hh * 64))}" for w, hh in self.topology_contact_template_sizes),
            ",".join(str(v * 64.0) for v in self.topology_contact_template_offsets),
            mask_res=64,
        )
        self.topology_contact_template_logit_bias = float(topology_contact_template_logit_bias)
        self.topology_contact_template_index_embedding = (
            bool(topology_contact_template_index_embedding) and self.topology_contact_template_proposal
        )
        self.topology_contact_template_combo_bias = (
            bool(topology_contact_template_combo_bias) and self.topology_contact_template_proposal
        )
        self.topology_contact_template_combo_bias_scale = float(topology_contact_template_combo_bias_scale)
        allow_mask = parse_combo_index_allowlist(
            topology_contact_template_combo_allowlist,
            int(self.topology_contact_template_count),
            num_combos=17,
        )
        if allow_mask is not None and self.topology_contact_template_proposal:
            self.register_buffer(
                "topology_contact_template_allow_mask",
                allow_mask,
                persistent=False,
            )
        else:
            self.topology_contact_template_allow_mask = None

        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        self.rgb_encoder = make_rgb_sidecar_encoder(c) if self.use_rgb_sidecar else None
        in_ch = (
            c
            + (c if self.use_rgb_sidecar else 0)
            + 2
            + (1 if self.use_tool_mask else 0)
            + (3 if self.use_tool_keypoints else 0)
        )
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.center_head = nn.Conv2d(c, self.proposal_count, 1)
        self.corner_heatmap_head = nn.Conv2d(c, 4, 1)
        self.candidate_embed = nn.Parameter(torch.zeros(1, 1, self.proposal_count, c))
        self.box_ref_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.candidate_in = nn.Sequential(nn.LayerNorm(3 * c + 4), nn.Linear(3 * c + 4, c), nn.SiLU())
        self.frame_norm = nn.LayerNorm(c)
        self.frame_attn = nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
        self.temporal_norm = nn.LayerNorm(c)
        self.temporal_attn = nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
        self.ffn_norm = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Dropout(0.10), nn.Linear(4 * c, c))
        self.delta_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 6))
        self.size_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.quality_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 1))
        if self.proposal_oicr_refine_head_enabled:
            self.proposal_oicr_refine_head = nn.Sequential(
                nn.LayerNorm(c + 8 + 1),
                nn.Linear(c + 8 + 1, c),
                nn.SiLU(),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.proposal_oicr_refine_head[-1].weight)
            nn.init.zeros_(self.proposal_oicr_refine_head[-1].bias)
        if self.proposal_final_fusion:
            self.proposal_final_fusion_candidate_mlp = nn.Sequential(
                nn.LayerNorm(c + 8 + 2),
                nn.Linear(c + 8 + 2, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_final_fusion_head = nn.Sequential(
                nn.LayerNorm(c + 8 + 4),
                nn.Linear(c + 8 + 4, c),
                nn.SiLU(),
                nn.Linear(c, 8),
            )
            nn.init.zeros_(self.proposal_final_fusion_head[-1].weight)
            nn.init.zeros_(self.proposal_final_fusion_head[-1].bias)
        if self.proposal_source_calibration:
            self.proposal_source_logit_bias = nn.Embedding(8, 1)
            nn.init.zeros_(self.proposal_source_logit_bias.weight)
        if self.proposal_handoff_source_router:
            self.proposal_handoff_source_router_candidate_head = nn.Sequential(
                nn.LayerNorm(c + 8 + 1),
                nn.Linear(c + 8 + 1, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            self.proposal_handoff_source_router_context_head = nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 8),
            )
            nn.init.zeros_(self.proposal_handoff_source_router_candidate_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source_router_candidate_head[-1].bias)
            nn.init.zeros_(self.proposal_handoff_source_router_context_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source_router_context_head[-1].bias)
        if self.proposal_geom_temporal_fusion:
            self.proposal_geom_temporal_value = nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
            )
            self.proposal_geom_temporal_token_proj = nn.Sequential(
                nn.LayerNorm(2 * c + 8),
                nn.Linear(2 * c + 8, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_geom_temporal_logit_head = nn.Sequential(
                nn.LayerNorm(2 * c + 8 + 1),
                nn.Linear(2 * c + 8 + 1, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.proposal_geom_temporal_token_proj[-1].weight)
            nn.init.zeros_(self.proposal_geom_temporal_token_proj[-1].bias)
            nn.init.zeros_(self.proposal_geom_temporal_logit_head[-1].weight)
            nn.init.zeros_(self.proposal_geom_temporal_logit_head[-1].bias)
        if self.proposal_clip_context_fusion:
            self.proposal_clip_time_embed = nn.Parameter(torch.zeros(1, 16, 1, c))
            self.proposal_clip_source_embed = nn.Embedding(8, c)
            self.proposal_clip_geom_mlp = nn.Sequential(
                nn.LayerNorm(8),
                nn.Linear(8, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_clip_context_norm = nn.LayerNorm(c)
            self.proposal_clip_context_attn = nn.MultiheadAttention(
                c,
                num_heads=4,
                dropout=self.proposal_clip_context_dropout,
                batch_first=True,
            )
            self.proposal_clip_context_ffn_norm = nn.LayerNorm(c)
            self.proposal_clip_context_ffn = nn.Sequential(
                nn.Linear(c, 4 * c),
                nn.GELU(),
                nn.Dropout(self.proposal_clip_context_dropout),
                nn.Linear(4 * c, c),
            )
            self.proposal_clip_context_token_proj = nn.Sequential(
                nn.LayerNorm(2 * c + 8),
                nn.Linear(2 * c + 8, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_clip_context_logit_head = nn.Sequential(
                nn.LayerNorm(2 * c + 8 + 1),
                nn.Linear(2 * c + 8 + 1, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.trunc_normal_(self.proposal_clip_time_embed, std=0.02)
            nn.init.normal_(self.proposal_clip_source_embed.weight, std=0.02)
            nn.init.zeros_(self.proposal_clip_context_token_proj[-1].weight)
            nn.init.zeros_(self.proposal_clip_context_token_proj[-1].bias)
            nn.init.zeros_(self.proposal_clip_context_logit_head[-1].weight)
            nn.init.zeros_(self.proposal_clip_context_logit_head[-1].bias)
        if self.proposal_frame_anchor_memory:
            self.proposal_frame_anchor_frame_mlp = nn.Sequential(
                nn.LayerNorm(3 * c + 14),
                nn.Linear(3 * c + 14, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_frame_anchor_score_head = nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            self.proposal_frame_anchor_token_proj = nn.Sequential(
                nn.LayerNorm(2 * c + 8),
                nn.Linear(2 * c + 8, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_frame_anchor_logit_head = nn.Sequential(
                nn.LayerNorm(2 * c + 8 + 1),
                nn.Linear(2 * c + 8 + 1, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.proposal_frame_anchor_score_head[-1].weight)
            nn.init.zeros_(self.proposal_frame_anchor_score_head[-1].bias)
            nn.init.zeros_(self.proposal_frame_anchor_token_proj[-1].weight)
            nn.init.zeros_(self.proposal_frame_anchor_token_proj[-1].bias)
            nn.init.zeros_(self.proposal_frame_anchor_logit_head[-1].weight)
            nn.init.zeros_(self.proposal_frame_anchor_logit_head[-1].bias)
        if self.token_affinity_proposal:
            self.token_affinity_proposal_token_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.token_affinity_proposal_quality_head = nn.Sequential(
                nn.LayerNorm(c + 8),
                nn.Linear(c + 8, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.token_affinity_proposal_quality_head[-1].weight)
            nn.init.zeros_(self.token_affinity_proposal_quality_head[-1].bias)
        if self.topology_contact_template_proposal:
            topo_count = max(int(self.topology_contact_template_count), 1)
            self.topology_contact_template_token_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.topology_contact_template_quality_head = nn.Sequential(
                nn.LayerNorm(c + 8),
                nn.Linear(c + 8, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.topology_contact_template_quality_head[-1].weight)
            nn.init.zeros_(self.topology_contact_template_quality_head[-1].bias)
            if self.topology_contact_template_index_embedding:
                self.topology_contact_template_index_embed = nn.Embedding(topo_count, c)
            if self.topology_contact_template_combo_bias:
                self.topology_contact_template_combo_bias_table = nn.Embedding(17, topo_count)
                nn.init.zeros_(self.topology_contact_template_combo_bias_table.weight)
        if self.proposal_support_moment_proposal:
            self.support_moment_proposal_token_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.support_moment_proposal_quality_head = nn.Sequential(
                nn.LayerNorm(c + 8),
                nn.Linear(c + 8, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.support_moment_proposal_quality_head[-1].weight)
            nn.init.zeros_(self.support_moment_proposal_quality_head[-1].bias)
        if self.corner_heatmap_proposal:
            self.corner_proposal_token_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.corner_proposal_quality_head = nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.corner_proposal_quality_head[-1].weight)
            nn.init.zeros_(self.corner_proposal_quality_head[-1].bias)
        if self.proposal_quality_readout:
            self.proposal_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            if self.proposal_quality_readout_pool == "attention":
                self.proposal_quality_readout_attn = nn.Sequential(
                    nn.LayerNorm(2 * c + 8),
                    nn.Linear(2 * c + 8, c),
                    nn.SiLU(),
                    nn.Linear(c, 1),
                )
            if self.proposal_quality_readout_traj_context:
                self.proposal_quality_traj_context_mlp = nn.Sequential(
                    nn.LayerNorm(13),
                    nn.Linear(13, c),
                    nn.SiLU(),
                    nn.Linear(c, c),
                )
            quality_in_dim = (3 * c + 8) if self.proposal_quality_readout_traj_context else (2 * c + 8)
            self.proposal_quality_readout_head = nn.Sequential(
                nn.LayerNorm(quality_in_dim),
                nn.Linear(quality_in_dim, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.proposal_quality_readout_head[-1].weight)
            nn.init.zeros_(self.proposal_quality_readout_head[-1].bias)
        if self.proposal_decoupled_selector_readout:
            self.proposal_decoupled_selector_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
        if self.proposal_decoupled_selector:
            self.proposal_decoupled_selector_source_emb = nn.Embedding(8, c)
            nn.init.normal_(self.proposal_decoupled_selector_source_emb.weight, std=0.02)
            selector_in_dim = 2 * c + 8 + 1
            if self.proposal_decoupled_selector_readout:
                selector_in_dim += c
            self.proposal_decoupled_selector_head = nn.Sequential(
                nn.LayerNorm(selector_in_dim),
                nn.Linear(selector_in_dim, c),
                nn.SiLU(),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.proposal_decoupled_selector_head[-1].weight)
            nn.init.zeros_(self.proposal_decoupled_selector_head[-1].bias)
        if self.proposal_set_refiner:
            self.proposal_set_refiner_source_emb = nn.Embedding(8, c)
            nn.init.normal_(self.proposal_set_refiner_source_emb.weight, std=0.02)
            self.proposal_set_refiner_item_mlp = nn.Sequential(
                nn.LayerNorm(2 * c + 8 + 1),
                nn.Linear(2 * c + 8 + 1, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_set_refiner_score_head = nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            self.proposal_set_refiner_delta_head = nn.Sequential(
                nn.LayerNorm(c + 8 + 4),
                nn.Linear(c + 8 + 4, c),
                nn.SiLU(),
                nn.Linear(c, 8),
            )
            nn.init.zeros_(self.proposal_set_refiner_score_head[-1].weight)
            nn.init.zeros_(self.proposal_set_refiner_score_head[-1].bias)
            nn.init.zeros_(self.proposal_set_refiner_delta_head[-1].weight)
            nn.init.zeros_(self.proposal_set_refiner_delta_head[-1].bias)
        if self.proposal_pointset_refine:
            self.proposal_pointset_corner_embed = nn.Parameter(torch.zeros(1, 1, 1, 4, c))
            self.proposal_pointset_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_pointset_delta_head = nn.Sequential(
                nn.LayerNorm(2 * c + 4),
                nn.Linear(2 * c + 4, c),
                nn.SiLU(),
                nn.Linear(c, 2),
            )
            self.proposal_pointset_quality_head = nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            if self.proposal_pointset_refine_update_token:
                self.proposal_pointset_token_proj = nn.Sequential(
                    nn.LayerNorm(c),
                    nn.Linear(c, c),
                    nn.SiLU(),
                    nn.Linear(c, c),
                )
                nn.init.zeros_(self.proposal_pointset_token_proj[-1].weight)
                nn.init.zeros_(self.proposal_pointset_token_proj[-1].bias)
            nn.init.trunc_normal_(self.proposal_pointset_corner_embed, std=0.02)
            nn.init.zeros_(self.proposal_pointset_delta_head[-1].weight)
            nn.init.zeros_(self.proposal_pointset_delta_head[-1].bias)
            nn.init.zeros_(self.proposal_pointset_quality_head[-1].weight)
            nn.init.zeros_(self.proposal_pointset_quality_head[-1].bias)
        active_tip_logits = None
        if self.proposal_active_tip_selector:
            # Candidate-level geometry features: current tip/path distances,
            # tool/motion alignment, frame role, proposal shape, and visibility.
            self.proposal_active_tip_selector_head = nn.Sequential(
                nn.LayerNorm(12),
                nn.Linear(12, c),
                nn.SiLU(),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.proposal_active_tip_selector_head[-1].weight)
            nn.init.zeros_(self.proposal_active_tip_selector_head[-1].bias)
        if self.proposal_active_tip_feature_fusion:
            self.proposal_active_tip_feature_mlp = nn.Sequential(
                nn.LayerNorm(12),
                nn.Linear(12, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.proposal_active_tip_feature_gate = nn.Sequential(
                nn.LayerNorm(2 * c + 8),
                nn.Linear(2 * c + 8, c),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.proposal_active_tip_feature_mlp[-1].weight)
            nn.init.zeros_(self.proposal_active_tip_feature_mlp[-1].bias)
        if self.proposal_trajectory_rerank or self.proposal_trajectory_refine:
            self.proposal_traj_mlp = nn.Sequential(nn.LayerNorm(13), nn.Linear(13, c), nn.SiLU(), nn.Linear(c, c))
            self.proposal_traj_time_pos = nn.Parameter(torch.zeros(1, 16, c))
            self.proposal_traj_query_norm = nn.LayerNorm(c)
            self.proposal_traj_memory_norm = nn.LayerNorm(c)
            self.proposal_traj_attn = nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            if self.proposal_trajectory_refine:
                self.proposal_traj_refine_head = nn.Sequential(
                    nn.LayerNorm(2 * c + 8),
                    nn.Linear(2 * c + 8, c),
                    nn.SiLU(),
                    nn.Linear(c, 6),
                )
                nn.init.zeros_(self.proposal_traj_refine_head[-1].weight)
                nn.init.zeros_(self.proposal_traj_refine_head[-1].bias)
            if self.proposal_trajectory_rerank:
                self.proposal_traj_head = nn.Sequential(
                    nn.LayerNorm(2 * c + 8),
                    nn.Linear(2 * c + 8, c),
                    nn.SiLU(),
                    nn.Linear(c, 1),
                )
                nn.init.zeros_(self.proposal_traj_head[-1].weight)
                nn.init.zeros_(self.proposal_traj_head[-1].bias)
            nn.init.trunc_normal_(self.proposal_traj_time_pos, std=0.02)
        if self.condition_prior_enabled:
            self.condition_prior_mlp = nn.Sequential(nn.LayerNorm(13), nn.Linear(13, c), nn.SiLU(), nn.Linear(c, c))
            self.condition_prior_time_pos = nn.Parameter(torch.zeros(1, 16, c))
            self.condition_prior_attn = nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            self.condition_prior_norm = nn.LayerNorm(c)
            self.condition_prior_ffn_norm = nn.LayerNorm(c)
            self.condition_prior_ffn = nn.Sequential(
                nn.Linear(c, 4 * c),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(4 * c, c),
            )
            self.condition_prior_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 6))
            self.condition_prior_quality_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 1))
            nn.init.trunc_normal_(self.condition_prior_time_pos, std=0.02)
            nn.init.zeros_(self.condition_prior_head[-1].weight)
            nn.init.zeros_(self.condition_prior_head[-1].bias)
            with torch.no_grad():
                self.condition_prior_head[-1].bias[2:4].fill_(-2.1)
        if self.h259_prior_proposal:
            self.h259_prior = H259TrajectoryPriorTemporalRegion(
                hidden_dim=c,
                residual_scale=min(max(float(residual_scale), 0.10), 0.50),
            )
            self.h259_prior_token_mlp = nn.Sequential(
                nn.LayerNorm(21),
                nn.Linear(21, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.h259_prior_quality_head = nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
            )
            nn.init.zeros_(self.h259_prior_quality_head[-1].weight)
            nn.init.zeros_(self.h259_prior_quality_head[-1].bias)
        if self.trajectory_query_proposal_count > 0:
            tq = self.trajectory_query_proposal_count
            traj_out_dim = 8 if self.trajectory_query_output_mode == "direct_quad" else 6
            self.traj_query_mlp = nn.Sequential(nn.LayerNorm(13), nn.Linear(13, c), nn.SiLU(), nn.Linear(c, c))
            self.traj_query_time_pos = nn.Parameter(torch.zeros(1, 16, c))
            self.traj_query_embed = nn.Parameter(torch.zeros(1, 1, tq, c))
            self.traj_query_norm = nn.LayerNorm(c)
            self.traj_memory_norm = nn.LayerNorm(c)
            self.traj_cross_attn = nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            self.traj_frame_norm = nn.LayerNorm(c)
            self.traj_frame_attn = nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            self.traj_temporal_norm = nn.LayerNorm(c)
            self.traj_temporal_attn = nn.MultiheadAttention(c, num_heads=4, dropout=0.10, batch_first=True)
            self.traj_query_ffn_norm = nn.LayerNorm(c)
            self.traj_query_ffn = nn.Sequential(
                nn.Linear(c, 4 * c),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(4 * c, c),
            )
            self.traj_query_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, traj_out_dim))
            self.traj_query_quality_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 1))
            nn.init.trunc_normal_(self.traj_query_time_pos, std=0.02)
            nn.init.trunc_normal_(self.traj_query_embed, std=0.02)
            nn.init.zeros_(self.traj_query_head[-1].weight)
            nn.init.zeros_(self.traj_query_head[-1].bias)
            nn.init.zeros_(self.traj_query_quality_head[-1].weight)
            nn.init.zeros_(self.traj_query_quality_head[-1].bias)
            if self.trajectory_query_output_mode == "oriented_rect":
                with torch.no_grad():
                    self.traj_query_head[-1].bias[2:4].fill_(-2.1)
            else:
                with torch.no_grad():
                    self.traj_query_quality_head[-1].bias.fill_(-1.25)
        if (
            self.trajectory_readout_bias
            or self.condition_prior_readout_bias
            or self.proposal_quality_readout
            or self.proposal_pointset_refine
            or self.proposal_local_corr_score
            or self.corner_heatmap_proposal
            or self.token_affinity_proposal
            or self.topology_contact_template_proposal
        ):
            self.traj_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * c),
                nn.Linear(3 * c, c),
                nn.SiLU(),
                nn.Linear(c, c),
            )
            self.traj_readout_gate = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.Sigmoid())
            sample_offsets = torch.tensor(
                [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
                dtype=torch.float32,
            )
            self.register_buffer("traj_readout_sample_offsets", sample_offsets, persistent=False)
            self.traj_readout_sample_radius = 1.25 / 24.0
        nn.init.trunc_normal_(self.candidate_embed, std=0.02)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.size_head[-1].weight)
        nn.init.constant_(self.size_head[-1].bias, -2.1)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def _proposal_geom_temporal_context(
        self,
        candidate_tokens: torch.Tensor,
        candidate_quads: torch.Tensor,
        proposal_source_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Adjacent-frame proposal memory matched by geometry, not candidate index."""
        b, t, _k, _c = candidate_tokens.shape
        if t < 2:
            return torch.zeros_like(candidate_tokens)
        value = self.proposal_geom_temporal_value(candidate_tokens)
        centers = candidate_quads.detach().mean(dim=-2)
        sources = proposal_source_ids.detach()
        ctx = torch.zeros_like(candidate_tokens)
        count = candidate_tokens.new_zeros((b, t, _k, 1))
        center_scale_sq = self.proposal_geom_temporal_center_scale ** 2
        source_penalty = self.proposal_geom_temporal_source_penalty

        cur_centers = centers[:, 1:]
        prev_centers = centers[:, :-1]
        dist2 = ((cur_centers[:, :, :, None, :] - prev_centers[:, :, None, :, :]) ** 2).sum(dim=-1)
        logits = -dist2 / center_scale_sq
        if source_penalty != 0.0:
            source_mismatch = sources[:, 1:, :, None] != sources[:, :-1, None, :]
            logits = logits - source_penalty * source_mismatch.to(dtype=logits.dtype)
        attn = F.softmax(logits, dim=-1)
        ctx[:, 1:] = ctx[:, 1:] + torch.einsum("btkj,btjc->btkc", attn, value[:, :-1])
        count[:, 1:] = count[:, 1:] + 1.0

        cur_centers = centers[:, :-1]
        next_centers = centers[:, 1:]
        dist2 = ((cur_centers[:, :, :, None, :] - next_centers[:, :, None, :, :]) ** 2).sum(dim=-1)
        logits = -dist2 / center_scale_sq
        if source_penalty != 0.0:
            source_mismatch = sources[:, :-1, :, None] != sources[:, 1:, None, :]
            logits = logits - source_penalty * source_mismatch.to(dtype=logits.dtype)
        attn = F.softmax(logits, dim=-1)
        ctx[:, :-1] = ctx[:, :-1] + torch.einsum("btkj,btjc->btkc", attn, value[:, 1:])
        count[:, :-1] = count[:, :-1] + 1.0
        return ctx / count.clamp_min(1.0)

    def _proposal_clip_context(
        self,
        candidate_tokens: torch.Tensor,
        candidate_quads: torch.Tensor,
        proposal_source_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Full-clip proposal context over all frame/candidate selectors."""
        b, t, k, c = candidate_tokens.shape
        device = candidate_tokens.device
        dtype = candidate_tokens.dtype
        geom = candidate_quads.detach().reshape(b, t, k, 8).to(device=device, dtype=dtype)
        source_ids = proposal_source_ids.detach().clamp(0, self.proposal_clip_source_embed.num_embeddings - 1)
        source_embed = self.proposal_clip_source_embed(source_ids).to(device=device, dtype=dtype)
        time_idx = torch.arange(t, device=device).clamp(max=self.proposal_clip_time_embed.shape[1] - 1)
        time_embed = self.proposal_clip_time_embed[:, time_idx].to(device=device, dtype=dtype)
        clip_tokens = (
            candidate_tokens
            + time_embed
            + source_embed
            + self.proposal_clip_geom_mlp(geom)
        )
        flat = clip_tokens.reshape(b, t * k, c)
        norm_flat = self.proposal_clip_context_norm(flat)
        attn_out, _ = self.proposal_clip_context_attn(norm_flat, norm_flat, norm_flat, need_weights=False)
        flat = flat + attn_out
        flat = flat + self.proposal_clip_context_ffn(self.proposal_clip_context_ffn_norm(flat))
        return flat.reshape(b, t, k, c)

    def _proposal_frame_anchor_memory(
        self,
        candidate_tokens: torch.Tensor,
        candidate_quads: torch.Tensor,
        proposal_source_ids: torch.Tensor,
        quality_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Restricted essential-frame memory over proposal evidence.

        This follows the RMem lesson at the proposal level: select a compact
        frame memory first, then broadcast it, instead of letting all candidates
        attend to all other candidates.
        """
        b, t, k, c = candidate_tokens.shape
        device = candidate_tokens.device
        dtype = candidate_tokens.dtype
        detached_logits = quality_logits.detach().float()
        candidate_weights = F.softmax(
            detached_logits / self.proposal_frame_anchor_candidate_temperature,
            dim=-1,
        ).to(device=device, dtype=dtype)
        weighted_token = (candidate_tokens * candidate_weights[..., None]).sum(dim=2)
        mean_token = candidate_tokens.mean(dim=2)
        max_token = candidate_tokens.max(dim=2).values

        weighted_quad = (
            candidate_quads.detach().reshape(b, t, k, 8).to(device=device, dtype=dtype)
            * candidate_weights[..., None]
        ).sum(dim=2)
        logit_mean = detached_logits.mean(dim=-1)
        logit_max = detached_logits.max(dim=-1).values
        logit_std = detached_logits.std(dim=-1, unbiased=False)
        entropy = -(
            candidate_weights.float().clamp_min(1e-6)
            * candidate_weights.float().clamp_min(1e-6).log()
        ).sum(dim=-1) / math.log(max(k, 2))
        source_ids = proposal_source_ids.detach().to(device=device)
        source0_mass = (candidate_weights * (source_ids == 0).to(dtype=dtype)).sum(dim=-1)
        source4_mass = (candidate_weights * (source_ids == 4).to(dtype=dtype)).sum(dim=-1)
        stats = torch.stack(
            [
                logit_mean,
                logit_max,
                logit_std,
                entropy.to(device=device, dtype=logit_mean.dtype),
                source0_mass.float(),
                source4_mass.float(),
            ],
            dim=-1,
        ).to(device=device, dtype=dtype)
        frame_token = self.proposal_frame_anchor_frame_mlp(
            torch.cat([weighted_token, mean_token, max_token, weighted_quad, stats], dim=-1)
        )
        frame_logits = self.proposal_frame_anchor_score_head(frame_token).squeeze(-1)
        frame_select_logits = frame_logits / self.proposal_frame_anchor_temperature
        topk = min(int(self.proposal_frame_anchor_topk), t)
        if topk > 0 and topk < t:
            topk_idx = frame_select_logits.topk(topk, dim=-1).indices
            masked_logits = torch.full_like(frame_select_logits, torch.finfo(frame_select_logits.dtype).min)
            frame_select_logits = masked_logits.scatter(
                -1,
                topk_idx,
                frame_select_logits.gather(-1, topk_idx),
            )
        frame_weights = F.softmax(frame_select_logits, dim=-1).to(dtype=dtype)
        memory = (frame_token * frame_weights[..., None]).sum(dim=1)
        context = memory[:, None, None, :].expand(-1, t, k, -1)
        return context, frame_logits, frame_weights

    def _active_tip_candidate_features(
        self,
        candidate_quads: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_id: torch.Tensor,
    ) -> torch.Tensor:
        b, _, k_select = candidate_quads.shape[:3]
        device = candidate_quads.device
        dtype = candidate_quads.dtype
        kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
        kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        is_hook = (
            tool_id.to(device=device).reshape(b, 1, 1, 1).to(dtype=torch.long) == 0
        )
        hook_tip_xy = kpt_xy[:, :, 2, :]
        hook_tip_vis = kpt_vis[:, :, 2, 0]
        tip_pair_vis = kpt_vis[:, :, 1:3, :]
        tip_pair_sum = tip_pair_vis.sum(dim=2).clamp_min(1e-4)
        tip_pair_xy = (kpt_xy[:, :, 1:3, :] * tip_pair_vis).sum(dim=2) / tip_pair_sum
        tip_pair_visible = tip_pair_vis.squeeze(-1).amax(dim=2)
        tip_xy = torch.where(is_hook.squeeze(-1), hook_tip_xy, tip_pair_xy)
        tip_vis = torch.where(is_hook.reshape(b, 1, 1).squeeze(-1), hook_tip_vis, tip_pair_visible)
        role = role_features_16.to(device=device, dtype=dtype)
        active = role[..., 0].clamp(0.0, 1.0)
        phase = role[..., 1].clamp(0.0, 1.0)
        dist_to_range = role[..., 3].abs().clamp(0.0, 2.0)

        cand_center = candidate_quads.mean(dim=-2)
        curr_tip_dist = torch.linalg.norm(cand_center - tip_xy[:, :, None, :], dim=-1)

        path_w = (active * tip_vis).clamp_min(0.0)
        fallback_w = tip_vis.clamp_min(0.0)
        has_active = path_w.sum(dim=1, keepdim=True) > 1e-4
        path_w = torch.where(has_active, path_w, fallback_w)
        path_w = path_w / path_w.sum(dim=1, keepdim=True).clamp_min(1e-4)
        active_tip_mean = (tip_xy * path_w[:, :, None]).sum(dim=1)
        mean_tip_dist = torch.linalg.norm(cand_center - active_tip_mean[:, None, None, :], dim=-1)

        dist_to_path = torch.linalg.norm(cand_center[:, :, :, None, :] - tip_xy[:, None, None, :, :], dim=-1)
        tau = 0.050
        path_logits = -dist_to_path / tau + (path_w[:, None, None, :] + 1e-6).log()
        path_alpha = F.softmax(path_logits, dim=-1)
        active_path_dist = (path_alpha * dist_to_path).sum(dim=-1)

        hook_anchor_vis = kpt_vis[:, :, 0:2, :]
        hook_anchor_sum = hook_anchor_vis.sum(dim=2).clamp_min(1e-4)
        hook_anchor_xy = (kpt_xy[:, :, 0:2, :] * hook_anchor_vis).sum(dim=2) / hook_anchor_sum
        nonhook_anchor_xy = kpt_xy[:, :, 0, :]
        anchor_xy = torch.where(is_hook.squeeze(-1), hook_anchor_xy, nonhook_anchor_xy)
        tool_axis = tip_xy - anchor_xy
        fallback_axis = torch.zeros_like(tool_axis)
        fallback_axis[..., 0] = 1.0
        tool_axis_norm = tool_axis.norm(dim=-1, keepdim=True)
        tool_axis = torch.where(
            tool_axis_norm > 1e-4,
            tool_axis / tool_axis_norm.clamp_min(1e-4),
            fallback_axis,
        )
        prev_tip = torch.cat([tip_xy[:, :1], tip_xy[:, :-1]], dim=1)
        next_tip = torch.cat([tip_xy[:, 1:], tip_xy[:, -1:]], dim=1)
        motion = next_tip - prev_tip
        motion_norm = motion.norm(dim=-1, keepdim=True)
        motion_axis = torch.where(motion_norm > 1e-4, motion / motion_norm.clamp_min(1e-4), tool_axis)

        cand_dirs, cand_lens = quad_axes(candidate_quads.reshape(-1, 4, 2))
        cand_axis = cand_dirs[:, 0, :].reshape(b, 16, k_select, 2)
        cand_lens = cand_lens.reshape(b, 16, k_select, 2)
        tool_align = (cand_axis * tool_axis[:, :, None, :]).sum(dim=-1).abs()
        motion_align = (cand_axis * motion_axis[:, :, None, :]).sum(dim=-1).abs()
        cand_area = (cand_lens[..., 0] * cand_lens[..., 1]).clamp(0.0, 1.0)
        cand_aspect = (cand_lens[..., 0] / cand_lens[..., 1].clamp_min(1e-4)).log().clamp(-3.0, 3.0) / 3.0
        active_count = path_w.gt(1e-4).float().sum(dim=1).clamp(0.0, 16.0) / 16.0

        return torch.stack(
            [
                (-curr_tip_dist / 0.25).clamp(-8.0, 0.0),
                (-active_path_dist / 0.25).clamp(-8.0, 0.0),
                (-mean_tip_dist / 0.25).clamp(-8.0, 0.0),
                tool_align,
                motion_align,
                active[:, :, None].expand(-1, -1, k_select),
                phase[:, :, None].expand(-1, -1, k_select),
                (-dist_to_range[:, :, None]).expand(-1, -1, k_select),
                cand_area,
                cand_aspect,
                tip_vis[:, :, None].expand(-1, -1, k_select),
                active_count[:, None, None].expand(-1, 16, k_select),
            ],
            dim=-1,
        )

    def _tool_anchor_quad(self, tool_kpt_vec_16: torch.Tensor) -> torch.Tensor:
        b = tool_kpt_vec_16.shape[0]
        device = tool_kpt_vec_16.device
        dtype = tool_kpt_vec_16.dtype
        kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
        kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        vis_sum = kpt_vis.sum(dim=2).clamp_min(1e-4)
        base_center = (kpt_xy * kpt_vis).sum(dim=2) / vis_sum
        base_axis = kpt_xy[:, :, 0] - kpt_xy[:, :, -1]
        fallback = torch.zeros_like(base_axis)
        fallback[..., 0] = 1.0
        axis_norm = base_axis.norm(dim=-1, keepdim=True)
        base_axis = torch.where(axis_norm > 1e-4, base_axis / axis_norm.clamp_min(1e-4), fallback)
        kpt_dist = (kpt_xy - base_center[:, :, None, :]).norm(dim=-1)
        tool_extent = (kpt_dist * kpt_vis.squeeze(-1)).sum(dim=2) / vis_sum.squeeze(-1)
        base_size = torch.stack(
            [
                (0.75 * tool_extent + 0.060).clamp(0.045, 0.24),
                (0.45 * tool_extent + 0.045).clamp(0.035, 0.18),
            ],
            dim=-1,
        )
        return oriented_rect_to_quad(base_center, base_size, base_axis).clamp(0.0, 1.0)

    def _unit_vec(self, vec: torch.Tensor, fallback: torch.Tensor | None = None) -> torch.Tensor:
        if fallback is None:
            fallback = torch.zeros_like(vec)
            fallback[..., 0] = 1.0
        fb_norm = fallback.norm(dim=-1, keepdim=True)
        fallback = torch.where(fb_norm > 1e-4, fallback / fb_norm.clamp_min(1e-4), torch.zeros_like(fallback))
        fallback[..., 0] = torch.where(fb_norm.squeeze(-1) > 1e-4, fallback[..., 0], torch.ones_like(fallback[..., 0]))
        norm = vec.norm(dim=-1, keepdim=True)
        return torch.where(norm > 1e-4, vec / norm.clamp_min(1e-4), fallback)

    def _topology_contact_template_quads(
        self,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic source7 candidates from the surgical-tool topology.

        Dataset keypoint schema is hook=[anchor0, anchor1, tip0] and
        grasper/scissors/clipper=[anchor0, tip0, tip1].  We generate a fixed
        padded bank so mixed-tool batches keep a stable candidate count.
        """
        b = tool_kpt_vec_16.shape[0]
        device = tool_kpt_vec_16.device
        dtype = tool_kpt_vec_16.dtype
        kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
        kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        sizes = self.topology_contact_template_sizes
        offsets = self.topology_contact_template_offsets
        endpoint_offsets = [0.0, 2.0 / 64.0, -2.0 / 64.0]

        def append_family(
            out: list[torch.Tensor],
            center: torch.Tensor,
            axis: torch.Tensor,
            family_offsets: list[float],
        ) -> None:
            axis = self._unit_vec(axis)
            for off in family_offsets:
                shifted = (center + float(off) * axis).clamp(0.0, 1.0)
                for long_size, short_size in sizes:
                    size = shifted.new_tensor([float(long_size), float(short_size)]).view(1, 1, 2).expand(b, 16, 2)
                    out.append(oriented_rect_to_quad(shifted, size, axis).clamp(0.0, 1.0))
                    swapped = shifted.new_tensor([float(short_size), float(long_size)]).view(1, 1, 2).expand(b, 16, 2)
                    out.append(oriented_rect_to_quad(shifted, swapped, axis).clamp(0.0, 1.0))

        hook_quads: list[torch.Tensor] = []
        anchor_mid = 0.5 * (kpt_xy[:, :, 0] + kpt_xy[:, :, 1])
        hook_tip = kpt_xy[:, :, 2]
        hook_axis = self._unit_vec(hook_tip - anchor_mid)
        hook_perp = torch.stack([-hook_axis[..., 1], hook_axis[..., 0]], dim=-1)
        append_family(hook_quads, hook_tip, hook_axis, offsets)
        append_family(hook_quads, hook_tip, hook_perp, [0.0])

        nonhook_quads: list[torch.Tensor] = []
        tip_w = kpt_vis[:, :, 1:3].clamp(0.05, 1.0)
        tip_mid = (kpt_xy[:, :, 1:3] * tip_w).sum(dim=2) / tip_w.sum(dim=2).clamp_min(1e-4)
        jaw_axis = self._unit_vec(kpt_xy[:, :, 2] - kpt_xy[:, :, 1], fallback=tip_mid - kpt_xy[:, :, 0])
        shaft_axis = self._unit_vec(tip_mid - kpt_xy[:, :, 0], fallback=jaw_axis)
        append_family(nonhook_quads, tip_mid, jaw_axis, offsets)
        append_family(nonhook_quads, tip_mid, shaft_axis, offsets)
        append_family(nonhook_quads, kpt_xy[:, :, 1], shaft_axis, endpoint_offsets)
        append_family(nonhook_quads, kpt_xy[:, :, 2], shaft_axis, endpoint_offsets)

        max_count = max(len(hook_quads), len(nonhook_quads), int(self.topology_contact_template_count), 1)

        def pad_stack(items: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            if not items:
                zero = kpt_xy[:, :, 0]
                fallback = oriented_rect_to_quad(
                    zero,
                    zero.new_tensor([4.0 / 64.0, 3.0 / 64.0]).view(1, 1, 2).expand(b, 16, 2),
                    self._unit_vec(zero.new_zeros(b, 16, 2)),
                )
                items = [fallback]
            valid_count = len(items)
            while len(items) < max_count:
                items.append(items[0])
            quads = torch.stack(items[:max_count], dim=2).clamp(0.0, 1.0)
            valid = torch.zeros((b, 16, max_count), device=device, dtype=torch.bool)
            valid[:, :, :valid_count] = True
            return quads, valid

        hook_bank, hook_valid = pad_stack(hook_quads)
        nonhook_bank, nonhook_valid = pad_stack(nonhook_quads)
        is_hook = (tool_id.to(device=device).reshape(b).to(dtype=torch.long) == 0).view(b, 1, 1, 1, 1)
        quads = torch.where(is_hook, hook_bank, nonhook_bank)
        valid = torch.where(is_hook.squeeze(-1).squeeze(-1), hook_valid, nonhook_valid)
        return quads.clamp(0.0, 1.0), valid

    def _sample_traj_readout_features(self, feat16: torch.Tensor, ref_xy: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = feat16.shape
        p = ref_xy.shape[2]
        offsets = (
            self.traj_readout_sample_offsets.to(device=feat16.device, dtype=feat16.dtype)
            * float(self.traj_readout_sample_radius)
        )
        pts = ref_xy[:, :, :, None, :] + offsets.view(1, 1, 1, -1, 2)
        grid = (pts.reshape(b * t, p * offsets.shape[0], 1, 2) * 2.0 - 1.0).clamp(-1.15, 1.15)
        sampled = F.grid_sample(
            feat16.reshape(b * t, c, h, w).float(),
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).to(feat16.dtype)
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, t, p, offsets.shape[0], c)
        center = sampled[:, :, :, 0]
        mean = sampled.mean(dim=3)
        maxv = sampled.max(dim=3).values
        return torch.cat([center, mean, maxv], dim=-1)

    def _decode_support_moment_proposal(self, support_logits_8: torch.Tensor) -> torch.Tensor:
        """Decode a tight support-peak moment into one 16-frame quad candidate."""
        if support_logits_8.ndim == 4:
            b, t8, h, w = support_logits_8.shape
            n = h * w
            logits_flat_in = support_logits_8.reshape(b, t8, n)
        elif support_logits_8.ndim == 3:
            b, t8, n = support_logits_8.shape
            h = int(round(math.sqrt(float(n))))
            w = h
            if h * w != n:
                raise ValueError(f"support logits token count is not square: n={n}")
            logits_flat_in = support_logits_8
        else:
            raise ValueError(f"unsupported support logits shape: {tuple(support_logits_8.shape)}")
        logits_flat = (logits_flat_in.detach() if self.proposal_support_moment_detach else logits_flat_in).float()
        top_n = max(4, min(n, int(round(float(self.proposal_support_moment_top_frac) * float(n)))))
        _top_val, top_idx = logits_flat.topk(top_n, dim=-1)
        masked_logits = torch.full_like(logits_flat, -1.0e4)
        masked_logits.scatter_(-1, top_idx, logits_flat.gather(-1, top_idx))
        weights = F.softmax(
            masked_logits / float(self.proposal_support_moment_temperature),
            dim=-1,
        ).to(dtype=support_logits_8.dtype)

        grid_flat = self.grid_xy.to(device=support_logits_8.device, dtype=support_logits_8.dtype)
        grid_flat = grid_flat.view(2, n).transpose(0, 1)
        center = torch.matmul(weights, grid_flat).clamp(0.0, 1.0)
        diff = grid_flat.view(1, 1, n, 2) - center.unsqueeze(-2)
        ww = weights.unsqueeze(-1)
        var_xx = (ww * diff[..., 0:1] * diff[..., 0:1]).sum(dim=-2).squeeze(-1)
        var_yy = (ww * diff[..., 1:2] * diff[..., 1:2]).sum(dim=-2).squeeze(-1)
        cov_xy = (ww * diff[..., 0:1] * diff[..., 1:2]).sum(dim=-2).squeeze(-1)

        angle = 0.5 * torch.atan2(2.0 * cov_xy, var_xx - var_yy + 1e-8)
        axis = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        delta = torch.sqrt((var_xx - var_yy).square() + 4.0 * cov_xy.square() + 1e-8)
        trace = (var_xx + var_yy).clamp_min(1e-8)
        eig1 = (0.5 * (trace + delta)).clamp_min(1e-8)
        eig2 = (0.5 * (trace - delta)).clamp_min(1e-8)
        token_floor = 2.0 / float(max(h, w))
        size = float(self.proposal_support_moment_size_scale) * torch.stack(
            [torch.sqrt(eig1), torch.sqrt(eig2)],
            dim=-1,
        )
        size = (size + token_floor).clamp(0.03, 0.75)
        quad8 = oriented_rect_to_quad(center, size, axis).clamp(0.0, 1.0)
        quad16 = F.interpolate(
            quad8.permute(0, 2, 3, 1).reshape(b * 4, 2, t8),
            size=16,
            mode="linear",
            align_corners=True,
        ).reshape(b, 4, 2, 16).permute(0, 3, 1, 2).contiguous()
        if self.proposal_support_moment_detach:
            quad16 = quad16.detach()
        return quad16.clamp(0.0, 1.0)

    def _decode_token_affinity_proposal(
        self,
        token_feat_8: torch.Tensor,
        center_logits_8: torch.Tensor,
    ) -> torch.Tensor:
        """Decode TokenCut/CutLER-style affinity regions into 16-frame quads.

        TokenCut/LOST/CutLER first use frozen transformer token affinity to
        discover a foreground region, then convert that region into a proposal.
        This lightweight version keeps the same contract but avoids a CPU
        eigensolve in the training loop: learned visual center logits choose
        seed tokens, cosine affinity expands each seed to a compact token
        region, and weighted second moments become an oriented rectangle.
        """
        b, t8, c, h, w = token_feat_8.shape
        n = h * w
        k = center_logits_8.shape[2]
        m = min(max(int(self.token_affinity_proposal_count), 1), int(k))
        tau = float(self.token_affinity_proposal_temperature)

        feat = token_feat_8.detach() if self.token_affinity_proposal_detach else token_feat_8
        feat_flat = feat.permute(0, 1, 3, 4, 2).reshape(b, t8, n, c).float()
        feat_flat = F.normalize(feat_flat, dim=-1, eps=1e-6)

        seed_logits_all = center_logits_8.detach() if self.token_affinity_proposal_detach else center_logits_8
        channel_conf = seed_logits_all.float().amax(dim=-1)
        seed_idx = channel_conf.topk(m, dim=-1).indices
        gather_idx = seed_idx[..., None].expand(-1, -1, -1, n)
        seed_logits = seed_logits_all.float().gather(2, gather_idx)
        seed_prob = F.softmax(seed_logits / tau, dim=-1)
        seed_feat = torch.einsum("btmn,btnc->btmc", seed_prob, feat_flat)
        seed_feat = F.normalize(seed_feat, dim=-1, eps=1e-6)

        grid_flat = self.grid_xy.to(device=token_feat_8.device, dtype=token_feat_8.dtype)
        grid_flat = grid_flat.view(2, n).transpose(0, 1)
        sim = torch.einsum("btnc,btmc->btmn", feat_flat, seed_feat)
        quad_variants = []
        for top_frac, size_scale in self.token_affinity_proposal_variants:
            top_n = max(4, min(n, int(round(float(top_frac) * float(n)))))
            _top_val, top_pos = sim.topk(top_n, dim=-1)
            top_mask = torch.zeros_like(sim, dtype=torch.bool).scatter_(-1, top_pos, True)
            masked_sim = sim.masked_fill(~top_mask, -1.0e4)
            region_w = F.softmax(masked_sim / tau, dim=-1).to(dtype=token_feat_8.dtype)

            center = torch.einsum("btmn,nc->btmc", region_w, grid_flat).clamp(0.0, 1.0)
            diff = grid_flat.view(1, 1, 1, n, 2) - center.unsqueeze(-2)
            ww = region_w.unsqueeze(-1)
            var_xx = (ww * diff[..., 0:1] * diff[..., 0:1]).sum(dim=-2).squeeze(-1)
            var_yy = (ww * diff[..., 1:2] * diff[..., 1:2]).sum(dim=-2).squeeze(-1)
            cov_xy = (ww * diff[..., 0:1] * diff[..., 1:2]).sum(dim=-2).squeeze(-1)

            angle = 0.5 * torch.atan2(2.0 * cov_xy, var_xx - var_yy + 1e-8)
            axis = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
            delta = torch.sqrt((var_xx - var_yy).square() + 4.0 * cov_xy.square() + 1e-8)
            trace = (var_xx + var_yy).clamp_min(1e-8)
            eig1 = (0.5 * (trace + delta)).clamp_min(1e-8)
            eig2 = (0.5 * (trace - delta)).clamp_min(1e-8)
            token_floor = 2.0 / float(max(h, w))
            size = float(size_scale) * torch.stack(
                [torch.sqrt(eig1), torch.sqrt(eig2)],
                dim=-1,
            )
            size = (size + token_floor).clamp(0.03, 0.75)
            quad_variants.append(oriented_rect_to_quad(center, size, axis).clamp(0.0, 1.0))

        quad8 = torch.cat(quad_variants, dim=2)
        mv = int(quad8.shape[2])
        quad16 = F.interpolate(
            quad8.permute(0, 2, 3, 4, 1).reshape(b * mv * 4, 2, t8),
            size=16,
            mode="linear",
            align_corners=True,
        ).reshape(b, mv, 4, 2, 16).permute(0, 4, 1, 2, 3).contiguous()
        if self.token_affinity_proposal_detach:
            quad16 = quad16.detach()
        return quad16.clamp(0.0, 1.0)

    def _decode_corner_heatmap_proposal(self, corner_logits_8: torch.Tensor) -> torch.Tensor:
        """Decode four ordered corner heatmaps into a differentiable 16-frame quad."""
        b, t8, corners, h, w = corner_logits_8.shape
        if corners != 4:
            raise ValueError(f"Expected 4 corner heatmap channels, got {corners}")
        tau = float(self.corner_heatmap_proposal_temperature)
        logits = corner_logits_8.reshape(b, t8, 4, h * w)
        prob = F.softmax((logits / tau).float(), dim=-1).to(dtype=corner_logits_8.dtype)
        grid_flat = self.grid_xy.to(device=corner_logits_8.device, dtype=corner_logits_8.dtype)
        grid_flat = grid_flat.view(2, h * w).transpose(0, 1)
        quad8 = torch.matmul(prob, grid_flat).clamp(0.0, 1.0)
        quad16 = F.interpolate(
            quad8.permute(0, 2, 3, 1).reshape(b * 4, 2, t8),
            size=16,
            mode="linear",
            align_corners=True,
        ).reshape(b, 4, 2, 16).permute(0, 3, 1, 2).contiguous()
        if self.corner_heatmap_proposal_detach:
            quad16 = quad16.detach()
        return quad16.clamp(0.0, 1.0)

    @staticmethod
    def _candidate_aligned_grid_points(candidate_quads: torch.Tensor, grid_size: int) -> torch.Tensor:
        """Sample a regular grid in each candidate quad's own coordinate frame.

        S2A-Net's AlignConv aligns convolution sampling locations to rotated
        anchors.  Here the feature map is a 24x24 V-JEPA token grid, so the
        lightweight equivalent is to bilinearly sample token features at a
        candidate-aligned grid rather than only at global center/corner points.
        """
        grid_size = max(int(grid_size), 2)
        q0 = candidate_quads[..., 0, :]
        q1 = candidate_quads[..., 1, :]
        q2 = candidate_quads[..., 2, :]
        q3 = candidate_quads[..., 3, :]
        coords = torch.linspace(
            0.0,
            1.0,
            grid_size,
            device=candidate_quads.device,
            dtype=candidate_quads.dtype,
        )
        vv, uu = torch.meshgrid(coords, coords, indexing="ij")
        u = uu.reshape(1, 1, 1, grid_size * grid_size, 1)
        v = vv.reshape(1, 1, 1, grid_size * grid_size, 1)
        pts = (
            (1.0 - u) * (1.0 - v) * q0.unsqueeze(-2)
            + u * (1.0 - v) * q1.unsqueeze(-2)
            + u * v * q2.unsqueeze(-2)
            + (1.0 - u) * v * q3.unsqueeze(-2)
        )
        return pts.clamp(0.0, 1.0)

    def _tool_mask_geom8(self, tool_mask_8: torch.Tensor) -> torch.Tensor:
        mask = tool_mask_8.squeeze(2)
        b, t, h, w = mask.shape
        grid = self.grid_xy.to(device=mask.device, dtype=mask.dtype)
        gx = grid[0].view(1, 1, h, w)
        gy = grid[1].view(1, 1, h, w)
        mass_raw = mask.sum(dim=(-1, -2))
        mass = mass_raw.clamp_min(1e-6)
        cx = (mask * gx).sum(dim=(-1, -2)) / mass
        cy = (mask * gy).sum(dim=(-1, -2)) / mass
        area = mass / float(h * w)
        sx = torch.sqrt(((gx - cx[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        sy = torch.sqrt(((gy - cy[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        empty = (mass_raw <= 0.0).to(mask.dtype)
        cx = cx * (1.0 - empty) + 0.5 * empty
        cy = cy * (1.0 - empty) + 0.5 * empty
        return torch.stack([cx, cy, area, sx, sy], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
        rgb_16: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, t8, h, w, _ = tokens.shape
        device = tokens.device
        dtype = tokens.dtype
        k = self.proposal_count

        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t8, -1, 1, 1)
        if self.use_tool_keypoints:
            kpt8 = tool_kpt_vec_16.reshape(b, 8, 2, 9).mean(dim=2)
            role_bias8 = role_bias8 + self.tool_kpt_mlp(kpt8).view(b, t8, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=device, dtype=dtype).view(1, 1, 2, h, w).expand(b, t8, -1, -1, -1)
        feats = [x]
        if self.use_rgb_sidecar:
            if self.rgb_encoder is None:
                raise RuntimeError("RGB sidecar is enabled without an encoder")
            feats.append(encode_rgb_sidecar_8(self.rgb_encoder, rgb_16, dtype=dtype))
        feats.append(grid)
        if self.use_tool_mask:
            feats.append(tool_mask_8.to(device=device, dtype=dtype))
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=device, dtype=dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t8, -1, h, w))
        corner_logits_8 = self.corner_heatmap_head(y).view(b, t8, 4, h, w)

        center_logits_8 = self.center_head(y).view(b, t8, k, h * w)
        prob = F.softmax(center_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=device, dtype=dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(prob, grid_flat)
        y_flat = y.flatten(2).view(b, t8, 1, -1, h * w)
        pooled8 = (y_flat * prob.unsqueeze(3)).sum(dim=-1)
        role16 = self.role_mlp(role_features_16)
        role8_as16 = role16.reshape(b, 8, 2, -1).mean(dim=2)
        cond8 = z[:, None, :].expand(-1, t8, -1)
        role_cond8 = role8_as16 + cond8
        if self.use_tool_mask:
            mask_geom8 = self._tool_mask_geom8(tool_mask_8.to(device=device, dtype=dtype))
            role_cond8 = role_cond8 + self.tool_mask_geom_mlp(mask_geom8)

        size8 = torch.sigmoid(self.size_head(pooled8 + self.candidate_embed.to(device=device, dtype=dtype))).clamp(0.02, 0.95)
        boxes8 = torch.cat([centers8, size8], dim=-1)
        q8 = self.candidate_in(
            torch.cat(
                [
                    pooled8,
                    role_cond8[:, :, None, :].expand(-1, -1, k, -1),
                    self.box_ref_mlp(boxes8),
                    boxes8,
                ],
                dim=-1,
            )
        )
        q8 = q8 + self.candidate_embed.to(device=device, dtype=dtype)

        q16 = F.interpolate(
            q8.permute(0, 2, 3, 1).reshape(b * k, -1, t8),
            size=16,
            mode="linear",
            align_corners=True,
        ).reshape(b, k, -1, 16).permute(0, 3, 1, 2).contiguous()
        boxes16 = F.interpolate(
            boxes8.permute(0, 2, 3, 1).reshape(b * k, 4, t8),
            size=16,
            mode="linear",
            align_corners=True,
        ).reshape(b, k, 4, 16).permute(0, 3, 1, 2).contiguous()

        active_tip_logits = None
        frame_anchor_logits = None
        frame_anchor_weights = None
        cond_quad = None
        cond_xyxy = None
        cond_cxcywh = None
        cond_logit = None
        h259_quad = None
        h259_xyxy = None
        h259_cxcywh = None
        h259_token = None
        h259_logit = None
        token_affinity_proposal_quad = None
        if self.condition_prior_enabled:
            cond_in = torch.cat(
                [
                    tool_kpt_vec_16.to(device=device, dtype=dtype),
                    role_features_16.to(device=device, dtype=dtype),
                ],
                dim=-1,
            )
            cond_tok = self.condition_prior_mlp(cond_in)
            cond_tok = cond_tok + z[:, None, :] + self.condition_prior_time_pos.to(device=device, dtype=dtype)
            cond_attn, _ = self.condition_prior_attn(
                self.condition_prior_norm(cond_tok),
                self.condition_prior_norm(cond_tok),
                self.condition_prior_norm(cond_tok),
                need_weights=False,
            )
            cond_tok = cond_tok + cond_attn
            cond_tok = cond_tok + self.condition_prior_ffn(self.condition_prior_ffn_norm(cond_tok))
            cond_raw = self.condition_prior_head(cond_tok)
            kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
            kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
            vis_sum = kpt_vis.sum(dim=2).clamp_min(1e-4)
            base_center = (kpt_xy * kpt_vis).sum(dim=2) / vis_sum
            base_axis = kpt_xy[:, :, 0] - kpt_xy[:, :, -1]
            fallback = torch.zeros_like(base_axis)
            fallback[..., 0] = 1.0
            axis_norm = base_axis.norm(dim=-1, keepdim=True)
            base_axis = torch.where(axis_norm > 1e-4, base_axis / axis_norm.clamp_min(1e-4), fallback)
            cond_center = (base_center + 0.35 * torch.tanh(cond_raw[..., :2])).clamp(0.0, 1.0)
            cond_size = torch.sigmoid(cond_raw[..., 2:4]).clamp(0.02, 0.95)
            cond_axis = F.normalize(base_axis + 0.75 * torch.tanh(cond_raw[..., 4:6]), dim=-1, eps=1e-6)
            cond_quad = oriented_rect_to_quad(cond_center, cond_size, cond_axis)
            cond_xyxy = quad_to_xyxy(cond_quad, min_size=0.02)
            cond_cxcywh = xyxy_to_cxcywh_torch(cond_xyxy)
            cond_logit = self.condition_prior_quality_head(cond_tok).squeeze(-1)
        if self.h259_prior_proposal:
            h259_out = self.h259_prior(
                tokens,
                role_features_16,
                tool_mask_8,
                tool_kpt_heatmaps_8,
                tool_kpt_vec_16,
                tool_id,
                act_id,
                combo_id,
            )
            h259_quad = h259_out["trajectory_prior_quad"].to(device=device, dtype=dtype)
            h259_xyxy = h259_out["trajectory_prior_xyxy"].to(device=device, dtype=dtype)
            h259_cxcywh = xyxy_to_cxcywh_torch(h259_xyxy)
            h259_geom = torch.cat(
                [
                    h259_quad.reshape(b, 16, 8),
                    tool_kpt_vec_16.to(device=device, dtype=dtype),
                    role_features_16.to(device=device, dtype=dtype),
                ],
                dim=-1,
            )
            h259_token = self.h259_prior_token_mlp(h259_geom)
            if self.proposal_quality_readout:
                mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
                feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
                feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
                h259_readout = self._sample_traj_readout_features(feat16_map, h259_quad)
                h259_readout = self.traj_readout_proj(h259_readout).mean(dim=2)
                h259_token = h259_token + h259_readout
            h259_logit = self.h259_prior_quality_head(h259_token).squeeze(-1)

            # Reuse the existing condition-prior metric/loss keys for this
            # stronger trajectory-prior proposal when it is enabled. This keeps
            # downstream logging and ablation scripts comparable to H260.
            cond_quad = h259_quad
            cond_xyxy = h259_xyxy
            cond_cxcywh = h259_cxcywh
            cond_logit = h259_logit

        if self.trajectory_readout_bias:
            traj_anchor_quad = self._tool_anchor_quad(tool_kpt_vec_16.to(device=device, dtype=dtype))
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            readout = self._sample_traj_readout_features(feat16_map, traj_anchor_quad)
            readout = self.traj_readout_proj(readout).mean(dim=2)
            readout_gate = self.traj_readout_gate(readout)
            q16 = q16 + 0.35 * readout_gate[:, :, None, :] * readout[:, :, None, :]
        if self.condition_prior_readout_bias and cond_quad is not None:
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            prior_readout = self._sample_traj_readout_features(feat16_map, cond_quad)
            prior_readout = self.traj_readout_proj(prior_readout).mean(dim=2)
            prior_gate = self.traj_readout_gate(prior_readout)
            q16 = q16 + self.condition_prior_readout_scale * prior_gate[:, :, None, :] * prior_readout[:, :, None, :]

        q_frame = q16.reshape(b * 16, k, -1)
        frame_out, _ = self.frame_attn(
            self.frame_norm(q_frame),
            self.frame_norm(q_frame),
            self.frame_norm(q_frame),
            need_weights=False,
        )
        q16 = (q_frame + frame_out).reshape(b, 16, k, -1)

        q_tem = q16.permute(0, 2, 1, 3).reshape(b * k, 16, -1)
        tem_out, _ = self.temporal_attn(
            self.temporal_norm(q_tem),
            self.temporal_norm(q_tem),
            self.temporal_norm(q_tem),
            need_weights=False,
        )
        q16 = (q_tem + tem_out).reshape(b, k, 16, -1).permute(0, 2, 1, 3)
        q16 = q16 + self.ffn(self.ffn_norm(q16))

        delta = torch.tanh(self.delta_head(q16))
        center = (boxes16[..., :2] + delta[..., :2] * self.residual_scale * boxes16[..., 2:].detach()).clamp(0.0, 1.0)
        size = (boxes16[..., 2:] * torch.exp(0.45 * delta[..., 2:4])).clamp(0.02, 0.95)
        base_axis = torch.zeros_like(delta[..., 4:6])
        base_axis[..., 0] = 1.0
        axis = F.normalize(base_axis + 0.75 * delta[..., 4:6], dim=-1, eps=1e-6)
        candidate_quads = oriented_rect_to_quad(center, size, axis)
        candidate_xyxy = quad_to_xyxy(candidate_quads.reshape(b, 16 * k, 4, 2), min_size=0.02).reshape(b, 16, k, 4)
        candidate_cxcywh = xyxy_to_cxcywh_torch(candidate_xyxy)
        if self.proposal_active_tip_feature_fusion:
            tip_feats = self._active_tip_candidate_features(
                candidate_quads,
                tool_kpt_vec_16,
                role_features_16,
                tool_id,
            )
            tip_feature = self.proposal_active_tip_feature_mlp(tip_feats)
            cand_geom = candidate_quads.reshape(b, 16, k, 8)
            tip_gate = self.proposal_active_tip_feature_gate(torch.cat([q16, tip_feature, cand_geom], dim=-1))
            q16 = q16 + self.proposal_active_tip_feature_scale * tip_gate * tip_feature
        quality_logits = self.quality_head(q16).squeeze(-1)
        candidate_tokens = q16
        proposal_source_ids = torch.zeros((b, 16, k), device=device, dtype=torch.long)
        support_logits_8 = torch.logsumexp(center_logits_8.float(), dim=2) - math.log(float(max(k, 1)))

        traj_query_quad = None
        traj_query_xyxy = None
        traj_query_cxcywh = None
        if self.trajectory_query_proposal_count > 0:
            tq = self.trajectory_query_proposal_count
            cond_in = torch.cat(
                [
                    tool_kpt_vec_16.to(device=device, dtype=dtype),
                    role_features_16.to(device=device, dtype=dtype),
                ],
                dim=-1,
            )
            traj_tok = self.traj_query_mlp(cond_in)
            traj_tok = traj_tok + z[:, None, :] + self.traj_query_time_pos.to(device=device, dtype=dtype)
            traj_q = traj_tok[:, :, None, :] + self.traj_query_embed.to(device=device, dtype=dtype)

            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            mem16_chw = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            mem16 = mem16_chw.permute(0, 2, 3, 4, 1).reshape(b * 16, h * w, -1)

            traj_q_flat = traj_q.reshape(b * 16, tq, -1)
            traj_cross, _ = self.traj_cross_attn(
                self.traj_query_norm(traj_q_flat),
                self.traj_memory_norm(mem16),
                self.traj_memory_norm(mem16),
                need_weights=False,
            )
            traj_q = (traj_q_flat + traj_cross).reshape(b, 16, tq, -1)

            traj_frame = traj_q.reshape(b * 16, tq, -1)
            traj_frame_out, _ = self.traj_frame_attn(
                self.traj_frame_norm(traj_frame),
                self.traj_frame_norm(traj_frame),
                self.traj_frame_norm(traj_frame),
                need_weights=False,
            )
            traj_q = (traj_frame + traj_frame_out).reshape(b, 16, tq, -1)

            traj_tem = traj_q.permute(0, 2, 1, 3).reshape(b * tq, 16, -1)
            traj_tem_out, _ = self.traj_temporal_attn(
                self.traj_temporal_norm(traj_tem),
                self.traj_temporal_norm(traj_tem),
                self.traj_temporal_norm(traj_tem),
                need_weights=False,
            )
            traj_q = (traj_tem + traj_tem_out).reshape(b, tq, 16, -1).permute(0, 2, 1, 3)
            traj_q = traj_q + self.traj_query_ffn(self.traj_query_ffn_norm(traj_q))

            traj_raw = self.traj_query_head(traj_q)
            kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
            kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
            vis_sum = kpt_vis.sum(dim=2).clamp_min(1e-4)
            base_center = (kpt_xy * kpt_vis).sum(dim=2) / vis_sum
            base_axis = kpt_xy[:, :, 0] - kpt_xy[:, :, -1]
            fallback = torch.zeros_like(base_axis)
            fallback[..., 0] = 1.0
            axis_norm = base_axis.norm(dim=-1, keepdim=True)
            base_axis = torch.where(axis_norm > 1e-4, base_axis / axis_norm.clamp_min(1e-4), fallback)
            if self.trajectory_query_output_mode == "direct_quad":
                kpt_dist = (kpt_xy - base_center[:, :, None, :]).norm(dim=-1)
                tool_extent = (kpt_dist * kpt_vis.squeeze(-1)).sum(dim=2) / vis_sum.squeeze(-1)
                base_size = torch.stack(
                    [
                        (0.75 * tool_extent + 0.060).clamp(0.045, 0.24),
                        (0.45 * tool_extent + 0.045).clamp(0.035, 0.18),
                    ],
                    dim=-1,
                )
                base_quad = oriented_rect_to_quad(
                    base_center[:, :, None, :].expand(-1, -1, tq, -1),
                    base_size[:, :, None, :].expand(-1, -1, tq, -1),
                    base_axis[:, :, None, :].expand(-1, -1, tq, -1),
                )
                corner_scale = (0.55 * base_size[:, :, None, None, :]).clamp(0.018, 0.16)
                traj_delta = torch.tanh(traj_raw.view(b, 16, tq, 4, 2)) * corner_scale
                traj_query_quad = (base_quad + traj_delta).clamp(0.0, 1.0)
            else:
                traj_center = (base_center[:, :, None, :] + 0.35 * torch.tanh(traj_raw[..., :2])).clamp(0.0, 1.0)
                traj_size = torch.sigmoid(traj_raw[..., 2:4]).clamp(0.02, 0.95)
                traj_axis = F.normalize(
                    base_axis[:, :, None, :] + 0.75 * torch.tanh(traj_raw[..., 4:6]),
                    dim=-1,
                    eps=1e-6,
                )
                traj_query_quad = oriented_rect_to_quad(traj_center, traj_size, traj_axis)
            traj_query_xyxy = quad_to_xyxy(traj_query_quad.reshape(b, 16 * tq, 4, 2), min_size=0.02).reshape(b, 16, tq, 4)
            traj_query_cxcywh = xyxy_to_cxcywh_torch(traj_query_xyxy)
            traj_query_logit = self.traj_query_quality_head(traj_q).squeeze(-1)

            candidate_quads = torch.cat([candidate_quads, traj_query_quad], dim=2)
            candidate_xyxy = torch.cat([candidate_xyxy, traj_query_xyxy], dim=2)
            candidate_cxcywh = torch.cat([candidate_cxcywh, traj_query_cxcywh], dim=2)
            quality_logits = torch.cat([quality_logits, traj_query_logit], dim=-1)
            candidate_tokens = torch.cat([candidate_tokens, traj_q], dim=2)
            proposal_source_ids = torch.cat(
                [
                    proposal_source_ids,
                    torch.full((b, 16, tq), 1, device=device, dtype=torch.long),
                ],
                dim=2,
            )

        if self.condition_prior_proposal and cond_quad is not None:
            candidate_quads = torch.cat([candidate_quads, cond_quad[:, :, None]], dim=2)
            candidate_xyxy = torch.cat([candidate_xyxy, cond_xyxy[:, :, None]], dim=2)
            candidate_cxcywh = torch.cat([candidate_cxcywh, cond_cxcywh[:, :, None]], dim=2)
            quality_logits = torch.cat([quality_logits, cond_logit[:, :, None]], dim=-1)
            candidate_tokens = torch.cat([candidate_tokens, cond_tok[:, :, None, :]], dim=2)
            proposal_source_ids = torch.cat(
                [
                    proposal_source_ids,
                    torch.full((b, 16, 1), 2, device=device, dtype=torch.long),
                ],
                dim=2,
            )
        if self.h259_prior_proposal and h259_quad is not None:
            candidate_quads = torch.cat([candidate_quads, h259_quad[:, :, None]], dim=2)
            candidate_xyxy = torch.cat([candidate_xyxy, h259_xyxy[:, :, None]], dim=2)
            candidate_cxcywh = torch.cat([candidate_cxcywh, h259_cxcywh[:, :, None]], dim=2)
            quality_logits = torch.cat([quality_logits, h259_logit[:, :, None]], dim=-1)
            candidate_tokens = torch.cat([candidate_tokens, h259_token[:, :, None, :]], dim=2)
            proposal_source_ids = torch.cat(
                [
                    proposal_source_ids,
                    torch.full((b, 16, 1), 3, device=device, dtype=torch.long),
                ],
                dim=2,
            )
        corner_proposal_quad = None
        if self.corner_heatmap_proposal:
            corner_proposal_quad = self._decode_corner_heatmap_proposal(corner_logits_8)
            corner_proposal_xyxy = quad_to_xyxy(corner_proposal_quad, min_size=0.02)
            corner_proposal_cxcywh = xyxy_to_cxcywh_torch(corner_proposal_xyxy)
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            corner_readout = self._sample_traj_readout_features(feat16_map, corner_proposal_quad)
            corner_token = self.corner_proposal_token_proj(corner_readout.mean(dim=2))
            corner_logit = self.corner_proposal_quality_head(corner_token).squeeze(-1)
            candidate_quads = torch.cat([candidate_quads, corner_proposal_quad[:, :, None]], dim=2)
            candidate_xyxy = torch.cat([candidate_xyxy, corner_proposal_xyxy[:, :, None]], dim=2)
            candidate_cxcywh = torch.cat([candidate_cxcywh, corner_proposal_cxcywh[:, :, None]], dim=2)
            quality_logits = torch.cat([quality_logits, corner_logit[:, :, None]], dim=-1)
            candidate_tokens = torch.cat([candidate_tokens, corner_token[:, :, None, :]], dim=2)
            proposal_source_ids = torch.cat(
                [
                    proposal_source_ids,
                    torch.full((b, 16, 1), 4, device=device, dtype=torch.long),
                ],
                dim=2,
            )
        if self.token_affinity_proposal:
            token_affinity_proposal_quad = self._decode_token_affinity_proposal(x, center_logits_8)
            ta_count = int(token_affinity_proposal_quad.shape[2])
            token_affinity_proposal_xyxy = quad_to_xyxy(
                token_affinity_proposal_quad.reshape(b, 16 * ta_count, 4, 2),
                min_size=0.02,
            ).reshape(b, 16, ta_count, 4)
            token_affinity_proposal_cxcywh = xyxy_to_cxcywh_torch(token_affinity_proposal_xyxy)
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            affinity_readout = self._sample_traj_readout_features(
                feat16_map,
                token_affinity_proposal_quad.reshape(b, 16, ta_count * 4, 2),
            )
            affinity_readout = affinity_readout.reshape(b, 16, ta_count, 4, -1).mean(dim=3)
            affinity_token = self.token_affinity_proposal_token_proj(affinity_readout)
            affinity_geom = token_affinity_proposal_quad.reshape(b, 16, ta_count, 8)
            affinity_logit = self.token_affinity_proposal_quality_head(
                torch.cat([affinity_token, affinity_geom], dim=-1)
            ).squeeze(-1)
            candidate_quads = torch.cat([candidate_quads, token_affinity_proposal_quad], dim=2)
            candidate_xyxy = torch.cat([candidate_xyxy, token_affinity_proposal_xyxy], dim=2)
            candidate_cxcywh = torch.cat([candidate_cxcywh, token_affinity_proposal_cxcywh], dim=2)
            quality_logits = torch.cat([quality_logits, affinity_logit], dim=-1)
            candidate_tokens = torch.cat([candidate_tokens, affinity_token], dim=2)
            proposal_source_ids = torch.cat(
                [
                    proposal_source_ids,
                    torch.full((b, 16, ta_count), 5, device=device, dtype=torch.long),
                ],
                dim=2,
            )

        topology_contact_template_quad = None
        if self.topology_contact_template_proposal:
            topology_contact_template_quad, topology_template_valid = self._topology_contact_template_quads(
                tool_kpt_vec_16.to(device=device, dtype=dtype),
                tool_id,
            )
            topo_count = int(topology_contact_template_quad.shape[2])
            topology_contact_template_xyxy = quad_to_xyxy(
                topology_contact_template_quad.reshape(b, 16 * topo_count, 4, 2),
                min_size=0.02,
            ).reshape(b, 16, topo_count, 4)
            topology_contact_template_cxcywh = xyxy_to_cxcywh_torch(topology_contact_template_xyxy)
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            topo_readout = self._sample_traj_readout_features(
                feat16_map,
                topology_contact_template_quad.reshape(b, 16, topo_count * 4, 2),
            )
            topo_readout = topo_readout.reshape(b, 16, topo_count, 4, -1).mean(dim=3)
            topology_token = self.topology_contact_template_token_proj(topo_readout)
            if self.topology_contact_template_index_embedding:
                topo_idx = torch.arange(topo_count, device=device, dtype=torch.long).clamp(
                    max=self.topology_contact_template_index_embed.num_embeddings - 1
                )
                topology_token = topology_token + self.topology_contact_template_index_embed(
                    topo_idx
                ).to(dtype=topology_token.dtype).view(1, 1, topo_count, -1)
            topology_geom = topology_contact_template_quad.reshape(b, 16, topo_count, 8)
            topology_logit = self.topology_contact_template_quality_head(
                torch.cat([topology_token, topology_geom], dim=-1)
            ).squeeze(-1)
            topology_logit = topology_logit + float(self.topology_contact_template_logit_bias)
            if self.topology_contact_template_combo_bias:
                combo_idx = combo_id.to(device=device, dtype=torch.long).clamp(
                    0,
                    self.topology_contact_template_combo_bias_table.num_embeddings - 1,
                )
                combo_bias = self.topology_contact_template_combo_bias_table(combo_idx)[
                    :, :topo_count
                ].to(dtype=topology_logit.dtype)
                topology_logit = topology_logit + float(
                    self.topology_contact_template_combo_bias_scale
                ) * combo_bias[:, None, :]
            allow_mask = getattr(self, "topology_contact_template_allow_mask", None)
            if allow_mask is not None:
                combo_idx = combo_id.to(device=device, dtype=torch.long).clamp(
                    0,
                    allow_mask.shape[0] - 1,
                )
                combo_allow = allow_mask.to(device=device)[combo_idx, :topo_count]
                topology_template_valid = topology_template_valid & combo_allow[:, None, :]
                topology_contact_template_quad = torch.where(
                    topology_template_valid[..., None, None],
                    topology_contact_template_quad,
                    torch.zeros_like(topology_contact_template_quad),
                )
                topology_contact_template_xyxy = quad_to_xyxy(
                    topology_contact_template_quad.reshape(b, 16 * topo_count, 4, 2),
                    min_size=0.02,
                ).reshape(b, 16, topo_count, 4)
                topology_contact_template_cxcywh = xyxy_to_cxcywh_torch(
                    topology_contact_template_xyxy
                )
            topology_logit = torch.where(
                topology_template_valid,
                topology_logit,
                torch.full_like(topology_logit, -50.0),
            )
            candidate_quads = torch.cat([candidate_quads, topology_contact_template_quad], dim=2)
            candidate_xyxy = torch.cat([candidate_xyxy, topology_contact_template_xyxy], dim=2)
            candidate_cxcywh = torch.cat([candidate_cxcywh, topology_contact_template_cxcywh], dim=2)
            quality_logits = torch.cat([quality_logits, topology_logit], dim=-1)
            candidate_tokens = torch.cat([candidate_tokens, topology_token], dim=2)
            topology_source_ids = torch.where(
                topology_template_valid,
                torch.full((b, 16, topo_count), 7, device=device, dtype=torch.long),
                torch.full((b, 16, topo_count), -1, device=device, dtype=torch.long),
            )
            proposal_source_ids = torch.cat(
                [
                    proposal_source_ids,
                    topology_source_ids,
                ],
                dim=2,
            )

        support_moment_proposal_quad = None
        if self.proposal_support_moment_proposal:
            support_moment_proposal_quad = self._decode_support_moment_proposal(support_logits_8)
            support_moment_proposal_xyxy = quad_to_xyxy(support_moment_proposal_quad, min_size=0.02)
            support_moment_proposal_cxcywh = xyxy_to_cxcywh_torch(support_moment_proposal_xyxy)
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            support_readout = self._sample_traj_readout_features(feat16_map, support_moment_proposal_quad)
            support_token = self.support_moment_proposal_token_proj(support_readout.mean(dim=2))
            support_geom = support_moment_proposal_quad.reshape(b, 16, 8)
            support_logit = self.support_moment_proposal_quality_head(
                torch.cat([support_token, support_geom], dim=-1)
            ).squeeze(-1)
            candidate_quads = torch.cat([candidate_quads, support_moment_proposal_quad[:, :, None]], dim=2)
            candidate_xyxy = torch.cat([candidate_xyxy, support_moment_proposal_xyxy[:, :, None]], dim=2)
            candidate_cxcywh = torch.cat([candidate_cxcywh, support_moment_proposal_cxcywh[:, :, None]], dim=2)
            quality_logits = torch.cat([quality_logits, support_logit[:, :, None]], dim=-1)
            candidate_tokens = torch.cat([candidate_tokens, support_token[:, :, None, :]], dim=2)
            proposal_source_ids = torch.cat(
                [
                    proposal_source_ids,
                    torch.full((b, 16, 1), 6, device=device, dtype=torch.long),
                ],
                dim=2,
            )

        k_select = candidate_quads.shape[2]
        if self.proposal_trajectory_refine:
            traj_in = torch.cat(
                [
                    tool_kpt_vec_16.to(device=device, dtype=dtype),
                    role_features_16.to(device=device, dtype=dtype),
                ],
                dim=-1,
            )
            traj_mem = self.proposal_traj_mlp(traj_in)
            traj_mem = traj_mem + z[:, None, :] + self.proposal_traj_time_pos.to(device=device, dtype=dtype)
            cand_q = candidate_tokens.reshape(b * 16, k_select, -1)
            traj_kv = traj_mem[:, None, :, :].expand(-1, 16, -1, -1).reshape(b * 16, 16, -1)
            traj_ctx, _ = self.proposal_traj_attn(
                self.proposal_traj_query_norm(cand_q),
                self.proposal_traj_memory_norm(traj_kv),
                self.proposal_traj_memory_norm(traj_kv),
                need_weights=False,
            )
            traj_ctx = traj_ctx.reshape(b, 16, k_select, -1)
            cand_geom = candidate_quads.reshape(b, 16, k_select, 8)
            refine_raw = torch.tanh(
                self.proposal_traj_refine_head(torch.cat([candidate_tokens, traj_ctx, cand_geom], dim=-1))
            )
            cand_center = candidate_quads.mean(dim=-2)
            cand_dirs, cand_lens = quad_axes(candidate_quads.reshape(-1, 4, 2))
            cand_dirs = cand_dirs.reshape(b, 16, k_select, 2, 2)
            cand_lens = cand_lens.reshape(b, 16, k_select, 2)
            refine_scale = float(self.proposal_trajectory_refine_scale)
            center_step = cand_lens.mean(dim=-1, keepdim=True).clamp(0.02, 0.35)
            refined_center = (cand_center + refine_scale * refine_raw[..., :2] * center_step).clamp(0.0, 1.0)
            refined_size = (cand_lens * torch.exp(0.35 * refine_scale * refine_raw[..., 2:4])).clamp(0.02, 0.95)
            refined_axis = F.normalize(
                cand_dirs[..., 0, :] + 0.75 * refine_scale * refine_raw[..., 4:6],
                dim=-1,
                eps=1e-6,
            )
            candidate_quads = oriented_rect_to_quad(refined_center, refined_size, refined_axis)
            candidate_xyxy = quad_to_xyxy(candidate_quads.reshape(b, 16 * k_select, 4, 2), min_size=0.02).reshape(
                b, 16, k_select, 4
            )
            candidate_cxcywh = xyxy_to_cxcywh_torch(candidate_xyxy)

        if self.proposal_pointset_refine:
            # OrientedRepPoints-style handoff: use the current 4 candidate
            # points as the representation, sample local token evidence at
            # those points, then predict a small point-wise residual.  The
            # final task remains free-quad regression; this avoids forcing the
            # GT through a rotated-rectangle surrogate.
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            gm = float(self.proposal_pointset_refine_grad_mul)
            sample_quads = candidate_quads.detach() * (1.0 - gm) + candidate_quads * gm
            point_readout = self._sample_traj_readout_features(
                feat16_map,
                sample_quads.reshape(b, 16, k_select * 4, 2),
            )
            point_readout = point_readout.reshape(b, 16, k_select, 4, -1)
            point_feat = self.proposal_pointset_readout_proj(point_readout)
            point_feat = point_feat + self.proposal_pointset_corner_embed.to(device=device, dtype=dtype)
            point_geom_center = sample_quads.mean(dim=-2, keepdim=True)
            point_geom = torch.cat([sample_quads, sample_quads - point_geom_center], dim=-1)
            cand_tok = candidate_tokens[:, :, :, None, :].expand(-1, -1, -1, 4, -1)
            point_delta = torch.tanh(
                self.proposal_pointset_delta_head(torch.cat([cand_tok, point_feat, point_geom], dim=-1))
            )
            _dirs, cand_lens = quad_axes(sample_quads.reshape(-1, 4, 2))
            cand_lens = cand_lens.reshape(b, 16, k_select, 2)
            point_step = (
                float(self.proposal_pointset_refine_scale)
                * cand_lens.mean(dim=-1).clamp(0.015, 0.35)
            )
            candidate_quads = (candidate_quads + point_delta * point_step[..., None, None]).clamp(0.0, 1.0)
            candidate_xyxy = quad_to_xyxy(candidate_quads.reshape(b, 16 * k_select, 4, 2), min_size=0.02).reshape(
                b, 16, k_select, 4
            )
            candidate_cxcywh = xyxy_to_cxcywh_torch(candidate_xyxy)
            point_context = point_feat.mean(dim=3)
            if self.proposal_pointset_refine_update_token:
                candidate_tokens = candidate_tokens + self.proposal_pointset_token_proj(point_context)
            if self.proposal_pointset_quality_scale != 0.0:
                quality_logits = quality_logits + (
                    self.proposal_pointset_quality_scale
                    * self.proposal_pointset_quality_head(point_context).squeeze(-1)
                )

        decoupled_selector_readout = None

        if self.proposal_quality_readout:
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            cand_center = candidate_quads.mean(dim=-2, keepdim=True)
            if self.proposal_quality_readout_points == "aligned_grid_5x5":
                readout_pts = self._candidate_aligned_grid_points(candidate_quads, grid_size=5)
                points_per_candidate = 25
            elif self.proposal_quality_readout_points == "aligned_grid_3x3":
                readout_pts = self._candidate_aligned_grid_points(candidate_quads, grid_size=3)
                points_per_candidate = 9
            elif self.proposal_quality_readout_points == "center_corners_edges":
                edge_midpoints = 0.5 * (candidate_quads + torch.roll(candidate_quads, shifts=-1, dims=-2))
                readout_pts = torch.cat([cand_center, candidate_quads, edge_midpoints], dim=3)
                points_per_candidate = 9
            else:
                readout_pts = torch.cat([cand_center, candidate_quads], dim=3)
                points_per_candidate = 5
            readout_pts = readout_pts.reshape(b, 16, k_select * points_per_candidate, 2)
            cand_geom = candidate_quads.reshape(b, 16, k_select, 8)
            cand_readout = self._sample_traj_readout_features(feat16_map, readout_pts)
            cand_readout = cand_readout.reshape(b, 16, k_select, points_per_candidate, -1)
            point_readout = self.proposal_readout_proj(cand_readout)
            if self.proposal_quality_readout_pool == "attention":
                attn_token = candidate_tokens[:, :, :, None, :].expand(-1, -1, -1, points_per_candidate, -1)
                attn_geom = cand_geom[:, :, :, None, :].expand(-1, -1, -1, points_per_candidate, -1)
                attn_in = torch.cat([point_readout, attn_token, attn_geom], dim=-1)
                if self.proposal_quality_readout_detach:
                    attn_in = attn_in.detach()
                point_weights = F.softmax(self.proposal_quality_readout_attn(attn_in).squeeze(-1), dim=3)
                cand_readout = (point_readout * point_weights[..., None]).sum(dim=3)
            else:
                cand_readout = point_readout.mean(dim=3)
            readout_tokens = candidate_tokens
            if self.proposal_quality_readout_detach:
                readout_tokens = readout_tokens.detach()
                cand_readout = cand_readout.detach()
                cand_geom = cand_geom.detach()
            quality_parts = [readout_tokens, cand_readout, cand_geom]
            if self.proposal_quality_readout_traj_context:
                traj_context = self.proposal_quality_traj_context_mlp(
                    torch.cat(
                        [
                            tool_kpt_vec_16.to(device=device, dtype=dtype),
                            role_features_16.to(device=device, dtype=dtype),
                        ],
                        dim=-1,
                    )
                )
                quality_parts.insert(2, traj_context[:, :, None, :].expand(-1, -1, k_select, -1))
            quality_delta = self.proposal_quality_readout_head(
                torch.cat(quality_parts, dim=-1)
            ).squeeze(-1)
            if self.proposal_quality_readout_replace_logits:
                quality_logits = quality_delta
            else:
                quality_logits = quality_logits + quality_delta

        if self.proposal_decoupled_selector_readout:
            mem8_chw = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            feat16_map = F.interpolate(mem8_chw, size=(16, h, w), mode="trilinear", align_corners=True)
            feat16_map = feat16_map.permute(0, 2, 1, 3, 4).contiguous()
            readout_quads = candidate_quads
            if self.proposal_decoupled_selector_detach:
                feat16_map = feat16_map.detach()
                readout_quads = readout_quads.detach()
            cand_center = readout_quads.mean(dim=-2, keepdim=True)
            if self.proposal_quality_readout_points == "aligned_grid_5x5":
                readout_pts = self._candidate_aligned_grid_points(readout_quads, grid_size=5)
                points_per_candidate = 25
            elif self.proposal_quality_readout_points == "aligned_grid_3x3":
                readout_pts = self._candidate_aligned_grid_points(readout_quads, grid_size=3)
                points_per_candidate = 9
            elif self.proposal_quality_readout_points == "center_corners_edges":
                edge_midpoints = 0.5 * (readout_quads + torch.roll(readout_quads, shifts=-1, dims=-2))
                readout_pts = torch.cat([cand_center, readout_quads, edge_midpoints], dim=3)
                points_per_candidate = 9
            else:
                readout_pts = torch.cat([cand_center, readout_quads], dim=3)
                points_per_candidate = 5
            readout_pts = readout_pts.reshape(b, 16, k_select * points_per_candidate, 2)
            selector_readout = self._sample_traj_readout_features(feat16_map, readout_pts)
            selector_readout = selector_readout.reshape(b, 16, k_select, points_per_candidate, -1)
            decoupled_selector_readout = self.proposal_decoupled_selector_readout_proj(selector_readout).mean(dim=3)

        if self.proposal_active_tip_selector:
            tip_feats = self._active_tip_candidate_features(
                candidate_quads,
                tool_kpt_vec_16,
                role_features_16,
                tool_id,
            )
            active_tip_delta = self.proposal_active_tip_selector_head(tip_feats).squeeze(-1)
            active_tip_logits = active_tip_delta
            if self.proposal_active_tip_selector_aux_only:
                if (not self.training) and self.proposal_active_tip_selector_eval_fuse_scale != 0.0:
                    quality_logits = (
                        quality_logits
                        + self.proposal_active_tip_selector_eval_fuse_scale * active_tip_delta
                    )
            else:
                quality_logits = quality_logits + self.proposal_active_tip_selector_scale * active_tip_delta

        if self.proposal_trajectory_rerank:
            traj_in = torch.cat(
                [
                    tool_kpt_vec_16.to(device=device, dtype=dtype),
                    role_features_16.to(device=device, dtype=dtype),
                ],
                dim=-1,
            )
            traj_mem = self.proposal_traj_mlp(traj_in)
            traj_mem = traj_mem + z[:, None, :] + self.proposal_traj_time_pos.to(device=device, dtype=dtype)
            cand_q = candidate_tokens.reshape(b * 16, k_select, -1)
            traj_kv = traj_mem[:, None, :, :].expand(-1, 16, -1, -1).reshape(b * 16, 16, -1)
            traj_ctx, _ = self.proposal_traj_attn(
                self.proposal_traj_query_norm(cand_q),
                self.proposal_traj_memory_norm(traj_kv),
                self.proposal_traj_memory_norm(traj_kv),
                need_weights=False,
            )
            traj_ctx = traj_ctx.reshape(b, 16, k_select, -1)
            cand_geom = candidate_quads.reshape(b, 16, k_select, 8)
            traj_delta = self.proposal_traj_head(
                torch.cat([candidate_tokens, traj_ctx, cand_geom], dim=-1)
            ).squeeze(-1)
            quality_logits = quality_logits + self.proposal_trajectory_rerank_scale * traj_delta

        if self.trajectory_anchor_logit_bias_scale != 0.0 and self.use_tool_keypoints:
            kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
            kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3, 1).to(device=device, dtype=dtype).clamp(0.0, 1.0)
            vis_sum = kpt_vis.sum(dim=2).clamp_min(1e-4)
            anchor_center = (kpt_xy * kpt_vis).sum(dim=2) / vis_sum
            cand_center = candidate_quads.mean(dim=-2)
            center_dist = torch.linalg.norm(cand_center - anchor_center[:, :, None, :], dim=-1)
            prior = -center_dist / self.trajectory_anchor_logit_bias_radius
            if self.trajectory_anchor_axis_bias_weight != 0.0:
                base_axis = kpt_xy[:, :, 0] - kpt_xy[:, :, -1]
                fallback = torch.zeros_like(base_axis)
                fallback[..., 0] = 1.0
                axis_norm = base_axis.norm(dim=-1, keepdim=True)
                base_axis = torch.where(axis_norm > 1e-4, base_axis / axis_norm.clamp_min(1e-4), fallback)
                cand_dirs, _cand_lens = quad_axes(candidate_quads.reshape(-1, 4, 2))
                cand_axis = cand_dirs[..., 0, :].reshape(b, 16, k_select, 2)
                axis_align = (cand_axis * base_axis[:, :, None, :]).sum(dim=-1).abs().pow(2)
                prior = prior + self.trajectory_anchor_axis_bias_weight * axis_align
            quality_logits = quality_logits + self.trajectory_anchor_logit_bias_scale * prior

        if self.proposal_source_calibration:
            source_bias = self.proposal_source_logit_bias(proposal_source_ids).squeeze(-1)
            quality_logits = quality_logits + self.proposal_source_calibration_scale * source_bias

        support_logits_8 = torch.logsumexp(center_logits_8.float(), dim=2) - math.log(float(max(k, 1)))
        proposal_support_overlap_scores = None
        proposal_support_geom_scores = None

        handoff_source_router_logits = None
        handoff_source_router_candidate_logits = None
        if self.proposal_handoff_source_router and proposal_source_ids.shape == quality_logits.shape:
            router_geom = candidate_quads.reshape(b, 16, k_select, 8)
            router_in = torch.cat(
                [
                    candidate_tokens,
                    router_geom,
                    quality_logits.detach().unsqueeze(-1),
                ],
                dim=-1,
            )
            handoff_source_router_candidate_logits = (
                self.proposal_handoff_source_router_candidate_head(router_in).squeeze(-1)
            )
            handoff_source_ids = proposal_source_ids.detach().clamp(0, 7)
            neg = torch.finfo(handoff_source_router_candidate_logits.dtype).min
            group_logits = []
            group_present = []
            for source_id in range(8):
                source_mask = handoff_source_ids == int(source_id)
                group_present.append(source_mask.any(dim=-1))
                source_scores = handoff_source_router_candidate_logits.masked_fill(~source_mask, neg)
                group_logits.append(source_scores.max(dim=-1).values)
            handoff_source_router_logits = torch.stack(group_logits, dim=-1)
            router_context = role16 + z[:, None, :].expand(-1, 16, -1)
            if self.use_tool_keypoints:
                router_context = router_context + self.tool_kpt_mlp(
                    tool_kpt_vec_16.to(device=device, dtype=dtype)
                )
            if self.use_tool_mask:
                mask_geom8 = self._tool_mask_geom8(tool_mask_8.to(device=device, dtype=dtype))
                mask_geom16 = F.interpolate(
                    mask_geom8.transpose(1, 2),
                    size=16,
                    mode="linear",
                    align_corners=True,
                ).transpose(1, 2)
                router_context = router_context + self.tool_mask_geom_mlp(mask_geom16)
            handoff_source_router_logits = (
                handoff_source_router_logits
                + self.proposal_handoff_source_router_context_head(router_context).to(
                    dtype=handoff_source_router_logits.dtype
                )
            )
            group_present_t = torch.stack(group_present, dim=-1)
            handoff_source_router_logits = handoff_source_router_logits.masked_fill(
                ~group_present_t,
                neg,
            )
            handoff_bonus = handoff_source_router_logits.gather(-1, handoff_source_ids)
            quality_logits = quality_logits + float(self.proposal_handoff_source_router_scale) * handoff_bonus.to(
                device=quality_logits.device,
                dtype=quality_logits.dtype,
            )
            if abs(float(self.proposal_handoff_source_router_candidate_select_scale)) > 0.0:
                quality_logits = quality_logits + float(
                    self.proposal_handoff_source_router_candidate_select_scale
                ) * handoff_source_router_candidate_logits.to(
                    device=quality_logits.device,
                    dtype=quality_logits.dtype,
                )
            if self.proposal_handoff_router_normalized_select and (not self.training):
                centered_logits = torch.full_like(quality_logits, -50.0)
                for source_id in range(8):
                    source_mask = handoff_source_ids == int(source_id)
                    if not bool(source_mask.any().detach().cpu()):
                        continue
                    source_count = source_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(
                        dtype=quality_logits.dtype
                    )
                    source_vals = quality_logits.masked_fill(~source_mask, 0.0)
                    source_mean = source_vals.sum(dim=-1, keepdim=True) / source_count
                    source_var = (
                        ((quality_logits - source_mean).masked_fill(~source_mask, 0.0) ** 2).sum(
                            dim=-1,
                            keepdim=True,
                        )
                        / source_count
                    )
                    source_std = torch.sqrt(source_var.clamp_min(1e-4))
                    centered_logits = torch.where(
                        source_mask,
                        (quality_logits - source_mean) / source_std,
                        centered_logits,
                    )
                router_bonus = handoff_source_router_logits.gather(-1, handoff_source_ids)
                quality_logits = (
                    float(self.proposal_handoff_router_normalized_within_scale)
                    * centered_logits
                    + float(self.proposal_handoff_router_normalized_source_scale)
                    * router_bonus.to(device=quality_logits.device, dtype=quality_logits.dtype)
                )

        if self.proposal_geom_temporal_fusion:
            geom_ctx = self._proposal_geom_temporal_context(
                candidate_tokens,
                candidate_quads,
                proposal_source_ids,
            )
            t_select = candidate_tokens.shape[1]
            cand_geom = candidate_quads.reshape(b, t_select, k_select, 8)
            token_delta = self.proposal_geom_temporal_token_proj(
                torch.cat([candidate_tokens, geom_ctx, cand_geom], dim=-1)
            )
            fusion_scale = float(self.proposal_geom_temporal_fusion_scale)
            candidate_tokens = candidate_tokens + fusion_scale * token_delta
            logit_delta = self.proposal_geom_temporal_logit_head(
                torch.cat(
                    [
                        candidate_tokens,
                        geom_ctx,
                        cand_geom,
                        quality_logits.detach().unsqueeze(-1),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            quality_logits = quality_logits + fusion_scale * logit_delta

        if self.proposal_clip_context_fusion:
            clip_ctx = self._proposal_clip_context(
                candidate_tokens,
                candidate_quads,
                proposal_source_ids,
            )
            cand_geom = candidate_quads.reshape(b, 16, k_select, 8)
            clip_scale = float(self.proposal_clip_context_scale)
            token_delta = self.proposal_clip_context_token_proj(
                torch.cat([candidate_tokens, clip_ctx, cand_geom], dim=-1)
            )
            candidate_tokens = candidate_tokens + clip_scale * token_delta
            logit_delta = self.proposal_clip_context_logit_head(
                torch.cat(
                    [
                        candidate_tokens,
                        clip_ctx,
                        cand_geom,
                        quality_logits.detach().unsqueeze(-1),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            quality_logits = quality_logits + clip_scale * logit_delta

        if self.proposal_frame_anchor_memory:
            frame_ctx, frame_anchor_logits, frame_anchor_weights = self._proposal_frame_anchor_memory(
                candidate_tokens,
                candidate_quads,
                proposal_source_ids,
                quality_logits,
            )
            cand_geom = candidate_quads.reshape(b, 16, k_select, 8)
            anchor_scale = float(self.proposal_frame_anchor_scale)
            token_delta = self.proposal_frame_anchor_token_proj(
                torch.cat([candidate_tokens, frame_ctx, cand_geom], dim=-1)
            )
            candidate_tokens = candidate_tokens + anchor_scale * token_delta
            logit_delta = self.proposal_frame_anchor_logit_head(
                torch.cat(
                    [
                        candidate_tokens,
                        frame_ctx,
                        cand_geom,
                        quality_logits.detach().unsqueeze(-1),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            quality_logits = quality_logits + anchor_scale * logit_delta

        if self.proposal_support_overlap_score and float(self.proposal_support_overlap_scale) != 0.0:
            support_for_score = support_logits_8.detach() if self.proposal_support_overlap_detach else support_logits_8
            support_flat = support_for_score.reshape(b, t8, h * w).transpose(1, 2)
            support16_flat = F.interpolate(
                support_flat,
                size=16,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
            support16 = support16_flat.reshape(b, 16, h, w)
            score_grid = int(self.proposal_support_overlap_grid_size)
            if score_grid != h:
                support16 = F.interpolate(
                    support16.reshape(b * 16, 1, h, w),
                    size=(score_grid, score_grid),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, 16, score_grid, score_grid)
            support_prob = support16.sigmoid().to(dtype=candidate_quads.dtype).unsqueeze(2)
            cand_mask = quad_soft_inside_grid(
                candidate_quads.reshape(-1, 4, 2),
                grid_size=score_grid,
                edge_sharpness=96.0,
            ).to(dtype=candidate_quads.dtype).reshape(b, 16, k_select, score_grid, score_grid)
            mask_area = cand_mask.sum(dim=(-1, -2)).clamp_min(1.0)
            support_mean = (cand_mask * support_prob).sum(dim=(-1, -2)) / mask_area
            proposal_support_overlap_scores = torch.logit(support_mean.clamp(1e-4, 1.0 - 1e-4))
            quality_logits = quality_logits + float(self.proposal_support_overlap_scale) * proposal_support_overlap_scores.to(
                dtype=quality_logits.dtype
            )

        if self.proposal_support_geom_score and float(self.proposal_support_geom_scale) != 0.0:
            support_for_geom = support_logits_8.detach() if self.proposal_support_geom_detach else support_logits_8
            support_flat = support_for_geom.reshape(b, t8, h * w).transpose(1, 2)
            support16_flat = F.interpolate(
                support_flat,
                size=16,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
            support16 = support16_flat.reshape(b, 16, h, w)
            geom_grid = int(self.proposal_support_geom_grid_size)
            if geom_grid != h:
                support16 = F.interpolate(
                    support16.reshape(b * 16, 1, h, w),
                    size=(geom_grid, geom_grid),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, 16, geom_grid, geom_grid)
            support_prob = support16.sigmoid().to(dtype=candidate_quads.dtype)
            coords = torch.linspace(
                0.0,
                1.0,
                geom_grid,
                device=device,
                dtype=candidate_quads.dtype,
            )
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            grid_xy = torch.stack([xx, yy], dim=-1)
            support_mass = support_prob.sum(dim=(-1, -2)).clamp_min(1e-4)
            support_center = (support_prob[..., None] * grid_xy).sum(dim=(-3, -2)) / support_mass[..., None]
            support_area = (support_mass / float(geom_grid * geom_grid)).clamp_min(1e-5)
            cand_center = candidate_quads.mean(dim=-2)
            cand_area = quad_polygon_area(candidate_quads.reshape(-1, 4, 2)).reshape(b, 16, k_select)
            center_err = (cand_center - support_center[:, :, None, :]).norm(dim=-1)
            log_area_err = torch.abs(torch.log(cand_area.clamp_min(1e-5) / support_area[:, :, None]))
            raw_geom_score = -(
                center_err / float(self.proposal_support_geom_center_tau)
                + log_area_err / float(self.proposal_support_geom_area_tau)
            )
            score_mean = raw_geom_score.mean(dim=-1, keepdim=True)
            score_std = raw_geom_score.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
            proposal_support_geom_scores = (raw_geom_score - score_mean) / score_std
            quality_logits = quality_logits + float(self.proposal_support_geom_scale) * proposal_support_geom_scores.to(
                dtype=quality_logits.dtype
            )

        decoupled_selector_logits = None
        pre_decoupled_quality_logits = None
        if self.proposal_decoupled_selector:
            pre_decoupled_quality_logits = quality_logits
            selector_tokens = candidate_tokens
            selector_geom = candidate_quads.reshape(b, 16, k_select, 8)
            selector_base = quality_logits.unsqueeze(-1)
            if self.proposal_decoupled_selector_detach:
                selector_tokens = selector_tokens.detach()
                selector_geom = selector_geom.detach()
                selector_base = selector_base.detach()
            selector_source = proposal_source_ids.detach().clamp(0, 7)
            selector_source_feat = self.proposal_decoupled_selector_source_emb(selector_source).to(
                dtype=selector_tokens.dtype
            )
            selector_parts = [selector_tokens, selector_source_feat, selector_geom, selector_base]
            if decoupled_selector_readout is not None:
                selector_parts.insert(2, decoupled_selector_readout.to(dtype=selector_tokens.dtype))
            selector_in = torch.cat(selector_parts, dim=-1)
            selector_delta = self.proposal_decoupled_selector_head(selector_in).squeeze(-1)
            if self.proposal_decoupled_selector_replace_logits:
                decoupled_selector_logits = (
                    float(self.proposal_decoupled_selector_base_scale)
                    * quality_logits.detach()
                    + float(self.proposal_decoupled_selector_scale) * selector_delta
                )
                quality_logits = decoupled_selector_logits
            else:
                decoupled_selector_logits = quality_logits + (
                    float(self.proposal_decoupled_selector_scale) * selector_delta
                )
                quality_logits = decoupled_selector_logits

        base_quality_logits = quality_logits
        oicr_refine_logits = None
        if self.proposal_oicr_refine_head_enabled:
            # PCL/OICR trains separate refinement classifiers from the base
            # MIL score.  Here the base selector logit is a detached input, and
            # the new head learns a residual refinement score.  Zero init makes
            # the handoff start exactly from the existing selector.
            oicr_in = torch.cat(
                [
                    candidate_tokens,
                    candidate_quads.reshape(b, 16, k_select, 8),
                    base_quality_logits.detach().unsqueeze(-1),
                ],
                dim=-1,
            )
            oicr_refine_logits = base_quality_logits + self.proposal_oicr_refine_head(oicr_in).squeeze(-1)
            if self.proposal_oicr_refine_use_for_selection:
                quality_logits = oicr_refine_logits

        select_quality_logits = quality_logits
        if self.training:
            noise_std = float(self.proposal_selection_train_logit_noise_std)
            if noise_std > 0.0:
                select_quality_logits = select_quality_logits + torch.randn_like(select_quality_logits) * noise_std
            drop_prob = float(self.proposal_selection_train_drop_prob)
            if drop_prob > 0.0 and k_select > 1:
                keep = torch.rand_like(select_quality_logits.float()) >= drop_prob
                has_keep = keep.any(dim=-1, keepdim=True)
                fallback_idx = quality_logits.detach().argmax(dim=-1, keepdim=True)
                fallback_keep = torch.zeros_like(keep, dtype=torch.bool).scatter_(-1, fallback_idx, True)
                keep = torch.where(has_keep, keep, fallback_keep)
                select_quality_logits = select_quality_logits.masked_fill(~keep, -50.0)

        use_hard = self.proposal_selection_mode == "hard" or (
            self.proposal_selection_mode == "train_soft_eval_hard" and not self.training
        )
        if use_hard:
            best = select_quality_logits.argmax(dim=-1)
            weights = F.one_hot(best, num_classes=k_select).to(dtype=select_quality_logits.dtype)
        else:
            select_logits = select_quality_logits / self.proposal_selection_temperature
            topk = min(int(self.proposal_selection_topk), k_select)
            if topk > 0 and topk < k_select:
                topk_idx = select_logits.topk(topk, dim=-1).indices
                masked_logits = torch.full_like(select_logits, torch.finfo(select_logits.dtype).min)
                select_logits = masked_logits.scatter(-1, topk_idx, select_logits.gather(-1, topk_idx))
            weights_soft = F.softmax(select_logits, dim=-1)
            if self.proposal_selection_straight_through:
                best = weights_soft.argmax(dim=-1)
                weights_hard = F.one_hot(best, num_classes=k_select).to(dtype=weights_soft.dtype)
                weights = weights_hard - weights_soft.detach() + weights_soft
            else:
                weights = weights_soft
        soft_quad = (candidate_quads * weights[..., None, None]).sum(dim=2).clamp(0.0, 1.0)
        quad = soft_quad
        if self.proposal_final_fusion:
            cand_geom = candidate_quads.reshape(b, 16, k_select, 8)
            quality_stat = torch.tanh(quality_logits)[..., None]
            cand_weight = weights[..., None]
            cand_fusion = self.proposal_final_fusion_candidate_mlp(
                torch.cat([candidate_tokens, cand_geom, cand_weight, quality_stat], dim=-1)
            )
            fused_token = (cand_fusion * cand_weight).sum(dim=2)
            weight_max = weights.max(dim=-1).values
            weight_entropy = -(
                weights.clamp_min(1e-6) * weights.clamp_min(1e-6).log()
            ).sum(dim=-1) / math.log(max(k_select, 2))
            quality_tanh = torch.tanh(quality_logits)
            quality_mean = (weights * quality_tanh).sum(dim=-1)
            quality_max = quality_tanh.max(dim=-1).values
            fusion_stats = torch.stack([weight_max, weight_entropy, quality_mean, quality_max], dim=-1)
            fusion_delta = torch.tanh(
                self.proposal_final_fusion_head(
                    torch.cat([fused_token, soft_quad.reshape(b, 16, 8), fusion_stats], dim=-1)
                )
            ).reshape(b, 16, 4, 2)
            _fusion_dirs, fusion_lens = quad_axes(soft_quad.reshape(-1, 4, 2))
            fusion_step = fusion_lens.mean(dim=-1).reshape(b, 16, 1, 1).clamp(0.015, 0.35)
            quad = (soft_quad + self.proposal_final_fusion_scale * fusion_delta * fusion_step).clamp(0.0, 1.0)
        set_refiner_weights = None
        set_refiner_base_quad = None
        set_refiner_scores = None
        if self.proposal_set_refiner:
            set_tokens = candidate_tokens
            set_quads = candidate_quads.reshape(b, 16, k_select, 8)
            set_logits = quality_logits
            if self.proposal_set_refiner_detach:
                set_tokens = set_tokens.detach()
                set_quads = set_quads.detach()
                set_logits = set_logits.detach()
            set_source = proposal_source_ids.detach().clamp(0, 7)
            set_source_feat = self.proposal_set_refiner_source_emb(set_source).to(dtype=set_tokens.dtype)
            set_item = self.proposal_set_refiner_item_mlp(
                torch.cat([set_tokens, set_source_feat, set_quads, set_logits.unsqueeze(-1)], dim=-1)
            )
            set_score_delta = self.proposal_set_refiner_score_head(set_item).squeeze(-1)
            set_scores = set_logits / float(self.proposal_set_refiner_temperature) + set_score_delta
            set_topk = min(int(self.proposal_set_refiner_topk), k_select)
            if set_topk > 0 and set_topk < k_select:
                topk_idx = set_logits.detach().topk(set_topk, dim=-1).indices
                masked_scores = torch.full_like(set_scores, torch.finfo(set_scores.dtype).min)
                set_scores = masked_scores.scatter(-1, topk_idx, set_scores.gather(-1, topk_idx))
            set_refiner_scores = set_scores
            set_refiner_weights = F.softmax(set_scores, dim=-1)
            set_candidate_quads = candidate_quads.detach() if self.proposal_set_refiner_detach else candidate_quads
            set_refiner_base_quad = (
                set_candidate_quads * set_refiner_weights[..., None, None]
            ).sum(dim=2).clamp(0.0, 1.0)
            set_pooled = (set_item * set_refiner_weights[..., None]).sum(dim=2)
            set_entropy = -(
                set_refiner_weights.clamp_min(1e-6)
                * set_refiner_weights.clamp_min(1e-6).log()
            ).sum(dim=-1) / math.log(max(k_select, 2))
            set_stats = torch.stack(
                [
                    set_refiner_weights.max(dim=-1).values,
                    set_entropy,
                    torch.tanh(set_logits).mean(dim=-1),
                    torch.tanh(set_logits).max(dim=-1).values,
                ],
                dim=-1,
            )
            set_delta = torch.tanh(
                self.proposal_set_refiner_delta_head(
                    torch.cat([set_pooled, set_refiner_base_quad.reshape(b, 16, 8), set_stats], dim=-1)
                )
            ).reshape(b, 16, 4, 2)
            _set_dirs, set_lens = quad_axes(set_refiner_base_quad.reshape(-1, 4, 2))
            set_step = set_lens.mean(dim=-1).reshape(b, 16, 1, 1).clamp(0.015, 0.35)
            quad = (
                set_refiner_base_quad
                + float(self.proposal_set_refiner_scale) * set_delta * set_step
            ).clamp(0.0, 1.0)
        box_xyxy = quad_to_xyxy(quad, min_size=0.02)
        out = {
            "boxes_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "boxes_xyxy": box_xyxy,
            "center_logits_8": center_logits_8.mean(dim=2).view(b, t8, h, w),
            "support_logits_8": support_logits_8.to(dtype=center_logits_8.dtype).view(b, t8, h, w),
            "coarse_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "coarse_xyxy": box_xyxy,
            "quad_xy": quad,
            "proposal_cxcywh": candidate_cxcywh,
            "proposal_xyxy": candidate_xyxy,
            "proposal_quads": candidate_quads,
            "proposal_logits": quality_logits,
            "proposal_source_ids": proposal_source_ids,
            "proposal_selection_weights": weights,
            "corner_logits_8": corner_logits_8,
        }
        if oicr_refine_logits is not None:
            out["proposal_base_logits"] = base_quality_logits
            out["proposal_oicr_logits"] = oicr_refine_logits
        if decoupled_selector_logits is not None:
            out["proposal_decoupled_selector_logits"] = decoupled_selector_logits
        if pre_decoupled_quality_logits is not None:
            out["proposal_pre_decoupled_logits"] = pre_decoupled_quality_logits
        if proposal_support_overlap_scores is not None:
            out["proposal_support_overlap_scores"] = proposal_support_overlap_scores
        if proposal_support_geom_scores is not None:
            out["proposal_support_geom_scores"] = proposal_support_geom_scores
        if self.proposal_final_fusion:
            out["proposal_fusion_base_quad"] = soft_quad
        if set_refiner_weights is not None:
            out["proposal_set_refiner_weights"] = set_refiner_weights
            out["proposal_set_refiner_base_quad"] = set_refiner_base_quad
        if set_refiner_scores is not None:
            out["proposal_set_refiner_scores"] = set_refiner_scores
        if active_tip_logits is not None:
            out["proposal_active_tip_logits"] = active_tip_logits
        if frame_anchor_logits is not None:
            out["proposal_frame_anchor_logits"] = frame_anchor_logits
        if frame_anchor_weights is not None:
            out["proposal_frame_anchor_weights"] = frame_anchor_weights
        if handoff_source_router_logits is not None:
            out["proposal_handoff_source_router_logits"] = handoff_source_router_logits
        if handoff_source_router_candidate_logits is not None:
            out["proposal_handoff_source_router_candidate_logits"] = handoff_source_router_candidate_logits
        if (self.condition_prior_enabled or self.h259_prior_proposal) and cond_quad is not None:
            out["condition_prior_quad"] = cond_quad
            out["condition_prior_xyxy"] = cond_xyxy
            out["condition_prior_cxcywh"] = cond_cxcywh
        if self.h259_prior_proposal and h259_quad is not None:
            out["h259_prior_quad"] = h259_quad
            out["h259_prior_xyxy"] = h259_xyxy
            out["h259_prior_cxcywh"] = h259_cxcywh
        if traj_query_quad is not None and traj_query_xyxy is not None and traj_query_cxcywh is not None:
            out["trajectory_query_quad"] = traj_query_quad
            out["trajectory_query_xyxy"] = traj_query_xyxy
            out["trajectory_query_cxcywh"] = traj_query_cxcywh
        if corner_proposal_quad is not None:
            out["corner_proposal_quad"] = corner_proposal_quad
        if token_affinity_proposal_quad is not None:
            out["token_affinity_proposal_quad"] = token_affinity_proposal_quad
        if support_moment_proposal_quad is not None:
            out["support_moment_proposal_quad"] = support_moment_proposal_quad
        return out


class H210ProposalPointTemporalRegion(H191QuadLocalPointTemporalRegion):
    """Proposal/rerank Stage1 plus CoTracker-style four-point Stage2."""

    def __init__(
        self,
        hidden_dim: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        residual_scale: float,
        decoder_layers: int,
        query_noise_scale: float = 0.0,
        dn_query_noise_scale: float = 0.0,
        dn_query_groups: int = 1,
        dn_query_noise_min_scale: float = 1.0,
        use_proposal_oracle_refiner: bool = False,
        proposal_oracle_refiner_noise_scale: float = 0.0,
        stage2_bridge_noise_mode: str = "local_only",
        use_tool_trajectory_bias: bool = False,
        use_tool_trajectory_attn: bool = False,
        use_tool_corner_query_bias: bool = False,
        stage1_drop_tool_keypoints: bool = False,
        detach_stage1_refiner: bool = False,
        proposal_count: int = 6,
        proposal_selection_mode: str = "train_soft_eval_hard",
        proposal_selection_temperature: float = 1.0,
        proposal_selection_topk: int = 0,
        proposal_selection_straight_through: bool = False,
        proposal_selection_train_logit_noise_std: float = 0.0,
        proposal_selection_train_drop_prob: float = 0.0,
        proposal_geom_temporal_fusion: bool = False,
        proposal_geom_temporal_fusion_scale: float = 0.30,
        proposal_geom_temporal_center_scale: float = 0.18,
        proposal_geom_temporal_source_penalty: float = 4.0,
        proposal_clip_context_fusion: bool = False,
        proposal_clip_context_scale: float = 0.22,
        proposal_clip_context_dropout: float = 0.10,
        proposal_frame_anchor_memory: bool = False,
        proposal_frame_anchor_scale: float = 0.20,
        proposal_frame_anchor_topk: int = 2,
        proposal_frame_anchor_temperature: float = 0.12,
        proposal_frame_anchor_candidate_temperature: float = 0.45,
        proposal_final_fusion: bool = False,
        proposal_final_fusion_scale: float = 0.18,
        proposal_quality_readout: bool = False,
        proposal_quality_readout_detach: bool = False,
        proposal_quality_readout_replace_logits: bool = False,
        proposal_quality_readout_traj_context: bool = False,
        proposal_quality_readout_points: str = "center_corners",
        proposal_quality_readout_pool: str = "mean",
        proposal_support_overlap_score: bool = False,
        proposal_support_overlap_scale: float = 0.0,
        proposal_support_overlap_grid_size: int = 24,
        proposal_support_overlap_detach: bool = False,
        proposal_support_geom_score: bool = False,
        proposal_support_geom_scale: float = 0.0,
        proposal_support_geom_grid_size: int = 24,
        proposal_support_geom_center_tau: float = 0.07,
        proposal_support_geom_area_tau: float = 0.75,
        proposal_support_geom_detach: bool = False,
        proposal_support_moment_proposal: bool = False,
        proposal_support_moment_top_frac: float = 0.035,
        proposal_support_moment_size_scale: float = 1.8,
        proposal_support_moment_temperature: float = 0.16,
        proposal_support_moment_detach: bool = False,
        proposal_decoupled_selector: bool = False,
        proposal_decoupled_selector_detach: bool = False,
        proposal_decoupled_selector_replace_logits: bool = False,
        proposal_decoupled_selector_scale: float = 1.0,
        proposal_decoupled_selector_base_scale: float = 0.0,
        proposal_decoupled_selector_readout: bool = False,
        proposal_oicr_refine_head: bool = False,
        proposal_oicr_refine_use_for_selection: bool = False,
        proposal_handoff_oicr_refine_times: int = 0,
        proposal_handoff_oicr_refine_scale: float = 1.0,
        proposal_handoff_oicr_refine_aggregate: str = "last",
        proposal_handoff_oicr_refine_detach: bool = False,
        proposal_handoff_oicr_refine_use_for_selection: bool = False,
        refined_quality_readout: bool = False,
        refined_quality_readout_detach: bool = False,
        refined_quality_readout_replace_logits: bool = False,
        refined_quality_readout_points: str = "aligned_grid_3x3",
        refiner_aligned_context: bool = False,
        refiner_aligned_context_points: str = "aligned_grid_3x3",
        refiner_aligned_context_scale: float = 0.35,
        proposal_pointset_refine: bool = False,
        proposal_pointset_refine_scale: float = 0.18,
        proposal_pointset_refine_grad_mul: float = 0.25,
        proposal_pointset_quality_scale: float = 1.0,
        proposal_pointset_refine_update_token: bool = False,
        proposal_set_refiner: bool = False,
        proposal_set_refiner_topk: int = 0,
        proposal_set_refiner_scale: float = 0.22,
        proposal_set_refiner_temperature: float = 0.90,
        proposal_set_refiner_detach: bool = False,
        proposal_local_corr_score: bool = False,
        proposal_local_corr_scale: float = 0.45,
        proposal_local_corr_grid_size: int = 3,
        proposal_local_corr_temperature: float = 0.08,
        proposal_local_corr_detach: bool = False,
        proposal_active_tip_selector: bool = False,
        proposal_active_tip_selector_scale: float = 1.0,
        proposal_active_tip_selector_aux_only: bool = False,
        proposal_active_tip_selector_eval_fuse_scale: float = 0.0,
        proposal_active_tip_feature_fusion: bool = False,
        proposal_active_tip_feature_scale: float = 0.35,
        proposal_trajectory_rerank: bool = False,
        proposal_trajectory_rerank_scale: float = 0.45,
        proposal_trajectory_refine: bool = False,
        proposal_trajectory_refine_scale: float = 0.30,
        refine_all_proposals: bool = False,
        refine_proposal_topk: int = 0,
        refine_recollect_stage1: bool = False,
        refine_recollect_refined_base_only: bool = False,
        condition_prior_proposal: bool = False,
        h259_prior_proposal: bool = False,
        trajectory_query_proposal_count: int = 0,
        trajectory_query_output_mode: str = "oriented_rect",
        trajectory_readout_bias: bool = False,
        condition_prior_readout_bias: bool = False,
        condition_prior_readout_scale: float = 0.45,
        trajectory_anchor_logit_bias_scale: float = 0.0,
        trajectory_anchor_logit_bias_radius: float = 0.22,
        trajectory_anchor_axis_bias_weight: float = 0.0,
        corner_heatmap_proposal: bool = False,
        corner_heatmap_proposal_temperature: float = 0.35,
        corner_heatmap_proposal_detach: bool = False,
        proposal_source_calibration: bool = False,
        proposal_source_calibration_scale: float = 1.0,
        proposal_handoff_source_calibration: bool = False,
        proposal_handoff_source_calibration_scale: float = 1.0,
        proposal_handoff_source4_init: float = 0.0,
        proposal_handoff_source_router: bool = False,
        proposal_handoff_source_router_scale: float = 1.0,
        proposal_handoff_source_router_candidate_select_scale: float = 0.0,
        proposal_handoff_source5_readout: bool = False,
        proposal_handoff_source5_readout_scale: float = 1.0,
        proposal_handoff_source5_readout_grid_size: int = 3,
        proposal_handoff_source5_readout_sources: str = "5",
        proposal_handoff_source5_readout_detach: bool = False,
        proposal_handoff_mask_quality: bool = False,
        proposal_handoff_mask_quality_scale: float = 0.50,
        proposal_handoff_mask_grid_size: int = 7,
        proposal_handoff_mask_expand_scale: float = 1.25,
        proposal_handoff_mask_refine_scale: float = 0.0,
        proposal_handoff_mask_refine_size_scale: float = 2.0,
        proposal_handoff_mask_refine_min_size_px: float = 2.0,
        proposal_handoff_mask_refine_max_ratio: float = 1.05,
        proposal_handoff_mask_quality_sources: str = "5,7",
        proposal_handoff_mask_detach: bool = False,
        proposal_handoff_mask_score_detach: bool = False,
        proposal_handoff_rank_adaptive_scorer: bool = False,
        proposal_handoff_rank_adaptive_scale: float = 0.45,
        proposal_handoff_rank_adaptive_max_rank: int = 512,
        proposal_handoff_rank_adaptive_detach: bool = False,
        proposal_handoff_sparse_set_scorer: bool = False,
        proposal_handoff_sparse_set_scorer_scale: float = 0.35,
        proposal_handoff_sparse_set_scorer_sources: str = "0,5,7",
        proposal_handoff_sparse_set_scorer_detach: bool = False,
        proposal_handoff_source5_refiner: bool = False,
        proposal_handoff_source5_refiner_topk: int = 12,
        proposal_handoff_source5_refiner_temperature: float = 0.75,
        proposal_handoff_source5_refiner_scale: float = 0.35,
        proposal_handoff_source5_refiner_logit_scale: float = 0.65,
        proposal_handoff_source5_refiner_init_logit_bias: float = 0.0,
        proposal_handoff_source5_refiner_detach: bool = False,
        proposal_handoff_source5_refiner_source_id: int = 5,
        proposal_handoff_source5_refiner_output_source_id: int = 6,
        proposal_handoff_topk_query_refiner: bool = False,
        proposal_handoff_topk_query_sources: str = "0,5",
        proposal_handoff_topk_query_topk_per_source: int = 12,
        proposal_handoff_topk_query_grid_size: int = 3,
        proposal_handoff_topk_query_scale: float = 0.30,
        proposal_handoff_topk_query_logit_scale: float = 0.75,
        proposal_handoff_topk_query_output_source_id: int = 6,
        proposal_handoff_topk_query_replace_logits: bool = False,
        proposal_handoff_topk_query_detach: bool = False,
        proposal_handoff_final_fusion: bool = False,
        proposal_handoff_final_fusion_scale: float = 0.18,
        proposal_handoff_detach_selection_geometry: bool = False,
        proposal_handoff_source_representatives_only: bool = False,
        proposal_handoff_source5_topk: int = 0,
        proposal_handoff_router_gated_source: bool = False,
        proposal_handoff_router_normalized_select: bool = False,
        proposal_handoff_router_normalized_source_scale: float = 1.0,
        proposal_handoff_router_normalized_within_scale: float = 1.0,
        token_affinity_proposal: bool = False,
        token_affinity_proposal_count: int = 1,
        token_affinity_proposal_temperature: float = 0.08,
        token_affinity_proposal_top_frac: float = 0.055,
        token_affinity_proposal_size_scale: float = 3.6,
        token_affinity_proposal_extra_top_fracs: str = "",
        token_affinity_proposal_extra_size_scales: str = "",
        token_affinity_proposal_detach: bool = True,
        topology_contact_template_proposal: bool = False,
        topology_contact_template_sizes_px: str = "4x3,6x4,8x4,10x5",
        topology_contact_template_offsets_px: str = "-2,0,2",
        topology_contact_template_logit_bias: float = 0.0,
        topology_contact_template_index_embedding: bool = False,
        topology_contact_template_combo_bias: bool = False,
        topology_contact_template_combo_bias_scale: float = 1.0,
        topology_contact_template_combo_allowlist: str = "",
        local_grid_refine: bool = False,
        local_grid_refine_grid_size: int = 5,
        local_grid_refine_scale: float = 1.35,
        local_grid_refine_temperature: float = 0.20,
        local_grid_refine_prior_scale: float = 6.0,
        local_grid_refine_prior_sigma: float = 0.28,
        local_grid_refine_detach_base: bool = False,
        local_grid_final_refine: bool = True,
        local_grid_confidence_score: bool = False,
        local_grid_confidence_scale: float = 0.35,
        local_grid_confidence_detach: bool = False,
        candidate_mask_quality: bool = False,
        candidate_mask_grid_size: int = 7,
        candidate_mask_score_scale: float = 0.0,
        candidate_mask_score_detach: bool = False,
        use_rgb_sidecar: bool = False,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            use_tool_mask=use_tool_mask,
            use_tool_keypoints=use_tool_keypoints,
            residual_scale=residual_scale,
            decoder_layers=decoder_layers,
            query_noise_scale=query_noise_scale,
            dn_query_noise_scale=dn_query_noise_scale,
            dn_query_groups=dn_query_groups,
            dn_query_noise_min_scale=dn_query_noise_min_scale,
            use_proposal_oracle_refiner=use_proposal_oracle_refiner,
            proposal_oracle_refiner_noise_scale=proposal_oracle_refiner_noise_scale,
            stage2_bridge_noise_mode=stage2_bridge_noise_mode,
            use_tool_trajectory_bias=use_tool_trajectory_bias,
            use_tool_trajectory_attn=use_tool_trajectory_attn,
            use_tool_corner_query_bias=use_tool_corner_query_bias,
            stage1_drop_tool_keypoints=stage1_drop_tool_keypoints,
            detach_stage1_refiner=detach_stage1_refiner,
            local_grid_refine=local_grid_refine,
            local_grid_refine_grid_size=local_grid_refine_grid_size,
            local_grid_refine_scale=local_grid_refine_scale,
            local_grid_refine_temperature=local_grid_refine_temperature,
            local_grid_refine_prior_scale=local_grid_refine_prior_scale,
            local_grid_refine_prior_sigma=local_grid_refine_prior_sigma,
            local_grid_refine_detach_base=local_grid_refine_detach_base,
            local_grid_final_refine=local_grid_final_refine,
            use_rgb_sidecar=use_rgb_sidecar,
        )
        stage1_use_tool_keypoints = bool(use_tool_keypoints) and not bool(stage1_drop_tool_keypoints)
        self.stage1 = H210ProposalOrientedStage1(
            hidden_dim=hidden_dim,
            use_tool_mask=use_tool_mask,
            use_tool_keypoints=stage1_use_tool_keypoints,
            residual_scale=residual_scale,
            proposal_count=proposal_count,
            proposal_selection_mode=proposal_selection_mode,
            proposal_selection_temperature=proposal_selection_temperature,
            proposal_selection_topk=proposal_selection_topk,
            proposal_selection_straight_through=proposal_selection_straight_through,
            proposal_selection_train_logit_noise_std=proposal_selection_train_logit_noise_std,
            proposal_selection_train_drop_prob=proposal_selection_train_drop_prob,
            proposal_geom_temporal_fusion=proposal_geom_temporal_fusion,
            proposal_geom_temporal_fusion_scale=proposal_geom_temporal_fusion_scale,
            proposal_geom_temporal_center_scale=proposal_geom_temporal_center_scale,
            proposal_geom_temporal_source_penalty=proposal_geom_temporal_source_penalty,
            proposal_clip_context_fusion=proposal_clip_context_fusion,
            proposal_clip_context_scale=proposal_clip_context_scale,
            proposal_clip_context_dropout=proposal_clip_context_dropout,
            proposal_frame_anchor_memory=proposal_frame_anchor_memory,
            proposal_frame_anchor_scale=proposal_frame_anchor_scale,
            proposal_frame_anchor_topk=proposal_frame_anchor_topk,
            proposal_frame_anchor_temperature=proposal_frame_anchor_temperature,
            proposal_frame_anchor_candidate_temperature=proposal_frame_anchor_candidate_temperature,
            proposal_final_fusion=proposal_final_fusion,
            proposal_final_fusion_scale=proposal_final_fusion_scale,
            proposal_quality_readout=proposal_quality_readout,
            proposal_quality_readout_detach=proposal_quality_readout_detach,
            proposal_quality_readout_replace_logits=proposal_quality_readout_replace_logits,
            proposal_quality_readout_traj_context=proposal_quality_readout_traj_context,
            proposal_quality_readout_points=proposal_quality_readout_points,
            proposal_quality_readout_pool=proposal_quality_readout_pool,
            proposal_support_overlap_score=proposal_support_overlap_score,
            proposal_support_overlap_scale=proposal_support_overlap_scale,
            proposal_support_overlap_grid_size=proposal_support_overlap_grid_size,
            proposal_support_overlap_detach=proposal_support_overlap_detach,
            proposal_support_geom_score=proposal_support_geom_score,
            proposal_support_geom_scale=proposal_support_geom_scale,
            proposal_support_geom_grid_size=proposal_support_geom_grid_size,
            proposal_support_geom_center_tau=proposal_support_geom_center_tau,
            proposal_support_geom_area_tau=proposal_support_geom_area_tau,
            proposal_support_geom_detach=proposal_support_geom_detach,
            proposal_support_moment_proposal=proposal_support_moment_proposal,
            proposal_support_moment_top_frac=proposal_support_moment_top_frac,
            proposal_support_moment_size_scale=proposal_support_moment_size_scale,
            proposal_support_moment_temperature=proposal_support_moment_temperature,
            proposal_support_moment_detach=proposal_support_moment_detach,
            proposal_decoupled_selector=proposal_decoupled_selector,
            proposal_decoupled_selector_detach=proposal_decoupled_selector_detach,
            proposal_decoupled_selector_replace_logits=proposal_decoupled_selector_replace_logits,
            proposal_decoupled_selector_scale=proposal_decoupled_selector_scale,
            proposal_decoupled_selector_base_scale=proposal_decoupled_selector_base_scale,
            proposal_decoupled_selector_readout=proposal_decoupled_selector_readout,
            proposal_oicr_refine_head=proposal_oicr_refine_head,
            proposal_oicr_refine_use_for_selection=proposal_oicr_refine_use_for_selection,
            proposal_pointset_refine=proposal_pointset_refine,
            proposal_pointset_refine_scale=proposal_pointset_refine_scale,
            proposal_pointset_refine_grad_mul=proposal_pointset_refine_grad_mul,
            proposal_pointset_quality_scale=proposal_pointset_quality_scale,
            proposal_pointset_refine_update_token=proposal_pointset_refine_update_token,
            proposal_set_refiner=proposal_set_refiner,
            proposal_set_refiner_topk=proposal_set_refiner_topk,
            proposal_set_refiner_scale=proposal_set_refiner_scale,
            proposal_set_refiner_temperature=proposal_set_refiner_temperature,
            proposal_set_refiner_detach=proposal_set_refiner_detach,
            proposal_local_corr_score=proposal_local_corr_score,
            proposal_local_corr_scale=proposal_local_corr_scale,
            proposal_local_corr_grid_size=proposal_local_corr_grid_size,
            proposal_local_corr_temperature=proposal_local_corr_temperature,
            proposal_local_corr_detach=proposal_local_corr_detach,
            proposal_active_tip_selector=proposal_active_tip_selector,
            proposal_active_tip_selector_scale=proposal_active_tip_selector_scale,
            proposal_active_tip_selector_aux_only=proposal_active_tip_selector_aux_only,
            proposal_active_tip_selector_eval_fuse_scale=proposal_active_tip_selector_eval_fuse_scale,
            proposal_active_tip_feature_fusion=proposal_active_tip_feature_fusion,
            proposal_active_tip_feature_scale=proposal_active_tip_feature_scale,
            proposal_trajectory_rerank=proposal_trajectory_rerank,
            proposal_trajectory_rerank_scale=proposal_trajectory_rerank_scale,
            proposal_trajectory_refine=proposal_trajectory_refine,
            proposal_trajectory_refine_scale=proposal_trajectory_refine_scale,
            condition_prior_proposal=condition_prior_proposal,
            h259_prior_proposal=h259_prior_proposal,
            trajectory_query_proposal_count=trajectory_query_proposal_count,
            trajectory_query_output_mode=trajectory_query_output_mode,
            trajectory_readout_bias=trajectory_readout_bias,
            condition_prior_readout_bias=condition_prior_readout_bias,
            condition_prior_readout_scale=condition_prior_readout_scale,
            trajectory_anchor_logit_bias_scale=trajectory_anchor_logit_bias_scale,
            trajectory_anchor_logit_bias_radius=trajectory_anchor_logit_bias_radius,
            trajectory_anchor_axis_bias_weight=trajectory_anchor_axis_bias_weight,
            corner_heatmap_proposal=corner_heatmap_proposal,
            corner_heatmap_proposal_temperature=corner_heatmap_proposal_temperature,
            corner_heatmap_proposal_detach=corner_heatmap_proposal_detach,
            proposal_source_calibration=proposal_source_calibration,
            proposal_source_calibration_scale=proposal_source_calibration_scale,
            proposal_handoff_source_router=proposal_handoff_source_router,
            proposal_handoff_source_router_scale=proposal_handoff_source_router_scale,
            proposal_handoff_router_normalized_select=proposal_handoff_router_normalized_select,
            proposal_handoff_router_normalized_source_scale=proposal_handoff_router_normalized_source_scale,
            proposal_handoff_router_normalized_within_scale=proposal_handoff_router_normalized_within_scale,
            token_affinity_proposal=token_affinity_proposal,
            token_affinity_proposal_count=token_affinity_proposal_count,
            token_affinity_proposal_temperature=token_affinity_proposal_temperature,
            token_affinity_proposal_top_frac=token_affinity_proposal_top_frac,
            token_affinity_proposal_size_scale=token_affinity_proposal_size_scale,
            token_affinity_proposal_extra_top_fracs=token_affinity_proposal_extra_top_fracs,
            token_affinity_proposal_extra_size_scales=token_affinity_proposal_extra_size_scales,
            token_affinity_proposal_detach=token_affinity_proposal_detach,
            topology_contact_template_proposal=topology_contact_template_proposal,
            topology_contact_template_sizes_px=topology_contact_template_sizes_px,
            topology_contact_template_offsets_px=topology_contact_template_offsets_px,
            topology_contact_template_logit_bias=topology_contact_template_logit_bias,
            topology_contact_template_index_embedding=topology_contact_template_index_embedding,
            topology_contact_template_combo_bias=topology_contact_template_combo_bias,
            topology_contact_template_combo_bias_scale=topology_contact_template_combo_bias_scale,
            topology_contact_template_combo_allowlist=topology_contact_template_combo_allowlist,
            use_rgb_sidecar=use_rgb_sidecar,
        )
        self.refine_all_proposals = bool(refine_all_proposals)
        self.refine_proposal_topk = max(int(refine_proposal_topk), 0)
        self.refine_recollect_stage1 = bool(refine_recollect_stage1)
        self.refine_recollect_refined_base_only = bool(refine_recollect_refined_base_only)
        self.local_grid_confidence_score = bool(local_grid_confidence_score)
        self.local_grid_confidence_scale = float(local_grid_confidence_scale)
        self.local_grid_confidence_detach = bool(local_grid_confidence_detach)
        self.candidate_mask_quality = bool(candidate_mask_quality)
        self.candidate_mask_grid_size = max(int(candidate_mask_grid_size), 2)
        self.candidate_mask_score_scale = float(candidate_mask_score_scale)
        self.candidate_mask_score_detach = bool(candidate_mask_score_detach)
        self.proposal_local_corr_score = bool(proposal_local_corr_score) and bool(use_tool_keypoints)
        self.proposal_local_corr_scale = float(proposal_local_corr_scale)
        self.proposal_local_corr_grid_size = max(int(proposal_local_corr_grid_size), 2)
        self.proposal_local_corr_temperature = max(float(proposal_local_corr_temperature), 1e-3)
        self.proposal_local_corr_detach = bool(proposal_local_corr_detach)
        self.proposal_handoff_source_calibration = bool(proposal_handoff_source_calibration)
        self.proposal_handoff_source_calibration_scale = float(proposal_handoff_source_calibration_scale)
        self.proposal_handoff_source_router = bool(proposal_handoff_source_router)
        self.proposal_handoff_source_router_scale = float(proposal_handoff_source_router_scale)
        self.proposal_handoff_source_router_candidate_select_scale = float(
            proposal_handoff_source_router_candidate_select_scale
        )
        self.proposal_handoff_source5_readout = bool(proposal_handoff_source5_readout)
        self.proposal_handoff_source5_readout_scale = float(proposal_handoff_source5_readout_scale)
        self.proposal_handoff_source5_readout_grid_size = max(
            int(proposal_handoff_source5_readout_grid_size),
            2,
        )
        readout_sources = sorted(
            {
                max(0, min(7, int(src)))
                for src in parse_int_csv(proposal_handoff_source5_readout_sources)
            }
        )
        if not readout_sources:
            readout_sources = [5]
        self.proposal_handoff_source5_readout_sources = tuple(readout_sources)
        self.proposal_handoff_source5_readout_detach = bool(
            proposal_handoff_source5_readout_detach
        )
        self.proposal_handoff_mask_quality = bool(proposal_handoff_mask_quality)
        self.proposal_handoff_mask_quality_scale = float(proposal_handoff_mask_quality_scale)
        self.proposal_handoff_mask_grid_size = max(int(proposal_handoff_mask_grid_size), 2)
        self.proposal_handoff_mask_expand_scale = max(float(proposal_handoff_mask_expand_scale), 0.25)
        self.proposal_handoff_mask_refine_scale = max(
            float(proposal_handoff_mask_refine_scale),
            0.0,
        )
        self.proposal_handoff_mask_refine_size_scale = max(
            float(proposal_handoff_mask_refine_size_scale),
            0.1,
        )
        self.proposal_handoff_mask_refine_min_size_px = max(
            float(proposal_handoff_mask_refine_min_size_px),
            0.25,
        )
        self.proposal_handoff_mask_refine_max_ratio = max(
            float(proposal_handoff_mask_refine_max_ratio),
            0.1,
        )
        handoff_mask_sources = sorted(
            {
                max(0, min(7, int(src)))
                for src in parse_int_csv(proposal_handoff_mask_quality_sources)
            }
        )
        if not handoff_mask_sources:
            handoff_mask_sources = [5, 7]
        self.proposal_handoff_mask_quality_sources = tuple(handoff_mask_sources)
        self.proposal_handoff_mask_detach = bool(proposal_handoff_mask_detach)
        self.proposal_handoff_mask_score_detach = bool(proposal_handoff_mask_score_detach)
        self.proposal_handoff_rank_adaptive_scorer = bool(
            proposal_handoff_rank_adaptive_scorer
        )
        self.proposal_handoff_rank_adaptive_scale = float(
            proposal_handoff_rank_adaptive_scale
        )
        self.proposal_handoff_rank_adaptive_max_rank = max(
            int(proposal_handoff_rank_adaptive_max_rank),
            2,
        )
        self.proposal_handoff_rank_adaptive_detach = bool(
            proposal_handoff_rank_adaptive_detach
        )
        self.proposal_handoff_sparse_set_scorer = bool(proposal_handoff_sparse_set_scorer)
        self.proposal_handoff_sparse_set_scorer_scale = float(
            proposal_handoff_sparse_set_scorer_scale
        )
        sparse_set_sources = sorted(
            {
                max(0, min(7, int(src)))
                for src in parse_int_csv(proposal_handoff_sparse_set_scorer_sources)
            }
        )
        if not sparse_set_sources:
            sparse_set_sources = [0, 5, 7]
        self.proposal_handoff_sparse_set_scorer_sources = tuple(sparse_set_sources)
        self.proposal_handoff_sparse_set_scorer_detach = bool(
            proposal_handoff_sparse_set_scorer_detach
        )
        self.proposal_handoff_source5_refiner = bool(proposal_handoff_source5_refiner)
        self.proposal_handoff_source5_refiner_topk = max(
            int(proposal_handoff_source5_refiner_topk),
            0,
        )
        self.proposal_handoff_source5_refiner_temperature = max(
            float(proposal_handoff_source5_refiner_temperature),
            1e-3,
        )
        self.proposal_handoff_source5_refiner_scale = max(
            float(proposal_handoff_source5_refiner_scale),
            0.0,
        )
        self.proposal_handoff_source5_refiner_logit_scale = float(
            proposal_handoff_source5_refiner_logit_scale
        )
        self.proposal_handoff_source5_refiner_init_logit_bias = float(
            proposal_handoff_source5_refiner_init_logit_bias
        )
        self.proposal_handoff_source5_refiner_detach = bool(
            proposal_handoff_source5_refiner_detach
        )
        self.proposal_handoff_source5_refiner_source_id = int(
            max(0, min(7, int(proposal_handoff_source5_refiner_source_id)))
        )
        self.proposal_handoff_source5_refiner_output_source_id = int(
            max(0, min(7, int(proposal_handoff_source5_refiner_output_source_id)))
        )
        self.proposal_handoff_topk_query_refiner = bool(
            proposal_handoff_topk_query_refiner
        )
        topk_query_sources = sorted(
            {
                int(max(0, min(7, src)))
                for src in parse_int_csv(proposal_handoff_topk_query_sources)
            }
        )
        if not topk_query_sources:
            topk_query_sources = [0, 5]
        self.proposal_handoff_topk_query_sources = tuple(topk_query_sources)
        self.proposal_handoff_topk_query_topk_per_source = max(
            int(proposal_handoff_topk_query_topk_per_source),
            1,
        )
        self.proposal_handoff_topk_query_grid_size = max(
            int(proposal_handoff_topk_query_grid_size),
            2,
        )
        self.proposal_handoff_topk_query_scale = max(
            float(proposal_handoff_topk_query_scale),
            0.0,
        )
        self.proposal_handoff_topk_query_logit_scale = float(
            proposal_handoff_topk_query_logit_scale
        )
        self.proposal_handoff_topk_query_output_source_id = int(
            max(0, min(7, int(proposal_handoff_topk_query_output_source_id)))
        )
        self.proposal_handoff_topk_query_replace_logits = bool(
            proposal_handoff_topk_query_replace_logits
        )
        self.proposal_handoff_topk_query_detach = bool(
            proposal_handoff_topk_query_detach
        )
        self.proposal_handoff_final_fusion = bool(proposal_handoff_final_fusion)
        self.proposal_handoff_final_fusion_scale = max(float(proposal_handoff_final_fusion_scale), 0.0)
        self.proposal_handoff_detach_selection_geometry = bool(
            proposal_handoff_detach_selection_geometry
        )
        self.proposal_handoff_source_representatives_only = bool(
            proposal_handoff_source_representatives_only
        )
        self.proposal_handoff_source5_topk = max(int(proposal_handoff_source5_topk), 0)
        self.proposal_handoff_router_gated_source = bool(proposal_handoff_router_gated_source)
        self.proposal_handoff_router_normalized_select = bool(
            proposal_handoff_router_normalized_select
        )
        self.proposal_handoff_router_normalized_source_scale = float(
            proposal_handoff_router_normalized_source_scale
        )
        self.proposal_handoff_router_normalized_within_scale = float(
            proposal_handoff_router_normalized_within_scale
        )
        self.proposal_handoff_oicr_refine_times = max(
            int(proposal_handoff_oicr_refine_times),
            0,
        )
        self.proposal_handoff_oicr_refine_scale = float(
            proposal_handoff_oicr_refine_scale
        )
        if proposal_handoff_oicr_refine_aggregate not in {"last", "mean", "mean_base"}:
            proposal_handoff_oicr_refine_aggregate = "last"
        self.proposal_handoff_oicr_refine_aggregate = str(
            proposal_handoff_oicr_refine_aggregate
        )
        self.proposal_handoff_oicr_refine_detach = bool(
            proposal_handoff_oicr_refine_detach
        )
        self.proposal_handoff_oicr_refine_use_for_selection = bool(
            proposal_handoff_oicr_refine_use_for_selection
        )
        self.refined_quality_readout = bool(refined_quality_readout)
        self.refined_quality_readout_detach = bool(refined_quality_readout_detach)
        self.refined_quality_readout_replace_logits = bool(refined_quality_readout_replace_logits)
        self.refiner_aligned_context = bool(refiner_aligned_context)
        if refiner_aligned_context_points not in {"aligned_grid_3x3", "aligned_grid_5x5"}:
            refiner_aligned_context_points = "aligned_grid_3x3"
        self.refiner_aligned_context_points = str(refiner_aligned_context_points)
        self.refiner_aligned_context_scale = float(refiner_aligned_context_scale)
        if refined_quality_readout_points not in {
            "center_corners",
            "center_corners_edges",
            "aligned_grid_3x3",
            "aligned_grid_5x5",
        }:
            refined_quality_readout_points = "aligned_grid_3x3"
        self.refined_quality_readout_points = str(refined_quality_readout_points)
        self.proposal_count = max(int(proposal_count), 2)
        extra_proposal_count = 0
        if bool(condition_prior_proposal):
            extra_proposal_count += 1
        if stage1_use_tool_keypoints:
            if bool(h259_prior_proposal):
                extra_proposal_count += 1
            extra_proposal_count += max(int(trajectory_query_proposal_count), 0)
        if bool(corner_heatmap_proposal):
            extra_proposal_count += 1
        if bool(token_affinity_proposal):
            variant_count = token_affinity_variant_count(
                token_affinity_proposal_extra_top_fracs,
                token_affinity_proposal_extra_size_scales,
            )
            extra_proposal_count += max(int(token_affinity_proposal_count), 1) * variant_count
        if stage1_use_tool_keypoints and bool(topology_contact_template_proposal):
            extra_proposal_count += topology_contact_template_count(
                topology_contact_template_sizes_px,
                topology_contact_template_offsets_px,
                mask_res=64,
            )
        self.refine_candidate_count = self.proposal_count + extra_proposal_count
        if self.refine_all_proposals:
            self.candidate_refine_embed = nn.Parameter(
                torch.zeros(1, 1, self.refine_candidate_count, 1, hidden_dim)
            )
            self.refined_quality_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            if self.refined_quality_readout:
                self.refined_quality_readout_proj = nn.Sequential(
                    nn.LayerNorm(3 * hidden_dim),
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.refined_quality_readout_head = nn.Sequential(
                    nn.LayerNorm(2 * hidden_dim + 8),
                    nn.Linear(2 * hidden_dim + 8, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                nn.init.zeros_(self.refined_quality_readout_head[-1].weight)
                nn.init.zeros_(self.refined_quality_readout_head[-1].bias)
            if self.refiner_aligned_context:
                self.refiner_aligned_context_proj = nn.Sequential(
                    nn.LayerNorm(3 * hidden_dim),
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            if self.candidate_mask_quality:
                mask_in_dim = 2 * hidden_dim + 10
                self.candidate_mask_readout_proj = nn.Sequential(
                    nn.LayerNorm(3 * hidden_dim),
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.candidate_mask_point_head = nn.Sequential(
                    nn.LayerNorm(mask_in_dim),
                    nn.Linear(mask_in_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                self.candidate_mask_quality_head = nn.Sequential(
                    nn.LayerNorm(mask_in_dim),
                    nn.Linear(mask_in_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
                nn.init.zeros_(self.candidate_mask_point_head[-1].weight)
                nn.init.zeros_(self.candidate_mask_point_head[-1].bias)
                nn.init.zeros_(self.candidate_mask_quality_head[-1].weight)
                nn.init.zeros_(self.candidate_mask_quality_head[-1].bias)
            nn.init.trunc_normal_(self.candidate_refine_embed, std=0.02)
        if self.proposal_handoff_source_calibration:
            self.proposal_handoff_source_logit_bias = nn.Embedding(8, 1)
            nn.init.zeros_(self.proposal_handoff_source_logit_bias.weight)
            with torch.no_grad():
                self.proposal_handoff_source_logit_bias.weight[4, 0] = float(proposal_handoff_source4_init)
        if self.proposal_handoff_source_router:
            self.proposal_handoff_source_router_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * hidden_dim),
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_source_router_candidate_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 1),
                nn.Linear(hidden_dim + 8 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.proposal_handoff_source_router_context_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 8),
            )
            nn.init.zeros_(self.proposal_handoff_source_router_candidate_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source_router_candidate_head[-1].bias)
            nn.init.zeros_(self.proposal_handoff_source_router_context_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source_router_context_head[-1].bias)
        if self.proposal_handoff_source5_readout:
            self.proposal_handoff_source5_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * hidden_dim),
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_source5_readout_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 1),
                nn.Linear(hidden_dim + 8 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.proposal_handoff_source5_readout_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source5_readout_head[-1].bias)
        if self.proposal_handoff_mask_quality:
            mask_point_in_dim = 3 * hidden_dim + 10
            mask_quality_in_dim = 3 * hidden_dim + 11
            self.proposal_handoff_mask_source_emb = nn.Embedding(8, hidden_dim)
            self.proposal_handoff_mask_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * hidden_dim),
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_mask_point_head = nn.Sequential(
                nn.LayerNorm(mask_point_in_dim),
                nn.Linear(mask_point_in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.proposal_handoff_mask_quality_head = nn.Sequential(
                nn.LayerNorm(mask_quality_in_dim),
                nn.Linear(mask_quality_in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.normal_(self.proposal_handoff_mask_source_emb.weight, std=0.02)
            nn.init.zeros_(self.proposal_handoff_mask_point_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_mask_point_head[-1].bias)
            nn.init.zeros_(self.proposal_handoff_mask_quality_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_mask_quality_head[-1].bias)
        if self.proposal_handoff_rank_adaptive_scorer:
            self.proposal_handoff_rank_adaptive_rank_emb = nn.Embedding(
                self.proposal_handoff_rank_adaptive_max_rank,
                hidden_dim,
            )
            self.proposal_handoff_rank_adaptive_source_emb = nn.Embedding(8, hidden_dim)
            self.proposal_handoff_rank_adaptive_geom_mlp = nn.Sequential(
                nn.LayerNorm(8),
                nn.Linear(8, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_rank_adaptive_logit_mlp = nn.Sequential(
                nn.LayerNorm(1),
                nn.Linear(1, hidden_dim),
            )
            self.proposal_handoff_rank_adaptive_score_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.normal_(self.proposal_handoff_rank_adaptive_rank_emb.weight, std=0.02)
            nn.init.normal_(self.proposal_handoff_rank_adaptive_source_emb.weight, std=0.02)
            nn.init.zeros_(self.proposal_handoff_rank_adaptive_score_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_rank_adaptive_score_head[-1].bias)
        if self.proposal_handoff_sparse_set_scorer:
            self.proposal_handoff_sparse_set_source_emb = nn.Embedding(8, hidden_dim)
            self.proposal_handoff_sparse_set_geom_mlp = nn.Sequential(
                nn.LayerNorm(8),
                nn.Linear(8, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_sparse_set_logit_mlp = nn.Sequential(
                nn.LayerNorm(1),
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_sparse_set_attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads=4,
                dropout=0.10,
                batch_first=True,
            )
            self.proposal_handoff_sparse_set_attn_norm = nn.LayerNorm(hidden_dim)
            self.proposal_handoff_sparse_set_ffn_norm = nn.LayerNorm(hidden_dim)
            self.proposal_handoff_sparse_set_ffn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            self.proposal_handoff_sparse_set_score_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.normal_(self.proposal_handoff_sparse_set_source_emb.weight, std=0.02)
            nn.init.zeros_(self.proposal_handoff_sparse_set_score_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_sparse_set_score_head[-1].bias)
        if self.proposal_handoff_oicr_refine_times > 0:
            self.proposal_handoff_oicr_refine_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_dim + 8 + 1),
                        nn.Linear(hidden_dim + 8 + 1, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, 1),
                    )
                    for _ in range(self.proposal_handoff_oicr_refine_times)
                ]
            )
            for head in self.proposal_handoff_oicr_refine_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        if self.proposal_handoff_source5_refiner:
            self.proposal_handoff_source5_refiner_item_mlp = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 1),
                nn.Linear(hidden_dim + 8 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_source5_refiner_score_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.proposal_handoff_source5_refiner_delta_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 4),
                nn.Linear(hidden_dim + 8 + 4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 8),
            )
            self.proposal_handoff_source5_refiner_logit_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 4),
                nn.Linear(hidden_dim + 8 + 4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.proposal_handoff_source5_refiner_score_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source5_refiner_score_head[-1].bias)
            nn.init.zeros_(self.proposal_handoff_source5_refiner_delta_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source5_refiner_delta_head[-1].bias)
            nn.init.zeros_(self.proposal_handoff_source5_refiner_logit_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_source5_refiner_logit_head[-1].bias)
        if self.proposal_handoff_topk_query_refiner:
            topk_query_in_dim = 3 * hidden_dim + 8 + 1
            self.proposal_handoff_topk_query_source_emb = nn.Embedding(8, hidden_dim)
            self.proposal_handoff_topk_query_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * hidden_dim),
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_topk_query_item_mlp = nn.Sequential(
                nn.LayerNorm(topk_query_in_dim),
                nn.Linear(topk_query_in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_topk_query_attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads=4,
                dropout=0.10,
                batch_first=True,
            )
            self.proposal_handoff_topk_query_attn_norm = nn.LayerNorm(hidden_dim)
            self.proposal_handoff_topk_query_ffn_norm = nn.LayerNorm(hidden_dim)
            self.proposal_handoff_topk_query_ffn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            self.proposal_handoff_topk_query_delta_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8),
                nn.Linear(hidden_dim + 8, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 8),
            )
            self.proposal_handoff_topk_query_logit_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8),
                nn.Linear(hidden_dim + 8, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.normal_(self.proposal_handoff_topk_query_source_emb.weight, std=0.02)
            nn.init.zeros_(self.proposal_handoff_topk_query_delta_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_topk_query_delta_head[-1].bias)
            nn.init.zeros_(self.proposal_handoff_topk_query_logit_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_topk_query_logit_head[-1].bias)
        if self.proposal_handoff_final_fusion:
            self.proposal_handoff_final_fusion_candidate_mlp = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 2),
                nn.Linear(hidden_dim + 8 + 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_handoff_final_fusion_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 4),
                nn.Linear(hidden_dim + 8 + 4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 8),
            )
            nn.init.zeros_(self.proposal_handoff_final_fusion_head[-1].weight)
            nn.init.zeros_(self.proposal_handoff_final_fusion_head[-1].bias)
        if self.proposal_local_corr_score:
            self.proposal_local_corr_readout_proj = nn.Sequential(
                nn.LayerNorm(3 * hidden_dim),
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.proposal_local_corr_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + 8 + 1 + 12 + 6),
                nn.Linear(hidden_dim + 8 + 1 + 12 + 6, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.proposal_local_corr_head[-1].weight)
            nn.init.zeros_(self.proposal_local_corr_head[-1].bias)

    def _proposal_local_corr_logits(
        self,
        feat16_map: torch.Tensor,
        candidate_quads: torch.Tensor,
        candidate_logits: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, k_select = candidate_quads.shape[:3]
        device = candidate_quads.device
        dtype = candidate_quads.dtype
        grid_size = int(self.proposal_local_corr_grid_size)
        cand_pts = self.stage1._candidate_aligned_grid_points(candidate_quads, grid_size=grid_size)
        cand_readout = self.stage1._sample_traj_readout_features(
            feat16_map,
            cand_pts.reshape(b, 16, k_select * grid_size * grid_size, 2),
        )
        cand_readout = cand_readout.reshape(b, 16, k_select, grid_size * grid_size, -1)
        cand_feat = self.proposal_local_corr_readout_proj(cand_readout)

        kpt_xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).to(device=device, dtype=dtype)
        kpt_vis = tool_kpt_vec_16[..., 6:9].reshape(b, 16, 3).to(device=device, dtype=dtype).clamp(0.0, 1.0)
        tool_readout = self.stage1._sample_traj_readout_features(feat16_map, kpt_xy.reshape(b, 16, 3, 2))
        tool_feat = self.proposal_local_corr_readout_proj(tool_readout)

        cand_norm = F.normalize(cand_feat.float(), dim=-1, eps=1e-6)
        tool_norm = F.normalize(tool_feat.float(), dim=-1, eps=1e-6)
        sim = torch.einsum("btkgc,btpc->btkgp", cand_norm, tool_norm)
        vis = kpt_vis[:, :, None, None, :].float()
        sim = sim.masked_fill(vis <= 0.05, -1.0)
        flat = sim.reshape(b, 16, k_select, -1)
        max_sim = flat.max(dim=-1).values
        valid = (flat > -0.999).float()
        valid_count = valid.sum(dim=-1).clamp_min(1.0)
        mean_sim = (flat * valid).sum(dim=-1) / valid_count
        var_sim = (((flat - mean_sim[..., None]) * valid) ** 2).sum(dim=-1) / valid_count
        std_sim = var_sim.clamp_min(0.0).sqrt()
        tau = float(self.proposal_local_corr_temperature)
        prob = F.softmax(flat / tau, dim=-1)
        entropy = -(
            prob.clamp_min(1e-6) * prob.clamp_min(1e-6).log()
        ).sum(dim=-1) / math.log(max(int(flat.shape[-1]), 2))
        vis_frac = kpt_vis.mean(dim=-1)[:, :, None].expand(-1, -1, k_select)
        corr_stats = torch.stack(
            [
                max_sim,
                mean_sim,
                std_sim,
                max_sim - mean_sim,
                1.0 - entropy.to(device=device, dtype=max_sim.dtype),
                vis_frac.to(device=device, dtype=max_sim.dtype),
            ],
            dim=-1,
        ).to(device=device, dtype=dtype)

        cand_context = cand_feat.mean(dim=3).to(dtype=dtype)
        tip_feats = self.stage1._active_tip_candidate_features(
            candidate_quads,
            tool_kpt_vec_16,
            role_features_16,
            tool_id,
        )
        corr_tail = torch.cat(
            [
                candidate_quads.reshape(b, 16, k_select, 8),
                candidate_logits.unsqueeze(-1),
                tip_feats,
                corr_stats,
            ],
            dim=-1,
        )
        if self.proposal_local_corr_detach:
            corr_tail = corr_tail.detach()
            cand_context = cand_context.detach()
        corr_logits = self.proposal_local_corr_head(
            torch.cat([cand_context, corr_tail], dim=-1)
        ).squeeze(-1)
        return corr_logits.to(device=device, dtype=dtype), corr_stats

    def _select_refined_proposals(
        self,
        candidate_quads: torch.Tensor,
        quality_logits: torch.Tensor,
        detach_geometry: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = candidate_quads.shape[2]
        use_hard = self.stage1.proposal_selection_mode == "hard" or (
            self.stage1.proposal_selection_mode == "train_soft_eval_hard" and not self.training
        )
        if use_hard:
            best = quality_logits.argmax(dim=-1)
            weights = F.one_hot(best, num_classes=k).to(dtype=quality_logits.dtype)
        else:
            select_logits = quality_logits / self.stage1.proposal_selection_temperature
            topk = min(int(self.stage1.proposal_selection_topk), k)
            if topk > 0 and topk < k:
                topk_idx = select_logits.topk(topk, dim=-1).indices
                masked_logits = torch.full_like(select_logits, torch.finfo(select_logits.dtype).min)
                select_logits = masked_logits.scatter(-1, topk_idx, select_logits.gather(-1, topk_idx))
            weights_soft = F.softmax(select_logits, dim=-1)
            if self.stage1.proposal_selection_straight_through:
                best = weights_soft.argmax(dim=-1)
                weights_hard = F.one_hot(best, num_classes=k).to(dtype=weights_soft.dtype)
                weights = weights_hard - weights_soft.detach() + weights_soft
            else:
                weights = weights_soft
        handoff_quads = candidate_quads.detach() if detach_geometry else candidate_quads
        quad = (handoff_quads * weights[..., None, None]).sum(dim=2).clamp(0.0, 1.0)
        return quad, weights

    def _local_grid_refine_candidate_quads(
        self,
        feat16_map: torch.Tensor,
        base_quads: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the local corner search head independently to every proposal."""
        b, t, k = base_quads.shape[:3]
        grid_size = int(self.local_grid_refine_grid_size)
        search_quads = base_quads.detach() if self.local_grid_refine_detach_base else base_quads
        grid_pts, rel = self._local_grid_points(
            search_quads,
            grid_size=grid_size,
            scale=float(self.local_grid_refine_scale),
        )
        p = grid_pts.shape[3]
        point_feat = self._sample_point_features(
            feat16_map,
            grid_pts.reshape(b, t, k * p, 2),
        ).reshape(b, t, k, p, -1)
        rel_feat = rel.to(device=point_feat.device, dtype=point_feat.dtype)
        rel_feat = rel_feat.reshape(1, 1, 1, p, 2).expand(b, t, k, -1, -1)
        grid_feat = self.local_grid_point_proj(torch.cat([point_feat, rel_feat], dim=-1))
        corner_feat = self.local_grid_corner_embed.to(device=grid_feat.device, dtype=grid_feat.dtype)
        corner_feat = corner_feat.reshape(1, 1, 1, 4, -1).expand(b, t, k, -1, -1)
        corner_grid_feat = grid_feat[:, :, :, None, :, :].expand(-1, -1, -1, 4, -1, -1)
        corner_token = corner_feat[:, :, :, :, None, :].expand(-1, -1, -1, -1, p, -1)
        grid_xy = grid_pts[:, :, :, None, :, :].expand(-1, -1, -1, 4, -1, -1)
        logits = self.local_grid_corner_head(
            torch.cat([corner_grid_feat, corner_token, grid_xy], dim=-1)
        ).squeeze(-1)
        if self.local_grid_refine_prior_scale != 0.0:
            prior_quad = search_quads.to(device=grid_pts.device, dtype=grid_pts.dtype)
            _dirs, lens = quad_axes(prior_quad.reshape(-1, 4, 2))
            side = lens.mean(dim=-1).reshape(b, t, k, 1, 1).clamp_min(1e-4)
            prior_dist2 = (grid_xy - prior_quad[:, :, :, :, None, :]).square().sum(dim=-1)
            sigma = float(self.local_grid_refine_prior_sigma) * side
            corner_prior = -prior_dist2 / (2.0 * sigma.square().clamp_min(1e-8))
            logits = logits + float(self.local_grid_refine_prior_scale) * corner_prior
        probs = F.softmax(logits / float(self.local_grid_refine_temperature), dim=-1)
        quads = (probs[..., None] * grid_xy).sum(dim=-2).clamp(0.0, 1.0)
        return quads, logits, grid_pts

    def _local_grid_confidence_logits(self, corner_logits: torch.Tensor) -> torch.Tensor:
        """Convert local-grid corner peak margins into per-candidate scores."""
        logits = corner_logits.float()
        if logits.shape[-1] < 2:
            conf = logits.mean(dim=(-1, -2))
        else:
            top2 = logits.topk(2, dim=-1).values
            conf = (top2[..., 0] - top2[..., 1]).mean(dim=-1)
        mean = conf.mean(dim=-1, keepdim=True)
        std = conf.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        return (conf - mean) / std

    def _candidate_mask_quality_outputs(
        self,
        feat16_map: torch.Tensor,
        candidate_quads: torch.Tensor,
        candidate_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict candidate-local contact masks and an IoU-aware quality logit."""
        b, t, k = candidate_quads.shape[:3]
        grid_size = int(self.candidate_mask_grid_size)
        p_grid = grid_size * grid_size
        mask_pts = self.stage1._candidate_aligned_grid_points(
            candidate_quads,
            grid_size=grid_size,
        )
        mask_readout = self.stage1._sample_traj_readout_features(
            feat16_map,
            mask_pts.reshape(b, t, k * p_grid, 2),
        )
        mask_readout = mask_readout.reshape(b, t, k, p_grid, -1)
        point_feat = self.candidate_mask_readout_proj(mask_readout)
        coords = torch.linspace(
            -1.0,
            1.0,
            grid_size,
            device=candidate_quads.device,
            dtype=candidate_quads.dtype,
        )
        vv, uu = torch.meshgrid(coords, coords, indexing="ij")
        uv = torch.stack([uu, vv], dim=-1).reshape(1, 1, 1, p_grid, 2)
        uv = uv.expand(b, t, k, -1, -1).to(device=point_feat.device, dtype=point_feat.dtype)
        geom = candidate_quads.reshape(b, t, k, 8).to(device=point_feat.device, dtype=point_feat.dtype)
        token = candidate_tokens.to(device=point_feat.device, dtype=point_feat.dtype)
        point_token = token[:, :, :, None, :].expand(-1, -1, -1, p_grid, -1)
        point_geom = geom[:, :, :, None, :].expand(-1, -1, -1, p_grid, -1)
        point_logits = self.candidate_mask_point_head(
            torch.cat([point_token, point_feat, point_geom, uv], dim=-1)
        ).squeeze(-1)
        prob = torch.sigmoid(point_logits.float()).to(dtype=point_feat.dtype)
        pooled_feat = (point_feat * prob[..., None]).sum(dim=3) / prob.sum(dim=3, keepdim=True).clamp_min(1e-4)
        mask_stats = torch.stack(
            [
                prob.mean(dim=3),
                prob.amax(dim=3),
            ],
            dim=-1,
        ).to(device=point_feat.device, dtype=point_feat.dtype)
        quality_logits = self.candidate_mask_quality_head(
            torch.cat([token, pooled_feat, geom, mask_stats], dim=-1)
        ).squeeze(-1)
        return point_logits, mask_pts, quality_logits

    def _proposal_handoff_mask_quality_outputs(
        self,
        feat16_map: torch.Tensor,
        candidate_quads: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_logits: torch.Tensor,
        candidate_source_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict source-conditioned local masks and IoU-quality logits.

        Deformable DETR samples sparse evidence around each reference box.
        Mask Scoring R-CNN then scores a predicted region using both region
        evidence and the predicted mask.  This lightweight version keeps our
        existing proposal bank fixed but gives source5/source7 candidates a
        deployable local evidence head before final handoff ranking.
        """
        b, t, k = candidate_quads.shape[:3]
        grid_size = int(self.proposal_handoff_mask_grid_size)
        p_grid = grid_size * grid_size
        sample_quads = candidate_quads
        expand = float(self.proposal_handoff_mask_expand_scale)
        if abs(expand - 1.0) > 1e-6:
            center = candidate_quads.mean(dim=-2, keepdim=True)
            sample_quads = (center + (candidate_quads - center) * expand).clamp(0.0, 1.0)
        mask_pts = self.stage1._candidate_aligned_grid_points(
            sample_quads,
            grid_size=grid_size,
        )
        mask_readout = self.stage1._sample_traj_readout_features(
            feat16_map,
            mask_pts.reshape(b, t, k * p_grid, 2),
        ).reshape(b, t, k, p_grid, -1)
        point_feat = self.proposal_handoff_mask_readout_proj(mask_readout)
        coords = torch.linspace(
            -1.0,
            1.0,
            grid_size,
            device=candidate_quads.device,
            dtype=candidate_quads.dtype,
        )
        vv, uu = torch.meshgrid(coords, coords, indexing="ij")
        uv = torch.stack([uu, vv], dim=-1).reshape(1, 1, 1, p_grid, 2)
        uv = uv.expand(b, t, k, -1, -1).to(device=point_feat.device, dtype=point_feat.dtype)
        source_ids = candidate_source_ids.detach().clamp(0, 7)
        source_emb = self.proposal_handoff_mask_source_emb(source_ids).to(dtype=point_feat.dtype)
        geom = candidate_quads.reshape(b, t, k, 8).to(device=point_feat.device, dtype=point_feat.dtype)
        token = candidate_tokens.to(device=point_feat.device, dtype=point_feat.dtype)
        base_logit = candidate_logits.detach().unsqueeze(-1).to(device=point_feat.device, dtype=point_feat.dtype)
        if self.proposal_handoff_mask_detach:
            point_feat = point_feat.detach()
            geom = geom.detach()
            token = token.detach()
            source_emb = source_emb.detach()
        point_token = token[:, :, :, None, :].expand(-1, -1, -1, p_grid, -1)
        point_source = source_emb[:, :, :, None, :].expand(-1, -1, -1, p_grid, -1)
        point_geom = geom[:, :, :, None, :].expand(-1, -1, -1, p_grid, -1)
        point_logits = self.proposal_handoff_mask_point_head(
            torch.cat([point_token, point_feat, point_source, point_geom, uv], dim=-1)
        ).squeeze(-1)
        prob = torch.sigmoid(point_logits.float()).to(dtype=point_feat.dtype)
        pooled_feat = (point_feat * prob[..., None]).sum(dim=3) / prob.sum(
            dim=3,
            keepdim=True,
        ).clamp_min(1e-4)
        mask_stats = torch.stack(
            [
                prob.mean(dim=3),
                prob.amax(dim=3),
            ],
            dim=-1,
        ).to(device=point_feat.device, dtype=point_feat.dtype)
        quality_logits = self.proposal_handoff_mask_quality_head(
            torch.cat([token, pooled_feat, source_emb, geom, mask_stats, base_logit], dim=-1)
        ).squeeze(-1)
        source_mask = torch.zeros_like(source_ids, dtype=torch.bool)
        for source_id in self.proposal_handoff_mask_quality_sources:
            source_mask = source_mask | (source_ids == int(source_id))
        return point_logits, mask_pts, quality_logits, source_mask

    def _proposal_handoff_mask_refined_quads(
        self,
        base_quads: torch.Tensor,
        mask_points: torch.Tensor,
        mask_logits: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode candidate-local point-mask logits into aligned moment quads."""
        b, t, k = base_quads.shape[:3]
        dtype = base_quads.dtype
        prob = torch.sigmoid(mask_logits.float()).to(dtype=dtype)
        weights = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-4)
        center = (weights[..., None] * mask_points.to(dtype=dtype)).sum(dim=-2)

        dirs, lens = quad_axes(base_quads.reshape(-1, 4, 2))
        dirs = dirs.reshape(b, t, k, 2, 2).to(device=base_quads.device, dtype=dtype)
        lens = lens.reshape(b, t, k, 2).to(device=base_quads.device, dtype=dtype)
        delta = mask_points.to(dtype=dtype) - center[..., None, :]
        proj = torch.einsum("btkpd,btkad->btkpa", delta, dirs)
        var = (weights[..., None] * proj.square()).sum(dim=-2).clamp_min(1e-8)
        min_size = float(self.proposal_handoff_mask_refine_min_size_px) / 64.0
        max_size = lens * float(self.proposal_handoff_mask_refine_max_ratio)
        size = (
            float(self.proposal_handoff_mask_refine_size_scale) * torch.sqrt(var)
        ).clamp_min(min_size)
        size = torch.minimum(size, max_size.clamp_min(min_size))
        moment_quad = oriented_rect_to_quad(center, size, dirs[..., 0, :]).to(dtype=dtype)
        refined = (
            base_quads
            + float(self.proposal_handoff_mask_refine_scale) * (moment_quad - base_quads)
        ).clamp(0.0, 1.0)
        return torch.where(source_mask[..., None, None], refined, base_quads)

    def _proposal_handoff_topk_query_outputs(
        self,
        feat16_map: torch.Tensor,
        candidate_quads: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_logits: torch.Tensor,
        candidate_source_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Refine a fixed top-k source subset as independent proposal queries."""
        b, t, k = candidate_quads.shape[:3]
        source_ids = candidate_source_ids.detach().clamp(0, 7)
        topk_each = min(int(self.proposal_handoff_topk_query_topk_per_source), k)
        query_indices = []
        query_valid = []
        for source_id in self.proposal_handoff_topk_query_sources:
            source_mask = source_ids == int(source_id)
            source_scores = candidate_logits.masked_fill(~source_mask, -50.0)
            vals, idx = source_scores.topk(topk_each, dim=-1)
            valid = source_mask.gather(-1, idx) & (vals > -49.0)
            query_indices.append(idx)
            query_valid.append(valid)
        topk_idx = torch.cat(query_indices, dim=-1)
        valid_mask = torch.cat(query_valid, dim=-1)
        q = int(topk_idx.shape[-1])

        gather_quad_idx = topk_idx[..., None, None].expand(-1, -1, -1, 4, 2)
        query_quads = candidate_quads.gather(2, gather_quad_idx)
        gather_token_idx = topk_idx[..., None].expand(-1, -1, -1, candidate_tokens.shape[-1])
        query_tokens = candidate_tokens.gather(2, gather_token_idx)
        query_logits = candidate_logits.gather(-1, topk_idx)
        query_sources = source_ids.gather(-1, topk_idx)

        grid_size = int(self.proposal_handoff_topk_query_grid_size)
        p_grid = grid_size * grid_size
        query_pts = self.stage1._candidate_aligned_grid_points(
            query_quads,
            grid_size=grid_size,
        )
        query_readout = self.stage1._sample_traj_readout_features(
            feat16_map,
            query_pts.reshape(b, t, q * p_grid, 2),
        ).reshape(b, t, q, p_grid, -1)
        query_readout = self.proposal_handoff_topk_query_readout_proj(
            query_readout.mean(dim=3)
        )
        query_geom = query_quads.reshape(b, t, q, 8)
        query_source_emb = self.proposal_handoff_topk_query_source_emb(query_sources).to(
            dtype=query_tokens.dtype
        )
        item_tokens = query_tokens
        item_geom = query_geom
        item_logits = query_logits
        if self.proposal_handoff_topk_query_detach:
            item_tokens = item_tokens.detach()
            item_geom = item_geom.detach()
            item_logits = item_logits.detach()
            query_readout = query_readout.detach()
            query_source_emb = query_source_emb.detach()
        query_item = self.proposal_handoff_topk_query_item_mlp(
            torch.cat(
                [
                    item_tokens,
                    query_readout.to(dtype=item_tokens.dtype),
                    query_source_emb,
                    item_geom.to(dtype=item_tokens.dtype),
                    item_logits.unsqueeze(-1).to(dtype=item_tokens.dtype),
                ],
                dim=-1,
            )
        )
        flat = query_item.reshape(b * t, q, -1)
        valid_flat = valid_mask.reshape(b * t, q)
        key_padding = ~valid_flat
        has_valid = valid_flat.any(dim=-1)
        if key_padding.numel() > 0:
            key_padding = key_padding.clone()
            key_padding[~has_valid, 0] = False
        attn, _ = self.proposal_handoff_topk_query_attn(
            flat,
            flat,
            flat,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        flat = self.proposal_handoff_topk_query_attn_norm(flat + attn)
        flat = self.proposal_handoff_topk_query_ffn_norm(
            flat + self.proposal_handoff_topk_query_ffn(flat)
        )
        flat = flat * valid_flat.unsqueeze(-1).to(dtype=flat.dtype)
        query_item = flat.reshape(b, t, q, -1)

        delta_in = torch.cat(
            [query_item, query_geom.to(device=query_item.device, dtype=query_item.dtype)],
            dim=-1,
        )
        query_delta = torch.tanh(
            self.proposal_handoff_topk_query_delta_head(delta_in)
        ).reshape(b, t, q, 4, 2)
        _query_dirs, query_lens = quad_axes(query_quads.reshape(-1, 4, 2))
        query_step = query_lens.mean(dim=-1).reshape(b, t, q, 1, 1).clamp(0.015, 0.35)
        refined_quads = (
            query_quads
            + float(self.proposal_handoff_topk_query_scale) * query_delta * query_step
        ).clamp(0.0, 1.0)
        query_logit_delta = self.proposal_handoff_topk_query_logit_head(delta_in).squeeze(-1)
        query_logit_delta = query_logit_delta.to(device=query_logits.device, dtype=query_logits.dtype)
        if self.proposal_handoff_topk_query_replace_logits:
            refined_logits = float(self.proposal_handoff_topk_query_logit_scale) * query_logit_delta
        else:
            refined_logits = query_logits + float(
                self.proposal_handoff_topk_query_logit_scale
            ) * query_logit_delta
        refined_logits = torch.where(
            valid_mask,
            refined_logits,
            torch.full_like(refined_logits, -50.0),
        )
        return refined_quads, query_quads, refined_logits, valid_mask, query_sources, query_item

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
        dn_quad_xy: torch.Tensor | None = None,
        rgb_16: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.refine_all_proposals:
            return super().forward(
                tokens=tokens,
                role_features_16=role_features_16,
                tool_mask_8=tool_mask_8,
                tool_kpt_heatmaps_8=tool_kpt_heatmaps_8,
                tool_kpt_vec_16=tool_kpt_vec_16,
                tool_id=tool_id,
                act_id=act_id,
                combo_id=combo_id,
                dn_quad_xy=dn_quad_xy,
                rgb_16=rgb_16,
            )

        stage1_kwargs: dict[str, torch.Tensor] = {
            "tokens": tokens,
            "role_features_16": role_features_16,
            "tool_mask_8": tool_mask_8,
            "tool_kpt_heatmaps_8": tool_kpt_heatmaps_8,
            "tool_kpt_vec_16": tool_kpt_vec_16,
            "tool_id": tool_id,
            "act_id": act_id,
            "combo_id": combo_id,
        }
        if getattr(self.stage1, "use_rgb_sidecar", False):
            stage1_kwargs["rgb_16"] = rgb_16
        stage1 = self.stage1(**stage1_kwargs)
        stage1_full_proposal_quads = stage1["proposal_quads"]
        stage1_full_proposal_xyxy = stage1["proposal_xyxy"]
        stage1_full_proposal_logits = stage1["proposal_logits"]
        topk_indices = None
        if self.refine_proposal_topk > 0 and self.refine_proposal_topk < stage1["proposal_quads"].shape[2]:
            topk = min(self.refine_proposal_topk, stage1["proposal_quads"].shape[2])
            topk_indices = stage1["proposal_logits"].topk(topk, dim=-1).indices

            def gather_candidates(x: torch.Tensor) -> torch.Tensor:
                expand_shape = topk_indices.shape + x.shape[3:]
                gather_idx = topk_indices.reshape(*topk_indices.shape, *([1] * (x.dim() - 3))).expand(expand_shape)
                return x.gather(2, gather_idx)

            stage1 = dict(stage1)
            for key in ("proposal_quads", "proposal_xyxy", "proposal_cxcywh"):
                if key in stage1:
                    stage1[key] = gather_candidates(stage1[key])
            stage1["proposal_logits"] = gather_candidates(stage1["proposal_logits"].unsqueeze(-1)).squeeze(-1)
            if "proposal_source_ids" in stage1:
                stage1["proposal_source_ids"] = gather_candidates(stage1["proposal_source_ids"])
            if "proposal_selection_weights" in stage1:
                stage1["proposal_selection_weights"] = gather_candidates(stage1["proposal_selection_weights"])

        b, t8, h, w, _ = tokens.shape
        device = tokens.device
        dtype = tokens.dtype
        k = stage1["proposal_quads"].shape[2]
        p = k * 4

        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        role_bias8 = self.role_mlp(role8).view(b, t8, -1, 1, 1)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gamma_beta = self.cond_to_gamma_beta(z).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=device, dtype=dtype).view(1, 1, 2, h, w).expand(b, t8, -1, -1, -1)
        spatial_parts = [x]
        if self.use_rgb_sidecar:
            if self.rgb_encoder is None:
                raise RuntimeError("RGB sidecar is enabled without an encoder")
            spatial_parts.append(encode_rgb_sidecar_8(self.rgb_encoder, rgb_16, dtype=dtype))
        spatial_parts.append(grid)
        if self.use_tool_mask:
            spatial_parts.append(tool_mask_8.to(device=device, dtype=dtype))
        if self.use_tool_keypoints:
            spatial_parts.append(tool_kpt_heatmaps_8.to(device=device, dtype=dtype))
        y = self.spatial(torch.cat(spatial_parts, dim=2).reshape(b * t8, -1, h, w))
        mem8 = y.view(b, t8, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
        mem16_chw = F.interpolate(mem8, size=(16, h, w), mode="trilinear", align_corners=True)
        feat16_map = mem16_chw.permute(0, 2, 1, 3, 4).contiguous()
        mem16 = mem16_chw.permute(0, 2, 3, 4, 1).reshape(b, 16, h * w, -1)

        query_base = self.role_mlp(role_features_16) + z[:, None, :].expand(-1, 16, -1)
        if self.use_tool_mask:
            mask_geom8 = self._tool_mask_geom8(tool_mask_8.to(device=device, dtype=dtype))
            mask_geom16 = F.interpolate(mask_geom8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
            query_base = query_base + self.tool_mask_geom_mlp(mask_geom16)
        if self.use_tool_keypoints:
            query_base = query_base + self.tool_kpt_mlp(tool_kpt_vec_16.to(device=device, dtype=dtype))
            if self.use_tool_trajectory_bias:
                traj_bias = self.tool_traj_mlp(tool_kpt_vec_16.to(device=device, dtype=dtype).reshape(b, 16 * 9))
                query_base = query_base + traj_bias[:, None, :]
        tool_memory = None
        if self.use_tool_trajectory_attn:
            tool_memory = self.tool_traj_token_mlp(
                torch.cat(
                    [
                        tool_kpt_vec_16.to(device=device, dtype=dtype),
                        role_features_16.to(device=device, dtype=dtype),
                    ],
                    dim=-1,
                )
            )
            tool_memory = tool_memory + z[:, None, :] + self.tool_time_pos.to(device=device, dtype=dtype)
        corner_query_bias = 0.0
        if self.use_tool_corner_query_bias:
            tool_anchor_quad = self._tool_anchor_quad(tool_kpt_vec_16.to(device=device, dtype=dtype))
            corner_query_bias = self.tool_corner_ref_mlp(tool_anchor_quad)
        corner_query_bias_all = (
            corner_query_bias[:, :, None, :, :] if isinstance(corner_query_bias, torch.Tensor) else 0.0
        )

        ref_clean = stage1["proposal_quads"].clamp(0.0, 1.0)
        initial_xyxy = stage1["proposal_xyxy"]
        if self.detach_stage1_refiner:
            ref_clean = ref_clean.detach()
            initial_xyxy = initial_xyxy.detach()
        initial_wh = xyxy_to_cxcywh_torch(initial_xyxy)[..., 2:].detach()
        step_scale = (self.residual_scale * initial_wh[:, :, :, None, :]).clamp(0.01, 0.50)
        step_scale_points = step_scale.expand(-1, -1, -1, 4, -1).reshape(b, 16, p, 2)

        if self.stage2_bridge_noise_mode == "replace":
            ref = self._maybe_perturb_points(ref_clean.reshape(b, 16, p, 2), step_scale_points)
            ref = ref.reshape(b, 16, k, 4, 2)
        else:
            ref = ref_clean

        q = (
            query_base[:, :, None, None, :]
            + self.candidate_refine_embed[:, :, :k].to(device=device, dtype=dtype)
            + self.point_index.to(device=device, dtype=dtype)[:, :, None, :, :]
            + self.time_pos.to(device=device, dtype=dtype).unsqueeze(2)
            + self.point_ref_mlp(ref)
            + corner_query_bias_all
        )
        q = self.query_norm(q).reshape(b, 16, p, -1)

        layer_boxes = []
        layer_xyxy = []
        layer_quads = []
        mem_bt = mem16.reshape(b * 16, h * w, -1)
        for i in range(self.decoder_layers):
            sample_ref = ref
            if self.stage2_bridge_noise_mode == "local_only":
                sample_ref = self._maybe_perturb_points(
                    ref.reshape(b, 16, p, 2),
                    step_scale_points,
                ).reshape(b, 16, k, 4, 2)
            if self.refiner_aligned_context:
                grid_size = 5 if self.refiner_aligned_context_points == "aligned_grid_5x5" else 3
                aligned_pts = self.stage1._candidate_aligned_grid_points(sample_ref, grid_size=grid_size)
                aligned_ctx = self.stage1._sample_traj_readout_features(
                    feat16_map,
                    aligned_pts.reshape(b, 16, k * grid_size * grid_size, 2),
                )
                aligned_ctx = aligned_ctx.reshape(b, 16, k, grid_size * grid_size, -1).mean(dim=3)
                aligned_ctx = self.refiner_aligned_context_proj(aligned_ctx)
                aligned_ctx = aligned_ctx[:, :, :, None, :].expand(-1, -1, -1, 4, -1).reshape(b, 16, p, -1)
                q = q + float(self.refiner_aligned_context_scale) * aligned_ctx
            q = q + self.local_point_proj(
                self._sample_point_features(feat16_map, sample_ref.reshape(b, 16, p, 2))
            )

            q_frame = q.reshape(b * 16, p, -1)
            frame_out, _ = self.frame_attn[i](
                self.frame_norms[i](q_frame),
                self.frame_norms[i](q_frame),
                self.frame_norms[i](q_frame),
                need_weights=False,
            )
            q = (q_frame + frame_out).reshape(b, 16, p, -1)

            q_tem = q.permute(0, 2, 1, 3).reshape(b * p, 16, -1)
            tem_out, _ = self.temporal_attn[i](
                self.temporal_norms[i](q_tem),
                self.temporal_norms[i](q_tem),
                self.temporal_norms[i](q_tem),
                need_weights=False,
            )
            q = (q_tem + tem_out).reshape(b, p, 16, -1).permute(0, 2, 1, 3)

            if tool_memory is not None:
                q_tool = q.reshape(b, 16 * p, -1)
                tool_out, _ = self.tool_traj_attn[i](
                    self.tool_query_norms[i](q_tool),
                    self.tool_memory_norms[i](tool_memory),
                    self.tool_memory_norms[i](tool_memory),
                    need_weights=False,
                )
                q = (q_tool + tool_out).reshape(b, 16, p, -1)

            ref_flat = ref.reshape(b, 16, p, 2)
            q_ref = q + self.point_ref_mlp(ref_flat)
            cross_q = self.cross_norms[i](q_ref).reshape(b * 16, p, -1)
            mem_norm = self.memory_norms[i](mem_bt)
            cross_out, _ = self.cross_attn[i](cross_q, mem_norm, mem_norm, need_weights=False)
            q = q + cross_out.reshape(b, 16, p, -1)
            q = q + self.ffns[i](self.ffn_norms[i](q))

            delta = self.point_delta_heads[i](q).reshape(b, 16, k, 4, 2)
            ref_new = (ref + torch.tanh(delta) * step_scale).clamp(0.0, 1.0)
            layer_quads.append(ref_new)
            box_xyxy = quad_to_xyxy(ref_new.reshape(b, 16 * k, 4, 2), min_size=0.02).reshape(b, 16, k, 4)
            layer_xyxy.append(box_xyxy)
            layer_boxes.append(xyxy_to_cxcywh_torch(box_xyxy))
            ref = ref_new.detach() if i + 1 < self.decoder_layers else ref_new

        refined_quads = layer_quads[-1]
        candidate_local_grid_logits = None
        candidate_local_grid_points = None
        candidate_local_grid_base_quads = None
        if self.local_grid_refine:
            candidate_local_grid_base_quads = refined_quads
            refined_quads, candidate_local_grid_logits, candidate_local_grid_points = (
                self._local_grid_refine_candidate_quads(
                    feat16_map,
                    refined_quads,
                )
            )
        refined_xyxy = quad_to_xyxy(refined_quads.reshape(b, 16 * k, 4, 2), min_size=0.02).reshape(b, 16, k, 4)
        refined_cxcywh = xyxy_to_cxcywh_torch(refined_xyxy)
        refined_tokens = q.reshape(b, 16, k, 4, -1).mean(dim=3)
        proposal_oracle_refiner_payload: dict[str, torch.Tensor] = {}
        if self.use_proposal_oracle_refiner and dn_quad_xy is not None and self.training:
            gt_ref = dn_quad_xy.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            with torch.no_grad():
                gt_prop = gt_ref[:, :, None, :, :].expand_as(ref_clean)
                prop_quad_iou = quad_iou_approx(
                    ref_clean.reshape(-1, 4, 2).float(),
                    gt_prop.reshape(-1, 4, 2).float(),
                    grid_size=24,
                ).reshape(ref_clean.shape[:-2])
                oracle_idx = prop_quad_iou.argmax(dim=-1)
            gather_quad_idx = oracle_idx[:, :, None, None, None].expand(-1, -1, 1, 4, 2)
            gather_box_idx = oracle_idx[:, :, None, None].expand(-1, -1, 1, 4)
            oracle_start = ref_clean.gather(2, gather_quad_idx).squeeze(2).detach()
            oracle_quad = refined_quads.gather(2, gather_quad_idx).squeeze(2)
            oracle_xyxy = refined_xyxy.gather(2, gather_box_idx).squeeze(2)
            oracle_cxcywh = refined_cxcywh.gather(2, gather_box_idx).squeeze(2)
            proposal_oracle_refiner_payload = {
                "proposal_oracle_refiner_start_quad": oracle_start,
                "proposal_oracle_refiner_quad": oracle_quad,
                "proposal_oracle_refiner_xyxy": oracle_xyxy,
                "proposal_oracle_refiner_cxcywh": oracle_cxcywh,
            }
        q_grouped = refined_tokens
        refined_logits = self.refined_quality_head(q_grouped).squeeze(-1)
        if self.refined_quality_readout:
            refined_center = refined_quads.mean(dim=-2, keepdim=True)
            if self.refined_quality_readout_points == "aligned_grid_5x5":
                readout_pts = self.stage1._candidate_aligned_grid_points(refined_quads, grid_size=5)
                points_per_candidate = 25
            elif self.refined_quality_readout_points == "aligned_grid_3x3":
                readout_pts = self.stage1._candidate_aligned_grid_points(refined_quads, grid_size=3)
                points_per_candidate = 9
            elif self.refined_quality_readout_points == "center_corners_edges":
                edge_midpoints = 0.5 * (refined_quads + torch.roll(refined_quads, shifts=-1, dims=-2))
                readout_pts = torch.cat([refined_center, refined_quads, edge_midpoints], dim=3)
                points_per_candidate = 9
            else:
                readout_pts = torch.cat([refined_center, refined_quads], dim=3)
                points_per_candidate = 5

            readout_pts = readout_pts.reshape(b, 16, k * points_per_candidate, 2)
            refined_readout = self.stage1._sample_traj_readout_features(feat16_map, readout_pts)
            refined_readout = refined_readout.reshape(b, 16, k, points_per_candidate, -1).mean(dim=3)
            refined_readout = self.refined_quality_readout_proj(refined_readout)
            refined_geom = refined_quads.reshape(b, 16, k, 8)
            readout_tokens = q_grouped
            if self.refined_quality_readout_detach:
                readout_tokens = readout_tokens.detach()
                refined_readout = refined_readout.detach()
                refined_geom = refined_geom.detach()
            refined_quality_delta = self.refined_quality_readout_head(
                torch.cat([readout_tokens, refined_readout, refined_geom], dim=-1)
            ).squeeze(-1)
            if self.refined_quality_readout_replace_logits:
                refined_logits = refined_quality_delta
            else:
                refined_logits = refined_logits + refined_quality_delta
        candidate_local_grid_confidence = None
        if self.local_grid_confidence_score and candidate_local_grid_logits is not None:
            candidate_local_grid_confidence = self._local_grid_confidence_logits(candidate_local_grid_logits)
            if self.local_grid_confidence_detach:
                candidate_local_grid_confidence = candidate_local_grid_confidence.detach()
            refined_logits = refined_logits + float(self.local_grid_confidence_scale) * candidate_local_grid_confidence.to(
                device=refined_logits.device,
                dtype=refined_logits.dtype,
            )
        candidate_mask_logits = None
        candidate_mask_points = None
        candidate_mask_quality_logits = None
        candidate_mask_base_quads = None
        if self.candidate_mask_quality:
            candidate_mask_base_quads = refined_quads
            (
                candidate_mask_logits,
                candidate_mask_points,
                candidate_mask_quality_logits,
            ) = self._candidate_mask_quality_outputs(
                feat16_map,
                refined_quads,
                refined_tokens,
            )
            mask_quality_score = candidate_mask_quality_logits
            if self.candidate_mask_score_detach:
                mask_quality_score = mask_quality_score.detach()
            if float(self.candidate_mask_score_scale) != 0.0:
                refined_logits = refined_logits + float(self.candidate_mask_score_scale) * mask_quality_score.to(
                    device=refined_logits.device,
                    dtype=refined_logits.dtype,
                )
        _refined_quad, refined_weights = self._select_refined_proposals(refined_quads, refined_logits)
        select_quads = refined_quads
        select_xyxy = refined_xyxy
        select_cxcywh = refined_cxcywh
        select_logits = refined_logits
        select_tokens = refined_tokens
        select_source_ids = stage1.get("proposal_source_ids")
        select_active_tip_logits = stage1.get("proposal_active_tip_logits")
        if self.refine_recollect_stage1:
            recollect_quads = refined_quads
            recollect_xyxy = refined_xyxy
            recollect_cxcywh = refined_cxcywh
            recollect_logits = refined_logits
            recollect_source_ids = stage1.get("proposal_source_ids")
            recollect_active_tip_logits = select_active_tip_logits
            if self.refine_recollect_refined_base_only:
                base_k = min(int(self.stage1.proposal_count), int(refined_quads.shape[2]))
                recollect_quads = recollect_quads[:, :, :base_k]
                recollect_xyxy = recollect_xyxy[:, :, :base_k]
                recollect_cxcywh = recollect_cxcywh[:, :, :base_k]
                recollect_logits = recollect_logits[:, :, :base_k]
                recollect_tokens = refined_tokens[:, :, :base_k]
                if recollect_source_ids is not None:
                    recollect_source_ids = recollect_source_ids[:, :, :base_k]
                if recollect_active_tip_logits is not None:
                    recollect_active_tip_logits = recollect_active_tip_logits[:, :, :base_k]
            else:
                recollect_tokens = refined_tokens
            select_quads = torch.cat([stage1["proposal_quads"], recollect_quads], dim=2)
            select_xyxy = torch.cat([stage1["proposal_xyxy"], recollect_xyxy], dim=2)
            select_cxcywh = torch.cat([stage1["proposal_cxcywh"], recollect_cxcywh], dim=2)
            select_logits = torch.cat([stage1["proposal_logits"], recollect_logits], dim=-1)
            stage1_tokens = refined_tokens[:, :, : stage1["proposal_quads"].shape[2]]
            select_tokens = torch.cat([stage1_tokens, recollect_tokens], dim=2)
            if "proposal_source_ids" in stage1:
                select_source_ids = torch.cat(
                    [stage1["proposal_source_ids"], recollect_source_ids],
                    dim=2,
                )
            if select_active_tip_logits is not None and recollect_active_tip_logits is not None:
                select_active_tip_logits = torch.cat(
                    [select_active_tip_logits, recollect_active_tip_logits],
                    dim=-1,
                )
        proposal_local_corr_logits = None
        proposal_local_corr_stats = None
        if self.proposal_local_corr_score and tool_kpt_vec_16 is not None:
            proposal_local_corr_logits, proposal_local_corr_stats = self._proposal_local_corr_logits(
                feat16_map,
                select_quads,
                select_logits,
                tool_kpt_vec_16,
                role_features_16,
                tool_id,
            )
            select_logits = select_logits + float(self.proposal_local_corr_scale) * proposal_local_corr_logits.to(
                device=select_logits.device,
                dtype=select_logits.dtype,
            )
        handoff_source5_readout_logits = None
        if (
            self.proposal_handoff_source5_readout
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
        ):
            select_k = int(select_quads.shape[2])
            source5_grid_size = int(self.proposal_handoff_source5_readout_grid_size)
            source5_pts = self.stage1._candidate_aligned_grid_points(
                select_quads,
                grid_size=source5_grid_size,
            )
            source5_readout = self.stage1._sample_traj_readout_features(
                feat16_map,
                source5_pts.reshape(b, 16, select_k * source5_grid_size * source5_grid_size, 2),
            )
            source5_readout = source5_readout.reshape(
                b,
                16,
                select_k,
                source5_grid_size * source5_grid_size,
                -1,
            ).mean(dim=3)
            source5_readout = self.proposal_handoff_source5_readout_proj(source5_readout)
            source5_geom = select_quads.reshape(b, 16, select_k, 8)
            source5_tail = torch.cat(
                [
                    source5_readout,
                    source5_geom,
                    select_logits.detach().unsqueeze(-1),
                ],
                dim=-1,
            )
            if self.proposal_handoff_source5_readout_detach:
                source5_tail = source5_tail.detach()
            handoff_source5_readout_logits = (
                self.proposal_handoff_source5_readout_head(source5_tail).squeeze(-1)
            )
            readout_source_ids = select_source_ids.detach().clamp(0, 7)
            source5_mask = torch.zeros_like(readout_source_ids, dtype=torch.bool)
            for source_id in self.proposal_handoff_source5_readout_sources:
                source5_mask = source5_mask | (readout_source_ids == int(source_id))
            select_logits = select_logits + (
                float(self.proposal_handoff_source5_readout_scale)
                * handoff_source5_readout_logits.to(
                    device=select_logits.device,
                    dtype=select_logits.dtype,
                )
                * source5_mask.to(device=select_logits.device, dtype=select_logits.dtype)
            )
        handoff_mask_logits = None
        handoff_mask_points = None
        handoff_mask_quality_logits = None
        handoff_mask_source_mask = None
        handoff_mask_base_quads = None
        handoff_mask_refined_quads = None
        if (
            self.proposal_handoff_mask_quality
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
            and select_tokens.shape[:3] == select_quads.shape[:3]
        ):
            handoff_mask_base_quads = select_quads
            (
                handoff_mask_logits,
                handoff_mask_points,
                handoff_mask_quality_logits,
                handoff_mask_source_mask,
            ) = self._proposal_handoff_mask_quality_outputs(
                feat16_map,
                select_quads,
                select_tokens,
                select_logits,
                select_source_ids,
            )
            if float(self.proposal_handoff_mask_refine_scale) > 0.0:
                handoff_mask_refined_quads = self._proposal_handoff_mask_refined_quads(
                    handoff_mask_base_quads,
                    handoff_mask_points,
                    handoff_mask_logits,
                    handoff_mask_source_mask,
                )
                select_quads = handoff_mask_refined_quads
                select_xyxy = quad_to_xyxy(select_quads.reshape(b, 16 * select_quads.shape[2], 4, 2), min_size=0.02).reshape(
                    b,
                    16,
                    select_quads.shape[2],
                    4,
                )
                select_cxcywh = xyxy_to_cxcywh_torch(select_xyxy)
            handoff_mask_score = handoff_mask_quality_logits
            if self.proposal_handoff_mask_score_detach:
                handoff_mask_score = handoff_mask_score.detach()
            select_logits = select_logits + (
                float(self.proposal_handoff_mask_quality_scale)
                * handoff_mask_score.to(device=select_logits.device, dtype=select_logits.dtype)
                * handoff_mask_source_mask.to(device=select_logits.device, dtype=select_logits.dtype)
            )
        handoff_rank_adaptive_delta = None
        handoff_rank_adaptive_rank = None
        if (
            self.proposal_handoff_rank_adaptive_scorer
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
            and select_tokens.shape[:3] == select_quads.shape[:3]
        ):
            # Rank-DETR reorders queries by current confidence and injects
            # rank-aware query content before later scoring.  We keep the bank
            # fixed, but expose that same detached rank context to a residual
            # score head so high-quality candidates can be calibrated by their
            # current position in the handoff list.
            rank_base_logits = select_logits.detach()
            rank_order = rank_base_logits.argsort(dim=-1, descending=True)
            handoff_rank_adaptive_rank = torch.empty_like(rank_order)
            rank_values = torch.arange(
                int(rank_order.shape[-1]),
                device=rank_order.device,
                dtype=rank_order.dtype,
            )
            handoff_rank_adaptive_rank.scatter_(
                -1,
                rank_order,
                rank_values.view(*([1] * (rank_order.dim() - 1)), -1).expand_as(rank_order),
            )
            rank_idx = handoff_rank_adaptive_rank.clamp(
                max=int(self.proposal_handoff_rank_adaptive_max_rank) - 1
            )
            rank_tokens = select_tokens
            rank_quads = select_quads
            rank_logits = select_logits
            if self.proposal_handoff_rank_adaptive_detach:
                rank_tokens = rank_tokens.detach()
                rank_quads = rank_quads.detach()
                rank_logits = rank_logits.detach()
            rank_source_ids = select_source_ids.detach().clamp(0, 7)
            rank_item = (
                rank_tokens
                + self.proposal_handoff_rank_adaptive_rank_emb(rank_idx).to(
                    dtype=rank_tokens.dtype
                )
                + self.proposal_handoff_rank_adaptive_source_emb(rank_source_ids).to(
                    dtype=rank_tokens.dtype
                )
                + self.proposal_handoff_rank_adaptive_geom_mlp(
                    rank_quads.reshape(b, 16, rank_quads.shape[2], 8)
                ).to(dtype=rank_tokens.dtype)
                + self.proposal_handoff_rank_adaptive_logit_mlp(
                    rank_logits.unsqueeze(-1)
                ).to(dtype=rank_tokens.dtype)
            )
            handoff_rank_adaptive_delta = (
                self.proposal_handoff_rank_adaptive_score_head(rank_item).squeeze(-1)
            )
            select_logits = select_logits + (
                float(self.proposal_handoff_rank_adaptive_scale)
                * handoff_rank_adaptive_delta.to(
                    device=select_logits.device,
                    dtype=select_logits.dtype,
                )
            )
        handoff_sparse_set_delta = None
        handoff_sparse_set_mask = None
        if (
            self.proposal_handoff_sparse_set_scorer
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
            and select_tokens.shape[:3] == select_quads.shape[:3]
        ):
            # Sparse R-CNN's dynamic head lets proposal features interact before
            # scoring/regression.  Here we keep the proposal bank fixed but add
            # the same set interaction before final handoff ranking.
            sparse_tokens = select_tokens
            sparse_quads = select_quads
            sparse_logits = select_logits
            if self.proposal_handoff_sparse_set_scorer_detach:
                sparse_tokens = sparse_tokens.detach()
                sparse_quads = sparse_quads.detach()
                sparse_logits = sparse_logits.detach()
            sparse_k = int(sparse_quads.shape[2])
            sparse_source_ids = select_source_ids.detach().clamp(0, 7)
            sparse_geom = sparse_quads.reshape(b, 16, sparse_k, 8)
            sparse_item = (
                sparse_tokens
                + self.proposal_handoff_sparse_set_source_emb(sparse_source_ids).to(
                    dtype=sparse_tokens.dtype
                )
                + self.proposal_handoff_sparse_set_geom_mlp(sparse_geom).to(
                    dtype=sparse_tokens.dtype
                )
                + self.proposal_handoff_sparse_set_logit_mlp(
                    sparse_logits.unsqueeze(-1)
                ).to(dtype=sparse_tokens.dtype)
            )
            sparse_flat = sparse_item.reshape(b * 16, sparse_k, -1)
            sparse_attn, _ = self.proposal_handoff_sparse_set_attn(
                sparse_flat,
                sparse_flat,
                sparse_flat,
                need_weights=False,
            )
            sparse_flat = self.proposal_handoff_sparse_set_attn_norm(sparse_flat + sparse_attn)
            sparse_flat = self.proposal_handoff_sparse_set_ffn_norm(
                sparse_flat + self.proposal_handoff_sparse_set_ffn(sparse_flat)
            )
            handoff_sparse_set_delta = self.proposal_handoff_sparse_set_score_head(
                sparse_flat
            ).reshape(b, 16, sparse_k)
            handoff_sparse_set_mask = torch.zeros_like(sparse_source_ids, dtype=torch.bool)
            for source_id in self.proposal_handoff_sparse_set_scorer_sources:
                handoff_sparse_set_mask = handoff_sparse_set_mask | (
                    sparse_source_ids == int(source_id)
                )
            select_logits = select_logits + (
                float(self.proposal_handoff_sparse_set_scorer_scale)
                * handoff_sparse_set_delta.to(device=select_logits.device, dtype=select_logits.dtype)
                * handoff_sparse_set_mask.to(device=select_logits.device, dtype=select_logits.dtype)
            )
        handoff_source5_refiner_quad = None
        handoff_source5_refiner_base_quad = None
        handoff_source5_refiner_scores = None
        handoff_source5_refiner_weights = None
        handoff_source5_refiner_valid = None
        handoff_source5_refiner_logit = None
        handoff_topk_query_refiner_quads = None
        handoff_topk_query_refiner_base_quads = None
        handoff_topk_query_refiner_logits = None
        handoff_topk_query_refiner_valid = None
        handoff_topk_query_refiner_source_ids = None
        if (
            self.proposal_handoff_source5_refiner
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
            and select_tokens.shape[:3] == select_quads.shape[:3]
        ):
            source5_refiner_source_id = int(self.proposal_handoff_source5_refiner_source_id)
            source5_ids = select_source_ids.detach().clamp(0, 7)
            source5_mask = source5_ids == source5_refiner_source_id
            source5_valid = source5_mask.any(dim=-1)
            ref_tokens = select_tokens
            ref_quads = select_quads
            ref_logits = select_logits
            if self.proposal_handoff_source5_refiner_detach:
                ref_tokens = ref_tokens.detach()
                ref_quads = ref_quads.detach()
                ref_logits = ref_logits.detach()
            ref_geom = ref_quads.reshape(b, 16, ref_quads.shape[2], 8)
            ref_item = self.proposal_handoff_source5_refiner_item_mlp(
                torch.cat([ref_tokens, ref_geom, ref_logits.unsqueeze(-1)], dim=-1)
            )
            ref_score_delta = self.proposal_handoff_source5_refiner_score_head(ref_item).squeeze(-1)
            ref_scores = (
                ref_logits / float(self.proposal_handoff_source5_refiner_temperature)
                + ref_score_delta.to(device=ref_logits.device, dtype=ref_logits.dtype)
            )
            source5_keep = source5_mask
            ref_topk = min(
                int(self.proposal_handoff_source5_refiner_topk),
                int(ref_scores.shape[-1]),
            )
            if ref_topk > 0 and ref_topk < int(ref_scores.shape[-1]):
                source5_rank_scores = ref_scores.masked_fill(~source5_mask, -50.0)
                source5_top_idx = source5_rank_scores.topk(ref_topk, dim=-1).indices
                source5_keep = torch.zeros_like(source5_mask, dtype=torch.bool)
                source5_keep.scatter_(
                    -1,
                    source5_top_idx,
                    source5_mask.gather(-1, source5_top_idx),
                )
            masked_ref_scores = ref_scores.masked_fill(~source5_keep, -50.0)
            source5_weights = F.softmax(masked_ref_scores, dim=-1)
            source5_weights = source5_weights * source5_valid.to(
                device=source5_weights.device,
                dtype=source5_weights.dtype,
            ).unsqueeze(-1)
            handoff_source5_refiner_scores = masked_ref_scores
            handoff_source5_refiner_weights = source5_weights
            handoff_source5_refiner_valid = source5_valid
            handoff_source5_refiner_base_quad = (
                ref_quads * source5_weights[..., None, None]
            ).sum(dim=2).clamp(0.0, 1.0)
            source5_pooled = (ref_item * source5_weights[..., None]).sum(dim=2)
            source5_entropy = -(
                source5_weights.clamp_min(1e-6)
                * source5_weights.clamp_min(1e-6).log()
            ).sum(dim=-1) / math.log(max(int(ref_scores.shape[-1]), 2))
            ref_logits_tanh = torch.tanh(ref_logits)
            source5_max_logit = ref_logits.masked_fill(~source5_mask, -50.0).max(dim=-1).values
            source5_stats = torch.stack(
                [
                    source5_weights.max(dim=-1).values,
                    source5_entropy.to(device=ref_logits.device, dtype=ref_logits.dtype),
                    (source5_weights * ref_logits_tanh).sum(dim=-1),
                    ref_logits_tanh.masked_fill(~source5_mask, -1.0).max(dim=-1).values,
                ],
                dim=-1,
            )
            source5_tail = torch.cat(
                [
                    source5_pooled,
                    handoff_source5_refiner_base_quad.reshape(b, 16, 8),
                    source5_stats,
                ],
                dim=-1,
            )
            source5_delta = torch.tanh(
                self.proposal_handoff_source5_refiner_delta_head(source5_tail)
            ).reshape(b, 16, 4, 2)
            _source5_dirs, source5_lens = quad_axes(
                handoff_source5_refiner_base_quad.reshape(-1, 4, 2)
            )
            source5_step = source5_lens.mean(dim=-1).reshape(b, 16, 1, 1).clamp(0.015, 0.35)
            handoff_source5_refiner_quad = (
                handoff_source5_refiner_base_quad
                + float(self.proposal_handoff_source5_refiner_scale)
                * source5_delta
                * source5_step
            ).clamp(0.0, 1.0)
            ref_logit_delta = self.proposal_handoff_source5_refiner_logit_head(source5_tail).squeeze(-1)
            handoff_source5_refiner_logit = (
                source5_max_logit
                + float(self.proposal_handoff_source5_refiner_init_logit_bias)
                + float(self.proposal_handoff_source5_refiner_logit_scale)
                * ref_logit_delta.to(device=ref_logits.device, dtype=ref_logits.dtype)
            )
            handoff_source5_refiner_logit = torch.where(
                source5_valid,
                handoff_source5_refiner_logit,
                torch.full_like(handoff_source5_refiner_logit, -50.0),
            )
            ref_xyxy = quad_to_xyxy(handoff_source5_refiner_quad, min_size=0.02)
            select_quads = torch.cat([select_quads, handoff_source5_refiner_quad[:, :, None]], dim=2)
            select_xyxy = torch.cat([select_xyxy, ref_xyxy[:, :, None]], dim=2)
            select_cxcywh = torch.cat(
                [select_cxcywh, xyxy_to_cxcywh_torch(ref_xyxy)[:, :, None]],
                dim=2,
            )
            select_logits = torch.cat([select_logits, handoff_source5_refiner_logit[:, :, None]], dim=-1)
            select_tokens = torch.cat([select_tokens, source5_pooled[:, :, None]], dim=2)
            select_source_ids = torch.cat(
                [
                    select_source_ids,
                    torch.full_like(
                        select_source_ids[:, :, :1],
                        int(self.proposal_handoff_source5_refiner_output_source_id),
                    ),
                ],
                dim=2,
            )
            if select_active_tip_logits is not None and select_active_tip_logits.shape[:2] == select_logits.shape[:2]:
                select_active_tip_logits = torch.cat(
                    [select_active_tip_logits, handoff_source5_refiner_logit[:, :, None]],
                    dim=-1,
                )
        if (
            self.proposal_handoff_topk_query_refiner
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
            and select_tokens.shape[:3] == select_quads.shape[:3]
        ):
            (
                handoff_topk_query_refiner_quads,
                handoff_topk_query_refiner_base_quads,
                handoff_topk_query_refiner_logits,
                handoff_topk_query_refiner_valid,
                handoff_topk_query_refiner_source_ids,
                handoff_topk_query_refiner_tokens,
            ) = self._proposal_handoff_topk_query_outputs(
                feat16_map,
                select_quads,
                select_tokens,
                select_logits,
                select_source_ids,
            )
            topk_query_xyxy = quad_to_xyxy(
                handoff_topk_query_refiner_quads.reshape(-1, 4, 2),
                min_size=0.02,
            ).reshape(b, 16, handoff_topk_query_refiner_quads.shape[2], 4)
            select_quads = torch.cat(
                [select_quads, handoff_topk_query_refiner_quads],
                dim=2,
            )
            select_xyxy = torch.cat([select_xyxy, topk_query_xyxy], dim=2)
            select_cxcywh = torch.cat(
                [select_cxcywh, xyxy_to_cxcywh_torch(topk_query_xyxy)],
                dim=2,
            )
            select_logits = torch.cat(
                [select_logits, handoff_topk_query_refiner_logits],
                dim=-1,
            )
            select_tokens = torch.cat(
                [select_tokens, handoff_topk_query_refiner_tokens],
                dim=2,
            )
            select_source_ids = torch.cat(
                [
                    select_source_ids,
                    torch.full_like(
                        handoff_topk_query_refiner_source_ids,
                        int(self.proposal_handoff_topk_query_output_source_id),
                    ),
                ],
                dim=2,
            )
            if select_active_tip_logits is not None and select_active_tip_logits.shape[:2] == select_logits.shape[:2]:
                select_active_tip_logits = torch.cat(
                    [select_active_tip_logits, handoff_topk_query_refiner_logits],
                    dim=-1,
                )
        handoff_oicr_logits_list = None
        if (
            self.proposal_handoff_oicr_refine_times > 0
            and select_tokens.shape[:3] == select_quads.shape[:3]
            and select_logits.shape == select_quads.shape[:3]
        ):
            # PCL/OICR uses several independent refinement streams, where
            # each stream is supervised from the previous proposal scores.
            # Here we apply the same idea to the final handoff bank so the
            # refinement logits align with the candidates actually selected.
            oicr_tokens = select_tokens
            oicr_quads = select_quads.reshape(b, 16, select_quads.shape[2], 8)
            if self.proposal_handoff_oicr_refine_detach:
                oicr_tokens = oicr_tokens.detach()
                oicr_quads = oicr_quads.detach()
            base_oicr_logits = select_logits
            prev_logits = select_logits
            oicr_logits_seq = []
            for head in self.proposal_handoff_oicr_refine_heads:
                prev_in = prev_logits.detach().unsqueeze(-1)
                oicr_in = torch.cat([oicr_tokens, oicr_quads, prev_in], dim=-1)
                delta = head(oicr_in).squeeze(-1)
                prev_logits = prev_logits + float(self.proposal_handoff_oicr_refine_scale) * delta.to(
                    device=prev_logits.device,
                    dtype=prev_logits.dtype,
                )
                oicr_logits_seq.append(prev_logits)
            handoff_oicr_logits_list = torch.stack(oicr_logits_seq, dim=0)
            if self.proposal_handoff_oicr_refine_use_for_selection:
                if self.proposal_handoff_oicr_refine_aggregate == "mean":
                    select_logits = handoff_oicr_logits_list.mean(dim=0)
                elif self.proposal_handoff_oicr_refine_aggregate == "mean_base":
                    select_logits = torch.cat(
                        [
                            base_oicr_logits.unsqueeze(0),
                            handoff_oicr_logits_list,
                        ],
                        dim=0,
                    ).mean(dim=0)
                else:
                    select_logits = prev_logits
        handoff_source_router_logits = None
        handoff_source_router_candidate_logits = None
        if self.proposal_handoff_source_router and select_source_ids is not None:
            if select_source_ids.shape == select_logits.shape:
                select_k = int(select_quads.shape[2])
                router_pts = self.stage1._candidate_aligned_grid_points(select_quads, grid_size=3)
                router_readout = self.stage1._sample_traj_readout_features(
                    feat16_map,
                    router_pts.reshape(b, 16, select_k * 9, 2),
                )
                router_readout = router_readout.reshape(b, 16, select_k, 9, -1).mean(dim=3)
                router_readout = self.proposal_handoff_source_router_readout_proj(router_readout)
                router_geom = select_quads.reshape(b, 16, select_k, 8)
                router_in = torch.cat(
                    [
                        router_readout,
                        router_geom,
                        select_logits.detach().unsqueeze(-1),
                    ],
                    dim=-1,
                )
                handoff_source_router_candidate_logits = (
                    self.proposal_handoff_source_router_candidate_head(router_in).squeeze(-1)
                )
                handoff_source_ids = select_source_ids.detach().clamp(0, 7)
                neg = torch.finfo(handoff_source_router_candidate_logits.dtype).min
                group_logits = []
                group_present = []
                for source_id in range(8):
                    source_mask = handoff_source_ids == int(source_id)
                    group_present.append(source_mask.any(dim=-1))
                    source_scores = handoff_source_router_candidate_logits.masked_fill(~source_mask, neg)
                    group_logits.append(source_scores.max(dim=-1).values)
                handoff_source_router_logits = torch.stack(group_logits, dim=-1)
                handoff_source_router_logits = (
                    handoff_source_router_logits
                    + self.proposal_handoff_source_router_context_head(query_base).to(
                        dtype=handoff_source_router_logits.dtype
                    )
                )
                group_present_t = torch.stack(group_present, dim=-1)
                handoff_source_router_logits = handoff_source_router_logits.masked_fill(
                    ~group_present_t,
                    neg,
                )
                handoff_bonus = handoff_source_router_logits.gather(-1, handoff_source_ids)
                select_logits = select_logits + float(self.proposal_handoff_source_router_scale) * handoff_bonus.to(
                    device=select_logits.device,
                    dtype=select_logits.dtype,
                )
                if abs(float(self.proposal_handoff_source_router_candidate_select_scale)) > 0.0:
                    select_logits = select_logits + float(
                        self.proposal_handoff_source_router_candidate_select_scale
                    ) * handoff_source_router_candidate_logits.to(
                        device=select_logits.device,
                        dtype=select_logits.dtype,
                    )
        if self.proposal_handoff_source_calibration and select_source_ids is not None:
            if select_source_ids.shape == select_logits.shape:
                handoff_source_ids = select_source_ids.detach().clamp(
                    0,
                    self.proposal_handoff_source_logit_bias.num_embeddings - 1,
                )
                handoff_bias = self.proposal_handoff_source_logit_bias(handoff_source_ids).squeeze(-1)
                select_logits = select_logits + float(self.proposal_handoff_source_calibration_scale) * handoff_bias.to(
                    device=select_logits.device,
                    dtype=select_logits.dtype,
                )
        if (
            self.proposal_handoff_source5_topk > 0
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
        ):
            handoff_source_ids = select_source_ids.detach().clamp(0, 7)
            source5_mask = handoff_source_ids == 5
            if bool(source5_mask.any().detach().cpu()):
                k5 = min(int(self.proposal_handoff_source5_topk), int(select_logits.shape[-1]))
                source5_scores = select_logits.masked_fill(~source5_mask, -50.0)
                source5_top_idx = source5_scores.topk(k5, dim=-1).indices
                source5_keep = torch.zeros_like(source5_mask, dtype=torch.bool)
                source5_keep.scatter_(
                    -1,
                    source5_top_idx,
                    source5_mask.gather(-1, source5_top_idx),
                )
                select_logits = torch.where(
                    source5_mask & ~source5_keep,
                    torch.full_like(select_logits, -50.0),
                    select_logits,
                )
        handoff_router_source_gate_mask = None
        if (
            self.proposal_handoff_router_normalized_select
            and (not self.training)
            and handoff_source_router_logits is not None
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
        ):
            handoff_source_ids = select_source_ids.detach().clamp(0, 7)
            centered_logits = torch.full_like(select_logits, -50.0)
            for source_id in range(8):
                source_mask = handoff_source_ids == int(source_id)
                if not bool(source_mask.any().detach().cpu()):
                    continue
                source_count = source_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(
                    dtype=select_logits.dtype
                )
                source_vals = select_logits.masked_fill(~source_mask, 0.0)
                source_mean = source_vals.sum(dim=-1, keepdim=True) / source_count
                source_var = (
                    ((select_logits - source_mean).masked_fill(~source_mask, 0.0) ** 2).sum(
                        dim=-1,
                        keepdim=True,
                    )
                    / source_count
                )
                source_std = torch.sqrt(source_var.clamp_min(1e-4))
                centered_logits = torch.where(
                    source_mask,
                    (select_logits - source_mean) / source_std,
                    centered_logits,
                )
            router_bonus = handoff_source_router_logits.gather(-1, handoff_source_ids)
            select_logits = (
                float(self.proposal_handoff_router_normalized_within_scale)
                * centered_logits
                + float(self.proposal_handoff_router_normalized_source_scale)
                * router_bonus.to(device=select_logits.device, dtype=select_logits.dtype)
            )
        if (
            self.proposal_handoff_router_gated_source
            and (not self.training)
            and handoff_source_router_logits is not None
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
        ):
            handoff_source_ids = select_source_ids.detach().clamp(0, 7)
            router_top_source = handoff_source_router_logits.detach().argmax(dim=-1)
            handoff_router_source_gate_mask = handoff_source_ids == router_top_source.unsqueeze(-1)
            select_logits = torch.where(
                handoff_router_source_gate_mask,
                select_logits,
                torch.full_like(select_logits, -50.0),
                )
        handoff_source_representative_mask = None
        if (
            self.proposal_handoff_source_representatives_only
            and (not self.training)
            and select_source_ids is not None
            and select_source_ids.shape == select_logits.shape
        ):
            handoff_source_ids = select_source_ids.detach().clamp(0, 7)
            handoff_source_representative_mask = torch.zeros_like(select_logits, dtype=torch.bool)
            neg = torch.full_like(select_logits, -50.0)
            for source_id in range(8):
                source_mask = handoff_source_ids == int(source_id)
                source_present = source_mask.any(dim=-1)
                source_scores = select_logits.masked_fill(~source_mask, -50.0)
                source_best = source_scores.argmax(dim=-1, keepdim=True)
                handoff_source_representative_mask.scatter_(
                    -1,
                    source_best,
                    source_present.unsqueeze(-1),
                )
            select_logits = torch.where(handoff_source_representative_mask, select_logits, neg)
        quad, _weights = self._select_refined_proposals(
            select_quads,
            select_logits,
            detach_geometry=bool(self.proposal_handoff_detach_selection_geometry),
        )
        handoff_fusion_base_quad = None
        if (
            self.proposal_handoff_final_fusion
            and select_tokens.shape[:3] == select_quads.shape[:3]
        ):
            handoff_fusion_base_quad = quad
            handoff_geom = select_quads.reshape(b, 16, select_quads.shape[2], 8)
            handoff_quality = torch.tanh(select_logits)[..., None]
            handoff_weight = _weights[..., None]
            handoff_item = self.proposal_handoff_final_fusion_candidate_mlp(
                torch.cat([select_tokens, handoff_geom, handoff_weight, handoff_quality], dim=-1)
            )
            handoff_token = (handoff_item * handoff_weight).sum(dim=2)
            handoff_entropy = -(
                _weights.clamp_min(1e-6) * _weights.clamp_min(1e-6).log()
            ).sum(dim=-1) / math.log(max(int(select_quads.shape[2]), 2))
            quality_tanh = torch.tanh(select_logits)
            handoff_stats = torch.stack(
                [
                    _weights.max(dim=-1).values,
                    handoff_entropy,
                    (_weights * quality_tanh).sum(dim=-1),
                    quality_tanh.max(dim=-1).values,
                ],
                dim=-1,
            )
            handoff_delta = torch.tanh(
                self.proposal_handoff_final_fusion_head(
                    torch.cat([handoff_token, quad.reshape(b, 16, 8), handoff_stats], dim=-1)
                )
            ).reshape(b, 16, 4, 2)
            _handoff_dirs, handoff_lens = quad_axes(quad.reshape(-1, 4, 2))
            handoff_step = handoff_lens.mean(dim=-1).reshape(b, 16, 1, 1).clamp(0.015, 0.35)
            quad = (
                quad + self.proposal_handoff_final_fusion_scale * handoff_delta * handoff_step
            ).clamp(0.0, 1.0)
        local_grid_corner_logits = None
        local_grid_points = None
        local_grid_base_quad = None
        if self.local_grid_refine and self.local_grid_final_refine:
            local_grid_base_quad = quad
            quad, local_grid_corner_logits, local_grid_points = self._local_grid_refine_quad(
                feat16_map,
                quad,
            )
        final_xyxy = quad_to_xyxy(quad, min_size=0.02)

        out = dict(stage1)
        out["boxes_xyxy"] = final_xyxy
        out["boxes_cxcywh"] = xyxy_to_cxcywh_torch(final_xyxy)
        out["coarse_xyxy"] = stage1["boxes_xyxy"]
        out["coarse_cxcywh"] = stage1["boxes_cxcywh"]
        out["decoder_layer_xyxy"] = torch.stack(
            [
                (
                    (cand * refined_weights[..., None]).sum(dim=2)
                    if self.refine_recollect_stage1
                    else (cand * _weights[..., None]).sum(dim=2)
                )
                for cand in layer_xyxy
            ],
            dim=0,
        )
        out["decoder_layer_cxcywh"] = torch.stack(
            [
                xyxy_to_cxcywh_torch(
                    (cand * refined_weights[..., None]).sum(dim=2)
                    if self.refine_recollect_stage1
                    else (cand * _weights[..., None]).sum(dim=2)
                )
                for cand in layer_xyxy
            ],
            dim=0,
        )
        out["quad_xy"] = quad
        out["proposal_quads"] = select_quads
        out["proposal_xyxy"] = select_xyxy
        out["proposal_cxcywh"] = select_cxcywh
        out["proposal_logits"] = select_logits
        if select_source_ids is not None:
            out["proposal_source_ids"] = select_source_ids
        if select_active_tip_logits is not None and select_active_tip_logits.shape == select_logits.shape:
            out["proposal_active_tip_logits"] = select_active_tip_logits
        else:
            out.pop("proposal_active_tip_logits", None)
        if handoff_source_router_logits is not None:
            out["proposal_handoff_source_router_logits"] = handoff_source_router_logits
        if handoff_source_router_candidate_logits is not None:
            out["proposal_handoff_source_router_candidate_logits"] = handoff_source_router_candidate_logits
        if handoff_source_representative_mask is not None:
            out["proposal_handoff_source_representative_mask"] = handoff_source_representative_mask
        if handoff_router_source_gate_mask is not None:
            out["proposal_handoff_router_source_gate_mask"] = handoff_router_source_gate_mask
        if handoff_fusion_base_quad is not None:
            out["proposal_handoff_fusion_base_quad"] = handoff_fusion_base_quad
        if proposal_local_corr_logits is not None:
            out["proposal_local_corr_logits"] = proposal_local_corr_logits
        if proposal_local_corr_stats is not None:
            out["proposal_local_corr_stats"] = proposal_local_corr_stats
        if handoff_source5_readout_logits is not None:
            out["proposal_handoff_source5_readout_logits"] = handoff_source5_readout_logits
        if (
            handoff_mask_logits is not None
            and handoff_mask_points is not None
            and handoff_mask_quality_logits is not None
            and handoff_mask_source_mask is not None
            and handoff_mask_base_quads is not None
        ):
            out["proposal_handoff_mask_logits"] = handoff_mask_logits
            out["proposal_handoff_mask_points"] = handoff_mask_points
            out["proposal_handoff_mask_quality_logits"] = handoff_mask_quality_logits
            out["proposal_handoff_mask_source_mask"] = handoff_mask_source_mask
            out["proposal_handoff_mask_base_quads"] = handoff_mask_base_quads
            if handoff_mask_refined_quads is not None:
                out["proposal_handoff_mask_refined_quads"] = handoff_mask_refined_quads
        if handoff_rank_adaptive_delta is not None:
            out["proposal_handoff_rank_adaptive_delta"] = handoff_rank_adaptive_delta
            out["proposal_handoff_rank_adaptive_rank"] = handoff_rank_adaptive_rank
        if handoff_sparse_set_delta is not None:
            out["proposal_handoff_sparse_set_delta"] = handoff_sparse_set_delta
            out["proposal_handoff_sparse_set_mask"] = handoff_sparse_set_mask
        if handoff_source5_refiner_quad is not None:
            out["proposal_handoff_source5_refiner_quad"] = handoff_source5_refiner_quad
            out["proposal_handoff_source5_refiner_base_quad"] = handoff_source5_refiner_base_quad
            out["proposal_handoff_source5_refiner_scores"] = handoff_source5_refiner_scores
            out["proposal_handoff_source5_refiner_weights"] = handoff_source5_refiner_weights
            out["proposal_handoff_source5_refiner_valid"] = handoff_source5_refiner_valid
            out["proposal_handoff_source5_refiner_logit"] = handoff_source5_refiner_logit
        if handoff_topk_query_refiner_quads is not None:
            out["proposal_handoff_topk_query_refiner_quads"] = (
                handoff_topk_query_refiner_quads
            )
            out["proposal_handoff_topk_query_refiner_base_quads"] = (
                handoff_topk_query_refiner_base_quads
            )
            out["proposal_handoff_topk_query_refiner_logits"] = (
                handoff_topk_query_refiner_logits
            )
            out["proposal_handoff_topk_query_refiner_valid"] = (
                handoff_topk_query_refiner_valid
            )
            out["proposal_handoff_topk_query_refiner_source_ids"] = (
                handoff_topk_query_refiner_source_ids
            )
        if handoff_oicr_logits_list is not None:
            out["proposal_oicr_logits_list"] = handoff_oicr_logits_list
            out["proposal_oicr_logits"] = handoff_oicr_logits_list[-1]
        out["proposal_selection_weights"] = _weights
        if local_grid_corner_logits is not None and local_grid_points is not None:
            out["local_grid_corner_logits"] = local_grid_corner_logits
            out["local_grid_points"] = local_grid_points
            if local_grid_base_quad is not None:
                out["local_grid_base_quad"] = local_grid_base_quad
        if candidate_local_grid_logits is not None and candidate_local_grid_points is not None:
            out["candidate_local_grid_corner_logits"] = candidate_local_grid_logits
            out["candidate_local_grid_points"] = candidate_local_grid_points
            if candidate_local_grid_base_quads is not None:
                out["candidate_local_grid_base_quads"] = candidate_local_grid_base_quads
        if candidate_local_grid_confidence is not None:
            out["candidate_local_grid_confidence_logits"] = candidate_local_grid_confidence
        if (
            candidate_mask_logits is not None
            and candidate_mask_points is not None
            and candidate_mask_quality_logits is not None
        ):
            out["candidate_mask_logits"] = candidate_mask_logits
            out["candidate_mask_points"] = candidate_mask_points
            out["candidate_mask_quality_logits"] = candidate_mask_quality_logits
            if candidate_mask_base_quads is not None:
                out["candidate_mask_base_quads"] = candidate_mask_base_quads
        out["stage1_proposal_quads"] = stage1["proposal_quads"]
        out["stage1_proposal_xyxy"] = stage1["proposal_xyxy"]
        out["stage1_full_proposal_quads"] = stage1_full_proposal_quads
        out["stage1_full_proposal_xyxy"] = stage1_full_proposal_xyxy
        out["stage1_full_proposal_logits"] = stage1_full_proposal_logits
        if topk_indices is not None:
            out["stage1_refine_topk_indices"] = topk_indices
        out.update(proposal_oracle_refiner_payload)
        return out


class H219TrajectoryContactStage1(nn.Module):
    """Trajectory-conditioned contact/hotspot bottleneck for oriented ROI.

    This mirrors the useful part of HOI-Forecast / VRB / Afformer style heads:
    predict a spatial contact/support distribution first, pool V-JEPA features
    through that distribution, then decode the region geometry.  The heatmap is
    not the final answer; it is the query prior for the tight oriented region.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        residual_scale: float,
        tool_mask_spatial: bool = False,
    ) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.use_tool_mask = bool(use_tool_mask)
        self.use_tool_mask_spatial = bool(use_tool_mask) and bool(tool_mask_spatial)
        self.use_tool_keypoints = bool(use_tool_keypoints)
        self.residual_scale = float(residual_scale)
        self.token_norm = nn.LayerNorm(1024)
        self.token_proj = nn.Linear(1024, c)
        in_ch = c + 2 + (1 if self.use_tool_mask_spatial else 0) + (3 if self.use_tool_keypoints else 0)
        self.role_mlp = nn.Sequential(nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_kpt_mlp = nn.Sequential(nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_mask_geom_mlp = nn.Sequential(nn.Linear(5, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_traj_mlp = nn.Sequential(nn.LayerNorm(16 * 9), nn.Linear(16 * 9, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_traj_token_mlp = nn.Sequential(nn.LayerNorm(13), nn.Linear(13, c), nn.SiLU(), nn.Linear(c, c))
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.cond_to_gamma_beta = nn.Sequential(nn.SiLU(), nn.Linear(c, 2 * c))
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
        )
        self.contact_head = nn.Conv2d(c, 1, 1)
        self.corner_heatmap_head = nn.Conv2d(c, 4, 1)
        seq_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=4,
            dim_feedforward=4 * c,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(seq_layer, num_layers=2)
        seq_in_dim = 4 * c + 2 + (c if self.use_tool_mask else 0)
        self.seq_in = nn.Sequential(nn.LayerNorm(seq_in_dim), nn.Linear(seq_in_dim, c), nn.SiLU())
        self.size_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.axis_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.point_offset_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 8))
        nn.init.zeros_(self.size_head[-1].weight)
        nn.init.constant_(self.size_head[-1].bias, -2.1)
        nn.init.zeros_(self.axis_head[-1].weight)
        nn.init.zeros_(self.axis_head[-1].bias)
        nn.init.zeros_(self.point_offset_head[-1].weight)
        nn.init.zeros_(self.point_offset_head[-1].bias)
        base = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            dtype=torch.float32,
        )
        self.register_buffer("base_corners", base, persistent=False)
        self.register_buffer("grid_xy", H161RoiCondLite._make_grid(24, 24), persistent=False)

    def _tool_mask_geom8(self, tool_mask_8: torch.Tensor) -> torch.Tensor:
        mask = tool_mask_8.squeeze(2)
        b, t, h, w = mask.shape
        grid = self.grid_xy.to(device=mask.device, dtype=mask.dtype)
        gx = grid[0].view(1, 1, h, w)
        gy = grid[1].view(1, 1, h, w)
        mass_raw = mask.sum(dim=(-1, -2))
        mass = mass_raw.clamp_min(1e-6)
        cx = (mask * gx).sum(dim=(-1, -2)) / mass
        cy = (mask * gy).sum(dim=(-1, -2)) / mass
        area = mass_raw / float(h * w)
        sx = torch.sqrt(((gx - cx[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        sy = torch.sqrt(((gy - cy[..., None, None]) ** 2 * mask).sum(dim=(-1, -2)) / mass).clamp(0.0, 1.0)
        empty = (mass_raw <= 0.0).to(mask.dtype)
        cx = cx * (1.0 - empty) + 0.5 * empty
        cy = cy * (1.0 - empty) + 0.5 * empty
        return torch.stack([cx, cy, area, sx, sy], dim=-1)

    def _quad_from_local(self, center: torch.Tensor, size: torch.Tensor, axis: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        axis_a = F.normalize(axis, dim=-1, eps=1e-6)
        axis_b = torch.stack([-axis_a[..., 1], axis_a[..., 0]], dim=-1)
        local = self.base_corners.to(device=center.device, dtype=center.dtype).view(1, 1, 4, 2)
        local = local + offsets.view(*center.shape[:2], 4, 2)
        local = local * size[:, :, None, :]
        quad = center[:, :, None, :] + local[..., :1] * axis_a[:, :, None, :] + local[..., 1:] * axis_b[:, :, None, :]
        return quad.clamp(0.0, 1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t8, h, w, _ = tokens.shape
        device = tokens.device
        dtype = tokens.dtype
        x = self.token_proj(self.token_norm(tokens)).permute(0, 1, 4, 2, 3).contiguous()
        role8 = role_features_16.reshape(b, 8, 2, 4).mean(dim=2)
        z = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        if self.use_tool_keypoints:
            traj_global = self.tool_traj_mlp(tool_kpt_vec_16.to(device=device, dtype=dtype).reshape(b, 16 * 9))
        else:
            traj_global = torch.zeros_like(z)
        cond_global = z + traj_global
        role_bias8 = self.role_mlp(role8).view(b, t8, -1, 1, 1) + cond_global[:, None, :, None, None]
        if self.use_tool_keypoints:
            kpt8 = tool_kpt_vec_16.reshape(b, 8, 2, 9).mean(dim=2)
            role_bias8 = role_bias8 + self.tool_kpt_mlp(kpt8).view(b, t8, -1, 1, 1)
        gamma_beta = self.cond_to_gamma_beta(cond_global).view(b, 1, 2, -1, 1, 1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_beta[:, :, 0])
        beta = 0.1 * torch.tanh(gamma_beta[:, :, 1])
        x = gamma * (x + role_bias8) + beta

        grid = self.grid_xy.to(device=device, dtype=dtype).view(1, 1, 2, h, w).expand(b, t8, -1, -1, -1)
        feats = [x, grid]
        if self.use_tool_mask_spatial:
            feats.append(tool_mask_8.to(device=device, dtype=dtype))
        if self.use_tool_keypoints:
            feats.append(tool_kpt_heatmaps_8.to(device=device, dtype=dtype))
        y = self.spatial(torch.cat(feats, dim=2).reshape(b * t8, -1, h, w))

        contact_logits_8 = self.contact_head(y).view(b, t8, h * w)
        contact_prob = F.softmax(contact_logits_8, dim=-1)
        grid_flat = self.grid_xy.to(device=device, dtype=dtype).view(2, h * w).transpose(0, 1)
        centers8 = torch.matmul(contact_prob, grid_flat)
        pooled8 = (y.flatten(2) * contact_prob.reshape(b * t8, 1, h * w)).sum(dim=-1).view(b, t8, -1)

        visual16 = F.interpolate(pooled8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        centers16 = F.interpolate(centers8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
        role16 = self.role_mlp(role_features_16)
        cond16 = cond_global[:, None, :].expand(-1, 16, -1)
        if self.use_tool_keypoints:
            kpt16 = self.tool_kpt_mlp(tool_kpt_vec_16.to(device=device, dtype=dtype))
            traj16 = self.tool_traj_token_mlp(
                torch.cat(
                    [
                        tool_kpt_vec_16.to(device=device, dtype=dtype),
                        role_features_16.to(device=device, dtype=dtype),
                    ],
                    dim=-1,
                )
            )
        else:
            kpt16 = torch.zeros_like(role16)
            traj16 = torch.zeros_like(role16)
        seq_parts = [visual16, role16 + cond16, kpt16, traj16, centers16]
        if self.use_tool_mask:
            mask_geom8 = self._tool_mask_geom8(tool_mask_8.to(device=device, dtype=dtype))
            mask_geom16 = F.interpolate(mask_geom8.transpose(1, 2), size=16, mode="linear", align_corners=True).transpose(1, 2)
            seq_parts.append(self.tool_mask_geom_mlp(mask_geom16))
        seq = self.seq_in(torch.cat(seq_parts, dim=-1))
        seq = self.temporal(seq)

        size = torch.sigmoid(self.size_head(seq)).clamp(0.02, 0.95)
        base_axis = torch.zeros_like(self.axis_head(seq))
        base_axis[..., 0] = 1.0
        axis = F.normalize(base_axis + 0.75 * torch.tanh(self.axis_head(seq)), dim=-1, eps=1e-6)
        offsets = 0.20 * torch.tanh(self.point_offset_head(seq).view(b, 16, 4, 2))
        quad = self._quad_from_local(centers16.clamp(0.0, 1.0), size, axis, offsets)
        box_xyxy = quad_to_xyxy(quad, min_size=0.02)
        return {
            "boxes_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "boxes_xyxy": box_xyxy,
            "center_logits_8": contact_logits_8.view(b, t8, h, w),
            "support_logits_8": contact_logits_8.view(b, t8, h, w),
            "corner_logits_8": self.corner_heatmap_head(y).view(b, t8, 4, h, w),
            "coarse_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "coarse_xyxy": box_xyxy,
            "quad_xy": quad,
        }


class H219TrajectoryContactPointTemporalRegion(H191QuadLocalPointTemporalRegion):
    """Contact-hotspot Stage1 plus V-JEPA local point temporal refinement."""

    def __init__(
        self,
        hidden_dim: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        residual_scale: float,
        decoder_layers: int,
        query_noise_scale: float = 0.0,
        stage2_bridge_noise_mode: str = "local_only",
        use_tool_trajectory_bias: bool = False,
        use_tool_trajectory_attn: bool = False,
        stage1_drop_tool_keypoints: bool = False,
        detach_stage1_refiner: bool = False,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            use_tool_mask=use_tool_mask,
            use_tool_keypoints=use_tool_keypoints,
            residual_scale=residual_scale,
            decoder_layers=decoder_layers,
            query_noise_scale=query_noise_scale,
            stage2_bridge_noise_mode=stage2_bridge_noise_mode,
            use_tool_trajectory_bias=use_tool_trajectory_bias,
            use_tool_trajectory_attn=use_tool_trajectory_attn,
            stage1_drop_tool_keypoints=stage1_drop_tool_keypoints,
            detach_stage1_refiner=detach_stage1_refiner,
        )
        stage1_use_tool_keypoints = bool(use_tool_keypoints) and not bool(stage1_drop_tool_keypoints)
        self.stage1 = H219TrajectoryContactStage1(
            hidden_dim=hidden_dim,
            use_tool_mask=use_tool_mask,
            use_tool_keypoints=stage1_use_tool_keypoints,
            residual_scale=residual_scale,
            tool_mask_spatial=False,
        )


class H259TrajectoryPriorTemporalRegion(nn.Module):
    """Trajectory/action-range prior without V-JEPA visual tokens.

    This model is a condition-use proof model.  Tool trajectory is not appended
    as a weak side feature; it provides the geometric reference center/axis.
    Action-range role and tool/action IDs gate that trajectory representation.
    A useful checkpoint must degrade under condition ablations.
    """

    def __init__(self, hidden_dim: int, residual_scale: float = 0.35) -> None:
        super().__init__()
        c = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.tool_mlp = nn.Sequential(nn.LayerNorm(9), nn.Linear(9, c), nn.SiLU(), nn.Linear(c, c))
        self.role_mlp = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, c), nn.SiLU(), nn.Linear(c, c))
        self.traj_global_mlp = nn.Sequential(
            nn.LayerNorm(16 * 13),
            nn.Linear(16 * 13, c),
            nn.SiLU(),
            nn.Linear(c, c),
        )
        self.tool_emb = nn.Embedding(9, c)
        self.act_emb = nn.Embedding(9, c)
        self.combo_emb = nn.Embedding(17, c)
        self.role_gate = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.Sigmoid())
        self.id_gate = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.Sigmoid())
        self.fuse = nn.Sequential(nn.LayerNorm(4 * c + 2), nn.Linear(4 * c + 2, c), nn.SiLU())
        seq_layer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=4,
            dim_feedforward=4 * c,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(seq_layer, num_layers=2)
        self.center_delta = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.size_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.axis_delta = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 2))
        self.corner_offset = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.SiLU(), nn.Linear(c, 8))
        self.time_pos = nn.Parameter(torch.zeros(1, 16, c))
        base = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            dtype=torch.float32,
        )
        self.register_buffer("base_corners", base, persistent=False)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.zeros_(self.center_delta[-1].weight)
        nn.init.zeros_(self.center_delta[-1].bias)
        nn.init.zeros_(self.size_head[-1].weight)
        nn.init.constant_(self.size_head[-1].bias, -2.1)
        nn.init.zeros_(self.axis_delta[-1].weight)
        nn.init.zeros_(self.axis_delta[-1].bias)
        nn.init.zeros_(self.corner_offset[-1].weight)
        nn.init.zeros_(self.corner_offset[-1].bias)

    @staticmethod
    def _tool_anchor_and_axis(tool_kpt_vec_16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = tool_kpt_vec_16.shape[0]
        xy = tool_kpt_vec_16[..., :6].reshape(b, 16, 3, 2).clamp(0.0, 1.0)
        vis = tool_kpt_vec_16[..., 6:9].clamp(0.0, 1.0)
        mass = vis.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        anchor = (xy * vis.unsqueeze(-1)).sum(dim=-2) / mass
        empty = (vis.sum(dim=-1, keepdim=True) <= 0.0).to(xy.dtype)
        anchor = anchor * (1.0 - empty) + 0.5 * empty

        shaft = xy[:, :, 0] - xy[:, :, 2]
        shaft_norm = shaft.norm(dim=-1, keepdim=True)
        default_axis = torch.zeros_like(shaft)
        default_axis[..., 0] = 1.0
        axis = torch.where(shaft_norm > 1e-4, shaft / shaft_norm.clamp_min(1e-6), default_axis)
        return anchor, axis

    def forward(
        self,
        tokens: torch.Tensor,
        role_features_16: torch.Tensor,
        tool_mask_8: torch.Tensor,
        tool_kpt_heatmaps_8: torch.Tensor,
        tool_kpt_vec_16: torch.Tensor,
        tool_id: torch.Tensor,
        act_id: torch.Tensor,
        combo_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del tokens, tool_mask_8, tool_kpt_heatmaps_8
        device = tool_kpt_vec_16.device
        dtype = tool_kpt_vec_16.dtype
        b = tool_kpt_vec_16.shape[0]
        role = role_features_16.to(device=device, dtype=dtype)
        tool_vec = tool_kpt_vec_16.to(device=device, dtype=dtype)
        anchor, tool_axis = self._tool_anchor_and_axis(tool_vec)

        tool_tok = self.tool_mlp(tool_vec)
        role_tok = self.role_mlp(role)
        cond_global_in = torch.cat([tool_vec, role], dim=-1).reshape(b, 16 * 13)
        global_tok = self.traj_global_mlp(cond_global_in)
        ids = self.tool_emb(tool_id) + self.act_emb(act_id) + self.combo_emb(combo_id)
        gate = 0.5 * (self.role_gate(role_tok) + self.id_gate(ids[:, None, :]))
        x = torch.cat(
            [
                tool_tok * (1.0 + gate),
                role_tok,
                global_tok[:, None, :].expand(-1, 16, -1),
                ids[:, None, :].expand(-1, 16, -1),
                anchor,
            ],
            dim=-1,
        )
        x = self.fuse(x) + self.time_pos.to(device=device, dtype=dtype)
        x = self.temporal(x)

        center = (anchor + float(self.residual_scale) * torch.tanh(self.center_delta(x))).clamp(0.0, 1.0)
        size = torch.sigmoid(self.size_head(x)).clamp(0.02, 0.95)
        axis = F.normalize(tool_axis + 0.75 * torch.tanh(self.axis_delta(x)), dim=-1, eps=1e-6)
        perp = torch.stack([-axis[..., 1], axis[..., 0]], dim=-1)
        base = self.base_corners.to(device=device, dtype=dtype).view(1, 1, 4, 2)
        local = base + 0.20 * torch.tanh(self.corner_offset(x).view(b, 16, 4, 2))
        local = local * size[:, :, None, :]
        quad = center[:, :, None, :] + local[..., :1] * axis[:, :, None, :] + local[..., 1:] * perp[:, :, None, :]
        quad = quad.clamp(0.0, 1.0)
        box_xyxy = quad_to_xyxy(quad, min_size=0.02)
        center_logits_8 = torch.zeros((b, 8, 24, 24), device=device, dtype=dtype)
        return {
            "boxes_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "boxes_xyxy": box_xyxy,
            "coarse_cxcywh": xyxy_to_cxcywh_torch(box_xyxy),
            "coarse_xyxy": box_xyxy,
            "center_logits_8": center_logits_8,
            "quad_xy": quad,
            "trajectory_prior_quad": quad,
            "trajectory_prior_xyxy": box_xyxy,
        }


class H270TrajectoryPriorPointTemporalRegion(H191QuadLocalPointTemporalRegion):
    """Trajectory-prior Stage1 plus V-JEPA local point-temporal refinement.

    Earlier proposal runs appended the trajectory prior as one selectable
    candidate, so the model could still ignore it and rely on visual proposal
    logits.  This variant makes the trajectory/action-range prior the mandatory
    Stage1 geometry.  The V-JEPA path is only asked to read/refine around that
    trajectory-conditioned quad, matching the H270 handoff hypothesis.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_tool_mask: bool,
        use_tool_keypoints: bool,
        residual_scale: float,
        decoder_layers: int,
        query_noise_scale: float = 0.0,
        dn_query_noise_scale: float = 0.0,
        stage2_bridge_noise_mode: str = "local_only",
        use_tool_trajectory_bias: bool = False,
        use_tool_trajectory_attn: bool = False,
        use_tool_corner_query_bias: bool = False,
        detach_stage1_refiner: bool = False,
    ) -> None:
        if not bool(use_tool_keypoints):
            raise ValueError("trajectory_prior_point_temporal requires --use-tool-keypoints")
        super().__init__(
            hidden_dim=hidden_dim,
            use_tool_mask=use_tool_mask,
            use_tool_keypoints=use_tool_keypoints,
            residual_scale=residual_scale,
            decoder_layers=decoder_layers,
            query_noise_scale=query_noise_scale,
            dn_query_noise_scale=dn_query_noise_scale,
            use_proposal_oracle_refiner=False,
            proposal_oracle_refiner_noise_scale=0.0,
            stage2_bridge_noise_mode=stage2_bridge_noise_mode,
            use_tool_trajectory_bias=use_tool_trajectory_bias,
            use_tool_trajectory_attn=use_tool_trajectory_attn,
            use_tool_corner_query_bias=use_tool_corner_query_bias,
            stage1_drop_tool_keypoints=False,
            detach_stage1_refiner=detach_stage1_refiner,
        )
        self.stage1 = H259TrajectoryPriorTemporalRegion(
            hidden_dim=hidden_dim,
            residual_scale=min(max(float(residual_scale), 0.10), 0.75),
        )


def target_center_indices(boxes_cxcywh: torch.Tensor, grid_size: int = 24) -> torch.Tensor:
    cxcy = boxes_cxcywh[:, :, :2].reshape(boxes_cxcywh.shape[0], 8, 2, 2).mean(dim=2)
    xy = (cxcy * grid_size).long().clamp(0, grid_size - 1)
    return xy[..., 1] * grid_size + xy[..., 0]


def target_corner_heatmaps_8(quad_xy: torch.Tensor, grid_size: int = 24, sigma_px: float = 1.25) -> torch.Tensor:
    quad8 = quad_xy.reshape(quad_xy.shape[0], 8, 2, 4, 2).mean(dim=2)
    yy, xx = torch.meshgrid(
        torch.arange(grid_size, device=quad_xy.device, dtype=quad_xy.dtype) + 0.5,
        torch.arange(grid_size, device=quad_xy.device, dtype=quad_xy.dtype) + 0.5,
        indexing="ij",
    )
    gx = xx.view(1, 1, 1, grid_size, grid_size)
    gy = yy.view(1, 1, 1, grid_size, grid_size)
    px = (quad8[..., 0] * grid_size).view(*quad8.shape[:3], 1, 1)
    py = (quad8[..., 1] * grid_size).view(*quad8.shape[:3], 1, 1)
    sigma2 = max(float(sigma_px), 1e-3) ** 2
    hm = torch.exp(-0.5 * ((gx - px) ** 2 + (gy - py) ** 2) / sigma2)
    hm = hm / hm.flatten(-2).amax(dim=-1).clamp_min(1e-6).view(*hm.shape[:3], 1, 1)
    return hm.clamp(0.0, 1.0)


def modified_focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits_f = logits.float()
    target_f = target.float()
    pred = torch.sigmoid(logits_f).clamp(1e-4, 1.0 - 1e-4)
    pos = target_f.eq(1.0).to(logits_f.dtype)
    neg = target_f.lt(1.0).to(logits_f.dtype)
    neg_weights = (1.0 - target_f).pow(4.0)
    pos_bce = F.binary_cross_entropy_with_logits(logits_f, torch.ones_like(logits_f), reduction="none")
    neg_bce = F.binary_cross_entropy_with_logits(logits_f, torch.zeros_like(logits_f), reduction="none")
    pos_loss = pos_bce * (1.0 - pred).pow(2.0) * pos
    neg_loss = neg_bce * pred.pow(2.0) * neg_weights * neg
    num_pos = pos.sum().clamp_min(1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def target_support_heatmaps_8(
    boxes_cxcywh: torch.Tensor,
    grid_size: int = 24,
    gamma: float = 0.5,
) -> torch.Tensor:
    box8 = boxes_cxcywh.reshape(boxes_cxcywh.shape[0], 8, 2, 4).mean(dim=2)
    cx, cy, bw, bh = box8.unbind(dim=-1)
    yy, xx = torch.meshgrid(
        torch.linspace(0.5 / grid_size, 1.0 - 0.5 / grid_size, grid_size, device=boxes_cxcywh.device),
        torch.linspace(0.5 / grid_size, 1.0 - 0.5 / grid_size, grid_size, device=boxes_cxcywh.device),
        indexing="ij",
    )
    gx = xx.view(1, 1, grid_size, grid_size)
    gy = yy.view(1, 1, grid_size, grid_size)
    half_w = (bw * 0.5).clamp_min(1e-4).view(*box8.shape[:2], 1, 1)
    half_h = (bh * 0.5).clamp_min(1e-4).view(*box8.shape[:2], 1, 1)
    cx = cx.view(*box8.shape[:2], 1, 1)
    cy = cy.view(*box8.shape[:2], 1, 1)
    sx = (1.0 - ((gx - cx).abs() / half_w)).clamp(0.0, 1.0)
    sy = (1.0 - ((gy - cy).abs() / half_h)).clamp(0.0, 1.0)
    support = torch.minimum(sx, sy).pow(float(gamma))
    return support.clamp(0.0, 1.0)


def target_quad_support_heatmaps_8(
    quad_xy: torch.Tensor,
    grid_size: int = 24,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Soft oriented support target from the native four-point region.

    The previous support target used the enclosing AABB, which teaches the
    contact bottleneck a rectangular context region.  For oriented tight ROI
    training, the support distribution should preserve the quad's center,
    orientation, and aspect before the final bbox/quad regression losses.
    """

    quad8 = quad_xy.reshape(quad_xy.shape[0], 8, 2, 4, 2).mean(dim=2).clamp(0.0, 1.0)
    center = quad8.mean(dim=-2)
    dirs, lens = quad_axes(quad8)
    half = (0.5 * lens).clamp_min(1e-4)
    yy, xx = torch.meshgrid(
        torch.linspace(0.5 / grid_size, 1.0 - 0.5 / grid_size, grid_size, device=quad_xy.device),
        torch.linspace(0.5 / grid_size, 1.0 - 0.5 / grid_size, grid_size, device=quad_xy.device),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).view(1, 1, grid_size, grid_size, 2)
    diff = grid - center[:, :, None, None, :]
    proj0 = (diff * dirs[:, :, None, None, 0, :]).sum(dim=-1).abs() / half[:, :, None, None, 0]
    proj1 = (diff * dirs[:, :, None, None, 1, :]).sum(dim=-1).abs() / half[:, :, None, None, 1]
    support = torch.minimum(1.0 - proj0, 1.0 - proj1).clamp(0.0, 1.0)
    return support.pow(float(gamma)).clamp(0.0, 1.0)


def soft_support_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 4.0,
) -> torch.Tensor:
    logits_f = logits.float()
    target_f = target.float()
    bce = F.binary_cross_entropy_with_logits(logits_f, target_f, reduction="none")
    weight = 1.0 + float(pos_weight) * target_f
    mse = F.mse_loss(torch.sigmoid(logits_f), target_f, reduction="none")
    return ((bce + mse) * weight).sum() / weight.sum().clamp_min(1e-6)


def target_box_masks_8(boxes_cxcywh: torch.Tensor, grid_size: int = 24) -> torch.Tensor:
    box8 = boxes_cxcywh.reshape(boxes_cxcywh.shape[0], 8, 2, 4).mean(dim=2)
    xyxy = cxcywh_to_xyxy(box8)
    yy, xx = torch.meshgrid(
        torch.linspace(0.5 / grid_size, 1.0 - 0.5 / grid_size, grid_size, device=boxes_cxcywh.device),
        torch.linspace(0.5 / grid_size, 1.0 - 0.5 / grid_size, grid_size, device=boxes_cxcywh.device),
        indexing="ij",
    )
    gx = xx.view(1, 1, grid_size, grid_size)
    gy = yy.view(1, 1, grid_size, grid_size)
    x0, y0, x1, y1 = xyxy.unbind(dim=-1)
    return ((gx >= x0[..., None, None]) & (gx <= x1[..., None, None]) & (gy >= y0[..., None, None]) & (gy <= y1[..., None, None])).float()


def target_edge_indices_8(boxes_xyxy: torch.Tensor, grid_size: int = 24) -> dict[str, torch.Tensor]:
    box8 = boxes_xyxy.reshape(boxes_xyxy.shape[0], 8, 2, 4).mean(dim=2)
    idx = (box8 * grid_size).long().clamp(0, grid_size - 1)
    return {
        "x0": idx[..., 0],
        "y0": idx[..., 1],
        "x1": idx[..., 2],
        "y1": idx[..., 3],
    }


def distribution_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    n_bins = logits.shape[-1]
    t = (target.clamp(0.0, 1.0) * (n_bins - 1)).clamp(0.0, n_bins - 1 - 1e-4)
    left = t.floor().long()
    right = (left + 1).clamp(max=n_bins - 1)
    wl = (right.to(t.dtype) - t).clamp(0.0, 1.0)
    wr = (t - left.to(t.dtype)).clamp(0.0, 1.0)
    flat_logits = logits.reshape(-1, n_bins)
    loss_l = F.cross_entropy(flat_logits, left.reshape(-1), reduction="none").reshape_as(t)
    loss_r = F.cross_entropy(flat_logits, right.reshape(-1), reduction="none").reshape_as(t)
    return loss_l * wl + loss_r * wr


def weighted_mean(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    while weight.ndim < x.ndim:
        weight = weight.unsqueeze(-1)
    return (x * weight).sum() / weight.sum().clamp_min(1e-6)


class ProposalRankSort(torch.autograd.Function):
    """Small-K adaptation of the official RankSortLoss selector gradient.

    Source mechanism:
      external_repos/selector_refs/RankSortLoss/mmdet/models/losses/ranking_losses.py

    The original detector loss ranks positives above negatives and sorts
    positives by localization quality.  Here each frame has only K proposal
    candidates, so we apply the same identity-update gradient per frame.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor, delta_rs: float = 0.50, eps: float = 1e-10):
        grads = torch.zeros_like(logits)
        fg_labels = targets > 0.0
        fg_logits = logits[fg_labels]
        fg_targets = targets[fg_labels]
        fg_num = int(fg_logits.numel())

        if fg_num == 0:
            ctx.save_for_backward(grads)
            zero = logits.sum() * 0.0
            return zero, zero

        delta = float(delta_rs)
        threshold_logit = torch.min(fg_logits) - delta
        bg_labels = (targets == 0.0) & (logits >= threshold_logit)
        bg_logits = logits[bg_labels]

        fg_grad = torch.zeros_like(fg_logits)
        bg_grad = torch.zeros_like(bg_logits)
        ranking_error = torch.zeros_like(fg_logits)
        sorting_error = torch.zeros_like(fg_logits)

        order = torch.argsort(fg_logits)
        for order_idx in order:
            ii = int(order_idx.item())
            fg_rel = fg_logits - fg_logits[ii]
            bg_rel = bg_logits - fg_logits[ii]

            if delta > 0:
                fg_rel = torch.clamp(fg_rel / (2.0 * delta) + 0.5, min=0.0, max=1.0)
                bg_rel = torch.clamp(bg_rel / (2.0 * delta) + 0.5, min=0.0, max=1.0)
            else:
                fg_rel = (fg_rel >= 0.0).float()
                bg_rel = (bg_rel >= 0.0).float()

            rank_pos = torch.sum(fg_rel)
            fp_num = torch.sum(bg_rel)
            rank = rank_pos + fp_num
            ranking_error[ii] = fp_num / rank.clamp_min(eps)

            current_sort = torch.sum(fg_rel * (1.0 - fg_targets)) / rank_pos.clamp_min(eps)
            iou_rel = fg_targets >= fg_targets[ii]
            target_sorted = iou_rel.float() * fg_rel
            rank_pos_target = torch.sum(target_sorted)
            target_sort = torch.sum(target_sorted * (1.0 - fg_targets)) / rank_pos_target.clamp_min(eps)
            sorting_error[ii] = current_sort - target_sort

            if fp_num > eps:
                fg_grad[ii] -= ranking_error[ii]
                bg_grad += bg_rel * (ranking_error[ii] / fp_num.clamp_min(eps))

            missorted = (~iou_rel).float() * fg_rel
            sorting_denom = torch.sum(missorted)
            if sorting_denom > eps:
                fg_grad[ii] -= sorting_error[ii]
                fg_grad += missorted * (sorting_error[ii] / sorting_denom.clamp_min(eps))

        grads[fg_labels] = fg_grad / float(fg_num)
        grads[bg_labels] = bg_grad / float(fg_num)
        ctx.save_for_backward(grads)
        return ranking_error.mean(), sorting_error.mean()

    @staticmethod
    def backward(ctx, grad_rank: torch.Tensor, grad_sort: torch.Tensor):
        (grads,) = ctx.saved_tensors
        # The saved identity-update gradient already includes both ranking and
        # sorting components, matching the official RankSortLoss usage.
        return grads * grad_rank, None, None, None


def proposal_ranksort_loss(
    logits: torch.Tensor,
    quality: torch.Tensor,
    frame_w: torch.Tensor,
    positive_threshold: float,
    delta_rs: float,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_quality = quality.detach().float().reshape(-1, quality.shape[-1]).clamp(0.0, 1.0)
    flat_weight = frame_w.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    pos_thr = float(positive_threshold)

    frame_losses = []
    for row_logits, row_quality in zip(flat_logits.float(), flat_quality):
        target = torch.where(row_quality >= pos_thr, row_quality, torch.zeros_like(row_quality))
        best = int(row_quality.argmax().item())
        target[best] = row_quality[best].clamp_min(max(pos_thr, 1e-4))
        rank_loss, sort_loss = ProposalRankSort.apply(row_logits, target, float(delta_rs))
        frame_losses.append(rank_loss + sort_loss)

    if not frame_losses:
        return logits.sum() * 0.0
    per_frame = torch.stack(frame_losses).to(dtype=logits.dtype).reshape_as(frame_w)
    return weighted_mean(per_frame, flat_weight.reshape_as(frame_w))


def neural_sort_top1_loss(
    logits: torch.Tensor,
    quality: torch.Tensor,
    frame_w: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_quality = quality.detach().float().reshape(-1, quality.shape[-1]).clamp(0.0, 1.0)
    flat_weight = frame_w.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    k = int(flat_logits.shape[-1])
    if k <= 1:
        return logits.sum() * 0.0

    scores = flat_logits.unsqueeze(-1)
    one = torch.ones(k, 1, device=flat_logits.device, dtype=flat_logits.dtype)
    pair_abs = torch.abs(scores - scores.transpose(1, 2))
    pair_sum = pair_abs.matmul(one.matmul(one.transpose(0, 1)))
    scaling = (k + 1 - 2 * (torch.arange(k, device=flat_logits.device, dtype=flat_logits.dtype) + 1))
    unary = scores.matmul(scaling.unsqueeze(0))
    soft_perm = F.softmax((unary - pair_sum).transpose(1, 2) / max(float(tau), 1e-4), dim=-1)

    oracle = flat_quality.argmax(dim=-1)
    top1_prob = soft_perm[:, 0, :].gather(-1, oracle.unsqueeze(-1)).squeeze(-1).clamp_min(1e-6)
    per_frame = -torch.log(top1_prob).to(dtype=logits.dtype).reshape_as(frame_w)
    return weighted_mean(per_frame, flat_weight.reshape_as(frame_w))


def quad_pairwise_shape_loss(pred_quad: torch.Tensor, gt_quad: torch.Tensor, frame_w: torch.Tensor) -> torch.Tensor:
    pred_d = torch.cdist(pred_quad.reshape(-1, 4, 2), pred_quad.reshape(-1, 4, 2), p=2).reshape(
        *pred_quad.shape[:2], 4, 4
    )
    gt_d = torch.cdist(gt_quad.reshape(-1, 4, 2), gt_quad.reshape(-1, 4, 2), p=2).reshape(
        *gt_quad.shape[:2], 4, 4
    )
    tri = torch.triu(torch.ones(4, 4, device=pred_quad.device, dtype=torch.bool), diagonal=1)
    per_frame = F.smooth_l1_loss(pred_d[..., tri], gt_d[..., tri], reduction="none").sum(dim=-1)
    return weighted_mean(per_frame, frame_w)


def quad_axes(quad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two sign-agnostic quad axes and their lengths.

    The input is an ordered quad [top-left, top-right, bottom-right,
    bottom-left] in normalized image coordinates.
    """
    p0, p1, p2, p3 = quad.unbind(dim=-2)
    axis_a = 0.5 * ((p1 - p0) + (p2 - p3))
    axis_b = 0.5 * ((p3 - p0) + (p2 - p1))
    len_a = torch.linalg.norm(axis_a, dim=-1).clamp_min(1e-4)
    len_b = torch.linalg.norm(axis_b, dim=-1).clamp_min(1e-4)
    dirs = torch.stack([axis_a / len_a[..., None], axis_b / len_b[..., None]], dim=-2)
    lens = torch.stack([len_a, len_b], dim=-1)
    return dirs, lens


def quad_axis_alignment_loss(
    pred_quad: torch.Tensor,
    gt_quad: torch.Tensor,
    frame_w: torch.Tensor,
    length_weight: float = 0.25,
) -> torch.Tensor:
    """Point-axis style orientation loss for ordered quad predictions.

    We avoid raw angle regression because rotated boxes have periodic boundary
    ambiguities.  Direction is sign-invariant and axis order can swap, so the
    loss uses the better of the no-swap and swapped assignments.
    """
    pred_dirs, pred_lens = quad_axes(pred_quad)
    gt_dirs, gt_lens = quad_axes(gt_quad)

    def dir_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dot = (a * b).sum(dim=-1).abs().clamp(0.0, 1.0)
        return 1.0 - dot.pow(2)

    no_swap_dir = dir_loss(pred_dirs[..., 0, :], gt_dirs[..., 0, :]) + dir_loss(
        pred_dirs[..., 1, :], gt_dirs[..., 1, :]
    )
    swap_dir = dir_loss(pred_dirs[..., 0, :], gt_dirs[..., 1, :]) + dir_loss(
        pred_dirs[..., 1, :], gt_dirs[..., 0, :]
    )
    len_w = float(length_weight)
    if len_w > 0.0:
        no_swap_len = F.smooth_l1_loss(
            torch.log(pred_lens), torch.log(gt_lens), reduction="none"
        ).sum(dim=-1)
        swap_len = F.smooth_l1_loss(
            torch.log(pred_lens), torch.log(gt_lens.flip(dims=(-1,))), reduction="none"
        ).sum(dim=-1)
    else:
        no_swap_len = torch.zeros_like(no_swap_dir)
        swap_len = torch.zeros_like(swap_dir)
    per_frame = torch.minimum(no_swap_dir + len_w * no_swap_len, swap_dir + len_w * swap_len)
    return weighted_mean(per_frame, frame_w)


def quad_topology_loss(
    pred_quad: torch.Tensor,
    gt_quad: torch.Tensor,
    frame_w: torch.Tensor,
    margin: float = 5e-4,
) -> torch.Tensor:
    """Penalize concave, self-crossing, or orientation-flipped ordered quads.

    The GT remains the native four-point polygon.  This loss only adds a
    Point2RBox/OrientedRepPoints-style layout consistency prior: consecutive
    edge turns should have the same signed orientation as the annotated quad.
    """

    def adjacent_cross(q: torch.Tensor) -> torch.Tensor:
        edge = torch.roll(q, shifts=-1, dims=-2) - q
        nxt = torch.roll(edge, shifts=-1, dims=-2)
        return edge[..., 0] * nxt[..., 1] - edge[..., 1] * nxt[..., 0]

    pred_cross = adjacent_cross(pred_quad.float())
    gt_cross = adjacent_cross(gt_quad.float()).detach()
    gt_sign = torch.sign(gt_cross.mean(dim=-1, keepdim=True))
    gt_sign = torch.where(gt_sign == 0.0, torch.ones_like(gt_sign), gt_sign)
    signed_turn = pred_cross * gt_sign
    per_frame = F.relu(float(margin) - signed_turn).mean(dim=-1)
    return weighted_mean(per_frame.to(dtype=pred_quad.dtype), frame_w)


def quad_to_xywhr(quad: torch.Tensor, min_size: float = 1e-3) -> torch.Tensor:
    """Convert the ordered free quad into an oriented-rectangle surrogate.

    This is an auxiliary representation only.  The prediction target remains
    the native 4-point quad, but KFIoU-style losses need center/size/angle.
    """
    center = quad.mean(dim=-2)
    dirs, lens = quad_axes(quad)
    axis = dirs[..., 0, :]
    angle = torch.atan2(axis[..., 1], axis[..., 0])
    return torch.cat([center, lens.clamp_min(float(min_size)), angle[..., None]], dim=-1)


def quad_polygon_area(quad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shoelace polygon area for an ordered free quad in normalized coordinates."""
    x = quad[..., :, 0]
    y = quad[..., :, 1]
    area = 0.5 * torch.abs(
        (x * torch.roll(y, shifts=-1, dims=-1)).sum(dim=-1)
        - (y * torch.roll(x, shifts=-1, dims=-1)).sum(dim=-1)
    )
    return area.clamp_min(float(eps))


def xywhr_to_xy_sigma(xywhr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MMRotate KFIoU conversion from rotated box to 2-D Gaussian."""
    shape = xywhr.shape
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-3, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4].reshape(-1)
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    rot = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    scale = 0.5 * torch.diag_embed(wh)
    sigma = rot.bmm(scale.square()).bmm(rot.transpose(1, 2)).reshape(shape[:-1] + (2, 2))
    return xy, sigma


def quad_kfiou_loss(
    pred_quad: torch.Tensor,
    gt_quad: torch.Tensor,
    frame_w: torch.Tensor,
    beta: float = 1.0 / 9.0,
    fun: str = "none",
    eps: float = 1e-6,
) -> torch.Tensor:
    """KFIoU-style auxiliary loss adapted from MMRotate.

    Ref: repro_workdirs/ref_code/oriented_roi/mmrotate/mmrotate/models/losses/kf_iou_loss.py
    """
    cm = torch.cuda.amp.autocast(enabled=False) if pred_quad.is_cuda else nullcontext()
    with cm:
        return _quad_kfiou_loss_impl(
            pred_quad=pred_quad,
            gt_quad=gt_quad,
            frame_w=frame_w,
            beta=beta,
            fun=fun,
            eps=eps,
        )


def _quad_kfiou_loss_impl(
    pred_quad: torch.Tensor,
    gt_quad: torch.Tensor,
    frame_w: torch.Tensor,
    beta: float,
    fun: str,
    eps: float,
) -> torch.Tensor:
    pred_quad = pred_quad.float()
    gt_quad = gt_quad.float()
    frame_w = frame_w.float()
    pred_box = quad_to_xywhr(pred_quad)
    gt_box = quad_to_xywhr(gt_quad)
    xy_p, sigma_p = xywhr_to_xy_sigma(pred_box)
    xy_t, sigma_t = xywhr_to_xy_sigma(gt_box)

    def det2(mat: torch.Tensor) -> torch.Tensor:
        return mat[..., 0, 0] * mat[..., 1, 1] - mat[..., 0, 1] * mat[..., 1, 0]

    def inv2(mat: torch.Tensor) -> torch.Tensor:
        det = det2(mat).clamp_min(eps)
        a = mat[..., 0, 0]
        b = mat[..., 0, 1]
        c = mat[..., 1, 0]
        d = mat[..., 1, 1]
        return torch.stack((d, -b, -c, a), dim=-1).reshape_as(mat) / det[..., None, None]

    diff = torch.abs(xy_p - xy_t)
    beta_t = max(float(beta), 1e-6)
    xy_loss = torch.where(diff < beta_t, 0.5 * diff * diff / beta_t, diff - 0.5 * beta_t).sum(dim=-1)

    sigma_p_f = sigma_p.reshape(-1, 2, 2)
    sigma_t_f = sigma_t.reshape(-1, 2, 2)
    eye = torch.eye(2, device=pred_quad.device, dtype=pred_quad.dtype).reshape(1, 2, 2)
    sigma_sum = sigma_p_f + sigma_t_f + eps * eye
    k_gain = sigma_p_f.bmm(inv2(sigma_sum))
    sigma = sigma_p_f - k_gain.bmm(sigma_p_f)

    vb_p = 4.0 * det2(sigma_p_f).clamp_min(0.0).sqrt()
    vb_t = 4.0 * det2(sigma_t_f).clamp_min(0.0).sqrt()
    vb = 4.0 * det2(sigma).clamp_min(0.0).sqrt()
    kfiou = (vb / (vb_p + vb_t - vb + eps)).clamp(0.0, 1.0).reshape(pred_quad.shape[:2])

    if fun == "ln":
        kf_loss = -torch.log(kfiou + eps)
    elif fun == "exp":
        kf_loss = torch.exp(1.0 - kfiou) - 1.0
    else:
        kf_loss = 1.0 - kfiou
    per_frame = torch.nan_to_num(xy_loss + kf_loss, nan=2.0, posinf=2.0, neginf=0.0).clamp(0.0, 2.0)
    return weighted_mean(per_frame, frame_w)


def quad_inside_grid(quad: torch.Tensor, grid_size: int = 64) -> torch.Tensor:
    q = quad.float().clamp(0.0, 1.0)
    g = max(int(grid_size), 8)
    yy, xx = torch.meshgrid(
        torch.linspace(0.5 / g, 1.0 - 0.5 / g, g, device=q.device, dtype=q.dtype),
        torch.linspace(0.5 / g, 1.0 - 0.5 / g, g, device=q.device, dtype=q.dtype),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).view(*([1] * (q.ndim - 2)), 1, g, g, 2)
    a = q
    b = torch.roll(q, shifts=-1, dims=-2)
    edge = b - a
    rel = grid - a[..., None, None, :]
    cross = edge[..., 0, None, None] * rel[..., 1] - edge[..., 1, None, None] * rel[..., 0]
    return (cross >= -1e-6).all(dim=-3) | (cross <= 1e-6).all(dim=-3)


def points_inside_quad(points: torch.Tensor, quad: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return whether arbitrary points fall inside a broadcastable ordered quad."""
    pts = points.float().clamp(0.0, 1.0)
    q = quad.float().clamp(0.0, 1.0)
    a = q
    b = torch.roll(q, shifts=-1, dims=-2)
    edge = b - a
    rel = pts.unsqueeze(-2) - a
    cross = edge[..., 0] * rel[..., 1] - edge[..., 1] * rel[..., 0]
    return (cross >= -float(eps)).all(dim=-1) | (cross <= float(eps)).all(dim=-1)


def points_signed_distance_to_quad(
    points: torch.Tensor,
    quad: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Signed Euclidean distance from sampled points to a broadcastable quad boundary.

    Negative values are inside the GT quad and positive values are outside,
    matching the signed distance-map convention used by BoundaryLoss.
    """
    pts = points.float().clamp(0.0, 1.0)
    q = quad.float().clamp(0.0, 1.0)
    a = q
    b = torch.roll(q, shifts=-1, dims=-2)
    edge = b - a
    rel = pts.unsqueeze(-2) - a
    edge_len2 = edge.square().sum(dim=-1).clamp_min(float(eps))
    proj = (rel * edge).sum(dim=-1) / edge_len2
    proj = proj.clamp(0.0, 1.0)
    closest = a + proj[..., None] * edge
    dist = torch.linalg.norm(pts.unsqueeze(-2) - closest, dim=-1).amin(dim=-1)
    cross = edge[..., 0] * rel[..., 1] - edge[..., 1] * rel[..., 0]
    inside = (cross >= -float(eps)).all(dim=-1) | (cross <= float(eps)).all(dim=-1)
    return torch.where(inside, -dist, dist)


def quad_soft_inside_grid(
    quad: torch.Tensor,
    grid_size: int = 64,
    edge_sharpness: float = 128.0,
) -> torch.Tensor:
    """Differentiable soft polygon rasterizer for an ordered free quad."""
    q = quad.float().clamp(0.0, 1.0)
    g = max(int(grid_size), 8)
    yy, xx = torch.meshgrid(
        torch.linspace(0.5 / g, 1.0 - 0.5 / g, g, device=q.device, dtype=q.dtype),
        torch.linspace(0.5 / g, 1.0 - 0.5 / g, g, device=q.device, dtype=q.dtype),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).view(*([1] * (q.ndim - 2)), 1, g, g, 2)
    a = q
    b = torch.roll(q, shifts=-1, dims=-2)
    edge = b - a
    rel = grid - a[..., None, None, :]
    cross = edge[..., 0, None, None] * rel[..., 1] - edge[..., 1, None, None] * rel[..., 0]
    nxt = torch.roll(edge, shifts=-1, dims=-2)
    turn = edge[..., 0] * nxt[..., 1] - edge[..., 1] * nxt[..., 0]
    orient = torch.sign(turn.mean(dim=-1, keepdim=True))
    orient = torch.where(orient == 0.0, torch.ones_like(orient), orient)
    edge_norm = torch.linalg.norm(edge, dim=-1).clamp_min(1e-4)
    signed_dist = cross * orient[..., None, None] / edge_norm[..., None, None]
    min_signed_dist = signed_dist.amin(dim=-3)
    return torch.sigmoid(float(edge_sharpness) * min_signed_dist)


def quad_soft_mask_iou_loss(
    pred_quad: torch.Tensor,
    gt_quad: torch.Tensor,
    frame_w: torch.Tensor,
    grid_size: int = 64,
    edge_sharpness: float = 128.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_mask = quad_soft_inside_grid(
        pred_quad,
        grid_size=grid_size,
        edge_sharpness=edge_sharpness,
    )
    with torch.no_grad():
        gt_mask = quad_inside_grid(gt_quad, grid_size=grid_size).to(dtype=pred_mask.dtype)
    inter = (pred_mask * gt_mask).sum(dim=(-1, -2))
    union = (pred_mask + gt_mask - pred_mask * gt_mask).sum(dim=(-1, -2)).clamp_min(1.0)
    soft_iou = (inter + 1.0) / (union + 1.0)
    soft_loss = 1.0 - soft_iou
    return weighted_mean(soft_loss, frame_w), weighted_mean(soft_iou, frame_w)


def quad_iou_approx(pred_quad: torch.Tensor, gt_quad: torch.Tensor, grid_size: int = 64) -> torch.Tensor:
    pred_m = quad_inside_grid(pred_quad, grid_size=grid_size)
    gt_m = quad_inside_grid(gt_quad, grid_size=grid_size)
    inter = (pred_m & gt_m).sum(dim=(-1, -2)).float()
    union = (pred_m | gt_m).sum(dim=(-1, -2)).float()
    return inter / union.clamp_min(1.0)


def nwd_similarity_cxcywh(
    pred_cxcywh: torch.Tensor,
    gt_cxcywh: torch.Tensor,
    c_px: float,
    image_size: float,
) -> torch.Tensor:
    pred = pred_cxcywh.float()
    gt = gt_cxcywh.float()
    d_center = (pred[..., :2] - gt[..., :2]).pow(2).sum(dim=-1)
    d_size = 0.25 * (pred[..., 2:] - gt[..., 2:]).pow(2).sum(dim=-1)
    w2 = (d_center + d_size).clamp_min(0.0)
    c_norm = max(float(c_px), 1e-6) / max(float(image_size), 1e-6)
    return torch.exp(-torch.sqrt(w2 + 1e-12) / c_norm).clamp(0.0, 1.0)


def propagate_cxcywh_from_active(boxes_cxcywh: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Fill inactive frames from active-frame predictions using action_range.

    This uses only predicted boxes and the action-range role label.  Inactive
    frames before/after the active span use the nearest active prediction;
    inactive frames between two active predictions use linear interpolation.
    """

    if boxes_cxcywh.ndim != 3 or boxes_cxcywh.shape[1] != 16:
        return boxes_cxcywh
    out = boxes_cxcywh.clone()
    active_bool = active > 0.5
    bsz, num_frames, _ = boxes_cxcywh.shape
    for b in range(bsz):
        idx = torch.nonzero(active_bool[b], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        for t in range(num_frames):
            if bool(active_bool[b, t]):
                continue
            prev_idx = idx[idx < t]
            next_idx = idx[idx > t]
            if prev_idx.numel() > 0 and next_idx.numel() > 0:
                p = int(prev_idx[-1].item())
                n = int(next_idx[0].item())
                alpha = float(t - p) / float(max(n - p, 1))
                out[b, t] = (1.0 - alpha) * boxes_cxcywh[b, p] + alpha * boxes_cxcywh[b, n]
            elif prev_idx.numel() > 0:
                out[b, t] = boxes_cxcywh[b, int(prev_idx[-1].item())]
            else:
                out[b, t] = boxes_cxcywh[b, int(next_idx[0].item())]
    return out


def propagate_quad_from_active(quad_xy: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Fill inactive frame quads from active-frame quads using action_range."""
    if quad_xy.ndim != 4 or quad_xy.shape[1:] != (16, 4, 2):
        return quad_xy
    out = quad_xy.clone()
    active_bool = active > 0.5
    bsz, num_frames = quad_xy.shape[:2]
    for b in range(bsz):
        idx = torch.nonzero(active_bool[b], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        for t in range(num_frames):
            if bool(active_bool[b, t]):
                continue
            prev_idx = idx[idx < t]
            next_idx = idx[idx > t]
            if prev_idx.numel() > 0 and next_idx.numel() > 0:
                p = int(prev_idx[-1].item())
                n = int(next_idx[0].item())
                alpha = float(t - p) / float(max(n - p, 1))
                out[b, t] = (1.0 - alpha) * quad_xy[b, p] + alpha * quad_xy[b, n]
            elif prev_idx.numel() > 0:
                out[b, t] = quad_xy[b, int(prev_idx[-1].item())]
            else:
                out[b, t] = quad_xy[b, int(next_idx[0].item())]
    return out.clamp(0.0, 1.0)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_pred_cxcywh = outputs["boxes_cxcywh"]
    active = batch["active_16"].to(raw_pred_cxcywh.device)
    if bool(args.active_prop_quads) and "quad_xy" in outputs:
        pred_quad_eval = propagate_quad_from_active(outputs["quad_xy"], active)
        pred_xyxy = quad_to_xyxy(pred_quad_eval, min_size=0.0)
        pred_cxcywh = xyxy_to_cxcywh_torch(pred_xyxy)
        outputs["quad_xy_eval"] = pred_quad_eval
        outputs["boxes_cxcywh_eval"] = pred_cxcywh
        outputs["boxes_xyxy_eval"] = pred_xyxy
    elif bool(args.active_prop_boxes):
        pred_cxcywh = propagate_cxcywh_from_active(raw_pred_cxcywh, active).clamp(0.0, 1.0)
        pred_xyxy = cxcywh_to_xyxy(pred_cxcywh)
        outputs["boxes_cxcywh_eval"] = pred_cxcywh
        outputs["boxes_xyxy_eval"] = pred_xyxy
    else:
        pred_cxcywh = raw_pred_cxcywh
        pred_xyxy = outputs["boxes_xyxy"]
    gt_xyxy = batch["boxes_xyxy"].to(pred_xyxy.device)
    gt_cxcywh = batch["boxes_cxcywh"].to(pred_xyxy.device)
    if bool(args.use_role_loss_weighting):
        frame_w = torch.where(active > 0.5, torch.full_like(active, args.active_loss_weight), torch.full_like(active, args.inactive_loss_weight))
    else:
        frame_w = torch.ones_like(active)
    small_box_factor = torch.ones_like(frame_w)
    if float(args.small_box_loss_weight) > 0.0:
        gt_area_for_weight = (gt_cxcywh[..., 2] * gt_cxcywh[..., 3]).clamp_min(1e-6)
        scale_boost = torch.sqrt(
            gt_area_for_weight.new_tensor(float(args.small_box_ref_area)) / gt_area_for_weight
        ).clamp(min=1.0, max=float(args.small_box_max_factor))
        small_box_factor = 1.0 + float(args.small_box_loss_weight) * (scale_boost - 1.0)
        frame_w = frame_w * small_box_factor
    hard_factor = torch.ones_like(frame_w)
    if float(args.hard_box_focal_weight) > 0.0:
        with torch.no_grad():
            pre_iou = box_iou_xyxy(pred_xyxy, gt_xyxy)
            threshold = max(float(args.hard_box_focal_threshold), 1e-3)
            hard = ((threshold - pre_iou).clamp_min(0.0) / threshold).pow(float(args.hard_box_focal_gamma))
            hard_factor = (1.0 + float(args.hard_box_focal_weight) * hard).clamp(
                max=float(args.hard_box_focal_max_factor)
            )
        frame_w = frame_w * hard_factor
    l1 = weighted_mean(F.smooth_l1_loss(pred_cxcywh, gt_cxcywh, reduction="none").sum(dim=-1), frame_w)
    giou = weighted_mean(giou_loss_xyxy(pred_xyxy, gt_xyxy), frame_w)
    ciou = pred_xyxy.new_tensor(0.0)
    if float(args.ciou_weight) > 0.0:
        ciou = weighted_mean(ciou_loss_xyxy(pred_xyxy, gt_xyxy), frame_w)
    center_idx = target_center_indices(gt_cxcywh).to(pred_xyxy.device)
    center_logits = outputs["center_logits_8"].flatten(2)
    ce = F.cross_entropy(center_logits.reshape(-1, center_logits.shape[-1]), center_idx.reshape(-1))
    smooth = F.smooth_l1_loss(pred_cxcywh[:, 1:], pred_cxcywh[:, :-1])
    loss = (
        args.l1_weight * l1
        + args.giou_weight * giou
        + float(args.ciou_weight) * ciou
        + args.center_ce_weight * ce
        + args.temporal_smooth_weight * smooth
    )
    box_mask_loss = pred_xyxy.new_tensor(0.0)
    if "region_logits_8" in outputs and float(args.box_mask_loss_weight) > 0.0:
        target_mask = target_box_masks_8(gt_cxcywh, grid_size=outputs["region_logits_8"].shape[-1]).to(pred_xyxy.device)
        bce = F.binary_cross_entropy_with_logits(outputs["region_logits_8"], target_mask, reduction="none")
        pixel_w = torch.where(target_mask > 0.5, torch.full_like(target_mask, float(args.box_mask_pos_weight)), torch.ones_like(target_mask))
        box_mask_loss = (bce * pixel_w).sum() / pixel_w.sum().clamp_min(1e-6)
        loss = loss + float(args.box_mask_loss_weight) * box_mask_loss
    edge_ce = pred_xyxy.new_tensor(0.0)
    if "edge_logits_8" in outputs and float(args.edge_ce_weight) > 0.0:
        edge_targets = target_edge_indices_8(gt_xyxy, grid_size=outputs["edge_logits_8"]["x0"].shape[-1])
        edge_losses = []
        for key, logits in outputs["edge_logits_8"].items():
            target = edge_targets[key].to(logits.device)
            edge_losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)))
        edge_ce = torch.stack(edge_losses).mean()
        loss = loss + float(args.edge_ce_weight) * edge_ce
    dfl = pred_xyxy.new_tensor(0.0)
    if "box_dist_logits_16" in outputs and float(args.dfl_weight) > 0.0:
        dfl_per_side = distribution_focal_loss(outputs["box_dist_logits_16"], gt_xyxy.to(pred_xyxy.device))
        dfl = weighted_mean(dfl_per_side.sum(dim=-1), frame_w)
        loss = loss + float(args.dfl_weight) * dfl
    corner_heatmap = pred_xyxy.new_tensor(0.0)
    if "corner_logits_8" in outputs and float(args.corner_heatmap_weight) > 0.0:
        target_corner = target_corner_heatmaps_8(
            batch["quad_xy"].to(pred_xyxy.device),
            grid_size=outputs["corner_logits_8"].shape[-1],
            sigma_px=float(args.corner_heatmap_sigma_px),
        )
        corner_heatmap = modified_focal_heatmap_loss(outputs["corner_logits_8"], target_corner)
        loss = loss + float(args.corner_heatmap_weight) * corner_heatmap
    support_heatmap = pred_xyxy.new_tensor(0.0)
    if "support_logits_8" in outputs and float(args.support_heatmap_weight) > 0.0:
        if str(getattr(args, "support_heatmap_target", "aabb")) == "quad" and "quad_xy" in batch:
            target_support = target_quad_support_heatmaps_8(
                batch["quad_xy"].to(pred_xyxy.device),
                grid_size=outputs["support_logits_8"].shape[-1],
                gamma=float(args.support_heatmap_gamma),
            )
        else:
            target_support = target_support_heatmaps_8(
                gt_cxcywh.to(pred_xyxy.device),
                grid_size=outputs["support_logits_8"].shape[-1],
                gamma=float(args.support_heatmap_gamma),
            )
        support_heatmap = soft_support_heatmap_loss(
            outputs["support_logits_8"],
            target_support,
            pos_weight=float(args.support_heatmap_pos_weight),
        )
        loss = loss + float(args.support_heatmap_weight) * support_heatmap
    area_loss = pred_xyxy.new_tensor(0.0)
    if float(args.area_weight) > 0.0:
        pred_area = (pred_cxcywh[..., 2] * pred_cxcywh[..., 3]).clamp_min(1e-5)
        gt_area = (gt_cxcywh[..., 2] * gt_cxcywh[..., 3]).clamp_min(1e-5)
        area_loss = weighted_mean(
            F.smooth_l1_loss(torch.log(pred_area), torch.log(gt_area), reduction="none"),
            frame_w,
        )
        loss = loss + float(args.area_weight) * area_loss
    size_loss = pred_xyxy.new_tensor(0.0)
    if float(args.size_weight) > 0.0:
        pred_wh = pred_cxcywh[..., 2:].clamp_min(1e-4)
        gt_wh = gt_cxcywh[..., 2:].clamp_min(1e-4)
        size_loss = weighted_mean(
            F.smooth_l1_loss(torch.log(pred_wh), torch.log(gt_wh), reduction="none").sum(dim=-1),
            frame_w,
        )
        loss = loss + float(args.size_weight) * size_loss
    aspect_loss = pred_xyxy.new_tensor(0.0)
    if float(args.aspect_weight) > 0.0:
        pred_wh = pred_cxcywh[..., 2:].clamp_min(1e-4)
        gt_wh = gt_cxcywh[..., 2:].clamp_min(1e-4)
        pred_aspect = torch.log(pred_wh[..., 0] / pred_wh[..., 1])
        gt_aspect = torch.log(gt_wh[..., 0] / gt_wh[..., 1])
        aspect_loss = weighted_mean(
            F.smooth_l1_loss(pred_aspect, gt_aspect, reduction="none"),
            frame_w,
        )
        loss = loss + float(args.aspect_weight) * aspect_loss
    nwd_loss = pred_xyxy.new_tensor(0.0)
    if float(args.nwd_weight) > 0.0:
        nwd = nwd_similarity_cxcywh(
            pred_cxcywh,
            gt_cxcywh,
            c_px=float(args.nwd_c_px),
            image_size=float(args.nwd_image_size),
        )
        nwd_loss = weighted_mean(1.0 - nwd, frame_w)
        loss = loss + float(args.nwd_weight) * nwd_loss
    clip_prior_l1 = pred_xyxy.new_tensor(0.0)
    clip_prior_giou = pred_xyxy.new_tensor(0.0)
    clip_prior_iou = pred_xyxy.new_tensor(0.0)
    if "clip_prior_cxcywh" in outputs and float(args.clip_prior_loss_weight) > 0.0:
        prior_cxcywh = outputs["clip_prior_cxcywh"]
        w = frame_w / frame_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        gt_clip_cxcywh = (gt_cxcywh * w[..., None]).sum(dim=1)
        gt_clip_xyxy = cxcywh_to_xyxy(gt_clip_cxcywh)
        prior_xyxy = outputs["clip_prior_xyxy"]
        clip_prior_l1 = F.smooth_l1_loss(prior_cxcywh, gt_clip_cxcywh, reduction="none").sum(dim=-1).mean()
        clip_prior_giou = giou_loss_xyxy(prior_xyxy, gt_clip_xyxy).mean()
        loss = loss + float(args.clip_prior_loss_weight) * (clip_prior_l1 + clip_prior_giou)
        with torch.no_grad():
            clip_prior_iou = box_iou_xyxy(prior_xyxy, gt_clip_xyxy).mean()
    anchor_l1 = pred_xyxy.new_tensor(0.0)
    anchor_giou = pred_xyxy.new_tensor(0.0)
    anchor_iou = pred_xyxy.new_tensor(0.0)
    if "anchor_cxcywh" in outputs and float(args.anchor_loss_weight) > 0.0:
        anchor_cxcywh = outputs["anchor_cxcywh"]
        anchor_xyxy = outputs["anchor_xyxy"]
        anchor_l1 = weighted_mean(
            F.smooth_l1_loss(anchor_cxcywh, gt_cxcywh, reduction="none").sum(dim=-1),
            frame_w,
        )
        anchor_giou = weighted_mean(giou_loss_xyxy(anchor_xyxy, gt_xyxy), frame_w)
        loss = loss + float(args.anchor_loss_weight) * (anchor_l1 + anchor_giou)
        with torch.no_grad():
            anchor_iou = box_iou_xyxy(anchor_xyxy, gt_xyxy).mean()
    coarse_l1 = pred_xyxy.new_tensor(0.0)
    coarse_giou = pred_xyxy.new_tensor(0.0)
    coarse_iou = pred_xyxy.new_tensor(0.0)
    if "coarse_cxcywh" in outputs and float(args.coarse_loss_weight) > 0.0:
        coarse_cxcywh = outputs["coarse_cxcywh"]
        coarse_xyxy = outputs["coarse_xyxy"]
        coarse_l1 = weighted_mean(
            F.smooth_l1_loss(coarse_cxcywh, gt_cxcywh, reduction="none").sum(dim=-1),
            frame_w,
        )
        coarse_giou = weighted_mean(giou_loss_xyxy(coarse_xyxy, gt_xyxy), frame_w)
        loss = loss + float(args.coarse_loss_weight) * (coarse_l1 + coarse_giou)
        with torch.no_grad():
            coarse_iou = box_iou_xyxy(coarse_xyxy, gt_xyxy).mean()
    decoder_aux_l1 = pred_xyxy.new_tensor(0.0)
    decoder_aux_giou = pred_xyxy.new_tensor(0.0)
    decoder_aux_iou = pred_xyxy.new_tensor(0.0)
    if "decoder_layer_cxcywh" in outputs and float(args.decoder_aux_loss_weight) > 0.0:
        layer_cxcywh = outputs["decoder_layer_cxcywh"]
        layer_xyxy = outputs["decoder_layer_xyxy"]
        # The final layer is already supervised by the main loss.  The aux loss
        # keeps earlier refinement steps honest without double-counting final.
        if layer_cxcywh.shape[0] > 1:
            aux_cxcywh = layer_cxcywh[:-1]
            aux_xyxy = layer_xyxy[:-1]
        else:
            aux_cxcywh = layer_cxcywh
            aux_xyxy = layer_xyxy
        cascade_pads = parse_float_list(getattr(args, "decoder_cascade_pad_ratios", ""))
        target_xyxy_layers = []
        target_cxcywh_layers = []
        for li in range(aux_xyxy.shape[0]):
            pad_ratio = cascade_pads[li] if li < len(cascade_pads) else 0.0
            layer_gt_xyxy = expand_xyxy_torch(gt_xyxy, pad_ratio=pad_ratio, min_size=0.0)
            target_xyxy_layers.append(layer_gt_xyxy)
            target_cxcywh_layers.append(xyxy_to_cxcywh_torch(layer_gt_xyxy))
        gt_x = torch.stack(target_xyxy_layers, dim=0)
        gt_cx = torch.stack(target_cxcywh_layers, dim=0)
        fw = frame_w.unsqueeze(0).expand(aux_cxcywh.shape[:-1])
        aux_l1_frame = F.smooth_l1_loss(aux_cxcywh, gt_cx, reduction="none").sum(dim=-1)
        aux_giou_frame = giou_loss_xyxy(aux_xyxy, gt_x)
        decoder_aux_l1 = weighted_mean(aux_l1_frame, fw)
        decoder_aux_giou = weighted_mean(aux_giou_frame, fw)
        loss = loss + float(args.decoder_aux_loss_weight) * (decoder_aux_l1 + decoder_aux_giou)
        with torch.no_grad():
            decoder_aux_iou = box_iou_xyxy(aux_xyxy, gt_x).mean()
    dn_refiner_loss = pred_xyxy.new_tensor(0.0)
    dn_refiner_l1 = pred_xyxy.new_tensor(0.0)
    dn_refiner_giou = pred_xyxy.new_tensor(0.0)
    dn_refiner_point = pred_xyxy.new_tensor(0.0)
    dn_refiner_axis = pred_xyxy.new_tensor(0.0)
    dn_refiner_iou = pred_xyxy.new_tensor(0.0)
    dn_refiner_quad_iou = pred_xyxy.new_tensor(0.0)
    if "dn_refiner_quad" in outputs and float(getattr(args, "dn_refiner_loss_weight", 0.0)) > 0.0:
        dn_quad = outputs["dn_refiner_quad"]
        dn_xyxy = outputs["dn_refiner_xyxy"]
        dn_cxcywh = outputs["dn_refiner_cxcywh"]
        gt_quad_dn = batch["quad_xy"].to(dn_quad.device)
        gt_cx_dn = gt_cxcywh.to(dn_cxcywh.device)
        gt_xy_dn = gt_xyxy.to(dn_xyxy.device)
        frame_w_dn = frame_w
        dn_group_count_raw = outputs.get("dn_refiner_group_count", 1)
        if torch.is_tensor(dn_group_count_raw):
            dn_group_count = int(dn_group_count_raw.detach().cpu().item())
        else:
            dn_group_count = int(dn_group_count_raw)
        if dn_group_count > 1 and dn_quad.shape[0] == gt_quad_dn.shape[0] * dn_group_count:
            b0 = gt_quad_dn.shape[0]
            gt_quad_dn = (
                gt_quad_dn[:, None]
                .expand(b0, dn_group_count, 16, 4, 2)
                .reshape(b0 * dn_group_count, 16, 4, 2)
                .contiguous()
            )
            gt_cx_dn = (
                gt_cx_dn[:, None]
                .expand(b0, dn_group_count, 16, 4)
                .reshape(b0 * dn_group_count, 16, 4)
                .contiguous()
            )
            gt_xy_dn = (
                gt_xy_dn[:, None]
                .expand(b0, dn_group_count, 16, 4)
                .reshape(b0 * dn_group_count, 16, 4)
                .contiguous()
            )
            frame_w_dn = (
                frame_w[:, None]
                .expand(b0, dn_group_count, 16)
                .reshape(b0 * dn_group_count, 16)
                .contiguous()
            )
        dn_l1_frame = F.smooth_l1_loss(dn_cxcywh, gt_cx_dn, reduction="none").sum(dim=-1)
        dn_giou_frame = giou_loss_xyxy(dn_xyxy, gt_xy_dn)
        dn_point_frame = F.smooth_l1_loss(dn_quad, gt_quad_dn, reduction="none").sum(dim=(-1, -2))
        dn_refiner_l1 = weighted_mean(dn_l1_frame, frame_w_dn)
        dn_refiner_giou = weighted_mean(dn_giou_frame, frame_w_dn)
        dn_refiner_point = weighted_mean(dn_point_frame, frame_w_dn)
        dn_refiner_axis = quad_axis_alignment_loss(
            dn_quad,
            gt_quad_dn,
            frame_w_dn,
            length_weight=float(args.quad_axis_length_weight),
        )
        dn_refiner_loss = dn_refiner_l1 + dn_refiner_giou + 0.50 * dn_refiner_point + 0.25 * dn_refiner_axis
        loss = loss + float(args.dn_refiner_loss_weight) * dn_refiner_loss
        with torch.no_grad():
            dn_refiner_iou = box_iou_xyxy(dn_xyxy, gt_xy_dn).mean()
            dn_grid = min(max(int(args.quad_iou_grid_size), 8), 32)
            dn_refiner_quad_iou = quad_iou_approx(dn_quad, gt_quad_dn, grid_size=dn_grid).mean()
    proposal_oracle_refiner_loss = pred_xyxy.new_tensor(0.0)
    proposal_oracle_refiner_l1 = pred_xyxy.new_tensor(0.0)
    proposal_oracle_refiner_giou = pred_xyxy.new_tensor(0.0)
    proposal_oracle_refiner_point = pred_xyxy.new_tensor(0.0)
    proposal_oracle_refiner_axis = pred_xyxy.new_tensor(0.0)
    proposal_oracle_refiner_iou = pred_xyxy.new_tensor(0.0)
    proposal_oracle_refiner_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_oracle_refiner_start_quad_iou = pred_xyxy.new_tensor(0.0)
    if (
        "proposal_oracle_refiner_quad" in outputs
        and float(getattr(args, "proposal_oracle_refiner_loss_weight", 0.0)) > 0.0
    ):
        oracle_quad = outputs["proposal_oracle_refiner_quad"]
        oracle_xyxy = outputs["proposal_oracle_refiner_xyxy"]
        oracle_cxcywh = outputs["proposal_oracle_refiner_cxcywh"]
        gt_quad_oracle = batch["quad_xy"].to(oracle_quad.device)
        gt_cx_oracle = gt_cxcywh.to(oracle_cxcywh.device)
        gt_xy_oracle = gt_xyxy.to(oracle_xyxy.device)
        oracle_l1_frame = F.smooth_l1_loss(oracle_cxcywh, gt_cx_oracle, reduction="none").sum(dim=-1)
        oracle_giou_frame = giou_loss_xyxy(oracle_xyxy, gt_xy_oracle)
        oracle_point_frame = F.smooth_l1_loss(oracle_quad, gt_quad_oracle, reduction="none").sum(dim=(-1, -2))
        proposal_oracle_refiner_l1 = weighted_mean(oracle_l1_frame, frame_w)
        proposal_oracle_refiner_giou = weighted_mean(oracle_giou_frame, frame_w)
        proposal_oracle_refiner_point = weighted_mean(oracle_point_frame, frame_w)
        proposal_oracle_refiner_axis = quad_axis_alignment_loss(
            oracle_quad,
            gt_quad_oracle,
            frame_w,
            length_weight=float(args.quad_axis_length_weight),
        )
        proposal_oracle_refiner_loss = (
            proposal_oracle_refiner_l1
            + proposal_oracle_refiner_giou
            + 0.50 * proposal_oracle_refiner_point
            + 0.25 * proposal_oracle_refiner_axis
        )
        loss = loss + float(args.proposal_oracle_refiner_loss_weight) * proposal_oracle_refiner_loss
        with torch.no_grad():
            proposal_oracle_refiner_iou = box_iou_xyxy(oracle_xyxy, gt_xy_oracle).mean()
            oracle_grid = min(max(int(args.quad_iou_grid_size), 8), 32)
            proposal_oracle_refiner_quad_iou = quad_iou_approx(
                oracle_quad,
                gt_quad_oracle,
                grid_size=oracle_grid,
            ).mean()
            if "proposal_oracle_refiner_start_quad" in outputs:
                proposal_oracle_refiner_start_quad_iou = quad_iou_approx(
                    outputs["proposal_oracle_refiner_start_quad"],
                    gt_quad_oracle,
                    grid_size=oracle_grid,
                ).mean()
    local_grid_corner_ce = pred_xyxy.new_tensor(0.0)
    if (
        "local_grid_corner_logits" in outputs
        and "local_grid_points" in outputs
        and float(getattr(args, "local_grid_corner_ce_weight", 0.0)) > 0.0
    ):
        corner_logits = outputs["local_grid_corner_logits"].float()
        local_grid_points = outputs["local_grid_points"].to(corner_logits.device).float()
        gt_grid_quad = batch["quad_xy"].to(corner_logits.device).float()
        dist = (
            local_grid_points[:, :, None, :, :]
            - gt_grid_quad[:, :, :, None, :]
        ).square().sum(dim=-1)
        corner_target = dist.argmin(dim=-1)
        b_ce, t_ce, c_ce, p_ce = corner_logits.shape
        ce_grid = F.cross_entropy(
            corner_logits.reshape(b_ce * t_ce * c_ce, p_ce),
            corner_target.reshape(b_ce * t_ce * c_ce),
            reduction="none",
        ).reshape(b_ce, t_ce, c_ce).mean(dim=-1)
        local_grid_corner_ce = weighted_mean(ce_grid, frame_w)
        loss = loss + float(args.local_grid_corner_ce_weight) * local_grid_corner_ce
    candidate_local_grid_corner_ce = pred_xyxy.new_tensor(0.0)
    if (
        "candidate_local_grid_corner_logits" in outputs
        and "candidate_local_grid_points" in outputs
        and "candidate_local_grid_base_quads" in outputs
        and float(getattr(args, "candidate_local_grid_corner_ce_weight", 0.0)) > 0.0
    ):
        cand_logits_all = outputs["candidate_local_grid_corner_logits"].float()
        cand_points_all = outputs["candidate_local_grid_points"].to(cand_logits_all.device).float()
        cand_base_quads = outputs["candidate_local_grid_base_quads"].to(cand_logits_all.device).float()
        gt_cand_quad = batch["quad_xy"].to(cand_logits_all.device).float()
        with torch.no_grad():
            gt_cand_expand = gt_cand_quad[:, :, None, :, :].expand_as(cand_base_quads)
            cand_grid = min(max(int(args.quad_iou_grid_size), 8), 32)
            cand_base_iou = quad_iou_approx(
                cand_base_quads.reshape(-1, 4, 2),
                gt_cand_expand.reshape(-1, 4, 2),
                grid_size=cand_grid,
            ).reshape(cand_base_quads.shape[:-2])
            cand_oracle_idx = cand_base_iou.argmax(dim=-1)
        b_cg, t_cg, _k_cg, c_cg, p_cg = cand_logits_all.shape
        gather_logits_idx = cand_oracle_idx[:, :, None, None, None].expand(-1, -1, 1, c_cg, p_cg)
        cand_logits = cand_logits_all.gather(2, gather_logits_idx).squeeze(2)
        gather_points_idx = cand_oracle_idx[:, :, None, None, None].expand(-1, -1, 1, p_cg, 2)
        cand_points = cand_points_all.gather(2, gather_points_idx).squeeze(2)
        cand_dist = (
            cand_points[:, :, None, :, :]
            - gt_cand_quad[:, :, :, None, :]
        ).square().sum(dim=-1)
        cand_target = cand_dist.argmin(dim=-1)
        cand_ce_grid = F.cross_entropy(
            cand_logits.reshape(b_cg * t_cg * c_cg, p_cg),
            cand_target.reshape(b_cg * t_cg * c_cg),
            reduction="none",
        ).reshape(b_cg, t_cg, c_cg).mean(dim=-1)
        candidate_local_grid_corner_ce = weighted_mean(cand_ce_grid, frame_w)
        loss = loss + float(args.candidate_local_grid_corner_ce_weight) * candidate_local_grid_corner_ce
    candidate_mask_bce = pred_xyxy.new_tensor(0.0)
    candidate_mask_quality = pred_xyxy.new_tensor(0.0)
    candidate_mask_top1_quad_iou = pred_xyxy.new_tensor(0.0)
    candidate_mask_iou_corr = pred_xyxy.new_tensor(0.0)
    proposal_handoff_mask_bce = pred_xyxy.new_tensor(0.0)
    proposal_handoff_mask_quality = pred_xyxy.new_tensor(0.0)
    proposal_handoff_mask_boundary = pred_xyxy.new_tensor(0.0)
    proposal_handoff_mask_top1_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_handoff_mask_iou_corr = pred_xyxy.new_tensor(0.0)
    if (
        "candidate_mask_logits" in outputs
        and "candidate_mask_points" in outputs
        and "candidate_mask_quality_logits" in outputs
        and "candidate_mask_base_quads" in outputs
    ):
        cand_mask_logits = outputs["candidate_mask_logits"].float()
        cand_mask_points = outputs["candidate_mask_points"].to(cand_mask_logits.device).float()
        cand_mask_quality_logits = outputs["candidate_mask_quality_logits"].float()
        cand_mask_quads = outputs["candidate_mask_base_quads"].to(cand_mask_logits.device).float()
        gt_mask_quad = batch["quad_xy"].to(cand_mask_logits.device).float()
        gt_for_points = gt_mask_quad[:, :, None, None, :, :]
        gt_for_quads = gt_mask_quad[:, :, None, :, :].expand_as(cand_mask_quads)
        with torch.no_grad():
            cand_mask_target = points_inside_quad(cand_mask_points, gt_for_points).to(cand_mask_logits.dtype)
            cand_mask_grid = min(max(int(args.quad_iou_grid_size), 8), 32)
            cand_mask_iou = quad_iou_approx(
                cand_mask_quads.reshape(-1, 4, 2),
                gt_for_quads.reshape(-1, 4, 2),
                grid_size=cand_mask_grid,
            ).reshape(cand_mask_quads.shape[:-2]).detach()
            pos_k = min(
                max(int(getattr(args, "candidate_mask_positive_topk", 3)), 1),
                int(cand_mask_iou.shape[-1]),
            )
            pos_idx = cand_mask_iou.topk(pos_k, dim=-1).indices
            cand_mask_pos = torch.zeros_like(cand_mask_iou, dtype=torch.bool)
            cand_mask_pos.scatter_(-1, pos_idx, True)
            mask_top1 = cand_mask_quality_logits.detach().argmax(dim=-1, keepdim=True)
            candidate_mask_top1_quad_iou = weighted_mean(
                cand_mask_iou.gather(-1, mask_top1).squeeze(-1),
                frame_w,
            )
            q_centered = cand_mask_iou.float() - cand_mask_iou.float().mean(dim=-1, keepdim=True)
            logit_centered = (
                cand_mask_quality_logits.detach().float()
                - cand_mask_quality_logits.detach().float().mean(dim=-1, keepdim=True)
            )
            corr_denom = (
                q_centered.square().sum(dim=-1).sqrt()
                * logit_centered.square().sum(dim=-1).sqrt()
            ).clamp_min(1e-6)
            candidate_mask_iou_corr = weighted_mean(
                (q_centered * logit_centered).sum(dim=-1) / corr_denom,
                frame_w,
            )
        if float(getattr(args, "candidate_mask_loss_weight", 0.0)) > 0.0:
            mask_bce = F.binary_cross_entropy_with_logits(
                cand_mask_logits,
                cand_mask_target,
                reduction="none",
            )
            point_w = 1.0 + float(getattr(args, "candidate_mask_pos_weight", 2.0)) * cand_mask_target
            mask_per_candidate = (mask_bce * point_w).sum(dim=-1) / point_w.sum(dim=-1).clamp_min(1.0)
            mask_frame = (mask_per_candidate * cand_mask_pos.to(mask_per_candidate.dtype)).sum(dim=-1)
            mask_frame = mask_frame / cand_mask_pos.sum(dim=-1).clamp_min(1).to(mask_frame.dtype)
            candidate_mask_bce = weighted_mean(mask_frame, frame_w)
            loss = loss + float(args.candidate_mask_loss_weight) * candidate_mask_bce
        if float(getattr(args, "candidate_mask_quality_weight", 0.0)) > 0.0:
            quality_target = cand_mask_iou.float().clamp(0.0, 1.0)
            quality_pred = torch.sigmoid(cand_mask_quality_logits).clamp(1e-4, 1.0 - 1e-4)
            quality_weight = quality_target + 0.75 * (quality_pred - quality_target).abs().pow(2.0)
            quality_bce = F.binary_cross_entropy_with_logits(
                cand_mask_quality_logits,
                quality_target,
                reduction="none",
            )
            candidate_mask_quality = weighted_mean((quality_bce * quality_weight).mean(dim=-1), frame_w)
            loss = loss + float(args.candidate_mask_quality_weight) * candidate_mask_quality
    if (
        "proposal_handoff_mask_logits" in outputs
        and "proposal_handoff_mask_points" in outputs
        and "proposal_handoff_mask_quality_logits" in outputs
        and "proposal_handoff_mask_base_quads" in outputs
        and "proposal_handoff_mask_source_mask" in outputs
    ):
        handoff_mask_logits = outputs["proposal_handoff_mask_logits"].float()
        handoff_mask_points = outputs["proposal_handoff_mask_points"].to(handoff_mask_logits.device).float()
        handoff_mask_quality_logits = outputs["proposal_handoff_mask_quality_logits"].float()
        handoff_mask_quads = outputs["proposal_handoff_mask_base_quads"].to(handoff_mask_logits.device).float()
        handoff_mask_source = outputs["proposal_handoff_mask_source_mask"].to(handoff_mask_logits.device).bool()
        gt_mask_quad = batch["quad_xy"].to(handoff_mask_logits.device).float()
        gt_for_points = gt_mask_quad[:, :, None, None, :, :]
        gt_for_quads = gt_mask_quad[:, :, None, :, :].expand_as(handoff_mask_quads)
        with torch.no_grad():
            handoff_mask_target = points_inside_quad(
                handoff_mask_points,
                gt_for_points,
            ).to(handoff_mask_logits.dtype)
            handoff_iou_grid = min(max(int(args.quad_iou_grid_size), 8), 32)
            handoff_mask_iou = quad_iou_approx(
                handoff_mask_quads.reshape(-1, 4, 2),
                gt_for_quads.reshape(-1, 4, 2),
                grid_size=handoff_iou_grid,
            ).reshape(handoff_mask_quads.shape[:-2]).detach()
            source_count = handoff_mask_source.float().sum(dim=-1, keepdim=True)
            has_source = source_count.squeeze(-1) > 0
            masked_logits = handoff_mask_quality_logits.detach().masked_fill(
                ~handoff_mask_source,
                -1.0e4,
            )
            handoff_top1 = masked_logits.argmax(dim=-1, keepdim=True)
            handoff_top1_iou = handoff_mask_iou.gather(-1, handoff_top1).squeeze(-1)
            proposal_handoff_mask_top1_quad_iou = weighted_mean(
                handoff_top1_iou,
                frame_w * has_source.to(device=frame_w.device, dtype=frame_w.dtype),
            )
            valid_count = source_count.clamp_min(1.0)
            valid_f = handoff_mask_source.to(handoff_mask_iou.dtype)
            q_mean = (handoff_mask_iou * valid_f).sum(dim=-1, keepdim=True) / valid_count
            logit_mean = (
                handoff_mask_quality_logits.detach() * valid_f
            ).sum(dim=-1, keepdim=True) / valid_count
            q_centered = (handoff_mask_iou - q_mean) * valid_f
            logit_centered = (handoff_mask_quality_logits.detach() - logit_mean) * valid_f
            corr_denom = (
                q_centered.square().sum(dim=-1).sqrt()
                * logit_centered.square().sum(dim=-1).sqrt()
            ).clamp_min(1e-6)
            corr = (q_centered * logit_centered).sum(dim=-1) / corr_denom
            corr = torch.where(has_source, corr, torch.zeros_like(corr))
            proposal_handoff_mask_iou_corr = weighted_mean(
                corr,
                frame_w * has_source.to(device=frame_w.device, dtype=frame_w.dtype),
            )
            pos_k = min(
                max(int(getattr(args, "proposal_handoff_mask_positive_topk", 6)), 1),
                int(handoff_mask_iou.shape[-1]),
            )
            masked_iou = handoff_mask_iou.masked_fill(~handoff_mask_source, -1.0)
            pos_idx = masked_iou.topk(pos_k, dim=-1).indices
            handoff_mask_pos = torch.zeros_like(handoff_mask_iou, dtype=torch.bool)
            pos_valid = handoff_mask_source.gather(-1, pos_idx) & (
                handoff_mask_iou.gather(-1, pos_idx) > 0.0
            )
            handoff_mask_pos.scatter_(-1, pos_idx, pos_valid)
        if float(getattr(args, "proposal_handoff_mask_loss_weight", 0.0)) > 0.0:
            handoff_bce = F.binary_cross_entropy_with_logits(
                handoff_mask_logits,
                handoff_mask_target,
                reduction="none",
            )
            point_w = 1.0 + float(getattr(args, "proposal_handoff_mask_pos_weight", 3.0)) * handoff_mask_target
            mask_per_candidate = (handoff_bce * point_w).sum(dim=-1) / point_w.sum(dim=-1).clamp_min(1.0)
            pos_f = handoff_mask_pos.to(mask_per_candidate.dtype)
            pos_den = pos_f.sum(dim=-1).clamp_min(1.0)
            mask_frame = (mask_per_candidate * pos_f).sum(dim=-1) / pos_den
            proposal_handoff_mask_bce = weighted_mean(
                mask_frame,
                frame_w * (pos_f.sum(dim=-1) > 0).to(device=frame_w.device, dtype=frame_w.dtype),
            )
            loss = loss + float(args.proposal_handoff_mask_loss_weight) * proposal_handoff_mask_bce
        if float(getattr(args, "proposal_handoff_mask_boundary_weight", 0.0)) > 0.0:
            signed_dist = points_signed_distance_to_quad(
                handoff_mask_points,
                gt_for_points,
            ).to(device=handoff_mask_logits.device, dtype=handoff_mask_logits.dtype)
            scale = max(float(getattr(args, "proposal_handoff_mask_boundary_scale_px", 8.0)), 1e-3) / 64.0
            signed_dist = (signed_dist / scale).clamp(-1.0, 1.0)
            prob = torch.sigmoid(handoff_mask_logits)
            boundary_per_candidate = (prob * signed_dist).mean(dim=-1)
            pos_f = handoff_mask_pos.to(boundary_per_candidate.dtype)
            pos_den = pos_f.sum(dim=-1).clamp_min(1.0)
            boundary_frame = (boundary_per_candidate * pos_f).sum(dim=-1) / pos_den
            proposal_handoff_mask_boundary = weighted_mean(
                boundary_frame,
                frame_w * (pos_f.sum(dim=-1) > 0).to(device=frame_w.device, dtype=frame_w.dtype),
            )
            loss = (
                loss
                + float(args.proposal_handoff_mask_boundary_weight)
                * proposal_handoff_mask_boundary
            )
        if float(getattr(args, "proposal_handoff_mask_quality_weight", 0.0)) > 0.0:
            quality_target = handoff_mask_iou.float().clamp(0.0, 1.0)
            quality_pred = torch.sigmoid(handoff_mask_quality_logits).clamp(1e-4, 1.0 - 1e-4)
            quality_l1 = F.smooth_l1_loss(quality_pred, quality_target, reduction="none")
            quality_weight = handoff_mask_source.to(quality_l1.dtype) * (0.25 + 2.0 * quality_target)
            quality_frame = (quality_l1 * quality_weight).sum(dim=-1)
            quality_frame = quality_frame / quality_weight.sum(dim=-1).clamp_min(1.0)
            proposal_handoff_mask_quality = weighted_mean(
                quality_frame,
                frame_w
                * (quality_weight.sum(dim=-1) > 0).to(device=frame_w.device, dtype=frame_w.dtype),
            )
            loss = loss + float(args.proposal_handoff_mask_quality_weight) * proposal_handoff_mask_quality
    proposal_loss = pred_xyxy.new_tensor(0.0)
    proposal_rank = pred_xyxy.new_tensor(0.0)
    proposal_oracle_iou = pred_xyxy.new_tensor(0.0)
    proposal_quad_loss = pred_xyxy.new_tensor(0.0)
    proposal_quad_oracle_iou = pred_xyxy.new_tensor(0.0)
    proposal_quality = pred_xyxy.new_tensor(0.0)
    proposal_varifocal = pred_xyxy.new_tensor(0.0)
    proposal_geom_quality = pred_xyxy.new_tensor(0.0)
    proposal_center_area = pred_xyxy.new_tensor(0.0)
    proposal_source_balanced_varifocal = pred_xyxy.new_tensor(0.0)
    proposal_source_balanced_task_aligned = pred_xyxy.new_tensor(0.0)
    proposal_source_group_ce = pred_xyxy.new_tensor(0.0)
    proposal_source_local_rank = pred_xyxy.new_tensor(0.0)
    proposal_handoff_source_router_ce = pred_xyxy.new_tensor(0.0)
    proposal_handoff_source_router_top1_match = pred_xyxy.new_tensor(0.0)
    proposal_handoff_source_router_candidate_quality = pred_xyxy.new_tensor(0.0)
    proposal_handoff_oracle_distill = pred_xyxy.new_tensor(0.0)
    proposal_handoff_oracle_distill_point = pred_xyxy.new_tensor(0.0)
    proposal_handoff_oracle_distill_axis = pred_xyxy.new_tensor(0.0)
    proposal_handoff_source5_readout_varifocal = pred_xyxy.new_tensor(0.0)
    proposal_handoff_source5_readout_soft_rank = pred_xyxy.new_tensor(0.0)
    proposal_handoff_source5_readout_top1_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_pcl_cluster_ce = pred_xyxy.new_tensor(0.0)
    proposal_oicr_refine_ce = pred_xyxy.new_tensor(0.0)
    proposal_source0_multimatch = pred_xyxy.new_tensor(0.0)
    proposal_source5_multimatch = pred_xyxy.new_tensor(0.0)
    proposal_source4_demote = pred_xyxy.new_tensor(0.0)
    proposal_source5_soft_owner = pred_xyxy.new_tensor(0.0)
    proposal_source5_mass_preserve = pred_xyxy.new_tensor(0.0)
    proposal_source7_multimatch = pred_xyxy.new_tensor(0.0)
    proposal_pairwise_rank = pred_xyxy.new_tensor(0.0)
    proposal_hardneg_rank = pred_xyxy.new_tensor(0.0)
    proposal_tool_hardneg = pred_xyxy.new_tensor(0.0)
    proposal_soft_rank = pred_xyxy.new_tensor(0.0)
    proposal_owner_band = pred_xyxy.new_tensor(0.0)
    proposal_neuralsort_top1 = pred_xyxy.new_tensor(0.0)
    proposal_ranksort = pred_xyxy.new_tensor(0.0)
    proposal_task_aligned = pred_xyxy.new_tensor(0.0)
    proposal_set_refiner_rank = pred_xyxy.new_tensor(0.0)
    proposal_set_refiner_soft_rank = pred_xyxy.new_tensor(0.0)
    proposal_set_refiner_top1_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_set_refiner_top3_oracle_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_set_refiner_top3_contains_oracle = pred_xyxy.new_tensor(0.0)
    proposal_handoff_source5_refiner_loss = pred_xyxy.new_tensor(0.0)
    proposal_handoff_source5_refiner_point = pred_xyxy.new_tensor(0.0)
    proposal_handoff_source5_refiner_axis = pred_xyxy.new_tensor(0.0)
    proposal_handoff_source5_refiner_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_handoff_source5_refiner_quality = pred_xyxy.new_tensor(0.0)
    proposal_handoff_source5_refiner_rank = pred_xyxy.new_tensor(0.0)
    proposal_handoff_source5_refiner_rank_top1_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_handoff_source5_refiner_rank_top1_match = pred_xyxy.new_tensor(0.0)
    proposal_handoff_topk_query_refiner_loss = pred_xyxy.new_tensor(0.0)
    proposal_handoff_topk_query_refiner_point = pred_xyxy.new_tensor(0.0)
    proposal_handoff_topk_query_refiner_axis = pred_xyxy.new_tensor(0.0)
    proposal_handoff_topk_query_refiner_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_handoff_topk_query_refiner_base_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_handoff_topk_query_refiner_quality = pred_xyxy.new_tensor(0.0)
    proposal_handoff_topk_query_refiner_rank = pred_xyxy.new_tensor(0.0)
    proposal_handoff_topk_query_refiner_soft_rank = pred_xyxy.new_tensor(0.0)
    proposal_handoff_topk_query_refiner_rank_top1_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_handoff_topk_query_refiner_rank_top1_match = pred_xyxy.new_tensor(0.0)
    proposal_c2f_assign = pred_xyxy.new_tensor(0.0)
    proposal_pcl_cluster = pred_xyxy.new_tensor(0.0)
    proposal_frame_anchor = pred_xyxy.new_tensor(0.0)
    proposal_frame_anchor_top1_oracle_iou = pred_xyxy.new_tensor(0.0)
    proposal_frame_anchor_entropy = pred_xyxy.new_tensor(0.0)
    proposal_logit_top1_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_logit_top3_oracle_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_logit_top3_contains_oracle = pred_xyxy.new_tensor(0.0)
    proposal_logit_iou_corr = pred_xyxy.new_tensor(0.0)
    proposal_source0_oracle_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_source0_logit_top1_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_source4_oracle_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_source4_logit_top1_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_source5_oracle_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_source5_logit_top1_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_source7_oracle_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_source7_logit_top1_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_top1_source4_frac = pred_xyxy.new_tensor(0.0)
    proposal_oracle_source4_frac = pred_xyxy.new_tensor(0.0)
    proposal_top1_source5_frac = pred_xyxy.new_tensor(0.0)
    proposal_oracle_source5_frac = pred_xyxy.new_tensor(0.0)
    proposal_top1_source7_frac = pred_xyxy.new_tensor(0.0)
    proposal_oracle_source7_frac = pred_xyxy.new_tensor(0.0)
    proposal_selection_source4_mass = pred_xyxy.new_tensor(0.0)
    proposal_selection_source5_mass = pred_xyxy.new_tensor(0.0)
    proposal_selection_source7_mass = pred_xyxy.new_tensor(0.0)
    proposal_top1_matches_oracle_source_frac = pred_xyxy.new_tensor(0.0)
    proposal_active_tip_varifocal = pred_xyxy.new_tensor(0.0)
    proposal_active_tip_top1_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_active_tip_top3_oracle_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_active_tip_top3_contains_oracle = pred_xyxy.new_tensor(0.0)
    proposal_active_tip_iou_corr = pred_xyxy.new_tensor(0.0)
    proposal_local_corr_top1_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_local_corr_top3_oracle_quad_iou = pred_xyxy.new_tensor(0.0)
    proposal_local_corr_iou_corr = pred_xyxy.new_tensor(0.0)
    stage1_full_proposal_oracle_iou = pred_xyxy.new_tensor(0.0)
    stage1_full_proposal_quad_oracle_iou = pred_xyxy.new_tensor(0.0)
    stage1_full_proposal_logit_top1_quad_iou = pred_xyxy.new_tensor(0.0)
    stage1_full_proposal_logit_top3_oracle_quad_iou = pred_xyxy.new_tensor(0.0)
    stage1_full_proposal_logit_top3_contains_oracle = pred_xyxy.new_tensor(0.0)
    stage1_full_proposal_logit_iou_corr = pred_xyxy.new_tensor(0.0)
    stage1_full_proposal_loss = pred_xyxy.new_tensor(0.0)
    stage1_full_proposal_quad_loss = pred_xyxy.new_tensor(0.0)
    stage1_full_proposal_quality = pred_xyxy.new_tensor(0.0)
    stage1_full_proposal_varifocal = pred_xyxy.new_tensor(0.0)
    if "stage1_full_proposal_quads" in outputs:
        with torch.no_grad():
            full_quads = outputs["stage1_full_proposal_quads"]
            gt_quad_full = batch["quad_xy"].to(full_quads.device)[:, :, None, :, :].expand_as(full_quads)
            full_grid = min(max(int(args.quad_iou_grid_size), 8), 32)
            full_quad_iou = quad_iou_approx(
                full_quads.reshape(-1, 4, 2),
                gt_quad_full.reshape(-1, 4, 2),
                grid_size=full_grid,
            ).reshape(full_quads.shape[:-2])
            stage1_full_proposal_quad_oracle_iou = weighted_mean(
                full_quad_iou.max(dim=-1).values,
                frame_w,
            )
            if "stage1_full_proposal_xyxy" in outputs:
                full_xyxy = outputs["stage1_full_proposal_xyxy"]
                gt_xy_full = gt_xyxy.to(full_xyxy.device)[:, :, None, :].expand_as(full_xyxy)
                full_box_iou = box_iou_xyxy(full_xyxy, gt_xy_full)
                stage1_full_proposal_oracle_iou = weighted_mean(
                    full_box_iou.max(dim=-1).values,
                    frame_w,
                )
            if "stage1_full_proposal_logits" in outputs:
                full_logits = outputs["stage1_full_proposal_logits"].detach().float()
                full_q = full_quad_iou.detach().float()
                full_top1 = full_logits.argmax(dim=-1, keepdim=True)
                stage1_full_proposal_logit_top1_quad_iou = weighted_mean(
                    full_q.gather(-1, full_top1).squeeze(-1),
                    frame_w,
                )
                full_topk_n = min(3, int(full_q.shape[-1]))
                full_topk = full_logits.topk(full_topk_n, dim=-1).indices
                full_topk_q = full_q.gather(-1, full_topk)
                stage1_full_proposal_logit_top3_oracle_quad_iou = weighted_mean(
                    full_topk_q.max(dim=-1).values,
                    frame_w,
                )
                full_oracle_idx = full_q.argmax(dim=-1, keepdim=True)
                full_contains = (full_topk == full_oracle_idx).any(dim=-1).float()
                stage1_full_proposal_logit_top3_contains_oracle = weighted_mean(
                    full_contains,
                    frame_w,
                )
                full_q_centered = full_q - full_q.mean(dim=-1, keepdim=True)
                full_logit_centered = full_logits - full_logits.mean(dim=-1, keepdim=True)
                full_denom = (
                    full_q_centered.square().sum(dim=-1).sqrt()
                    * full_logit_centered.square().sum(dim=-1).sqrt()
                ).clamp_min(1e-6)
                full_corr = (full_q_centered * full_logit_centered).sum(dim=-1) / full_denom
                stage1_full_proposal_logit_iou_corr = weighted_mean(full_corr, frame_w)
    if "proposal_cxcywh" in outputs and float(getattr(args, "proposal_loss_weight", 0.0)) > 0.0:
        proposal_cxcywh = outputs["proposal_cxcywh"]
        proposal_xyxy = outputs["proposal_xyxy"]
        gt_cx_p = gt_cxcywh[:, :, None, :].expand_as(proposal_cxcywh)
        gt_xy_p = gt_xyxy[:, :, None, :].expand_as(proposal_xyxy)
        prop_l1 = F.smooth_l1_loss(proposal_cxcywh, gt_cx_p, reduction="none").sum(dim=-1)
        prop_giou = giou_loss_xyxy(proposal_xyxy, gt_xy_p)
        prop_total = prop_l1 + prop_giou
        prop_best = prop_total.min(dim=-1).values
        proposal_loss = weighted_mean(prop_best, frame_w)
        loss = loss + float(args.proposal_loss_weight) * proposal_loss
        with torch.no_grad():
            prop_iou = box_iou_xyxy(proposal_xyxy, gt_xy_p)
            proposal_oracle_iou = prop_iou.max(dim=-1).values.mean()
        prop_quad_l1 = None
        prop_quad_iou = None
        if "proposal_quads" in outputs:
            proposal_quads = outputs["proposal_quads"]
            gt_quad_prop = batch["quad_xy"].to(proposal_quads.device)[:, :, None, :, :].expand_as(proposal_quads)
            prop_quad_l1 = F.smooth_l1_loss(proposal_quads, gt_quad_prop, reduction="none").sum(dim=(-1, -2))
            if float(getattr(args, "proposal_quad_loss_weight", 0.0)) > 0.0:
                proposal_quad_loss = weighted_mean(prop_quad_l1.min(dim=-1).values, frame_w)
                loss = loss + float(args.proposal_quad_loss_weight) * proposal_quad_loss
            with torch.no_grad():
                prop_quad_grid = min(max(int(args.quad_iou_grid_size), 8), 32)
                prop_quad_iou = quad_iou_approx(
                    proposal_quads.reshape(-1, 4, 2),
                    gt_quad_prop.reshape(-1, 4, 2),
                    grid_size=prop_quad_grid,
                ).reshape(proposal_quads.shape[:-2])
                proposal_quad_oracle_iou = prop_quad_iou.max(dim=-1).values.mean()
            if (
                "proposal_logits" in outputs
                and float(getattr(args, "proposal_geom_quality_weight", 0.0)) > 0.0
            ):
                logits = outputs["proposal_logits"].float()
                with torch.no_grad():
                    center_tau = max(float(getattr(args, "proposal_geom_center_tau", 2.0)), 1e-4)
                    area_tau = max(float(getattr(args, "proposal_geom_area_tau", 0.75)), 1e-4)
                    cand_center = proposal_quads.detach().float().mean(dim=-2)
                    gt_center = batch["quad_xy"].to(cand_center.device).detach().float().mean(dim=-2)
                    center_err = torch.linalg.norm(cand_center - gt_center[:, :, None, :], dim=-1)
                    center_err64 = center_err * float(args.quad_iou_grid_size)
                    pred_area = quad_polygon_area(proposal_quads.detach().float())
                    gt_area = quad_polygon_area(gt_quad_prop.detach().float())
                    log_area_abs = (torch.log(pred_area) - torch.log(gt_area)).abs()
                    geom = torch.exp(-center_err64 / center_tau) * torch.exp(-log_area_abs / area_tau)
                    target = (prop_quad_iou.detach().float().clamp(0.0, 1.0) * geom).clamp(0.0, 1.0)
                    oracle_idx = prop_quad_iou.detach().float().argmax(dim=-1, keepdim=True)
                    oracle_q = prop_quad_iou.detach().float().gather(-1, oracle_idx)
                    target.scatter_(-1, oracle_idx, torch.maximum(target.gather(-1, oracle_idx), oracle_q))
                pred = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
                focal_weight = target + float(args.proposal_varifocal_alpha) * (
                    pred - target
                ).abs().pow(float(args.proposal_varifocal_gamma))
                geom_vfl = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * focal_weight
                proposal_geom_quality = weighted_mean(geom_vfl.mean(dim=-1), frame_w)
                loss = loss + float(args.proposal_geom_quality_weight) * proposal_geom_quality
            if float(getattr(args, "proposal_center_area_loss_weight", 0.0)) > 0.0:
                q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
                topk = min(max(int(getattr(args, "proposal_center_area_topk", 2)), 1), int(q.shape[-1]))
                top_idx = q.topk(topk, dim=-1).indices
                top_mask = torch.zeros_like(q, dtype=torch.bool)
                top_mask.scatter_(-1, top_idx, True)
                cand_center = proposal_quads.float().mean(dim=-2)
                gt_center = batch["quad_xy"].to(cand_center.device).float().mean(dim=-2)
                center_delta_px = (cand_center - gt_center[:, :, None, :]) * float(args.quad_iou_grid_size)
                center_loss = F.smooth_l1_loss(
                    center_delta_px,
                    torch.zeros_like(center_delta_px),
                    reduction="none",
                ).sum(dim=-1)
                pred_area = quad_polygon_area(proposal_quads.float())
                gt_area = quad_polygon_area(gt_quad_prop.detach().float())
                log_area = torch.log(pred_area) - torch.log(gt_area)
                area_loss_prop = F.smooth_l1_loss(log_area, torch.zeros_like(log_area), reduction="none")
                over_loss_prop = F.smooth_l1_loss(
                    log_area.clamp_min(0.0),
                    torch.zeros_like(log_area),
                    reduction="none",
                )
                geom_frame = (
                    0.25 * center_loss
                    + area_loss_prop
                    + float(getattr(args, "proposal_center_area_over_weight", 0.5)) * over_loss_prop
                )
                geom_frame = (geom_frame * top_mask.float()).sum(dim=-1) / top_mask.float().sum(dim=-1).clamp_min(1.0)
                proposal_center_area = weighted_mean(geom_frame, frame_w)
                loss = loss + float(args.proposal_center_area_loss_weight) * proposal_center_area
            if (
                "quad_xy" in outputs
                and float(getattr(args, "proposal_handoff_oracle_distill_weight", 0.0)) > 0.0
            ):
                with torch.no_grad():
                    oracle_idx = prop_quad_iou.detach().float().argmax(dim=-1)
                    gather_idx = oracle_idx[:, :, None, None, None].expand(-1, -1, 1, 4, 2)
                    oracle_quad = proposal_quads.detach().gather(2, gather_idx).squeeze(2).clamp(0.0, 1.0)
                handoff_quad = outputs["quad_xy"].float().clamp(0.0, 1.0)
                proposal_handoff_oracle_distill_point = weighted_mean(
                    F.smooth_l1_loss(handoff_quad, oracle_quad, reduction="none").sum(dim=(-1, -2)),
                    frame_w,
                )
                proposal_handoff_oracle_distill_axis = quad_axis_alignment_loss(
                    handoff_quad,
                    oracle_quad,
                    frame_w,
                    length_weight=float(args.quad_axis_length_weight),
                )
                proposal_handoff_oracle_distill = (
                    proposal_handoff_oracle_distill_point
                    + float(getattr(args, "proposal_handoff_oracle_distill_axis_weight", 0.25))
                    * proposal_handoff_oracle_distill_axis
                )
                loss = (
                    loss
                    + float(args.proposal_handoff_oracle_distill_weight)
                    * proposal_handoff_oracle_distill
                )
            if (
                "proposal_handoff_source5_readout_logits" in outputs
                and "proposal_source_ids" in outputs
                and prop_quad_iou is not None
                and (
                    float(getattr(args, "proposal_handoff_source5_readout_varifocal_weight", 0.0)) > 0.0
                    or float(getattr(args, "proposal_handoff_source5_readout_soft_rank_weight", 0.0)) > 0.0
                )
            ):
                readout_logits = outputs["proposal_handoff_source5_readout_logits"].float()
                source_ids = outputs["proposal_source_ids"].to(device=readout_logits.device, dtype=torch.long)
                q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
                if readout_logits.shape == q.shape and source_ids.shape == q.shape:
                    readout_sources = parse_int_csv(
                        str(getattr(args, "proposal_handoff_source5_readout_sources", "5"))
                    )
                    if not readout_sources:
                        readout_sources = [5]
                    source_mask = torch.zeros_like(source_ids, dtype=torch.bool)
                    for source_id in readout_sources:
                        source_mask = source_mask | (source_ids == int(source_id))
                    source_count = source_mask.float().sum(dim=-1)
                    valid_frame = source_count > 0
                    neg = torch.finfo(readout_logits.dtype).min
                    readout_top1 = readout_logits.masked_fill(~source_mask, neg).argmax(
                        dim=-1,
                        keepdim=True,
                    )
                    readout_top1_q = q.gather(-1, readout_top1).squeeze(-1)
                    proposal_handoff_source5_readout_top1_quad_iou = weighted_mean(
                        torch.where(valid_frame, readout_top1_q, torch.zeros_like(readout_top1_q)),
                        frame_w,
                    )
                    if (
                        float(getattr(args, "proposal_handoff_source5_readout_varifocal_weight", 0.0))
                        > 0.0
                    ):
                        topk = min(
                            max(int(getattr(args, "proposal_handoff_source5_readout_loss_topk", 3)), 1),
                            int(q.shape[-1]),
                        )
                        source_q = q.masked_fill(~source_mask, -1.0)
                        top_idx = source_q.topk(topk, dim=-1).indices
                        top_q = q.gather(-1, top_idx)
                        top_valid = source_mask.gather(-1, top_idx) & (top_q > 0.0)
                        target = torch.zeros_like(q)
                        target.scatter_(-1, top_idx, torch.where(top_valid, top_q, torch.zeros_like(top_q)))
                        pred = readout_logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
                        focal_weight = target + float(args.proposal_varifocal_alpha) * (
                            pred - target
                        ).abs().pow(float(args.proposal_varifocal_gamma))
                        vfl = (
                            F.binary_cross_entropy_with_logits(readout_logits, target, reduction="none")
                            * focal_weight
                            * source_mask.float()
                        )
                        frame_loss = vfl.sum(dim=-1) / source_count.clamp_min(1.0)
                        frame_loss = torch.where(valid_frame, frame_loss, torch.zeros_like(frame_loss))
                        proposal_handoff_source5_readout_varifocal = weighted_mean(frame_loss, frame_w)
                        loss = loss + float(
                            args.proposal_handoff_source5_readout_varifocal_weight
                        ) * proposal_handoff_source5_readout_varifocal
                    if (
                        float(getattr(args, "proposal_handoff_source5_readout_soft_rank_weight", 0.0))
                        > 0.0
                    ):
                        tau = max(
                            float(getattr(args, "proposal_handoff_source5_readout_soft_rank_tau", 0.08)),
                            1e-4,
                        )
                        masked_logits = readout_logits.masked_fill(~source_mask, -50.0)
                        target_dist = F.softmax(q.masked_fill(~source_mask, -50.0) / tau, dim=-1)
                        log_prob = F.log_softmax(masked_logits, dim=-1)
                        sr = F.kl_div(log_prob, target_dist, reduction="none").sum(dim=-1)
                        sr = torch.where(valid_frame, sr, torch.zeros_like(sr))
                        proposal_handoff_source5_readout_soft_rank = weighted_mean(sr, frame_w)
                        loss = loss + float(
                            args.proposal_handoff_source5_readout_soft_rank_weight
                        ) * proposal_handoff_source5_readout_soft_rank

        def proposal_quality_target_from_iou(q: torch.Tensor) -> torch.Tensor:
            q_target = q.detach().float().clamp(0.0, 1.0)
            mode = str(getattr(args, "proposal_quality_target_mode", "all"))
            thr = float(getattr(args, "proposal_quality_positive_threshold", 0.0))
            if mode == "all":
                return q_target
            if mode.startswith("adaptive_meanstd"):
                mean = q_target.mean(dim=-1, keepdim=True)
                std = q_target.std(dim=-1, keepdim=True, unbiased=False)
                scale = float(getattr(args, "proposal_quality_adaptive_std_scale", 1.0))
                adaptive_thr = mean + scale * std
                pos = q_target >= adaptive_thr
                if mode.endswith("_or_oracle"):
                    oracle_idx = q_target.argmax(dim=-1, keepdim=True)
                    pos.scatter_(-1, oracle_idx, True)
                return torch.where(pos, q_target, torch.zeros_like(q_target))
            if mode.startswith("adaptive_topk"):
                k = int(getattr(args, "proposal_quality_adaptive_topk", 2))
                k = max(1, min(k, q_target.shape[-1]))
                top_idx = q_target.topk(k, dim=-1).indices
                pos = torch.zeros_like(q_target, dtype=torch.bool)
                pos.scatter_(-1, top_idx, True)
                if mode.endswith("_or_oracle"):
                    oracle_idx = q_target.argmax(dim=-1, keepdim=True)
                    pos.scatter_(-1, oracle_idx, True)
                return torch.where(pos, q_target, torch.zeros_like(q_target))
            if thr <= 0.0:
                return q_target
            if mode == "oracle":
                pos = torch.zeros_like(q_target, dtype=torch.bool)
            else:
                pos = q_target >= thr
            if mode in {"threshold_or_oracle", "oracle"}:
                oracle_idx = q_target.argmax(dim=-1, keepdim=True)
                pos.scatter_(-1, oracle_idx, True)
            return torch.where(pos, q_target, torch.zeros_like(q_target))

        if "stage1_full_proposal_quads" in outputs:
            # H304-style safeguard: when refine-all/top-k replaces
            # proposal_quads with refined candidates, keep the original
            # Stage1 K-candidate bank directly supervised.  Otherwise the
            # refiner can learn while proposal recall quietly collapses.
            full_quads_train = outputs["stage1_full_proposal_quads"]
            gt_quad_full_train = batch["quad_xy"].to(full_quads_train.device)[:, :, None, :, :].expand_as(
                full_quads_train
            )
            if (
                "stage1_full_proposal_xyxy" in outputs
                and float(getattr(args, "stage1_full_proposal_loss_weight", 0.0)) > 0.0
            ):
                full_xyxy_train = outputs["stage1_full_proposal_xyxy"]
                full_cxcywh_train = xyxy_to_cxcywh_torch(full_xyxy_train)
                gt_cx_full_train = gt_cxcywh.to(full_cxcywh_train.device)[:, :, None, :].expand_as(
                    full_cxcywh_train
                )
                gt_xy_full_train = gt_xyxy.to(full_xyxy_train.device)[:, :, None, :].expand_as(full_xyxy_train)
                full_l1_train = F.smooth_l1_loss(
                    full_cxcywh_train,
                    gt_cx_full_train,
                    reduction="none",
                ).sum(dim=-1)
                full_giou_train = giou_loss_xyxy(full_xyxy_train, gt_xy_full_train)
                stage1_full_proposal_loss = weighted_mean(
                    (full_l1_train + full_giou_train).min(dim=-1).values,
                    frame_w,
                )
                loss = loss + float(args.stage1_full_proposal_loss_weight) * stage1_full_proposal_loss

            if float(getattr(args, "stage1_full_proposal_quad_loss_weight", 0.0)) > 0.0:
                full_quad_l1_train = F.smooth_l1_loss(
                    full_quads_train,
                    gt_quad_full_train,
                    reduction="none",
                ).sum(dim=(-1, -2))
                stage1_full_proposal_quad_loss = weighted_mean(
                    full_quad_l1_train.min(dim=-1).values,
                    frame_w,
                )
                loss = loss + float(args.stage1_full_proposal_quad_loss_weight) * stage1_full_proposal_quad_loss

            if (
                "stage1_full_proposal_logits" in outputs
                and (
                    float(getattr(args, "stage1_full_proposal_quality_weight", 0.0)) > 0.0
                    or float(getattr(args, "stage1_full_proposal_varifocal_weight", 0.0)) > 0.0
                )
            ):
                full_logits_train = outputs["stage1_full_proposal_logits"].float()
                with torch.no_grad():
                    full_quality_grid = min(max(int(args.quad_iou_grid_size), 8), 32)
                    full_quality_iou = quad_iou_approx(
                        full_quads_train.reshape(-1, 4, 2),
                        gt_quad_full_train.reshape(-1, 4, 2),
                        grid_size=full_quality_grid,
                    ).reshape(full_quads_train.shape[:-2])
                    full_quality_target = proposal_quality_target_from_iou(full_quality_iou)
                if float(getattr(args, "stage1_full_proposal_quality_weight", 0.0)) > 0.0:
                    full_quality_pred = torch.sigmoid(full_logits_train)
                    full_quality_loss = F.smooth_l1_loss(
                        full_quality_pred,
                        full_quality_target.float(),
                        reduction="none",
                    )
                    stage1_full_proposal_quality = weighted_mean(full_quality_loss.mean(dim=-1), frame_w)
                    loss = loss + float(args.stage1_full_proposal_quality_weight) * stage1_full_proposal_quality
                if float(getattr(args, "stage1_full_proposal_varifocal_weight", 0.0)) > 0.0:
                    full_quality_pred = torch.sigmoid(full_logits_train).clamp(1e-4, 1.0 - 1e-4)
                    full_focal_weight = full_quality_target + float(args.proposal_varifocal_alpha) * (
                        full_quality_pred - full_quality_target
                    ).abs().pow(float(args.proposal_varifocal_gamma))
                    full_vfl = (
                        F.binary_cross_entropy_with_logits(
                            full_logits_train,
                            full_quality_target,
                            reduction="none",
                        )
                        * full_focal_weight
                    )
                    stage1_full_proposal_varifocal = weighted_mean(full_vfl.mean(dim=-1), frame_w)
                    loss = loss + float(args.stage1_full_proposal_varifocal_weight) * stage1_full_proposal_varifocal

        if "proposal_logits" in outputs and float(getattr(args, "proposal_rank_weight", 0.0)) > 0.0:
            with torch.no_grad():
                if prop_quad_iou is not None and float(getattr(args, "proposal_quad_loss_weight", 0.0)) > 0.0:
                    best_idx = prop_quad_iou.argmax(dim=-1)
                else:
                    best_idx = prop_total.argmin(dim=-1)
            rank_ce = F.cross_entropy(
                outputs["proposal_logits"].reshape(-1, outputs["proposal_logits"].shape[-1]),
                best_idx.reshape(-1),
                reduction="none",
            ).reshape_as(best_idx)
            proposal_rank = weighted_mean(rank_ce, frame_w)
            loss = loss + float(args.proposal_rank_weight) * proposal_rank
        if "proposal_logits" in outputs and prop_quad_iou is not None:
            with torch.no_grad():
                logits_eval = outputs["proposal_logits"].detach().float()
                q_eval = prop_quad_iou.detach().float()
                k_eval = q_eval.shape[-1]
                logit_top1 = logits_eval.argmax(dim=-1, keepdim=True)
                proposal_logit_top1_quad_iou = weighted_mean(
                    q_eval.gather(-1, logit_top1).squeeze(-1),
                    frame_w,
                )
                topk_eval = min(3, int(k_eval))
                logit_topk = logits_eval.topk(topk_eval, dim=-1).indices
                topk_q = q_eval.gather(-1, logit_topk)
                proposal_logit_top3_oracle_quad_iou = weighted_mean(topk_q.max(dim=-1).values, frame_w)
                oracle_idx = q_eval.argmax(dim=-1, keepdim=True)
                contains = (logit_topk == oracle_idx).any(dim=-1).float()
                proposal_logit_top3_contains_oracle = weighted_mean(contains, frame_w)
                q_centered = q_eval - q_eval.mean(dim=-1, keepdim=True)
                logit_centered = logits_eval - logits_eval.mean(dim=-1, keepdim=True)
                denom = (
                    q_centered.square().sum(dim=-1).sqrt()
                    * logit_centered.square().sum(dim=-1).sqrt()
                ).clamp_min(1e-6)
                corr = (q_centered * logit_centered).sum(dim=-1) / denom
                proposal_logit_iou_corr = weighted_mean(corr, frame_w)
                if "proposal_source_ids" in outputs:
                    source_ids = outputs["proposal_source_ids"].to(device=q_eval.device, dtype=torch.long)
                    if source_ids.shape == q_eval.shape:
                        top1_source = source_ids.gather(-1, logit_top1).squeeze(-1)
                        oracle_source = source_ids.gather(-1, oracle_idx).squeeze(-1)
                        proposal_top1_source4_frac = weighted_mean((top1_source == 4).float(), frame_w)
                        proposal_oracle_source4_frac = weighted_mean((oracle_source == 4).float(), frame_w)
                        proposal_top1_source5_frac = weighted_mean((top1_source == 5).float(), frame_w)
                        proposal_oracle_source5_frac = weighted_mean((oracle_source == 5).float(), frame_w)
                        proposal_top1_source7_frac = weighted_mean((top1_source == 7).float(), frame_w)
                        proposal_oracle_source7_frac = weighted_mean((oracle_source == 7).float(), frame_w)
                        proposal_top1_matches_oracle_source_frac = weighted_mean(
                            (top1_source == oracle_source).float(),
                            frame_w,
                        )
                        if "proposal_selection_weights" in outputs:
                            sel_w = outputs["proposal_selection_weights"].detach().float().to(q_eval.device)
                            if sel_w.shape == q_eval.shape:
                                proposal_selection_source4_mass = weighted_mean(
                                    (sel_w * (source_ids == 4).float()).sum(dim=-1),
                                    frame_w,
                                )
                                proposal_selection_source5_mass = weighted_mean(
                                    (sel_w * (source_ids == 5).float()).sum(dim=-1),
                                    frame_w,
                                )
                                proposal_selection_source7_mass = weighted_mean(
                                    (sel_w * (source_ids == 7).float()).sum(dim=-1),
                                    frame_w,
                                )

                        def source_metrics(source_id: int) -> tuple[torch.Tensor, torch.Tensor]:
                            source_mask = source_ids == int(source_id)
                            source_frame_w = frame_w * source_mask.any(dim=-1).float()
                            q_masked = q_eval.masked_fill(~source_mask, -1.0)
                            source_oracle = q_masked.max(dim=-1).values.clamp_min(0.0)
                            logits_masked = logits_eval.masked_fill(
                                ~source_mask,
                                torch.finfo(logits_eval.dtype).min,
                            )
                            source_top1 = logits_masked.argmax(dim=-1, keepdim=True)
                            source_top1_iou = q_eval.gather(-1, source_top1).squeeze(-1)
                            return (
                                weighted_mean(source_oracle, source_frame_w),
                                weighted_mean(source_top1_iou, source_frame_w),
                            )

                        (
                            proposal_source0_oracle_quad_iou,
                            proposal_source0_logit_top1_quad_iou,
                        ) = source_metrics(0)
                        (
                            proposal_source4_oracle_quad_iou,
                            proposal_source4_logit_top1_quad_iou,
                        ) = source_metrics(4)
                        (
                            proposal_source5_oracle_quad_iou,
                            proposal_source5_logit_top1_quad_iou,
                        ) = source_metrics(5)
                        (
                            proposal_source7_oracle_quad_iou,
                            proposal_source7_logit_top1_quad_iou,
                        ) = source_metrics(7)
        if "proposal_set_refiner_scores" in outputs and prop_quad_iou is not None:
            set_scores = outputs["proposal_set_refiner_scores"].float()
            if set_scores.shape == prop_quad_iou.shape:
                with torch.no_grad():
                    q_eval = prop_quad_iou.detach().float()
                    set_top1 = set_scores.detach().argmax(dim=-1, keepdim=True)
                    proposal_set_refiner_top1_quad_iou = weighted_mean(
                        q_eval.gather(-1, set_top1).squeeze(-1),
                        frame_w,
                    )
                    set_topk_n = min(3, int(q_eval.shape[-1]))
                    set_topk = set_scores.detach().topk(set_topk_n, dim=-1).indices
                    set_topk_q = q_eval.gather(-1, set_topk)
                    proposal_set_refiner_top3_oracle_quad_iou = weighted_mean(
                        set_topk_q.max(dim=-1).values,
                        frame_w,
                    )
                    set_oracle_idx = q_eval.argmax(dim=-1, keepdim=True)
                    set_contains = (set_topk == set_oracle_idx).any(dim=-1).float()
                    proposal_set_refiner_top3_contains_oracle = weighted_mean(set_contains, frame_w)
                if float(getattr(args, "proposal_set_refiner_rank_weight", 0.0)) > 0.0:
                    with torch.no_grad():
                        best_idx = prop_quad_iou.detach().float().argmax(dim=-1)
                    set_rank_ce = F.cross_entropy(
                        set_scores.reshape(-1, set_scores.shape[-1]),
                        best_idx.reshape(-1),
                        reduction="none",
                    ).reshape_as(best_idx)
                    proposal_set_refiner_rank = weighted_mean(set_rank_ce, frame_w)
                    loss = loss + float(args.proposal_set_refiner_rank_weight) * proposal_set_refiner_rank
                if float(getattr(args, "proposal_set_refiner_soft_rank_weight", 0.0)) > 0.0:
                    with torch.no_grad():
                        tau = max(float(getattr(args, "proposal_set_refiner_soft_rank_tau", 0.07)), 1e-4)
                        target_dist = F.softmax(prop_quad_iou.detach().float().clamp(0.0, 1.0) / tau, dim=-1)
                    set_log_prob = F.log_softmax(set_scores, dim=-1)
                    set_kl = F.kl_div(set_log_prob, target_dist, reduction="none").sum(dim=-1)
                    proposal_set_refiner_soft_rank = weighted_mean(set_kl, frame_w)
                    loss = (
                        loss
                        + float(args.proposal_set_refiner_soft_rank_weight) * proposal_set_refiner_soft_rank
                    )
        if "proposal_handoff_source5_refiner_quad" in outputs:
            source5_ref_quad = outputs["proposal_handoff_source5_refiner_quad"].float().clamp(0.0, 1.0)
            gt_source5_ref = batch["quad_xy"].to(source5_ref_quad.device).float()
            source5_ref_w = frame_w.to(source5_ref_quad.device)
            source5_refiner_source_id = int(
                max(0, min(7, int(getattr(args, "proposal_handoff_source5_refiner_source_id", 5))))
            )
            if "proposal_handoff_source5_refiner_valid" in outputs:
                source5_ref_w = source5_ref_w * outputs[
                    "proposal_handoff_source5_refiner_valid"
                ].to(device=source5_ref_quad.device, dtype=source5_ref_w.dtype)
            min_source5_iou = float(
                getattr(args, "proposal_handoff_source5_refiner_min_source5_iou", 0.0)
            )
            if (
                min_source5_iou > 0.0
                and prop_quad_iou is not None
                and "proposal_source_ids" in outputs
            ):
                with torch.no_grad():
                    source_ids_ref = outputs["proposal_source_ids"].to(
                        device=prop_quad_iou.device,
                        dtype=torch.long,
                    )
                    if source_ids_ref.shape == prop_quad_iou.shape:
                        source5_existing = source_ids_ref == source5_refiner_source_id
                        source5_oracle_ref = prop_quad_iou.detach().float().masked_fill(
                            ~source5_existing,
                            -1.0,
                        ).max(dim=-1).values
                        source5_ref_w = source5_ref_w * (
                            source5_oracle_ref.to(source5_ref_w.device) >= min_source5_iou
                        ).float()
            with torch.no_grad():
                ref_iou_grid = min(max(int(args.quad_iou_grid_size), 8), 32)
                ref_quad_iou = quad_iou_approx(
                    source5_ref_quad.reshape(-1, 4, 2),
                    gt_source5_ref.reshape(-1, 4, 2),
                    grid_size=ref_iou_grid,
                ).reshape(source5_ref_quad.shape[:2])
                proposal_handoff_source5_refiner_quad_iou = weighted_mean(
                    ref_quad_iou,
                    source5_ref_w,
                )
            if float(getattr(args, "proposal_handoff_source5_refiner_loss_weight", 0.0)) > 0.0:
                proposal_handoff_source5_refiner_point = weighted_mean(
                    F.smooth_l1_loss(
                        source5_ref_quad,
                        gt_source5_ref,
                        reduction="none",
                    ).sum(dim=(-1, -2)),
                    source5_ref_w,
                )
                proposal_handoff_source5_refiner_axis = quad_axis_alignment_loss(
                    source5_ref_quad,
                    gt_source5_ref,
                    source5_ref_w,
                    length_weight=float(args.quad_axis_length_weight),
                )
                proposal_handoff_source5_refiner_loss = (
                    proposal_handoff_source5_refiner_point
                    + float(getattr(args, "proposal_handoff_source5_refiner_axis_weight", 0.25))
                    * proposal_handoff_source5_refiner_axis
                )
                loss = (
                    loss
                    + float(args.proposal_handoff_source5_refiner_loss_weight)
                    * proposal_handoff_source5_refiner_loss
                )
            if (
                float(getattr(args, "proposal_handoff_source5_refiner_rank_weight", 0.0))
                > 0.0
                and prop_quad_iou is not None
                and "proposal_handoff_source5_refiner_scores" in outputs
                and "proposal_source_ids" in outputs
            ):
                rank_logits = outputs["proposal_handoff_source5_refiner_scores"].float()
                rank_k = int(rank_logits.shape[-1])
                if rank_k <= int(prop_quad_iou.shape[-1]):
                    rank_quality = prop_quad_iou.detach().float()[..., :rank_k].clamp(0.0, 1.0)
                    rank_source_ids = outputs["proposal_source_ids"].to(
                        device=rank_quality.device,
                        dtype=torch.long,
                    )[..., :rank_k]
                    rank_source5 = rank_source_ids == source5_refiner_source_id
                    target_quality = rank_quality.masked_fill(~rank_source5, -1.0)
                    target_idx = target_quality.argmax(dim=-1)
                    rank_valid = source5_ref_w * rank_source5.any(dim=-1).to(
                        device=source5_ref_w.device,
                        dtype=source5_ref_w.dtype,
                    )
                    rank_ce = F.cross_entropy(
                        rank_logits.reshape(-1, rank_k),
                        target_idx.reshape(-1),
                        reduction="none",
                    ).reshape_as(target_idx)
                    proposal_handoff_source5_refiner_rank = weighted_mean(rank_ce, rank_valid)
                    rank_top1_idx = rank_logits.detach().argmax(dim=-1, keepdim=True)
                    rank_top1_q = rank_quality.gather(-1, rank_top1_idx).squeeze(-1)
                    proposal_handoff_source5_refiner_rank_top1_quad_iou = weighted_mean(
                        rank_top1_q,
                        rank_valid,
                    )
                    rank_match = (rank_top1_idx.squeeze(-1) == target_idx).float()
                    proposal_handoff_source5_refiner_rank_top1_match = weighted_mean(
                        rank_match,
                        rank_valid,
                    )
                    loss = (
                        loss
                        + float(args.proposal_handoff_source5_refiner_rank_weight)
                        * proposal_handoff_source5_refiner_rank
                    )
            if (
                float(getattr(args, "proposal_handoff_source5_refiner_quality_weight", 0.0))
                > 0.0
                and "proposal_handoff_source5_refiner_logit" in outputs
            ):
                ref_quality_logits = outputs["proposal_handoff_source5_refiner_logit"].float()
                ref_quality_target = ref_quad_iou.detach().float().clamp(0.0, 1.0)
                ref_quality_pred = torch.sigmoid(ref_quality_logits).clamp(1e-4, 1.0 - 1e-4)
                ref_focal_weight = ref_quality_target + float(args.proposal_varifocal_alpha) * (
                    ref_quality_pred - ref_quality_target
                ).abs().pow(float(args.proposal_varifocal_gamma))
                ref_quality_bce = F.binary_cross_entropy_with_logits(
                    ref_quality_logits,
                    ref_quality_target,
                    reduction="none",
                )
                proposal_handoff_source5_refiner_quality = weighted_mean(
                    ref_quality_bce * ref_focal_weight,
                    source5_ref_w,
                )
                loss = (
                    loss
                    + float(args.proposal_handoff_source5_refiner_quality_weight)
                    * proposal_handoff_source5_refiner_quality
                )
        if "proposal_handoff_topk_query_refiner_quads" in outputs:
            topk_ref_quad = outputs["proposal_handoff_topk_query_refiner_quads"].float().clamp(0.0, 1.0)
            topk_base_quad = outputs["proposal_handoff_topk_query_refiner_base_quads"].float().clamp(0.0, 1.0)
            topk_valid = outputs["proposal_handoff_topk_query_refiner_valid"].to(
                device=topk_ref_quad.device,
                dtype=topk_ref_quad.dtype,
            )
            gt_topk_ref = batch["quad_xy"].to(topk_ref_quad.device).float()
            gt_topk_expand = gt_topk_ref[:, :, None, :, :].expand_as(topk_ref_quad)
            topk_w = frame_w.to(topk_ref_quad.device, dtype=topk_ref_quad.dtype).unsqueeze(-1) * topk_valid
            with torch.no_grad():
                topk_grid = min(max(int(args.quad_iou_grid_size), 8), 32)
                topk_ref_iou = quad_iou_approx(
                    topk_ref_quad.reshape(-1, 4, 2),
                    gt_topk_expand.reshape(-1, 4, 2),
                    grid_size=topk_grid,
                ).reshape(topk_ref_quad.shape[:3])
                topk_base_iou = quad_iou_approx(
                    topk_base_quad.reshape(-1, 4, 2),
                    gt_topk_expand.reshape(-1, 4, 2),
                    grid_size=topk_grid,
                ).reshape(topk_ref_quad.shape[:3])
                min_base_iou = float(
                    getattr(args, "proposal_handoff_topk_query_min_base_iou", 0.0)
                )
                if min_base_iou > 0.0:
                    topk_w = topk_w * (topk_base_iou >= min_base_iou).to(
                        device=topk_w.device,
                        dtype=topk_w.dtype,
                    )
                proposal_handoff_topk_query_refiner_quad_iou = weighted_mean(
                    topk_ref_iou,
                    topk_w,
                )
                proposal_handoff_topk_query_refiner_base_quad_iou = weighted_mean(
                    topk_base_iou,
                    topk_w,
                )
            if float(getattr(args, "proposal_handoff_topk_query_loss_weight", 0.0)) > 0.0:
                proposal_handoff_topk_query_refiner_point = weighted_mean(
                    F.smooth_l1_loss(
                        topk_ref_quad,
                        gt_topk_expand,
                        reduction="none",
                    ).sum(dim=(-1, -2)),
                    topk_w,
                )
                proposal_handoff_topk_query_refiner_axis = quad_axis_alignment_loss(
                    topk_ref_quad.reshape(-1, 4, 2),
                    gt_topk_expand.reshape(-1, 4, 2),
                    topk_w.reshape(-1),
                    length_weight=float(args.quad_axis_length_weight),
                )
                proposal_handoff_topk_query_refiner_loss = (
                    proposal_handoff_topk_query_refiner_point
                    + 0.25 * proposal_handoff_topk_query_refiner_axis
                )
                loss = (
                    loss
                    + float(args.proposal_handoff_topk_query_loss_weight)
                    * proposal_handoff_topk_query_refiner_loss
                )
            if (
                float(getattr(args, "proposal_handoff_topk_query_quality_weight", 0.0))
                > 0.0
                and "proposal_handoff_topk_query_refiner_logits" in outputs
            ):
                topk_quality_logits = outputs[
                    "proposal_handoff_topk_query_refiner_logits"
                ].float()
                topk_quality_target = topk_ref_iou.detach().float().clamp(0.0, 1.0)
                topk_quality_pred = torch.sigmoid(topk_quality_logits).clamp(1e-4, 1.0 - 1e-4)
                topk_focal_weight = topk_quality_target + float(args.proposal_varifocal_alpha) * (
                    topk_quality_pred - topk_quality_target
                ).abs().pow(float(args.proposal_varifocal_gamma))
                topk_quality_bce = F.binary_cross_entropy_with_logits(
                    topk_quality_logits,
                    topk_quality_target,
                    reduction="none",
                )
                proposal_handoff_topk_query_refiner_quality = weighted_mean(
                    topk_quality_bce * topk_focal_weight,
                    topk_w,
                )
                loss = (
                    loss
                    + float(args.proposal_handoff_topk_query_quality_weight)
                    * proposal_handoff_topk_query_refiner_quality
                )
            if (
                float(getattr(args, "proposal_handoff_topk_query_rank_weight", 0.0))
                > 0.0
                and "proposal_handoff_topk_query_refiner_logits" in outputs
            ):
                topk_rank_logits = outputs["proposal_handoff_topk_query_refiner_logits"].float()
                rank_quality = topk_ref_iou.detach().float().masked_fill(topk_w <= 0, -1.0)
                target_idx = rank_quality.argmax(dim=-1)
                rank_valid = (topk_w.sum(dim=-1) > 0).to(device=frame_w.device, dtype=frame_w.dtype) * frame_w
                topk_rank_ce = F.cross_entropy(
                    topk_rank_logits.reshape(-1, topk_rank_logits.shape[-1]),
                    target_idx.reshape(-1),
                    reduction="none",
                ).reshape_as(target_idx)
                proposal_handoff_topk_query_refiner_rank = weighted_mean(topk_rank_ce, rank_valid)
                rank_top1_idx = topk_rank_logits.detach().argmax(dim=-1, keepdim=True)
                rank_top1_q = topk_ref_iou.detach().float().gather(-1, rank_top1_idx).squeeze(-1)
                proposal_handoff_topk_query_refiner_rank_top1_quad_iou = weighted_mean(
                    rank_top1_q,
                    rank_valid,
                )
                proposal_handoff_topk_query_refiner_rank_top1_match = weighted_mean(
                    (rank_top1_idx.squeeze(-1) == target_idx).float(),
                    rank_valid,
                )
                loss = (
                    loss
                    + float(args.proposal_handoff_topk_query_rank_weight)
                    * proposal_handoff_topk_query_refiner_rank
                )
            if (
                float(getattr(args, "proposal_handoff_topk_query_soft_rank_weight", 0.0))
                > 0.0
                and "proposal_handoff_topk_query_refiner_logits" in outputs
            ):
                topk_soft_logits = outputs["proposal_handoff_topk_query_refiner_logits"].float()
                soft_valid = topk_w > 0
                soft_rank_valid = (
                    soft_valid.any(dim=-1).to(device=frame_w.device, dtype=frame_w.dtype)
                    * frame_w
                )
                tau = max(float(getattr(args, "proposal_handoff_topk_query_soft_rank_tau", 0.05)), 1e-4)
                soft_quality = topk_ref_iou.detach().float().masked_fill(~soft_valid, -1e4)
                soft_target = F.softmax(soft_quality / tau, dim=-1) * soft_valid.to(
                    dtype=topk_soft_logits.dtype
                )
                soft_target = soft_target / soft_target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                soft_log_prob = F.log_softmax(
                    topk_soft_logits.masked_fill(~soft_valid, -1e4),
                    dim=-1,
                )
                soft_rank_ce = -(soft_target * soft_log_prob).sum(dim=-1)
                proposal_handoff_topk_query_refiner_soft_rank = weighted_mean(
                    soft_rank_ce,
                    soft_rank_valid,
                )
                loss = (
                    loss
                    + float(args.proposal_handoff_topk_query_soft_rank_weight)
                    * proposal_handoff_topk_query_refiner_soft_rank
                )
        if "proposal_frame_anchor_logits" in outputs and prop_quad_iou is not None:
            anchor_logits = outputs["proposal_frame_anchor_logits"].float()
            with torch.no_grad():
                frame_quality = prop_quad_iou.detach().float().clamp(0.0, 1.0).max(dim=-1).values
                anchor_top1 = anchor_logits.detach().argmax(dim=-1, keepdim=True)
                proposal_frame_anchor_top1_oracle_iou = frame_quality.gather(-1, anchor_top1).mean()
                if "proposal_frame_anchor_weights" in outputs:
                    anchor_weights_eval = outputs["proposal_frame_anchor_weights"].detach().float()
                else:
                    anchor_weights_eval = F.softmax(anchor_logits.detach(), dim=-1)
                proposal_frame_anchor_entropy = -(
                    anchor_weights_eval.clamp_min(1e-6)
                    * anchor_weights_eval.clamp_min(1e-6).log()
                ).sum(dim=-1).mean() / math.log(max(anchor_weights_eval.shape[-1], 2))
            if float(getattr(args, "proposal_frame_anchor_loss_weight", 0.0)) > 0.0:
                tau = max(float(getattr(args, "proposal_frame_anchor_loss_tau", 0.05)), 1e-4)
                with torch.no_grad():
                    frame_quality = prop_quad_iou.detach().float().clamp(0.0, 1.0).max(dim=-1).values
                    target_dist = F.softmax(frame_quality / tau, dim=-1)
                log_prob = F.log_softmax(anchor_logits, dim=-1)
                anchor_kl = F.kl_div(log_prob, target_dist, reduction="none").sum(dim=-1)
                proposal_frame_anchor = anchor_kl.mean()
                loss = loss + float(args.proposal_frame_anchor_loss_weight) * proposal_frame_anchor
        if "proposal_active_tip_logits" in outputs and prop_quad_iou is not None:
            active_tip_logits = outputs["proposal_active_tip_logits"].float()
            with torch.no_grad():
                active_tip_eval = active_tip_logits.detach()
                q_eval = prop_quad_iou.detach().float()
                k_eval = q_eval.shape[-1]
                active_top1 = active_tip_eval.argmax(dim=-1, keepdim=True)
                proposal_active_tip_top1_quad_iou = weighted_mean(
                    q_eval.gather(-1, active_top1).squeeze(-1),
                    frame_w,
                )
                topk_eval = min(3, int(k_eval))
                active_topk = active_tip_eval.topk(topk_eval, dim=-1).indices
                active_topk_q = q_eval.gather(-1, active_topk)
                proposal_active_tip_top3_oracle_quad_iou = weighted_mean(
                    active_topk_q.max(dim=-1).values,
                    frame_w,
                )
                oracle_idx = q_eval.argmax(dim=-1, keepdim=True)
                active_contains = (active_topk == oracle_idx).any(dim=-1).float()
                proposal_active_tip_top3_contains_oracle = weighted_mean(active_contains, frame_w)
                q_centered = q_eval - q_eval.mean(dim=-1, keepdim=True)
                tip_centered = active_tip_eval - active_tip_eval.mean(dim=-1, keepdim=True)
                denom = (
                    q_centered.square().sum(dim=-1).sqrt()
                    * tip_centered.square().sum(dim=-1).sqrt()
                ).clamp_min(1e-6)
                active_corr = (q_centered * tip_centered).sum(dim=-1) / denom
                proposal_active_tip_iou_corr = weighted_mean(active_corr, frame_w)
            if float(getattr(args, "proposal_active_tip_selector_loss_weight", 0.0)) > 0.0:
                quality_target = prop_quad_iou.detach().float().clamp(0.0, 1.0)
                quality_pred = torch.sigmoid(active_tip_logits).clamp(1e-4, 1.0 - 1e-4)
                focal_weight = quality_target + float(args.proposal_varifocal_alpha) * (
                    quality_pred - quality_target
                ).abs().pow(float(args.proposal_varifocal_gamma))
                tip_vfl = (
                    F.binary_cross_entropy_with_logits(
                        active_tip_logits,
                        quality_target,
                        reduction="none",
                    )
                    * focal_weight
                )
                proposal_active_tip_varifocal = weighted_mean(tip_vfl.mean(dim=-1), frame_w)
                loss = loss + float(args.proposal_active_tip_selector_loss_weight) * proposal_active_tip_varifocal
        if "proposal_local_corr_logits" in outputs and prop_quad_iou is not None:
            local_corr_logits = outputs["proposal_local_corr_logits"].float()
            if local_corr_logits.shape == prop_quad_iou.shape:
                with torch.no_grad():
                    local_corr_eval = local_corr_logits.detach()
                    q_eval = prop_quad_iou.detach().float()
                    k_eval = q_eval.shape[-1]
                    corr_top1 = local_corr_eval.argmax(dim=-1, keepdim=True)
                    proposal_local_corr_top1_quad_iou = weighted_mean(
                        q_eval.gather(-1, corr_top1).squeeze(-1),
                        frame_w,
                    )
                    topk_eval = min(3, int(k_eval))
                    corr_topk = local_corr_eval.topk(topk_eval, dim=-1).indices
                    corr_topk_q = q_eval.gather(-1, corr_topk)
                    proposal_local_corr_top3_oracle_quad_iou = weighted_mean(
                        corr_topk_q.max(dim=-1).values,
                        frame_w,
                    )
                    q_centered = q_eval - q_eval.mean(dim=-1, keepdim=True)
                    corr_centered = local_corr_eval - local_corr_eval.mean(dim=-1, keepdim=True)
                    denom = (
                        q_centered.square().sum(dim=-1).sqrt()
                        * corr_centered.square().sum(dim=-1).sqrt()
                    ).clamp_min(1e-6)
                    local_corr = (q_centered * corr_centered).sum(dim=-1) / denom
                    proposal_local_corr_iou_corr = weighted_mean(local_corr, frame_w)
        if (
            "proposal_logits" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_pairwise_rank_weight", 0.0)) > 0.0
        ):
            logits = outputs["proposal_logits"].float()
            with torch.no_grad():
                q = prop_quad_iou.detach().float()
                q_gap = q.unsqueeze(-1) - q.unsqueeze(-2)
                pair_mask = q_gap > float(args.proposal_pairwise_iou_gap)
            logit_gap = logits.unsqueeze(-1) - logits.unsqueeze(-2)
            pair_loss = F.softplus(float(args.proposal_pairwise_rank_margin) - logit_gap)
            pair_loss = pair_loss * pair_mask.float()
            pair_count = pair_mask.float().sum(dim=(-1, -2))
            pair_frame = pair_loss.sum(dim=(-1, -2)) / pair_count.clamp_min(1.0)
            pair_frame = torch.where(pair_count > 0, pair_frame, torch.zeros_like(pair_frame))
            proposal_pairwise_rank = weighted_mean(pair_frame, frame_w)
            loss = loss + float(args.proposal_pairwise_rank_weight) * proposal_pairwise_rank
        if (
            "proposal_logits" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_hardneg_rank_weight", 0.0)) > 0.0
        ):
            logits = outputs["proposal_logits"].float()
            with torch.no_grad():
                q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
                oracle_idx = q.argmax(dim=-1, keepdim=True)
                oracle_q = q.gather(-1, oracle_idx)
                neg_mask = q < (oracle_q - float(args.proposal_hardneg_rank_iou_gap))
                neg_mask.scatter_(-1, oracle_idx, False)
                neg_count = neg_mask.float().sum(dim=-1)
                topk = min(max(int(args.proposal_hardneg_rank_topk), 1), int(q.shape[-1]))
                neg_logits_for_topk = logits.detach().masked_fill(~neg_mask, -50.0)
                hard_neg_idx = neg_logits_for_topk.topk(topk, dim=-1).indices
                hard_neg_valid = neg_mask.gather(-1, hard_neg_idx)
            pos_logits = logits.gather(-1, oracle_idx)
            hard_neg_logits = logits.gather(-1, hard_neg_idx)
            hardneg_loss = F.softplus(
                float(args.proposal_hardneg_rank_margin) - (pos_logits - hard_neg_logits)
            )
            hardneg_loss = hardneg_loss * hard_neg_valid.float()
            hardneg_frame = hardneg_loss.sum(dim=-1) / hard_neg_valid.float().sum(dim=-1).clamp_min(1.0)
            hardneg_frame = torch.where(neg_count > 0, hardneg_frame, torch.zeros_like(hardneg_frame))
            proposal_hardneg_rank = weighted_mean(hardneg_frame, frame_w)
            loss = loss + float(args.proposal_hardneg_rank_weight) * proposal_hardneg_rank
        if (
            "proposal_logits" in outputs
            and "proposal_source_ids" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_source4_demote_weight", 0.0)) > 0.0
        ):
            logits = outputs["proposal_logits"].float()
            source_ids = outputs["proposal_source_ids"].to(device=logits.device, dtype=torch.long)
            if source_ids.shape == logits.shape:
                with torch.no_grad():
                    q = prop_quad_iou.detach().float().to(device=logits.device).clamp(0.0, 1.0)
                    oracle_idx = q.argmax(dim=-1, keepdim=True)
                    oracle_q = q.gather(-1, oracle_idx)
                    gap = float(args.proposal_source4_demote_iou_gap)
                    positive_sources = parse_int_csv(args.proposal_source4_demote_positive_sources)
                    if not positive_sources:
                        positive_sources = [0, 5, 7, 8]
                    pos_source_mask = torch.zeros_like(source_ids, dtype=torch.bool)
                    for source_id in positive_sources:
                        pos_source_mask = pos_source_mask | (source_ids == int(source_id))
                    pos_mask = pos_source_mask & (q >= (oracle_q - gap))
                    pos_present = pos_mask.any(dim=-1, keepdim=True)

                    bad_source4 = (source_ids == 4) & (q < (oracle_q - gap))
                    bad_source4.scatter_(-1, oracle_idx, False)
                    bad_count = bad_source4.float().sum(dim=-1)
                    topk = min(max(int(args.proposal_source4_demote_topk), 1), int(q.shape[-1]))
                    bad_rank_logits = logits.detach().masked_fill(~bad_source4, -50.0)
                    bad_idx = bad_rank_logits.topk(topk, dim=-1).indices
                    bad_valid = bad_source4.gather(-1, bad_idx) & pos_present
                pos_logits = logits.masked_fill(~pos_mask, -50.0).max(dim=-1, keepdim=True).values
                bad_logits = logits.gather(-1, bad_idx)
                demote_loss = F.softplus(
                    float(args.proposal_source4_demote_margin) - (pos_logits - bad_logits)
                )
                demote_loss = demote_loss * bad_valid.float()
                demote_count = bad_valid.float().sum(dim=-1)
                demote_frame = demote_loss.sum(dim=-1) / demote_count.clamp_min(1.0)
                demote_frame = torch.where(
                    (bad_count > 0) & (demote_count > 0),
                    demote_frame,
                    torch.zeros_like(demote_frame),
                )
                proposal_source4_demote = weighted_mean(demote_frame, frame_w)
                loss = loss + float(args.proposal_source4_demote_weight) * proposal_source4_demote
        if (
            "proposal_logits" in outputs
            and "proposal_quads" in outputs
            and prop_quad_iou is not None
            and "tool_kpt_vec_16" in batch
            and float(getattr(args, "proposal_tool_hardneg_weight", 0.0)) > 0.0
        ):
            logits = outputs["proposal_logits"].float()
            quads = outputs["proposal_quads"].to(device=logits.device, dtype=logits.dtype)
            tool_vec = batch["tool_kpt_vec_16"].to(device=logits.device, dtype=logits.dtype)
            if (
                logits.ndim == 3
                and quads.ndim == 5
                and tool_vec.ndim == 3
                and tool_vec.shape[:2] == logits.shape[:2]
                and quads.shape[:3] == logits.shape
            ):
                with torch.no_grad():
                    q = prop_quad_iou.detach().to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
                    centers = quads.detach().mean(dim=-2)
                    kpt_xy = tool_vec[..., :6].reshape(tool_vec.shape[0], tool_vec.shape[1], 3, 2).clamp(0.0, 1.0)
                    kpt_vis = tool_vec[..., 6:9].reshape(tool_vec.shape[0], tool_vec.shape[1], 3).clamp(0.0, 1.0)
                    seg_a = kpt_xy[:, :, :2, :]
                    seg_b = kpt_xy[:, :, 1:, :]
                    point = centers[:, :, :, None, :]
                    a = seg_a[:, :, None, :, :]
                    b_seg = seg_b[:, :, None, :, :]
                    ab = b_seg - a
                    ap = point - a
                    denom = ab.square().sum(dim=-1).clamp_min(1e-6)
                    seg_t = ((ap * ab).sum(dim=-1) / denom).clamp(0.0, 1.0)
                    proj = a + seg_t[..., None] * ab
                    dist = (point - proj).norm(dim=-1)
                    seg_vis = torch.minimum(kpt_vis[:, :, :2], kpt_vis[:, :, 1:])
                    sigma = max(float(args.proposal_tool_hardneg_sigma), 1e-4)
                    tool_score = (
                        torch.exp(-dist / sigma) * seg_vis[:, :, None, :]
                    ).max(dim=-1).values
                    oracle_idx = q.argmax(dim=-1, keepdim=True)
                    oracle_q = q.gather(-1, oracle_idx)
                    neg_mask = (
                        (tool_score >= float(args.proposal_tool_hardneg_min_score))
                        & (q < (oracle_q - float(args.proposal_tool_hardneg_iou_gap)))
                    )
                    neg_mask.scatter_(-1, oracle_idx, False)
                    tool_rank_score = logits.detach() + tool_score.clamp_min(1e-6).log()
                    topk = min(max(int(args.proposal_tool_hardneg_topk), 1), int(q.shape[-1]))
                    hard_tool_idx = tool_rank_score.masked_fill(~neg_mask, -50.0).topk(topk, dim=-1).indices
                    hard_tool_valid = neg_mask.gather(-1, hard_tool_idx)
                    hard_tool_weight = tool_score.gather(-1, hard_tool_idx).detach() * hard_tool_valid.float()
                pos_logits = logits.gather(-1, oracle_idx)
                hard_tool_logits = logits.gather(-1, hard_tool_idx)
                tool_hardneg_loss = F.softplus(
                    float(args.proposal_tool_hardneg_margin) - (pos_logits - hard_tool_logits)
                )
                tool_hardneg_loss = tool_hardneg_loss * hard_tool_weight
                hard_tool_weight_sum = hard_tool_weight.sum(dim=-1)
                tool_hardneg_frame = tool_hardneg_loss.sum(dim=-1) / hard_tool_weight_sum.clamp_min(1e-6)
                tool_hardneg_frame = torch.where(
                    hard_tool_weight_sum > 0,
                    tool_hardneg_frame,
                    torch.zeros_like(tool_hardneg_frame),
                )
                proposal_tool_hardneg = weighted_mean(tool_hardneg_frame, frame_w)
                loss = loss + float(args.proposal_tool_hardneg_weight) * proposal_tool_hardneg
        if (
            "proposal_logits" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_soft_rank_weight", 0.0)) > 0.0
        ):
            logits = outputs["proposal_logits"].float()
            with torch.no_grad():
                tau = max(float(args.proposal_soft_rank_tau), 1e-4)
                target_dist = F.softmax(prop_quad_iou.detach().float() / tau, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            soft_rank_frame = F.kl_div(log_prob, target_dist, reduction="none").sum(dim=-1)
            proposal_soft_rank = weighted_mean(soft_rank_frame, frame_w)
            loss = loss + float(args.proposal_soft_rank_weight) * proposal_soft_rank
        if (
            "proposal_logits" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_owner_band_weight", 0.0)) > 0.0
        ):
            logits = outputs["proposal_logits"].float()
            with torch.no_grad():
                q = prop_quad_iou.detach().float().to(device=logits.device).clamp(0.0, 1.0)
                qmax, oracle_idx = q.max(dim=-1, keepdim=True)
                gap = max(float(getattr(args, "proposal_owner_band_gap", 0.04)), 0.0)
                min_iou = max(float(getattr(args, "proposal_owner_band_min_iou", 0.05)), 0.0)
                band = (q >= (qmax - gap)) & (q >= min_iou)
                band.scatter_(-1, oracle_idx, True)
                topk = int(getattr(args, "proposal_owner_band_topk", 0))
                if topk > 0:
                    topk = min(max(topk, 1), int(q.shape[-1]))
                    topk_idx = q.topk(topk, dim=-1).indices
                    topk_mask = torch.zeros_like(band)
                    topk_mask.scatter_(-1, topk_idx, True)
                    band = band & topk_mask
                    band.scatter_(-1, oracle_idx, True)

                tau = max(float(getattr(args, "proposal_owner_band_tau", 0.05)), 1e-4)
                raw_mass = torch.exp(((q - qmax) / tau).clamp(min=-50.0, max=0.0)) * band.float()
                if (
                    bool(getattr(args, "proposal_owner_band_source_balanced", False))
                    and "proposal_source_ids" in outputs
                    and outputs["proposal_source_ids"].shape == q.shape
                ):
                    source_ids = outputs["proposal_source_ids"].to(device=q.device, dtype=torch.long)
                    target_raw = torch.zeros_like(q)
                    source_count = torch.zeros_like(qmax.squeeze(-1))
                    max_source_id = int(source_ids.detach().max().clamp_min(0).cpu().item()) if source_ids.numel() else 0
                    for source_id in range(max_source_id + 1):
                        source_pos = band & (source_ids == int(source_id))
                        source_raw = raw_mass * source_pos.float()
                        source_sum = source_raw.sum(dim=-1, keepdim=True)
                        source_present = source_sum.squeeze(-1) > 0
                        target_raw = target_raw + source_raw / source_sum.clamp_min(1e-8)
                        source_count = source_count + source_present.float()
                    target_dist = target_raw / source_count.clamp_min(1.0).unsqueeze(-1)
                else:
                    target_dist = raw_mass / raw_mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            log_prob = F.log_softmax(logits, dim=-1)
            owner_band_frame = F.kl_div(log_prob, target_dist, reduction="none").sum(dim=-1)
            proposal_owner_band = weighted_mean(owner_band_frame, frame_w)
            loss = loss + float(args.proposal_owner_band_weight) * proposal_owner_band
        if (
            "proposal_logits" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_neuralsort_top1_weight", 0.0)) > 0.0
        ):
            proposal_neuralsort_top1 = neural_sort_top1_loss(
                outputs["proposal_logits"].float(),
                prop_quad_iou.detach().float(),
                frame_w,
                tau=float(args.proposal_neuralsort_tau),
            )
            loss = loss + float(args.proposal_neuralsort_top1_weight) * proposal_neuralsort_top1
        if (
            "proposal_logits" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_ranksort_weight", 0.0)) > 0.0
        ):
            proposal_ranksort = proposal_ranksort_loss(
                outputs["proposal_logits"].float(),
                prop_quad_iou.detach().float(),
                frame_w,
                positive_threshold=float(args.proposal_ranksort_positive_threshold),
                delta_rs=float(args.proposal_ranksort_delta),
            )
            loss = loss + float(args.proposal_ranksort_weight) * proposal_ranksort
        if (
            "proposal_logits" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_task_aligned_weight", 0.0)) > 0.0
        ):
            # Adapt TOOD Task Alignment Learning to the per-frame candidate
            # selector: positives are the candidates whose detached score and
            # localization quality agree.  Unlike all-candidate VFL, non-top-k
            # candidates become true negatives, which directly attacks the
            # observed oracle-gap.
            logits = outputs["proposal_logits"].float()
            with torch.no_grad():
                q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
                score = logits.detach().sigmoid().clamp(1e-4, 1.0 - 1e-4)
                alpha = max(float(args.proposal_task_aligned_alpha), 0.0)
                beta = max(float(args.proposal_task_aligned_beta), 0.0)
                align = score.pow(alpha) * q.pow(beta)
                topk = min(max(int(args.proposal_task_aligned_topk), 1), int(align.shape[-1]))
                topk_idx = align.topk(topk, dim=-1).indices
                target = torch.zeros_like(q)
                topk_align = align.gather(-1, topk_idx)
                topk_q = q.gather(-1, topk_idx)
                norm = topk_align.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
                # Same spirit as task-aligned normalized assignment: retain
                # localization quality scale while using alignment for which
                # candidates are supervised as positives.
                topk_target = (topk_align / norm * topk_q.max(dim=-1, keepdim=True).values).clamp(0.0, 1.0)
                target.scatter_(-1, topk_idx, topk_target)
            pred = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
            focal_weight = target + float(args.proposal_varifocal_alpha) * (
                pred - target
            ).abs().pow(float(args.proposal_varifocal_gamma))
            tal_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * focal_weight
            proposal_task_aligned = weighted_mean(tal_loss.mean(dim=-1), frame_w)
            loss = loss + float(args.proposal_task_aligned_weight) * proposal_task_aligned
        if (
            "proposal_logits" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_c2f_assign_weight", 0.0)) > 0.0
        ):
            # Adapt DCFL/C2F assignment to the per-frame candidate selector.
            # The coarse step is score-independent, so high-IoU candidates do
            # not disappear merely because the current selector scores are bad.
            # The fine step then lets the current score break ties among the
            # coarse high-localization candidates.
            logits = outputs["proposal_logits"].float()
            with torch.no_grad():
                q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
                score = logits.detach().sigmoid().clamp(1e-4, 1.0 - 1e-4)
                coarse_k = min(max(int(args.proposal_c2f_topk), 1), int(q.shape[-1]))
                fine_q = min(max(int(args.proposal_c2f_topq), 1), coarse_k)
                coarse_idx = q.topk(coarse_k, dim=-1).indices
                coarse_q = q.gather(-1, coarse_idx)
                coarse_score = score.gather(-1, coarse_idx)
                fine_quality = coarse_q + float(args.proposal_c2f_score_weight) * coarse_score
                fine_idx_in_coarse = fine_quality.topk(fine_q, dim=-1).indices
                fine_idx = coarse_idx.gather(-1, fine_idx_in_coarse)

                target = torch.zeros_like(q)
                fine_target = q.gather(-1, fine_idx)
                target.scatter_(-1, fine_idx, fine_target)

                # Always keep the oracle candidate positive. This preserves
                # the DCFL spirit of low-quality matching while preventing the
                # selector from training against the best available proposal.
                oracle_idx = q.argmax(dim=-1, keepdim=True)
                target.scatter_(-1, oracle_idx, q.gather(-1, oracle_idx))
            pred = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
            focal_weight = target + float(args.proposal_varifocal_alpha) * (
                pred - target
            ).abs().pow(float(args.proposal_varifocal_gamma))
            c2f_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * focal_weight
            proposal_c2f_assign = weighted_mean(c2f_loss.mean(dim=-1), frame_w)
            loss = loss + float(args.proposal_c2f_assign_weight) * proposal_c2f_assign
        if (
            "proposal_logits" in outputs
            and "proposal_xyxy" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_pcl_cluster_weight", 0.0)) > 0.0
        ):
            # PCL-style proposal cluster mining: instead of treating only one
            # oracle candidate as positive, create soft positive clusters
            # around the detached high-IoU candidates.  This targets the
            # observed failure where the candidate bank has good quads but the
            # selector over-commits to a brittle source/top1 hypothesis.
            logits = outputs["proposal_logits"].float()
            with torch.no_grad():
                q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
                boxes = outputs["proposal_xyxy"].detach().float()
                pair_iou = box_iou_xyxy(boxes.unsqueeze(-2), boxes.unsqueeze(-3)).clamp(0.0, 1.0)
                center_k = min(max(int(args.proposal_pcl_cluster_topk), 1), int(q.shape[-1]))
                center_idx = q.topk(center_k, dim=-1).indices
                center_mask = torch.zeros_like(q, dtype=torch.bool)
                center_mask.scatter_(-1, center_idx, True)

                center_overlap, center_owner = pair_iou.masked_fill(
                    ~center_mask.unsqueeze(-2),
                    -1.0,
                ).max(dim=-1)
                center_owner = center_owner.clamp_min(0)
                center_quality = q.gather(-1, center_owner)

                fg_thr = float(args.proposal_pcl_cluster_iou_threshold)
                bg_thr = float(args.proposal_pcl_cluster_bg_threshold)
                pos = center_overlap >= fg_thr
                neg = center_overlap < bg_thr
                pos.scatter_(-1, center_idx, True)
                valid = pos | neg

                target = torch.zeros_like(q)
                # Propagate cluster-center quality to nearby proposals, but
                # keep the candidate's own IoU if it is better.  The overlap
                # factor avoids making weak fringe proposals identical to the
                # center.
                cluster_quality = (center_quality * center_overlap.clamp(0.0, 1.0)).clamp(0.0, 1.0)
                pos_target = torch.maximum(q, cluster_quality)
                target = torch.where(pos, pos_target, target)

            pred = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
            focal_weight = target + float(args.proposal_varifocal_alpha) * (
                pred - target
            ).abs().pow(float(args.proposal_varifocal_gamma))
            pcl_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * focal_weight
            pcl_loss = pcl_loss * valid.float()
            pcl_count = valid.float().sum(dim=-1)
            pcl_frame = pcl_loss.sum(dim=-1) / pcl_count.clamp_min(1.0)
            pcl_frame = torch.where(pcl_count > 0, pcl_frame, torch.zeros_like(pcl_frame))
            proposal_pcl_cluster = weighted_mean(pcl_frame, frame_w)
            loss = loss + float(args.proposal_pcl_cluster_weight) * proposal_pcl_cluster
        if (
            "proposal_logits" in outputs
            and "proposal_xyxy" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_pcl_cluster_ce_weight", 0.0)) > 0.0
        ):
            # C-MIL/PCL-style local-consistency objective.  H342's PCL term
            # supervises candidate logits independently.  This term instead
            # groups neighboring candidates around the top-IoU centers and
            # trains the aggregate cluster score to prefer the oracle cluster.
            # It keeps multiple locally consistent proposals alive without
            # requiring the exact oracle candidate to be the scalar top1.
            logits = outputs["proposal_logits"].float()
            with torch.no_grad():
                q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
                boxes = outputs["proposal_xyxy"].detach().float()
                pair_iou = box_iou_xyxy(boxes.unsqueeze(-2), boxes.unsqueeze(-3)).clamp(0.0, 1.0)
                center_k = min(max(int(args.proposal_pcl_cluster_topk), 1), int(q.shape[-1]))
                center_idx = q.topk(center_k, dim=-1).indices
                center_pair_iou = pair_iou.gather(
                    -1,
                    center_idx.unsqueeze(-2).expand(*q.shape, center_k),
                )
                cluster_owner = center_pair_iou.argmax(dim=-1)
            neg = torch.finfo(logits.dtype).min
            cluster_scores = []
            for cluster_id in range(center_k):
                cluster_mask = cluster_owner == int(cluster_id)
                cluster_logits = logits.masked_fill(~cluster_mask, neg)
                cluster_scores.append(torch.logsumexp(cluster_logits, dim=-1))
            cluster_scores_t = torch.stack(cluster_scores, dim=-1)
            cluster_target = torch.zeros_like(frame_w, dtype=torch.long)
            cluster_ce = F.cross_entropy(
                cluster_scores_t.reshape(-1, center_k),
                cluster_target.reshape(-1),
                reduction="none",
            ).reshape_as(frame_w)
            proposal_pcl_cluster_ce = weighted_mean(cluster_ce, frame_w)
            loss = loss + float(args.proposal_pcl_cluster_ce_weight) * proposal_pcl_cluster_ce
        if (
            ("proposal_oicr_logits" in outputs or "proposal_oicr_logits_list" in outputs)
            and "proposal_xyxy" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_oicr_refine_ce_weight", 0.0)) > 0.0
        ):
            # PCL/OICR-style separate refine_score supervision.  If multiple
            # refinement heads are present, supervise every refinement stream
            # and average their CE, matching OICR's multi-stream intent while
            # keeping the target bank aligned to our final proposals.
            logits_terms: list[torch.Tensor] = []
            if "proposal_oicr_logits_list" in outputs:
                seq = outputs["proposal_oicr_logits_list"].float()
                logits_terms.extend([seq[i] for i in range(int(seq.shape[0]))])
            elif "proposal_oicr_logits" in outputs:
                logits_terms.append(outputs["proposal_oicr_logits"].float())
            with torch.no_grad():
                q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
                boxes = outputs["proposal_xyxy"].detach().float()
                pair_iou = box_iou_xyxy(boxes.unsqueeze(-2), boxes.unsqueeze(-3)).clamp(0.0, 1.0)
                center_k = min(max(int(args.proposal_pcl_cluster_topk), 1), int(q.shape[-1]))
                center_idx = q.topk(center_k, dim=-1).indices
                center_pair_iou = pair_iou.gather(
                    -1,
                    center_idx.unsqueeze(-2).expand(*q.shape, center_k),
                )
                cluster_owner = center_pair_iou.argmax(dim=-1)
            oicr_losses = []
            for logits in logits_terms:
                if logits.shape != q.shape:
                    continue
                neg = torch.finfo(logits.dtype).min
                cluster_scores = []
                for cluster_id in range(center_k):
                    cluster_mask = cluster_owner == int(cluster_id)
                    cluster_logits = logits.masked_fill(~cluster_mask, neg)
                    cluster_scores.append(torch.logsumexp(cluster_logits, dim=-1))
                cluster_scores_t = torch.stack(cluster_scores, dim=-1)
                cluster_target = torch.zeros_like(frame_w, dtype=torch.long)
                oicr_ce = F.cross_entropy(
                    cluster_scores_t.reshape(-1, center_k),
                    cluster_target.reshape(-1),
                    reduction="none",
                ).reshape_as(frame_w)
                oicr_losses.append(weighted_mean(oicr_ce, frame_w))
            if oicr_losses:
                proposal_oicr_refine_ce = torch.stack(oicr_losses).mean()
                loss = loss + float(args.proposal_oicr_refine_ce_weight) * proposal_oicr_refine_ce
        if (
            "proposal_logits" in outputs
            and "proposal_source_ids" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_source0_multimatch_weight", 0.0)) > 0.0
        ):
            # SPE-style multi-match assignment for the base visual proposal
            # source.  Instead of one oracle candidate owning the frame, the
            # top-K source0 candidates by localization quality are trained as
            # positives.  This aims to make the visual proposal bank itself
            # competitive with the source4 corner proposal, without adding a
            # fixed source penalty.
            logits = outputs["proposal_logits"].float()
            source_ids = outputs["proposal_source_ids"].to(device=logits.device, dtype=torch.long)
            with torch.no_grad():
                q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
                source0 = source_ids == 0
                q_source0 = q.masked_fill(~source0, -1.0)
                topk = min(max(int(args.proposal_source0_multimatch_topk), 1), int(q.shape[-1]))
                top_idx = q_source0.topk(topk, dim=-1).indices
                target = torch.zeros_like(q)
                top_q = q.gather(-1, top_idx)
                top_valid = source0.gather(-1, top_idx) & (top_q > 0.0)
                target.scatter_(-1, top_idx, torch.where(top_valid, top_q, torch.zeros_like(top_q)))
                valid = source0
            pred = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
            focal_weight = target + float(args.proposal_varifocal_alpha) * (
                pred - target
            ).abs().pow(float(args.proposal_varifocal_gamma))
            mm_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * focal_weight
            mm_loss = mm_loss * valid.float()
            mm_count = valid.float().sum(dim=-1)
            mm_frame = mm_loss.sum(dim=-1) / mm_count.clamp_min(1.0)
            mm_frame = torch.where(mm_count > 0, mm_frame, torch.zeros_like(mm_frame))
            proposal_source0_multimatch = weighted_mean(mm_frame, frame_w)
            loss = loss + float(args.proposal_source0_multimatch_weight) * proposal_source0_multimatch
        if (
            "proposal_logits" in outputs
            and "proposal_source_ids" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_source5_multimatch_weight", 0.0)) > 0.0
        ):
            # Same assignment as source0 multimatch, but applied to the
            # token-affinity proposal source. H414/H415 found source5 oracle
            # candidates that were almost never selected.
            logits = outputs["proposal_logits"].float()
            source_ids = outputs["proposal_source_ids"].to(device=logits.device, dtype=torch.long)
            with torch.no_grad():
                q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
                source5 = source_ids == 5
                q_source5 = q.masked_fill(~source5, -1.0)
                topk = min(max(int(args.proposal_source5_multimatch_topk), 1), int(q.shape[-1]))
                top_idx = q_source5.topk(topk, dim=-1).indices
                target = torch.zeros_like(q)
                top_q = q.gather(-1, top_idx)
                top_valid = source5.gather(-1, top_idx) & (top_q > 0.0)
                target.scatter_(-1, top_idx, torch.where(top_valid, top_q, torch.zeros_like(top_q)))
                valid = source5
            pred = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
            focal_weight = target + float(args.proposal_varifocal_alpha) * (
                pred - target
            ).abs().pow(float(args.proposal_varifocal_gamma))
            mm_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * focal_weight
            mm_loss = mm_loss * valid.float()
            mm_count = valid.float().sum(dim=-1)
            mm_frame = mm_loss.sum(dim=-1) / mm_count.clamp_min(1.0)
            mm_frame = torch.where(mm_count > 0, mm_frame, torch.zeros_like(mm_frame))
            proposal_source5_multimatch = weighted_mean(mm_frame, frame_w)
            loss = loss + float(args.proposal_source5_multimatch_weight) * proposal_source5_multimatch
        if (
            "proposal_logits" in outputs
            and "proposal_source_ids" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_source5_soft_owner_weight", 0.0)) > 0.0
        ):
            logits = outputs["proposal_logits"].float()
            source_ids = outputs["proposal_source_ids"].to(device=logits.device, dtype=torch.long)
            if source_ids.shape == logits.shape:
                with torch.no_grad():
                    q = prop_quad_iou.detach().float().to(device=logits.device).clamp(0.0, 1.0)
                    source5 = source_ids == 5
                    source5_present = source5.any(dim=-1)
                    topk = min(max(int(args.proposal_source5_soft_owner_topk), 1), int(q.shape[-1]))
                    q_source5 = q.masked_fill(~source5, -50.0)
                    owner_idx = q_source5.topk(topk, dim=-1).indices
                    owner_valid = source5.gather(-1, owner_idx)
                    owner_q = q.gather(-1, owner_idx).masked_fill(~owner_valid, -50.0)
                    tau = max(float(args.proposal_source5_soft_owner_tau), 1e-4)
                    owner_target = F.softmax(owner_q / tau, dim=-1).masked_fill(~owner_valid, 0.0)
                    target = torch.zeros_like(q)
                    target.scatter_(-1, owner_idx, owner_target)
                log_prob = F.log_softmax(logits.masked_fill(~source5, -50.0), dim=-1)
                soft_owner_frame = (
                    target * (target.clamp_min(1e-8).log() - log_prob)
                ).sum(dim=-1)
                soft_owner_frame = torch.where(
                    source5_present,
                    soft_owner_frame,
                    torch.zeros_like(soft_owner_frame),
                )
                proposal_source5_soft_owner = weighted_mean(
                    soft_owner_frame,
                    frame_w * source5_present.to(device=frame_w.device, dtype=frame_w.dtype),
                )
                loss = loss + float(args.proposal_source5_soft_owner_weight) * proposal_source5_soft_owner
        if (
            "proposal_logits" in outputs
            and "proposal_source_ids" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_source5_mass_preserve_weight", 0.0)) > 0.0
        ):
            logits = outputs["proposal_logits"].float()
            source_ids = outputs["proposal_source_ids"].to(device=logits.device, dtype=torch.long)
            if source_ids.shape == logits.shape:
                with torch.no_grad():
                    q = prop_quad_iou.detach().float().to(device=logits.device).clamp(0.0, 1.0)
                    source5 = source_ids == 5
                    non5 = ~source5
                    has5 = source5.any(dim=-1)
                    has_non5 = non5.any(dim=-1)
                    best5 = q.masked_fill(~source5, -1.0).max(dim=-1).values
                    best_non5 = q.masked_fill(~non5, -1.0).max(dim=-1).values
                    tau = max(float(args.proposal_source5_mass_preserve_tau), 1e-4)
                    target_mass = torch.sigmoid((best5 - best_non5) / tau)
                    lo = float(args.proposal_source5_mass_preserve_min)
                    hi = float(args.proposal_source5_mass_preserve_max)
                    if hi < lo:
                        lo, hi = hi, lo
                    target_mass = target_mass.clamp(lo, hi)
                    mass_valid = has5 & has_non5
                sel_temp = max(float(getattr(args, "proposal_selection_temperature", 1.0)), 1e-4)
                selection_prob = F.softmax(logits / sel_temp, dim=-1)
                source5_mass = (selection_prob * (source_ids == 5).float()).sum(dim=-1).clamp(1e-5, 1.0 - 1e-5)
                mass_loss = -(
                    target_mass * source5_mass.log()
                    + (1.0 - target_mass) * (1.0 - source5_mass).log()
                )
                mass_loss = torch.where(mass_valid, mass_loss, torch.zeros_like(mass_loss))
                proposal_source5_mass_preserve = weighted_mean(
                    mass_loss,
                    frame_w * mass_valid.to(device=frame_w.device, dtype=frame_w.dtype),
                )
                loss = loss + (
                    float(args.proposal_source5_mass_preserve_weight)
                    * proposal_source5_mass_preserve
                )
        if (
            "proposal_logits" in outputs
            and "proposal_source_ids" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_source7_multimatch_weight", 0.0)) > 0.0
        ):
            # Same assignment as source5 multimatch, but applied to the
            # topology-template proposal source. H498 found source7 oracle
            # candidates on many frames, but the selector almost never chose
            # them.
            logits = outputs["proposal_logits"].float()
            source_ids = outputs["proposal_source_ids"].to(device=logits.device, dtype=torch.long)
            with torch.no_grad():
                q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
                source7 = source_ids == 7
                q_source7 = q.masked_fill(~source7, -1.0)
                topk = min(max(int(args.proposal_source7_multimatch_topk), 1), int(q.shape[-1]))
                top_idx = q_source7.topk(topk, dim=-1).indices
                target = torch.zeros_like(q)
                top_q = q.gather(-1, top_idx)
                top_valid = source7.gather(-1, top_idx) & (top_q > 0.0)
                target.scatter_(-1, top_idx, torch.where(top_valid, top_q, torch.zeros_like(top_q)))
                valid = source7
            pred = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
            focal_weight = target + float(args.proposal_varifocal_alpha) * (
                pred - target
            ).abs().pow(float(args.proposal_varifocal_gamma))
            mm_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * focal_weight
            mm_loss = mm_loss * valid.float()
            mm_count = valid.float().sum(dim=-1)
            mm_frame = mm_loss.sum(dim=-1) / mm_count.clamp_min(1.0)
            mm_frame = torch.where(mm_count > 0, mm_frame, torch.zeros_like(mm_frame))
            proposal_source7_multimatch = weighted_mean(mm_frame, frame_w)
            loss = loss + float(args.proposal_source7_multimatch_weight) * proposal_source7_multimatch
        if (
            "proposal_logits" in outputs
            and "proposal_source_ids" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_source_balanced_varifocal_weight", 0.0)) > 0.0
        ):
            # Source-balanced assignment: keep one localization-quality
            # positive per proposal source, plus the global oracle candidate.
            # This adapts the ATSS/PAA idea that assignment should depend on
            # candidate quality distribution, not on current selector score.
            logits = outputs["proposal_logits"].float()
            source_ids = outputs["proposal_source_ids"].to(device=logits.device, dtype=torch.long)
            q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
            target = torch.zeros_like(q)
            for source_id in range(8):
                source_mask = source_ids == int(source_id)
                source_present = source_mask.any(dim=-1, keepdim=True)
                if bool(source_present.any().detach().cpu()):
                    q_source = q.masked_fill(~source_mask, -1.0)
                    source_idx = q_source.argmax(dim=-1, keepdim=True)
                    source_target = q.gather(-1, source_idx) * source_present.float()
                    target.scatter_(-1, source_idx, torch.maximum(target.gather(-1, source_idx), source_target))
            oracle_idx = q.argmax(dim=-1, keepdim=True)
            target.scatter_(-1, oracle_idx, q.gather(-1, oracle_idx))

            pred = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
            focal_weight = target + float(args.proposal_varifocal_alpha) * (
                pred - target
            ).abs().pow(float(args.proposal_varifocal_gamma))
            src_balanced_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * focal_weight
            proposal_source_balanced_varifocal = weighted_mean(src_balanced_loss.mean(dim=-1), frame_w)
            loss = loss + float(args.proposal_source_balanced_varifocal_weight) * proposal_source_balanced_varifocal
        if (
            "proposal_logits" in outputs
            and "proposal_source_ids" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_source_balanced_task_aligned_weight", 0.0)) > 0.0
        ):
            # TOOD's TaskAlignedAssigner chooses top-k candidates by
            # score^alpha * IoU^beta.  Apply that selection inside each source
            # family so source5/source7 learn quality ranking before their
            # logits compete with source0 globally.
            logits = outputs["proposal_logits"].float()
            source_ids = outputs["proposal_source_ids"].to(device=logits.device, dtype=torch.long)
            with torch.no_grad():
                q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
                score = logits.detach().sigmoid().clamp(1e-4, 1.0 - 1e-4)
                alpha = max(float(args.proposal_task_aligned_alpha), 0.0)
                beta = max(float(args.proposal_task_aligned_beta), 0.0)
                align = score.pow(alpha) * q.pow(beta)
                target = torch.zeros_like(q)
                per_source_topk = min(
                    max(int(args.proposal_source_balanced_task_aligned_topk), 1),
                    int(align.shape[-1]),
                )
                for source_id in range(8):
                    source_mask = source_ids == int(source_id)
                    if not bool(source_mask.any().detach().cpu()):
                        continue
                    source_align = align.masked_fill(~source_mask, -1.0)
                    topk_idx = source_align.topk(per_source_topk, dim=-1).indices
                    topk_valid = source_mask.gather(-1, topk_idx)
                    topk_align = align.gather(-1, topk_idx).masked_fill(~topk_valid, 0.0)
                    topk_q = q.gather(-1, topk_idx).masked_fill(~topk_valid, 0.0)
                    norm = topk_align.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
                    qmax = topk_q.max(dim=-1, keepdim=True).values
                    topk_target = (topk_align / norm * qmax).clamp(0.0, 1.0)
                    target.scatter_(-1, topk_idx, torch.maximum(target.gather(-1, topk_idx), topk_target))
                oracle_idx = q.argmax(dim=-1, keepdim=True)
                target.scatter_(-1, oracle_idx, q.gather(-1, oracle_idx))
            pred = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
            focal_weight = target + float(args.proposal_varifocal_alpha) * (
                pred - target
            ).abs().pow(float(args.proposal_varifocal_gamma))
            src_tal_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * focal_weight
            proposal_source_balanced_task_aligned = weighted_mean(src_tal_loss.mean(dim=-1), frame_w)
            loss = loss + (
                float(args.proposal_source_balanced_task_aligned_weight)
                * proposal_source_balanced_task_aligned
            )
        if (
            "proposal_logits" in outputs
            and "proposal_source_ids" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_source_group_ce_weight", 0.0)) > 0.0
        ):
            # Group-level handoff: first choose which candidate source should
            # own the frame, then let within-source logits pick the candidate.
            # This directly targets the H315/H316 failure where source4 was
            # selected far more often than it was the oracle source.
            logits = outputs["proposal_logits"].float()
            source_ids = outputs["proposal_source_ids"].to(device=logits.device, dtype=torch.long)
            q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
            oracle_idx = q.argmax(dim=-1, keepdim=True)
            oracle_source = source_ids.gather(-1, oracle_idx).squeeze(-1).clamp(0, 7)
            group_scores = []
            neg = torch.finfo(logits.dtype).min
            for source_id in range(8):
                source_mask = source_ids == int(source_id)
                source_logits = logits.masked_fill(~source_mask, neg)
                group_scores.append(source_logits.max(dim=-1).values)
            group_scores = torch.stack(group_scores, dim=-1)
            group_ce = F.cross_entropy(
                group_scores.reshape(-1, group_scores.shape[-1]),
                oracle_source.reshape(-1),
                reduction="none",
            ).reshape_as(frame_w)
            proposal_source_group_ce = weighted_mean(group_ce, frame_w)
            loss = loss + float(args.proposal_source_group_ce_weight) * proposal_source_group_ce
        if (
            "proposal_logits" in outputs
            and "proposal_source_ids" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_source_local_rank_weight", 0.0)) > 0.0
        ):
            logits = outputs["proposal_logits"].float()
            source_ids = outputs["proposal_source_ids"].to(device=logits.device, dtype=torch.long)
            q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
            local_num = logits.new_tensor(0.0)
            local_den = logits.new_tensor(0.0)
            for source_id in range(8):
                source_mask = source_ids == int(source_id)
                source_present = source_mask.any(dim=-1)
                if not bool(source_present.any().detach().cpu()):
                    continue
                q_source = q.masked_fill(~source_mask, -1.0)
                source_target = q_source.argmax(dim=-1)
                source_logits = logits.masked_fill(~source_mask, -50.0)
                source_ce = F.cross_entropy(
                    source_logits.reshape(-1, source_logits.shape[-1]),
                    source_target.reshape(-1),
                    reduction="none",
                ).reshape_as(frame_w)
                source_w = frame_w * source_present.float()
                local_num = local_num + (source_ce * source_w).sum()
                local_den = local_den + source_w.sum()
            proposal_source_local_rank = local_num / local_den.clamp_min(1.0)
            loss = loss + float(args.proposal_source_local_rank_weight) * proposal_source_local_rank
        if (
            "proposal_handoff_source_router_logits" in outputs
            and "proposal_source_ids" in outputs
            and prop_quad_iou is not None
        ):
            router_logits = outputs["proposal_handoff_source_router_logits"].float()
            source_ids = outputs["proposal_source_ids"].to(device=router_logits.device, dtype=torch.long)
            q = prop_quad_iou.detach().float().clamp(0.0, 1.0)
            if source_ids.shape == q.shape and router_logits.shape[:2] == q.shape[:2]:
                oracle_idx = q.argmax(dim=-1, keepdim=True)
                oracle_source = source_ids.gather(-1, oracle_idx).squeeze(-1).clamp(0, 7)
                router_top1 = router_logits.detach().argmax(dim=-1)
                proposal_handoff_source_router_top1_match = weighted_mean(
                    (router_top1 == oracle_source).float(),
                    frame_w,
                )
                if float(getattr(args, "proposal_handoff_source_router_ce_weight", 0.0)) > 0.0:
                    router_ce = F.cross_entropy(
                        router_logits.reshape(-1, router_logits.shape[-1]),
                        oracle_source.reshape(-1),
                        reduction="none",
                    ).reshape_as(frame_w)
                    proposal_handoff_source_router_ce = weighted_mean(router_ce, frame_w)
                    loss = (
                        loss
                        + float(args.proposal_handoff_source_router_ce_weight)
                        * proposal_handoff_source_router_ce
                    )
            cand_quality_weight = float(
                getattr(args, "proposal_handoff_source_router_candidate_quality_weight", 0.0)
            )
            if (
                cand_quality_weight > 0.0
                and "proposal_handoff_source_router_candidate_logits" in outputs
            ):
                cand_logits = outputs["proposal_handoff_source_router_candidate_logits"].float()
                if cand_logits.shape == q.shape:
                    quality_target = proposal_quality_target_from_iou(q)
                    cand_pred = cand_logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
                    focal_weight = quality_target + float(args.proposal_varifocal_alpha) * (
                        cand_pred - quality_target
                    ).abs().pow(float(args.proposal_varifocal_gamma))
                    cand_vfl = F.binary_cross_entropy_with_logits(
                        cand_logits,
                        quality_target,
                        reduction="none",
                    ) * focal_weight
                    proposal_handoff_source_router_candidate_quality = weighted_mean(
                        cand_vfl.mean(dim=-1),
                        frame_w,
                    )
                    loss = loss + cand_quality_weight * proposal_handoff_source_router_candidate_quality
        if (
            "proposal_logits" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_quality_weight", 0.0)) > 0.0
        ):
            quality_target = proposal_quality_target_from_iou(prop_quad_iou)
            quality_pred = torch.sigmoid(outputs["proposal_logits"].float())
            quality_loss = F.smooth_l1_loss(quality_pred, quality_target.float(), reduction="none")
            proposal_quality = weighted_mean(quality_loss.mean(dim=-1), frame_w)
            loss = loss + float(args.proposal_quality_weight) * proposal_quality
        if (
            "proposal_logits" in outputs
            and prop_quad_iou is not None
            and float(getattr(args, "proposal_varifocal_weight", 0.0)) > 0.0
        ):
            logits = outputs["proposal_logits"].float()
            quality_target = proposal_quality_target_from_iou(prop_quad_iou)
            quality_pred = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
            focal_weight = quality_target + float(args.proposal_varifocal_alpha) * (
                quality_pred - quality_target
            ).abs().pow(float(args.proposal_varifocal_gamma))
            vfl = F.binary_cross_entropy_with_logits(logits, quality_target, reduction="none") * focal_weight
            proposal_varifocal = weighted_mean(vfl.mean(dim=-1), frame_w)
            loss = loss + float(args.proposal_varifocal_weight) * proposal_varifocal
    context_l1 = pred_xyxy.new_tensor(0.0)
    context_giou = pred_xyxy.new_tensor(0.0)
    context_iou = pred_xyxy.new_tensor(0.0)
    if "context_cxcywh" in outputs and float(args.context_loss_weight) > 0.0:
        context_gt_cxcywh = batch["context_boxes_cxcywh"].to(pred_xyxy.device)
        context_gt_xyxy = batch["context_boxes_xyxy"].to(pred_xyxy.device)
        context_l1 = weighted_mean(
            F.smooth_l1_loss(outputs["context_cxcywh"], context_gt_cxcywh, reduction="none").sum(dim=-1),
            frame_w,
        )
        context_giou = weighted_mean(giou_loss_xyxy(outputs["context_xyxy"], context_gt_xyxy), frame_w)
        loss = loss + float(args.context_loss_weight) * (context_l1 + context_giou)
        with torch.no_grad():
            context_iou = box_iou_xyxy(outputs["context_xyxy"], context_gt_xyxy).mean()
    condition_prior_loss = pred_xyxy.new_tensor(0.0)
    condition_prior_box_iou = pred_xyxy.new_tensor(0.0)
    condition_prior_quad_iou = pred_xyxy.new_tensor(0.0)
    if "condition_prior_quad" in outputs and float(getattr(args, "condition_prior_loss_weight", 0.0)) > 0.0:
        prior_quad = outputs["condition_prior_quad"]
        prior_xyxy = outputs["condition_prior_xyxy"]
        prior_cxcywh = outputs["condition_prior_cxcywh"]
        gt_quad = batch["quad_xy"].to(prior_quad.device)
        prior_l1 = F.smooth_l1_loss(prior_quad, gt_quad, reduction="none").sum(dim=(-1, -2))
        prior_box_l1 = F.smooth_l1_loss(prior_cxcywh, gt_cxcywh.to(prior_cxcywh.device), reduction="none").sum(dim=-1)
        prior_giou = giou_loss_xyxy(prior_xyxy, gt_xyxy.to(prior_xyxy.device))
        condition_prior_loss = weighted_mean(prior_l1 + 0.35 * prior_box_l1 + 0.35 * prior_giou, frame_w)
        loss = loss + float(args.condition_prior_loss_weight) * condition_prior_loss
        with torch.no_grad():
            condition_prior_box_iou = box_iou_xyxy(prior_xyxy, gt_xyxy.to(prior_xyxy.device)).mean()
            prior_quad_grid = min(max(int(args.quad_iou_grid_size), 8), 32)
            condition_prior_quad_iou = quad_iou_approx(prior_quad, gt_quad, grid_size=prior_quad_grid).mean()
    trajectory_query_loss = pred_xyxy.new_tensor(0.0)
    trajectory_query_box_iou = pred_xyxy.new_tensor(0.0)
    trajectory_query_quad_iou = pred_xyxy.new_tensor(0.0)
    if "trajectory_query_quad" in outputs:
        tq_quad = outputs["trajectory_query_quad"]
        tq_xyxy = outputs["trajectory_query_xyxy"]
        tq_cxcywh = outputs["trajectory_query_cxcywh"]
        gt_quad = batch["quad_xy"].to(tq_quad.device)
        gt_quad_tq = gt_quad[:, :, None, :, :].expand_as(tq_quad)
        gt_cx_tq = gt_cxcywh.to(tq_cxcywh.device)[:, :, None, :].expand_as(tq_cxcywh)
        gt_xy_tq = gt_xyxy.to(tq_xyxy.device)[:, :, None, :].expand_as(tq_xyxy)
        tq_l1 = F.smooth_l1_loss(tq_quad, gt_quad_tq, reduction="none").sum(dim=(-1, -2))
        tq_box_l1 = F.smooth_l1_loss(tq_cxcywh, gt_cx_tq, reduction="none").sum(dim=-1)
        tq_giou = giou_loss_xyxy(tq_xyxy, gt_xy_tq)
        tq_total = tq_l1 + 0.35 * tq_box_l1 + 0.35 * tq_giou
        trajectory_query_loss = weighted_mean(tq_total.min(dim=-1).values, frame_w)
        if float(getattr(args, "trajectory_query_loss_weight", 0.0)) > 0.0:
            loss = loss + float(args.trajectory_query_loss_weight) * trajectory_query_loss
        with torch.no_grad():
            trajectory_query_box_iou = box_iou_xyxy(tq_xyxy, gt_xy_tq).max(dim=-1).values.mean()
            tq_quad_grid = min(max(int(args.quad_iou_grid_size), 8), 32)
            tq_quad_iou = quad_iou_approx(
                tq_quad.reshape(-1, 4, 2),
                gt_quad_tq.reshape(-1, 4, 2),
                grid_size=tq_quad_grid,
            ).reshape(tq_quad.shape[:-2])
            trajectory_query_quad_iou = tq_quad_iou.max(dim=-1).values.mean()
    point_l1 = pred_xyxy.new_tensor(0.0)
    point_shape = pred_xyxy.new_tensor(0.0)
    point_temporal = pred_xyxy.new_tensor(0.0)
    quad_axis = pred_xyxy.new_tensor(0.0)
    quad_center = pred_xyxy.new_tensor(0.0)
    quad_log_area = pred_xyxy.new_tensor(0.0)
    quad_over_area = pred_xyxy.new_tensor(0.0)
    quad_soft_mask = pred_xyxy.new_tensor(0.0)
    quad_soft_mask_iou = pred_xyxy.new_tensor(0.0)
    quad_topology = pred_xyxy.new_tensor(0.0)
    quad_kfiou = pred_xyxy.new_tensor(0.0)
    quad_iou_metric = pred_xyxy.new_tensor(0.0)
    if "quad_xy" in outputs:
        pred_quad = outputs.get("quad_xy_eval", outputs["quad_xy"])
        gt_quad = batch["quad_xy"].to(pred_quad.device)
        if float(args.point_loss_weight) > 0.0:
            point_l1 = weighted_mean(
                F.smooth_l1_loss(pred_quad, gt_quad, reduction="none").sum(dim=(-1, -2)),
                frame_w,
            )
            loss = loss + float(args.point_loss_weight) * point_l1
        if float(args.point_shape_loss_weight) > 0.0:
            point_shape = quad_pairwise_shape_loss(pred_quad, gt_quad, frame_w)
            loss = loss + float(args.point_shape_loss_weight) * point_shape
        if float(args.point_temporal_loss_weight) > 0.0:
            pred_delta = pred_quad[:, 1:] - pred_quad[:, :-1]
            gt_delta = gt_quad[:, 1:] - gt_quad[:, :-1]
            delta_w = 0.5 * (frame_w[:, 1:] + frame_w[:, :-1])
            point_temporal = weighted_mean(
                F.smooth_l1_loss(pred_delta, gt_delta, reduction="none").sum(dim=(-1, -2)),
                delta_w,
            )
            loss = loss + float(args.point_temporal_loss_weight) * point_temporal
        if float(args.quad_axis_loss_weight) > 0.0:
            quad_axis = quad_axis_alignment_loss(
                pred_quad,
                gt_quad,
                frame_w,
                length_weight=float(args.quad_axis_length_weight),
            )
            loss = loss + float(args.quad_axis_loss_weight) * quad_axis
        if float(getattr(args, "quad_center_loss_weight", 0.0)) > 0.0:
            pred_center = pred_quad.mean(dim=-2)
            gt_center = gt_quad.mean(dim=-2)
            quad_center = weighted_mean(
                F.smooth_l1_loss(pred_center, gt_center, reduction="none").sum(dim=-1),
                frame_w,
            )
            loss = loss + float(args.quad_center_loss_weight) * quad_center
        if (
            float(getattr(args, "quad_log_area_loss_weight", 0.0)) > 0.0
            or float(getattr(args, "quad_over_area_loss_weight", 0.0)) > 0.0
        ):
            pred_poly_area = quad_polygon_area(pred_quad)
            gt_poly_area = quad_polygon_area(gt_quad).detach()
            log_area_ratio = torch.log(pred_poly_area) - torch.log(gt_poly_area)
            if float(getattr(args, "quad_log_area_loss_weight", 0.0)) > 0.0:
                quad_log_area = weighted_mean(
                    F.smooth_l1_loss(log_area_ratio, torch.zeros_like(log_area_ratio), reduction="none"),
                    frame_w,
                )
                loss = loss + float(args.quad_log_area_loss_weight) * quad_log_area
            if float(getattr(args, "quad_over_area_loss_weight", 0.0)) > 0.0:
                over_ratio = log_area_ratio.clamp_min(0.0)
                quad_over_area = weighted_mean(
                    F.smooth_l1_loss(over_ratio, torch.zeros_like(over_ratio), reduction="none"),
                    frame_w,
                )
                loss = loss + float(args.quad_over_area_loss_weight) * quad_over_area
        if float(getattr(args, "quad_soft_mask_loss_weight", 0.0)) > 0.0:
            quad_soft_mask, quad_soft_mask_iou = quad_soft_mask_iou_loss(
                pred_quad,
                gt_quad,
                frame_w,
                grid_size=int(getattr(args, "quad_soft_mask_grid_size", 64)),
                edge_sharpness=float(getattr(args, "quad_soft_mask_edge_sharpness", 128.0)),
            )
            loss = loss + float(args.quad_soft_mask_loss_weight) * quad_soft_mask
        if float(getattr(args, "quad_topology_loss_weight", 0.0)) > 0.0:
            quad_topology = quad_topology_loss(
                pred_quad,
                gt_quad,
                frame_w,
                margin=float(getattr(args, "quad_topology_margin", 5e-4)),
            )
            loss = loss + float(args.quad_topology_loss_weight) * quad_topology
        if float(getattr(args, "quad_kfiou_loss_weight", 0.0)) > 0.0:
            quad_kfiou = quad_kfiou_loss(
                pred_quad,
                gt_quad,
                frame_w,
                beta=float(getattr(args, "quad_kfiou_beta", 1.0 / 9.0)),
                fun=str(getattr(args, "quad_kfiou_fun", "none")),
            )
            loss = loss + float(args.quad_kfiou_loss_weight) * quad_kfiou
        with torch.no_grad():
            train_quad_grid = min(max(int(args.quad_iou_grid_size), 8), 32)
            quad_iou_metric = quad_iou_approx(pred_quad, gt_quad, grid_size=train_quad_grid).mean()
    with torch.no_grad():
        iou = box_iou_xyxy(pred_xyxy, gt_xyxy)
    metrics = {
        "loss": float(loss.detach().cpu()),
        "l1": float(l1.detach().cpu()),
        "giou": float(giou.detach().cpu()),
        "ciou": float(ciou.detach().cpu()),
        "center_ce": float(ce.detach().cpu()),
        "smooth": float(smooth.detach().cpu()),
        "box_mask": float(box_mask_loss.detach().cpu()),
        "edge_ce": float(edge_ce.detach().cpu()),
        "dfl": float(dfl.detach().cpu()),
        "corner_heatmap": float(corner_heatmap.detach().cpu()),
        "support_heatmap": float(support_heatmap.detach().cpu()),
        "area": float(area_loss.detach().cpu()),
        "size": float(size_loss.detach().cpu()),
        "aspect": float(aspect_loss.detach().cpu()),
        "nwd": float(nwd_loss.detach().cpu()),
        "clip_prior_l1": float(clip_prior_l1.detach().cpu()),
        "clip_prior_giou": float(clip_prior_giou.detach().cpu()),
        "clip_prior_iou": float(clip_prior_iou.detach().cpu()),
        "box_iou": float(iou.mean().detach().cpu()),
        "hard_factor": float(hard_factor.mean().detach().cpu()),
        "small_box_factor": float(small_box_factor.mean().detach().cpu()),
        "anchor_l1": float(anchor_l1.detach().cpu()),
        "anchor_giou": float(anchor_giou.detach().cpu()),
        "anchor_iou": float(anchor_iou.detach().cpu()),
        "coarse_l1": float(coarse_l1.detach().cpu()),
        "coarse_giou": float(coarse_giou.detach().cpu()),
        "coarse_iou": float(coarse_iou.detach().cpu()),
        "decoder_aux_l1": float(decoder_aux_l1.detach().cpu()),
        "decoder_aux_giou": float(decoder_aux_giou.detach().cpu()),
        "decoder_aux_iou": float(decoder_aux_iou.detach().cpu()),
        "dn_refiner_loss": float(dn_refiner_loss.detach().cpu()),
        "dn_refiner_l1": float(dn_refiner_l1.detach().cpu()),
        "dn_refiner_giou": float(dn_refiner_giou.detach().cpu()),
        "dn_refiner_point": float(dn_refiner_point.detach().cpu()),
        "dn_refiner_axis": float(dn_refiner_axis.detach().cpu()),
        "dn_refiner_iou": float(dn_refiner_iou.detach().cpu()),
        "dn_refiner_quad_iou": float(dn_refiner_quad_iou.detach().cpu()),
        "proposal_oracle_refiner_loss": float(proposal_oracle_refiner_loss.detach().cpu()),
        "proposal_oracle_refiner_l1": float(proposal_oracle_refiner_l1.detach().cpu()),
        "proposal_oracle_refiner_giou": float(proposal_oracle_refiner_giou.detach().cpu()),
        "proposal_oracle_refiner_point": float(proposal_oracle_refiner_point.detach().cpu()),
        "proposal_oracle_refiner_axis": float(proposal_oracle_refiner_axis.detach().cpu()),
        "proposal_oracle_refiner_iou": float(proposal_oracle_refiner_iou.detach().cpu()),
        "proposal_oracle_refiner_quad_iou": float(proposal_oracle_refiner_quad_iou.detach().cpu()),
        "proposal_oracle_refiner_start_quad_iou": float(
            proposal_oracle_refiner_start_quad_iou.detach().cpu()
        ),
        "local_grid_corner_ce": float(local_grid_corner_ce.detach().cpu()),
        "candidate_local_grid_corner_ce": float(candidate_local_grid_corner_ce.detach().cpu()),
        "candidate_mask_bce": float(candidate_mask_bce.detach().cpu()),
        "candidate_mask_quality": float(candidate_mask_quality.detach().cpu()),
        "candidate_mask_top1_quad_iou": float(candidate_mask_top1_quad_iou.detach().cpu()),
        "candidate_mask_iou_corr": float(candidate_mask_iou_corr.detach().cpu()),
        "proposal_handoff_mask_bce": float(proposal_handoff_mask_bce.detach().cpu()),
        "proposal_handoff_mask_quality": float(
            proposal_handoff_mask_quality.detach().cpu()
        ),
        "proposal_handoff_mask_boundary": float(
            proposal_handoff_mask_boundary.detach().cpu()
        ),
        "proposal_handoff_mask_top1_quad_iou": float(
            proposal_handoff_mask_top1_quad_iou.detach().cpu()
        ),
        "proposal_handoff_mask_iou_corr": float(
            proposal_handoff_mask_iou_corr.detach().cpu()
        ),
        "proposal_loss": float(proposal_loss.detach().cpu()),
        "proposal_quad_loss": float(proposal_quad_loss.detach().cpu()),
        "proposal_rank": float(proposal_rank.detach().cpu()),
        "proposal_quality": float(proposal_quality.detach().cpu()),
        "proposal_varifocal": float(proposal_varifocal.detach().cpu()),
        "proposal_geom_quality": float(proposal_geom_quality.detach().cpu()),
        "proposal_center_area": float(proposal_center_area.detach().cpu()),
        "proposal_source_balanced_varifocal": float(
            proposal_source_balanced_varifocal.detach().cpu()
        ),
        "proposal_source_balanced_task_aligned": float(
            proposal_source_balanced_task_aligned.detach().cpu()
        ),
        "proposal_source_group_ce": float(proposal_source_group_ce.detach().cpu()),
        "proposal_source_local_rank": float(proposal_source_local_rank.detach().cpu()),
        "proposal_handoff_source_router_ce": float(
            proposal_handoff_source_router_ce.detach().cpu()
        ),
        "proposal_handoff_source_router_top1_match": float(
            proposal_handoff_source_router_top1_match.detach().cpu()
        ),
        "proposal_handoff_source_router_candidate_quality": float(
            proposal_handoff_source_router_candidate_quality.detach().cpu()
        ),
        "proposal_handoff_oracle_distill": float(
            proposal_handoff_oracle_distill.detach().cpu()
        ),
        "proposal_handoff_oracle_distill_point": float(
            proposal_handoff_oracle_distill_point.detach().cpu()
        ),
        "proposal_handoff_oracle_distill_axis": float(
            proposal_handoff_oracle_distill_axis.detach().cpu()
        ),
        "proposal_handoff_source5_readout_varifocal": float(
            proposal_handoff_source5_readout_varifocal.detach().cpu()
        ),
        "proposal_handoff_source5_readout_soft_rank": float(
            proposal_handoff_source5_readout_soft_rank.detach().cpu()
        ),
        "proposal_handoff_source5_readout_top1_quad_iou": float(
            proposal_handoff_source5_readout_top1_quad_iou.detach().cpu()
        ),
        "proposal_pcl_cluster_ce": float(proposal_pcl_cluster_ce.detach().cpu()),
        "proposal_oicr_refine_ce": float(proposal_oicr_refine_ce.detach().cpu()),
        "proposal_source0_multimatch": float(proposal_source0_multimatch.detach().cpu()),
        "proposal_source5_multimatch": float(proposal_source5_multimatch.detach().cpu()),
        "proposal_source4_demote": float(proposal_source4_demote.detach().cpu()),
        "proposal_source5_soft_owner": float(proposal_source5_soft_owner.detach().cpu()),
        "proposal_source5_mass_preserve": float(proposal_source5_mass_preserve.detach().cpu()),
        "proposal_source7_multimatch": float(proposal_source7_multimatch.detach().cpu()),
        "proposal_pairwise_rank": float(proposal_pairwise_rank.detach().cpu()),
        "proposal_hardneg_rank": float(proposal_hardneg_rank.detach().cpu()),
        "proposal_tool_hardneg": float(proposal_tool_hardneg.detach().cpu()),
        "proposal_soft_rank": float(proposal_soft_rank.detach().cpu()),
        "proposal_owner_band": float(proposal_owner_band.detach().cpu()),
        "proposal_neuralsort_top1": float(proposal_neuralsort_top1.detach().cpu()),
        "proposal_ranksort": float(proposal_ranksort.detach().cpu()),
        "proposal_task_aligned": float(proposal_task_aligned.detach().cpu()),
        "proposal_set_refiner_rank": float(proposal_set_refiner_rank.detach().cpu()),
        "proposal_set_refiner_soft_rank": float(proposal_set_refiner_soft_rank.detach().cpu()),
        "proposal_set_refiner_top1_quad_iou": float(
            proposal_set_refiner_top1_quad_iou.detach().cpu()
        ),
        "proposal_set_refiner_top3_oracle_quad_iou": float(
            proposal_set_refiner_top3_oracle_quad_iou.detach().cpu()
        ),
        "proposal_set_refiner_top3_contains_oracle": float(
            proposal_set_refiner_top3_contains_oracle.detach().cpu()
        ),
        "proposal_handoff_source5_refiner_loss": float(
            proposal_handoff_source5_refiner_loss.detach().cpu()
        ),
        "proposal_handoff_source5_refiner_point": float(
            proposal_handoff_source5_refiner_point.detach().cpu()
        ),
        "proposal_handoff_source5_refiner_axis": float(
            proposal_handoff_source5_refiner_axis.detach().cpu()
        ),
        "proposal_handoff_source5_refiner_quad_iou": float(
            proposal_handoff_source5_refiner_quad_iou.detach().cpu()
        ),
        "proposal_handoff_source5_refiner_quality": float(
            proposal_handoff_source5_refiner_quality.detach().cpu()
        ),
        "proposal_handoff_source5_refiner_rank": float(
            proposal_handoff_source5_refiner_rank.detach().cpu()
        ),
        "proposal_handoff_source5_refiner_rank_top1_quad_iou": float(
            proposal_handoff_source5_refiner_rank_top1_quad_iou.detach().cpu()
        ),
        "proposal_handoff_source5_refiner_rank_top1_match": float(
            proposal_handoff_source5_refiner_rank_top1_match.detach().cpu()
        ),
        "proposal_handoff_topk_query_refiner_loss": float(
            proposal_handoff_topk_query_refiner_loss.detach().cpu()
        ),
        "proposal_handoff_topk_query_refiner_point": float(
            proposal_handoff_topk_query_refiner_point.detach().cpu()
        ),
        "proposal_handoff_topk_query_refiner_axis": float(
            proposal_handoff_topk_query_refiner_axis.detach().cpu()
        ),
        "proposal_handoff_topk_query_refiner_quad_iou": float(
            proposal_handoff_topk_query_refiner_quad_iou.detach().cpu()
        ),
        "proposal_handoff_topk_query_refiner_base_quad_iou": float(
            proposal_handoff_topk_query_refiner_base_quad_iou.detach().cpu()
        ),
        "proposal_handoff_topk_query_refiner_quality": float(
            proposal_handoff_topk_query_refiner_quality.detach().cpu()
        ),
        "proposal_handoff_topk_query_refiner_rank": float(
            proposal_handoff_topk_query_refiner_rank.detach().cpu()
        ),
        "proposal_handoff_topk_query_refiner_soft_rank": float(
            proposal_handoff_topk_query_refiner_soft_rank.detach().cpu()
        ),
        "proposal_handoff_topk_query_refiner_rank_top1_quad_iou": float(
            proposal_handoff_topk_query_refiner_rank_top1_quad_iou.detach().cpu()
        ),
        "proposal_handoff_topk_query_refiner_rank_top1_match": float(
            proposal_handoff_topk_query_refiner_rank_top1_match.detach().cpu()
        ),
        "proposal_c2f_assign": float(proposal_c2f_assign.detach().cpu()),
        "proposal_pcl_cluster": float(proposal_pcl_cluster.detach().cpu()),
        "proposal_frame_anchor": float(proposal_frame_anchor.detach().cpu()),
        "proposal_frame_anchor_top1_oracle_iou": float(
            proposal_frame_anchor_top1_oracle_iou.detach().cpu()
        ),
        "proposal_frame_anchor_entropy": float(proposal_frame_anchor_entropy.detach().cpu()),
        "proposal_logit_top1_quad_iou": float(proposal_logit_top1_quad_iou.detach().cpu()),
        "proposal_logit_top3_oracle_quad_iou": float(proposal_logit_top3_oracle_quad_iou.detach().cpu()),
        "proposal_logit_top3_contains_oracle": float(proposal_logit_top3_contains_oracle.detach().cpu()),
        "proposal_logit_iou_corr": float(proposal_logit_iou_corr.detach().cpu()),
        "proposal_source0_oracle_quad_iou": float(proposal_source0_oracle_quad_iou.detach().cpu()),
        "proposal_source0_logit_top1_quad_iou": float(
            proposal_source0_logit_top1_quad_iou.detach().cpu()
        ),
        "proposal_source4_oracle_quad_iou": float(proposal_source4_oracle_quad_iou.detach().cpu()),
        "proposal_source4_logit_top1_quad_iou": float(
            proposal_source4_logit_top1_quad_iou.detach().cpu()
        ),
        "proposal_source5_oracle_quad_iou": float(proposal_source5_oracle_quad_iou.detach().cpu()),
        "proposal_source5_logit_top1_quad_iou": float(
            proposal_source5_logit_top1_quad_iou.detach().cpu()
        ),
        "proposal_source7_oracle_quad_iou": float(proposal_source7_oracle_quad_iou.detach().cpu()),
        "proposal_source7_logit_top1_quad_iou": float(
            proposal_source7_logit_top1_quad_iou.detach().cpu()
        ),
        "proposal_top1_source4_frac": float(proposal_top1_source4_frac.detach().cpu()),
        "proposal_oracle_source4_frac": float(proposal_oracle_source4_frac.detach().cpu()),
        "proposal_top1_source5_frac": float(proposal_top1_source5_frac.detach().cpu()),
        "proposal_oracle_source5_frac": float(proposal_oracle_source5_frac.detach().cpu()),
        "proposal_top1_source7_frac": float(proposal_top1_source7_frac.detach().cpu()),
        "proposal_oracle_source7_frac": float(proposal_oracle_source7_frac.detach().cpu()),
        "proposal_selection_source4_mass": float(proposal_selection_source4_mass.detach().cpu()),
        "proposal_selection_source5_mass": float(proposal_selection_source5_mass.detach().cpu()),
        "proposal_selection_source7_mass": float(proposal_selection_source7_mass.detach().cpu()),
        "proposal_top1_matches_oracle_source_frac": float(
            proposal_top1_matches_oracle_source_frac.detach().cpu()
        ),
        "proposal_active_tip_varifocal": float(proposal_active_tip_varifocal.detach().cpu()),
        "proposal_active_tip_top1_quad_iou": float(proposal_active_tip_top1_quad_iou.detach().cpu()),
        "proposal_active_tip_top3_oracle_quad_iou": float(
            proposal_active_tip_top3_oracle_quad_iou.detach().cpu()
        ),
        "proposal_active_tip_top3_contains_oracle": float(
            proposal_active_tip_top3_contains_oracle.detach().cpu()
        ),
        "proposal_active_tip_iou_corr": float(proposal_active_tip_iou_corr.detach().cpu()),
        "proposal_local_corr_top1_quad_iou": float(
            proposal_local_corr_top1_quad_iou.detach().cpu()
        ),
        "proposal_local_corr_top3_oracle_quad_iou": float(
            proposal_local_corr_top3_oracle_quad_iou.detach().cpu()
        ),
        "proposal_local_corr_iou_corr": float(proposal_local_corr_iou_corr.detach().cpu()),
        "proposal_oracle_iou": float(proposal_oracle_iou.detach().cpu()),
        "proposal_quad_oracle_iou": float(proposal_quad_oracle_iou.detach().cpu()),
        "stage1_full_proposal_oracle_iou": float(stage1_full_proposal_oracle_iou.detach().cpu()),
        "stage1_full_proposal_quad_oracle_iou": float(
            stage1_full_proposal_quad_oracle_iou.detach().cpu()
        ),
        "stage1_full_proposal_logit_top1_quad_iou": float(
            stage1_full_proposal_logit_top1_quad_iou.detach().cpu()
        ),
        "stage1_full_proposal_logit_top3_oracle_quad_iou": float(
            stage1_full_proposal_logit_top3_oracle_quad_iou.detach().cpu()
        ),
        "stage1_full_proposal_logit_top3_contains_oracle": float(
            stage1_full_proposal_logit_top3_contains_oracle.detach().cpu()
        ),
        "stage1_full_proposal_logit_iou_corr": float(
            stage1_full_proposal_logit_iou_corr.detach().cpu()
        ),
        "stage1_full_proposal_loss": float(stage1_full_proposal_loss.detach().cpu()),
        "stage1_full_proposal_quad_loss": float(stage1_full_proposal_quad_loss.detach().cpu()),
        "stage1_full_proposal_quality": float(stage1_full_proposal_quality.detach().cpu()),
        "stage1_full_proposal_varifocal": float(stage1_full_proposal_varifocal.detach().cpu()),
        "context_l1": float(context_l1.detach().cpu()),
        "context_giou": float(context_giou.detach().cpu()),
        "context_iou": float(context_iou.detach().cpu()),
        "condition_prior_loss": float(condition_prior_loss.detach().cpu()),
        "condition_prior_box_iou": float(condition_prior_box_iou.detach().cpu()),
        "condition_prior_quad_iou": float(condition_prior_quad_iou.detach().cpu()),
        "trajectory_query_loss": float(trajectory_query_loss.detach().cpu()),
        "trajectory_query_box_iou": float(trajectory_query_box_iou.detach().cpu()),
        "trajectory_query_quad_iou": float(trajectory_query_quad_iou.detach().cpu()),
        "point_l1": float(point_l1.detach().cpu()),
        "point_shape": float(point_shape.detach().cpu()),
        "point_temporal": float(point_temporal.detach().cpu()),
        "quad_axis": float(quad_axis.detach().cpu()),
        "quad_center": float(quad_center.detach().cpu()),
        "quad_log_area": float(quad_log_area.detach().cpu()),
        "quad_over_area": float(quad_over_area.detach().cpu()),
        "quad_soft_mask": float(quad_soft_mask.detach().cpu()),
        "quad_soft_mask_iou": float(quad_soft_mask_iou.detach().cpu()),
        "quad_topology": float(quad_topology.detach().cpu()),
        "quad_kfiou": float(quad_kfiou.detach().cpu()),
        "quad_iou": float(quad_iou_metric.detach().cpu()),
    }
    for extra_name in (
        "eval_tubelet_dp_switch_rate",
        "eval_tubelet_dp_adj_aabb_iou",
        "eval_tubelet_dp_path_score",
        "eval_token_corner_track_snap_dist",
        "eval_token_corner_track_step",
        "eval_token_corner_track_seed_frame",
        "eval_candidate_memory_similarity",
        "eval_candidate_memory_seed_frame",
    ):
        if extra_name in outputs and torch.is_tensor(outputs[extra_name]):
            metrics[extra_name] = float(outputs[extra_name].detach().float().mean().cpu())
    return loss, metrics


@torch.no_grad()
def apply_eval_proposal_source_bias(
    outputs: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    """Recompute eval selection after a source-specific logit calibration.

    This is intentionally diagnostic-only.  It lets H324 test whether source4
    over-confidence is a direct cause of poor selected quads without retraining
    or changing the proposal bank.
    """
    source4_bias = float(getattr(args, "proposal_eval_source4_logit_bias", 0.0))
    if source4_bias == 0.0:
        return outputs
    required = {"proposal_logits", "proposal_source_ids", "proposal_quads"}
    if not required.issubset(outputs):
        return outputs

    logits = outputs["proposal_logits"].float()
    source_ids = outputs["proposal_source_ids"].to(device=logits.device, dtype=torch.long)
    if source_ids.shape != logits.shape:
        return outputs

    adjusted_logits = logits + source4_bias * (source_ids == 4).float()
    quads = outputs["proposal_quads"]
    k_select = int(adjusted_logits.shape[-1])

    mode = str(getattr(args, "proposal_selection_mode", "train_soft_eval_hard"))
    use_hard = mode in {"hard", "train_soft_eval_hard"}
    if use_hard:
        best = adjusted_logits.argmax(dim=-1)
        weights = F.one_hot(best, num_classes=k_select).to(dtype=quads.dtype)
    else:
        temperature = max(float(getattr(args, "proposal_selection_temperature", 1.0)), 1e-6)
        select_logits = adjusted_logits / temperature
        topk = min(int(getattr(args, "proposal_selection_topk", 0)), k_select)
        if topk > 0 and topk < k_select:
            topk_idx = select_logits.topk(topk, dim=-1).indices
            masked_logits = torch.full_like(select_logits, torch.finfo(select_logits.dtype).min)
            select_logits = masked_logits.scatter(-1, topk_idx, select_logits.gather(-1, topk_idx))
        weights = F.softmax(select_logits, dim=-1).to(dtype=quads.dtype)

    quad = (quads * weights[..., None, None]).sum(dim=2).clamp(0.0, 1.0)
    box_xyxy = quad_to_xyxy(quad, min_size=0.02)
    updated = dict(outputs)
    updated["proposal_logits"] = adjusted_logits.to(dtype=outputs["proposal_logits"].dtype)
    updated["proposal_selection_weights"] = weights
    updated["quad_xy"] = quad
    updated["boxes_xyxy"] = box_xyxy
    updated["boxes_cxcywh"] = xyxy_to_cxcywh_torch(box_xyxy)
    updated["coarse_xyxy"] = box_xyxy
    updated["coarse_cxcywh"] = updated["boxes_cxcywh"]
    return updated


@torch.no_grad()
def apply_eval_center_identity_shift(
    outputs: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    """Eval-only H558 center handoff.

    The H557 audit showed that center identity has much larger counterfactual
    headroom than area shrink.  This diagnostic keeps H484's final quad shape
    and only translates it toward a deployable proposal-bank center.
    """
    if not bool(getattr(args, "eval_center_identity_shift", False)):
        return outputs
    required = {"quad_xy", "proposal_quads", "proposal_logits", "proposal_source_ids"}
    if not required.issubset(outputs):
        return outputs

    final = outputs["quad_xy"].detach().float().clamp(0.0, 1.0)
    quads = outputs["proposal_quads"].detach().float().clamp(0.0, 1.0)
    logits = outputs["proposal_logits"].detach().float()
    sources = outputs["proposal_source_ids"].to(device=logits.device, dtype=torch.long)
    if final.ndim != 4 or quads.ndim != 5 or logits.ndim != 3 or sources.shape != logits.shape:
        return outputs

    mode = str(getattr(args, "eval_center_identity_mode", "weight_no4"))
    alpha = float(getattr(args, "eval_center_identity_alpha", 0.50))
    temp = max(float(getattr(args, "eval_center_identity_temperature", 1.00)), 1e-6)
    topk = int(getattr(args, "eval_center_identity_topk", 13))
    b, t, k = logits.shape

    if mode == "top1":
        idx = logits.argmax(dim=-1)
        target_center = quads.gather(
            2,
            idx[:, :, None, None, None].expand(-1, -1, 1, 4, 2),
        ).squeeze(2).mean(dim=-2)
    else:
        if mode.startswith("weight_"):
            weights = outputs.get("proposal_selection_weights")
            if torch.is_tensor(weights) and weights.shape == logits.shape:
                score = weights.detach().float().to(device=logits.device).clamp_min(0.0)
            else:
                score = F.softmax(logits, dim=-1)
        elif mode.startswith("logit_"):
            score = logits
        else:
            return outputs

        if mode.endswith("_no4"):
            mask = sources != 4
        elif mode.endswith("_s5"):
            mask = sources == 5
        elif mode.endswith("_s0s5"):
            mask = (sources == 0) | (sources == 5)
        else:
            mask = torch.ones_like(sources, dtype=torch.bool)

        masked_score = score.masked_fill(~mask, -1e4)
        keep = min(max(topk, 1), k)
        if keep < k:
            idx = masked_score.topk(keep, dim=-1).indices
            sel_score = score.gather(-1, idx)
            sel_quads = quads.gather(2, idx[:, :, :, None, None].expand(-1, -1, -1, 4, 2))
            sel_mask = mask.gather(-1, idx)
        else:
            sel_score = score
            sel_quads = quads
            sel_mask = mask

        sel_score = sel_score.masked_fill(~sel_mask, -1e4)
        valid = sel_mask.any(dim=-1, keepdim=True)
        centered_score = sel_score - sel_score.max(dim=-1, keepdim=True).values
        center_weights = F.softmax(centered_score / temp, dim=-1) * sel_mask.float()
        center_weights = center_weights / center_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        candidate_center = sel_quads.mean(dim=-2)
        target_center = (candidate_center * center_weights[..., None]).sum(dim=-2)
        target_center = torch.where(valid.squeeze(-1)[..., None], target_center, final.mean(dim=-2))

    final_center = final.mean(dim=-2, keepdim=True)
    shifted = (final + alpha * (target_center[:, :, None, :] - final_center)).clamp(0.0, 1.0)
    box_xyxy = quad_to_xyxy(shifted, min_size=0.02)
    updated = dict(outputs)
    updated["quad_xy"] = shifted.to(dtype=outputs["quad_xy"].dtype)
    updated["boxes_xyxy"] = box_xyxy.to(dtype=outputs["boxes_xyxy"].dtype if "boxes_xyxy" in outputs else shifted.dtype)
    updated["boxes_cxcywh"] = xyxy_to_cxcywh_torch(updated["boxes_xyxy"])
    updated["coarse_xyxy"] = updated["boxes_xyxy"]
    updated["coarse_cxcywh"] = updated["boxes_cxcywh"]
    return updated


@torch.no_grad()
def _weighted_geometric_median(points: torch.Tensor, weights: torch.Tensor, iters: int) -> torch.Tensor:
    """Small Weiszfeld solver for H334 eval-only quad voting.

    `points` is [M, 8] flattened quads. This is used only at validation time
    over tiny candidate clusters, so a short Python loop is acceptable.
    """
    weights = weights.clamp_min(1e-6)
    x = (points * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(1e-6)
    for _ in range(max(int(iters), 1)):
        dist = (points - x[None]).norm(dim=-1).clamp_min(1e-5)
        inv = weights / dist
        x = (points * inv[:, None]).sum(dim=0) / inv.sum().clamp_min(1e-6)
    return x


@torch.no_grad()
def apply_eval_quad_voting(
    outputs: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    """Eval-only WBF-style quad voting over proposal candidates.

    This adapts Weighted Boxes Fusion to the free-quad setting: cluster proposal
    hypotheses by their AABB overlap, select one high-score cluster, then fuse
    ordered quad points by weighted mean or weighted geometric median.
    """
    mode = str(getattr(args, "eval_quad_voting_mode", "none"))
    if mode == "none":
        return outputs
    required = {"proposal_logits", "proposal_quads"}
    if not required.issubset(outputs):
        return outputs

    logits = outputs["proposal_logits"].detach().float()
    quads = outputs["proposal_quads"].detach().float().clamp(0.0, 1.0)
    if logits.ndim != 3 or quads.ndim != 5:
        return outputs

    b, t, k = logits.shape
    topk = int(getattr(args, "eval_quad_voting_topk", 0))
    if topk <= 0 or topk > k:
        topk = k
    temp = max(float(getattr(args, "eval_quad_voting_temperature", 0.45)), 1e-6)
    iou_thr = float(getattr(args, "eval_quad_voting_iou_threshold", 0.20))
    score_mode = str(getattr(args, "eval_quad_voting_score_mode", "sum"))
    max_iters = int(getattr(args, "eval_quad_voting_max_geomedian_iters", 8))

    flat_logits = logits.reshape(b * t, k)
    flat_quads = quads.reshape(b * t, k, 4, 2)
    topk_idx_full = None
    if topk < k:
        idx = flat_logits.topk(topk, dim=-1).indices
        topk_idx_full = idx
        flat_logits = flat_logits.gather(-1, idx)
        flat_quads = flat_quads.gather(
            1,
            idx[:, :, None, None].expand(-1, -1, 4, 2),
        )
    scores = F.softmax(flat_logits / temp, dim=-1).clamp_min(1e-6)
    voted = torch.empty((b * t, 4, 2), device=quads.device, dtype=quads.dtype)
    voted_weights = torch.zeros((b * t, topk), device=quads.device, dtype=quads.dtype)

    for n in range(b * t):
        qn = flat_quads[n]
        sn = scores[n]
        boxes = quad_to_xyxy(qn, min_size=0.02)
        pair_iou = box_iou_xyxy(boxes[:, None, :], boxes[None, :, :])
        order = torch.argsort(sn, descending=True)
        best_mask = None
        best_score = None
        for seed in order.tolist():
            mask = pair_iou[seed] >= iou_thr
            if not bool(mask.any()):
                mask = torch.zeros_like(mask, dtype=torch.bool)
                mask[seed] = True
            if score_mode == "max":
                cluster_score = sn[mask].max()
            elif score_mode == "seed":
                cluster_score = sn[seed]
            else:
                cluster_score = sn[mask].sum()
            if best_score is None or float(cluster_score) > float(best_score):
                best_score = cluster_score
                best_mask = mask
        assert best_mask is not None
        w = sn[best_mask]
        pts = qn[best_mask].reshape(-1, 8)
        if mode == "cluster_geomedian" and pts.shape[0] > 1:
            fused = _weighted_geometric_median(pts, w, max_iters).reshape(4, 2)
        else:
            fused = (pts * w[:, None]).sum(dim=0) / w.sum().clamp_min(1e-6)
            fused = fused.reshape(4, 2)
        voted[n] = roll_quad_start_min_sum(fused).clamp(0.0, 1.0)
        voted_weights[n, best_mask] = w / w.sum().clamp_min(1e-6)

    quad = voted.reshape(b, t, 4, 2)
    box_xyxy = quad_to_xyxy(quad, min_size=0.02)
    updated = dict(outputs)
    updated["quad_xy"] = quad.to(dtype=outputs["quad_xy"].dtype if "quad_xy" in outputs else quads.dtype)
    updated["boxes_xyxy"] = box_xyxy.to(dtype=outputs["boxes_xyxy"].dtype if "boxes_xyxy" in outputs else quads.dtype)
    updated["boxes_cxcywh"] = xyxy_to_cxcywh_torch(updated["boxes_xyxy"])
    updated["coarse_xyxy"] = updated["boxes_xyxy"]
    updated["coarse_cxcywh"] = updated["boxes_cxcywh"]
    if topk_idx_full is not None:
        full_weights = torch.zeros((b * t, k), device=quads.device, dtype=quads.dtype)
        full_weights.scatter_(1, topk_idx_full, voted_weights)
    else:
        full_weights = voted_weights
    updated["proposal_selection_weights"] = full_weights.reshape(b, t, k).to(dtype=quads.dtype)
    return updated


@torch.no_grad()
def apply_eval_tubelet_dp(
    outputs: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    """Eval-only Viterbi path selection over per-frame proposal candidates.

    H336 tests whether the handoff failure is partly caused by independent
    per-frame selection.  This function keeps the trained proposal bank fixed
    and selects one temporally coherent 16-frame candidate path using only
    proposal logits and adjacent-frame geometry consistency.
    """
    if not bool(getattr(args, "eval_tubelet_dp", False)):
        return outputs
    required = {"proposal_logits", "proposal_quads"}
    if not required.issubset(outputs):
        return outputs

    logits_full = outputs["proposal_logits"].detach().float()
    quads_full = outputs["proposal_quads"].detach().float().clamp(0.0, 1.0)
    if logits_full.ndim != 3 or quads_full.ndim != 5:
        return outputs

    b, t, k = logits_full.shape
    if t <= 1 or k <= 0:
        return outputs

    topk = int(getattr(args, "eval_tubelet_dp_topk", 0))
    if topk <= 0 or topk > k:
        topk = k

    if topk < k:
        topk_idx = logits_full.topk(topk, dim=-1).indices
        logits = logits_full.gather(-1, topk_idx)
        quads = quads_full.gather(
            2,
            topk_idx[:, :, :, None, None].expand(-1, -1, -1, 4, 2),
        )
    else:
        topk_idx = None
        logits = logits_full
        quads = quads_full

    node_weight = float(getattr(args, "eval_tubelet_dp_node_weight", 0.25))
    edge_weight = float(getattr(args, "eval_tubelet_dp_edge_weight", 1.0))
    edge_iou_weight = float(getattr(args, "eval_tubelet_dp_edge_iou_weight", 1.0))
    center_weight = float(getattr(args, "eval_tubelet_dp_center_weight", 0.5))
    corner_weight = float(getattr(args, "eval_tubelet_dp_corner_weight", 0.5))
    area_weight = float(getattr(args, "eval_tubelet_dp_area_weight", 0.25))
    aspect_weight = float(getattr(args, "eval_tubelet_dp_aspect_weight", 0.25))
    center_scale = max(float(getattr(args, "eval_tubelet_dp_center_scale", 0.18)), 1e-6)
    corner_scale = max(float(getattr(args, "eval_tubelet_dp_corner_scale", 0.18)), 1e-6)

    device = quads.device
    selected_quads = torch.empty((b, t, 4, 2), device=device, dtype=quads.dtype)
    full_weights = torch.zeros((b, t, k), device=device, dtype=quads.dtype)
    switch_counts = []
    adjacent_ious = []
    path_scores = []

    frame_arange = torch.arange(t, device=device)

    for bi in range(b):
        q = quads[bi]
        logit = logits[bi]
        node = F.log_softmax(logit, dim=-1) * node_weight
        boxes = quad_to_xyxy(q, min_size=0.0)
        centers = q.mean(dim=-2)
        wh = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-6)
        areas = (wh[..., 0] * wh[..., 1]).clamp_min(1e-6)
        aspects = (wh[..., 0] / wh[..., 1]).clamp_min(1e-6)

        dp = node[0].clone()
        backptrs: list[torch.Tensor] = []
        adj_iou_mats: list[torch.Tensor] = []
        for ti in range(1, t):
            prev_box = boxes[ti - 1]
            next_box = boxes[ti]
            pair_iou = box_iou_xyxy(prev_box[:, None, :], next_box[None, :, :])
            center_dist = torch.cdist(centers[ti - 1], centers[ti], p=2)
            center_sim = torch.exp(-center_dist / center_scale)
            corner_dist = torch.linalg.norm(
                q[ti - 1, :, None, :, :] - q[ti, None, :, :, :],
                dim=-1,
            ).mean(dim=-1)
            corner_sim = torch.exp(-corner_dist / corner_scale)
            area_sim = torch.exp(-torch.abs(torch.log(areas[ti - 1, :, None] / areas[ti, None, :])))
            aspect_sim = torch.exp(-torch.abs(torch.log(aspects[ti - 1, :, None] / aspects[ti, None, :])))
            edge = edge_weight * (
                edge_iou_weight * pair_iou
                + center_weight * center_sim
                + corner_weight * corner_sim
                + area_weight * area_sim
                + aspect_weight * aspect_sim
            )
            score = dp[:, None] + edge + node[ti][None, :]
            dp, prev_idx = score.max(dim=0)
            backptrs.append(prev_idx)
            adj_iou_mats.append(pair_iou)

        path_local = torch.empty((t,), device=device, dtype=torch.long)
        path_local[-1] = dp.argmax()
        for ti in range(t - 1, 0, -1):
            path_local[ti - 1] = backptrs[ti - 1][path_local[ti]]

        selected_quads[bi] = q[frame_arange, path_local]
        if topk_idx is not None:
            path_full = topk_idx[bi, frame_arange, path_local]
        else:
            path_full = path_local
        full_weights[bi, frame_arange, path_full] = 1.0
        if t > 1:
            switch_counts.append((path_full[1:] != path_full[:-1]).float().mean())
            adjacent_ious.append(
                torch.stack(
                    [
                        adj_iou_mats[ti][path_local[ti], path_local[ti + 1]]
                        for ti in range(t - 1)
                    ]
                ).mean()
            )
        path_scores.append(dp.max())

    quad = selected_quads.clamp(0.0, 1.0)
    box_xyxy = quad_to_xyxy(quad, min_size=0.02)
    updated = dict(outputs)
    updated["quad_xy"] = quad.to(dtype=outputs["quad_xy"].dtype if "quad_xy" in outputs else quads_full.dtype)
    updated["boxes_xyxy"] = box_xyxy.to(dtype=outputs["boxes_xyxy"].dtype if "boxes_xyxy" in outputs else quads_full.dtype)
    updated["boxes_cxcywh"] = xyxy_to_cxcywh_torch(updated["boxes_xyxy"])
    updated["coarse_xyxy"] = updated["boxes_xyxy"]
    updated["coarse_cxcywh"] = updated["boxes_cxcywh"]
    updated["proposal_selection_weights"] = full_weights.to(dtype=quads_full.dtype)
    if switch_counts:
        updated["eval_tubelet_dp_switch_rate"] = torch.stack(switch_counts).mean()
    if adjacent_ious:
        updated["eval_tubelet_dp_adj_aabb_iou"] = torch.stack(adjacent_ious).mean()
    if path_scores:
        updated["eval_tubelet_dp_path_score"] = torch.stack(path_scores).mean()
    return updated


@torch.no_grad()
def apply_eval_token_corner_tracking(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    """Eval-only token-space propagation of the four predicted quad corners.

    This is a lightweight H337 analogue of point tracking: initialize the four
    operation-region corners on one confident frame, track each point forward
    and backward by local cosine matching over the frozen V-JEPA token grid,
    then either use the tracked quad directly or snap each frame to the nearest
    existing proposal candidate.
    """
    if not bool(getattr(args, "eval_token_corner_track", False)):
        return outputs
    required = {"quad_xy", "proposal_logits", "proposal_quads"}
    if not required.issubset(outputs) or "tokens" not in batch:
        return outputs

    tokens_raw = batch["tokens"]
    logits_full = outputs["proposal_logits"].detach().float()
    quads_full = outputs["proposal_quads"].detach().float().clamp(0.0, 1.0)
    seed_quads = outputs["quad_xy"].detach().float().clamp(0.0, 1.0)
    if tokens_raw.ndim != 5 or logits_full.ndim != 3 or quads_full.ndim != 5:
        return outputs

    b, t, k = logits_full.shape
    bt, ttok, h, w, c = tokens_raw.shape
    if bt != b or t <= 1 or ttok <= 0:
        return outputs

    mode = str(getattr(args, "eval_token_corner_track_mode", "candidate_snap"))
    seed_mode = str(getattr(args, "eval_token_corner_track_seed", "max_margin"))
    radius = max(float(getattr(args, "eval_token_corner_track_radius", 0.22)), 0.0)
    temp = max(float(getattr(args, "eval_token_corner_track_temperature", 0.05)), 1e-6)
    use_soft = bool(getattr(args, "eval_token_corner_track_softargmax", False))
    logit_weight = float(getattr(args, "eval_token_corner_track_logit_weight", 0.20))
    dist_weight = float(getattr(args, "eval_token_corner_track_dist_weight", 1.0))
    dist_scale = max(float(getattr(args, "eval_token_corner_track_dist_scale", 0.12)), 1e-6)

    device = quads_full.device
    tokens = tokens_raw.to(device=device, dtype=torch.float32)
    tokens = F.normalize(tokens.reshape(b, ttok, h * w, c), dim=-1, eps=1e-6)
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, h, device=device),
        torch.linspace(0.0, 1.0, w, device=device),
        indexing="ij",
    )
    grid_xy = torch.stack([xx, yy], dim=-1).reshape(h * w, 2)
    token_frame = torch.round(torch.linspace(0, ttok - 1, t, device=device)).long()
    frame_arange = torch.arange(t, device=device)

    tracked = torch.empty((b, t, 4, 2), device=device, dtype=quads_full.dtype)
    full_weights = torch.zeros((b, t, k), device=device, dtype=quads_full.dtype)
    snap_dists = []
    track_steps = []
    seed_frames = []

    def choose_seed(logits_b: torch.Tensor) -> int:
        if seed_mode == "middle":
            return int(t // 2)
        top2 = logits_b.topk(min(2, logits_b.shape[-1]), dim=-1).values
        if seed_mode == "max_logit" or top2.shape[-1] == 1:
            return int(top2[:, 0].argmax().item())
        margin = top2[:, 0] - top2[:, 1]
        return int(margin.argmax().item())

    def nearest_feature(feat: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        dist = torch.linalg.norm(grid_xy - point[None, :], dim=-1)
        idx = dist.argmin()
        return feat[idx]

    def track_one_direction(
        feat_b: torch.Tensor,
        seed_frame: int,
        seed_quad: torch.Tensor,
        forward: bool,
        out: torch.Tensor,
    ) -> list[torch.Tensor]:
        desc = torch.stack(
            [
                nearest_feature(feat_b[int(token_frame[seed_frame])], seed_quad[p])
                for p in range(4)
            ],
            dim=0,
        )
        prev = seed_quad.clone()
        moved = []
        frame_iter = range(seed_frame + 1, t) if forward else range(seed_frame - 1, -1, -1)
        for ti in frame_iter:
            feat = feat_b[int(token_frame[ti])]
            pts = []
            for pi in range(4):
                dist = torch.linalg.norm(grid_xy - prev[pi][None, :], dim=-1)
                score = feat @ desc[pi]
                if radius > 0.0:
                    score = score.masked_fill(dist > radius, -1e4)
                if use_soft:
                    prob = F.softmax(score / temp, dim=0)
                    pt = (grid_xy * prob[:, None]).sum(dim=0)
                else:
                    pt = grid_xy[score.argmax()]
                pts.append(pt)
            cur = torch.stack(pts, dim=0).clamp(0.0, 1.0)
            out[ti] = roll_quad_start_min_sum(cur)
            moved.append(torch.linalg.norm(cur - prev, dim=-1).mean())
            prev = cur
        return moved

    for bi in range(b):
        seed_frame = choose_seed(logits_full[bi])
        seed_frames.append(torch.tensor(float(seed_frame), device=device))
        tracked_b = torch.empty((t, 4, 2), device=device, dtype=quads_full.dtype)
        tracked_b[seed_frame] = seed_quads[bi, seed_frame]
        moves = []
        moves.extend(track_one_direction(tokens[bi], seed_frame, tracked_b[seed_frame], True, tracked_b))
        moves.extend(track_one_direction(tokens[bi], seed_frame, tracked_b[seed_frame], False, tracked_b))
        tracked[bi] = tracked_b.clamp(0.0, 1.0)
        if moves:
            track_steps.append(torch.stack(moves).mean())

        if mode == "candidate_snap":
            corner_dist = torch.linalg.norm(
                quads_full[bi] - tracked_b[:, None, :, :],
                dim=-1,
            ).mean(dim=-1)
            logprob = F.log_softmax(logits_full[bi], dim=-1)
            score = logit_weight * logprob - dist_weight * (corner_dist / dist_scale)
            best = score.argmax(dim=-1)
            tracked[bi] = quads_full[bi, frame_arange, best]
            full_weights[bi, frame_arange, best] = 1.0
            snap_dists.append(corner_dist[frame_arange, best].mean())
        else:
            snap_dists.append(
                torch.linalg.norm(quads_full[bi] - tracked_b[:, None, :, :], dim=-1).mean(dim=-1).min(dim=-1).values.mean()
            )
            nearest = torch.linalg.norm(quads_full[bi] - tracked_b[:, None, :, :], dim=-1).mean(dim=-1).argmin(dim=-1)
            full_weights[bi, frame_arange, nearest] = 1.0

    quad = tracked.clamp(0.0, 1.0)
    box_xyxy = quad_to_xyxy(quad, min_size=0.02)
    updated = dict(outputs)
    updated["quad_xy"] = quad.to(dtype=outputs["quad_xy"].dtype if "quad_xy" in outputs else quads_full.dtype)
    updated["boxes_xyxy"] = box_xyxy.to(dtype=outputs["boxes_xyxy"].dtype if "boxes_xyxy" in outputs else quads_full.dtype)
    updated["boxes_cxcywh"] = xyxy_to_cxcywh_torch(updated["boxes_xyxy"])
    updated["coarse_xyxy"] = updated["boxes_xyxy"]
    updated["coarse_cxcywh"] = updated["boxes_cxcywh"]
    updated["proposal_selection_weights"] = full_weights.to(dtype=quads_full.dtype)
    if snap_dists:
        updated["eval_token_corner_track_snap_dist"] = torch.stack(snap_dists).mean()
    if track_steps:
        updated["eval_token_corner_track_step"] = torch.stack(track_steps).mean()
    if seed_frames:
        updated["eval_token_corner_track_seed_frame"] = torch.stack(seed_frames).mean()
    return updated


@torch.no_grad()
def apply_eval_candidate_memory(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    """Eval-only object-level candidate memory selection.

    H338 moves from point tracking to proposal-level memory: each candidate is
    represented by a 9-point V-JEPA readout, confident frames provide memory
    descriptors, and every frame selects the candidate most compatible with
    that memory while retaining a configurable proposal-logit prior.
    """
    if not bool(getattr(args, "eval_candidate_memory", False)):
        return outputs
    required = {"proposal_logits", "proposal_quads"}
    if not required.issubset(outputs) or "tokens" not in batch:
        return outputs

    tokens_raw = batch["tokens"]
    logits_full = outputs["proposal_logits"].detach().float()
    quads_full = outputs["proposal_quads"].detach().float().clamp(0.0, 1.0)
    if tokens_raw.ndim != 5 or logits_full.ndim != 3 or quads_full.ndim != 5:
        return outputs

    b, t, k = logits_full.shape
    bt, ttok, h, w, c = tokens_raw.shape
    if bt != b or t <= 0 or k <= 0:
        return outputs

    seed_mode = str(getattr(args, "eval_candidate_memory_seed", "max_margin"))
    mem_frames = max(int(getattr(args, "eval_candidate_memory_frames", 3)), 1)
    mem_frames = min(mem_frames, t)
    mem_source = str(getattr(args, "eval_candidate_memory_source", "selected"))
    memory_weight = float(getattr(args, "eval_candidate_memory_weight", 1.0))
    logit_weight = float(getattr(args, "eval_candidate_memory_logit_weight", 0.25))

    device = quads_full.device
    tokens = tokens_raw.to(device=device, dtype=torch.float32)
    token_frame = torch.round(torch.linspace(0, ttok - 1, t, device=device)).long()
    tokens16 = tokens[:, token_frame]
    feat = tokens16.reshape(b * t, h, w, c).permute(0, 3, 1, 2).contiguous()

    p0, p1, p2, p3 = quads_full.unbind(dim=-2)
    center = quads_full.mean(dim=-2)
    edge01 = 0.5 * (p0 + p1)
    edge12 = 0.5 * (p1 + p2)
    edge23 = 0.5 * (p2 + p3)
    edge30 = 0.5 * (p3 + p0)
    pts = torch.stack([center, p0, p1, p2, p3, edge01, edge12, edge23, edge30], dim=-2)
    grid = (pts.reshape(b * t, k * 9, 2) * 2.0 - 1.0).clamp(-1.0, 1.0)
    sampled = F.grid_sample(
        feat,
        grid[:, :, None, :],
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, t, k, 9, c)
    desc = F.normalize(sampled.mean(dim=3), dim=-1, eps=1e-6)

    top2 = logits_full.topk(min(2, k), dim=-1).values
    fallback_frame_score = top2[:, :, 0] if top2.shape[-1] == 1 else top2[:, :, 0] - top2[:, :, 1]
    if seed_mode == "middle":
        frame_score = -torch.abs(torch.arange(t, device=device, dtype=torch.float32)[None, :] - float(t // 2))
        frame_score = frame_score.expand(b, -1)
    elif seed_mode == "max_logit" or top2.shape[-1] == 1:
        frame_score = top2[:, :, 0]
    elif seed_mode in {"active", "active_mid", "active_center"} and torch.is_tensor(batch.get("active_16")):
        active = batch["active_16"].to(device=device, dtype=torch.float32)
        if active.shape != (b, t):
            frame_score = fallback_frame_score
        else:
            has_active = active.sum(dim=1, keepdim=True) > 0.5
            if seed_mode == "active_mid":
                frame_idx = torch.arange(t, device=device, dtype=torch.float32)[None, :]
                active_center = (active * frame_idx).sum(dim=1, keepdim=True) / active.sum(dim=1, keepdim=True).clamp_min(1e-6)
                mid_bonus = 1.0 - (frame_idx - active_center).abs() / max(float(t - 1), 1.0)
                frame_score = active + 0.25 * mid_bonus.clamp(0.0, 1.0)
            elif seed_mode == "active_center" and torch.is_tensor(batch.get("role_features_16")):
                role = batch["role_features_16"].to(device=device, dtype=torch.float32)
                if role.shape[:2] == (b, t) and role.shape[-1] >= 2:
                    phase = role[..., 1].clamp(0.0, 1.0)
                    center_bonus = 1.0 - (phase - 0.5).abs() * 2.0
                    frame_score = active + 0.25 * center_bonus.clamp(0.0, 1.0)
                else:
                    frame_score = active
            else:
                frame_score = active
            frame_score = torch.where(has_active, frame_score, fallback_frame_score)
    else:
        frame_score = fallback_frame_score
    mem_idx = frame_score.topk(mem_frames, dim=-1).indices

    selected_weights = outputs.get("proposal_selection_weights")
    if torch.is_tensor(selected_weights) and selected_weights.shape == logits_full.shape:
        selected_weights = selected_weights.detach().float().to(device=device)
    else:
        selected_weights = None

    selected_quads = torch.empty((b, t, 4, 2), device=device, dtype=quads_full.dtype)
    full_weights = torch.zeros((b, t, k), device=device, dtype=quads_full.dtype)
    mem_sims = []
    seed_means = []
    frame_arange = torch.arange(t, device=device)

    for bi in range(b):
        idx = mem_idx[bi]
        if mem_source == "selected" and selected_weights is not None:
            w_mem = selected_weights[bi, idx].clamp_min(0.0)
            w_mem = w_mem / w_mem.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            mem_desc = (desc[bi, idx] * w_mem[:, :, None]).sum(dim=1).mean(dim=0)
        else:
            best_mem = logits_full[bi, idx].argmax(dim=-1)
            mem_desc = desc[bi, idx, best_mem].mean(dim=0)
        mem_desc = F.normalize(mem_desc, dim=-1, eps=1e-6)
        sim = (desc[bi] * mem_desc[None, None, :]).sum(dim=-1)
        logprob = F.log_softmax(logits_full[bi], dim=-1)
        score = memory_weight * sim + logit_weight * logprob
        best = score.argmax(dim=-1)
        selected_quads[bi] = quads_full[bi, frame_arange, best]
        full_weights[bi, frame_arange, best] = 1.0
        mem_sims.append(sim[frame_arange, best].mean())
        seed_means.append(idx.float().mean())

    quad = selected_quads.clamp(0.0, 1.0)
    box_xyxy = quad_to_xyxy(quad, min_size=0.02)
    updated = dict(outputs)
    updated["quad_xy"] = quad.to(dtype=outputs["quad_xy"].dtype if "quad_xy" in outputs else quads_full.dtype)
    updated["boxes_xyxy"] = box_xyxy.to(dtype=outputs["boxes_xyxy"].dtype if "boxes_xyxy" in outputs else quads_full.dtype)
    updated["boxes_cxcywh"] = xyxy_to_cxcywh_torch(updated["boxes_xyxy"])
    updated["coarse_xyxy"] = updated["boxes_xyxy"]
    updated["coarse_cxcywh"] = updated["boxes_cxcywh"]
    updated["proposal_selection_weights"] = full_weights.to(dtype=quads_full.dtype)
    if mem_sims:
        updated["eval_candidate_memory_similarity"] = torch.stack(mem_sims).mean()
    if seed_means:
        updated["eval_candidate_memory_seed_frame"] = torch.stack(seed_means).mean()
    return updated


def combine_ensemble_outputs(
    outputs_list: list[dict[str, torch.Tensor]],
    *,
    mode: str,
    temperature: float,
    consensus_threshold: float = 0.02,
    blend_alpha: float = 0.50,
) -> dict[str, torch.Tensor]:
    if len(outputs_list) == 1:
        return outputs_list[0]
    first = outputs_list[0]
    updated = dict(first)
    selected_quads = [out["quad_xy"].float().clamp(0.0, 1.0) for out in outputs_list if "quad_xy" in out]
    if not selected_quads:
        return first

    def consensus_weighted_selected_mean(stack: torch.Tensor) -> torch.Tensor:
        if stack.shape[0] <= 1:
            return stack[0].clamp(0.0, 1.0)
        temp = max(float(temperature), 1e-6)
        pair_dist = torch.linalg.norm(stack[:, None] - stack[None, :], dim=-1).mean(dim=-1)
        avg_dist = pair_dist.sum(dim=1) / float(max(stack.shape[0] - 1, 1))
        weights = F.softmax(-avg_dist / temp, dim=0)
        updated["eval_ensemble_consensus_weight_entropy"] = (
            -(weights * weights.clamp_min(1e-8).log()).sum(dim=0).mean()
        )
        updated["eval_ensemble_consensus_weight_max"] = weights.max(dim=0).values.mean()
        return (stack * weights[..., None, None]).sum(dim=0).clamp(0.0, 1.0)

    proposal_quads = [out["proposal_quads"].float().clamp(0.0, 1.0) for out in outputs_list if "proposal_quads" in out]
    proposal_logits = [out["proposal_logits"].float() for out in outputs_list if "proposal_logits" in out]
    if len(proposal_quads) == len(outputs_list) and len(proposal_logits) == len(outputs_list):
        union_quads = torch.cat(proposal_quads, dim=2)
        union_logits = torch.cat(proposal_logits, dim=-1)
        updated["proposal_quads"] = union_quads.to(dtype=first["proposal_quads"].dtype)
        updated["proposal_logits"] = union_logits.to(dtype=first["proposal_logits"].dtype)
        if all("proposal_xyxy" in out for out in outputs_list):
            updated["proposal_xyxy"] = torch.cat([out["proposal_xyxy"].float() for out in outputs_list], dim=2).to(
                dtype=first["proposal_xyxy"].dtype
            )
        else:
            b, t, k = union_quads.shape[:3]
            updated["proposal_xyxy"] = quad_to_xyxy(union_quads.reshape(b, t * k, 4, 2), min_size=0.02).reshape(
                b, t, k, 4
            )
        updated["proposal_cxcywh"] = xyxy_to_cxcywh_torch(updated["proposal_xyxy"])
        if all("proposal_source_ids" in out for out in outputs_list):
            updated["proposal_source_ids"] = torch.cat([out["proposal_source_ids"] for out in outputs_list], dim=2)
        if mode in {"proposal_soft", "proposal_hard"}:
            if mode == "proposal_hard":
                best = union_logits.argmax(dim=-1)
                weights = F.one_hot(best, num_classes=union_logits.shape[-1]).to(dtype=union_logits.dtype)
            else:
                temp = max(float(temperature), 1e-6)
                weights = F.softmax(union_logits / temp, dim=-1)
            quad = (union_quads * weights[..., None, None]).sum(dim=2).clamp(0.0, 1.0)
            updated["proposal_selection_weights"] = weights.to(dtype=union_quads.dtype)
        else:
            weights_parts = []
            for out in outputs_list:
                weights = out.get("proposal_selection_weights")
                if torch.is_tensor(weights) and weights.shape == out["proposal_logits"].shape:
                    weights_parts.append(weights.float() / float(len(outputs_list)))
                else:
                    weights_parts.append(torch.zeros_like(out["proposal_logits"].float()))
            updated["proposal_selection_weights"] = torch.cat(weights_parts, dim=-1).to(dtype=union_quads.dtype)
            stack = torch.stack(selected_quads, dim=0)
            if mode == "selected_median":
                quad = stack.median(dim=0).values.clamp(0.0, 1.0)
            elif mode == "selected_mean_median_blend":
                alpha = float(np.clip(blend_alpha, 0.0, 1.0))
                mean_quad = stack.mean(dim=0)
                median_quad = stack.median(dim=0).values
                quad = (alpha * mean_quad + (1.0 - alpha) * median_quad).clamp(0.0, 1.0)
            elif mode == "selected_consensus_weighted_mean":
                quad = consensus_weighted_selected_mean(stack)
            elif mode in {"selected_mean_gate_first", "selected_mean_gate_median"}:
                mean_quad = stack.mean(dim=0).clamp(0.0, 1.0)
                disagreement = torch.linalg.norm(stack - mean_quad.unsqueeze(0), dim=-1).mean(dim=(0, 3))
                use_mean = disagreement <= max(float(consensus_threshold), 0.0)
                fallback = first["quad_xy"].float().clamp(0.0, 1.0)
                if mode == "selected_mean_gate_median":
                    fallback = stack.median(dim=0).values.clamp(0.0, 1.0)
                quad = torch.where(use_mean[..., None, None], mean_quad, fallback)
                updated["eval_ensemble_disagreement"] = disagreement.mean()
                updated["eval_ensemble_mean_gate_frac"] = use_mean.float().mean()
            else:
                quad = stack.mean(dim=0).clamp(0.0, 1.0)
    else:
        stack = torch.stack(selected_quads, dim=0)
        if mode == "selected_median":
            quad = stack.median(dim=0).values.clamp(0.0, 1.0)
        elif mode == "selected_mean_median_blend":
            alpha = float(np.clip(blend_alpha, 0.0, 1.0))
            mean_quad = stack.mean(dim=0)
            median_quad = stack.median(dim=0).values
            quad = (alpha * mean_quad + (1.0 - alpha) * median_quad).clamp(0.0, 1.0)
        elif mode == "selected_consensus_weighted_mean":
            quad = consensus_weighted_selected_mean(stack)
        elif mode in {"selected_mean_gate_first", "selected_mean_gate_median"}:
            mean_quad = stack.mean(dim=0).clamp(0.0, 1.0)
            disagreement = torch.linalg.norm(stack - mean_quad.unsqueeze(0), dim=-1).mean(dim=(0, 3))
            use_mean = disagreement <= max(float(consensus_threshold), 0.0)
            fallback = first["quad_xy"].float().clamp(0.0, 1.0)
            if mode == "selected_mean_gate_median":
                fallback = stack.median(dim=0).values.clamp(0.0, 1.0)
            quad = torch.where(use_mean[..., None, None], mean_quad, fallback)
            updated["eval_ensemble_disagreement"] = disagreement.mean()
            updated["eval_ensemble_mean_gate_frac"] = use_mean.float().mean()
        else:
            quad = stack.mean(dim=0).clamp(0.0, 1.0)

    box_xyxy = quad_to_xyxy(quad, min_size=0.02)
    updated["quad_xy"] = quad.to(dtype=first["quad_xy"].dtype)
    updated["boxes_xyxy"] = box_xyxy.to(dtype=first["boxes_xyxy"].dtype)
    updated["boxes_cxcywh"] = xyxy_to_cxcywh_torch(updated["boxes_xyxy"])
    updated["coarse_xyxy"] = updated["boxes_xyxy"]
    updated["coarse_cxcywh"] = updated["boxes_cxcywh"]
    return updated


class EvalCheckpointEnsemble(nn.Module):
    def __init__(
        self,
        models: list[nn.Module],
        *,
        mode: str,
        temperature: float,
        consensus_threshold: float = 0.02,
        blend_alpha: float = 0.50,
    ) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.mode = str(mode)
        self.temperature = float(temperature)
        self.consensus_threshold = float(consensus_threshold)
        self.blend_alpha = float(blend_alpha)

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = [model(*args, **kwargs) for model in self.models]
        return combine_ensemble_outputs(
            outputs,
            mode=self.mode,
            temperature=self.temperature,
            consensus_threshold=self.consensus_threshold,
            blend_alpha=self.blend_alpha,
        )


def forward_roi_model_for_batch(
    model: nn.Module,
    args: argparse.Namespace,
    batch_t: dict[str, torch.Tensor],
    *,
    tokens_in: torch.Tensor | None = None,
    role_in: torch.Tensor | None = None,
    tool_mask_in: torch.Tensor | None = None,
    tool_hm_in: torch.Tensor | None = None,
    tool_vec_in: torch.Tensor | None = None,
    tool_id_in: torch.Tensor | None = None,
    act_id_in: torch.Tensor | None = None,
    combo_id_in: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    tokens = batch_t["tokens"] if tokens_in is None else tokens_in
    role = batch_t["role_features_16"] if role_in is None else role_in
    tool_mask = batch_t["tool_mask_8"] if tool_mask_in is None else tool_mask_in
    tool_hm = batch_t["tool_kpt_heatmaps_8"] if tool_hm_in is None else tool_hm_in
    tool_vec = batch_t["tool_kpt_vec_16"] if tool_vec_in is None else tool_vec_in
    tool_id = batch_t["tool_id"] if tool_id_in is None else tool_id_in
    act_id = batch_t["act_id"] if act_id_in is None else act_id_in
    combo_id = batch_t["combo_id"] if combo_id_in is None else combo_id_in
    if args.model_mode == "rgb_sidecar_temporal_residual":
        return model(tokens, role, tool_mask, tool_hm, tool_vec, tool_id, act_id, combo_id, batch_t["rgb_16"])
    rgb_16 = (
        batch_t["rgb_16"]
        if bool(getattr(args, "use_rgb_sidecar", False))
        and args.model_mode in {"quad_local_point_temporal", "proposal_point_temporal"}
        else None
    )
    if (
        args.model_mode
        in {
            "quad_local_point_temporal",
            "proposal_point_temporal",
            "trajectory_prior_point_temporal",
        }
        and (
            float(args.dn_refiner_loss_weight) > 0.0
            or float(args.proposal_oracle_refiner_loss_weight) > 0.0
        )
    ):
        return model(
            tokens,
            role,
            tool_mask,
            tool_hm,
            tool_vec,
            tool_id,
            act_id,
            combo_id,
            dn_quad_xy=batch_t["quad_xy"],
            rgb_16=rgb_16,
        )
    if rgb_16 is not None:
        return model(tokens, role, tool_mask, tool_hm, tool_vec, tool_id, act_id, combo_id, rgb_16=rgb_16)
    return model(tokens, role, tool_mask, tool_hm, tool_vec, tool_id, act_id, combo_id)


def write_teacher_cache(
    teacher_model: nn.Module,
    datasets: dict[str, Dataset],
    args: argparse.Namespace,
    device: torch.device,
    cache_root: Path,
    *,
    splits: set[str] | None = None,
    include_proposals: bool = False,
) -> dict[str, int]:
    teacher_model.eval()
    counts: dict[str, int] = {}
    cache_root.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for split, dataset in datasets.items():
            if splits is not None and split not in splits:
                continue
            split_dir = cache_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            loader = make_loader(dataset, args, shuffle=False)
            written = 0
            for batch in loader:
                batch_t = tensor_batch_to_device(batch, device)
                outputs = forward_roi_model_for_batch(teacher_model, args, batch_t)
                quad = outputs["quad_xy"].float().clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float16)
                boxes = outputs["boxes_cxcywh"].float().clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float16)
                proposal_payload: dict[str, np.ndarray] = {}
                if include_proposals:
                    if "proposal_quads" in outputs:
                        proposal_payload["proposal_quads"] = (
                            outputs["proposal_quads"]
                            .float()
                            .clamp(0.0, 1.0)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_logits" in outputs:
                        proposal_payload["proposal_logits"] = (
                            outputs["proposal_logits"].float().detach().cpu().numpy().astype(np.float16)
                        )
                    if "proposal_source_ids" in outputs:
                        proposal_payload["proposal_source_ids"] = (
                            outputs["proposal_source_ids"].detach().cpu().numpy().astype(np.int16)
                        )
                    if "proposal_selection_weights" in outputs:
                        proposal_payload["proposal_selection_weights"] = (
                            outputs["proposal_selection_weights"].float().detach().cpu().numpy().astype(np.float16)
                        )
                    if "proposal_support_overlap_scores" in outputs:
                        proposal_payload["proposal_support_overlap_scores"] = (
                            outputs["proposal_support_overlap_scores"]
                            .float()
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_support_geom_scores" in outputs:
                        proposal_payload["proposal_support_geom_scores"] = (
                            outputs["proposal_support_geom_scores"]
                            .float()
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "support_moment_proposal_quad" in outputs:
                        proposal_payload["support_moment_proposal_quad"] = (
                            outputs["support_moment_proposal_quad"]
                            .float()
                            .clamp(0.0, 1.0)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_set_refiner_weights" in outputs:
                        proposal_payload["proposal_set_refiner_weights"] = (
                            outputs["proposal_set_refiner_weights"].float().detach().cpu().numpy().astype(np.float16)
                        )
                    if "proposal_set_refiner_scores" in outputs:
                        proposal_payload["proposal_set_refiner_scores"] = (
                            outputs["proposal_set_refiner_scores"].float().detach().cpu().numpy().astype(np.float16)
                        )
                    if "proposal_set_refiner_base_quad" in outputs:
                        proposal_payload["proposal_set_refiner_base_quad"] = (
                            outputs["proposal_set_refiner_base_quad"]
                            .float()
                            .clamp(0.0, 1.0)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_handoff_source_router_logits" in outputs:
                        proposal_payload["proposal_handoff_source_router_logits"] = (
                            outputs["proposal_handoff_source_router_logits"]
                            .float()
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_handoff_source_router_candidate_logits" in outputs:
                        proposal_payload["proposal_handoff_source_router_candidate_logits"] = (
                            outputs["proposal_handoff_source_router_candidate_logits"]
                            .float()
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_handoff_source5_readout_logits" in outputs:
                        proposal_payload["proposal_handoff_source5_readout_logits"] = (
                            outputs["proposal_handoff_source5_readout_logits"]
                            .float()
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_handoff_mask_quality_logits" in outputs:
                        proposal_payload["proposal_handoff_mask_quality_logits"] = (
                            outputs["proposal_handoff_mask_quality_logits"]
                            .float()
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_handoff_mask_source_mask" in outputs:
                        proposal_payload["proposal_handoff_mask_source_mask"] = (
                            outputs["proposal_handoff_mask_source_mask"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.bool_)
                        )
                    if "proposal_handoff_rank_adaptive_delta" in outputs:
                        proposal_payload["proposal_handoff_rank_adaptive_delta"] = (
                            outputs["proposal_handoff_rank_adaptive_delta"]
                            .float()
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_handoff_rank_adaptive_rank" in outputs:
                        proposal_payload["proposal_handoff_rank_adaptive_rank"] = (
                            outputs["proposal_handoff_rank_adaptive_rank"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int16)
                        )
                    if "proposal_handoff_sparse_set_delta" in outputs:
                        proposal_payload["proposal_handoff_sparse_set_delta"] = (
                            outputs["proposal_handoff_sparse_set_delta"]
                            .float()
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_handoff_sparse_set_mask" in outputs:
                        proposal_payload["proposal_handoff_sparse_set_mask"] = (
                            outputs["proposal_handoff_sparse_set_mask"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.bool_)
                        )
                    if "proposal_handoff_source5_refiner_quad" in outputs:
                        proposal_payload["proposal_handoff_source5_refiner_quad"] = (
                            outputs["proposal_handoff_source5_refiner_quad"]
                            .float()
                            .clamp(0.0, 1.0)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_handoff_source5_refiner_base_quad" in outputs:
                        proposal_payload["proposal_handoff_source5_refiner_base_quad"] = (
                            outputs["proposal_handoff_source5_refiner_base_quad"]
                            .float()
                            .clamp(0.0, 1.0)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_handoff_source5_refiner_scores" in outputs:
                        proposal_payload["proposal_handoff_source5_refiner_scores"] = (
                            outputs["proposal_handoff_source5_refiner_scores"]
                            .float()
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_handoff_source5_refiner_weights" in outputs:
                        proposal_payload["proposal_handoff_source5_refiner_weights"] = (
                            outputs["proposal_handoff_source5_refiner_weights"]
                            .float()
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_handoff_source5_refiner_valid" in outputs:
                        proposal_payload["proposal_handoff_source5_refiner_valid"] = (
                            outputs["proposal_handoff_source5_refiner_valid"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.bool_)
                        )
                    if "proposal_handoff_topk_query_refiner_quads" in outputs:
                        proposal_payload["proposal_handoff_topk_query_refiner_quads"] = (
                            outputs["proposal_handoff_topk_query_refiner_quads"]
                            .float()
                            .clamp(0.0, 1.0)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_handoff_topk_query_refiner_base_quads" in outputs:
                        proposal_payload["proposal_handoff_topk_query_refiner_base_quads"] = (
                            outputs["proposal_handoff_topk_query_refiner_base_quads"]
                            .float()
                            .clamp(0.0, 1.0)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_handoff_topk_query_refiner_logits" in outputs:
                        proposal_payload["proposal_handoff_topk_query_refiner_logits"] = (
                            outputs["proposal_handoff_topk_query_refiner_logits"]
                            .float()
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    if "proposal_handoff_topk_query_refiner_valid" in outputs:
                        proposal_payload["proposal_handoff_topk_query_refiner_valid"] = (
                            outputs["proposal_handoff_topk_query_refiner_valid"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.bool_)
                        )
                    if "proposal_handoff_topk_query_refiner_source_ids" in outputs:
                        proposal_payload["proposal_handoff_topk_query_refiner_source_ids"] = (
                            outputs["proposal_handoff_topk_query_refiner_source_ids"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int16)
                        )
                    if "proposal_handoff_source_representative_mask" in outputs:
                        proposal_payload["proposal_handoff_source_representative_mask"] = (
                            outputs["proposal_handoff_source_representative_mask"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.bool_)
                        )
                    if "proposal_handoff_router_source_gate_mask" in outputs:
                        proposal_payload["proposal_handoff_router_source_gate_mask"] = (
                            outputs["proposal_handoff_router_source_gate_mask"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.bool_)
                        )
                for i, (sid, quad_i, box_i) in enumerate(zip(batch_t["sid"], quad, boxes)):
                    payload: dict[str, np.ndarray] = {"quad_xy": quad_i, "boxes_cxcywh": box_i}
                    for key, value in proposal_payload.items():
                        payload[key] = value[i]
                    np.savez(split_dir / f"{sid}.npz", **payload)
                    written += 1
            counts[split] = written
    return counts


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    condition_ablation: str = "clean",
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    n_batches = 0
    ious_all = []
    ious_active = []
    ious_inactive = []
    quad_ious_all = []
    quad_ious_active = []
    quad_ious_inactive = []
    coverage_all = []
    per_combo: dict[int, list[float]] = {}
    per_combo_quad: dict[int, list[float]] = {}
    for batch in loader:
        batch_t = tensor_batch_to_device(batch, device)
        if bool(getattr(args, "force_zero_tokens", False)) and torch.is_tensor(batch_t.get("tokens")):
            batch_t = dict(batch_t)
            batch_t["tokens"] = torch.zeros_like(batch_t["tokens"])
        batch_t = apply_condition_ablation(batch_t, condition_ablation)
        outputs = forward_roi_model_for_batch(model, args, batch_t)
        outputs = apply_eval_proposal_source_bias(outputs, args)
        outputs = apply_eval_center_identity_shift(outputs, args)
        outputs = apply_eval_quad_voting(outputs, args)
        outputs = apply_eval_tubelet_dp(outputs, args)
        outputs = apply_eval_token_corner_tracking(outputs, batch_t, args)
        outputs = apply_eval_candidate_memory(outputs, batch_t, args)
        _, metrics = compute_loss(outputs, batch_t, args)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n_batches += 1
        gt = batch_t["boxes_xyxy"]
        pred = outputs.get("boxes_xyxy_eval", outputs["boxes_xyxy"])
        iou = box_iou_xyxy(pred, gt)
        active = batch_t["active_16"] > 0.5
        if "quad_xy" in outputs:
            pred_quad_eval = outputs.get("quad_xy_eval", outputs["quad_xy"])
            qiou = quad_iou_approx(
                pred_quad_eval,
                batch_t["quad_xy"],
                grid_size=int(args.quad_iou_grid_size),
            )
            quad_ious_all.append(qiou.reshape(-1).detach().cpu())
            if active.any():
                quad_ious_active.append(qiou[active].reshape(-1).detach().cpu())
            if (~active).any():
                quad_ious_inactive.append(qiou[~active].reshape(-1).detach().cpu())
            for combo_id, vals in zip(batch_t["combo_id"].detach().cpu().tolist(), qiou.detach().cpu()):
                per_combo_quad.setdefault(int(combo_id), []).extend([float(x) for x in vals.reshape(-1).tolist()])
        inter = box_area_xyxy(torch.cat([torch.maximum(pred[..., :2], gt[..., :2]), torch.minimum(pred[..., 2:], gt[..., 2:])], dim=-1))
        coverage = inter / box_area_xyxy(gt).clamp_min(1e-6)
        ious_all.append(iou.reshape(-1).detach().cpu())
        coverage_all.append(coverage.reshape(-1).detach().cpu())
        if active.any():
            ious_active.append(iou[active].reshape(-1).detach().cpu())
        if (~active).any():
            ious_inactive.append(iou[~active].reshape(-1).detach().cpu())
        for combo_id, vals in zip(batch_t["combo_id"].detach().cpu().tolist(), iou.detach().cpu()):
            per_combo.setdefault(int(combo_id), []).extend([float(x) for x in vals.reshape(-1).tolist()])
    out = {k: v / max(n_batches, 1) for k, v in totals.items()}
    out["eval_frames"] = float(sum(x.numel() for x in ious_all))
    out["box_iou_mean"] = float(torch.cat(ious_all).mean()) if ious_all else float("nan")
    out["box_iou_active"] = float(torch.cat(ious_active).mean()) if ious_active else float("nan")
    out["box_iou_inactive"] = float(torch.cat(ious_inactive).mean()) if ious_inactive else float("nan")
    out["quad_iou_mean"] = float(torch.cat(quad_ious_all).mean()) if quad_ious_all else float("nan")
    out["quad_iou_active"] = float(torch.cat(quad_ious_active).mean()) if quad_ious_active else float("nan")
    out["quad_iou_inactive"] = float(torch.cat(quad_ious_inactive).mean()) if quad_ious_inactive else float("nan")
    out["target_box_coverage"] = float(torch.cat(coverage_all).mean()) if coverage_all else float("nan")
    for cid, vals in sorted(per_combo.items()):
        out[f"combo_{cid}_box_iou"] = float(np.mean(vals)) if vals else float("nan")
    for cid, vals in sorted(per_combo_quad.items()):
        out[f"combo_{cid}_quad_iou"] = float(np.mean(vals)) if vals else float("nan")
    return out


def clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state.items()}


def clone_state_dict_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def update_ema_state(ema_state: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    model_state = model.state_dict()
    with torch.no_grad():
        for k, v in model_state.items():
            if torch.is_floating_point(v):
                ema_state[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
            else:
                ema_state[k].copy_(v.detach())


@torch.no_grad()
def evaluate_with_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    current_state = clone_state_dict(model.state_dict())
    try:
        model.load_state_dict(state, strict=True)
        return evaluate(model, loader, device, args)
    finally:
        model.load_state_dict(current_state, strict=True)


def tensor_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def parse_variant_list(text: str) -> list[str]:
    variants = [x.strip() for x in str(text).split(",") if x.strip()]
    return variants or ["clean"]


def apply_condition_ablation(batch_t: dict[str, Any], variant: str) -> dict[str, Any]:
    """Return a shallow batch copy with one condition stream ablated for eval.

    This is intentionally eval-only.  It measures whether a trained checkpoint
    actually depends on each condition, without retraining a second stochastic
    run whose seed/noise could hide the effect.
    """
    name = str(variant)
    out = dict(batch_t)
    if name == "clean":
        return out
    if name in {"zero_tokens", "zero_all"} and torch.is_tensor(out.get("tokens")):
        out["tokens"] = torch.zeros_like(out["tokens"])
    if name in {"zero_role", "zero_tool_role", "zero_tool_role_ids", "zero_all"} and torch.is_tensor(
        out.get("role_features_16")
    ):
        out["role_features_16"] = torch.zeros_like(out["role_features_16"])
    if name in {"zero_tool_kpt_all", "zero_tool_role", "zero_tool_role_ids", "zero_all"}:
        if torch.is_tensor(out.get("tool_kpt_heatmaps_8")):
            out["tool_kpt_heatmaps_8"] = torch.zeros_like(out["tool_kpt_heatmaps_8"])
        if torch.is_tensor(out.get("tool_kpt_vec_16")):
            out["tool_kpt_vec_16"] = torch.zeros_like(out["tool_kpt_vec_16"])
    if name == "zero_tool_kpt_heatmaps" and torch.is_tensor(out.get("tool_kpt_heatmaps_8")):
        out["tool_kpt_heatmaps_8"] = torch.zeros_like(out["tool_kpt_heatmaps_8"])
    if name == "zero_tool_kpt_vec" and torch.is_tensor(out.get("tool_kpt_vec_16")):
        out["tool_kpt_vec_16"] = torch.zeros_like(out["tool_kpt_vec_16"])
    if name == "reverse_tool_kpt_time":
        if torch.is_tensor(out.get("tool_kpt_heatmaps_8")):
            out["tool_kpt_heatmaps_8"] = torch.flip(out["tool_kpt_heatmaps_8"], dims=[1])
        if torch.is_tensor(out.get("tool_kpt_vec_16")):
            out["tool_kpt_vec_16"] = torch.flip(out["tool_kpt_vec_16"], dims=[1])
    if name in {"null_ids", "zero_tool_role_ids", "zero_all"}:
        if torch.is_tensor(out.get("tool_id")):
            out["tool_id"] = torch.full_like(out["tool_id"], 8)
        if torch.is_tensor(out.get("act_id")):
            out["act_id"] = torch.full_like(out["act_id"], 8)
        if torch.is_tensor(out.get("combo_id")):
            out["combo_id"] = torch.full_like(out["combo_id"], 16)
    return out


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    data_root = args.data_root.resolve()
    cache_root = (args.cache_root or (data_root / "vjepa_token_cache")).resolve()
    video_root = (args.video_root or (data_root / "video_16_npy_96")).resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    teacher_cache_root = args.teacher_cache_root.resolve() if args.teacher_cache_root is not None else None

    train_ds = H161RoiDataset(
        data_root,
        cache_root,
        "train",
        video_root=video_root,
        tool_mask_root=args.tool_mask_root,
        only_cached=bool(args.only_cached),
        max_samples=int(args.max_train_samples),
        bbox_pad_ratio=float(args.bbox_pad_ratio),
        context_aux_pad_ratio=float(args.context_aux_pad_ratio),
        min_box_size=float(args.min_box_size),
        use_role_features=bool(args.use_role_features),
        use_rgb_sidecar=bool(args.use_rgb_sidecar),
        rgb_sidecar_size=int(args.rgb_sidecar_size),
        use_tool_mask=bool(args.use_tool_mask),
        use_tool_keypoints=bool(args.use_tool_keypoints),
        tool_keypoint_sigma_px=float(args.tool_keypoint_sigma_px),
        allow_missing_tool_mask=(not bool(args.strict_tool_mask)) and bool(args.allow_missing_tool_mask),
        teacher_cache_root=teacher_cache_root,
    )
    val_ds = H161RoiDataset(
        data_root,
        cache_root,
        "val",
        video_root=video_root,
        tool_mask_root=args.tool_mask_root,
        only_cached=bool(args.only_cached),
        max_samples=int(args.max_val_samples),
        bbox_pad_ratio=float(args.bbox_pad_ratio),
        context_aux_pad_ratio=float(args.context_aux_pad_ratio),
        min_box_size=float(args.min_box_size),
        use_role_features=bool(args.use_role_features),
        use_rgb_sidecar=bool(args.use_rgb_sidecar),
        rgb_sidecar_size=int(args.rgb_sidecar_size),
        use_tool_mask=bool(args.use_tool_mask),
        use_tool_keypoints=bool(args.use_tool_keypoints),
        tool_keypoint_sigma_px=float(args.tool_keypoint_sigma_px),
        allow_missing_tool_mask=(not bool(args.strict_tool_mask)) and bool(args.allow_missing_tool_mask),
        teacher_cache_root=teacher_cache_root,
    )
    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)
    if args.model_mode == "anchor_residual":
        if not bool(args.use_tool_keypoints):
            raise ValueError("--model-mode anchor_residual requires --use-tool-keypoints")
        model = H161RoiAnchorResidual(
            hidden_dim=int(args.hidden_dim),
            residual_scale=float(args.residual_scale),
        ).to(device)
    elif args.model_mode == "tool_mask_anchor_residual":
        if not bool(args.use_tool_mask):
            raise ValueError("--model-mode tool_mask_anchor_residual requires --use-tool-mask")
        model = H162ToolMaskAnchorResidual(
            hidden_dim=int(args.hidden_dim),
            residual_scale=float(args.residual_scale),
        ).to(device)
    elif args.model_mode == "temporal_residual":
        model = H161RoiTemporalResidual(
            hidden_dim=int(args.hidden_dim),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
        ).to(device)
    elif args.model_mode == "tool_mask_geom_temporal_residual":
        if not bool(args.use_tool_mask):
            raise ValueError("--model-mode tool_mask_geom_temporal_residual requires --use-tool-mask")
        if bool(args.use_tool_keypoints):
            raise ValueError("--model-mode tool_mask_geom_temporal_residual should not use --use-tool-keypoints")
        model = H170ToolMaskGeomTemporalResidual(
            hidden_dim=int(args.hidden_dim),
            residual_scale=float(args.residual_scale),
        ).to(device)
    elif args.model_mode == "combo_expert_temporal_residual":
        model = H166ComboExpertTemporalResidual(
            hidden_dim=int(args.hidden_dim),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
        ).to(device)
    elif args.model_mode == "rgb_sidecar_temporal_residual":
        if not bool(args.use_rgb_sidecar):
            raise ValueError("--model-mode rgb_sidecar_temporal_residual requires --use-rgb-sidecar")
        model = H167RgbSidecarTemporalResidual(
            hidden_dim=int(args.hidden_dim),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
        ).to(device)
    elif args.model_mode == "temporal_transformer_residual":
        model = H164TemporalTransformerResidual(
            hidden_dim=int(args.hidden_dim),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
        ).to(device)
    elif args.model_mode == "local_refine_residual":
        model = H165LocalRefineResidual(
            hidden_dim=int(args.hidden_dim),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
            use_gates=False,
        ).to(device)
    elif args.model_mode == "local_refine_gated":
        model = H165LocalRefineResidual(
            hidden_dim=int(args.hidden_dim),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
            use_gates=True,
        ).to(device)
    elif args.model_mode == "clip_prior_temporal":
        model = H163ClipPriorTemporalResidual(
            hidden_dim=int(args.hidden_dim),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
            clip_prior_output_mix=float(args.clip_prior_output_mix),
        ).to(device)
    elif args.model_mode == "quad_temporal":
        model = H161QuadTemporalRegion(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
            tool_mask_spatial=not bool(args.tool_mask_geom_only),
        ).to(device)
    elif args.model_mode in {"query_temporal", "query_temporal_dab"}:
        model = H180TokenQueryTemporalRegion(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
            dab_refine=args.model_mode == "query_temporal_dab",
        ).to(device)
    elif args.model_mode == "dab_refine_temporal":
        if not bool(args.use_tool_mask):
            raise ValueError("--model-mode dab_refine_temporal requires --use-tool-mask")
        if bool(args.use_tool_keypoints):
            raise ValueError("--model-mode dab_refine_temporal should not use --use-tool-keypoints")
        model = H185DabRefineTemporalRegion(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
            decoder_layers=int(args.decoder_layers),
            query_noise_scale=float(args.decoder_query_noise_scale),
            stage2_bridge_noise_mode=str(args.stage2_bridge_noise_mode),
            use_tool_trajectory_bias=bool(args.use_tool_trajectory_bias),
        ).to(device)
    elif args.model_mode in {"point_token_temporal", "point_token_local_temporal", "point_token_local_rect_temporal"}:
        if bool(args.use_tool_keypoints):
            raise ValueError(f"--model-mode {args.model_mode} should not use --use-tool-keypoints")
        model = H187PointTokenTemporalRegion(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
            decoder_layers=int(args.decoder_layers),
            query_noise_scale=float(args.decoder_query_noise_scale),
            use_local_point_features=args.model_mode in {"point_token_local_temporal", "point_token_local_rect_temporal"},
            constrain_oriented_rect=args.model_mode == "point_token_local_rect_temporal",
        ).to(device)
    elif args.model_mode == "quad_local_point_temporal":
        model = H191QuadLocalPointTemporalRegion(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
            decoder_layers=int(args.decoder_layers),
            query_noise_scale=float(args.decoder_query_noise_scale),
            dn_query_noise_scale=float(args.dn_refiner_noise_scale),
            dn_query_groups=int(args.dn_refiner_groups),
            dn_query_noise_min_scale=float(args.dn_refiner_noise_min_scale),
            use_proposal_oracle_refiner=float(args.proposal_oracle_refiner_loss_weight) > 0.0,
            proposal_oracle_refiner_noise_scale=float(args.proposal_oracle_refiner_noise_scale),
            stage2_bridge_noise_mode=str(args.stage2_bridge_noise_mode),
            use_tool_trajectory_bias=bool(args.use_tool_trajectory_bias),
            use_tool_trajectory_attn=bool(args.use_tool_trajectory_attn),
            use_tool_corner_query_bias=bool(args.use_tool_corner_query_bias),
            stage1_drop_tool_keypoints=bool(args.stage1_drop_tool_keypoints),
            detach_stage1_refiner=bool(args.detach_stage1_refiner),
            local_grid_refine=bool(args.local_grid_refine),
            local_grid_refine_grid_size=int(args.local_grid_refine_grid_size),
            local_grid_refine_scale=float(args.local_grid_refine_scale),
            local_grid_refine_temperature=float(args.local_grid_refine_temperature),
            local_grid_refine_prior_scale=float(args.local_grid_refine_prior_scale),
            local_grid_refine_prior_sigma=float(args.local_grid_refine_prior_sigma),
            local_grid_refine_detach_base=bool(args.local_grid_refine_detach_base),
            local_grid_final_refine=not bool(args.disable_final_local_grid_refine),
            local_grid_confidence_score=bool(args.local_grid_confidence_score),
            local_grid_confidence_scale=float(args.local_grid_confidence_scale),
            local_grid_confidence_detach=bool(args.local_grid_confidence_detach),
            candidate_mask_quality=bool(args.candidate_mask_quality),
            candidate_mask_grid_size=int(args.candidate_mask_grid_size),
            candidate_mask_score_scale=float(args.candidate_mask_score_scale),
            candidate_mask_score_detach=bool(args.candidate_mask_score_detach),
            use_rgb_sidecar=bool(args.use_rgb_sidecar),
        ).to(device)
    elif args.model_mode == "proposal_point_temporal":
        model = H210ProposalPointTemporalRegion(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
            decoder_layers=int(args.decoder_layers),
            query_noise_scale=float(args.decoder_query_noise_scale),
            dn_query_noise_scale=float(args.dn_refiner_noise_scale),
            dn_query_groups=int(args.dn_refiner_groups),
            dn_query_noise_min_scale=float(args.dn_refiner_noise_min_scale),
            use_proposal_oracle_refiner=float(args.proposal_oracle_refiner_loss_weight) > 0.0,
            proposal_oracle_refiner_noise_scale=float(args.proposal_oracle_refiner_noise_scale),
            stage2_bridge_noise_mode=str(args.stage2_bridge_noise_mode),
            use_tool_trajectory_bias=bool(args.use_tool_trajectory_bias),
            use_tool_trajectory_attn=bool(args.use_tool_trajectory_attn),
            use_tool_corner_query_bias=bool(args.use_tool_corner_query_bias),
            stage1_drop_tool_keypoints=bool(args.stage1_drop_tool_keypoints),
            detach_stage1_refiner=bool(args.detach_stage1_refiner),
            proposal_count=int(args.proposal_count),
            proposal_selection_mode=str(args.proposal_selection_mode),
            proposal_selection_temperature=float(args.proposal_selection_temperature),
            proposal_selection_topk=int(args.proposal_selection_topk),
            proposal_selection_straight_through=bool(args.proposal_selection_straight_through),
            proposal_selection_train_logit_noise_std=float(args.proposal_selection_train_logit_noise_std),
            proposal_selection_train_drop_prob=float(args.proposal_selection_train_drop_prob),
            proposal_geom_temporal_fusion=bool(args.proposal_geom_temporal_fusion),
            proposal_geom_temporal_fusion_scale=float(args.proposal_geom_temporal_fusion_scale),
            proposal_geom_temporal_center_scale=float(args.proposal_geom_temporal_center_scale),
            proposal_geom_temporal_source_penalty=float(args.proposal_geom_temporal_source_penalty),
            proposal_clip_context_fusion=bool(args.proposal_clip_context_fusion),
            proposal_clip_context_scale=float(args.proposal_clip_context_scale),
            proposal_clip_context_dropout=float(args.proposal_clip_context_dropout),
            proposal_frame_anchor_memory=bool(args.proposal_frame_anchor_memory),
            proposal_frame_anchor_scale=float(args.proposal_frame_anchor_scale),
            proposal_frame_anchor_topk=int(args.proposal_frame_anchor_topk),
            proposal_frame_anchor_temperature=float(args.proposal_frame_anchor_temperature),
            proposal_frame_anchor_candidate_temperature=float(args.proposal_frame_anchor_candidate_temperature),
            proposal_final_fusion=bool(args.proposal_final_fusion),
            proposal_final_fusion_scale=float(args.proposal_final_fusion_scale),
            proposal_quality_readout=bool(args.proposal_quality_readout),
            proposal_quality_readout_detach=bool(args.proposal_quality_readout_detach),
            proposal_quality_readout_replace_logits=bool(args.proposal_quality_readout_replace_logits),
            proposal_quality_readout_traj_context=bool(args.proposal_quality_readout_traj_context),
            proposal_quality_readout_points=str(args.proposal_quality_readout_points),
            proposal_quality_readout_pool=str(args.proposal_quality_readout_pool),
            proposal_support_overlap_score=bool(args.proposal_support_overlap_score),
            proposal_support_overlap_scale=float(args.proposal_support_overlap_scale),
            proposal_support_overlap_grid_size=int(args.proposal_support_overlap_grid_size),
            proposal_support_overlap_detach=bool(args.proposal_support_overlap_detach),
            proposal_support_geom_score=bool(args.proposal_support_geom_score),
            proposal_support_geom_scale=float(args.proposal_support_geom_scale),
            proposal_support_geom_grid_size=int(args.proposal_support_geom_grid_size),
            proposal_support_geom_center_tau=float(args.proposal_support_geom_center_tau),
            proposal_support_geom_area_tau=float(args.proposal_support_geom_area_tau),
            proposal_support_geom_detach=bool(args.proposal_support_geom_detach),
            proposal_support_moment_proposal=bool(args.proposal_support_moment_proposal),
            proposal_support_moment_top_frac=float(args.proposal_support_moment_top_frac),
            proposal_support_moment_size_scale=float(args.proposal_support_moment_size_scale),
            proposal_support_moment_temperature=float(args.proposal_support_moment_temperature),
            proposal_support_moment_detach=bool(args.proposal_support_moment_detach),
            proposal_decoupled_selector=bool(args.proposal_decoupled_selector),
            proposal_decoupled_selector_detach=bool(args.proposal_decoupled_selector_detach),
            proposal_decoupled_selector_replace_logits=bool(args.proposal_decoupled_selector_replace_logits),
            proposal_decoupled_selector_scale=float(args.proposal_decoupled_selector_scale),
            proposal_decoupled_selector_base_scale=float(args.proposal_decoupled_selector_base_scale),
            proposal_decoupled_selector_readout=bool(args.proposal_decoupled_selector_readout),
            proposal_oicr_refine_head=bool(args.proposal_oicr_refine_head),
            proposal_oicr_refine_use_for_selection=bool(args.proposal_oicr_refine_use_for_selection),
            proposal_handoff_oicr_refine_times=int(args.proposal_handoff_oicr_refine_times),
            proposal_handoff_oicr_refine_scale=float(args.proposal_handoff_oicr_refine_scale),
            proposal_handoff_oicr_refine_aggregate=str(
                args.proposal_handoff_oicr_refine_aggregate
            ),
            proposal_handoff_oicr_refine_detach=bool(args.proposal_handoff_oicr_refine_detach),
            proposal_handoff_oicr_refine_use_for_selection=bool(
                args.proposal_handoff_oicr_refine_use_for_selection
            ),
            refined_quality_readout=bool(args.refined_quality_readout),
            refined_quality_readout_detach=bool(args.refined_quality_readout_detach),
            refined_quality_readout_replace_logits=bool(args.refined_quality_readout_replace_logits),
            refined_quality_readout_points=str(args.refined_quality_readout_points),
            refiner_aligned_context=bool(args.refiner_aligned_context),
            refiner_aligned_context_points=str(args.refiner_aligned_context_points),
            refiner_aligned_context_scale=float(args.refiner_aligned_context_scale),
            proposal_pointset_refine=bool(args.proposal_pointset_refine),
            proposal_pointset_refine_scale=float(args.proposal_pointset_refine_scale),
            proposal_pointset_refine_grad_mul=float(args.proposal_pointset_refine_grad_mul),
            proposal_pointset_quality_scale=float(args.proposal_pointset_quality_scale),
            proposal_pointset_refine_update_token=bool(args.proposal_pointset_refine_update_token),
            proposal_set_refiner=bool(args.proposal_set_refiner),
            proposal_set_refiner_topk=int(args.proposal_set_refiner_topk),
            proposal_set_refiner_scale=float(args.proposal_set_refiner_scale),
            proposal_set_refiner_temperature=float(args.proposal_set_refiner_temperature),
            proposal_set_refiner_detach=bool(args.proposal_set_refiner_detach),
            proposal_local_corr_score=bool(args.proposal_local_corr_score),
            proposal_local_corr_scale=float(args.proposal_local_corr_scale),
            proposal_local_corr_grid_size=int(args.proposal_local_corr_grid_size),
            proposal_local_corr_temperature=float(args.proposal_local_corr_temperature),
            proposal_local_corr_detach=bool(args.proposal_local_corr_detach),
            proposal_active_tip_selector=bool(args.proposal_active_tip_selector),
            proposal_active_tip_selector_scale=float(args.proposal_active_tip_selector_scale),
            proposal_active_tip_selector_aux_only=bool(args.proposal_active_tip_selector_aux_only),
            proposal_active_tip_selector_eval_fuse_scale=float(args.proposal_active_tip_selector_eval_fuse_scale),
            proposal_active_tip_feature_fusion=bool(args.proposal_active_tip_feature_fusion),
            proposal_active_tip_feature_scale=float(args.proposal_active_tip_feature_scale),
            proposal_trajectory_rerank=bool(args.proposal_trajectory_rerank),
            proposal_trajectory_rerank_scale=float(args.proposal_trajectory_rerank_scale),
            proposal_trajectory_refine=bool(args.proposal_trajectory_refine),
            proposal_trajectory_refine_scale=float(args.proposal_trajectory_refine_scale),
            refine_all_proposals=bool(args.refine_all_proposals),
            refine_proposal_topk=int(args.refine_proposal_topk),
            refine_recollect_stage1=bool(args.refine_recollect_stage1),
            refine_recollect_refined_base_only=bool(args.refine_recollect_refined_base_only),
            condition_prior_proposal=bool(args.condition_prior_proposal),
            h259_prior_proposal=bool(args.use_h259_prior_proposal),
            trajectory_query_proposal_count=int(args.trajectory_query_proposal_count),
            trajectory_query_output_mode=str(args.trajectory_query_output_mode),
            trajectory_readout_bias=bool(args.use_trajectory_readout_bias),
            condition_prior_readout_bias=bool(args.use_condition_prior_readout_bias),
            condition_prior_readout_scale=float(args.condition_prior_readout_scale),
            trajectory_anchor_logit_bias_scale=float(args.trajectory_anchor_logit_bias_scale),
            trajectory_anchor_logit_bias_radius=float(args.trajectory_anchor_logit_bias_radius),
            trajectory_anchor_axis_bias_weight=float(args.trajectory_anchor_axis_bias_weight),
            corner_heatmap_proposal=bool(args.corner_heatmap_proposal),
            corner_heatmap_proposal_temperature=float(args.corner_heatmap_proposal_temperature),
            corner_heatmap_proposal_detach=bool(args.corner_heatmap_proposal_detach),
            proposal_source_calibration=bool(args.proposal_source_calibration),
            proposal_source_calibration_scale=float(args.proposal_source_calibration_scale),
            proposal_handoff_source_calibration=bool(args.proposal_handoff_source_calibration),
            proposal_handoff_source_calibration_scale=float(args.proposal_handoff_source_calibration_scale),
            proposal_handoff_source4_init=float(args.proposal_handoff_source4_init),
            proposal_handoff_source_router=bool(args.proposal_handoff_source_router),
            proposal_handoff_source_router_scale=float(args.proposal_handoff_source_router_scale),
            proposal_handoff_source_router_candidate_select_scale=float(
                args.proposal_handoff_source_router_candidate_select_scale
            ),
            proposal_handoff_source5_readout=bool(args.proposal_handoff_source5_readout),
            proposal_handoff_source5_readout_scale=float(
                args.proposal_handoff_source5_readout_scale
            ),
            proposal_handoff_source5_readout_grid_size=int(
                args.proposal_handoff_source5_readout_grid_size
            ),
            proposal_handoff_source5_readout_sources=str(
                args.proposal_handoff_source5_readout_sources
            ),
            proposal_handoff_source5_readout_detach=bool(
                args.proposal_handoff_source5_readout_detach
            ),
            proposal_handoff_mask_quality=bool(args.proposal_handoff_mask_quality),
            proposal_handoff_mask_quality_scale=float(
                args.proposal_handoff_mask_quality_scale
            ),
            proposal_handoff_mask_grid_size=int(args.proposal_handoff_mask_grid_size),
            proposal_handoff_mask_expand_scale=float(
                args.proposal_handoff_mask_expand_scale
            ),
            proposal_handoff_mask_refine_scale=float(
                args.proposal_handoff_mask_refine_scale
            ),
            proposal_handoff_mask_refine_size_scale=float(
                args.proposal_handoff_mask_refine_size_scale
            ),
            proposal_handoff_mask_refine_min_size_px=float(
                args.proposal_handoff_mask_refine_min_size_px
            ),
            proposal_handoff_mask_refine_max_ratio=float(
                args.proposal_handoff_mask_refine_max_ratio
            ),
            proposal_handoff_mask_quality_sources=str(
                args.proposal_handoff_mask_quality_sources
            ),
            proposal_handoff_mask_detach=bool(args.proposal_handoff_mask_detach),
            proposal_handoff_mask_score_detach=bool(
                args.proposal_handoff_mask_score_detach
            ),
            proposal_handoff_rank_adaptive_scorer=bool(
                args.proposal_handoff_rank_adaptive_scorer
            ),
            proposal_handoff_rank_adaptive_scale=float(
                args.proposal_handoff_rank_adaptive_scale
            ),
            proposal_handoff_rank_adaptive_max_rank=int(
                args.proposal_handoff_rank_adaptive_max_rank
            ),
            proposal_handoff_rank_adaptive_detach=bool(
                args.proposal_handoff_rank_adaptive_detach
            ),
            proposal_handoff_sparse_set_scorer=bool(
                args.proposal_handoff_sparse_set_scorer
            ),
            proposal_handoff_sparse_set_scorer_scale=float(
                args.proposal_handoff_sparse_set_scorer_scale
            ),
            proposal_handoff_sparse_set_scorer_sources=str(
                args.proposal_handoff_sparse_set_scorer_sources
            ),
            proposal_handoff_sparse_set_scorer_detach=bool(
                args.proposal_handoff_sparse_set_scorer_detach
            ),
            proposal_handoff_source5_refiner=bool(args.proposal_handoff_source5_refiner),
            proposal_handoff_source5_refiner_topk=int(
                args.proposal_handoff_source5_refiner_topk
            ),
            proposal_handoff_source5_refiner_temperature=float(
                args.proposal_handoff_source5_refiner_temperature
            ),
            proposal_handoff_source5_refiner_scale=float(
                args.proposal_handoff_source5_refiner_scale
            ),
            proposal_handoff_source5_refiner_logit_scale=float(
                args.proposal_handoff_source5_refiner_logit_scale
            ),
            proposal_handoff_source5_refiner_init_logit_bias=float(
                args.proposal_handoff_source5_refiner_init_logit_bias
            ),
            proposal_handoff_source5_refiner_detach=bool(
                args.proposal_handoff_source5_refiner_detach
            ),
            proposal_handoff_source5_refiner_source_id=int(
                args.proposal_handoff_source5_refiner_source_id
            ),
            proposal_handoff_source5_refiner_output_source_id=int(
                args.proposal_handoff_source5_refiner_output_source_id
            ),
            proposal_handoff_topk_query_refiner=bool(
                args.proposal_handoff_topk_query_refiner
            ),
            proposal_handoff_topk_query_sources=str(
                args.proposal_handoff_topk_query_sources
            ),
            proposal_handoff_topk_query_topk_per_source=int(
                args.proposal_handoff_topk_query_topk_per_source
            ),
            proposal_handoff_topk_query_grid_size=int(
                args.proposal_handoff_topk_query_grid_size
            ),
            proposal_handoff_topk_query_scale=float(
                args.proposal_handoff_topk_query_scale
            ),
            proposal_handoff_topk_query_logit_scale=float(
                args.proposal_handoff_topk_query_logit_scale
            ),
            proposal_handoff_topk_query_output_source_id=int(
                args.proposal_handoff_topk_query_output_source_id
            ),
            proposal_handoff_topk_query_replace_logits=bool(
                args.proposal_handoff_topk_query_replace_logits
            ),
            proposal_handoff_topk_query_detach=bool(
                args.proposal_handoff_topk_query_detach
            ),
            proposal_handoff_final_fusion=bool(args.proposal_handoff_final_fusion),
            proposal_handoff_final_fusion_scale=float(args.proposal_handoff_final_fusion_scale),
            proposal_handoff_detach_selection_geometry=bool(
                args.proposal_handoff_detach_selection_geometry
            ),
            proposal_handoff_source_representatives_only=bool(
                args.proposal_handoff_source_representatives_only
            ),
            proposal_handoff_source5_topk=int(args.proposal_handoff_source5_topk),
            proposal_handoff_router_gated_source=bool(args.proposal_handoff_router_gated_source),
            proposal_handoff_router_normalized_select=bool(
                args.proposal_handoff_router_normalized_select
            ),
            proposal_handoff_router_normalized_source_scale=float(
                args.proposal_handoff_router_normalized_source_scale
            ),
            proposal_handoff_router_normalized_within_scale=float(
                args.proposal_handoff_router_normalized_within_scale
            ),
            token_affinity_proposal=bool(args.token_affinity_proposal),
            token_affinity_proposal_count=int(args.token_affinity_proposal_count),
            token_affinity_proposal_temperature=float(args.token_affinity_proposal_temperature),
            token_affinity_proposal_top_frac=float(args.token_affinity_proposal_top_frac),
            token_affinity_proposal_size_scale=float(args.token_affinity_proposal_size_scale),
            token_affinity_proposal_extra_top_fracs=str(args.token_affinity_proposal_extra_top_fracs),
            token_affinity_proposal_extra_size_scales=str(args.token_affinity_proposal_extra_size_scales),
            token_affinity_proposal_detach=bool(args.token_affinity_proposal_detach),
            topology_contact_template_proposal=bool(args.topology_contact_template_proposal),
            topology_contact_template_sizes_px=str(args.topology_contact_template_sizes_px),
            topology_contact_template_offsets_px=str(args.topology_contact_template_offsets_px),
            topology_contact_template_logit_bias=float(args.topology_contact_template_logit_bias),
            topology_contact_template_index_embedding=bool(
                args.topology_contact_template_index_embedding
            ),
            topology_contact_template_combo_bias=bool(args.topology_contact_template_combo_bias),
            topology_contact_template_combo_bias_scale=float(
                args.topology_contact_template_combo_bias_scale
            ),
            topology_contact_template_combo_allowlist=str(
                args.topology_contact_template_combo_allowlist
            ),
            local_grid_refine=bool(args.local_grid_refine),
            local_grid_refine_grid_size=int(args.local_grid_refine_grid_size),
            local_grid_refine_scale=float(args.local_grid_refine_scale),
            local_grid_refine_temperature=float(args.local_grid_refine_temperature),
            local_grid_refine_prior_scale=float(args.local_grid_refine_prior_scale),
            local_grid_refine_prior_sigma=float(args.local_grid_refine_prior_sigma),
            local_grid_refine_detach_base=bool(args.local_grid_refine_detach_base),
            local_grid_final_refine=not bool(args.disable_final_local_grid_refine),
            local_grid_confidence_score=bool(args.local_grid_confidence_score),
            local_grid_confidence_scale=float(args.local_grid_confidence_scale),
            local_grid_confidence_detach=bool(args.local_grid_confidence_detach),
            candidate_mask_quality=bool(args.candidate_mask_quality),
            candidate_mask_grid_size=int(args.candidate_mask_grid_size),
            candidate_mask_score_scale=float(args.candidate_mask_score_scale),
            candidate_mask_score_detach=bool(args.candidate_mask_score_detach),
            use_rgb_sidecar=bool(args.use_rgb_sidecar),
        ).to(device)
    elif args.model_mode == "trajectory_prior_temporal":
        if not bool(args.use_tool_keypoints):
            raise ValueError("--model-mode trajectory_prior_temporal requires --use-tool-keypoints")
        model = H259TrajectoryPriorTemporalRegion(
            hidden_dim=int(args.hidden_dim),
            residual_scale=float(args.residual_scale),
        ).to(device)
    elif args.model_mode == "trajectory_prior_point_temporal":
        if not bool(args.use_tool_keypoints):
            raise ValueError("--model-mode trajectory_prior_point_temporal requires --use-tool-keypoints")
        model = H270TrajectoryPriorPointTemporalRegion(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
            decoder_layers=int(args.decoder_layers),
            query_noise_scale=float(args.decoder_query_noise_scale),
            dn_query_noise_scale=float(args.dn_refiner_noise_scale),
            stage2_bridge_noise_mode=str(args.stage2_bridge_noise_mode),
            use_tool_trajectory_bias=bool(args.use_tool_trajectory_bias),
            use_tool_trajectory_attn=bool(args.use_tool_trajectory_attn),
            use_tool_corner_query_bias=bool(args.use_tool_corner_query_bias),
            detach_stage1_refiner=bool(args.detach_stage1_refiner),
        ).to(device)
    elif args.model_mode == "traj_contact_point_temporal":
        model = H219TrajectoryContactPointTemporalRegion(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            residual_scale=float(args.residual_scale),
            decoder_layers=int(args.decoder_layers),
            query_noise_scale=float(args.decoder_query_noise_scale),
            stage2_bridge_noise_mode=str(args.stage2_bridge_noise_mode),
            use_tool_trajectory_bias=bool(args.use_tool_trajectory_bias),
            use_tool_trajectory_attn=bool(args.use_tool_trajectory_attn),
            stage1_drop_tool_keypoints=bool(args.stage1_drop_tool_keypoints),
            detach_stage1_refiner=bool(args.detach_stage1_refiner),
        ).to(device)
    elif args.model_mode == "context_refine":
        model = H161RoiContextRefine(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
        ).to(device)
    elif args.model_mode == "box_dist":
        model = H161RoiBoxDist(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            n_bins=int(args.box_dist_bins),
        ).to(device)
    elif args.model_mode == "corner_heatmap":
        model = H162CornerHeatmapRegion(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
        ).to(device)
    elif args.model_mode == "support_heatmap":
        model = H162SupportHeatmapRegion(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
        ).to(device)
    elif args.model_mode == "support_moment":
        model = H162SupportMomentRegion(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
            moment_size_scale=float(args.moment_size_scale),
        ).to(device)
    elif args.model_mode == "edge_dist":
        model = H161RoiEdgeDist(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
        ).to(device)
    else:
        model = H161RoiCondLite(
            hidden_dim=int(args.hidden_dim),
            use_tool_mask=bool(args.use_tool_mask),
            use_tool_keypoints=bool(args.use_tool_keypoints),
        ).to(device)
    if str(args.token_dropout_mode) == "mask_token" and not hasattr(model, "token_mask_value"):
        model.register_parameter("token_mask_value", nn.Parameter(torch.zeros(1024, device=device)))
    if args.eval_only_checkpoint is not None:
        ckpt_path = args.eval_only_checkpoint.resolve()
        ensemble_extra = [
            Path(x.strip()).resolve()
            for x in str(getattr(args, "eval_ensemble_checkpoints", "")).split(",")
            if x.strip()
        ]
        ensemble_paths = [ckpt_path] + ensemble_extra
        ensemble_ckpts: list[dict[str, Any] | dict[str, torch.Tensor]] = []
        if ensemble_extra:
            members: list[nn.Module] = []
            for idx, member_path in enumerate(ensemble_paths):
                member = model if idx == 0 else copy.deepcopy(model).to(device)
                member_ckpt = torch.load(member_path, map_location=device, weights_only=False)
                member_state = member_ckpt["model"] if isinstance(member_ckpt, dict) and "model" in member_ckpt else member_ckpt
                member.load_state_dict(member_state, strict=True)
                member.eval()
                members.append(member)
                ensemble_ckpts.append(member_ckpt)
            model = EvalCheckpointEnsemble(
                members,
                mode=str(args.eval_ensemble_mode),
                temperature=float(args.eval_ensemble_temperature),
                consensus_threshold=float(args.eval_ensemble_consensus_threshold),
                blend_alpha=float(args.eval_ensemble_blend_alpha),
            ).to(device)
            ckpt = ensemble_ckpts[0]
        else:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state, strict=True)
        variants = parse_variant_list(args.condition_ablation_variants) if bool(args.eval_condition_ablations) else ["clean"]
        eval_rows: dict[str, dict[str, float]] = {}
        clean_quad = None
        clean_box = None
        for variant in variants:
            metrics = evaluate(model, val_loader, device, args, condition_ablation=variant)
            if variant == "clean":
                clean_quad = float(metrics.get("quad_iou_mean", float("nan")))
                clean_box = float(metrics.get("box_iou_mean", float("nan")))
            eval_rows[variant] = metrics
        if clean_quad is None and "clean" in eval_rows:
            clean_quad = float(eval_rows["clean"].get("quad_iou_mean", float("nan")))
            clean_box = float(eval_rows["clean"].get("box_iou_mean", float("nan")))
        deltas: dict[str, dict[str, float]] = {}
        for variant, metrics in eval_rows.items():
            deltas[variant] = {
                "quad_iou_drop_vs_clean": float(clean_quad - float(metrics.get("quad_iou_mean", float("nan")))),
                "box_iou_drop_vs_clean": float(clean_box - float(metrics.get("box_iou_mean", float("nan")))),
            }
        payload = {
            "checkpoint": str(ckpt_path),
            "ensemble_checkpoints": [str(path) for path in ensemble_paths],
            "ensemble_mode": str(args.eval_ensemble_mode) if ensemble_extra else None,
            "ensemble_temperature": float(args.eval_ensemble_temperature) if ensemble_extra else None,
            "selection_source": ckpt.get("selection_source") if isinstance(ckpt, dict) else None,
            "checkpoint_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "variants": eval_rows,
            "deltas": deltas,
        }
        (out_dir / "eval_condition_ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return
    if args.init_checkpoint is not None:
        init_path = args.init_checkpoint.resolve()
        init_ckpt = torch.load(init_path, map_location=device, weights_only=False)
        init_state = init_ckpt["model"] if isinstance(init_ckpt, dict) and "model" in init_ckpt else init_ckpt
        model.load_state_dict(init_state, strict=True)
        print(f"[h161-roi] initialized from {init_path}", flush=True)
    teacher_model: nn.Module | None = None
    teacher_paths = [
        Path(x.strip()).resolve()
        for x in str(getattr(args, "teacher_ensemble_checkpoints", "")).split(",")
        if x.strip()
    ]
    if teacher_paths and (
        float(getattr(args, "teacher_quad_distill_weight", 0.0)) > 0.0
        or float(getattr(args, "teacher_box_distill_weight", 0.0)) > 0.0
        or args.write_teacher_cache_root is not None
    ):
        teacher_members: list[nn.Module] = []
        for teacher_path in teacher_paths:
            member = copy.deepcopy(model).to(device)
            teacher_ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
            teacher_state = teacher_ckpt["model"] if isinstance(teacher_ckpt, dict) and "model" in teacher_ckpt else teacher_ckpt
            member.load_state_dict(teacher_state, strict=True)
            member.eval()
            for param in member.parameters():
                param.requires_grad_(False)
            teacher_members.append(member)
        if len(teacher_members) == 1:
            teacher_model = teacher_members[0]
        else:
            teacher_model = EvalCheckpointEnsemble(
                teacher_members,
                mode=str(args.teacher_ensemble_mode),
                temperature=float(args.teacher_ensemble_temperature),
                consensus_threshold=float(args.teacher_ensemble_consensus_threshold),
                blend_alpha=float(args.teacher_ensemble_blend_alpha),
            ).to(device)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad_(False)
        print(
            "[h161-roi] teacher_distill "
            f"members={len(teacher_members)} mode={args.teacher_ensemble_mode} "
            f"quad_w={float(args.teacher_quad_distill_weight)} "
            f"box_w={float(args.teacher_box_distill_weight)}",
            flush=True,
        )
    if args.write_teacher_cache_root is not None:
        if teacher_model is None:
            raise ValueError("--write-teacher-cache-root requires --teacher-ensemble-checkpoints")
        cache_out = args.write_teacher_cache_root.resolve()
        counts = write_teacher_cache(
            teacher_model,
            {"train": train_ds, "val": val_ds},
            args,
            device,
            cache_out,
            splits={x.strip() for x in str(args.write_teacher_cache_splits).split(",") if x.strip()},
            include_proposals=bool(args.write_teacher_cache_proposals),
        )
        payload = {"teacher_cache_root": str(cache_out), "counts": counts}
        (out_dir / "teacher_cache_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")
    ema_decay = float(args.ema_decay)
    ema_state = clone_state_dict(model.state_dict()) if ema_decay > 0.0 else None
    best_iou = -1.0
    best_proposal_oracle = -1.0
    history: list[dict[str, Any]] = []
    print(
        f"[h161-roi] data={data_root} cache={cache_root} train={len(train_ds)} val={len(val_ds)} "
        f"device={device} mode={args.model_mode} role={args.use_role_features} rgb={args.use_rgb_sidecar} "
        f"video={video_root} tool_mask={args.use_tool_mask} "
        f"tool_keypoints={args.use_tool_keypoints} anchor_w={args.anchor_loss_weight} ema_decay={ema_decay}",
        flush=True,
    )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        t0 = time.time()
        meter: dict[str, float] = {}
        n = 0
        for step, batch in enumerate(train_loader, start=1):
            batch_t = tensor_batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp) and device.type == "cuda"):
                reg_scale = regularization_scale(args, epoch)
                tokens_in = batch_t["tokens"]
                if bool(args.force_zero_tokens):
                    tokens_in = torch.zeros_like(tokens_in)
                role_in = batch_t["role_features_16"]
                tool_hm_in = batch_t["tool_kpt_heatmaps_8"]
                tool_vec_in = batch_t["tool_kpt_vec_16"]
                if not bool(args.force_zero_tokens):
                    tokens_in = apply_token_regularization(
                        tokens_in,
                        args,
                        reg_scale,
                        mask_token=getattr(model, "token_mask_value", None),
                    )
                if float(args.condition_dropout) * reg_scale > 0.0:
                    p_drop = min(max(float(args.condition_dropout) * reg_scale, 0.0), 0.95)
                    keep_role = (torch.rand(role_in.shape[0], 1, 1, device=role_in.device) > p_drop).to(role_in.dtype)
                    keep_tool = (torch.rand(tool_vec_in.shape[0], 1, 1, device=tool_vec_in.device) > p_drop).to(tool_vec_in.dtype)
                    role_in = role_in * keep_role
                    tool_vec_in = tool_vec_in * keep_tool
                    tool_hm_in = tool_hm_in * keep_tool.view(tool_hm_in.shape[0], 1, 1, 1, 1)
                if bool(args.use_tool_keypoints):
                    tool_hm_in, tool_vec_in = apply_tool_keypoint_regularization(
                        tool_hm_in,
                        tool_vec_in,
                        args,
                        reg_scale,
                    )
                tool_mask_in = batch_t["tool_mask_8"]
                if bool(args.use_tool_mask):
                    tool_mask_in = apply_tool_mask_regularization(tool_mask_in, args, reg_scale)
                tool_id_in, act_id_in, combo_id_in = corrupt_condition_ids(
                    batch_t["tool_id"],
                    batch_t["act_id"],
                    batch_t["combo_id"],
                    min(max(float(args.condition_id_dropout) * reg_scale, 0.0), 0.95),
                    mode=str(args.condition_id_drop_mode),
                )
                outputs = forward_roi_model_for_batch(
                    model,
                    args,
                    batch_t,
                    tokens_in=tokens_in,
                    role_in=role_in,
                    tool_mask_in=tool_mask_in,
                    tool_hm_in=tool_hm_in,
                    tool_vec_in=tool_vec_in,
                    tool_id_in=tool_id_in,
                    act_id_in=act_id_in,
                    combo_id_in=combo_id_in,
                )
                loss, metrics = compute_loss(outputs, batch_t, args)
                teacher_quad_w = float(getattr(args, "teacher_quad_distill_weight", 0.0))
                teacher_box_w = float(getattr(args, "teacher_box_distill_weight", 0.0))
                if teacher_quad_w > 0.0 or teacher_box_w > 0.0:
                    teacher_outputs: dict[str, torch.Tensor] | None = None
                    with torch.no_grad():
                        if teacher_model is not None:
                            teacher_model.eval()
                            teacher_outputs = forward_roi_model_for_batch(teacher_model, args, batch_t)
                        elif (
                            "teacher_distill_mask" in batch_t
                            and float(batch_t["teacher_distill_mask"].float().max().detach().cpu()) > 0.0
                        ):
                            teacher_outputs = {
                                "quad_xy": batch_t["teacher_quad_xy"],
                                "boxes_cxcywh": batch_t["teacher_boxes_cxcywh"],
                            }
                    if teacher_outputs is None:
                        raise ValueError(
                            "teacher distill weights are enabled but neither online teacher nor teacher cache is available"
                        )
                    active_teacher = batch_t["active_16"].to(outputs["boxes_cxcywh"].device)
                    if bool(args.use_role_loss_weighting):
                        teacher_frame_w = torch.where(
                            active_teacher > 0.5,
                            torch.full_like(active_teacher, float(args.active_loss_weight)),
                            torch.full_like(active_teacher, float(args.inactive_loss_weight)),
                        )
                    else:
                        teacher_frame_w = torch.ones_like(active_teacher)
                    if "teacher_distill_mask" in batch_t:
                        teacher_frame_w = teacher_frame_w * batch_t["teacher_distill_mask"].to(
                            device=teacher_frame_w.device,
                            dtype=teacher_frame_w.dtype,
                        )
                    if teacher_quad_w > 0.0 and "quad_xy" in outputs and "quad_xy" in teacher_outputs:
                        teacher_quad = teacher_outputs["quad_xy"].float().detach().clamp(0.0, 1.0)
                        student_quad = outputs["quad_xy"].float()
                        teacher_quad_loss = weighted_mean(
                            F.smooth_l1_loss(student_quad, teacher_quad, reduction="none").sum(dim=(-1, -2)),
                            teacher_frame_w,
                        )
                        loss = loss + teacher_quad_w * teacher_quad_loss
                        metrics["teacher_quad_distill"] = float(teacher_quad_loss.detach())
                        with torch.no_grad():
                            tq_grid = min(max(int(args.quad_iou_grid_size), 8), 32)
                            metrics["teacher_quad_iou"] = float(
                                quad_iou_approx(
                                    student_quad.detach(),
                                    teacher_quad,
                                    grid_size=tq_grid,
                                )
                                .mean()
                                .detach()
                            )
                    if teacher_box_w > 0.0 and "boxes_cxcywh" in teacher_outputs:
                        teacher_box = teacher_outputs["boxes_cxcywh"].float().detach()
                        student_box = outputs["boxes_cxcywh"].float()
                        teacher_box_loss = weighted_mean(
                            F.smooth_l1_loss(student_box, teacher_box, reduction="none").sum(dim=-1),
                            teacher_frame_w,
                        )
                        loss = loss + teacher_box_w * teacher_box_loss
                        metrics["teacher_box_distill"] = float(teacher_box_loss.detach())
                clean_pass_w = float(args.clean_pass_loss_weight)
                consistency_w = float(args.clean_corrupt_consistency_weight)
                if clean_pass_w > 0.0 or consistency_w > 0.0:
                    clean_tokens = torch.zeros_like(batch_t["tokens"]) if bool(args.force_zero_tokens) else batch_t["tokens"]
                    outputs_clean = forward_roi_model_for_batch(
                        model,
                        args,
                        batch_t,
                        tokens_in=clean_tokens,
                        role_in=batch_t["role_features_16"],
                        tool_mask_in=batch_t["tool_mask_8"],
                        tool_hm_in=batch_t["tool_kpt_heatmaps_8"],
                        tool_vec_in=batch_t["tool_kpt_vec_16"],
                        tool_id_in=batch_t["tool_id"],
                        act_id_in=batch_t["act_id"],
                        combo_id_in=batch_t["combo_id"],
                    )
                    if clean_pass_w > 0.0:
                        clean_loss, clean_metrics = compute_loss(outputs_clean, batch_t, args)
                        loss = loss + clean_pass_w * clean_loss
                        metrics["clean_loss"] = float(clean_loss.detach())
                        metrics["clean_box_iou"] = float(clean_metrics.get("box_iou", 0.0))
                    if consistency_w > 0.0:
                        consistency = F.smooth_l1_loss(
                            outputs["boxes_cxcywh"].float(),
                            outputs_clean["boxes_cxcywh"].float().detach(),
                        )
                        if "quad_xy" in outputs and "quad_xy" in outputs_clean:
                            consistency = consistency + 0.5 * F.smooth_l1_loss(
                                outputs["quad_xy"].float(),
                                outputs_clean["quad_xy"].float().detach(),
                            )
                        loss = loss + consistency_w * consistency
                        metrics["clean_corrupt_consistency"] = float(consistency.detach())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch} step={step}: {metrics}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            if ema_state is not None:
                update_ema_state(ema_state, model, ema_decay)
            for k, v in metrics.items():
                meter[k] = meter.get(k, 0.0) + float(v)
            n += 1
            if step % max(int(args.log_interval), 1) == 0:
                avg = {k: v / max(n, 1) for k, v in meter.items()}
                print(f"[epoch {epoch:03d} step {step:04d}] train {avg}", flush=True)
        train_avg = {f"train_{k}": v / max(n, 1) for k, v in meter.items()}
        raw_val = evaluate(model, val_loader, device, args)
        if ema_state is not None:
            val = evaluate_with_state(model, ema_state, val_loader, device, args)
            rec = {
                "epoch": epoch,
                "seconds": time.time() - t0,
                "val_selection_source": "ema",
                **train_avg,
                **{f"raw_val_{k}": v for k, v in raw_val.items()},
                **{f"val_{k}": v for k, v in val.items()},
            }
        else:
            val = raw_val
            rec = {"epoch": epoch, "seconds": time.time() - t0, **train_avg, **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print("[epoch %03d] %s" % (epoch, json.dumps(rec, sort_keys=True)), flush=True)
        selection_metric = str(args.selection_metric)
        val_iou = float(val.get(selection_metric, val.get("box_iou_mean", val.get("box_iou", -1.0))))
        if val_iou > best_iou:
            best_iou = val_iou
            selected_state = ema_state if ema_state is not None else model.state_dict()
            torch.save(
                {
                    "model": clone_state_dict_cpu(selected_state),
                    "args": vars(args),
                    "epoch": epoch,
                    "val": val,
                    "raw_val": raw_val,
                    "selection_source": "ema" if ema_state is not None else "raw",
                },
                out_dir / "best.pt",
            )
            (out_dir / "best_metrics.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            print(f"[h161-roi] new best epoch={epoch} val_{selection_metric}={val_iou:.4f}", flush=True)
        proposal_oracle_iou = float(val.get("proposal_quad_oracle_iou", -1.0))
        if proposal_oracle_iou > best_proposal_oracle:
            best_proposal_oracle = proposal_oracle_iou
            selected_state = ema_state if ema_state is not None else model.state_dict()
            torch.save(
                {
                    "model": clone_state_dict_cpu(selected_state),
                    "args": vars(args),
                    "epoch": epoch,
                    "val": val,
                    "raw_val": raw_val,
                    "selection_source": "ema" if ema_state is not None else "raw",
                },
                out_dir / "best_proposal_oracle.pt",
            )
            (out_dir / "best_proposal_oracle_metrics.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            print(
                f"[h161-roi] new best proposal oracle epoch={epoch} "
                f"val_proposal_quad_oracle_iou={proposal_oracle_iou:.4f}",
                flush=True,
            )
    print(f"[h161-roi] done best_val_{args.selection_metric}={best_iou:.4f} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
