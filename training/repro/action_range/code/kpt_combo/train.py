import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

if __package__ in (None, ""):
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))
    from dataset import SurgKptComboDataset
    from loss import align_targets_to_tprime, compute_batch_losses, compute_batch_losses_per_sample
    from model import KeypointComboNet
    from vjepa_wrapper import VJEPAWrapper
else:
    from .dataset import SurgKptComboDataset
    from .loss import align_targets_to_tprime, compute_batch_losses, compute_batch_losses_per_sample
    from .model import KeypointComboNet
    from .vjepa_wrapper import VJEPAWrapper

COMBO_CE_WEIGHT = 0.1
RANGE_START_CE_WEIGHT = 0.1
RANGE_END_CE_WEIGHT = 0.1
RANGE_NUM_BINS = 10
RANGE_LABEL_SMOOTHING = 0.05
RANGE_FOCAL_GAMMA = 1.5
RANGE_NEIGHBOR_SMOOTHING = 0.20
TRACK_PCK_THRESHOLDS_PX = (2.0, 4.0, 8.0, 16.0)
TRACK_SURVIVAL_THRESHOLD_PX = 50.0
TRACK_AJ_THRESHOLDS_PX = (1.0, 2.0, 4.0, 8.0, 16.0)
TRACK_CONF_VIS_THRESHOLD = 0.5
RANGE_TIOU_THRESHOLDS = (0.3, 0.5, 0.7)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
DEFAULT_INPUT_SIZE = 256
DEFAULT_PROJ_DIM = 512
DEFAULT_VJEPA_REPO = "vjepa2"
DEFAULT_VJEPA_ARCH = "vit_large"
DEFAULT_VJEPA_HF_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"


def _track_pck_keys() -> List[int]:
    return [int(float(thr)) for thr in TRACK_PCK_THRESHOLDS_PX]
DEFAULT_DATA_ROOT = str(Path(__file__).resolve().parents[1] / "surg_act_video_toy_phase4p5_toolstart_16f_kpt_minimal")
LAYERWISE_STATECARRY_REFINER_MODES = (
    "m2_layerwise_statecarry_refine",
    "m3_layerwise_statecarry_dynamiccentroid",
    "m4_layerwise_statecarry_dynamiccentroid_contextpoints",
    "m5_layerwise_statecarry_dynamiccentroid_templatecorr",
    "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr",
)
STRICT_S3_CROSSLAYER_REFINER_MODES = (
    "s3p2x_finalrefine_l18",
)
SINGLELAYER_STATECARRY_REFINER_MODES = (
    "s3p3_singlelayer_statecarry_refine",
    "s3p3x_singlelayer_statecarry_refine_plus1",
    "s3p8_singlelayer_statecarry_fourierbelief",
    "s3p8a_singlelayer_statecarry_residualfourierbelief",
    "s3p9_singlelayer_statecarry_anchoronly",
    "s3p9d_singlelayer_statecarry_anchoronly_detached",
    "s3p9e_singlelayer_statecarry_anchoronly_ramped",
    "s3p9f_singlelayer_statecarry_anchorverifier_l18",
    "s3p10_singlelayer_statecarry_factorizedtoolresidual",
    "s3p4_singlelayer_statecarry_dynamiccentroid",
    "s3p5_singlelayer_statecarry_deformablesupport",
    "s3p6_singlelayer_statecarry_lategated_deformablesupport",
    "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
    "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
)
DEFAULT_LAYERWISE_TRAJ_STAGE_WEIGHTS: Tuple[float, ...] = (0.35, 0.45, 0.60, 0.80, 1.00)
DEFAULT_T2B_PHASE1_STAGE_WEIGHTS: Tuple[float, ...] = (0.60, 0.50, 0.45, 0.40, 0.50)
DEFAULT_T2B_PHASE3_STAGE_WEIGHTS: Tuple[float, ...] = (0.15, 0.25, 0.50, 0.90, 1.00)


def _parse_layer_indices_arg(spec: str) -> List[int]:
    spec = str(spec).strip()
    if not spec:
        return []
    out: List[int] = []
    seen = set()
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        val = int(item)
        if val <= 0:
            raise ValueError(f"`--vjepa-layer-indices` must contain positive 1-based ids, got {val}")
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _parse_stage_weight_spec(spec: str, *, arg_name: str) -> List[float]:
    values: List[float] = []
    for part in str(spec).split(","):
        item = part.strip()
        if not item:
            continue
        values.append(float(item))
    if len(values) != 5:
        raise ValueError(f"{arg_name} must contain exactly 5 comma-separated floats")
    if any(v < 0.0 for v in values):
        raise ValueError(f"{arg_name} must contain non-negative floats")
    if sum(values) <= 0.0:
        raise ValueError(f"{arg_name} must sum to > 0")
    return values


def _parse_zero_based_stage_indices_arg(spec: str, *, arg_name: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for part in str(spec).split(","):
        item = part.strip()
        if not item:
            continue
        val = int(item)
        if val < 0:
            raise ValueError(f"{arg_name} must contain non-negative zero-based stage ids, got {val}")
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    if not out:
        raise ValueError(f"{arg_name} must contain at least one stage id")
    return out


def _format_stage_weight_spec(weights: Sequence[float]) -> str:
    return ",".join(f"{float(w):.2f}" for w in weights)


def _lerp_stage_weights(
    start_weights: Sequence[float],
    end_weights: Sequence[float],
    alpha: float,
) -> List[float]:
    a = min(max(float(alpha), 0.0), 1.0)
    return [
        float(sw) + (float(ew) - float(sw)) * a
        for sw, ew in zip(start_weights, end_weights)
    ]


def _resolve_train_traj_stage_weights(args: argparse.Namespace, epoch: int) -> Optional[List[float]]:
    if not bool(getattr(args, "traj_stage_curriculum_enable", False)):
        return None
    phase1 = list(getattr(args, "traj_stage_curriculum_phase1_weights"))
    phase2 = list(getattr(args, "traj_stage_curriculum_phase2_weights"))
    phase3 = list(getattr(args, "traj_stage_curriculum_phase3_weights"))
    phase1_end = int(getattr(args, "traj_stage_curriculum_phase1_end_epoch", 0))
    ramp12_end = int(getattr(args, "traj_stage_curriculum_ramp12_end_epoch", phase1_end))
    phase2_end = int(getattr(args, "traj_stage_curriculum_phase2_end_epoch", ramp12_end))
    ramp23_end = int(getattr(args, "traj_stage_curriculum_ramp23_end_epoch", phase2_end))
    if epoch <= phase1_end:
        return phase1
    if epoch <= ramp12_end:
        denom = max(ramp12_end - phase1_end, 1)
        alpha = float(epoch - phase1_end) / float(denom)
        return _lerp_stage_weights(phase1, phase2, alpha)
    if epoch <= phase2_end:
        return phase2
    if epoch <= ramp23_end:
        denom = max(ramp23_end - phase2_end, 1)
        alpha = float(epoch - phase2_end) / float(denom)
        return _lerp_stage_weights(phase2, phase3, alpha)
    return phase3


def _resolve_eval_traj_stage_weights(args: argparse.Namespace) -> Optional[List[float]]:
    if not bool(getattr(args, "traj_stage_curriculum_enable", False)):
        return None
    return list(DEFAULT_LAYERWISE_TRAJ_STAGE_WEIGHTS)


def _resolve_train_traj_stage_phase_name(args: argparse.Namespace, epoch: int) -> str:
    if not bool(getattr(args, "traj_stage_curriculum_enable", False)):
        return "disabled"
    phase1_end = int(getattr(args, "traj_stage_curriculum_phase1_end_epoch", 0))
    ramp12_end = int(getattr(args, "traj_stage_curriculum_ramp12_end_epoch", phase1_end))
    phase2_end = int(getattr(args, "traj_stage_curriculum_phase2_end_epoch", ramp12_end))
    ramp23_end = int(getattr(args, "traj_stage_curriculum_ramp23_end_epoch", phase2_end))
    if epoch <= phase1_end:
        return "phase1_coarse"
    if epoch <= ramp12_end:
        return "ramp12"
    if epoch <= phase2_end:
        return "phase2_default"
    if epoch <= ramp23_end:
        return "ramp23"
    return "phase3_refine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Train combo classification + keypoint regression on frozen VJEPA features"
    )

    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-split-file", type=str, default="")
    parser.add_argument("--val-split-file", type=str, default="")
    parser.add_argument("--n-folds", type=int, default=0)
    parser.add_argument("--n-fold-epochs", type=int, default=100)
    parser.add_argument(
        "--nfold-cls3-report",
        dest="nfold_cls3_report",
        action="store_true",
        help="In n-fold mode, auto-run combo/action/tool classification analysis after each fold.",
    )
    parser.add_argument(
        "--no-nfold-cls3-report",
        dest="nfold_cls3_report",
        action="store_false",
        help="Disable auto-run of combo/action/tool classification analysis in n-fold mode.",
    )
    parser.set_defaults(nfold_cls3_report=True)
    parser.add_argument(
        "--nfold-cls3-top-k",
        type=int,
        default=20,
        help="Top-K confusion pairs per task for auto n-fold cls3 report.",
    )
    parser.add_argument(
        "--nfold-cls3-target-epoch",
        type=int,
        default=0,
        help="Target epoch for val_combo_fpfn file in cls3 report (0 uses --n-fold-epochs).",
    )
    parser.add_argument(
        "--nfold-cls3-output-prefix",
        type=str,
        default="nfold_cls3",
        help="Output prefix for auto n-fold cls3 report files.",
    )
    parser.add_argument(
        "--nfold-cls3-write-markdown",
        dest="nfold_cls3_write_markdown",
        action="store_true",
        help="Write markdown summary for auto n-fold cls3 report.",
    )
    parser.add_argument(
        "--no-nfold-cls3-write-markdown",
        dest="nfold_cls3_write_markdown",
        action="store_false",
        help="Do not write markdown summary for auto n-fold cls3 report.",
    )
    parser.set_defaults(nfold_cls3_write_markdown=True)
    parser.add_argument(
        "--nfold-cls3-debug",
        action="store_true",
        help="Enable debug logs for auto n-fold cls3 report.",
    )
    parser.add_argument(
        "--nfold-cls3-strict",
        dest="nfold_cls3_strict",
        action="store_true",
        help="Fail n-fold training if auto cls3 report fails.",
    )
    parser.add_argument(
        "--no-nfold-cls3-strict",
        dest="nfold_cls3_strict",
        action="store_false",
        help="Warn (do not fail) when auto cls3 report fails.",
    )
    parser.set_defaults(nfold_cls3_strict=False)
    parser.add_argument("--output-dir", type=str, default="work_dirs/kpt_combo")
    parser.add_argument("--vjepa-repo", type=str, default=DEFAULT_VJEPA_REPO)
    parser.add_argument("--vjepa-arch", type=str, default=DEFAULT_VJEPA_ARCH)
    parser.add_argument("--vjepa-hf-repo", type=str, default=DEFAULT_VJEPA_HF_REPO)
    parser.add_argument("--vjepa-hf-local-files-only", action="store_true")
    parser.add_argument("--vjepa-pretrained-hub", action="store_true")
    parser.add_argument("--vjepa-checkpoint", type=str, default="")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)

    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument(
        "--val-interval",
        type=int,
        default=0,
        help="Validation cadence in epochs. Values <=0 fall back to --save-interval.",
    )
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        help="Skip regular epoch checkpoints and only update best.pt when validation improves.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many consecutive validations without meaningful best-monitor improvement. <=0 disables early stop.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum absolute best-monitor improvement required to reset early-stop patience.",
    )
    parser.add_argument(
        "--early-stop-warmup-epochs",
        type=int,
        default=0,
        help="Do not count early-stop patience before this epoch.",
    )
    parser.add_argument(
        "--target-range-tiou",
        type=float,
        default=0.0,
        help="Stop immediately after a validation epoch once val range_tiou reaches this threshold. <=0 disables target-stop.",
    )
    parser.add_argument(
        "--keyframe-selector-only",
        action="store_true",
        help=(
            "Run keyframe selector training stage only (frozen VJEPA token cache + soft labels), "
            "then exit."
        ),
    )
    parser.add_argument(
        "--keyframe-softlabel-root",
        type=str,
        default="",
        help=(
            "Root containing keyframe soft labels (.npz) with train/val subdirs. "
            "Required when --keyframe-selector-only is set."
        ),
    )
    parser.add_argument(
        "--keyframe-cache-root",
        type=str,
        default="",
        help=(
            "Optional token-cache root for keyframe selector stage. "
            "Default: <data-root>/vjepa_token_cache"
        ),
    )
    parser.add_argument("--temporal-upsample", type=str, default="deconv", choices=["interp", "deconv"])
    parser.add_argument("--heatmap-softmax-temp", type=float, default=1.0)
    parser.add_argument("--disable-delta-branch", action="store_true")
    parser.add_argument("--temporal-mean-residual-alpha", type=float, default=0.5)
    parser.add_argument("--refiner-mode", type=str, default="lite", choices=["none", "lite", "attn"])
    parser.add_argument("--trajectory-only", action="store_true")
    parser.add_argument("--trajectory-target", type=str, default="tool", choices=["both", "tool", "ct"])
    parser.add_argument(
        "--traj-num-subframes",
        type=int,
        default=2,
        help="Trajectory-only temporal expansion factor per encoder-time token block.",
    )
    parser.add_argument(
        "--traj-condition-mode",
        type=str,
        default="tool",
        choices=["tool", "tool_action", "combo"],
        help="Conditioning signal for trajectory-only heads.",
    )
    parser.add_argument(
        "--range-condition-mode",
        type=str,
        default="tool_action",
        choices=["tool", "tool_action", "combo"],
        help="Conditioning signal for range-only head.",
    )
    parser.add_argument(
        "--range-positive-mode",
        type=str,
        default="span",
        choices=["span", "center"],
        help="Positive target mode for range offset loss.",
    )
    parser.add_argument(
        "--range-sensitivity-mode",
        type=str,
        default="none",
        choices=["none", "asl"],
        help="Optional ASL-style frame-sensitivity reweighting for range-only offset training.",
    )
    parser.add_argument(
        "--range-sensitivity-cls-sigma",
        type=float,
        default=0.5,
        help="Relative-width sigma for center-focused classification sensitivity when enabled.",
    )
    parser.add_argument(
        "--range-sensitivity-loc-sigma",
        type=float,
        default=0.35,
        help="Relative-width sigma for boundary-focused localization sensitivity when enabled.",
    )
    parser.add_argument(
        "--range-sensitivity-min-weight",
        type=float,
        default=0.25,
        help="Minimum positive-point weight for ASL-style range sensitivity reweighting.",
    )
    parser.add_argument(
        "--range-cls-target-mode",
        type=str,
        default="binary",
        choices=["binary", "varifocal_iou", "hybrid_varifocal"],
        help="How to supervise range actionness: binary span labels or IoU-aware varifocal targets.",
    )
    parser.add_argument(
        "--range-cls-varifocal-alpha",
        type=float,
        default=0.75,
        help="Alpha term for varifocal actionness when --range-cls-target-mode=varifocal_iou.",
    )
    parser.add_argument(
        "--range-cls-varifocal-weight",
        type=float,
        default=1.0,
        help="Blend weight for the IoU-aware varifocal term when --range-cls-target-mode=hybrid_varifocal.",
    )
    parser.add_argument(
        "--range-quality-mode",
        type=str,
        default="none",
        choices=["none", "dgqp"],
        help="Optional DGQP-style localization-quality branch for quality-aware point scoring.",
    )
    parser.add_argument(
        "--range-quality-loss-weight",
        type=float,
        default=0.0,
        help="Loss weight for the range quality branch when enabled.",
    )
    parser.add_argument(
        "--range-scale-invariant-mode",
        type=str,
        default="none",
        choices=["none", "cls", "cls_quality"],
        help=(
            "Optional TSI-style scale-invariant weighting for span-positive range supervision. "
            "'cls' applies it only to actionness classification; "
            "'cls_quality' applies it to actionness and quality."
        ),
    )
    parser.add_argument(
        "--range-scale-invariant-alpha",
        type=float,
        default=0.8,
        help="Alpha in [0,1] for TSI-style scale-invariant weighting. Higher values emphasize negatives more.",
    )
    parser.add_argument(
        "--range-head-mode",
        type=str,
        default="scalar",
        choices=["scalar", "trident"],
        help="Range head decode mode: plain scalar offsets or TriDet-style boundary-distribution decoding.",
    )
    parser.add_argument(
        "--range-proposal-mode",
        type=str,
        default="none",
        choices=["none", "bmn"],
        help="Optional BMN-style proposal scorer on top of the range head.",
    )
    parser.add_argument(
        "--range-proposal-decode-mode",
        type=str,
        default="argmax",
        choices=["argmax", "none"],
        help="How proposal logits affect final range decode: argmax pair decode or auxiliary-only.",
    )
    parser.add_argument(
        "--range-boundary-refine-mode",
        type=str,
        default="none",
        choices=["none", "local_argmax"],
        help="Optional boundary-likelihood refinement applied after the base range decode.",
    )
    parser.add_argument(
        "--range-boundary-refine-radius",
        type=int,
        default=2,
        help="Local search radius for boundary refinement when enabled.",
    )
    parser.add_argument(
        "--range-temporal-context-mode",
        type=str,
        default="none",
        choices=["none", "temporalmaxer", "fpn"],
        help="Optional temporal context block before the range head.",
    )
    parser.add_argument(
        "--range-temporal-context-kernel-size",
        type=int,
        default=3,
        help="Kernel size for the temporal context block. Must be a positive odd integer.",
    )
    parser.add_argument(
        "--range-temporal-context-layers",
        type=int,
        default=1,
        help="Number of temporal context blocks stacked before the range head.",
    )
    parser.add_argument(
        "--range-temporal-context-levels",
        type=int,
        default=3,
        help="Number of pyramid levels when --range-temporal-context-mode=fpn.",
    )
    parser.add_argument(
        "--range-spatial-pool-mode",
        type=str,
        default="mean_max",
        choices=["mean_max", "mean_max_condtoken", "mean_max_condtoken_locresid"],
        help=(
            "How to summarize frozen spatial tokens before the range temporal head. "
            "'mean_max' is the legacy global pool; "
            "'mean_max_condtoken' adds tool/action-conditioned TokenLearner-style local evidence; "
            "'mean_max_condtoken_locresid' keeps the baseline score stream and injects local evidence only as a "
            "zero-init residual into the localization stream."
        ),
    )
    parser.add_argument(
        "--range-spatial-pool-num-tokens",
        type=int,
        default=8,
        help="Number of conditioned local spatial tokens when --range-spatial-pool-mode=mean_max_condtoken.",
    )
    parser.add_argument(
        "--range-adapter-mode",
        type=str,
        default="none",
        choices=["none", "sola", "losa", "dyfa", "sola_dyfa", "sola_dyfa_livegate", "sola_dyfa_laf"],
        help=(
            "Optional temporal adapter on frozen range features: SoLa, LoSA-lite, DyFA-lite, "
            "legacy SoLa+DyFA, the live-gate SoLa+DyFA fix, or local attentive fusion."
        ),
    )
    parser.add_argument(
        "--range-adapter-kernel-size",
        type=int,
        default=5,
        help="Kernel size for the range adapter when enabled. Must be a positive odd integer.",
    )
    parser.add_argument(
        "--range-adapter-num-heads",
        type=int,
        default=4,
        help="Number of attention heads for the LoSA-lite long-range adapter branch.",
    )
    parser.add_argument(
        "--range-task-decouple-mode",
        type=str,
        default="none",
        choices=["none", "score_loc", "score_loc_align"],
        help=(
            "Optional task-decoupled range head. "
            "'score_loc' routes actionness/proposal/quality and boundary regression through separate towers; "
            "'score_loc_align' additionally aligns actionness/quality back toward the localization stream."
        ),
    )
    parser.add_argument(
        "--range-duration-prior-mode",
        type=str,
        default="none",
        choices=["none", "combo_gaussian"],
        help=(
            "Optional combo-conditioned duration prior used only in point ranking decode. "
            "'combo_gaussian' multiplies point scores by a Gaussian prior over predicted span duration."
        ),
    )
    parser.add_argument(
        "--range-duration-prior-weight",
        type=float,
        default=0.0,
        help="Non-negative exponent for the combo-conditioned duration prior when enabled.",
    )
    parser.add_argument(
        "--range-duration-prior-min-std",
        type=float,
        default=0.05,
        help="Minimum normalized duration std used by the combo-conditioned duration prior.",
    )
    parser.add_argument(
        "--range-combo-reweight-mode",
        type=str,
        default="none",
        choices=["none", "effective_num"],
        help=(
            "Optional combo-aware tail reweighting for range losses. "
            "Applies only to the selected range-loss components."
        ),
    )
    parser.add_argument(
        "--range-combo-reweight-targets",
        type=str,
        default="cls,quality",
        help=(
            "Comma-separated range-loss components to reweight by combo frequency. "
            "Allowed: start,end,cls,reg,boundary,proposal,quality,sola"
        ),
    )
    parser.add_argument(
        "--range-combo-reweight-beta",
        type=float,
        default=0.9997,
        help="Effective-number beta for combo-aware range reweighting when enabled.",
    )
    parser.add_argument(
        "--range-combo-reweight-strength",
        type=float,
        default=0.75,
        help=(
            "Blend strength in [0,1] for combo-aware range reweighting. "
            "0 keeps unit weights; 1 uses the full effective-number weights."
        ),
    )
    parser.add_argument(
        "--range-combo-reweight-max-factor",
        type=float,
        default=4.0,
        help="Maximum clip factor for combo-aware range reweighting weights.",
    )
    parser.add_argument(
        "--range-combo-reweight-start-epoch",
        type=int,
        default=1,
        help=(
            "1-based epoch at which combo-aware range reweighting starts to activate. "
            "Use >1 for deferred reweighting."
        ),
    )
    parser.add_argument(
        "--range-combo-reweight-ramp-epochs",
        type=int,
        default=1,
        help=(
            "Number of epochs to ramp combo-aware range reweighting from 0 to full strength "
            "after it starts. 1 means immediate full activation."
        ),
    )
    parser.add_argument(
        "--range-trident-num-bins",
        type=int,
        default=15,
        help="Number of relative distance bins per side for TriDet-style range decoding.",
    )
    parser.add_argument(
        "--range-boundary-aux-weight",
        type=float,
        default=0.0,
        help="Auxiliary weight for start/end boundary classification on the range head.",
    )
    parser.add_argument(
        "--range-proposal-weight",
        type=float,
        default=0.0,
        help="Weight for BMN-style proposal-map supervision on the range head.",
    )
    parser.add_argument(
        "--range-sola-aux-weight",
        type=float,
        default=0.0,
        help="Weight for the SoLa-style inter-frame similarity auxiliary loss.",
    )
    parser.add_argument(
        "--range-sola-sim-k",
        type=float,
        default=16.0,
        help="Similarity target sharpness constant for the SoLa auxiliary loss.",
    )
    parser.add_argument(
        "--traj-refiner-mode",
        type=str,
        default="none",
        choices=[
            "none",
            "lite",
            "attn",
            "s1_offset",
            "s2_iter_offset",
            "s3_support3x3",
            "s3p2x_finalrefine_l18",
            "s3p3_singlelayer_statecarry_refine",
            "s3p3x_singlelayer_statecarry_refine_plus1",
            "s3p8_singlelayer_statecarry_fourierbelief",
            "s3p8a_singlelayer_statecarry_residualfourierbelief",
            "s3p9_singlelayer_statecarry_anchoronly",
            "s3p9d_singlelayer_statecarry_anchoronly_detached",
            "s3p9e_singlelayer_statecarry_anchoronly_ramped",
            "s3p9f_singlelayer_statecarry_anchorverifier_l18",
            "s3p10_singlelayer_statecarry_factorizedtoolresidual",
            "s3p4_singlelayer_statecarry_dynamiccentroid",
            "s3p5_singlelayer_statecarry_deformablesupport",
            "s3p6_singlelayer_statecarry_lategated_deformablesupport",
            "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
            "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
            "s4a_template_centercorr_3x3",
            "s4b_template_allpaircorr_3x3",
            "m1_progressive_multilayer_movingrefine",
            "m1_strict_s3_multilayer_mean",
            "m2_layerwise_statecarry_refine",
            "m3_layerwise_statecarry_dynamiccentroid",
            "m4_layerwise_statecarry_dynamiccentroid_contextpoints",
            "m5_layerwise_statecarry_dynamiccentroid_templatecorr",
            "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr",
        ],
        help="Optional trajectory-only point refiner on top of raw trajectory head outputs.",
    )
    parser.add_argument(
        "--tool-kpt-only-data",
        action="store_true",
        help=(
            "Dataset contract only requires video_16_npy + combo_ids + gt_tool_xy/gt_tool_vis. "
            "ct/action targets are synthesized as zeros in loader."
        ),
    )
    parser.add_argument(
        "--tool-centroid-only",
        action="store_true",
        help=(
            "Use a single tool trajectory query/point by collapsing 3 tool GT points to a centroid "
            "(mean over tool points; vis=max over tool vis)."
        ),
    )
    parser.add_argument(
        "--traj-tool-centroid-offset-head",
        action="store_true",
        help=(
            "Trajectory-only tool head: per subframe, predict absolute centroid + per-keypoint offsets "
            "(no temporal delta accumulation)."
        ),
    )
    parser.add_argument(
        "--traj-tool-offset-max-norm",
        type=float,
        default=0.20,
        help="Max normalized magnitude for tool keypoint offsets around predicted centroid.",
    )
    parser.add_argument(
        "--token-mask-enable",
        action="store_true",
        help="Enable train-time KV token masking in trajectory-only path.",
    )
    parser.add_argument(
        "--token-mask-eval",
        action="store_true",
        help="Also apply token masking during eval (for side-eval robustness checks).",
    )
    parser.add_argument(
        "--token-mask-strategy",
        type=str,
        default="random",
        choices=["random", "tube"],
        help="Token masking strategy for KV tokens.",
    )
    parser.add_argument(
        "--token-mask-value",
        type=str,
        default="learned_pos",
        choices=["learned_pos", "zero"],
        help="Replacement token type for masked KV tokens.",
    )
    parser.add_argument(
        "--token-mask-p-max",
        type=float,
        default=0.10,
        help="Legacy max masking probability for early ramp (used when --token-mask-p-early is not set).",
    )
    parser.add_argument(
        "--token-mask-p-early",
        type=float,
        default=None,
        help="Early-ramp target masking probability (delay->warmup stage). Defaults to --token-mask-p-max.",
    )
    parser.add_argument(
        "--token-mask-delay-epochs",
        type=int,
        default=5,
        help="Epochs to keep masking disabled before warmup starts.",
    )
    parser.add_argument(
        "--token-mask-warmup-epochs",
        type=int,
        default=20,
        help="Linear warmup length (epochs) from 0 to p_early.",
    )
    parser.add_argument(
        "--token-mask-late-start-epoch",
        type=int,
        default=25,
        help="Epoch to start late-ramp masking schedule.",
    )
    parser.add_argument(
        "--token-mask-p-late",
        type=float,
        default=0.15,
        help="Late-ramp target masking probability.",
    )
    parser.add_argument(
        "--token-mask-late-warmup-epochs",
        type=int,
        default=40,
        help="Late-ramp warmup length (epochs) from p(late_start) to p_late.",
    )
    parser.add_argument(
        "--token-mask-keep-k",
        type=int,
        default=64,
        help="Per-frame minimum number of tokens to keep.",
    )
    parser.add_argument(
        "--token-mask-mix-801010",
        dest="token_mask_mix_801010",
        action="store_true",
        help="Use 80/10/10 mixed replacement for selected masked tokens.",
    )
    parser.add_argument(
        "--traj-stage0-centroid-noise-prob",
        type=float,
        default=0.0,
        help="Training-time probability of adding shared centroid carry noise before the first refine stage.",
    )
    parser.add_argument(
        "--traj-stage0-centroid-noise-sigma",
        type=float,
        default=0.0,
        help="Stddev of training-time Gaussian centroid carry noise in normalized xy units.",
    )
    parser.add_argument(
        "--traj-stage-curriculum-enable",
        action="store_true",
        help="Enable train-only epoch-wise stage-loss curriculum for layerwise/statecarry trajectory modes.",
    )
    parser.add_argument(
        "--traj-stage-curriculum-phase1-end-epoch",
        type=int,
        default=16,
        help="Inclusive end epoch of coarse-heavy phase.",
    )
    parser.add_argument(
        "--traj-stage-curriculum-ramp12-end-epoch",
        type=int,
        default=20,
        help="Inclusive end epoch of linear ramp from phase1 to phase2.",
    )
    parser.add_argument(
        "--traj-stage-curriculum-phase2-end-epoch",
        type=int,
        default=44,
        help="Inclusive end epoch of default-weight phase before late ramp.",
    )
    parser.add_argument(
        "--traj-stage-curriculum-ramp23-end-epoch",
        type=int,
        default=50,
        help="Inclusive end epoch of linear ramp from phase2 to phase3.",
    )
    parser.add_argument(
        "--traj-stage-curriculum-phase1-weights",
        type=str,
        default=_format_stage_weight_spec(DEFAULT_T2B_PHASE1_STAGE_WEIGHTS),
        help="Comma-separated 5-stage weights for coarse-heavy phase.",
    )
    parser.add_argument(
        "--traj-stage-curriculum-phase2-weights",
        type=str,
        default=_format_stage_weight_spec(DEFAULT_LAYERWISE_TRAJ_STAGE_WEIGHTS),
        help="Comma-separated 5-stage weights for default phase.",
    )
    parser.add_argument(
        "--traj-stage-curriculum-phase3-weights",
        type=str,
        default=_format_stage_weight_spec(DEFAULT_T2B_PHASE3_STAGE_WEIGHTS),
        help="Comma-separated 5-stage weights for refine-heavy phase.",
    )
    parser.add_argument(
        "--traj-late-focal-enable",
        action="store_true",
        help="Enable training-only late-stage focal reweighting on targeted trajectory stages.",
    )
    parser.add_argument(
        "--traj-late-focal-stage-indices",
        type=str,
        default="3,4",
        help="Comma-separated zero-based stage ids to reweight (e.g. 3,4).",
    )
    parser.add_argument(
        "--traj-late-focal-gamma",
        type=float,
        default=1.5,
        help="Exponent for late-stage focal reweighting.",
    )
    parser.add_argument(
        "--traj-late-focal-clip-min",
        type=float,
        default=0.30,
        help="Minimum focal weight after clipping.",
    )
    parser.add_argument(
        "--traj-late-focal-clip-max",
        type=float,
        default=3.0,
        help="Maximum focal weight after clipping.",
    )
    parser.add_argument(
        "--traj-late-focal-ema-decay",
        type=float,
        default=0.95,
        help="EMA decay for detached late-stage residual normalization.",
    )
    parser.add_argument(
        "--traj-late-focal-warmup-steps",
        type=int,
        default=350,
        help="Warmup steps before enabling focal reweighting; EMA still updates during warmup.",
    )
    parser.add_argument(
        "--no-token-mask-mix-801010",
        dest="token_mask_mix_801010",
        action="store_false",
        help="Disable 80/10/10 mixed replacement.",
    )
    parser.set_defaults(token_mask_mix_801010=True)
    parser.add_argument("--combo-only", action="store_true")
    parser.add_argument("--range-only", action="store_true")
    parser.add_argument("--tool-only", action="store_true")
    parser.add_argument("--combo-probe-attn", action="store_true")
    parser.add_argument("--combo-probe-dual-attn", action="store_true")
    parser.add_argument(
        "--dual-attn-tool-prior",
        action="store_true",
        help="In dual-attn probe mode, add tool-conditioned bias to action query: q_action += alpha * Linear(softmax(tool_logits)).",
    )
    parser.add_argument(
        "--dual-attn-tool-prior-init-alpha",
        type=float,
        default=0.0,
        help="Initial scalar alpha for dual-attn tool-conditioned action prior.",
    )
    parser.add_argument(
        "--dual-attn-tool-prior-detach",
        dest="dual_attn_tool_prior_detach",
        action="store_true",
        help="Detach tool probabilities before tool->action query-bias projection (recommended for stability).",
    )
    parser.add_argument(
        "--no-dual-attn-tool-prior-detach",
        dest="dual_attn_tool_prior_detach",
        action="store_false",
        help="Allow action loss gradients to flow into tool logits via tool->action query-bias path.",
    )
    parser.set_defaults(dual_attn_tool_prior_detach=True)
    parser.add_argument(
        "--combo-only-pure",
        action="store_true",
        help="Pure combo mode: dataset reads only video_16_npy + combo_ids, aux labels are synthetic placeholders.",
    )
    parser.add_argument("--combo-ce-weight", type=float, default=COMBO_CE_WEIGHT)
    parser.add_argument("--combo-tool-ce-weight", type=float, default=0.5)
    parser.add_argument("--combo-action-ce-weight", type=float, default=0.5)
    parser.add_argument("--traj-heatmap-ce-weight", type=float, default=0.0)
    parser.add_argument("--traj-heatmap-sigma", type=float, default=2.45)
    parser.add_argument("--traj-head-hidden-dim", type=int, default=256)
    parser.add_argument("--traj-head-up-layers", type=int, default=2)
    parser.add_argument("--traj-recurrent-max-delta-norm", type=float, default=0.06)
    parser.add_argument(
        "--traj-dynamic-centroid-max-beta",
        type=float,
        default=0.75,
        help="Max late-stage centroid refresh blend for M3/M4 dynamic-centroid refine modes.",
    )
    parser.add_argument("--combo-head-dropout", type=float, default=0.0)
    parser.add_argument("--combo-feat-dropout", type=float, default=0.0)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--kpt-loss-type",
        type=str,
        default="cotracker_huber",
        choices=["l1", "l2", "l1l2", "cotracker_l1", "cotracker_huber"],
    )
    parser.add_argument("--traj-loss-pixel-scale", type=float, default=256.0)
    parser.add_argument("--traj-huber-delta", type=float, default=6.0)
    parser.add_argument("--traj-t0-loss-weight", type=float, default=1.0)
    parser.add_argument("--traj-loss-visible-only", action="store_true", default=False)
    parser.add_argument("--traj-loss-include-nonvis", action="store_false", dest="traj_loss_visible_only")
    parser.add_argument(
        "--symmetry-swap-invariant-loss",
        dest="symmetry_swap_invariant_loss",
        action="store_true",
        help="Apply pair-only swap-invariant trajectory loss on the symmetric tool-point pair.",
    )
    parser.add_argument(
        "--no-symmetry-swap-invariant-loss",
        dest="symmetry_swap_invariant_loss",
        action="store_false",
        help="Disable pair-only swap-invariant trajectory loss.",
    )
    parser.set_defaults(symmetry_swap_invariant_loss=False)
    parser.add_argument(
        "--symmetry-pair-separation-loss",
        dest="symmetry_pair_separation_loss",
        action="store_true",
        help="Apply a weak symmetric-pair separation auxiliary loss on tool trajectories.",
    )
    parser.add_argument(
        "--no-symmetry-pair-separation-loss",
        dest="symmetry_pair_separation_loss",
        action="store_false",
        help="Disable the symmetric-pair separation auxiliary loss.",
    )
    parser.set_defaults(symmetry_pair_separation_loss=False)
    parser.add_argument("--symmetry-pair-separation-weight", type=float, default=0.05)
    parser.add_argument("--symmetry-pair-separation-min-pred-gap-px", type=float, default=8.0)
    parser.add_argument("--symmetry-pair-separation-gt-min-gap-px", type=float, default=16.0)
    parser.add_argument("--traj-centroid-aux-weight", type=float, default=0.0)
    parser.add_argument("--traj-offset-aux-weight", type=float, default=0.0)
    parser.add_argument("--no-vjepa-cache", action="store_true")
    parser.add_argument("--vjepa-cache-dir", type=str, default="")
    parser.add_argument("--vjepa-cache-dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument(
        "--vjepa-layer-indices",
        type=str,
        default="",
        help=(
            "Optional comma-separated 1-based VJEPA encoder layer ids to keep when using the HF backend. "
            "Example: 6,12,18,24. Required automatically for multi-layer trajectory refiner modes."
        ),
    )
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--train-platform", type=str, default="none", choices=["none", "tensorboard"])
    parser.add_argument("--tb-dir", type=str, default="")
    parser.add_argument("--vis-scale", type=int, default=3)
    parser.add_argument("--no-progress", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-init-eval-vis", action="store_true")
    parser.add_argument(
        "--no-checkpoint-vis",
        action="store_true",
        help="Disable train/val visualization export at save checkpoints.",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable all visualization exports (init snapshot + checkpoint visuals).",
    )
    parser.add_argument("--no-weighted-ce", action="store_true")
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument("--reweight-power", type=float, default=0.5)
    parser.add_argument("--reweight-max-factor", type=float, default=10.0)

    # Backward-compatible alias for old flag.
    parser.add_argument("--save-every", type=int, default=None, help=argparse.SUPPRESS)
    # Hidden advanced debug switches (kept off by default for clean CLI).
    parser.add_argument("--debug-sanity", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--debug-train-check", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--debug-check-interval", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--debug-save-step-report", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--debug-model-timing", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--debug-model-shapes", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.save_every is not None:
        args.save_interval = args.save_every
    if int(args.save_interval) <= 0:
        raise ValueError("--save-interval must be > 0")
    if int(args.val_interval) < 0:
        raise ValueError("--val-interval must be >= 0")
    if int(args.val_interval) == 0:
        args.val_interval = int(args.save_interval)
    if int(args.early_stop_patience) < 0:
        raise ValueError("--early-stop-patience must be >= 0")
    if float(args.early_stop_min_delta) < 0.0:
        raise ValueError("--early-stop-min-delta must be >= 0")
    if float(getattr(args, "range_boundary_aux_weight", 0.0)) < 0.0:
        raise ValueError("--range-boundary-aux-weight must be >= 0")
    if float(getattr(args, "range_proposal_weight", 0.0)) < 0.0:
        raise ValueError("--range-proposal-weight must be >= 0")
    if float(getattr(args, "range_quality_loss_weight", 0.0)) < 0.0:
        raise ValueError("--range-quality-loss-weight must be >= 0")
    if int(getattr(args, "range_trident_num_bins", 15)) <= 0:
        raise ValueError("--range-trident-num-bins must be > 0")
    if int(args.early_stop_warmup_epochs) < 0:
        raise ValueError("--early-stop-warmup-epochs must be >= 0")
    if float(getattr(args, "target_range_tiou", 0.0)) < 0.0:
        raise ValueError("--target-range-tiou must be >= 0")
    if (
        (int(args.early_stop_patience) > 0 or float(getattr(args, "target_range_tiou", 0.0)) > 0.0)
        and int(args.val_interval) != 1
    ):
        print(
            "[args] forcing --val-interval=1 because early-stop / target-range-tiou "
            f"requires per-epoch validation (got {int(args.val_interval)})"
        )
        args.val_interval = 1
    if int(args.n_folds) < 0:
        raise ValueError("--n-folds must be >= 0")
    if int(args.n_fold_epochs) <= 0:
        raise ValueError("--n-fold-epochs must be > 0")
    if bool(getattr(args, "keyframe_selector_only", False)):
        if int(args.n_folds) > 1:
            raise ValueError("--keyframe-selector-only does not support --n-folds > 1")
        if not str(getattr(args, "keyframe_softlabel_root", "")).strip():
            raise ValueError("--keyframe-selector-only requires --keyframe-softlabel-root")
    if bool(args.combo_only_pure):
        args.combo_only = True
    if bool(args.tool_only):
        args.combo_only = True
        # tool-only runs only tool classification; auxiliary range/kpt files are not required.
        args.combo_only_pure = True
    if bool(args.range_only):
        args.combo_ce_weight = 0.0
        args.combo_tool_ce_weight = 0.0
        args.combo_action_ce_weight = 0.0
    if bool(args.combo_probe_attn):
        args.combo_only = True
        # combo probe runs only classification heads; auxiliary range/kpt files are synthetic placeholders.
        args.combo_only_pure = True
        args.combo_tool_ce_weight = 0.0
        args.combo_action_ce_weight = 0.0
    if bool(args.combo_probe_dual_attn):
        args.combo_only = True
        # dual-attn probe runs classification-only heads; auxiliary range/kpt files are synthetic placeholders.
        args.combo_only_pure = True
    if bool(args.trajectory_only) and bool(args.combo_only):
        raise ValueError("--trajectory-only and --combo-only are mutually exclusive")
    if bool(args.trajectory_only) and bool(args.range_only):
        raise ValueError("--trajectory-only and --range-only are mutually exclusive")
    if bool(args.trajectory_only) and bool(args.combo_only_pure):
        raise ValueError("--trajectory-only and --combo-only-pure are mutually exclusive")
    if bool(args.trajectory_only) and bool(args.tool_only):
        raise ValueError("--trajectory-only and --tool-only are mutually exclusive")
    if bool(args.trajectory_only) and bool(args.combo_probe_attn):
        raise ValueError("--trajectory-only and --combo-probe-attn are mutually exclusive")
    if bool(args.trajectory_only) and bool(args.combo_probe_dual_attn):
        raise ValueError("--trajectory-only and --combo-probe-dual-attn are mutually exclusive")
    if bool(getattr(args, "tool_kpt_only_data", False)):
        traj_target = str(getattr(args, "trajectory_target", "tool")).lower()
        if (not bool(args.trajectory_only)) or (traj_target != "tool"):
            raise ValueError("--tool-kpt-only-data requires --trajectory-only --trajectory-target tool")
        if bool(getattr(args, "range_only", False)):
            raise ValueError("--tool-kpt-only-data is incompatible with --range-only")
    if bool(getattr(args, "range_only", False)):
        if bool(args.combo_only):
            raise ValueError("--range-only cannot be combined with --combo-only")
        if bool(args.combo_only_pure):
            raise ValueError("--range-only cannot be combined with --combo-only-pure")
        if bool(args.tool_only):
            raise ValueError("--range-only cannot be combined with --tool-only")
        if bool(args.combo_probe_attn):
            raise ValueError("--range-only cannot be combined with --combo-probe-attn")
        if bool(args.combo_probe_dual_attn):
            raise ValueError("--range-only cannot be combined with --combo-probe-dual-attn")
    if bool(getattr(args, "tool_centroid_only", False)):
        traj_target = str(getattr(args, "trajectory_target", "tool")).lower()
        if (not bool(args.trajectory_only)) or (traj_target != "tool"):
            raise ValueError("--tool-centroid-only requires --trajectory-only --trajectory-target tool")
    if bool(getattr(args, "traj_tool_centroid_offset_head", False)):
        traj_target = str(getattr(args, "trajectory_target", "tool")).lower()
        if (not bool(args.trajectory_only)) or (traj_target not in ("tool", "both")):
            raise ValueError("--traj-tool-centroid-offset-head requires --trajectory-only --trajectory-target tool|both")
    if bool(getattr(args, "tool_centroid_only", False)) and bool(getattr(args, "traj_tool_centroid_offset_head", False)):
        raise ValueError("--tool-centroid-only and --traj-tool-centroid-offset-head are mutually exclusive")
    if bool(getattr(args, "symmetry_swap_invariant_loss", False)):
        traj_target = str(getattr(args, "trajectory_target", "tool")).lower()
        if not bool(getattr(args, "trajectory_only", False)):
            raise ValueError("--symmetry-swap-invariant-loss currently requires --trajectory-only")
        if traj_target not in ("tool", "both"):
            raise ValueError("--symmetry-swap-invariant-loss requires trajectory target to include tool points")
        if bool(getattr(args, "tool_centroid_only", False)):
            raise ValueError("--symmetry-swap-invariant-loss is incompatible with --tool-centroid-only")
    if bool(getattr(args, "symmetry_pair_separation_loss", False)):
        traj_target = str(getattr(args, "trajectory_target", "tool")).lower()
        if not bool(getattr(args, "trajectory_only", False)):
            raise ValueError("--symmetry-pair-separation-loss currently requires --trajectory-only")
        if traj_target not in ("tool", "both"):
            raise ValueError("--symmetry-pair-separation-loss requires trajectory target to include tool points")
        if bool(getattr(args, "tool_centroid_only", False)):
            raise ValueError("--symmetry-pair-separation-loss is incompatible with --tool-centroid-only")
    if float(getattr(args, "symmetry_pair_separation_weight", 0.0)) < 0.0:
        raise ValueError("--symmetry-pair-separation-weight must be >= 0")
    if float(getattr(args, "symmetry_pair_separation_min_pred_gap_px", 0.0)) < 0.0:
        raise ValueError("--symmetry-pair-separation-min-pred-gap-px must be >= 0")
    if float(getattr(args, "symmetry_pair_separation_gt_min_gap_px", 0.0)) < 0.0:
        raise ValueError("--symmetry-pair-separation-gt-min-gap-px must be >= 0")
    if bool(getattr(args, "token_mask_enable", False)) and (not bool(getattr(args, "trajectory_only", False))):
        raise ValueError("--token-mask-enable currently requires --trajectory-only")
    if bool(getattr(args, "token_mask_eval", False)) and (not bool(getattr(args, "trajectory_only", False))):
        raise ValueError("--token-mask-eval currently requires --trajectory-only")
    if int(getattr(args, "traj_num_subframes", 2)) <= 0:
        raise ValueError("--traj-num-subframes must be > 0")
    traj_condition_mode = str(getattr(args, "traj_condition_mode", "tool")).lower()
    if traj_condition_mode not in ("tool", "tool_action", "combo"):
        raise ValueError("--traj-condition-mode must be one of: tool, tool_action, combo")
    range_condition_mode = str(getattr(args, "range_condition_mode", "tool_action")).lower()
    if range_condition_mode not in ("tool", "tool_action", "combo"):
        raise ValueError("--range-condition-mode must be one of: tool, tool_action, combo")
    range_sensitivity_mode = str(getattr(args, "range_sensitivity_mode", "none")).lower()
    if range_sensitivity_mode not in ("none", "asl"):
        raise ValueError("--range-sensitivity-mode must be one of: none, asl")
    if float(getattr(args, "range_sensitivity_cls_sigma", 0.5)) <= 0.0:
        raise ValueError("--range-sensitivity-cls-sigma must be > 0")
    if float(getattr(args, "range_sensitivity_loc_sigma", 0.35)) <= 0.0:
        raise ValueError("--range-sensitivity-loc-sigma must be > 0")
    range_sensitivity_min_weight = float(getattr(args, "range_sensitivity_min_weight", 0.25))
    if range_sensitivity_min_weight < 0.0 or range_sensitivity_min_weight > 1.0:
        raise ValueError("--range-sensitivity-min-weight must be in [0,1]")
    range_cls_target_mode = str(getattr(args, "range_cls_target_mode", "binary")).lower()
    if range_cls_target_mode not in ("binary", "varifocal_iou", "hybrid_varifocal"):
        raise ValueError("--range-cls-target-mode must be one of: binary, varifocal_iou, hybrid_varifocal")
    if float(getattr(args, "range_cls_varifocal_alpha", 0.75)) < 0.0:
        raise ValueError("--range-cls-varifocal-alpha must be >= 0")
    range_cls_varifocal_weight = float(getattr(args, "range_cls_varifocal_weight", 1.0))
    if range_cls_varifocal_weight < 0.0 or range_cls_varifocal_weight > 1.0:
        raise ValueError("--range-cls-varifocal-weight must be in [0,1]")
    range_proposal_mode = str(getattr(args, "range_proposal_mode", "none")).lower()
    if range_proposal_mode not in ("none", "bmn"):
        raise ValueError("--range-proposal-mode must be one of: none, bmn")
    range_quality_mode = str(getattr(args, "range_quality_mode", "none")).lower()
    if range_quality_mode not in ("none", "dgqp"):
        raise ValueError("--range-quality-mode must be one of: none, dgqp")
    range_proposal_decode_mode = str(getattr(args, "range_proposal_decode_mode", "argmax")).lower()
    if range_proposal_decode_mode not in ("argmax", "none"):
        raise ValueError("--range-proposal-decode-mode must be one of: argmax, none")
    range_boundary_refine_mode = str(getattr(args, "range_boundary_refine_mode", "none")).lower()
    if range_boundary_refine_mode not in ("none", "local_argmax"):
        raise ValueError("--range-boundary-refine-mode must be one of: none, local_argmax")
    if int(getattr(args, "range_boundary_refine_radius", 2)) <= 0:
        raise ValueError("--range-boundary-refine-radius must be > 0")
    range_temporal_context_mode = str(getattr(args, "range_temporal_context_mode", "none")).lower()
    if range_temporal_context_mode not in ("none", "temporalmaxer", "fpn"):
        raise ValueError("--range-temporal-context-mode must be one of: none, temporalmaxer, fpn")
    if int(getattr(args, "range_temporal_context_kernel_size", 3)) <= 0:
        raise ValueError("--range-temporal-context-kernel-size must be > 0")
    if int(getattr(args, "range_temporal_context_kernel_size", 3)) % 2 == 0:
        raise ValueError("--range-temporal-context-kernel-size must be odd")
    if int(getattr(args, "range_temporal_context_layers", 1)) <= 0:
        raise ValueError("--range-temporal-context-layers must be > 0")
    if int(getattr(args, "range_temporal_context_levels", 3)) <= 0:
        raise ValueError("--range-temporal-context-levels must be > 0")
    range_spatial_pool_mode = str(getattr(args, "range_spatial_pool_mode", "mean_max")).lower()
    if range_spatial_pool_mode not in ("mean_max", "mean_max_condtoken", "mean_max_condtoken_locresid"):
        raise ValueError(
            "--range-spatial-pool-mode must be one of: mean_max, mean_max_condtoken, mean_max_condtoken_locresid"
        )
    args.range_spatial_pool_mode = range_spatial_pool_mode
    if int(getattr(args, "range_spatial_pool_num_tokens", 8)) <= 0:
        raise ValueError("--range-spatial-pool-num-tokens must be > 0")
    range_adapter_mode = str(getattr(args, "range_adapter_mode", "none")).lower()
    if range_adapter_mode not in ("none", "sola", "losa", "dyfa", "sola_dyfa", "sola_dyfa_livegate", "sola_dyfa_laf"):
        raise ValueError(
            "--range-adapter-mode must be one of: none, sola, losa, dyfa, sola_dyfa, sola_dyfa_livegate, sola_dyfa_laf"
        )
    if int(getattr(args, "range_adapter_kernel_size", 5)) <= 0:
        raise ValueError("--range-adapter-kernel-size must be > 0")
    if int(getattr(args, "range_adapter_kernel_size", 5)) % 2 == 0:
        raise ValueError("--range-adapter-kernel-size must be odd")
    if int(getattr(args, "range_adapter_num_heads", 4)) <= 0:
        raise ValueError("--range-adapter-num-heads must be > 0")
    range_task_decouple_mode = str(getattr(args, "range_task_decouple_mode", "none")).lower()
    if range_task_decouple_mode not in ("none", "score_loc", "score_loc_align"):
        raise ValueError("--range-task-decouple-mode must be one of: none, score_loc, score_loc_align")
    args.range_task_decouple_mode = range_task_decouple_mode
    range_duration_prior_mode = str(getattr(args, "range_duration_prior_mode", "none")).lower()
    if range_duration_prior_mode not in ("none", "combo_gaussian"):
        raise ValueError("--range-duration-prior-mode must be one of: none, combo_gaussian")
    args.range_duration_prior_mode = range_duration_prior_mode
    if float(getattr(args, "range_duration_prior_weight", 0.0)) < 0.0:
        raise ValueError("--range-duration-prior-weight must be >= 0")
    if float(getattr(args, "range_duration_prior_min_std", 0.05)) <= 0.0:
        raise ValueError("--range-duration-prior-min-std must be > 0")
    range_scale_invariant_mode = str(getattr(args, "range_scale_invariant_mode", "none")).lower()
    if range_scale_invariant_mode not in ("none", "cls", "cls_quality"):
        raise ValueError("--range-scale-invariant-mode must be one of: none, cls, cls_quality")
    args.range_scale_invariant_mode = range_scale_invariant_mode
    if float(getattr(args, "range_scale_invariant_alpha", 0.8)) < 0.0 or float(
        getattr(args, "range_scale_invariant_alpha", 0.8)
    ) > 1.0:
        raise ValueError("--range-scale-invariant-alpha must be in [0,1]")
    range_combo_reweight_mode = str(getattr(args, "range_combo_reweight_mode", "none")).lower()
    if range_combo_reweight_mode not in ("none", "effective_num"):
        raise ValueError("--range-combo-reweight-mode must be one of: none, effective_num")
    args.range_combo_reweight_mode = range_combo_reweight_mode
    args.range_combo_reweight_targets = ",".join(
        _parse_range_combo_reweight_targets(getattr(args, "range_combo_reweight_targets", ""))
    )
    range_combo_reweight_beta = float(getattr(args, "range_combo_reweight_beta", 0.9997))
    if range_combo_reweight_beta < 0.0 or range_combo_reweight_beta >= 1.0:
        raise ValueError("--range-combo-reweight-beta must be in [0,1)")
    range_combo_reweight_strength = float(getattr(args, "range_combo_reweight_strength", 0.75))
    if range_combo_reweight_strength < 0.0 or range_combo_reweight_strength > 1.0:
        raise ValueError("--range-combo-reweight-strength must be in [0,1]")
    if float(getattr(args, "range_combo_reweight_max_factor", 4.0)) <= 0.0:
        raise ValueError("--range-combo-reweight-max-factor must be > 0")
    if int(getattr(args, "range_combo_reweight_start_epoch", 1)) < 1:
        raise ValueError("--range-combo-reweight-start-epoch must be >= 1")
    if int(getattr(args, "range_combo_reweight_ramp_epochs", 1)) < 1:
        raise ValueError("--range-combo-reweight-ramp-epochs must be >= 1")
    if float(getattr(args, "range_sola_aux_weight", 0.0)) < 0.0:
        raise ValueError("--range-sola-aux-weight must be >= 0")
    if float(getattr(args, "range_sola_sim_k", 16.0)) <= 0.0:
        raise ValueError("--range-sola-sim-k must be > 0")
    traj_refiner_mode = str(getattr(args, "traj_refiner_mode", "none")).lower()
    if traj_refiner_mode not in (
        "none",
        "lite",
        "attn",
        "s1_offset",
        "s2_iter_offset",
        "s3_support3x3",
        "s3p2x_finalrefine_l18",
        "s3p3_singlelayer_statecarry_refine",
        "s3p3x_singlelayer_statecarry_refine_plus1",
        "s3p8_singlelayer_statecarry_fourierbelief",
        "s3p8a_singlelayer_statecarry_residualfourierbelief",
        "s3p9_singlelayer_statecarry_anchoronly",
        "s3p9d_singlelayer_statecarry_anchoronly_detached",
        "s3p9e_singlelayer_statecarry_anchoronly_ramped",
        "s3p9f_singlelayer_statecarry_anchorverifier_l18",
        "s3p10_singlelayer_statecarry_factorizedtoolresidual",
        "s3p4_singlelayer_statecarry_dynamiccentroid",
        "s3p5_singlelayer_statecarry_deformablesupport",
        "s3p6_singlelayer_statecarry_lategated_deformablesupport",
        "s3p6a_singlelayer_statecarry_pointgated_deformablesupport",
        "s3p7_singlelayer_statecarry_jointpoint_deformablesupport",
        "s4a_template_centercorr_3x3",
        "s4b_template_allpaircorr_3x3",
        "m1_progressive_multilayer_movingrefine",
        "m1_strict_s3_multilayer_mean",
        "m2_layerwise_statecarry_refine",
        "m3_layerwise_statecarry_dynamiccentroid",
        "m4_layerwise_statecarry_dynamiccentroid_contextpoints",
        "m5_layerwise_statecarry_dynamiccentroid_templatecorr",
        "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr",
    ):
        raise ValueError(
            "--traj-refiner-mode must be one of: "
            "none, lite, attn, s1_offset, s2_iter_offset, s3_support3x3, "
            "s3p2x_finalrefine_l18, "
            "s3p3_singlelayer_statecarry_refine, s3p3x_singlelayer_statecarry_refine_plus1, "
            "s3p8_singlelayer_statecarry_fourierbelief, "
            "s3p8a_singlelayer_statecarry_residualfourierbelief, "
            "s3p9_singlelayer_statecarry_anchoronly, "
            "s3p9d_singlelayer_statecarry_anchoronly_detached, "
            "s3p9e_singlelayer_statecarry_anchoronly_ramped, "
            "s3p9f_singlelayer_statecarry_anchorverifier_l18, "
            "s3p10_singlelayer_statecarry_factorizedtoolresidual, "
            "s3p4_singlelayer_statecarry_dynamiccentroid, "
            "s3p5_singlelayer_statecarry_deformablesupport, "
            "s3p6_singlelayer_statecarry_lategated_deformablesupport, "
            "s3p6a_singlelayer_statecarry_pointgated_deformablesupport, "
            "s3p7_singlelayer_statecarry_jointpoint_deformablesupport, "
            "s4a_template_centercorr_3x3, s4b_template_allpaircorr_3x3, "
            "m1_progressive_multilayer_movingrefine, m1_strict_s3_multilayer_mean, "
            "m2_layerwise_statecarry_refine, m3_layerwise_statecarry_dynamiccentroid, "
            "m4_layerwise_statecarry_dynamiccentroid_contextpoints, "
            "m5_layerwise_statecarry_dynamiccentroid_templatecorr, "
            "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr"
        )
    if (
        traj_refiner_mode in (
            *STRICT_S3_CROSSLAYER_REFINER_MODES,
            "m1_progressive_multilayer_movingrefine",
            "m1_strict_s3_multilayer_mean",
            *SINGLELAYER_STATECARRY_REFINER_MODES,
            "m2_layerwise_statecarry_refine",
            "m3_layerwise_statecarry_dynamiccentroid",
            "m4_layerwise_statecarry_dynamiccentroid_contextpoints",
            "m5_layerwise_statecarry_dynamiccentroid_templatecorr",
            "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr",
        )
        and (not bool(getattr(args, "trajectory_only", False)))
    ):
        raise ValueError(f"--traj-refiner-mode {traj_refiner_mode} requires --trajectory-only")
    if (
        traj_refiner_mode in STRICT_S3_CROSSLAYER_REFINER_MODES
        and (not str(getattr(args, "vjepa_layer_indices", "")).strip())
    ):
        raise ValueError("`s3p2x_finalrefine_l18` requires `--vjepa-layer-indices 18,24`.")
    if (
        traj_refiner_mode in (
            "m1_progressive_multilayer_movingrefine",
            "m1_strict_s3_multilayer_mean",
            "m2_layerwise_statecarry_refine",
            "m3_layerwise_statecarry_dynamiccentroid",
            "m4_layerwise_statecarry_dynamiccentroid_contextpoints",
            "m5_layerwise_statecarry_dynamiccentroid_templatecorr",
            "m5p1_layerwise_statecarry_dynamiccentroid_templatecorr",
        )
        and (not str(getattr(args, "vjepa_layer_indices", "")).strip())
    ):
        args.vjepa_layer_indices = "6,12,18,24"
    if (
        float(getattr(args, "traj_dynamic_centroid_max_beta", 0.75)) < 0.0
        or float(getattr(args, "traj_dynamic_centroid_max_beta", 0.75)) > 1.0
    ):
        raise ValueError("--traj-dynamic-centroid-max-beta must be in [0,1]")
    args.vjepa_layer_indices = _parse_layer_indices_arg(str(getattr(args, "vjepa_layer_indices", "")))
    if traj_refiner_mode in STRICT_S3_CROSSLAYER_REFINER_MODES and args.vjepa_layer_indices != [18, 24]:
        raise ValueError(
            "`s3p2x_finalrefine_l18` requires exactly `--vjepa-layer-indices 18,24` "
            f"(got {args.vjepa_layer_indices})."
        )
    if args.vjepa_layer_indices and (not str(getattr(args, "vjepa_hf_repo", "")).strip()):
        raise ValueError("--vjepa-layer-indices currently requires the HF VJEPA backend (--vjepa-hf-repo).")
    if float(getattr(args, "token_mask_p_max", 0.0)) < 0.0 or float(getattr(args, "token_mask_p_max", 0.0)) >= 1.0:
        raise ValueError("--token-mask-p-max must be in [0,1)")
    if getattr(args, "token_mask_p_early", None) is None:
        args.token_mask_p_early = float(getattr(args, "token_mask_p_max", 0.10))
    if float(getattr(args, "token_mask_p_early", 0.0)) < 0.0 or float(getattr(args, "token_mask_p_early", 0.0)) >= 1.0:
        raise ValueError("--token-mask-p-early must be in [0,1)")
    if int(getattr(args, "token_mask_delay_epochs", 0)) < 0:
        raise ValueError("--token-mask-delay-epochs must be >= 0")
    if int(getattr(args, "token_mask_warmup_epochs", 0)) < 0:
        raise ValueError("--token-mask-warmup-epochs must be >= 0")
    if int(getattr(args, "token_mask_late_start_epoch", 0)) < 0:
        raise ValueError("--token-mask-late-start-epoch must be >= 0")
    if float(getattr(args, "token_mask_p_late", 0.0)) < 0.0 or float(getattr(args, "token_mask_p_late", 0.0)) >= 1.0:
        raise ValueError("--token-mask-p-late must be in [0,1)")
    if int(getattr(args, "token_mask_late_warmup_epochs", 0)) < 0:
        raise ValueError("--token-mask-late-warmup-epochs must be >= 0")
    if int(getattr(args, "token_mask_keep_k", 0)) <= 0:
        raise ValueError("--token-mask-keep-k must be > 0")
    args.traj_stage_curriculum_phase1_weights = _parse_stage_weight_spec(
        getattr(args, "traj_stage_curriculum_phase1_weights", ""),
        arg_name="--traj-stage-curriculum-phase1-weights",
    )
    args.traj_stage_curriculum_phase2_weights = _parse_stage_weight_spec(
        getattr(args, "traj_stage_curriculum_phase2_weights", ""),
        arg_name="--traj-stage-curriculum-phase2-weights",
    )
    args.traj_stage_curriculum_phase3_weights = _parse_stage_weight_spec(
        getattr(args, "traj_stage_curriculum_phase3_weights", ""),
        arg_name="--traj-stage-curriculum-phase3-weights",
    )
    args.traj_late_focal_stage_indices = _parse_zero_based_stage_indices_arg(
        getattr(args, "traj_late_focal_stage_indices", ""),
        arg_name="--traj-late-focal-stage-indices",
    )
    if float(getattr(args, "traj_late_focal_gamma", 0.0)) <= 0.0:
        raise ValueError("--traj-late-focal-gamma must be > 0")
    if float(getattr(args, "traj_late_focal_clip_min", 0.0)) <= 0.0:
        raise ValueError("--traj-late-focal-clip-min must be > 0")
    if float(getattr(args, "traj_late_focal_clip_max", 0.0)) < float(getattr(args, "traj_late_focal_clip_min", 0.0)):
        raise ValueError("--traj-late-focal-clip-max must be >= --traj-late-focal-clip-min")
    if not (0.0 <= float(getattr(args, "traj_late_focal_ema_decay", 0.95)) < 1.0):
        raise ValueError("--traj-late-focal-ema-decay must be in [0,1)")
    if int(getattr(args, "traj_late_focal_warmup_steps", 0)) < 0:
        raise ValueError("--traj-late-focal-warmup-steps must be >= 0")
    phase1_end = int(getattr(args, "traj_stage_curriculum_phase1_end_epoch", 0))
    ramp12_end = int(getattr(args, "traj_stage_curriculum_ramp12_end_epoch", 0))
    phase2_end = int(getattr(args, "traj_stage_curriculum_phase2_end_epoch", 0))
    ramp23_end = int(getattr(args, "traj_stage_curriculum_ramp23_end_epoch", 0))
    if phase1_end <= 0:
        raise ValueError("--traj-stage-curriculum-phase1-end-epoch must be > 0")
    if not (phase1_end <= ramp12_end <= phase2_end <= ramp23_end):
        raise ValueError(
            "--traj-stage-curriculum epochs must satisfy "
            "phase1_end <= ramp12_end <= phase2_end <= ramp23_end"
        )
    if bool(getattr(args, "traj_stage_curriculum_enable", False)):
        if not bool(getattr(args, "trajectory_only", False)):
            raise ValueError("--traj-stage-curriculum-enable currently requires --trajectory-only")
        if traj_refiner_mode not in LAYERWISE_STATECARRY_REFINER_MODES and traj_refiner_mode not in SINGLELAYER_STATECARRY_REFINER_MODES:
            raise ValueError(
                "--traj-stage-curriculum-enable currently requires a layerwise/singlelayer statecarry refiner mode"
            )
    if bool(getattr(args, "traj_late_focal_enable", False)):
        if not bool(getattr(args, "trajectory_only", False)):
            raise ValueError("--traj-late-focal-enable currently requires --trajectory-only")
        if str(getattr(args, "trajectory_target", "tool")).lower() not in ("tool", "both"):
            raise ValueError("--traj-late-focal-enable currently requires --trajectory-target tool|both")
        if traj_refiner_mode not in LAYERWISE_STATECARRY_REFINER_MODES and traj_refiner_mode not in SINGLELAYER_STATECARRY_REFINER_MODES:
            raise ValueError(
                "--traj-late-focal-enable currently requires a layerwise/singlelayer statecarry refiner mode"
            )
    if bool(args.tool_only) and bool(args.combo_probe_attn):
        raise ValueError("--tool-only and --combo-probe-attn are mutually exclusive")
    if bool(args.tool_only) and bool(args.combo_probe_dual_attn):
        raise ValueError("--tool-only and --combo-probe-dual-attn are mutually exclusive")
    if bool(args.combo_probe_attn) and bool(args.combo_probe_dual_attn):
        raise ValueError("--combo-probe-attn and --combo-probe-dual-attn are mutually exclusive")
    if bool(getattr(args, "dual_attn_tool_prior", False)) and (not bool(args.combo_probe_dual_attn)):
        raise ValueError("--dual-attn-tool-prior requires --combo-probe-dual-attn")
    if bool(args.no_vis):
        args.no_init_eval_vis = True
        args.no_checkpoint_vis = True

    # Fixed defaults kept here only for options that remain intentionally internal.
    args.input_size = int(getattr(args, "input_size", DEFAULT_INPUT_SIZE))
    args.proj_dim = DEFAULT_PROJ_DIM
    if int(args.input_size) <= 0:
        raise ValueError("--input-size must be > 0")
    if int(args.traj_head_hidden_dim) <= 0:
        args.traj_head_hidden_dim = int(args.proj_dim)
    if int(args.traj_head_up_layers) <= 0:
        raise ValueError("--traj-head-up-layers must be > 0")
    if float(args.combo_head_dropout) < 0.0 or float(args.combo_head_dropout) >= 1.0:
        raise ValueError("--combo-head-dropout must be in [0,1)")
    if float(args.combo_feat_dropout) < 0.0 or float(args.combo_feat_dropout) >= 1.0:
        raise ValueError("--combo-feat-dropout must be in [0,1)")
    if bool(args.ema):
        if float(args.ema_decay) <= 0.0 or float(args.ema_decay) >= 1.0:
            raise ValueError("--ema-decay must be in (0,1)")
    if float(args.combo_ce_weight) < 0.0:
        raise ValueError("--combo-ce-weight must be >= 0")
    if float(args.combo_tool_ce_weight) < 0.0:
        raise ValueError("--combo-tool-ce-weight must be >= 0")
    if float(args.combo_action_ce_weight) < 0.0:
        raise ValueError("--combo-action-ce-weight must be >= 0")
    if float(getattr(args, "traj_tool_offset_max_norm", 0.20)) <= 0.0:
        raise ValueError("--traj-tool-offset-max-norm must be > 0")
    args.range_num_bins = RANGE_NUM_BINS
    args.range_label_smoothing = RANGE_LABEL_SMOOTHING
    args.range_focal_gamma = RANGE_FOCAL_GAMMA
    args.range_neighbor_smoothing = RANGE_NEIGHBOR_SMOOTHING
    args.allow_random_vjepa_init = False
    args.vjepa_cache = not bool(args.no_vjepa_cache)
    args.use_weighted_ce = not bool(args.no_weighted_ce)
    args.use_balanced_sampler = not bool(args.no_balanced_sampler)

    # Optional debug/visualization knobs stay internal with stable defaults.
    args.max_train_steps = 0
    args.max_val_steps = 0
    args.vis_samples_per_ckpt = 6
    args.vis_max_frames = 16
    args.vis_heatmap_alpha = 0.45
    args.run_init_eval_vis = not bool(args.no_init_eval_vis)
    args.run_checkpoint_vis = not bool(args.no_checkpoint_vis)
    if not args.run_checkpoint_vis:
        args.vis_samples_per_ckpt = 0
    args.use_progress = not bool(args.no_progress)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace, num_combo_classes: int) -> KeypointComboNet:
    vjepa_num_frames = 16
    vjepa_tubelet_size = 2
    traj_num_subframes = int(getattr(args, "traj_num_subframes", 2))
    output_num_frames = int(vjepa_num_frames)
    if bool(getattr(args, "trajectory_only", False)):
        output_num_frames = int((vjepa_num_frames // vjepa_tubelet_size) * traj_num_subframes)
    vjepa = VJEPAWrapper(
        vjepa_repo=args.vjepa_repo,
        arch=args.vjepa_arch,
        img_size=args.input_size,
        num_frames=vjepa_num_frames,
        patch_size=16,
        tubelet_size=vjepa_tubelet_size,
        hf_repo=args.vjepa_hf_repo,
        hf_local_files_only=args.vjepa_hf_local_files_only,
        pretrained_from_hub=args.vjepa_pretrained_hub,
        checkpoint_path=args.vjepa_checkpoint,
        freeze=True,
        layer_indices=list(getattr(args, "vjepa_layer_indices", [])),
    )
    combo_feature_mode = (
        "direct_spatiotemporal"
        if bool(
            getattr(args, "combo_only", False)
            or getattr(args, "tool_only", False)
            or getattr(args, "range_only", False)
        )
        else "tool_weighted"
    )
    model = KeypointComboNet(
        vjepa=vjepa,
        proj_dim=args.proj_dim,
        num_combo_classes=num_combo_classes,
        num_tool_points=(1 if bool(getattr(args, "tool_centroid_only", False)) else 3),
        num_contact_points=4,
        output_num_frames=output_num_frames,
        num_range_bins=int(getattr(args, "range_num_bins", RANGE_NUM_BINS)),
        temporal_upsample=args.temporal_upsample,
        heatmap_softmax_temp=float(getattr(args, "heatmap_softmax_temp", 1.0)),
        enable_delta_branch=(not bool(getattr(args, "disable_delta_branch", False))),
        temporal_mean_residual_alpha=args.temporal_mean_residual_alpha,
        refiner_mode=args.refiner_mode,
        trajectory_only=bool(getattr(args, "trajectory_only", False)),
        range_only=bool(getattr(args, "range_only", False)),
        trajectory_target=str(getattr(args, "trajectory_target", "tool")),
        tool_only=bool(getattr(args, "tool_only", False)),
        combo_probe_attn=bool(getattr(args, "combo_probe_attn", False)),
        combo_probe_dual_attn=bool(getattr(args, "combo_probe_dual_attn", False)),
        combo_feature_mode=combo_feature_mode,
        combo_head_dropout=float(getattr(args, "combo_head_dropout", 0.0)),
        combo_feat_dropout=float(getattr(args, "combo_feat_dropout", 0.0)),
        dual_attn_tool_prior=bool(getattr(args, "dual_attn_tool_prior", False)),
        dual_attn_tool_prior_init_alpha=float(getattr(args, "dual_attn_tool_prior_init_alpha", 0.0)),
        dual_attn_tool_prior_detach=bool(getattr(args, "dual_attn_tool_prior_detach", True)),
        num_tool_classes=4,
        num_action_classes=4,
        traj_head_hidden_dim=int(getattr(args, "traj_head_hidden_dim", args.proj_dim)),
        traj_num_subframes=traj_num_subframes,
        traj_condition_mode=str(getattr(args, "traj_condition_mode", "tool")),
        range_condition_mode=str(getattr(args, "range_condition_mode", "tool_action")),
        range_head_mode=str(getattr(args, "range_head_mode", "scalar")),
        range_trident_num_bins=int(getattr(args, "range_trident_num_bins", 15)),
        range_proposal_mode=str(getattr(args, "range_proposal_mode", "none")),
        range_proposal_decode_mode=str(getattr(args, "range_proposal_decode_mode", "argmax")),
        range_quality_mode=str(getattr(args, "range_quality_mode", "none")),
        range_boundary_refine_mode=str(getattr(args, "range_boundary_refine_mode", "none")),
        range_boundary_refine_radius=int(getattr(args, "range_boundary_refine_radius", 2)),
        range_temporal_context_mode=str(getattr(args, "range_temporal_context_mode", "none")),
        range_temporal_context_kernel_size=int(getattr(args, "range_temporal_context_kernel_size", 3)),
        range_temporal_context_layers=int(getattr(args, "range_temporal_context_layers", 1)),
        range_temporal_context_levels=int(getattr(args, "range_temporal_context_levels", 3)),
        range_spatial_pool_mode=str(getattr(args, "range_spatial_pool_mode", "mean_max")),
        range_spatial_pool_num_tokens=int(getattr(args, "range_spatial_pool_num_tokens", 8)),
        range_adapter_mode=str(getattr(args, "range_adapter_mode", "none")),
        range_adapter_kernel_size=int(getattr(args, "range_adapter_kernel_size", 5)),
        range_adapter_num_heads=int(getattr(args, "range_adapter_num_heads", 4)),
        range_task_decouple_mode=str(getattr(args, "range_task_decouple_mode", "none")),
        range_duration_prior_mode=str(getattr(args, "range_duration_prior_mode", "none")),
        range_duration_prior_weight=float(getattr(args, "range_duration_prior_weight", 0.0)),
        range_duration_prior_means=getattr(args, "range_duration_prior_means", None),
        range_duration_prior_stds=getattr(args, "range_duration_prior_stds", None),
        range_duration_prior_min_std=float(getattr(args, "range_duration_prior_min_std", 0.05)),
        traj_refiner_mode=str(getattr(args, "traj_refiner_mode", "none")),
        traj_head_up_layers=int(getattr(args, "traj_head_up_layers", 2)),
        traj_recurrent_max_delta_norm=float(getattr(args, "traj_recurrent_max_delta_norm", 0.06)),
        traj_dynamic_centroid_max_beta=float(getattr(args, "traj_dynamic_centroid_max_beta", 0.75)),
        traj_tool_centroid_offset_head=bool(getattr(args, "traj_tool_centroid_offset_head", False)),
        traj_tool_offset_max_norm=float(getattr(args, "traj_tool_offset_max_norm", 0.20)),
        token_mask_enable=bool(getattr(args, "token_mask_enable", False)),
        token_mask_eval=bool(getattr(args, "token_mask_eval", False)),
        token_mask_strategy=str(getattr(args, "token_mask_strategy", "random")),
        token_mask_value=str(getattr(args, "token_mask_value", "learned_pos")),
        token_mask_p_max=float(getattr(args, "token_mask_p_max", 0.10)),
        token_mask_p_early=(
            float(getattr(args, "token_mask_p_max", 0.10))
            if (getattr(args, "token_mask_p_early", None) is None)
            else float(getattr(args, "token_mask_p_early"))
        ),
        token_mask_delay_epochs=int(getattr(args, "token_mask_delay_epochs", 5)),
        token_mask_warmup_epochs=int(getattr(args, "token_mask_warmup_epochs", 20)),
        token_mask_late_start_epoch=int(getattr(args, "token_mask_late_start_epoch", 25)),
        token_mask_p_late=float(getattr(args, "token_mask_p_late", 0.15)),
        token_mask_late_warmup_epochs=int(getattr(args, "token_mask_late_warmup_epochs", 40)),
        token_mask_keep_k=int(getattr(args, "token_mask_keep_k", 64)),
        token_mask_mix_801010=bool(getattr(args, "token_mask_mix_801010", True)),
        traj_stage0_centroid_noise_prob=float(getattr(args, "traj_stage0_centroid_noise_prob", 0.0)),
        traj_stage0_centroid_noise_sigma=float(getattr(args, "traj_stage0_centroid_noise_sigma", 0.0)),
        debug_timing=bool(getattr(args, "debug_model_timing", False)),
        debug_shapes=bool(getattr(args, "debug_model_shapes", False)),
    )
    return model


def _to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _get_amp_dtype(amp_dtype: str) -> torch.dtype:
    if amp_dtype == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_keyframe_data_root(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if (p / "train.txt").is_file() and (p / "combo_ids").is_dir():
        return p
    p_data = p / "data"
    if (p_data / "train.txt").is_file() and (p_data / "combo_ids").is_dir():
        return p_data
    raise FileNotFoundError(
        f"Cannot resolve dataset root from: {p}\n"
        "Expected either <root>/train.txt or <root>/data/train.txt"
    )


def _validate_range_only_data_root(path: str) -> None:
    root = Path(path).expanduser().resolve()
    required_dirs = [
        "sampled_abs_frames",
        "action_range_abs",
        "action_range_mask",
    ]
    missing = [name for name in required_dirs if not (root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            "range_only requires a full range-capable dataset root.\n"
            f"Checked: {root}\n"
            f"Missing directories: {missing}"
        )


def _best_monitor_name(args: argparse.Namespace) -> str:
    if bool(getattr(args, "range_only", False)):
        return "range_tiou"
    return "loss_total"


def _best_monitor_mode(args: argparse.Namespace) -> str:
    if bool(getattr(args, "range_only", False)):
        return "max"
    return "min"


def _initial_best_monitor_value(args: argparse.Namespace) -> float:
    return float("-inf") if _best_monitor_mode(args) == "max" else float("inf")


def _extract_best_monitor_value(args: argparse.Namespace, val_metrics: Dict[str, Any], default: float) -> float:
    key = _best_monitor_name(args)
    if key not in val_metrics:
        return float(default)
    try:
        return float(val_metrics[key])
    except Exception:
        return float(default)


def _is_better_best_monitor(args: argparse.Namespace, current: float, best: float) -> bool:
    if _best_monitor_mode(args) == "max":
        return float(current) > float(best)
    return float(current) < float(best)


def _is_better_best_monitor_with_margin(
    args: argparse.Namespace,
    current: float,
    best: float,
    *,
    min_delta: float = 0.0,
) -> bool:
    delta = max(float(min_delta), 0.0)
    if _best_monitor_mode(args) == "max":
        return float(current) > (float(best) + delta)
    return float(current) < (float(best) - delta)


def _sanitize_metrics_for_json(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(metrics, dict):
        return out
    for key, value in metrics.items():
        if isinstance(value, bool):
            out[str(key)] = bool(value)
            continue
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
            continue
        if isinstance(value, str):
            out[str(key)] = value
            continue
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_early_stop_bundle(
    out_dir: Path,
    *,
    report: Dict[str, Any],
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "early_stop_report.json"
    prompt_path = out_dir / "early_stop_prompt.txt"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    prompt_lines = [
        "Analyze this stopped training run based on the saved logs and summary.",
        "",
        f"Output dir: {out_dir}",
        f"Report: {report_path}",
        f"Stop reason: {report.get('reason', 'unknown')}",
        f"Epoch history: {out_dir / 'epoch_metrics.jsonl'}",
        f"Best checkpoint: {out_dir / 'checkpoints' / 'best.pt'}",
        "",
        "Focus on:",
        "1. Why validation tIoU stopped improving or regressed.",
        "2. Whether the issue looks like overfitting, optimization saturation, conditioning weakness, or label/data noise.",
        "3. What concrete model, loss, data, or training changes should be made next to improve tIoU.",
        "4. Rank the top next experiments by expected impact.",
    ]
    prompt_path.write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    return report_path, prompt_path


def _resolve_keyframe_split_file(data_root: Path, split: str, override: str) -> Path:
    if str(override).strip():
        p = Path(override).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Missing split file for {split}: {p}")
        return p
    p = data_root / f"{split}.txt"
    if not p.is_file():
        raise FileNotFoundError(f"Missing split file for {split}: {p}")
    return p


def _keyframe_soft_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _resize_time_distribution(x: torch.Tensor, out_t: int) -> torch.Tensor:
    in_t = int(x.shape[-1])
    if in_t == int(out_t):
        return x
    x_up = F.interpolate(x.unsqueeze(1), size=int(out_t), mode="linear", align_corners=False).squeeze(1)
    x_up = x_up.clamp_min(0.0)
    return x_up / x_up.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _resize_time_values(x: torch.Tensor, out_t: int) -> torch.Tensor:
    in_t = int(x.shape[-1])
    if in_t == int(out_t):
        return x
    return F.interpolate(x.unsqueeze(1), size=int(out_t), mode="linear", align_corners=False).squeeze(1)


class KeyframeTokenDataset(Dataset):
    def __init__(
        self,
        *,
        split: str,
        split_file: Path,
        cache_root: Path,
        softlabel_root: Path,
    ) -> None:
        super().__init__()
        sids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not sids:
            raise RuntimeError(f"No samples found in split file: {split_file}")

        samples: List[Dict[str, Any]] = []
        missing_token = 0
        missing_label = 0
        for sid in sids:
            token_path = cache_root / split / f"{sid}.npy"
            label_path = softlabel_root / split / f"{sid}.npz"
            if not token_path.is_file():
                missing_token += 1
                continue
            if not label_path.is_file():
                missing_label += 1
                continue
            samples.append(
                {
                    "sid": str(sid),
                    "token_path": token_path,
                    "softlabel_path": label_path,
                }
            )

        if not samples:
            raise RuntimeError(
                f"No valid samples for split={split}. "
                f"missing_token={missing_token}, missing_label={missing_label}"
            )

        self.split = str(split)
        self.samples = samples
        self.missing_token = int(missing_token)
        self.missing_label = int(missing_label)
        self.total_sid = int(len(sids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        token_path = Path(sample["token_path"])
        label_path = Path(sample["softlabel_path"])

        tok_np = _coerce_cached_token_grid_np(np.load(token_path, allow_pickle=False), source=str(token_path))

        z = np.load(label_path, allow_pickle=False)
        soft = z["soft_label"].astype(np.float32, copy=False)
        ferr = z["frame_error_px"].astype(np.float32, copy=False)
        fvalid = z["frame_valid"].astype(np.float32, copy=False)
        best = int(z["best_frame"].reshape(-1)[0])

        return {
            "sid": str(sample["sid"]),
            "tokens_grid": torch.from_numpy(tok_np.astype(np.float32, copy=False)),
            "soft_label": torch.from_numpy(soft),
            "frame_error_px": torch.from_numpy(ferr),
            "frame_valid": torch.from_numpy(fvalid),
            "best_frame": torch.tensor(best, dtype=torch.long),
        }


class KeyframeSelectorOfficialHead(torch.nn.Module):
    """VJEPA official attentive pooler + per-frame linear classifier."""

    def __init__(
        self,
        *,
        hf_repo: str,
        target_frames: int = 16,
        hf_local_files_only: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig
        from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2AttentivePooler

        cfg = AutoConfig.from_pretrained(hf_repo, local_files_only=bool(hf_local_files_only))
        if hasattr(cfg, "_attn_implementation") and getattr(cfg, "_attn_implementation") is None:
            cfg._attn_implementation = "eager"

        self.hidden_size = int(getattr(cfg, "hidden_size"))
        self.pooler = VJEPA2AttentivePooler(cfg)
        self.classifier = torch.nn.Linear(self.hidden_size, 1, bias=True)
        self.target_frames = int(target_frames)

    def _upsample_logits(self, logits: torch.Tensor) -> torch.Tensor:
        in_t = int(logits.shape[1])
        out_t = int(self.target_frames)
        if in_t == out_t:
            return logits
        if out_t % in_t == 0:
            rep = out_t // in_t
            return logits.repeat_interleave(rep, dim=1)
        return F.interpolate(logits.unsqueeze(1), size=out_t, mode="linear", align_corners=False).squeeze(1)

    def forward(self, tokens_grid: torch.Tensor) -> torch.Tensor:
        if tokens_grid.ndim != 5:
            raise ValueError(f"`tokens_grid` must be [B,T',H,W,D], got {tuple(tokens_grid.shape)}")

        bsz, t_prime, h, w, d = tokens_grid.shape
        if int(d) != int(self.hidden_size):
            raise ValueError(f"Token dim mismatch: got {d}, expected {self.hidden_size}")

        x = tokens_grid.reshape(bsz * t_prime, h * w, d)
        frame_feat = self.pooler(x)
        logits = self.classifier(frame_feat).reshape(bsz, t_prime)
        logits = self._upsample_logits(logits)
        return logits


def _keyframe_update_meter(
    meter: Dict[str, float],
    *,
    loss_value: float,
    logits: torch.Tensor,
    frame_error_px: torch.Tensor,
    best_frame: torch.Tensor,
) -> None:
    bsz = int(logits.shape[0])
    meter["loss_sum"] += float(loss_value) * float(bsz)
    meter["n"] += float(bsz)

    pred_idx = torch.argmax(logits, dim=-1)
    b_idx = torch.arange(bsz, device=logits.device)
    sel_err = frame_error_px[b_idx, pred_idx]
    valid_sel = torch.isfinite(sel_err)
    if bool(valid_sel.any().item()):
        meter["sel_sum"] += float(sel_err[valid_sel].sum().item())
        meter["sel_n"] += float(valid_sel.sum().item())

    ferr_masked = torch.where(
        torch.isfinite(frame_error_px),
        frame_error_px,
        torch.full_like(frame_error_px, float("inf")),
    )
    oracle = torch.min(ferr_masked, dim=-1).values
    valid_oracle = torch.isfinite(oracle)
    if bool(valid_oracle.any().item()):
        meter["oracle_sum"] += float(oracle[valid_oracle].sum().item())
        meter["oracle_n"] += float(valid_oracle.sum().item())

    valid_best = (best_frame >= 0) & (best_frame < int(logits.shape[1]))
    if bool(valid_best.any().item()):
        hits = (pred_idx[valid_best] == best_frame[valid_best]).float().sum()
        meter["top1_sum"] += float(hits.item())
        meter["top1_n"] += float(valid_best.sum().item())


def _keyframe_finalize_meter(meter: Dict[str, float]) -> Dict[str, float]:
    loss = _safe_div(meter.get("loss_sum", 0.0), meter.get("n", 0.0), default=float("nan"))
    sel = _safe_div(meter.get("sel_sum", 0.0), meter.get("sel_n", 0.0), default=float("nan"))
    oracle = _safe_div(meter.get("oracle_sum", 0.0), meter.get("oracle_n", 0.0), default=float("nan"))
    top1 = _safe_div(meter.get("top1_sum", 0.0), meter.get("top1_n", 0.0), default=float("nan"))
    gap = float(sel - oracle) if (np.isfinite(sel) and np.isfinite(oracle)) else float("nan")
    return {
        "loss_total": float(loss),
        "loss_keyframe_ce": float(loss),
        "sel_ade_px": float(sel),
        "oracle_ade_px": float(oracle),
        "gap_px": float(gap),
        "top1_best_frame_acc": float(top1),
    }


def _evaluate_keyframe_selector(
    *,
    model: KeyframeSelectorOfficialHead,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    meter: Dict[str, float] = defaultdict(float)
    use_progress = _is_progress_enabled(args)
    total_steps = len(loader)
    if args.max_val_steps > 0:
        total_steps = min(total_steps, int(args.max_val_steps))

    iterator = loader
    pbar = None
    if use_progress:
        pbar = tqdm(
            loader,
            total=total_steps,
            desc=f"Val E{epoch:03d}",
            dynamic_ncols=True,
            leave=False,
        )
        iterator = pbar

    with torch.no_grad():
        for step, batch in enumerate(iterator, start=1):
            if args.max_val_steps > 0 and step > args.max_val_steps:
                break

            tokens_grid = batch["tokens_grid"].to(device=device, non_blocking=True)
            soft = batch["soft_label"].to(device=device, dtype=torch.float32, non_blocking=True)
            ferr = batch["frame_error_px"].to(device=device, dtype=torch.float32, non_blocking=True)
            fvalid = batch["frame_valid"].to(device=device, dtype=torch.float32, non_blocking=True)
            best_frame = batch["best_frame"].to(device=device, dtype=torch.long, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(tokens_grid)
                if int(logits.shape[1]) != int(soft.shape[1]):
                    out_t = int(logits.shape[1])
                    soft = _resize_time_distribution(soft, out_t)
                    ferr = _resize_time_values(torch.nan_to_num(ferr, nan=0.0).clamp_min(0.0), out_t)
                    fvalid = _resize_time_values(fvalid.clamp_min(0.0), out_t)
                    ferr = torch.where(fvalid > 0.0, ferr, torch.full_like(ferr, float("inf")))
                    best_frame = torch.argmax(soft, dim=-1)
                loss = _keyframe_soft_ce_loss(logits, soft)

            _keyframe_update_meter(
                meter,
                loss_value=float(loss.item()),
                logits=logits.detach(),
                frame_error_px=ferr.detach(),
                best_frame=best_frame.detach(),
            )

            if pbar is not None:
                m = _keyframe_finalize_meter(meter)
                pbar.set_postfix(
                    loss=f"{m['loss_total']:.4f}",
                    ade=f"{m['sel_ade_px']:.2f}",
                    top1=f"{m['top1_best_frame_acc']:.3f}",
                )

    if pbar is not None:
        pbar.close()
    return _keyframe_finalize_meter(meter)


def _run_keyframe_selector_training(args: argparse.Namespace) -> None:
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp_dtype = _get_amp_dtype(args.amp_dtype)
    amp_enabled = bool(args.amp and device.type == "cuda")

    data_root = _resolve_keyframe_data_root(str(args.data_root))
    softlabel_root = Path(args.keyframe_softlabel_root).expanduser().resolve()
    if not softlabel_root.is_dir():
        raise FileNotFoundError(f"Softlabel root not found: {softlabel_root}")
    cache_root = (
        Path(args.keyframe_cache_root).expanduser().resolve()
        if str(args.keyframe_cache_root).strip()
        else (data_root / "vjepa_token_cache")
    )
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Token cache root not found: {cache_root}")

    train_split_file = _resolve_keyframe_split_file(data_root, "train", str(args.train_split_file))
    val_split_file = _resolve_keyframe_split_file(data_root, "val", str(args.val_split_file))

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pin_memory = device.type == "cuda"
    train_ds = KeyframeTokenDataset(
        split="train",
        split_file=train_split_file,
        cache_root=cache_root,
        softlabel_root=softlabel_root,
    )
    val_ds = KeyframeTokenDataset(
        split="val",
        split_file=val_split_file,
        cache_root=cache_root,
        softlabel_root=softlabel_root,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = KeyframeSelectorOfficialHead(
        hf_repo=str(args.vjepa_hf_repo),
        target_frames=16,
        hf_local_files_only=bool(args.vjepa_hf_local_files_only),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scaler_enabled = bool(amp_enabled and amp_dtype == torch.float16)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    run_cfg = {
        "mode": "keyframe_selector_only",
        "data_root": str(data_root),
        "train_split_file": str(train_split_file),
        "val_split_file": str(val_split_file),
        "softlabel_root": str(softlabel_root),
        "cache_root": str(cache_root),
        "output_dir": str(out_dir),
        "hf_repo": str(args.vjepa_hf_repo),
        "hf_local_files_only": bool(args.vjepa_hf_local_files_only),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "seed": int(args.seed),
        "device": str(device),
        "amp": bool(args.amp),
        "amp_dtype": str(args.amp_dtype),
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "train_total_sid": int(train_ds.total_sid),
        "val_total_sid": int(val_ds.total_sid),
        "train_missing_token": int(train_ds.missing_token),
        "train_missing_label": int(train_ds.missing_label),
        "val_missing_token": int(val_ds.missing_token),
        "val_missing_label": int(val_ds.missing_label),
        "save_interval": int(args.save_interval),
        "log_interval": int(args.log_interval),
    }
    run_cfg_path = out_dir / "run_config.json"
    run_cfg_path.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    tb_writer: Optional["SummaryWriter"] = None
    if args.train_platform == "tensorboard":
        if SummaryWriter is None:
            print("[warn] TensorBoard unavailable: pip install tensorboard")
        else:
            tb_dir = Path(args.tb_dir) if str(args.tb_dir).strip() else (out_dir / "tensorboard")
            tb_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))

    _print_stage_header("KEYFRAME SELECTOR SETUP")
    print("[setup]")
    for k in sorted(run_cfg.keys()):
        print(f"  {k}={run_cfg[k]}")
    print(
        f"[setup] grad_contract: "
        f"pooler={_module_grad_stats(model.pooler)['requires_grad_params']} "
        f"classifier={_module_grad_stats(model.classifier)['requires_grad_params']}"
    )

    best_val = float("inf")
    history: List[Dict[str, float]] = []
    global_step = 0

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        _print_stage_header(f"EPOCH {epoch:03d}/{args.epochs:03d}")

        model.train()
        meter: Dict[str, float] = defaultdict(float)
        log_meter: Dict[str, float] = defaultdict(float)
        use_progress = _is_progress_enabled(args)
        total_steps = len(train_loader)
        if args.max_train_steps > 0:
            total_steps = min(total_steps, int(args.max_train_steps))

        iterator = train_loader
        pbar = None
        if use_progress:
            pbar = tqdm(
                train_loader,
                total=total_steps,
                desc=f"Train E{epoch:03d}",
                dynamic_ncols=True,
                leave=False,
            )
            iterator = pbar
        log_write = pbar.write if pbar is not None else print

        for step, batch in enumerate(iterator, start=1):
            if args.max_train_steps > 0 and step > args.max_train_steps:
                break

            tokens_grid = batch["tokens_grid"].to(device=device, non_blocking=True)
            soft = batch["soft_label"].to(device=device, dtype=torch.float32, non_blocking=True)
            ferr = batch["frame_error_px"].to(device=device, dtype=torch.float32, non_blocking=True)
            fvalid = batch["frame_valid"].to(device=device, dtype=torch.float32, non_blocking=True)
            best_frame = batch["best_frame"].to(device=device, dtype=torch.long, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(tokens_grid)
                if int(logits.shape[1]) != int(soft.shape[1]):
                    out_t = int(logits.shape[1])
                    soft = _resize_time_distribution(soft, out_t)
                    ferr = _resize_time_values(torch.nan_to_num(ferr, nan=0.0).clamp_min(0.0), out_t)
                    fvalid = _resize_time_values(fvalid.clamp_min(0.0), out_t)
                    ferr = torch.where(fvalid > 0.0, ferr, torch.full_like(ferr, float("inf")))
                    best_frame = torch.argmax(soft, dim=-1)
                loss = _keyframe_soft_ce_loss(logits, soft)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if float(args.grad_clip) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(args.grad_clip) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optimizer.step()

            global_step += 1
            _keyframe_update_meter(
                meter,
                loss_value=float(loss.item()),
                logits=logits.detach(),
                frame_error_px=ferr.detach(),
                best_frame=best_frame.detach(),
            )
            _keyframe_update_meter(
                log_meter,
                loss_value=float(loss.item()),
                logits=logits.detach(),
                frame_error_px=ferr.detach(),
                best_frame=best_frame.detach(),
            )

            if (args.log_interval > 0) and (global_step % int(args.log_interval) == 0):
                log_metrics = _keyframe_finalize_meter(log_meter)
                lr_now = float(optimizer.param_groups[0]["lr"])
                log_write(
                    f"[train][epoch {epoch:03d}][step {step:05d}][gstep {global_step:07d}] "
                    f"total={log_metrics['loss_total']:.5f} "
                    f"keyframe_ce={log_metrics['loss_keyframe_ce']:.5f} "
                    f"sel_ade_px={log_metrics['sel_ade_px']:.3f} "
                    f"oracle_ade_px={log_metrics['oracle_ade_px']:.3f} "
                    f"gap_px={log_metrics['gap_px']:.3f} "
                    f"top1={log_metrics['top1_best_frame_acc']:.4f} "
                    f"lr={lr_now:.2e}"
                )
                if pbar is not None:
                    pbar.set_postfix(
                        loss=f"{log_metrics['loss_total']:.4f}",
                        ade=f"{log_metrics['sel_ade_px']:.2f}",
                        top1=f"{log_metrics['top1_best_frame_acc']:.3f}",
                        lr=f"{lr_now:.2e}",
                    )
                log_meter = defaultdict(float)

        if pbar is not None:
            pbar.close()

        train_metrics = _keyframe_finalize_meter(meter)
        val_metrics = _evaluate_keyframe_selector(
            model=model,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            epoch=epoch,
            args=args,
        )

        print(
            f"[epoch {epoch:03d}] train: "
            f"total={train_metrics['loss_total']:.5f} "
            f"keyframe_ce={train_metrics['loss_keyframe_ce']:.5f} "
            f"sel_ade_px={train_metrics['sel_ade_px']:.3f} "
            f"oracle_ade_px={train_metrics['oracle_ade_px']:.3f} "
            f"gap_px={train_metrics['gap_px']:.3f} "
            f"top1={train_metrics['top1_best_frame_acc']:.4f}"
        )
        print(
            f"[epoch {epoch:03d}]   val: "
            f"total={val_metrics['loss_total']:.5f} "
            f"keyframe_ce={val_metrics['loss_keyframe_ce']:.5f} "
            f"sel_ade_px={val_metrics['sel_ade_px']:.3f} "
            f"oracle_ade_px={val_metrics['oracle_ade_px']:.3f} "
            f"gap_px={val_metrics['gap_px']:.3f} "
            f"top1={val_metrics['top1_best_frame_acc']:.4f}"
        )

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch_loss_total", train_metrics["loss_total"], epoch)
            tb_writer.add_scalar("train/epoch_sel_ade_px", train_metrics["sel_ade_px"], epoch)
            tb_writer.add_scalar("train/epoch_top1", train_metrics["top1_best_frame_acc"], epoch)
            tb_writer.add_scalar("val/epoch_loss_total", val_metrics["loss_total"], epoch)
            tb_writer.add_scalar("val/epoch_sel_ade_px", val_metrics["sel_ade_px"], epoch)
            tb_writer.add_scalar("val/epoch_oracle_ade_px", val_metrics["oracle_ade_px"], epoch)
            tb_writer.add_scalar("val/epoch_gap_px", val_metrics["gap_px"], epoch)
            tb_writer.add_scalar("val/epoch_top1", val_metrics["top1_best_frame_acc"], epoch)

        row = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train_loss_total": float(train_metrics["loss_total"]),
            "train_sel_ade_px": float(train_metrics["sel_ade_px"]),
            "train_oracle_ade_px": float(train_metrics["oracle_ade_px"]),
            "train_gap_px": float(train_metrics["gap_px"]),
            "train_top1": float(train_metrics["top1_best_frame_acc"]),
            "val_loss_total": float(val_metrics["loss_total"]),
            "val_sel_ade_px": float(val_metrics["sel_ade_px"]),
            "val_oracle_ade_px": float(val_metrics["oracle_ade_px"]),
            "val_gap_px": float(val_metrics["gap_px"]),
            "val_top1": float(val_metrics["top1_best_frame_acc"]),
        }
        history.append(row)
        history_path = out_dir / "history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        ckpt = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "run_cfg": run_cfg,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        if scaler.is_enabled():
            ckpt["scaler"] = scaler.state_dict()

        save_epoch_checkpoint = (
            ((epoch % int(args.save_interval) == 0) or (epoch == int(args.epochs)))
            and (not bool(getattr(args, "save_best_only", False)))
        )
        if save_epoch_checkpoint:
            ep_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save(ckpt, ep_path)
            print(f"[checkpoint] saved: {ep_path}")

        val_sel = float(val_metrics["sel_ade_px"])
        if np.isfinite(val_sel) and val_sel < float(best_val):
            best_val = float(val_sel)
            best_path = ckpt_dir / "best.pt"
            torch.save(ckpt, best_path)
            print(f"[checkpoint] updated best: {best_path} (val_sel_ade_px={best_val:.6f})")

        print(f"[epoch {epoch:03d}] elapsed={time.time() - t0:.1f}s")

    if tb_writer is not None:
        tb_writer.close()
    print(f"[done] history={out_dir / 'history.json'}")


def _collapse_tool_gt_to_centroid(
    gt_tool_xy: torch.Tensor,
    gt_tool_vis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if gt_tool_xy.ndim != 4 or gt_tool_xy.shape[-1] != 2:
        raise ValueError(f"`gt_tool_xy` must be [B,T,P,2], got {tuple(gt_tool_xy.shape)}")
    if gt_tool_vis.ndim != 3:
        raise ValueError(f"`gt_tool_vis` must be [B,T,P], got {tuple(gt_tool_vis.shape)}")
    if tuple(gt_tool_vis.shape) != tuple(gt_tool_xy.shape[:3]):
        raise ValueError(
            f"`gt_tool_vis` shape must match gt_tool_xy[:3], got {tuple(gt_tool_vis.shape)} vs {tuple(gt_tool_xy.shape[:3])}"
        )
    if int(gt_tool_xy.shape[2]) <= 0:
        raise ValueError(f"`gt_tool_xy` must have P>0, got {tuple(gt_tool_xy.shape)}")
    centroid_xy = gt_tool_xy.mean(dim=2, keepdim=True)
    centroid_vis = gt_tool_vis.amax(dim=2, keepdim=True)
    return centroid_xy, centroid_vis


def _maybe_use_tool_centroid_gt(
    gt_tool_xy: torch.Tensor,
    gt_tool_vis: torch.Tensor,
    *,
    use_tool_centroid_only: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not bool(use_tool_centroid_only):
        return gt_tool_xy, gt_tool_vis
    return _collapse_tool_gt_to_centroid(gt_tool_xy=gt_tool_xy, gt_tool_vis=gt_tool_vis)


def _derive_action_norm_from_bins(
    gt_action_start_bin: torch.Tensor,
    gt_action_end_bin: torch.Tensor,
    num_bins: int = RANGE_NUM_BINS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    den = float(max(int(num_bins) - 1, 1))
    s = gt_action_start_bin.float() / den
    e = gt_action_end_bin.float() / den
    e = torch.maximum(e, s)
    return s.clamp(0.0, 1.0), e.clamp(0.0, 1.0)


def _derive_tool_action_id_from_combo_id(combo_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # combo_id mapping:
    # 0 hook|dissect, 1 grasper|grasp, 2 clipper|clip,
    # 3 scissors|cut, 4 scissors|dissect, 5 grasper|dissect
    # tool_id:   0 hook, 1 grasper, 2 clipper, 3 scissors
    # action_id: 0 dissect, 1 grasp, 2 clip, 3 cut
    combo = combo_id.long()
    combo_to_tool = torch.tensor([0, 1, 2, 3, 3, 1], device=combo.device, dtype=torch.long)
    combo_to_action = torch.tensor([0, 1, 2, 3, 0, 0], device=combo.device, dtype=torch.long)
    if bool((combo < 0).any().item()) or bool((combo >= int(combo_to_tool.shape[0])).any().item()):
        raise ValueError(
            f"`combo_id` out of mapping range [0,{int(combo_to_tool.shape[0]) - 1}] "
            f"(min={int(combo.min().item())}, max={int(combo.max().item())})"
        )
    return combo_to_tool[combo], combo_to_action[combo]


def _structured_combo_logits_from_tool_action(
    combo_logits: torch.Tensor,
    combo_tool_logits: torch.Tensor,
    combo_action_logits: torch.Tensor,
) -> torch.Tensor:
    """Build structured combo logits from tool/action logits.

    combo_id mapping (fixed):
      0 hook|dissect
      1 grasper|grasp
      2 clipper|clip
      3 scissors|cut
      4 scissors|dissect
      5 grasper|dissect
    """
    if combo_logits.ndim != 2:
        return combo_logits
    if combo_tool_logits.ndim != 2 or combo_action_logits.ndim != 2:
        return combo_logits
    if int(combo_tool_logits.shape[0]) != int(combo_logits.shape[0]):
        return combo_logits
    if int(combo_action_logits.shape[0]) != int(combo_logits.shape[0]):
        return combo_logits

    bsz, n_combo = combo_logits.shape
    combo_to_tool = torch.tensor([0, 1, 2, 3, 3, 1], device=combo_logits.device, dtype=torch.long)
    combo_to_action = torch.tensor([0, 1, 2, 3, 0, 0], device=combo_logits.device, dtype=torch.long)
    if int(n_combo) != int(combo_to_tool.shape[0]):
        return combo_logits

    n_tool = int(combo_tool_logits.shape[1])
    n_action = int(combo_action_logits.shape[1])
    structured = combo_logits.new_full((bsz, n_combo), -1e9)

    valid = (combo_to_tool >= 0) & (combo_to_tool < n_tool) & (combo_to_action >= 0) & (combo_to_action < n_action)
    valid_ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if int(valid_ids.numel()) == 0:
        return combo_logits

    for cid_t in valid_ids:
        cid = int(cid_t.item())
        tool_idx = int(combo_to_tool[cid].item())
        action_idx = int(combo_to_action[cid].item())
        structured[:, cid] = combo_tool_logits[:, tool_idx] + combo_action_logits[:, action_idx]
    return structured


def _loss_value_for_sample(x: torch.Tensor, sample_idx: int) -> float:
    if not torch.is_tensor(x):
        return float(x)
    if x.ndim == 0:
        return float(x.item())
    return float(x.reshape(-1)[int(sample_idx)].item())


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if float(den) <= 0.0:
        return float(default)
    return float(num) / float(den)


def _compute_classification_report_from_confusion(
    *,
    class_ids: List[int],
    class_names: List[str],
    confusion_matrix: np.ndarray,
) -> Dict[str, Any]:
    cm = np.asarray(confusion_matrix, dtype=np.int64)
    if cm.ndim != 2 or int(cm.shape[0]) != int(cm.shape[1]):
        raise ValueError(f"Invalid confusion matrix shape: {tuple(cm.shape)}")
    if int(cm.shape[0]) != int(len(class_ids)) or int(cm.shape[0]) != int(len(class_names)):
        raise ValueError(
            f"Confusion matrix / class metadata mismatch: cm={tuple(cm.shape)} ids={len(class_ids)} names={len(class_names)}"
        )

    total = int(cm.sum())
    diag = np.diag(cm)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    correct = int(diag.sum())
    errors = int(total - correct)

    per_class: List[Dict[str, Any]] = []
    f1_vals: List[float] = []
    precision_vals: List[float] = []
    recall_vals: List[float] = []
    support_vals: List[int] = []

    for i, cid in enumerate(class_ids):
        tp = int(diag[i])
        fp = int(col_sum[i] - tp)
        fn = int(row_sum[i] - tp)
        tn = int(total - tp - fp - fn)
        support = int(row_sum[i])
        precision = _safe_div(tp, tp + fp, default=0.0)
        recall = _safe_div(tp, tp + fn, default=0.0)
        f1 = _safe_div(2.0 * precision * recall, precision + recall, default=0.0)

        fn_to: List[Dict[str, Any]] = []
        if fn > 0:
            for j in np.argsort(-cm[i, :]):
                jj = int(j)
                if jj == i:
                    continue
                c = int(cm[i, jj])
                if c <= 0:
                    continue
                fn_to.append(
                    {
                        "pred_combo_id": int(class_ids[jj]),
                        "pred_combo_name": str(class_names[jj]),
                        "count": c,
                        "ratio_in_fn": _safe_div(c, fn, default=0.0),
                    }
                )

        fp_from: List[Dict[str, Any]] = []
        if fp > 0:
            col = cm[:, i]
            for j in np.argsort(-col):
                jj = int(j)
                if jj == i:
                    continue
                c = int(cm[jj, i])
                if c <= 0:
                    continue
                fp_from.append(
                    {
                        "true_combo_id": int(class_ids[jj]),
                        "true_combo_name": str(class_names[jj]),
                        "count": c,
                        "ratio_in_fp": _safe_div(c, fp, default=0.0),
                    }
                )

        per_class.append(
            {
                "combo_id": int(cid),
                "combo_name": str(class_names[i]),
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "false_negative_to": fn_to,
                "false_positive_from": fp_from,
            }
        )
        f1_vals.append(float(f1))
        precision_vals.append(float(precision))
        recall_vals.append(float(recall))
        support_vals.append(int(support))

    macro_precision = float(np.mean(precision_vals)) if precision_vals else float("nan")
    macro_recall = float(np.mean(recall_vals)) if recall_vals else float("nan")
    macro_f1 = float(np.mean(f1_vals)) if f1_vals else float("nan")
    weighted_f1 = (
        float(np.average(np.asarray(f1_vals, dtype=np.float64), weights=np.asarray(support_vals, dtype=np.float64)))
        if support_vals and (sum(support_vals) > 0)
        else float("nan")
    )

    mistake_pairs: List[Dict[str, Any]] = []
    for i, true_cid in enumerate(class_ids):
        true_total = int(row_sum[i])
        if true_total <= 0:
            continue
        for j, pred_cid in enumerate(class_ids):
            if i == j:
                continue
            c = int(cm[i, j])
            if c <= 0:
                continue
            mistake_pairs.append(
                {
                    "true_combo_id": int(true_cid),
                    "true_combo_name": str(class_names[i]),
                    "pred_combo_id": int(pred_cid),
                    "pred_combo_name": str(class_names[j]),
                    "count": c,
                    "ratio_in_true_class": _safe_div(c, true_total, default=0.0),
                }
            )
    mistake_pairs.sort(
        key=lambda x: (
            -int(x["count"]),
            -float(x["ratio_in_true_class"]),
            str(x["true_combo_name"]),
            str(x["pred_combo_name"]),
        )
    )

    return {
        "num_classes": int(len(class_ids)),
        "class_ids": [int(x) for x in class_ids],
        "class_names": [str(x) for x in class_names],
        "combo_id_to_name": {str(cid): str(name) for cid, name in zip(class_ids, class_names)},
        "total_samples": int(total),
        "correct_samples": int(correct),
        "error_samples": int(errors),
        "accuracy": _safe_div(correct, total, default=0.0),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix_true_rows_pred_cols": cm.astype(np.int64).tolist(),
        "per_class": per_class,
        "mistake_pairs": mistake_pairs,
    }


def _build_combo_fpfn_report(
    *,
    true_combo_ids: List[int],
    pred_combo_ids: List[int],
    sids: List[str],
    combo_id_to_name: Dict[int, str],
) -> Dict[str, Any]:
    if len(true_combo_ids) != len(pred_combo_ids):
        raise ValueError(
            f"Length mismatch true/pred: {len(true_combo_ids)} vs {len(pred_combo_ids)}"
        )
    if len(sids) != len(true_combo_ids):
        raise ValueError(f"Length mismatch sid/preds: {len(sids)} vs {len(true_combo_ids)}")

    max_id = -1
    for x in true_combo_ids:
        max_id = max(max_id, int(x))
    for x in pred_combo_ids:
        max_id = max(max_id, int(x))
    for x in combo_id_to_name.keys():
        max_id = max(max_id, int(x))
    if max_id < 0:
        return {
            "num_classes": 0,
            "class_ids": [],
            "class_names": [],
            "combo_id_to_name": {},
            "total_samples": 0,
            "correct_samples": 0,
            "error_samples": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "confusion_matrix_true_rows_pred_cols": [],
            "per_class": [],
            "mistake_pairs": [],
            "misclassified_samples": [],
        }

    class_ids = list(range(int(max_id) + 1))
    class_names = [str(combo_id_to_name.get(int(cid), f"combo_{int(cid)}")) for cid in class_ids]
    n_cls = int(len(class_ids))
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    misclassified_samples: List[Dict[str, Any]] = []
    for sid, t_raw, p_raw in zip(sids, true_combo_ids, pred_combo_ids):
        t = int(t_raw)
        p = int(p_raw)
        if (t < 0) or (t >= n_cls) or (p < 0) or (p >= n_cls):
            continue
        cm[t, p] += 1
        if t != p:
            misclassified_samples.append(
                {
                    "sid": str(sid),
                    "true_combo_id": int(t),
                    "true_combo_name": str(class_names[t]),
                    "pred_combo_id": int(p),
                    "pred_combo_name": str(class_names[p]),
                }
            )

    out = _compute_classification_report_from_confusion(
        class_ids=class_ids,
        class_names=class_names,
        confusion_matrix=cm,
    )
    out["misclassified_samples"] = misclassified_samples
    return out


def _compute_refine_stats(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    gate_vals: List[float] = []
    delta_vals: List[float] = []
    corr_entropy_vals: List[float] = []
    corr_margin_vals: List[float] = []
    centroid_shift_vals: List[float] = []
    context_entropy_vals: List[float] = []
    context_margin_vals: List[float] = []
    context_offset_abs_vals: List[float] = []
    context_gate_vals: List[float] = []
    belief_gate_vals: List[float] = []
    anchor_delta_centroid_vals: List[float] = []
    anchor_alpha_vals: List[float] = []
    factor_anchor_gate_vals: List[float] = []
    factor_offset_gate_vals: List[float] = []
    factor_delta_offset_abs_vals: List[float] = []
    anchor_verify_scale_vals: List[float] = []
    anchor_verify_residual_vals: List[float] = []
    stage0_centroid_noise_vals: List[float] = []
    joint_gate_vals: List[float] = []
    template_entropy_vals: List[float] = []
    template_margin_vals: List[float] = []
    template_gate_vals: List[float] = []
    template_offset_abs_vals: List[float] = []
    template_delta_abs_vals: List[float] = []

    gate_tool = outputs.get("traj_refine_delta_tool_gate", None)
    gate_ct = outputs.get("traj_refine_delta_ct_gate", None)
    delta_tool = outputs.get("traj_refine_delta_tool_xy", None)
    delta_ct = outputs.get("traj_refine_delta_ct_xy", None)
    corr_entropy_tool = outputs.get("traj_refine_corr_entropy_tool", None)
    corr_entropy_ct = outputs.get("traj_refine_corr_entropy_ct", None)
    corr_margin_tool = outputs.get("traj_refine_corr_margin_tool", None)
    corr_margin_ct = outputs.get("traj_refine_corr_margin_ct", None)
    centroid_shift_tool = outputs.get("traj_refine_centroid_shift_tool", None)
    centroid_shift_ct = outputs.get("traj_refine_centroid_shift_ct", None)
    context_entropy_tool = outputs.get("traj_refine_context_entropy_tool", None)
    context_entropy_ct = outputs.get("traj_refine_context_entropy_ct", None)
    context_margin_tool = outputs.get("traj_refine_context_margin_tool", None)
    context_margin_ct = outputs.get("traj_refine_context_margin_ct", None)
    context_offset_abs_tool = outputs.get("traj_refine_context_offset_abs_tool", None)
    context_offset_abs_ct = outputs.get("traj_refine_context_offset_abs_ct", None)
    context_gate_tool = outputs.get("traj_refine_context_gate_tool", None)
    context_gate_ct = outputs.get("traj_refine_context_gate_ct", None)
    belief_gate_tool = outputs.get("traj_refine_belief_gate_tool", None)
    belief_gate_ct = outputs.get("traj_refine_belief_gate_ct", None)
    anchor_delta_centroid_tool = outputs.get("traj_refine_anchor_delta_centroid_tool", None)
    anchor_delta_centroid_ct = outputs.get("traj_refine_anchor_delta_centroid_ct", None)
    anchor_alpha_tool = outputs.get("traj_refine_anchor_alpha_tool", None)
    anchor_alpha_ct = outputs.get("traj_refine_anchor_alpha_ct", None)
    factor_anchor_gate_tool = outputs.get("traj_refine_factor_anchor_gate_tool", None)
    factor_anchor_gate_ct = outputs.get("traj_refine_factor_anchor_gate_ct", None)
    factor_offset_gate_tool = outputs.get("traj_refine_factor_offset_gate_tool", None)
    factor_offset_gate_ct = outputs.get("traj_refine_factor_offset_gate_ct", None)
    factor_delta_offset_abs_tool = outputs.get("traj_refine_factor_delta_offset_abs_tool", None)
    factor_delta_offset_abs_ct = outputs.get("traj_refine_factor_delta_offset_abs_ct", None)
    anchor_verify_scale_tool = outputs.get("traj_refine_anchor_verify_scale_tool", None)
    anchor_verify_scale_ct = outputs.get("traj_refine_anchor_verify_scale_ct", None)
    anchor_verify_residual_tool = outputs.get("traj_refine_anchor_verify_residual_tool", None)
    anchor_verify_residual_ct = outputs.get("traj_refine_anchor_verify_residual_ct", None)
    stage0_centroid_noise_tool = outputs.get("traj_refine_stage0_centroid_noise_abs_tool", None)
    stage0_centroid_noise_ct = outputs.get("traj_refine_stage0_centroid_noise_abs_ct", None)
    joint_gate_tool = outputs.get("traj_refine_joint_gate_tool", None)
    joint_gate_ct = outputs.get("traj_refine_joint_gate_ct", None)
    template_entropy_tool = outputs.get("traj_refine_template_entropy_tool", None)
    template_entropy_ct = outputs.get("traj_refine_template_entropy_ct", None)
    template_margin_tool = outputs.get("traj_refine_template_margin_tool", None)
    template_margin_ct = outputs.get("traj_refine_template_margin_ct", None)
    template_gate_tool = outputs.get("traj_refine_template_gate_tool", None)
    template_gate_ct = outputs.get("traj_refine_template_gate_ct", None)
    template_offset_abs_tool = outputs.get("traj_refine_template_offset_abs_tool", None)
    template_offset_abs_ct = outputs.get("traj_refine_template_offset_abs_ct", None)
    template_delta_abs_tool = outputs.get("traj_refine_template_delta_abs_tool", None)
    template_delta_abs_ct = outputs.get("traj_refine_template_delta_abs_ct", None)

    if torch.is_tensor(gate_tool):
        gate_vals.append(float(gate_tool.detach().float().mean().item()))
    if torch.is_tensor(gate_ct):
        gate_vals.append(float(gate_ct.detach().float().mean().item()))
    if torch.is_tensor(delta_tool):
        delta_vals.append(float(delta_tool.detach().float().abs().mean().item()))
    if torch.is_tensor(delta_ct):
        delta_vals.append(float(delta_ct.detach().float().abs().mean().item()))
    if torch.is_tensor(corr_entropy_tool):
        corr_entropy_vals.append(float(corr_entropy_tool.detach().float().mean().item()))
    if torch.is_tensor(corr_entropy_ct):
        corr_entropy_vals.append(float(corr_entropy_ct.detach().float().mean().item()))
    if torch.is_tensor(corr_margin_tool):
        corr_margin_vals.append(float(corr_margin_tool.detach().float().mean().item()))
    if torch.is_tensor(corr_margin_ct):
        corr_margin_vals.append(float(corr_margin_ct.detach().float().mean().item()))
    if torch.is_tensor(centroid_shift_tool):
        centroid_shift_vals.append(float(centroid_shift_tool.detach().float().abs().mean().item()))
    if torch.is_tensor(centroid_shift_ct):
        centroid_shift_vals.append(float(centroid_shift_ct.detach().float().abs().mean().item()))
    if torch.is_tensor(context_entropy_tool):
        context_entropy_vals.append(float(context_entropy_tool.detach().float().mean().item()))
    if torch.is_tensor(context_entropy_ct):
        context_entropy_vals.append(float(context_entropy_ct.detach().float().mean().item()))
    if torch.is_tensor(context_margin_tool):
        context_margin_vals.append(float(context_margin_tool.detach().float().mean().item()))
    if torch.is_tensor(context_margin_ct):
        context_margin_vals.append(float(context_margin_ct.detach().float().mean().item()))
    if torch.is_tensor(context_offset_abs_tool):
        context_offset_abs_vals.append(float(context_offset_abs_tool.detach().float().mean().item()))
    if torch.is_tensor(context_offset_abs_ct):
        context_offset_abs_vals.append(float(context_offset_abs_ct.detach().float().mean().item()))
    if torch.is_tensor(context_gate_tool):
        context_gate_vals.append(float(context_gate_tool.detach().float().mean().item()))
    if torch.is_tensor(context_gate_ct):
        context_gate_vals.append(float(context_gate_ct.detach().float().mean().item()))
    if torch.is_tensor(belief_gate_tool):
        belief_gate_vals.append(float(belief_gate_tool.detach().float().mean().item()))
    if torch.is_tensor(belief_gate_ct):
        belief_gate_vals.append(float(belief_gate_ct.detach().float().mean().item()))
    if torch.is_tensor(anchor_delta_centroid_tool):
        anchor_delta_centroid_vals.append(float(anchor_delta_centroid_tool.detach().float().mean().item()))
    if torch.is_tensor(anchor_delta_centroid_ct):
        anchor_delta_centroid_vals.append(float(anchor_delta_centroid_ct.detach().float().mean().item()))
    if torch.is_tensor(anchor_alpha_tool):
        anchor_alpha_vals.append(float(anchor_alpha_tool.detach().float().mean().item()))
    if torch.is_tensor(anchor_alpha_ct):
        anchor_alpha_vals.append(float(anchor_alpha_ct.detach().float().mean().item()))
    if torch.is_tensor(factor_anchor_gate_tool):
        factor_anchor_gate_vals.append(float(factor_anchor_gate_tool.detach().float().mean().item()))
    if torch.is_tensor(factor_anchor_gate_ct):
        factor_anchor_gate_vals.append(float(factor_anchor_gate_ct.detach().float().mean().item()))
    if torch.is_tensor(factor_offset_gate_tool):
        factor_offset_gate_vals.append(float(factor_offset_gate_tool.detach().float().mean().item()))
    if torch.is_tensor(factor_offset_gate_ct):
        factor_offset_gate_vals.append(float(factor_offset_gate_ct.detach().float().mean().item()))
    if torch.is_tensor(factor_delta_offset_abs_tool):
        factor_delta_offset_abs_vals.append(float(factor_delta_offset_abs_tool.detach().float().mean().item()))
    if torch.is_tensor(factor_delta_offset_abs_ct):
        factor_delta_offset_abs_vals.append(float(factor_delta_offset_abs_ct.detach().float().mean().item()))
    if torch.is_tensor(anchor_verify_scale_tool):
        anchor_verify_scale_vals.append(float(anchor_verify_scale_tool.detach().float().mean().item()))
    if torch.is_tensor(anchor_verify_scale_ct):
        anchor_verify_scale_vals.append(float(anchor_verify_scale_ct.detach().float().mean().item()))
    if torch.is_tensor(anchor_verify_residual_tool):
        anchor_verify_residual_vals.append(float(anchor_verify_residual_tool.detach().float().mean().item()))
    if torch.is_tensor(anchor_verify_residual_ct):
        anchor_verify_residual_vals.append(float(anchor_verify_residual_ct.detach().float().mean().item()))
    if torch.is_tensor(stage0_centroid_noise_tool):
        stage0_centroid_noise_vals.append(float(stage0_centroid_noise_tool.detach().float().mean().item()))
    if torch.is_tensor(stage0_centroid_noise_ct):
        stage0_centroid_noise_vals.append(float(stage0_centroid_noise_ct.detach().float().mean().item()))
    if torch.is_tensor(joint_gate_tool):
        joint_gate_vals.append(float(joint_gate_tool.detach().float().mean().item()))
    if torch.is_tensor(joint_gate_ct):
        joint_gate_vals.append(float(joint_gate_ct.detach().float().mean().item()))
    if torch.is_tensor(template_entropy_tool):
        template_entropy_vals.append(float(template_entropy_tool.detach().float().mean().item()))
    if torch.is_tensor(template_entropy_ct):
        template_entropy_vals.append(float(template_entropy_ct.detach().float().mean().item()))
    if torch.is_tensor(template_margin_tool):
        template_margin_vals.append(float(template_margin_tool.detach().float().mean().item()))
    if torch.is_tensor(template_margin_ct):
        template_margin_vals.append(float(template_margin_ct.detach().float().mean().item()))
    if torch.is_tensor(template_gate_tool):
        template_gate_vals.append(float(template_gate_tool.detach().float().mean().item()))
    if torch.is_tensor(template_gate_ct):
        template_gate_vals.append(float(template_gate_ct.detach().float().mean().item()))
    if torch.is_tensor(template_offset_abs_tool):
        template_offset_abs_vals.append(float(template_offset_abs_tool.detach().float().mean().item()))
    if torch.is_tensor(template_offset_abs_ct):
        template_offset_abs_vals.append(float(template_offset_abs_ct.detach().float().mean().item()))
    if torch.is_tensor(template_delta_abs_tool):
        template_delta_abs_vals.append(float(template_delta_abs_tool.detach().float().mean().item()))
    if torch.is_tensor(template_delta_abs_ct):
        template_delta_abs_vals.append(float(template_delta_abs_ct.detach().float().mean().item()))

    # Fallback to merged keys if group-specific keys are absent.
    if not gate_vals:
        gate_all = outputs.get("traj_refine_delta_gate", None)
        if torch.is_tensor(gate_all):
            gate_vals.append(float(gate_all.detach().float().mean().item()))
    if not delta_vals:
        delta_all = outputs.get("traj_refine_delta_xy", None)
        if torch.is_tensor(delta_all):
            delta_vals.append(float(delta_all.detach().float().abs().mean().item()))

    return {
        "refine_gate_mean": float(sum(gate_vals) / max(len(gate_vals), 1)),
        "refine_delta_abs_mean": float(sum(delta_vals) / max(len(delta_vals), 1)),
        "corr_entropy_mean": float(sum(corr_entropy_vals) / max(len(corr_entropy_vals), 1)),
        "corr_top1_minus_top2_mean": float(sum(corr_margin_vals) / max(len(corr_margin_vals), 1)),
        "refine_centroid_shift_abs_mean": float(sum(centroid_shift_vals) / max(len(centroid_shift_vals), 1)),
        "context_entropy_mean": float(sum(context_entropy_vals) / max(len(context_entropy_vals), 1)),
        "context_top1_minus_top2_mean": float(sum(context_margin_vals) / max(len(context_margin_vals), 1)),
        "context_offset_abs_mean": float(sum(context_offset_abs_vals) / max(len(context_offset_abs_vals), 1)),
        "context_gate_mean": float(sum(context_gate_vals) / max(len(context_gate_vals), 1)),
        "belief_gate_mean": float(sum(belief_gate_vals) / max(len(belief_gate_vals), 1)),
        "anchor_delta_centroid_abs_mean": float(
            sum(anchor_delta_centroid_vals) / max(len(anchor_delta_centroid_vals), 1)
        ),
        "anchor_alpha_mean": float(sum(anchor_alpha_vals) / max(len(anchor_alpha_vals), 1)),
        "anchor_verify_scale_mean": float(sum(anchor_verify_scale_vals) / max(len(anchor_verify_scale_vals), 1)),
        "anchor_verify_residual_mean": float(
            sum(anchor_verify_residual_vals) / max(len(anchor_verify_residual_vals), 1)
        ),
        "stage0_centroid_noise_abs_mean": float(
            sum(stage0_centroid_noise_vals) / max(len(stage0_centroid_noise_vals), 1)
        ),
        "factor_anchor_gate_mean": float(sum(factor_anchor_gate_vals) / max(len(factor_anchor_gate_vals), 1)),
        "factor_offset_gate_mean": float(sum(factor_offset_gate_vals) / max(len(factor_offset_gate_vals), 1)),
        "factor_delta_offset_abs_mean": float(
            sum(factor_delta_offset_abs_vals) / max(len(factor_delta_offset_abs_vals), 1)
        ),
        "joint_gate_mean": float(sum(joint_gate_vals) / max(len(joint_gate_vals), 1)),
        "template_entropy_mean": float(sum(template_entropy_vals) / max(len(template_entropy_vals), 1)),
        "template_top1_minus_top2_mean": float(sum(template_margin_vals) / max(len(template_margin_vals), 1)),
        "template_gate_mean": float(sum(template_gate_vals) / max(len(template_gate_vals), 1)),
        "template_offset_abs_mean": float(sum(template_offset_abs_vals) / max(len(template_offset_abs_vals), 1)),
        "template_delta_abs_mean": float(sum(template_delta_abs_vals) / max(len(template_delta_abs_vals), 1)),
    }


_TRAJ_STAGE_IDS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
_TRAJ_LATE_FOCAL_BAND_IDS: Tuple[str, ...] = (
    "le4",
    "gt4_le8",
    "gt8_le12",
    "gt12_le24",
    "gt24_le32",
    "gt32",
)
_TRAJ_LATE_FOCAL_BASE_METRIC_KEYS: Tuple[str, ...] = (
    "traj_late_focal_ema_error",
    "traj_late_focal_batch_error_mean",
    "traj_late_focal_weight_mean",
    "traj_late_focal_weight_max",
    "traj_late_focal_weight_p50",
    "traj_late_focal_weight_p75",
    "traj_late_focal_weight_p90",
    "traj_late_focal_weight_p95",
)


def _traj_late_focal_metric_keys() -> Tuple[str, ...]:
    keys: List[str] = list(_TRAJ_LATE_FOCAL_BASE_METRIC_KEYS)
    for stage_idx in _TRAJ_STAGE_IDS:
        keys.append(f"traj_late_focal_weight_stage{stage_idx}_mean")
    for band_name in _TRAJ_LATE_FOCAL_BAND_IDS:
        keys.append(f"traj_late_focal_weight_band_{band_name}_mean")
    return tuple(keys)


_TRAJ_LATE_FOCAL_METRIC_KEYS: Tuple[str, ...] = _traj_late_focal_metric_keys()


def _accumulate_traj_stage_losses(meter: Dict[str, float], losses: Dict[str, torch.Tensor], bsz: int) -> None:
    for prefix in ("loss_tool_xy", "loss_ct_xy"):
        fallback = losses[prefix] * 0.0
        for stage_idx in _TRAJ_STAGE_IDS:
            meter[f"{prefix}_stage{stage_idx}"] += losses.get(
                f"{prefix}_stage{stage_idx}",
                fallback,
            ).item() * bsz


def _accumulate_traj_late_focal_metrics(meter: Dict[str, float], losses: Dict[str, torch.Tensor], bsz: int) -> None:
    fallback = losses["loss_total"] * 0.0
    for key in _TRAJ_LATE_FOCAL_METRIC_KEYS:
        meter[key] += losses.get(key, fallback).item() * bsz


def _format_traj_stage_summary(metrics: Dict[str, float]) -> str:
    return " ".join(
        f"s{stage_idx}={metrics.get(f'loss_tool_xy_stage{stage_idx}', 0.0):.5f}"
        for stage_idx in _TRAJ_STAGE_IDS
    )


def _compute_track_metrics_per_sample(
    pred_tool_xy: torch.Tensor,
    pred_ct_xy: torch.Tensor,
    pred_tool_conf: torch.Tensor,
    pred_ct_conf: torch.Tensor,
    gt_tool_xy: torch.Tensor,
    gt_ct_xy: torch.Tensor,
    gt_tool_vis: torch.Tensor,
    gt_ct_vis: torch.Tensor,
    image_size: int,
    trajectory_target: str = "both",
) -> Dict[str, torch.Tensor]:
    scale = float(max(int(image_size) - 1, 1))
    err_tool = torch.norm((pred_tool_xy.float() - gt_tool_xy.float()) * scale, dim=-1)  # [B,T,P_tool]
    err_ct = torch.norm((pred_ct_xy.float() - gt_ct_xy.float()) * scale, dim=-1)  # [B,T,P_ct]
    tgt = str(trajectory_target).lower()
    vis_tool = gt_tool_vis.float().clamp_min(0.0)
    vis_ct = gt_ct_vis.float().clamp_min(0.0)
    if tgt == "tool":
        err = err_tool
        vis = vis_tool
        pred_conf = pred_tool_conf.float()
    elif tgt == "ct":
        err = err_ct
        vis = vis_ct
        pred_conf = pred_ct_conf.float()
    else:
        err = torch.cat([err_tool, err_ct], dim=2)  # [B,T,7]
        vis = torch.cat([vis_tool, vis_ct], dim=2)  # [B,T,7]
        pred_conf = torch.cat([pred_tool_conf.float(), pred_ct_conf.float()], dim=2)

    bsz = int(err.shape[0])
    den = vis.sum(dim=(1, 2))  # [B]
    valid = (den > 0.0).float()
    ade_num_s = (err * vis).sum(dim=(1, 2))
    ade_px = torch.where(valid > 0.0, ade_num_s / den.clamp_min(1e-6), torch.zeros_like(den))

    fde_w = vis[:, -1, :]
    fde_den = fde_w.sum(dim=1)
    fde_valid = (fde_den > 0.0).float()
    fde_num_s = (err[:, -1, :] * fde_w).sum(dim=1)
    fde_px = torch.where(fde_valid > 0.0, fde_num_s / fde_den.clamp_min(1e-6), torch.zeros_like(fde_den))

    out: Dict[str, torch.Tensor] = {
        "traj_valid": valid,
        "traj_ade_px": ade_px,
        "traj_fde_valid": fde_valid,
        "traj_fde_px": fde_px,
        "traj_ade_num": ade_num_s.sum(),
        "traj_ade_den": den.sum(),
        "traj_fde_num": fde_num_s.sum(),
        "traj_fde_den": fde_den.sum(),
    }

    pck_vals = []
    den_total = den.sum()
    for thr in TRACK_PCK_THRESHOLDS_PX:
        key = int(thr)
        hit = (err <= float(thr)).to(vis.dtype)
        hit_num_s = (hit * vis).sum(dim=(1, 2))
        pck = torch.where(valid > 0.0, hit_num_s / den.clamp_min(1e-6), torch.zeros_like(den))
        out[f"traj_pck_{key}"] = pck
        out[f"traj_pck_{key}_num"] = hit_num_s.sum()
        pck_vals.append(pck)
    out["traj_pck_den"] = den_total
    out["traj_pck_auc"] = torch.stack(pck_vals, dim=1).mean(dim=1) if pck_vals else torch.zeros((bsz,), device=err.device)

    valid_mask = vis > 0.5
    has_track = valid_mask.any(dim=1)  # [B,7]
    surv_ok = torch.logical_or(err <= float(TRACK_SURVIVAL_THRESHOLD_PX), ~valid_mask).all(dim=1)  # [B,7]
    surv_num_s = (surv_ok & has_track).float().sum(dim=1)
    surv_den_s = has_track.float().sum(dim=1)
    surv_valid = (surv_den_s > 0.0).float()
    survival = torch.where(surv_valid > 0.0, surv_num_s / surv_den_s.clamp_min(1.0), torch.zeros_like(surv_den_s))
    out["traj_survival_valid"] = surv_valid
    out["traj_survival_50"] = survival
    out["traj_survival_num"] = surv_num_s.sum()
    out["traj_survival_den"] = surv_den_s.sum()

    # CoTracker-style AJ (Average Jaccard) over pixel thresholds.
    gt_visible = vis > 0.5
    pred_visible = pred_conf > float(TRACK_CONF_VIS_THRESHOLD)
    aj_vals = []
    aj_valid = []
    for thr in TRACK_AJ_THRESHOLDS_PX:
        key = int(thr)
        within = err <= float(thr)
        tp_s = (within & gt_visible & pred_visible).float().sum(dim=(1, 2))
        gt_pos_s = gt_visible.float().sum(dim=(1, 2))
        fp_mask = ((~gt_visible) & pred_visible) | ((~within) & pred_visible)
        fp_s = fp_mask.float().sum(dim=(1, 2))
        den_s = gt_pos_s + fp_s
        valid_s = (den_s > 0.0).float()
        jac_s = torch.where(valid_s > 0.0, tp_s / den_s.clamp_min(1.0), torch.zeros_like(den_s))
        out[f"traj_jaccard_{key}"] = jac_s
        out[f"traj_jaccard_{key}_valid"] = valid_s
        aj_vals.append(jac_s)
        aj_valid.append(valid_s)
    if aj_vals:
        aj_stack = torch.stack(aj_vals, dim=1)
        aj_valid_stack = torch.stack(aj_valid, dim=1)
        aj_den_s = aj_valid_stack.sum(dim=1)
        aj_s = torch.where(
            aj_den_s > 0.0,
            (aj_stack * aj_valid_stack).sum(dim=1) / aj_den_s.clamp_min(1.0),
            torch.zeros_like(aj_den_s),
        )
        aj_valid_s = (aj_den_s > 0.0).float()
    else:
        aj_s = torch.zeros((bsz,), device=err.device)
        aj_valid_s = torch.zeros((bsz,), device=err.device)
    out["traj_aj"] = aj_s
    out["traj_aj_valid"] = aj_valid_s
    out["traj_aj_sum"] = (aj_s * aj_valid_s).sum()
    out["traj_aj_den"] = aj_valid_s.sum()
    return out


def _zero_track_metrics_per_sample(bsz: int, device: torch.device) -> Dict[str, torch.Tensor]:
    z = torch.zeros((int(bsz),), device=device, dtype=torch.float32)
    s = torch.zeros((), device=device, dtype=torch.float32)
    out: Dict[str, torch.Tensor] = {
        "traj_valid": z,
        "traj_ade_px": z,
        "traj_fde_valid": z,
        "traj_fde_px": z,
        "traj_ade_num": s,
        "traj_ade_den": s,
        "traj_fde_num": s,
        "traj_fde_den": s,
        "traj_pck_den": s,
        "traj_pck_auc": z,
        "traj_survival_valid": z,
        "traj_survival_50": z,
        "traj_survival_num": s,
        "traj_survival_den": s,
        "traj_aj": z,
        "traj_aj_valid": z,
        "traj_aj_sum": s,
        "traj_aj_den": s,
    }
    for thr in TRACK_PCK_THRESHOLDS_PX:
        key = int(thr)
        out[f"traj_pck_{key}"] = z
        out[f"traj_pck_{key}_num"] = s
    for thr in TRACK_AJ_THRESHOLDS_PX:
        key = int(thr)
        out[f"traj_jaccard_{key}"] = z
        out[f"traj_jaccard_{key}_valid"] = z
    return out


def _compute_range_metrics_per_sample(
    pred_action_start_norm: torch.Tensor,
    pred_action_end_norm: torch.Tensor,
    gt_action_start_norm: torch.Tensor,
    gt_action_end_norm: torch.Tensor,
    action_range_mask: torch.Tensor,
    sampled_abs_frames: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    pred_s = pred_action_start_norm.reshape(-1).float().clamp(0.0, 1.0)
    pred_e = pred_action_end_norm.reshape(-1).float().clamp(0.0, 1.0)
    pred_e = torch.maximum(pred_e, pred_s)
    gt_s = gt_action_start_norm.reshape(-1).float().clamp(0.0, 1.0)
    gt_e = gt_action_end_norm.reshape(-1).float().clamp(0.0, 1.0)
    gt_e = torch.maximum(gt_e, gt_s)

    valid = (action_range_mask.reshape(-1).float() > 0.5).float()
    inter = (torch.minimum(pred_e, gt_e) - torch.maximum(pred_s, gt_s)).clamp_min(0.0)
    union = (torch.maximum(pred_e, gt_e) - torch.minimum(pred_s, gt_s)).clamp_min(1e-6)
    tiou = inter / union

    start_err = (pred_s - gt_s).abs()
    end_err = (pred_e - gt_e).abs()
    center_err = (0.5 * (pred_s + pred_e) - 0.5 * (gt_s + gt_e)).abs()
    dur_err = ((pred_e - pred_s) - (gt_e - gt_s)).abs()

    if (sampled_abs_frames is not None) and torch.is_tensor(sampled_abs_frames) and sampled_abs_frames.ndim >= 2:
        clip_span = (sampled_abs_frames[:, -1].float() - sampled_abs_frames[:, 0].float()).abs().clamp_min(1.0)
    else:
        clip_span = pred_s.new_full(pred_s.shape, 15.0)
    start_err_abs = start_err * clip_span
    end_err_abs = end_err * clip_span

    out: Dict[str, torch.Tensor] = {
        "range_valid": valid,
        "range_tiou": torch.where(valid > 0.0, tiou, torch.zeros_like(tiou)),
        "range_start_mae_norm": torch.where(valid > 0.0, start_err, torch.zeros_like(start_err)),
        "range_end_mae_norm": torch.where(valid > 0.0, end_err, torch.zeros_like(end_err)),
        "range_center_mae_norm": torch.where(valid > 0.0, center_err, torch.zeros_like(center_err)),
        "range_duration_mae_norm": torch.where(valid > 0.0, dur_err, torch.zeros_like(dur_err)),
        "range_start_mae_abs": torch.where(valid > 0.0, start_err_abs, torch.zeros_like(start_err_abs)),
        "range_end_mae_abs": torch.where(valid > 0.0, end_err_abs, torch.zeros_like(end_err_abs)),
        "range_tiou_sum": (tiou * valid).sum(),
        "range_valid_den": valid.sum(),
        "range_start_mae_norm_sum": (start_err * valid).sum(),
        "range_end_mae_norm_sum": (end_err * valid).sum(),
        "range_center_mae_norm_sum": (center_err * valid).sum(),
        "range_duration_mae_norm_sum": (dur_err * valid).sum(),
        "range_start_mae_abs_sum": (start_err_abs * valid).sum(),
        "range_end_mae_abs_sum": (end_err_abs * valid).sum(),
    }
    for thr in RANGE_TIOU_THRESHOLDS:
        key = str(thr).replace(".", "")
        out[f"range_r{key}"] = torch.where(valid > 0.0, (tiou >= float(thr)).float(), torch.zeros_like(tiou))
        out[f"range_r{key}_sum"] = ((tiou >= float(thr)).float() * valid).sum()
    return out


def _cache_np_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "fp16":
        return np.float16
    if dtype_name == "fp32":
        return np.float32
    raise ValueError(f"Unsupported cache dtype: {dtype_name}")


_MULTILAYER_CACHE_NOTICE_EMITTED = False


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _coerce_cached_token_grid_np(tok_np: np.ndarray, *, source: str, keep_multilayer: bool = False) -> np.ndarray:
    global _MULTILAYER_CACHE_NOTICE_EMITTED
    if tok_np.ndim == 4:
        return tok_np
    if tok_np.ndim == 5:
        if int(tok_np.shape[0]) <= 0:
            raise ValueError(f"Multi-layer token cache has invalid layer axis for {source}: {tuple(tok_np.shape)}")
        if bool(keep_multilayer):
            return tok_np
        if not _MULTILAYER_CACHE_NOTICE_EMITTED:
            print(
                "[cache] detected multi-layer VJEPA token cache; "
                "current training utilities will use the last cached layer for compatibility."
            )
            _MULTILAYER_CACHE_NOTICE_EMITTED = True
        return tok_np[-1]
    raise ValueError(
        f"Token cache must be 4D [T',H,W,D] or 5D [K,T',H,W,D], got {tuple(tok_np.shape)} for {source}"
    )


class VJEPATokenCache:
    """On-disk cache for frozen VJEPA token grids.

    Per-sample file format:
      root/{split}/{sid}.npy -> [T',H',W',D] or [K,T',H',W',D]
    """

    def __init__(self, root: Path, enabled: bool, dtype_name: str = "fp16") -> None:
        self.root = Path(root)
        self.enabled = bool(enabled)
        self.dtype_name = str(dtype_name)
        self.np_dtype = _cache_np_dtype(self.dtype_name)
        self.hits = 0
        self.misses = 0
        self.writes = 0

        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, split: str, sid: str) -> Path:
        return self.root / str(split) / f"{sid}.npy"

    def has_all(self, split: str, sids: List[str]) -> bool:
        if not self.enabled:
            return False
        return all(self._path(split=split, sid=str(s)).is_file() for s in sids)

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "root": str(self.root),
            "dtype": self.dtype_name,
            "hits": int(self.hits),
            "misses": int(self.misses),
            "writes": int(self.writes),
        }

    def format_stats(self) -> str:
        s = self.stats()
        return (
            f"enabled={s['enabled']} root={s['root']} dtype={s['dtype']} "
            f"hits={s['hits']} misses={s['misses']} writes={s['writes']}"
        )

    def get_or_compute(
        self,
        *,
        model: KeypointComboNet,
        video: Optional[torch.Tensor],
        sids: List[str],
        split: str,
        target_device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None

        if video is not None and len(sids) != int(video.shape[0]):
            raise ValueError(
                f"Cache sid count mismatch: len(sids)={len(sids)} vs batch={int(video.shape[0])}"
            )

        paths = [self._path(split=split, sid=s) for s in sids]
        all_exist = all(p.is_file() for p in paths)
        keep_multilayer = bool(getattr(model, "requires_multilayer_vjepa_tokens", False))
        if all_exist:
            try:
                arrs = [
                    _coerce_cached_token_grid_np(
                        np.load(p, allow_pickle=False),
                        source=str(p),
                        keep_multilayer=keep_multilayer,
                    )
                    for p in paths
                ]
                if not arrs:
                    raise ValueError("cached token grid list is empty")
                tok_np = np.stack(arrs, axis=0)
                tok = torch.from_numpy(tok_np).to(
                    device=target_device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                self.hits += len(paths)
                return tok
            except Exception:
                # If any cached sample is corrupted, recompute current batch and refresh files.
                pass

        self.misses += len(paths)
        if video is None:
            raise RuntimeError(
                "VJEPA token cache miss encountered but `video` is not available in batch. "
                "Either prebuild cache first or run with video-loading dataset until cache is warm."
            )
        with torch.no_grad():
            tokens_grid, _, _, _ = model.vjepa(video)
        tok_cpu = tokens_grid.detach().float().cpu().numpy()
        for i, p in enumerate(paths):
            _atomic_save_npy(p, tok_cpu[i].astype(self.np_dtype, copy=False))
            self.writes += 1
        return tokens_grid


def _model_forward_with_cache(
    model: KeypointComboNet,
    video: Optional[torch.Tensor],
    sids: List[str],
    split: str,
    token_cache: Optional[VJEPATokenCache],
    tool_id: Optional[torch.Tensor] = None,
    action_id: Optional[torch.Tensor] = None,
    combo_id: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    target_device = video.device if video is not None else next(model.parameters()).device
    if token_cache is None:
        if video is None:
            raise ValueError("`video` is required when token_cache is None.")
        return model(video, tool_id=tool_id, action_id=action_id, combo_id=combo_id)
    tokens_grid = token_cache.get_or_compute(
        model=model,
        video=video,
        sids=sids,
        split=split,
        target_device=target_device,
    )
    if tokens_grid is None:
        if video is None:
            raise ValueError("`video` is required when cache is disabled.")
        return model(video, tool_id=tool_id, action_id=action_id, combo_id=combo_id)
    return model(
        video=None,
        tokens_grid=tokens_grid,
        token_order="cached_t_major_then_h_w",
        tool_id=tool_id,
        action_id=action_id,
        combo_id=combo_id,
    )


@torch.no_grad()
def _warmup_token_cache_for_split(
    *,
    model: KeypointComboNet,
    ds_video: SurgKptComboDataset,
    split: str,
    token_cache: VJEPATokenCache,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_progress: bool,
) -> Dict[str, int]:
    all_sids = [str(x) for x in ds_video.sids]
    if token_cache.has_all(split, all_sids):
        return {"seen": 0, "hits": 0, "misses": 0, "writes": 0}

    cache_before = token_cache.stats()
    loader = DataLoader(
        ds_video,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )
    iterator = loader
    pbar = None
    if use_progress and (tqdm is not None):
        pbar = tqdm(loader, total=len(loader), desc=f"CacheWarm {split}", dynamic_ncols=True, leave=False)
        iterator = pbar

    seen = 0
    for batch in iterator:
        batch = _to_device(batch, device)
        video = batch.get("video", None)
        if video is None:
            raise RuntimeError(f"Cache warmup for split={split} requires video tensors in dataset.")
        sid_raw = batch.get("sid", [])
        if isinstance(sid_raw, (list, tuple)):
            sid_list = [str(x) for x in sid_raw]
        elif isinstance(sid_raw, str):
            sid_list = [sid_raw]
        else:
            sid_list = [str(sid_raw)]
        _ = token_cache.get_or_compute(
            model=model,
            video=video,
            sids=sid_list,
            split=split,
            target_device=device,
        )
        seen += len(sid_list)

    if pbar is not None:
        pbar.close()

    cache_after = token_cache.stats()
    return {
        "seen": int(seen),
        "hits": int(cache_after["hits"] - cache_before["hits"]),
        "misses": int(cache_after["misses"] - cache_before["misses"]),
        "writes": int(cache_after["writes"] - cache_before["writes"]),
    }


def _tensor_stats(x: torch.Tensor) -> Dict[str, Any]:
    xf = x.detach().float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(xf.min().item()),
        "max": float(xf.max().item()),
        "mean": float(xf.mean().item()),
        "std": float(xf.std(unbiased=False).item()),
        "finite": bool(torch.isfinite(xf).all().item()),
    }


def _assert_finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"Non-finite values found in `{name}`")


def _should_debug_step(step: int, args: argparse.Namespace) -> bool:
    if step == 1:
        return True
    return args.debug_check_interval > 0 and (step % args.debug_check_interval == 0)


def _validate_train_batch_contract(
    video: torch.Tensor,
    gt_tool_xy: torch.Tensor,
    gt_tool_vis: torch.Tensor,
    gt_ct_xy: torch.Tensor,
    gt_ct_vis: torch.Tensor,
    gt_action_start_bin: torch.Tensor,
    gt_action_end_bin: torch.Tensor,
    action_range_mask: torch.Tensor,
    combo_id: torch.Tensor,
) -> None:
    if video.ndim != 5:
        raise ValueError(f"`video` must be 5D [B,T,3,H,W], got {tuple(video.shape)}")
    if video.shape[1] != 16 or video.shape[2] != 3:
        raise ValueError(f"`video` must be [B,16,3,H,W], got {tuple(video.shape)}")

    bsz, t, _, _, _ = video.shape
    if gt_tool_xy.ndim != 4 or gt_tool_xy.shape[0] != bsz or gt_tool_xy.shape[1] != t or gt_tool_xy.shape[-1] != 2:
        raise ValueError(f"`gt_tool_xy` must be [B,T,P,2], got {tuple(gt_tool_xy.shape)}")
    if gt_tool_vis.ndim != 3 or tuple(gt_tool_vis.shape) != tuple(gt_tool_xy.shape[:3]):
        raise ValueError(f"`gt_tool_vis` must be [B,T,P] matching gt_tool_xy[:3], got {tuple(gt_tool_vis.shape)}")
    if int(gt_tool_xy.shape[2]) <= 0:
        raise ValueError(f"`gt_tool_xy` must have P>0, got {tuple(gt_tool_xy.shape)}")
    if tuple(gt_ct_xy.shape) != (bsz, t, 4, 2):
        raise ValueError(f"`gt_ct_xy` must be [B,T,4,2], got {tuple(gt_ct_xy.shape)}")
    if tuple(gt_ct_vis.shape) != (bsz, t, 4):
        raise ValueError(f"`gt_ct_vis` must be [B,T,4], got {tuple(gt_ct_vis.shape)}")
    if gt_action_start_bin.ndim != 1 or gt_action_start_bin.shape[0] != bsz:
        raise ValueError(f"`gt_action_start_bin` must be [B], got {tuple(gt_action_start_bin.shape)}")
    if gt_action_end_bin.ndim != 1 or gt_action_end_bin.shape[0] != bsz:
        raise ValueError(f"`gt_action_end_bin` must be [B], got {tuple(gt_action_end_bin.shape)}")
    if action_range_mask.ndim != 1 or action_range_mask.shape[0] != bsz:
        raise ValueError(f"`action_range_mask` must be [B], got {tuple(action_range_mask.shape)}")
    if combo_id.ndim != 1 or combo_id.shape[0] != bsz:
        raise ValueError(f"`combo_id` must be [B], got {tuple(combo_id.shape)}")

    _assert_finite("video", video)
    _assert_finite("gt_tool_xy", gt_tool_xy)
    _assert_finite("gt_tool_vis", gt_tool_vis)
    _assert_finite("gt_ct_xy", gt_ct_xy)
    _assert_finite("gt_ct_vis", gt_ct_vis)
    _assert_finite("action_range_mask", action_range_mask)

    tool_xy_min = float(gt_tool_xy.min().item())
    tool_xy_max = float(gt_tool_xy.max().item())
    ct_xy_min = float(gt_ct_xy.min().item())
    ct_xy_max = float(gt_ct_xy.max().item())
    if tool_xy_min < -0.2 or tool_xy_max > 1.2:
        raise ValueError(f"`gt_tool_xy` out of normalized range: min={tool_xy_min:.4f}, max={tool_xy_max:.4f}")
    if ct_xy_min < -0.2 or ct_xy_max > 1.2:
        raise ValueError(f"`gt_ct_xy` out of normalized range: min={ct_xy_min:.4f}, max={ct_xy_max:.4f}")
    if bool((gt_action_start_bin < 0).any().item()) or bool((gt_action_start_bin >= RANGE_NUM_BINS).any().item()):
        raise ValueError("`gt_action_start_bin` out of valid bin range")
    if bool((gt_action_end_bin < 0).any().item()) or bool((gt_action_end_bin >= RANGE_NUM_BINS).any().item()):
        raise ValueError("`gt_action_end_bin` out of valid bin range")
    if bool((gt_action_end_bin < gt_action_start_bin).any().item()):
        raise ValueError("`gt_action_end_bin` must be >= `gt_action_start_bin`")


def _validate_train_output_contract(outputs: Dict[str, torch.Tensor], bsz: int) -> None:
    required = [
        "pred_tool_xy",
        "pred_ct_xy",
        "combo_logits",
        "combo_tool_logits",
        "combo_action_logits",
        "logits_start_bin",
        "logits_end_bin",
        "conf_tool",
        "conf_ct",
        "probs_tool",
        "probs_ct",
        "heatmap_tool",
        "heatmap_ct",
    ]
    for k in required:
        if k not in outputs:
            raise KeyError(f"Missing model output key: `{k}`")

    pred_tool_xy = outputs["pred_tool_xy"]
    pred_ct_xy = outputs["pred_ct_xy"]
    combo_logits = outputs["combo_logits"]
    combo_tool_logits = outputs["combo_tool_logits"]
    combo_action_logits = outputs["combo_action_logits"]
    logits_start_bin = outputs["logits_start_bin"]
    logits_end_bin = outputs["logits_end_bin"]
    conf_tool = outputs["conf_tool"]
    conf_ct = outputs["conf_ct"]
    probs_tool = outputs["probs_tool"]
    probs_ct = outputs["probs_ct"]
    heatmap_tool = outputs["heatmap_tool"]
    heatmap_ct = outputs["heatmap_ct"]

    if pred_tool_xy.ndim != 4 or pred_tool_xy.shape[0] != bsz or pred_tool_xy.shape[-1] != 2:
        raise ValueError(f"`pred_tool_xy` bad shape: {tuple(pred_tool_xy.shape)}")
    p_tool = int(pred_tool_xy.shape[2])
    if p_tool <= 0:
        raise ValueError(f"`pred_tool_xy` must have P>0, got {tuple(pred_tool_xy.shape)}")
    if pred_ct_xy.ndim != 4 or pred_ct_xy.shape[0] != bsz or pred_ct_xy.shape[2:] != (4, 2):
        raise ValueError(f"`pred_ct_xy` bad shape: {tuple(pred_ct_xy.shape)}")
    if combo_logits.ndim != 2 or combo_logits.shape[0] != bsz:
        raise ValueError(f"`combo_logits` bad shape: {tuple(combo_logits.shape)}")
    if combo_tool_logits.ndim != 2 or combo_tool_logits.shape[0] != bsz:
        raise ValueError(f"`combo_tool_logits` bad shape: {tuple(combo_tool_logits.shape)}")
    if combo_action_logits.ndim != 2 or combo_action_logits.shape[0] != bsz:
        raise ValueError(f"`combo_action_logits` bad shape: {tuple(combo_action_logits.shape)}")
    if tuple(logits_start_bin.shape) != (bsz, RANGE_NUM_BINS):
        raise ValueError(f"`logits_start_bin` bad shape: {tuple(logits_start_bin.shape)}")
    if tuple(logits_end_bin.shape) != (bsz, RANGE_NUM_BINS):
        raise ValueError(f"`logits_end_bin` bad shape: {tuple(logits_end_bin.shape)}")

    t_prime = int(pred_tool_xy.shape[1])
    if tuple(conf_tool.shape) != (bsz, t_prime, p_tool):
        raise ValueError(f"`conf_tool` bad shape: {tuple(conf_tool.shape)}")
    if tuple(conf_ct.shape) != (bsz, t_prime, 4):
        raise ValueError(f"`conf_ct` bad shape: {tuple(conf_ct.shape)}")
    if probs_tool.ndim != 4 or probs_tool.shape[:3] != (bsz, t_prime, p_tool):
        raise ValueError(f"`probs_tool` bad shape: {tuple(probs_tool.shape)}")
    if probs_ct.ndim != 4 or probs_ct.shape[:3] != (bsz, t_prime, 4):
        raise ValueError(f"`probs_ct` bad shape: {tuple(probs_ct.shape)}")

    if heatmap_tool.ndim != 5 or heatmap_tool.shape[:3] != (bsz, t_prime, p_tool):
        raise ValueError(f"`heatmap_tool` bad shape: {tuple(heatmap_tool.shape)}")
    if heatmap_ct.ndim != 5 or heatmap_ct.shape[:3] != (bsz, t_prime, 4):
        raise ValueError(f"`heatmap_ct` bad shape: {tuple(heatmap_ct.shape)}")

    h_prime = int(heatmap_tool.shape[-2])
    w_prime = int(heatmap_tool.shape[-1])
    s = int(h_prime * w_prime)
    if probs_tool.shape[-1] != s or probs_ct.shape[-1] != s:
        raise ValueError(
            f"Inconsistent S between probs and heatmap: "
            f"S_tool={probs_tool.shape[-1]} S_ct={probs_ct.shape[-1]} H'={h_prime} W'={w_prime}"
        )

    meta = outputs.get("meta", {})
    if isinstance(meta, dict):
        t_meta = int(meta["t_out"]) if "t_out" in meta else int(meta.get("t_prime", t_prime))
        if t_meta != t_prime:
            raise ValueError(f"meta.t_out/t_prime mismatch: {t_meta} vs {t_prime}")
        if "h_prime" in meta and int(meta["h_prime"]) != h_prime:
            raise ValueError(f"meta.h_prime mismatch: {meta['h_prime']} vs {h_prime}")
        if "w_prime" in meta and int(meta["w_prime"]) != w_prime:
            raise ValueError(f"meta.w_prime mismatch: {meta['w_prime']} vs {w_prime}")
        t_for_tokens = int(meta["t_encoder"]) if "t_encoder" in meta else t_meta
        if "n_tokens" in meta and int(meta["n_tokens"]) != (t_for_tokens * h_prime * w_prime):
            raise ValueError("meta.n_tokens mismatch with T_encoder*H'*W'")

    _assert_finite("pred_tool_xy", pred_tool_xy)
    _assert_finite("pred_ct_xy", pred_ct_xy)
    _assert_finite("combo_logits", combo_logits)
    _assert_finite("combo_tool_logits", combo_tool_logits)
    _assert_finite("combo_action_logits", combo_action_logits)
    _assert_finite("logits_start_bin", logits_start_bin)
    _assert_finite("logits_end_bin", logits_end_bin)
    _assert_finite("conf_tool", conf_tool)
    _assert_finite("conf_ct", conf_ct)
    _assert_finite("probs_tool", probs_tool)
    _assert_finite("probs_ct", probs_ct)


def _write_train_step_debug_report(report: Dict[str, Any], out_dir: Path, epoch: int, step: int) -> Path:
    dbg_dir = out_dir / "debug" / "train_steps"
    dbg_dir.mkdir(parents=True, exist_ok=True)
    p = dbg_dir / f"epoch_{epoch:03d}_step_{step:05d}.json"
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p


def _denorm_video_to_uint8(video_tchw_norm: torch.Tensor) -> np.ndarray:
    # video_tchw_norm: [T,3,H,W] after ImageNet normalize
    mean = IMAGENET_MEAN.to(video_tchw_norm.device)
    std = IMAGENET_STD.to(video_tchw_norm.device)
    video = (video_tchw_norm * std + mean).clamp(0.0, 1.0)
    video = (video * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return video


def _load_raw_video_u8(root: Path, sid: str) -> np.ndarray:
    p = root / "video_16_npy" / f"{sid}.npy"
    if not p.is_file():
        raise FileNotFoundError(f"Missing raw video file: {p}")
    v = np.load(p)
    if v.ndim != 4 or v.shape[-1] != 3:
        raise ValueError(f"raw video must be [T,H,W,3], got {v.shape} for sid={sid}")
    if v.dtype != np.uint8:
        v = np.clip(v, 0, 255).astype(np.uint8)
    # Dataset export uses OpenCV frames (BGR). Convert for PIL/overlay display.
    v = np.ascontiguousarray(v[..., ::-1])
    return v


def _norm_xy_to_px(xy: np.ndarray, w: int, h: int) -> Tuple[float, float]:
    x = float(np.clip(xy[0], 0.0, 1.0) * (w - 1))
    y = float(np.clip(xy[1], 0.0, 1.0) * (h - 1))
    return x, y


def _overlay_heatmap(base_rgb_u8: np.ndarray, heatmap_hw: torch.Tensor, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    hm = heatmap_hw.detach().cpu().float()
    hm = hm - hm.min()
    hm = hm / hm.max().clamp_min(1e-6)
    hm_np = hm.numpy()

    base = base_rgb_u8.astype(np.float32)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    blend = np.clip(alpha, 0.0, 1.0) * hm_np[..., None]
    out = base * (1.0 - blend) + color_arr * blend
    return out.clip(0.0, 255.0).astype(np.uint8)


def _save_vis_image(path: Path, rgb_u8: np.ndarray, scale: int = 2) -> None:
    img = Image.fromarray(rgb_u8)
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), resample=Image.BICUBIC)
    img.save(path)


def _draw_gt_overlay(
    base_rgb_u8: np.ndarray,
    gt_tool_xy: np.ndarray,
    gt_tool_vis: np.ndarray,
    gt_ct_xy: np.ndarray,
    gt_ct_vis: np.ndarray,
) -> np.ndarray:
    img = Image.fromarray(base_rgb_u8.copy())
    draw = ImageDraw.Draw(img)
    w, h = img.size
    r = max(9, int(round(min(w, h) * 0.028)))
    gt_w = 3

    # tool GT: green ring
    for i in range(gt_tool_xy.shape[0]):
        if float(gt_tool_vis[i]) > 0.0:
            gx, gy = _norm_xy_to_px(gt_tool_xy[i], w, h)
            draw.ellipse((gx - r, gy - r, gx + r, gy + r), outline=(80, 255, 80), width=gt_w)

    # contact GT: cyan ring
    for i in range(gt_ct_xy.shape[0]):
        if float(gt_ct_vis[i]) > 0.0:
            gx, gy = _norm_xy_to_px(gt_ct_xy[i], w, h)
            draw.ellipse((gx - r, gy - r, gx + r, gy + r), outline=(80, 230, 255), width=gt_w)

    x2 = min(w - 6, 1200)
    draw.rectangle((6, 6, x2, 52), fill=(0, 0, 0))
    draw.text(
        (10, 11),
        "GT only | Tool=green ring | Contact=cyan ring",
        fill=(255, 255, 255),
    )
    return np.asarray(img)


def _draw_pred_overlay(
    base_rgb_u8: np.ndarray,
    pred_tool_xy: np.ndarray,
    pred_ct_xy: np.ndarray,
) -> np.ndarray:
    img = Image.fromarray(base_rgb_u8.copy())
    draw = ImageDraw.Draw(img)
    w, h = img.size
    r = max(9, int(round(min(w, h) * 0.028)))
    pred_w = 3
    cross_len = max(8, int(round(r * 0.9)))

    def _draw_pred_cross(px: float, py: float, color: Tuple[int, int, int]) -> None:
        draw.line((px - cross_len, py, px + cross_len, py), fill=(0, 0, 0), width=pred_w + 2)
        draw.line((px, py - cross_len, px, py + cross_len), fill=(0, 0, 0), width=pred_w + 2)
        draw.line((px - cross_len, py, px + cross_len, py), fill=color, width=pred_w)
        draw.line((px, py - cross_len, px, py + cross_len), fill=color, width=pred_w)

    # tool pred: green cross
    for i in range(pred_tool_xy.shape[0]):
        px, py = _norm_xy_to_px(pred_tool_xy[i], w, h)
        _draw_pred_cross(px, py, color=(70, 255, 70))

    # contact pred: blue cross
    for i in range(pred_ct_xy.shape[0]):
        px, py = _norm_xy_to_px(pred_ct_xy[i], w, h)
        _draw_pred_cross(px, py, color=(80, 170, 255))

    x2 = min(w - 6, 1200)
    draw.rectangle((6, 6, x2, 52), fill=(0, 0, 0))
    draw.text(
        (10, 11),
        "Pred only | Tool=green cross | Contact=blue cross",
        fill=(255, 255, 255),
    )
    return np.asarray(img)


def _select_vis_indices(
    ds: SurgKptComboDataset,
    max_samples: int,
    selection_mode: str = "top_vis",
    rng_seed: int = 42,
) -> List[Tuple[int, int, str, float]]:
    """
    Returns list of tuples: (combo_id, dataset_index, sid, vis_score)
    selection_mode:
      - "top_vis": pick highest vis_score per combo
      - "random": pick random sample per combo
    """
    if max_samples <= 0:
        return []

    sid_vis_score: Dict[str, float] = {}
    meta_path = ds.root / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, list):
                for row in meta:
                    if not isinstance(row, dict):
                        continue
                    sid_val = row.get("sid")
                    vis_val = row.get("vis_score")
                    if sid_val is None or vis_val is None:
                        continue
                    sid_vis_score[str(sid_val)] = float(vis_val)
        except Exception:
            sid_vis_score = {}

    def _fallback_vis_score(sid: str) -> float:
        kp_path = ds.root / "keypoint_vis" / f"{sid}.npy"
        ct_path = ds.root / "contact_vis_raw" / f"{sid}.npy"
        try:
            kp_vis = np.load(kp_path).astype(np.float32)
            ct_vis = np.load(ct_path).astype(np.float32)
            return float(0.5 * (float(kp_vis.mean()) + float(ct_vis.mean())))
        except Exception:
            return -1.0

    combo_to_items: Dict[int, List[Tuple[int, str, float]]] = defaultdict(list)
    for idx, sid in enumerate(ds.sids):
        combo_path = ds.root / "combo_ids" / f"{sid}.npy"
        cid = int(np.load(combo_path).item())
        sid_str = str(sid)
        score = sid_vis_score.get(sid_str)
        if score is None:
            score = _fallback_vis_score(sid_str)
        combo_to_items[cid].append((idx, sid_str, float(score)))

    selected: List[Tuple[int, int, str, float]] = []
    rng = random.Random(int(rng_seed))
    for cid in sorted(combo_to_items.keys()):
        items = combo_to_items[cid]
        if not items:
            continue
        if selection_mode == "random":
            idx, sid_str, score = rng.choice(items)
        else:
            idx, sid_str, score = max(items, key=lambda t: (t[2], -t[0]))
        selected.append((cid, idx, sid_str, score))

    if max_samples > 0:
        selected = selected[:max_samples]
    return selected


@torch.no_grad()
def run_checkpoint_visualization(
    model: KeypointComboNet,
    val_set: SurgKptComboDataset,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    output_dir: Path,
    combo_id_to_name: Dict[int, str],
    token_cache: Optional[VJEPATokenCache] = None,
    range_combo_reweight_weights: Optional[torch.Tensor] = None,
    split_name: str = "val",
    selection_mode: str = "top_vis",
) -> Dict[str, str]:
    if not bool(getattr(args, "run_checkpoint_vis", True)):
        return {"enabled": "false", "saved_samples": "0", "path": ""}
    if args.vis_samples_per_ckpt <= 0:
        return {"enabled": "false", "saved_samples": "0", "path": ""}

    vis_root = output_dir / "visuals" / f"epoch_{epoch:03d}" / split_name
    vis_root.mkdir(parents=True, exist_ok=True)
    video_root = val_set.root / "video_16_npy"
    if not video_root.is_dir():
        reason = f"missing video_16_npy root: {video_root}"
        print(f"[vis {split_name}] skipped: {reason}")
        return {
            "enabled": "true",
            "saved_samples": "0",
            "path": str(vis_root),
            "skipped_reason": reason,
        }

    selected = _select_vis_indices(
        ds=val_set,
        max_samples=args.vis_samples_per_ckpt,
        selection_mode=selection_mode,
        rng_seed=int(args.seed) + int(epoch),
    )
    if not selected:
        return {"enabled": "true", "saved_samples": "0", "path": str(vis_root)}

    selected_meta: List[Dict[str, Any]] = []
    for cid, idx, sid, score in selected:
        selected_meta.append(
            {
                "combo_id": int(cid),
                "combo_name": combo_id_to_name.get(int(cid), f"combo_{int(cid)}"),
                "sid": str(sid),
                "dataset_index": int(idx),
                "vis_score": float(score),
            }
        )
    (vis_root / "selected_samples.json").write_text(json.dumps(selected_meta, indent=2), encoding="utf-8")
    print(f"[vis {split_name}] selection_mode={selection_mode} samples={len(selected_meta)}")
    for row in selected_meta:
        print(
            f"[vis {split_name}] combo={row['combo_name']} sid={row['sid']} "
            f"vis_score={row['vis_score']:.4f}"
        )

    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = _get_amp_dtype(args.amp_dtype)

    saved = 0
    for _, idx, sid, _ in selected:
        try:
            item = val_set[idx]
        except FileNotFoundError as exc:
            print(f"[vis {split_name}] skip sid={sid}: {exc}")
            continue
        sample_dir = vis_root / sid
        sample_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = sample_dir / "raw"
        hm_tool_dir = sample_dir / "heatmap_tool"
        hm_contact_dir = sample_dir / "heatmap_contact"
        gt_dir = sample_dir / "gt"
        pred_dir = sample_dir / "pred"
        raw_dir.mkdir(parents=True, exist_ok=True)
        hm_tool_dir.mkdir(parents=True, exist_ok=True)
        hm_contact_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        video = item["video"].unsqueeze(0).to(device)
        gt_tool_xy = item["gt_tool_xy"].unsqueeze(0).to(device)
        gt_tool_vis = item["gt_tool_vis"].unsqueeze(0).to(device)
        gt_tool_xy, gt_tool_vis = _maybe_use_tool_centroid_gt(
            gt_tool_xy=gt_tool_xy,
            gt_tool_vis=gt_tool_vis,
            use_tool_centroid_only=bool(getattr(args, "tool_centroid_only", False)),
        )
        gt_ct_xy = item["gt_ct_xy"].unsqueeze(0).to(device)
        gt_ct_vis = item["gt_ct_vis"].unsqueeze(0).to(device)
        gt_action_start_bin = item["gt_action_start_bin"].reshape(1).long().to(device)
        gt_action_end_bin = item["gt_action_end_bin"].reshape(1).long().to(device)
        if ("gt_action_start_norm" in item) and ("gt_action_end_norm" in item):
            gt_action_start_norm = item["gt_action_start_norm"].reshape(1).float().to(device)
            gt_action_end_norm = item["gt_action_end_norm"].reshape(1).float().to(device)
        else:
            gt_action_start_norm, gt_action_end_norm = _derive_action_norm_from_bins(
                gt_action_start_bin=gt_action_start_bin,
                gt_action_end_bin=gt_action_end_bin,
                num_bins=RANGE_NUM_BINS,
            )
        action_range_mask = item["action_range_mask"].reshape(1).float().to(device)
        combo_id = item["combo_id"].reshape(1).long().to(device)
        tool_pt_role = item.get("pt_role", None)
        if torch.is_tensor(tool_pt_role):
            tool_pt_role = tool_pt_role.reshape(1, -1).long().to(device)
        tool_id, action_id = _derive_tool_action_id_from_combo_id(combo_id)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = _model_forward_with_cache(
                model=model,
                video=video,
                sids=[str(sid)],
                split="val",
                token_cache=token_cache,
                tool_id=tool_id,
                action_id=action_id,
                combo_id=combo_id,
            )
            t_prime = outputs["pred_tool_xy"].shape[1]
            if (
                bool(getattr(args, "tool_only", False))
                or bool(getattr(args, "combo_probe_attn", False))
                or bool(getattr(args, "combo_probe_dual_attn", False))
            ):
                aligned = {
                    "gt_tool_xy": gt_tool_xy,
                    "gt_tool_vis": gt_tool_vis,
                    "gt_ct_xy": gt_ct_xy,
                    "gt_ct_vis": gt_ct_vis,
                }
            else:
                aligned = align_targets_to_tprime(gt_tool_xy, gt_tool_vis, gt_ct_xy, gt_ct_vis, t_prime=t_prime)
            ps_losses = compute_batch_losses_per_sample(
                outputs=outputs,
                gt_tool_xy=aligned["gt_tool_xy"],
                gt_tool_vis=aligned["gt_tool_vis"],
                gt_ct_xy=aligned["gt_ct_xy"],
                gt_ct_vis=aligned["gt_ct_vis"],
                combo_id=combo_id,
                gt_action_start_bin=gt_action_start_bin,
                gt_action_end_bin=gt_action_end_bin,
                action_range_mask=action_range_mask,
                gt_action_start_norm=gt_action_start_norm,
                gt_action_end_norm=gt_action_end_norm,
                combo_ce_weight=float(getattr(args, "combo_ce_weight", COMBO_CE_WEIGHT)),
                combo_tool_ce_weight=float(getattr(args, "combo_tool_ce_weight", 0.5)),
                combo_action_ce_weight=float(getattr(args, "combo_action_ce_weight", 0.5)),
                start_bin_loss_weight=RANGE_START_CE_WEIGHT,
                end_bin_loss_weight=RANGE_END_CE_WEIGHT,
                range_label_smoothing=RANGE_LABEL_SMOOTHING,
                range_positive_mode=str(getattr(args, "range_positive_mode", "span")),
                range_sensitivity_mode=str(getattr(args, "range_sensitivity_mode", "none")),
                range_sensitivity_cls_sigma=float(getattr(args, "range_sensitivity_cls_sigma", 0.5)),
                range_sensitivity_loc_sigma=float(getattr(args, "range_sensitivity_loc_sigma", 0.35)),
                range_sensitivity_min_weight=float(getattr(args, "range_sensitivity_min_weight", 0.25)),
                range_scale_invariant_mode=str(getattr(args, "range_scale_invariant_mode", "none")),
                range_scale_invariant_alpha=float(getattr(args, "range_scale_invariant_alpha", 0.8)),
                range_cls_target_mode=str(getattr(args, "range_cls_target_mode", "binary")),
                range_cls_varifocal_alpha=float(getattr(args, "range_cls_varifocal_alpha", 0.75)),
                range_cls_varifocal_weight=float(getattr(args, "range_cls_varifocal_weight", 1.0)),
                range_quality_mode=str(getattr(args, "range_quality_mode", "none")),
                range_quality_loss_weight=float(getattr(args, "range_quality_loss_weight", 0.0)),
                range_sola_aux_weight=float(getattr(args, "range_sola_aux_weight", 0.0)),
                range_sola_similarity_k=float(getattr(args, "range_sola_sim_k", 16.0)),
                range_focal_gamma=RANGE_FOCAL_GAMMA,
                range_neighbor_smoothing=RANGE_NEIGHBOR_SMOOTHING,
                range_boundary_aux_weight=float(getattr(args, "range_boundary_aux_weight", 0.0)),
                range_proposal_weight=float(getattr(args, "range_proposal_weight", 0.0)),
                range_combo_reweight_weights=range_combo_reweight_weights,
                range_combo_reweight_targets=str(getattr(args, "range_combo_reweight_targets", "")),
                kpt_loss_type=str(getattr(args, "kpt_loss_type", "l1l2")),
                traj_loss_pixel_scale=float(getattr(args, "traj_loss_pixel_scale", 256.0)),
                traj_huber_delta=float(getattr(args, "traj_huber_delta", 6.0)),
                traj_t0_loss_weight=float(getattr(args, "traj_t0_loss_weight", 1.0)),
                traj_loss_visible_only=bool(getattr(args, "traj_loss_visible_only", False)),
                traj_centroid_aux_weight=float(getattr(args, "traj_centroid_aux_weight", 0.0)),
                traj_offset_aux_weight=float(getattr(args, "traj_offset_aux_weight", 0.0)),
                trajectory_target=(
                    str(getattr(args, "trajectory_target", "tool"))
                    if bool(getattr(args, "trajectory_only", False))
                    else "both"
                ),
                trajectory_only=bool(getattr(args, "trajectory_only", False)),
                combo_only=bool(getattr(args, "combo_only", False)),
                tool_only=bool(getattr(args, "tool_only", False)),
                traj_heatmap_ce_weight=float(getattr(args, "traj_heatmap_ce_weight", 0.0)),
                traj_heatmap_sigma=float(getattr(args, "traj_heatmap_sigma", 2.45)),
                tool_pt_role=tool_pt_role,
                symmetry_swap_invariant_loss=bool(getattr(args, "symmetry_swap_invariant_loss", False)),
                symmetry_pair_separation_loss=bool(getattr(args, "symmetry_pair_separation_loss", False)),
                symmetry_pair_separation_weight=float(getattr(args, "symmetry_pair_separation_weight", 0.05)),
                symmetry_pair_separation_min_pred_gap_px=float(
                    getattr(args, "symmetry_pair_separation_min_pred_gap_px", 8.0)
                ),
                symmetry_pair_separation_gt_min_gap_px=float(
                    getattr(args, "symmetry_pair_separation_gt_min_gap_px", 16.0)
                ),
                traj_m2_stage_weights=_resolve_eval_traj_stage_weights(args),
            )

        # Visualize on original-resolution frames (video_16_npy) for faithful projection.
        try:
            video_u8 = _load_raw_video_u8(val_set.root, sid)
            video_source = "raw_video_16_npy"
        except Exception:
            # Fallback for robustness: use de-normalized training tensor.
            video_u8 = _denorm_video_to_uint8(item["video"])
            video_source = "denorm_train_tensor"
        total_t = int(video_u8.shape[0])
        ratio = float(total_t) / float(t_prime)
        frame_cap = int(t_prime if args.vis_max_frames <= 0 else min(t_prime, args.vis_max_frames))

        if bool(getattr(args, "combo_probe_attn", False)):
            combo_logits_struct = outputs["combo_logits"]
        else:
            combo_logits_struct = _structured_combo_logits_from_tool_action(
                combo_logits=outputs["combo_logits"],
                combo_tool_logits=outputs.get("combo_tool_logits", outputs["combo_logits"]),
                combo_action_logits=outputs.get("combo_action_logits", outputs["combo_logits"]),
            )
        pred_combo = int(combo_logits_struct[0].argmax(dim=-1).item())
        pred_prob = float(torch.softmax(combo_logits_struct[0], dim=-1)[pred_combo].item())
        gt_combo = int(combo_id.item())
        pred_start_bin = int(outputs["logits_start_bin"][0].argmax(dim=-1).item())
        pred_end_bin = int(outputs["logits_end_bin"][0].argmax(dim=-1).item())
        pred_end_bin = max(pred_end_bin, pred_start_bin)
        gt_start_bin = int(gt_action_start_bin[0].item())
        gt_end_bin = int(gt_action_end_bin[0].item())
        gt_start_norm = float(gt_action_start_norm[0].item())
        gt_end_norm = float(gt_action_end_norm[0].item())
        pred_start_norm = float(outputs.get("pred_action_start_norm", gt_action_start_norm)[0].item())
        pred_end_norm = float(outputs.get("pred_action_end_norm", gt_action_end_norm)[0].item())
        vis_scale = max(1, int(getattr(args, "vis_scale", 3)))

        pred_tool_xy_np = outputs["pred_tool_xy"][0].detach().float().cpu().numpy()
        pred_ct_xy_np = outputs["pred_ct_xy"][0].detach().float().cpu().numpy()
        heatmap_tool_np = outputs["heatmap_tool"][0].detach().float().cpu().numpy()
        heatmap_ct_np = outputs["heatmap_ct"][0].detach().float().cpu().numpy()
        np.save(sample_dir / "pred_tool_xy.npy", pred_tool_xy_np)
        np.save(sample_dir / "pred_ct_xy.npy", pred_ct_xy_np)
        np.save(sample_dir / "start_logits_bin.npy", outputs["logits_start_bin"][0].detach().float().cpu().numpy().astype(np.float32))
        np.save(sample_dir / "end_logits_bin.npy", outputs["logits_end_bin"][0].detach().float().cpu().numpy().astype(np.float32))
        np.save(sample_dir / "heatmap_tool.npy", heatmap_tool_np)
        np.save(sample_dir / "heatmap_ct.npy", heatmap_ct_np)

        for t in range(frame_cap):
            seg_start = int(round(t * ratio))
            seg_end = int(round((t + 1) * ratio)) - 1
            seg_start = max(0, min(seg_start, total_t - 1))
            seg_end = max(seg_start, min(seg_end, total_t - 1))
            frame_idx = (seg_start + seg_end) // 2

            frame = video_u8[frame_idx]
            h, w = frame.shape[:2]

            hm_tool = outputs["heatmap_tool"][0, t].float().sum(dim=0, keepdim=False).unsqueeze(0).unsqueeze(0)
            hm_ct = outputs["heatmap_ct"][0, t].float().sum(dim=0, keepdim=False).unsqueeze(0).unsqueeze(0)
            hm_tool_up = F.interpolate(hm_tool, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            hm_ct_up = F.interpolate(hm_ct, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            hm_tool_overlay = _overlay_heatmap(frame, hm_tool_up, color=(60, 220, 60), alpha=args.vis_heatmap_alpha)
            hm_contact_overlay = _overlay_heatmap(frame, hm_ct_up, color=(60, 120, 255), alpha=args.vis_heatmap_alpha)

            gt_overlay = _draw_gt_overlay(
                base_rgb_u8=frame,
                gt_tool_xy=aligned["gt_tool_xy"][0, t].detach().float().cpu().numpy(),
                gt_tool_vis=aligned["gt_tool_vis"][0, t].detach().float().cpu().numpy(),
                gt_ct_xy=aligned["gt_ct_xy"][0, t].detach().float().cpu().numpy(),
                gt_ct_vis=aligned["gt_ct_vis"][0, t].detach().float().cpu().numpy(),
            )
            pred_overlay = _draw_pred_overlay(
                base_rgb_u8=frame,
                pred_tool_xy=pred_tool_xy_np[t],
                pred_ct_xy=pred_ct_xy_np[t],
            )

            fname = f"t{t:02d}_f{frame_idx:02d}.png"
            _save_vis_image(raw_dir / fname, frame, scale=vis_scale)
            _save_vis_image(hm_tool_dir / fname, hm_tool_overlay, scale=vis_scale)
            _save_vis_image(hm_contact_dir / fname, hm_contact_overlay, scale=vis_scale)
            _save_vis_image(gt_dir / fname, gt_overlay, scale=vis_scale)
            _save_vis_image(pred_dir / fname, pred_overlay, scale=vis_scale)

        vis_meta = {
            "sid": sid,
            "epoch": epoch,
            "split": split_name,
            "selection_mode": selection_mode,
            "video_source": video_source,
            "video_shape_thwc": list(video_u8.shape),
            "gt_combo_id": gt_combo,
            "gt_combo_name": combo_id_to_name.get(gt_combo, f"combo_{gt_combo}"),
            "pred_combo_id": pred_combo,
            "pred_combo_name": combo_id_to_name.get(pred_combo, f"combo_{pred_combo}"),
            "pred_combo_prob": pred_prob,
            "gt_action_start_bin": gt_start_bin,
            "gt_action_end_bin": gt_end_bin,
            "gt_action_start_norm": gt_start_norm,
            "gt_action_end_norm": gt_end_norm,
            "gt_action_range_mask": float(action_range_mask[0].item()),
            "pred_action_start_bin": pred_start_bin,
            "pred_action_end_bin": pred_end_bin,
            "pred_action_start_norm": pred_start_norm,
            "pred_action_end_norm": pred_end_norm,
            "t_prime": int(t_prime),
            "visualized_frames": int(frame_cap),
            "visual_layout": ["raw", "heatmap_tool", "heatmap_contact", "gt", "pred"],
            "visual_scale": int(vis_scale),
            "loss_total": float(ps_losses["loss_total"][0].item()),
            "loss_tool_xy": float(ps_losses["loss_tool_xy"][0].item()),
            "loss_ct_xy": float(ps_losses["loss_ct_xy"][0].item()),
            "loss_tool_centroid_aux": float(ps_losses.get("loss_tool_centroid_aux", ps_losses["loss_total"] * 0.0)[0].item()),
            "loss_tool_offset_aux": float(ps_losses.get("loss_tool_offset_aux", ps_losses["loss_total"] * 0.0)[0].item()),
            "loss_tool_pair_sep_aux": float(ps_losses.get("loss_tool_pair_sep_aux", ps_losses["loss_total"] * 0.0)[0].item()),
            "loss_combo_ce": float(ps_losses["loss_combo_ce"][0].item()),
            "loss_combo_tool_ce": float(ps_losses.get("loss_combo_tool_ce", ps_losses["loss_total"] * 0.0)[0].item()),
            "loss_combo_action_ce": float(ps_losses.get("loss_combo_action_ce", ps_losses["loss_total"] * 0.0)[0].item()),
            "loss_start_bin_ce": float(ps_losses["loss_start_bin_ce"][0].item()),
            "loss_end_bin_ce": float(ps_losses["loss_end_bin_ce"][0].item()),
            "loss_start_l1": float(ps_losses["loss_start_bin_ce"][0].item()),
            "loss_end_l1": float(ps_losses["loss_end_bin_ce"][0].item()),
            "loss_range_cls": _loss_value_for_sample(
                ps_losses.get("loss_range_cls", ps_losses["loss_range"] * 0.0),
                sample_idx=0,
            ),
            "loss_range_reg": _loss_value_for_sample(
                ps_losses.get("loss_range_reg", ps_losses["loss_range"] * 0.0),
                sample_idx=0,
            ),
            "loss_range": float(ps_losses["loss_range"][0].item()),
        }
        (sample_dir / "meta.json").write_text(json.dumps(vis_meta, indent=2), encoding="utf-8")
        saved += 1

    return {"enabled": "true", "saved_samples": str(saved), "path": str(vis_root)}


def _finalize_meter(meter: Dict[str, float]) -> Dict[str, float]:
    n = max(int(meter.get("n", 0)), 1)
    range_n = float(meter.get("range_count", 0.0))
    loss_start_l1 = meter.get("loss_start_bin_ce", 0.0) / n
    loss_end_l1 = meter.get("loss_end_bin_ce", 0.0) / n
    track_pck_den = float(meter.get("traj_pck_den", 0.0))
    range_den = max(float(meter.get("range_valid_den", 0.0)), 1.0)
    track_pck_keys = _track_pck_keys()
    out = {
        "loss_total": meter["loss_total"] / n,
        "loss_tool_xy": meter["loss_tool_xy"] / n,
        "loss_ct_xy": meter["loss_ct_xy"] / n,
        "loss_tool_centroid_aux": meter.get("loss_tool_centroid_aux", 0.0) / n,
        "loss_tool_offset_aux": meter.get("loss_tool_offset_aux", 0.0) / n,
        "loss_tool_pair_sep_aux": meter.get("loss_tool_pair_sep_aux", 0.0) / n,
        "loss_tool_heatmap_ce": meter.get("loss_tool_heatmap_ce", 0.0) / n,
        "loss_ct_heatmap_ce": meter.get("loss_ct_heatmap_ce", 0.0) / n,
        "loss_combo_ce": meter["loss_combo_ce"] / n,
        "loss_combo_tool_ce": meter.get("loss_combo_tool_ce", 0.0) / n,
        "loss_combo_action_ce": meter.get("loss_combo_action_ce", 0.0) / n,
        "loss_start_bin_ce": loss_start_l1,
        "loss_end_bin_ce": loss_end_l1,
        "loss_start_l1": loss_start_l1,
        "loss_end_l1": loss_end_l1,
        "loss_range_cls": meter.get("loss_range_cls", 0.0) / n,
        "loss_range_reg": meter.get("loss_range_reg", 0.0) / n,
        "loss_range_proposal": meter.get("loss_range_proposal", 0.0) / n,
        "loss_range_quality": meter.get("loss_range_quality", 0.0) / n,
        "loss_range_sola": meter.get("loss_range_sola", 0.0) / n,
        "loss_range": meter["loss_range"] / n,
        "combo_acc": meter["combo_correct"] / n,
        "tool_acc": meter.get("tool_correct", 0.0) / n,
        "action_acc": meter.get("action_correct", 0.0) / n,
        "traj_ade_px": _safe_div(meter.get("traj_ade_num", 0.0), meter.get("traj_ade_den", 0.0)),
        "traj_fde_px": _safe_div(meter.get("traj_fde_num", 0.0), meter.get("traj_fde_den", 0.0)),
        "traj_survival_50": _safe_div(meter.get("traj_survival_num", 0.0), meter.get("traj_survival_den", 0.0)),
        "traj_aj": _safe_div(meter.get("traj_aj_sum", 0.0), meter.get("traj_aj_den", 0.0)),
        "range_tiou": _safe_div(meter.get("range_tiou_sum", 0.0), range_den),
        "range_r03": _safe_div(meter.get("range_r03_sum", 0.0), range_den),
        "range_r05": _safe_div(meter.get("range_r05_sum", 0.0), range_den),
        "range_r07": _safe_div(meter.get("range_r07_sum", 0.0), range_den),
        "range_start_mae_norm": _safe_div(meter.get("range_start_mae_norm_sum", 0.0), range_den),
        "range_end_mae_norm": _safe_div(meter.get("range_end_mae_norm_sum", 0.0), range_den),
        "range_center_mae_norm": _safe_div(meter.get("range_center_mae_norm_sum", 0.0), range_den),
        "range_duration_mae_norm": _safe_div(meter.get("range_duration_mae_norm_sum", 0.0), range_den),
        "range_start_mae_abs": _safe_div(meter.get("range_start_mae_abs_sum", 0.0), range_den),
        "range_end_mae_abs": _safe_div(meter.get("range_end_mae_abs_sum", 0.0), range_den),
        "range_count": range_n,
        "refine_gate_mean": meter.get("refine_gate_mean", 0.0) / n,
        "refine_delta_abs_mean": meter.get("refine_delta_abs_mean", 0.0) / n,
        "refine_centroid_shift_abs_mean": meter.get("refine_centroid_shift_abs_mean", 0.0) / n,
        "corr_entropy_mean": meter.get("corr_entropy_mean", 0.0) / n,
        "corr_top1_minus_top2_mean": meter.get("corr_top1_minus_top2_mean", 0.0) / n,
        "context_entropy_mean": meter.get("context_entropy_mean", 0.0) / n,
        "context_top1_minus_top2_mean": meter.get("context_top1_minus_top2_mean", 0.0) / n,
        "context_offset_abs_mean": meter.get("context_offset_abs_mean", 0.0) / n,
        "context_gate_mean": meter.get("context_gate_mean", 0.0) / n,
        "belief_gate_mean": meter.get("belief_gate_mean", 0.0) / n,
        "anchor_delta_centroid_abs_mean": meter.get("anchor_delta_centroid_abs_mean", 0.0) / n,
        "anchor_alpha_mean": meter.get("anchor_alpha_mean", 0.0) / n,
        "anchor_verify_scale_mean": meter.get("anchor_verify_scale_mean", 0.0) / n,
        "anchor_verify_residual_mean": meter.get("anchor_verify_residual_mean", 0.0) / n,
        "stage0_centroid_noise_abs_mean": meter.get("stage0_centroid_noise_abs_mean", 0.0) / n,
        "factor_anchor_gate_mean": meter.get("factor_anchor_gate_mean", 0.0) / n,
        "factor_offset_gate_mean": meter.get("factor_offset_gate_mean", 0.0) / n,
        "factor_delta_offset_abs_mean": meter.get("factor_delta_offset_abs_mean", 0.0) / n,
        "joint_gate_mean": meter.get("joint_gate_mean", 0.0) / n,
        "template_entropy_mean": meter.get("template_entropy_mean", 0.0) / n,
        "template_top1_minus_top2_mean": meter.get("template_top1_minus_top2_mean", 0.0) / n,
        "template_gate_mean": meter.get("template_gate_mean", 0.0) / n,
        "template_offset_abs_mean": meter.get("template_offset_abs_mean", 0.0) / n,
        "template_delta_abs_mean": meter.get("template_delta_abs_mean", 0.0) / n,
        "conf_tool_mean": meter.get("conf_tool_mean", 0.0) / n,
        "conf_ct_mean": meter.get("conf_ct_mean", 0.0) / n,
        "count": n,
    }
    pck_vals: List[float] = []
    for key in track_pck_keys:
        val = _safe_div(meter.get(f"traj_pck_{key}_num", 0.0), track_pck_den)
        out[f"traj_pck_{key}"] = val
        pck_vals.append(val)
    out["traj_pck_auc"] = (sum(pck_vals) / float(len(pck_vals))) if pck_vals else 0.0
    for stage_idx in _TRAJ_STAGE_IDS:
        out[f"loss_tool_xy_stage{stage_idx}"] = meter.get(f"loss_tool_xy_stage{stage_idx}", 0.0) / n
        out[f"loss_ct_xy_stage{stage_idx}"] = meter.get(f"loss_ct_xy_stage{stage_idx}", 0.0) / n
    for key in _TRAJ_LATE_FOCAL_METRIC_KEYS:
        out[key] = meter.get(key, 0.0) / n
    for k, v in meter.items():
        if str(k).startswith("timing_"):
            out[str(k)] = float(v) / float(n)
    return out


def _tb_safe_name(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "_-.") else "_" for ch in name)


def _print_model_shape_debug(log_write, shape_debug: Dict[str, object], epoch: int, step: int) -> None:
    if not isinstance(shape_debug, dict) or (len(shape_debug) == 0):
        return

    def _fmt_shape(v: object) -> str:
        if isinstance(v, (list, tuple)):
            try:
                return str(tuple(int(x) for x in v))
            except Exception:
                return str(v)
        return str(v)

    log_write(f"[shape][epoch {epoch:03d}][step {step:05d}] BEFORE REFINER")
    main_keys = [
        "tokens_grid",
        "x_tokens",
        "x_proj",
        "feat_btchw",
        "feat_flat_btsd",
        "tool_probs",
        "ct_probs",
        "tool_pred_xy_pre_refine",
        "ct_pred_xy_pre_refine",
        "feat_refine_btsd",
        "tool_local_feat",
        "ct_local_feat",
    ]
    for k in main_keys:
        if k in shape_debug:
            log_write(f"  {k}: {_fmt_shape(shape_debug[k])}")

    for group_key, title in (("tool_refiner", "TOOL REFINER INPUT PIPE"), ("contact_refiner", "CONTACT REFINER INPUT PIPE")):
        sub = shape_debug.get(group_key, None)
        if not isinstance(sub, dict):
            continue
        log_write(f"[shape][epoch {epoch:03d}][step {step:05d}] {title}")
        sub_keys = [
            "local_feat_in",
            "local_feat_small",
            "token",
            "h_point",
            "h_spatial",
            "h_time",
            "h_out",
            "delta_xy_raw",
            "delta_gate",
        ]
        for sk in sub_keys:
            if sk in sub:
                log_write(f"  {group_key}.{sk}: {_fmt_shape(sub[sk])}")


def _print_stage_header(title: str, line_len: int = 96) -> None:
    line = "=" * line_len
    print(f"\n{line}\n{title}\n{line}")


def _is_progress_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "use_progress", True)) and (tqdm is not None)


def _module_grad_stats(module: torch.nn.Module) -> Dict[str, int]:
    total_params = 0
    requires_grad_params = 0
    grad_params = 0
    for p in module.parameters():
        n = int(p.numel())
        total_params += n
        if p.requires_grad:
            requires_grad_params += n
            if p.grad is not None:
                grad_params += n
    return {
        "total_params": total_params,
        "requires_grad_params": requires_grad_params,
        "grad_params": grad_params,
    }


def _collect_grad_contract(model: KeypointComboNet) -> Dict[str, Dict[str, int]]:
    report: Dict[str, Dict[str, int]] = {
        "vjepa_encoder": _module_grad_stats(model.vjepa.encoder),
    }
    for name, module in model.trainable_modules.items():
        report[name] = _module_grad_stats(module)
    extra_params = _collect_extra_named_trainable_params(model)
    if extra_params:
        total_params = 0
        requires_grad_params = 0
        grad_params = 0
        for _, p in extra_params:
            n = int(p.numel())
            total_params += n
            if p.requires_grad:
                requires_grad_params += n
                if p.grad is not None:
                    grad_params += n
        report["extra_trainable_params"] = {
            "total_params": total_params,
            "requires_grad_params": requires_grad_params,
            "grad_params": grad_params,
        }
    return report


def _print_grad_contract(report: Dict[str, Dict[str, int]]) -> None:
    print("[grad contract]")
    for name, stats in report.items():
        print(
            f"  {name:16s} total={stats['total_params']} "
            f"requires_grad={stats['requires_grad_params']} grad={stats['grad_params']}"
        )


def _set_module_requires_grad(module: Optional[torch.nn.Module], enabled: bool) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = bool(enabled)


def _apply_dual_probe_branch_freeze(model: KeypointComboNet, args: argparse.Namespace) -> Dict[str, bool]:
    """
    For dual-attn probe mode, hard-freeze idle branches according to CE weights:
      - action-only (tool=0, combo=0, action>0): freeze tool branch
      - tool-only   (action=0, combo=0, tool>0): freeze action branch
    """
    freeze_tool = False
    freeze_action = False
    if bool(getattr(args, "combo_probe_dual_attn", False)):
        w_combo = float(getattr(args, "combo_ce_weight", 0.0))
        w_tool = float(getattr(args, "combo_tool_ce_weight", 0.0))
        w_action = float(getattr(args, "combo_action_ce_weight", 0.0))
        if (w_combo == 0.0) and (w_tool == 0.0) and (w_action > 0.0):
            freeze_tool = True
        if (w_combo == 0.0) and (w_action == 0.0) and (w_tool > 0.0):
            freeze_action = True

    if freeze_tool:
        _set_module_requires_grad(getattr(model, "combo_tool_spatial_pool", None), False)
        _set_module_requires_grad(getattr(model, "combo_tool_head", None), False)
    if freeze_action:
        _set_module_requires_grad(getattr(model, "combo_action_temporal_pool", None), False)
        _set_module_requires_grad(getattr(model, "combo_action_head", None), False)
        _set_module_requires_grad(getattr(model, "combo_action_tool_prior", None), False)
    return {
        "freeze_tool_branch": bool(freeze_tool),
        "freeze_action_branch": bool(freeze_action),
    }


def _load_sid_to_case(data_root: Path) -> Dict[str, str]:
    meta_path = data_root / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    sid_to_case: Dict[str, str] = {}
    if isinstance(meta, list):
        for it in meta:
            if not isinstance(it, dict):
                continue
            sid = str(it.get("sid", "")).strip()
            case = str(it.get("case", "")).strip()
            if sid and case:
                sid_to_case[sid] = case
    return sid_to_case


def _count_cases(sids: List[str], sid_to_case: Dict[str, str]) -> Tuple[Dict[str, int], int]:
    counts = Counter()
    unknown = 0
    for sid in sids:
        case_name = sid_to_case.get(str(sid), "")
        if case_name:
            counts[case_name] += 1
        else:
            unknown += 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0])), int(unknown)


def _print_case_counts(split: str, counts: Dict[str, int], unknown: int) -> None:
    known_n = int(sum(int(v) for v in counts.values()))
    total_n = int(known_n + int(unknown))
    print(f"[setup] {split}_case_counts: total={total_n} known={known_n} unknown={unknown}")
    for case_name, n in counts.items():
        print(f"  {case_name:40s} {int(n)}")


def _compute_action_bins_for_sid(data_root: Path, sid: str, n_bins: int) -> Tuple[int, int, float]:
    sampled_abs = np.load(data_root / "sampled_abs_frames" / f"{sid}.npy").reshape(-1).astype(np.float64)
    action_abs = np.load(data_root / "action_range_abs" / f"{sid}.npy").reshape(-1).astype(np.float64)
    action_mask = np.load(data_root / "action_range_mask" / f"{sid}.npy").reshape(-1).astype(np.float64)

    if sampled_abs.size < 2 or action_abs.size < 2 or action_mask.size < 1:
        return 0, 0, 0.0

    mask = 1.0 if float(action_mask[0]) > 0.5 else 0.0
    clip_s = float(sampled_abs[0])
    clip_e = float(sampled_abs[-1])
    den = clip_e - clip_s
    if mask <= 0.0 or den <= 0.0:
        return 0, 0, 0.0

    start_norm = (float(action_abs[0]) - clip_s) / den
    end_norm = (float(action_abs[1]) - clip_s) / den
    start_norm = float(np.clip(start_norm, 0.0, 1.0))
    end_norm = float(np.clip(end_norm, 0.0, 1.0))
    if end_norm < start_norm:
        end_norm = start_norm

    sb = int(np.floor(start_norm * n_bins))
    eb = int(np.floor(end_norm * n_bins))
    sb = int(np.clip(sb, 0, n_bins - 1))
    eb = int(np.clip(eb, 0, n_bins - 1))
    if eb < sb:
        eb = sb
    return sb, eb, 1.0


def _compute_action_duration_norm_for_sid(data_root: Path, sid: str) -> Tuple[float, float]:
    sampled_abs = np.load(data_root / "sampled_abs_frames" / f"{sid}.npy").reshape(-1).astype(np.float64)
    action_abs = np.load(data_root / "action_range_abs" / f"{sid}.npy").reshape(-1).astype(np.float64)
    action_mask = np.load(data_root / "action_range_mask" / f"{sid}.npy").reshape(-1).astype(np.float64)

    if sampled_abs.size < 2 or action_abs.size < 2 or action_mask.size < 1:
        return 0.0, 0.0

    mask = 1.0 if float(action_mask[0]) > 0.5 else 0.0
    clip_s = float(sampled_abs[0])
    clip_e = float(sampled_abs[-1])
    den = clip_e - clip_s
    if mask <= 0.0 or den <= 0.0:
        return 0.0, 0.0

    start_norm = float(np.clip((float(action_abs[0]) - clip_s) / den, 0.0, 1.0))
    end_norm = float(np.clip((float(action_abs[1]) - clip_s) / den, 0.0, 1.0))
    if end_norm < start_norm:
        end_norm = start_norm
    return float(end_norm - start_norm), 1.0


def _compute_combo_duration_prior_stats(
    data_root: Path,
    sids: List[str],
    num_combo_classes: int,
    min_std: float = 0.05,
) -> Dict[str, Any]:
    values: List[List[float]] = [[] for _ in range(int(num_combo_classes))]
    missing = 0
    invalid = 0
    for sid in sids:
        combo_path = data_root / "combo_ids" / f"{sid}.npy"
        if not combo_path.is_file():
            missing += 1
            continue
        try:
            combo_np = np.load(combo_path).reshape(-1)
            if combo_np.size <= 0:
                invalid += 1
                continue
            combo_id = int(combo_np[0])
            if combo_id < 0 or combo_id >= int(num_combo_classes):
                invalid += 1
                continue
            dur_norm, valid = _compute_action_duration_norm_for_sid(data_root=data_root, sid=str(sid))
            if valid <= 0.0:
                continue
            values[combo_id].append(float(dur_norm))
        except Exception:
            invalid += 1
            continue
    means: List[float] = []
    stds: List[float] = []
    counts: List[int] = []
    for vals in values:
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size <= 0:
            means.append(0.5)
            stds.append(float(min_std))
            counts.append(0)
            continue
        means.append(float(arr.mean()))
        stds.append(float(max(float(arr.std(ddof=0)), float(min_std))))
        counts.append(int(arr.size))
    return {
        "means": means,
        "stds": stds,
        "counts": counts,
        "missing_combo_ids": int(missing),
        "invalid_rows": int(invalid),
    }


def _make_reweight_vector(counts: np.ndarray, power: float = 0.5, max_factor: float = 10.0) -> np.ndarray:
    cnt = np.asarray(counts, dtype=np.float64).reshape(-1)
    pos = cnt[cnt > 0]
    fallback = float(pos.min()) if pos.size > 0 else 1.0
    cnt = np.where(cnt > 0, cnt, fallback)
    w = np.power(cnt, -float(power))
    w = w / max(float(w.mean()), 1e-12)
    w = np.clip(w, 0.0, float(max_factor))
    return w.astype(np.float32)


def _make_effective_num_vector(counts: np.ndarray, beta: float = 0.9997, max_factor: float = 4.0) -> np.ndarray:
    cnt = np.asarray(counts, dtype=np.float64).reshape(-1)
    pos = cnt[cnt > 0]
    fallback = float(pos.min()) if pos.size > 0 else 1.0
    cnt = np.where(cnt > 0, cnt, fallback)
    beta = float(beta)
    if beta < 0.0 or beta >= 1.0:
        raise ValueError(f"`beta` must be in [0,1), got {beta}")
    effective_num = 1.0 - np.power(beta, cnt)
    effective_num = np.clip(effective_num, 1e-12, None)
    w = (1.0 - beta) / effective_num
    w = w / max(float(w.mean()), 1e-12)
    w = np.clip(w, 0.0, float(max_factor))
    return w.astype(np.float32)


def _range_combo_reweight_schedule_scale(
    epoch: int,
    start_epoch: int = 1,
    ramp_epochs: int = 1,
) -> float:
    cur_epoch = int(epoch)
    start_epoch = max(1, int(start_epoch))
    ramp_epochs = max(1, int(ramp_epochs))
    if cur_epoch < start_epoch:
        return 0.0
    return min(1.0, float(cur_epoch - start_epoch + 1) / float(ramp_epochs))


def _blend_combo_reweight_vector(base_weights: np.ndarray, scale: float) -> np.ndarray:
    w = np.asarray(base_weights, dtype=np.float32).reshape(-1)
    alpha = float(np.clip(scale, 0.0, 1.0))
    return (1.0 + alpha * (w - 1.0)).astype(np.float32)


def _parse_range_combo_reweight_targets(spec: str) -> List[str]:
    allowed = ("start", "end", "cls", "reg", "boundary", "proposal", "quality", "sola")
    parts: List[str] = []
    seen = set()
    raw = str(spec).strip().lower()
    if (not raw) or raw == "none":
        return parts
    for item in raw.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in allowed:
            raise ValueError(
                "--range-combo-reweight-targets must contain only: "
                + ",".join(allowed)
                + f"; got `{name}`"
            )
        if name in seen:
            continue
        seen.add(name)
        parts.append(name)
    return parts


def _build_train_reweighting(
    data_root: Path,
    train_sids: List[str],
    num_combo_classes: int,
    num_range_bins: int,
    power: float,
    max_factor: float,
    include_range_bins: bool = True,
) -> Dict[str, Any]:
    combo_counts = np.zeros((int(num_combo_classes),), dtype=np.int64)
    start_counts = np.zeros((int(num_range_bins),), dtype=np.int64)
    end_counts = np.zeros((int(num_range_bins),), dtype=np.int64)
    valid_range = 0

    sid_to_combo_id: Dict[str, int] = {}
    for sid in train_sids:
        combo_np = np.load(data_root / "combo_ids" / f"{sid}.npy")
        combo_id = int(combo_np.reshape(-1)[0])
        if combo_id < 0 or combo_id >= int(num_combo_classes):
            raise ValueError(f"combo_id out of range for sid={sid}: {combo_id}")
        combo_counts[combo_id] += 1
        sid_to_combo_id[str(sid)] = combo_id

        if include_range_bins:
            sb, eb, m = _compute_action_bins_for_sid(data_root=data_root, sid=str(sid), n_bins=int(num_range_bins))
            if m > 0.5:
                valid_range += 1
                start_counts[sb] += 1
                end_counts[eb] += 1

    combo_weights = _make_reweight_vector(combo_counts, power=power, max_factor=max_factor)
    start_bin_weights = _make_reweight_vector(start_counts, power=power, max_factor=max_factor)
    end_bin_weights = _make_reweight_vector(end_counts, power=power, max_factor=max_factor)

    sample_weight_by_sid = {sid: float(combo_weights[cid]) for sid, cid in sid_to_combo_id.items()}
    report = {
        "power": float(power),
        "max_factor": float(max_factor),
        "combo_counts": [int(x) for x in combo_counts.tolist()],
        "combo_weights": [float(x) for x in combo_weights.tolist()],
        "start_bin_counts": [int(x) for x in start_counts.tolist()],
        "start_bin_weights": [float(x) for x in start_bin_weights.tolist()],
        "end_bin_counts": [int(x) for x in end_counts.tolist()],
        "end_bin_weights": [float(x) for x in end_bin_weights.tolist()],
        "range_valid_count": int(valid_range),
        "range_valid_ratio": float(valid_range / max(len(train_sids), 1)),
        "range_bins_enabled": bool(include_range_bins),
    }
    return {
        "combo_weights": combo_weights,
        "start_bin_weights": start_bin_weights,
        "end_bin_weights": end_bin_weights,
        "sample_weight_by_sid": sample_weight_by_sid,
        "report": report,
    }


def _make_train_sampler(ds: SurgKptComboDataset, sample_weight_by_sid: Dict[str, float]) -> WeightedRandomSampler:
    weights = [float(sample_weight_by_sid.get(str(sid), 1.0)) for sid in ds.sids]
    if not weights:
        raise ValueError("Cannot build train sampler: empty dataset.")
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


@torch.no_grad()
def run_debug_sanity(
    model: KeypointComboNet,
    train_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path,
    combo_class_weights: Optional[torch.Tensor] = None,
    start_bin_class_weights: Optional[torch.Tensor] = None,
    end_bin_class_weights: Optional[torch.Tensor] = None,
    range_combo_reweight_weights: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    model.eval()
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = _get_amp_dtype(args.amp_dtype)
    traj_m2_stage_weights = _resolve_train_traj_stage_weights(args, epoch=1)

    batch = next(iter(train_loader))
    batch = _to_device(batch, device)
    video = batch["video"]
    gt_tool_xy = batch["gt_tool_xy"]
    gt_tool_vis = batch["gt_tool_vis"]
    gt_tool_xy, gt_tool_vis = _maybe_use_tool_centroid_gt(
        gt_tool_xy=gt_tool_xy,
        gt_tool_vis=gt_tool_vis,
        use_tool_centroid_only=bool(getattr(args, "tool_centroid_only", False)),
    )
    gt_ct_xy = batch["gt_ct_xy"]
    gt_ct_vis = batch["gt_ct_vis"]
    gt_action_start_bin = batch["gt_action_start_bin"].long()
    gt_action_end_bin = batch["gt_action_end_bin"].long()
    if ("gt_action_start_norm" in batch) and ("gt_action_end_norm" in batch):
        gt_action_start_norm = batch["gt_action_start_norm"].float()
        gt_action_end_norm = batch["gt_action_end_norm"].float()
    else:
        gt_action_start_norm, gt_action_end_norm = _derive_action_norm_from_bins(
            gt_action_start_bin=gt_action_start_bin,
            gt_action_end_bin=gt_action_end_bin,
            num_bins=RANGE_NUM_BINS,
        )
    action_range_mask = batch["action_range_mask"].float()
    combo_id = batch["combo_id"].long()
    tool_pt_role = batch.get("pt_role", None)
    tool_id, action_id = _derive_tool_action_id_from_combo_id(combo_id)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        outputs = model(video, tool_id=tool_id, action_id=action_id, combo_id=combo_id)
        if (
            bool(getattr(args, "tool_only", False))
            or bool(getattr(args, "combo_probe_attn", False))
            or bool(getattr(args, "combo_probe_dual_attn", False))
        ):
            aligned = {
                "gt_tool_xy": gt_tool_xy,
                "gt_tool_vis": gt_tool_vis,
                "gt_ct_xy": gt_ct_xy,
                "gt_ct_vis": gt_ct_vis,
            }
        else:
            t_prime = outputs["pred_tool_xy"].shape[1]
            aligned = align_targets_to_tprime(gt_tool_xy, gt_tool_vis, gt_ct_xy, gt_ct_vis, t_prime=t_prime)
        losses = compute_batch_losses(
            outputs=outputs,
            gt_tool_xy=aligned["gt_tool_xy"],
            gt_tool_vis=aligned["gt_tool_vis"],
            gt_ct_xy=aligned["gt_ct_xy"],
            gt_ct_vis=aligned["gt_ct_vis"],
            combo_id=combo_id,
            gt_action_start_bin=gt_action_start_bin,
            gt_action_end_bin=gt_action_end_bin,
            action_range_mask=action_range_mask,
            gt_action_start_norm=gt_action_start_norm,
            gt_action_end_norm=gt_action_end_norm,
            combo_ce_weight=float(getattr(args, "combo_ce_weight", COMBO_CE_WEIGHT)),
            combo_tool_ce_weight=float(getattr(args, "combo_tool_ce_weight", 0.5)),
            combo_action_ce_weight=float(getattr(args, "combo_action_ce_weight", 0.5)),
            start_bin_loss_weight=RANGE_START_CE_WEIGHT,
            end_bin_loss_weight=RANGE_END_CE_WEIGHT,
            range_label_smoothing=RANGE_LABEL_SMOOTHING,
            range_positive_mode=str(getattr(args, "range_positive_mode", "span")),
            range_sensitivity_mode=str(getattr(args, "range_sensitivity_mode", "none")),
            range_sensitivity_cls_sigma=float(getattr(args, "range_sensitivity_cls_sigma", 0.5)),
            range_sensitivity_loc_sigma=float(getattr(args, "range_sensitivity_loc_sigma", 0.35)),
            range_sensitivity_min_weight=float(getattr(args, "range_sensitivity_min_weight", 0.25)),
            range_cls_target_mode=str(getattr(args, "range_cls_target_mode", "binary")),
            range_cls_varifocal_alpha=float(getattr(args, "range_cls_varifocal_alpha", 0.75)),
            range_cls_varifocal_weight=float(getattr(args, "range_cls_varifocal_weight", 1.0)),
            range_quality_mode=str(getattr(args, "range_quality_mode", "none")),
            range_quality_loss_weight=float(getattr(args, "range_quality_loss_weight", 0.0)),
            range_sola_aux_weight=float(getattr(args, "range_sola_aux_weight", 0.0)),
            range_sola_similarity_k=float(getattr(args, "range_sola_sim_k", 16.0)),
            range_focal_gamma=RANGE_FOCAL_GAMMA,
            range_neighbor_smoothing=RANGE_NEIGHBOR_SMOOTHING,
            range_boundary_aux_weight=float(getattr(args, "range_boundary_aux_weight", 0.0)),
            range_proposal_weight=float(getattr(args, "range_proposal_weight", 0.0)),
            range_combo_reweight_weights=range_combo_reweight_weights,
            range_combo_reweight_targets=str(getattr(args, "range_combo_reweight_targets", "")),
            combo_class_weights=combo_class_weights,
            start_bin_class_weights=start_bin_class_weights,
            end_bin_class_weights=end_bin_class_weights,
            kpt_loss_type=str(getattr(args, "kpt_loss_type", "l1l2")),
            traj_loss_pixel_scale=float(getattr(args, "traj_loss_pixel_scale", 256.0)),
            traj_huber_delta=float(getattr(args, "traj_huber_delta", 6.0)),
            traj_t0_loss_weight=float(getattr(args, "traj_t0_loss_weight", 1.0)),
            traj_loss_visible_only=bool(getattr(args, "traj_loss_visible_only", False)),
            traj_centroid_aux_weight=float(getattr(args, "traj_centroid_aux_weight", 0.0)),
            traj_offset_aux_weight=float(getattr(args, "traj_offset_aux_weight", 0.0)),
            trajectory_target=(
                str(getattr(args, "trajectory_target", "tool"))
                if bool(getattr(args, "trajectory_only", False))
                else "both"
            ),
            trajectory_only=bool(getattr(args, "trajectory_only", False)),
            range_only=bool(getattr(args, "range_only", False)),
            combo_only=bool(getattr(args, "combo_only", False)),
            tool_only=bool(getattr(args, "tool_only", False)),
            traj_heatmap_ce_weight=float(getattr(args, "traj_heatmap_ce_weight", 0.0)),
            traj_heatmap_sigma=float(getattr(args, "traj_heatmap_sigma", 2.45)),
            tool_pt_role=tool_pt_role,
            symmetry_swap_invariant_loss=bool(getattr(args, "symmetry_swap_invariant_loss", False)),
            symmetry_pair_separation_loss=bool(getattr(args, "symmetry_pair_separation_loss", False)),
            symmetry_pair_separation_weight=float(getattr(args, "symmetry_pair_separation_weight", 0.05)),
            symmetry_pair_separation_min_pred_gap_px=float(
                getattr(args, "symmetry_pair_separation_min_pred_gap_px", 8.0)
            ),
            symmetry_pair_separation_gt_min_gap_px=float(
                getattr(args, "symmetry_pair_separation_gt_min_gap_px", 16.0)
            ),
            traj_m2_stage_weights=traj_m2_stage_weights,
        )

    trainable_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total_count = int(sum(p.numel() for p in model.parameters()))
    encoder_trainable = any(p.requires_grad for p in model.vjepa.encoder.parameters())
    refine_stats = _compute_refine_stats(outputs)

    report: Dict[str, object] = {
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "amp_dtype": args.amp_dtype,
        "vjepa_arch": args.vjepa_arch,
        "vjepa_hf_repo": args.vjepa_hf_repo,
        "vjepa_loaded_sources": list(getattr(model.vjepa, "loaded_sources", [])),
        "vjepa_has_pretrained": bool(getattr(model.vjepa, "has_pretrained", False)),
        "vjepa_load_report": dict(getattr(model.vjepa, "load_report", {})),
        "vjepa_component_report": dict(getattr(model.vjepa, "component_report", {})),
        "encoder_trainable_any": bool(encoder_trainable),
        "param_count_total": total_count,
        "param_count_trainable": trainable_count,
        "input_shape_video": list(video.shape),
        "input_shape_gt_tool_xy": list(gt_tool_xy.shape),
        "input_shape_gt_ct_xy": list(gt_ct_xy.shape),
        "input_shape_gt_action_start_bin": list(gt_action_start_bin.shape),
        "input_shape_gt_action_end_bin": list(gt_action_end_bin.shape),
        "symmetry_swap_invariant_loss": bool(getattr(args, "symmetry_swap_invariant_loss", False)),
        "symmetry_pair_separation_loss": bool(getattr(args, "symmetry_pair_separation_loss", False)),
        "input_tool_vis_minmax": [float(gt_tool_vis.min().item()), float(gt_tool_vis.max().item())],
        "input_ct_vis_minmax": [float(gt_ct_vis.min().item()), float(gt_ct_vis.max().item())],
        "input_action_range_mask_minmax": [float(action_range_mask.min().item()), float(action_range_mask.max().item())],
        "output_shape_pred_tool_xy": list(outputs["pred_tool_xy"].shape),
        "output_shape_pred_ct_xy": list(outputs["pred_ct_xy"].shape),
        "output_shape_combo_logits": list(outputs["combo_logits"].shape),
        "output_shape_combo_tool_logits": list(outputs.get("combo_tool_logits", outputs["combo_logits"]).shape),
        "output_shape_combo_action_logits": list(outputs.get("combo_action_logits", outputs["combo_logits"]).shape),
        "output_shape_logits_start_bin": list(outputs["logits_start_bin"].shape),
        "output_shape_logits_end_bin": list(outputs["logits_end_bin"].shape),
        "output_shape_heatmap_tool": list(outputs["heatmap_tool"].shape),
        "output_shape_heatmap_ct": list(outputs["heatmap_ct"].shape),
        "meta": outputs["meta"],
        "loss_total": float(losses["loss_total"].item()),
        "loss_tool_xy": float(losses["loss_tool_xy"].item()),
        "loss_ct_xy": float(losses["loss_ct_xy"].item()),
        "loss_tool_centroid_aux": float(losses.get("loss_tool_centroid_aux", losses["loss_total"] * 0.0).item()),
        "loss_tool_offset_aux": float(losses.get("loss_tool_offset_aux", losses["loss_total"] * 0.0).item()),
        "loss_tool_pair_sep_aux": float(losses.get("loss_tool_pair_sep_aux", losses["loss_total"] * 0.0).item()),
        "loss_combo_ce": float(losses["loss_combo_ce"].item()),
        "loss_combo_tool_ce": float(losses.get("loss_combo_tool_ce", losses["loss_total"] * 0.0).item()),
        "loss_combo_action_ce": float(losses.get("loss_combo_action_ce", losses["loss_total"] * 0.0).item()),
        "loss_start_bin_ce": float(losses["loss_start_bin_ce"].item()),
        "loss_end_bin_ce": float(losses["loss_end_bin_ce"].item()),
        "loss_start_l1": float(losses["loss_start_bin_ce"].item()),
        "loss_end_l1": float(losses["loss_end_bin_ce"].item()),
        "loss_range_cls": float(losses.get("loss_range_cls", losses["loss_range"] * 0.0).item()),
        "loss_range_reg": float(losses.get("loss_range_reg", losses["loss_range"] * 0.0).item()),
        "loss_range_proposal": float(losses.get("loss_range_proposal", losses["loss_range"] * 0.0).item()),
        "loss_range": float(losses["loss_range"].item()),
        "refine_gate_mean": float(refine_stats["refine_gate_mean"]),
        "refine_delta_abs_mean": float(refine_stats["refine_delta_abs_mean"]),
        "refine_centroid_shift_abs_mean": float(refine_stats.get("refine_centroid_shift_abs_mean", 0.0)),
        "anchor_delta_centroid_abs_mean": float(refine_stats.get("anchor_delta_centroid_abs_mean", 0.0)),
        "anchor_alpha_mean": float(refine_stats.get("anchor_alpha_mean", 0.0)),
        "anchor_verify_scale_mean": float(refine_stats.get("anchor_verify_scale_mean", 1.0)),
        "anchor_verify_residual_mean": float(refine_stats.get("anchor_verify_residual_mean", 0.0)),
        "stage0_centroid_noise_abs_mean": float(refine_stats.get("stage0_centroid_noise_abs_mean", 0.0)),
        "factor_anchor_gate_mean": float(refine_stats.get("factor_anchor_gate_mean", 0.0)),
        "factor_offset_gate_mean": float(refine_stats.get("factor_offset_gate_mean", 0.0)),
        "factor_delta_offset_abs_mean": float(refine_stats.get("factor_delta_offset_abs_mean", 0.0)),
        "context_gate_mean": float(refine_stats.get("context_gate_mean", 0.0)),
        "template_gate_mean": float(refine_stats.get("template_gate_mean", 0.0)),
        "template_entropy_mean": float(refine_stats.get("template_entropy_mean", 0.0)),
        "grad_contract": _collect_grad_contract(model),
    }

    dbg_dir = out_dir / "debug"
    dbg_dir.mkdir(parents=True, exist_ok=True)
    report_path = dbg_dir / "sanity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[debug] sanity check complete")
    print(f"[debug] report: {report_path}")
    print(
        f"[debug] shapes: video={tuple(video.shape)} pred_tool={tuple(outputs['pred_tool_xy'].shape)} "
        f"pred_ct={tuple(outputs['pred_ct_xy'].shape)} combo={tuple(outputs['combo_logits'].shape)} "
        f"tool_cls={tuple(outputs.get('combo_tool_logits', outputs['combo_logits']).shape)} "
        f"action_cls={tuple(outputs.get('combo_action_logits', outputs['combo_logits']).shape)} "
        f"range_start={tuple(outputs['logits_start_bin'].shape)} range_end={tuple(outputs['logits_end_bin'].shape)}"
    )
    print(
        f"[debug] losses: total={report['loss_total']:.5f} tool={report['loss_tool_xy']:.5f} "
        f"ct={report['loss_ct_xy']:.5f} combo_ce={report['loss_combo_ce']:.5f} "
        f"sep_aux={report['loss_tool_pair_sep_aux']:.5f} "
        f"tool_ce={report['loss_combo_tool_ce']:.5f} action_ce={report['loss_combo_action_ce']:.5f} "
        f"start_l1={report['loss_start_l1']:.5f} end_l1={report['loss_end_l1']:.5f} "
        f"range={report['loss_range']:.5f} "
        f"range_cls={report['loss_range_cls']:.5f} range_reg={report['loss_range_reg']:.5f} "
        f"range_prop={report.get('loss_range_proposal', 0.0):.5f} "
        f"ref_gate={report['refine_gate_mean']:.4f} ref_dxy={report['refine_delta_abs_mean']:.5f} "
        f"ref_cshift={report['refine_centroid_shift_abs_mean']:.5f} "
        f"anchor_cshift={report.get('anchor_delta_centroid_abs_mean', 0.0):.5f} "
        f"st0_noise={report.get('stage0_centroid_noise_abs_mean', 0.0):.5f} "
        f"anchor_alpha={report.get('anchor_alpha_mean', 0.0):.4f} "
        f"anchor_vfy={report.get('anchor_verify_scale_mean', 1.0):.4f} "
        f"factor_ag={report.get('factor_anchor_gate_mean', 0.0):.4f} "
        f"factor_og={report.get('factor_offset_gate_mean', 0.0):.4f} "
        f"factor_off={report.get('factor_delta_offset_abs_mean', 0.0):.5f} "
        f"ctx_gate={report['context_gate_mean']:.4f} "
        f"belief_gate={report.get('belief_gate_mean', 0.0):.4f} "
        f"jnt_gate={report.get('joint_gate_mean', 0.0):.4f} "
        f"tpl_gate={report['template_gate_mean']:.4f} tpl_ent={report['template_entropy_mean']:.4f}"
    )
    return report


def train_one_epoch(
    model: KeypointComboNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    out_dir: Path,
    writer: Optional["SummaryWriter"] = None,
    global_step_start: int = 0,
    token_cache: Optional[VJEPATokenCache] = None,
    combo_class_weights: Optional[torch.Tensor] = None,
    start_bin_class_weights: Optional[torch.Tensor] = None,
    end_bin_class_weights: Optional[torch.Tensor] = None,
    range_combo_reweight_weights: Optional[torch.Tensor] = None,
    ema: Optional["TrainableEMA"] = None,
    traj_late_focal_state: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], int]:
    model.train()
    model.vjepa.encoder.eval()
    model.set_token_mask_epoch(int(epoch))

    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = _get_amp_dtype(args.amp_dtype)
    use_progress = _is_progress_enabled(args)
    is_traj_only = bool(getattr(args, "trajectory_only", False))
    is_combo_only = bool(getattr(args, "combo_only", False))
    is_tool_only = bool(getattr(args, "tool_only", False))
    is_combo_probe_attn = bool(getattr(args, "combo_probe_attn", False))
    is_combo_probe_dual_attn = bool(getattr(args, "combo_probe_dual_attn", False))
    traj_late_focal_enable = bool(getattr(args, "traj_late_focal_enable", False))
    train_traj_m2_stage_weights = _resolve_train_traj_stage_weights(args, epoch)

    meter = defaultdict(float)
    log_meter = defaultdict(float)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found.")

    total_steps = len(loader)
    if args.max_train_steps > 0:
        total_steps = min(total_steps, int(args.max_train_steps))

    iterator = loader
    pbar = None
    if use_progress:
        pbar = tqdm(
            loader,
            total=total_steps,
            desc=f"Train E{epoch:03d}",
            dynamic_ncols=True,
            leave=False,
        )
        iterator = pbar

    log_write = pbar.write if pbar is not None else print
    last_step = 0
    shape_debug_printed = False
    for step, batch in enumerate(iterator, start=1):
        last_step = step
        if args.max_train_steps > 0 and step > args.max_train_steps:
            break

        batch = _to_device(batch, device)
        video = batch.get("video", None)  # optional when cache-only mode is active
        gt_tool_xy = batch["gt_tool_xy"]
        gt_tool_vis = batch["gt_tool_vis"]
        gt_tool_xy, gt_tool_vis = _maybe_use_tool_centroid_gt(
            gt_tool_xy=gt_tool_xy,
            gt_tool_vis=gt_tool_vis,
            use_tool_centroid_only=bool(getattr(args, "tool_centroid_only", False)),
        )
        gt_ct_xy = batch["gt_ct_xy"]
        gt_ct_vis = batch["gt_ct_vis"]
        gt_action_start_bin = batch["gt_action_start_bin"].long()
        gt_action_end_bin = batch["gt_action_end_bin"].long()
        if ("gt_action_start_norm" in batch) and ("gt_action_end_norm" in batch):
            gt_action_start_norm = batch["gt_action_start_norm"].float()
            gt_action_end_norm = batch["gt_action_end_norm"].float()
        else:
            gt_action_start_norm, gt_action_end_norm = _derive_action_norm_from_bins(
                gt_action_start_bin=gt_action_start_bin,
                gt_action_end_bin=gt_action_end_bin,
                num_bins=RANGE_NUM_BINS,
            )
        action_range_mask = batch["action_range_mask"].float()
        combo_id = batch["combo_id"].long()
        tool_pt_role = batch.get("pt_role", None)
        tool_id, action_id = _derive_tool_action_id_from_combo_id(combo_id)
        sid_raw = batch.get("sid", [])
        if isinstance(sid_raw, (list, tuple)):
            sid_list = [str(x) for x in sid_raw]
        elif isinstance(sid_raw, str):
            sid_list = [sid_raw]
        else:
            sid_list = [str(sid_raw)]
        debug_this_step = bool(args.debug_train_check and _should_debug_step(step, args))

        if debug_this_step and (video is not None):
            _validate_train_batch_contract(
                video,
                gt_tool_xy,
                gt_tool_vis,
                gt_ct_xy,
                gt_ct_vis,
                gt_action_start_bin,
                gt_action_end_bin,
                action_range_mask,
                combo_id,
            )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = _model_forward_with_cache(
                model=model,
                video=video,
                sids=sid_list,
                split="train",
                token_cache=token_cache,
                tool_id=tool_id,
                action_id=action_id,
                combo_id=combo_id,
            )
            if is_tool_only or is_combo_probe_attn or is_combo_probe_dual_attn:
                aligned = {
                    "gt_tool_xy": gt_tool_xy,
                    "gt_tool_vis": gt_tool_vis,
                    "gt_ct_xy": gt_ct_xy,
                    "gt_ct_vis": gt_ct_vis,
                }
            else:
                t_prime = outputs["pred_tool_xy"].shape[1]
                aligned = align_targets_to_tprime(gt_tool_xy, gt_tool_vis, gt_ct_xy, gt_ct_vis, t_prime=t_prime)
            losses = compute_batch_losses(
                outputs=outputs,
                gt_tool_xy=aligned["gt_tool_xy"],
                gt_tool_vis=aligned["gt_tool_vis"],
                gt_ct_xy=aligned["gt_ct_xy"],
                gt_ct_vis=aligned["gt_ct_vis"],
                combo_id=combo_id,
                gt_action_start_bin=gt_action_start_bin,
                gt_action_end_bin=gt_action_end_bin,
                action_range_mask=action_range_mask,
                gt_action_start_norm=gt_action_start_norm,
                gt_action_end_norm=gt_action_end_norm,
                combo_ce_weight=float(getattr(args, "combo_ce_weight", COMBO_CE_WEIGHT)),
                combo_tool_ce_weight=float(getattr(args, "combo_tool_ce_weight", 0.5)),
                combo_action_ce_weight=float(getattr(args, "combo_action_ce_weight", 0.5)),
                start_bin_loss_weight=RANGE_START_CE_WEIGHT,
                end_bin_loss_weight=RANGE_END_CE_WEIGHT,
                range_label_smoothing=RANGE_LABEL_SMOOTHING,
                range_positive_mode=str(getattr(args, "range_positive_mode", "span")),
                range_sensitivity_mode=str(getattr(args, "range_sensitivity_mode", "none")),
                range_sensitivity_cls_sigma=float(getattr(args, "range_sensitivity_cls_sigma", 0.5)),
                range_sensitivity_loc_sigma=float(getattr(args, "range_sensitivity_loc_sigma", 0.35)),
                range_sensitivity_min_weight=float(getattr(args, "range_sensitivity_min_weight", 0.25)),
                range_scale_invariant_mode=str(getattr(args, "range_scale_invariant_mode", "none")),
                range_scale_invariant_alpha=float(getattr(args, "range_scale_invariant_alpha", 0.8)),
                range_cls_target_mode=str(getattr(args, "range_cls_target_mode", "binary")),
                range_cls_varifocal_alpha=float(getattr(args, "range_cls_varifocal_alpha", 0.75)),
                range_cls_varifocal_weight=float(getattr(args, "range_cls_varifocal_weight", 1.0)),
                range_quality_mode=str(getattr(args, "range_quality_mode", "none")),
                range_quality_loss_weight=float(getattr(args, "range_quality_loss_weight", 0.0)),
                range_sola_aux_weight=float(getattr(args, "range_sola_aux_weight", 0.0)),
                range_sola_similarity_k=float(getattr(args, "range_sola_sim_k", 16.0)),
                range_focal_gamma=RANGE_FOCAL_GAMMA,
                range_neighbor_smoothing=RANGE_NEIGHBOR_SMOOTHING,
                range_boundary_aux_weight=float(getattr(args, "range_boundary_aux_weight", 0.0)),
                range_proposal_weight=float(getattr(args, "range_proposal_weight", 0.0)),
                range_combo_reweight_weights=range_combo_reweight_weights,
                range_combo_reweight_targets=str(getattr(args, "range_combo_reweight_targets", "")),
                combo_class_weights=combo_class_weights,
                start_bin_class_weights=start_bin_class_weights,
                end_bin_class_weights=end_bin_class_weights,
                kpt_loss_type=str(getattr(args, "kpt_loss_type", "l1l2")),
                traj_loss_pixel_scale=float(getattr(args, "traj_loss_pixel_scale", 256.0)),
                traj_huber_delta=float(getattr(args, "traj_huber_delta", 6.0)),
                traj_t0_loss_weight=float(getattr(args, "traj_t0_loss_weight", 1.0)),
                traj_loss_visible_only=bool(getattr(args, "traj_loss_visible_only", False)),
                traj_centroid_aux_weight=float(getattr(args, "traj_centroid_aux_weight", 0.0)),
                traj_offset_aux_weight=float(getattr(args, "traj_offset_aux_weight", 0.0)),
                trajectory_target=(
                    str(getattr(args, "trajectory_target", "tool"))
                    if bool(getattr(args, "trajectory_only", False))
                    else "both"
                ),
                trajectory_only=bool(getattr(args, "trajectory_only", False)),
                range_only=bool(getattr(args, "range_only", False)),
                combo_only=is_combo_only,
                tool_only=bool(getattr(args, "tool_only", False)),
                traj_heatmap_ce_weight=float(getattr(args, "traj_heatmap_ce_weight", 0.0)),
                traj_heatmap_sigma=float(getattr(args, "traj_heatmap_sigma", 2.45)),
                tool_pt_role=tool_pt_role,
                symmetry_swap_invariant_loss=bool(getattr(args, "symmetry_swap_invariant_loss", False)),
                traj_m2_stage_weights=train_traj_m2_stage_weights,
                traj_late_focal_enable=traj_late_focal_enable,
                traj_late_focal_stage_indices=getattr(args, "traj_late_focal_stage_indices", None),
                traj_late_focal_gamma=float(getattr(args, "traj_late_focal_gamma", 1.5)),
                traj_late_focal_clip_min=float(getattr(args, "traj_late_focal_clip_min", 0.30)),
                traj_late_focal_clip_max=float(getattr(args, "traj_late_focal_clip_max", 3.0)),
                traj_late_focal_ema_decay=float(getattr(args, "traj_late_focal_ema_decay", 0.95)),
                traj_late_focal_warmup_steps=int(getattr(args, "traj_late_focal_warmup_steps", 0)),
                traj_late_focal_state=traj_late_focal_state,
            )

        if bool(getattr(args, "debug_model_shapes", False)) and (not shape_debug_printed):
            meta = outputs.get("meta", {})
            if isinstance(meta, dict):
                shape_debug = meta.get("shape_debug", {})
                _print_model_shape_debug(log_write, shape_debug=shape_debug, epoch=epoch, step=step)
                shape_debug_printed = True

        if debug_this_step:
            _validate_train_output_contract(outputs, bsz=int(combo_id.shape[0]))
            for k in (
                "loss_total",
                "loss_tool_xy",
                "loss_ct_xy",
                "loss_combo_ce",
                "loss_start_bin_ce",
                "loss_end_bin_ce",
                "loss_range",
            ):
                _assert_finite(k, losses[k])

        grad_norm = torch.tensor(0.0, device=combo_id.device)
        if scaler.is_enabled():
            scaler.scale(losses["loss_total"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss_total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
            optimizer.step()
        if ema is not None:
            ema.update(model)

        if debug_this_step:
            grad_norm_v = float(torch.as_tensor(grad_norm).detach().cpu().item())
            if not np.isfinite(grad_norm_v):
                raise ValueError(f"Non-finite grad norm at epoch={epoch}, step={step}")
            grad_contract = _collect_grad_contract(model)
            enc_rg = int(grad_contract["vjepa_encoder"]["requires_grad_params"])
            if enc_rg != 0:
                raise ValueError(f"Frozen encoder violated: requires_grad_params={enc_rg}")
            refine_stats_dbg = _compute_refine_stats(outputs)

            report = {
                "epoch": int(epoch),
                "step": int(step),
                "sid": list(batch.get("sid", [])),
                "device": str(device),
                "amp_enabled": bool(amp_enabled),
                "amp_dtype": args.amp_dtype,
                "grad_norm": grad_norm_v,
                "loss_total": float(losses["loss_total"].item()),
                "loss_tool_xy": float(losses["loss_tool_xy"].item()),
                "loss_ct_xy": float(losses["loss_ct_xy"].item()),
                "loss_tool_centroid_aux": float(losses.get("loss_tool_centroid_aux", losses["loss_total"] * 0.0).item()),
                "loss_tool_offset_aux": float(losses.get("loss_tool_offset_aux", losses["loss_total"] * 0.0).item()),
                "loss_tool_pair_sep_aux": float(losses.get("loss_tool_pair_sep_aux", losses["loss_total"] * 0.0).item()),
                "loss_combo_ce": float(losses["loss_combo_ce"].item()),
                "loss_combo_tool_ce": float(losses.get("loss_combo_tool_ce", losses["loss_total"] * 0.0).item()),
                "loss_combo_action_ce": float(losses.get("loss_combo_action_ce", losses["loss_total"] * 0.0).item()),
                "loss_start_bin_ce": float(losses["loss_start_bin_ce"].item()),
                "loss_end_bin_ce": float(losses["loss_end_bin_ce"].item()),
                "loss_start_l1": float(losses["loss_start_bin_ce"].item()),
                "loss_end_l1": float(losses["loss_end_bin_ce"].item()),
                "loss_range_cls": float(losses.get("loss_range_cls", losses["loss_range"] * 0.0).item()),
                "loss_range_reg": float(losses.get("loss_range_reg", losses["loss_range"] * 0.0).item()),
                "loss_range_proposal": float(losses.get("loss_range_proposal", losses["loss_range"] * 0.0).item()),
                "loss_range": float(losses["loss_range"].item()),
                "refine_gate_mean": float(refine_stats_dbg["refine_gate_mean"]),
                "refine_delta_abs_mean": float(refine_stats_dbg["refine_delta_abs_mean"]),
                "refine_centroid_shift_abs_mean": float(refine_stats_dbg.get("refine_centroid_shift_abs_mean", 0.0)),
                "anchor_delta_centroid_abs_mean": float(refine_stats_dbg.get("anchor_delta_centroid_abs_mean", 0.0)),
                "anchor_alpha_mean": float(refine_stats_dbg.get("anchor_alpha_mean", 0.0)),
                "anchor_verify_scale_mean": float(refine_stats_dbg.get("anchor_verify_scale_mean", 1.0)),
                "anchor_verify_residual_mean": float(refine_stats_dbg.get("anchor_verify_residual_mean", 0.0)),
                "factor_anchor_gate_mean": float(refine_stats_dbg.get("factor_anchor_gate_mean", 0.0)),
                "factor_offset_gate_mean": float(refine_stats_dbg.get("factor_offset_gate_mean", 0.0)),
                "factor_delta_offset_abs_mean": float(refine_stats_dbg.get("factor_delta_offset_abs_mean", 0.0)),
                "input_video": (_tensor_stats(video) if video is not None else {"available": False}),
                "input_gt_tool_xy": _tensor_stats(gt_tool_xy),
                "input_gt_tool_vis": _tensor_stats(gt_tool_vis),
                "input_gt_ct_xy": _tensor_stats(gt_ct_xy),
                "input_gt_ct_vis": _tensor_stats(gt_ct_vis),
                "input_gt_action_start_bin": _tensor_stats(gt_action_start_bin.float()),
                "input_gt_action_end_bin": _tensor_stats(gt_action_end_bin.float()),
                "input_action_range_mask": _tensor_stats(action_range_mask),
                "aligned_gt_tool_vis": _tensor_stats(aligned["gt_tool_vis"]),
                "aligned_gt_ct_vis": _tensor_stats(aligned["gt_ct_vis"]),
                "output_pred_tool_xy": _tensor_stats(outputs["pred_tool_xy"]),
                "output_pred_ct_xy": _tensor_stats(outputs["pred_ct_xy"]),
                "output_combo_logits": _tensor_stats(outputs["combo_logits"]),
                "output_combo_tool_logits": _tensor_stats(outputs.get("combo_tool_logits", outputs["combo_logits"])),
                "output_combo_action_logits": _tensor_stats(outputs.get("combo_action_logits", outputs["combo_logits"])),
                "output_logits_start_bin": _tensor_stats(outputs["logits_start_bin"]),
                "output_logits_end_bin": _tensor_stats(outputs["logits_end_bin"]),
                "output_conf_tool": _tensor_stats(outputs["conf_tool"]),
                "output_conf_ct": _tensor_stats(outputs["conf_ct"]),
                "meta": outputs.get("meta", {}),
                "vjepa_loaded_sources": list(getattr(model.vjepa, "loaded_sources", [])),
                "vjepa_component_report": dict(getattr(model.vjepa, "component_report", {})),
                "grad_contract": grad_contract,
            }
            print(
                    f"[debug][train][epoch {epoch:03d}][step {step:05d}] "
                    f"loss={report['loss_total']:.5f} grad_norm={grad_norm_v:.5f} "
                    f"sep_aux={report['loss_tool_pair_sep_aux']:.5f} "
                    f"ref_gate={report['refine_gate_mean']:.4f} "
                    f"ref_dxy={report['refine_delta_abs_mean']:.5f} "
                    f"ref_cshift={report['refine_centroid_shift_abs_mean']:.5f} "
                    f"anchor_cshift={report.get('anchor_delta_centroid_abs_mean', 0.0):.5f} "
                    f"anchor_alpha={report.get('anchor_alpha_mean', 0.0):.4f} "
                    f"anchor_vfy={report.get('anchor_verify_scale_mean', 1.0):.4f} "
                    f"factor_ag={report.get('factor_anchor_gate_mean', 0.0):.4f} "
                    f"factor_og={report.get('factor_offset_gate_mean', 0.0):.4f} "
                    f"factor_off={report.get('factor_delta_offset_abs_mean', 0.0):.5f} "
                    f"video={'none' if video is None else tuple(video.shape)} t'={outputs['pred_tool_xy'].shape[1]}"
                )
            if args.debug_save_step_report:
                rp = _write_train_step_debug_report(report=report, out_dir=out_dir, epoch=epoch, step=step)
                print(f"[debug][train] step report: {rp}")

        with torch.no_grad():
            bsz = int(combo_id.shape[0])
            if is_combo_probe_attn:
                pred_combo = outputs["combo_logits"].argmax(dim=-1)
                combo_correct = (pred_combo == combo_id).float().sum().item()
                tool_correct = 0.0
                action_correct = 0.0
            else:
                tool_id, action_id = _derive_tool_action_id_from_combo_id(combo_id)
                pred_tool_cls = outputs["combo_tool_logits"].argmax(dim=-1)
                tool_correct = (pred_tool_cls == tool_id).float().sum().item()
                if is_tool_only:
                    combo_correct = 0.0
                    action_correct = 0.0
                else:
                    pred_combo = outputs["combo_logits"].argmax(dim=-1)
                    combo_correct = (pred_combo == combo_id).float().sum().item()
                    pred_action_cls = outputs["combo_action_logits"].argmax(dim=-1)
                    action_correct = (pred_action_cls == action_id).float().sum().item()
            if is_tool_only:
                combo_correct = 0.0
                action_correct = 0.0
            pred_start_norm = outputs.get("pred_action_start_norm", gt_action_start_norm).reshape(-1)
            pred_end_norm = outputs.get("pred_action_end_norm", gt_action_end_norm).reshape(-1)
            refine_stats = _compute_refine_stats(outputs)
            timing_stats = {}
            meta = outputs.get("meta", {})
            if isinstance(meta, dict):
                tms = meta.get("timing_ms", {})
                if isinstance(tms, dict):
                    for k, v in tms.items():
                        try:
                            timing_stats[str(k)] = float(v)
                        except Exception:
                            continue
            if is_tool_only or is_combo_probe_attn or is_combo_probe_dual_attn:
                track_stats = _zero_track_metrics_per_sample(bsz=bsz, device=combo_id.device)
            else:
                track_stats = _compute_track_metrics_per_sample(
                    pred_tool_xy=outputs["pred_tool_xy"],
                    pred_ct_xy=outputs["pred_ct_xy"],
                    pred_tool_conf=outputs["conf_tool"],
                    pred_ct_conf=outputs["conf_ct"],
                    gt_tool_xy=aligned["gt_tool_xy"],
                    gt_ct_xy=aligned["gt_ct_xy"],
                    gt_tool_vis=aligned["gt_tool_vis"],
                    gt_ct_vis=aligned["gt_ct_vis"],
                    image_size=int(args.input_size),
                    trajectory_target=(str(getattr(args, "trajectory_target", "tool")) if is_traj_only else "both"),
                )
            range_stats = _compute_range_metrics_per_sample(
                pred_action_start_norm=pred_start_norm,
                pred_action_end_norm=pred_end_norm,
                gt_action_start_norm=gt_action_start_norm,
                gt_action_end_norm=gt_action_end_norm,
                action_range_mask=action_range_mask,
                sampled_abs_frames=batch.get("sampled_abs_frames", None),
            )

            meter["n"] += bsz
            meter["loss_total"] += losses["loss_total"].item() * bsz
            meter["loss_tool_xy"] += losses["loss_tool_xy"].item() * bsz
            meter["loss_ct_xy"] += losses["loss_ct_xy"].item() * bsz
            _accumulate_traj_stage_losses(meter, losses, bsz)
            _accumulate_traj_late_focal_metrics(meter, losses, bsz)
            meter["loss_tool_centroid_aux"] += losses.get("loss_tool_centroid_aux", losses["loss_total"] * 0.0).item() * bsz
            meter["loss_tool_offset_aux"] += losses.get("loss_tool_offset_aux", losses["loss_total"] * 0.0).item() * bsz
            meter["loss_tool_pair_sep_aux"] += losses.get("loss_tool_pair_sep_aux", losses["loss_total"] * 0.0).item() * bsz
            meter["loss_tool_heatmap_ce"] += losses.get("loss_tool_heatmap_ce", losses["loss_total"] * 0.0).item() * bsz
            meter["loss_ct_heatmap_ce"] += losses.get("loss_ct_heatmap_ce", losses["loss_total"] * 0.0).item() * bsz
            meter["loss_combo_ce"] += losses["loss_combo_ce"].item() * bsz
            meter["loss_combo_tool_ce"] += losses.get("loss_combo_tool_ce", losses["loss_total"] * 0.0).item() * bsz
            meter["loss_combo_action_ce"] += losses.get("loss_combo_action_ce", losses["loss_total"] * 0.0).item() * bsz
            meter["loss_start_bin_ce"] += losses["loss_start_bin_ce"].item() * bsz
            meter["loss_end_bin_ce"] += losses["loss_end_bin_ce"].item() * bsz
            meter["loss_range_cls"] += losses.get("loss_range_cls", losses["loss_range"] * 0.0).item() * bsz
            meter["loss_range_reg"] += losses.get("loss_range_reg", losses["loss_range"] * 0.0).item() * bsz
            meter["loss_range_proposal"] += losses.get("loss_range_proposal", losses["loss_range"] * 0.0).item() * bsz
            meter["loss_range_quality"] += losses.get("loss_range_quality", losses["loss_range"] * 0.0).item() * bsz
            meter["loss_range_sola"] += losses.get("loss_range_sola", losses["loss_range"] * 0.0).item() * bsz
            meter["loss_range"] += losses["loss_range"].item() * bsz
            meter["combo_correct"] += combo_correct
            meter["tool_correct"] += tool_correct
            meter["action_correct"] += action_correct
            meter["traj_ade_num"] += float(track_stats["traj_ade_num"].item())
            meter["traj_ade_den"] += float(track_stats["traj_ade_den"].item())
            meter["traj_fde_num"] += float(track_stats["traj_fde_num"].item())
            meter["traj_fde_den"] += float(track_stats["traj_fde_den"].item())
            meter["traj_pck_den"] += float(track_stats["traj_pck_den"].item())
            meter["traj_pck_2_num"] += float(track_stats["traj_pck_2_num"].item())
            meter["traj_pck_4_num"] += float(track_stats["traj_pck_4_num"].item())
            meter["traj_pck_8_num"] += float(track_stats["traj_pck_8_num"].item())
            meter["traj_pck_16_num"] += float(track_stats["traj_pck_16_num"].item())
            meter["traj_survival_num"] += float(track_stats["traj_survival_num"].item())
            meter["traj_survival_den"] += float(track_stats["traj_survival_den"].item())
            meter["traj_aj_sum"] += float(track_stats["traj_aj_sum"].item())
            meter["traj_aj_den"] += float(track_stats["traj_aj_den"].item())
            meter["range_valid_den"] += float(range_stats["range_valid_den"].item())
            meter["range_count"] += float(range_stats["range_valid_den"].item())
            meter["range_tiou_sum"] += float(range_stats["range_tiou_sum"].item())
            meter["range_r03_sum"] += float(range_stats["range_r03_sum"].item())
            meter["range_r05_sum"] += float(range_stats["range_r05_sum"].item())
            meter["range_r07_sum"] += float(range_stats["range_r07_sum"].item())
            meter["range_start_mae_norm_sum"] += float(range_stats["range_start_mae_norm_sum"].item())
            meter["range_end_mae_norm_sum"] += float(range_stats["range_end_mae_norm_sum"].item())
            meter["range_center_mae_norm_sum"] += float(range_stats["range_center_mae_norm_sum"].item())
            meter["range_duration_mae_norm_sum"] += float(range_stats["range_duration_mae_norm_sum"].item())
            meter["range_start_mae_abs_sum"] += float(range_stats["range_start_mae_abs_sum"].item())
            meter["range_end_mae_abs_sum"] += float(range_stats["range_end_mae_abs_sum"].item())
            meter["refine_gate_mean"] += float(refine_stats["refine_gate_mean"]) * bsz
            meter["refine_delta_abs_mean"] += float(refine_stats["refine_delta_abs_mean"]) * bsz
            meter["refine_centroid_shift_abs_mean"] += float(refine_stats.get("refine_centroid_shift_abs_mean", 0.0)) * bsz
            meter["anchor_delta_centroid_abs_mean"] += float(refine_stats.get("anchor_delta_centroid_abs_mean", 0.0)) * bsz
            meter["anchor_alpha_mean"] += float(refine_stats.get("anchor_alpha_mean", 0.0)) * bsz
            meter["anchor_verify_scale_mean"] += float(refine_stats.get("anchor_verify_scale_mean", 0.0)) * bsz
            meter["anchor_verify_residual_mean"] += float(refine_stats.get("anchor_verify_residual_mean", 0.0)) * bsz
            meter["stage0_centroid_noise_abs_mean"] += float(
                refine_stats.get("stage0_centroid_noise_abs_mean", 0.0)
            ) * bsz
            meter["factor_anchor_gate_mean"] += float(refine_stats.get("factor_anchor_gate_mean", 0.0)) * bsz
            meter["factor_offset_gate_mean"] += float(refine_stats.get("factor_offset_gate_mean", 0.0)) * bsz
            meter["factor_delta_offset_abs_mean"] += float(refine_stats.get("factor_delta_offset_abs_mean", 0.0)) * bsz
            meter["corr_entropy_mean"] += float(refine_stats.get("corr_entropy_mean", 0.0)) * bsz
            meter["corr_top1_minus_top2_mean"] += float(refine_stats.get("corr_top1_minus_top2_mean", 0.0)) * bsz
            meter["context_entropy_mean"] += float(refine_stats.get("context_entropy_mean", 0.0)) * bsz
            meter["context_top1_minus_top2_mean"] += float(refine_stats.get("context_top1_minus_top2_mean", 0.0)) * bsz
            meter["context_offset_abs_mean"] += float(refine_stats.get("context_offset_abs_mean", 0.0)) * bsz
            meter["context_gate_mean"] += float(refine_stats.get("context_gate_mean", 0.0)) * bsz
            meter["belief_gate_mean"] += float(refine_stats.get("belief_gate_mean", 0.0)) * bsz
            meter["joint_gate_mean"] += float(refine_stats.get("joint_gate_mean", 0.0)) * bsz
            meter["template_entropy_mean"] += float(refine_stats.get("template_entropy_mean", 0.0)) * bsz
            meter["template_top1_minus_top2_mean"] += float(refine_stats.get("template_top1_minus_top2_mean", 0.0)) * bsz
            meter["template_gate_mean"] += float(refine_stats.get("template_gate_mean", 0.0)) * bsz
            meter["template_offset_abs_mean"] += float(refine_stats.get("template_offset_abs_mean", 0.0)) * bsz
            meter["template_delta_abs_mean"] += float(refine_stats.get("template_delta_abs_mean", 0.0)) * bsz
            for tk, tv in timing_stats.items():
                meter[f"timing_{tk}"] += float(tv) * bsz
            meter["conf_tool_mean"] += outputs["conf_tool"].mean().item() * bsz
            meter["conf_ct_mean"] += outputs["conf_ct"].mean().item() * bsz

            log_meter["n"] += bsz
            log_meter["loss_total"] += losses["loss_total"].item() * bsz
            log_meter["loss_tool_xy"] += losses["loss_tool_xy"].item() * bsz
            log_meter["loss_ct_xy"] += losses["loss_ct_xy"].item() * bsz
            _accumulate_traj_stage_losses(log_meter, losses, bsz)
            _accumulate_traj_late_focal_metrics(log_meter, losses, bsz)
            log_meter["loss_tool_centroid_aux"] += losses.get("loss_tool_centroid_aux", losses["loss_total"] * 0.0).item() * bsz
            log_meter["loss_tool_offset_aux"] += losses.get("loss_tool_offset_aux", losses["loss_total"] * 0.0).item() * bsz
            log_meter["loss_tool_pair_sep_aux"] += losses.get("loss_tool_pair_sep_aux", losses["loss_total"] * 0.0).item() * bsz
            log_meter["loss_tool_heatmap_ce"] += losses.get("loss_tool_heatmap_ce", losses["loss_total"] * 0.0).item() * bsz
            log_meter["loss_ct_heatmap_ce"] += losses.get("loss_ct_heatmap_ce", losses["loss_total"] * 0.0).item() * bsz
            log_meter["loss_combo_ce"] += losses["loss_combo_ce"].item() * bsz
            log_meter["loss_combo_tool_ce"] += losses.get("loss_combo_tool_ce", losses["loss_total"] * 0.0).item() * bsz
            log_meter["loss_combo_action_ce"] += losses.get("loss_combo_action_ce", losses["loss_total"] * 0.0).item() * bsz
            log_meter["loss_start_bin_ce"] += losses["loss_start_bin_ce"].item() * bsz
            log_meter["loss_end_bin_ce"] += losses["loss_end_bin_ce"].item() * bsz
            log_meter["loss_range_cls"] += losses.get("loss_range_cls", losses["loss_range"] * 0.0).item() * bsz
            log_meter["loss_range_reg"] += losses.get("loss_range_reg", losses["loss_range"] * 0.0).item() * bsz
            log_meter["loss_range_proposal"] += losses.get("loss_range_proposal", losses["loss_range"] * 0.0).item() * bsz
            log_meter["loss_range_quality"] += losses.get("loss_range_quality", losses["loss_range"] * 0.0).item() * bsz
            log_meter["loss_range_sola"] += losses.get("loss_range_sola", losses["loss_range"] * 0.0).item() * bsz
            log_meter["loss_range"] += losses["loss_range"].item() * bsz
            log_meter["combo_correct"] += combo_correct
            log_meter["tool_correct"] += tool_correct
            log_meter["action_correct"] += action_correct
            log_meter["traj_ade_num"] += float(track_stats["traj_ade_num"].item())
            log_meter["traj_ade_den"] += float(track_stats["traj_ade_den"].item())
            log_meter["traj_fde_num"] += float(track_stats["traj_fde_num"].item())
            log_meter["traj_fde_den"] += float(track_stats["traj_fde_den"].item())
            log_meter["traj_pck_den"] += float(track_stats["traj_pck_den"].item())
            log_meter["traj_pck_2_num"] += float(track_stats["traj_pck_2_num"].item())
            log_meter["traj_pck_4_num"] += float(track_stats["traj_pck_4_num"].item())
            log_meter["traj_pck_8_num"] += float(track_stats["traj_pck_8_num"].item())
            log_meter["traj_pck_16_num"] += float(track_stats["traj_pck_16_num"].item())
            log_meter["traj_survival_num"] += float(track_stats["traj_survival_num"].item())
            log_meter["traj_survival_den"] += float(track_stats["traj_survival_den"].item())
            log_meter["traj_aj_sum"] += float(track_stats["traj_aj_sum"].item())
            log_meter["traj_aj_den"] += float(track_stats["traj_aj_den"].item())
            log_meter["range_valid_den"] += float(range_stats["range_valid_den"].item())
            log_meter["range_tiou_sum"] += float(range_stats["range_tiou_sum"].item())
            log_meter["range_r03_sum"] += float(range_stats["range_r03_sum"].item())
            log_meter["range_r05_sum"] += float(range_stats["range_r05_sum"].item())
            log_meter["range_r07_sum"] += float(range_stats["range_r07_sum"].item())
            log_meter["range_start_mae_norm_sum"] += float(range_stats["range_start_mae_norm_sum"].item())
            log_meter["range_end_mae_norm_sum"] += float(range_stats["range_end_mae_norm_sum"].item())
            log_meter["range_center_mae_norm_sum"] += float(range_stats["range_center_mae_norm_sum"].item())
            log_meter["range_duration_mae_norm_sum"] += float(range_stats["range_duration_mae_norm_sum"].item())
            log_meter["range_start_mae_abs_sum"] += float(range_stats["range_start_mae_abs_sum"].item())
            log_meter["range_end_mae_abs_sum"] += float(range_stats["range_end_mae_abs_sum"].item())
            log_meter["refine_gate_mean"] += float(refine_stats["refine_gate_mean"]) * bsz
            log_meter["refine_delta_abs_mean"] += float(refine_stats["refine_delta_abs_mean"]) * bsz
            log_meter["refine_centroid_shift_abs_mean"] += float(refine_stats.get("refine_centroid_shift_abs_mean", 0.0)) * bsz
            log_meter["anchor_delta_centroid_abs_mean"] += float(refine_stats.get("anchor_delta_centroid_abs_mean", 0.0)) * bsz
            log_meter["anchor_alpha_mean"] += float(refine_stats.get("anchor_alpha_mean", 0.0)) * bsz
            log_meter["anchor_verify_scale_mean"] += float(refine_stats.get("anchor_verify_scale_mean", 0.0)) * bsz
            log_meter["anchor_verify_residual_mean"] += float(refine_stats.get("anchor_verify_residual_mean", 0.0)) * bsz
            log_meter["stage0_centroid_noise_abs_mean"] += float(
                refine_stats.get("stage0_centroid_noise_abs_mean", 0.0)
            ) * bsz
            log_meter["factor_anchor_gate_mean"] += float(refine_stats.get("factor_anchor_gate_mean", 0.0)) * bsz
            log_meter["factor_offset_gate_mean"] += float(refine_stats.get("factor_offset_gate_mean", 0.0)) * bsz
            log_meter["factor_delta_offset_abs_mean"] += float(refine_stats.get("factor_delta_offset_abs_mean", 0.0)) * bsz
            log_meter["corr_entropy_mean"] += float(refine_stats.get("corr_entropy_mean", 0.0)) * bsz
            log_meter["corr_top1_minus_top2_mean"] += float(refine_stats.get("corr_top1_minus_top2_mean", 0.0)) * bsz
            log_meter["context_entropy_mean"] += float(refine_stats.get("context_entropy_mean", 0.0)) * bsz
            log_meter["context_top1_minus_top2_mean"] += float(refine_stats.get("context_top1_minus_top2_mean", 0.0)) * bsz
            log_meter["context_offset_abs_mean"] += float(refine_stats.get("context_offset_abs_mean", 0.0)) * bsz
            log_meter["context_gate_mean"] += float(refine_stats.get("context_gate_mean", 0.0)) * bsz
            log_meter["belief_gate_mean"] += float(refine_stats.get("belief_gate_mean", 0.0)) * bsz
            log_meter["joint_gate_mean"] += float(refine_stats.get("joint_gate_mean", 0.0)) * bsz
            log_meter["template_entropy_mean"] += float(refine_stats.get("template_entropy_mean", 0.0)) * bsz
            log_meter["template_top1_minus_top2_mean"] += float(refine_stats.get("template_top1_minus_top2_mean", 0.0)) * bsz
            log_meter["template_gate_mean"] += float(refine_stats.get("template_gate_mean", 0.0)) * bsz
            log_meter["template_offset_abs_mean"] += float(refine_stats.get("template_offset_abs_mean", 0.0)) * bsz
            log_meter["template_delta_abs_mean"] += float(refine_stats.get("template_delta_abs_mean", 0.0)) * bsz
            for tk, tv in timing_stats.items():
                log_meter[f"timing_{tk}"] += float(tv) * bsz

        global_step_now = int(global_step_start + step)
        if (args.log_interval > 0) and (global_step_now % args.log_interval == 0):
            log_metrics = _finalize_meter(log_meter)
            train_total = log_metrics["loss_total"]
            train_tool = log_metrics["loss_tool_xy"]
            train_ct = log_metrics["loss_ct_xy"]
            train_tool_centroid_aux = log_metrics.get("loss_tool_centroid_aux", 0.0)
            train_tool_offset_aux = log_metrics.get("loss_tool_offset_aux", 0.0)
            train_tool_pair_sep_aux = log_metrics.get("loss_tool_pair_sep_aux", 0.0)
            train_tool_heatmap_ce = log_metrics.get("loss_tool_heatmap_ce", 0.0)
            train_ct_heatmap_ce = log_metrics.get("loss_ct_heatmap_ce", 0.0)
            train_combo_ce = log_metrics["loss_combo_ce"]
            train_combo_tool_ce = log_metrics.get("loss_combo_tool_ce", 0.0)
            train_combo_action_ce = log_metrics.get("loss_combo_action_ce", 0.0)
            train_start_l1 = log_metrics["loss_start_l1"]
            train_end_l1 = log_metrics["loss_end_l1"]
            train_range_cls = log_metrics["loss_range_cls"]
            train_range_reg = log_metrics["loss_range_reg"]
            train_range_prop = log_metrics.get("loss_range_proposal", 0.0)
            train_range_sola = log_metrics.get("loss_range_sola", 0.0)
            train_range = log_metrics["loss_range"]
            combo_acc_now = float(log_metrics["combo_acc"])
            tool_acc_now = float(log_metrics.get("tool_acc", 0.0))
            action_acc_now = float(log_metrics.get("action_acc", 0.0))
            traj_ade_now = float(log_metrics["traj_ade_px"])
            traj_aj_now = float(log_metrics.get("traj_aj", 0.0))
            traj_pck8_now = float(log_metrics["traj_pck_8"])
            range_tiou_now = float(log_metrics["range_tiou"])
            range_r05_now = float(log_metrics["range_r05"])
            ref_gate_now = float(log_metrics.get("refine_gate_mean", 0.0))
            ref_dxy_now = float(log_metrics.get("refine_delta_abs_mean", 0.0))
            ref_cshift_now = float(log_metrics.get("refine_centroid_shift_abs_mean", 0.0))
            anchor_cshift_now = float(log_metrics.get("anchor_delta_centroid_abs_mean", 0.0))
            stage0_noise_now = float(log_metrics.get("stage0_centroid_noise_abs_mean", 0.0))
            anchor_alpha_now = float(log_metrics.get("anchor_alpha_mean", 0.0))
            anchor_verify_scale_now = float(log_metrics.get("anchor_verify_scale_mean", 1.0))
            anchor_verify_resid_now = float(log_metrics.get("anchor_verify_residual_mean", 0.0))
            factor_anchor_gate_now = float(log_metrics.get("factor_anchor_gate_mean", 0.0))
            factor_offset_gate_now = float(log_metrics.get("factor_offset_gate_mean", 0.0))
            factor_delta_offset_now = float(log_metrics.get("factor_delta_offset_abs_mean", 0.0))
            focal_ema_now = float(log_metrics.get("traj_late_focal_ema_error", 0.0))
            focal_w_now = float(log_metrics.get("traj_late_focal_weight_mean", 0.0))
            focal_w_p90_now = float(log_metrics.get("traj_late_focal_weight_p90", 0.0))
            t_model_ms = float(log_metrics.get("timing_model_total_ms", 0.0))
            t_trunk_ms = float(log_metrics.get("timing_shared_trunk_ms", 0.0))
            t_conv_ms = float(log_metrics.get("timing_conv_refine_ms", 0.0))
            t_heads_ms = float(log_metrics.get("timing_spatial_heads_decode_ms", 0.0))
            t_up_ms = float(log_metrics.get("timing_temporal_upsample_ms", 0.0))
            t_feat_ms = float(log_metrics.get("timing_local_feat_weighted_ms", 0.0))
            t_ref_t_ms = float(log_metrics.get("timing_refiner_tool_ms", 0.0))
            t_ref_c_ms = float(log_metrics.get("timing_refiner_contact_ms", 0.0))
            t_combo_ms = float(log_metrics.get("timing_combo_head_ms", 0.0))
            t_range_ms = float(log_metrics.get("timing_range_head_ms", 0.0))
            lr_now = float(optimizer.param_groups[0]["lr"])
            if is_traj_only:
                log_write(
                    f"[train][epoch {epoch:03d}][step {step:05d}][gstep {global_step_now:07d}] "
                    f"total={train_total:.5f} "
                    f"tool={train_tool:.5f} "
                    f"{_format_traj_stage_summary(log_metrics)} "
                    f"ctr_aux={train_tool_centroid_aux:.5f} "
                    f"off_aux={train_tool_offset_aux:.5f} "
                    f"sep_aux={train_tool_pair_sep_aux:.5f} "
                    f"traj_ade_px={traj_ade_now:.3f} "
                    f"ref_cshift={ref_cshift_now:.5f} "
                    f"anchor_cshift={anchor_cshift_now:.5f} "
                    f"st0_noise={stage0_noise_now:.5f} "
                    f"anchor_alpha={anchor_alpha_now:.4f} "
                    f"anchor_vfy={anchor_verify_scale_now:.4f} "
                    f"factor_ag={factor_anchor_gate_now:.4f} "
                    f"factor_og={factor_offset_gate_now:.4f} "
                    f"factor_off={factor_delta_offset_now:.5f} "
                    f"focal_w={focal_w_now:.4f} "
                    f"focal_p90={focal_w_p90_now:.4f} "
                    f"focal_ema={focal_ema_now:.4f} "
                    f"lr={lr_now:.2e} "
                    f"ms(total={t_model_ms:.2f},trunk={t_trunk_ms:.2f},conv={t_conv_ms:.2f},"
                    f"heads={t_heads_ms:.2f},up={t_up_ms:.2f},feat={t_feat_ms:.2f},"
                    f"ref_t={t_ref_t_ms:.2f},ref_c={t_ref_c_ms:.2f})"
                )
            elif is_tool_only:
                log_write(
                    f"[train][epoch {epoch:03d}][step {step:05d}][gstep {global_step_now:07d}] "
                    f"total={train_total:.5f} "
                    f"tool_ce={train_combo_tool_ce:.5f} "
                    f"tool_acc={tool_acc_now:.4f} "
                    f"lr={lr_now:.2e}"
                )
            elif is_combo_probe_attn:
                log_write(
                    f"[train][epoch {epoch:03d}][step {step:05d}][gstep {global_step_now:07d}] "
                    f"total={train_total:.5f} "
                    f"combo_ce={train_combo_ce:.5f} "
                    f"combo_acc={combo_acc_now:.4f} "
                    f"lr={lr_now:.2e}"
                )
            elif is_combo_only:
                log_write(
                    f"[train][epoch {epoch:03d}][step {step:05d}][gstep {global_step_now:07d}] "
                    f"total={train_total:.5f} "
                    f"combo_ce={train_combo_ce:.5f} "
                    f"tool_ce={train_combo_tool_ce:.5f} "
                    f"action_ce={train_combo_action_ce:.5f} "
                    f"combo_acc={combo_acc_now:.4f} "
                    f"tool_acc={tool_acc_now:.4f} "
                    f"action_acc={action_acc_now:.4f} "
                    f"lr={lr_now:.2e}"
                )
            else:
                log_write(
                    f"[train][epoch {epoch:03d}][step {step:05d}][gstep {global_step_now:07d}] "
                    f"total={train_total:.5f} "
                    f"tool={train_tool:.5f} "
                    f"ct={train_ct:.5f} "
                    f"combo_ce={train_combo_ce:.5f} "
                    f"start_l1={train_start_l1:.5f} "
                    f"end_l1={train_end_l1:.5f} "
                    f"range={train_range:.5f} "
                    f"range_cls={train_range_cls:.5f} "
                    f"range_reg={train_range_reg:.5f} "
                    f"range_prop={train_range_prop:.5f} "
                    f"range_sola={train_range_sola:.5f} "
                    f"combo_acc={combo_acc_now:.4f} "
                    f"traj_ade_px={traj_ade_now:.3f} "
                    f"traj_aj={traj_aj_now:.4f} "
                    f"traj_pck8={traj_pck8_now:.4f} "
                    f"range_tiou={range_tiou_now:.4f} "
                    f"range_r05={range_r05_now:.4f} "
                    f"ref_gate={ref_gate_now:.4f} "
                    f"ref_dxy={ref_dxy_now:.5f} "
                    f"lr={lr_now:.2e} "
                    f"ms(total={t_model_ms:.2f},trunk={t_trunk_ms:.2f},conv={t_conv_ms:.2f},"
                    f"heads={t_heads_ms:.2f},up={t_up_ms:.2f},feat={t_feat_ms:.2f},"
                    f"ref_t={t_ref_t_ms:.2f},ref_c={t_ref_c_ms:.2f},combo={t_combo_ms:.2f},range={t_range_ms:.2f})"
                )
            if pbar is not None:
                if is_traj_only:
                    pbar.set_postfix(
                        loss=f"{train_total:.4f}",
                        tool=f"{train_tool:.4f}",
                        ctr=f"{train_tool_centroid_aux:.4f}",
                        off=f"{train_tool_offset_aux:.4f}",
                        sep=f"{train_tool_pair_sep_aux:.4f}",
                        ade=f"{traj_ade_now:.2f}",
                        fw=f"{focal_w_now:.2f}",
                        ms=f"{t_model_ms:.1f}",
                        lr=f"{lr_now:.2e}",
                    )
                elif is_tool_only:
                    pbar.set_postfix(
                        loss=f"{train_total:.4f}",
                        tce=f"{train_combo_tool_ce:.4f}",
                        ta=f"{tool_acc_now:.3f}",
                        ms=f"{t_model_ms:.1f}",
                        lr=f"{lr_now:.2e}",
                    )
                elif is_combo_probe_attn:
                    pbar.set_postfix(
                        loss=f"{train_total:.4f}",
                        ce=f"{train_combo_ce:.4f}",
                        ca=f"{combo_acc_now:.3f}",
                        ms=f"{t_model_ms:.1f}",
                        lr=f"{lr_now:.2e}",
                    )
                elif is_combo_only:
                    pbar.set_postfix(
                        loss=f"{train_total:.4f}",
                        ce=f"{train_combo_ce:.4f}",
                        tce=f"{train_combo_tool_ce:.4f}",
                        ace=f"{train_combo_action_ce:.4f}",
                        ca=f"{combo_acc_now:.3f}",
                        ta=f"{tool_acc_now:.3f}",
                        aa=f"{action_acc_now:.3f}",
                        ms=f"{t_model_ms:.1f}",
                        lr=f"{lr_now:.2e}",
                    )
                else:
                    pbar.set_postfix(
                        loss=f"{train_total:.4f}",
                        tool=f"{train_tool:.4f}",
                        ct=f"{train_ct:.4f}",
                        ce=f"{train_combo_ce:.4f}",
                        sl=f"{train_start_l1:.4f}",
                        el=f"{train_end_l1:.4f}",
                        rng=f"{train_range:.4f}",
                        rc=f"{train_range_cls:.4f}",
                        rr=f"{train_range_reg:.4f}",
                        rp=f"{train_range_prop:.4f}",
                        rs=f"{train_range_sola:.4f}",
                        ca=f"{combo_acc_now:.3f}",
                        ade=f"{traj_ade_now:.2f}",
                        aj=f"{traj_aj_now:.3f}",
                        p8=f"{traj_pck8_now:.3f}",
                        tiou=f"{range_tiou_now:.3f}",
                        r05=f"{range_r05_now:.3f}",
                        rfg=f"{ref_gate_now:.3f}",
                        rd=f"{ref_dxy_now:.4f}",
                        ms=f"{t_model_ms:.1f}",
                        lr=f"{lr_now:.2e}",
                    )
            if writer is not None:
                writer.add_scalar("train/step_loss_total", train_total, global_step_now)
                writer.add_scalar("train/step_loss_tool_xy", train_tool, global_step_now)
                writer.add_scalar("train/step_loss_ct_xy", train_ct, global_step_now)
                writer.add_scalar("train/step_loss_tool_centroid_aux", train_tool_centroid_aux, global_step_now)
                writer.add_scalar("train/step_loss_tool_offset_aux", train_tool_offset_aux, global_step_now)
                writer.add_scalar("train/step_loss_tool_pair_sep_aux", train_tool_pair_sep_aux, global_step_now)
                writer.add_scalar("train/step_loss_tool_heatmap_ce", train_tool_heatmap_ce, global_step_now)
                writer.add_scalar("train/step_loss_ct_heatmap_ce", train_ct_heatmap_ce, global_step_now)
                if is_tool_only:
                    writer.add_scalar("train/step_loss_combo_tool_ce", train_combo_tool_ce, global_step_now)
                    writer.add_scalar("train/step_tool_acc", tool_acc_now, global_step_now)
                elif is_combo_probe_attn:
                    writer.add_scalar("train/step_loss_combo_ce", train_combo_ce, global_step_now)
                    writer.add_scalar("train/step_combo_acc", combo_acc_now, global_step_now)
                elif not is_traj_only:
                    writer.add_scalar("train/step_loss_combo_ce", train_combo_ce, global_step_now)
                    writer.add_scalar("train/step_loss_combo_tool_ce", train_combo_tool_ce, global_step_now)
                    writer.add_scalar("train/step_loss_combo_action_ce", train_combo_action_ce, global_step_now)
                    writer.add_scalar("train/step_tool_acc", tool_acc_now, global_step_now)
                    writer.add_scalar("train/step_action_acc", action_acc_now, global_step_now)
                    writer.add_scalar("train/step_loss_start_l1", train_start_l1, global_step_now)
                    writer.add_scalar("train/step_loss_end_l1", train_end_l1, global_step_now)
                    writer.add_scalar("train/step_loss_range", train_range, global_step_now)
                    writer.add_scalar("train/step_loss_range_cls", train_range_cls, global_step_now)
                    writer.add_scalar("train/step_loss_range_reg", train_range_reg, global_step_now)
                    writer.add_scalar("train/step_loss_range_proposal", train_range_prop, global_step_now)
                    writer.add_scalar("train/step_loss_range_sola", train_range_sola, global_step_now)
                    writer.add_scalar("train/step_combo_acc", combo_acc_now, global_step_now)
                writer.add_scalar("train/step_traj_ade_px", traj_ade_now, global_step_now)
                writer.add_scalar("train/step_traj_aj", traj_aj_now, global_step_now)
                writer.add_scalar("train/step_traj_pck_8", traj_pck8_now, global_step_now)
                if (not is_traj_only) and (not is_tool_only):
                    writer.add_scalar("train/step_range_tiou", range_tiou_now, global_step_now)
                writer.add_scalar("train/step_range_r05", range_r05_now, global_step_now)
                writer.add_scalar("train/step_refine_gate_mean", ref_gate_now, global_step_now)
                writer.add_scalar("train/step_refine_delta_abs_mean", ref_dxy_now, global_step_now)
                writer.add_scalar("train/step_refine_centroid_shift_abs_mean", ref_cshift_now, global_step_now)
                writer.add_scalar("train/step_anchor_delta_centroid_abs_mean", anchor_cshift_now, global_step_now)
                writer.add_scalar("train/step_anchor_alpha_mean", anchor_alpha_now, global_step_now)
                writer.add_scalar("train/step_anchor_verify_scale_mean", anchor_verify_scale_now, global_step_now)
                writer.add_scalar("train/step_anchor_verify_residual_mean", anchor_verify_resid_now, global_step_now)
                writer.add_scalar("train/step_factor_anchor_gate_mean", factor_anchor_gate_now, global_step_now)
                writer.add_scalar("train/step_factor_offset_gate_mean", factor_offset_gate_now, global_step_now)
                writer.add_scalar("train/step_factor_delta_offset_abs_mean", factor_delta_offset_now, global_step_now)
                if traj_late_focal_enable:
                    for key in _TRAJ_LATE_FOCAL_METRIC_KEYS:
                        writer.add_scalar(f"train/step_{key}", log_metrics.get(key, 0.0), global_step_now)
                writer.add_scalar("train/step_timing_model_total_ms", t_model_ms, global_step_now)
                writer.add_scalar("train/step_timing_shared_trunk_ms", t_trunk_ms, global_step_now)
                writer.add_scalar("train/step_timing_conv_refine_ms", t_conv_ms, global_step_now)
                writer.add_scalar("train/step_timing_spatial_heads_decode_ms", t_heads_ms, global_step_now)
                writer.add_scalar("train/step_timing_temporal_upsample_ms", t_up_ms, global_step_now)
                writer.add_scalar("train/step_timing_local_feat_weighted_ms", t_feat_ms, global_step_now)
                writer.add_scalar("train/step_timing_refiner_tool_ms", t_ref_t_ms, global_step_now)
                writer.add_scalar("train/step_timing_refiner_contact_ms", t_ref_c_ms, global_step_now)
                if (not is_traj_only) and (not is_tool_only):
                    writer.add_scalar("train/step_timing_combo_head_ms", t_combo_ms, global_step_now)
                    writer.add_scalar("train/step_timing_range_head_ms", t_range_ms, global_step_now)
            log_meter = defaultdict(float)

    if pbar is not None:
        pbar.close()

    return _finalize_meter(meter), int(global_step_start + last_step)


@torch.no_grad()
def evaluate(
    model: KeypointComboNet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    combo_id_to_name: Dict[int, str],
    epoch: Optional[int] = None,
    token_cache: Optional[VJEPATokenCache] = None,
    range_combo_reweight_weights: Optional[torch.Tensor] = None,
    return_combo_fpfn_report: bool = False,
) -> Tuple[Any, ...]:
    model.eval()
    model.set_token_mask_epoch(int(0 if epoch is None else epoch))

    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = _get_amp_dtype(args.amp_dtype)
    use_progress = _is_progress_enabled(args)
    is_traj_only = bool(getattr(args, "trajectory_only", False))
    is_combo_only = bool(getattr(args, "combo_only", False))
    is_tool_only = bool(getattr(args, "tool_only", False))
    is_combo_probe_attn = bool(getattr(args, "combo_probe_attn", False))
    is_combo_probe_dual_attn = bool(getattr(args, "combo_probe_dual_attn", False))
    eval_traj_m2_stage_weights = _resolve_eval_traj_stage_weights(args)

    meter = defaultdict(float)
    combo_meter = defaultdict(lambda: defaultdict(float))
    collect_combo_fpfn = bool(return_combo_fpfn_report) and (not is_traj_only) and (not is_tool_only)
    combo_true_ids: List[int] = []
    combo_pred_ids: List[int] = []
    combo_sids: List[str] = []

    total_steps = len(loader)
    if args.max_val_steps > 0:
        total_steps = min(total_steps, int(args.max_val_steps))

    iterator = loader
    pbar = None
    if use_progress:
        ep = "???" if epoch is None else f"{int(epoch):03d}"
        pbar = tqdm(
            loader,
            total=total_steps,
            desc=f"Val   E{ep}",
            dynamic_ncols=True,
            leave=False,
        )
        iterator = pbar

    for step, batch in enumerate(iterator, start=1):
        if args.max_val_steps > 0 and step > args.max_val_steps:
            break

        batch = _to_device(batch, device)
        video = batch.get("video", None)
        gt_tool_xy = batch["gt_tool_xy"]
        gt_tool_vis = batch["gt_tool_vis"]
        gt_tool_xy, gt_tool_vis = _maybe_use_tool_centroid_gt(
            gt_tool_xy=gt_tool_xy,
            gt_tool_vis=gt_tool_vis,
            use_tool_centroid_only=bool(getattr(args, "tool_centroid_only", False)),
        )
        gt_ct_xy = batch["gt_ct_xy"]
        gt_ct_vis = batch["gt_ct_vis"]
        gt_action_start_bin = batch["gt_action_start_bin"].long()
        gt_action_end_bin = batch["gt_action_end_bin"].long()
        if ("gt_action_start_norm" in batch) and ("gt_action_end_norm" in batch):
            gt_action_start_norm = batch["gt_action_start_norm"].float()
            gt_action_end_norm = batch["gt_action_end_norm"].float()
        else:
            gt_action_start_norm, gt_action_end_norm = _derive_action_norm_from_bins(
                gt_action_start_bin=gt_action_start_bin,
                gt_action_end_bin=gt_action_end_bin,
                num_bins=RANGE_NUM_BINS,
            )
        action_range_mask = batch["action_range_mask"].float()
        combo_id = batch["combo_id"].long()
        tool_pt_role = batch.get("pt_role", None)
        tool_id, action_id = _derive_tool_action_id_from_combo_id(combo_id)
        sid_raw = batch.get("sid", [])
        if isinstance(sid_raw, (list, tuple)):
            sid_list = [str(x) for x in sid_raw]
        elif isinstance(sid_raw, str):
            sid_list = [sid_raw]
        else:
            sid_list = [str(sid_raw)]

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = _model_forward_with_cache(
                model=model,
                video=video,
                sids=sid_list,
                split="val",
                token_cache=token_cache,
                tool_id=tool_id,
                action_id=action_id,
                combo_id=combo_id,
            )
            if is_tool_only or is_combo_probe_attn or is_combo_probe_dual_attn:
                aligned = {
                    "gt_tool_xy": gt_tool_xy,
                    "gt_tool_vis": gt_tool_vis,
                    "gt_ct_xy": gt_ct_xy,
                    "gt_ct_vis": gt_ct_vis,
                }
            else:
                t_prime = outputs["pred_tool_xy"].shape[1]
                aligned = align_targets_to_tprime(gt_tool_xy, gt_tool_vis, gt_ct_xy, gt_ct_vis, t_prime=t_prime)
            losses = compute_batch_losses(
                outputs=outputs,
                gt_tool_xy=aligned["gt_tool_xy"],
                gt_tool_vis=aligned["gt_tool_vis"],
                gt_ct_xy=aligned["gt_ct_xy"],
                gt_ct_vis=aligned["gt_ct_vis"],
                combo_id=combo_id,
                gt_action_start_bin=gt_action_start_bin,
                gt_action_end_bin=gt_action_end_bin,
                action_range_mask=action_range_mask,
                gt_action_start_norm=gt_action_start_norm,
                gt_action_end_norm=gt_action_end_norm,
                combo_ce_weight=float(getattr(args, "combo_ce_weight", COMBO_CE_WEIGHT)),
                combo_tool_ce_weight=float(getattr(args, "combo_tool_ce_weight", 0.5)),
                combo_action_ce_weight=float(getattr(args, "combo_action_ce_weight", 0.5)),
                start_bin_loss_weight=RANGE_START_CE_WEIGHT,
                end_bin_loss_weight=RANGE_END_CE_WEIGHT,
                range_label_smoothing=RANGE_LABEL_SMOOTHING,
                range_positive_mode=str(getattr(args, "range_positive_mode", "span")),
                range_sensitivity_mode=str(getattr(args, "range_sensitivity_mode", "none")),
                range_sensitivity_cls_sigma=float(getattr(args, "range_sensitivity_cls_sigma", 0.5)),
                range_sensitivity_loc_sigma=float(getattr(args, "range_sensitivity_loc_sigma", 0.35)),
                range_sensitivity_min_weight=float(getattr(args, "range_sensitivity_min_weight", 0.25)),
                range_cls_target_mode=str(getattr(args, "range_cls_target_mode", "binary")),
                range_cls_varifocal_alpha=float(getattr(args, "range_cls_varifocal_alpha", 0.75)),
                range_cls_varifocal_weight=float(getattr(args, "range_cls_varifocal_weight", 1.0)),
                range_quality_mode=str(getattr(args, "range_quality_mode", "none")),
                range_quality_loss_weight=float(getattr(args, "range_quality_loss_weight", 0.0)),
                range_sola_aux_weight=float(getattr(args, "range_sola_aux_weight", 0.0)),
                range_sola_similarity_k=float(getattr(args, "range_sola_sim_k", 16.0)),
                range_focal_gamma=RANGE_FOCAL_GAMMA,
                range_neighbor_smoothing=RANGE_NEIGHBOR_SMOOTHING,
                range_boundary_aux_weight=float(getattr(args, "range_boundary_aux_weight", 0.0)),
                range_proposal_weight=float(getattr(args, "range_proposal_weight", 0.0)),
                range_combo_reweight_weights=range_combo_reweight_weights,
                range_combo_reweight_targets=str(getattr(args, "range_combo_reweight_targets", "")),
                kpt_loss_type=str(getattr(args, "kpt_loss_type", "l1l2")),
                traj_loss_pixel_scale=float(getattr(args, "traj_loss_pixel_scale", 256.0)),
                traj_huber_delta=float(getattr(args, "traj_huber_delta", 6.0)),
                traj_t0_loss_weight=float(getattr(args, "traj_t0_loss_weight", 1.0)),
                traj_loss_visible_only=bool(getattr(args, "traj_loss_visible_only", False)),
                traj_centroid_aux_weight=float(getattr(args, "traj_centroid_aux_weight", 0.0)),
                traj_offset_aux_weight=float(getattr(args, "traj_offset_aux_weight", 0.0)),
                trajectory_target=(str(getattr(args, "trajectory_target", "tool")) if is_traj_only else "both"),
                trajectory_only=is_traj_only,
                range_only=bool(getattr(args, "range_only", False)),
                combo_only=is_combo_only,
                tool_only=bool(getattr(args, "tool_only", False)),
                traj_heatmap_ce_weight=float(getattr(args, "traj_heatmap_ce_weight", 0.0)),
                traj_heatmap_sigma=float(getattr(args, "traj_heatmap_sigma", 2.45)),
                tool_pt_role=tool_pt_role,
                symmetry_swap_invariant_loss=bool(getattr(args, "symmetry_swap_invariant_loss", False)),
                symmetry_pair_separation_loss=bool(getattr(args, "symmetry_pair_separation_loss", False)),
                symmetry_pair_separation_weight=float(getattr(args, "symmetry_pair_separation_weight", 0.05)),
                symmetry_pair_separation_min_pred_gap_px=float(
                    getattr(args, "symmetry_pair_separation_min_pred_gap_px", 8.0)
                ),
                symmetry_pair_separation_gt_min_gap_px=float(
                    getattr(args, "symmetry_pair_separation_gt_min_gap_px", 16.0)
                ),
                traj_m2_stage_weights=eval_traj_m2_stage_weights,
            )
            ps_losses = compute_batch_losses_per_sample(
                outputs=outputs,
                gt_tool_xy=aligned["gt_tool_xy"],
                gt_tool_vis=aligned["gt_tool_vis"],
                gt_ct_xy=aligned["gt_ct_xy"],
                gt_ct_vis=aligned["gt_ct_vis"],
                combo_id=combo_id,
                gt_action_start_bin=gt_action_start_bin,
                gt_action_end_bin=gt_action_end_bin,
                action_range_mask=action_range_mask,
                gt_action_start_norm=gt_action_start_norm,
                gt_action_end_norm=gt_action_end_norm,
                combo_ce_weight=float(getattr(args, "combo_ce_weight", COMBO_CE_WEIGHT)),
                combo_tool_ce_weight=float(getattr(args, "combo_tool_ce_weight", 0.5)),
                combo_action_ce_weight=float(getattr(args, "combo_action_ce_weight", 0.5)),
                start_bin_loss_weight=RANGE_START_CE_WEIGHT,
                end_bin_loss_weight=RANGE_END_CE_WEIGHT,
                range_label_smoothing=RANGE_LABEL_SMOOTHING,
                range_positive_mode=str(getattr(args, "range_positive_mode", "span")),
                range_sensitivity_mode=str(getattr(args, "range_sensitivity_mode", "none")),
                range_sensitivity_cls_sigma=float(getattr(args, "range_sensitivity_cls_sigma", 0.5)),
                range_sensitivity_loc_sigma=float(getattr(args, "range_sensitivity_loc_sigma", 0.35)),
                range_sensitivity_min_weight=float(getattr(args, "range_sensitivity_min_weight", 0.25)),
                range_cls_target_mode=str(getattr(args, "range_cls_target_mode", "binary")),
                range_cls_varifocal_alpha=float(getattr(args, "range_cls_varifocal_alpha", 0.75)),
                range_cls_varifocal_weight=float(getattr(args, "range_cls_varifocal_weight", 1.0)),
                range_quality_mode=str(getattr(args, "range_quality_mode", "none")),
                range_quality_loss_weight=float(getattr(args, "range_quality_loss_weight", 0.0)),
                range_sola_aux_weight=float(getattr(args, "range_sola_aux_weight", 0.0)),
                range_sola_similarity_k=float(getattr(args, "range_sola_sim_k", 16.0)),
                range_focal_gamma=RANGE_FOCAL_GAMMA,
                range_neighbor_smoothing=RANGE_NEIGHBOR_SMOOTHING,
                range_boundary_aux_weight=float(getattr(args, "range_boundary_aux_weight", 0.0)),
                range_proposal_weight=float(getattr(args, "range_proposal_weight", 0.0)),
                range_combo_reweight_weights=range_combo_reweight_weights,
                range_combo_reweight_targets=str(getattr(args, "range_combo_reweight_targets", "")),
                kpt_loss_type=str(getattr(args, "kpt_loss_type", "l1l2")),
                traj_loss_pixel_scale=float(getattr(args, "traj_loss_pixel_scale", 256.0)),
                traj_huber_delta=float(getattr(args, "traj_huber_delta", 6.0)),
                traj_t0_loss_weight=float(getattr(args, "traj_t0_loss_weight", 1.0)),
                traj_loss_visible_only=bool(getattr(args, "traj_loss_visible_only", False)),
                traj_centroid_aux_weight=float(getattr(args, "traj_centroid_aux_weight", 0.0)),
                traj_offset_aux_weight=float(getattr(args, "traj_offset_aux_weight", 0.0)),
                trajectory_target=(str(getattr(args, "trajectory_target", "tool")) if is_traj_only else "both"),
                trajectory_only=is_traj_only,
                combo_only=is_combo_only,
                tool_only=bool(getattr(args, "tool_only", False)),
                traj_heatmap_ce_weight=float(getattr(args, "traj_heatmap_ce_weight", 0.0)),
                traj_heatmap_sigma=float(getattr(args, "traj_heatmap_sigma", 2.45)),
                tool_pt_role=tool_pt_role,
                symmetry_swap_invariant_loss=bool(getattr(args, "symmetry_swap_invariant_loss", False)),
                symmetry_pair_separation_loss=bool(getattr(args, "symmetry_pair_separation_loss", False)),
                symmetry_pair_separation_weight=float(getattr(args, "symmetry_pair_separation_weight", 0.05)),
                symmetry_pair_separation_min_pred_gap_px=float(
                    getattr(args, "symmetry_pair_separation_min_pred_gap_px", 8.0)
                ),
                symmetry_pair_separation_gt_min_gap_px=float(
                    getattr(args, "symmetry_pair_separation_gt_min_gap_px", 16.0)
                ),
                traj_m2_stage_weights=eval_traj_m2_stage_weights,
            )

        bsz = int(combo_id.shape[0])
        if is_combo_probe_attn:
            combo_logits_struct = outputs["combo_logits"]
        else:
            combo_logits_struct = _structured_combo_logits_from_tool_action(
                combo_logits=outputs["combo_logits"],
                combo_tool_logits=outputs.get("combo_tool_logits", outputs["combo_logits"]),
                combo_action_logits=outputs.get("combo_action_logits", outputs["combo_logits"]),
            )
        tool_id, action_id = _derive_tool_action_id_from_combo_id(combo_id)
        pred_combo: Optional[torch.Tensor] = None
        if is_combo_probe_attn:
            pred_combo = combo_logits_struct.argmax(dim=-1)
            combo_correct = (pred_combo == combo_id).float()
            tool_correct = torch.zeros_like(combo_correct)
            action_correct = torch.zeros_like(combo_correct)
        else:
            pred_tool_cls = outputs["combo_tool_logits"].argmax(dim=-1)
            tool_correct = (pred_tool_cls == tool_id).float()
            if is_tool_only:
                combo_correct = torch.zeros_like(tool_correct)
                action_correct = torch.zeros_like(tool_correct)
            else:
                pred_combo = combo_logits_struct.argmax(dim=-1)
                combo_correct = (pred_combo == combo_id).float()
                pred_action_cls = outputs["combo_action_logits"].argmax(dim=-1)
                action_correct = (pred_action_cls == action_id).float()
        if is_tool_only:
            combo_correct = torch.zeros_like(tool_correct)
            action_correct = torch.zeros_like(tool_correct)
        if collect_combo_fpfn and (pred_combo is not None):
            combo_true_ids.extend([int(x) for x in combo_id.detach().cpu().tolist()])
            combo_pred_ids.extend([int(x) for x in pred_combo.detach().cpu().tolist()])
            for i in range(bsz):
                sid_i = sid_list[i] if i < len(sid_list) else f"idx_{step:06d}_{i:03d}"
                combo_sids.append(str(sid_i))
        pred_start_norm = outputs.get("pred_action_start_norm", gt_action_start_norm).reshape(-1)
        pred_end_norm = outputs.get("pred_action_end_norm", gt_action_end_norm).reshape(-1)
        refine_stats = _compute_refine_stats(outputs)
        timing_stats = {}
        meta = outputs.get("meta", {})
        if isinstance(meta, dict):
            tms = meta.get("timing_ms", {})
            if isinstance(tms, dict):
                for k, v in tms.items():
                    try:
                        timing_stats[str(k)] = float(v)
                    except Exception:
                        continue
        if is_tool_only or is_combo_probe_attn or is_combo_probe_dual_attn:
            track_stats = _zero_track_metrics_per_sample(bsz=bsz, device=combo_id.device)
        else:
            track_stats = _compute_track_metrics_per_sample(
                pred_tool_xy=outputs["pred_tool_xy"],
                pred_ct_xy=outputs["pred_ct_xy"],
                pred_tool_conf=outputs["conf_tool"],
                pred_ct_conf=outputs["conf_ct"],
                gt_tool_xy=aligned["gt_tool_xy"],
                gt_ct_xy=aligned["gt_ct_xy"],
                gt_tool_vis=aligned["gt_tool_vis"],
                gt_ct_vis=aligned["gt_ct_vis"],
                image_size=int(args.input_size),
                trajectory_target=(str(getattr(args, "trajectory_target", "tool")) if is_traj_only else "both"),
            )
        range_stats = _compute_range_metrics_per_sample(
            pred_action_start_norm=pred_start_norm,
            pred_action_end_norm=pred_end_norm,
            gt_action_start_norm=gt_action_start_norm,
            gt_action_end_norm=gt_action_end_norm,
            action_range_mask=action_range_mask,
            sampled_abs_frames=batch.get("sampled_abs_frames", None),
        )

        meter["n"] += bsz
        meter["loss_total"] += losses["loss_total"].item() * bsz
        meter["loss_tool_xy"] += losses["loss_tool_xy"].item() * bsz
        meter["loss_ct_xy"] += losses["loss_ct_xy"].item() * bsz
        _accumulate_traj_stage_losses(meter, losses, bsz)
        meter["loss_tool_centroid_aux"] += losses.get("loss_tool_centroid_aux", losses["loss_total"] * 0.0).item() * bsz
        meter["loss_tool_offset_aux"] += losses.get("loss_tool_offset_aux", losses["loss_total"] * 0.0).item() * bsz
        meter["loss_tool_pair_sep_aux"] += losses.get("loss_tool_pair_sep_aux", losses["loss_total"] * 0.0).item() * bsz
        meter["loss_tool_heatmap_ce"] += losses.get("loss_tool_heatmap_ce", losses["loss_total"] * 0.0).item() * bsz
        meter["loss_ct_heatmap_ce"] += losses.get("loss_ct_heatmap_ce", losses["loss_total"] * 0.0).item() * bsz
        meter["loss_combo_ce"] += losses["loss_combo_ce"].item() * bsz
        meter["loss_combo_tool_ce"] += losses.get("loss_combo_tool_ce", losses["loss_total"] * 0.0).item() * bsz
        meter["loss_combo_action_ce"] += losses.get("loss_combo_action_ce", losses["loss_total"] * 0.0).item() * bsz
        meter["loss_start_bin_ce"] += losses["loss_start_bin_ce"].item() * bsz
        meter["loss_end_bin_ce"] += losses["loss_end_bin_ce"].item() * bsz
        meter["loss_range_cls"] += losses.get("loss_range_cls", losses["loss_range"] * 0.0).item() * bsz
        meter["loss_range_reg"] += losses.get("loss_range_reg", losses["loss_range"] * 0.0).item() * bsz
        meter["loss_range_proposal"] += losses.get("loss_range_proposal", losses["loss_range"] * 0.0).item() * bsz
        meter["loss_range_quality"] += losses.get("loss_range_quality", losses["loss_range"] * 0.0).item() * bsz
        meter["loss_range_sola"] += losses.get("loss_range_sola", losses["loss_range"] * 0.0).item() * bsz
        meter["loss_range"] += losses["loss_range"].item() * bsz
        meter["combo_correct"] += combo_correct.sum().item()
        meter["tool_correct"] += tool_correct.sum().item()
        meter["action_correct"] += action_correct.sum().item()
        meter["traj_ade_num"] += float(track_stats["traj_ade_num"].item())
        meter["traj_ade_den"] += float(track_stats["traj_ade_den"].item())
        meter["traj_fde_num"] += float(track_stats["traj_fde_num"].item())
        meter["traj_fde_den"] += float(track_stats["traj_fde_den"].item())
        meter["traj_pck_den"] += float(track_stats["traj_pck_den"].item())
        for key in _track_pck_keys():
            meter[f"traj_pck_{key}_num"] += float(track_stats[f"traj_pck_{key}_num"].item())
        meter["traj_survival_num"] += float(track_stats["traj_survival_num"].item())
        meter["traj_survival_den"] += float(track_stats["traj_survival_den"].item())
        meter["traj_aj_sum"] += float(track_stats["traj_aj_sum"].item())
        meter["traj_aj_den"] += float(track_stats["traj_aj_den"].item())
        meter["range_valid_den"] += float(range_stats["range_valid_den"].item())
        meter["range_count"] += float(range_stats["range_valid_den"].item())
        meter["range_tiou_sum"] += float(range_stats["range_tiou_sum"].item())
        meter["range_r03_sum"] += float(range_stats["range_r03_sum"].item())
        meter["range_r05_sum"] += float(range_stats["range_r05_sum"].item())
        meter["range_r07_sum"] += float(range_stats["range_r07_sum"].item())
        meter["range_start_mae_norm_sum"] += float(range_stats["range_start_mae_norm_sum"].item())
        meter["range_end_mae_norm_sum"] += float(range_stats["range_end_mae_norm_sum"].item())
        meter["range_center_mae_norm_sum"] += float(range_stats["range_center_mae_norm_sum"].item())
        meter["range_duration_mae_norm_sum"] += float(range_stats["range_duration_mae_norm_sum"].item())
        meter["range_start_mae_abs_sum"] += float(range_stats["range_start_mae_abs_sum"].item())
        meter["range_end_mae_abs_sum"] += float(range_stats["range_end_mae_abs_sum"].item())
        meter["refine_gate_mean"] += float(refine_stats["refine_gate_mean"]) * bsz
        meter["refine_delta_abs_mean"] += float(refine_stats["refine_delta_abs_mean"]) * bsz
        meter["refine_centroid_shift_abs_mean"] += float(refine_stats.get("refine_centroid_shift_abs_mean", 0.0)) * bsz
        meter["anchor_delta_centroid_abs_mean"] += float(refine_stats.get("anchor_delta_centroid_abs_mean", 0.0)) * bsz
        meter["anchor_alpha_mean"] += float(refine_stats.get("anchor_alpha_mean", 0.0)) * bsz
        meter["anchor_verify_scale_mean"] += float(refine_stats.get("anchor_verify_scale_mean", 0.0)) * bsz
        meter["anchor_verify_residual_mean"] += float(refine_stats.get("anchor_verify_residual_mean", 0.0)) * bsz
        meter["stage0_centroid_noise_abs_mean"] += float(
            refine_stats.get("stage0_centroid_noise_abs_mean", 0.0)
        ) * bsz
        meter["factor_anchor_gate_mean"] += float(refine_stats.get("factor_anchor_gate_mean", 0.0)) * bsz
        meter["factor_offset_gate_mean"] += float(refine_stats.get("factor_offset_gate_mean", 0.0)) * bsz
        meter["factor_delta_offset_abs_mean"] += float(refine_stats.get("factor_delta_offset_abs_mean", 0.0)) * bsz
        meter["corr_entropy_mean"] += float(refine_stats.get("corr_entropy_mean", 0.0)) * bsz
        meter["corr_top1_minus_top2_mean"] += float(refine_stats.get("corr_top1_minus_top2_mean", 0.0)) * bsz
        meter["context_entropy_mean"] += float(refine_stats.get("context_entropy_mean", 0.0)) * bsz
        meter["context_top1_minus_top2_mean"] += float(refine_stats.get("context_top1_minus_top2_mean", 0.0)) * bsz
        meter["context_offset_abs_mean"] += float(refine_stats.get("context_offset_abs_mean", 0.0)) * bsz
        meter["context_gate_mean"] += float(refine_stats.get("context_gate_mean", 0.0)) * bsz
        meter["belief_gate_mean"] += float(refine_stats.get("belief_gate_mean", 0.0)) * bsz
        meter["joint_gate_mean"] += float(refine_stats.get("joint_gate_mean", 0.0)) * bsz
        meter["template_entropy_mean"] += float(refine_stats.get("template_entropy_mean", 0.0)) * bsz
        meter["template_top1_minus_top2_mean"] += float(refine_stats.get("template_top1_minus_top2_mean", 0.0)) * bsz
        meter["template_gate_mean"] += float(refine_stats.get("template_gate_mean", 0.0)) * bsz
        meter["template_offset_abs_mean"] += float(refine_stats.get("template_offset_abs_mean", 0.0)) * bsz
        meter["template_delta_abs_mean"] += float(refine_stats.get("template_delta_abs_mean", 0.0)) * bsz
        for tk, tv in timing_stats.items():
            meter[f"timing_{tk}"] += float(tv) * bsz
        meter["conf_tool_mean"] += outputs["conf_tool"].mean().item() * bsz
        meter["conf_ct_mean"] += outputs["conf_ct"].mean().item() * bsz

        for i in range(bsz):
            cid = int(combo_id[i].item())
            m = combo_meter[cid]
            m["count"] += 1
            m["loss_total"] += float(ps_losses["loss_total"][i].item())
            m["loss_tool_xy"] += float(ps_losses["loss_tool_xy"][i].item())
            m["loss_ct_xy"] += float(ps_losses["loss_ct_xy"][i].item())
            m["loss_combo_ce"] += float(ps_losses["loss_combo_ce"][i].item())
            m["loss_tool_pair_sep_aux"] += float(ps_losses.get("loss_tool_pair_sep_aux", ps_losses["loss_total"] * 0.0)[i].item())
            m["loss_combo_tool_ce"] += float(ps_losses.get("loss_combo_tool_ce", ps_losses["loss_total"] * 0.0)[i].item())
            m["loss_combo_action_ce"] += float(ps_losses.get("loss_combo_action_ce", ps_losses["loss_total"] * 0.0)[i].item())
            m["loss_start_bin_ce"] += float(ps_losses["loss_start_bin_ce"][i].item())
            m["loss_end_bin_ce"] += float(ps_losses["loss_end_bin_ce"][i].item())
            m["loss_range_cls"] += _loss_value_for_sample(
                ps_losses.get("loss_range_cls", ps_losses["loss_range"] * 0.0),
                sample_idx=i,
            )
            m["loss_range_reg"] += _loss_value_for_sample(
                ps_losses.get("loss_range_reg", ps_losses["loss_range"] * 0.0),
                sample_idx=i,
            )
            m["loss_range_proposal"] += _loss_value_for_sample(
                ps_losses.get("loss_range_proposal", ps_losses["loss_range"] * 0.0),
                sample_idx=i,
            )
            m["loss_range_quality"] += _loss_value_for_sample(
                ps_losses.get("loss_range_quality", ps_losses["loss_range"] * 0.0),
                sample_idx=i,
            )
            m["loss_range_sola"] += _loss_value_for_sample(
                ps_losses.get("loss_range_sola", ps_losses["loss_range"] * 0.0),
                sample_idx=i,
            )
            m["loss_range"] += float(ps_losses["loss_range"][i].item())
            m["combo_acc"] += float(combo_correct[i].item())
            m["tool_acc"] += float(tool_correct[i].item())
            m["action_acc"] += float(action_correct[i].item())
            m["traj_valid_count"] += float(track_stats["traj_valid"][i].item())
            m["traj_fde_valid_count"] += float(track_stats["traj_fde_valid"][i].item())
            m["traj_survival_valid_count"] += float(track_stats["traj_survival_valid"][i].item())
            m["traj_ade_sum"] += float(track_stats["traj_ade_px"][i].item() * track_stats["traj_valid"][i].item())
            m["traj_fde_sum"] += float(track_stats["traj_fde_px"][i].item() * track_stats["traj_fde_valid"][i].item())
            for key in _track_pck_keys():
                m[f"traj_pck_{key}_sum"] += float(
                    track_stats[f"traj_pck_{key}"][i].item() * track_stats["traj_valid"][i].item()
                )
            m["traj_pck_auc_sum"] += float(track_stats["traj_pck_auc"][i].item() * track_stats["traj_valid"][i].item())
            m["traj_survival_sum"] += float(
                track_stats["traj_survival_50"][i].item() * track_stats["traj_survival_valid"][i].item()
            )
            m["traj_aj_sum"] += float(track_stats["traj_aj"][i].item() * track_stats["traj_aj_valid"][i].item())
            m["traj_aj_valid_count"] += float(track_stats["traj_aj_valid"][i].item())
            m["range_count"] += float(range_stats["range_valid"][i].item())
            m["range_tiou_sum"] += float(range_stats["range_tiou"][i].item() * range_stats["range_valid"][i].item())
            m["range_r03_sum"] += float(range_stats["range_r03"][i].item() * range_stats["range_valid"][i].item())
            m["range_r05_sum"] += float(range_stats["range_r05"][i].item() * range_stats["range_valid"][i].item())
            m["range_r07_sum"] += float(range_stats["range_r07"][i].item() * range_stats["range_valid"][i].item())
            m["range_start_mae_norm_sum"] += float(
                range_stats["range_start_mae_norm"][i].item() * range_stats["range_valid"][i].item()
            )
            m["range_end_mae_norm_sum"] += float(
                range_stats["range_end_mae_norm"][i].item() * range_stats["range_valid"][i].item()
            )
            m["range_start_mae_abs_sum"] += float(
                range_stats["range_start_mae_abs"][i].item() * range_stats["range_valid"][i].item()
            )
            m["range_end_mae_abs_sum"] += float(
                range_stats["range_end_mae_abs"][i].item() * range_stats["range_valid"][i].item()
            )

        if pbar is not None:
            n = max(int(meter["n"]), 1)
            track_den_now = max(float(meter.get("traj_pck_den", 0.0)), 1.0)
            range_count_now = max(float(meter.get("range_valid_den", 0.0)), 1.0)
            if is_traj_only:
                pbar.set_postfix(
                    loss=f"{meter['loss_total']/n:.4f}",
                    tool=f"{meter['loss_tool_xy']/n:.4f}",
                    ct=f"{meter['loss_ct_xy']/n:.4f}",
                    thm=f"{meter.get('loss_tool_heatmap_ce', 0.0)/n:.4f}",
                    chm=f"{meter.get('loss_ct_heatmap_ce', 0.0)/n:.4f}",
                    ade=f"{_safe_div(meter.get('traj_ade_num', 0.0), meter.get('traj_ade_den', 0.0)):.2f}",
                    aj=f"{_safe_div(meter.get('traj_aj_sum', 0.0), meter.get('traj_aj_den', 0.0)):.3f}",
                    p8=f"{_safe_div(meter.get('traj_pck_8_num', 0.0), track_den_now):.3f}",
                    rfg=f"{meter.get('refine_gate_mean', 0.0)/n:.3f}",
                    rd=f"{meter.get('refine_delta_abs_mean', 0.0)/n:.4f}",
                )
            elif is_tool_only:
                pbar.set_postfix(
                    loss=f"{meter['loss_total']/n:.4f}",
                    tce=f"{meter.get('loss_combo_tool_ce', 0.0)/n:.4f}",
                    ta=f"{meter.get('tool_correct', 0.0)/n:.3f}",
                )
            elif is_combo_probe_attn:
                pbar.set_postfix(
                    loss=f"{meter['loss_total']/n:.4f}",
                    ce=f"{meter['loss_combo_ce']/n:.4f}",
                    ca=f"{meter['combo_correct']/n:.3f}",
                )
            elif is_combo_only:
                pbar.set_postfix(
                    loss=f"{meter['loss_total']/n:.4f}",
                    ce=f"{meter['loss_combo_ce']/n:.4f}",
                    tce=f"{meter.get('loss_combo_tool_ce', 0.0)/n:.4f}",
                    ace=f"{meter.get('loss_combo_action_ce', 0.0)/n:.4f}",
                    ca=f"{meter['combo_correct']/n:.3f}",
                    ta=f"{meter.get('tool_correct', 0.0)/n:.3f}",
                    aa=f"{meter.get('action_correct', 0.0)/n:.3f}",
                )
            else:
                pbar.set_postfix(
                    loss=f"{meter['loss_total']/n:.4f}",
                    tool=f"{meter['loss_tool_xy']/n:.4f}",
                    ct=f"{meter['loss_ct_xy']/n:.4f}",
                    ce=f"{meter['loss_combo_ce']/n:.4f}",
                    sl=f"{meter['loss_start_bin_ce']/n:.4f}",
                    el=f"{meter['loss_end_bin_ce']/n:.4f}",
                    rc=f"{meter['loss_range_cls']/n:.4f}",
                    rr=f"{meter['loss_range_reg']/n:.4f}",
                    rp=f"{meter.get('loss_range_proposal', 0.0)/n:.4f}",
                    rng=f"{meter['loss_range']/n:.4f}",
                    ca=f"{meter['combo_correct']/n:.3f}",
                    ade=f"{_safe_div(meter.get('traj_ade_num', 0.0), meter.get('traj_ade_den', 0.0)):.2f}",
                    aj=f"{_safe_div(meter.get('traj_aj_sum', 0.0), meter.get('traj_aj_den', 0.0)):.3f}",
                    p8=f"{_safe_div(meter.get('traj_pck_8_num', 0.0), track_den_now):.3f}",
                    tiou=f"{_safe_div(meter.get('range_tiou_sum', 0.0), range_count_now):.3f}",
                    r05=f"{_safe_div(meter.get('range_r05_sum', 0.0), range_count_now):.3f}",
                    rfg=f"{meter.get('refine_gate_mean', 0.0)/n:.3f}",
                    rd=f"{meter.get('refine_delta_abs_mean', 0.0)/n:.4f}",
                )

    if pbar is not None:
        pbar.close()

    global_metrics = _finalize_meter(meter)

    by_combo: Dict[str, Dict[str, float]] = {}
    for cid in sorted(combo_meter.keys()):
        cm = combo_meter[cid]
        c = max(int(cm["count"]), 1)
        range_c = max(float(cm.get("range_count", 0.0)), 1.0)
        traj_c = max(float(cm.get("traj_valid_count", 0.0)), 1.0)
        traj_fde_c = max(float(cm.get("traj_fde_valid_count", 0.0)), 1.0)
        traj_surv_c = max(float(cm.get("traj_survival_valid_count", 0.0)), 1.0)
        traj_aj_c = max(float(cm.get("traj_aj_valid_count", 0.0)), 1.0)
        name = combo_id_to_name.get(cid, f"combo_{cid}")
        combo_row = {
            "combo_id": cid,
            "count": c,
            "traj_valid_count": float(cm.get("traj_valid_count", 0.0)),
            "traj_fde_valid_count": float(cm.get("traj_fde_valid_count", 0.0)),
            "traj_survival_valid_count": float(cm.get("traj_survival_valid_count", 0.0)),
            "traj_aj_valid_count": float(cm.get("traj_aj_valid_count", 0.0)),
            "loss_total": cm["loss_total"] / c,
            "loss_tool_xy": cm["loss_tool_xy"] / c,
            "loss_ct_xy": cm["loss_ct_xy"] / c,
            "loss_tool_pair_sep_aux": cm.get("loss_tool_pair_sep_aux", 0.0) / c,
            "loss_combo_ce": cm["loss_combo_ce"] / c,
            "loss_combo_tool_ce": cm.get("loss_combo_tool_ce", 0.0) / c,
            "loss_combo_action_ce": cm.get("loss_combo_action_ce", 0.0) / c,
            "loss_start_bin_ce": cm["loss_start_bin_ce"] / c,
            "loss_end_bin_ce": cm["loss_end_bin_ce"] / c,
            "loss_start_l1": cm["loss_start_bin_ce"] / c,
            "loss_end_l1": cm["loss_end_bin_ce"] / c,
            "loss_range_cls": cm.get("loss_range_cls", 0.0) / c,
            "loss_range_reg": cm.get("loss_range_reg", 0.0) / c,
            "loss_range_proposal": cm.get("loss_range_proposal", 0.0) / c,
            "loss_range_quality": cm.get("loss_range_quality", 0.0) / c,
            "loss_range_sola": cm.get("loss_range_sola", 0.0) / c,
            "loss_range": cm["loss_range"] / c,
            "combo_acc": cm["combo_acc"] / c,
            "tool_acc": cm.get("tool_acc", 0.0) / c,
            "action_acc": cm.get("action_acc", 0.0) / c,
            "traj_ade_px": cm.get("traj_ade_sum", 0.0) / traj_c,
            "traj_fde_px": cm.get("traj_fde_sum", 0.0) / traj_fde_c,
            "traj_pck_auc": cm.get("traj_pck_auc_sum", 0.0) / traj_c,
            "traj_survival_50": cm.get("traj_survival_sum", 0.0) / traj_surv_c,
            "traj_aj": cm.get("traj_aj_sum", 0.0) / traj_aj_c,
            "range_tiou": cm.get("range_tiou_sum", 0.0) / range_c,
            "range_r03": cm.get("range_r03_sum", 0.0) / range_c,
            "range_r05": cm.get("range_r05_sum", 0.0) / range_c,
            "range_r07": cm.get("range_r07_sum", 0.0) / range_c,
            "range_start_mae_norm": cm.get("range_start_mae_norm_sum", 0.0) / range_c,
            "range_end_mae_norm": cm.get("range_end_mae_norm_sum", 0.0) / range_c,
            "range_start_mae_abs": cm.get("range_start_mae_abs_sum", 0.0) / range_c,
            "range_end_mae_abs": cm.get("range_end_mae_abs_sum", 0.0) / range_c,
            "range_count": float(cm.get("range_count", 0.0)),
        }
        for key in _track_pck_keys():
            combo_row[f"traj_pck_{key}"] = cm.get(f"traj_pck_{key}_sum", 0.0) / traj_c
        by_combo[name] = combo_row

    combo_fpfn_report: Optional[Dict[str, Any]] = None
    if collect_combo_fpfn:
        combo_fpfn_report = _build_combo_fpfn_report(
            true_combo_ids=combo_true_ids,
            pred_combo_ids=combo_pred_ids,
            sids=combo_sids,
            combo_id_to_name=combo_id_to_name,
        )

    if bool(return_combo_fpfn_report):
        return global_metrics, by_combo, combo_fpfn_report
    return global_metrics, by_combo


EXTRA_TRAINABLE_PARAM_KEY = "__extra_trainable_params__"


def _collect_extra_named_trainable_params(model: KeypointComboNet):
    covered_param_ids = set()
    for _, module in model.trainable_modules.items():
        for _, p in module.named_parameters(recurse=True):
            covered_param_ids.add(id(p))
    extra = []
    for name, p in model.named_parameters(recurse=True):
        if (not p.requires_grad) or (id(p) in covered_param_ids):
            continue
        extra.append((name, p))
    return extra


def trainable_state_dict(model: KeypointComboNet) -> Dict[str, Dict[str, torch.Tensor]]:
    out = {name: module.state_dict() for name, module in model.trainable_modules.items()}
    extra = {name: p.detach().cpu() for name, p in _collect_extra_named_trainable_params(model)}
    if extra:
        out[EXTRA_TRAINABLE_PARAM_KEY] = extra
    return out


def load_trainable_state_dict(model: KeypointComboNet, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
    loaded_exact: List[str] = []
    loaded_mapped: List[str] = []
    missing: List[str] = []
    extra_loaded: List[str] = []
    extra_missing: List[str] = []
    used_src_keys: List[str] = []

    for name, module in model.trainable_modules.items():
        if name in state:
            module.load_state_dict(state[name], strict=True)
            loaded_exact.append(name)
            used_src_keys.append(name)
            continue

        # Backward-compatible warm-start:
        # tool-only checkpoints have `combo_attn_pool`, while dual-attn expects
        # both `combo_tool_spatial_pool` and `combo_action_temporal_pool`.
        if name in ("combo_tool_spatial_pool", "combo_action_temporal_pool") and ("combo_attn_pool" in state):
            module.load_state_dict(state["combo_attn_pool"], strict=True)
            loaded_mapped.append(f"{name}<-combo_attn_pool")
            used_src_keys.append("combo_attn_pool")
            continue

        missing.append(name)

    extra_state = state.get(EXTRA_TRAINABLE_PARAM_KEY, None)
    if isinstance(extra_state, dict):
        with torch.no_grad():
            for name, p in _collect_extra_named_trainable_params(model):
                if name in extra_state:
                    p.copy_(extra_state[name].to(device=p.device, dtype=p.dtype))
                    extra_loaded.append(name)
                else:
                    extra_missing.append(name)
        used_src_keys.append(EXTRA_TRAINABLE_PARAM_KEY)
    else:
        extra_missing = [name for name, _ in _collect_extra_named_trainable_params(model)]

    unused = [k for k in state.keys() if k not in set(used_src_keys)]
    print(
        "[resume] trainable load summary: "
        f"exact={len(loaded_exact)} mapped={len(loaded_mapped)} "
        f"missing={len(missing)} extra_loaded={len(extra_loaded)} "
        f"extra_missing={len(extra_missing)} unused_src={len(unused)}"
    )
    if loaded_mapped:
        print(f"[resume] mapped: {', '.join(loaded_mapped)}")
    if missing:
        print(f"[resume] missing target modules (kept random init): {', '.join(missing)}")
    if extra_missing:
        print(
            "[resume] missing extra trainable params (kept random init): "
            f"{', '.join(extra_missing)}"
        )
    if unused:
        print(f"[resume] unused source modules: {', '.join(unused)}")


def _iter_named_trainable_params(model: KeypointComboNet):
    for mod_name, module in model.trainable_modules.items():
        for name, p in module.named_parameters(recurse=True):
            yield f"{mod_name}.{name}", p
    for name, p in _collect_extra_named_trainable_params(model):
        yield f"__extra__.{name}", p


class TrainableEMA:
    def __init__(self, model: KeypointComboNet, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.reset(model)

    def reset(self, model: KeypointComboNet) -> None:
        self.shadow = {}
        with torch.no_grad():
            for key, p in _iter_named_trainable_params(model):
                if (not p.requires_grad) or (not torch.is_floating_point(p)):
                    continue
                self.shadow[key] = p.detach().clone().float()

    @torch.no_grad()
    def update(self, model: KeypointComboNet) -> None:
        one_minus = 1.0 - self.decay
        for key, p in _iter_named_trainable_params(model):
            if key not in self.shadow:
                continue
            src = p.detach()
            tgt = self.shadow[key]
            tgt.mul_(self.decay).add_(src.to(dtype=torch.float32), alpha=one_minus)

    @torch.no_grad()
    def apply_shadow(self, model: KeypointComboNet) -> Dict[str, torch.Tensor]:
        backup: Dict[str, torch.Tensor] = {}
        for key, p in _iter_named_trainable_params(model):
            if key not in self.shadow:
                continue
            backup[key] = p.detach().clone()
            p.copy_(self.shadow[key].to(device=p.device, dtype=p.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: KeypointComboNet, backup: Dict[str, torch.Tensor]) -> None:
        for key, p in _iter_named_trainable_params(model):
            if key not in backup:
                continue
            p.copy_(backup[key].to(device=p.device, dtype=p.dtype))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": float(self.decay),
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, Any], model: KeypointComboNet) -> None:
        decay = state.get("decay", None)
        if decay is not None:
            self.decay = float(decay)
        src_shadow = state.get("shadow", {})
        if not isinstance(src_shadow, dict):
            raise ValueError("Invalid EMA state: expected dict at key `shadow`.")
        loaded = 0
        missed = 0
        with torch.no_grad():
            for key, p in _iter_named_trainable_params(model):
                if key in src_shadow:
                    self.shadow[key] = src_shadow[key].to(device=p.device, dtype=torch.float32).detach().clone()
                    loaded += 1
                else:
                    missed += 1
        print(f"[ema] loaded shadow params: loaded={loaded} missed={missed} decay={self.decay:.6f}")


@contextmanager
def _use_ema_weights(model: KeypointComboNet, ema: Optional[TrainableEMA]):
    if ema is None:
        yield
        return
    backup = ema.apply_shadow(model)
    try:
        yield
    finally:
        ema.restore(model, backup)


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    model: KeypointComboNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    args: argparse.Namespace,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    val_by_combo: Dict[str, Dict[str, float]],
    val_by_tool: Optional[Dict[str, Dict[str, Any]]] = None,
    ema: Optional[TrainableEMA] = None,
    traj_late_focal_state: Optional[Dict[str, Any]] = None,
    best_monitor_name: Optional[str] = None,
    best_monitor_value: Optional[float] = None,
    best_monitor_epoch: Optional[int] = None,
    early_stop_ref_value: Optional[float] = None,
    early_stop_ref_epoch: Optional[int] = None,
    early_stop_bad_epochs: Optional[int] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "global_step": int(global_step),
        "modules": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "args": vars(args),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "val_by_combo": val_by_combo,
        "vjepa": {
            "arch": args.vjepa_arch,
            "repo": args.vjepa_repo,
            "hf_repo": args.vjepa_hf_repo,
            "hf_local_files_only": args.vjepa_hf_local_files_only,
            "checkpoint": args.vjepa_checkpoint,
            "pretrained_hub": args.vjepa_pretrained_hub,
            "frozen": True,
        },
    }
    if val_by_tool is not None:
        ckpt["val_by_tool"] = val_by_tool
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    if traj_late_focal_state is not None:
        ckpt["traj_late_focal_state"] = dict(traj_late_focal_state)
    if (best_monitor_name is not None) or (best_monitor_value is not None) or (best_monitor_epoch is not None):
        ckpt["best_monitor"] = {
            "name": str(best_monitor_name or ""),
            "value": (None if best_monitor_value is None else float(best_monitor_value)),
            "epoch": (None if best_monitor_epoch is None else int(best_monitor_epoch)),
            "mode": _best_monitor_mode(args),
        }
    if (
        (early_stop_ref_value is not None)
        or (early_stop_ref_epoch is not None)
        or (early_stop_bad_epochs is not None)
    ):
        ckpt["early_stop"] = {
            "ref_value": (None if early_stop_ref_value is None else float(early_stop_ref_value)),
            "ref_epoch": (None if early_stop_ref_epoch is None else int(early_stop_ref_epoch)),
            "bad_epochs": (None if early_stop_bad_epochs is None else int(early_stop_bad_epochs)),
        }
    torch.save(ckpt, path)


def print_epoch_summary(epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
    train_start_l1 = float(train_metrics.get("loss_start_l1", train_metrics.get("loss_start_bin_ce", 0.0)))
    train_end_l1 = float(train_metrics.get("loss_end_l1", train_metrics.get("loss_end_bin_ce", 0.0)))
    train_range_cls = float(train_metrics.get("loss_range_cls", 0.0))
    train_range_reg = float(train_metrics.get("loss_range_reg", 0.0))
    train_range_prop = float(train_metrics.get("loss_range_proposal", 0.0))
    train_range_qual = float(train_metrics.get("loss_range_quality", 0.0))
    train_range_sola = float(train_metrics.get("loss_range_sola", 0.0))
    train_ref_gate = float(train_metrics.get("refine_gate_mean", 0.0))
    train_ref_dxy = float(train_metrics.get("refine_delta_abs_mean", 0.0))
    train_ref_cshift = float(train_metrics.get("refine_centroid_shift_abs_mean", 0.0))
    train_anchor_cshift = float(train_metrics.get("anchor_delta_centroid_abs_mean", 0.0))
    train_stage0_noise = float(train_metrics.get("stage0_centroid_noise_abs_mean", 0.0))
    train_anchor_alpha = float(train_metrics.get("anchor_alpha_mean", 0.0))
    train_anchor_vfy = float(train_metrics.get("anchor_verify_scale_mean", 1.0))
    train_factor_ag = float(train_metrics.get("factor_anchor_gate_mean", 0.0))
    train_factor_og = float(train_metrics.get("factor_offset_gate_mean", 0.0))
    train_factor_off = float(train_metrics.get("factor_delta_offset_abs_mean", 0.0))
    val_start_l1 = float(val_metrics.get("loss_start_l1", val_metrics.get("loss_start_bin_ce", 0.0)))
    val_end_l1 = float(val_metrics.get("loss_end_l1", val_metrics.get("loss_end_bin_ce", 0.0)))
    val_range_cls = float(val_metrics.get("loss_range_cls", 0.0))
    val_range_reg = float(val_metrics.get("loss_range_reg", 0.0))
    val_range_prop = float(val_metrics.get("loss_range_proposal", 0.0))
    val_range_qual = float(val_metrics.get("loss_range_quality", 0.0))
    val_range_sola = float(val_metrics.get("loss_range_sola", 0.0))
    val_ref_gate = float(val_metrics.get("refine_gate_mean", 0.0))
    val_ref_dxy = float(val_metrics.get("refine_delta_abs_mean", 0.0))
    val_ref_cshift = float(val_metrics.get("refine_centroid_shift_abs_mean", 0.0))
    val_anchor_cshift = float(val_metrics.get("anchor_delta_centroid_abs_mean", 0.0))
    val_stage0_noise = float(val_metrics.get("stage0_centroid_noise_abs_mean", 0.0))
    val_anchor_alpha = float(val_metrics.get("anchor_alpha_mean", 0.0))
    val_anchor_vfy = float(val_metrics.get("anchor_verify_scale_mean", 1.0))
    val_factor_ag = float(val_metrics.get("factor_anchor_gate_mean", 0.0))
    val_factor_og = float(val_metrics.get("factor_offset_gate_mean", 0.0))
    val_factor_off = float(val_metrics.get("factor_delta_offset_abs_mean", 0.0))
    print(
        f"[epoch {epoch:03d}] train: total={train_metrics['loss_total']:.5f} "
        f"tool={train_metrics['loss_tool_xy']:.5f} ct={train_metrics['loss_ct_xy']:.5f} "
        f"{_format_traj_stage_summary(train_metrics)} "
        f"combo_ce={train_metrics['loss_combo_ce']:.5f} "
        f"start_l1={train_start_l1:.5f} "
        f"end_l1={train_end_l1:.5f} "
        f"range={train_metrics['loss_range']:.5f} "
        f"range_cls={train_range_cls:.5f} range_reg={train_range_reg:.5f} "
        f"range_prop={train_range_prop:.5f} range_qual={train_range_qual:.5f} "
        f"range_sola={train_range_sola:.5f} "
        f"combo_acc={train_metrics['combo_acc']:.4f} "
        f"traj_ade_px={train_metrics.get('traj_ade_px', 0.0):.3f} "
        f"traj_aj={train_metrics.get('traj_aj', 0.0):.4f} "
        f"traj_pck8={train_metrics.get('traj_pck_8', 0.0):.4f} "
        f"range_tiou={train_metrics.get('range_tiou', 0.0):.4f} "
        f"range_r05={train_metrics.get('range_r05', 0.0):.4f} "
        f"ref_gate={train_ref_gate:.4f} ref_dxy={train_ref_dxy:.5f} ref_cshift={train_ref_cshift:.5f} "
        f"anchor_cshift={train_anchor_cshift:.5f} st0_noise={train_stage0_noise:.5f} "
        f"anchor_alpha={train_anchor_alpha:.4f} "
        f"anchor_vfy={train_anchor_vfy:.4f} "
        f"factor_ag={train_factor_ag:.4f} factor_og={train_factor_og:.4f} factor_off={train_factor_off:.5f} "
        f"ms(total={train_metrics.get('timing_model_total_ms', 0.0):.2f},"
        f"trunk={train_metrics.get('timing_shared_trunk_ms', 0.0):.2f},"
        f"conv={train_metrics.get('timing_conv_refine_ms', 0.0):.2f},"
        f"heads={train_metrics.get('timing_spatial_heads_decode_ms', 0.0):.2f},"
        f"up={train_metrics.get('timing_temporal_upsample_ms', 0.0):.2f},"
        f"feat={train_metrics.get('timing_local_feat_weighted_ms', 0.0):.2f},"
        f"ref_t={train_metrics.get('timing_refiner_tool_ms', 0.0):.2f},"
        f"ref_c={train_metrics.get('timing_refiner_contact_ms', 0.0):.2f},"
        f"combo={train_metrics.get('timing_combo_head_ms', 0.0):.2f},"
        f"range={train_metrics.get('timing_range_head_ms', 0.0):.2f})"
    )
    print(
        f"[epoch {epoch:03d}]   val: total={val_metrics['loss_total']:.5f} "
        f"tool={val_metrics['loss_tool_xy']:.5f} ct={val_metrics['loss_ct_xy']:.5f} "
        f"{_format_traj_stage_summary(val_metrics)} "
        f"combo_ce={val_metrics['loss_combo_ce']:.5f} "
        f"start_l1={val_start_l1:.5f} "
        f"end_l1={val_end_l1:.5f} "
        f"range={val_metrics['loss_range']:.5f} "
        f"range_cls={val_range_cls:.5f} range_reg={val_range_reg:.5f} "
        f"range_prop={val_range_prop:.5f} range_qual={val_range_qual:.5f} "
        f"range_sola={val_range_sola:.5f} "
        f"combo_acc={val_metrics['combo_acc']:.4f} "
        f"traj_ade_px={val_metrics.get('traj_ade_px', 0.0):.3f} "
        f"traj_aj={val_metrics.get('traj_aj', 0.0):.4f} "
        f"traj_pck8={val_metrics.get('traj_pck_8', 0.0):.4f} "
        f"range_tiou={val_metrics.get('range_tiou', 0.0):.4f} "
        f"range_r05={val_metrics.get('range_r05', 0.0):.4f} "
        f"ref_gate={val_ref_gate:.4f} ref_dxy={val_ref_dxy:.5f} ref_cshift={val_ref_cshift:.5f} "
        f"anchor_cshift={val_anchor_cshift:.5f} st0_noise={val_stage0_noise:.5f} "
        f"anchor_alpha={val_anchor_alpha:.4f} "
        f"anchor_vfy={val_anchor_vfy:.4f} "
        f"factor_ag={val_factor_ag:.4f} factor_og={val_factor_og:.4f} factor_off={val_factor_off:.5f} "
        f"ms(total={val_metrics.get('timing_model_total_ms', 0.0):.2f},"
        f"trunk={val_metrics.get('timing_shared_trunk_ms', 0.0):.2f},"
        f"conv={val_metrics.get('timing_conv_refine_ms', 0.0):.2f},"
        f"heads={val_metrics.get('timing_spatial_heads_decode_ms', 0.0):.2f},"
        f"up={val_metrics.get('timing_temporal_upsample_ms', 0.0):.2f},"
        f"feat={val_metrics.get('timing_local_feat_weighted_ms', 0.0):.2f},"
        f"ref_t={val_metrics.get('timing_refiner_tool_ms', 0.0):.2f},"
        f"ref_c={val_metrics.get('timing_refiner_contact_ms', 0.0):.2f},"
        f"combo={val_metrics.get('timing_combo_head_ms', 0.0):.2f},"
        f"range={val_metrics.get('timing_range_head_ms', 0.0):.2f}) "
        f"conf_tool={val_metrics['conf_tool_mean']:.4f} conf_ct={val_metrics['conf_ct_mean']:.4f}"
    )


def print_val_by_combo(
    val_by_combo: Dict[str, Dict[str, float]],
    combo_only: bool = False,
    tool_only: bool = False,
    combo_probe_attn: bool = False,
) -> None:
    print("[val per combo]")
    for combo_name, m in val_by_combo.items():
        if tool_only:
            print(
                f"  {combo_name:24s} n={m['count']:4d} total={m['loss_total']:.5f} "
                f"tool_ce={m.get('loss_combo_tool_ce', 0.0):.5f} "
                f"tool_acc={m.get('tool_acc', 0.0):.4f}"
            )
        elif combo_probe_attn:
            print(
                f"  {combo_name:24s} n={m['count']:4d} total={m['loss_total']:.5f} "
                f"combo_ce={m['loss_combo_ce']:.5f} "
                f"combo_acc={m['combo_acc']:.4f}"
            )
        elif combo_only:
            print(
                f"  {combo_name:24s} n={m['count']:4d} total={m['loss_total']:.5f} "
                f"combo_ce={m['loss_combo_ce']:.5f} "
                f"tool_ce={m.get('loss_combo_tool_ce', 0.0):.5f} "
                f"action_ce={m.get('loss_combo_action_ce', 0.0):.5f} "
                f"combo_acc={m['combo_acc']:.4f} "
                f"tool_acc={m.get('tool_acc', 0.0):.4f} "
                f"action_acc={m.get('action_acc', 0.0):.4f}"
            )
        else:
            start_l1 = float(m.get("loss_start_l1", m.get("loss_start_bin_ce", 0.0)))
            end_l1 = float(m.get("loss_end_l1", m.get("loss_end_bin_ce", 0.0)))
            range_cls = float(m.get("loss_range_cls", 0.0))
            range_reg = float(m.get("loss_range_reg", 0.0))
            range_prop = float(m.get("loss_range_proposal", 0.0))
            print(
                f"  {combo_name:24s} n={m['count']:4d} total={m['loss_total']:.5f} "
                f"tool={m['loss_tool_xy']:.5f} ct={m['loss_ct_xy']:.5f} "
                f"combo_ce={m['loss_combo_ce']:.5f} "
                f"start_l1={start_l1:.5f} "
                f"end_l1={end_l1:.5f} "
                f"range={m['loss_range']:.5f} "
                f"range_cls={range_cls:.5f} range_reg={range_reg:.5f} "
                f"range_prop={range_prop:.5f} "
                f"combo_acc={m['combo_acc']:.4f} "
                f"traj_ade_px={m.get('traj_ade_px', 0.0):.3f} "
                f"traj_aj={m.get('traj_aj', 0.0):.4f} "
                f"traj_pck8={m.get('traj_pck_8', 0.0):.4f} "
                f"range_tiou={m.get('range_tiou', 0.0):.4f} "
                f"range_r05={m.get('range_r05', 0.0):.4f}"
            )


def aggregate_val_by_tool(
    val_by_combo: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    tool_sums: Dict[str, Dict[str, Any]] = {}
    for combo_name, m in val_by_combo.items():
        tool_name = str(combo_name).split("|", 1)[0].strip()
        if not tool_name:
            tool_name = str(combo_name)
        acc = tool_sums.setdefault(
            tool_name,
            {
                "combo_names": [],
                "count": 0.0,
                "traj_valid_count": 0.0,
                "traj_fde_valid_count": 0.0,
                "traj_survival_valid_count": 0.0,
                "traj_aj_valid_count": 0.0,
                "range_count": 0.0,
                "loss_total_sum": 0.0,
                "loss_tool_xy_sum": 0.0,
                "loss_ct_xy_sum": 0.0,
                "loss_tool_pair_sep_aux_sum": 0.0,
                "loss_combo_ce_sum": 0.0,
                "loss_combo_tool_ce_sum": 0.0,
                "loss_combo_action_ce_sum": 0.0,
                "loss_start_bin_ce_sum": 0.0,
                "loss_end_bin_ce_sum": 0.0,
                "loss_start_l1_sum": 0.0,
                "loss_end_l1_sum": 0.0,
                "loss_range_cls_sum": 0.0,
                "loss_range_reg_sum": 0.0,
                "loss_range_quality_sum": 0.0,
                "loss_range_sola_sum": 0.0,
                "loss_range_sum": 0.0,
                "combo_acc_sum": 0.0,
                "tool_acc_sum": 0.0,
                "action_acc_sum": 0.0,
                "traj_ade_sum": 0.0,
                "traj_fde_sum": 0.0,
                "traj_pck_2_sum": 0.0,
                "traj_pck_4_sum": 0.0,
                "traj_pck_8_sum": 0.0,
                "traj_pck_16_sum": 0.0,
                "traj_pck_auc_sum": 0.0,
                "traj_survival_sum": 0.0,
                "traj_aj_sum": 0.0,
                "range_tiou_sum": 0.0,
                "range_r03_sum": 0.0,
                "range_r05_sum": 0.0,
                "range_r07_sum": 0.0,
                "range_start_mae_norm_sum": 0.0,
                "range_end_mae_norm_sum": 0.0,
                "range_start_mae_abs_sum": 0.0,
                "range_end_mae_abs_sum": 0.0,
            },
        )
        acc["combo_names"].append(str(combo_name))
        count = float(m.get("count", 0.0))
        traj_c = float(m.get("traj_valid_count", count))
        traj_fde_c = float(m.get("traj_fde_valid_count", count))
        traj_surv_c = float(m.get("traj_survival_valid_count", count))
        traj_aj_c = float(m.get("traj_aj_valid_count", count))
        range_c = float(m.get("range_count", 0.0))

        acc["count"] += count
        acc["traj_valid_count"] += traj_c
        acc["traj_fde_valid_count"] += traj_fde_c
        acc["traj_survival_valid_count"] += traj_surv_c
        acc["traj_aj_valid_count"] += traj_aj_c
        acc["range_count"] += range_c
        acc["loss_total_sum"] += float(m.get("loss_total", 0.0)) * count
        acc["loss_tool_xy_sum"] += float(m.get("loss_tool_xy", 0.0)) * count
        acc["loss_ct_xy_sum"] += float(m.get("loss_ct_xy", 0.0)) * count
        acc["loss_tool_pair_sep_aux_sum"] += float(m.get("loss_tool_pair_sep_aux", 0.0)) * count
        acc["loss_combo_ce_sum"] += float(m.get("loss_combo_ce", 0.0)) * count
        acc["loss_combo_tool_ce_sum"] += float(m.get("loss_combo_tool_ce", 0.0)) * count
        acc["loss_combo_action_ce_sum"] += float(m.get("loss_combo_action_ce", 0.0)) * count
        acc["loss_start_bin_ce_sum"] += float(m.get("loss_start_bin_ce", 0.0)) * count
        acc["loss_end_bin_ce_sum"] += float(m.get("loss_end_bin_ce", 0.0)) * count
        acc["loss_start_l1_sum"] += float(m.get("loss_start_l1", m.get("loss_start_bin_ce", 0.0))) * count
        acc["loss_end_l1_sum"] += float(m.get("loss_end_l1", m.get("loss_end_bin_ce", 0.0))) * count
        acc["loss_range_cls_sum"] += float(m.get("loss_range_cls", 0.0)) * count
        acc["loss_range_reg_sum"] += float(m.get("loss_range_reg", 0.0)) * count
        acc["loss_range_quality_sum"] += float(m.get("loss_range_quality", 0.0)) * count
        acc["loss_range_sola_sum"] += float(m.get("loss_range_sola", 0.0)) * count
        acc["loss_range_sum"] += float(m.get("loss_range", 0.0)) * count
        acc["combo_acc_sum"] += float(m.get("combo_acc", 0.0)) * count
        acc["tool_acc_sum"] += float(m.get("tool_acc", 0.0)) * count
        acc["action_acc_sum"] += float(m.get("action_acc", 0.0)) * count
        acc["traj_ade_sum"] += float(m.get("traj_ade_px", 0.0)) * traj_c
        acc["traj_fde_sum"] += float(m.get("traj_fde_px", 0.0)) * traj_fde_c
        acc["traj_pck_2_sum"] += float(m.get("traj_pck_2", 0.0)) * traj_c
        acc["traj_pck_4_sum"] += float(m.get("traj_pck_4", 0.0)) * traj_c
        acc["traj_pck_8_sum"] += float(m.get("traj_pck_8", 0.0)) * traj_c
        acc["traj_pck_16_sum"] += float(m.get("traj_pck_16", 0.0)) * traj_c
        acc["traj_pck_auc_sum"] += float(m.get("traj_pck_auc", 0.0)) * traj_c
        acc["traj_survival_sum"] += float(m.get("traj_survival_50", 0.0)) * traj_surv_c
        acc["traj_aj_sum"] += float(m.get("traj_aj", 0.0)) * traj_aj_c
        acc["range_tiou_sum"] += float(m.get("range_tiou", 0.0)) * range_c
        acc["range_r03_sum"] += float(m.get("range_r03", 0.0)) * range_c
        acc["range_r05_sum"] += float(m.get("range_r05", 0.0)) * range_c
        acc["range_r07_sum"] += float(m.get("range_r07", 0.0)) * range_c
        acc["range_start_mae_norm_sum"] += float(m.get("range_start_mae_norm", 0.0)) * range_c
        acc["range_end_mae_norm_sum"] += float(m.get("range_end_mae_norm", 0.0)) * range_c
        acc["range_start_mae_abs_sum"] += float(m.get("range_start_mae_abs", 0.0)) * range_c
        acc["range_end_mae_abs_sum"] += float(m.get("range_end_mae_abs", 0.0)) * range_c

    by_tool: Dict[str, Dict[str, Any]] = {}
    for tool_name in sorted(tool_sums.keys()):
        acc = tool_sums[tool_name]
        count = max(float(acc.get("count", 0.0)), 1.0)
        traj_c = max(float(acc.get("traj_valid_count", 0.0)), 1.0)
        traj_fde_c = max(float(acc.get("traj_fde_valid_count", 0.0)), 1.0)
        traj_surv_c = max(float(acc.get("traj_survival_valid_count", 0.0)), 1.0)
        traj_aj_c = max(float(acc.get("traj_aj_valid_count", 0.0)), 1.0)
        range_c = max(float(acc.get("range_count", 0.0)), 1.0)
        by_tool[tool_name] = {
            "tool": str(tool_name),
            "combo_names": sorted([str(x) for x in acc.get("combo_names", [])]),
            "count": int(round(float(acc.get("count", 0.0)))),
            "traj_valid_count": float(acc.get("traj_valid_count", 0.0)),
            "traj_fde_valid_count": float(acc.get("traj_fde_valid_count", 0.0)),
            "traj_survival_valid_count": float(acc.get("traj_survival_valid_count", 0.0)),
            "traj_aj_valid_count": float(acc.get("traj_aj_valid_count", 0.0)),
            "range_count": float(acc.get("range_count", 0.0)),
            "loss_total": float(acc.get("loss_total_sum", 0.0)) / count,
            "loss_tool_xy": float(acc.get("loss_tool_xy_sum", 0.0)) / count,
            "loss_ct_xy": float(acc.get("loss_ct_xy_sum", 0.0)) / count,
            "loss_tool_pair_sep_aux": float(acc.get("loss_tool_pair_sep_aux_sum", 0.0)) / count,
            "loss_combo_ce": float(acc.get("loss_combo_ce_sum", 0.0)) / count,
            "loss_combo_tool_ce": float(acc.get("loss_combo_tool_ce_sum", 0.0)) / count,
            "loss_combo_action_ce": float(acc.get("loss_combo_action_ce_sum", 0.0)) / count,
            "loss_start_bin_ce": float(acc.get("loss_start_bin_ce_sum", 0.0)) / count,
            "loss_end_bin_ce": float(acc.get("loss_end_bin_ce_sum", 0.0)) / count,
            "loss_start_l1": float(acc.get("loss_start_l1_sum", 0.0)) / count,
            "loss_end_l1": float(acc.get("loss_end_l1_sum", 0.0)) / count,
            "loss_range_cls": float(acc.get("loss_range_cls_sum", 0.0)) / count,
            "loss_range_reg": float(acc.get("loss_range_reg_sum", 0.0)) / count,
            "loss_range_quality": float(acc.get("loss_range_quality_sum", 0.0)) / count,
            "loss_range_sola": float(acc.get("loss_range_sola_sum", 0.0)) / count,
            "loss_range": float(acc.get("loss_range_sum", 0.0)) / count,
            "combo_acc": float(acc.get("combo_acc_sum", 0.0)) / count,
            "tool_acc": float(acc.get("tool_acc_sum", 0.0)) / count,
            "action_acc": float(acc.get("action_acc_sum", 0.0)) / count,
            "traj_ade_px": float(acc.get("traj_ade_sum", 0.0)) / traj_c,
            "traj_fde_px": float(acc.get("traj_fde_sum", 0.0)) / traj_fde_c,
            "traj_pck_2": float(acc.get("traj_pck_2_sum", 0.0)) / traj_c,
            "traj_pck_4": float(acc.get("traj_pck_4_sum", 0.0)) / traj_c,
            "traj_pck_8": float(acc.get("traj_pck_8_sum", 0.0)) / traj_c,
            "traj_pck_16": float(acc.get("traj_pck_16_sum", 0.0)) / traj_c,
            "traj_pck_auc": float(acc.get("traj_pck_auc_sum", 0.0)) / traj_c,
            "traj_survival_50": float(acc.get("traj_survival_sum", 0.0)) / traj_surv_c,
            "traj_aj": float(acc.get("traj_aj_sum", 0.0)) / traj_aj_c,
            "range_tiou": float(acc.get("range_tiou_sum", 0.0)) / range_c,
            "range_r03": float(acc.get("range_r03_sum", 0.0)) / range_c,
            "range_r05": float(acc.get("range_r05_sum", 0.0)) / range_c,
            "range_r07": float(acc.get("range_r07_sum", 0.0)) / range_c,
            "range_start_mae_norm": float(acc.get("range_start_mae_norm_sum", 0.0)) / range_c,
            "range_end_mae_norm": float(acc.get("range_end_mae_norm_sum", 0.0)) / range_c,
            "range_start_mae_abs": float(acc.get("range_start_mae_abs_sum", 0.0)) / range_c,
            "range_end_mae_abs": float(acc.get("range_end_mae_abs_sum", 0.0)) / range_c,
        }
    return by_tool


def _zero_train_metrics() -> Dict[str, float]:
    out = {
        "loss_total": 0.0,
        "loss_tool_xy": 0.0,
        "loss_ct_xy": 0.0,
        "loss_tool_centroid_aux": 0.0,
        "loss_tool_offset_aux": 0.0,
        "loss_tool_pair_sep_aux": 0.0,
        "loss_tool_heatmap_ce": 0.0,
        "loss_ct_heatmap_ce": 0.0,
        "loss_combo_ce": 0.0,
        "loss_combo_tool_ce": 0.0,
        "loss_combo_action_ce": 0.0,
        "loss_start_bin_ce": 0.0,
        "loss_end_bin_ce": 0.0,
        "loss_start_l1": 0.0,
        "loss_end_l1": 0.0,
        "loss_range_cls": 0.0,
        "loss_range_reg": 0.0,
        "loss_range_quality": 0.0,
        "loss_range_sola": 0.0,
        "loss_range": 0.0,
        "combo_acc": 0.0,
        "tool_acc": 0.0,
        "action_acc": 0.0,
        "traj_ade_px": 0.0,
        "traj_fde_px": 0.0,
        "traj_pck_2": 0.0,
        "traj_pck_4": 0.0,
        "traj_pck_8": 0.0,
        "traj_pck_16": 0.0,
        "traj_pck_auc": 0.0,
        "traj_survival_50": 0.0,
        "traj_aj": 0.0,
        "range_tiou": 0.0,
        "range_r03": 0.0,
        "range_r05": 0.0,
        "range_r07": 0.0,
        "range_start_mae_norm": 0.0,
        "range_end_mae_norm": 0.0,
        "range_center_mae_norm": 0.0,
        "range_duration_mae_norm": 0.0,
        "range_start_mae_abs": 0.0,
        "range_end_mae_abs": 0.0,
        "range_count": 0.0,
        "refine_gate_mean": 0.0,
        "refine_delta_abs_mean": 0.0,
        "refine_centroid_shift_abs_mean": 0.0,
        "anchor_delta_centroid_abs_mean": 0.0,
        "anchor_alpha_mean": 0.0,
        "anchor_verify_scale_mean": 0.0,
        "anchor_verify_residual_mean": 0.0,
        "stage0_centroid_noise_abs_mean": 0.0,
        "factor_anchor_gate_mean": 0.0,
        "factor_offset_gate_mean": 0.0,
        "factor_delta_offset_abs_mean": 0.0,
        "corr_entropy_mean": 0.0,
        "corr_top1_minus_top2_mean": 0.0,
        "context_entropy_mean": 0.0,
        "context_top1_minus_top2_mean": 0.0,
        "context_offset_abs_mean": 0.0,
        "context_gate_mean": 0.0,
        "joint_gate_mean": 0.0,
        "template_entropy_mean": 0.0,
        "template_top1_minus_top2_mean": 0.0,
        "template_gate_mean": 0.0,
        "template_offset_abs_mean": 0.0,
        "template_delta_abs_mean": 0.0,
        "conf_tool_mean": 0.0,
        "conf_ct_mean": 0.0,
        "count": 0.0,
    }
    for stage_idx in _TRAJ_STAGE_IDS:
        out[f"loss_tool_xy_stage{stage_idx}"] = 0.0
        out[f"loss_ct_xy_stage{stage_idx}"] = 0.0
    for key in _TRAJ_LATE_FOCAL_METRIC_KEYS:
        out[key] = 0.0
    return out


def _resolve_split_file_path(data_root: Path, split: str, split_file: str) -> Path:
    if split_file.strip():
        p = Path(split_file)
        if not p.is_absolute():
            p = data_root / p
        return p
    return data_root / f"{split}.txt"


def _read_split_ids(split_file: Path) -> List[str]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    return [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_split_ids(path: Path, sids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join([str(x) for x in sids])
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _load_sid_case_combo_map_for_reports(data_root: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    meta_path = data_root / "meta.json"
    if not meta_path.is_file():
        return out
    try:
        arr = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(arr, list):
        return out
    for row in arr:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("sid", "")).strip()
        if not sid:
            continue
        tool = str(row.get("tool", "")).strip()
        action = str(row.get("action", "")).strip()
        combo = str(row.get("combo", "")).strip()
        if not combo:
            combo = f"{tool}|{action}" if (tool or action) else "unknown"
        out[sid] = {
            "case": str(row.get("case", "unknown")),
            "combo": combo,
        }
    return out


def _build_split_case_combo_composition_report(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    sid_case_combo: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    split_map = {
        "train": list(train_ids),
        "val": list(val_ids),
        "test": list(test_ids),
    }
    per_split_case_counts: Dict[str, Dict[str, int]] = {k: {} for k in split_map.keys()}
    per_split_combo_counts: Dict[str, Dict[str, int]] = {k: {} for k in split_map.keys()}
    per_split_case_combo_counts: Dict[str, Dict[str, Dict[str, int]]] = {k: {} for k in split_map.keys()}
    sid_details: List[Dict[str, str]] = []

    for split, sids in split_map.items():
        for sid in sids:
            cc = sid_case_combo.get(str(sid), {})
            case = str(cc.get("case", "unknown"))
            combo = str(cc.get("combo", "unknown"))
            per_split_case_counts[split][case] = int(per_split_case_counts[split].get(case, 0) + 1)
            per_split_combo_counts[split][combo] = int(per_split_combo_counts[split].get(combo, 0) + 1)
            case_combo = per_split_case_combo_counts[split].setdefault(case, {})
            case_combo[combo] = int(case_combo.get(combo, 0) + 1)
            sid_details.append(
                {
                    "sid": str(sid),
                    "split": str(split),
                    "case": case,
                    "combo": combo,
                }
            )

    rows: List[Dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for case in sorted(per_split_case_combo_counts[split].keys()):
            cc = per_split_case_combo_counts[split][case]
            for combo in sorted(cc.keys()):
                rows.append(
                    {
                        "split": split,
                        "case": case,
                        "combo": combo,
                        "count": int(cc[combo]),
                    }
                )
    sid_details = sorted(
        sid_details,
        key=lambda x: (x["split"], x["case"], x["combo"], x["sid"]),
    )
    return {
        "split_totals": {
            "train": int(len(train_ids)),
            "val": int(len(val_ids)),
            "test": int(len(test_ids)),
            "all": int(len(train_ids) + len(val_ids) + len(test_ids)),
        },
        "per_split_case_counts": {
            k: {kk: int(vv) for kk, vv in sorted(v.items(), key=lambda kv: kv[0])}
            for k, v in per_split_case_counts.items()
        },
        "per_split_combo_counts": {
            k: {kk: int(vv) for kk, vv in sorted(v.items(), key=lambda kv: kv[0])}
            for k, v in per_split_combo_counts.items()
        },
        "rows": rows,
        "sid_details": sid_details,
    }


def _write_fold_split_composition_report(
    split_root: Path,
    fold_idx: int,
    report: Dict[str, Any],
) -> Tuple[Path, Path, Path]:
    json_path = split_root / f"fold_{int(fold_idx):02d}_split_composition.json"
    tsv_path = split_root / f"fold_{int(fold_idx):02d}_split_composition.tsv"
    sid_tsv_path = split_root / f"fold_{int(fold_idx):02d}_split_sid_details.tsv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("split\tcase\tcombo\tcount\n")
        for row in report.get("rows", []):
            f.write(f"{row['split']}\t{row['case']}\t{row['combo']}\t{int(row['count'])}\n")

    with open(sid_tsv_path, "w", encoding="utf-8") as f:
        f.write("sid\tsplit\tcase\tcombo\n")
        for row in report.get("sid_details", []):
            f.write(f"{row['sid']}\t{row['split']}\t{row['case']}\t{row['combo']}\n")
    return json_path, tsv_path, sid_tsv_path


def _build_case_disjoint_combo_stratified_folds(
    data_root: Path,
    sids: List[str],
    sid_case_combo: Dict[str, Dict[str, str]],
    n_folds: int,
    seed: int,
) -> Dict[str, Any]:
    """Build case-disjoint folds with approximate combo balancing.

    Split unit is `case`:
      - all samples from one case go to exactly one fold
      - val/train case sets are strictly disjoint per fold
    """
    if int(n_folds) <= 1:
        raise ValueError("n_folds must be > 1 for case-disjoint fold build.")
    if len(sids) < int(n_folds):
        raise ValueError(f"Not enough samples for n_folds={n_folds}: n_samples={len(sids)}")

    case_to_sids: Dict[str, List[str]] = defaultdict(list)
    case_to_combo_counts: Dict[str, Counter] = defaultdict(Counter)
    combo_global = Counter()
    sid_to_case_resolved: Dict[str, str] = {}
    unknown_case_sids = 0

    for sid in sids:
        sid_str = str(sid)
        cc = sid_case_combo.get(sid_str, {})
        case_name = str(cc.get("case", "")).strip()
        if (not case_name) or (case_name.lower() == "unknown"):
            # Keep unknown samples case-disjoint by assigning one synthetic case per sid.
            case_name = f"__unknown_case__{sid_str}"
            unknown_case_sids += 1

        combo_name = str(cc.get("combo", "")).strip()
        if not combo_name:
            combo_np = np.load(data_root / "combo_ids" / f"{sid_str}.npy")
            combo_id = int(combo_np.reshape(-1)[0])
            combo_name = f"combo_id_{combo_id}"

        sid_to_case_resolved[sid_str] = case_name
        case_to_sids[case_name].append(sid_str)
        case_to_combo_counts[case_name][combo_name] += 1
        combo_global[combo_name] += 1

    all_cases = sorted(case_to_sids.keys())
    if len(all_cases) < int(n_folds):
        raise ValueError(
            f"Not enough unique cases for n_folds={n_folds}: n_cases={len(all_cases)}"
        )

    target_n = float(len(sids)) / float(int(n_folds))
    target_case_n = float(len(all_cases)) / float(int(n_folds))
    combo_targets = {k: float(v) / float(int(n_folds)) for k, v in combo_global.items()}

    rng = random.Random(int(seed))
    case_order = list(all_cases)
    rng.shuffle(case_order)
    case_order.sort(
        key=lambda c: (
            -len(case_to_sids[c]),
            -sum(int(v) for v in case_to_combo_counts[c].values()),
            c,
        )
    )

    fold_cases: List[List[str]] = [[] for _ in range(int(n_folds))]
    fold_combo_counts: List[Counter] = [Counter() for _ in range(int(n_folds))]
    fold_sample_counts = [0 for _ in range(int(n_folds))]

    # Seed each fold with one case so no fold ends up empty.
    for fi in range(int(n_folds)):
        seed_case = case_order[fi]
        fold_cases[fi].append(seed_case)
        fold_combo_counts[fi].update(case_to_combo_counts[seed_case])
        fold_sample_counts[fi] += int(len(case_to_sids[seed_case]))

    for case_name in case_order[int(n_folds):]:
        case_n = int(len(case_to_sids[case_name]))
        case_combo = case_to_combo_counts[case_name]

        best_fold = 0
        best_score = None
        best_tie = None
        for fi in range(int(n_folds)):
            n_after = float(fold_sample_counts[fi] + case_n)
            size_penalty = ((n_after - target_n) / max(target_n, 1.0)) ** 2
            case_penalty = ((float(len(fold_cases[fi]) + 1) - target_case_n) / max(target_case_n, 1.0)) ** 2

            combo_penalty = 0.0
            for combo_name, cnt in case_combo.items():
                # Avoid over-penalizing extremely rare combos in fold assignment.
                target = max(float(combo_targets.get(combo_name, 0.0)), 10.0)
                after = float(fold_combo_counts[fi].get(combo_name, 0) + int(cnt))
                combo_penalty += ((after - target) / target) ** 2

            # Prioritize fold size/case-count balance; combo acts only as a tie-breaker.
            score = 1.25 * size_penalty + 0.75 * case_penalty
            tie = (fold_sample_counts[fi], len(fold_cases[fi]), combo_penalty, fi)
            if (best_score is None) or (score < best_score) or (abs(score - best_score) <= 1e-12 and tie < best_tie):
                best_score = score
                best_fold = fi
                best_tie = tie

        fold_cases[best_fold].append(case_name)
        fold_combo_counts[best_fold].update(case_combo)
        fold_sample_counts[best_fold] += case_n

    # Rebalance by case-count so each fold has enough validation cases.
    min_cases_per_fold = max(1, int(len(all_cases) // int(n_folds)))
    max_rebalance_steps = max(int(len(all_cases) * 4), 1)
    for _ in range(max_rebalance_steps):
        case_counts = [int(len(x)) for x in fold_cases]
        low_idx = int(min(range(int(n_folds)), key=lambda i: case_counts[i]))
        high_idx = int(max(range(int(n_folds)), key=lambda i: case_counts[i]))
        if case_counts[low_idx] >= min_cases_per_fold:
            break
        if case_counts[high_idx] <= max(min_cases_per_fold, 1):
            break
        if low_idx == high_idx:
            break

        moved = False
        donor_cases = sorted(
            [c for c in fold_cases[high_idx]],
            key=lambda c: (len(case_to_sids[c]), c),
        )
        for case_name in donor_cases:
            if len(fold_cases[high_idx]) <= 1:
                break
            case_n = int(len(case_to_sids[case_name]))
            cur_dev = abs(float(fold_sample_counts[low_idx]) - target_n) + abs(float(fold_sample_counts[high_idx]) - target_n)
            next_dev = abs(float(fold_sample_counts[low_idx] + case_n) - target_n) + abs(
                float(fold_sample_counts[high_idx] - case_n) - target_n
            )
            # Allow slight sample-imbalance increase to improve case balance.
            if next_dev <= (cur_dev + max(0.05 * target_n, 5.0)):
                fold_cases[high_idx].remove(case_name)
                fold_cases[low_idx].append(case_name)
                fold_combo_counts[high_idx].subtract(case_to_combo_counts[case_name])
                for k in list(fold_combo_counts[high_idx].keys()):
                    if fold_combo_counts[high_idx][k] <= 0:
                        del fold_combo_counts[high_idx][k]
                fold_combo_counts[low_idx].update(case_to_combo_counts[case_name])
                fold_sample_counts[high_idx] -= case_n
                fold_sample_counts[low_idx] += case_n
                moved = True
                break
        if not moved:
            break

    fold_val_ids: List[List[str]] = []
    fold_val_cases: List[List[str]] = []
    for fi in range(int(n_folds)):
        cases = sorted(fold_cases[fi])
        sids_fold: List[str] = []
        for case_name in cases:
            sids_fold.extend(case_to_sids[case_name])
        sids_fold = sorted([str(x) for x in sids_fold])
        if not sids_fold:
            raise RuntimeError(f"Fold {fi + 1} has no validation samples; cannot continue.")
        fold_val_cases.append(cases)
        fold_val_ids.append(sids_fold)

    return {
        "fold_val_ids": fold_val_ids,
        "fold_val_cases": fold_val_cases,
        "sid_to_case_resolved": sid_to_case_resolved,
        "all_cases": sorted(all_cases),
        "unknown_case_sid_count": int(unknown_case_sids),
    }


def _strip_cli_args_for_nfold(raw_args: List[str]) -> List[str]:
    # Remove args that must be controlled by n-fold orchestrator.
    options_with_value = {
        "--n-folds",
        "--n-fold-epochs",
        "--output-dir",
        "--train-split-file",
        "--val-split-file",
        "--resume",
        "--epochs",
        "--save-interval",
        "--save-every",
        "--tb-dir",
    }
    out: List[str] = []
    i = 0
    while i < len(raw_args):
        a = raw_args[i]
        if not a.startswith("--"):
            out.append(a)
            i += 1
            continue
        key = a.split("=", 1)[0]
        if key in options_with_value:
            if "=" in a:
                i += 1
                continue
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def _cleanup_cache_split_dir(split_dir: Path, keep_sids: set[str]) -> Dict[str, int]:
    out = {
        "files_seen": 0,
        "files_removed": 0,
        "bytes_removed": 0,
        "errors": 0,
    }
    if not split_dir.is_dir():
        return out
    for p in split_dir.glob("*.npy"):
        if not p.is_file():
            continue
        out["files_seen"] += 1
        sid = str(p.stem)
        if sid in keep_sids:
            continue
        try:
            size_bytes = int(p.stat().st_size)
        except Exception:
            size_bytes = 0
        try:
            p.unlink()
            out["files_removed"] += 1
            out["bytes_removed"] += size_bytes
        except Exception:
            out["errors"] += 1
    return out


def _cleanup_nfold_cache_root(
    *,
    cache_root: Path,
    keep_train_sids: set[str],
    keep_val_sids: set[str],
) -> Dict[str, Any]:
    train_stats = _cleanup_cache_split_dir(cache_root / "train", keep_train_sids)
    val_stats = _cleanup_cache_split_dir(cache_root / "val", keep_val_sids)
    removed_files = int(train_stats["files_removed"]) + int(val_stats["files_removed"])
    removed_bytes = int(train_stats["bytes_removed"]) + int(val_stats["bytes_removed"])
    return {
        "cache_root": str(cache_root),
        "keep_train": int(len(keep_train_sids)),
        "keep_val": int(len(keep_val_sids)),
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
        "errors": int(train_stats["errors"]) + int(val_stats["errors"]),
        "train": train_stats,
        "val": val_stats,
    }


def _load_fold_val_metrics(fold_out_dir: Path, n_fold_epochs: int) -> Dict[str, Any]:
    ckpt_dir = fold_out_dir / "checkpoints"
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / f"epoch_{int(n_fold_epochs):03d}.pt"

    pick = best_path if best_path.is_file() else last_path
    if not pick.is_file():
        raise FileNotFoundError(
            f"No fold checkpoint found under {ckpt_dir} (expected {best_path.name} or {last_path.name})."
        )
    ckpt = torch.load(pick, map_location="cpu")
    val_metrics = dict(ckpt.get("val_metrics", {}))
    val_by_combo = dict(ckpt.get("val_by_combo", {}))
    val_by_tool = dict(ckpt.get("val_by_tool", {}))
    if (not val_by_tool) and val_by_combo:
        val_by_tool = aggregate_val_by_tool(val_by_combo)
    return {
        "picked_checkpoint": str(pick),
        "epoch": int(ckpt.get("epoch", -1)),
        "global_step": int(ckpt.get("global_step", 0)),
        "val_metrics": val_metrics,
        "val_by_combo": val_by_combo,
        "val_by_tool": val_by_tool,
    }


def _compute_nfold_average(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_keys = [
        "val_total",
        "val_combo_acc",
            "val_traj_ade_px",
            "val_traj_aj",
            "val_traj_pck_8",
            "val_range_tiou",
            "val_range_r05",
        "val_tool",
        "val_ct",
    ]
    out: Dict[str, Any] = {
        "fold_count": int(len(rows)),
        "train_n_total": int(sum(int(r.get("train_n", 0)) for r in rows)),
        "val_n_total": int(sum(int(r.get("val_n", 0)) for r in rows)),
        "train_case_n_total": int(sum(int(r.get("train_case_n", 0)) for r in rows)),
        "val_case_n_total": int(sum(int(r.get("val_case_n", 0)) for r in rows)),
        "train_n_mean": float(np.mean([int(r.get("train_n", 0)) for r in rows])) if rows else float("nan"),
        "val_n_mean": float(np.mean([int(r.get("val_n", 0)) for r in rows])) if rows else float("nan"),
        "train_case_n_mean": float(np.mean([int(r.get("train_case_n", 0)) for r in rows])) if rows else float("nan"),
        "val_case_n_mean": float(np.mean([int(r.get("val_case_n", 0)) for r in rows])) if rows else float("nan"),
    }
    for k in metric_keys:
        vals = [float(r.get(k, float("nan"))) for r in rows]
        vals = [v for v in vals if np.isfinite(v)]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def _write_nfold_summary(base_out_dir: Path, rows: List[Dict[str, Any]]) -> Tuple[Path, Path, Path]:
    json_path = base_out_dir / "nfold_results_summary.json"
    tsv_path = base_out_dir / "nfold_results_summary.tsv"
    avg = _compute_nfold_average(rows)
    json_path.write_text(
        json.dumps({"folds": rows, "average": avg}, indent=2),
        encoding="utf-8",
    )

    header = [
        "fold",
        "train_n",
        "val_n",
        "train_case_n",
        "val_case_n",
        "best_epoch",
        "val_total",
        "val_combo_acc",
        "val_traj_ade_px",
        "val_traj_aj",
        "val_traj_pck_8",
        "val_range_tiou",
        "val_range_r05",
        "val_tool",
        "val_ct",
        "picked_checkpoint",
    ]
    lines = ["\t".join(header)]
    for r in rows:
        lines.append(
            "\t".join(
                [
                    str(r.get("fold", "")),
                    str(r.get("train_n", "")),
                    str(r.get("val_n", "")),
                    str(r.get("train_case_n", "")),
                    str(r.get("val_case_n", "")),
                    str(r.get("best_epoch", "")),
                    f"{float(r.get('val_total', float('nan'))):.6f}",
                    f"{float(r.get('val_combo_acc', float('nan'))):.6f}",
                    f"{float(r.get('val_traj_ade_px', float('nan'))):.6f}",
                    f"{float(r.get('val_traj_aj', float('nan'))):.6f}",
                    f"{float(r.get('val_traj_pck_8', float('nan'))):.6f}",
                    f"{float(r.get('val_range_tiou', float('nan'))):.6f}",
                    f"{float(r.get('val_range_r05', float('nan'))):.6f}",
                    f"{float(r.get('val_tool', float('nan'))):.6f}",
                    f"{float(r.get('val_ct', float('nan'))):.6f}",
                    str(r.get("picked_checkpoint", "")),
                ]
            )
        )
    lines.append(
        "\t".join(
            [
                "avg",
                f"{float(avg.get('train_n_mean', float('nan'))):.1f}",
                f"{float(avg.get('val_n_mean', float('nan'))):.1f}",
                f"{float(avg.get('train_case_n_mean', float('nan'))):.1f}",
                f"{float(avg.get('val_case_n_mean', float('nan'))):.1f}",
                "",
                f"{float(avg.get('val_total', float('nan'))):.6f}",
                f"{float(avg.get('val_combo_acc', float('nan'))):.6f}",
                f"{float(avg.get('val_traj_ade_px', float('nan'))):.6f}",
                f"{float(avg.get('val_traj_aj', float('nan'))):.6f}",
                f"{float(avg.get('val_traj_pck_8', float('nan'))):.6f}",
                f"{float(avg.get('val_range_tiou', float('nan'))):.6f}",
                f"{float(avg.get('val_range_r05', float('nan'))):.6f}",
                f"{float(avg.get('val_tool', float('nan'))):.6f}",
                f"{float(avg.get('val_ct', float('nan'))):.6f}",
                "",
            ]
        )
    )
    tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    avg_json_path = base_out_dir / "nfold_results_average.json"
    avg_json_path.write_text(json.dumps(avg, indent=2), encoding="utf-8")
    return json_path, tsv_path, avg_json_path


def _run_nfold_cls3_report(
    *,
    args: argparse.Namespace,
    base_out_dir: Path,
    stage_tag: str,
) -> Optional[Dict[str, Any]]:
    if not bool(getattr(args, "nfold_cls3_report", True)):
        return None
    script_path = Path(__file__).resolve().parent / "analyze_nfold_cls3.py"
    if not script_path.is_file():
        msg = f"[nfold] cls3 report skipped: missing script {script_path}"
        if bool(getattr(args, "nfold_cls3_strict", False)):
            raise FileNotFoundError(msg)
        print(msg)
        return None
    fpfn_files = sorted(base_out_dir.glob("fold_*/checkpoints/val_combo_fpfn_epoch_*.json"))
    if len(fpfn_files) == 0:
        print(f"[nfold] cls3 report skipped ({stage_tag}): no val_combo_fpfn files yet under {base_out_dir}")
        return None

    target_epoch = int(getattr(args, "nfold_cls3_target_epoch", 0))
    if target_epoch <= 0:
        target_epoch = int(getattr(args, "n_fold_epochs", 0))
    cmd = [
        sys.executable,
        str(script_path),
        "--nfold-root",
        str(base_out_dir),
        "--out-dir",
        str(base_out_dir),
        "--fold-glob",
        "fold_*",
        "--combo-fpfn-pattern",
        "val_combo_fpfn_epoch_*.json",
        "--target-epoch",
        str(target_epoch),
        "--top-k",
        str(int(getattr(args, "nfold_cls3_top_k", 20))),
        "--output-prefix",
        str(getattr(args, "nfold_cls3_output_prefix", "nfold_cls3")),
    ]
    if bool(getattr(args, "nfold_cls3_write_markdown", True)):
        cmd.append("--write-markdown")
    if bool(getattr(args, "nfold_cls3_debug", False)):
        cmd.append("--debug")
    if bool(getattr(args, "nfold_cls3_strict", False)):
        cmd.append("--strict")
    else:
        cmd.append("--no-strict")

    try:
        if bool(getattr(args, "nfold_cls3_debug", False)):
            print(f"[nfold] cls3 report ({stage_tag}) cmd: {' '.join(cmd)}")
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if bool(getattr(args, "nfold_cls3_debug", False)):
            out_txt = (proc.stdout or "").strip()
            if out_txt:
                for ln in out_txt.splitlines():
                    print(f"[nfold][cls3] {ln}")
        summary_json = base_out_dir / f"{str(getattr(args, 'nfold_cls3_output_prefix', 'nfold_cls3'))}_summary.json"
        if summary_json.is_file():
            print(f"[nfold] cls3 report updated ({stage_tag}): {summary_json}")
        return {
            "stage": str(stage_tag),
            "cmd": cmd,
            "summary_json": str(summary_json),
        }
    except subprocess.CalledProcessError as e:
        stdout_txt = (e.stdout or "").strip()
        stderr_txt = (e.stderr or "").strip()
        msg = (
            f"[nfold] cls3 report failed at {stage_tag} (exit={int(e.returncode)}): "
            f"stdout='{stdout_txt[:400]}' stderr='{stderr_txt[:400]}'"
        )
        if bool(getattr(args, "nfold_cls3_strict", False)):
            raise RuntimeError(msg) from e
        print(msg)
        return None


def _run_nfold_training(args: argparse.Namespace) -> None:
    if int(args.n_folds) <= 1:
        raise ValueError("n-fold mode requires --n-folds > 1")
    if args.resume:
        raise ValueError("--resume is not supported together with n-fold mode.")

    data_root = Path(args.data_root)
    base_out_dir = Path(args.output_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    (base_out_dir / "nfold_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    base_train_file = _resolve_split_file_path(data_root=data_root, split="train", split_file=str(args.train_split_file))
    base_val_file = _resolve_split_file_path(data_root=data_root, split="val", split_file=str(args.val_split_file))
    base_train_ids = _read_split_ids(base_train_file)
    base_val_ids = _read_split_ids(base_val_file)
    all_ids = sorted(set([str(x) for x in base_train_ids] + [str(x) for x in base_val_ids]))
    if not all_ids:
        raise RuntimeError("n-fold source ids are empty (train+val union).")
    sid_case_combo = _load_sid_case_combo_map_for_reports(data_root=data_root)

    fold_plan = _build_case_disjoint_combo_stratified_folds(
        data_root=data_root,
        sids=all_ids,
        sid_case_combo=sid_case_combo,
        n_folds=int(args.n_folds),
        seed=int(args.seed),
    )
    folds_val_ids = list(fold_plan["fold_val_ids"])
    folds_val_cases = list(fold_plan["fold_val_cases"])
    sid_to_case_resolved = dict(fold_plan["sid_to_case_resolved"])
    all_cases = sorted([str(x) for x in fold_plan["all_cases"]])
    all_case_set = set(all_cases)
    split_root = base_out_dir / "nfold_splits"
    split_root.mkdir(parents=True, exist_ok=True)
    folds_manifest: List[Dict[str, Any]] = []

    print("[nfold]")
    print(f"  n_folds={int(args.n_folds)} epochs_per_fold={int(args.n_fold_epochs)}")
    print("  split_unit=case (strict train/val case disjoint)")
    print(f"  source_train={base_train_file}")
    print(f"  source_val={base_val_file}")
    print(f"  source_union_samples={len(all_ids)}")
    print(f"  source_union_cases={len(all_cases)}")
    print(f"  source_unknown_case_sids={int(fold_plan.get('unknown_case_sid_count', 0))}")

    raw_args = list(sys.argv[1:])
    child_args = _strip_cli_args_for_nfold(raw_args)
    if "--no-init-eval-vis" not in child_args:
        child_args.append("--no-init-eval-vis")

    use_shared_cache = bool(args.vjepa_cache) and (str(args.vjepa_cache_dir).strip() == "")
    nfold_cache_dir: Optional[Path] = None
    nfold_cache_gc_enabled = False
    if bool(args.vjepa_cache):
        if use_shared_cache:
            nfold_cache_dir = base_out_dir / "nfold_shared_vjepa_cache"
            nfold_cache_gc_enabled = True
        else:
            nfold_cache_dir = Path(args.vjepa_cache_dir)
    fold_train_ids_plan: List[List[str]] = []
    fold_val_ids_plan: List[List[str]] = []
    for fi in range(int(args.n_folds)):
        val_ids_i = sorted([str(x) for x in folds_val_ids[fi]])
        val_set_i = set(val_ids_i)
        train_ids_i = sorted([sid for sid in all_ids if sid not in val_set_i])
        fold_train_ids_plan.append(train_ids_i)
        fold_val_ids_plan.append(val_ids_i)
    remain_train_uses = Counter()
    remain_val_uses = Counter()
    for ids_i in fold_train_ids_plan:
        remain_train_uses.update([str(x) for x in ids_i])
    for ids_i in fold_val_ids_plan:
        remain_val_uses.update([str(x) for x in ids_i])
    cache_gc_events: List[Dict[str, Any]] = []
    cls3_report_events: List[Dict[str, Any]] = []

    rows: List[Dict[str, Any]] = []
    all_id_set = set(all_ids)
    if nfold_cache_dir is not None:
        print(
            f"  nfold_cache_root={nfold_cache_dir} "
            f"(shared={use_shared_cache} gc={nfold_cache_gc_enabled})"
        )
    if nfold_cache_gc_enabled and (nfold_cache_dir is not None):
        keep_train_initial = set([str(k) for k, v in remain_train_uses.items() if int(v) > 0])
        keep_val_initial = set([str(k) for k, v in remain_val_uses.items() if int(v) > 0])
        gc_report = _cleanup_nfold_cache_root(
            cache_root=nfold_cache_dir,
            keep_train_sids=keep_train_initial,
            keep_val_sids=keep_val_initial,
        )
        gc_report["event"] = "initial"
        cache_gc_events.append(gc_report)
        print(
            f"  nfold_cache: root={nfold_cache_dir} reuse=True gc=True "
            f"removed_files={gc_report['removed_files']} "
            f"removed_mb={gc_report['removed_bytes'] / (1024.0 * 1024.0):.1f} "
            f"errors={gc_report['errors']}"
        )
    for fi in range(int(args.n_folds)):
        fold_idx = fi + 1
        val_ids = list(fold_val_ids_plan[fi])
        val_cases = sorted([str(x) for x in folds_val_cases[fi]])
        val_case_set = set(val_cases)
        train_cases = sorted([c for c in all_cases if c not in val_case_set])
        val_set = set(val_ids)
        train_ids = list(fold_train_ids_plan[fi])
        if not train_ids or not val_ids:
            raise RuntimeError(f"Fold {fold_idx}: empty train/val split (train={len(train_ids)} val={len(val_ids)}).")
        if not train_cases or not val_cases:
            raise RuntimeError(
                f"Fold {fold_idx}: empty train/val case split (train_cases={len(train_cases)} val_cases={len(val_cases)})."
            )

        fold_train_file = split_root / f"fold_{fold_idx:02d}_train.txt"
        fold_val_file = split_root / f"fold_{fold_idx:02d}_val.txt"
        fold_test_file = split_root / f"fold_{fold_idx:02d}_test.txt"
        fold_train_cases_file = split_root / f"fold_{fold_idx:02d}_train_cases.txt"
        fold_val_cases_file = split_root / f"fold_{fold_idx:02d}_val_cases.txt"
        _write_split_ids(fold_train_file, train_ids)
        _write_split_ids(fold_val_file, val_ids)
        _write_split_ids(fold_test_file, [])
        _write_split_ids(fold_train_cases_file, train_cases)
        _write_split_ids(fold_val_cases_file, val_cases)
        split_composition_report = _build_split_case_combo_composition_report(
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=[],
            sid_case_combo=sid_case_combo,
        )
        comp_json_path, comp_tsv_path, comp_sid_tsv_path = _write_fold_split_composition_report(
            split_root=split_root,
            fold_idx=fold_idx,
            report=split_composition_report,
        )
        folds_manifest.append(
            {
                "fold": int(fold_idx),
                "train_file": str(fold_train_file.resolve()),
                "val_file": str(fold_val_file.resolve()),
                "test_file": str(fold_test_file.resolve()),
                "train_n": int(len(train_ids)),
                "val_n": int(len(val_ids)),
                "train_case_n": int(len(train_cases)),
                "val_case_n": int(len(val_cases)),
                "train_cases_file": str(fold_train_cases_file.resolve()),
                "val_cases_file": str(fold_val_cases_file.resolve()),
                "split_composition_json": str(comp_json_path.resolve()),
                "split_composition_tsv": str(comp_tsv_path.resolve()),
                "split_sid_details_tsv": str(comp_sid_tsv_path.resolve()),
            }
        )

        fold_out_dir = base_out_dir / f"fold_{fold_idx:02d}"
        fold_out_dir.mkdir(parents=True, exist_ok=True)
        fold_tb_dir = fold_out_dir / "tb"

        cmd: List[str] = [
            sys.executable,
            str(Path(__file__).resolve()),
            *child_args,
            "--output-dir",
            str(fold_out_dir),
            "--train-split-file",
            str(fold_train_file.resolve()),
            "--val-split-file",
            str(fold_val_file.resolve()),
            "--epochs",
            str(int(args.n_fold_epochs)),
            "--save-interval",
            str(int(args.n_fold_epochs)),
        ]
        if args.train_platform == "tensorboard":
            cmd.extend(["--tb-dir", str(fold_tb_dir)])
        if use_shared_cache:
            cmd.extend(["--vjepa-cache-dir", str(base_out_dir / "nfold_shared_vjepa_cache")])

        print("")
        print(
            f"[nfold] fold {fold_idx}/{int(args.n_folds)}: "
            f"train={len(train_ids)} val={len(val_ids)} "
            f"train_cases={len(train_cases)} val_cases={len(val_cases)}"
        )
        print(f"[nfold] fold {fold_idx}: output={fold_out_dir}")
        print(
            f"[nfold] fold {fold_idx}: split composition="
            f"{comp_json_path.name}, {comp_tsv_path.name}, {comp_sid_tsv_path.name}"
        )
        print(
            f"[nfold] fold {fold_idx}: case lists="
            f"{fold_train_cases_file.name}, {fold_val_cases_file.name}"
        )
        subprocess.run(cmd, check=True)

        fold_result = _load_fold_val_metrics(fold_out_dir=fold_out_dir, n_fold_epochs=int(args.n_fold_epochs))
        vm = dict(fold_result.get("val_metrics", {}))
        row = {
            "fold": int(fold_idx),
            "train_n": int(len(train_ids)),
            "val_n": int(len(val_ids)),
            "train_case_n": int(len(train_cases)),
            "val_case_n": int(len(val_cases)),
            "best_epoch": int(fold_result.get("epoch", -1)),
            "val_total": float(vm.get("loss_total", float("nan"))),
            "val_combo_acc": float(vm.get("combo_acc", float("nan"))),
            "val_traj_ade_px": float(vm.get("traj_ade_px", float("nan"))),
            "val_traj_aj": float(vm.get("traj_aj", float("nan"))),
            "val_traj_pck_8": float(vm.get("traj_pck_8", float("nan"))),
            "val_range_tiou": float(vm.get("range_tiou", float("nan"))),
            "val_range_r05": float(vm.get("range_r05", float("nan"))),
            "val_tool": float(vm.get("loss_tool_xy", float("nan"))),
            "val_ct": float(vm.get("loss_ct_xy", float("nan"))),
            "picked_checkpoint": str(fold_result.get("picked_checkpoint", "")),
        }
        rows.append(row)
        print(
            f"[nfold] fold {fold_idx} result: total={row['val_total']:.5f} "
            f"combo_acc={row['val_combo_acc']:.4f} "
            f"traj_ade_px={row['val_traj_ade_px']:.3f} "
            f"traj_aj={row['val_traj_aj']:.4f} "
            f"traj_pck8={row['val_traj_pck_8']:.4f} "
            f"range_tiou={row['val_range_tiou']:.4f} "
            f"range_r05={row['val_range_r05']:.4f}"
        )

        # Basic safety: ensure split partition is valid.
        if set(train_ids).intersection(val_set):
            raise RuntimeError(f"Fold {fold_idx}: train/val overlap detected.")
        if set(train_ids).union(val_set) != all_id_set:
            raise RuntimeError(f"Fold {fold_idx}: train/val union does not match source ids.")
        train_case_actual = {sid_to_case_resolved[str(s)] for s in train_ids}
        val_case_actual = {sid_to_case_resolved[str(s)] for s in val_ids}
        if train_case_actual.intersection(val_case_actual):
            raise RuntimeError(f"Fold {fold_idx}: train/val case overlap detected.")
        if val_case_actual != val_case_set:
            raise RuntimeError(f"Fold {fold_idx}: val case set mismatch with fold planner.")
        if train_case_actual.union(val_case_actual) != all_case_set:
            raise RuntimeError(f"Fold {fold_idx}: train/val case union does not match source cases.")

        if nfold_cache_gc_enabled and (nfold_cache_dir is not None):
            for sid in train_ids:
                sid_s = str(sid)
                next_cnt = int(remain_train_uses.get(sid_s, 0)) - 1
                if next_cnt > 0:
                    remain_train_uses[sid_s] = next_cnt
                else:
                    remain_train_uses.pop(sid_s, None)
            for sid in val_ids:
                sid_s = str(sid)
                next_cnt = int(remain_val_uses.get(sid_s, 0)) - 1
                if next_cnt > 0:
                    remain_val_uses[sid_s] = next_cnt
                else:
                    remain_val_uses.pop(sid_s, None)
            keep_train_now = set([str(k) for k, v in remain_train_uses.items() if int(v) > 0])
            keep_val_now = set([str(k) for k, v in remain_val_uses.items() if int(v) > 0])
            gc_report = _cleanup_nfold_cache_root(
                cache_root=nfold_cache_dir,
                keep_train_sids=keep_train_now,
                keep_val_sids=keep_val_now,
            )
            gc_report["event"] = f"after_fold_{fold_idx:02d}"
            cache_gc_events.append(gc_report)
            print(
                f"[nfold] fold {fold_idx}: cache_gc removed_files={gc_report['removed_files']} "
                f"removed_mb={gc_report['removed_bytes'] / (1024.0 * 1024.0):.1f} "
                f"keep_train={len(keep_train_now)} keep_val={len(keep_val_now)} "
                f"errors={gc_report['errors']}"
            )

        cls3_event = _run_nfold_cls3_report(
            args=args,
            base_out_dir=base_out_dir,
            stage_tag=f"after_fold_{fold_idx:02d}",
        )
        if cls3_event is not None:
            cls3_report_events.append(cls3_event)

    json_path, tsv_path, avg_json_path = _write_nfold_summary(base_out_dir=base_out_dir, rows=rows)
    (base_out_dir / "nfold_splits_manifest.json").write_text(json.dumps(folds_manifest, indent=2), encoding="utf-8")
    cache_gc_report_path: Optional[Path] = None
    if nfold_cache_gc_enabled and (nfold_cache_dir is not None):
        gc_final = _cleanup_nfold_cache_root(
            cache_root=nfold_cache_dir,
            keep_train_sids=set(),
            keep_val_sids=set(),
        )
        gc_final["event"] = "final"
        cache_gc_events.append(gc_final)
        cache_gc_report_path = base_out_dir / "nfold_cache_gc_report.json"
        cache_gc_report_path.write_text(json.dumps(cache_gc_events, indent=2), encoding="utf-8")
        print(
            f"[nfold] cache_gc final: removed_files={gc_final['removed_files']} "
            f"removed_mb={gc_final['removed_bytes'] / (1024.0 * 1024.0):.1f} "
            f"errors={gc_final['errors']}"
        )
    avg = _compute_nfold_average(rows)
    print("")
    print("[nfold] summary")
    for r in rows:
        print(
            f"  fold={r['fold']:02d} "
            f"train={r['train_n']} val={r['val_n']} "
            f"train_cases={r.get('train_case_n', 0)} val_cases={r.get('val_case_n', 0)} "
            f"total={r['val_total']:.5f} combo_acc={r['val_combo_acc']:.4f} "
            f"traj_ade_px={r['val_traj_ade_px']:.3f} "
            f"traj_aj={r['val_traj_aj']:.4f} "
            f"traj_pck8={r['val_traj_pck_8']:.4f} "
            f"range_tiou={r['val_range_tiou']:.4f} "
            f"range_r05={r['val_range_r05']:.4f}"
        )
    print(
        f"  avg: train_n={float(avg.get('train_n_mean', float('nan'))):.1f} "
        f"val_n={float(avg.get('val_n_mean', float('nan'))):.1f} "
        f"train_cases={float(avg.get('train_case_n_mean', float('nan'))):.1f} "
        f"val_cases={float(avg.get('val_case_n_mean', float('nan'))):.1f} "
        f"total={float(avg.get('val_total', float('nan'))):.5f} "
        f"combo_acc={float(avg.get('val_combo_acc', float('nan'))):.4f} "
        f"traj_ade_px={float(avg.get('val_traj_ade_px', float('nan'))):.3f} "
        f"traj_aj={float(avg.get('val_traj_aj', float('nan'))):.4f} "
        f"traj_pck8={float(avg.get('val_traj_pck_8', float('nan'))):.4f} "
        f"range_tiou={float(avg.get('val_range_tiou', float('nan'))):.4f} "
        f"range_r05={float(avg.get('val_range_r05', float('nan'))):.4f}"
    )
    print(f"[nfold] saved: {json_path}")
    print(f"[nfold] saved: {tsv_path}")
    print(f"[nfold] saved: {avg_json_path}")
    if cache_gc_report_path is not None:
        print(f"[nfold] saved: {cache_gc_report_path}")
    cls3_final_event = _run_nfold_cls3_report(
        args=args,
        base_out_dir=base_out_dir,
        stage_tag="final",
    )
    if cls3_final_event is not None:
        cls3_report_events.append(cls3_final_event)
    if cls3_report_events:
        cls3_event_path = base_out_dir / "nfold_cls3_report_events.json"
        cls3_event_path.write_text(json.dumps(cls3_report_events, indent=2), encoding="utf-8")
        print(f"[nfold] saved: {cls3_event_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.vjepa_hf_repo and (args.vjepa_pretrained_hub or args.vjepa_checkpoint):
        raise ValueError("`--vjepa-hf-repo` cannot be combined with `--vjepa-pretrained-hub` or `--vjepa-checkpoint`.")

    has_pretrained_source = bool(args.vjepa_hf_repo) or bool(args.vjepa_pretrained_hub) or bool(args.vjepa_checkpoint)
    if (not has_pretrained_source) and (not args.allow_random_vjepa_init):
        raise ValueError(
            "VJEPA pretrained weights are required by default. "
            "Pass --vjepa-hf-repo facebook/vjepa2-vitl-fpc16-256-ssv2, "
            "--vjepa-checkpoint /path/to/checkpoint.pt, or --vjepa-pretrained-hub. "
            "If you really want random frozen encoder, pass --allow-random-vjepa-init."
        )

    if bool(getattr(args, "keyframe_selector_only", False)):
        _run_keyframe_selector_training(args)
        return

    if int(args.n_folds) > 1:
        _run_nfold_training(args)
        return

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.vjepa_cache_dir) if args.vjepa_cache_dir else (out_dir / "vjepa_token_cache")
    token_cache = VJEPATokenCache(
        root=cache_root,
        enabled=bool(args.vjepa_cache),
        dtype_name=str(args.vjepa_cache_dtype),
    )
    tb_writer: Optional["SummaryWriter"] = None
    if args.train_platform == "tensorboard":
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard writer is unavailable. Please install `tensorboard` in this environment.")
        tb_dir = Path(args.tb_dir) if args.tb_dir else (out_dir / "tb")
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"[setup] tensorboard_logdir={tb_dir}")

    (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    if bool(getattr(args, "range_only", False)):
        _validate_range_only_data_root(args.data_root)

    # Build both video-loading and cache-only dataset views.
    train_set_video = SurgKptComboDataset(
        root=args.data_root,
        split="train",
        split_file=(args.train_split_file or None),
        input_size=args.input_size,
        normalize=True,
        load_video=True,
        combo_only_pure=bool(getattr(args, "combo_only_pure", False)),
        tool_kpt_only=bool(getattr(args, "tool_kpt_only_data", False)),
        range_only_contract=bool(getattr(args, "range_only", False)),
    )
    train_set_cache = SurgKptComboDataset(
        root=args.data_root,
        split="train",
        split_file=(args.train_split_file or None),
        input_size=args.input_size,
        normalize=True,
        load_video=False,
        combo_only_pure=bool(getattr(args, "combo_only_pure", False)),
        tool_kpt_only=bool(getattr(args, "tool_kpt_only_data", False)),
        range_only_contract=bool(getattr(args, "range_only", False)),
    )
    val_set_video = SurgKptComboDataset(
        root=args.data_root,
        split="val",
        split_file=(args.val_split_file or None),
        input_size=args.input_size,
        normalize=True,
        load_video=True,
        combo_only_pure=bool(getattr(args, "combo_only_pure", False)),
        tool_kpt_only=bool(getattr(args, "tool_kpt_only_data", False)),
        range_only_contract=bool(getattr(args, "range_only", False)),
    )
    val_set_cache = SurgKptComboDataset(
        root=args.data_root,
        split="val",
        split_file=(args.val_split_file or None),
        input_size=args.input_size,
        normalize=True,
        load_video=False,
        combo_only_pure=bool(getattr(args, "combo_only_pure", False)),
        tool_kpt_only=bool(getattr(args, "tool_kpt_only_data", False)),
        range_only_contract=bool(getattr(args, "range_only", False)),
    )
    # Visualization always needs video tensors.
    val_set_vis = val_set_video

    num_combo_classes = max(train_set_video.num_combos, val_set_video.num_combos)
    combo_id_to_name = train_set_video.combo_id_to_name or val_set_video.combo_id_to_name
    sid_to_case = _load_sid_to_case(Path(args.data_root))
    train_case_counts, train_unknown_cases = _count_cases(train_set_cache.sids, sid_to_case)
    val_case_counts, val_unknown_cases = _count_cases(val_set_cache.sids, sid_to_case)
    # Range-bin reweighting needs sampled_abs/action_range files, which are not
    # present in tool-kpt-only trajectory datasets and are irrelevant in trajectory-only runs.
    use_range_bin_reweight = (
        (not bool(getattr(args, "combo_only", False)))
        and (not bool(getattr(args, "trajectory_only", False)))
        and (not bool(getattr(args, "tool_kpt_only_data", False)))
    )
    reweight = _build_train_reweighting(
        data_root=Path(args.data_root),
        train_sids=[str(x) for x in train_set_cache.sids],
        num_combo_classes=int(num_combo_classes),
        num_range_bins=int(RANGE_NUM_BINS),
        power=float(args.reweight_power),
        max_factor=float(args.reweight_max_factor),
        include_range_bins=use_range_bin_reweight,
    )
    reweight_report_path = out_dir / "train_reweight_report.json"
    reweight_report_path.write_text(json.dumps(reweight["report"], indent=2), encoding="utf-8")
    combo_class_weights = (
        torch.tensor(reweight["combo_weights"], dtype=torch.float32, device=device)
        if args.use_weighted_ce else None
    )
    range_combo_reweight_weights: Optional[torch.Tensor] = None
    range_combo_reweight_base_weights_np: Optional[np.ndarray] = None
    range_combo_reweight_report: Dict[str, Any] = {
        "mode": str(getattr(args, "range_combo_reweight_mode", "none")),
        "targets": _parse_range_combo_reweight_targets(getattr(args, "range_combo_reweight_targets", "")),
        "enabled": False,
    }
    if str(getattr(args, "range_combo_reweight_mode", "none")).lower() != "none":
        combo_counts = np.asarray(reweight["report"]["combo_counts"], dtype=np.int64)
        combo_weights_tail = _make_effective_num_vector(
            combo_counts,
            beta=float(getattr(args, "range_combo_reweight_beta", 0.9997)),
            max_factor=float(getattr(args, "range_combo_reweight_max_factor", 4.0)),
        )
        tail_strength = float(getattr(args, "range_combo_reweight_strength", 0.75))
        combo_weights_tail = 1.0 + tail_strength * (combo_weights_tail - 1.0)
        range_combo_reweight_base_weights_np = combo_weights_tail.astype(np.float32)
        init_scale = _range_combo_reweight_schedule_scale(
            epoch=0,
            start_epoch=int(getattr(args, "range_combo_reweight_start_epoch", 1)),
            ramp_epochs=int(getattr(args, "range_combo_reweight_ramp_epochs", 1)),
        )
        init_weights = _blend_combo_reweight_vector(range_combo_reweight_base_weights_np, init_scale)
        range_combo_reweight_weights = torch.tensor(init_weights, dtype=torch.float32, device=device)
        range_combo_reweight_report.update(
            {
                "enabled": True,
                "beta": float(getattr(args, "range_combo_reweight_beta", 0.9997)),
                "strength": tail_strength,
                "max_factor": float(getattr(args, "range_combo_reweight_max_factor", 4.0)),
                "start_epoch": int(getattr(args, "range_combo_reweight_start_epoch", 1)),
                "ramp_epochs": int(getattr(args, "range_combo_reweight_ramp_epochs", 1)),
                "init_scale": float(init_scale),
                "combo_counts": [int(x) for x in combo_counts.tolist()],
                "base_combo_weights": [float(x) for x in range_combo_reweight_base_weights_np.tolist()],
                "init_combo_weights": [float(x) for x in init_weights.tolist()],
                "combo_names": [
                    str(combo_id_to_name.get(int(cid), f"combo_{int(cid)}"))
                    for cid in range(int(num_combo_classes))
                ],
            }
        )
    (out_dir / "range_combo_reweight_report.json").write_text(
        json.dumps(range_combo_reweight_report, indent=2),
        encoding="utf-8",
    )
    start_bin_class_weights = (
        torch.tensor(reweight["start_bin_weights"], dtype=torch.float32, device=device)
        if args.use_weighted_ce else None
    )
    end_bin_class_weights = (
        torch.tensor(reweight["end_bin_weights"], dtype=torch.float32, device=device)
        if args.use_weighted_ce else None
    )
    train_sample_weight_by_sid = reweight["sample_weight_by_sid"] if args.use_balanced_sampler else {}

    pin_memory = device.type == "cuda"

    def _make_loader(ds: SurgKptComboDataset, split: str, shuffle: bool) -> DataLoader:
        sampler = None
        actual_shuffle = bool(shuffle)
        if split == "train" and args.use_balanced_sampler:
            sampler = _make_train_sampler(ds=ds, sample_weight_by_sid=train_sample_weight_by_sid)
            actual_shuffle = False
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=actual_shuffle,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    use_train_video_loader = True
    use_val_video_loader = True
    if token_cache.enabled:
        if token_cache.has_all("train", train_set_cache.sids):
            use_train_video_loader = False
        if token_cache.has_all("val", val_set_cache.sids):
            use_val_video_loader = False

    train_loader = _make_loader(
        train_set_video if use_train_video_loader else train_set_cache,
        split="train",
        shuffle=True,
    )
    val_loader = _make_loader(
        val_set_video if use_val_video_loader else val_set_cache,
        split="val",
        shuffle=False,
    )

    range_duration_prior_report: Dict[str, Any] = {
        "mode": str(getattr(args, "range_duration_prior_mode", "none")),
        "enabled": False,
    }
    if str(getattr(args, "range_duration_prior_mode", "none")).lower() != "none":
        duration_stats = _compute_combo_duration_prior_stats(
            data_root=Path(args.data_root),
            sids=[str(s) for s in train_set_video.sids],
            num_combo_classes=int(num_combo_classes),
            min_std=float(getattr(args, "range_duration_prior_min_std", 0.05)),
        )
        args.range_duration_prior_means = [float(x) for x in duration_stats["means"]]
        args.range_duration_prior_stds = [float(x) for x in duration_stats["stds"]]
        range_duration_prior_report.update(
            {
                "enabled": True,
                "weight": float(getattr(args, "range_duration_prior_weight", 0.0)),
                "min_std": float(getattr(args, "range_duration_prior_min_std", 0.05)),
                "combo_names": [
                    str(combo_id_to_name.get(int(cid), f"combo_{int(cid)}"))
                    for cid in range(int(num_combo_classes))
                ],
                "combo_means": [float(x) for x in duration_stats["means"]],
                "combo_stds": [float(x) for x in duration_stats["stds"]],
                "combo_counts": [int(x) for x in duration_stats["counts"]],
                "missing_combo_ids": int(duration_stats["missing_combo_ids"]),
                "invalid_rows": int(duration_stats["invalid_rows"]),
            }
        )
    else:
        args.range_duration_prior_means = None
        args.range_duration_prior_stds = None
    (out_dir / "range_duration_prior_report.json").write_text(
        json.dumps(range_duration_prior_report, indent=2),
        encoding="utf-8",
    )

    model = build_model(args=args, num_combo_classes=num_combo_classes).to(device)
    probe_freeze = _apply_dual_probe_branch_freeze(model=model, args=args)
    print(
        f"[setup] vjepa_loaded_sources={getattr(model.vjepa, 'loaded_sources', [])} "
        f"has_pretrained={getattr(model.vjepa, 'has_pretrained', False)}"
    )
    if (not getattr(model.vjepa, "has_pretrained", False)) and (not args.allow_random_vjepa_init):
        raise RuntimeError("VJEPA encoder has no pretrained source loaded. Aborting due to safety check.")
    grad_contract = _collect_grad_contract(model)
    _print_grad_contract(grad_contract)
    if int(grad_contract["vjepa_encoder"]["requires_grad_params"]) != 0:
        raise RuntimeError("VJEPA encoder must be frozen (requires_grad_params should be 0).")
    if bool(getattr(args, "trajectory_only", False)):
        traj_target = str(getattr(args, "trajectory_target", "tool")).lower()
        required_trainable = []
        if traj_target in ("both", "tool"):
            required_trainable.append("tool_prob_head")
        if traj_target in ("both", "ct"):
            required_trainable.append("contact_prob_head")
        if str(getattr(args, "traj_refiner_mode", "none")).lower() != "none":
            if traj_target in ("both", "tool"):
                required_trainable.append("tool_refiner")
            if traj_target in ("both", "ct"):
                required_trainable.append("contact_refiner")
        if len(required_trainable) == 0:
            raise RuntimeError("trajectory_only requires at least one trajectory head (tool/ct).")
    elif bool(getattr(args, "tool_only", False)):
        required_trainable = [
            "combo_attn_pool",
            "combo_tool_head",
        ]
    elif bool(getattr(args, "range_only", False)):
        required_trainable = [
            "range_temporal_pool",
            "range_head",
        ]
    elif bool(getattr(args, "combo_probe_attn", False)):
        required_trainable = [
            "combo_attn_pool",
            "combo_head",
        ]
    elif bool(getattr(args, "combo_probe_dual_attn", False)):
        required_trainable = []
        if not probe_freeze.get("freeze_tool_branch", False):
            required_trainable.extend(["combo_tool_spatial_pool", "combo_tool_head"])
        if not probe_freeze.get("freeze_action_branch", False):
            required_trainable.extend(["combo_action_temporal_pool", "combo_action_head"])
            if bool(getattr(args, "dual_attn_tool_prior", False)):
                required_trainable.append("combo_action_tool_prior")
        if len(required_trainable) == 0:
            raise RuntimeError(
                "Dual-attn probe has no trainable branch. "
                "Check combo/tool/action CE weights."
            )
    else:
        required_trainable = [
            "shared_trunk",
            "conv_refine",
            "tool_head",
            "contact_head",
            "combo_head",
            "range_head",
        ]
        if getattr(args, "refiner_mode", "lite") != "none":
            required_trainable.extend(["tool_refiner", "contact_refiner"])
    if (
        args.temporal_upsample == "deconv"
        and (not bool(getattr(args, "tool_only", False)))
        and (not bool(getattr(args, "combo_probe_attn", False)))
        and (not bool(getattr(args, "combo_probe_dual_attn", False)))
        and (not bool(getattr(args, "trajectory_only", False)))
        and (not bool(getattr(args, "range_only", False)))
    ):
        required_trainable.extend(["tool_time_up_logits", "contact_time_up_logits"])
        if (not bool(getattr(args, "disable_delta_branch", False))) and (not bool(getattr(args, "trajectory_only", False))):
            required_trainable.extend(["tool_time_up_delta", "contact_time_up_delta"])
    for name in required_trainable:
        if int(grad_contract.get(name, {}).get("requires_grad_params", 0)) <= 0:
            raise RuntimeError(f"Trainable module `{name}` has no trainable params.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    amp_enabled = args.amp and device.type == "cuda"
    scaler_enabled = bool(amp_enabled and args.amp_dtype == "fp16")
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    start_epoch = 1
    best_metric_name = _best_monitor_name(args)
    best_metric_mode = _best_monitor_mode(args)
    best_val = _initial_best_monitor_value(args)
    best_epoch = 0
    early_stop_ref_val = best_val
    early_stop_ref_epoch = 0
    early_stop_bad_epochs = 0
    early_stop_enabled = int(getattr(args, "early_stop_patience", 0)) > 0
    target_range_tiou = float(getattr(args, "target_range_tiou", 0.0))
    target_range_tiou_enabled = target_range_tiou > 0.0
    global_step = 0
    resume_ckpt: Optional[Dict[str, Any]] = None
    traj_late_focal_state: Dict[str, Any] = {}
    epoch_metrics_path = out_dir / "epoch_metrics.jsonl"
    val_history_tail: List[Dict[str, Any]] = []

    if args.resume:
        resume_path = Path(args.resume)
        resume_ckpt = torch.load(resume_path, map_location="cpu")
        load_trainable_state_dict(model, resume_ckpt["modules"])
        if "optimizer" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
            except Exception as e:
                print(
                    f"[resume] warning: optimizer state not loaded due to mismatch "
                    f"({type(e).__name__}: {e}); continuing with fresh optimizer state."
                )
        if scaler.is_enabled() and resume_ckpt.get("scaler") is not None:
            try:
                scaler.load_state_dict(resume_ckpt["scaler"])
            except Exception as e:
                print(
                    f"[resume] warning: scaler state not loaded due to mismatch "
                    f"({type(e).__name__}: {e}); continuing with fresh scaler state."
                )
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        best_state = resume_ckpt.get("best_monitor")
        if isinstance(best_state, dict):
            try:
                best_val = float(best_state.get("value", best_val))
            except Exception:
                best_val = _extract_best_monitor_value(args, resume_ckpt.get("val_metrics", {}), best_val)
            best_epoch = int(best_state.get("epoch", resume_ckpt.get("epoch", 0)) or 0)
        else:
            best_val = _extract_best_monitor_value(args, resume_ckpt.get("val_metrics", {}), best_val)
            best_epoch = int(resume_ckpt.get("epoch", 0))
        early_stop_state = resume_ckpt.get("early_stop")
        if isinstance(early_stop_state, dict):
            try:
                early_stop_ref_val = float(early_stop_state.get("ref_value", best_val))
            except Exception:
                early_stop_ref_val = float(best_val)
            early_stop_ref_epoch = int(early_stop_state.get("ref_epoch", best_epoch) or 0)
            early_stop_bad_epochs = int(early_stop_state.get("bad_epochs", 0) or 0)
        else:
            early_stop_ref_val = float(best_val)
            early_stop_ref_epoch = int(best_epoch)
            early_stop_bad_epochs = 0
        if isinstance(resume_ckpt.get("traj_late_focal_state"), dict):
            traj_late_focal_state = dict(resume_ckpt["traj_late_focal_state"])
        print(
            f"[resume] loaded {resume_path}, start_epoch={start_epoch}, "
            f"global_step={global_step}, best_{best_metric_name}={best_val:.6f} "
            f"(epoch={best_epoch:03d})"
        )
        if early_stop_enabled:
            print(
                f"[resume] early_stop_state: ref_{best_metric_name}={early_stop_ref_val:.6f} "
                f"(epoch={early_stop_ref_epoch:03d}) bad_epochs={early_stop_bad_epochs}"
            )

    ema: Optional[TrainableEMA] = None
    if bool(getattr(args, "ema", False)):
        ema = TrainableEMA(model=model, decay=float(getattr(args, "ema_decay", 0.999)))
        if resume_ckpt is not None and ("ema" in resume_ckpt):
            try:
                ema.load_state_dict(resume_ckpt["ema"], model=model)
            except Exception as e:
                print(f"[ema] warning: failed to load EMA state ({type(e).__name__}: {e}); reset from current model.")
                ema.reset(model)
        elif resume_ckpt is not None:
            print("[ema] resume checkpoint has no EMA state; initialized from current model weights.")
        print(f"[ema] enabled=True decay={float(getattr(args, 'ema_decay', 0.999)):.6f}")

    if token_cache.enabled:
        _print_stage_header("CACHE WARMUP")
        use_progress = _is_progress_enabled(args)
        if use_train_video_loader:
            stats_tr = _warmup_token_cache_for_split(
                model=model,
                ds_video=train_set_video,
                split="train",
                token_cache=token_cache,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                use_progress=use_progress,
            )
            print(
                f"[cache warmup][train] seen={stats_tr['seen']} hits={stats_tr['hits']} "
                f"misses={stats_tr['misses']} writes={stats_tr['writes']}"
            )
            if token_cache.has_all("train", train_set_cache.sids):
                use_train_video_loader = False
                train_loader = _make_loader(train_set_cache, split="train", shuffle=True)
                print("[cache warmup][train] complete -> switched train loader to cache-only mode.")

        if use_val_video_loader:
            stats_val = _warmup_token_cache_for_split(
                model=model,
                ds_video=val_set_video,
                split="val",
                token_cache=token_cache,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                use_progress=use_progress,
            )
            print(
                f"[cache warmup][val] seen={stats_val['seen']} hits={stats_val['hits']} "
                f"misses={stats_val['misses']} writes={stats_val['writes']}"
            )
            if token_cache.has_all("val", val_set_cache.sids):
                use_val_video_loader = False
                val_loader = _make_loader(val_set_cache, split="val", shuffle=False)
                print("[cache warmup][val] complete -> switched val loader to cache-only mode.")

    is_traj_only = bool(getattr(args, "trajectory_only", False))
    is_combo_only = bool(getattr(args, "combo_only", False))
    is_tool_only = bool(getattr(args, "tool_only", False))
    is_combo_probe_attn = bool(getattr(args, "combo_probe_attn", False))
    is_combo_probe_dual_attn = bool(getattr(args, "combo_probe_dual_attn", False))
    is_combo_only_pure = bool(getattr(args, "combo_only_pure", False))

    print("[setup]")
    print(f"  device={device}")
    print(f"  best_monitor={best_metric_name} mode={best_metric_mode}")
    if early_stop_enabled:
        print(
            f"  early_stop=enabled patience={int(getattr(args, 'early_stop_patience', 0))} "
            f"min_delta={float(getattr(args, 'early_stop_min_delta', 0.0)):.6f} "
            f"warmup_epochs={int(getattr(args, 'early_stop_warmup_epochs', 0))}"
        )
    else:
        print("  early_stop=disabled")
    if target_range_tiou_enabled:
        print(f"  target_range_tiou_stop=enabled threshold={target_range_tiou:.6f}")
    else:
        print("  target_range_tiou_stop=disabled")
    print(f"  train_split_file={args.train_split_file or 'train.txt'}")
    print(f"  val_split_file={args.val_split_file or 'val.txt'}")
    print(f"  train_samples={len(train_set_cache)} val_samples={len(val_set_cache)}")
    _print_case_counts("train", train_case_counts, train_unknown_cases)
    _print_case_counts("val", val_case_counts, val_unknown_cases)
    print(
        f"  loader_mode: train_video={use_train_video_loader} val_video={use_val_video_loader} "
        f"(cache_enabled={token_cache.enabled})"
    )
    print(f"  weighted_ce={args.use_weighted_ce} balanced_sampler={args.use_balanced_sampler}")
    print(
        f"  reweight(power={args.reweight_power:.3f}, max_factor={args.reweight_max_factor:.3f}) "
        f"report={reweight_report_path}"
    )
    rw = reweight["report"]
    print(f"  combo_only_pure={is_combo_only_pure} range_bins_enabled={bool(rw.get('range_bins_enabled', True))}")
    print(f"  num_combo_classes={num_combo_classes}")
    print(f"  ema={bool(getattr(args, 'ema', False))} ema_decay={float(getattr(args, 'ema_decay', 0.999)):.6f}")
    if is_tool_only:
        # Keep tool-only setup logs minimal and focused.
        print(f"  tool_ce_weight={float(getattr(args, 'combo_tool_ce_weight', 0.5))}")
        print(f"  train_combo_counts={rw['combo_counts']}")
        if args.use_weighted_ce:
            print(f"  combo_class_weights={rw['combo_weights']}")
        print(f"  vjepa_arch={args.vjepa_arch} frozen=True hf_repo={args.vjepa_hf_repo or '(none)'}")
        print("  model_path=tool_only (xattn_pool + linear_tool_head)")
        print(f"  combo_only={is_combo_only}")
        print(f"  tool_only={is_tool_only}")
        print(f"  vjepa_token_cache: {token_cache.format_stats()}")
        print(
            f"  save_interval={args.save_interval} val_interval={args.val_interval} "
            f"save_best_only={bool(args.save_best_only)} log_interval={args.log_interval}"
        )
        print(
            f"  visualization: init_eval_vis={args.run_init_eval_vis} "
            f"checkpoint_vis={args.run_checkpoint_vis}"
        )
    elif is_combo_probe_attn:
        # Keep combo probe setup logs minimal and focused.
        print(f"  combo_ce_weight={float(getattr(args, 'combo_ce_weight', COMBO_CE_WEIGHT))}")
        print(f"  combo_tool_ce_weight={float(getattr(args, 'combo_tool_ce_weight', 0.5))}")
        print(f"  combo_action_ce_weight={float(getattr(args, 'combo_action_ce_weight', 0.5))}")
        print(f"  train_combo_counts={rw['combo_counts']}")
        if args.use_weighted_ce:
            print(f"  combo_class_weights={rw['combo_weights']}")
        print(f"  vjepa_arch={args.vjepa_arch} frozen=True hf_repo={args.vjepa_hf_repo or '(none)'}")
        print("  model_path=combo_probe_attn (xattn_pool + linear combo head)")
        print(f"  combo_only={is_combo_only}")
        print(f"  combo_probe_attn={is_combo_probe_attn}")
        print(f"  vjepa_token_cache: {token_cache.format_stats()}")
        print(
            f"  save_interval={args.save_interval} val_interval={args.val_interval} "
            f"save_best_only={bool(args.save_best_only)} log_interval={args.log_interval}"
        )
        print(
            f"  visualization: init_eval_vis={args.run_init_eval_vis} "
            f"checkpoint_vis={args.run_checkpoint_vis}"
        )
    elif is_combo_probe_dual_attn:
        # Keep dual-attn probe setup logs minimal and focused.
        print(f"  combo_ce_weight={float(getattr(args, 'combo_ce_weight', COMBO_CE_WEIGHT))}")
        print(f"  combo_tool_ce_weight={float(getattr(args, 'combo_tool_ce_weight', 0.5))}")
        print(f"  combo_action_ce_weight={float(getattr(args, 'combo_action_ce_weight', 0.5))}")
        print(f"  dual_attn_tool_prior={bool(getattr(args, 'dual_attn_tool_prior', False))}")
        print(f"  dual_attn_tool_prior_init_alpha={float(getattr(args, 'dual_attn_tool_prior_init_alpha', 0.0)):.6f}")
        print(f"  dual_attn_tool_prior_detach={bool(getattr(args, 'dual_attn_tool_prior_detach', True))}")
        print(f"  freeze_tool_branch={bool(probe_freeze.get('freeze_tool_branch', False))}")
        print(f"  freeze_action_branch={bool(probe_freeze.get('freeze_action_branch', False))}")
        print(f"  train_combo_counts={rw['combo_counts']}")
        if args.use_weighted_ce:
            print(f"  combo_class_weights={rw['combo_weights']}")
        print(f"  vjepa_arch={args.vjepa_arch} frozen=True hf_repo={args.vjepa_hf_repo or '(none)'}")
        print("  model_path=combo_probe_dual_attn (token xattn branch #1 -> tool, token xattn branch #2 -> action, combo=tool+action)")
        print(f"  combo_only={is_combo_only}")
        print(f"  combo_probe_dual_attn={is_combo_probe_dual_attn}")
        print(f"  vjepa_token_cache: {token_cache.format_stats()}")
        print(
            f"  save_interval={args.save_interval} val_interval={args.val_interval} "
            f"save_best_only={bool(args.save_best_only)} log_interval={args.log_interval}"
        )
        print(
            f"  visualization: init_eval_vis={args.run_init_eval_vis} "
            f"checkpoint_vis={args.run_checkpoint_vis}"
        )
    else:
        if not is_traj_only:
            print(f"  combo_ce_weight={float(getattr(args, 'combo_ce_weight', COMBO_CE_WEIGHT))}")
            if is_combo_only:
                print(f"  combo_tool_ce_weight={float(getattr(args, 'combo_tool_ce_weight', 0.5))}")
                print(f"  combo_action_ce_weight={float(getattr(args, 'combo_action_ce_weight', 0.5))}")
            else:
                print(f"  range_start_l1_weight={RANGE_START_CE_WEIGHT} (fixed)")
                print(f"  range_end_l1_weight={RANGE_END_CE_WEIGHT} (fixed)")
                print(f"  range_label_smoothing={RANGE_LABEL_SMOOTHING} (fixed)")
                print(f"  range_focal_gamma={RANGE_FOCAL_GAMMA} (fixed)")
                print(f"  range_neighbor_smoothing={RANGE_NEIGHBOR_SMOOTHING} (fixed)")
                print(f"  range_num_bins={RANGE_NUM_BINS} (fixed)")
        if not is_traj_only:
            print(f"  train_combo_counts={rw['combo_counts']}")
            if not is_combo_only:
                print(f"  train_start_bin_counts={rw['start_bin_counts']}")
                print(f"  train_end_bin_counts={rw['end_bin_counts']}")
        if args.use_weighted_ce and (not is_traj_only):
            print(f"  combo_class_weights={rw['combo_weights']}")
            if not is_combo_only:
                print(f"  start_bin_class_weights={rw['start_bin_weights']}")
                print(f"  end_bin_class_weights={rw['end_bin_weights']}")
        print(f"  vjepa_arch={args.vjepa_arch} frozen=True hf_repo={args.vjepa_hf_repo or '(none)'}")
        print(f"  temporal_upsample={args.temporal_upsample}")
        print(f"  heatmap_softmax_temp={float(getattr(args, 'heatmap_softmax_temp', 1.0))}")
        print(f"  disable_delta_branch={bool(getattr(args, 'disable_delta_branch', False))}")
        print(f"  temporal_mean_residual_alpha={args.temporal_mean_residual_alpha}")
        print(f"  refiner_mode={args.refiner_mode}")
        print(f"  trajectory_only={is_traj_only}")
        print(f"  trajectory_target={str(getattr(args, 'trajectory_target', 'tool'))}")
        print(f"  tool_kpt_only_data={bool(getattr(args, 'tool_kpt_only_data', False))}")
        print(f"  combo_only={is_combo_only}")
        print(f"  tool_only={is_tool_only}")
        if bool(getattr(args, "range_only", False)):
            print(f"  range_condition_mode={str(getattr(args, 'range_condition_mode', 'tool_action'))}")
            print(f"  range_positive_mode={str(getattr(args, 'range_positive_mode', 'span'))}")
            print(f"  range_sensitivity_mode={str(getattr(args, 'range_sensitivity_mode', 'none'))}")
            print(f"  range_cls_target_mode={str(getattr(args, 'range_cls_target_mode', 'binary'))}")
            print(f"  range_cls_varifocal_weight={float(getattr(args, 'range_cls_varifocal_weight', 1.0))}")
            print(f"  range_quality_mode={str(getattr(args, 'range_quality_mode', 'none'))}")
            print(
                "  range_sensitivity_cls_sigma="
                f"{float(getattr(args, 'range_sensitivity_cls_sigma', 0.5))}"
            )
            print(
                "  range_sensitivity_loc_sigma="
                f"{float(getattr(args, 'range_sensitivity_loc_sigma', 0.35))}"
            )
            print(
                "  range_sensitivity_min_weight="
                f"{float(getattr(args, 'range_sensitivity_min_weight', 0.25))}"
            )
            print(f"  range_head_mode={str(getattr(args, 'range_head_mode', 'scalar'))}")
            print(f"  range_proposal_mode={str(getattr(args, 'range_proposal_mode', 'none'))}")
            print(f"  range_quality_loss_weight={float(getattr(args, 'range_quality_loss_weight', 0.0))}")
            print(f"  range_cls_varifocal_alpha={float(getattr(args, 'range_cls_varifocal_alpha', 0.75))}")
            print(
                "  range_proposal_decode_mode="
                f"{str(getattr(args, 'range_proposal_decode_mode', 'argmax'))}"
            )
            print(
                "  range_boundary_refine_mode="
                f"{str(getattr(args, 'range_boundary_refine_mode', 'none'))}"
            )
            print(
                "  range_boundary_refine_radius="
                f"{int(getattr(args, 'range_boundary_refine_radius', 2))}"
            )
            print(
                "  range_temporal_context_mode="
                f"{str(getattr(args, 'range_temporal_context_mode', 'none'))}"
            )
            print(
                "  range_temporal_context_kernel_size="
                f"{int(getattr(args, 'range_temporal_context_kernel_size', 3))}"
            )
            print(
                "  range_temporal_context_layers="
                f"{int(getattr(args, 'range_temporal_context_layers', 1))}"
            )
            print(
                "  range_temporal_context_levels="
                f"{int(getattr(args, 'range_temporal_context_levels', 3))}"
            )
            print(f"  range_spatial_pool_mode={str(getattr(args, 'range_spatial_pool_mode', 'mean_max'))}")
            print(f"  range_spatial_pool_num_tokens={int(getattr(args, 'range_spatial_pool_num_tokens', 8))}")
            print(f"  range_adapter_mode={str(getattr(args, 'range_adapter_mode', 'none'))}")
            print(f"  range_adapter_kernel_size={int(getattr(args, 'range_adapter_kernel_size', 5))}")
            print(f"  range_adapter_num_heads={int(getattr(args, 'range_adapter_num_heads', 4))}")
            print(f"  range_duration_prior_mode={str(getattr(args, 'range_duration_prior_mode', 'none'))}")
            print(f"  range_duration_prior_weight={float(getattr(args, 'range_duration_prior_weight', 0.0))}")
            print(f"  range_duration_prior_min_std={float(getattr(args, 'range_duration_prior_min_std', 0.05))}")
            print(f"  range_scale_invariant_mode={str(getattr(args, 'range_scale_invariant_mode', 'none'))}")
            print(f"  range_scale_invariant_alpha={float(getattr(args, 'range_scale_invariant_alpha', 0.8))}")
            print(f"  range_trident_num_bins={int(getattr(args, 'range_trident_num_bins', 15))}")
            print(f"  range_boundary_aux_weight={float(getattr(args, 'range_boundary_aux_weight', 0.0))}")
            print(f"  range_proposal_weight={float(getattr(args, 'range_proposal_weight', 0.0))}")
            print(f"  range_sola_aux_weight={float(getattr(args, 'range_sola_aux_weight', 0.0))}")
            print(f"  range_sola_sim_k={float(getattr(args, 'range_sola_sim_k', 16.0))}")
            print(f"  range_combo_reweight_mode={str(getattr(args, 'range_combo_reweight_mode', 'none'))}")
            print(
                "  range_combo_reweight_targets="
                f"{str(getattr(args, 'range_combo_reweight_targets', ''))}"
            )
            print(
                "  range_combo_reweight_beta="
                f"{float(getattr(args, 'range_combo_reweight_beta', 0.9997))}"
            )
            print(
                "  range_combo_reweight_strength="
                f"{float(getattr(args, 'range_combo_reweight_strength', 0.75))}"
            )
            print(
                "  range_combo_reweight_start_epoch="
                f"{int(getattr(args, 'range_combo_reweight_start_epoch', 1))}"
            )
            print(
                "  range_combo_reweight_ramp_epochs="
                f"{int(getattr(args, 'range_combo_reweight_ramp_epochs', 1))}"
            )
        if is_traj_only:
            print(f"  traj_heatmap_ce_weight={float(getattr(args, 'traj_heatmap_ce_weight', 0.0))}")
            print(f"  traj_heatmap_sigma={float(getattr(args, 'traj_heatmap_sigma', 2.45))}")
            print(f"  traj_num_subframes={int(getattr(args, 'traj_num_subframes', 2))}")
            print(f"  traj_condition_mode={str(getattr(args, 'traj_condition_mode', 'tool'))}")
            print(f"  traj_refiner_mode={str(getattr(args, 'traj_refiner_mode', 'none'))}")
            if str(getattr(args, "traj_refiner_mode", "none")) == "s3p9e_singlelayer_statecarry_anchoronly_ramped":
                print("  anchor_activation_delay_epochs=10")
                print("  anchor_activation_ramp_epochs=10")
                print("  anchor_activation_skip_when_zero=True")
            if str(getattr(args, "traj_refiner_mode", "none")) == "s3p9f_singlelayer_statecarry_anchorverifier_l18":
                print("  anchor_verifier_layer=18")
                print("  anchor_verifier_support_radius=1")
                print("  anchor_verifier_identity_init=True")
            print(f"  traj_centroid_aux_weight={float(getattr(args, 'traj_centroid_aux_weight', 0.0))}")
            print(f"  traj_offset_aux_weight={float(getattr(args, 'traj_offset_aux_weight', 0.0))}")
            print(f"  traj_head_hidden_dim={int(getattr(args, 'traj_head_hidden_dim', args.proj_dim))}")
            print(f"  traj_head_up_layers={int(getattr(args, 'traj_head_up_layers', 2))}")
            print(f"  traj_recurrent_max_delta_norm={float(getattr(args, 'traj_recurrent_max_delta_norm', 0.06))}")
            print(f"  traj_dynamic_centroid_max_beta={float(getattr(args, 'traj_dynamic_centroid_max_beta', 0.75))}")
            print(f"  token_mask_enable={bool(getattr(args, 'token_mask_enable', False))}")
            print(f"  token_mask_eval={bool(getattr(args, 'token_mask_eval', False))}")
            print(f"  token_mask_strategy={str(getattr(args, 'token_mask_strategy', 'random'))}")
            print(f"  token_mask_value={str(getattr(args, 'token_mask_value', 'learned_pos'))}")
            print(f"  token_mask_p_max={float(getattr(args, 'token_mask_p_max', 0.10)):.4f}")
            print(f"  token_mask_p_early={float(getattr(args, 'token_mask_p_early', getattr(args, 'token_mask_p_max', 0.10))):.4f}")
            print(f"  token_mask_delay_epochs={int(getattr(args, 'token_mask_delay_epochs', 5))}")
            print(f"  token_mask_warmup_epochs={int(getattr(args, 'token_mask_warmup_epochs', 20))}")
            print(f"  token_mask_late_start_epoch={int(getattr(args, 'token_mask_late_start_epoch', 25))}")
            print(f"  token_mask_p_late={float(getattr(args, 'token_mask_p_late', 0.15)):.4f}")
            print(f"  token_mask_late_warmup_epochs={int(getattr(args, 'token_mask_late_warmup_epochs', 40))}")
            print(f"  token_mask_keep_k={int(getattr(args, 'token_mask_keep_k', 64))}")
            print(f"  token_mask_mix_801010={bool(getattr(args, 'token_mask_mix_801010', True))}")
            print(
                f"  traj_stage0_centroid_noise_prob={float(getattr(args, 'traj_stage0_centroid_noise_prob', 0.0)):.4f}"
            )
            print(
                f"  traj_stage0_centroid_noise_sigma={float(getattr(args, 'traj_stage0_centroid_noise_sigma', 0.0)):.4f}"
            )
            print(f"  traj_stage_curriculum_enable={bool(getattr(args, 'traj_stage_curriculum_enable', False))}")
            if bool(getattr(args, "traj_stage_curriculum_enable", False)):
                print(
                    f"  traj_stage_curriculum_phase1_end_epoch="
                    f"{int(getattr(args, 'traj_stage_curriculum_phase1_end_epoch', 16))}"
                )
                print(
                    f"  traj_stage_curriculum_ramp12_end_epoch="
                    f"{int(getattr(args, 'traj_stage_curriculum_ramp12_end_epoch', 20))}"
                )
                print(
                    f"  traj_stage_curriculum_phase2_end_epoch="
                    f"{int(getattr(args, 'traj_stage_curriculum_phase2_end_epoch', 44))}"
                )
                print(
                    f"  traj_stage_curriculum_ramp23_end_epoch="
                    f"{int(getattr(args, 'traj_stage_curriculum_ramp23_end_epoch', 50))}"
                )
                print(
                    "  traj_stage_curriculum_phase1_weights="
                    f"{_format_stage_weight_spec(getattr(args, 'traj_stage_curriculum_phase1_weights'))}"
                )
                print(
                    "  traj_stage_curriculum_phase2_weights="
                    f"{_format_stage_weight_spec(getattr(args, 'traj_stage_curriculum_phase2_weights'))}"
                )
                print(
                    "  traj_stage_curriculum_phase3_weights="
                    f"{_format_stage_weight_spec(getattr(args, 'traj_stage_curriculum_phase3_weights'))}"
                )
            print(f"  traj_late_focal_enable={bool(getattr(args, 'traj_late_focal_enable', False))}")
            if bool(getattr(args, "traj_late_focal_enable", False)):
                print(
                    "  traj_late_focal_stage_indices="
                    f"{','.join(str(int(x)) for x in getattr(args, 'traj_late_focal_stage_indices', []))}"
                )
                print(f"  traj_late_focal_gamma={float(getattr(args, 'traj_late_focal_gamma', 1.5)):.4f}")
                print(f"  traj_late_focal_clip_min={float(getattr(args, 'traj_late_focal_clip_min', 0.30)):.4f}")
                print(f"  traj_late_focal_clip_max={float(getattr(args, 'traj_late_focal_clip_max', 3.0)):.4f}")
                print(f"  traj_late_focal_ema_decay={float(getattr(args, 'traj_late_focal_ema_decay', 0.95)):.4f}")
                print(
                    f"  traj_late_focal_warmup_steps={int(getattr(args, 'traj_late_focal_warmup_steps', 0))}"
                )
        print(f"  kpt_loss_type={args.kpt_loss_type}")
        print(f"  traj_loss_pixel_scale={float(getattr(args, 'traj_loss_pixel_scale', 256.0))}")
        print(f"  traj_huber_delta={float(getattr(args, 'traj_huber_delta', 6.0))}")
        print(f"  traj_t0_loss_weight={float(getattr(args, 'traj_t0_loss_weight', 1.0))}")
        print(f"  traj_loss_visible_only={bool(getattr(args, 'traj_loss_visible_only', False))}")
        print(f"  debug_model_timing={bool(getattr(args, 'debug_model_timing', False))}")
        print(f"  vjepa_token_cache: {token_cache.format_stats()}")
        print(
            f"  save_interval={args.save_interval} val_interval={args.val_interval} "
            f"save_best_only={bool(args.save_best_only)} log_interval={args.log_interval}"
        )
        print(
            f"  visualization: init_eval_vis={args.run_init_eval_vis} "
            f"checkpoint_vis={args.run_checkpoint_vis}"
        )
        print(
            f"  ckpt_visualization: samples_per_ckpt={args.vis_samples_per_ckpt}, "
            f"max_frames={args.vis_max_frames}, heatmap_alpha={args.vis_heatmap_alpha}, vis_scale={args.vis_scale}"
        )
    if args.debug_sanity:
        sanity_loader = train_loader
        if not use_train_video_loader:
            sanity_loader = _make_loader(train_set_video, split="train", shuffle=False)
        run_debug_sanity(
            model=model,
            train_loader=sanity_loader,
            device=device,
            args=args,
            out_dir=out_dir,
            combo_class_weights=combo_class_weights,
            start_bin_class_weights=start_bin_class_weights,
            end_bin_class_weights=end_bin_class_weights,
            range_combo_reweight_weights=range_combo_reweight_weights,
        )

    if args.run_init_eval_vis:
        _print_stage_header("INIT SNAPSHOT (EPOCH 000): val + visualization + init checkpoint")
        print("[init] running pre-train val + visualization snapshot (epoch_000)")
        with _use_ema_weights(model, ema):
            init_val_metrics, init_val_by_combo = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                args=args,
                combo_id_to_name=combo_id_to_name,
                epoch=0,
                token_cache=token_cache,
                range_combo_reweight_weights=range_combo_reweight_weights,
            )
        init_val_by_tool = aggregate_val_by_tool(init_val_by_combo)
        if bool(getattr(args, "trajectory_only", False)):
            print(
                f"[epoch 000] init val: total={init_val_metrics['loss_total']:.5f} "
                f"tool={init_val_metrics['loss_tool_xy']:.5f} ct={init_val_metrics['loss_ct_xy']:.5f} "
                f"tool_hm_ce={init_val_metrics.get('loss_tool_heatmap_ce', 0.0):.5f} "
                f"ct_hm_ce={init_val_metrics.get('loss_ct_heatmap_ce', 0.0):.5f} "
                f"traj_ade_px={init_val_metrics.get('traj_ade_px', 0.0):.3f} "
                f"ref_gate={init_val_metrics.get('refine_gate_mean', 0.0):.4f} "
                f"ref_dxy={init_val_metrics.get('refine_delta_abs_mean', 0.0):.5f} "
                f"ref_cshift={init_val_metrics.get('refine_centroid_shift_abs_mean', 0.0):.5f} "
                f"anchor_cshift={init_val_metrics.get('anchor_delta_centroid_abs_mean', 0.0):.5f} "
                f"anchor_alpha={init_val_metrics.get('anchor_alpha_mean', 0.0):.4f} "
                f"anchor_vfy={init_val_metrics.get('anchor_verify_scale_mean', 1.0):.4f} "
                f"factor_ag={init_val_metrics.get('factor_anchor_gate_mean', 0.0):.4f} "
                f"factor_og={init_val_metrics.get('factor_offset_gate_mean', 0.0):.4f} "
                f"factor_off={init_val_metrics.get('factor_delta_offset_abs_mean', 0.0):.5f}"
            )
        elif bool(getattr(args, "combo_only", False)):
            if bool(getattr(args, "tool_only", False)):
                print(
                    f"[epoch 000] init val: total={init_val_metrics['loss_total']:.5f} "
                    f"tool_ce={init_val_metrics.get('loss_combo_tool_ce', 0.0):.5f} "
                    f"tool_acc={init_val_metrics.get('tool_acc', 0.0):.4f}"
                )
            elif bool(getattr(args, "combo_probe_attn", False)):
                print(
                    f"[epoch 000] init val: total={init_val_metrics['loss_total']:.5f} "
                    f"combo_ce={init_val_metrics['loss_combo_ce']:.5f} "
                    f"combo_acc={init_val_metrics['combo_acc']:.4f}"
                )
            else:
                print(
                    f"[epoch 000] init val: total={init_val_metrics['loss_total']:.5f} "
                    f"combo_ce={init_val_metrics['loss_combo_ce']:.5f} "
                    f"tool_ce={init_val_metrics.get('loss_combo_tool_ce', 0.0):.5f} "
                    f"action_ce={init_val_metrics.get('loss_combo_action_ce', 0.0):.5f} "
                    f"combo_acc={init_val_metrics['combo_acc']:.4f} "
                    f"tool_acc={init_val_metrics.get('tool_acc', 0.0):.4f} "
                    f"action_acc={init_val_metrics.get('action_acc', 0.0):.4f}"
                )
            print_val_by_combo(
                init_val_by_combo,
                combo_only=bool(getattr(args, "combo_only", False)),
                tool_only=bool(getattr(args, "tool_only", False)),
                combo_probe_attn=bool(getattr(args, "combo_probe_attn", False)),
            )
        else:
            print(
                f"[epoch 000] init val: total={init_val_metrics['loss_total']:.5f} "
                f"tool={init_val_metrics['loss_tool_xy']:.5f} ct={init_val_metrics['loss_ct_xy']:.5f} "
                f"combo_ce={init_val_metrics['loss_combo_ce']:.5f} "
                f"start_l1={init_val_metrics.get('loss_start_l1', init_val_metrics['loss_start_bin_ce']):.5f} "
                f"end_l1={init_val_metrics.get('loss_end_l1', init_val_metrics['loss_end_bin_ce']):.5f} "
                f"range={init_val_metrics['loss_range']:.5f} "
                f"range_cls={init_val_metrics.get('loss_range_cls', 0.0):.5f} "
                f"range_reg={init_val_metrics.get('loss_range_reg', 0.0):.5f} "
                f"range_prop={init_val_metrics.get('loss_range_proposal', 0.0):.5f} "
                f"range_sola={init_val_metrics.get('loss_range_sola', 0.0):.5f} "
                f"combo_acc={init_val_metrics['combo_acc']:.4f} "
                f"traj_ade_px={init_val_metrics.get('traj_ade_px', 0.0):.3f} "
                f"traj_aj={init_val_metrics.get('traj_aj', 0.0):.4f} "
                f"traj_pck8={init_val_metrics.get('traj_pck_8', 0.0):.4f} "
                f"range_tiou={init_val_metrics.get('range_tiou', 0.0):.4f} "
                f"range_r05={init_val_metrics.get('range_r05', 0.0):.4f} "
                f"ref_gate={init_val_metrics.get('refine_gate_mean', 0.0):.4f} "
                f"ref_dxy={init_val_metrics.get('refine_delta_abs_mean', 0.0):.5f} "
                f"ref_cshift={init_val_metrics.get('refine_centroid_shift_abs_mean', 0.0):.5f} "
                f"anchor_cshift={init_val_metrics.get('anchor_delta_centroid_abs_mean', 0.0):.5f} "
                f"anchor_alpha={init_val_metrics.get('anchor_alpha_mean', 0.0):.4f} "
                f"anchor_vfy={init_val_metrics.get('anchor_verify_scale_mean', 1.0):.4f} "
                f"factor_ag={init_val_metrics.get('factor_anchor_gate_mean', 0.0):.4f} "
                f"factor_og={init_val_metrics.get('factor_offset_gate_mean', 0.0):.4f} "
                f"factor_off={init_val_metrics.get('factor_delta_offset_abs_mean', 0.0):.5f}"
            )
            print_val_by_combo(
                init_val_by_combo,
                combo_only=bool(getattr(args, "combo_only", False)),
                tool_only=bool(getattr(args, "tool_only", False)),
                combo_probe_attn=bool(getattr(args, "combo_probe_attn", False)),
            )
        init_combo_json_path = ckpt_dir / "val_by_combo_epoch_000.json"
        init_combo_json_path.write_text(json.dumps(init_val_by_combo, indent=2), encoding="utf-8")
        init_tool_json_path = ckpt_dir / "val_by_tool_epoch_000.json"
        init_tool_json_path.write_text(json.dumps(init_val_by_tool, indent=2), encoding="utf-8")
        print(f"[init] saved: {init_combo_json_path}")
        print(f"[init] saved: {init_tool_json_path}")
        with _use_ema_weights(model, ema):
            init_vis_val_info = run_checkpoint_visualization(
                model=model,
                val_set=val_set_vis,
                device=device,
                args=args,
                epoch=0,
                output_dir=out_dir,
                combo_id_to_name=combo_id_to_name,
                token_cache=token_cache,
                range_combo_reweight_weights=range_combo_reweight_weights,
                split_name="val",
                selection_mode="top_vis",
            )
        if init_vis_val_info.get("enabled") == "true":
            print(
                f"[init] visuals[val]: samples={init_vis_val_info.get('saved_samples','0')} "
                f"path={init_vis_val_info.get('path','')}"
            )
        with _use_ema_weights(model, ema):
            init_vis_train_info = run_checkpoint_visualization(
                model=model,
                val_set=train_set_video,
                device=device,
                args=args,
                epoch=0,
                output_dir=out_dir,
                combo_id_to_name=combo_id_to_name,
                token_cache=token_cache,
                range_combo_reweight_weights=range_combo_reweight_weights,
                split_name="train",
                selection_mode="random",
            )
        if init_vis_train_info.get("enabled") == "true":
            print(
                f"[init] visuals[train]: samples={init_vis_train_info.get('saved_samples','0')} "
                f"path={init_vis_train_info.get('path','')}"
            )
        init_ckpt_path = ckpt_dir / "epoch_000_init.pt"
        save_checkpoint(
            path=init_ckpt_path,
            epoch=0,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            train_metrics=_zero_train_metrics(),
            val_metrics=init_val_metrics,
            val_by_combo=init_val_by_combo,
            val_by_tool=init_val_by_tool,
            ema=ema,
            traj_late_focal_state=traj_late_focal_state,
            best_monitor_name=best_metric_name,
            best_monitor_value=best_val,
            best_monitor_epoch=best_epoch,
            early_stop_ref_value=early_stop_ref_val,
            early_stop_ref_epoch=early_stop_ref_epoch,
            early_stop_bad_epochs=early_stop_bad_epochs,
        )
        print(f"[init] saved: {init_ckpt_path}")
        if tb_writer is not None:
            tb_writer.add_scalar("val/epoch_loss_total", init_val_metrics["loss_total"], 0)
            tb_writer.add_scalar("val/epoch_loss_tool_xy", init_val_metrics["loss_tool_xy"], 0)
            tb_writer.add_scalar("val/epoch_loss_ct_xy", init_val_metrics["loss_ct_xy"], 0)
            tb_writer.add_scalar("val/epoch_loss_tool_pair_sep_aux", init_val_metrics.get("loss_tool_pair_sep_aux", 0.0), 0)
            tb_writer.add_scalar("val/epoch_loss_tool_heatmap_ce", init_val_metrics.get("loss_tool_heatmap_ce", 0.0), 0)
            tb_writer.add_scalar("val/epoch_loss_ct_heatmap_ce", init_val_metrics.get("loss_ct_heatmap_ce", 0.0), 0)
            if not bool(getattr(args, "trajectory_only", False)):
                tb_writer.add_scalar("val/epoch_loss_combo_ce", init_val_metrics["loss_combo_ce"], 0)
                tb_writer.add_scalar("val/epoch_loss_combo_tool_ce", init_val_metrics.get("loss_combo_tool_ce", 0.0), 0)
                tb_writer.add_scalar("val/epoch_loss_combo_action_ce", init_val_metrics.get("loss_combo_action_ce", 0.0), 0)
                tb_writer.add_scalar("val/epoch_combo_acc", init_val_metrics["combo_acc"], 0)
                tb_writer.add_scalar("val/epoch_tool_acc", init_val_metrics.get("tool_acc", 0.0), 0)
                tb_writer.add_scalar("val/epoch_action_acc", init_val_metrics.get("action_acc", 0.0), 0)
                if not bool(getattr(args, "combo_only", False)):
                    tb_writer.add_scalar(
                        "val/epoch_loss_start_l1",
                        init_val_metrics.get("loss_start_l1", init_val_metrics["loss_start_bin_ce"]),
                        0,
                    )
                    tb_writer.add_scalar(
                        "val/epoch_loss_end_l1",
                        init_val_metrics.get("loss_end_l1", init_val_metrics["loss_end_bin_ce"]),
                        0,
                    )
                    tb_writer.add_scalar("val/epoch_loss_range", init_val_metrics["loss_range"], 0)
                    tb_writer.add_scalar("val/epoch_loss_range_cls", init_val_metrics.get("loss_range_cls", 0.0), 0)
                    tb_writer.add_scalar("val/epoch_loss_range_reg", init_val_metrics.get("loss_range_reg", 0.0), 0)
                    tb_writer.add_scalar("val/epoch_loss_range_proposal", init_val_metrics.get("loss_range_proposal", 0.0), 0)
                    tb_writer.add_scalar("val/epoch_loss_range_sola", init_val_metrics.get("loss_range_sola", 0.0), 0)
            tb_writer.add_scalar("val/epoch_traj_ade_px", init_val_metrics.get("traj_ade_px", 0.0), 0)
            tb_writer.add_scalar("val/epoch_traj_aj", init_val_metrics.get("traj_aj", 0.0), 0)
            tb_writer.add_scalar("val/epoch_traj_pck_8", init_val_metrics.get("traj_pck_8", 0.0), 0)
            if (not bool(getattr(args, "trajectory_only", False))) and (not bool(getattr(args, "combo_only", False))):
                tb_writer.add_scalar("val/epoch_range_tiou", init_val_metrics.get("range_tiou", 0.0), 0)
                tb_writer.add_scalar("val/epoch_range_r05", init_val_metrics.get("range_r05", 0.0), 0)
            tb_writer.add_scalar("val/epoch_refine_gate_mean", init_val_metrics.get("refine_gate_mean", 0.0), 0)
            tb_writer.add_scalar(
                "val/epoch_refine_delta_abs_mean",
                init_val_metrics.get("refine_delta_abs_mean", 0.0),
                0,
            )
            tb_writer.add_scalar(
                "val/epoch_refine_centroid_shift_abs_mean",
                init_val_metrics.get("refine_centroid_shift_abs_mean", 0.0),
                0,
            )
            tb_writer.add_scalar(
                "val/epoch_anchor_delta_centroid_abs_mean",
                init_val_metrics.get("anchor_delta_centroid_abs_mean", 0.0),
                0,
            )
            tb_writer.add_scalar(
                "val/epoch_anchor_alpha_mean",
                init_val_metrics.get("anchor_alpha_mean", 0.0),
                0,
            )
            tb_writer.add_scalar(
                "val/epoch_anchor_verify_scale_mean",
                init_val_metrics.get("anchor_verify_scale_mean", 1.0),
                0,
            )
            tb_writer.add_scalar(
                "val/epoch_anchor_verify_residual_mean",
                init_val_metrics.get("anchor_verify_residual_mean", 0.0),
                0,
            )
            tb_writer.add_scalar(
                "val/epoch_factor_anchor_gate_mean",
                init_val_metrics.get("factor_anchor_gate_mean", 0.0),
                0,
            )
            tb_writer.add_scalar(
                "val/epoch_factor_offset_gate_mean",
                init_val_metrics.get("factor_offset_gate_mean", 0.0),
                0,
            )
            tb_writer.add_scalar(
                "val/epoch_factor_delta_offset_abs_mean",
                init_val_metrics.get("factor_delta_offset_abs_mean", 0.0),
                0,
            )

    for epoch in range(start_epoch, args.epochs + 1):
        if token_cache.enabled and use_train_video_loader and token_cache.has_all("train", train_set_cache.sids):
            use_train_video_loader = False
            train_loader = _make_loader(train_set_cache, split="train", shuffle=True)
            print(f"[epoch {epoch:03d}] cache warmup complete for train -> switched loader to cache-only mode.")
        if token_cache.enabled and use_val_video_loader and token_cache.has_all("val", val_set_cache.sids):
            use_val_video_loader = False
            val_loader = _make_loader(val_set_cache, split="val", shuffle=False)
            print(f"[epoch {epoch:03d}] cache warmup complete for val -> switched loader to cache-only mode.")

        current_train_stage_weights = _resolve_train_traj_stage_weights(args, epoch)
        current_train_stage_phase = _resolve_train_traj_stage_phase_name(args, epoch)
        _print_stage_header(f"EPOCH {epoch:03d}/{args.epochs:03d}")
        if current_train_stage_weights is not None:
            print(
                f"[epoch {epoch:03d}] stage_curriculum: phase={current_train_stage_phase} "
                f"weights={_format_stage_weight_spec(current_train_stage_weights)}"
            )
        t0 = time.time()
        cache_before = token_cache.stats()
        current_range_combo_reweight_scale = 0.0
        if range_combo_reweight_base_weights_np is not None:
            current_range_combo_reweight_scale = _range_combo_reweight_schedule_scale(
                epoch=epoch,
                start_epoch=int(getattr(args, "range_combo_reweight_start_epoch", 1)),
                ramp_epochs=int(getattr(args, "range_combo_reweight_ramp_epochs", 1)),
            )
            current_weights_np = _blend_combo_reweight_vector(
                range_combo_reweight_base_weights_np,
                current_range_combo_reweight_scale,
            )
            range_combo_reweight_weights = torch.tensor(current_weights_np, dtype=torch.float32, device=device)
            print(
                f"[epoch {epoch:03d}] range_combo_reweight: "
                f"scale={current_range_combo_reweight_scale:.4f} "
                f"start_epoch={int(getattr(args, 'range_combo_reweight_start_epoch', 1))} "
                f"ramp_epochs={int(getattr(args, 'range_combo_reweight_ramp_epochs', 1))}"
            )
        else:
            range_combo_reweight_weights = None
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
            out_dir=out_dir,
            writer=tb_writer,
            global_step_start=global_step,
            token_cache=token_cache,
            combo_class_weights=combo_class_weights,
            start_bin_class_weights=start_bin_class_weights,
            end_bin_class_weights=end_bin_class_weights,
            range_combo_reweight_weights=range_combo_reweight_weights,
            ema=ema,
            traj_late_focal_state=traj_late_focal_state,
        )

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch_loss_total", train_metrics["loss_total"], epoch)
            tb_writer.add_scalar("train/epoch_loss_tool_xy", train_metrics["loss_tool_xy"], epoch)
            tb_writer.add_scalar("train/epoch_loss_ct_xy", train_metrics["loss_ct_xy"], epoch)
            if current_train_stage_weights is not None:
                for stage_idx, stage_weight in enumerate(current_train_stage_weights):
                    tb_writer.add_scalar(
                        f"train/epoch_stage_weight_s{stage_idx}",
                        float(stage_weight),
                        epoch,
                    )
            if bool(getattr(args, "traj_late_focal_enable", False)):
                for key in _TRAJ_LATE_FOCAL_METRIC_KEYS:
                    tb_writer.add_scalar(f"train/epoch_{key}", train_metrics.get(key, 0.0), epoch)
            tb_writer.add_scalar("train/epoch_loss_tool_centroid_aux", train_metrics.get("loss_tool_centroid_aux", 0.0), epoch)
            tb_writer.add_scalar("train/epoch_loss_tool_offset_aux", train_metrics.get("loss_tool_offset_aux", 0.0), epoch)
            tb_writer.add_scalar("train/epoch_loss_tool_pair_sep_aux", train_metrics.get("loss_tool_pair_sep_aux", 0.0), epoch)
            tb_writer.add_scalar("train/epoch_loss_tool_heatmap_ce", train_metrics.get("loss_tool_heatmap_ce", 0.0), epoch)
            tb_writer.add_scalar("train/epoch_loss_ct_heatmap_ce", train_metrics.get("loss_ct_heatmap_ce", 0.0), epoch)
            if not bool(getattr(args, "trajectory_only", False)):
                tb_writer.add_scalar("train/epoch_loss_combo_ce", train_metrics["loss_combo_ce"], epoch)
                tb_writer.add_scalar("train/epoch_loss_combo_tool_ce", train_metrics.get("loss_combo_tool_ce", 0.0), epoch)
                tb_writer.add_scalar("train/epoch_loss_combo_action_ce", train_metrics.get("loss_combo_action_ce", 0.0), epoch)
                tb_writer.add_scalar("train/epoch_combo_acc", train_metrics["combo_acc"], epoch)
                tb_writer.add_scalar("train/epoch_tool_acc", train_metrics.get("tool_acc", 0.0), epoch)
                tb_writer.add_scalar("train/epoch_action_acc", train_metrics.get("action_acc", 0.0), epoch)
                if not bool(getattr(args, "combo_only", False)):
                    tb_writer.add_scalar(
                        "train/epoch_loss_start_l1",
                        train_metrics.get("loss_start_l1", train_metrics["loss_start_bin_ce"]),
                        epoch,
                    )
                    tb_writer.add_scalar(
                        "train/epoch_loss_end_l1",
                        train_metrics.get("loss_end_l1", train_metrics["loss_end_bin_ce"]),
                        epoch,
                    )
                    tb_writer.add_scalar("train/epoch_loss_range", train_metrics["loss_range"], epoch)
                    tb_writer.add_scalar("train/epoch_loss_range_cls", train_metrics.get("loss_range_cls", 0.0), epoch)
                    tb_writer.add_scalar("train/epoch_loss_range_reg", train_metrics.get("loss_range_reg", 0.0), epoch)
                    tb_writer.add_scalar("train/epoch_loss_range_proposal", train_metrics.get("loss_range_proposal", 0.0), epoch)
                    tb_writer.add_scalar("train/epoch_loss_range_sola", train_metrics.get("loss_range_sola", 0.0), epoch)
            tb_writer.add_scalar("train/epoch_traj_ade_px", train_metrics.get("traj_ade_px", 0.0), epoch)
            tb_writer.add_scalar("train/epoch_traj_aj", train_metrics.get("traj_aj", 0.0), epoch)
            tb_writer.add_scalar("train/epoch_traj_pck_8", train_metrics.get("traj_pck_8", 0.0), epoch)
            if (not bool(getattr(args, "trajectory_only", False))) and (not bool(getattr(args, "combo_only", False))):
                tb_writer.add_scalar("train/epoch_range_tiou", train_metrics.get("range_tiou", 0.0), epoch)
                tb_writer.add_scalar("train/epoch_range_r05", train_metrics.get("range_r05", 0.0), epoch)
            tb_writer.add_scalar("train/epoch_refine_gate_mean", train_metrics.get("refine_gate_mean", 0.0), epoch)
            tb_writer.add_scalar(
                "train/epoch_refine_delta_abs_mean",
                train_metrics.get("refine_delta_abs_mean", 0.0),
                epoch,
            )
            tb_writer.add_scalar(
                "train/epoch_refine_centroid_shift_abs_mean",
                train_metrics.get("refine_centroid_shift_abs_mean", 0.0),
                epoch,
            )
            tb_writer.add_scalar(
                "train/epoch_anchor_delta_centroid_abs_mean",
                train_metrics.get("anchor_delta_centroid_abs_mean", 0.0),
                epoch,
            )
            tb_writer.add_scalar(
                "train/epoch_anchor_alpha_mean",
                train_metrics.get("anchor_alpha_mean", 0.0),
                epoch,
            )
            tb_writer.add_scalar(
                "train/epoch_anchor_verify_scale_mean",
                train_metrics.get("anchor_verify_scale_mean", 1.0),
                epoch,
            )
            tb_writer.add_scalar(
                "train/epoch_anchor_verify_residual_mean",
                train_metrics.get("anchor_verify_residual_mean", 0.0),
                epoch,
            )
            tb_writer.add_scalar(
                "train/epoch_factor_anchor_gate_mean",
                train_metrics.get("factor_anchor_gate_mean", 0.0),
                epoch,
            )
            tb_writer.add_scalar(
                "train/epoch_factor_offset_gate_mean",
                train_metrics.get("factor_offset_gate_mean", 0.0),
                epoch,
            )
            tb_writer.add_scalar(
                "train/epoch_factor_delta_offset_abs_mean",
                train_metrics.get("factor_delta_offset_abs_mean", 0.0),
                epoch,
            )

        if bool(getattr(args, "trajectory_only", False)):
            print(
                f"[epoch {epoch:03d}] train: total={train_metrics['loss_total']:.5f} "
                f"tool={train_metrics['loss_tool_xy']:.5f} ct={train_metrics['loss_ct_xy']:.5f} "
                f"{_format_traj_stage_summary(train_metrics)} "
                f"ctr_aux={train_metrics.get('loss_tool_centroid_aux', 0.0):.5f} "
                f"off_aux={train_metrics.get('loss_tool_offset_aux', 0.0):.5f} "
                f"sep_aux={train_metrics.get('loss_tool_pair_sep_aux', 0.0):.5f} "
                f"tool_hm_ce={train_metrics.get('loss_tool_heatmap_ce', 0.0):.5f} "
                f"ct_hm_ce={train_metrics.get('loss_ct_heatmap_ce', 0.0):.5f} "
                f"traj_ade_px={train_metrics.get('traj_ade_px', 0.0):.3f} "
                f"ref_gate={train_metrics.get('refine_gate_mean', 0.0):.4f} "
                f"ref_dxy={train_metrics.get('refine_delta_abs_mean', 0.0):.5f} "
                f"ref_cshift={train_metrics.get('refine_centroid_shift_abs_mean', 0.0):.5f} "
                f"anchor_cshift={train_metrics.get('anchor_delta_centroid_abs_mean', 0.0):.5f} "
                f"st0_noise={train_metrics.get('stage0_centroid_noise_abs_mean', 0.0):.5f} "
                f"anchor_alpha={train_metrics.get('anchor_alpha_mean', 0.0):.4f} "
                f"anchor_vfy={train_metrics.get('anchor_verify_scale_mean', 1.0):.4f} "
                f"factor_ag={train_metrics.get('factor_anchor_gate_mean', 0.0):.4f} "
                f"factor_og={train_metrics.get('factor_offset_gate_mean', 0.0):.4f} "
                f"factor_off={train_metrics.get('factor_delta_offset_abs_mean', 0.0):.5f} "
                f"ctx_gate={train_metrics.get('context_gate_mean', 0.0):.4f} "
                f"belief_gate={train_metrics.get('belief_gate_mean', 0.0):.4f} "
                f"jnt_gate={train_metrics.get('joint_gate_mean', 0.0):.4f} "
                f"tpl_gate={train_metrics.get('template_gate_mean', 0.0):.4f} "
                f"tpl_ent={train_metrics.get('template_entropy_mean', 0.0):.4f}"
            )
        elif bool(getattr(args, "combo_only", False)):
            if bool(getattr(args, "tool_only", False)):
                print(
                    f"[epoch {epoch:03d}] train: total={train_metrics['loss_total']:.5f} "
                    f"tool_ce={train_metrics.get('loss_combo_tool_ce', 0.0):.5f} "
                    f"tool_acc={train_metrics.get('tool_acc', 0.0):.4f}"
                )
            elif bool(getattr(args, "combo_probe_attn", False)):
                print(
                    f"[epoch {epoch:03d}] train: total={train_metrics['loss_total']:.5f} "
                    f"combo_ce={train_metrics['loss_combo_ce']:.5f} "
                    f"combo_acc={train_metrics['combo_acc']:.4f}"
                )
            else:
                print(
                    f"[epoch {epoch:03d}] train: total={train_metrics['loss_total']:.5f} "
                    f"combo_ce={train_metrics['loss_combo_ce']:.5f} "
                    f"tool_ce={train_metrics.get('loss_combo_tool_ce', 0.0):.5f} "
                    f"action_ce={train_metrics.get('loss_combo_action_ce', 0.0):.5f} "
                    f"combo_acc={train_metrics['combo_acc']:.4f} "
                    f"tool_acc={train_metrics.get('tool_acc', 0.0):.4f} "
                    f"action_acc={train_metrics.get('action_acc', 0.0):.4f}"
                )
        else:
            print(
                f"[epoch {epoch:03d}] train: total={train_metrics['loss_total']:.5f} "
                f"tool={train_metrics['loss_tool_xy']:.5f} ct={train_metrics['loss_ct_xy']:.5f} "
                f"combo_ce={train_metrics['loss_combo_ce']:.5f} "
                f"start_l1={train_metrics.get('loss_start_l1', train_metrics['loss_start_bin_ce']):.5f} "
                f"end_l1={train_metrics.get('loss_end_l1', train_metrics['loss_end_bin_ce']):.5f} "
                f"range={train_metrics['loss_range']:.5f} "
                f"range_cls={train_metrics.get('loss_range_cls', 0.0):.5f} "
                f"range_reg={train_metrics.get('loss_range_reg', 0.0):.5f} "
                f"range_prop={train_metrics.get('loss_range_proposal', 0.0):.5f} "
                f"range_qual={train_metrics.get('loss_range_quality', 0.0):.5f} "
                f"range_sola={train_metrics.get('loss_range_sola', 0.0):.5f} "
                f"combo_acc={train_metrics['combo_acc']:.4f} "
                f"traj_ade_px={train_metrics.get('traj_ade_px', 0.0):.3f} "
                f"traj_aj={train_metrics.get('traj_aj', 0.0):.4f} "
                f"traj_pck8={train_metrics.get('traj_pck_8', 0.0):.4f} "
                f"range_tiou={train_metrics.get('range_tiou', 0.0):.4f} "
                f"range_r05={train_metrics.get('range_r05', 0.0):.4f} "
                f"ref_gate={train_metrics.get('refine_gate_mean', 0.0):.4f} "
                f"ref_dxy={train_metrics.get('refine_delta_abs_mean', 0.0):.5f} "
                f"ref_cshift={train_metrics.get('refine_centroid_shift_abs_mean', 0.0):.5f} "
                f"anchor_cshift={train_metrics.get('anchor_delta_centroid_abs_mean', 0.0):.5f} "
                f"anchor_alpha={train_metrics.get('anchor_alpha_mean', 0.0):.4f} "
                f"anchor_vfy={train_metrics.get('anchor_verify_scale_mean', 1.0):.4f}"
            )

        val_due = (epoch % int(args.val_interval) == 0) or (epoch == int(args.epochs))
        save_epoch_checkpoint = (
            ((epoch % int(args.save_interval) == 0) or (epoch == int(args.epochs)))
            and (not bool(getattr(args, "save_best_only", False)))
        )
        val_metrics_epoch: Optional[Dict[str, Any]] = None
        current_best_metric: Optional[float] = None
        should_stop_after_epoch = False
        stop_reason = ""
        if val_due:
            with _use_ema_weights(model, ema):
                val_metrics, val_by_combo, val_combo_fpfn = evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    args=args,
                    combo_id_to_name=combo_id_to_name,
                    epoch=epoch,
                    token_cache=token_cache,
                    range_combo_reweight_weights=range_combo_reweight_weights,
                    return_combo_fpfn_report=True,
                )
            val_by_tool = aggregate_val_by_tool(val_by_combo)
            if bool(getattr(args, "trajectory_only", False)):
                print(
                    f"[epoch {epoch:03d}]   val: total={val_metrics['loss_total']:.5f} "
                    f"tool={val_metrics['loss_tool_xy']:.5f} ct={val_metrics['loss_ct_xy']:.5f} "
                    f"{_format_traj_stage_summary(val_metrics)} "
                    f"ctr_aux={val_metrics.get('loss_tool_centroid_aux', 0.0):.5f} "
                    f"off_aux={val_metrics.get('loss_tool_offset_aux', 0.0):.5f} "
                    f"sep_aux={val_metrics.get('loss_tool_pair_sep_aux', 0.0):.5f} "
                    f"tool_hm_ce={val_metrics.get('loss_tool_heatmap_ce', 0.0):.5f} "
                    f"ct_hm_ce={val_metrics.get('loss_ct_heatmap_ce', 0.0):.5f} "
                    f"traj_ade_px={val_metrics.get('traj_ade_px', 0.0):.3f} "
                    f"ref_gate={val_metrics.get('refine_gate_mean', 0.0):.4f} "
                    f"ref_dxy={val_metrics.get('refine_delta_abs_mean', 0.0):.5f} "
                    f"ref_cshift={val_metrics.get('refine_centroid_shift_abs_mean', 0.0):.5f} "
                    f"anchor_cshift={val_metrics.get('anchor_delta_centroid_abs_mean', 0.0):.5f} "
                    f"st0_noise={val_metrics.get('stage0_centroid_noise_abs_mean', 0.0):.5f} "
                    f"anchor_alpha={val_metrics.get('anchor_alpha_mean', 0.0):.4f} "
                    f"anchor_vfy={val_metrics.get('anchor_verify_scale_mean', 1.0):.4f} "
                    f"factor_ag={val_metrics.get('factor_anchor_gate_mean', 0.0):.4f} "
                    f"factor_og={val_metrics.get('factor_offset_gate_mean', 0.0):.4f} "
                    f"factor_off={val_metrics.get('factor_delta_offset_abs_mean', 0.0):.5f} "
                    f"ctx_gate={val_metrics.get('context_gate_mean', 0.0):.4f} "
                    f"belief_gate={val_metrics.get('belief_gate_mean', 0.0):.4f} "
                    f"jnt_gate={val_metrics.get('joint_gate_mean', 0.0):.4f} "
                    f"tpl_gate={val_metrics.get('template_gate_mean', 0.0):.4f} "
                    f"tpl_ent={val_metrics.get('template_entropy_mean', 0.0):.4f} "
                    f"conf_tool={val_metrics['conf_tool_mean']:.4f} conf_ct={val_metrics['conf_ct_mean']:.4f}"
                )
            elif bool(getattr(args, "combo_only", False)):
                if bool(getattr(args, "tool_only", False)):
                    print(
                        f"[epoch {epoch:03d}]   val: total={val_metrics['loss_total']:.5f} "
                        f"tool_ce={val_metrics.get('loss_combo_tool_ce', 0.0):.5f} "
                        f"tool_acc={val_metrics.get('tool_acc', 0.0):.4f}"
                    )
                elif bool(getattr(args, "combo_probe_attn", False)):
                    print(
                        f"[epoch {epoch:03d}]   val: total={val_metrics['loss_total']:.5f} "
                        f"combo_ce={val_metrics['loss_combo_ce']:.5f} "
                        f"combo_acc={val_metrics['combo_acc']:.4f}"
                    )
                else:
                    print(
                        f"[epoch {epoch:03d}]   val: total={val_metrics['loss_total']:.5f} "
                        f"combo_ce={val_metrics['loss_combo_ce']:.5f} "
                        f"tool_ce={val_metrics.get('loss_combo_tool_ce', 0.0):.5f} "
                        f"action_ce={val_metrics.get('loss_combo_action_ce', 0.0):.5f} "
                        f"combo_acc={val_metrics['combo_acc']:.4f} "
                        f"tool_acc={val_metrics.get('tool_acc', 0.0):.4f} "
                        f"action_acc={val_metrics.get('action_acc', 0.0):.4f}"
                    )
                print_val_by_combo(
                    val_by_combo,
                    combo_only=bool(getattr(args, "combo_only", False)),
                    tool_only=bool(getattr(args, "tool_only", False)),
                    combo_probe_attn=bool(getattr(args, "combo_probe_attn", False)),
                )
            else:
                print(
                    f"[epoch {epoch:03d}]   val: total={val_metrics['loss_total']:.5f} "
                    f"tool={val_metrics['loss_tool_xy']:.5f} ct={val_metrics['loss_ct_xy']:.5f} "
                    f"combo_ce={val_metrics['loss_combo_ce']:.5f} "
                    f"start_l1={val_metrics.get('loss_start_l1', val_metrics['loss_start_bin_ce']):.5f} "
                    f"end_l1={val_metrics.get('loss_end_l1', val_metrics['loss_end_bin_ce']):.5f} "
                    f"range={val_metrics['loss_range']:.5f} "
                    f"range_cls={val_metrics.get('loss_range_cls', 0.0):.5f} "
                    f"range_reg={val_metrics.get('loss_range_reg', 0.0):.5f} "
                    f"range_prop={val_metrics.get('loss_range_proposal', 0.0):.5f} "
                    f"range_qual={val_metrics.get('loss_range_quality', 0.0):.5f} "
                    f"range_sola={val_metrics.get('loss_range_sola', 0.0):.5f} "
                    f"combo_acc={val_metrics['combo_acc']:.4f} "
                    f"traj_ade_px={val_metrics.get('traj_ade_px', 0.0):.3f} "
                    f"traj_aj={val_metrics.get('traj_aj', 0.0):.4f} "
                    f"traj_pck8={val_metrics.get('traj_pck_8', 0.0):.4f} "
                    f"range_tiou={val_metrics.get('range_tiou', 0.0):.4f} "
                    f"range_r05={val_metrics.get('range_r05', 0.0):.4f} "
                    f"ref_gate={val_metrics.get('refine_gate_mean', 0.0):.4f} "
                    f"ref_dxy={val_metrics.get('refine_delta_abs_mean', 0.0):.5f} "
                    f"ref_cshift={val_metrics.get('refine_centroid_shift_abs_mean', 0.0):.5f} "
                    f"anchor_cshift={val_metrics.get('anchor_delta_centroid_abs_mean', 0.0):.5f} "
                    f"anchor_alpha={val_metrics.get('anchor_alpha_mean', 0.0):.4f} "
                    f"anchor_vfy={val_metrics.get('anchor_verify_scale_mean', 1.0):.4f} "
                    f"factor_ag={val_metrics.get('factor_anchor_gate_mean', 0.0):.4f} "
                    f"factor_og={val_metrics.get('factor_offset_gate_mean', 0.0):.4f} "
                    f"factor_off={val_metrics.get('factor_delta_offset_abs_mean', 0.0):.5f} "
                    f"conf_tool={val_metrics['conf_tool_mean']:.4f} conf_ct={val_metrics['conf_ct_mean']:.4f}"
                )
                print_val_by_combo(
                    val_by_combo,
                    combo_only=bool(getattr(args, "combo_only", False)),
                    tool_only=bool(getattr(args, "tool_only", False)),
                    combo_probe_attn=bool(getattr(args, "combo_probe_attn", False)),
                )
            val_metrics_epoch = val_metrics

            if tb_writer is not None:
                tb_writer.add_scalar("val/epoch_loss_total", val_metrics["loss_total"], epoch)
                tb_writer.add_scalar("val/epoch_loss_tool_xy", val_metrics["loss_tool_xy"], epoch)
                tb_writer.add_scalar("val/epoch_loss_ct_xy", val_metrics["loss_ct_xy"], epoch)
                tb_writer.add_scalar("val/epoch_loss_tool_centroid_aux", val_metrics.get("loss_tool_centroid_aux", 0.0), epoch)
                tb_writer.add_scalar("val/epoch_loss_tool_offset_aux", val_metrics.get("loss_tool_offset_aux", 0.0), epoch)
                tb_writer.add_scalar("val/epoch_loss_tool_pair_sep_aux", val_metrics.get("loss_tool_pair_sep_aux", 0.0), epoch)
                tb_writer.add_scalar("val/epoch_loss_tool_heatmap_ce", val_metrics.get("loss_tool_heatmap_ce", 0.0), epoch)
                tb_writer.add_scalar("val/epoch_loss_ct_heatmap_ce", val_metrics.get("loss_ct_heatmap_ce", 0.0), epoch)
                if not bool(getattr(args, "trajectory_only", False)):
                    tb_writer.add_scalar("val/epoch_loss_combo_ce", val_metrics["loss_combo_ce"], epoch)
                    tb_writer.add_scalar("val/epoch_loss_combo_tool_ce", val_metrics.get("loss_combo_tool_ce", 0.0), epoch)
                    tb_writer.add_scalar("val/epoch_loss_combo_action_ce", val_metrics.get("loss_combo_action_ce", 0.0), epoch)
                    tb_writer.add_scalar("val/epoch_combo_acc", val_metrics["combo_acc"], epoch)
                    tb_writer.add_scalar("val/epoch_tool_acc", val_metrics.get("tool_acc", 0.0), epoch)
                    tb_writer.add_scalar("val/epoch_action_acc", val_metrics.get("action_acc", 0.0), epoch)
                    if not bool(getattr(args, "combo_only", False)):
                        tb_writer.add_scalar(
                            "val/epoch_loss_start_l1",
                            val_metrics.get("loss_start_l1", val_metrics["loss_start_bin_ce"]),
                            epoch,
                        )
                        tb_writer.add_scalar(
                            "val/epoch_loss_end_l1",
                            val_metrics.get("loss_end_l1", val_metrics["loss_end_bin_ce"]),
                            epoch,
                        )
                        tb_writer.add_scalar("val/epoch_loss_range", val_metrics["loss_range"], epoch)
                        tb_writer.add_scalar("val/epoch_loss_range_cls", val_metrics.get("loss_range_cls", 0.0), epoch)
                        tb_writer.add_scalar("val/epoch_loss_range_reg", val_metrics.get("loss_range_reg", 0.0), epoch)
                        tb_writer.add_scalar("val/epoch_loss_range_proposal", val_metrics.get("loss_range_proposal", 0.0), epoch)
                        tb_writer.add_scalar("val/epoch_loss_range_sola", val_metrics.get("loss_range_sola", 0.0), epoch)
                tb_writer.add_scalar("val/epoch_traj_ade_px", val_metrics.get("traj_ade_px", 0.0), epoch)
                tb_writer.add_scalar("val/epoch_traj_aj", val_metrics.get("traj_aj", 0.0), epoch)
                tb_writer.add_scalar("val/epoch_traj_pck_8", val_metrics.get("traj_pck_8", 0.0), epoch)
                if (not bool(getattr(args, "trajectory_only", False))) and (not bool(getattr(args, "combo_only", False))):
                    tb_writer.add_scalar("val/epoch_range_tiou", val_metrics.get("range_tiou", 0.0), epoch)
                    tb_writer.add_scalar("val/epoch_range_r05", val_metrics.get("range_r05", 0.0), epoch)
                tb_writer.add_scalar("val/epoch_refine_gate_mean", val_metrics.get("refine_gate_mean", 0.0), epoch)
                tb_writer.add_scalar(
                    "val/epoch_refine_delta_abs_mean",
                    val_metrics.get("refine_delta_abs_mean", 0.0),
                    epoch,
                )
                tb_writer.add_scalar(
                    "val/epoch_refine_centroid_shift_abs_mean",
                    val_metrics.get("refine_centroid_shift_abs_mean", 0.0),
                    epoch,
                )
                tb_writer.add_scalar(
                    "val/epoch_anchor_delta_centroid_abs_mean",
                    val_metrics.get("anchor_delta_centroid_abs_mean", 0.0),
                    epoch,
                )
                tb_writer.add_scalar(
                    "val/epoch_anchor_alpha_mean",
                    val_metrics.get("anchor_alpha_mean", 0.0),
                    epoch,
                )
                tb_writer.add_scalar(
                    "val/epoch_anchor_verify_scale_mean",
                    val_metrics.get("anchor_verify_scale_mean", 1.0),
                    epoch,
                )
                tb_writer.add_scalar(
                    "val/epoch_anchor_verify_residual_mean",
                    val_metrics.get("anchor_verify_residual_mean", 0.0),
                    epoch,
                )
                tb_writer.add_scalar(
                    "val/epoch_factor_anchor_gate_mean",
                    val_metrics.get("factor_anchor_gate_mean", 0.0),
                    epoch,
                )
                tb_writer.add_scalar(
                    "val/epoch_factor_offset_gate_mean",
                    val_metrics.get("factor_offset_gate_mean", 0.0),
                    epoch,
                )
                tb_writer.add_scalar(
                    "val/epoch_factor_delta_offset_abs_mean",
                    val_metrics.get("factor_delta_offset_abs_mean", 0.0),
                    epoch,
                )
                tb_writer.add_scalar("val/epoch_conf_tool_mean", val_metrics["conf_tool_mean"], epoch)
                tb_writer.add_scalar("val/epoch_conf_ct_mean", val_metrics["conf_ct_mean"], epoch)
                if not bool(getattr(args, "trajectory_only", False)):
                    for combo_name, m in val_by_combo.items():
                        safe = _tb_safe_name(combo_name)
                        tb_writer.add_scalar(f"val_combo/{safe}_combo_acc", m["combo_acc"], epoch)
                        tb_writer.add_scalar(f"val_combo/{safe}_tool_acc", m.get("tool_acc", 0.0), epoch)
                        tb_writer.add_scalar(f"val_combo/{safe}_action_acc", m.get("action_acc", 0.0), epoch)
                        tb_writer.add_scalar(f"val_combo/{safe}_traj_ade_px", m.get("traj_ade_px", 0.0), epoch)
                        tb_writer.add_scalar(f"val_combo/{safe}_traj_aj", m.get("traj_aj", 0.0), epoch)
                        tb_writer.add_scalar(f"val_combo/{safe}_traj_pck_8", m.get("traj_pck_8", 0.0), epoch)
                        tb_writer.add_scalar(f"val_combo/{safe}_range_tiou", m.get("range_tiou", 0.0), epoch)
                        tb_writer.add_scalar(f"val_combo/{safe}_range_r05", m.get("range_r05", 0.0), epoch)
                        tb_writer.add_scalar(f"val_combo/{safe}_loss_total", m["loss_total"], epoch)

            current_best_metric = _extract_best_monitor_value(args, val_metrics, best_val)
            val_history_row = {
                "epoch": int(epoch),
                "metric": float(current_best_metric),
                "loss_total": float(val_metrics.get("loss_total", 0.0)),
                "range_tiou": float(val_metrics.get("range_tiou", 0.0)),
                "range_r05": float(val_metrics.get("range_r05", 0.0)),
            }
            val_history_tail.append(val_history_row)
            if len(val_history_tail) > 12:
                val_history_tail = val_history_tail[-12:]

            target_reached = (
                target_range_tiou_enabled
                and (not bool(getattr(args, "trajectory_only", False)))
                and (not bool(getattr(args, "combo_only", False)))
                and float(val_metrics.get("range_tiou", 0.0)) >= target_range_tiou
            )
            if target_reached:
                should_stop_after_epoch = True
                stop_reason = "target_range_tiou_reached"
                print(
                    f"[target-stop] reached at epoch {epoch:03d}: "
                    f"val_range_tiou={float(val_metrics.get('range_tiou', 0.0)):.6f} "
                    f">= target={target_range_tiou:.6f}"
                )
            elif early_stop_enabled:
                early_stop_improved = _is_better_best_monitor_with_margin(
                    args,
                    current_best_metric,
                    early_stop_ref_val,
                    min_delta=float(getattr(args, "early_stop_min_delta", 0.0)),
                )
                if early_stop_improved:
                    early_stop_ref_val = float(current_best_metric)
                    early_stop_ref_epoch = int(epoch)
                    early_stop_bad_epochs = 0
                    print(
                        f"[early-stop] reset at epoch {epoch:03d}: "
                        f"val_{best_metric_name}={early_stop_ref_val:.6f}"
                    )
                elif int(epoch) >= int(getattr(args, "early_stop_warmup_epochs", 0)):
                    early_stop_bad_epochs += 1
                    print(
                        f"[early-stop] no meaningful val_{best_metric_name} improvement "
                        f"for {early_stop_bad_epochs}/{int(getattr(args, 'early_stop_patience', 0))} "
                        f"val(s); ref_epoch={early_stop_ref_epoch:03d} "
                        f"ref_{best_metric_name}={early_stop_ref_val:.6f}"
                    )
                    if early_stop_bad_epochs >= int(getattr(args, "early_stop_patience", 0)):
                        should_stop_after_epoch = True
                        stop_reason = "early_stop_patience_exhausted"
                else:
                    print(
                        f"[early-stop] warmup active at epoch {epoch:03d}; "
                        f"counting starts from epoch {int(getattr(args, 'early_stop_warmup_epochs', 0)):03d}"
                    )

            if save_epoch_checkpoint:
                ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
                save_checkpoint(
                    path=ckpt_path,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    args=args,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    val_by_combo=val_by_combo,
                    val_by_tool=val_by_tool,
                    ema=ema,
                    traj_late_focal_state=traj_late_focal_state,
                    best_monitor_name=best_metric_name,
                    best_monitor_value=best_val,
                    best_monitor_epoch=best_epoch,
                    early_stop_ref_value=early_stop_ref_val,
                    early_stop_ref_epoch=early_stop_ref_epoch,
                    early_stop_bad_epochs=early_stop_bad_epochs,
                )
                combo_json_path = ckpt_dir / f"val_by_combo_epoch_{epoch:03d}.json"
                combo_json_path.write_text(json.dumps(val_by_combo, indent=2), encoding="utf-8")
                tool_json_path = ckpt_dir / f"val_by_tool_epoch_{epoch:03d}.json"
                tool_json_path.write_text(json.dumps(val_by_tool, indent=2), encoding="utf-8")
                print(f"[checkpoint] saved: {ckpt_path}")
                print(f"[checkpoint] saved: {combo_json_path}")
                print(f"[checkpoint] saved: {tool_json_path}")
                if val_combo_fpfn is not None:
                    combo_fpfn_json_path = ckpt_dir / f"val_combo_fpfn_epoch_{epoch:03d}.json"
                    combo_fpfn_json_path.write_text(json.dumps(val_combo_fpfn, indent=2), encoding="utf-8")
                    print(f"[checkpoint] saved: {combo_fpfn_json_path}")

                with _use_ema_weights(model, ema):
                    vis_val_info = run_checkpoint_visualization(
                        model=model,
                        val_set=val_set_vis,
                        device=device,
                        args=args,
                        epoch=epoch,
                        output_dir=out_dir,
                        combo_id_to_name=combo_id_to_name,
                        token_cache=token_cache,
                        range_combo_reweight_weights=range_combo_reweight_weights,
                        split_name="val",
                        selection_mode="top_vis",
                    )
                if vis_val_info.get("enabled") == "true":
                    print(
                        f"[checkpoint] visuals[val]: samples={vis_val_info.get('saved_samples','0')} "
                        f"path={vis_val_info.get('path','')}"
                    )
                with _use_ema_weights(model, ema):
                    vis_train_info = run_checkpoint_visualization(
                        model=model,
                        val_set=train_set_video,
                        device=device,
                        args=args,
                        epoch=epoch,
                        output_dir=out_dir,
                        combo_id_to_name=combo_id_to_name,
                        token_cache=token_cache,
                        range_combo_reweight_weights=range_combo_reweight_weights,
                        split_name="train",
                        selection_mode="random",
                    )
                if vis_train_info.get("enabled") == "true":
                    print(
                        f"[checkpoint] visuals[train]: samples={vis_train_info.get('saved_samples','0')} "
                        f"path={vis_train_info.get('path','')}"
                    )

            if _is_better_best_monitor(args, current_best_metric, best_val):
                best_val = current_best_metric
                best_epoch = int(epoch)
                best_path = ckpt_dir / "best.pt"
                save_checkpoint(
                    path=best_path,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    args=args,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    val_by_combo=val_by_combo,
                    val_by_tool=val_by_tool,
                    ema=ema,
                    traj_late_focal_state=traj_late_focal_state,
                    best_monitor_name=best_metric_name,
                    best_monitor_value=best_val,
                    best_monitor_epoch=best_epoch,
                    early_stop_ref_value=early_stop_ref_val,
                    early_stop_ref_epoch=early_stop_ref_epoch,
                    early_stop_bad_epochs=early_stop_bad_epochs,
                )
                print(
                    f"[checkpoint] updated best: {best_path} "
                    f"(val_{best_metric_name}={best_val:.6f})"
                )
            if should_stop_after_epoch:
                report = {
                    "reason": str(stop_reason or "manual_stop"),
                    "stop_epoch": int(epoch),
                    "best_monitor_name": str(best_metric_name),
                    "best_monitor_mode": str(best_metric_mode),
                    "best_epoch": int(best_epoch),
                    "best_value": float(best_val),
                    "stop_value": float(current_best_metric),
                    "early_stop_patience": int(getattr(args, "early_stop_patience", 0)),
                    "early_stop_min_delta": float(getattr(args, "early_stop_min_delta", 0.0)),
                    "early_stop_warmup_epochs": int(getattr(args, "early_stop_warmup_epochs", 0)),
                    "early_stop_ref_epoch": int(early_stop_ref_epoch),
                    "early_stop_ref_value": float(early_stop_ref_val),
                    "early_stop_bad_epochs": int(early_stop_bad_epochs),
                    "target_range_tiou": float(target_range_tiou),
                    "target_range_tiou_enabled": bool(target_range_tiou_enabled),
                    "recent_val_history": list(val_history_tail[-8:]),
                    "best_checkpoint": str(ckpt_dir / "best.pt"),
                    "epoch_metrics_jsonl": str(epoch_metrics_path),
                }
                report_path, prompt_path = _write_early_stop_bundle(out_dir, report=report)
                stop_prefix = "[target-stop]" if stop_reason == "target_range_tiou_reached" else "[early-stop]"
                print(f"{stop_prefix} wrote report: {report_path}")
                print(f"{stop_prefix} wrote prompt: {prompt_path}")
                print(
                    f"{stop_prefix} stopping at epoch {epoch:03d}; "
                    f"best_epoch={best_epoch:03d} best_{best_metric_name}={best_val:.6f}"
                )
        else:
            print(f"[epoch {epoch:03d}]   val: skipped (will run on val interval={args.val_interval})")

        dt = time.time() - t0
        cache_after = token_cache.stats()
        dh = int(cache_after["hits"]) - int(cache_before["hits"])
        dm = int(cache_after["misses"]) - int(cache_before["misses"])
        dw = int(cache_after["writes"]) - int(cache_before["writes"])
        print(
            f"[epoch {epoch:03d}] cache_delta: hits={dh} misses={dm} writes={dw} "
            f"(cum hits={cache_after['hits']} misses={cache_after['misses']} writes={cache_after['writes']})"
        )
        print(f"[epoch {epoch:03d}] elapsed={dt:.1f}s")
        _append_jsonl(
            epoch_metrics_path,
            {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "elapsed_sec": float(dt),
                "val_due": bool(val_due),
                "train_metrics": _sanitize_metrics_for_json(train_metrics),
                "val_metrics": _sanitize_metrics_for_json(val_metrics_epoch),
                "best_monitor": {
                    "name": str(best_metric_name),
                    "mode": str(best_metric_mode),
                    "value": float(best_val),
                    "epoch": int(best_epoch),
                },
                "early_stop": {
                    "enabled": bool(early_stop_enabled),
                    "ref_value": float(early_stop_ref_val),
                    "ref_epoch": int(early_stop_ref_epoch),
                    "bad_epochs": int(early_stop_bad_epochs),
                    "patience": int(getattr(args, "early_stop_patience", 0)),
                    "min_delta": float(getattr(args, "early_stop_min_delta", 0.0)),
                    "warmup_epochs": int(getattr(args, "early_stop_warmup_epochs", 0)),
                    "target_range_tiou": float(target_range_tiou),
                    "target_range_tiou_enabled": bool(target_range_tiou_enabled),
                    "reason": str(stop_reason),
                    "triggered": bool(should_stop_after_epoch),
                },
                "cache_delta": {
                    "hits": int(dh),
                    "misses": int(dm),
                    "writes": int(dw),
                },
                "range_combo_reweight": {
                    "enabled": bool(range_combo_reweight_base_weights_np is not None),
                    "scale": float(current_range_combo_reweight_scale),
                    "start_epoch": int(getattr(args, "range_combo_reweight_start_epoch", 1)),
                    "ramp_epochs": int(getattr(args, "range_combo_reweight_ramp_epochs", 1)),
                },
            },
        )
        if should_stop_after_epoch:
            break

    if tb_writer is not None:
        tb_writer.close()

    print(f"[done] vjepa_token_cache: {token_cache.format_stats()}")
    print("[done] training finished")


if __name__ == "__main__":
    main()
