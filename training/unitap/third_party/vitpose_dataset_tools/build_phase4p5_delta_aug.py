#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-4.5 builder (fixed-16f clips + combo classification targets + keypoint supervision)

Key decisions (FINAL):
1) Sampling window starts from clip first frame (`affordance_range.start` / min frameIndex).
   We do not search for a "best" start frame.

2) Each sample is uniformly sampled to TARGET_FRAMES (=16) absolute frames.
   Files:
     keypoints/{sid}.npy        -> (16,3,2), raw sequence
     keypoint_vis/{sid}.npy     -> (16,3), raw visibility
     contact_vis_raw/{sid}.npy  -> (16,4), raw contact visibility
     sampled_abs_frames/{sid}.npy -> (16,)
     action_range_abs/{sid}.npy -> (2,), [action_start_abs, action_end_abs]
     action_range_mask/{sid}.npy -> scalar float32, 1=valid action_range in annotation, 0=missing/invalid
   If `INTERPOLATE_VIS0_TO_FULL_SUPERVISION=True`, exported `gt_*_vis` become soft weights
   while `gt_*_xy` coordinates stay unchanged.

3) Motions saved to motions/{sid}.npy are STEP-DELTA / RELATIVE representations:
    step[t] = abs[t+1] - abs[t]
    Shape (L,6), float32 where L=T-1.

4) contacts/{sid}.npy stores 4 contact points from t=0 frame:
   - canonical CCW order
   - start from "top-right" (max x, tie max y)
   - if missing, fallback to a synthetic tool-centered box

5) Split rules by Tool|Action combo:
   - If combo_count >= 100:
       val = 2 (diverse across cases)
       train = 400 (diverse across cases)
       test = remaining
       NO augmentation for this combo
   - If combo_count < 100:
       val = 2 (diverse across cases)
       train = remaining
       test = empty
       Augmentation ONLY for train, with per-combo multipliers:
         clipper|clip      : +1 per train sample
         scissors|cut      : +1 per train sample
         scissors|dissect  : +1 or +2 (half 2, half 1)
         grasper|dissect   : +7 per train sample

6) Debug images for geometric augmentation:
   For each augmented sample using geometric transform, save BEFORE/AFTER overlays:
     debug_aug/{aug_sid}_before.jpg (base)
     debug_aug/{aug_sid}_after.jpg  (augmented)

Expected dataset layout under root_dir (folder containing case dirs):
  case_x/
    annotation/*.json
    clips/{json_stem}.mp4
The mp4 is assumed to start at affordance_range.start (frame 0 corresponds to that start).
"""

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import cv2

# =========================
# Config
# =========================
# OUT_NAME = "surg_act_video_toy_phase4p5_delta_contactstart"
OUT_NAME_BASE = "surg_act_video_toy_phase4p5_toolstart_16f_kpt_2"
# Output mode:
#   - "full": keep all legacy exports
#   - "minimal_kpt_combo": keep only files needed by current training idea
#   - "tool_kpt_only": keep only tool-trajectory training contract
#   - "combo_only_train": keep only combo classification training contract
OUTPUT_MODE = "combo_only_train"



PRINT_NORMALIZED_MOTION = True   # 只打印，不存
PRINT_NORMALIZED_GEO = False     # 只打印，不存（你现在要关）
# Export t=0 geo features (contact_area/tool_area/center_dist/edge_cos).
# Disabled by default for current training pipeline.
EXPORT_GEO_FEATURES = False


SEED = 1234

CONTACT_MIN = 4
CONTACT_KEEP = 4
CONTACT_PICK_MODE = "even"  # evenly spaced along canonical CCW order
START_RATIO_TH = 0.4

# -------------------------
# Keypoint clip settings
# -------------------------
# Fixed-length clip for ViT *-16f backbones.
TARGET_FRAMES = 16
# Train-only offset-base expansion (applied before offline augmentation).
# Multiplier means total windows per original base clip (including original start).
# Example: multiplier=3 -> original + 2 offset windows.
USE_OFFSET_BASE_EXPANSION = True
OFFSET_BASE_MIN_SEG_LEN = 40
OFFSET_BASE_MULTIPLIER_BY_COMBO: Dict[str, int] = {
    "hook|dissect": 1,
    "grasper|grasp": 1,
    "grasper|dissect": 2,
    "clipper|clip": 3,
    "scissors|cut": 3,
    "scissors|dissect": 2,
}
# Optional per-combo stage-1 target after offset-base expansion.
# This lets temporal boundary diversity increase train-base support without
# consuming the full final target that should still be completed by offline aug.
OFFSET_BASE_TARGET_PER_COMBO: Dict[str, int] = {}
# Start frame policy: by default use clip first frame for sampling window.
START_REQUIRE_VISIBLE_TOOL = False
ALLOW_START_FALLBACK_ANY_TOOL = True
# Use clip first frame as the start of sampling window.
USE_CLIP_START_AS_ABS_START = True
# Visibility-based quality filtering on sampled 16-frame clip.
MIN_VIS_FRAMES_PER_POINT = 0
MIN_VIS_TOTAL = 0
# Optional legacy filter (usually disable for fixed-16f training).
ENFORCE_HOOK_DISSECT_MIN_STEPS = False
# Pseudo-label supervision for vis==0 frames
ENABLE_PSEUDO_SUPERVISION = True
PSEUDO_WEIGHT = 0.05
# Interpolate vis==0 frames and output dense soft-supervision masks.
INTERPOLATE_VIS0_TO_FULL_SUPERVISION = True
# Soft visibility minimum weight for interpolated / unobserved frames.
# Example: vis pattern 1-0-1 => middle frame gets 0.5.
SOFT_VIS_MIN_WEIGHT = 0.05

# Action-range binning config used by training dataset.
ACTION_BIN_SIZE = 0.1
ACTION_NUM_BINS = int(round(1.0 / ACTION_BIN_SIZE))

# tool topology -> 3 points only
# 2A1T: [anchor0, anchor1, tip0]
# 1A2T: [anchor0, tip0, tip1]
TOOL2TOPO = {
    "hook": "2A1T",
    "grasper": "1A2T",
    "scissors": "1A2T",
    "clipper": "1A2T",
}

TOOL2ID = {"hook": 0, "grasper": 1, "scissors": 2, "clipper": 3}
ACT2ID = {"dissect": 0, "grasp": 1, "cut": 2, "clip": 3}
# Final classification target: 6 valid tool|action combos in this dataset.
COMBO2ID = {
    "hook|dissect": 0,
    "grasper|grasp": 1,
    "clipper|clip": 2,
    "scissors|cut": 3,
    "scissors|dissect": 4,
    "grasper|dissect": 5,
}


def combo_name(tool: str, action: str) -> str:
    return f"{tool}|{action}"

# Split sizes (requested):
#   - big combos (>=100): train=400, val=2, rest=test
#   - small combos (<100): val=2, rest=train
VAL_K_BIG = 2
TRAIN_K_BIG = 400
VAL_K_SMALL = 2
# Optional hard caps applied to every combo (override big/small policy when >0).
TRAIN_K_PER_COMBO: Optional[int] = None
VAL_K_PER_COMBO: Optional[int] = None

# =========================
# Split mode
# =========================
SPLIT_MODE = "legacy"   # "legacy" or "neighbor"
VAL_EVERY_K = 2           # 每 K 个 train，插 1 个 val（只影响 neighbor 模式的“分布形态”）

# =========================
# Folder-level split (selected train/val folders)
# =========================
# When enabled:
#   1) only clips from SELECTED_CASE_FOLDERS are exported
#   2) train/val is assigned strictly by folder name
#   3) split is determined by VAL_CASE_FOLDERS (all other selected -> train), no test
#   4) if TRAIN_K_PER_COMBO / VAL_K_PER_COMBO is set, sampling is capped per combo
#      within folder-disjoint train/val pools (selected folders are not fully consumed)
#   5) optional val backfill can pull only missing-combo samples from train folders
#      (sample-level move, not whole-folder move) to meet val caps for rare combos
USE_FOLDER_SPLIT_10_1 = True
SELECTED_CASE_FOLDERS: List[str] = [
    "case_002_video_part_002_segment_2",
    "case_034_video_part_001_segment_4",
    "case_057_video_part_002_segment_3",
    "case_010_video_part_002_segment_2",
    "case_015_video_part_001_segment_2",
    "case_042_video_part_002_segment_2",
    "case_049_video_part_002_segment_1",
    "cholec80_02",
    "cholec80_09",
    "cholec80_10",
    "cholec80_12",
    "cholec80_13",
    "cholec80_19",
    "cholec80_21",
    "cholec80_22",
    "cholec80_23",
    "cholec80_27",
    "cholec80_28",
    "comprehensive_005_left",
    "heichole_001",
    "heichole_002",
    "heichole_003",
    "heichole_004",
    "heichole_006",
    "heichole_009",
    "heichole_012",
    "youtube_cholecystectomy_chenyan_00000",
    "youtube_cholecystectomy_chenyan_00002",
    "youtube_cholecystectomy_chenyan_00005",
    "youtube_cholecystectomy_chenyan_00010",
    "youtube_cholecystectomy_chenyan_00012",
    "youtube_cholecystectomy_chenyan_00013",
    "youtube_cholecystectomy_chenyan_00016",
    "youtube_cholecystectomy_chenyan_00020",
]
# Additional train-folder expansion discovered via combo distribution search
# (mainly to increase train counts for clipper|clip / scissors|cut / grasper|dissect).
ADDITIONAL_TRAIN_CASE_FOLDERS: List[str] = [
    "youtube_cholecystectomy_chenyan_00001",
    "case_004_video_part_002_segment_2",
    "case_056_video_part_002_segment_3",
    "case_013_video_part_002_segment_2",
    "youtube_cholecystectomy_chenyan_00006",
    "case_032_video_part_002_segment_2",
    "youtube_cholecystectomy_chenyan_00023",
    "youtube_cholecystectomy_chenyan_00007",
    "cholec80_17",
    "case_025_video_part_002_segment_1",
    "youtube_cholecystectomy_chenyan_00017",
    "case_038_video_part_002_segment_3",
    "youtube_cholecystectomy_chenyan_00003",
    "youtube_cholecystectomy_chenyan_00009",
    "cholec80_30",
    "youtube_cholecystectomy_chenyan_00019",
    "youtube_cholecystectomy_chenyan_00011",
    "youtube_cholecystectomy_chenyan_00008",
    "case_022_video_part_002_segment_2",
    "case_064_video_part_002_segment_1",
    "cholec80_20",
    "cholec80_06",
    "heichole_011",
    "cholec80_14",
    "youtube_cholecystectomy_chenyan_00028",
    "cholec80_07",
    "cholec80_18",
    "cholec80_29",
    "cholec80_25",
    "heichole_008",
    "cholec80_11",
    "heichole_010",
    "cholec80_24",
    "cholec80_16",
    "cholec80_05",
    "cholec80_03",
    "cholec80_26",
    "cholec80_15",
    "youtube_cholecystectomy_chenyan_00004",
    "cholec80_08",
    "cholec80_04",
]
SELECTED_CASE_FOLDERS = sorted(list(dict.fromkeys(SELECTED_CASE_FOLDERS + ADDITIONAL_TRAIN_CASE_FOLDERS)))
VAL_CASE_FOLDERS: List[str] = [
    "case_002_video_part_002_segment_2",
    "case_042_video_part_002_segment_2",
    "heichole_001",
]
FOLDER_SPLIT_VAL_BACKFILL = True
# Disable optional filtering rules for this run.
# Hard validity checks (I/O errors, malformed labels, missing required tool points) still apply.
DISABLE_FILTER_RULES = True

# Visibility-priority sampling:
# for large combos, prefer clips with higher average visibility.
VIS_PRIORITY_FOR_BIG = False
VIS_PRIORITY_MIN_COMBO_COUNT = 100

# Weak-combo augmentation policy:
#   - apply only to explicitly listed weak combos (TRAIN split)
#   - use mild ratio-based augmentation with a hard cap per combo
#   - do not try to match dominant classes (e.g., hook|dissect)
ENABLE_WEAK_COMBO_AUG = True
WEAK_COMBO_AUG_RATIO: Dict[str, float] = {
    "clipper|clip": 0.35,
    "scissors|cut": 0.40,
    "scissors|dissect": 0.20,
    "grasper|dissect": 0.15,
}
WEAK_COMBO_AUG_MAX_EXTRA: Dict[str, int] = {
    "clipper|clip": 220,
    "scissors|cut": 220,
    "scissors|dissect": 180,
    "grasper|dissect": 180,
}

# Aug type distribution for weak-combo augmentation (sum ~= 1.0).
AUG_PROBS = {
    "A": 0.60,  # translation + rotation (geometric, warp image)
    "B": 0.20,  # translation + scale (geometric, warp image)
    "C": 0.10,  # time warp (no image warp)
    "D": 0.10,  # low-frequency noise (no image warp)
}

# Geometric augmentation magnitudes (normalized)
TRANS_MAX = 0.02          # +-2%
ROT_DEG_MAX = 10.0        # +-10 degrees
SCALE_MIN, SCALE_MAX = 0.95, 1.05

# Time warp
TIME_RATIO_MIN, TIME_RATIO_MAX = 0.90, 1.10

# Noise
NOISE_STD = 0.002
NOISE_EMA = 0.90

# =========================
# Fold-1 clip-level post augmentation (classification-focused)
# =========================
# This policy augments TRAIN clips only and is restricted to fold-1 train sids.
# It operates on exported `video_16_npy` + synchronized gt arrays, i.e. AFTER
# black-bar/UI-bar removal and coordinate remapping performed in base extraction.
USE_FOLD1_CLIP_POST_AUG = True
FOLD1_SPLIT_DIR = Path(__file__).resolve().parents[1] / "work_dirs" / "10fold_case_sep_abpointrefiner_b32_trajloss_nfold_v1cfg" / "nfold_splits"
FOLD1_TRAIN_SIDS_FILE = FOLD1_SPLIT_DIR / "fold_01_train.txt"
FOLD1_VAL_SIDS_FILE = FOLD1_SPLIT_DIR / "fold_01_val.txt"
FOLD1_TRAIN_CASES_FILE = FOLD1_SPLIT_DIR / "fold_01_train_cases.txt"
FOLD1_VAL_CASES_FILE = FOLD1_SPLIT_DIR / "fold_01_val_cases.txt"
# If True, fold-1 post-augmentation scope is selected ONLY by case lists
# (fold_01_train_cases.txt / fold_01_val_cases.txt), not by sid files.
FOLD1_AUG_SCOPE_BY_CASE_ONLY = True
FOLD1_AUG_SERIES_NAME = "default"

# Target train counts (post-aug) per combo for fold-1 experiments.
FOLD1_TARGET_TRAIN_PER_COMBO: Dict[str, int] = {
    # Pro-Max v3.2 (20k) quota targets from PLAN_PRO_MAX_AUG_MIGRATION.md
    "hook|dissect": 4200,
    "grasper|grasp": 5000,
    "grasper|dissect": 4000,
    "clipper|clip": 3000,
    "scissors|cut": 2500,
    "scissors|dissect": 1300,
}
FOLD1_SKIP_AUG_COMBOS: Set[str] = set()
# Optional per-combo cap on eligible TRAIN base samples before augmentation.
# Useful when one combo is overly dominant in selected cases.
FOLD1_TRAIN_BASE_CAP_PER_COMBO: Dict[str, int] = {}
# For capped combos (currently hook|dissect), sample evenly across cases first.
FOLD1_CAP_SAMPLE_BY_CASE_BALANCED = False
# Apply capped eligible fold1 train ids to final train split, not augmentation-only.
FOLD1_APPLY_CAP_TO_FINAL_TRAIN = False

# Pro-Max v3 strength policy:
# Each generated sample is tagged as light/medium/heavy per-combo by this mix,
# then operator probabilities/magnitudes are chosen from FOLD1_STRENGTH_PRESETS.
FOLD1_USE_V3_STRENGTH_MIX = True
FOLD1_STRENGTH_MIX_BY_COMBO: Dict[str, Dict[str, float]] = {
    "grasper|dissect": {"light": 0.10, "medium": 0.40, "heavy": 0.42, "extreme": 0.08},
    "grasper|grasp": {"light": 0.15, "medium": 0.45, "heavy": 0.35, "extreme": 0.05},
    "clipper|clip": {"light": 0.10, "medium": 0.35, "heavy": 0.45, "extreme": 0.10},
    "scissors|cut": {"light": 0.10, "medium": 0.35, "heavy": 0.45, "extreme": 0.10},
    "hook|dissect": {"light": 0.25, "medium": 0.60, "heavy": 0.14, "extreme": 0.01},
    "scissors|dissect": {"light": 0.25, "medium": 0.55, "heavy": 0.18, "extreme": 0.02},
    "default": {"light": 0.20, "medium": 0.50, "heavy": 0.27, "extreme": 0.03},
}

# Extreme strategy controls.
FOLD1_EXTREME_ONLY_HARD = True
FOLD1_EXTREME_HARD_TOP_FRAC = 0.30
FOLD1_EXTREME_HARD_MIN_SAMPLES = 8
FOLD1_EXTREME_PRIORITY_COMBOS: Set[str] = {
    "grasper|dissect",
    "grasper|grasp",
    "clipper|clip",
    "scissors|cut",
}

# Hard lock: force generated offline samples to high-difficulty buckets.
# When enabled, light/medium are promoted to heavy, and extreme share is
# enforced via hard-pool promotion (for label-clean pressure training).
FOLD1_FORCE_HIGH_DIFFICULTY_ONLY = True
FOLD1_FORCE_HIGH_DIFFICULTY_MIN_EXTREME_PRIORITY = 0.30
FOLD1_FORCE_HIGH_DIFFICULTY_MIN_EXTREME_DEFAULT = 0.15

# Donor diversity constraints for high-k generation (to avoid near-duplicate variants).
FOLD1_DONOR_DIVERSITY_K_THRESHOLD = 4
FOLD1_DONOR_DIVERSITY_MIN_UNIQUE = 6

# Per-strength augmentation profile (values not listed here fallback to current globals).
FOLD1_STRENGTH_PRESETS: Dict[str, Dict[str, float]] = {
    "light": {
        "temp_warp_p": 0.08,
        "hflip_p": 0.25,
        "affine_p": 0.60,
        "affine_rot_deg": 6.0,
        "affine_scale_min": 0.97,
        "affine_scale_max": 1.03,
        "affine_trans_frac": 0.015,
        "bg_replace_p": 0.35,
        "bg_same_combo_p": 0.30,
        "color_p": 0.70,
        "brightness": 0.08,
        "contrast": 0.12,
        "saturation": 0.12,
        "hue": 0.01,
        "gamma_p": 0.30,
        "gamma_min": 0.90,
        "gamma_max": 1.15,
        "blur_p": 0.10,
        "blur_sigma_min": 0.30,
        "blur_sigma_max": 0.90,
        "noise_p": 0.10,
        "noise_std_min": 1.0,
        "noise_std_max": 6.0,
        "jpeg_p": 0.12,
        "jpeg_q_min": 60.0,
        "jpeg_q_max": 95.0,
        "occlusion_event_p": 0.00,
        "pstm_event_p": 0.00,
    },
    "medium": {
        "temp_warp_p": 0.25,
        "hflip_p": 0.50,
        "affine_p": 0.85,
        "affine_rot_deg": 10.0,
        "affine_scale_min": 0.95,
        "affine_scale_max": 1.06,
        "affine_trans_frac": 0.02,
        "bg_replace_p": 0.75,
        "bg_same_combo_p": 0.15,
        "color_p": 0.95,
        "brightness": 0.15,
        "contrast": 0.20,
        "saturation": 0.20,
        "hue": 0.02,
        "gamma_p": 0.50,
        "gamma_min": 0.80,
        "gamma_max": 1.25,
        "blur_p": 0.20,
        "blur_sigma_min": 0.40,
        "blur_sigma_max": 1.20,
        "noise_p": 0.20,
        "noise_std_min": 2.0,
        "noise_std_max": 10.0,
        "jpeg_p": 0.25,
        "jpeg_q_min": 45.0,
        "jpeg_q_max": 95.0,
        "occlusion_event_p": 0.25,
        "pstm_event_p": 0.15,
    },
    "heavy": {
        "temp_warp_p": 0.45,
        "hflip_p": 0.60,
        "affine_p": 0.95,
        "affine_rot_deg": 15.0,
        "affine_scale_min": 0.90,
        "affine_scale_max": 1.10,
        "affine_trans_frac": 0.05,
        "bg_replace_p": 0.92,
        "bg_same_combo_p": 0.10,
        "color_p": 0.98,
        "brightness": 0.22,
        "contrast": 0.28,
        "saturation": 0.26,
        "hue": 0.03,
        "gamma_p": 0.70,
        "gamma_min": 0.70,
        "gamma_max": 1.35,
        "blur_p": 0.45,
        "blur_sigma_min": 0.60,
        "blur_sigma_max": 2.20,
        "noise_p": 0.45,
        "noise_std_min": 4.0,
        "noise_std_max": 16.0,
        "jpeg_p": 0.60,
        "jpeg_q_min": 30.0,
        "jpeg_q_max": 70.0,
        "occlusion_event_p": 0.75,
        "pstm_event_p": 0.65,
    },
    "extreme": {
        "temp_warp_p": 0.60,
        "hflip_p": 0.65,
        "affine_p": 0.98,
        "affine_rot_deg": 18.0,
        "affine_scale_min": 0.86,
        "affine_scale_max": 1.16,
        "affine_trans_frac": 0.06,
        "bg_replace_p": 0.95,
        "bg_same_combo_p": 0.08,
        "color_p": 0.99,
        "brightness": 0.25,
        "contrast": 0.32,
        "saturation": 0.30,
        "hue": 0.04,
        "gamma_p": 0.80,
        "gamma_min": 0.65,
        "gamma_max": 1.40,
        "blur_p": 0.65,
        "blur_sigma_min": 0.90,
        "blur_sigma_max": 2.80,
        "noise_p": 0.60,
        "noise_std_min": 6.0,
        "noise_std_max": 18.0,
        "jpeg_p": 0.80,
        "jpeg_q_min": 25.0,
        "jpeg_q_max": 55.0,
        "occlusion_event_p": 0.90,
        "pstm_event_p": 0.82,
    },
}
FOLD1_STRENGTH_PRESET_OVERRIDES_BY_COMBO: Dict[str, Dict[str, Dict[str, float]]] = {}

# Crop policy (context / hinge / tip+contact), biased for scissors/clipper.
FOLD1_CROP_MODE_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Bias toward wider context to avoid overly-tight local crops.
    "default": {"context": 0.70, "hinge": 0.18, "tipcontact": 0.12},
    "scissors|cut": {"context": 0.62, "hinge": 0.24, "tipcontact": 0.14},
    "scissors|dissect": {"context": 0.60, "hinge": 0.25, "tipcontact": 0.15},
    "clipper|clip": {"context": 0.62, "hinge": 0.22, "tipcontact": 0.16},
}
FOLD1_CROP_PARAMS: Dict[str, Dict[str, float]] = {
    # Ultra-loose crops: explicitly include more tissue around keypoint regions.
    "context": {"pad_scale": 4.6, "min_size_frac": 0.97, "jitter_center_frac": 0.005, "jitter_scale_lo": 1.13, "jitter_scale_hi": 1.36},
    "hinge": {"pad_scale": 4.2, "min_size_frac": 0.92, "jitter_center_frac": 0.007, "jitter_scale_lo": 1.11, "jitter_scale_hi": 1.33},
    "tipcontact": {"pad_scale": 4.0, "min_size_frac": 0.88, "jitter_center_frac": 0.007, "jitter_scale_lo": 1.09, "jitter_scale_hi": 1.35},
}
# Final hard expansion ring around crop bbox:
# expand by max(FOLD1_CROP_EXTRA_MARGIN_FRAC * min(W,H), FOLD1_CROP_EXTRA_MARGIN_MIN_PX)
FOLD1_CROP_EXTRA_MARGIN_FRAC = 0.18
FOLD1_CROP_EXTRA_MARGIN_MIN_PX = 96

FOLD1_TEMP_WARP_P = 0.25
FOLD1_TEMP_WARP_RATIO_MIN = 0.90
FOLD1_TEMP_WARP_RATIO_MAX = 1.12
# Randomly move temporal start inside first 10% boundary of the sampled clip.
# This is applied during augmentation only, before temporal warp.
FOLD1_TEMP_START_JITTER_P = 0.50
FOLD1_TEMP_START_JITTER_MAX_FRAC = 0.10

FOLD1_HFLIP_P = 0.50
FOLD1_AFFINE_P = 0.85
FOLD1_AFFINE_ROT_DEG = 10.0
FOLD1_AFFINE_SCALE_MIN = 0.95
FOLD1_AFFINE_SCALE_MAX = 1.06
FOLD1_AFFINE_TRANS_FRAC = 0.02

FOLD1_BG_REPLACE_P = 0.75
FOLD1_BG_SAME_COMBO_P = 0.15
FOLD1_MASK_DILATE_K = 35
FOLD1_MASK_BLUR_K = 25
FOLD1_MASK_SKELETON_THICK_FRAC = 0.018
FOLD1_MASK_POINT_RADIUS_FRAC = 0.095
FOLD1_MASK_POINT_RADIUS_MIN_PX = 48
FOLD1_MASK_FILL_ALL_KPT_HULL = True

FOLD1_COLOR_P = 0.95
FOLD1_BRIGHTNESS = 0.15
FOLD1_CONTRAST = 0.20
FOLD1_SATURATION = 0.20
FOLD1_HUE = 0.02
FOLD1_GAMMA_P = 0.50
FOLD1_GAMMA_MIN = 0.80
FOLD1_GAMMA_MAX = 1.25
FOLD1_BLUR_P = 0.20
FOLD1_BLUR_SIGMA_MIN = 0.40
FOLD1_BLUR_SIGMA_MAX = 1.20
FOLD1_NOISE_P = 0.20
FOLD1_NOISE_STD_MIN = 2.0
FOLD1_NOISE_STD_MAX = 10.0
FOLD1_JPEG_P = 0.25
FOLD1_JPEG_Q_MIN = 45
FOLD1_JPEG_Q_MAX = 95
FOLD1_DEBUG_SAVE_PROB = 0.03
DEBUG_AUG_ONE_EXAMPLE_PER_TYPE = True
FOLD1_DEBUG_FIXED_PAIR_COUNT = 10
FOLD1_DEBUG_FIXED_DIR_NAME = "debug_aug_10"

# Augmentation logging (runtime progress verbosity)
AUG_LOG_EVERY_N_SAMPLES = 100
AUG_LOG_PER_BASE_SAMPLE = False

# Debug images
SAVE_DEBUG_AUG_IMAGES = True
DEBUG_DIR_NAME = "debug_aug"
DEBUG_DRAW_RADIUS = 5

# Random clip-level visualization sanity check:
# sample K base clips and render all 16 sampled frames with stored keypoints + vis.
EXPORT_RANDOM_CLIP_VIS = False
RANDOM_CLIP_VIS_SAMPLES = 3
RANDOM_CLIP_VIS_DIR_NAME = "debug_clip_vis_16f"

# Pre-export sampled 16 cropped frames as npy for fast training loading.
EXPORT_VIDEO16_NPY = True
VIDEO16_DIR_NAME = "video_16_npy"
# Optional external/staging directory for video_16_npy.
# When set, we create/require out_root/video_16_npy -> <override> symlink.
VIDEO16_DIR_OVERRIDE = ""
# Dump helper: avoid inlining extremely large arrays into debug json print.
DEBUG_INLINE_ARRAY_MAX_ELEMS = 20000

# =========================
# Mask image export (NEW)
# =========================
# If True, also save a black/white pose-contact mask image per sample
# (mask is built from anchor_tip + contact points at t=0).
ENABLE_MASK_IMAGE = (OUTPUT_MODE == "full")
MASK_DIR_NAME = "mask_images"
MASK_RADIUS_PX = 2
MASK_BORDER_PX = 2  # outline thickness for clearer boundary
MASK_BORDER_COLOR = 0  # 0=black outline on white fill

def _mode_suffix(mode: str) -> str:
    if mode == "minimal_kpt_combo":
        return "_minimal"
    if mode == "tool_kpt_only":
        return "_toolkpt"
    if mode == "combo_only_train":
        return "_combo"
    return ""


def _refresh_output_mode_dependent_config() -> None:
    global ENABLE_MASK_IMAGE, OUT_NAME
    ENABLE_MASK_IMAGE = (OUTPUT_MODE == "full")
    OUT_NAME = OUT_NAME_BASE + _mode_suffix(OUTPUT_MODE) + ("_mask" if ENABLE_MASK_IMAGE else "")


_refresh_output_mode_dependent_config()

# Valid length
MIN_T = 2  # require at least 2 frames (delta meaningful)
# hook|dissect 轨迹最小 step 数（step-delta 的长度 L）
HOOK_DISSECT_MIN_STEPS = 150

# =========================
# Coordinate mode
# =========================
USE_PIXEL_COORDS = False     # ✅ True: x,y 都用像素；False: 归一化到[0,1]
STD_EPS = 1e-12

# =========================
# Black bar removal (pillarbox crop)
# =========================
REMOVE_BLACK_BARS = True        # ✅ True: crop black bars in saved images AND remap all coordinates
CROP_TOP_BOTTOM = False        # 只裁左右黑边（你现在的需求）
BAR_GRAY_THR = 10              # 灰度阈值：<=thr 认为“黑”
BAR_MIN_NONBLACK_FRAC = 0.02   # 一列中 >thr 的像素占比 >= 2% 才算“内容列”
BAR_MIN_BORDER_PX = 8          # 小于这个宽度的边不裁
BAR_MIN_KEEP_FRAC = 0.55       # 裁完后内容区域至少保留原来的 55%（防误裁）
# 2026/02/02

# 2026/02/03
# =========================
# UI bar removal (top/bottom overlay crop)
# =========================
REMOVE_UI_BARS = True          # ✅ True: 自动裁上/下 UI 条（警告条/底部工具栏/边缘时间条）
UI_TOP_SCAN_FRAC = 0.25        # 只在图像上方 25% 内找“上边界”
UI_BOT_SCAN_FRAC = 0.30        # 只在图像下方 30% 内找“下边界”

UI_SMOOTH_K = 11               # 纵向梯度平滑核（奇数）
UI_PEAK_RATIO = 0.55           # 峰值阈值比例（越大越保守）
UI_PAD_Y = 2                   # 边界额外再裁 2px（去掉分隔线）

UI_DARK_THR = 35               # “黑”阈值（用于判断上/下区域像不像 UI bar）
UI_BRIGHT_THR = 220            # “白”阈值（用于判断时间条那种白底条）
UI_BAR_STD_RATIO = 0.85        # bar 区域的 std 必须明显低于内容区域 std
UI_MIN_BAR_PX = 8              # 小于 8px 的裁剪忽略
UI_MIN_KEEP_FRAC_H = 0.60      # 裁完后高度至少保留 60%，否则认为误裁 -> 不裁
# =========================
# Point-aware crop (NEW)
# =========================
# 目的：如果 detect_content_bbox 裁完后有任何点（tool/motion/contact）出界，
#      不再回退到 full-frame，而是把 bbox 向外扩到“包含所有点”为止。
POINT_AWARE_CROP = True          # ✅ True: bbox 会被扩张以包含所有点
POINT_CROP_PAD_PX = 2            # 点到边界额外留的像素 padding
POINT_CROP_PAD_FRAC = 0.0        # 额外按比例 padding：min(W,H)*frac（可设 0.005 等）
# 2026/02/03

# 2026/01/30
# =========================
# Motion representation (NEW)
# =========================
# "stepdelta": (L,6) where L=T-1, step[t]=abs[t+1]-abs[t]
# "relative":  (L,6) where L=T-1, rel[t]=abs[t+1]-abs[0]   (relative-to-first, cumulative)
MOTION_REP = "relative"   # <-- relative 模式

# If MOTION_REP == "relative":
#   False: store (T-1,6) without the explicit t=0 zero row (recommended)
#   True : store (T,6) with rel[0]=0
REL_INCLUDE_T0 = False
# 2026/01/30

# =========================
# Debug: print delta distributions & mean checks
# =========================
PRINT_TRAIN_DELTA_STATS = True

# =========================
# Data containers
# =========================
@dataclass
class BaseItem:
    sid: str
    case: str
    ann_relpath: str
    tool: str
    action: str
    combo: str
    tool_id: int
    act_id: int
    combo_id: int
    topology: str
    pt_role: List[int]
    W: int
    H: int

    # absolute frame indices in original/global space (from annotation)
    video_base_abs: int  # affordance_range.start (assumed to be video frame 0)
    abs_start: int       # sampling-window start (clip first frame when configured)
    abs_end: int         # end (global abs frameIndex)
    num_frames: int      # T
    action_start_abs: int
    action_end_abs: int
    action_range_mask: int

    # stats
    max_contact_visible_in_window: int
    vis_count_per_point: Optional[List[int]] = None
    tool_vis_mean: Optional[float] = None
    contact_vis_mean: Optional[float] = None
    vis_score: Optional[float] = None
    crop_bbox_xyxy: Optional[List[int]] = None

    # split assigned later
    split: str = "unset"  # train/val/test

    # augmentation fields (for augmented items)
    source_sid: Optional[str] = None
    aug_type: Optional[str] = None
    aug_params: Optional[dict] = None


# class RunningStats:
#     """Streaming mean/std for D dims."""
#     def __init__(self, D: int):
#         self.D = D
#         self.n = 0
#         self.mean = np.zeros((D,), dtype=np.float64)
#         self.M2 = np.zeros((D,), dtype=np.float64)
#
#     def update(self, X: np.ndarray):
#         # X: (N, D)
#         X = np.asarray(X, dtype=np.float64)
#         if X.ndim != 2 or X.shape[1] != self.D:
#             raise ValueError(f"RunningStats.update expects (N,{self.D}), got {X.shape}")
#         for i in range(X.shape[0]):
#             self.n += 1
#             delta = X[i] - self.mean
#             self.mean += delta / self.n
#             delta2 = X[i] - self.mean
#             self.M2 += delta * delta2
#
#     def finalize(self, eps=1e-8):
#         if self.n < 2:
#             raise RuntimeError("Not enough samples to compute std.")
#         var = self.M2 / self.n
#         std = np.sqrt(var) + eps
#         return self.mean.astype(np.float32), std.astype(np.float32)


# =========================
# Utility helpers
# =========================
def print_percentiles(name, arr, ps=(5, 50, 95)):
    """
    arr: (N, ...) numpy array
    """
    arr = np.asarray(arr).reshape(-1)
    out = {}
    for p in ps:
        out[p] = float(np.percentile(arr, p))
    print(f"[PCTL] {name}: " + ", ".join([f"p{p}={out[p]:.4g}" for p in ps]))
    return out

def _poly_area_xy(xy: np.ndarray) -> float:
    x = xy[:, 0].astype(np.float64)
    y = xy[:, 1].astype(np.float64)
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    return 0.5 * float(np.abs(np.sum(x * y2 - y * x2)))

def _tri_area_xy(tri: np.ndarray) -> float:
    a, b, c = tri
    ab = b - a
    ac = c - a
    return 0.5 * float(abs(ab[0] * ac[1] - ab[1] * ac[0]))

def _longest_edge_dir(contact4: np.ndarray) -> np.ndarray:
    edges = []
    lens2 = []
    for i in range(4):
        j = (i + 1) % 4
        e = contact4[j] - contact4[i]
        edges.append(e)
        lens2.append(float(e[0] ** 2 + e[1] ** 2))
    return edges[int(np.argmax(lens2))]

def _cos_abs_2d(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    return abs(max(-1.0, min(1.0, cos)))

def is_vis(p: dict) -> bool:
    v = p.get("vis", True)
    return bool(v)


def normalize_xy(x, y, W, H):
    """
    If USE_PIXEL_COORDS:
        return pixel coordinates (float)
    else:
        return normalized coordinates in [0,1]
    """
    if USE_PIXEL_COORDS:
        return float(x), float(y)
    return float(x) / float(W), float(y) / float(H)



def sort_xy_asc(xy: np.ndarray):
    """Sort points by (x asc, y asc)."""
    xy = np.asarray(xy, dtype=np.float32)
    if xy.shape[0] == 0:
        return xy
    order = np.lexsort((xy[:, 1], xy[:, 0]))
    return xy[order]


def canonical_order_xy(xy: np.ndarray):
    """
    xy: (K,2) float
    return: CCW order starting from top-right (max x, tie max y)
    """
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"canonical_order_xy expects (K,2), got {xy.shape}")
    if xy.shape[0] == 0:
        return xy

    center = xy.mean(axis=0)
    angles = np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0])
    order_ccw = np.argsort(angles)
    xy_ccw = xy[order_ccw]

    # start: max x, tie max y
    start_idx = np.lexsort((-xy_ccw[:, 1], -xy_ccw[:, 0]))[0]
    xy_ccw = np.roll(xy_ccw, -start_idx, axis=0)
    return xy_ccw


def pick_contact4(contact_xy_kx2: np.ndarray):
    """
    Input: (K,2), K>=4
    Output: (4,2) representative points, canonical CCW (top-right start).
    Strategy:
      - canonical all points
      - pick 4 evenly spread indices if K>4
      - canonical again on chosen 4
    """
    xy = np.asarray(contact_xy_kx2, dtype=np.float32)
    k = xy.shape[0]
    if k < CONTACT_KEEP:
        raise ValueError("pick_contact4 requires K>=4")

    xy_ccw = canonical_order_xy(xy)

    if k == CONTACT_KEEP:
        xy4 = xy_ccw
    else:
        if CONTACT_PICK_MODE == "even":
            # better tail coverage than linspace(0,k,4,endpoint=False)
            idx = np.linspace(0, k - 1, CONTACT_KEEP).round().astype(int)
            idx = np.unique(idx)
            # ensure 4 distinct indices (rare edge cases)
            if idx.shape[0] < CONTACT_KEEP:
                pool = [i for i in range(k) if i not in set(idx.tolist())]
                while idx.shape[0] < CONTACT_KEEP and len(pool) > 0:
                    idx = np.append(idx, pool.pop(0))
            idx = idx[:CONTACT_KEEP]
            xy4 = xy_ccw[idx]
        else:
            xy4 = xy_ccw[:CONTACT_KEEP]

    return canonical_order_xy(xy4).astype(np.float32)


def get_topology(tool: str) -> str:
    return TOOL2TOPO.get(tool, "2A1T")


def topology_roles(topology: str) -> np.ndarray:
    if topology == "2A1T":
        return np.array([0, 0, 1], dtype=np.int64)
    if topology == "1A2T":
        return np.array([0, 1, 1], dtype=np.int64)
    return np.array([0, 0, 1], dtype=np.int64)


def _collect_tool_points(
    items: List[dict],
    W: int,
    H: int,
    ptype: str,
    require_vis: bool,
) -> np.ndarray:
    """Collect points as (x, y, vis), ordered by uid first to keep point identity stable."""
    rows = []
    for p in items:
        if p.get("type") != ptype:
            continue
        vis = bool(is_vis(p))
        if require_vis and (not vis):
            continue
        try:
            x, y = normalize_xy(p["x"], p["y"], W, H)
        except Exception:
            continue
        uid_raw = p.get("uid", None)
        uid_val = None
        try:
            if uid_raw is not None:
                uid_val = int(uid_raw)
        except Exception:
            uid_val = None
        rows.append((uid_val, x, y, 1.0 if vis else 0.0))

    if len(rows) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    rows.sort(key=lambda z: (
        1 if z[0] is None else 0,
        (10 ** 9) if z[0] is None else z[0],
        z[1],
        z[2],
    ))
    arr = np.asarray([[x, y, v] for (_, x, y, v) in rows], dtype=np.float32)
    return arr


def _pick_first_n(points_xyv: np.ndarray, n: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if points_xyv.shape[0] < n:
        return None

    picked = points_xyv[:n]
    xy = picked[:, :2].astype(np.float32)
    vis = picked[:, 2].astype(np.float32)
    return xy, vis


def extract_points_and_vis_for_frame(
    items: List[dict],
    W: int,
    H: int,
    topology: str,
    require_vis: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Return tool keypoints and visibility for one frame.
    Output:
      - pts3: (3,2)
      - vis3: (3,) in {0,1}
    """
    anchors = _collect_tool_points(items, W, H, "tool_anchor", require_vis=require_vis)
    tips = _collect_tool_points(items, W, H, "tool_tip", require_vis=require_vis)

    if topology == "2A1T":
        sel_a = _pick_first_n(anchors, 2)
        sel_t = _pick_first_n(tips, 1)
        if sel_a is None or sel_t is None:
            return None
        a_xy, a_vis = sel_a
        t_xy, t_vis = sel_t
        pts3 = np.concatenate([a_xy, t_xy], axis=0).astype(np.float32)
        vis3 = np.concatenate([a_vis, t_vis], axis=0).astype(np.float32)
        return pts3, vis3

    if topology == "1A2T":
        sel_a = _pick_first_n(anchors, 1)
        sel_t = _pick_first_n(tips, 2)
        if sel_a is None or sel_t is None:
            return None
        a_xy, a_vis = sel_a
        t_xy, t_vis = sel_t
        pts3 = np.concatenate([a_xy, t_xy], axis=0).astype(np.float32)
        vis3 = np.concatenate([a_vis, t_vis], axis=0).astype(np.float32)
        return pts3, vis3

    return None


def extract_points_for_frame(
    items: List[dict],
    W: int,
    H: int,
    topology: str,
    require_vis: bool = True,
) -> Optional[np.ndarray]:
    got = extract_points_and_vis_for_frame(
        items=items,
        W=W,
        H=H,
        topology=topology,
        require_vis=require_vis,
    )
    if got is None:
        return None
    return got[0]


def canonical_order_xyv(xyv: np.ndarray) -> np.ndarray:
    """CCW order on xy, keep the 3rd channel (e.g., vis) aligned."""
    arr = np.asarray(xyv, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"canonical_order_xyv expects (K,>=3), got {arr.shape}")
    if arr.shape[0] == 0:
        return arr

    xy = arr[:, :2]
    center = xy.mean(axis=0)
    angles = np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0])
    order_ccw = np.argsort(angles)
    out = arr[order_ccw]

    start_idx = np.lexsort((-out[:, 1], -out[:, 0]))[0]
    out = np.roll(out, -start_idx, axis=0)
    return out.astype(np.float32)


def pick_contact4_xyv(contact_xyv_kx3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input: (K,3) where channels are (x,y,vis)
    Output:
      - xy4: (4,2), canonical CCW
      - vis4: (4,), aligned with xy4
    """
    arr = np.asarray(contact_xyv_kx3, dtype=np.float32)
    k = arr.shape[0]
    if k < CONTACT_KEEP:
        raise ValueError("pick_contact4_xyv requires K>=4")

    arr_ccw = canonical_order_xyv(arr)

    if k == CONTACT_KEEP:
        arr4 = arr_ccw
    else:
        if CONTACT_PICK_MODE == "even":
            idx = np.linspace(0, k - 1, CONTACT_KEEP).round().astype(int)
            idx = np.unique(idx)
            if idx.shape[0] < CONTACT_KEEP:
                pool = [i for i in range(k) if i not in set(idx.tolist())]
                while idx.shape[0] < CONTACT_KEEP and len(pool) > 0:
                    idx = np.append(idx, pool.pop(0))
            idx = idx[:CONTACT_KEEP]
            arr4 = arr_ccw[idx]
        else:
            arr4 = arr_ccw[:CONTACT_KEEP]

    arr4 = canonical_order_xyv(arr4)
    xy4 = arr4[:, :2].astype(np.float32)
    vis4 = (arr4[:, 2] > 0.5).astype(np.float32)
    return xy4, vis4


def extract_contact4_and_vis_for_frame(
    items: List[dict],
    W: int,
    H: int,
    require_vis: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    rows = []
    for p in items:
        if p.get("type") != "contact":
            continue
        vis = bool(is_vis(p))
        if require_vis and (not vis):
            continue
        try:
            x, y = normalize_xy(p["x"], p["y"], W, H)
        except Exception:
            continue
        rows.append([x, y, 1.0 if vis else 0.0])

    if len(rows) < CONTACT_MIN:
        return None
    try:
        return pick_contact4_xyv(np.asarray(rows, dtype=np.float32))
    except Exception:
        return None


def extract_contact4_for_frame(items: List[dict], W: int, H: int, require_vis: bool = True) -> Optional[np.ndarray]:
    got = extract_contact4_and_vis_for_frame(items, W, H, require_vis=require_vis)
    if got is None:
        return None
    return got[0]


def sample_fixed_abs_indices(abs_start: int, abs_end: int, n_frames: int) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n_frames == 1:
        return np.array([int(abs_start)], dtype=np.int64)
    if abs_end <= abs_start:
        return np.full((n_frames,), int(abs_start), dtype=np.int64)
    return np.round(np.linspace(abs_start, abs_end, n_frames)).astype(np.int64)


def offset_start_positions(L: int, clip_len: int, K_total: int) -> List[int]:
    """
    Generate offset window starts for a segment.
    Uses both:
      - start_i = round(i * max_start / (K_total - 1))
      - stride = max(clip_len // 2, floor((L - clip_len) / K_total))
    and always keeps the last window (max_start) to avoid dropping the final 16 frames.
    """
    l = int(L)
    c = int(max(1, clip_len))
    k = int(max(1, K_total))
    if k <= 1 or l <= c:
        return [0]

    max_start = int(max(0, l - c))
    if max_start <= 0:
        return [0]

    raw = [int(round(float(i) * float(max_start) / float(max(1, k - 1)))) for i in range(k)]
    starts: List[int] = []
    for s in raw:
        ss = int(np.clip(int(s), 0, max_start))
        if len(starts) == 0 or ss != starts[-1]:
            starts.append(ss)
    if len(starts) <= 1:
        return [0]

    stride_rule = int(max(int(c // 2), int(math.floor(float(l - c) / float(max(1, k))))))
    if stride_rule <= 0:
        return starts

    kept: List[int] = [starts[0]]
    for i, s in enumerate(starts[1:], start=1):
        is_last = (i == len(starts) - 1)
        if is_last:
            if int(s) != int(kept[-1]):
                kept.append(int(s))
            continue
        if int(s) - int(kept[-1]) >= stride_rule:
            kept.append(int(s))

    # Ensure the final window is always preserved.
    if kept[-1] != max_start:
        kept.append(int(max_start))

    # If stride_rule is too strict, collapse middle windows but keep first/last.
    if len(kept) >= 3:
        last = int(kept[-1])
        prev = int(kept[-2])
        if last - prev < stride_rule:
            kept = [int(kept[0]), int(last)]

    out: List[int] = []
    for s in kept:
        ss = int(np.clip(int(s), 0, max_start))
        if len(out) == 0 or ss != out[-1]:
            out.append(ss)
    return out if len(out) > 0 else [0]


def order_offset_extra_starts(starts: List[int], base_offset: int) -> List[int]:
    """
    Order extra temporal windows by boundary diversity, not by raw offset value.
    For tool-trajectory A-series runs we want window diversity without exploding
    counts, so we prioritize the windows farthest from the base clip start first.
    """
    base = int(base_offset)
    dedup: List[int] = []
    seen: Set[int] = set()
    for s in starts:
        ss = int(s)
        if ss == base or ss in seen:
            continue
        seen.add(ss)
        dedup.append(ss)
    dedup.sort(key=lambda x: (-abs(int(x) - base), int(x)))
    return dedup


def parse_action_range_abs(
    ann_data: dict,
    default_start_abs: int,
    default_end_abs: int,
    stats: dict,
) -> Tuple[int, int, int]:
    """
    Parse action_range from annotation.
    Returns:
      action_start_abs, action_end_abs, mask
    mask=1 when action_range exists and is valid, else 0.
    """
    act = ann_data.get("action_range", None)
    if not (isinstance(act, dict) and ("start" in act) and ("end" in act)):
        stats["warn_no_action_range"] += 1
        return int(default_start_abs), int(default_end_abs), 0

    try:
        s = int(act["start"])
        e = int(act["end"])
    except Exception:
        stats["warn_bad_action_range"] += 1
        return int(default_start_abs), int(default_end_abs), 0

    if e < s:
        stats["warn_bad_action_range"] += 1
        s, e = e, s
    return int(s), int(e), 1


def fallback_contact_from_tool_points(tool_pts3: np.ndarray, W: int, H: int) -> np.ndarray:
    """Create a synthetic 4-point contact box around tool points when contact is missing."""
    pts = np.asarray(tool_pts3, dtype=np.float32).reshape(3, 2)
    xmin = float(pts[:, 0].min())
    xmax = float(pts[:, 0].max())
    ymin = float(pts[:, 1].min())
    ymax = float(pts[:, 1].max())

    if USE_PIXEL_COORDS:
        pad_x = max(2.0, 0.01 * float(max(1, W)))
        pad_y = max(2.0, 0.01 * float(max(1, H)))
        xmin = max(0.0, xmin - pad_x)
        xmax = min(float(W), xmax + pad_x)
        ymin = max(0.0, ymin - pad_y)
        ymax = min(float(H), ymax + pad_y)
    else:
        pad_x = max(1.0 / float(max(1, W)), 0.01)
        pad_y = max(1.0 / float(max(1, H)), 0.01)
        xmin = max(0.0, xmin - pad_x)
        xmax = min(1.0, xmax + pad_x)
        ymin = max(0.0, ymin - pad_y)
        ymax = min(1.0, ymax + pad_y)

    if xmax <= xmin:
        xmax = min((xmin + pad_x), float(W) if USE_PIXEL_COORDS else 1.0)
    if ymax <= ymin:
        ymax = min((ymin + pad_y), float(H) if USE_PIXEL_COORDS else 1.0)

    box = np.array(
        [
            [xmax, ymin],
            [xmin, ymin],
            [xmin, ymax],
            [xmax, ymax],
        ],
        dtype=np.float32,
    )
    return canonical_order_xy(box).astype(np.float32)


def build_linear_pseudo_targets(
    xy: np.ndarray,
    vis: np.ndarray,
    pseudo_weight: float = PSEUDO_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build low-weight pseudo targets for vis==0 points by linear interpolation over time.
    Inputs:
      - xy:  (T,P,2)
      - vis: (T,P), >0 means visible
    Outputs:
      - pseudo_xy: (T,P,2)
      - pseudo_w : (T,P) in [0, pseudo_weight]
    """
    xy = np.asarray(xy, dtype=np.float32)
    vis = np.asarray(vis, dtype=np.float32)
    T, P = int(xy.shape[0]), int(xy.shape[1])

    pseudo_xy = xy.copy()
    pseudo_w = np.zeros((T, P), dtype=np.float32)

    if T <= 2:
        return pseudo_xy, pseudo_w

    for k in range(P):
        visible = np.where(vis[:, k] > 0.5)[0].tolist()
        if len(visible) < 2:
            continue

        visible_set = set(visible)
        for t in range(T):
            if t in visible_set:
                continue

            left = -1
            right = -1
            for vv in visible:
                if vv < t:
                    left = vv
                elif vv > t:
                    right = vv
                    break

            if left < 0 or right < 0 or right <= left:
                continue

            alpha = float(t - left) / float(right - left)
            pseudo_xy[t, k] = (1.0 - alpha) * xy[left, k] + alpha * xy[right, k]
            pseudo_w[t, k] = float(pseudo_weight)

    return pseudo_xy.astype(np.float32), pseudo_w.astype(np.float32)


def interpolate_vis0_to_full_supervision(
    xy: np.ndarray,
    vis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep coordinates unchanged; only convert visibility from hard {0,1}
    to soft supervision weights in [SOFT_VIS_MIN_WEIGHT, 1].
    - originally visible frames keep weight 1
    - missing frames between two visible frames get linear distance weight
      (e.g. 1-0-1 => 0.5 for the middle frame)
    Returns:
      - xy_same : (T,P,2), unchanged from input
      - vis_soft : (T,P) soft supervision weights
      - interp_mask: (T,P) 1 where point was originally vis==0
    """
    xy = np.asarray(xy, dtype=np.float32)
    vis = np.asarray(vis, dtype=np.float32)
    T, P = int(xy.shape[0]), int(xy.shape[1])

    out = xy.copy()
    interp_mask = (vis <= 0.5).astype(np.float32)
    vis_soft = np.zeros((T, P), dtype=np.float32)
    min_w = float(np.clip(SOFT_VIS_MIN_WEIGHT, 0.0, 1.0))

    for k in range(P):
        valid = np.where(vis[:, k] > 0.5)[0]

        # build soft visibility weights
        if valid.size == 0:
            vis_soft[:, k] = min_w
        else:
            vis_soft[:, k] = min_w
            vis_soft[valid, k] = 1.0
            # assign linear distance-based weights for originally invisible entries
            for t in range(T):
                if vis[t, k] > 0.5:
                    continue
                pos = int(np.searchsorted(valid, t))
                left = int(valid[pos - 1]) if pos - 1 >= 0 else None
                right = int(valid[pos]) if pos < valid.size else None
                if left is not None and right is not None and right > left:
                    span = float(right - left)
                    dist_to_nearest = float(min(t - left, right - t))
                    w = dist_to_nearest / span  # in (0, 0.5] inside interval
                    vis_soft[t, k] = max(min_w, float(w))
                else:
                    # one-sided extrapolation region: keep small but non-zero weight
                    vis_soft[t, k] = min_w

    return out.astype(np.float32), vis_soft.astype(np.float32), interp_mask.astype(np.float32)


def make_prompt(tool: str, action: str, contact_xy_4x2: np.ndarray):
    parts = [f"x {x:.2f} y {y:.2f}" for (x, y) in contact_xy_4x2.tolist()]
    return f"{tool} {action} contact " + " ; ".join(parts)


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def npy_save(path: Path, arr: np.ndarray, dtype=None):
    if dtype is not None:
        arr = np.asarray(arr, dtype=dtype)
    np.save(path, arr)


def cleanup_paths(paths: List[Path]):
    for p in paths:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def prune_output_tree_for_mode(out_root: Path, mode: str) -> Tuple[int, int, List[str], List[str]]:
    """
    Remove non-essential outputs for compact modes.
    Returns:
      removed_dir_count, removed_file_count, removed_dirs, removed_files
    """
    if mode == "full":
        return 0, 0, [], []

    if mode == "minimal_kpt_combo":
        # Strict minimal set used by kpt_combo/dataset.py + train.py
        keep_dirs = {
            "combo_ids",
            "gt_tool_xy",
            "gt_tool_vis",
            "gt_ct_xy",
            "gt_ct_vis",
            "sampled_abs_frames",
            "action_range_abs",
            "action_range_mask",
        }
        if EXPORT_VIDEO16_NPY:
            keep_dirs.add(VIDEO16_DIR_NAME)
        keep_dirs.add(FOLD1_DEBUG_FIXED_DIR_NAME)
        keep_files = {
            "train.txt",
            "val.txt",
            "test.txt",
            "labels.json",
            "meta.json",
            "index.txt",
            "selected_case_folders.json",
            "selected_case_folders.tsv",
            "fold1_aug_detailed_report.json",
            "fold1_aug_detailed_report.tsv",
        }
    elif mode == "tool_kpt_only":
        # Tool-trajectory contract: only keep tool keypoints + combo ids (+ video clips).
        keep_dirs = {
            "combo_ids",
            "gt_tool_xy",
            "gt_tool_vis",
        }
        if EXPORT_VIDEO16_NPY:
            keep_dirs.add(VIDEO16_DIR_NAME)
        keep_dirs.add(FOLD1_DEBUG_FIXED_DIR_NAME)
        keep_files = {
            "train.txt",
            "val.txt",
            "test.txt",
            "labels.json",
            "meta.json",
            "index.txt",
            "selected_case_folders.json",
            "selected_case_folders.tsv",
            "fold1_aug_detailed_report.json",
            "fold1_aug_detailed_report.tsv",
        }
    elif mode == "combo_only_train":
        # Combo-only contract: only video clips + combo labels + split/meta files.
        keep_dirs = {"combo_ids"}
        if EXPORT_VIDEO16_NPY:
            keep_dirs.add(VIDEO16_DIR_NAME)
        keep_dirs.add(FOLD1_DEBUG_FIXED_DIR_NAME)
        keep_files = {
            "train.txt",
            "val.txt",
            "test.txt",
            "labels.json",
            "meta.json",
            "index.txt",
            "selected_case_folders.json",
            "selected_case_folders.tsv",
            "fold1_aug_detailed_report.json",
            "fold1_aug_detailed_report.tsv",
        }
    else:
        raise ValueError(f"Unknown output mode for prune: {mode}")

    removed_dirs: List[str] = []
    removed_files: List[str] = []

    for p in sorted(out_root.iterdir(), key=lambda x: x.name):
        name = p.name
        if p.is_dir():
            if name not in keep_dirs:
                shutil.rmtree(p, ignore_errors=True)
                removed_dirs.append(name)
        else:
            if name not in keep_files:
                try:
                    p.unlink()
                    removed_files.append(name)
                except Exception:
                    pass

    return len(removed_dirs), len(removed_files), removed_dirs, removed_files


def build_comboid_split_report(
    sid2item: Dict[str, "BaseItem"],
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> Dict:
    split_map = {
        "train": list(train_ids),
        "val": list(val_ids),
        "test": list(test_ids),
    }
    id2combo = {int(v): str(k) for k, v in COMBO2ID.items()}
    rows: Dict[int, Dict[str, int]] = {}
    for cid in sorted(id2combo.keys()):
        rows[cid] = {"train": 0, "val": 0, "test": 0, "total": 0}

    for split, ids in split_map.items():
        for sid in ids:
            it = sid2item.get(sid)
            if it is None:
                continue
            cid = int(it.combo_id)
            if cid not in rows:
                rows[cid] = {"train": 0, "val": 0, "test": 0, "total": 0}
            rows[cid][split] += 1
            rows[cid]["total"] += 1

    return {
        "combo2id_6class": {k: int(v) for k, v in COMBO2ID.items()},
        "id2combo_6class": {str(k): v for k, v in sorted(id2combo.items(), key=lambda kv: kv[0])},
        "split_totals": {
            "train": int(len(train_ids)),
            "val": int(len(val_ids)),
            "test": int(len(test_ids)),
            "all": int(len(train_ids) + len(val_ids) + len(test_ids)),
        },
        "by_combo_id": {
            str(cid): {
                "combo": id2combo.get(cid, "unknown"),
                "train": int(rows[cid]["train"]),
                "val": int(rows[cid]["val"]),
                "test": int(rows[cid]["test"]),
                "total": int(rows[cid]["total"]),
            }
            for cid in sorted(rows.keys())
        },
    }


def write_comboid_split_report(out_root: Path, report: Dict) -> Tuple[Path, Path]:
    json_path = out_root / "comboid_split_report.json"
    tsv_path = out_root / "comboid_split_report.tsv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("combo_id\tcombo\ttrain\tval\ttest\ttotal\n")
        by_combo_id = report.get("by_combo_id", {})
        for cid in sorted([int(x) for x in by_combo_id.keys()]):
            row = by_combo_id[str(cid)]
            f.write(
                f"{cid}\t{row['combo']}\t{row['train']}\t{row['val']}\t{row['test']}\t{row['total']}\n"
            )
    return json_path, tsv_path


def build_split_case_combo_report(
    sid2item: Dict[str, "BaseItem"],
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> Dict:
    split_map = {
        "train": list(train_ids),
        "val": list(val_ids),
        "test": list(test_ids),
    }
    per_split_case_counts: Dict[str, Dict[str, int]] = {k: {} for k in split_map.keys()}
    per_split_combo_counts: Dict[str, Dict[str, int]] = {k: {} for k in split_map.keys()}
    per_split_case_combo_counts: Dict[str, Dict[str, Dict[str, int]]] = {k: {} for k in split_map.keys()}
    sid_details: List[Dict[str, str]] = []

    for split, ids in split_map.items():
        for sid in ids:
            it = sid2item.get(sid)
            if it is None:
                continue
            case = str(it.case)
            combo = f"{it.tool}|{it.action}"
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

    rows: List[Dict[str, object]] = []
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


def write_split_case_combo_report(out_root: Path, report: Dict) -> Tuple[Path, Path, Path]:
    json_path = out_root / "split_case_combo_report.json"
    tsv_path = out_root / "split_case_combo_report.tsv"
    sid_tsv_path = out_root / "split_sid_details.tsv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("split\tcase\tcombo\tcount\n")
        for row in report.get("rows", []):
            f.write(f"{row['split']}\t{row['case']}\t{row['combo']}\t{int(row['count'])}\n")

    with open(sid_tsv_path, "w", encoding="utf-8") as f:
        f.write("sid\tsplit\tcase\tcombo\n")
        for row in report.get("sid_details", []):
            f.write(f"{row['sid']}\t{row['split']}\t{row['case']}\t{row['combo']}\n")
    return json_path, tsv_path, sid_tsv_path


def _inc_nested_counter(dst: Dict[str, Dict[str, int]], k1: str, k2: str, delta: int = 1) -> None:
    bucket = dst.setdefault(str(k1), {})
    bucket[str(k2)] = int(bucket.get(str(k2), 0) + int(delta))


def _normalize_strength_mix(row: Dict[str, float]) -> Dict[str, float]:
    out = {
        "light": max(0.0, float(row.get("light", 0.0))),
        "medium": max(0.0, float(row.get("medium", 0.0))),
        "heavy": max(0.0, float(row.get("heavy", 0.0))),
        "extreme": max(0.0, float(row.get("extreme", 0.0))),
    }
    s = float(out["light"] + out["medium"] + out["heavy"] + out["extreme"])
    if s <= 0.0:
        return {"light": 0.2, "medium": 0.5, "heavy": 0.27, "extreme": 0.03}
    out["light"] /= s
    out["medium"] /= s
    out["heavy"] /= s
    out["extreme"] /= s
    return out


def _plan_strength_buckets_for_combo(combo: str, n_samples: int, seed: int) -> List[str]:
    if n_samples <= 0:
        return []
    row = FOLD1_STRENGTH_MIX_BY_COMBO.get(combo, FOLD1_STRENGTH_MIX_BY_COMBO.get("default", {}))
    mix = _normalize_strength_mix(row)
    n_light = int(round(float(n_samples) * mix["light"]))
    n_medium = int(round(float(n_samples) * mix["medium"]))
    n_heavy = int(round(float(n_samples) * mix["heavy"]))
    n_extreme = int(n_samples - n_light - n_medium - n_heavy)
    if n_heavy < 0:
        n_heavy = 0
    if n_extreme < 0:
        n_extreme = 0
    buckets = (
        (["light"] * n_light)
        + (["medium"] * n_medium)
        + (["heavy"] * n_heavy)
        + (["extreme"] * n_extreme)
    )
    while len(buckets) < int(n_samples):
        buckets.append("medium")
    if len(buckets) > int(n_samples):
        buckets = buckets[: int(n_samples)]
    rng_local = random.Random(int(seed) + 1931 + sum(ord(ch) for ch in str(combo)))
    rng_local.shuffle(buckets)
    return buckets


def _strength_profile(bucket: str, combo: Optional[str] = None) -> Dict[str, float]:
    b = str(bucket).lower().strip()
    if b not in FOLD1_STRENGTH_PRESETS:
        b = "medium"
    out = dict(FOLD1_STRENGTH_PRESETS[b])
    override_roots: List[str] = ["default"]
    if combo is not None:
        override_roots.append(str(combo))
    for key in override_roots:
        row = FOLD1_STRENGTH_PRESET_OVERRIDES_BY_COMBO.get(str(key), {}).get(b, None)
        if isinstance(row, dict):
            out.update({str(k): float(v) for k, v in row.items()})
    return out


def _hard_score_of_item(it: "BaseItem") -> float:
    # Lower visibility generally indicates harder trajectory supervision.
    vis = float(it.vis_score) if (it.vis_score is not None) else 0.5
    return float(1.0 - vis)


def _select_hard_sid_set_for_combo(
    combo: str,
    sids: List[str],
    sid2item: Dict[str, "BaseItem"],
    top_frac: float,
    min_keep: int,
) -> Set[str]:
    if len(sids) == 0:
        return set()
    arr: List[Tuple[float, str]] = []
    for sid in sids:
        it = sid2item.get(sid, None)
        if it is None:
            arr.append((0.5, sid))
        else:
            arr.append((_hard_score_of_item(it), sid))
    arr = sorted(arr, key=lambda x: (-float(x[0]), str(x[1])))
    n = int(len(arr))
    k = int(max(int(min_keep), int(round(float(n) * float(top_frac)))))
    k = int(min(max(1, k), n))
    picked = [sid for _, sid in arr[:k]]
    return set(picked)


def _enforce_extreme_on_hard_tasks(
    combo: str,
    task_base_sids: List[str],
    strength_schedule: List[str],
    hard_sid_set: Set[str],
    rng_py: random.Random,
) -> Tuple[List[str], Dict[str, int]]:
    schedule = list(strength_schedule)
    moved_nonhard = 0
    promoted_hard = 0
    extreme_before = int(sum(1 for b in schedule if str(b) == "extreme"))
    nonhard_extreme_idx = [
        i for i, b in enumerate(schedule)
        if (str(b) == "extreme") and (task_base_sids[i] not in hard_sid_set)
    ]
    for i in nonhard_extreme_idx:
        schedule[i] = "heavy"
    moved_nonhard = int(len(nonhard_extreme_idx))
    need = int(moved_nonhard)
    if need > 0 and len(hard_sid_set) > 0:
        cand = [
            i for i, b in enumerate(schedule)
            if (task_base_sids[i] in hard_sid_set) and (str(b) != "extreme")
        ]
        rng_py.shuffle(cand)
        for i in cand[:need]:
            schedule[i] = "extreme"
        promoted_hard = int(min(len(cand), need))
    extreme_after = int(sum(1 for b in schedule if str(b) == "extreme"))
    stats = {
        "extreme_before": int(extreme_before),
        "extreme_after": int(extreme_after),
        "moved_nonhard_to_heavy": int(moved_nonhard),
        "promoted_hard_to_extreme": int(promoted_hard),
        "hard_pool_size": int(len(hard_sid_set)),
        "combo_is_priority": int(1 if combo in FOLD1_EXTREME_PRIORITY_COMBOS else 0),
    }
    return schedule, stats


def _enforce_high_difficulty_only_schedule(
    combo: str,
    task_base_sids: List[str],
    strength_schedule: List[str],
    hard_sid_set: Set[str],
    rng_py: random.Random,
) -> Tuple[List[str], Dict[str, int]]:
    schedule = list(strength_schedule)
    if len(schedule) == 0:
        return schedule, {
            "before_light": 0,
            "before_medium": 0,
            "before_heavy": 0,
            "before_extreme": 0,
            "promoted_to_heavy": 0,
            "promoted_to_extreme": 0,
            "target_extreme": 0,
            "after_light": 0,
            "after_medium": 0,
            "after_heavy": 0,
            "after_extreme": 0,
        }

    before_counts = {
        "light": int(sum(1 for x in schedule if str(x) == "light")),
        "medium": int(sum(1 for x in schedule if str(x) == "medium")),
        "heavy": int(sum(1 for x in schedule if str(x) == "heavy")),
        "extreme": int(sum(1 for x in schedule if str(x) == "extreme")),
    }

    # Step 1: hard-lock all non-high buckets to heavy.
    promoted_to_heavy = 0
    for i, b in enumerate(schedule):
        sb = str(b)
        if sb in ("light", "medium"):
            schedule[i] = "heavy"
            promoted_to_heavy += 1

    # Step 2: enforce a minimum extreme ratio on hard pool.
    n = int(len(schedule))
    min_extreme_ratio = (
        float(FOLD1_FORCE_HIGH_DIFFICULTY_MIN_EXTREME_PRIORITY)
        if combo in FOLD1_EXTREME_PRIORITY_COMBOS
        else float(FOLD1_FORCE_HIGH_DIFFICULTY_MIN_EXTREME_DEFAULT)
    )
    target_extreme = int(max(1, round(min_extreme_ratio * float(n))))
    current_extreme = int(sum(1 for x in schedule if str(x) == "extreme"))
    need = int(max(0, target_extreme - current_extreme))
    promoted_to_extreme = 0
    if need > 0 and len(hard_sid_set) > 0:
        cand = [
            i for i, b in enumerate(schedule)
            if (task_base_sids[i] in hard_sid_set) and (str(b) == "heavy")
        ]
        rng_py.shuffle(cand)
        for i in cand[:need]:
            schedule[i] = "extreme"
        promoted_to_extreme = int(min(len(cand), need))

    after_counts = {
        "light": int(sum(1 for x in schedule if str(x) == "light")),
        "medium": int(sum(1 for x in schedule if str(x) == "medium")),
        "heavy": int(sum(1 for x in schedule if str(x) == "heavy")),
        "extreme": int(sum(1 for x in schedule if str(x) == "extreme")),
    }
    return schedule, {
        "before_light": int(before_counts["light"]),
        "before_medium": int(before_counts["medium"]),
        "before_heavy": int(before_counts["heavy"]),
        "before_extreme": int(before_counts["extreme"]),
        "promoted_to_heavy": int(promoted_to_heavy),
        "promoted_to_extreme": int(promoted_to_extreme),
        "target_extreme": int(target_extreme),
        "after_light": int(after_counts["light"]),
        "after_medium": int(after_counts["medium"]),
        "after_heavy": int(after_counts["heavy"]),
        "after_extreme": int(after_counts["extreme"]),
    }


def _build_donor_hint_sequence(
    base_sid: str,
    k: int,
    donor_pool_all: List[str],
    seed: int,
) -> List[Optional[str]]:
    kk = int(max(0, k))
    if kk <= 0:
        return []
    cands = [sid for sid in donor_pool_all if sid != base_sid]
    if len(cands) == 0:
        return [None for _ in range(kk)]
    rng_local = random.Random(int(seed) + 881 + sum(ord(ch) for ch in str(base_sid)))
    cands = sorted(cands)
    rng_local.shuffle(cands)
    if kk < int(FOLD1_DONOR_DIVERSITY_K_THRESHOLD):
        return [cands[i % len(cands)] for i in range(kk)]
    min_unique = int(min(len(cands), kk, int(FOLD1_DONOR_DIVERSITY_MIN_UNIQUE)))
    unique_part = cands[:min_unique]
    seq: List[Optional[str]] = []
    for i in range(kk):
        seq.append(unique_part[i % max(1, len(unique_part))])
    for i in range(kk):
        if rng_local.random() < 0.35:
            seq[i] = cands[rng_local.randrange(len(cands))]
    return seq


def _sample_sids_balanced_by_case(
    sids: List[str],
    sid2item: Dict[str, "BaseItem"],
    cap_n: int,
    combo: str,
    seed: int,
) -> List[str]:
    """
    Deterministic case-balanced sampling:
    round-robin one sid per case until cap_n is reached.
    """
    if cap_n <= 0 or len(sids) == 0:
        return []
    if len(sids) <= cap_n:
        return sorted(sids)

    case2sids: Dict[str, List[str]] = {}
    for sid in sorted(sids):
        it = sid2item.get(sid, None)
        case_name = str(it.case) if it is not None else "unknown_case"
        case2sids.setdefault(case_name, []).append(sid)

    # Deterministic shuffle inside each case to avoid always taking earliest items.
    for case_name, arr in case2sids.items():
        rng_local = random.Random(int(seed) + 9107 + sum(ord(ch) for ch in combo + "|" + case_name))
        rng_local.shuffle(arr)

    case_names = sorted(case2sids.keys())
    ptr = {k: 0 for k in case_names}
    kept: List[str] = []

    while len(kept) < int(cap_n):
        progressed = False
        for case_name in case_names:
            arr = case2sids[case_name]
            i = ptr[case_name]
            if i >= len(arr):
                continue
            kept.append(arr[i])
            ptr[case_name] = i + 1
            progressed = True
            if len(kept) >= int(cap_n):
                break
        if not progressed:
            break

    return sorted(kept[: int(cap_n)])


def write_fold1_aug_detailed_report(out_root: Path, report: Dict[str, object]) -> Tuple[Path, Path]:
    json_path = out_root / "fold1_aug_detailed_report.json"
    tsv_path = out_root / "fold1_aug_detailed_report.tsv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write(
            "combo\tbase_before_cap\tbase_after_cap\ttarget\tplanned_extra\tgenerated_extra\t"
            "gen_light\tgen_medium\tgen_heavy\tgen_extreme\tfinal_train_est\n"
        )
        combos = sorted(set(report.get("base_by_combo_before_cap", {}).keys()) | set(report.get("base_by_combo_after_cap", {}).keys()))
        targets = dict(report.get("target_train_per_combo", {}))
        planned = dict(report.get("planned_extra_by_combo", {}))
        generated = dict(report.get("generated_extra_by_combo", {}))
        before = dict(report.get("base_by_combo_before_cap", {}))
        after = dict(report.get("base_by_combo_after_cap", {}))
        by_strength = dict(report.get("generated_by_combo_and_strength", {}))
        for combo in combos:
            b0 = int(before.get(combo, 0))
            b1 = int(after.get(combo, 0))
            t = targets.get(combo, "")
            p = int(planned.get(combo, 0))
            g = int(generated.get(combo, 0))
            strength_row = by_strength.get(combo, {}) if isinstance(by_strength.get(combo, {}), dict) else {}
            g_light = int(strength_row.get("light", 0))
            g_medium = int(strength_row.get("medium", 0))
            g_heavy = int(strength_row.get("heavy", 0))
            g_extreme = int(strength_row.get("extreme", 0))
            final_est = int(b1 + g)
            f.write(
                f"{combo}\t{b0}\t{b1}\t{t}\t{p}\t{g}\t"
                f"{g_light}\t{g_medium}\t{g_heavy}\t{g_extreme}\t{final_est}\n"
            )

    return json_path, tsv_path


def write_kpt_combo_contract_audit(out_root: Path, export_video16: bool, mode: str) -> Path:
    if mode == "combo_only_train":
        required_dirs = ["combo_ids"]
        if export_video16:
            required_dirs.append(VIDEO16_DIR_NAME)
        target_name = "combo_only training dataset contract"
    elif mode == "tool_kpt_only":
        required_dirs = [
            "combo_ids",
            "gt_tool_xy",
            "gt_tool_vis",
        ]
        if export_video16:
            required_dirs.append(VIDEO16_DIR_NAME)
        target_name = "tool trajectory-only training dataset contract"
    else:
        required_dirs = [
            "combo_ids",
            "gt_tool_xy",
            "gt_tool_vis",
            "gt_ct_xy",
            "gt_ct_vis",
            "sampled_abs_frames",
            "action_range_abs",
            "action_range_mask",
        ]
        if export_video16:
            required_dirs.append(VIDEO16_DIR_NAME)
        target_name = "kpt_combo training dataset contract"

    required_files = [
        "train.txt",
        "val.txt",
        "test.txt",
        "labels.json",
        "meta.json",
        "index.txt",
    ]

    dir_status = {d: (out_root / d).is_dir() for d in required_dirs}
    file_status = {f: (out_root / f).is_file() for f in required_files}
    all_ok = all(dir_status.values()) and all(file_status.values())

    audit = {
        "target": target_name,
        "output_mode": str(mode),
        "all_required_present": bool(all_ok),
        "required_dirs": dir_status,
        "required_files": file_status,
    }
    p = out_root / "kpt_combo_contract_audit.json"
    p.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return p

def read_video_frame_at_local_index(video_path: Path, local_frame_idx: int) -> Optional[np.ndarray]:
    """
    Read frame at local index from a clip mp4.
    local_frame_idx is 0-based (frame0).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None

    # Try seeking
    if local_frame_idx >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(local_frame_idx))
    ok, frame = cap.read()
    if ok and frame is not None:
        cap.release()
        return frame

    # Fallback: rewind and sequential read
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
    cur = 0
    frame = None
    while cur <= local_frame_idx:
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None
        cur += 1
    cap.release()
    return frame


def read_video_frames_at_local_indices(video_path: Path, local_indices: List[int]) -> List[Optional[np.ndarray]]:
    """
    Read multiple local frame indices from one clip using a single VideoCapture handle.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return [None for _ in local_indices]

    out: List[Optional[np.ndarray]] = []
    for idx in local_indices:
        idx = int(idx)
        if idx < 0:
            out.append(None)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            out.append(None)
        else:
            out.append(frame)
    cap.release()
    return out


def build_video16_npy_from_sampled_frames(
    video_path: Path,
    sampled_abs: np.ndarray,
    video_base_abs: int,
    bbox_xyxy: Tuple[int, int, int, int],
    start_frame_full: np.ndarray,
    stats: dict,
) -> np.ndarray:
    """
    Build sampled 16-frame clip as uint8 npy tensor in cropped space.
    Output shape: (T, Hc, Wc, 3)
    """
    sampled_abs = np.asarray(sampled_abs, dtype=np.int64).reshape(-1)
    frames_crop: List[np.ndarray] = []
    prev_full: Optional[np.ndarray] = None
    local_indices = [int(v) - int(video_base_abs) for v in sampled_abs.tolist()]
    full_list = read_video_frames_at_local_indices(video_path, local_indices)

    for i, full in enumerate(full_list):
        local_idx = int(local_indices[i])
        if i == 0 and local_idx >= 0 and full is None:
            full = start_frame_full.copy()
        elif full is None:
            stats["warn_video16_frame_fallback"] += 1
            if prev_full is not None:
                full = prev_full.copy()
            else:
                full = start_frame_full.copy()
        prev_full = full

        crop = crop_frame(full, bbox_xyxy)
        frames_crop.append(crop.astype(np.uint8))

    if len(frames_crop) == 0:
        # should not happen in normal flow
        h0, w0 = start_frame_full.shape[:2]
        x0, y0, x1, y1 = bbox_xyxy
        h0 = max(1, int(y1 - y0))
        w0 = max(1, int(x1 - x0))
        return np.zeros((0, h0, w0, 3), dtype=np.uint8)

    # safety: enforce same shape across frames
    h_ref, w_ref = frames_crop[0].shape[:2]
    for i in range(len(frames_crop)):
        if frames_crop[i].shape[:2] != (h_ref, w_ref):
            stats["warn_video16_frame_resized_mismatch"] += 1
            frames_crop[i] = cv2.resize(
                frames_crop[i],
                (w_ref, h_ref),
                interpolation=cv2.INTER_LINEAR,
            )

    return np.stack(frames_crop, axis=0).astype(np.uint8)

# 2026/02/02
def detect_black_bars_bbox(
    frame_bgr: np.ndarray,
    thr: int = BAR_GRAY_THR,
    min_frac: float = BAR_MIN_NONBLACK_FRAC,
    min_border_px: int = BAR_MIN_BORDER_PX,
    crop_top_bottom: bool = CROP_TOP_BOTTOM,
    min_keep_frac: float = BAR_MIN_KEEP_FRAC,
) -> Tuple[int, int, int, int]:
    """
    Return bbox (x0, y0, x1, y1) in pixel coords, where x1/y1 are EXCLUSIVE.
    Designed for pillarbox black bars.
    """
    if frame_bgr is None:
        return (0, 0, 0, 0)
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return (0, 0, w, h)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > thr)

    # per-column non-black fraction
    col_frac = mask.mean(axis=0)  # (w,)
    cols = np.where(col_frac > min_frac)[0]
    if cols.size == 0:
        return (0, 0, w, h)

    x0 = int(cols[0])
    x1 = int(cols[-1] + 1)

    # ignore tiny borders
    if x0 < min_border_px:
        x0 = 0
    if (w - x1) < min_border_px:
        x1 = w

    y0, y1 = 0, h
    if crop_top_bottom:
        row_frac = mask.mean(axis=1)  # (h,)
        rows = np.where(row_frac > min_frac)[0]
        if rows.size > 0:
            y0 = int(rows[0])
            y1 = int(rows[-1] + 1)
            if y0 < min_border_px:
                y0 = 0
            if (h - y1) < min_border_px:
                y1 = h

    # sanity: prevent over-cropping
    if (x1 - x0) < int(min_keep_frac * w):
        x0, x1 = 0, w
    if (y1 - y0) < int(min_keep_frac * h):
        y0, y1 = 0, h

    return (x0, y0, x1, y1)

# 2026/02/03
def _smooth1d(x: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k <= 1:
        return x.astype(np.float32)
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k,), dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def _looks_like_bar_region(gray_region: np.ndarray, content_std: float) -> bool:
    """
    判断某个区域(上方/下方)像不像“UI bar / 时间条”：
      - 要么偏黑，要么偏白
      - 且整体纹理 std 显著低于内容区
    """
    if gray_region is None or gray_region.size == 0:
        return False
    h = gray_region.shape[0]
    if h < UI_MIN_BAR_PX:
        return False

    region_std = float(gray_region.std())
    dark_frac = float((gray_region < UI_DARK_THR).mean())
    bright_frac = float((gray_region > UI_BRIGHT_THR).mean())

    # 偏黑或偏白 + 纹理比内容区更“平”
    return ((dark_frac > 0.35) or (bright_frac > 0.35)) and (region_std < content_std * float(UI_BAR_STD_RATIO))


def detect_ui_bars_tb_bbox(gray: np.ndarray) -> Tuple[int, int]:
    """
    输入：左右黑边裁掉后的灰度图 gray (H,W)
    输出：y0, y1（EXCLUSIVE），用于裁掉上/下 UI 条
    方法：用纵向梯度能量找上/下边界 + bar 外观验证（黑/白 + 低纹理）
    """
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return 0, h

    # 纵向梯度能量：每一条“行边界”的平均 |dy|
    grad = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(axis=1)  # (h-1,)
    grad = _smooth1d(grad, UI_SMOOTH_K)

    # 内容区 std 参考：中间 50% 高度
    mid0 = int(0.25 * h)
    mid1 = int(0.75 * h)
    content_std = float(gray[mid0:mid1].std()) + 1e-6

    # ---------- top ----------
    top_max = max(1, int(UI_TOP_SCAN_FRAC * h))
    top_seg = grad[:top_max]
    y0 = 0
    if top_seg.size > 0:
        peak = float(top_seg.max())
        if peak > 1.0:
            cand = np.where(top_seg >= peak * float(UI_PEAK_RATIO))[0]
            if cand.size > 0:
                y0 = int(cand.max() + 1)  # 取“最靠下”的强边界
                y0 = min(h - 1, y0 + int(UI_PAD_Y))

                # bar 外观验证：上方区域像不像 UI bar
                if not _looks_like_bar_region(gray[:y0, :], content_std):
                    y0 = 0

    # ---------- bottom ----------
    bot_min = int((1.0 - float(UI_BOT_SCAN_FRAC)) * h)
    bot_min = max(0, min(h - 2, bot_min))
    bot_seg = grad[bot_min:]
    y1 = h
    if bot_seg.size > 0:
        peak = float(bot_seg.max())
        if peak > 1.0:
            cand = np.where(bot_seg >= peak * float(UI_PEAK_RATIO))[0]
            if cand.size > 0:
                y1 = int(bot_min + cand.min() + 1)  # 取“最靠上”的强边界
                y1 = max(1, y1 - int(UI_PAD_Y))

                # bar 外观验证：下方区域像不像 UI bar
                if not _looks_like_bar_region(gray[y1:, :], content_std):
                    y1 = h

    # ---------- sanity ----------
    if y0 < UI_MIN_BAR_PX:
        y0 = 0
    if (h - y1) < UI_MIN_BAR_PX:
        y1 = h

    if (y1 - y0) < int(UI_MIN_KEEP_FRAC_H * h):
        # 防误裁：裁完后剩余太少 -> 取消上下裁剪
        y0, y1 = 0, h

    return y0, y1


def detect_content_bbox(frame_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """
    统一输出内容区域 bbox (x0,y0,x1,y1)：
      1) 先裁左右黑边（detect_black_bars_bbox）
      2) 再在左右裁后的区域里检测上下 UI bar（detect_ui_bars_tb_bbox）
    """
    if frame_bgr is None:
        return (0, 0, 0, 0)

    Hf, Wf = frame_bgr.shape[:2]

    # 1) 左右黑边
    if REMOVE_BLACK_BARS:
        x0, _, x1, _ = detect_black_bars_bbox(frame_bgr, crop_top_bottom=False)
    else:
        x0, x1 = 0, Wf

    x0 = max(0, min(Wf, int(x0)))
    x1 = max(0, min(Wf, int(x1)))
    if x1 <= x0:
        x0, x1 = 0, Wf

    # 2) 上下 UI 条（可选）
    y0, y1 = 0, Hf
    if REMOVE_UI_BARS:
        lr = frame_bgr[:, x0:x1]
        gray = cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY)
        y0_ui, y1_ui = detect_ui_bars_tb_bbox(gray)
        y0 = int(y0_ui)
        y1 = int(y1_ui)

    # clamp + 防呆
    y0 = max(0, min(Hf, y0))
    y1 = max(0, min(Hf, y1))
    if y1 <= y0:
        y0, y1 = 0, Hf

    # 保证整体保留比例（复用你原来的 BAR_MIN_KEEP_FRAC 也可以）
    if (x1 - x0) < int(BAR_MIN_KEEP_FRAC * Wf):
        x0, x1 = 0, Wf
    if (y1 - y0) < int(UI_MIN_KEEP_FRAC_H * Hf):
        y0, y1 = 0, Hf

    return (x0, y0, x1, y1)
# =========================
# Point-aware crop helpers (NEW)
# =========================
def points_to_frame_px(
    xy: np.ndarray,
    ann_W: int, ann_H: int,
    frame_W: int, frame_H: int,
) -> np.ndarray:
    """
    Convert points in annotation coord system to FRAME pixel coord system.

    Input xy shape: (...,2)
      - if USE_PIXEL_COORDS==True : xy is in annotation pixel coords
      - else                      : xy is normalized in annotation coords ([0,1] wrt ann_W/ann_H)

    Output: (...,2) in FRAME pixel coords
    """
    arr = np.asarray(xy, dtype=np.float32)
    out = arr.copy()

    if USE_PIXEL_COORDS:
        sx = float(frame_W) / float(max(1, ann_W))
        sy = float(frame_H) / float(max(1, ann_H))
        out[..., 0] = out[..., 0] * sx
        out[..., 1] = out[..., 1] * sy
    else:
        out[..., 0] = out[..., 0] * float(frame_W)
        out[..., 1] = out[..., 1] * float(frame_H)

    return out.astype(np.float32)


def expand_bbox_to_include_points(
    bbox_xyxy: Tuple[int, int, int, int],
    pts_frame_px: np.ndarray,
    frame_W: int, frame_H: int,
    pad_px: int = 2,
) -> Tuple[int, int, int, int]:
    """
    Expand bbox (x0,y0,x1,y1 EXCLUSIVE) outward so that all pts (in FRAME px) are included.
    Only expands (never shrinks).
    """
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]

    pts = np.asarray(pts_frame_px, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return (x0, y0, x1, y1)

    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if pts.shape[0] == 0:
        return (x0, y0, x1, y1)

    # Clamp points to frame bounds for bbox expansion (points outside frame can't be covered anyway)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(frame_W))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(frame_H))

    xmin = float(pts[:, 0].min())
    xmax = float(pts[:, 0].max())
    ymin = float(pts[:, 1].min())
    ymax = float(pts[:, 1].max())

    pad = int(max(0, int(round(pad_px))))

    x0_new = min(x0, int(np.floor(xmin - pad)))
    y0_new = min(y0, int(np.floor(ymin - pad)))
    x1_new = max(x1, int(np.ceil(xmax + pad)))
    y1_new = max(y1, int(np.ceil(ymax + pad)))

    # Clamp bbox to frame range; x1/y1 are EXCLUSIVE so they can equal frame_W/frame_H
    if frame_W <= 1:
        x0_new, x1_new = 0, int(frame_W)
    else:
        x0_new = max(0, min(int(frame_W - 1), x0_new))
        x1_new = max(1, min(int(frame_W), x1_new))

    if frame_H <= 1:
        y0_new, y1_new = 0, int(frame_H)
    else:
        y0_new = max(0, min(int(frame_H - 1), y0_new))
        y1_new = max(1, min(int(frame_H), y1_new))

    # final safety
    if x1_new <= x0_new:
        x0_new, x1_new = 0, int(frame_W)
    if y1_new <= y0_new:
        y0_new, y1_new = 0, int(frame_H)

    return (int(x0_new), int(y0_new), int(x1_new), int(y1_new))


def point_aware_expand_bbox(
    bbox_xyxy: Tuple[int, int, int, int],
    motion_abs: np.ndarray,
    contact_xy4: np.ndarray,
    ann_W: int, ann_H: int,
    frame_W: int, frame_H: int,
) -> Tuple[int, int, int, int]:
    """
    Given a base bbox from content detection, expand it to include ALL points:
      - all tool points over time (motion_abs: T,3,2)
      - contact points (contact_xy4: 4,2)
    """
    if not POINT_AWARE_CROP:
        return bbox_xyxy

    pad_px = int(
        max(
            float(POINT_CROP_PAD_PX),
            float(POINT_CROP_PAD_FRAC) * float(min(frame_W, frame_H)),
        )
    )

    m_px = points_to_frame_px(motion_abs, ann_W, ann_H, frame_W, frame_H).reshape(-1, 2)
    c_px = points_to_frame_px(contact_xy4, ann_W, ann_H, frame_W, frame_H).reshape(-1, 2)
    pts_all = np.concatenate([m_px, c_px], axis=0)

    return expand_bbox_to_include_points(
        bbox_xyxy=bbox_xyxy,
        pts_frame_px=pts_all,
        frame_W=frame_W,
        frame_H=frame_H,
        pad_px=pad_px,
    )

# 2026/02/03

def crop_frame(frame_bgr: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox_xyxy
    return frame_bgr[y0:y1, x0:x1].copy()


def transform_points_to_crop(
    xy: np.ndarray,
    ann_W: int, ann_H: int,
    frame_W: int, frame_H: int,
    bbox_xyxy: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Transform points from annotation coordinate system (what your JSON provides)
    into the cropped-image coordinate system.

    Input xy shape: (..., 2)
      - if USE_PIXEL_COORDS==True : xy is in annotation pixel coords
      - else                      : xy is normalized in annotation coords ([0,1] wrt ann_W/ann_H)

    Output:
      - if USE_PIXEL_COORDS==True : pixel coords in cropped image
      - else                      : normalized coords in cropped image ([0,1] wrt crop_W/crop_H)
    """
    x0, y0, x1, y1 = bbox_xyxy
    crop_W = max(1, int(x1 - x0))
    crop_H = max(1, int(y1 - y0))

    arr = np.asarray(xy, dtype=np.float32)
    out = arr.copy()

    if USE_PIXEL_COORDS:
        # annotation px -> frame px -> crop px
        sx = float(frame_W) / float(ann_W)
        sy = float(frame_H) / float(ann_H)
        out[..., 0] = out[..., 0] * sx - float(x0)
        out[..., 1] = out[..., 1] * sy - float(y0)
        return out.astype(np.float32)

    # annotation-normalized -> frame px -> crop px -> crop-normalized
    out[..., 0] = out[..., 0] * float(frame_W) - float(x0)
    out[..., 1] = out[..., 1] * float(frame_H) - float(y0)
    out[..., 0] /= float(crop_W)
    out[..., 1] /= float(crop_H)
    return out.astype(np.float32)
# 2026/02/02

def draw_overlay(image_bgr: np.ndarray,
                 tool_pts3_norm: np.ndarray,
                 pt_role: np.ndarray,
                 contact_xy4_norm: np.ndarray,
                 W: int, H: int) -> np.ndarray:
    """
    Draw contacts (red), anchors (blue), tips (green) on a copy of image.
    Inputs are in the same coordinate space as normalize_xy():
      - pixel if USE_PIXEL_COORDS
      - normalized if not
    """
    img = image_bgr.copy()

    def to_px(x, y):
        if USE_PIXEL_COORDS:
            return int(round(float(x))), int(round(float(y)))
        return int(round(float(x) * W)), int(round(float(y) * H))

    # contacts in red
    for i, (x, y) in enumerate(contact_xy4_norm.astype(np.float32)):
        px, py = to_px(x, y)
        cv2.circle(img, (px, py), DEBUG_DRAW_RADIUS, (0, 0, 255), -1)
        cv2.putText(img, f"c{i}", (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # tool points
    for i, (x, y) in enumerate(tool_pts3_norm.astype(np.float32)):
        px, py = to_px(x, y)
        role = int(pt_role[i])
        if role == 0:
            color = (255, 0, 0)  # anchor blue
            tag = f"a{i}"
        else:
            color = (0, 255, 0)  # tip green
            tag = f"t{i}"
        cv2.circle(img, (px, py), DEBUG_DRAW_RADIUS, color, -1)
        cv2.putText(img, tag, (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return img


def _save_debug_aug_pair_once(
    debug_dir: Path,
    key: str,
    before: np.ndarray,
    after: np.ndarray,
    saved_keys: Optional[Set[str]] = None,
) -> bool:
    """
    Save one before/after pair for this augmentation key.
    Returns True if written in this call, False if skipped.
    """
    safe_key = str(key).strip().replace("|", "_").replace("/", "_").replace(" ", "_")
    if not safe_key:
        safe_key = "unknown"
    if DEBUG_AUG_ONE_EXAMPLE_PER_TYPE and (saved_keys is not None):
        if safe_key in saved_keys:
            return False
        saved_keys.add(safe_key)
    cv2.imwrite(str(debug_dir / f"example_{safe_key}_before.jpg"), before)
    cv2.imwrite(str(debug_dir / f"example_{safe_key}_after.jpg"), after)
    return True


def _save_debug_aug_pair_indexed(
    debug_dir: Path,
    pair_idx: int,
    aug_sid: str,
    base_sid: str,
    combo: str,
    before: np.ndarray,
    after: np.ndarray,
    params: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Save deterministic indexed before/after pair for fold1 post-augmentation.
    """
    idx = int(pair_idx)
    prefix = f"{idx:02d}_{str(aug_sid)}"
    before_name = f"{prefix}_before.jpg"
    after_name = f"{prefix}_after.jpg"
    cv2.imwrite(str(debug_dir / before_name), before)
    cv2.imwrite(str(debug_dir / after_name), after)
    return {
        "pair_index": idx,
        "aug_sid": str(aug_sid),
        "source_sid": str(base_sid),
        "combo": str(combo),
        "before": before_name,
        "after": after_name,
        "crop_mode": None if params is None else params.get("crop_mode", None),
    }


def draw_overlay_with_vis(
    image_bgr: np.ndarray,
    tool_pts3: np.ndarray,
    tool_vis3: np.ndarray,
    pt_role: np.ndarray,
    contact_xy4: np.ndarray,
    contact_vis4: np.ndarray,
    W: int,
    H: int,
    title: str = "",
) -> np.ndarray:
    """
    Draw one frame overlay using stored coordinates and visibility.
    Coordinates are assumed to be in the dataset coordinate space:
      - pixel if USE_PIXEL_COORDS
      - normalized if not
    """
    img = image_bgr.copy()

    def to_px(x, y):
        if USE_PIXEL_COORDS:
            px, py = int(round(float(x))), int(round(float(y)))
        else:
            px = int(round(float(x) * max(1, W - 1)))
            py = int(round(float(y) * max(1, H - 1)))
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))
        return px, py

    # contacts: red
    for i, (xy, vv) in enumerate(zip(np.asarray(contact_xy4, dtype=np.float32), np.asarray(contact_vis4, dtype=np.float32))):
        px, py = to_px(float(xy[0]), float(xy[1]))
        cv2.circle(img, (px, py), DEBUG_DRAW_RADIUS, (0, 0, 255), -1)
        cv2.putText(
            img,
            f"c{i}:{float(vv):.2f}",
            (px + 5, py - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # tool points: anchors blue / tips green
    roles = np.asarray(pt_role, dtype=np.int64).reshape(-1)
    for i, (xy, vv) in enumerate(zip(np.asarray(tool_pts3, dtype=np.float32), np.asarray(tool_vis3, dtype=np.float32))):
        px, py = to_px(float(xy[0]), float(xy[1]))
        role = int(roles[i]) if i < roles.shape[0] else 0
        if role == 0:
            color = (255, 0, 0)
            tag = f"a{i}"
        else:
            color = (0, 255, 0)
            tag = f"t{i}"
        cv2.circle(img, (px, py), DEBUG_DRAW_RADIUS, color, -1)
        cv2.putText(
            img,
            f"{tag}:{float(vv):.2f}",
            (px + 5, py + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if title:
        cv2.putText(
            img,
            title,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return img


def export_random_clip_visualizations(
    base_items: List[BaseItem],
    root_dir: Path,
    out_root: Path,
    dirs: Dict[str, Path],
    rng_py: random.Random,
    n_samples: int = RANDOM_CLIP_VIS_SAMPLES,
) -> List[str]:
    """
    Randomly sample base clips and export all sampled 16 frames with keypoint overlays.
    Uses stored coordinates/vis arrays and keeps the same crop bbox as extraction.
    """
    if n_samples <= 0 or len(base_items) == 0:
        return []

    k = min(int(n_samples), len(base_items))
    picked = rng_py.sample(base_items, k) if len(base_items) > k else list(base_items)

    vis_root = out_root / RANDOM_CLIP_VIS_DIR_NAME
    safe_mkdir(vis_root)
    exported: List[str] = []

    for it in picked:
        ann_path = root_dir / it.ann_relpath
        video_path = ann_path.parent.parent / "clips" / f"{ann_path.stem}.mp4"
        if not video_path.exists():
            continue

        sid = it.sid
        sample_dir = vis_root / f"{sid}_{it.combo.replace('|', '_')}"
        safe_mkdir(sample_dir)

        try:
            sampled_abs = np.load(dirs["FRAME_INDEX_DIR"] / f"{sid}.npy").astype(np.int64).reshape(-1)
            tool_xy = np.load(dirs["GT_TOOL_XY_DIR"] / f"{sid}.npy").astype(np.float32)     # (16,3,2)
            tool_vis = np.load(dirs["KEYPOINT_VIS_DIR"] / f"{sid}.npy").astype(np.float32)  # raw vis (16,3)
            ct_xy = np.load(dirs["GT_CT_XY_DIR"] / f"{sid}.npy").astype(np.float32)          # (16,4,2)
            ct_vis = np.load(dirs["CONTACT_VIS_DIR"] / f"{sid}.npy").astype(np.float32)      # raw vis (16,4)
        except Exception:
            continue

        bbox = None
        if it.crop_bbox_xyxy is not None and len(it.crop_bbox_xyxy) == 4:
            x0, y0, x1, y1 = [int(v) for v in it.crop_bbox_xyxy]
            if x1 > x0 and y1 > y0:
                bbox = (x0, y0, x1, y1)

        T = int(min(sampled_abs.shape[0], tool_xy.shape[0], tool_vis.shape[0], ct_xy.shape[0], ct_vis.shape[0]))
        for t in range(T):
            abs_idx = int(sampled_abs[t])
            local_idx = abs_idx - int(it.video_base_abs)
            frame_full = read_video_frame_at_local_index(video_path, local_idx)
            if frame_full is None:
                continue
            frame = crop_frame(frame_full, bbox) if bbox is not None else frame_full
            Hf, Wf = frame.shape[:2]
            title = f"{sid} t={t:02d} abs={abs_idx}"
            vis_img = draw_overlay_with_vis(
                image_bgr=frame,
                tool_pts3=tool_xy[t],
                tool_vis3=tool_vis[t],
                pt_role=np.asarray(it.pt_role, dtype=np.int64),
                contact_xy4=ct_xy[t],
                contact_vis4=ct_vis[t],
                W=Wf,
                H=Hf,
                title=title,
            )
            cv2.imwrite(str(sample_dir / f"{t:02d}_abs{abs_idx}.jpg"), vis_img)

        exported.append(sid)

    return exported


def dump_selected_samples_all_data(
    selected_sids: List[str],
    sid2item: Dict[str, BaseItem],
    dirs: Dict[str, Path],
    out_root: Path,
) -> List[Path]:
    """
    Print and save all per-sample arrays/metadata for selected SIDs.
    """
    if len(selected_sids) == 0:
        return []

    dump_dir = out_root / RANDOM_CLIP_VIS_DIR_NAME / "sample_data_dump"
    safe_mkdir(dump_dir)
    out_paths: List[Path] = []

    for sid in selected_sids:
        item = sid2item.get(sid, None)
        payload = {
            "sid": sid,
            "meta": (asdict(item) if item is not None else None),
            "files": {},
        }

        for key, dpath in sorted(dirs.items(), key=lambda kv: kv[0]):
            npy_path = dpath / f"{sid}.npy"
            txt_path = dpath / f"{sid}.txt"
            jpg_path = dpath / f"{sid}.jpg"

            if npy_path.exists():
                try:
                    arr = np.load(npy_path)
                    node = {
                        "path": str(npy_path),
                        "dtype": str(arr.dtype),
                        "shape": list(arr.shape),
                    }
                    if int(arr.size) <= int(DEBUG_INLINE_ARRAY_MAX_ELEMS):
                        node["values"] = arr.tolist()
                    else:
                        node["values"] = "omitted_large_array"
                        node["size"] = int(arr.size)
                        node["min"] = float(np.min(arr))
                        node["max"] = float(np.max(arr))
                        node["mean"] = float(np.mean(arr))
                    payload["files"][key] = node
                except Exception as e:
                    payload["files"][key] = {
                        "path": str(npy_path),
                        "load_error": str(e),
                    }
            elif txt_path.exists():
                try:
                    payload["files"][key] = {
                        "path": str(txt_path),
                        "text": txt_path.read_text(encoding="utf-8").strip(),
                    }
                except Exception as e:
                    payload["files"][key] = {
                        "path": str(txt_path),
                        "load_error": str(e),
                    }
            elif jpg_path.exists():
                img = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
                if img is None:
                    payload["files"][key] = {
                        "path": str(jpg_path),
                        "image_shape": None,
                    }
                else:
                    payload["files"][key] = {
                        "path": str(jpg_path),
                        "image_shape": [int(v) for v in img.shape],
                    }

        out_path = dump_dir / f"{sid}_all_data.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        out_paths.append(out_path)

        print("\n[DEBUG_SAMPLE_DATA]")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return out_paths


def build_pose_mask_image(anchor_tip_norm: np.ndarray,
                          contact_xy4_norm: np.ndarray,
                          W: int, H: int,
                          radius_px: int = 2) -> np.ndarray:
    """
    Build a black/white mask image from anchor_tip + contact points.
    Inputs are in the same coordinate space as normalize_xy():
      - pixel if USE_PIXEL_COORDS
      - normalized if not
    Returns BGR uint8 image (H,W,3).
    """
    mask = np.zeros((H, W), dtype=np.uint8)

    def to_px(x, y):
        if USE_PIXEL_COORDS:
            px = int(round(float(x)))
            py = int(round(float(y)))
        else:
            px = int(round(float(x) * W))
            py = int(round(float(y) * H))
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))
        return px, py

    anchor_tip_norm = np.asarray(anchor_tip_norm, dtype=np.float32)
    if anchor_tip_norm.ndim == 1 and anchor_tip_norm.size == 6:
        anchor_pts = anchor_tip_norm.reshape(3, 2)
    else:
        anchor_pts = anchor_tip_norm.reshape(-1, 2)

    contact_pts = np.asarray(contact_xy4_norm, dtype=np.float32).reshape(-1, 2)

    # fill contact polygon (connect 4 points and fill)
    if contact_pts.shape[0] >= 3:
        poly = []
        for (x, y) in contact_pts:
            px, py = to_px(x, y)
            poly.append([px, py])
        poly = np.array([poly], dtype=np.int32)
        cv2.fillPoly(mask, poly, 255)
        if MASK_BORDER_PX > 0:
            cv2.polylines(mask, poly, isClosed=True, color=MASK_BORDER_COLOR, thickness=MASK_BORDER_PX)

    # fill anchor tip triangle (if 3 points)
    if anchor_pts.shape[0] >= 3:
        poly_a = []
        for (x, y) in anchor_pts:
            px, py = to_px(x, y)
            poly_a.append([px, py])
        poly_a = np.array([poly_a], dtype=np.int32)
        cv2.fillPoly(mask, poly_a, 255)
        if MASK_BORDER_PX > 0:
            cv2.polylines(mask, poly_a, isClosed=True, color=MASK_BORDER_COLOR, thickness=MASK_BORDER_PX)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask_bgr


def abs_to_delta(motion_abs: np.ndarray) -> np.ndarray:
    """
    motion_abs: (T,3,2) float32
    return delta_flat: (T,6) float32, delta[0]=0
    """
    motion_abs = np.asarray(motion_abs, dtype=np.float32)
    T = motion_abs.shape[0]
    delta = np.zeros_like(motion_abs, dtype=np.float32)
    if T > 1:
        delta[1:] = motion_abs[1:] - motion_abs[:-1]
    delta_flat = delta.reshape(T, 6).astype(np.float32)
    # enforce exact zeros at t=0
    delta_flat[0, :] = 0.0
    return delta_flat

# motion[0]表示的就是第一个step了！
def abs_to_stepdelta(motion_abs: np.ndarray) -> np.ndarray:
    """
    motion_abs: (T,3,2) float32
    return step_delta_flat: (T-1,6) float32
      step_delta[t] = abs[t+1] - abs[t]
    """
    motion_abs = np.asarray(motion_abs, dtype=np.float32)
    T = int(motion_abs.shape[0])
    if T <= 1:
        return np.zeros((0, 6), dtype=np.float32)

    step = motion_abs[1:] - motion_abs[:-1]  # (T-1,3,2)
    return step.reshape(T - 1, 6).astype(np.float32)

# 2026/01/30
def abs_to_relative(motion_abs: np.ndarray, include_t0: bool = False) -> np.ndarray:
    """
    Convert ABS motion to "relative-to-first" representation.

    motion_abs: (T,3,2)
    if include_t0:
        return rel_flat: (T,6) where rel[t] = abs[t] - abs[0], rel[0]=0
    else:
        return rel_flat: (T-1,6) where rel[t] = abs[t+1] - abs[0]
                         (so motion[0] == abs[1]-abs[0])
    """
    motion_abs = np.asarray(motion_abs, dtype=np.float32)
    T = int(motion_abs.shape[0])
    if T <= 1:
        return np.zeros((0, 6), dtype=np.float32) if (not include_t0) else np.zeros((T, 6), dtype=np.float32)

    rel = motion_abs - motion_abs[:1]  # (T,3,2), rel[0]=0
    if include_t0:
        rel_flat = rel.reshape(T, 6).astype(np.float32)
        rel_flat[0, :] = 0.0
        return rel_flat
    else:
        rel_flat = rel[1:].reshape(T - 1, 6).astype(np.float32)
        return rel_flat


def relative_to_abs(rel_flat: np.ndarray, anchor_tip_flat: np.ndarray, include_t0: bool = False) -> np.ndarray:
    """
    Inverse of abs_to_relative given anchor_tip (abs at t=0).

    anchor_tip_flat: (6,) = abs[0]
    if include_t0:
        rel_flat: (T,6), rel[0]=0
        return abs: (T,3,2), abs[t] = abs[0] + rel[t]
    else:
        rel_flat: (L,6) where L=T-1, rel[t] = abs[t+1]-abs[0]
        return abs: (L+1,3,2), abs[0]=abs0, abs[t+1]=abs0+rel[t]
    """
    rel_flat = np.asarray(rel_flat, dtype=np.float32)
    abs0 = np.asarray(anchor_tip_flat, dtype=np.float32).reshape(3, 2)

    if include_t0:
        T = int(rel_flat.shape[0])
        motion_abs = np.zeros((T, 3, 2), dtype=np.float32)
        if T > 0:
            motion_abs[:] = abs0[None] + rel_flat.reshape(T, 3, 2)
            motion_abs[0] = abs0  # enforce exact anchor
        return motion_abs

    else:
        L = int(rel_flat.shape[0])
        motion_abs = np.zeros((L + 1, 3, 2), dtype=np.float32)
        motion_abs[0] = abs0
        if L > 0:
            motion_abs[1:] = abs0[None] + rel_flat.reshape(L, 3, 2)
        return motion_abs


def encode_motion(motion_abs: np.ndarray) -> np.ndarray:
    """
    Encode ABS motion into the configured representation (MOTION_REP).
    """
    if MOTION_REP == "stepdelta":
        return abs_to_stepdelta(motion_abs)  # (T-1,6)
    elif MOTION_REP == "relative":
        return abs_to_relative(motion_abs, include_t0=REL_INCLUDE_T0)
    else:
        raise ValueError(f"Unknown MOTION_REP: {MOTION_REP}")


def decode_motion(motion_repr: np.ndarray, anchor_tip_flat: np.ndarray) -> np.ndarray:
    """
    Decode stored motion representation back to ABS (T,3,2).
    """
    if MOTION_REP == "stepdelta":
        return stepdelta_to_abs(motion_repr, anchor_tip_flat)  # (L+1,3,2)
    elif MOTION_REP == "relative":
        return relative_to_abs(motion_repr, anchor_tip_flat, include_t0=REL_INCLUDE_T0)
    else:
        raise ValueError(f"Unknown MOTION_REP: {MOTION_REP}")

# 2026/01/31

def delta_to_abs(delta_flat: np.ndarray, anchor_tip_flat: np.ndarray) -> np.ndarray:
    """
    Reconstruct abs from delta and anchor_tip.
    delta_flat: (T,6)
    anchor_tip_flat: (6,)
    return motion_abs: (T,3,2)
    """
    delta_flat = np.asarray(delta_flat, dtype=np.float32)
    T = delta_flat.shape[0]
    motion_abs = np.zeros((T, 3, 2), dtype=np.float32)
    motion_abs[0] = np.asarray(anchor_tip_flat, dtype=np.float32).reshape(3, 2)
    if T > 1:
        for t in range(1, T):
            motion_abs[t] = motion_abs[t - 1] + delta_flat[t].reshape(3, 2)
    return motion_abs

# 这个是配合motion[0]是第一个step的case的！
def stepdelta_to_abs(step_delta_flat: np.ndarray, anchor_tip_flat: np.ndarray) -> np.ndarray:
    """
    step_delta_flat: (L,6) where L = T-1
    anchor_tip_flat: (6,) abs at t=0

    return motion_abs: (L+1,3,2) i.e. original T
      abs[0] = anchor_tip
      abs[t+1] = abs[t] + step_delta[t]
    """
    step_delta_flat = np.asarray(step_delta_flat, dtype=np.float32)
    L = int(step_delta_flat.shape[0])

    motion_abs = np.zeros((L + 1, 3, 2), dtype=np.float32)
    motion_abs[0] = np.asarray(anchor_tip_flat, dtype=np.float32).reshape(3, 2)

    for t in range(L):
        motion_abs[t + 1] = motion_abs[t] + step_delta_flat[t].reshape(3, 2)

    return motion_abs



# =========================
# Augmentation helpers
# =========================
def sample_aug_type(rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    for k, p in AUG_PROBS.items():
        cum += p
        if r <= cum:
            return k
    return "A"


def compute_center_norm(motion_abs: np.ndarray, contact_xy4: np.ndarray) -> Tuple[float, float]:
    all_pts = np.concatenate([motion_abs.reshape(-1, 2), contact_xy4.reshape(-1, 2)], axis=0)
    c = all_pts.mean(axis=0)
    return float(c[0]), float(c[1])


def affine_matrix_px(W: int, H: int,
                     center_xy: Tuple[float, float],
                     angle_deg: float,
                     scale: float,
                     trans_xy: Tuple[float, float]) -> np.ndarray:
    """
    Build cv2 affine matrix in pixel space (2x3).
    center_xy / trans_xy are in:
      - pixel if USE_PIXEL_COORDS
      - normalized if not
    """
    cx, cy = center_xy
    tx, ty = trans_xy

    if USE_PIXEL_COORDS:
        center_px = (float(cx), float(cy))
        tx_px, ty_px = float(tx), float(ty)
    else:
        center_px = (float(cx) * W, float(cy) * H)
        tx_px, ty_px = float(tx) * W, float(ty) * H

    M = cv2.getRotationMatrix2D(center_px, angle_deg, scale)
    M[0, 2] += tx_px
    M[1, 2] += ty_px
    return M.astype(np.float32)


def apply_affine_to_points_norm(xy_norm: np.ndarray, M: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Apply affine to points in either pixel or normalized space.
    If USE_PIXEL_COORDS:
        xy_norm is actually xy_px (float), returned in pixel.
    Else:
        xy_norm is normalized, returned normalized.
    """
    xy = np.asarray(xy_norm, dtype=np.float32)
    orig_shape = xy.shape
    pts = xy.reshape(-1, 2)

    if USE_PIXEL_COORDS:
        pts_px = pts
    else:
        pts_px = pts.copy()
        pts_px[:, 0] *= float(W)
        pts_px[:, 1] *= float(H)

    ones = np.ones((pts_px.shape[0], 1), dtype=np.float32)
    pts1 = np.concatenate([pts_px, ones], axis=1)  # Nx3
    out_px = (pts1 @ M.T).astype(np.float32)       # Nx2

    if USE_PIXEL_COORDS:
        out = out_px
    else:
        out = out_px.copy()
        out[:, 0] /= float(W)
        out[:, 1] /= float(H)

    return out.reshape(orig_shape).astype(np.float32)



def in_bounds(xy_norm: np.ndarray, W: int, H: int, margin: float = 0.02) -> bool:
    xy = np.asarray(xy_norm, dtype=np.float32)
    if USE_PIXEL_COORDS:
        m = float(margin)  # 这里 margin 解释为 “像素容忍”
        xmin, xmax = float(xy[..., 0].min()), float(xy[..., 0].max())
        ymin, ymax = float(xy[..., 1].min()), float(xy[..., 1].max())
        return (xmin >= -m) and (xmax <= float(W) + m) and (ymin >= -m) and (ymax <= float(H) + m)
    else:
        m = float(margin)
        return bool(np.all(xy >= -m) and np.all(xy <= 1.0 + m))



def time_warp_abs(motion_abs: np.ndarray, ratio: float) -> np.ndarray:
    """
    Keep length T, but warp time parameter u -> u' = clamp(u*ratio, 0,1)
    motion_abs: (T,3,2)
    """
    motion_abs = np.asarray(motion_abs, dtype=np.float32)
    T = motion_abs.shape[0]
    if T <= 1:
        return motion_abs.copy()

    u = np.linspace(0.0, 1.0, T, dtype=np.float32)
    u2 = np.clip(u * float(ratio), 0.0, 1.0)

    # interpolate per dimension
    out = np.zeros_like(motion_abs, dtype=np.float32)
    for j in range(3):
        for d in range(2):
            out[:, j, d] = np.interp(u2, u, motion_abs[:, j, d]).astype(np.float32)
    # ensure start frame stays identical
    out[0] = motion_abs[0]
    return out


def lowfreq_noise_abs(motion_abs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Add smooth noise (EMA filtered) to abs motion, with noise[0]=0.
    """
    motion_abs = np.asarray(motion_abs, dtype=np.float32)
    T = motion_abs.shape[0]
    if T <= 1:
        return motion_abs.copy()

    noise = rng.normal(0.0, NOISE_STD, size=motion_abs.shape).astype(np.float32)
    noise[0] = 0.0

    # EMA along time
    for t in range(1, T):
        noise[t] = NOISE_EMA * noise[t - 1] + (1.0 - NOISE_EMA) * noise[t]

    if USE_PIXEL_COORDS:
        out = motion_abs + noise
    else:
        out = np.clip(motion_abs + noise, 0.0, 1.0)

    out[0] = motion_abs[0]

    return out.astype(np.float32)


def _read_sid_set(path: Path) -> Set[str]:
    if path is None or (not path.is_file()):
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def _weighted_choice_rng(weight_map: Dict[str, float], rng_py: random.Random) -> str:
    keys = list(weight_map.keys())
    ws = np.array([float(weight_map[k]) for k in keys], dtype=np.float64)
    s = float(ws.sum())
    if s <= 0.0:
        return keys[0]
    ws = ws / s
    r = rng_py.random()
    cum = 0.0
    for k, p in zip(keys, ws.tolist()):
        cum += float(p)
        if r <= cum:
            return str(k)
    return str(keys[-1])


def _norm_to_px(xy_norm: np.ndarray, W: int, H: int) -> np.ndarray:
    out = np.asarray(xy_norm, dtype=np.float32).copy()
    out[..., 0] *= float(W)
    out[..., 1] *= float(H)
    return out


def _px_to_norm(xy_px: np.ndarray, W: int, H: int) -> np.ndarray:
    out = np.asarray(xy_px, dtype=np.float32).copy()
    out[..., 0] /= float(max(1, W))
    out[..., 1] /= float(max(1, H))
    return out


def _temporal_warp_indices(T: int, ratio: float) -> np.ndarray:
    if T <= 1:
        return np.zeros((max(1, T),), dtype=np.int64)
    u = np.linspace(0.0, 1.0, T, dtype=np.float32)
    u2 = np.clip(u * float(ratio), 0.0, 1.0)
    idx = np.rint(u2 * float(T - 1)).astype(np.int64)
    return idx


def _temporal_start_shift_indices(T: int, shift_frames: int) -> np.ndarray:
    if T <= 1:
        return np.zeros((max(1, T),), dtype=np.int64)
    s = int(max(0, shift_frames))
    if s <= 0:
        return np.arange(T, dtype=np.int64)
    idx = np.arange(T, dtype=np.int64) + int(s)
    idx = np.clip(idx, 0, T - 1)
    return idx.astype(np.int64)


def fold1_temporal_start_jitter_clip(
    video16: np.ndarray,
    gt_tool_xy: np.ndarray,
    gt_tool_vis: np.ndarray,
    gt_ct_xy: np.ndarray,
    gt_ct_vis: np.ndarray,
    sampled_abs: np.ndarray,
    shift_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = int(video16.shape[0])
    idx = _temporal_start_shift_indices(T, shift_frames=shift_frames)
    return (
        video16[idx].copy(),
        gt_tool_xy[idx].astype(np.float32),
        gt_tool_vis[idx].astype(np.float32),
        gt_ct_xy[idx].astype(np.float32),
        gt_ct_vis[idx].astype(np.float32),
        sampled_abs[idx].astype(np.int64),
    )


def fold1_temporal_warp_clip(
    video16: np.ndarray,
    gt_tool_xy: np.ndarray,
    gt_tool_vis: np.ndarray,
    gt_ct_xy: np.ndarray,
    gt_ct_vis: np.ndarray,
    sampled_abs: np.ndarray,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = int(video16.shape[0])
    idx = _temporal_warp_indices(T, ratio)
    return (
        video16[idx].copy(),
        gt_tool_xy[idx].astype(np.float32),
        gt_tool_vis[idx].astype(np.float32),
        gt_ct_xy[idx].astype(np.float32),
        gt_ct_vis[idx].astype(np.float32),
        sampled_abs[idx].astype(np.int64),
    )


def _compute_fold1_crop_bbox(
    mode: str,
    gt_tool_xy: np.ndarray,  # (T,3,2) normalized
    gt_ct_xy: np.ndarray,    # (T,4,2) normalized
    pt_role: List[int],
    W: int,
    H: int,
    rng_py: random.Random,
) -> Tuple[int, int, int, int]:
    params = FOLD1_CROP_PARAMS[mode]
    pad_scale = float(params["pad_scale"])
    min_size_frac = float(params["min_size_frac"])
    jitter_center_frac = float(params["jitter_center_frac"])
    jitter_scale_lo = float(params["jitter_scale_lo"])
    jitter_scale_hi = float(params["jitter_scale_hi"])

    tool_px = _norm_to_px(gt_tool_xy, W, H)  # (T,3,2)
    ct_px = _norm_to_px(gt_ct_xy, W, H)      # (T,4,2)
    tool_flat = tool_px.reshape(-1, 2)
    ct_flat = ct_px.reshape(-1, 2)

    anchors = [i for i, r in enumerate(pt_role) if int(r) == 0]
    tips = [i for i, r in enumerate(pt_role) if int(r) == 1]
    if len(anchors) == 0:
        anchors = [0]
    if len(tips) == 0:
        tips = [1, 2] if int(tool_px.shape[1]) >= 3 else [0]

    if mode == "context":
        pts_for_bbox = np.concatenate([tool_flat, ct_flat], axis=0)
        center = pts_for_bbox.mean(axis=0)
    elif mode == "hinge":
        anchor_pts = tool_px[:, anchors, :].reshape(-1, 2)
        pts_for_bbox = tool_flat
        center = anchor_pts.mean(axis=0)
    else:
        tip_pts = tool_px[:, tips, :].reshape(-1, 2)
        pts_for_bbox = np.concatenate([tip_pts, ct_flat], axis=0)
        center = pts_for_bbox.mean(axis=0)

    xmin = float(np.min(pts_for_bbox[:, 0]))
    xmax = float(np.max(pts_for_bbox[:, 0]))
    ymin = float(np.min(pts_for_bbox[:, 1]))
    ymax = float(np.max(pts_for_bbox[:, 1]))
    bw = max(2.0, xmax - xmin)
    bh = max(2.0, ymax - ymin)

    side = max(bw, bh) * pad_scale
    side = max(side, min_size_frac * float(min(W, H)))
    side = side * float(rng_py.uniform(jitter_scale_lo, jitter_scale_hi))

    jitter = jitter_center_frac * side
    cx = float(center[0] + rng_py.uniform(-jitter, jitter))
    cy = float(center[1] + rng_py.uniform(-jitter, jitter))
    cx = float(np.clip(cx, 0.0, float(W)))
    cy = float(np.clip(cy, 0.0, float(H)))

    x0 = int(round(cx - side * 0.5))
    y0 = int(round(cy - side * 0.5))
    x1 = int(round(cx + side * 0.5))
    y1 = int(round(cy + side * 0.5))
    x0 = int(np.clip(x0, 0, W - 2))
    y0 = int(np.clip(y0, 0, H - 2))
    x1 = int(np.clip(x1, x0 + 2, W))
    y1 = int(np.clip(y1, y0 + 2, H))

    # ensure all points remain inside crop, expand minimally when needed.
    all_pts = np.concatenate([tool_flat, ct_flat], axis=0)
    need_x0 = int(math.floor(float(all_pts[:, 0].min())))
    need_y0 = int(math.floor(float(all_pts[:, 1].min())))
    need_x1 = int(math.ceil(float(all_pts[:, 0].max())))
    need_y1 = int(math.ceil(float(all_pts[:, 1].max())))
    if need_x0 < x0 or need_y0 < y0 or need_x1 > x1 or need_y1 > y1:
        x0 = min(x0, need_x0)
        y0 = min(y0, need_y0)
        x1 = max(x1, need_x1)
        y1 = max(y1, need_y1)
        x0 = int(np.clip(x0, 0, W - 2))
        y0 = int(np.clip(y0, 0, H - 2))
        x1 = int(np.clip(x1, x0 + 2, W))
        y1 = int(np.clip(y1, y0 + 2, H))

    # Extra hard margin ring to avoid tight crop around keypoint neighborhoods.
    margin_px = int(
        round(
            max(
                float(FOLD1_CROP_EXTRA_MARGIN_MIN_PX),
                float(FOLD1_CROP_EXTRA_MARGIN_FRAC) * float(min(W, H)),
            )
        )
    )
    if margin_px > 0:
        x0 = int(np.clip(x0 - margin_px, 0, W - 2))
        y0 = int(np.clip(y0 - margin_px, 0, H - 2))
        x1 = int(np.clip(x1 + margin_px, x0 + 2, W))
        y1 = int(np.clip(y1 + margin_px, y0 + 2, H))
    return x0, y0, x1, y1


def _apply_fold1_crop_clip(
    video16: np.ndarray,
    gt_tool_xy: np.ndarray,
    gt_ct_xy: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = bbox_xyxy
    T, H, W, _ = video16.shape
    x0 = int(np.clip(x0, 0, W - 2))
    y0 = int(np.clip(y0, 0, H - 2))
    x1 = int(np.clip(x1, x0 + 2, W))
    y1 = int(np.clip(y1, y0 + 2, H))

    video_c = video16[:, y0:y1, x0:x1, :].copy()
    Hc, Wc = int(video_c.shape[1]), int(video_c.shape[2])

    tool_px = _norm_to_px(gt_tool_xy, W, H)
    ct_px = _norm_to_px(gt_ct_xy, W, H)
    tool_px[..., 0] -= float(x0)
    tool_px[..., 1] -= float(y0)
    ct_px[..., 0] -= float(x0)
    ct_px[..., 1] -= float(y0)
    tool_xy_c = _px_to_norm(tool_px, Wc, Hc)
    ct_xy_c = _px_to_norm(ct_px, Wc, Hc)
    return video_c, np.clip(tool_xy_c, 0.0, 1.0), np.clip(ct_xy_c, 0.0, 1.0)


def _apply_fold1_hflip_clip(
    video16: np.ndarray,
    gt_tool_xy: np.ndarray,
    gt_ct_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_video = video16[:, :, ::-1, :].copy()
    out_tool = np.asarray(gt_tool_xy, dtype=np.float32).copy()
    out_ct = np.asarray(gt_ct_xy, dtype=np.float32).copy()
    out_tool[..., 0] = 1.0 - out_tool[..., 0]
    out_ct[..., 0] = 1.0 - out_ct[..., 0]
    return out_video, np.clip(out_tool, 0.0, 1.0), np.clip(out_ct, 0.0, 1.0)


def _affine_matrix_center_px(W: int, H: int, rot_deg: float, scale: float, tx: float, ty: float) -> np.ndarray:
    cx = float(W) * 0.5
    cy = float(H) * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), float(rot_deg), float(scale))
    M[0, 2] += float(tx)
    M[1, 2] += float(ty)
    return M.astype(np.float32)


def _apply_affine_points_px(xy_px: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy_px, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts1 = np.concatenate([pts, ones], axis=1)
    out = (pts1 @ M.T).astype(np.float32)
    return out.reshape(xy_px.shape)


def _all_points_in_bounds_norm(gt_tool_xy: np.ndarray, gt_ct_xy: np.ndarray, margin: float = 0.01) -> bool:
    lo = float(0.0 + margin)
    hi = float(1.0 - margin)
    if np.any(gt_tool_xy[..., 0] < lo) or np.any(gt_tool_xy[..., 0] > hi):
        return False
    if np.any(gt_tool_xy[..., 1] < lo) or np.any(gt_tool_xy[..., 1] > hi):
        return False
    if np.any(gt_ct_xy[..., 0] < lo) or np.any(gt_ct_xy[..., 0] > hi):
        return False
    if np.any(gt_ct_xy[..., 1] < lo) or np.any(gt_ct_xy[..., 1] > hi):
        return False
    return True


def _apply_fold1_affine_clip(
    video16: np.ndarray,
    gt_tool_xy: np.ndarray,
    gt_ct_xy: np.ndarray,
    rng_py: random.Random,
    max_tries: int = 10,
    rot_deg_max: Optional[float] = None,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    trans_frac: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    T, H, W, _ = video16.shape
    rot_deg_max = float(FOLD1_AFFINE_ROT_DEG if rot_deg_max is None else rot_deg_max)
    scale_min = float(FOLD1_AFFINE_SCALE_MIN if scale_min is None else scale_min)
    scale_max = float(FOLD1_AFFINE_SCALE_MAX if scale_max is None else scale_max)
    trans_frac = float(FOLD1_AFFINE_TRANS_FRAC if trans_frac is None else trans_frac)
    for _ in range(max_tries):
        rot = float(rng_py.uniform(-rot_deg_max, rot_deg_max))
        scale = float(rng_py.uniform(scale_min, scale_max))
        tmax = trans_frac * float(min(W, H))
        tx = float(rng_py.uniform(-tmax, tmax))
        ty = float(rng_py.uniform(-tmax, tmax))
        M = _affine_matrix_center_px(W, H, rot, scale, tx, ty)

        tool_px = _norm_to_px(gt_tool_xy, W, H)
        ct_px = _norm_to_px(gt_ct_xy, W, H)
        tool_px2 = _apply_affine_points_px(tool_px, M)
        ct_px2 = _apply_affine_points_px(ct_px, M)
        tool_xy2 = _px_to_norm(tool_px2, W, H)
        ct_xy2 = _px_to_norm(ct_px2, W, H)
        if not _all_points_in_bounds_norm(tool_xy2, ct_xy2, margin=0.01):
            continue

        out_video = np.empty_like(video16)
        for t in range(T):
            out_video[t] = cv2.warpAffine(
                video16[t],
                M,
                dsize=(W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        return out_video, np.clip(tool_xy2, 0.0, 1.0), np.clip(ct_xy2, 0.0, 1.0), M
    return video16, gt_tool_xy, gt_ct_xy, None


def _build_fold1_kpt_mask_frame(
    tool_xy_t: np.ndarray,   # (3,2) normalized
    ct_xy_t: np.ndarray,     # (4,2) normalized
    pt_role: List[int],
    W: int,
    H: int,
) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    tool_px = np.rint(_norm_to_px(tool_xy_t, W, H)).astype(np.int32)
    ct_px = np.rint(_norm_to_px(ct_xy_t, W, H)).astype(np.int32)

    if ct_px.shape[0] >= 3:
        cv2.fillPoly(mask, [ct_px.reshape(-1, 1, 2)], 255)
    if tool_px.shape[0] == 3:
        cv2.fillPoly(mask, [tool_px.reshape(-1, 1, 2)], 255)

    anchors = [i for i, r in enumerate(pt_role) if int(r) == 0]
    tips = [i for i, r in enumerate(pt_role) if int(r) == 1]
    if len(anchors) == 0:
        anchors = [0]
    if len(tips) == 0:
        tips = [1, 2] if int(tool_px.shape[0]) >= 3 else [0]

    thickness = max(2, int(round(FOLD1_MASK_SKELETON_THICK_FRAC * float(min(W, H)))))
    for a in anchors:
        for tp in tips:
            pa = tuple(tool_px[a].tolist())
            pb = tuple(tool_px[tp].tolist())
            cv2.line(mask, pa, pb, 255, thickness=thickness, lineType=cv2.LINE_AA)
    if len(tips) >= 2:
        p0 = tuple(tool_px[tips[0]].tolist())
        p1 = tuple(tool_px[tips[1]].tolist())
        cv2.line(mask, p0, p1, 255, thickness=max(1, thickness // 2), lineType=cv2.LINE_AA)

    # Keep a larger neighborhood around each keypoint (tool + contact).
    pt_radius = int(
        round(
            max(
                float(FOLD1_MASK_POINT_RADIUS_MIN_PX),
                float(FOLD1_MASK_POINT_RADIUS_FRAC) * float(min(W, H)),
            )
        )
    )
    if pt_radius > 0:
        for p in tool_px:
            cv2.circle(mask, tuple(p.tolist()), pt_radius, 255, thickness=-1, lineType=cv2.LINE_AA)
        for p in ct_px:
            cv2.circle(mask, tuple(p.tolist()), pt_radius, 255, thickness=-1, lineType=cv2.LINE_AA)

    # Optional: fill convex hull over all keypoints to preserve nearby tissue region.
    if bool(FOLD1_MASK_FILL_ALL_KPT_HULL):
        all_pts = np.concatenate([tool_px.reshape(-1, 2), ct_px.reshape(-1, 2)], axis=0)
        if all_pts.shape[0] >= 3:
            hull = cv2.convexHull(all_pts.reshape(-1, 1, 2))
            cv2.fillConvexPoly(mask, hull, 255)

    kd = int(FOLD1_MASK_DILATE_K)
    if kd % 2 == 0:
        kd += 1
    kd = max(1, kd)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
    mask = cv2.dilate(mask, kernel, iterations=1)

    kb = int(FOLD1_MASK_BLUR_K)
    if kb % 2 == 0:
        kb += 1
    if kb >= 3:
        mask_f = cv2.GaussianBlur(mask.astype(np.float32), (kb, kb), 0)
    else:
        mask_f = mask.astype(np.float32)
    mask_f = np.clip(mask_f / 255.0, 0.0, 1.0)
    return mask_f[..., None].astype(np.float32)


def _replace_bg_with_donor_clip(
    video16: np.ndarray,
    gt_tool_xy: np.ndarray,
    gt_ct_xy: np.ndarray,
    pt_role: List[int],
    donor_video16: np.ndarray,
    donor_core_soft_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    T, H, W, _ = video16.shape
    donor = np.asarray(donor_video16, dtype=np.uint8)
    if donor.shape[0] != T:
        idx = np.arange(T, dtype=np.int64) % int(donor.shape[0])
        donor = donor[idx]
    donor_rs = np.empty_like(video16)
    for t in range(T):
        donor_rs[t] = cv2.resize(donor[t], (W, H), interpolation=cv2.INTER_LINEAR)

    out = np.empty_like(video16)
    for t in range(T):
        m_base = _build_fold1_kpt_mask_frame(gt_tool_xy[t], gt_ct_xy[t], pt_role, W, H)
        m = m_base
        if donor_core_soft_mask is not None and donor_core_soft_mask.shape[0] > 0:
            md = donor_core_soft_mask[int(t % int(donor_core_soft_mask.shape[0]))].astype(np.float32)
            m = np.clip(np.maximum(m_base, md), 0.0, 1.0).astype(np.float32)
        fg = video16[t].astype(np.float32)
        bg = donor_rs[t].astype(np.float32)
        out_t = fg * m + bg * (1.0 - m)
        out[t] = np.clip(out_t, 0.0, 255.0).astype(np.uint8)
    return out


def _build_core_soft_mask_seq(
    gt_tool_xy: np.ndarray,
    gt_ct_xy: np.ndarray,
    pt_role: List[int],
    H: int,
    W: int,
) -> np.ndarray:
    T = int(gt_tool_xy.shape[0])
    out = np.zeros((T, H, W, 1), dtype=np.float32)
    for t in range(T):
        out[t] = _build_fold1_kpt_mask_frame(gt_tool_xy[t], gt_ct_xy[t], pt_role, W, H)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _build_donor_core_soft_mask_seq_for_sid(
    dirs: Dict[str, Path],
    donor_sid: str,
    pt_role: List[int],
    T: int,
    H: int,
    W: int,
) -> Optional[np.ndarray]:
    tool_path = dirs["GT_TOOL_XY_DIR"] / f"{donor_sid}.npy"
    ct_path = dirs["GT_CT_XY_DIR"] / f"{donor_sid}.npy"
    if (not tool_path.is_file()) or (not ct_path.is_file()):
        return None
    try:
        donor_tool_xy = np.load(tool_path).astype(np.float32)
        donor_ct_xy = np.load(ct_path).astype(np.float32)
    except Exception:
        return None
    if donor_tool_xy.ndim != 3 or donor_tool_xy.shape[-1] != 2 or donor_tool_xy.shape[0] <= 0:
        return None
    if donor_ct_xy.ndim != 3 or donor_ct_xy.shape[-1] != 2 or donor_ct_xy.shape[0] <= 0:
        return None
    Td_tool = int(donor_tool_xy.shape[0])
    Td_ct = int(donor_ct_xy.shape[0])
    out = np.zeros((int(T), int(H), int(W), 1), dtype=np.float32)
    for t in range(int(T)):
        tt_tool = int(t % Td_tool)
        tt_ct = int(t % Td_ct)
        out[t] = _build_fold1_kpt_mask_frame(
            np.clip(donor_tool_xy[tt_tool], 0.0, 1.0),
            np.clip(donor_ct_xy[tt_ct], 0.0, 1.0),
            pt_role,
            int(W),
            int(H),
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _sample_event_window(T: int, rng_py: random.Random, min_len: int, max_len: int) -> Tuple[int, int]:
    if T <= 0:
        return 0, 0
    min_len = int(max(1, min_len))
    max_len = int(max(min_len, max_len))
    win_len = int(min(T, rng_py.randint(min_len, max_len)))
    if win_len >= T:
        return 0, int(T)
    t0 = int(rng_py.randint(0, int(T - win_len)))
    return t0, int(t0 + win_len)


def _sample_mask_family(
    H: int,
    W: int,
    rng_py: random.Random,
    family: str,
    drift_x: int = 0,
    drift_y: int = 0,
) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.float32)
    fam = str(family)
    if fam == "rect":
        ww = int(round(rng_py.uniform(0.18, 0.50) * float(W)))
        hh = int(round(rng_py.uniform(0.18, 0.50) * float(H)))
        cx = int(round(rng_py.uniform(0.15, 0.85) * float(W))) + int(drift_x)
        cy = int(round(rng_py.uniform(0.15, 0.85) * float(H))) + int(drift_y)
        x0 = int(np.clip(cx - ww // 2, 0, W - 1))
        y0 = int(np.clip(cy - hh // 2, 0, H - 1))
        x1 = int(np.clip(x0 + ww, x0 + 1, W))
        y1 = int(np.clip(y0 + hh, y0 + 1, H))
        m[y0:y1, x0:x1] = 1.0
    elif fam == "blob":
        n_blob = int(rng_py.randint(2, 6))
        for _ in range(n_blob):
            rr = int(round(rng_py.uniform(0.06, 0.18) * float(min(H, W))))
            cx = int(round(rng_py.uniform(0.0, 1.0) * float(W - 1))) + int(drift_x)
            cy = int(round(rng_py.uniform(0.0, 1.0) * float(H - 1))) + int(drift_y)
            cv2.circle(m, (int(np.clip(cx, 0, W - 1)), int(np.clip(cy, 0, H - 1))), max(3, rr), 1.0, thickness=-1)
    elif fam == "wedge":
        cx = int(round(rng_py.uniform(0.20, 0.80) * float(W))) + int(drift_x)
        cy = int(round(rng_py.uniform(0.20, 0.80) * float(H))) + int(drift_y)
        span = int(round(rng_py.uniform(0.30, 0.75) * float(min(H, W))))
        p0 = [int(np.clip(cx, 0, W - 1)), int(np.clip(cy, 0, H - 1))]
        p1 = [int(np.clip(cx - span, 0, W - 1)), int(np.clip(cy + span, 0, H - 1))]
        p2 = [int(np.clip(cx + span, 0, W - 1)), int(np.clip(cy + span // 2, 0, H - 1))]
        poly = np.array([p0, p1, p2], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [poly], 1.0)
    elif fam == "stripe":
        pitch = max(10, int(round(rng_py.uniform(0.06, 0.14) * float(min(H, W)))))
        width = max(3, int(round(rng_py.uniform(0.25, 0.50) * float(pitch))))
        for y in range((drift_y % pitch), H, pitch):
            y0 = int(np.clip(y, 0, H - 1))
            y1 = int(np.clip(y + width, y0 + 1, H))
            m[y0:y1, :] = 1.0
        if rng_py.random() < 0.5:
            for x in range((drift_x % pitch), W, pitch):
                x0 = int(np.clip(x, 0, W - 1))
                x1 = int(np.clip(x + width, x0 + 1, W))
                m[:, x0:x1] = 1.0
    else:
        noise = np.array([[rng_py.random() for _ in range(W)] for _ in range(H)], dtype=np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(2.0, 0.03 * min(H, W)))
        thr = float(np.quantile(noise, 0.70))
        m = (noise >= thr).astype(np.float32)
    kb = max(3, int(round(0.02 * min(H, W))) | 1)
    m = cv2.GaussianBlur(m, (kb, kb), 0)
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def _strong_transform_donor_frame(
    frame: np.ndarray,
    rng_py: random.Random,
    strength_bucket: str = "heavy",
) -> np.ndarray:
    H, W = int(frame.shape[0]), int(frame.shape[1])
    out = np.asarray(frame, dtype=np.uint8)
    k = int(rng_py.randint(0, 3))
    if k > 0:
        out = np.ascontiguousarray(np.rot90(out, k=k))
        if out.shape[0] != H or out.shape[1] != W:
            out = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
    sb = str(strength_bucket).lower().strip()
    if sb == "extreme":
        rot = float(rng_py.uniform(-20.0, 20.0))
        scale = float(rng_py.uniform(0.78, 1.30))
        tx = float(rng_py.uniform(-0.16, 0.16) * float(W))
        ty = float(rng_py.uniform(-0.16, 0.16) * float(H))
    else:
        rot = float(rng_py.uniform(-14.0, 14.0))
        scale = float(rng_py.uniform(0.86, 1.20))
        tx = float(rng_py.uniform(-0.10, 0.10) * float(W))
        ty = float(rng_py.uniform(-0.10, 0.10) * float(H))
    M = _affine_matrix_center_px(W=W, H=H, rot_deg=rot, scale=scale, tx=tx, ty=ty)
    out = cv2.warpAffine(out, M, dsize=(W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return out


def _apply_random_erasing_outside_core_event(
    video16: np.ndarray,
    core_soft: np.ndarray,
    rng_py: random.Random,
    strength_bucket: str,
) -> np.ndarray:
    T, H, W, _ = video16.shape
    out = np.asarray(video16, dtype=np.uint8).copy()
    if T <= 0:
        return out
    sb = str(strength_bucket).lower().strip()
    if sb == "extreme":
        min_len, max_len = 6, 13
        area_min, area_max = 0.05, 0.28
        block_n_min, block_n_max = 3, 7
    elif sb == "heavy":
        min_len, max_len = 4, 10
        area_min, area_max = 0.03, 0.20
        block_n_min, block_n_max = 2, 5
    else:
        min_len, max_len = 3, 8
        area_min, area_max = 0.02, 0.12
        block_n_min, block_n_max = 1, 3
    t0, t1 = _sample_event_window(T=T, rng_py=rng_py, min_len=min_len, max_len=max_len)
    area = float(H * W)
    for t in range(t0, t1):
        outside = np.clip(1.0 - core_soft[t], 0.0, 1.0)
        base = out[t].astype(np.float32)
        n_blk = int(rng_py.randint(block_n_min, block_n_max))
        for _ in range(n_blk):
            target_area = float(rng_py.uniform(area_min, area_max) * area / max(1, n_blk))
            aspect = float(np.exp(rng_py.uniform(np.log(0.3), np.log(3.2))))
            hh = int(round(np.sqrt(target_area * aspect)))
            ww = int(round(np.sqrt(target_area / aspect)))
            if hh < 3 or ww < 3 or hh >= H or ww >= W:
                continue
            y0 = int(rng_py.randint(0, max(1, H - hh)))
            x0 = int(rng_py.randint(0, max(1, W - ww)))
            y1 = int(min(H, y0 + hh))
            x1 = int(min(W, x0 + ww))
            mask = np.zeros((H, W, 1), dtype=np.float32)
            mask[y0:y1, x0:x1, 0] = 1.0
            mask = mask * outside
            fill_color = np.array([rng_py.uniform(0, 255), rng_py.uniform(0, 255), rng_py.uniform(0, 255)], dtype=np.float32)
            fill = np.ones_like(base, dtype=np.float32) * fill_color.reshape(1, 1, 3)
            base = base * (1.0 - mask) + fill * mask
        out[t] = np.clip(base, 0.0, 255.0).astype(np.uint8)
    return out


def _apply_grid_occlusion_outside_core_event(
    video16: np.ndarray,
    core_soft: np.ndarray,
    rng_py: random.Random,
    strength_bucket: str,
) -> np.ndarray:
    T, H, W, _ = video16.shape
    out = np.asarray(video16, dtype=np.uint8).copy()
    if T <= 0:
        return out
    sb = str(strength_bucket).lower().strip()
    if sb == "extreme":
        min_len, max_len = 6, 12
    elif sb == "heavy":
        min_len, max_len = 4, 10
    else:
        min_len, max_len = 3, 7
    t0, t1 = _sample_event_window(T=T, rng_py=rng_py, min_len=min_len, max_len=max_len)
    if sb == "extreme":
        pitch = int(round(rng_py.uniform(10.0, 24.0)))
        line_w = int(round(rng_py.uniform(5.0, 14.0)))
    else:
        pitch = int(round(rng_py.uniform(14.0, 32.0)))
        line_w = int(round(rng_py.uniform(3.0, 10.0)))
    for t in range(t0, t1):
        outside = np.clip(1.0 - core_soft[t], 0.0, 1.0)
        gy = int((t - t0) * rng_py.uniform(-2.0, 2.0))
        gx = int((t - t0) * rng_py.uniform(-2.0, 2.0))
        g = np.zeros((H, W), dtype=np.float32)
        y = gy % max(1, pitch)
        while y < H:
            y0 = int(np.clip(y, 0, H - 1))
            y1 = int(np.clip(y + line_w, y0 + 1, H))
            g[y0:y1, :] = 1.0
            y += pitch
        x = gx % max(1, pitch)
        while x < W:
            x0 = int(np.clip(x, 0, W - 1))
            x1 = int(np.clip(x + line_w, x0 + 1, W))
            g[:, x0:x1] = 1.0
            x += pitch
        mask = (g[..., None] * outside).astype(np.float32)
        base = out[t].astype(np.float32)
        fill = np.full_like(base, float(rng_py.uniform(0.0, 20.0)), dtype=np.float32)
        out_t = base * (1.0 - mask) + fill * mask
        out[t] = np.clip(out_t, 0.0, 255.0).astype(np.uint8)
    return out


def _apply_pstm_mix_event(
    video16: np.ndarray,
    donor_video16: np.ndarray,
    core_soft: np.ndarray,
    rng_py: random.Random,
    strength_bucket: str,
    donor_core_soft_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    T, H, W, _ = video16.shape
    out = np.asarray(video16, dtype=np.uint8).copy()
    donor = np.asarray(donor_video16, dtype=np.uint8)
    if donor.shape[0] != T:
        idx = np.arange(T, dtype=np.int64) % int(max(1, donor.shape[0]))
        donor = donor[idx]
    donor_rs = np.empty_like(out)
    for t in range(T):
        donor_rs[t] = cv2.resize(donor[t], (W, H), interpolation=cv2.INTER_LINEAR)
    sb = str(strength_bucket).lower().strip()
    if sb == "extreme":
        min_len, max_len = 6, 12
        drift_px = int(round(rng_py.uniform(4.0, 12.0)))
    elif sb == "heavy":
        min_len, max_len = 4, 10
        drift_px = int(round(rng_py.uniform(2.0, 8.0)))
    else:
        min_len, max_len = 3, 7
        drift_px = int(round(rng_py.uniform(2.0, 8.0)))
    t0, t1 = _sample_event_window(T=T, rng_py=rng_py, min_len=min_len, max_len=max_len)
    family = rng_py.choice(["rect", "blob", "wedge", "stripe", "cow"])
    for t in range(t0, t1):
        rel = int(t - t0)
        dx = int(round(rng_py.uniform(-1.0, 1.0) * drift_px * rel / max(1, (t1 - t0))))
        dy = int(round(rng_py.uniform(-1.0, 1.0) * drift_px * rel / max(1, (t1 - t0))))
        m2d = _sample_mask_family(H=H, W=W, rng_py=rng_py, family=family, drift_x=dx, drift_y=dy)
        mask = np.clip(m2d[..., None], 0.0, 1.0)
        donor_t = _strong_transform_donor_frame(
            donor_rs[t],
            rng_py=rng_py,
            strength_bucket=sb,
        ).astype(np.float32)
        base_t = out[t].astype(np.float32)
        if donor_core_soft_mask is not None and donor_core_soft_mask.shape[0] > 0:
            md = donor_core_soft_mask[int(t % int(donor_core_soft_mask.shape[0]))].astype(np.float32)
            donor_t = donor_t * (1.0 - md) + base_t * md
        out_hard = base_t * (1.0 - mask) + donor_t * mask
        soft = np.clip(core_soft[t] * 0.98, 0.0, 1.0)
        out_soft = out_hard * (1.0 - soft) + base_t * soft
        out[t] = np.clip(out_soft, 0.0, 255.0).astype(np.uint8)
    return out


def _color_augment_clip(
    video16: np.ndarray,
    rng_py: random.Random,
    rng_np: np.random.Generator,
    cfg: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    info: Dict[str, object] = {
        "enabled": False,
        "brightness": 0.0,
        "contrast": 0.0,
        "saturation": 0.0,
        "hue": 0.0,
        "do_gamma": False,
        "gamma": 1.0,
        "do_blur": False,
        "blur_sigma": 0.0,
        "do_noise": False,
        "noise_std": 0.0,
        "do_jpeg": False,
        "jpeg_q": 95,
    }
    cfg_map = dict(cfg) if isinstance(cfg, dict) else {}
    color_p = float(cfg_map.get("color_p", FOLD1_COLOR_P))
    brightness_max = float(cfg_map.get("brightness", FOLD1_BRIGHTNESS))
    contrast_max = float(cfg_map.get("contrast", FOLD1_CONTRAST))
    saturation_max = float(cfg_map.get("saturation", FOLD1_SATURATION))
    hue_max = float(cfg_map.get("hue", FOLD1_HUE))
    gamma_p = float(cfg_map.get("gamma_p", FOLD1_GAMMA_P))
    gamma_min = float(cfg_map.get("gamma_min", FOLD1_GAMMA_MIN))
    gamma_max = float(cfg_map.get("gamma_max", FOLD1_GAMMA_MAX))
    blur_p = float(cfg_map.get("blur_p", FOLD1_BLUR_P))
    blur_sigma_min = float(cfg_map.get("blur_sigma_min", FOLD1_BLUR_SIGMA_MIN))
    blur_sigma_max = float(cfg_map.get("blur_sigma_max", FOLD1_BLUR_SIGMA_MAX))
    noise_p = float(cfg_map.get("noise_p", FOLD1_NOISE_P))
    noise_std_min = float(cfg_map.get("noise_std_min", FOLD1_NOISE_STD_MIN))
    noise_std_max = float(cfg_map.get("noise_std_max", FOLD1_NOISE_STD_MAX))
    jpeg_p = float(cfg_map.get("jpeg_p", FOLD1_JPEG_P))
    jpeg_q_min = float(cfg_map.get("jpeg_q_min", FOLD1_JPEG_Q_MIN))
    jpeg_q_max = float(cfg_map.get("jpeg_q_max", FOLD1_JPEG_Q_MAX))

    if rng_py.random() > color_p:
        return video16, info
    info["enabled"] = True
    out = video16.astype(np.float32)

    b = float(rng_py.uniform(-brightness_max, brightness_max))
    c = float(rng_py.uniform(-contrast_max, contrast_max))
    s = float(rng_py.uniform(-saturation_max, saturation_max))
    h = float(rng_py.uniform(-hue_max, hue_max))

    do_gamma = (rng_py.random() < gamma_p)
    gamma = float(rng_py.uniform(gamma_min, gamma_max)) if do_gamma else 1.0
    do_blur = (rng_py.random() < blur_p)
    blur_sigma = float(rng_py.uniform(blur_sigma_min, blur_sigma_max)) if do_blur else 0.0
    do_noise = (rng_py.random() < noise_p)
    noise_std = float(rng_py.uniform(noise_std_min, noise_std_max)) if do_noise else 0.0
    do_jpeg = (rng_py.random() < jpeg_p)
    jpeg_q = int(round(rng_py.uniform(jpeg_q_min, jpeg_q_max))) if do_jpeg else 95
    info.update(
        {
            "brightness": float(b),
            "contrast": float(c),
            "saturation": float(s),
            "hue": float(h),
            "do_gamma": bool(do_gamma),
            "gamma": float(gamma),
            "do_blur": bool(do_blur),
            "blur_sigma": float(blur_sigma),
            "do_noise": bool(do_noise),
            "noise_std": float(noise_std),
            "do_jpeg": bool(do_jpeg),
            "jpeg_q": int(jpeg_q),
        }
    )

    mean = out.mean(axis=(1, 2, 3), keepdims=True)
    out = (out - mean) * (1.0 + c) + mean
    out = out + b * 255.0
    out = np.clip(out, 0.0, 255.0)

    if abs(s) > 1e-6 or abs(h) > 1e-6:
        out2 = np.empty_like(out)
        for t in range(out.shape[0]):
            bgr = out[t].astype(np.uint8)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + h * 179.0) % 180.0
            hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + s), 0.0, 255.0)
            out2[t] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        out = out2

    if do_gamma and abs(gamma - 1.0) > 1e-3:
        out = (out / 255.0) ** gamma
        out = np.clip(out * 255.0, 0.0, 255.0)

    if do_blur:
        k = int(round(blur_sigma * 3.0) * 2 + 1)
        k = max(3, k)
        for t in range(out.shape[0]):
            out[t] = cv2.GaussianBlur(out[t], (k, k), blur_sigma)

    if do_noise:
        noise = rng_np.normal(0.0, noise_std, size=out.shape).astype(np.float32)
        out = out + noise
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    if do_jpeg:
        out2 = np.empty_like(out)
        for t in range(out.shape[0]):
            ok, enc = cv2.imencode(".jpg", out[t], [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
            if not ok:
                out2[t] = out[t]
                continue
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            out2[t] = out[t] if dec is None else dec
        out = out2
    return out, info


# =========================
# Core extraction (base)
# =========================
def extract_base_sample(
    ann_path: Path,
    root_dir: Path,
    sid: str,
    dirs: Dict[str, Path],
    stats: dict,
    start_offset_frames: int = 0,
) -> Optional[BaseItem]:
    """
    Create ONE base sample from an annotation:
      - uses clip-first sampling window and fixed 16-frame uniform sampling
      - builds abs trajectory with forward fill
      - saves delta motion and other arrays
      - saves image at start frame
    """
    # Load JSON
    try:
        data = json.loads(ann_path.read_text())
    except Exception:
        stats["skip_bad_json"] += 1
        return None

    tool = data.get("tool", None)
    action = data.get("action", None)
    if tool is None or action is None:
        stats["skip_no_tool_action"] += 1
        return None
    if tool not in TOOL2ID or action not in ACT2ID:
        stats["skip_unknown_tool_or_action"] += 1
        return None
    combo = combo_name(tool, action)
    if combo not in COMBO2ID:
        stats["skip_unknown_combo"] += 1
        return None

    dims = data.get("imageDimensions", None)
    if dims is None or "width" not in dims or "height" not in dims:
        stats["skip_no_dims"] += 1
        return None
    W, H = int(dims["width"]), int(dims["height"])
    # 2026/02/02
    W_ann, H_ann = W, H
    # 2026/02/02

    pts = data.get("points", [])
    if not isinstance(pts, list) or len(pts) == 0:
        stats["skip_empty_points"] += 1
        return None

    # Video path
    video_path = ann_path.parent.parent / "clips" / (ann_path.stem + ".mp4")
    if not video_path.exists():
        stats["skip_missing_video"] += 1
        return None

    # Determine video base/end frame indices in the annotation coordinate system
    aff = data.get("affordance_range", None)
    if aff and "start" in aff and "end" in aff:
        video_base_abs = int(aff["start"])
        abs_end = int(aff["end"])
    else:
        # fallback: assume clip starts at min frameIndex in points
        try:
            frame_inds = [int(p["frameIndex"]) for p in pts if "frameIndex" in p]
            video_base_abs = int(min(frame_inds))
            abs_end = int(max(frame_inds))
        except Exception:
            stats["skip_bad_range"] += 1
            return None
        stats["warn_no_affordance_range"] += 1

    if abs_end < video_base_abs:
        stats["skip_bad_range"] += 1
        return None

    action_start_abs, action_end_abs, action_range_mask = parse_action_range_abs(
        ann_data=data,
        default_start_abs=int(video_base_abs),
        default_end_abs=int(abs_end),
        stats=stats,
    )

    topology = get_topology(tool)
    pt_role = topology_roles(topology)

    # Group points by absolute frameIndex
    frames_abs: Dict[int, List[dict]] = {}
    for p in pts:
        try:
            f = int(p["frameIndex"])
        except Exception:
            continue
        if video_base_abs <= f <= abs_end:
            frames_abs.setdefault(f, []).append(p)

    # ==============================
    # Start frame policy
    # ==============================
    max_contact_in_clip = 0
    for f in range(video_base_abs, abs_end + 1):
        items = frames_abs.get(f, [])
        contacts_vis = [p for p in items if (p.get("type") == "contact" and is_vis(p))]
        max_contact_in_clip = max(max_contact_in_clip, len(contacts_vis))

    # Optional start offset (used by train-only offset-base expansion).
    seg_len = int(abs_end - video_base_abs + 1)
    max_start_offset = int(max(0, seg_len - int(TARGET_FRAMES)))
    start_offset = int(np.clip(int(start_offset_frames), 0, max_start_offset))
    abs_start_target = int(video_base_abs + start_offset)

    # Use clip start directly if configured; otherwise fallback to old search behavior.
    if USE_CLIP_START_AS_ABS_START:
        abs_start = int(abs_start_target)
    else:
        abs_start = None

    tool_pts3_start = None
    search_start = int(abs_start_target)
    for f in range(search_start, abs_end + 1):
        items = frames_abs.get(f, [])
        got = extract_points_and_vis_for_frame(
            items=items,
            W=W,
            H=H,
            topology=topology,
            require_vis=False,
        )
        if got is None:
            continue
        tool_pts3_start = got[0]
        if abs_start is None:
            abs_start = f
        break

    if abs_start is None or tool_pts3_start is None:
        stats["skip_no_valid_start"] += 1
        return None
    if USE_CLIP_START_AS_ABS_START and abs_start != int(abs_start_target):
        abs_start = int(abs_start_target)

    # Contact points are optional for keypoint-detect training.
    # If missing at t=0, fallback to a small tool-centered box.
    start_items = frames_abs.get(abs_start, [])
    contact_start = extract_contact4_and_vis_for_frame(start_items, W, H, require_vis=True)
    if contact_start is None:
        contact_start = extract_contact4_and_vis_for_frame(start_items, W, H, require_vis=False)
    if contact_start is None:
        # fallback: find earliest frame with 4 contacts in clip
        for f in range(search_start, abs_end + 1):
            items = frames_abs.get(f, [])
            contact_start = extract_contact4_and_vis_for_frame(items, W, H, require_vis=False)
            if contact_start is not None:
                break
    if contact_start is None:
        contact_xy4 = fallback_contact_from_tool_points(tool_pts3_start, W, H)
        stats["warn_contact_fallback_to_tool_box"] += 1
    else:
        contact_xy4, _ = contact_start

    T_raw = abs_end - abs_start + 1
    if T_raw < MIN_T:
        stats["skip_too_short"] += 1
        return None
    if (not DISABLE_FILTER_RULES) and ENFORCE_HOOK_DISSECT_MIN_STEPS:
        L_raw = T_raw - 1
        if tool == "hook" and action == "dissect" and L_raw < HOOK_DISSECT_MIN_STEPS:
            stats["skip_hook_dissect_too_short"] += 1
            return None

    # fixed 16-frame clip sampling for ViT-*16f
    sampled_abs = sample_fixed_abs_indices(abs_start, abs_end, TARGET_FRAMES)
    T = int(sampled_abs.shape[0])

    # Build keypoint sequence + per-point visibility.
    motion_abs = np.zeros((T, 3, 2), dtype=np.float32)
    keypoint_vis = np.zeros((T, 3), dtype=np.float32)
    contact_abs = np.zeros((T, 4, 2), dtype=np.float32)
    contact_vis = np.zeros((T, 4), dtype=np.float32)
    prev = tool_pts3_start.astype(np.float32).copy()
    prev_ct = contact_xy4.astype(np.float32).copy()
    missing_frames = 0
    missing_contact_frames = 0
    for t, f in enumerate(sampled_abs.tolist()):
        items = frames_abs.get(int(f), [])

        # tool (3 points)
        got = extract_points_and_vis_for_frame(
            items=items,
            W=W,
            H=H,
            topology=topology,
            require_vis=False,
        )
        if got is None:
            motion_abs[t] = prev
            keypoint_vis[t] = 0.0
            missing_frames += 1
        else:
            pts3, vis3 = got
            motion_abs[t] = pts3.astype(np.float32)
            keypoint_vis[t] = vis3.astype(np.float32)
            prev = pts3.astype(np.float32)

        # contact (4 points)
        got_ct = extract_contact4_and_vis_for_frame(
            items=items,
            W=W,
            H=H,
            require_vis=False,
        )
        if got_ct is None:
            contact_abs[t] = prev_ct
            contact_vis[t] = 0.0
            missing_contact_frames += 1
        else:
            c_xy4, c_vis4 = got_ct
            contact_abs[t] = c_xy4.astype(np.float32)
            contact_vis[t] = c_vis4.astype(np.float32)
            prev_ct = c_xy4.astype(np.float32)

    if missing_frames > 0:
        stats["warn_sampled_frame_missing_tool"] += 1
    if missing_contact_frames > 0:
        stats["warn_sampled_frame_missing_contact"] += 1

    # visibility quality filters
    vis_count_per_point = keypoint_vis.sum(axis=0).astype(np.int64)
    tool_vis_mean = float(keypoint_vis.mean())
    contact_vis_mean = float(contact_vis.mean())
    vis_score = 0.5 * (tool_vis_mean + contact_vis_mean)
    vis_total = int(keypoint_vis.sum())
    if (not DISABLE_FILTER_RULES) and int(MIN_VIS_FRAMES_PER_POINT) > 0 and int(vis_count_per_point.min()) < int(MIN_VIS_FRAMES_PER_POINT):
        stats["skip_low_vis_per_point"] += 1
        return None
    if (not DISABLE_FILTER_RULES) and int(MIN_VIS_TOTAL) > 0 and vis_total < int(MIN_VIS_TOTAL):
        stats["skip_low_vis_total"] += 1
        return None

    # # contacts flatten
    # contact_flat = contact_xy4.reshape(-1).astype(np.float32)  # (8,)
    #
    # # =========================
    # # Geo features (t = 0)
    # # =========================
    #
    # # tool_pts3 = motion_abs[0]  # (3,2) tool points at t=0
    # #
    # # # contact area (pixel^2)
    # # contact_area = _poly_area_xy(contact_xy4)
    # #
    # # # tool triangle area (pixel^2)
    # # tool_area = _tri_area_xy(tool_pts3)
    # #
    # # # center distance (pixel)
    # # contact_center = contact_xy4.mean(axis=0)
    # # tool_center = tool_pts3.mean(axis=0)
    # # center_dist = float(np.linalg.norm(tool_center - contact_center))
    # #
    # # # edge cosine
    # # edge_dir = _longest_edge_dir(contact_xy4)
    #
    # # 2026/01/30
    # tool_pts3 = motion_abs[0]  # (3,2) tool points at t=0
    #
    # # ===== compute GEO in PIXEL space ALWAYS =====
    # contact_xy4_px = contact_xy4.astype(np.float32).copy()
    # tool_pts3_px = tool_pts3.astype(np.float32).copy()
    # if not USE_PIXEL_COORDS:
    #     contact_xy4_px[:, 0] *= float(W)
    #     contact_xy4_px[:, 1] *= float(H)
    #     tool_pts3_px[:, 0] *= float(W)
    #     tool_pts3_px[:, 1] *= float(H)
    #
    # # contact area (pixel^2)
    # contact_area = _poly_area_xy(contact_xy4_px)
    #
    # # tool triangle area (pixel^2)
    # tool_area = _tri_area_xy(tool_pts3_px)
    #
    # # center distance (pixel)
    # contact_center = contact_xy4_px.mean(axis=0)
    # tool_center = tool_pts3_px.mean(axis=0)
    # center_dist = float(np.linalg.norm(tool_center - contact_center))
    #
    # # edge cosine (unitless)
    # edge_dir = _longest_edge_dir(contact_xy4_px)
    # vec_ct = tool_center - contact_center
    # edge_cos = _cos_abs_2d(edge_dir, vec_ct)
    # # 2026/01/30
    #
    # # vec_ct = tool_center - contact_center
    # # edge_cos = _cos_abs_2d(edge_dir, vec_ct)
    #
    # # # delta motion (T,6)
    # # motion_delta = abs_to_delta(motion_abs)  # (T,6)
    # #
    # # # anchor_tips is ABS of start frame
    # # anchor_tip = motion_abs[0].reshape(6).astype(np.float32)
    #
    # # 2026/01/21
    # # step-delta motion (T-1,6)  ✅ NO dummy frame
    # # motion_delta = abs_to_stepdelta(motion_abs)  # (T-1,6)
    # # 2026/01/30+
    # # motion stored in configurable representation
    # motion_repr = encode_motion(motion_abs)  # stepdelta or relative
    #
    # # anchor_tips is ABS of start frame
    # anchor_tip = motion_abs[0].reshape(6).astype(np.float32)
    #
    # # cond_points (ABS) from contact frame (which is now t=0)
    # cond_points = motion_abs[0].astype(np.float32)
    #
    # # Image at start frame: local index within clip mp4
    # local_start_idx = abs_start - video_base_abs
    # frame0 = read_video_frame_at_local_index(video_path, local_start_idx)
    # if frame0 is None:
    #     stats["skip_bad_video_read"] += 1
    #     return None
    #
    # # Save all base files

    # 2026/02/02
    # =========================================================
    # (NEW) Read start frame, remove black bars, and remap coords
    # =========================================================
    local_start_idx = abs_start - video_base_abs
    frame0_full = read_video_frame_at_local_index(video_path, local_start_idx)
    if frame0_full is None:
        stats["skip_bad_video_read"] += 1
        return None

    frameH, frameW = frame0_full.shape[:2]

    # if REMOVE_BLACK_BARS:
    #     bbox = detect_black_bars_bbox(frame0_full)
    # else:
    #     bbox = (0, 0, frameW, frameH)

    if REMOVE_BLACK_BARS:
        bbox = detect_content_bbox(frame0_full)  # ✅ 新的：左右黑边 + 上下 UI
    else:
        bbox = (0, 0, frameW, frameH)

    x0, y0, x1, y1 = bbox
    W_crop, H_crop = int(x1 - x0), int(y1 - y0)

    # remap motion + contacts into cropped coordinate system
    motion_abs_crop = transform_points_to_crop(motion_abs, W_ann, H_ann, frameW, frameH, bbox)
    contact_abs_crop = transform_points_to_crop(contact_abs, W_ann, H_ann, frameW, frameH, bbox)

    # safety: if crop makes points go out-of-range, disable crop for THIS sample
    if REMOVE_BLACK_BARS:
        bbox = detect_content_bbox(frame0_full)  # ✅ 左右黑边 + 上下 UI

        # =========================================================
        # POINT-AWARE CROP (NEW)
        # 如果 bbox 把任何点裁掉了（点出界），就把 bbox 向外扩到能包含所有点为止
        # （不再回退 full-frame）
        # =========================================================
        bbox = point_aware_expand_bbox(
            bbox_xyxy=bbox,
            motion_abs=motion_abs,  # still in annotation coord space
            contact_xy4=contact_abs,  # still in annotation coord space
            ann_W=W_ann,
            ann_H=H_ann,
            frame_W=frameW,
            frame_H=frameH,
        )
    else:
        bbox = (0, 0, frameW, frameH)

    x0, y0, x1, y1 = bbox
    W_crop, H_crop = int(x1 - x0), int(y1 - y0)

    # remap motion + contacts into cropped coordinate system
    motion_abs_crop = transform_points_to_crop(motion_abs, W_ann, H_ann, frameW, frameH, bbox)
    contact_abs_crop = transform_points_to_crop(contact_abs, W_ann, H_ann, frameW, frameH, bbox)

    # NOTE:
    #   这里不再做 “点出界 -> bbox=full-frame” 的回退逻辑。
    #   经过 point_aware_expand_bbox 后，bbox 已经保证尽量覆盖所有点；
    #   后面仍会对微小浮点溢出做 clip。

    # crop the image
    frame0 = crop_frame(frame0_full, bbox)

    # overwrite dims to cropped dims from here on
    W, H = W_crop, H_crop
    motion_abs = motion_abs_crop.astype(np.float32)
    contact_abs = contact_abs_crop.astype(np.float32)
    contact_xy4 = canonical_order_xy(contact_abs[0]).astype(np.float32)

    # clamp tiny numerical spillover
    if USE_PIXEL_COORDS:
        motion_abs[..., 0] = np.clip(motion_abs[..., 0], 0.0, float(W))
        motion_abs[..., 1] = np.clip(motion_abs[..., 1], 0.0, float(H))
        contact_abs[..., 0] = np.clip(contact_abs[..., 0], 0.0, float(W))
        contact_abs[..., 1] = np.clip(contact_abs[..., 1], 0.0, float(H))
    else:
        motion_abs = np.clip(motion_abs, 0.0, 1.0)
        contact_abs = np.clip(contact_abs, 0.0, 1.0)

    contact_xy4 = canonical_order_xy(contact_abs[0]).astype(np.float32)

    # contacts flatten (NOW in cropped coord space)
    contact_flat = contact_xy4.reshape(-1).astype(np.float32)  # (8,)

    # =========================
    # Geo features (t = 0), optional export.
    # =========================
    contact_area = None
    tool_area = None
    center_dist = None
    edge_cos = None
    if EXPORT_GEO_FEATURES:
        tool_pts3 = motion_abs[0]  # (3,2) tool points at t=0

        contact_xy4_px = contact_xy4.astype(np.float32).copy()
        tool_pts3_px = tool_pts3.astype(np.float32).copy()
        if not USE_PIXEL_COORDS:
            contact_xy4_px[:, 0] *= float(W)
            contact_xy4_px[:, 1] *= float(H)
            tool_pts3_px[:, 0] *= float(W)
            tool_pts3_px[:, 1] *= float(H)

        contact_area = _poly_area_xy(contact_xy4_px)      # pixel^2
        tool_area = _tri_area_xy(tool_pts3_px)            # pixel^2
        contact_center = contact_xy4_px.mean(axis=0)
        tool_center = tool_pts3_px.mean(axis=0)
        center_dist = float(np.linalg.norm(tool_center - contact_center))  # pixel
        edge_dir = _longest_edge_dir(contact_xy4_px)
        vec_ct = tool_center - contact_center
        edge_cos = _cos_abs_2d(edge_dir, vec_ct)          # unitless

    # =========================
    # Motion representation
    # =========================
    keypoint_seq_raw = motion_abs.astype(np.float32)      # (16,3,2) in cropped coord space
    keypoint_vis_raw = keypoint_vis.astype(np.float32)    # (16,3)
    contact_seq_raw = contact_abs.astype(np.float32)      # (16,4,2)
    contact_vis_raw = contact_vis.astype(np.float32)      # (16,4)

    if INTERPOLATE_VIS0_TO_FULL_SUPERVISION:
        keypoint_seq, keypoint_vis, tool_interp_mask = interpolate_vis0_to_full_supervision(
            keypoint_seq_raw,
            keypoint_vis_raw,
        )
        contact_seq, contact_vis_seq, ct_interp_mask = interpolate_vis0_to_full_supervision(
            contact_seq_raw,
            contact_vis_raw,
        )
    else:
        keypoint_seq, keypoint_vis = keypoint_seq_raw, keypoint_vis_raw
        contact_seq, contact_vis_seq = contact_seq_raw, contact_vis_raw
        tool_interp_mask = np.zeros_like(keypoint_vis_raw, dtype=np.float32)
        ct_interp_mask = np.zeros_like(contact_vis_raw, dtype=np.float32)
    combo_id_val = np.array(COMBO2ID[combo], dtype=np.int64)
    if ENABLE_PSEUDO_SUPERVISION:
        tool_pseudo_xy = keypoint_seq.copy()
        ct_pseudo_xy = contact_seq.copy()
        tool_pseudo_w = (tool_interp_mask * float(PSEUDO_WEIGHT)).astype(np.float32)
        ct_pseudo_w = (ct_interp_mask * float(PSEUDO_WEIGHT)).astype(np.float32)
    else:
        tool_pseudo_xy = keypoint_seq.copy()
        ct_pseudo_xy = contact_seq.copy()
        tool_pseudo_w = np.zeros_like(keypoint_vis, dtype=np.float32)
        ct_pseudo_w = np.zeros_like(contact_vis_seq, dtype=np.float32)
    sampled_abs = sampled_abs.astype(np.int64)        # (16,)

    video16_clip = None
    if EXPORT_VIDEO16_NPY:
        video16_clip = build_video16_npy_from_sampled_frames(
            video_path=video_path,
            sampled_abs=sampled_abs,
            video_base_abs=int(video_base_abs),
            bbox_xyxy=(int(x0), int(y0), int(x1), int(y1)),
            start_frame_full=frame0_full,
            stats=stats,
        )

    motion_repr = encode_motion(keypoint_seq)  # stepdelta or relative

    # anchor_tips is ABS of start frame
    anchor_tip = keypoint_seq[0].reshape(6).astype(np.float32)

    # cond_points (ABS) from contact frame (t=0)
    cond_points = keypoint_seq[0].astype(np.float32)
    # 2026/02/02

    paths_to_cleanup = [
        dirs["MOTION_DIR"] / f"{sid}.npy",
        dirs["ANCHOR_TIP_DIR"] / f"{sid}.npy",
        dirs["COND_POINTS_DIR"] / f"{sid}.npy",
        dirs["PT_ROLE_DIR"] / f"{sid}.npy",
        dirs["TOOL_ID_DIR"] / f"{sid}.npy",
        dirs["ACT_ID_DIR"] / f"{sid}.npy",
        dirs["CONTACT_DIR"] / f"{sid}.npy",
        dirs["IMAGE_DIR"] / f"{sid}.jpg",
        dirs["TEXT_DIR"] / f"{sid}.txt",
        dirs["KEYPOINT_DIR"] / f"{sid}.npy",
        dirs["KEYPOINT_VIS_DIR"] / f"{sid}.npy",
        dirs["CONTACT_VIS_DIR"] / f"{sid}.npy",
        dirs["FRAME_INDEX_DIR"] / f"{sid}.npy",
        dirs["ACTION_RANGE_ABS_DIR"] / f"{sid}.npy",
        dirs["ACTION_RANGE_MASK_DIR"] / f"{sid}.npy",
        dirs["GT_TOOL_XY_DIR"] / f"{sid}.npy",
        dirs["GT_TOOL_VIS_DIR"] / f"{sid}.npy",
        dirs["GT_CT_XY_DIR"] / f"{sid}.npy",
        dirs["GT_CT_VIS_DIR"] / f"{sid}.npy",
        dirs["COMBO_ID_DIR"] / f"{sid}.npy",
        dirs["GT_TOOL_XY_PSEUDO_DIR"] / f"{sid}.npy",
        dirs["GT_TOOL_PSEUDO_W_DIR"] / f"{sid}.npy",
        dirs["GT_CT_XY_PSEUDO_DIR"] / f"{sid}.npy",
        dirs["GT_CT_PSEUDO_W_DIR"] / f"{sid}.npy",
    ]
    if EXPORT_GEO_FEATURES:
        paths_to_cleanup.extend([
            dirs["CONTACT_AREA_DIR"] / f"{sid}.npy",
            dirs["TOOL_AREA_DIR"] / f"{sid}.npy",
            dirs["CENTER_DIST_DIR"] / f"{sid}.npy",
            dirs["EDGE_COS_DIR"] / f"{sid}.npy",
        ])
    if EXPORT_VIDEO16_NPY:
        paths_to_cleanup.append(dirs["VIDEO16_DIR"] / f"{sid}.npy")
    if ENABLE_MASK_IMAGE:
        paths_to_cleanup.append(dirs["MASK_DIR"] / f"{sid}.jpg")

    try:
        # npy_save(dirs["MOTION_DIR"] / f"{sid}.npy", motion_delta, dtype=np.float32)
        npy_save(dirs["MOTION_DIR"] / f"{sid}.npy", motion_repr, dtype=np.float32)
        npy_save(dirs["ANCHOR_TIP_DIR"] / f"{sid}.npy", anchor_tip, dtype=np.float32)
        npy_save(dirs["COND_POINTS_DIR"] / f"{sid}.npy", cond_points, dtype=np.float32)
        npy_save(dirs["PT_ROLE_DIR"] / f"{sid}.npy", pt_role, dtype=np.int64)
        npy_save(dirs["TOOL_ID_DIR"] / f"{sid}.npy", np.array(TOOL2ID[tool], dtype=np.int64))
        npy_save(dirs["ACT_ID_DIR"] / f"{sid}.npy", np.array(ACT2ID[action], dtype=np.int64))
        npy_save(dirs["CONTACT_DIR"] / f"{sid}.npy", contact_flat, dtype=np.float32)
        npy_save(dirs["KEYPOINT_DIR"] / f"{sid}.npy", keypoint_seq_raw, dtype=np.float32)
        npy_save(dirs["KEYPOINT_VIS_DIR"] / f"{sid}.npy", keypoint_vis_raw, dtype=np.float32)
        npy_save(dirs["CONTACT_VIS_DIR"] / f"{sid}.npy", contact_vis_raw, dtype=np.float32)
        npy_save(dirs["FRAME_INDEX_DIR"] / f"{sid}.npy", sampled_abs, dtype=np.int64)
        npy_save(
            dirs["ACTION_RANGE_ABS_DIR"] / f"{sid}.npy",
            np.array([int(action_start_abs), int(action_end_abs)], dtype=np.int64),
            dtype=np.int64,
        )
        npy_save(
            dirs["ACTION_RANGE_MASK_DIR"] / f"{sid}.npy",
            np.array(float(action_range_mask), dtype=np.float32),
            dtype=np.float32,
        )
        npy_save(dirs["GT_TOOL_XY_DIR"] / f"{sid}.npy", keypoint_seq, dtype=np.float32)
        npy_save(dirs["GT_TOOL_VIS_DIR"] / f"{sid}.npy", keypoint_vis, dtype=np.float32)
        npy_save(dirs["GT_CT_XY_DIR"] / f"{sid}.npy", contact_seq, dtype=np.float32)
        npy_save(dirs["GT_CT_VIS_DIR"] / f"{sid}.npy", contact_vis_seq, dtype=np.float32)
        npy_save(dirs["COMBO_ID_DIR"] / f"{sid}.npy", combo_id_val, dtype=np.int64)
        npy_save(dirs["GT_TOOL_XY_PSEUDO_DIR"] / f"{sid}.npy", tool_pseudo_xy, dtype=np.float32)
        npy_save(dirs["GT_TOOL_PSEUDO_W_DIR"] / f"{sid}.npy", tool_pseudo_w, dtype=np.float32)
        npy_save(dirs["GT_CT_XY_PSEUDO_DIR"] / f"{sid}.npy", ct_pseudo_xy, dtype=np.float32)
        npy_save(dirs["GT_CT_PSEUDO_W_DIR"] / f"{sid}.npy", ct_pseudo_w, dtype=np.float32)
        if EXPORT_VIDEO16_NPY and video16_clip is not None:
            npy_save(dirs["VIDEO16_DIR"] / f"{sid}.npy", video16_clip, dtype=np.uint8)
        if EXPORT_GEO_FEATURES:
            npy_save(
                dirs["CONTACT_AREA_DIR"] / f"{sid}.npy",
                np.array([contact_area], dtype=np.float32),
            )
            npy_save(
                dirs["TOOL_AREA_DIR"] / f"{sid}.npy",
                np.array([tool_area], dtype=np.float32),
            )
            npy_save(
                dirs["CENTER_DIST_DIR"] / f"{sid}.npy",
                np.array([center_dist], dtype=np.float32),
            )
            npy_save(
                dirs["EDGE_COS_DIR"] / f"{sid}.npy",
                np.array([edge_cos], dtype=np.float32),
            )

        cv2.imwrite(str(dirs["IMAGE_DIR"] / f"{sid}.jpg"), frame0)
        if ENABLE_MASK_IMAGE:
            mask_bgr = build_pose_mask_image(anchor_tip, contact_xy4, W, H, radius_px=MASK_RADIUS_PX)
            cv2.imwrite(str(dirs["MASK_DIR"] / f"{sid}.jpg"), mask_bgr)

        prompt = make_prompt(tool, action, contact_xy4)
        (dirs["TEXT_DIR"] / f"{sid}.txt").write_text(prompt + "\n", encoding="utf-8")
    except Exception:
        cleanup_paths(paths_to_cleanup)
        stats["skip_write_error"] += 1
        return None

    # Build BaseItem
    case_name = ann_path.parent.parent.name
    ann_relpath = str(ann_path.relative_to(root_dir))

    item = BaseItem(
        sid=sid,
        case=case_name,
        ann_relpath=ann_relpath,
        tool=tool,
        action=action,
        combo=combo,
        tool_id=int(TOOL2ID[tool]),
        act_id=int(ACT2ID[action]),
        combo_id=int(COMBO2ID[combo]),
        topology=topology,
        pt_role=pt_role.tolist(),
        W=W,
        H=H,
        video_base_abs=int(video_base_abs),
        abs_start=int(abs_start),
        abs_end=int(abs_end),
        num_frames=int(motion_abs.shape[0]),  # fixed 16 frames
        action_start_abs=int(action_start_abs),
        action_end_abs=int(action_end_abs),
        action_range_mask=int(action_range_mask),
        max_contact_visible_in_window=int(max_contact_in_clip),
        vis_count_per_point=vis_count_per_point.tolist(),
        tool_vis_mean=float(tool_vis_mean),
        contact_vis_mean=float(contact_vis_mean),
        vis_score=float(vis_score),
        crop_bbox_xyxy=[int(x0), int(y0), int(x1), int(y1)],
        split="unset",
        source_sid=None,
        aug_type=None,
        aug_params=None,
    )
    return item


# =========================
# Split helpers (diverse sampling)
# =========================
def group_by_case(items: List[BaseItem]) -> Dict[str, List[BaseItem]]:
    d: Dict[str, List[BaseItem]] = {}
    for it in items:
        d.setdefault(it.case, []).append(it)
    return d


def diverse_sample(items: List[BaseItem], k: int, rng: random.Random) -> List[BaseItem]:
    """
    Round-robin pick across cases for diversity.
    Deterministic given rng state.
    """
    if k <= 0 or len(items) == 0:
        return []
    if k >= len(items):
        return list(items)

    by_case = group_by_case(items)
    cases = list(by_case.keys())
    rng.shuffle(cases)
    for c in cases:
        rng.shuffle(by_case[c])

    out: List[BaseItem] = []
    case_idx = 0
    while len(out) < k and len(cases) > 0:
        c = cases[case_idx % len(cases)]
        if len(by_case[c]) > 0:
            out.append(by_case[c].pop(0))
        # prune empty cases
        cases = [cc for cc in cases if len(by_case[cc]) > 0]
        if len(cases) == 0:
            break
        case_idx += 1

    # If still not enough, fill randomly from remaining
    if len(out) < k:
        remaining = []
        for v in by_case.values():
            remaining.extend(v)
        rng.shuffle(remaining)
        out.extend(remaining[: (k - len(out))])

    return out[:k]


def _item_vis_score(it: BaseItem) -> float:
    if it.vis_score is not None:
        return float(it.vis_score)
    tool_mean = 0.0
    if it.tool_vis_mean is not None:
        tool_mean = float(it.tool_vis_mean)
    elif it.vis_count_per_point is not None and len(it.vis_count_per_point) > 0:
        denom = float(max(TARGET_FRAMES * len(it.vis_count_per_point), 1))
        tool_mean = float(sum(it.vis_count_per_point)) / denom
    ct_mean = float(it.contact_vis_mean) if it.contact_vis_mean is not None else 0.0
    return 0.5 * (tool_mean + ct_mean)


def _vis_sorted(items: List[BaseItem]) -> List[BaseItem]:
    return sorted(items, key=lambda it: (_item_vis_score(it), -int(it.sid)), reverse=True)


def vis_diverse_sample(items: List[BaseItem], k: int, rng: random.Random) -> List[BaseItem]:
    """
    Case-diverse sampling that prioritizes higher visibility scores inside each case.
    """
    if k <= 0 or len(items) == 0:
        return []
    if k >= len(items):
        return list(items)

    by_case = group_by_case(items)
    cases = list(by_case.keys())
    rng.shuffle(cases)
    for c in cases:
        by_case[c] = _vis_sorted(by_case[c])

    out: List[BaseItem] = []
    case_idx = 0
    while len(out) < k and len(cases) > 0:
        c = cases[case_idx % len(cases)]
        if len(by_case[c]) > 0:
            out.append(by_case[c].pop(0))
        cases = [cc for cc in cases if len(by_case[cc]) > 0]
        if len(cases) == 0:
            break
        case_idx += 1

    if len(out) < k:
        remaining = []
        for v in by_case.values():
            remaining.extend(v)
        remaining = _vis_sorted(remaining)
        out.extend(remaining[: (k - len(out))])

    return out[:k]


def _quick_read_combo_from_ann(ann_path: Path) -> Optional[Tuple[str, str, str]]:
    """
    Lightweight annotation parse for pre-selection.
    Returns (tool, action, combo) or None if invalid/unreadable.
    """
    try:
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    tool = str(ann.get("tool", "")).strip()
    action = str(ann.get("action", "")).strip()
    combo = combo_name(tool, action)
    if combo not in COMBO2ID:
        return None
    return tool, action, combo


def _sample_ann_paths_balanced_by_case(
    case2paths: Dict[str, List[Path]],
    cap_n: int,
    seed: int,
) -> List[Path]:
    """
    Deterministic, case-balanced path sampling.
    """
    if cap_n <= 0:
        return []
    all_paths = []
    for _, paths in case2paths.items():
        all_paths.extend(paths)
    if cap_n >= len(all_paths):
        return sorted(all_paths, key=lambda p: str(p))

    pools: Dict[str, List[Path]] = {
        c: sorted(v, key=lambda p: str(p))
        for c, v in case2paths.items()
    }
    cases = sorted(pools.keys())
    rng = random.Random(int(seed))
    rng.shuffle(cases)

    out: List[Path] = []
    idx = 0
    active = list(cases)
    while len(out) < cap_n and len(active) > 0:
        c = active[idx % len(active)]
        lst = pools[c]
        if len(lst) > 0:
            out.append(lst.pop(0))
        active = [cc for cc in active if len(pools[cc]) > 0]
        if len(active) == 0:
            break
        idx += 1
    return out[:cap_n]


def _should_use_vis_priority(combo_count: int) -> bool:
    return bool(VIS_PRIORITY_FOR_BIG) and int(combo_count) >= int(VIS_PRIORITY_MIN_COMBO_COUNT)

# 2026/02/02
def _even_pick_indices(cands: List[int], k: int) -> List[int]:
    """Pick k indices from sorted candidate list with roughly even spacing."""
    if k <= 0:
        return []
    if k >= len(cands):
        return list(cands)
    idxs = np.linspace(0, len(cands) - 1, k).round().astype(int).tolist()
    out = []
    used = set()
    for ii in idxs:
        ii = max(0, min(len(cands) - 1, int(ii)))
        if cands[ii] not in used:
            out.append(cands[ii])
            used.add(cands[ii])
    # fill if duplicates reduced count
    j = 0
    while len(out) < k and j < len(cands):
        if cands[j] not in used:
            out.append(cands[j])
            used.add(cands[j])
        j += 1
    return out[:k]


def assign_splits_by_selected_folders(
    meta: List[BaseItem],
    selected_cases: List[str],
    val_cases: List[str],
    rng: random.Random,
    train_k_per_combo: Optional[int] = None,
    val_k_per_combo: Optional[int] = None,
    allow_val_backfill_from_train: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Folder-disjoint split:
      - export only selected_cases
      - val clips are those whose case is in val_cases
      - all other selected clips are train
      - optional per-combo sampling caps can be applied independently
        inside train/val pools
      - optional val backfill can move only deficit samples (per combo)
        from train-case pool to val, without moving full folders
      - test is empty
    """
    selected_set = set(selected_cases)
    val_set = set(val_cases)
    if len(selected_set) == 0:
        raise ValueError("selected_cases is empty in folder split mode.")
    if not val_set.issubset(selected_set):
        bad = sorted(val_set - selected_set)
        raise ValueError(f"VAL_CASE_FOLDERS must be subset of SELECTED_CASE_FOLDERS, bad={bad}")

    train_case_set = selected_set - val_set
    if len(val_set) == 0:
        raise ValueError("Folder split expects at least one val folder in VAL_CASE_FOLDERS.")
    if len(train_case_set) == 0:
        raise ValueError(
            "Folder split expects at least one train folder in SELECTED_CASE_FOLDERS "
            f"after removing val folders; got selected={len(selected_set)} val={len(val_set)}"
        )

    for it in meta:
        it.split = "drop"

    selected_items = [it for it in meta if it.case in selected_set]
    train_pool = [it for it in selected_items if it.case not in val_set]
    val_pool = [it for it in selected_items if it.case in val_set]

    # Backward-compatible behavior: use all selected data if no caps are provided.
    if train_k_per_combo is None and val_k_per_combo is None:
        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []
        for it in selected_items:
            if it.case in val_set:
                it.split = "val"
                val_ids.append(it.sid)
            else:
                it.split = "train"
                train_ids.append(it.sid)
        return sorted(train_ids), sorted(val_ids), test_ids

    combo_order = sorted(COMBO2ID.keys(), key=lambda c: int(COMBO2ID[c]))
    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for combo in combo_order:
        train_combo_pool = [it for it in train_pool if f"{it.tool}|{it.action}" == combo]
        val_combo_pool = [it for it in val_pool if f"{it.tool}|{it.action}" == combo]
        combo_total = len(train_combo_pool) + len(val_combo_pool)
        use_vis_priority = bool(VIS_PRIORITY_FOR_BIG and combo_total >= int(VIS_PRIORITY_MIN_COMBO_COUNT))

        if val_k_per_combo is None:
            val_target = len(val_combo_pool)
            val_keep = list(val_combo_pool)
        else:
            k_val_req = int(val_k_per_combo)
            if combo_total > 1:
                # Keep at least one sample for train whenever possible.
                k_val_req = min(k_val_req, combo_total - 1)
            else:
                k_val_req = min(k_val_req, combo_total)
            val_target = max(0, int(k_val_req))
            k_val = min(val_target, len(val_combo_pool))
            if use_vis_priority:
                val_keep = vis_diverse_sample(val_combo_pool, k_val, rng)
            else:
                val_keep = diverse_sample(val_combo_pool, k_val, rng)

        moved_to_val_sid: Set[str] = set()
        if allow_val_backfill_from_train and (len(val_keep) < int(val_target)):
            need = int(val_target) - len(val_keep)
            if need > 0 and len(train_combo_pool) > 0:
                if use_vis_priority:
                    extra_val = vis_diverse_sample(train_combo_pool, need, rng)
                else:
                    extra_val = diverse_sample(train_combo_pool, need, rng)
                val_keep = list(val_keep) + list(extra_val)
                moved_to_val_sid = {str(it.sid) for it in extra_val}

        train_combo_pool_eff = [it for it in train_combo_pool if str(it.sid) not in moved_to_val_sid]
        if train_k_per_combo is None:
            train_keep = list(train_combo_pool_eff)
        else:
            k_train = min(int(train_k_per_combo), len(train_combo_pool_eff))
            if use_vis_priority:
                train_keep = vis_diverse_sample(train_combo_pool_eff, k_train, rng)
            else:
                train_keep = diverse_sample(train_combo_pool_eff, k_train, rng)

        for it in val_keep:
            it.split = "val"
            val_ids.append(it.sid)
        for it in train_keep:
            it.split = "train"
            train_ids.append(it.sid)

    return sorted(train_ids), sorted(val_ids), test_ids


def _resolve_split_targets(n: int) -> Tuple[int, int, bool]:
    """
    Return (val_k, train_k, is_big_combo) for one combo with n samples.
    If *_K_PER_COMBO is set, it overrides big/small default policy.
    """
    is_big = (n >= 100)

    if VAL_K_PER_COMBO is not None:
        val_k = min(int(VAL_K_PER_COMBO), n)
    else:
        val_k = min(VAL_K_BIG if is_big else VAL_K_SMALL, n)

    if n > 1:
        val_k = min(val_k, n - 1)

    nonval_n = max(n - val_k, 0)
    if TRAIN_K_PER_COMBO is not None:
        train_k = min(int(TRAIN_K_PER_COMBO), nonval_n)
    else:
        train_k = min(TRAIN_K_BIG, nonval_n) if is_big else nonval_n

    return val_k, train_k, is_big


def assign_splits_neighbor(meta: List[BaseItem]) -> Tuple[List[str], List[str], List[str]]:
    """
    Neighbor-aware split inside EACH tool|action combo:
      - sort by sid (which follows folder scan order)
      - choose val indices spread across the combo sequence
      - enforce each val has at least one immediate neighbor in TRAIN (not TEST)
      - keep original counts:
          if combo_count >=100: val=VAL_K_BIG, train=TRAIN_K_BIG, test=rest
          else:                val=VAL_K_SMALL, train=rest, test=empty
    """
    combo2items: Dict[str, List[BaseItem]] = {}
    for it in meta:
        combo2items.setdefault(f"{it.tool}|{it.action}", []).append(it)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for combo, items in sorted(combo2items.items(), key=lambda kv: kv[0]):
        items_sorted = sorted(items, key=lambda x: x.sid)
        n = len(items_sorted)
        if n == 0:
            continue

        val_k, train_k, is_big = _resolve_split_targets(n)

        # --- pick val indices: spread, but never choose index 0 (so it has a left neighbor) ---
        # if val_k <= 0:
        #     val_idx_set = set()
        # else:
        #     raw = np.linspace(1, n - 1, val_k).round().astype(int).tolist()  # start at 1
        #     val_idx_set = set()
        #     for i in raw:
        #         i = max(1, min(n - 1, int(i)))
        #         val_idx_set.add(i)
        #
        #     # fill if rounding caused duplicates
        #     i = 1
        #     while len(val_idx_set) < val_k and i < n:
        #         if i not in val_idx_set:
        #             val_idx_set.add(i)
        #         i += 1
        # 2026/02/02
        # --- pick val indices by fixed stride (neighbor-aware) ---
        # Every (VAL_EVERY_K + 1)-th sample becomes VAL.
        # Example: VAL_EVERY_K = 5  -> indices: 5, 11, 17, ...
        if val_k <= 0:
            val_idx_set = set()
        else:
            step = int(VAL_EVERY_K) + 1
            raw = list(range(step - 1, n, step))  # e.g. 5, 11, 17, ...

            # ensure valid range [1, n-1] (never choose index 0)
            raw = [max(1, min(n - 1, i)) for i in raw]

            # cap to desired number of val samples (stop once val_k is reached)
            raw = raw[:val_k]
            val_idx_set = set(raw)

            # if still not enough (short sequence), fill from remaining indices
            i = 1
            while len(val_idx_set) < val_k and i < n:
                if i not in val_idx_set:
                    val_idx_set.add(i)
                i += 1
            # 2026/02/02

            # avoid consecutive val as much as possible (best-effort)
            val_sorted = sorted(val_idx_set)
            fixed = []
            used = set()
            for v in val_sorted:
                vv = v
                if (vv - 1) in used:
                    # shift right if possible
                    if vv + 1 < n and (vv + 1) not in used:
                        vv = vv + 1
                if vv not in used and vv < n:
                    fixed.append(vv)
                    used.add(vv)
            val_idx_set = set(fixed[:val_k])

        # --- determine train/test counts ---
        idx_all = list(range(n))
        idx_val = sorted(list(val_idx_set))
        idx_nonval = [i for i in idx_all if i not in val_idx_set]

        train_k = min(train_k, len(idx_nonval))

        # --- enforce neighbor: for each val index, force at least one neighbor into TRAIN ---
        must_train = set()
        for v in idx_val:
            left = v - 1
            right = v + 1
            if left >= 0 and left not in val_idx_set:
                must_train.add(left)
            elif right < n and right not in val_idx_set:
                must_train.add(right)

        # train selection
        train_set = set(must_train)
        remaining = [i for i in idx_nonval if i not in train_set]

        need = train_k - len(train_set)
        if need > 0:
            if _should_use_vis_priority(n):
                remaining_items = [items_sorted[i] for i in remaining]
                combo_seed = int(SEED) + sum(ord(ch) for ch in combo)
                picked_items = vis_diverse_sample(remaining_items, need, random.Random(combo_seed))
                picked_ids = set(it.sid for it in picked_items)
                extra = [i for i in remaining if items_sorted[i].sid in picked_ids]
                if len(extra) > need:
                    extra = extra[:need]
            else:
                extra = _even_pick_indices(remaining, need)
            train_set.update(extra)

        # Default behavior keeps small-combo leftovers in TRAIN unless a fixed train cap is requested.
        if is_big or (TRAIN_K_PER_COMBO is not None):
            test_set = set(idx_nonval) - set(train_set)
        else:
            test_set = set()

        # assign splits
        for i, it in enumerate(items_sorted):
            if i in val_idx_set:
                it.split = "val"
                val_ids.append(it.sid)
            elif i in train_set:
                it.split = "train"
                train_ids.append(it.sid)
            else:
                it.split = "test"
                test_ids.append(it.sid)

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)
# 2026/02/02

def assign_splits(meta: List[BaseItem], rng: random.Random) -> Tuple[List[str], List[str], List[str]]:
    """
    Assign split field for base items and return (train_ids, val_ids, test_ids) for BASE items only.
    Augmented items will be appended later to train_ids.
    """
    combo2items: Dict[str, List[BaseItem]] = {}
    for it in meta:
        combo2items.setdefault(f"{it.tool}|{it.action}", []).append(it)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for combo, items in sorted(combo2items.items(), key=lambda kv: kv[0]):
        n = len(items)
        items_copy = list(items)
        val_k, train_k, is_big = _resolve_split_targets(n)
        use_vis_priority = _should_use_vis_priority(n)

        if use_vis_priority:
            val_sel = vis_diverse_sample(items_copy, val_k, rng)
        else:
            val_sel = diverse_sample(items_copy, val_k, rng)
        val_set = set(x.sid for x in val_sel)

        remaining = [it for it in items_copy if it.sid not in val_set]
        train_k = min(train_k, len(remaining))
        if use_vis_priority:
            train_sel = vis_diverse_sample(remaining, train_k, rng)
        else:
            train_sel = diverse_sample(remaining, train_k, rng)
        train_set = set(x.sid for x in train_sel)

        if is_big or (TRAIN_K_PER_COMBO is not None):
            test_sel = [it for it in remaining if it.sid not in train_set]
        else:
            # Keep small-combo leftover samples in TRAIN in default mode.
            extra_train = [it for it in remaining if it.sid not in train_set]
            train_sel = list(train_sel) + extra_train
            test_sel = []

        for it in train_sel:
            it.split = "train"
            train_ids.append(it.sid)
        for it in val_sel:
            it.split = "val"
            val_ids.append(it.sid)
        for it in test_sel:
            it.split = "test"
            test_ids.append(it.sid)

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


# =========================
# Augmentation generation
# =========================
def load_base_arrays(dirs: Dict[str, Path], sid: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load (delta_flat, anchor_tip_flat, contacts_xy4, image_bgr)
    """
    motion_repr = np.load(dirs["MOTION_DIR"] / f"{sid}.npy")
    # delta_flat = np.load(dirs["MOTION_DIR"] / f"{sid}.npy")  # (T,6) delta
    anchor_tip = np.load(dirs["ANCHOR_TIP_DIR"] / f"{sid}.npy")  # (6,) abs
    contact_flat = np.load(dirs["CONTACT_DIR"] / f"{sid}.npy")  # (8,) abs
    contact_xy4 = contact_flat.reshape(4, 2).astype(np.float32)
    img = cv2.imread(str(dirs["IMAGE_DIR"] / f"{sid}.jpg"), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read base image: {sid}")
    # return delta_flat.astype(np.float32), anchor_tip.astype(np.float32), contact_xy4, img
    return motion_repr.astype(np.float32), anchor_tip.astype(np.float32), contact_xy4, img


def load_base_clip_arrays(dirs: Dict[str, Path], sid: str) -> Dict[str, np.ndarray]:
    """
    Load synchronized arrays for clip-level post augmentation.
    """
    out = {
        "motion_repr": np.load(dirs["MOTION_DIR"] / f"{sid}.npy").astype(np.float32),
        "anchor_tip": np.load(dirs["ANCHOR_TIP_DIR"] / f"{sid}.npy").astype(np.float32),
        "contact_xy4": np.load(dirs["CONTACT_DIR"] / f"{sid}.npy").reshape(4, 2).astype(np.float32),
        "image_bgr": cv2.imread(str(dirs["IMAGE_DIR"] / f"{sid}.jpg"), cv2.IMREAD_COLOR),
        "gt_tool_xy": np.load(dirs["GT_TOOL_XY_DIR"] / f"{sid}.npy").astype(np.float32),
        "gt_tool_vis": np.load(dirs["GT_TOOL_VIS_DIR"] / f"{sid}.npy").astype(np.float32),
        "gt_ct_xy": np.load(dirs["GT_CT_XY_DIR"] / f"{sid}.npy").astype(np.float32),
        "gt_ct_vis": np.load(dirs["GT_CT_VIS_DIR"] / f"{sid}.npy").astype(np.float32),
        "sampled_abs": np.load(dirs["FRAME_INDEX_DIR"] / f"{sid}.npy").astype(np.int64),
    }
    if EXPORT_VIDEO16_NPY:
        out["video16"] = np.load(dirs["VIDEO16_DIR"] / f"{sid}.npy").astype(np.uint8)
    else:
        out["video16"] = None
    if out["image_bgr"] is None:
        raise RuntimeError(f"Cannot read base image: {sid}")
    if EXPORT_VIDEO16_NPY and out["video16"] is None:
        raise RuntimeError(f"Cannot read base video16: {sid}")
    return out


def save_aug_sample(
    aug_sid: str,
    base_sid: str,
    base_item: BaseItem,
    dirs: Dict[str, Path],
    motion_delta_aug: np.ndarray,
    anchor_tip_aug: np.ndarray,
    cond_points_aug: np.ndarray,
    contact_xy4_aug: np.ndarray,
    image_aug_bgr: np.ndarray,
    prompt: str,
    gt_tool_xy_aug: Optional[np.ndarray] = None,
    gt_tool_vis_aug: Optional[np.ndarray] = None,
    gt_ct_xy_aug: Optional[np.ndarray] = None,
    gt_ct_vis_aug: Optional[np.ndarray] = None,
    sampled_abs_aug: Optional[np.ndarray] = None,
    video16_aug: Optional[np.ndarray] = None,
    keypoint_raw_aug: Optional[np.ndarray] = None,
    contact_vis_raw_aug: Optional[np.ndarray] = None,
):
    # Keep training-contract files complete for augmented samples.
    # For current kpt_combo pipeline, augmented items must have the same required
    # files as base items (video16/gt*/sampled_abs/action_range/combo_id).
    required_copy_keys = [
        "GT_TOOL_XY_DIR",
        "GT_TOOL_VIS_DIR",
        "GT_CT_XY_DIR",
        "GT_CT_VIS_DIR",
        "FRAME_INDEX_DIR",
        "ACTION_RANGE_ABS_DIR",
        "ACTION_RANGE_MASK_DIR",
        "COMBO_ID_DIR",
    ]
    if EXPORT_VIDEO16_NPY:
        required_copy_keys.append("VIDEO16_DIR")

    optional_copy_keys = [
        "KEYPOINT_DIR",
        "KEYPOINT_VIS_DIR",
        "CONTACT_VIS_DIR",
        "GT_TOOL_XY_PSEUDO_DIR",
        "GT_TOOL_PSEUDO_W_DIR",
        "GT_CT_XY_PSEUDO_DIR",
        "GT_CT_PSEUDO_W_DIR",
    ]

    for key in required_copy_keys:
        src = dirs[key] / f"{base_sid}.npy"
        dst = dirs[key] / f"{aug_sid}.npy"
        if not src.is_file():
            raise FileNotFoundError(f"Missing base file for aug copy: {src}")
        shutil.copy2(src, dst)

    for key in optional_copy_keys:
        src = dirs[key] / f"{base_sid}.npy"
        dst = dirs[key] / f"{aug_sid}.npy"
        if src.is_file():
            shutil.copy2(src, dst)

    npy_save(dirs["MOTION_DIR"] / f"{aug_sid}.npy", motion_delta_aug, dtype=np.float32)
    npy_save(dirs["ANCHOR_TIP_DIR"] / f"{aug_sid}.npy", anchor_tip_aug, dtype=np.float32)
    npy_save(dirs["COND_POINTS_DIR"] / f"{aug_sid}.npy", cond_points_aug, dtype=np.float32)
    npy_save(dirs["PT_ROLE_DIR"] / f"{aug_sid}.npy", np.array(base_item.pt_role, dtype=np.int64), dtype=np.int64)
    npy_save(dirs["TOOL_ID_DIR"] / f"{aug_sid}.npy", np.array(base_item.tool_id, dtype=np.int64))
    npy_save(dirs["ACT_ID_DIR"] / f"{aug_sid}.npy", np.array(base_item.act_id, dtype=np.int64))
    npy_save(dirs["COMBO_ID_DIR"] / f"{aug_sid}.npy", np.array(base_item.combo_id, dtype=np.int64))
    npy_save(
        dirs["ACTION_RANGE_ABS_DIR"] / f"{aug_sid}.npy",
        np.array([int(base_item.action_start_abs), int(base_item.action_end_abs)], dtype=np.int64),
        dtype=np.int64,
    )
    npy_save(
        dirs["ACTION_RANGE_MASK_DIR"] / f"{aug_sid}.npy",
        np.array(float(base_item.action_range_mask), dtype=np.float32),
        dtype=np.float32,
    )
    if gt_tool_xy_aug is not None:
        npy_save(dirs["GT_TOOL_XY_DIR"] / f"{aug_sid}.npy", np.asarray(gt_tool_xy_aug, dtype=np.float32), dtype=np.float32)
    if gt_tool_vis_aug is not None:
        npy_save(dirs["GT_TOOL_VIS_DIR"] / f"{aug_sid}.npy", np.asarray(gt_tool_vis_aug, dtype=np.float32), dtype=np.float32)
    if gt_ct_xy_aug is not None:
        npy_save(dirs["GT_CT_XY_DIR"] / f"{aug_sid}.npy", np.asarray(gt_ct_xy_aug, dtype=np.float32), dtype=np.float32)
    if gt_ct_vis_aug is not None:
        npy_save(dirs["GT_CT_VIS_DIR"] / f"{aug_sid}.npy", np.asarray(gt_ct_vis_aug, dtype=np.float32), dtype=np.float32)
    if sampled_abs_aug is not None:
        npy_save(dirs["FRAME_INDEX_DIR"] / f"{aug_sid}.npy", np.asarray(sampled_abs_aug, dtype=np.int64), dtype=np.int64)
    if EXPORT_VIDEO16_NPY and (video16_aug is not None):
        npy_save(dirs["VIDEO16_DIR"] / f"{aug_sid}.npy", np.asarray(video16_aug, dtype=np.uint8), dtype=np.uint8)
    if keypoint_raw_aug is not None and (dirs["KEYPOINT_DIR"] / f"{base_sid}.npy").is_file():
        npy_save(dirs["KEYPOINT_DIR"] / f"{aug_sid}.npy", np.asarray(keypoint_raw_aug, dtype=np.float32), dtype=np.float32)
    if contact_vis_raw_aug is not None and (dirs["CONTACT_VIS_DIR"] / f"{base_sid}.npy").is_file():
        npy_save(dirs["CONTACT_VIS_DIR"] / f"{aug_sid}.npy", np.asarray(contact_vis_raw_aug, dtype=np.float32), dtype=np.float32)

    npy_save(dirs["CONTACT_DIR"] / f"{aug_sid}.npy", contact_xy4_aug.reshape(-1), dtype=np.float32)
    cv2.imwrite(str(dirs["IMAGE_DIR"] / f"{aug_sid}.jpg"), image_aug_bgr)
    if ENABLE_MASK_IMAGE:
        H, W = image_aug_bgr.shape[:2]
        mask_bgr = build_pose_mask_image(anchor_tip_aug, contact_xy4_aug, W, H, radius_px=MASK_RADIUS_PX)
        cv2.imwrite(str(dirs["MASK_DIR"] / f"{aug_sid}.jpg"), mask_bgr)
    (dirs["TEXT_DIR"] / f"{aug_sid}.txt").write_text(prompt + "\n", encoding="utf-8")


def generate_one_aug(
    base_item: BaseItem,
    base_sid: str,
    aug_sid: str,
    dirs: Dict[str, Path],
    rng_py: random.Random,
    rng_np: np.random.Generator,
    debug_dir: Path,
    donor_pool_by_combo: Optional[Dict[str, List[str]]] = None,
    donor_pool_all: Optional[List[str]] = None,
    donor_sid_hint: Optional[str] = None,
    debug_saved_keys: Optional[Set[str]] = None,
    debug_fixed_dir: Optional[Path] = None,
    debug_fixed_pairs: Optional[List[Dict[str, object]]] = None,
    strength_bucket: str = "medium",
) -> Tuple[BaseItem, bool]:
    """
    Generate one augmented sample from base_sid.
    Returns (aug_item, is_geometric)
    """
    if USE_FOLD1_CLIP_POST_AUG:
        arr = load_base_clip_arrays(dirs, base_sid)
        pt_role = list(base_item.pt_role)

        video16 = np.asarray(arr["video16"], dtype=np.uint8)
        gt_tool_xy = np.asarray(arr["gt_tool_xy"], dtype=np.float32)
        gt_tool_vis = np.asarray(arr["gt_tool_vis"], dtype=np.float32)
        gt_ct_xy = np.asarray(arr["gt_ct_xy"], dtype=np.float32)
        gt_ct_vis = np.asarray(arr["gt_ct_vis"], dtype=np.float32)
        sampled_abs = np.asarray(arr["sampled_abs"], dtype=np.int64)

        profile = _strength_profile(
            strength_bucket if FOLD1_USE_V3_STRENGTH_MIX else "medium",
            combo=base_item.combo,
        )
        params: Dict[str, object] = {"strength_bucket": str(strength_bucket)}

        # 0) temporal start jitter in the first 10% boundary (augmentation only).
        t_len = int(video16.shape[0])
        start_jitter_p = float(profile.get("temp_start_jitter_p", FOLD1_TEMP_START_JITTER_P))
        start_jitter_max_frac = float(
            profile.get("temp_start_jitter_max_frac", FOLD1_TEMP_START_JITTER_MAX_FRAC)
        )
        max_shift = int(max(0, math.floor(float(max(0, t_len - 1)) * float(max(0.0, start_jitter_max_frac)))))
        params["start_jitter_p"] = float(start_jitter_p)
        params["start_jitter_max_frac"] = float(start_jitter_max_frac)
        params["start_jitter_max_shift"] = int(max_shift)
        if max_shift > 0 and (rng_py.random() < start_jitter_p):
            shift_frames = int(rng_py.randint(0, max_shift))
            if shift_frames > 0:
                video16, gt_tool_xy, gt_tool_vis, gt_ct_xy, gt_ct_vis, sampled_abs = fold1_temporal_start_jitter_clip(
                    video16=video16,
                    gt_tool_xy=gt_tool_xy,
                    gt_tool_vis=gt_tool_vis,
                    gt_ct_xy=gt_ct_xy,
                    gt_ct_vis=gt_ct_vis,
                    sampled_abs=sampled_abs,
                    shift_frames=shift_frames,
                )
                params["start_jitter_shift_frames"] = int(shift_frames)

        # 1) temporal warp on clip tensors.
        temp_warp_p = float(profile.get("temp_warp_p", FOLD1_TEMP_WARP_P))
        temp_warp_ratio_min = float(profile.get("temp_warp_ratio_min", FOLD1_TEMP_WARP_RATIO_MIN))
        temp_warp_ratio_max = float(profile.get("temp_warp_ratio_max", FOLD1_TEMP_WARP_RATIO_MAX))
        if temp_warp_ratio_max < temp_warp_ratio_min:
            temp_warp_ratio_min, temp_warp_ratio_max = temp_warp_ratio_max, temp_warp_ratio_min
        if rng_py.random() < temp_warp_p:
            ratio = float(rng_py.uniform(temp_warp_ratio_min, temp_warp_ratio_max))
            video16, gt_tool_xy, gt_tool_vis, gt_ct_xy, gt_ct_vis, sampled_abs = fold1_temporal_warp_clip(
                video16=video16,
                gt_tool_xy=gt_tool_xy,
                gt_tool_vis=gt_tool_vis,
                gt_ct_xy=gt_ct_xy,
                gt_ct_vis=gt_ct_vis,
                sampled_abs=sampled_abs,
                ratio=ratio,
            )
            params["temp_warp_ratio"] = float(ratio)
        params["temp_warp_p"] = float(temp_warp_p)
        params["temp_warp_ratio_min"] = float(temp_warp_ratio_min)
        params["temp_warp_ratio_max"] = float(temp_warp_ratio_max)

        # 2) keypoint-guided crop (mandatory in fold1 policy).
        crop_weights = dict(FOLD1_CROP_MODE_WEIGHTS.get(base_item.combo, FOLD1_CROP_MODE_WEIGHTS["default"]))
        if str(strength_bucket) == "light":
            crop_weights["context"] = float(crop_weights.get("context", 0.0) + 0.10)
        elif str(strength_bucket) == "heavy":
            crop_weights["context"] = float(max(0.05, crop_weights.get("context", 0.0) - 0.10))
            crop_weights["hinge"] = float(crop_weights.get("hinge", 0.0) + 0.05)
            crop_weights["tipcontact"] = float(crop_weights.get("tipcontact", 0.0) + 0.05)
        wsum = float(sum(max(0.0, float(v)) for v in crop_weights.values()))
        if wsum > 0.0:
            for k in list(crop_weights.keys()):
                crop_weights[k] = float(max(0.0, float(crop_weights[k])) / wsum)
        crop_mode = _weighted_choice_rng(crop_weights, rng_py)
        H0, W0 = int(video16.shape[1]), int(video16.shape[2])
        crop_bbox = _compute_fold1_crop_bbox(
            mode=crop_mode,
            gt_tool_xy=gt_tool_xy,
            gt_ct_xy=gt_ct_xy,
            pt_role=pt_role,
            W=W0,
            H=H0,
            rng_py=rng_py,
        )
        video16, gt_tool_xy, gt_ct_xy = _apply_fold1_crop_clip(video16, gt_tool_xy, gt_ct_xy, crop_bbox)
        params["crop_mode"] = str(crop_mode)
        params["crop_bbox_xyxy"] = [int(v) for v in crop_bbox]

        # 3) horizontal flip.
        hflip_p = float(profile.get("hflip_p", FOLD1_HFLIP_P))
        do_hflip = (rng_py.random() < hflip_p)
        if do_hflip:
            video16, gt_tool_xy, gt_ct_xy = _apply_fold1_hflip_clip(video16, gt_tool_xy, gt_ct_xy)
        params["hflip"] = bool(do_hflip)
        params["hflip_p"] = float(hflip_p)

        # 4) affine on clip.
        M_used = None
        affine_p = float(profile.get("affine_p", FOLD1_AFFINE_P))
        if rng_py.random() < affine_p:
            video16, gt_tool_xy, gt_ct_xy, M_used = _apply_fold1_affine_clip(
                video16=video16,
                gt_tool_xy=gt_tool_xy,
                gt_ct_xy=gt_ct_xy,
                rng_py=rng_py,
                rot_deg_max=float(profile.get("affine_rot_deg", FOLD1_AFFINE_ROT_DEG)),
                scale_min=float(profile.get("affine_scale_min", FOLD1_AFFINE_SCALE_MIN)),
                scale_max=float(profile.get("affine_scale_max", FOLD1_AFFINE_SCALE_MAX)),
                trans_frac=float(profile.get("affine_trans_frac", FOLD1_AFFINE_TRANS_FRAC)),
            )
        params["affine_M"] = None if M_used is None else M_used.tolist()
        params["affine_p"] = float(affine_p)

        # 5) keypoint-preserving background replacement.
        donor_video16: Optional[np.ndarray] = None
        donor_core_soft: Optional[np.ndarray] = None
        bg_replace_p = float(profile.get("bg_replace_p", FOLD1_BG_REPLACE_P))
        bg_same_combo_p = float(profile.get("bg_same_combo_p", FOLD1_BG_SAME_COMBO_P))
        do_bg = bool(
            (rng_py.random() < bg_replace_p)
            and (donor_pool_all is not None)
            and (len(donor_pool_all) > 0)
            and EXPORT_VIDEO16_NPY
        )
        if do_bg:
            use_same_combo = bool(
                (rng_py.random() < bg_same_combo_p)
                and (donor_pool_by_combo is not None)
                and (base_item.combo in donor_pool_by_combo)
                and (len(donor_pool_by_combo[base_item.combo]) > 0)
            )
            donor_list = donor_pool_by_combo[base_item.combo] if use_same_combo else donor_pool_all
            donor_sid = base_sid
            if (
                donor_sid_hint is not None
                and str(donor_sid_hint) != str(base_sid)
                and str(donor_sid_hint) in set([str(x) for x in donor_list])
            ):
                donor_sid = str(donor_sid_hint)
            if len(donor_list) > 1:
                # avoid self as donor when possible
                for _ in range(8):
                    sid_try = donor_list[rng_py.randrange(len(donor_list))]
                    if sid_try != base_sid:
                        donor_sid = sid_try
                        break
            elif len(donor_list) == 1:
                donor_sid = donor_list[0]
            donor_video16 = np.load(dirs["VIDEO16_DIR"] / f"{donor_sid}.npy").astype(np.uint8)
            donor_core_soft = _build_donor_core_soft_mask_seq_for_sid(
                dirs=dirs,
                donor_sid=str(donor_sid),
                pt_role=pt_role,
                T=int(video16.shape[0]),
                H=int(video16.shape[1]),
                W=int(video16.shape[2]),
            )
            video16 = _replace_bg_with_donor_clip(
                video16=video16,
                gt_tool_xy=gt_tool_xy,
                gt_ct_xy=gt_ct_xy,
                pt_role=pt_role,
                donor_video16=donor_video16,
                donor_core_soft_mask=donor_core_soft,
            )
            params["bg_replace"] = True
            params["bg_same_combo"] = bool(use_same_combo)
            params["bg_donor_sid"] = str(donor_sid)
            params["bg_donor_kpt_guard"] = bool(donor_core_soft is not None)
        else:
            params["bg_replace"] = False
        params["bg_replace_p"] = float(bg_replace_p)
        params["bg_same_combo_p"] = float(bg_same_combo_p)

        # 6) event-style hard ops (PSTM + occlusion) with core/soft protection.
        core_soft = _build_core_soft_mask_seq(
            gt_tool_xy=gt_tool_xy,
            gt_ct_xy=gt_ct_xy,
            pt_role=pt_role,
            H=int(video16.shape[1]),
            W=int(video16.shape[2]),
        )
        pstm_event_p = float(profile.get("pstm_event_p", 0.0))
        occlusion_event_p = float(profile.get("occlusion_event_p", 0.0))
        applied_ops: List[str] = []
        if (
            rng_py.random() < pstm_event_p
            and (donor_pool_all is not None)
            and (len(donor_pool_all) > 0)
            and EXPORT_VIDEO16_NPY
        ):
            if donor_video16 is None:
                if (
                    donor_sid_hint is not None
                    and str(donor_sid_hint) != str(base_sid)
                    and str(donor_sid_hint) in set([str(x) for x in donor_pool_all])
                ):
                    donor_sid = str(donor_sid_hint)
                else:
                    donor_sid = donor_pool_all[rng_py.randrange(len(donor_pool_all))]
                donor_video16 = np.load(dirs["VIDEO16_DIR"] / f"{donor_sid}.npy").astype(np.uint8)
                params["pstm_donor_sid"] = str(donor_sid)
                donor_core_soft = _build_donor_core_soft_mask_seq_for_sid(
                    dirs=dirs,
                    donor_sid=str(donor_sid),
                    pt_role=pt_role,
                    T=int(video16.shape[0]),
                    H=int(video16.shape[1]),
                    W=int(video16.shape[2]),
                )
            video16 = _apply_pstm_mix_event(
                video16=video16,
                donor_video16=donor_video16,
                core_soft=core_soft,
                rng_py=rng_py,
                strength_bucket=strength_bucket,
                donor_core_soft_mask=donor_core_soft,
            )
            applied_ops.append("pstm_mix")
            params["pstm_donor_kpt_guard"] = bool(donor_core_soft is not None)
        if rng_py.random() < occlusion_event_p:
            video16 = _apply_random_erasing_outside_core_event(
                video16=video16,
                core_soft=core_soft,
                rng_py=rng_py,
                strength_bucket=strength_bucket,
            )
            applied_ops.append("random_erasing")
            if rng_py.random() < 0.55:
                video16 = _apply_grid_occlusion_outside_core_event(
                    video16=video16,
                    core_soft=core_soft,
                    rng_py=rng_py,
                    strength_bucket=strength_bucket,
                )
                applied_ops.append("grid_occlusion")
        params["event_ops"] = applied_ops
        params["pstm_event_p"] = float(pstm_event_p)
        params["occlusion_event_p"] = float(occlusion_event_p)

        # 7) color / blur / noise / jpeg.
        video16, color_info = _color_augment_clip(video16, rng_py=rng_py, rng_np=rng_np, cfg=profile)
        params["color"] = color_info

        # Final synchronized arrays.
        gt_tool_xy = np.clip(gt_tool_xy.astype(np.float32), 0.0, 1.0)
        gt_ct_xy = np.clip(gt_ct_xy.astype(np.float32), 0.0, 1.0)
        gt_tool_vis = gt_tool_vis.astype(np.float32)
        gt_ct_vis = gt_ct_vis.astype(np.float32)
        sampled_abs = sampled_abs.astype(np.int64)
        motion_abs_aug = gt_tool_xy.astype(np.float32)
        contact_seq_aug = gt_ct_xy.astype(np.float32)
        contact_aug = canonical_order_xy(contact_seq_aug[0]).astype(np.float32)
        anchor_tip_aug = motion_abs_aug[0].reshape(6).astype(np.float32)
        cond_points_aug = motion_abs_aug[0].astype(np.float32)
        motion_repr_aug = encode_motion(motion_abs_aug)
        img_aug = video16[0].copy()
        prompt = make_prompt(base_item.tool, base_item.action, contact_aug)

        if SAVE_DEBUG_AUG_IMAGES:
            base_img = np.asarray(arr["image_bgr"], dtype=np.uint8)
            base_tool_t0 = np.asarray(arr["gt_tool_xy"], dtype=np.float32)[0]
            base_ct_t0 = np.asarray(arr["gt_ct_xy"], dtype=np.float32)[0]
            before = draw_overlay(base_img, base_tool_t0, np.array(pt_role, dtype=np.int64), base_ct_t0, base_img.shape[1], base_img.shape[0])
            after = draw_overlay(img_aug, motion_abs_aug[0], np.array(pt_role, dtype=np.int64), contact_aug, img_aug.shape[1], img_aug.shape[0])
            if (
                debug_fixed_dir is not None
                and (debug_fixed_pairs is not None)
                and (len(debug_fixed_pairs) < int(FOLD1_DEBUG_FIXED_PAIR_COUNT))
            ):
                rec = _save_debug_aug_pair_indexed(
                    debug_dir=debug_fixed_dir,
                    pair_idx=(len(debug_fixed_pairs) + 1),
                    aug_sid=aug_sid,
                    base_sid=base_sid,
                    combo=base_item.combo,
                    before=before,
                    after=after,
                    params=params,
                )
                debug_fixed_pairs.append(rec)
            if DEBUG_AUG_ONE_EXAMPLE_PER_TYPE:
                candidate_keys: List[str] = [f"fold1_crop_{str(crop_mode)}"]
                if do_hflip:
                    candidate_keys.append("fold1_hflip")
                if M_used is not None:
                    candidate_keys.append("fold1_affine")
                if bool(params.get("bg_replace", False)):
                    candidate_keys.append("fold1_bg_replace")
                for op_name in params.get("event_ops", []):
                    candidate_keys.append(f"fold1_{str(op_name)}")
                candidate_keys.append("fold1_color")
                for key in candidate_keys:
                    _save_debug_aug_pair_once(
                        debug_dir=debug_dir,
                        key=key,
                        before=before,
                        after=after,
                        saved_keys=debug_saved_keys,
                    )
            elif rng_py.random() < FOLD1_DEBUG_SAVE_PROB:
                cv2.imwrite(str(debug_dir / f"{aug_sid}_before.jpg"), before)
                cv2.imwrite(str(debug_dir / f"{aug_sid}_after.jpg"), after)

        save_aug_sample(
            aug_sid=aug_sid,
            base_sid=base_sid,
            base_item=base_item,
            dirs=dirs,
            motion_delta_aug=motion_repr_aug,
            anchor_tip_aug=anchor_tip_aug,
            cond_points_aug=cond_points_aug,
            contact_xy4_aug=contact_aug,
            image_aug_bgr=img_aug,
            prompt=prompt,
            gt_tool_xy_aug=motion_abs_aug,
            gt_tool_vis_aug=gt_tool_vis,
            gt_ct_xy_aug=contact_seq_aug,
            gt_ct_vis_aug=gt_ct_vis,
            sampled_abs_aug=sampled_abs,
            video16_aug=video16,
            keypoint_raw_aug=motion_abs_aug,
            contact_vis_raw_aug=gt_ct_vis,
        )

        Hn, Wn = int(video16.shape[1]), int(video16.shape[2])
        aug_item = BaseItem(
            sid=aug_sid,
            case=base_item.case,
            ann_relpath=base_item.ann_relpath,
            tool=base_item.tool,
            action=base_item.action,
            combo=base_item.combo,
            tool_id=base_item.tool_id,
            act_id=base_item.act_id,
            combo_id=base_item.combo_id,
            topology=base_item.topology,
            pt_role=base_item.pt_role,
            W=Wn,
            H=Hn,
            video_base_abs=base_item.video_base_abs,
            abs_start=base_item.abs_start,
            abs_end=base_item.abs_end,
            num_frames=int(motion_abs_aug.shape[0]),
            action_start_abs=int(base_item.action_start_abs),
            action_end_abs=int(base_item.action_end_abs),
            action_range_mask=int(base_item.action_range_mask),
            max_contact_visible_in_window=base_item.max_contact_visible_in_window,
            vis_count_per_point=base_item.vis_count_per_point,
            tool_vis_mean=float(np.mean(gt_tool_vis)),
            contact_vis_mean=float(np.mean(gt_ct_vis)),
            vis_score=base_item.vis_score,
            crop_bbox_xyxy=[0, 0, Wn, Hn],
            split="train",
            source_sid=base_sid,
            aug_type="fold1_clip",
            aug_params=params,
        )
        return aug_item, True

    # ---------- legacy augmentation fallback ----------
    motion_repr, anchor_tip, contact_xy4, img = load_base_arrays(dirs, base_sid)
    motion_abs = decode_motion(motion_repr, anchor_tip)  # -> (T,3,2)
    pt_role = np.array(base_item.pt_role, dtype=np.int64)
    aug_type = sample_aug_type(rng_py)
    W, H = base_item.W, base_item.H
    is_geometric = aug_type in ("A", "B")

    if aug_type == "A" or aug_type == "B":
        max_tries = 10
        motion_abs_aug = None
        contact_aug = None
        img_aug = None
        params = None
        for _ in range(max_tries):
            if USE_PIXEL_COORDS:
                tx = rng_py.uniform(-TRANS_MAX, TRANS_MAX) * float(W)
                ty = rng_py.uniform(-TRANS_MAX, TRANS_MAX) * float(H)
            else:
                tx = rng_py.uniform(-TRANS_MAX, TRANS_MAX)
                ty = rng_py.uniform(-TRANS_MAX, TRANS_MAX)
            if aug_type == "A":
                angle = rng_py.uniform(-ROT_DEG_MAX, ROT_DEG_MAX)
                scale = 1.0
            else:
                angle = 0.0
                scale = rng_py.uniform(SCALE_MIN, SCALE_MAX)

            center_norm = compute_center_norm(motion_abs, contact_xy4)
            M = affine_matrix_px(W, H, center_xy=center_norm, angle_deg=angle, scale=scale, trans_xy=(tx, ty))
            m_aug = apply_affine_to_points_norm(motion_abs, M, W, H)
            c_aug = apply_affine_to_points_norm(contact_xy4, M, W, H)
            if in_bounds(m_aug, W, H) and in_bounds(c_aug, W, H):
                if USE_PIXEL_COORDS:
                    motion_abs_aug = m_aug.astype(np.float32)
                    contact_aug = c_aug.astype(np.float32)
                    motion_abs_aug[..., 0] = np.clip(motion_abs_aug[..., 0], 0.0, float(W))
                    motion_abs_aug[..., 1] = np.clip(motion_abs_aug[..., 1], 0.0, float(H))
                    contact_aug[..., 0] = np.clip(contact_aug[..., 0], 0.0, float(W))
                    contact_aug[..., 1] = np.clip(contact_aug[..., 1], 0.0, float(H))
                else:
                    motion_abs_aug = np.clip(m_aug, 0.0, 1.0).astype(np.float32)
                    contact_aug = np.clip(c_aug, 0.0, 1.0).astype(np.float32)
                contact_aug = canonical_order_xy(contact_aug).astype(np.float32)
                img_aug = cv2.warpAffine(
                    img, M, dsize=(W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                params = {"tx": tx, "ty": ty, "angle_deg": angle, "scale": scale, "center_norm": center_norm}
                break
        if motion_abs_aug is None or contact_aug is None or img_aug is None:
            motion_abs_aug = motion_abs.copy()
            contact_aug = contact_xy4.copy()
            img_aug = img.copy()
            params = {"fallback": "identity"}
    elif aug_type == "C":
        ratio = rng_py.uniform(TIME_RATIO_MIN, TIME_RATIO_MAX)
        motion_abs_aug = time_warp_abs(motion_abs, ratio=ratio)
        contact_aug = contact_xy4.copy()
        img_aug = img.copy()
        params = {"ratio": float(ratio)}
    else:
        motion_abs_aug = lowfreq_noise_abs(motion_abs, rng_np)
        contact_aug = contact_xy4.copy()
        img_aug = img.copy()
        params = {"noise_std": float(NOISE_STD), "noise_ema": float(NOISE_EMA)}

    anchor_tip_aug = motion_abs_aug[0].reshape(6).astype(np.float32)
    cond_points_aug = motion_abs_aug[0].astype(np.float32)
    motion_repr_aug = encode_motion(motion_abs_aug)
    prompt = make_prompt(base_item.tool, base_item.action, contact_aug)
    if SAVE_DEBUG_AUG_IMAGES and is_geometric:
        before = draw_overlay(img, motion_abs[0], pt_role, contact_xy4, W, H)
        after = draw_overlay(img_aug, motion_abs_aug[0], pt_role, contact_aug, W, H)
        if DEBUG_AUG_ONE_EXAMPLE_PER_TYPE:
            _save_debug_aug_pair_once(
                debug_dir=debug_dir,
                key=f"legacy_{aug_type}",
                before=before,
                after=after,
                saved_keys=debug_saved_keys,
            )
        else:
            cv2.imwrite(str(debug_dir / f"{aug_sid}_before.jpg"), before)
            cv2.imwrite(str(debug_dir / f"{aug_sid}_after.jpg"), after)

    save_aug_sample(
        aug_sid=aug_sid,
        base_sid=base_sid,
        base_item=base_item,
        dirs=dirs,
        motion_delta_aug=motion_repr_aug,
        anchor_tip_aug=anchor_tip_aug,
        cond_points_aug=cond_points_aug,
        contact_xy4_aug=contact_aug,
        image_aug_bgr=img_aug,
        prompt=prompt,
    )

    aug_item = BaseItem(
        sid=aug_sid,
        case=base_item.case,
        ann_relpath=base_item.ann_relpath,
        tool=base_item.tool,
        action=base_item.action,
        combo=base_item.combo,
        tool_id=base_item.tool_id,
        act_id=base_item.act_id,
        combo_id=base_item.combo_id,
        topology=base_item.topology,
        pt_role=base_item.pt_role,
        W=base_item.W,
        H=base_item.H,
        video_base_abs=base_item.video_base_abs,
        abs_start=base_item.abs_start,
        abs_end=base_item.abs_end,
        num_frames=int(motion_abs_aug.shape[0]),
        action_start_abs=int(base_item.action_start_abs),
        action_end_abs=int(base_item.action_end_abs),
        action_range_mask=int(base_item.action_range_mask),
        max_contact_visible_in_window=base_item.max_contact_visible_in_window,
        vis_count_per_point=base_item.vis_count_per_point,
        tool_vis_mean=base_item.tool_vis_mean,
        contact_vis_mean=base_item.contact_vis_mean,
        vis_score=base_item.vis_score,
        crop_bbox_xyxy=base_item.crop_bbox_xyxy,
        split="train",
        source_sid=base_sid,
        aug_type=aug_type,
        aug_params=params,
    )
    return aug_item, is_geometric


def planned_aug_counts_for_combo(combo: str, n_train_in_combo: int) -> List[int]:
    """
    Return per-sample augmentation counts for one TRAIN combo.
    Uses ratio-based weak-combo policy and caps total extras per combo.
    """
    if n_train_in_combo <= 0:
        return []
    if combo in FOLD1_SKIP_AUG_COMBOS:
        return [0 for _ in range(n_train_in_combo)]
    if USE_FOLD1_CLIP_POST_AUG:
        target = int(FOLD1_TARGET_TRAIN_PER_COMBO.get(combo, n_train_in_combo))
        extra_total = max(0, int(target) - int(n_train_in_combo))
        if extra_total <= 0:
            return [0 for _ in range(n_train_in_combo)]
        base = extra_total // n_train_in_combo
        rem = extra_total % n_train_in_combo
        return [int(base + (1 if i < rem else 0)) for i in range(n_train_in_combo)]

    if not ENABLE_WEAK_COMBO_AUG:
        return []
    ratio = float(WEAK_COMBO_AUG_RATIO.get(combo, 0.0))
    if ratio <= 0.0:
        return []

    extra_total = int(round(float(n_train_in_combo) * ratio))
    cap_total = int(WEAK_COMBO_AUG_MAX_EXTRA.get(combo, extra_total))
    extra_total = min(extra_total, cap_total)
    if extra_total <= 0:
        return []

    base = extra_total // n_train_in_combo
    rem = extra_total % n_train_in_combo
    return [int(base + (1 if i < rem else 0)) for i in range(n_train_in_combo)]

# def check_stepdelta_inverse(motion_delta, anchor_tip, eps=1e-6):
#     """
#     motion_delta : (L,6) step-delta
#     anchor_tip   : (6,) abs at t=0
#
#     return: max reconstruction error
#     """
#     # 重建 abs
#     abs_rec = stepdelta_to_abs(motion_delta, anchor_tip)  # (L+1,3,2)
#
#     # 再算回 step-delta
#     delta_rec = abs_to_stepdelta(abs_rec)  # (L,6)
#
#     err = np.linalg.norm(delta_rec - motion_delta, axis=1)
#     return err.max(), err.mean()

def check_motion_inverse(motion_repr, anchor_tip, eps=1e-6):
    """
    Generic inverse consistency check for current MOTION_REP.
    motion_repr: stored motion array
    anchor_tip : (6,) abs at t=0
    """
    motion_repr = np.asarray(motion_repr, dtype=np.float32)
    abs_rec = decode_motion(motion_repr, anchor_tip)          # -> abs
    repr_rec = encode_motion(abs_rec)                         # -> repr again

    if repr_rec.shape != motion_repr.shape:
        return float("inf"), float("inf")

    if motion_repr.size == 0:
        return 0.0, 0.0

    err = np.linalg.norm(repr_rec - motion_repr, axis=1)
    return float(err.max()), float(err.mean())

def check_normalize_range(motion_abs, W=None, H=None, margin=1e-4):
    """
    motion_abs: (T,3,2)
    If USE_PIXEL_COORDS:
        check x in [0,W], y in [0,H]
    else:
        check in [0,1]
    """
    motion_abs = np.asarray(motion_abs, dtype=np.float32)

    if USE_PIXEL_COORDS:
        if W is None or H is None:
            return False, None
        x = motion_abs[..., 0]
        y = motion_abs[..., 1]
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        ok = (xmin >= -margin) and (xmax <= float(W) + margin) and (ymin >= -margin) and (ymax <= float(H) + margin)
        return ok, (xmin, xmax, ymin, ymax)

    else:
        min_v = float(motion_abs.min())
        max_v = float(motion_abs.max())
        ok = (-margin <= min_v) and (max_v <= 1.0 + margin)
        return ok, (min_v, max_v)


    ok = (-margin <= min_v) and (max_v <= 1.0 + margin)
    return ok, float(min_v), float(max_v)

def check_step0_not_zero(motion_delta, eps=1e-6):
    """
    motion_delta: (L,6)
    """
    norm0 = np.linalg.norm(motion_delta[0])
    return norm0 > eps, norm0

def run_sanity_checks(dataset_root, sid2wh=None, max_check=50):
    motions_dir = Path(dataset_root) / "motions"
    anchor_dir  = Path(dataset_root) / "anchor_tips"

    sids = sorted([p.stem for p in motions_dir.glob("*.npy")])
    print(f"[CHECK] total samples: {len(sids)}")

    for i, sid in enumerate(sids[:max_check]):
        motion_delta = np.load(motions_dir / f"{sid}.npy")   # (L,6)
        anchor_tip   = np.load(anchor_dir  / f"{sid}.npy")   # (6,)

        # # 1. inverse consistency (step-delta <-> abs)
        # max_err, mean_err = check_stepdelta_inverse(motion_delta, anchor_tip)
        #
        # # 2. reconstruct abs trajectory
        # abs_rec = stepdelta_to_abs(motion_delta, anchor_tip)  # (L+1,3,2)

        max_err, mean_err = check_motion_inverse(motion_delta, anchor_tip)
        abs_rec = decode_motion(motion_delta, anchor_tip)

        # 3. range check (pixel-aware or normalized-aware)
        W, H = None, None
        if sid2wh is not None and sid in sid2wh:
            W, H = sid2wh[sid]

        ok_range, range_info = check_normalize_range(abs_rec, W=W, H=H)

        # 4. step[0] magnitude (should usually be non-zero, but not mandatory)
        # ok_step0, step0_norm = check_step0_not_zero(motion_delta)
        if MOTION_REP == "relative" and REL_INCLUDE_T0:
            step0_norm = float(np.linalg.norm(motion_delta[0])) if motion_delta.shape[0] > 0 else 0.0
            ok_step0 = step0_norm < 1e-6
        else:
            ok_step0, step0_norm = check_step0_not_zero(motion_delta)

        # ---------- pretty print ----------
        if USE_PIXEL_COORDS:
            if range_info is not None:
                xmin, xmax, ymin, ymax = range_info
                range_str = f"x=({xmin:.1f},{xmax:.1f}) y=({ymin:.1f},{ymax:.1f})"
            else:
                range_str = "range=None"
        else:
            if range_info is not None:
                min_v, max_v = range_info
                range_str = f"({min_v:.3f},{max_v:.3f})"
            else:
                range_str = "range=None"

        print(
            f"[{sid}] "
            f"inv_err(max/mean)=({max_err:.2e}/{mean_err:.2e}) | "
            f"range={range_str} | "
            f"step0_norm={step0_norm:.4f}"
        )

        # ---------- warnings ----------
        if max_err > 1e-5:
            print("  ❌ inverse consistency FAILED")
        if not ok_range:
            print("  ❌ range check FAILED")
        if not ok_step0:
            print("  ⚠️ step[0] is zero (can be OK for early-static motion)")


# =========================
# Main
# =========================
def main():
    global SEED, SPLIT_MODE, TRAIN_K_BIG, VAL_K_BIG, VAL_K_SMALL, VAL_EVERY_K, TRAIN_K_PER_COMBO, VAL_K_PER_COMBO
    global FOLDER_SPLIT_VAL_BACKFILL
    global VIS_PRIORITY_FOR_BIG, VIS_PRIORITY_MIN_COMBO_COUNT
    global OUTPUT_MODE
    global VIDEO16_DIR_OVERRIDE

    parser = argparse.ArgumentParser("Build phase4p5 dataset with fixed 16f keypoint/combo targets")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "surg_act_video_toy"),
        help="Raw dataset root containing case directories and index.json",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="",
        help="Output directory path. Empty means <data-root-parent>/<OUT_NAME>.",
    )
    parser.add_argument(
        "--video16-dir-override",
        type=str,
        default="",
        help="Optional external/staging directory for video_16_npy.",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default=OUTPUT_MODE,
        choices=["full", "minimal_kpt_combo", "tool_kpt_only", "combo_only_train"],
        help=(
            "Export mode: full keeps all artifacts; minimal_kpt_combo keeps kpt+combo contract; "
            "tool_kpt_only keeps tool keypoint trajectory contract only; "
            "combo_only_train keeps only combo classification contract."
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for split/sampling.")
    parser.add_argument(
        "--split-mode",
        type=str,
        default=SPLIT_MODE,
        choices=["legacy", "neighbor"],
        help="Split policy: legacy diverse sampling or neighbor-aware split.",
    )
    parser.add_argument(
        "--train-k-big",
        type=int,
        default=TRAIN_K_BIG,
        help="For big combos (count>=100), keep this many TRAIN samples.",
    )
    parser.add_argument(
        "--val-k-big",
        type=int,
        default=VAL_K_BIG,
        help="For big combos (count>=100), keep this many VAL samples.",
    )
    parser.add_argument(
        "--val-k-small",
        type=int,
        default=VAL_K_SMALL,
        help="For small combos (count<100), keep this many VAL samples.",
    )
    parser.add_argument(
        "--val-every-k",
        type=int,
        default=VAL_EVERY_K,
        help="Neighbor mode stride control: every (K+1)-th index is a VAL candidate.",
    )
    parser.add_argument(
        "--train-k-per-combo",
        type=int,
        default=0,
        help="If >0, override big/small policy and cap TRAIN count per combo to this value.",
    )
    parser.add_argument(
        "--val-k-per-combo",
        type=int,
        default=0,
        help="If >0, override big/small policy and cap VAL count per combo to this value.",
    )
    parser.add_argument(
        "--no-folder-split-val-backfill",
        action="store_true",
        help=(
            "Disable val backfill in folder split mode. "
            "By default, missing VAL samples for a combo can be moved from train-case pool."
        ),
    )
    parser.add_argument(
        "--max-base-per-combo",
        type=int,
        default=0,
        help="If >0, stop exporting new base samples for a combo after this count is reached.",
    )
    parser.add_argument(
        "--prefer-high-vis-big",
        action="store_true",
        help="For big combos, prefer higher-visibility clips when selecting split samples.",
    )
    parser.add_argument(
        "--vis-priority-min-combo-count",
        type=int,
        default=VIS_PRIORITY_MIN_COMBO_COUNT,
        help="Apply visibility-priority only when combo count is at least this value.",
    )
    parser.add_argument(
        "--max-aug-samples",
        type=int,
        default=0,
        help="If >0, stop fold1/train augmentation after generating this many augmented samples.",
    )
    parser.add_argument(
        "--target-total-samples",
        type=int,
        default=0,
        help=(
            "If >0, enforce final total samples (train+val+test) by capping generated "
            "augmentation count to max(0, target_total - base_total_after_split)."
        ),
    )
    args = parser.parse_args()

    if args.train_k_big < 0 or args.val_k_big < 0 or args.val_k_small < 0:
        raise ValueError("train/val k arguments must be non-negative.")
    if args.val_every_k < 0:
        raise ValueError("--val-every-k must be >= 0.")
    if args.train_k_per_combo < 0 or args.val_k_per_combo < 0:
        raise ValueError("--train-k-per-combo / --val-k-per-combo must be >= 0.")
    if args.max_base_per_combo < 0:
        raise ValueError("--max-base-per-combo must be >= 0.")
    if args.vis_priority_min_combo_count < 0:
        raise ValueError("--vis-priority-min-combo-count must be >= 0.")
    if args.max_aug_samples < 0:
        raise ValueError("--max-aug-samples must be >= 0.")
    if args.target_total_samples < 0:
        raise ValueError("--target-total-samples must be >= 0.")

    SEED = int(args.seed)
    OUTPUT_MODE = str(args.output_mode)
    _refresh_output_mode_dependent_config()
    SPLIT_MODE = str(args.split_mode)
    TRAIN_K_BIG = int(args.train_k_big)
    VAL_K_BIG = int(args.val_k_big)
    VAL_K_SMALL = int(args.val_k_small)
    VAL_EVERY_K = int(args.val_every_k)
    TRAIN_K_PER_COMBO = int(args.train_k_per_combo) if args.train_k_per_combo > 0 else None
    VAL_K_PER_COMBO = int(args.val_k_per_combo) if args.val_k_per_combo > 0 else None
    FOLDER_SPLIT_VAL_BACKFILL = not bool(args.no_folder_split_val_backfill)
    VIS_PRIORITY_FOR_BIG = bool(args.prefer_high_vis_big)
    VIS_PRIORITY_MIN_COMBO_COUNT = int(args.vis_priority_min_combo_count)

    rng_py = random.Random(SEED)
    rng_np = np.random.default_rng(SEED)

    root_dir = Path(args.data_root).resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"data root not found: {root_dir}")
    out_root = Path(args.out_root).resolve() if args.out_root else (root_dir.parent / OUT_NAME)
    safe_mkdir(out_root)
    video16_override_arg = str(getattr(args, "video16_dir_override", "")).strip()
    video16_override = video16_override_arg if video16_override_arg else str(VIDEO16_DIR_OVERRIDE).strip()
    video16_link_path = out_root / VIDEO16_DIR_NAME
    video16_write_dir = video16_link_path
    if video16_override:
        video16_write_dir = Path(video16_override).expanduser().resolve()
        safe_mkdir(video16_write_dir)
        if video16_link_path.exists() or video16_link_path.is_symlink():
            if video16_link_path.is_symlink():
                cur_target = video16_link_path.resolve()
                if cur_target != video16_write_dir:
                    raise RuntimeError(
                        f"video16 link exists with different target: {video16_link_path} -> {cur_target}, "
                        f"expected {video16_write_dir}"
                    )
            else:
                cur_real = video16_link_path.resolve()
                if cur_real != video16_write_dir:
                    raise RuntimeError(
                        f"video16 path already exists and is not staging target: {video16_link_path} (real={cur_real}), "
                        f"expected {video16_write_dir}"
                    )
        else:
            video16_link_path.symlink_to(video16_write_dir, target_is_directory=True)

    max_base_per_combo = int(args.max_base_per_combo) if args.max_base_per_combo > 0 else None
    max_aug_samples = int(args.max_aug_samples) if args.max_aug_samples > 0 else None
    target_total_samples = int(args.target_total_samples) if args.target_total_samples > 0 else None
    print(f"[setup] data_root={root_dir}")
    print(f"[setup] out_root={out_root}")
    print(f"[setup] video16_write_dir={video16_write_dir}")
    if video16_override:
        print(f"[setup] video16_link={video16_link_path} -> {video16_write_dir}")
    print(
        "[setup] split_config: "
        f"mode={SPLIT_MODE} train_k_big={TRAIN_K_BIG} val_k_big={VAL_K_BIG} "
        f"val_k_small={VAL_K_SMALL} val_every_k={VAL_EVERY_K} "
        f"train_k_per_combo={TRAIN_K_PER_COMBO} val_k_per_combo={VAL_K_PER_COMBO} "
        f"max_base_per_combo={max_base_per_combo} max_aug_samples={max_aug_samples} "
        f"target_total_samples={target_total_samples} "
        f"folder_split_val_backfill={FOLDER_SPLIT_VAL_BACKFILL} "
        f"vis_priority_for_big={VIS_PRIORITY_FOR_BIG} "
        f"vis_priority_min_combo_count={VIS_PRIORITY_MIN_COMBO_COUNT} "
        f"disable_filter_rules={DISABLE_FILTER_RULES}"
    )

    combo_aux_root: Optional[Path] = None
    if OUTPUT_MODE in ("combo_only_train", "tool_kpt_only"):
        combo_aux_root = Path("/tmp") / f"vitpose_compact_aux_{os.getpid()}"
        safe_mkdir(combo_aux_root)
        print(f"[setup] compact mode aux_root={combo_aux_root}")

    dirs = {
        "MOTION_DIR": out_root / "motions",
        "ANCHOR_TIP_DIR": out_root / "anchor_tips",
        "COND_POINTS_DIR": out_root / "cond_points",
        "VIDEO16_DIR": video16_write_dir,
        "KEYPOINT_DIR": out_root / "keypoints",
        "KEYPOINT_VIS_DIR": out_root / "keypoint_vis",
        "CONTACT_VIS_DIR": out_root / "contact_vis_raw",
        "FRAME_INDEX_DIR": out_root / "sampled_abs_frames",
        "ACTION_RANGE_ABS_DIR": out_root / "action_range_abs",
        "ACTION_RANGE_MASK_DIR": out_root / "action_range_mask",
        "GT_TOOL_XY_DIR": out_root / "gt_tool_xy",
        "GT_TOOL_VIS_DIR": out_root / "gt_tool_vis",
        "GT_CT_XY_DIR": out_root / "gt_ct_xy",
        "GT_CT_VIS_DIR": out_root / "gt_ct_vis",
        "COMBO_ID_DIR": out_root / "combo_ids",
        "GT_TOOL_XY_PSEUDO_DIR": out_root / "gt_tool_xy_pseudo",
        "GT_TOOL_PSEUDO_W_DIR": out_root / "gt_tool_pseudo_w",
        "GT_CT_XY_PSEUDO_DIR": out_root / "gt_ct_xy_pseudo",
        "GT_CT_PSEUDO_W_DIR": out_root / "gt_ct_pseudo_w",
        "PT_ROLE_DIR": out_root / "pt_roles",
        "TOOL_ID_DIR": out_root / "tool_ids",
        "ACT_ID_DIR": out_root / "act_ids",
        "IMAGE_DIR": out_root / "images",
        "CONTACT_DIR": out_root / "contacts",
        "TEXT_DIR": out_root / "texts",
    }
    if combo_aux_root is not None:
        keep_out_keys = {"VIDEO16_DIR", "COMBO_ID_DIR"}
        if OUTPUT_MODE == "tool_kpt_only":
            keep_out_keys.update({"GT_TOOL_XY_DIR", "GT_TOOL_VIS_DIR"})
        for k, p in list(dirs.items()):
            if k not in keep_out_keys:
                dirs[k] = combo_aux_root / p.name
    if EXPORT_GEO_FEATURES:
        dirs.update({
            "CONTACT_AREA_DIR": out_root / "contact_areas",
            "TOOL_AREA_DIR": out_root / "tool_areas",
            "CENTER_DIST_DIR": out_root / "center_dists",
            "EDGE_COS_DIR": out_root / "edge_cos",
        })
    if ENABLE_MASK_IMAGE:
        dirs["MASK_DIR"] = out_root / MASK_DIR_NAME

    for d in dirs.values():
        safe_mkdir(d)
    debug_dir = (combo_aux_root / DEBUG_DIR_NAME) if combo_aux_root is not None else (out_root / DEBUG_DIR_NAME)
    safe_mkdir(debug_dir)
    debug_fixed_dir = out_root / FOLD1_DEBUG_FIXED_DIR_NAME
    safe_mkdir(debug_fixed_dir)

    stats = {
        "skip_bad_json": 0,
        "skip_no_tool_action": 0,
        "skip_unknown_tool_or_action": 0,
        "skip_unknown_combo": 0,
        "skip_no_dims": 0,
        "skip_empty_points": 0,
        "warn_no_affordance_range": 0,
        "warn_no_action_range": 0,
        "warn_bad_action_range": 0,
        "skip_bad_range": 0,
        "skip_too_short": 0,
        "skip_missing_video": 0,
        "skip_bad_video_read": 0,
        "skip_no_valid_start": 0,
        "skip_write_error": 0,
        "skip_hook_dissect_too_short": 0,
        "warn_start_fallback_any_tool": 0,
        "warn_contact_fallback_to_tool_box": 0,
        "warn_sampled_frame_missing_tool": 0,
        "warn_sampled_frame_missing_contact": 0,
        "warn_video16_frame_fallback": 0,
        "warn_video16_frame_resized_mismatch": 0,
        "skip_low_vis_per_point": 0,
        "skip_low_vis_total": 0,
        "skip_cap_reached": 0,
    }

    meta: List[BaseItem] = []
    scanned = 0

    # deterministic scan of case dirs
    case_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
    selected_case_folders_runtime: List[str] = list(SELECTED_CASE_FOLDERS)
    val_case_folders_runtime: List[str] = list(VAL_CASE_FOLDERS)
    if USE_FOLDER_SPLIT_10_1 and USE_FOLD1_CLIP_POST_AUG and FOLD1_AUG_SCOPE_BY_CASE_ONLY:
        if FOLD1_TRAIN_CASES_FILE.is_file() and FOLD1_VAL_CASES_FILE.is_file():
            fold1_train_cases = sorted(_read_sid_set(FOLD1_TRAIN_CASES_FILE))
            fold1_val_cases = sorted(_read_sid_set(FOLD1_VAL_CASES_FILE))
            if len(fold1_train_cases) > 0 and len(fold1_val_cases) > 0:
                selected_case_folders_runtime = sorted(list(dict.fromkeys(fold1_train_cases + fold1_val_cases)))
                val_case_folders_runtime = list(fold1_val_cases)
                print(
                    f"[setup] fold1_case_override=True "
                    f"selected={len(selected_case_folders_runtime)} "
                    f"train={len(fold1_train_cases)} val={len(fold1_val_cases)}"
                )
            else:
                print(
                    "[setup] fold1_case_override skipped: empty fold1 case files, "
                    "fallback to configured SELECTED_CASE_FOLDERS/VAL_CASE_FOLDERS"
                )
        else:
            print(
                "[setup] fold1_case_override skipped: fold1 case files missing, "
                "fallback to configured SELECTED_CASE_FOLDERS/VAL_CASE_FOLDERS"
            )

    selected_case_set = set(selected_case_folders_runtime) if USE_FOLDER_SPLIT_10_1 else set()
    val_case_set = set(val_case_folders_runtime) if USE_FOLDER_SPLIT_10_1 else set()
    if USE_FOLDER_SPLIT_10_1:
        existing_case_names = {p.name for p in case_dirs}
        missing_selected = sorted(selected_case_set - existing_case_names)
        if len(missing_selected) > 0:
            raise FileNotFoundError(f"Missing selected case folders under {root_dir}: {missing_selected}")
        if not val_case_set.issubset(selected_case_set):
            bad = sorted(val_case_set - selected_case_set)
            raise ValueError(f"VAL_CASE_FOLDERS must be subset of SELECTED_CASE_FOLDERS, bad={bad}")
        train_case_set = selected_case_set - val_case_set
        if len(val_case_set) == 0:
            raise ValueError(
                "Folder split requires at least one val folder in VAL_CASE_FOLDERS."
            )
        if len(train_case_set) == 0:
            raise ValueError(
                "Folder split requires at least one train folder in SELECTED_CASE_FOLDERS "
                f"after removing val folders; got selected={len(selected_case_set)} val={len(val_case_set)}"
            )
        case_dirs = [p for p in case_dirs if p.name in selected_case_set]
        print(
            f"[setup] folder_split_selected=True selected={len(selected_case_set)} "
            f"train={len(train_case_set)} val={len(val_case_set)}"
        )
        print(f"[setup] val_folders={sorted(val_case_set)}")

    next_idx = 0
    combo_kept_counts: Dict[str, int] = {k: 0 for k in COMBO2ID.keys()}

    # Fold-1 strict path: preselect final BASE annotation list first, then export files.
    # This avoids "write all base then cap later".
    use_fold1_preselect = bool(
        USE_FOLD1_CLIP_POST_AUG
        and USE_FOLDER_SPLIT_10_1
        and FOLD1_AUG_SCOPE_BY_CASE_ONLY
    )
    if use_fold1_preselect:
        train_case_set_runtime = set(selected_case_set - val_case_set)
        train_pool_by_combo_case: Dict[str, Dict[str, List[Path]]] = {}
        val_pool_by_combo: Dict[str, List[Path]] = {}
        train_pool_count_by_combo: Dict[str, int] = {}
        val_pool_count_by_combo: Dict[str, int] = {}

        # Pass-1: lightweight scan only (no base file export).
        for case_dir in case_dirs:
            case_name = case_dir.name
            ann_dir = case_dir / "annotation"
            if not ann_dir.exists():
                continue
            ann_paths = sorted(list(ann_dir.glob("*.json")))
            for ann_path in ann_paths:
                scanned += 1
                parsed = _quick_read_combo_from_ann(ann_path)
                if parsed is None:
                    continue
                _, _, combo_fast = parsed
                if combo_fast not in COMBO2ID:
                    continue
                if case_name in val_case_set:
                    val_pool_by_combo.setdefault(combo_fast, []).append(ann_path)
                    val_pool_count_by_combo[combo_fast] = int(val_pool_count_by_combo.get(combo_fast, 0) + 1)
                elif case_name in train_case_set_runtime:
                    by_case = train_pool_by_combo_case.setdefault(combo_fast, {})
                    by_case.setdefault(case_name, []).append(ann_path)
                    train_pool_count_by_combo[combo_fast] = int(train_pool_count_by_combo.get(combo_fast, 0) + 1)

        # Pass-2: apply per-combo cap on TRAIN pool before any file export.
        selected_train_ann_paths: List[Path] = []
        cap_report_lines: List[str] = []
        for combo in sorted(COMBO2ID.keys(), key=lambda c: int(COMBO2ID[c])):
            case2paths = train_pool_by_combo_case.get(combo, {})
            all_paths_combo: List[Path] = []
            for c in sorted(case2paths.keys()):
                all_paths_combo.extend(sorted(case2paths[c], key=lambda p: str(p)))
            before_n = int(len(all_paths_combo))
            cap_v = FOLD1_TRAIN_BASE_CAP_PER_COMBO.get(combo, None)
            if cap_v is None:
                kept_paths = all_paths_combo
                after_n = int(len(kept_paths))
            else:
                cap_n = max(0, int(cap_v))
                if before_n <= cap_n:
                    kept_paths = all_paths_combo
                else:
                    if FOLD1_CAP_SAMPLE_BY_CASE_BALANCED and combo == "hook|dissect":
                        kept_paths = _sample_ann_paths_balanced_by_case(
                            case2paths=case2paths,
                            cap_n=cap_n,
                            seed=int(SEED) + 1701 + sum(ord(ch) for ch in combo),
                        )
                    else:
                        work = sorted(all_paths_combo, key=lambda p: str(p))
                        rng_cap = random.Random(int(SEED) + 1701 + sum(ord(ch) for ch in combo))
                        rng_cap.shuffle(work)
                        kept_paths = sorted(work[:cap_n], key=lambda p: str(p))
                after_n = int(len(kept_paths))
                dropped_n = int(before_n - after_n)
                cap_report_lines.append(
                    f"  - {combo:20s}: before={before_n} after={after_n} dropped={dropped_n} cap={cap_n}"
                )
            selected_train_ann_paths.extend(kept_paths)
            train_pool_count_by_combo[combo] = int(after_n if cap_v is not None else before_n)

        selected_val_ann_paths: List[Path] = []
        for combo in sorted(COMBO2ID.keys(), key=lambda c: int(COMBO2ID[c])):
            selected_val_ann_paths.extend(sorted(val_pool_by_combo.get(combo, []), key=lambda p: str(p)))

        selected_ann_paths = sorted(list(dict.fromkeys(selected_train_ann_paths + selected_val_ann_paths)), key=lambda p: str(p))
        print(
            f"[SCAN][FOLD1] preselect first, then export: "
            f"train_selected={len(selected_train_ann_paths)} val_selected={len(selected_val_ann_paths)} "
            f"total_selected={len(selected_ann_paths)}"
        )
        if len(cap_report_lines) > 0:
            print("[SCAN][FOLD1] train cap applied before export:")
            for ln in cap_report_lines:
                print(ln)
        for combo in sorted(COMBO2ID.keys(), key=lambda c: int(COMBO2ID[c])):
            tr_n = int(train_pool_count_by_combo.get(combo, 0))
            va_n = int(val_pool_count_by_combo.get(combo, 0))
            if tr_n > 0 or va_n > 0:
                print(f"[SCAN][FOLD1][BASE] {combo:20s}: train={tr_n:4d} val={va_n:4d}")

        # Pass-3: export only selected BASE items.
        for ann_path in selected_ann_paths:
            if max_base_per_combo is not None:
                parsed = _quick_read_combo_from_ann(ann_path)
                combo_fast = parsed[2] if parsed is not None else None
                if combo_fast in COMBO2ID and combo_kept_counts.get(combo_fast, 0) >= max_base_per_combo:
                    stats["skip_cap_reached"] += 1
                    continue
            sid = f"{next_idx:06d}"
            item = extract_base_sample(
                ann_path=ann_path,
                root_dir=root_dir,
                sid=sid,
                dirs=dirs,
                stats=stats,
            )
            if item is None:
                continue
            combo_item = f"{item.tool}|{item.action}"
            meta.append(item)
            next_idx += 1
            if combo_item in combo_kept_counts:
                combo_kept_counts[combo_item] = int(combo_kept_counts[combo_item]) + 1
    else:
        stop_sampling = False
        for case_dir in case_dirs:
            if stop_sampling:
                break
            ann_dir = case_dir / "annotation"
            if not ann_dir.exists():
                continue
            ann_paths = sorted(list(ann_dir.glob("*.json")))
            for ann_path in ann_paths:
                if stop_sampling:
                    break
                scanned += 1

                if max_base_per_combo is not None:
                    combo_fast: Optional[str] = None
                    parsed = _quick_read_combo_from_ann(ann_path)
                    if parsed is not None:
                        _, _, combo_fast = parsed
                    if combo_fast in COMBO2ID and combo_kept_counts.get(combo_fast, 0) >= max_base_per_combo:
                        stats["skip_cap_reached"] += 1
                        continue

                sid = f"{next_idx:06d}"
                item = extract_base_sample(
                    ann_path=ann_path,
                    root_dir=root_dir,
                    sid=sid,
                    dirs=dirs,
                    stats=stats,
                )
                if item is not None:
                    combo_item = f"{item.tool}|{item.action}"
                    meta.append(item)
                    next_idx += 1
                    if combo_item in combo_kept_counts:
                        combo_kept_counts[combo_item] = int(combo_kept_counts[combo_item]) + 1

                    if max_base_per_combo is not None:
                        if all(combo_kept_counts.get(c, 0) >= max_base_per_combo for c in COMBO2ID.keys()):
                            stop_sampling = True
                            break

    if len(meta) == 0:
        raise RuntimeError("No base samples generated. Check filters / paths / mappings.")

    # Report combo counts (base)
    combo_counts: Dict[str, int] = {}
    for it in meta:
        combo_counts[f"{it.tool}|{it.action}"] = combo_counts.get(f"{it.tool}|{it.action}", 0) + 1

    print("---- Base Tool × Action combos (counts) ----")
    for combo in sorted(combo_counts.keys()):
        print(f"{combo:20s}: {combo_counts[combo]}")

    # Assign splits for BASE items
    # train_base_ids, val_ids, test_ids = assign_splits(meta, rng_py)
    # 2026/02/02
    if USE_FOLDER_SPLIT_10_1:
        train_base_ids, val_ids, test_ids = assign_splits_by_selected_folders(
            meta=meta,
            selected_cases=selected_case_folders_runtime,
            val_cases=val_case_folders_runtime,
            rng=rng_py,
            train_k_per_combo=TRAIN_K_PER_COMBO,
            val_k_per_combo=VAL_K_PER_COMBO,
            allow_val_backfill_from_train=FOLDER_SPLIT_VAL_BACKFILL,
        )
    elif SPLIT_MODE == "neighbor":
        train_base_ids, val_ids, test_ids = assign_splits_neighbor(meta)
    else:
        train_base_ids, val_ids, test_ids = assign_splits(meta, rng_py)

    # Split summary by combo (base only)
    sid2base = {it.sid: it for it in meta}
    print("---- Split summary by combo (base) ----")
    for combo in sorted(combo_counts.keys()):
        tr = 0
        va = 0
        te = 0
        for sid in train_base_ids:
            it = sid2base[sid]
            if f"{it.tool}|{it.action}" == combo:
                tr += 1
        for sid in val_ids:
            it = sid2base[sid]
            if f"{it.tool}|{it.action}" == combo:
                va += 1
        for sid in test_ids:
            it = sid2base[sid]
            if f"{it.tool}|{it.action}" == combo:
                te += 1
        print(f"{combo:20s}: train={tr:4d}  val={va:3d}  test={te:4d}")

    # Train-only offset-base expansion stage (before offline augmentation).
    offset_base_added_by_combo: Dict[str, int] = {}
    offset_base_planned_by_combo: Dict[str, int] = {}
    offset_base_available_by_combo: Dict[str, int] = {}
    if USE_OFFSET_BASE_EXPANSION:
        sid2base = {it.sid: it for it in meta}
        train_base_snapshot = list(train_base_ids)
        raw_train_count_by_combo: Dict[str, int] = {}
        offset_plan_by_combo: Dict[str, List[Dict[str, object]]] = {}
        print(
            f"[OFFSET][BASE] enabled=True train_base={len(train_base_snapshot)} "
            f"min_seg_len={int(OFFSET_BASE_MIN_SEG_LEN)} clip_len={int(TARGET_FRAMES)}"
        )
        for base_sid in train_base_snapshot:
            base_item = sid2base.get(base_sid, None)
            if base_item is None:
                continue
            combo = f"{base_item.tool}|{base_item.action}"
            raw_train_count_by_combo[combo] = int(raw_train_count_by_combo.get(combo, 0) + 1)
            mult = int(OFFSET_BASE_MULTIPLIER_BY_COMBO.get(combo, 1))
            if mult <= 1:
                continue
            seg_len = int(base_item.abs_end - base_item.video_base_abs + 1)
            if seg_len < int(OFFSET_BASE_MIN_SEG_LEN):
                continue
            starts = offset_start_positions(
                L=int(seg_len),
                clip_len=int(TARGET_FRAMES),
                K_total=int(mult),
            )
            base_offset = int(max(0, int(base_item.abs_start) - int(base_item.video_base_abs)))
            extra_offsets = order_offset_extra_starts(starts=starts, base_offset=base_offset)
            if len(extra_offsets) <= 0:
                continue
            ann_path = root_dir / str(base_item.ann_relpath)
            if not ann_path.is_file():
                continue
            offset_plan_by_combo.setdefault(combo, []).append(
                {
                    "base_sid": str(base_sid),
                    "base_item": base_item,
                    "ann_path": ann_path,
                    "seg_len": int(seg_len),
                    "mult": int(mult),
                    "extra_offsets": [int(x) for x in extra_offsets],
                }
            )

        if USE_FOLD1_CLIP_POST_AUG and str(OUTPUT_MODE) == "tool_kpt_only" and len(FOLD1_TARGET_TRAIN_PER_COMBO) > 0:
            impossible_lines: List[str] = []
            target_total = 0
            floor_total_after_offset = 0
            combos_in_scope = sorted(
                set([str(k) for k in raw_train_count_by_combo.keys()])
                | set([str(k) for k in OFFSET_BASE_TARGET_PER_COMBO.keys()])
                | set([str(k) for k in FOLD1_TARGET_TRAIN_PER_COMBO.keys()])
            )
            for combo in combos_in_scope:
                raw_n = int(raw_train_count_by_combo.get(combo, 0))
                offset_target_v = OFFSET_BASE_TARGET_PER_COMBO.get(combo, None)
                floor_after_offset = int(raw_n)
                if offset_target_v is not None and int(offset_target_v) > floor_after_offset:
                    floor_after_offset = int(offset_target_v)

                target_v = FOLD1_TARGET_TRAIN_PER_COMBO.get(combo, None)
                if target_v is None:
                    continue
                target_n = int(target_v)
                target_total += target_n
                floor_total_after_offset += floor_after_offset
                if floor_after_offset > target_n:
                    impossible_lines.append(
                        f"combo={combo} raw_train={raw_n} "
                        f"offset_floor={floor_after_offset} target={target_n}"
                    )

            if floor_total_after_offset > target_total:
                impossible_lines.append(
                    f"total_offset_floor={floor_total_after_offset} target_total={target_total}"
                )

            if len(impossible_lines) > 0:
                msg = "\n".join(impossible_lines)
                raise ValueError(
                    "Fold1 target/train config is impossible before offline augmentation. "
                    "Current base split and offset floor already exceed the requested target(s). "
                    "Add/restore train_base_cap_per_combo or lower offset/target settings.\n"
                    f"{msg}"
                )

        selected_offset_tasks: List[Tuple[str, Dict[str, object], int, int]] = []
        for combo in sorted(offset_plan_by_combo.keys()):
            entries = sorted(offset_plan_by_combo[combo], key=lambda x: str(x["base_sid"]))
            raw_n = int(raw_train_count_by_combo.get(combo, 0))
            available_extra = int(sum(len(rec["extra_offsets"]) for rec in entries))
            offset_base_available_by_combo[combo] = int(available_extra)

            # Tool-trajectory A-series expects offset expansion to increase temporal-window
            # diversity, not to blindly multiply final sample counts. Respect the strongest
            # explicit combo budget available: train-base cap first, then offset-stage
            # target, and only fall back to final target if no stage-1 target was provided.
            budget_total: Optional[int] = None
            budget_source = "all"
            cap_v = FOLD1_TRAIN_BASE_CAP_PER_COMBO.get(combo, None)
            if cap_v is not None and int(cap_v) > 0:
                budget_total = int(cap_v)
                budget_source = "train_base_cap"
            if budget_total is None:
                offset_target_v = OFFSET_BASE_TARGET_PER_COMBO.get(combo, None)
                if offset_target_v is not None and int(offset_target_v) > 0:
                    budget_total = int(offset_target_v)
                    budget_source = "offset_base_target"
            if budget_total is None and USE_FOLD1_CLIP_POST_AUG and str(OUTPUT_MODE) == "tool_kpt_only":
                target_v = FOLD1_TARGET_TRAIN_PER_COMBO.get(combo, None)
                if target_v is not None and int(target_v) > 0:
                    budget_total = int(target_v)
                    budget_source = "final_target"

            extra_budget: Optional[int] = None
            if budget_total is not None:
                extra_budget = max(0, int(budget_total) - int(raw_n))

            selected_extra = 0
            pass_idx = 0
            while True:
                if extra_budget is not None and selected_extra >= int(extra_budget):
                    break
                added_this_pass = False
                for rec in entries:
                    offs = rec["extra_offsets"]
                    if pass_idx >= len(offs):
                        continue
                    selected_offset_tasks.append((combo, rec, int(offs[pass_idx]), int(pass_idx)))
                    selected_extra += 1
                    added_this_pass = True
                    if extra_budget is not None and selected_extra >= int(extra_budget):
                        break
                if not added_this_pass:
                    break
                pass_idx += 1

            offset_base_planned_by_combo[combo] = int(selected_extra)
            budget_str = "all" if extra_budget is None else str(int(extra_budget))
            target_str = "-" if budget_total is None else str(int(budget_total))
            print(
                f"[OFFSET][BASE][PLAN] combo={combo:20s} "
                f"raw={raw_n:4d} target={target_str:>4s} "
                f"source={budget_source:>18s} "
                f"available_extra={available_extra:4d} "
                f"selected_extra={selected_extra:4d} "
                f"budget_extra={budget_str}"
            )

        for combo, rec, off, pass_idx in selected_offset_tasks:
            base_sid = str(rec["base_sid"])
            base_item = rec["base_item"]
            ann_path = rec["ann_path"]
            seg_len = int(rec["seg_len"])
            mult = int(rec["mult"])
            sid_new = f"{next_idx:06d}"
            item_new = extract_base_sample(
                ann_path=ann_path,
                root_dir=root_dir,
                sid=sid_new,
                dirs=dirs,
                stats=stats,
                start_offset_frames=int(off),
            )
            if item_new is None:
                continue
            item_new.split = "train"
            item_new.source_sid = str(base_sid)
            item_new.aug_type = "offset_base"
            item_new.aug_params = {
                "offset_start_frames": int(off),
                "offset_total_windows": int(mult),
                "offset_seg_len": int(seg_len),
                "offset_clip_len": int(TARGET_FRAMES),
                "offset_selection_pass": int(pass_idx),
                "offset_selection_mode": "budgeted_round_robin_boundary_first",
            }
            meta.append(item_new)
            train_base_ids.append(item_new.sid)
            next_idx += 1
            offset_base_added_by_combo[combo] = int(offset_base_added_by_combo.get(combo, 0) + 1)

        if len(offset_base_added_by_combo) > 0:
            print("[OFFSET][BASE] added by combo:")
            for combo in sorted(offset_base_added_by_combo.keys()):
                print(
                    f"  - {combo:20s}: "
                    f"+{int(offset_base_added_by_combo[combo])} "
                    f"(planned={int(offset_base_planned_by_combo.get(combo, 0))}, "
                    f"available={int(offset_base_available_by_combo.get(combo, 0))})"
                )
        print(f"[OFFSET][BASE] train_base_after_offset={len(train_base_ids)}")

    # Optional hard cap by final desired total samples.
    base_total_after_split = int(len(train_base_ids) + len(val_ids) + len(test_ids))
    max_aug_from_target_total: Optional[int] = None
    if target_total_samples is not None:
        max_aug_from_target_total = max(0, int(target_total_samples) - base_total_after_split)
        print(
            f"[AUG][TARGET_TOTAL] base_total_after_split={base_total_after_split} "
            f"target_total={int(target_total_samples)} "
            f"max_aug_from_target_total={int(max_aug_from_target_total)}"
        )
    effective_max_aug_samples: Optional[int]
    if (max_aug_samples is None) and (max_aug_from_target_total is None):
        effective_max_aug_samples = None
    elif max_aug_samples is None:
        effective_max_aug_samples = int(max_aug_from_target_total)
    elif max_aug_from_target_total is None:
        effective_max_aug_samples = int(max_aug_samples)
    else:
        effective_max_aug_samples = min(int(max_aug_samples), int(max_aug_from_target_total))
    print(f"[AUG][LIMIT] effective_max_aug_samples={effective_max_aug_samples}")

    augmented_items: List[BaseItem] = []
    aug_total = 0
    aug_plan_by_combo: Dict[str, int] = {}
    aug_generated_by_combo: Dict[str, int] = {}
    aug_generated_by_type: Dict[str, int] = {}
    aug_generated_by_combo_and_type: Dict[str, Dict[str, int]] = {}
    aug_generated_by_strength: Dict[str, int] = {}
    aug_generated_by_combo_and_strength: Dict[str, Dict[str, int]] = {}
    aug_extreme_by_combo: Dict[str, Dict[str, int]] = {}
    aug_hard_pool_by_combo: Dict[str, Dict[str, float]] = {}
    aug_ops_global: Dict[str, int] = {}
    aug_ops_by_combo: Dict[str, Dict[str, int]] = {}
    aug_examples_by_combo: Dict[str, List[Dict[str, object]]] = {}
    aug_base_by_combo_before_cap: Dict[str, int] = {}
    aug_base_by_combo_after_cap: Dict[str, int] = {}
    aug_cap_applied_by_combo: Dict[str, Dict[str, int]] = {}
    fold1_aug_report: Dict[str, object] = {}
    debug_saved_keys: Set[str] = set()
    debug_fixed_pairs: List[Dict[str, object]] = []

    if USE_FOLD1_CLIP_POST_AUG and (not EXPORT_VIDEO16_NPY):
        raise ValueError("USE_FOLD1_CLIP_POST_AUG=True requires EXPORT_VIDEO16_NPY=True.")

    if ENABLE_WEAK_COMBO_AUG:
        # Generate augmentations for configured weak combos (TRAIN only).
        meta_by_sid = {it.sid: it for it in meta}

        # Optional: restrict augmentation to fold-1 train split.
        # When FOLD1_AUG_SCOPE_BY_CASE_ONLY=True, we rely on case files only.
        fold1_train_sid_set: Set[str] = set()
        fold1_val_sid_set: Set[str] = set()
        fold1_train_case_set: Set[str] = set()
        fold1_val_case_set: Set[str] = set()
        if USE_FOLD1_CLIP_POST_AUG:
            if FOLD1_TRAIN_CASES_FILE.is_file():
                fold1_train_case_set = _read_sid_set(FOLD1_TRAIN_CASES_FILE)
            if FOLD1_VAL_CASES_FILE.is_file():
                fold1_val_case_set = _read_sid_set(FOLD1_VAL_CASES_FILE)
            if FOLD1_AUG_SCOPE_BY_CASE_ONLY:
                if len(fold1_train_case_set) == 0:
                    raise FileNotFoundError(
                        f"Missing/empty fold1 train case file: {FOLD1_TRAIN_CASES_FILE}"
                    )
                print(
                    f"[AUG][FOLD1] enabled(case-only): "
                    f"train_cases={len(fold1_train_case_set)} val_cases={len(fold1_val_case_set)}"
                )
            else:
                if not FOLD1_TRAIN_SIDS_FILE.is_file():
                    raise FileNotFoundError(f"Missing fold1 train sid file: {FOLD1_TRAIN_SIDS_FILE}")
                if not FOLD1_VAL_SIDS_FILE.is_file():
                    raise FileNotFoundError(f"Missing fold1 val sid file: {FOLD1_VAL_SIDS_FILE}")
                fold1_train_sid_set = _read_sid_set(FOLD1_TRAIN_SIDS_FILE)
                fold1_val_sid_set = _read_sid_set(FOLD1_VAL_SIDS_FILE)
                print(
                    f"[AUG][FOLD1] enabled(sid): train_sids={len(fold1_train_sid_set)} "
                    f"val_sids={len(fold1_val_sid_set)}"
                )

        eligible_train_base_ids: List[str] = []
        for sid in train_base_ids:
            if sid not in meta_by_sid:
                continue
            if USE_FOLD1_CLIP_POST_AUG:
                if FOLD1_AUG_SCOPE_BY_CASE_ONLY:
                    case_name = str(meta_by_sid[sid].case)
                    if case_name not in fold1_train_case_set:
                        continue
                    if case_name in fold1_val_case_set:
                        continue
                else:
                    if sid not in fold1_train_sid_set:
                        continue
                    if sid in fold1_val_sid_set:
                        continue
            eligible_train_base_ids.append(sid)

        # Fallback by case names when sid mapping differs across builds.
        if (
            USE_FOLD1_CLIP_POST_AUG
            and (not FOLD1_AUG_SCOPE_BY_CASE_ONLY)
            and len(eligible_train_base_ids) == 0
            and len(fold1_train_case_set) > 0
        ):
            for sid in train_base_ids:
                if sid not in meta_by_sid:
                    continue
                case_name = str(meta_by_sid[sid].case)
                if case_name not in fold1_train_case_set:
                    continue
                if case_name in fold1_val_case_set:
                    continue
                eligible_train_base_ids.append(sid)
            print(
                f"[AUG][FOLD1] sid-based match empty; fallback to case-based match, "
                f"eligible_train_base={len(eligible_train_base_ids)}"
            )

        if USE_FOLD1_CLIP_POST_AUG:
            print(
                f"[AUG][FOLD1] eligible_train_base={len(eligible_train_base_ids)} "
                f"(from global_train_base={len(train_base_ids)})"
            )

        # Group train base by combo.
        for sid in eligible_train_base_ids:
            it = meta_by_sid[sid]
            combo = f"{it.tool}|{it.action}"
            aug_base_by_combo_before_cap[combo] = int(aug_base_by_combo_before_cap.get(combo, 0) + 1)

        # Optional cap on eligible TRAIN base clips (before any augmentation planning).
        if USE_FOLD1_CLIP_POST_AUG and len(FOLD1_TRAIN_BASE_CAP_PER_COMBO) > 0:
            grouped_before_cap: Dict[str, List[str]] = {}
            for sid in eligible_train_base_ids:
                it = meta_by_sid[sid]
                combo = f"{it.tool}|{it.action}"
                grouped_before_cap.setdefault(combo, []).append(sid)

            capped_ids: List[str] = []
            for combo, sids in sorted(grouped_before_cap.items()):
                cap_v = FOLD1_TRAIN_BASE_CAP_PER_COMBO.get(combo, None)
                if cap_v is None:
                    capped_ids.extend(sids)
                    continue
                cap_n = int(cap_v)
                if cap_n <= 0:
                    continue
                if len(sids) <= cap_n:
                    capped_ids.extend(sids)
                    continue

                # Deterministic sampling for stability/reproducibility.
                # For hook-dominant combo, use case-balanced sampling.
                if FOLD1_CAP_SAMPLE_BY_CASE_BALANCED and combo == "hook|dissect":
                    kept = _sample_sids_balanced_by_case(
                        sids=sids,
                        sid2item=meta_by_sid,
                        cap_n=cap_n,
                        combo=combo,
                        seed=int(SEED),
                    )
                else:
                    sids_work = sorted(sids)
                    rng_cap = random.Random(int(SEED) + 1701 + sum(ord(ch) for ch in combo))
                    rng_cap.shuffle(sids_work)
                    kept = sorted(sids_work[:cap_n])
                dropped = int(len(sids) - len(kept))
                aug_cap_applied_by_combo[combo] = {
                    "before": int(len(sids)),
                    "after": int(len(kept)),
                    "dropped": int(dropped),
                    "cap": int(cap_n),
                }
                capped_ids.extend(kept)

            eligible_train_base_ids = sorted(capped_ids)
            if len(aug_cap_applied_by_combo) > 0:
                print("[AUG][CAP] fold1 train base cap applied:")
                for combo in sorted(aug_cap_applied_by_combo.keys()):
                    r = aug_cap_applied_by_combo[combo]
                    print(
                        f"  - {combo:20s}: before={r['before']} after={r['after']} "
                        f"dropped={r['dropped']} cap={r['cap']}"
                    )

            if FOLD1_APPLY_CAP_TO_FINAL_TRAIN:
                kept_train_set = set(eligible_train_base_ids)
                old_train_n = int(len(train_base_ids))
                train_base_ids = sorted([sid for sid in train_base_ids if sid in kept_train_set])
                if len(train_base_ids) != old_train_n:
                    print(
                        f"[AUG][CAP] applied-to-final-train: "
                        f"before={old_train_n} after={len(train_base_ids)} "
                        f"dropped={old_train_n - len(train_base_ids)}"
                    )

        for sid in eligible_train_base_ids:
            it = meta_by_sid[sid]
            combo = f"{it.tool}|{it.action}"
            aug_base_by_combo_after_cap[combo] = int(aug_base_by_combo_after_cap.get(combo, 0) + 1)

        # Group train base by combo.
        train_base_by_combo: Dict[str, List[str]] = {}
        for sid in eligible_train_base_ids:
            it = meta_by_sid[sid]
            combo = f"{it.tool}|{it.action}"
            train_base_by_combo.setdefault(combo, []).append(sid)

        donor_pool_all: List[str] = sorted(eligible_train_base_ids)
        donor_pool_by_combo: Dict[str, List[str]] = {}
        for sid in donor_pool_all:
            it = meta_by_sid[sid]
            combo = f"{it.tool}|{it.action}"
            donor_pool_by_combo.setdefault(combo, []).append(sid)

        print(
            f"[AUG][PLAN] combos_in_scope={len(train_base_by_combo)} "
            f"donor_pool={len(donor_pool_all)} "
            f"log_every={AUG_LOG_EVERY_N_SAMPLES}"
        )
        for combo in sorted(train_base_by_combo.keys()):
            print(f"[AUG][PLAN][BASE] combo={combo:20s} base_train={len(train_base_by_combo[combo])}")

        stop_aug_early = False
        for combo, sids in sorted(train_base_by_combo.items()):
            if stop_aug_early:
                break
            # deterministic order
            sids_sorted = sorted(sids)
            per_sample_counts = planned_aug_counts_for_combo(combo, len(sids_sorted))
            if len(per_sample_counts) == 0:
                continue
            # Expand per-base counts into per-task list to support:
            #   1) strength scheduling with hard-pool constraints
            #   2) donor diversity hints for high-k combos
            task_base_sids: List[str] = []
            task_donor_hints: List[Optional[str]] = []
            for base_sid, k in zip(sids_sorted, per_sample_counts):
                kk = int(max(0, k))
                if kk <= 0:
                    continue
                donor_seq = _build_donor_hint_sequence(
                    base_sid=base_sid,
                    k=kk,
                    donor_pool_all=donor_pool_all,
                    seed=int(SEED) + int(len(sids_sorted)) + sum(ord(ch) for ch in str(combo)),
                )
                for j in range(kk):
                    task_base_sids.append(str(base_sid))
                    if j < len(donor_seq):
                        task_donor_hints.append(donor_seq[j])
                    else:
                        task_donor_hints.append(None)

            planned_extra = int(len(task_base_sids))
            if planned_extra <= 0:
                continue
            aug_plan_by_combo[combo] = planned_extra
            aug_generated_by_combo.setdefault(combo, 0)
            if FOLD1_USE_V3_STRENGTH_MIX:
                strength_schedule = _plan_strength_buckets_for_combo(
                    combo=combo,
                    n_samples=planned_extra,
                    seed=int(SEED) + int(len(sids_sorted)),
                )
            else:
                strength_schedule = ["medium" for _ in range(planned_extra)]
            hard_sid_set = _select_hard_sid_set_for_combo(
                combo=combo,
                sids=sids_sorted,
                sid2item=meta_by_sid,
                top_frac=float(FOLD1_EXTREME_HARD_TOP_FRAC),
                min_keep=int(FOLD1_EXTREME_HARD_MIN_SAMPLES),
            )
            aug_hard_pool_by_combo[combo] = {
                "pool_size": float(len(hard_sid_set)),
                "pool_frac": float(len(hard_sid_set) / max(1, len(sids_sorted))),
            }
            highdiff_fix_stats: Dict[str, int] = {}
            if FOLD1_EXTREME_ONLY_HARD:
                strength_schedule, extreme_fix_stats = _enforce_extreme_on_hard_tasks(
                    combo=combo,
                    task_base_sids=task_base_sids,
                    strength_schedule=strength_schedule,
                    hard_sid_set=hard_sid_set,
                    rng_py=rng_py,
                )
                aug_extreme_by_combo.setdefault(combo, {})
                aug_extreme_by_combo[combo]["fix_extreme_before"] = int(extreme_fix_stats.get("extreme_before", 0))
                aug_extreme_by_combo[combo]["fix_extreme_after"] = int(extreme_fix_stats.get("extreme_after", 0))
                aug_extreme_by_combo[combo]["fix_moved_nonhard"] = int(extreme_fix_stats.get("moved_nonhard_to_heavy", 0))
            if FOLD1_FORCE_HIGH_DIFFICULTY_ONLY:
                strength_schedule, highdiff_fix_stats = _enforce_high_difficulty_only_schedule(
                    combo=combo,
                    task_base_sids=task_base_sids,
                    strength_schedule=strength_schedule,
                    hard_sid_set=hard_sid_set,
                    rng_py=rng_py,
                )
                aug_extreme_by_combo.setdefault(combo, {})
                for k, v in highdiff_fix_stats.items():
                    aug_extreme_by_combo[combo][f"highdiff_{k}"] = int(v)
            strength_plan_counter: Dict[str, int] = {}
            for b in strength_schedule:
                strength_plan_counter[str(b)] = int(strength_plan_counter.get(str(b), 0) + 1)
            if USE_FOLD1_CLIP_POST_AUG:
                target = int(FOLD1_TARGET_TRAIN_PER_COMBO.get(combo, len(sids_sorted)))
                report_tail = f"target={target}"
            else:
                report_tail = f"ratio={WEAK_COMBO_AUG_RATIO.get(combo, 0.0):.3f}"
            print(
                f"[AUG] combo={combo} train_base={len(sids_sorted)} "
                f"aug_extra={planned_extra} "
                f"{report_tail}"
            )
            if len(strength_plan_counter) > 0:
                print(
                    f"[AUG] combo={combo} strength_plan="
                    f"light:{strength_plan_counter.get('light', 0)} "
                    f"medium:{strength_plan_counter.get('medium', 0)} "
                    f"heavy:{strength_plan_counter.get('heavy', 0)} "
                    f"extreme:{strength_plan_counter.get('extreme', 0)}"
                )

            for task_idx, base_sid in enumerate(task_base_sids):
                if stop_aug_early:
                    break
                base_item = meta_by_sid[base_sid]
                if (effective_max_aug_samples is not None) and (aug_total >= effective_max_aug_samples):
                    stop_aug_early = True
                    break
                strength_bucket = (
                    str(strength_schedule[task_idx])
                    if task_idx < len(strength_schedule) else "medium"
                )
                donor_sid_hint = (
                    task_donor_hints[task_idx]
                    if task_idx < len(task_donor_hints) else None
                )
                aug_sid = f"{next_idx:06d}"
                next_idx += 1

                aug_item, _ = generate_one_aug(
                    base_item=base_item,
                    base_sid=base_sid,
                    aug_sid=aug_sid,
                    dirs=dirs,
                    rng_py=rng_py,
                    rng_np=rng_np,
                    debug_dir=debug_dir,
                    donor_pool_by_combo=donor_pool_by_combo,
                    donor_pool_all=donor_pool_all,
                    donor_sid_hint=donor_sid_hint,
                    debug_saved_keys=debug_saved_keys,
                    debug_fixed_dir=debug_fixed_dir,
                    debug_fixed_pairs=debug_fixed_pairs,
                    strength_bucket=strength_bucket,
                )
                augmented_items.append(aug_item)
                aug_total += 1
                aug_generated_by_combo[combo] = aug_generated_by_combo.get(combo, 0) + 1
                aug_type = str(aug_item.aug_type) if aug_item.aug_type is not None else "none"
                aug_generated_by_type[aug_type] = aug_generated_by_type.get(aug_type, 0) + 1
                _inc_nested_counter(aug_generated_by_combo_and_type, combo, aug_type, 1)
                aug_generated_by_strength[strength_bucket] = int(
                    aug_generated_by_strength.get(strength_bucket, 0) + 1
                )
                _inc_nested_counter(aug_generated_by_combo_and_strength, combo, strength_bucket, 1)
                if str(strength_bucket) == "extreme":
                    rec_ext = aug_extreme_by_combo.setdefault(combo, {})
                    rec_ext["generated_total"] = int(rec_ext.get("generated_total", 0) + 1)
                    if base_sid in hard_sid_set:
                        rec_ext["generated_from_hard"] = int(rec_ext.get("generated_from_hard", 0) + 1)

                params_dict = aug_item.aug_params if isinstance(aug_item.aug_params, dict) else {}
                crop_mode = str(params_dict.get("crop_mode", "unknown"))
                _inc_nested_counter(aug_ops_by_combo, combo, f"crop:{crop_mode}", 1)
                aug_ops_global[f"crop:{crop_mode}"] = int(aug_ops_global.get(f"crop:{crop_mode}", 0) + 1)
                if "temp_warp_ratio" in params_dict:
                    _inc_nested_counter(aug_ops_by_combo, combo, "temporal_warp", 1)
                    aug_ops_global["temporal_warp"] = int(aug_ops_global.get("temporal_warp", 0) + 1)
                if bool(params_dict.get("hflip", False)):
                    _inc_nested_counter(aug_ops_by_combo, combo, "hflip", 1)
                    aug_ops_global["hflip"] = int(aug_ops_global.get("hflip", 0) + 1)
                if params_dict.get("affine_M", None) is not None:
                    _inc_nested_counter(aug_ops_by_combo, combo, "affine", 1)
                    aug_ops_global["affine"] = int(aug_ops_global.get("affine", 0) + 1)
                if bool(params_dict.get("bg_replace", False)):
                    _inc_nested_counter(aug_ops_by_combo, combo, "bg_replace", 1)
                    aug_ops_global["bg_replace"] = int(aug_ops_global.get("bg_replace", 0) + 1)
                    if bool(params_dict.get("bg_same_combo", False)):
                        _inc_nested_counter(aug_ops_by_combo, combo, "bg_same_combo", 1)
                        aug_ops_global["bg_same_combo"] = int(aug_ops_global.get("bg_same_combo", 0) + 1)
                    else:
                        _inc_nested_counter(aug_ops_by_combo, combo, "bg_cross_combo", 1)
                        aug_ops_global["bg_cross_combo"] = int(aug_ops_global.get("bg_cross_combo", 0) + 1)

                color_info = params_dict.get("color", {}) if isinstance(params_dict.get("color", {}), dict) else {}
                if bool(color_info.get("enabled", False)):
                    _inc_nested_counter(aug_ops_by_combo, combo, "color_enabled", 1)
                    aug_ops_global["color_enabled"] = int(aug_ops_global.get("color_enabled", 0) + 1)
                if bool(color_info.get("do_gamma", False)):
                    _inc_nested_counter(aug_ops_by_combo, combo, "color_gamma", 1)
                    aug_ops_global["color_gamma"] = int(aug_ops_global.get("color_gamma", 0) + 1)
                if bool(color_info.get("do_blur", False)):
                    _inc_nested_counter(aug_ops_by_combo, combo, "color_blur", 1)
                    aug_ops_global["color_blur"] = int(aug_ops_global.get("color_blur", 0) + 1)
                if bool(color_info.get("do_noise", False)):
                    _inc_nested_counter(aug_ops_by_combo, combo, "color_noise", 1)
                    aug_ops_global["color_noise"] = int(aug_ops_global.get("color_noise", 0) + 1)
                if bool(color_info.get("do_jpeg", False)):
                    _inc_nested_counter(aug_ops_by_combo, combo, "color_jpeg", 1)
                    aug_ops_global["color_jpeg"] = int(aug_ops_global.get("color_jpeg", 0) + 1)
                for opn in params_dict.get("event_ops", []):
                    op_key = f"event_{str(opn)}"
                    _inc_nested_counter(aug_ops_by_combo, combo, op_key, 1)
                    aug_ops_global[op_key] = int(aug_ops_global.get(op_key, 0) + 1)

                ex_list = aug_examples_by_combo.setdefault(combo, [])
                if len(ex_list) < 20:
                    ex_list.append(
                        {
                            "sid": str(aug_sid),
                            "source_sid": str(base_sid),
                            "aug_type": aug_type,
                            "params": params_dict,
                        }
                    )

                if (
                    AUG_LOG_EVERY_N_SAMPLES > 0
                    and (aug_total % AUG_LOG_EVERY_N_SAMPLES == 0)
                ):
                    planned_total = int(sum(aug_plan_by_combo.values()))
                    print(
                        f"[AUG][PROGRESS] generated={aug_total}/{planned_total} "
                        f"latest_combo={combo} latest_sid={aug_sid}"
                    )

            print(
                f"[AUG][DONE][COMBO] combo={combo:20s} "
                f"generated={aug_generated_by_combo.get(combo, 0)} "
                f"planned={aug_plan_by_combo.get(combo, 0)}"
            )
            if stop_aug_early:
                print(
                    f"[AUG][STOP] reached effective_max_aug_samples={effective_max_aug_samples}; "
                    f"early stop augmentation at combo={combo}"
                )

        if len(aug_plan_by_combo) > 0:
            print("[AUG][SUMMARY] per-combo generated/planned:")
            for combo in sorted(aug_plan_by_combo.keys()):
                print(
                    f"  - {combo:20s}: "
                    f"{aug_generated_by_combo.get(combo, 0)}/{aug_plan_by_combo.get(combo, 0)}"
                )
            print("[AUG][SUMMARY] aug_type counts:")
            for aug_type in sorted(aug_generated_by_type.keys()):
                print(f"  - {aug_type}: {aug_generated_by_type[aug_type]}")
            if len(aug_generated_by_strength) > 0:
                print("[AUG][SUMMARY] strength counts:")
                for sb in ("light", "medium", "heavy", "extreme"):
                    if sb in aug_generated_by_strength:
                        print(f"  - {sb}: {aug_generated_by_strength[sb]}")
                n_light = int(aug_generated_by_strength.get("light", 0))
                n_medium = int(aug_generated_by_strength.get("medium", 0))
                n_heavy = int(aug_generated_by_strength.get("heavy", 0))
                n_extreme = int(aug_generated_by_strength.get("extreme", 0))
                n_total = int(max(1, n_light + n_medium + n_heavy + n_extreme))
                n_high = int(n_heavy + n_extreme)
                print(
                    f"[AUG][SUMMARY] high_difficulty_ratio={n_high}/{n_total} "
                    f"({100.0 * float(n_high) / float(n_total):.2f}%)"
                )
            print(
                f"[AUG][SUMMARY] total_generated={aug_total} "
                f"total_planned={sum(aug_plan_by_combo.values())}"
            )
            if SAVE_DEBUG_AUG_IMAGES:
                print(
                    f"[AUG][SUMMARY] debug_examples_saved={len(debug_saved_keys)} "
                    f"keys={sorted(debug_saved_keys)}"
                )

        if USE_FOLD1_CLIP_POST_AUG:
            fold1_aug_report = {
                "scope_mode": "case_only" if FOLD1_AUG_SCOPE_BY_CASE_ONLY else "sid",
                "aug_series_name": str(FOLD1_AUG_SERIES_NAME),
                "fold1_train_cases_file": str(FOLD1_TRAIN_CASES_FILE),
                "fold1_val_cases_file": str(FOLD1_VAL_CASES_FILE),
                "fold1_train_sids_file": str(FOLD1_TRAIN_SIDS_FILE),
                "fold1_val_sids_file": str(FOLD1_VAL_SIDS_FILE),
                "base_by_combo_before_cap": {str(k): int(v) for k, v in sorted(aug_base_by_combo_before_cap.items())},
                "base_by_combo_after_cap": {str(k): int(v) for k, v in sorted(aug_base_by_combo_after_cap.items())},
                "base_cap_per_combo": {str(k): int(v) for k, v in sorted(FOLD1_TRAIN_BASE_CAP_PER_COMBO.items())},
                "cap_sample_by_case_balanced": bool(FOLD1_CAP_SAMPLE_BY_CASE_BALANCED),
                "cap_applied_to_final_train": bool(FOLD1_APPLY_CAP_TO_FINAL_TRAIN),
                "base_cap_applied_by_combo": {str(k): v for k, v in sorted(aug_cap_applied_by_combo.items())},
                "target_train_per_combo": {str(k): int(v) for k, v in sorted(FOLD1_TARGET_TRAIN_PER_COMBO.items())},
                "skip_aug_combos": sorted([str(x) for x in FOLD1_SKIP_AUG_COMBOS]),
                "use_v3_strength_mix": bool(FOLD1_USE_V3_STRENGTH_MIX),
                "strength_mix_by_combo": {
                    str(k): {str(kk): float(vv) for kk, vv in sorted(v.items())}
                    for k, v in sorted(FOLD1_STRENGTH_MIX_BY_COMBO.items())
                },
                "strength_preset_overrides_by_combo": {
                    str(combo): {
                        str(bucket): {str(k): float(v) for k, v in sorted(vals.items())}
                        for bucket, vals in sorted(bucket_map.items())
                    }
                    for combo, bucket_map in sorted(FOLD1_STRENGTH_PRESET_OVERRIDES_BY_COMBO.items())
                },
                "offset_base_multiplier_by_combo": {
                    str(k): int(v) for k, v in sorted(OFFSET_BASE_MULTIPLIER_BY_COMBO.items())
                },
                "offset_base_target_per_combo": {
                    str(k): int(v) for k, v in sorted(OFFSET_BASE_TARGET_PER_COMBO.items())
                },
                "planned_extra_by_combo": {str(k): int(v) for k, v in sorted(aug_plan_by_combo.items())},
                "generated_extra_by_combo": {str(k): int(v) for k, v in sorted(aug_generated_by_combo.items())},
                "generated_by_type": {str(k): int(v) for k, v in sorted(aug_generated_by_type.items())},
                "generated_by_strength": {str(k): int(v) for k, v in sorted(aug_generated_by_strength.items())},
                "generated_by_combo_and_type": {
                    str(k): {str(kk): int(vv) for kk, vv in sorted(v.items())}
                    for k, v in sorted(aug_generated_by_combo_and_type.items())
                },
                "generated_by_combo_and_strength": {
                    str(k): {str(kk): int(vv) for kk, vv in sorted(v.items())}
                    for k, v in sorted(aug_generated_by_combo_and_strength.items())
                },
                "extreme_hard_pool_by_combo": {
                    str(k): {str(kk): float(vv) for kk, vv in sorted(v.items())}
                    for k, v in sorted(aug_hard_pool_by_combo.items())
                },
                "extreme_generation_by_combo": {
                    str(k): {str(kk): int(vv) for kk, vv in sorted(v.items())}
                    for k, v in sorted(aug_extreme_by_combo.items())
                },
                "ops_global": {str(k): int(v) for k, v in sorted(aug_ops_global.items())},
                "ops_by_combo": {
                    str(k): {str(kk): int(vv) for kk, vv in sorted(v.items())}
                    for k, v in sorted(aug_ops_by_combo.items())
                },
                "examples_by_combo": {str(k): v for k, v in sorted(aug_examples_by_combo.items())},
                "debug_fixed_pairs": {
                    "enabled": bool(SAVE_DEBUG_AUG_IMAGES),
                    "dir": str(debug_fixed_dir),
                    "target_count": int(FOLD1_DEBUG_FIXED_PAIR_COUNT),
                    "saved_count": int(len(debug_fixed_pairs)),
                    "pairs": list(debug_fixed_pairs),
                },
                "high_difficulty_policy": {
                    "enabled": bool(FOLD1_FORCE_HIGH_DIFFICULTY_ONLY),
                    "min_extreme_priority": float(FOLD1_FORCE_HIGH_DIFFICULTY_MIN_EXTREME_PRIORITY),
                    "min_extreme_default": float(FOLD1_FORCE_HIGH_DIFFICULTY_MIN_EXTREME_DEFAULT),
                },
                "totals": {
                    "eligible_train_base_before_cap": int(sum(aug_base_by_combo_before_cap.values())),
                    "eligible_train_base_after_cap": int(sum(aug_base_by_combo_after_cap.values())),
                    "planned_extra_total": int(sum(aug_plan_by_combo.values())),
                    "generated_extra_total": int(aug_total),
                    "generated_high_difficulty_total": int(
                        aug_generated_by_strength.get("heavy", 0) + aug_generated_by_strength.get("extreme", 0)
                    ),
                },
            }

    # Final train ids include augmented ids.
    train_ids = sorted(train_base_ids + [it.sid for it in augmented_items])

    # Set split for augmented items explicitly (train)
    for it in augmented_items:
        it.split = "train"

    # Merge meta
    meta_all: List[BaseItem] = meta + augmented_items

    # =========================
    # Collect stats (TRAIN BASE ONLY)
    # =========================
    motion_all = []  # (N, 6)
    motion_l2 = []  # (N,)
    contact_area_all = []
    tool_area_all = []
    center_dist_all = []
    edge_cos_all = []
    keypoint_vis_all = []  # each: (T=16,3)
    contact_vis_all = []   # each: (T=16,4)

    for sid in train_base_ids:
        mot = np.load(dirs["MOTION_DIR"] / f"{sid}.npy")  # (L,6)
        kpt_vis = np.load(dirs["KEYPOINT_VIS_DIR"] / f"{sid}.npy")  # (16,3)
        ct_vis = np.load(dirs["CONTACT_VIS_DIR"] / f"{sid}.npy")  # (16,4) raw visibility

        # ---------- motion ----------
        motion_all.append(mot.reshape(-1, 6))

        mot_pts = mot.reshape(-1, 3, 2)
        motion_l2.append(np.linalg.norm(mot_pts, axis=2).reshape(-1))
        keypoint_vis_all.append(kpt_vis.reshape(-1, 3).astype(np.float32))
        contact_vis_all.append(ct_vis.reshape(-1, 4).astype(np.float32))

        if EXPORT_GEO_FEATURES:
            contact_area_all.append(np.load(dirs["CONTACT_AREA_DIR"] / f"{sid}.npy")[0])
            tool_area_all.append(np.load(dirs["TOOL_AREA_DIR"] / f"{sid}.npy")[0])
            center_dist_all.append(np.load(dirs["CENTER_DIST_DIR"] / f"{sid}.npy")[0])
            edge_cos_all.append(np.load(dirs["EDGE_COS_DIR"] / f"{sid}.npy")[0])

    # Compute Mean/Std on TRAIN only (delta motions)
    # rs = RunningStats(D=6)
    # for sid in train_ids:
    #     mot = np.load(dirs["MOTION_DIR"] / f"{sid}.npy")
    #     if mot.ndim != 2 or mot.shape[1] != 6:
    #         raise RuntimeError(f"Bad motion shape for {sid}: {mot.shape}")
    #     rs.update(mot.astype(np.float32))
    # 下面这个是不做data aug的！
    # =========================================================
    # Compute Mean/Std on BASE TRAIN ONLY (NO AUG)
    # =========================================================
    # rs = RunningStats(D=6)

    # mean, std = rs.finalize(eps=1e-8)
    motion_all = np.concatenate(motion_all, axis=0)  # (N,6)
    motion_l2 = np.concatenate(motion_l2, axis=0)
    keypoint_vis_all = np.concatenate(keypoint_vis_all, axis=0)  # (N*16, 3)
    contact_vis_all = np.concatenate(contact_vis_all, axis=0)    # (N*16, 4)

    mean = motion_all.mean(axis=0)
    std = motion_all.std(axis=0) + 1e-8

    np.save(out_root / "Mean.npy", mean.astype(np.float32))
    np.save(out_root / "Std.npy", std.astype(np.float32))

    # =========================
    # Print distribution stats
    # =========================
    print("\n========== DISTRIBUTION STATS (TRAIN BASE) ==========")

    # print("\n-- Motion step-delta (pixel) --")
    # if MOTION_REP == "stepdelta":
    #     print("\n-- Motion step-delta --")
    # else:
    #     print("\n-- Motion relative-to-first (cumulative) --")
    if MOTION_REP == "stepdelta":
        print("\n-- Motion step-delta --")
    else:
        print("\n-- Motion relative-to-first (cumulative) --")

    # ===== motion normalized (NOT saved) =====
    motion_norm = (motion_all - mean) / std

    print("\n-- Motion step-delta (NORMALIZED, NOT stored) --")
    for i in range(6):
        print_percentiles(f"motion_norm_dim[{i}]", motion_norm[:, i], ps=(5, 50, 95))

    # L2 per-point after norm (still 3 points, each has (x,y))
    motion_norm_pts = motion_norm.reshape(-1, 3, 2)
    motion_norm_l2 = np.linalg.norm(motion_norm_pts, axis=2).reshape(-1)
    print_percentiles("motion_norm_L2", motion_norm_l2, ps=(5, 50, 95))
    for i in range(6):
        print_percentiles(f"motion_dim[{i}]", motion_all[:, i])

    print_percentiles("motion_L2", motion_l2)
    print("\n-- Keypoint visibility (from sampled 16f clips) --")
    vis_ratio_per_point = keypoint_vis_all.mean(axis=0)
    vis_points_per_frame = keypoint_vis_all.sum(axis=1)
    ct_vis_ratio_per_point = contact_vis_all.mean(axis=0)
    ct_vis_points_per_frame = contact_vis_all.sum(axis=1)
    print(f"vis_ratio anchor/tip dims: {vis_ratio_per_point.tolist()}")
    print_percentiles("visible_points_per_frame", vis_points_per_frame, ps=(5, 50, 95))
    print(f"vis_ratio contact4 dims: {ct_vis_ratio_per_point.tolist()}")
    print_percentiles("visible_contact_points_per_frame", ct_vis_points_per_frame, ps=(5, 50, 95))

    if EXPORT_GEO_FEATURES:
        print("\n-- Geo features (t=0) --")
        print("\n-- Geo features (RAW, stored as-is in dataset) --")
        print_percentiles("contact_area_raw (pixel^2)", contact_area_all, ps=(5, 50, 95))
        print_percentiles("tool_area_raw (pixel^2)", tool_area_all, ps=(5, 50, 95))
        print_percentiles("center_dist_raw (pixel)", center_dist_all, ps=(5, 50, 95))
        print_percentiles("edge_cos_raw (unitless)", edge_cos_all, ps=(5, 50, 95))

        if PRINT_NORMALIZED_GEO:
            # ===== NOTE: geo normalization is ONLY for printing (NOT saved to dataset) =====
            contact_area_all = np.asarray(contact_area_all, dtype=np.float64)
            tool_area_all = np.asarray(tool_area_all, dtype=np.float64)
            center_dist_all = np.asarray(center_dist_all, dtype=np.float64)
            edge_cos_all = np.asarray(edge_cos_all, dtype=np.float64)

            geo_mean = {
                "contact_area": float(contact_area_all.mean()),
                "tool_area": float(tool_area_all.mean()),
                "center_dist": float(center_dist_all.mean()),
                "edge_cos": float(edge_cos_all.mean()),
            }
            geo_std = {
                "contact_area": float(contact_area_all.std() + 1e-8),
                "tool_area": float(tool_area_all.std() + 1e-8),
                "center_dist": float(center_dist_all.std() + 1e-8),
                "edge_cos": float(edge_cos_all.std() + 1e-8),
            }

            # normalized (NOT saved)
            contact_area_norm = (contact_area_all - geo_mean["contact_area"]) / geo_std["contact_area"]
            tool_area_norm = (tool_area_all - geo_mean["tool_area"]) / geo_std["tool_area"]
            center_dist_norm = (center_dist_all - geo_mean["center_dist"]) / geo_std["center_dist"]
            edge_cos_norm = (edge_cos_all - geo_mean["edge_cos"]) / geo_std["edge_cos"]

            print("\n-- Geo features (NORMALIZED, NOT stored; for inspection only) --")
            print_percentiles("contact_area_norm", contact_area_norm, ps=(5, 50, 95))
            print_percentiles("tool_area_norm", tool_area_norm, ps=(5, 50, 95))
            print_percentiles("center_dist_norm", center_dist_norm, ps=(5, 50, 95))
            print_percentiles("edge_cos_norm", edge_cos_norm, ps=(5, 50, 95))
            print("\n-- Geo norm params (train-base only, for training-time transforms) --")
            print(
                f"[GEO_NORM] mean: contact_area={geo_mean['contact_area']:.6g}, "
                f"tool_area={geo_mean['tool_area']:.6g}, center_dist={geo_mean['center_dist']:.6g}, "
                f"edge_cos={geo_mean['edge_cos']:.6g}"
            )
            print(
                f"[GEO_NORM] std : contact_area={geo_std['contact_area']:.6g}, "
                f"tool_area={geo_std['tool_area']:.6g}, center_dist={geo_std['center_dist']:.6g}, "
                f"edge_cos={geo_std['edge_cos']:.6g}"
            )

    print("====================================================\n")
    # =========================================================
    # Sanity check: step-delta / normalize / inverse
    # =========================================================
    # run_sanity_checks(out_root, max_check=50)
    sid2wh = {it.sid: (it.W, it.H) for it in meta_all}
    run_sanity_checks(out_root, sid2wh=sid2wh, max_check=50)

    vis_summary = {
        "target_frames": int(TARGET_FRAMES),
        "min_vis_frames_per_point": int(MIN_VIS_FRAMES_PER_POINT),
        "min_vis_total": int(MIN_VIS_TOTAL),
        "interpolate_vis0_to_full_supervision": bool(INTERPOLATE_VIS0_TO_FULL_SUPERVISION),
        "soft_vis_min_weight": float(SOFT_VIS_MIN_WEIGHT),
        "pseudo_supervision_enabled": bool(ENABLE_PSEUDO_SUPERVISION),
        "pseudo_weight": float(PSEUDO_WEIGHT),
        "train_base_samples": int(len(train_base_ids)),
        "vis_ratio_per_point": [float(x) for x in vis_ratio_per_point.tolist()],
        "contact_vis_ratio_per_point": [float(x) for x in ct_vis_ratio_per_point.tolist()],
        "visible_points_per_frame_percentiles": {
            "p5": float(np.percentile(vis_points_per_frame, 5)),
            "p50": float(np.percentile(vis_points_per_frame, 50)),
            "p95": float(np.percentile(vis_points_per_frame, 95)),
        },
        "visible_contact_points_per_frame_percentiles": {
            "p5": float(np.percentile(ct_vis_points_per_frame, 5)),
            "p50": float(np.percentile(ct_vis_points_per_frame, 50)),
            "p95": float(np.percentile(ct_vis_points_per_frame, 95)),
        },
    }
    (out_root / "vis_summary_train.json").write_text(
        json.dumps(vis_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write split files
    (out_root / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (out_root / "val.txt").write_text("\n".join(sorted(val_ids)) + "\n", encoding="utf-8")
    (out_root / "test.txt").write_text("\n".join(sorted(test_ids)) + "\n", encoding="utf-8")

    # 2026/02/02
    # =========================================================
    # (NEW) Val summary files: which val is which tool/action
    # =========================================================
    sid2item = {it.sid: it for it in meta_all}

    # 1) Flat table
    val_summary_path = out_root / "val_summary.tsv"
    with open(val_summary_path, "w", encoding="utf-8") as f:
        f.write(
            "sid\tsplit\ttool\taction\tcombo\tcase\tann_relpath\t"
            "abs_start\tabs_end\taction_start_abs\taction_end_abs\taction_range_mask\n"
        )
        for sid in sorted(val_ids):
            it = sid2item[sid]
            combo = f"{it.tool}|{it.action}"
            f.write(
                f"{it.sid}\t{it.split}\t{it.tool}\t{it.action}\t{combo}\t{it.case}\t{it.ann_relpath}\t"
                f"{it.abs_start}\t{it.abs_end}\t{it.action_start_abs}\t{it.action_end_abs}\t{it.action_range_mask}\n"
            )

    # 2) Grouped json: by combo/tool/action
    val_by_combo: Dict[str, List[str]] = {}
    val_by_tool: Dict[str, List[str]] = {}
    val_by_action: Dict[str, List[str]] = {}

    for sid in sorted(val_ids):
        it = sid2item[sid]
        combo = f"{it.tool}|{it.action}"
        val_by_combo.setdefault(combo, []).append(sid)
        val_by_tool.setdefault(it.tool, []).append(sid)
        val_by_action.setdefault(it.action, []).append(sid)

    (out_root / "val_by_combo.json").write_text(json.dumps(val_by_combo, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_root / "val_by_tool.json").write_text(json.dumps(val_by_tool, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_root / "val_by_action.json").write_text(json.dumps(val_by_action, indent=2, ensure_ascii=False), encoding="utf-8")

    if USE_FOLDER_SPLIT_10_1:
        selected_case_set = set(selected_case_folders_runtime)
        val_case_set = set(val_case_folders_runtime)

        case_split_actual: Dict[str, str] = {}
        case_combo_counts: Dict[str, Dict[str, int]] = {}
        case_total_counts: Dict[str, int] = {}
        case_used_counts: Dict[str, int] = {}
        case_has_train: Dict[str, bool] = {}
        case_has_val: Dict[str, bool] = {}
        case_has_test: Dict[str, bool] = {}
        for it in meta:
            if it.case not in selected_case_set:
                continue
            case_total_counts[it.case] = case_total_counts.get(it.case, 0) + 1
            if it.split in ("train", "val", "test"):
                case_used_counts[it.case] = case_used_counts.get(it.case, 0) + 1
            if it.split == "train":
                case_has_train[it.case] = True
            elif it.split == "val":
                case_has_val[it.case] = True
            elif it.split == "test":
                case_has_test[it.case] = True
            combo = f"{it.tool}|{it.action}"
            cc = case_combo_counts.setdefault(it.case, {})
            cc[combo] = cc.get(combo, 0) + 1

        for case_name in sorted(case_total_counts.keys()):
            has_tr = bool(case_has_train.get(case_name, False))
            has_va = bool(case_has_val.get(case_name, False))
            if has_tr and has_va:
                case_split_actual[case_name] = "mixed"
            elif has_va:
                case_split_actual[case_name] = "val"
            elif has_tr:
                case_split_actual[case_name] = "train"
            elif bool(case_has_test.get(case_name, False)):
                case_split_actual[case_name] = "test"
            else:
                # selected but fully dropped after caps/filters
                case_split_actual[case_name] = "drop"

        train_case_names = sorted([k for k, v in case_split_actual.items() if v in ("train", "mixed")])
        val_case_names = sorted([k for k, v in case_split_actual.items() if v in ("val", "mixed")])
        mixed_case_names = sorted([k for k, v in case_split_actual.items() if v == "mixed"])
        report = {
            "mode": "folder_split_selected_cases",
            "selected_case_folders_config": list(selected_case_folders_runtime),
            "val_case_folders_config": list(val_case_folders_runtime),
            "val_backfill_from_train_pool": bool(FOLDER_SPLIT_VAL_BACKFILL),
            "train_case_folders_actual": train_case_names,
            "val_case_folders_actual": val_case_names,
            "mixed_case_folders_actual": mixed_case_names,
            "selected_case_count": int(len(case_total_counts)),
            "train_case_count": int(len(train_case_names)),
            "val_case_count": int(len(val_case_names)),
            "mixed_case_count": int(len(mixed_case_names)),
            "train_sample_count": int(len(train_base_ids)),
            "val_sample_count": int(len(val_ids)),
            "test_sample_count": int(len(test_ids)),
            "unused_selected_sample_count": int(max(len(meta) - len(train_base_ids) - len(val_ids) - len(test_ids), 0)),
            "per_case_split_usage": {k: str(v) for k, v in sorted(case_split_actual.items(), key=lambda kv: kv[0])},
            "per_case_used_samples": {k: int(case_used_counts.get(k, 0)) for k in sorted(case_total_counts.keys())},
            "per_case_total_samples": {k: int(case_total_counts.get(k, 0)) for k in sorted(case_total_counts.keys())},
            "per_case_combo_counts": {
                k: {kk: int(vv) for kk, vv in sorted(case_combo_counts.get(k, {}).items(), key=lambda kv: kv[0])}
                for k in sorted(case_combo_counts.keys())
            },
        }
        (out_root / "selected_case_folders.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        tsv_path = out_root / "selected_case_folders.tsv"
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("case\tsplit_usage\tused_samples\ttotal_samples\tcombo_counts\n")
            for case_name in sorted(case_split_actual.keys()):
                cdict = report["per_case_combo_counts"].get(case_name, {})
                combo_str = "; ".join([f"{k}:{v}" for k, v in cdict.items()])
                f.write(
                    f"{case_name}\t{case_split_actual[case_name]}\t"
                    f"{int(case_used_counts.get(case_name, 0))}\t"
                    f"{int(case_total_counts.get(case_name, 0))}\t{combo_str}\n"
                )
        print(f"[SPLIT] wrote: selected_case_folders.json, {tsv_path.name}")

    print(f"[VAL] wrote: {val_summary_path.name}, val_by_combo.json, val_by_tool.json, val_by_action.json")
    # 20206/02/02

    # labels.json
    labels = {
        "tool2id": TOOL2ID,
        "act2id": ACT2ID,
        "combo2id_6class": COMBO2ID,
        "output_mode": OUTPUT_MODE,
        "tool2topology": TOOL2TOPO,
        "topology_roles": {"2A1T": [0, 0, 1], "1A2T": [0, 1, 1]},
        "tool_point_order": {
            "2A1T": "[anchor0, anchor1, tip0]",
            "1A2T": "[anchor0, tip0, tip1]",
            "note": "order is fixed by topology and uid ordering; no permutation matching expected in training",
        },
        "contact_min": CONTACT_MIN,
        "contact_keep": CONTACT_KEEP,
        "contact_pick_mode": CONTACT_PICK_MODE,
        "target_frames": TARGET_FRAMES,
        "hook_dissect_min_steps": (HOOK_DISSECT_MIN_STEPS if ENFORCE_HOOK_DISSECT_MIN_STEPS else None),
        "filter_rules": {
            "disabled": DISABLE_FILTER_RULES,
            "note": "when disabled, optional visibility/length thresholds are bypassed",
        },
        "folder_split_10_1": {
            "enabled": USE_FOLDER_SPLIT_10_1,
            "selected_case_folders": selected_case_folders_runtime,
            "val_case_folders": val_case_folders_runtime,
            "train_k_per_combo": TRAIN_K_PER_COMBO,
            "val_k_per_combo": VAL_K_PER_COMBO,
            "val_backfill_from_train_pool": FOLDER_SPLIT_VAL_BACKFILL,
            "note": (
                "when train/val_k_per_combo is set, sampling is capped per combo within "
                "selected folders; optional val backfill can move only deficit combo samples "
                "from train-case pool to val"
            ),
        },
        "visibility_filter": {
            "min_vis_frames_per_point": MIN_VIS_FRAMES_PER_POINT,
            "min_vis_total": MIN_VIS_TOTAL,
        },
        "keypoint_sequence_files": {
            "keypoints": f"keypoints/{{sid}}.npy, shape=({TARGET_FRAMES},3,2)",
            "keypoint_vis": f"keypoint_vis/{{sid}}.npy, shape=({TARGET_FRAMES},3), 1=visible",
            "contact_vis_raw": f"contact_vis_raw/{{sid}}.npy, shape=({TARGET_FRAMES},4), 1=visible",
            "sampled_abs_frames": f"sampled_abs_frames/{{sid}}.npy, shape=({TARGET_FRAMES},), absolute frameIndex",
            "action_range_abs": "action_range_abs/{sid}.npy, shape=(2,), [action_start_abs, action_end_abs]",
            "action_range_mask": "action_range_mask/{sid}.npy, scalar float32, 1=valid annotation else 0",
            "video_16_npy": (
                f"{dirs['VIDEO16_DIR']}/{{sid}}.npy, shape=({TARGET_FRAMES},H,W,3), uint8, cropped frames "
                f"(dataset link: {out_root / VIDEO16_DIR_NAME}/{{sid}}.npy)"
                if EXPORT_VIDEO16_NPY else
                "disabled"
            ),
        },
        "gt_tensor_files": {
            "gt_tool_xy": f"gt_tool_xy/{{sid}}.npy, shape=({TARGET_FRAMES},3,2), normalized",
            "gt_tool_vis": (
                f"gt_tool_vis/{{sid}}.npy, shape=({TARGET_FRAMES},3), soft weights in [{SOFT_VIS_MIN_WEIGHT},1] after vis interpolation"
                if INTERPOLATE_VIS0_TO_FULL_SUPERVISION else
                f"gt_tool_vis/{{sid}}.npy, shape=({TARGET_FRAMES},3), >0 means visible"
            ),
            "gt_ct_xy": f"gt_ct_xy/{{sid}}.npy, shape=({TARGET_FRAMES},4,2), canonical order",
            "gt_ct_vis": (
                f"gt_ct_vis/{{sid}}.npy, shape=({TARGET_FRAMES},4), soft weights in [{SOFT_VIS_MIN_WEIGHT},1] after vis interpolation"
                if INTERPOLATE_VIS0_TO_FULL_SUPERVISION else
                f"gt_ct_vis/{{sid}}.npy, shape=({TARGET_FRAMES},4), >0 means visible"
            ),
            "combo_id": "combo_ids/{sid}.npy, scalar int64, combo class for tool|action",
            "gt_tool_xy_pseudo": f"gt_tool_xy_pseudo/{{sid}}.npy, shape=({TARGET_FRAMES},3,2)",
            "gt_tool_pseudo_w": f"gt_tool_pseudo_w/{{sid}}.npy, shape=({TARGET_FRAMES},3), default {PSEUDO_WEIGHT} at vis==0 interp frames",
            "gt_ct_xy_pseudo": f"gt_ct_xy_pseudo/{{sid}}.npy, shape=({TARGET_FRAMES},4,2)",
            "gt_ct_pseudo_w": f"gt_ct_pseudo_w/{{sid}}.npy, shape=({TARGET_FRAMES},4), default {PSEUDO_WEIGHT} at vis==0 interp frames",
        },
        "classification_target": {
            "name": "combo_id",
            "num_classes": 6,
            "desc": "joint Tool|Action classification target",
        },
        "temporal_range_target": {
            "name": "action_range_bins",
            "num_bins": ACTION_NUM_BINS,
            "bin_size": ACTION_BIN_SIZE,
            "dataset_mapping": (
                "Dataset maps action_range_abs to clip-relative [0,1] by sampled_abs_frames, "
                "then bins by floor(norm*num_bins) with clamp and end>=start"
            ),
            "missing_label_policy": "action_range_mask=0 => range loss masked out",
        },
        "pseudo_supervision": {
            "enabled": ENABLE_PSEUDO_SUPERVISION,
            "method": "visibility interpolation between nearest previous/next visible frames (xy unchanged)",
            "base_weight": PSEUDO_WEIGHT,
            "soft_vis_min_weight": SOFT_VIS_MIN_WEIGHT,
            "train_note": "apply pseudo loss only when model confidence is high enough (gating in training loop)",
        },
        "random_clip_visualization": {
            "enabled": EXPORT_RANDOM_CLIP_VIS,
            "num_samples": RANDOM_CLIP_VIS_SAMPLES,
            "dir": RANDOM_CLIP_VIS_DIR_NAME,
            "content": "for each sampled clip, 16 sampled frames with stored keypoints and raw vis overlays",
            "crop_note": "uses the same crop bbox as dataset extraction",
            "sample_data_dump": f"{RANDOM_CLIP_VIS_DIR_NAME}/sample_data_dump/{{sid}}_all_data.json",
        },
        # "motion_format": "(L,6) float32 STEP-DELTA; L=T-1, step[t]=abs[t+1]-abs[t]",
        # "coord_space": "normalized",
        "motion_format": (
            "(L,6) float32 STEP-DELTA; L=T-1, step[t]=abs[t+1]-abs[t]"
            if MOTION_REP == "stepdelta" else
            ("(T,6) float32 RELATIVE-TO-FIRST; rel[t]=abs[t]-abs[0], rel[0]=0"
             if REL_INCLUDE_T0 else
             "(L,6) float32 RELATIVE-TO-FIRST; L=T-1, rel[t]=abs[t+1]-abs[0]")
        ),
        "coord_space": ("pixel" if USE_PIXEL_COORDS else "normalized"),
        "black_bar_removal": {
            "enabled": REMOVE_BLACK_BARS,
            "crop_top_bottom": CROP_TOP_BOTTOM,
            "gray_thr": BAR_GRAY_THR,
            "min_nonblack_frac": BAR_MIN_NONBLACK_FRAC,
            "min_border_px": BAR_MIN_BORDER_PX,
            "min_keep_frac": BAR_MIN_KEEP_FRAC,
            "note": "images are saved AFTER cropping; all coords/motions are remapped to the cropped coordinate system."
        },
        "geo_features": (
            {
                "enabled": True,
                "contact_area": "pixel^2",
                "tool_area": "pixel^2",
                "center_dist": "pixel",
                "edge_cos": "|cos(theta)|",
            }
            if EXPORT_GEO_FEATURES
            else {"enabled": False}
        ),
        "anchor_tip_format": "(6,) float32 ABS at t=0 of sampled clip",
        "split_mode": SPLIT_MODE,
        "split_policy": {
            "big_combo_rule": f"if count>=100: val={VAL_K_BIG}, train={TRAIN_K_BIG}, test=rest",
            "small_combo_rule": f"if count<100: val={VAL_K_SMALL}, train=rest, test=0",
        },
        "neighbor_split": {
            "enabled_if_split_mode_neighbor": True,
            "order": "per tool|action combo, sorted by sid (folder scan order)",
            "pattern": f"val indices spread across sequence; forced neighbor in train; VAL_EVERY_K={VAL_EVERY_K}"
        },
        "vis_priority_sampling": {
            "enabled": bool(VIS_PRIORITY_FOR_BIG),
            "apply_when_combo_count_ge": int(VIS_PRIORITY_MIN_COMBO_COUNT),
            "score": "0.5 * (mean(tool_vis_16x3) + mean(contact_vis_16x4))",
            "note": "when enabled, train selection for big combos prefers higher-vis clips (case-diverse)",
        },
        "augmentation_policy": {
            "enabled": bool(ENABLE_WEAK_COMBO_AUG),
            "enabled_for": "train & combo in weak list",
            "weak_combo_aug_ratio": WEAK_COMBO_AUG_RATIO,
            "weak_combo_aug_max_extra": WEAK_COMBO_AUG_MAX_EXTRA,
            "aug_probs": AUG_PROBS,
            "geom": {"trans_max": TRANS_MAX, "rot_deg_max": ROT_DEG_MAX, "scale_min": SCALE_MIN, "scale_max": SCALE_MAX},
            "time_warp": {"ratio_min": TIME_RATIO_MIN, "ratio_max": TIME_RATIO_MAX},
            "noise": {"std": NOISE_STD, "ema": NOISE_EMA},
            "fold1_clip_post_aug": {
                "enabled": bool(USE_FOLD1_CLIP_POST_AUG),
                "scope_by_case_only": bool(FOLD1_AUG_SCOPE_BY_CASE_ONLY),
                "aug_series_name": str(FOLD1_AUG_SERIES_NAME),
                "fold1_train_sids_file": str(FOLD1_TRAIN_SIDS_FILE),
                "fold1_val_sids_file": str(FOLD1_VAL_SIDS_FILE),
                "fold1_train_cases_file": str(FOLD1_TRAIN_CASES_FILE),
                "fold1_val_cases_file": str(FOLD1_VAL_CASES_FILE),
                "target_train_per_combo": FOLD1_TARGET_TRAIN_PER_COMBO,
                "train_base_cap_per_combo": FOLD1_TRAIN_BASE_CAP_PER_COMBO,
                "skip_aug_combos": sorted(FOLD1_SKIP_AUG_COMBOS),
                "use_v3_strength_mix": bool(FOLD1_USE_V3_STRENGTH_MIX),
                "strength_mix_by_combo": FOLD1_STRENGTH_MIX_BY_COMBO,
                "strength_presets": FOLD1_STRENGTH_PRESETS,
                "strength_preset_overrides_by_combo": FOLD1_STRENGTH_PRESET_OVERRIDES_BY_COMBO,
                "offset_base_multiplier_by_combo": OFFSET_BASE_MULTIPLIER_BY_COMBO,
                "offset_base_target_per_combo": OFFSET_BASE_TARGET_PER_COMBO,
                "extreme_only_hard": bool(FOLD1_EXTREME_ONLY_HARD),
                "extreme_hard_top_frac": float(FOLD1_EXTREME_HARD_TOP_FRAC),
                "extreme_hard_min_samples": int(FOLD1_EXTREME_HARD_MIN_SAMPLES),
                "extreme_priority_combos": sorted([str(x) for x in FOLD1_EXTREME_PRIORITY_COMBOS]),
                "force_high_difficulty_only": bool(FOLD1_FORCE_HIGH_DIFFICULTY_ONLY),
                "force_high_difficulty_min_extreme_priority": float(FOLD1_FORCE_HIGH_DIFFICULTY_MIN_EXTREME_PRIORITY),
                "force_high_difficulty_min_extreme_default": float(FOLD1_FORCE_HIGH_DIFFICULTY_MIN_EXTREME_DEFAULT),
                "donor_diversity_k_threshold": int(FOLD1_DONOR_DIVERSITY_K_THRESHOLD),
                "donor_diversity_min_unique": int(FOLD1_DONOR_DIVERSITY_MIN_UNIQUE),
                "crop_mode_weights": FOLD1_CROP_MODE_WEIGHTS,
                "temp_start_jitter_p": float(FOLD1_TEMP_START_JITTER_P),
                "temp_start_jitter_max_frac": float(FOLD1_TEMP_START_JITTER_MAX_FRAC),
                "temp_warp_p": FOLD1_TEMP_WARP_P,
                "hflip_p": FOLD1_HFLIP_P,
                "affine_p": FOLD1_AFFINE_P,
                "bg_replace_p": FOLD1_BG_REPLACE_P,
                "bg_same_combo_p": FOLD1_BG_SAME_COMBO_P,
            },
        },
        "debug_aug_images": {
            "enabled": SAVE_DEBUG_AUG_IMAGES,
            "dir": DEBUG_DIR_NAME,
            "one_example_per_type": DEBUG_AUG_ONE_EXAMPLE_PER_TYPE,
            "content": "start-frame overlay before/after; fold1 mode saves one example per augmentation key",
            "fixed_first_n_pairs_dir": FOLD1_DEBUG_FIXED_DIR_NAME,
            "fixed_first_n_pairs_count": int(FOLD1_DEBUG_FIXED_PAIR_COUNT),
        },
        "start_frame_definition": (
            "clip first frame (video_base_abs) when USE_CLIP_START_AS_ABS_START=True; "
            "tool-start search is used only as fallback initialization source for missing early annotations"
            if USE_CLIP_START_AS_ABS_START else
            ("earliest frame with valid tool topology; "
             f"START_REQUIRE_VISIBLE_TOOL={START_REQUIRE_VISIBLE_TOOL}, "
             f"ALLOW_START_FALLBACK_ANY_TOOL={ALLOW_START_FALLBACK_ANY_TOOL}")
        ),
        "video_assumption": "clip mp4 frame0 corresponds to affordance_range.start (video_base_abs).",
    }
    if ENABLE_MASK_IMAGE:
        labels["mask_images"] = {
            "enabled": True,
            "dir": MASK_DIR_NAME,
            "content": "black/white pose mask from anchor_tip triangle + contact polygon (t=0), same size as images",
            "radius_px": MASK_RADIUS_PX,
            "border_px": MASK_BORDER_PX,
        }
    (out_root / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")

    # meta.json
    meta_json = [asdict(it) for it in meta_all]
    (out_root / "meta.json").write_text(json.dumps(meta_json, indent=2, ensure_ascii=False), encoding="utf-8")

    # Random visual sanity check: sample K base clips and render all 16 frames.
    vis_clip_ids: List[str] = []
    sample_dump_paths: List[Path] = []
    if EXPORT_RANDOM_CLIP_VIS:
        vis_clip_ids = export_random_clip_visualizations(
            base_items=meta,
            root_dir=root_dir,
            out_root=out_root,
            dirs=dirs,
            rng_py=rng_py,
            n_samples=RANDOM_CLIP_VIS_SAMPLES,
        )
        sample_dump_paths = dump_selected_samples_all_data(
            selected_sids=vis_clip_ids,
            sid2item={it.sid: it for it in meta},
            dirs=dirs,
            out_root=out_root,
        )

    # Prune non-essential artifacts for minimal mode.
    pruned_dir_n, pruned_file_n, pruned_dirs, pruned_files = prune_output_tree_for_mode(
        out_root=out_root,
        mode=OUTPUT_MODE,
    )

    # index.txt
    # sid \t split \t tool|action \t ann_relpath \t source_sid \t aug_type
    with open(out_root / "index.txt", "w", encoding="utf-8") as f:
        for it in sorted(meta_all, key=lambda x: x.sid):
            combo = f"{it.tool}|{it.action}"
            source = it.source_sid if it.source_sid is not None else ""
            augt = it.aug_type if it.aug_type is not None else ""
            f.write(f"{it.sid}\t{it.split}\t{combo}\t{it.ann_relpath}\t{source}\t{augt}\n")

    comboid_report = build_comboid_split_report(
        sid2item=sid2item,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )
    comboid_json_path, comboid_tsv_path = write_comboid_split_report(out_root=out_root, report=comboid_report)
    split_case_combo_report = build_split_case_combo_report(
        sid2item=sid2item,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )
    split_case_combo_json_path, split_case_combo_tsv_path, split_sid_tsv_path = write_split_case_combo_report(
        out_root=out_root,
        report=split_case_combo_report,
    )
    fold1_aug_json_path: Optional[Path] = None
    fold1_aug_tsv_path: Optional[Path] = None
    fold1_aug_v3_json_path: Optional[Path] = None
    fold1_aug_v3_tsv_path: Optional[Path] = None
    if len(fold1_aug_report) > 0:
        fold1_aug_json_path, fold1_aug_tsv_path = write_fold1_aug_detailed_report(
            out_root=out_root,
            report=fold1_aug_report,
        )
        fold1_aug_v3_json_path = out_root / "fold1_aug_v3_detailed_report.json"
        fold1_aug_v3_tsv_path = out_root / "fold1_aug_v3_detailed_report.tsv"
        shutil.copy2(fold1_aug_json_path, fold1_aug_v3_json_path)
        shutil.copy2(fold1_aug_tsv_path, fold1_aug_v3_tsv_path)
    contract_audit_path = write_kpt_combo_contract_audit(
        out_root=out_root,
        export_video16=EXPORT_VIDEO16_NPY,
        mode=OUTPUT_MODE,
    )

    combo_aux_removed = False
    if combo_aux_root is not None and combo_aux_root.exists():
        try:
            shutil.rmtree(combo_aux_root, ignore_errors=True)
            combo_aux_removed = True
        except Exception:
            combo_aux_removed = False

    # final report
    print("=================================================")
    print(f"[DONE] {OUT_NAME} generated")
    print(f"  scanned(base ann files) : {scanned}")
    print(f"  kept(base)              : {len(meta)}")
    print(f"  augmented(train only)   : {aug_total}")
    print(f"  total samples           : {len(meta_all)}")
    print(f"  fixed input frames      : {TARGET_FRAMES}")
    print(f"  clip start policy       : {'clip_first_frame' if USE_CLIP_START_AS_ABS_START else 'tool_first_valid'}")
    print(f"  vis interpolation       : {INTERPOLATE_VIS0_TO_FULL_SUPERVISION}")
    print(f"  filter rules disabled   : {DISABLE_FILTER_RULES}")
    if USE_FOLDER_SPLIT_10_1:
        print(
            f"  folder split (selected) : enabled "
            f"(selected={len(selected_case_folders_runtime)} "
            f"train={len(selected_case_folders_runtime) - len(val_case_folders_runtime)} "
            f"val={len(val_case_folders_runtime)})"
        )
        print(
            f"  folder split caps       : train_k_per_combo={TRAIN_K_PER_COMBO} "
            f"val_k_per_combo={VAL_K_PER_COMBO} "
            f"val_backfill_from_train={FOLDER_SPLIT_VAL_BACKFILL}"
        )
    if MOTION_REP == "stepdelta":
        print(f"  motion format           : (L,6) float32 STEP-DELTA")
    elif REL_INCLUDE_T0:
        print(f"  motion format           : (T,6) float32 RELATIVE-TO-FIRST")
    else:
        print(f"  motion format           : (L,6) float32 RELATIVE-TO-FIRST")
    # print(f"  Mean/Std computed on    : train (base+aug)")
    print(f"  Mean/Std computed on    : train_base_only")
    print(f"  train / val / test      : {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")
    print(f"  out dir                 : {out_root}")
    print(f"  output mode             : {OUTPUT_MODE}")
    print(f"  comboid report          : {comboid_json_path.name}, {comboid_tsv_path.name}")
    print(
        "  split composition       : "
        f"{split_case_combo_json_path.name}, {split_case_combo_tsv_path.name}, {split_sid_tsv_path.name}"
    )
    if fold1_aug_json_path is not None and fold1_aug_tsv_path is not None:
        print(f"  fold1 aug report        : {fold1_aug_json_path.name}, {fold1_aug_tsv_path.name}")
    if fold1_aug_v3_json_path is not None and fold1_aug_v3_tsv_path is not None:
        print(f"  fold1 aug v3 report     : {fold1_aug_v3_json_path.name}, {fold1_aug_v3_tsv_path.name}")
    print(f"  training contract audit : {contract_audit_path.name}")
    if combo_aux_root is not None:
        print(f"  combo_only aux cleanup  : {'done' if combo_aux_removed else 'failed'} ({combo_aux_root})")
    if EXPORT_VIDEO16_NPY:
        print(f"  video16 npy write dir   : {dirs['VIDEO16_DIR']}")
        print(f"  video16 npy dataset link: {out_root / VIDEO16_DIR_NAME}")
    print(f"  debug aug dir           : {debug_dir}  (geometric only)")
    print(
        f"  debug fixed pairs       : {debug_fixed_dir} "
        f"(saved={len(debug_fixed_pairs)}/{int(FOLD1_DEBUG_FIXED_PAIR_COUNT)})"
    )
    if EXPORT_RANDOM_CLIP_VIS:
        print(f"  debug clip vis dir      : {out_root / RANDOM_CLIP_VIS_DIR_NAME}  ({len(vis_clip_ids)} sampled clips)")
        if len(vis_clip_ids) > 0:
            print(f"  debug clip vis sids     : {', '.join(vis_clip_ids)}")
        if len(sample_dump_paths) > 0:
            print(f"  debug sample dumps      : {len(sample_dump_paths)} json files")
            for p in sample_dump_paths:
                print(f"    - {p}")
    if OUTPUT_MODE != "full":
        print(f"  pruned dirs/files       : {pruned_dir_n} / {pruned_file_n}")
        if pruned_dir_n > 0:
            print(f"  pruned dir names        : {', '.join(pruned_dirs)}")
        if pruned_file_n > 0:
            print(f"  pruned file names       : {', '.join(pruned_files)}")
    vis_summary_path = out_root / "vis_summary_train.json"
    if vis_summary_path.exists():
        print(f"  vis summary             : {vis_summary_path}")
    else:
        print("  vis summary             : (pruned in minimal mode)")
    print("---- stats ----")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("=================================================")


if __name__ == "__main__":
    main()
