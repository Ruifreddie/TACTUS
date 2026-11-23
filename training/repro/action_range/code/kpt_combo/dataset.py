import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
ACTION_BIN_SIZE = 0.1
ACTION_NUM_BINS = int(round(1.0 / ACTION_BIN_SIZE))


class SurgKptComboDataset(Dataset):
    """Dataset for fixed-16f clips with combo labels and normalized keypoints.

    Each sample returns:
      - video: [T, 3, H, W] float32 (already normalized if `normalize=True`)
      - gt_tool_xy: [T, 3, 2]
      - gt_tool_vis: [T, 3] (soft weights)
      - gt_ct_xy: [T, 4, 2]
      - gt_ct_vis: [T, 4] (soft weights)
      - gt_action_start_bin: scalar long in [0, ACTION_NUM_BINS)
      - gt_action_end_bin: scalar long in [0, ACTION_NUM_BINS)
      - gt_action_start_norm: scalar float in [0,1]
      - gt_action_end_norm: scalar float in [0,1]
      - action_range_mask: scalar float in {0,1}
      - combo_id: scalar long in [0, num_combos)
      - sid: string sample id

    Optional tool-kpt-only contract (`tool_kpt_only=True`):
      - required on disk: video_16_npy, combo_ids, gt_tool_xy, gt_tool_vis (+ split/meta files)
      - synthesized by loader: gt_ct_*, action_range_*, sampled_abs_frames placeholders

    Optional range-only contract (`range_only_contract=True`):
      - required on disk: combo_ids, gt_tool_xy, gt_tool_vis, sampled_abs_frames,
        action_range_abs, action_range_mask (+ split/meta files)
      - gt_ct_* may be absent and will be synthesized as zeros because range-only
        training does not supervise contact trajectories.
    """

    def __init__(
        self,
        root: str,
        split: str,
        split_file: Optional[str] = None,
        input_size: int = 256,
        normalize: bool = True,
        enforce_16f: bool = True,
        load_video: bool = True,
        combo_only_pure: bool = False,
        tool_kpt_only: bool = False,
        range_only_contract: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.input_size = int(input_size)
        self.normalize = bool(normalize)
        self.enforce_16f = bool(enforce_16f)
        self.load_video = bool(load_video)
        self.combo_only_pure = bool(combo_only_pure)
        self.tool_kpt_only = bool(tool_kpt_only)
        self.range_only_contract = bool(range_only_contract)

        if split_file is None or str(split_file).strip() == "":
            split_path = self.root / f"{split}.txt"
        else:
            split_path = Path(str(split_file))
            if not split_path.is_absolute():
                split_path = self.root / split_path
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        self.sids = [x.strip() for x in split_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not self.sids:
            raise RuntimeError(f"No sample ids found in {split_path}")

        labels_path = self.root / "labels.json"
        self.labels: Dict = {}
        if labels_path.is_file():
            self.labels = json.loads(labels_path.read_text(encoding="utf-8"))

        combo2id = self.labels.get("combo2id_6class", {})
        self.combo_id_to_name = {int(v): str(k) for k, v in combo2id.items()}
        self.num_combos = len(self.combo_id_to_name) if self.combo_id_to_name else 6
        self.tool2topology = {str(k): str(v) for k, v in self.labels.get("tool2topology", {}).items()}
        self.topology_roles = {
            str(k): [int(x) for x in list(v)]
            for k, v in self.labels.get("topology_roles", {}).items()
            if isinstance(v, list)
        }
        self.sid_to_pt_role = self._load_sid_to_pt_role()

    def __len__(self) -> int:
        return len(self.sids)

    def _load_npy(self, folder: str, sid: str, dtype: Optional[np.dtype] = None) -> np.ndarray:
        p = self.root / folder / f"{sid}.npy"
        if not p.is_file():
            raise FileNotFoundError(f"Missing data file: {p}")
        arr = np.load(p)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def _load_sid_to_pt_role(self) -> Dict[str, List[int]]:
        meta_path = self.root / "meta.json"
        if not meta_path.is_file():
            return {}
        try:
            rows = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(rows, list):
            return {}
        out: Dict[str, List[int]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("sid", "")).strip()
            pt_role = row.get("pt_role", None)
            if (not sid) or (not isinstance(pt_role, list)):
                continue
            try:
                out[sid] = [int(x) for x in pt_role]
            except Exception:
                continue
        return out

    def _infer_pt_role(self, sid: str, combo_id: int) -> np.ndarray:
        pt_role = self.sid_to_pt_role.get(str(sid), None)
        if isinstance(pt_role, list) and len(pt_role) > 0:
            return np.asarray(pt_role, dtype=np.int64)
        combo_name = self.combo_id_to_name.get(int(combo_id), "")
        tool_name = str(combo_name).split("|", 1)[0] if "|" in str(combo_name) else str(combo_name)
        topology = self.tool2topology.get(tool_name, "")
        role_list = self.topology_roles.get(str(topology), None)
        if isinstance(role_list, list) and len(role_list) > 0:
            return np.asarray(role_list, dtype=np.int64)
        return np.zeros((3,), dtype=np.int64)

    def _prepare_video(self, video_thwc_u8: np.ndarray) -> torch.Tensor:
        if video_thwc_u8.ndim != 4 or video_thwc_u8.shape[-1] != 3:
            raise ValueError(f"video_16_npy must be [T,H,W,3], got {video_thwc_u8.shape}")

        # video_16_npy is exported from OpenCV decode path (BGR); convert to RGB for VJEPA/ImageNet stats.
        video_rgb = np.ascontiguousarray(video_thwc_u8[..., ::-1])
        video = torch.from_numpy(video_rgb).permute(0, 3, 1, 2).contiguous().float() / 255.0
        if self.enforce_16f and video.shape[0] != 16:
            raise ValueError(f"Expected 16 frames, got {video.shape[0]}")

        if self.input_size > 0:
            video = F.interpolate(video, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)

        if self.normalize:
            video = (video - IMAGENET_MEAN) / IMAGENET_STD

        return video

    @staticmethod
    def _compute_action_targets(
        sampled_abs_frames: np.ndarray,
        action_start_abs: float,
        action_end_abs: float,
        range_mask: float,
    ) -> tuple[float, float, int, int, float]:
        mask = 1.0 if float(range_mask) > 0.5 else 0.0
        if mask <= 0.0:
            return 0.0, 0.0, 0, 0, 0.0

        frames = np.asarray(sampled_abs_frames, dtype=np.float64).reshape(-1)
        if frames.size < 2:
            return 0.0, 0.0, 0, 0, 0.0
        clip_s = float(frames[0])
        clip_e = float(frames[-1])
        den = clip_e - clip_s
        if den <= 0.0:
            return 0.0, 0.0, 0, 0, 0.0

        start_norm = (float(action_start_abs) - clip_s) / den
        end_norm = (float(action_end_abs) - clip_s) / den
        start_norm = float(np.clip(start_norm, 0.0, 1.0))
        end_norm = float(np.clip(end_norm, 0.0, 1.0))
        if end_norm < start_norm:
            end_norm = start_norm

        n_bins = int(ACTION_NUM_BINS)
        start_bin = int(np.floor(start_norm * n_bins))
        end_bin = int(np.floor(end_norm * n_bins))
        start_bin = int(np.clip(start_bin, 0, n_bins - 1))
        end_bin = int(np.clip(end_bin, 0, n_bins - 1))
        if end_bin < start_bin:
            end_bin = start_bin
        return start_norm, end_norm, start_bin, end_bin, 1.0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid = self.sids[idx]
        combo_np = self._load_npy("combo_ids", sid)
        combo_id = int(combo_np.item()) if combo_np.ndim == 0 else int(combo_np.reshape(-1)[0])

        if self.combo_only_pure:
            # Pure combo mode: only read combo label (+ video if requested); all auxiliary
            # supervision is returned as zeros to keep train/eval loops unchanged.
            t = 16
            item: Dict[str, torch.Tensor] = {
                "sid": sid,
                "gt_tool_xy": torch.zeros((t, 3, 2), dtype=torch.float32),
                "gt_tool_vis": torch.zeros((t, 3), dtype=torch.float32),
                "gt_ct_xy": torch.zeros((t, 4, 2), dtype=torch.float32),
                "gt_ct_vis": torch.zeros((t, 4), dtype=torch.float32),
                "gt_action_start_bin": torch.tensor(0, dtype=torch.long),
                "gt_action_end_bin": torch.tensor(0, dtype=torch.long),
                "gt_action_start_norm": torch.tensor(0.0, dtype=torch.float32),
                "gt_action_end_norm": torch.tensor(0.0, dtype=torch.float32),
                "action_range_mask": torch.tensor(0.0, dtype=torch.float32),
                "sampled_abs_frames": torch.arange(t, dtype=torch.int64),
                "combo_id": torch.tensor(combo_id, dtype=torch.long),
                "pt_role": torch.from_numpy(self._infer_pt_role(sid=sid, combo_id=combo_id)),
            }
            if self.load_video:
                video = self._prepare_video(self._load_npy("video_16_npy", sid))
                item["video"] = video
            return item

        gt_tool_xy = torch.from_numpy(self._load_npy("gt_tool_xy", sid, np.float32))
        gt_tool_vis = torch.from_numpy(self._load_npy("gt_tool_vis", sid, np.float32))
        t = int(gt_tool_xy.shape[0])
        if self.tool_kpt_only:
            if self.enforce_16f and t != 16:
                raise ValueError(f"Expected 16 frames for tool_kpt_only dataset, got {t}")
            gt_ct_xy = torch.zeros((t, 4, 2), dtype=torch.float32)
            gt_ct_vis = torch.zeros((t, 4), dtype=torch.float32)
            sampled_abs = np.arange(t, dtype=np.int64)
            gt_start_norm = 0.0
            gt_end_norm = 0.0
            gt_start_bin = 0
            gt_end_bin = 0
            range_mask = 0.0
        else:
            if self.enforce_16f and t != 16:
                raise ValueError(f"Expected 16 frames for range-capable dataset, got {t}")
            if self.range_only_contract:
                gt_ct_xy_path = self.root / "gt_ct_xy" / f"{sid}.npy"
                gt_ct_vis_path = self.root / "gt_ct_vis" / f"{sid}.npy"
                if gt_ct_xy_path.is_file() and gt_ct_vis_path.is_file():
                    gt_ct_xy = torch.from_numpy(self._load_npy("gt_ct_xy", sid, np.float32))
                    gt_ct_vis = torch.from_numpy(self._load_npy("gt_ct_vis", sid, np.float32))
                else:
                    gt_ct_xy = torch.zeros((t, 4, 2), dtype=torch.float32)
                    gt_ct_vis = torch.zeros((t, 4), dtype=torch.float32)
            else:
                gt_ct_xy = torch.from_numpy(self._load_npy("gt_ct_xy", sid, np.float32))
                gt_ct_vis = torch.from_numpy(self._load_npy("gt_ct_vis", sid, np.float32))
            sampled_abs = self._load_npy("sampled_abs_frames", sid, np.int64).reshape(-1)

            action_abs_path = self.root / "action_range_abs" / f"{sid}.npy"
            action_mask_path = self.root / "action_range_mask" / f"{sid}.npy"
            action_start_abs = 0.0
            action_end_abs = 0.0
            range_mask = 0.0
            if action_abs_path.is_file():
                action_abs = np.load(action_abs_path).reshape(-1)
                if action_abs.size >= 2:
                    action_start_abs = float(action_abs[0])
                    action_end_abs = float(action_abs[1])
                    range_mask = 1.0
            if action_mask_path.is_file():
                action_mask_arr = np.load(action_mask_path).reshape(-1)
                if action_mask_arr.size > 0:
                    range_mask = float(action_mask_arr[0])
            gt_start_norm, gt_end_norm, gt_start_bin, gt_end_bin, range_mask = self._compute_action_targets(
                sampled_abs_frames=sampled_abs,
                action_start_abs=action_start_abs,
                action_end_abs=action_end_abs,
                range_mask=range_mask,
            )

        item = {
            "sid": sid,
            "gt_tool_xy": gt_tool_xy,  # [T,3,2]
            "gt_tool_vis": gt_tool_vis,  # [T,3]
            "gt_ct_xy": gt_ct_xy,  # [T,4,2]
            "gt_ct_vis": gt_ct_vis,  # [T,4]
            "gt_action_start_bin": torch.tensor(int(gt_start_bin), dtype=torch.long),
            "gt_action_end_bin": torch.tensor(int(gt_end_bin), dtype=torch.long),
            "gt_action_start_norm": torch.tensor(float(gt_start_norm), dtype=torch.float32),
            "gt_action_end_norm": torch.tensor(float(gt_end_norm), dtype=torch.float32),
            "action_range_mask": torch.tensor(float(range_mask), dtype=torch.float32),
            "sampled_abs_frames": torch.from_numpy(sampled_abs.astype(np.int64)),
            "combo_id": torch.tensor(combo_id, dtype=torch.long),
            "pt_role": torch.from_numpy(self._infer_pt_role(sid=sid, combo_id=combo_id)),
        }
        if self.load_video:
            video = self._prepare_video(self._load_npy("video_16_npy", sid))
            item["video"] = video  # [T,3,H,W]
        return item
