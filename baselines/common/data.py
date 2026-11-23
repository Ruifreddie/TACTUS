"""Read the shared TACTUS export: the same per-clip arrays TactusNet trains and is scored on.

Conventions (identical to UniTAP / TactusNet):
- A clip is identified by its sid (6 digits). Splits: all_{train,val,test,silver}.txt. Baselines train on Gold train only.
- frames(): the 16 protocol frames, content crop at NATIVE resolution (e.g. 894x601), uint8. Stored as BGR; returned as RGB.
  Models resize them to their own input size (TactusNet squashes to 384x384).
- Points are normalized to the content crop, (x, y) in [0, 1]. Spatial metrics multiply by 384 in both axes.
- Tool points [16,3,2]: hook (topology 2A1T) = anchor, anchor, tip; grasper/scissors/clipper (1A2T) = anchor, tip, tip.
- Contact points [16,4,2]: four tracks with one identity each over the clip (UniTAP's uid order: first visible frame, then
  angle around the mean position). The contact region is their convex hull (see hull.py).
- Pairs: the 6 valid tool-action pairs, ids from labels.json "combo2id_6class". Which/What are read from the pair.
- Interval: [start, end] on the window axis, normalized by the first and last sampled frame (UniTAP's
  _compute_action_targets), so 0 = first protocol frame, 1 = last.
"""
import json, os
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config

SOURCES = ("SurgVU", "Cholec80", "HeiChole", "CRCD", "YouTube")
TOPOLOGY_ROLES = {"1A2T": (0, 1, 1), "2A1T": (0, 0, 1)}  # 0 = anchor, 1 = tip (UniTAP infer_pipeline.py)


def source_of(task: str) -> str:
    if task.startswith("case_"): return "SurgVU"
    if task.startswith("cholec80_"): return "Cholec80"
    if task.startswith("heichole_"): return "HeiChole"
    if task.startswith("comprehensive_"): return "CRCD"
    if task.startswith("youtube_"): return "YouTube"
    raise ValueError(f"unknown source for task {task!r}")


class Export:
    def __init__(self, root=None, hf_raw=None):
        self.root = Path(root or config.EXPORT)
        self.hf_raw = Path(hf_raw or config.HF_RAW)
        self.labels = json.loads((self.root / "labels.json").read_text())
        self.meta = {m["sid"]: m for m in json.loads((self.root / "meta.json").read_text())}
        self.tool2id = self.labels["tool2id"]; self.act2id = self.labels["act2id"]; self.pair2id = self.labels["combo2id_6class"]
        self.tools = sorted(self.tool2id, key=self.tool2id.get); self.actions = sorted(self.act2id, key=self.act2id.get)
        self.pairs = sorted(self.pair2id, key=self.pair2id.get)                      # pair names in id order, "tool|action"
        self.pair_tool = np.array([self.tool2id[p.split("|")[0]] for p in self.pairs])  # pair id -> tool id
        self.pair_act = np.array([self.act2id[p.split("|")[1]] for p in self.pairs])    # pair id -> action id
        self.tool_roles = {t: TOPOLOGY_ROLES[self.labels["tool2topology"][t]] for t in self.tools}

    # ---- lists and labels ----
    def sids(self, split: str):
        """split in train / val / test / silver. The test list is the evaluation set of every row (the Gold test clips)."""
        p = self.root / f"all_{split}.txt"
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]

    def task(self, sid): return self.meta[sid]["clip_key"].split("/")[0]       # one task = one source video
    def source(self, sid): return source_of(self.task(sid))
    def pair_id(self, sid): return int(self.pair2id[self.meta[sid]["combo"]])
    def tool_id(self, sid): return int(self.pair_tool[self.pair_id(sid)])
    def act_id(self, sid): return int(self.pair_act[self.pair_id(sid)])
    def tool_name(self, sid): return self.meta[sid]["tool"]

    def tip_index(self, sid):
        """Indices of the tip points among the 3 tool points (hook: [2]; jawed tools: [1, 2])."""
        return [k for k, r in enumerate(self.tool_roles[self.tool_name(sid)]) if r == 1]

    # ---- arrays ----
    def _npy(self, d, sid, mmap=False):
        return np.load(self.root / d / f"{sid}.npy", mmap_mode="r" if mmap else None)

    def frames(self, sid, rgb=True):
        """uint8 [16,H,W,3]: the protocol frames, content crop at native resolution.
        Reads the export's video_16_npy when present; otherwise decodes the same frames from the raw clip (identical
        pixels, checked by tests/test_common.py), so the pod does not need the 500 GB of exported frames."""
        p = self.root / "video_16_npy" / f"{sid}.npy"
        if p.is_file():
            f = np.load(p)
            return np.ascontiguousarray(f[..., ::-1]) if rgb else f
        cache = os.environ.get("BL_FRAME_CACHE")
        if cache:                                   # decoded once per clip, then JPEG (q95) on the pod disk: ~2.5 MB per clip
            f = _cache_read(Path(cache) / f"{sid}.npz")
            if f is None:
                f = self.raw_frames(sid, self.sampled_abs(sid), rgb=False); _cache_write(Path(cache) / f"{sid}.npz", f)
            return np.ascontiguousarray(f[..., ::-1]) if rgb else f
        return self.raw_frames(sid, self.sampled_abs(sid), rgb=rgb)

    def tool_points(self, sid):
        """xy [16,3,2] normalized to the content crop, vis [16,3] in {0,1}."""
        return self._npy("gt_tool_xy", sid).astype(np.float32), (self._npy("gt_tool_vis", sid) > 0.5).astype(np.float32)

    def contact_points(self, sid):
        """xy [16,4,2] normalized to the content crop (uid order), vis [16,4] in {0,1}."""
        return self._npy("gt_ct_xy", sid).astype(np.float32), (self._npy("gt_ct_vis", sid) > 0.5).astype(np.float32)

    def sampled_abs(self, sid):
        return self._npy("sampled_abs_frames", sid).reshape(-1).astype(np.int64)

    def interval(self, sid):
        """(start, end, valid) on the window axis, exactly as UniTAP's dataset._compute_action_targets."""
        sab = self.sampled_abs(sid).astype(np.float64)
        p = self.root / "action_range_abs" / f"{sid}.npy"; q = self.root / "action_range_mask" / f"{sid}.npy"
        s = e = 0.0; mask = 0.0
        if p.is_file():
            a = np.load(p).reshape(-1)
            if a.size >= 2: s, e, mask = float(a[0]), float(a[1]), 1.0
        if q.is_file():
            m = np.load(q).reshape(-1)
            if m.size > 0: mask = float(m[0])
        if mask <= 0.5 or sab.size < 2 or sab[-1] - sab[0] <= 0: return 0.0, 0.0, 0.0
        den = sab[-1] - sab[0]
        s = float(np.clip((s - sab[0]) / den, 0.0, 1.0)); e = float(np.clip((e - sab[0]) / den, 0.0, 1.0))
        return s, max(s, e), 1.0

    FALLBACK_FPS = {"SurgVU": 60.0, "CRCD": 60.0, "Cholec80": 25.0, "HeiChole": 25.0, "YouTube": 30.0}

    def fps(self, sid):
        """Frame rate of the clip's video (read once from the mp4; source default if the file is missing)."""
        if not hasattr(self, "_fps"): self._fps = {}
        if sid not in self._fps:
            f = 0.0
            try:
                import cv2
                cap = cv2.VideoCapture(str(self.clip_path(sid))); f = float(cap.get(cv2.CAP_PROP_FPS)); cap.release()
            except Exception:
                f = 0.0
            self._fps[sid] = f if f > 0 else self.FALLBACK_FPS[self.source(sid)]
        return self._fps[sid]

    def frames_1fps(self, sid):
        """Indices (0..15) of the protocol frames nearest to each whole second of the window (0 s, 1 s, 2 s, ...).
        Image methods TRAIN on these frames only (about one frame per second); every method is TESTED on all 16."""
        sab = self.sampled_abs(sid).astype(np.float64); t = (sab - sab[0]) / self.fps(sid)
        marks = np.arange(0.0, t[-1] + 1e-9, 1.0)
        return sorted({int(np.argmin(np.abs(t - m))) for m in marks})

    def tokens(self, sid, tok_root=None):
        """TactusNet's frozen V-JEPA 2.1 tokens [8,24,24,1024] (fp16, memory-mapped)."""
        return np.load(Path(tok_root or config.TOKENS) / f"{sid}.npy", mmap_mode="r")

    # ---- raw frames (for methods that need frames other than the 16 protocol frames) ----
    def clip_path(self, sid):
        task, stem = self.meta[sid]["clip_key"].split("/")
        return self.hf_raw / task / "clips" / f"{stem}.mp4"

    def raw_frames(self, sid, abs_indices, rgb=True):
        """Frames at absolute indices (the scale of sampled_abs), cropped to the same content box as frames().
        Indices outside the clip are clamped to its first/last frame. Returns uint8 [N,H,W,3].
        Decoded with OpenCV exactly like the exporter, so raw_frames(sid, sampled_abs(sid)) == frames(sid)."""
        import cv2
        m = self.meta[sid]; base = int(m["abs_start"]); x0, y0, x1, y1 = [int(v) for v in m["crop_bbox_xyxy"]]
        cap = cv2.VideoCapture(str(self.clip_path(sid))); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0: raise IOError(f"cannot open {self.clip_path(sid)}")
        local = np.clip(np.asarray(abs_indices, dtype=np.int64) - base, 0, n - 1)
        want = sorted(set(local.tolist())); got = {}; pos = None
        for i in want:
            if pos is None or i < pos or i - pos > 8:      # seek for big jumps, read forward for small ones
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); pos = i
            while pos <= i:
                ok, f = cap.read()
                if not ok: raise IOError(f"cannot read frame {pos} of {self.clip_path(sid)}")
                if pos == i: got[i] = f[y0:y1, x0:x1]
                pos += 1
        cap.release()
        out = np.stack([got[i] for i in local.tolist()])
        return np.ascontiguousarray(out[..., ::-1]) if rgb else out


def _cache_read(path):
    """BGR uint8 [T,H,W,3] from a JPEG frame cache file, or None if it is missing or unreadable."""
    try:
        import cv2
        z = np.load(path); data, offs = z["data"], z["offs"]
        return np.stack([cv2.imdecode(data[offs[i]:offs[i + 1]], cv2.IMREAD_COLOR) for i in range(len(offs) - 1)])
    except Exception:
        return None


def _cache_write(path, frames_bgr):
    import cv2
    bufs = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].reshape(-1) for f in frames_bgr]
    offs = np.cumsum([0] + [len(b) for b in bufs]).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.stem}.{os.getpid()}.npz")
    np.savez(tmp, data=np.concatenate(bufs), offs=offs); os.replace(tmp, path)     # atomic: workers may race


@lru_cache(maxsize=4)
def get_export(root=None, hf_raw=None) -> Export:
    return Export(root, hf_raw)


def resize_frames(frames, size):
    """[T,H,W,3] uint8 -> [T,size,size,3] uint8, squashed like TactusNet's preprocessing (bilinear)."""
    import cv2
    if isinstance(size, int): size = (size, size)
    return np.stack([cv2.resize(f, (size[1], size[0]), interpolation=cv2.INTER_LINEAR) for f in frames])
