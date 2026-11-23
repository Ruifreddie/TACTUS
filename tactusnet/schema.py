"""Plain data containers that cross layer boundaries.  Everything is JSON-serialisable through `Annotation.to_json()`."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

N_SAMPLED = 16   # frames a clip is sampled into; also the length of the temporal axis of every head


def sample_local_indices(n: int, k: int = N_SAMPLED) -> np.ndarray:
    """k indices uniformly over the clip's n frames; identical to UniTAP's `sample_fixed_abs_indices(0, n - 1, k)`."""
    if n < 1:
        raise ValueError(f"a clip needs at least one frame, got n={n}")
    return np.round(np.linspace(0, n - 1, k)).astype(int)


@dataclass
class ClipSpec:
    """The window the whole pyramid works on (an 'affordance range')."""
    a0: int                       # absolute index of the first frame
    n: int                        # number of frames
    fps: float = 60.0
    video: Optional[str] = None

    @property
    def a1(self) -> int:
        return self.a0 + self.n - 1

    def sampled_local(self) -> np.ndarray:
        return sample_local_indices(self.n)


@dataclass
class SemanticOut:
    combo_id: int
    combo: str
    tool: str
    action: str
    conf: float
    probs: List[float]
    source: str                   # 'predicted' or 'given'


@dataclass
class TemporalOut:
    start_t: float                # on the 16-step axis, 0..15
    end_t: float
    start_local: int              # frame index inside the clip, 0..n-1
    end_local: int
    score: float                  # best per-step action score (for ranking, not calibrated)
    source: str                   # 'predicted' or 'given'


@dataclass
class SpatialOut:
    sampled_local: np.ndarray     # [16] frame indices inside the clip
    tool_xy: np.ndarray           # [16, 3, 2] pixels of the original frame
    tool_conf: np.ndarray         # [16, 3]
    contact_xy: np.ndarray        # [16, 4, 2]
    contact_conf: np.ndarray      # [16, 4]
    roles: List[str]              # 3 entries from {'tool_anchor', 'tool_tip'}
    image_wh: Tuple[int, int]


@dataclass
class Seeds:
    tool_local: int               # sampled frame (index inside the clip) the tool points are seeded on
    contact_local: int
    fallback: bool                # True when no sampled frame fell inside the action range
    inside_n: int                 # number of sampled frames inside the action range


@dataclass
class Annotation:
    clip: ClipSpec
    semantic: SemanticOut
    temporal: TemporalOut
    spatial: SpatialOut
    seeds: Seeds
    prompt: List[Dict[str, Any]]
    tracks: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        sp, a0 = self.spatial, self.clip.a0
        points = []
        for i, f in enumerate(sp.sampled_local):
            for k in range(3):
                points.append(dict(frameIndex=a0 + int(f), type=sp.roles[k], x=float(sp.tool_xy[i, k, 0]), y=float(sp.tool_xy[i, k, 1]), conf=float(sp.tool_conf[i, k])))
            for k in range(4):
                points.append(dict(frameIndex=a0 + int(f), type="contact", x=float(sp.contact_xy[i, k, 0]), y=float(sp.contact_xy[i, k, 1]), conf=float(sp.contact_conf[i, k])))
        s, t = self.semantic, self.temporal
        return dict(
            clip=dict(a0=self.clip.a0, a1=self.clip.a1, n=self.clip.n, fps=self.clip.fps, video=self.clip.video),
            semantic=dict(combo_id=s.combo_id, combo=s.combo, tool=s.tool, action=s.action, conf=s.conf, probs=s.probs, source=s.source),
            action_range=dict(start=a0 + t.start_local, end=a0 + t.end_local, start_t=t.start_t, end_t=t.end_t, score=t.score, source=t.source),
            sampled_frames=[a0 + int(f) for f in sp.sampled_local], image_wh=list(sp.image_wh), points=points,
            seeds=dict(tool=a0 + self.seeds.tool_local, contact=a0 + self.seeds.contact_local, fallback=self.seeds.fallback, inside_n=self.seeds.inside_n),
            prompt=self.prompt, tracks=self.tracks, extra=self.extra,
        )
