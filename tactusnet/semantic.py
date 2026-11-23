"""Semantic layer (Which / What): the 6-way tool-action classifier on frozen tokens.

Module attribute names (`combo_attn_pool`, `combo_head`, `combo_tool_spatial_pool`, `combo_action_temporal_pool`, `combo_tool_head`,
`combo_action_head`, `combo_action_tool_prior`) are the ones used by `kpt_combo/model.py`, so a probe checkpoint from that code base
loads into this module after stripping its prefix.  The building blocks are the original classes (vendored verbatim).
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._vendored_heads import CrossAttentionPooler, LinearClassifierHead, ToolConditionedActionPrior
from .labels import ACTIONS, COMBOS, TOOLS, combo_to_tool_action_index, split_combo
from .schema import SemanticOut


def structured_combo_logits(tool_logits: torch.Tensor, action_logits: torch.Tensor, convention: str = "kpt_combo") -> torch.Tensor:
    """Combo logit = tool logit + action logit for each of the 6 legal pairs; no other pair exists, so there is nothing to mask."""
    to_tool, to_action = combo_to_tool_action_index(convention)
    out = tool_logits.new_full((tool_logits.shape[0], len(COMBOS)), -1e9)
    for cid in range(len(COMBOS)):
        out[:, cid] = tool_logits[:, to_tool[cid]] + action_logits[:, to_action[cid]]
    return out


class SemanticHead(nn.Module):
    """mode='joint': one cross-attention pooler -> 6-way linear head.
       mode='dual' : a tool pooler + head (4 classes) and an action pooler + head (4 classes); the action query can be biased by the
                     tool probabilities (zero-initialised, detached); combo logits are composed from the two."""

    def __init__(self, mode: str = "dual", embed_dim: int = 1024, tool_prior: bool = True, feat_dropout: float = 0.0, convention: str = "kpt_combo") -> None:
        super().__init__()
        if mode not in ("joint", "dual"):
            raise ValueError(f"mode must be 'joint' or 'dual', got {mode!r}")
        self.mode, self.convention, self.feat_dropout = mode, convention, float(feat_dropout)
        if mode == "joint":
            self.combo_attn_pool = CrossAttentionPooler(in_dim=embed_dim)
            self.combo_head = LinearClassifierHead(in_dim=embed_dim, num_classes=len(COMBOS))
        else:
            self.combo_tool_spatial_pool = CrossAttentionPooler(in_dim=embed_dim)
            self.combo_action_temporal_pool = CrossAttentionPooler(in_dim=embed_dim)
            self.combo_tool_head = LinearClassifierHead(in_dim=embed_dim, num_classes=len(TOOLS))
            self.combo_action_head = LinearClassifierHead(in_dim=embed_dim, num_classes=len(ACTIONS))
            self.combo_action_tool_prior = ToolConditionedActionPrior(num_tool_classes=len(TOOLS), query_dim=embed_dim, init_alpha=0.0, detach_tool_prob=True) if tool_prior else None

    def _drop(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.feat_dropout, training=self.training) if self.feat_dropout > 0.0 else x

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        if tokens.ndim != 5:
            raise ValueError(f"tokens must be [B,T,H,W,D], got {tuple(tokens.shape)}")
        b, t, h, w, d = tokens.shape
        seq = tokens.reshape(b, t * h * w, d)
        if self.mode == "joint":
            return {"combo_logits": self.combo_head(self._drop(self.combo_attn_pool(seq)))}
        tool_logits = self.combo_tool_head(self._drop(self.combo_tool_spatial_pool(seq)))
        bias = None
        if self.combo_action_tool_prior is not None:
            bias, _ = self.combo_action_tool_prior(tool_logits)
        action_logits = self.combo_action_head(self._drop(self.combo_action_temporal_pool(seq, query_bias=bias)))
        return {"combo_logits": structured_combo_logits(tool_logits, action_logits, self.convention), "tool_logits": tool_logits, "action_logits": action_logits}

    @torch.inference_mode()
    def predict(self, tokens: torch.Tensor) -> SemanticOut:
        if tokens.shape[0] != 1:
            raise ValueError("predict() takes one clip at a time")
        probs = F.softmax(self.forward(tokens)["combo_logits"].float(), dim=-1)[0]
        cid = int(probs.argmax())
        tool, action = split_combo(cid)
        return SemanticOut(combo_id=cid, combo=COMBOS[cid], tool=tool, action=action, conf=float(probs[cid]), probs=[float(p) for p in probs], source="predicted")


def semantic_loss(out: Dict[str, torch.Tensor], combo_id: torch.Tensor, convention: str = "kpt_combo",
                  w_combo: float = 0.1, w_tool: float = 0.5, w_action: float = 0.5) -> torch.Tensor:
    """The objective of the reference configuration: 0.1*CE(combo) + 0.5*CE(tool) + 0.5*CE(action) (dual); CE(combo) alone (joint)."""
    if "tool_logits" not in out:
        return F.cross_entropy(out["combo_logits"], combo_id)
    to_tool, to_action = combo_to_tool_action_index(convention)
    tool_t = torch.tensor(to_tool, device=combo_id.device)[combo_id]
    action_t = torch.tensor(to_action, device=combo_id.device)[combo_id]
    return (w_combo * F.cross_entropy(out["combo_logits"], combo_id) + w_tool * F.cross_entropy(out["tool_logits"], tool_t)
            + w_action * F.cross_entropy(out["action_logits"], action_t))
