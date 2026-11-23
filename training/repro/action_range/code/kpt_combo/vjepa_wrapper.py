import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _clean_backbone_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "").replace("backbone.", "")
        out[nk] = v
    return out


def _resolve_encoder_state(raw: Dict) -> Dict[str, torch.Tensor]:
    if "encoder" in raw and isinstance(raw["encoder"], dict):
        return _clean_backbone_keys(raw["encoder"])

    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        s = raw["state_dict"]
        if any(k.startswith("encoder.") for k in s.keys()):
            trimmed = {k[len("encoder.") :]: v for k, v in s.items() if k.startswith("encoder.")}
            return _clean_backbone_keys(trimmed)
        return _clean_backbone_keys(s)

    if all(isinstance(k, str) for k in raw.keys()):
        return _clean_backbone_keys(raw)

    raise ValueError("Unsupported checkpoint structure for VJEPA encoder.")


class VJEPAWrapper(nn.Module):
    """Frozen VJEPA encoder wrapper.

    forward(video) returns:
      - tokens_grid: [B, T', H', W', D]
      - (T', H', W', D)
      - has_cls: bool
      - token_order: str
    """

    def __init__(
        self,
        vjepa_repo: str = "",
        arch: str = "vit_large",
        img_size: int = 256,
        num_frames: int = 16,
        patch_size: int = 16,
        tubelet_size: int = 2,
        hf_repo: str = "",
        hf_local_files_only: bool = False,
        pretrained_from_hub: bool = False,
        checkpoint_path: str = "",
        freeze: bool = True,
        layer_indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.loaded_sources = []
        self.load_report: Dict[str, str] = {}
        self.component_report: Dict[str, Any] = {}
        self.backend = "repo"
        self.arch = arch
        self.img_size = int(img_size)
        parsed_layer_indices = [int(x) for x in (layer_indices or [])]
        if any(int(x) <= 0 for x in parsed_layer_indices):
            raise ValueError(f"`layer_indices` must be positive 1-based layer ids, got {parsed_layer_indices}")
        self.layer_indices = parsed_layer_indices

        self.hf_repo = hf_repo.strip()
        if self.hf_repo and (pretrained_from_hub or checkpoint_path):
            raise ValueError(
                "Conflicting VJEPA source arguments: `hf_repo` cannot be combined with "
                "`pretrained_from_hub` or `checkpoint_path`."
            )

        if self.hf_repo:
            from transformers import AutoModel

            hf_model = AutoModel.from_pretrained(self.hf_repo, local_files_only=bool(hf_local_files_only))
            self.encoder = hf_model.encoder
            self.backend = "hf"
            self.arch = str(getattr(hf_model.config, "architectures", ["VJEPA2Model"])[0])
            self.patch_size = int(getattr(hf_model.config, "patch_size", patch_size))
            self.tubelet_size = int(getattr(hf_model.config, "tubelet_size", tubelet_size))
            hidden_size = getattr(hf_model.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(self.encoder, "embed_dim")
            self.embed_dim = int(hidden_size)
            self.num_encoder_layers = int(getattr(hf_model.config, "num_hidden_layers", 0))
            if self.layer_indices:
                if self.num_encoder_layers <= 0:
                    raise ValueError("Cannot validate `layer_indices`: missing `num_hidden_layers` in HF config.")
                bad = [x for x in self.layer_indices if int(x) > self.num_encoder_layers]
                if bad:
                    raise ValueError(
                        f"`layer_indices` out of range for encoder depth={self.num_encoder_layers}: {bad}"
                    )
            self.loaded_sources.append("hf_repo")
            self.load_report["hf_repo"] = self.hf_repo
            self.load_report["checkpoint_path"] = ""
            self.load_report["missing_keys"] = "n/a"
            self.load_report["unexpected_keys"] = "n/a"
            available_components = []
            for comp_name in ("encoder", "predictor", "pooler", "classifier"):
                if hasattr(hf_model, comp_name):
                    available_components.append(comp_name)
            self.component_report = {
                "backend": "hf",
                "full_model_class": hf_model.__class__.__name__,
                "available_components": available_components,
                "used_component": "encoder",
                "ignored_components": [x for x in available_components if x != "encoder"],
                "frames_per_clip": int(getattr(hf_model.config, "frames_per_clip", num_frames)),
                "patch_size": self.patch_size,
                "tubelet_size": self.tubelet_size,
                "hidden_size": self.embed_dim,
                "num_encoder_layers": self.num_encoder_layers,
                "layer_indices": list(self.layer_indices),
            }
        else:
            if self.layer_indices:
                raise ValueError("`layer_indices` multi-layer output is only supported with HF VJEPA backend.")
            self.vjepa_repo = Path(vjepa_repo).resolve()
            if not self.vjepa_repo.is_dir():
                raise FileNotFoundError(f"VJEPA repo directory not found: {self.vjepa_repo}")
            if str(self.vjepa_repo) not in sys.path:
                sys.path.insert(0, str(self.vjepa_repo))

            from src.hub import backbones as hub_backbones

            official_base_url = "https://dl.fbaipublicfiles.com/vjepa2"
            if getattr(hub_backbones, "VJEPA_BASE_URL", "") != official_base_url:
                setattr(hub_backbones, "VJEPA_BASE_URL", official_base_url)

            arch_to_factory = {
                "vit_large": getattr(hub_backbones, "vjepa2_vit_large"),
                "vit_huge": getattr(hub_backbones, "vjepa2_vit_huge"),
                "vit_giant": getattr(hub_backbones, "vjepa2_vit_giant"),
                "vit_giant_384": getattr(hub_backbones, "vjepa2_vit_giant_384"),
            }
            for extra_arch in (
                "vjepa2_1_vit_base_384",
                "vjepa2_1_vit_large_384",
                "vjepa2_1_vit_giant_384",
                "vjepa2_1_vit_gigantic_384",
            ):
                factory = getattr(hub_backbones, extra_arch, None)
                if factory is not None:
                    arch_to_factory[extra_arch] = factory
            if arch not in arch_to_factory:
                raise ValueError(f"Unsupported VJEPA arch: {arch}")

            factory = arch_to_factory[arch]
            encoder, predictor = factory(
                pretrained=pretrained_from_hub,
                num_frames=num_frames,
                patch_size=patch_size,
                tubelet_size=tubelet_size,
            )
            self.encoder = encoder
            if pretrained_from_hub:
                self.loaded_sources.append("hub")

            if checkpoint_path:
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                enc_state = _resolve_encoder_state(ckpt)
                msg = self.encoder.load_state_dict(enc_state, strict=False)
                self.loaded_sources.append("checkpoint")
                self.load_report["checkpoint_path"] = str(checkpoint_path)
                self.load_report["missing_keys"] = str(len(msg.missing_keys))
                self.load_report["unexpected_keys"] = str(len(msg.unexpected_keys))
                print(
                    f"[VJEPA] loaded encoder weights from {checkpoint_path} "
                    f"(missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})"
                )
            else:
                self.load_report["checkpoint_path"] = ""
                self.load_report["missing_keys"] = "n/a"
                self.load_report["unexpected_keys"] = "n/a"
            self.load_report["hf_repo"] = ""
            self.patch_size = int(getattr(self.encoder, "patch_size", patch_size))
            self.tubelet_size = int(getattr(self.encoder, "tubelet_size", tubelet_size))
            self.embed_dim = int(getattr(self.encoder, "embed_dim"))
            self.num_encoder_layers = int(getattr(self.encoder, "num_layers", 0))
            available_components = ["encoder"]
            if predictor is not None:
                available_components.append("predictor")
            self.component_report = {
                "backend": "repo",
                "full_model_class": f"vjepa2_{arch}",
                "available_components": available_components,
                "used_component": "encoder",
                "ignored_components": [x for x in available_components if x != "encoder"],
                "pretrained_from_hub": bool(pretrained_from_hub),
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
                "patch_size": self.patch_size,
                "tubelet_size": self.tubelet_size,
                "hidden_size": self.embed_dim,
                "num_encoder_layers": self.num_encoder_layers,
                "layer_indices": list(self.layer_indices),
            }

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    @property
    def has_pretrained(self) -> bool:
        return len(self.loaded_sources) > 0

    def _tokens_flat_to_grid(self, tokens_flat: torch.Tensor, x_btchw: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int], bool]:
        bsz, n_tokens, d = tokens_flat.shape
        t_prime = int(x_btchw.shape[1] // self.tubelet_size)
        h_prime = int(x_btchw.shape[3] // self.patch_size)
        w_prime = int(x_btchw.shape[4] // self.patch_size)
        hw = h_prime * w_prime

        has_cls = False
        tokens_body = tokens_flat
        expected = t_prime * hw

        # Prefer config-derived T', but support encoder outputs with/without CLS and fallback T' inference.
        if n_tokens == expected + 1:
            has_cls = True
            tokens_body = tokens_flat[:, 1:, :]
            n_tokens = expected
        elif n_tokens != expected:
            if hw > 0 and (n_tokens - 1) > 0 and (n_tokens - 1) % hw == 0:
                has_cls = True
                tokens_body = tokens_flat[:, 1:, :]
                n_tokens = n_tokens - 1
            if hw > 0 and n_tokens % hw == 0:
                t_prime = n_tokens // hw
                expected = t_prime * hw
            if expected != n_tokens:
                raise ValueError(
                    f"Token count mismatch: N={tokens_flat.shape[1]}, body={n_tokens}, "
                    f"expected={expected} from T'={t_prime}, H'={h_prime}, W'={w_prime}"
                )

        tokens_grid = tokens_body.reshape(bsz, t_prime, h_prime, w_prime, d).contiguous()
        return tokens_grid, (t_prime, h_prime, w_prime, d), has_cls

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int], bool, str]:
        """
        Args:
          video: [B,T,3,H,W] or [B,3,T,H,W]
        """
        if video.ndim != 5:
            raise ValueError(f"video must be 5D, got shape={tuple(video.shape)}")

        if video.shape[2] == 3:
            x_btchw = video.contiguous()  # [B,T,3,H,W]
        elif video.shape[1] == 3:
            x_btchw = video.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,3,H,W]
        else:
            raise ValueError(f"Cannot infer channel dimension from shape={tuple(video.shape)}")

        with torch.no_grad():
            if self.backend == "hf":
                use_multi_layer = len(self.layer_indices) > 0
                out = self.encoder(
                    pixel_values_videos=x_btchw,
                    output_hidden_states=bool(use_multi_layer),
                )
                if use_multi_layer:
                    hidden_states = getattr(out, "hidden_states", None)
                    if hidden_states is None:
                        raise RuntimeError("HF encoder did not return hidden_states with output_hidden_states=True.")
                    grids = []
                    has_cls_any = False
                    meta_shape: Optional[Tuple[int, int, int, int]] = None
                    for layer_idx_1based in self.layer_indices:
                        flat = hidden_states[int(layer_idx_1based)]
                        grid_i, meta_i, has_cls_i = self._tokens_flat_to_grid(flat, x_btchw)
                        grids.append(grid_i)
                        has_cls_any = has_cls_any or bool(has_cls_i)
                        if meta_shape is None:
                            meta_shape = meta_i
                        elif meta_i != meta_shape:
                            raise ValueError(f"Inconsistent grid shape across layers: {meta_shape} vs {meta_i}")
                    tokens_grid = torch.stack(grids, dim=1)  # [B,K,T',H',W',D]
                    token_order = "t_major_then_h_w"
                    return tokens_grid, meta_shape, has_cls_any, token_order

                tokens_flat = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            else:
                x_bcthw = x_btchw.permute(0, 2, 1, 3, 4).contiguous()  # [B,3,T,H,W]
                tokens_flat = self.encoder(x_bcthw)  # [B,N,D], no CLS in VJEPA encoder output

        token_order = "t_major_then_h_w"  # PatchEmbed3D flatten order from Conv3D output
        tokens_grid, meta, has_cls = self._tokens_flat_to_grid(tokens_flat, x_btchw)
        return tokens_grid, meta, has_cls, token_order
