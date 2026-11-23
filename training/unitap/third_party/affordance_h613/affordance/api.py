"""Small inference API around the checkpoint-compatible region model."""

import json
from pathlib import Path, PosixPath

import torch
import torch.nn.functional as F

from .model import H210ProposalPointTemporalRegion


def build_model(checkpoint=None, device='cpu'):
    """Build H613; optionally load an explicit, locally supplied checkpoint."""
    config = json.loads(Path(__file__).with_name('h613.json').read_text())
    model = H210ProposalPointTemporalRegion(**config)
    model.register_parameter('token_mask_value', torch.nn.Parameter(torch.zeros(1024)))
    if checkpoint is not None:
        # Historical H613 checkpoints also serialize Path-valued run arguments.
        with torch.serialization.safe_globals([PosixPath]):
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict_regions(model, tokens, tool_id, action_id, combo_id):
    """Return normalized xy corners [B,16,4,2] from [B,8,24,24,1024]."""
    if tokens.ndim != 5 or tuple(tokens.shape[1:]) != (8, 24, 24, 1024):
        raise ValueError('Expected tokens [B,8,24,24,1024], final encoder layer.')
    parameter = next(model.parameters())
    tokens = tokens.to(device=parameter.device, dtype=parameter.dtype)
    if not torch.isfinite(tokens).all():
        raise ValueError('Tokens must be finite.')
    batch = tokens.shape[0]
    ids = []
    for name, value, limit in [('tool', tool_id, 9), ('action', action_id, 9), ('combo', combo_id, 17)]:
        value = torch.as_tensor(value, device=tokens.device)
        if value.is_floating_point() or value.dtype == torch.bool:
            raise ValueError(f'{name} IDs must be integers.')
        value = value.long().reshape(-1)
        if value.numel() != batch or (value < 0).any() or (value >= limit).any():
            raise ValueError(f'{name} IDs must have B entries in [0,{limit}).')
        ids.append(value)
    model.eval()
    outputs = model(tokens, tokens.new_zeros(batch, 16, 4),
                    tokens.new_zeros(batch, 8, 1, 24, 24),
                    tokens.new_zeros(batch, 8, 3, 24, 24),
                    tokens.new_zeros(batch, 16, 9), *ids)
    return outputs['quad_xy']


def preprocess_rgb(frames):
    """Convert uint8 RGB [B,16,H,W,3] into normalized [B,16,3,384,384]."""
    if frames.dtype != torch.uint8 or frames.ndim != 5 or frames.shape[1] != 16 or frames.shape[-1] != 3:
        raise ValueError('Expected uint8 RGB frames [B,16,H,W,3].')
    b, t, h, w, _ = frames.shape
    frames = frames.permute(0, 1, 4, 2, 3).reshape(b*t, 3, h, w).float() / 255.0
    frames = F.interpolate(frames, size=(384, 384), mode='bilinear', align_corners=False)
    mean = frames.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = frames.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return ((frames - mean) / std).reshape(b, t, 3, 384, 384)


@torch.inference_mode()
def encode_video(encoder, rgb_frames):
    """Use the supplied frozen VJEPAWrapper to produce final-layer tokens."""
    device = next(encoder.parameters()).device
    encoder.eval()
    tokens, _, _, _ = encoder(preprocess_rgb(rgb_frames).to(device))
    if tuple(tokens.shape[1:]) != (8, 24, 24, 1024):
        raise ValueError(f'Unexpected V-JEPA2.1 feature shape: {tuple(tokens.shape)}')
    return tokens
