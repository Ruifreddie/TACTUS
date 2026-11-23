"""Heatmap codecs of the two heatmap baselines, ported from their official code (see THIRD_PARTY.md).

DarkCodec (HRNet-W32 + DARK), from DarkPose (github.com/ilovepose/DarkPose, Apache-2.0, commit 0185b427):
  encode  lib/dataset/JointsDataset.py generate_target + adjust_target_weight: a full-grid Gaussian (sigma 3) at the exact
          sub-pixel heatmap location ("unbiased" encoding); weight 0 when the 3-sigma box lies outside the heatmap.
  decode  lib/core/inference.py get_final_preds: argmax (0 where the max is <= 0), Gaussian modulation (zero-padded
          cv2.GaussianBlur, rescaled to the original maximum, kernel TEST.BLUR_KERNEL = 11), log, second-order Taylor step.
  Heatmap coordinates are the plain rescale of the frame to the heatmap grid (DarkPose maps joints with a separate
  heatmap-size affine), i.e. hm = (in + 0.5) * hm_size / in_size - 0.5 in pixel-centre coordinates.
UDPCodec (ViTPose-B), from the official ViTPose code (github.com/ViTAE-Transformer/ViTPose, mmpose 0.x, Apache-2.0):
  encode  mmpose/datasets/pipelines/top_down_transform.py TopDownGenerateTarget(encoding='UDP', GaussianHeatmap), sigma 2:
          hm = in * (hm_size - 1) / (in_size - 1); Gaussian on a (6 sigma + 1)^2 window around int(hm + 0.5).
  decode  mmpose/core/evaluation/top_down_eval.py keypoints_from_heatmaps(use_udp=True, post_process='default',
          kernel=11): _get_max_preds (-1 where the max is <= 0) + post_dark_udp (in-place cv2.GaussianBlur, clip
          [0.001, 50], log, Newton step with the 3x3 Hessian); back to input pixels with (in_size - 1) / (hm_size - 1).
Encoding runs batched on the GPU for the 7 selected channels; decoding is numpy, as in the official code.
"""
import cv2
import numpy as np
import torch


def _grid(H, W, dev):
    return (torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, 1, W),
            torch.arange(H, device=dev, dtype=torch.float32).view(1, 1, H, 1))


class DarkCodec:
    def __init__(self, input_wh, heatmap_wh, sigma=3, kernel=11):
        self.input_wh, self.heatmap_wh, self.sigma, self.kernel = tuple(input_wh), tuple(heatmap_wh), sigma, kernel
        self.ratio = np.array(heatmap_wh, np.float64) / np.array(input_wh, np.float64)

    def to_hm(self, kp_in):
        r = torch.as_tensor(self.ratio, dtype=torch.float32, device=kp_in.device); return (kp_in + 0.5) * r - 0.5

    def encode(self, kp_in, vis):
        """kp_in [B,K,2] input pixels, vis [B,K] -> target heatmaps [B,K,H,W], weights [B,K]."""
        W, H = self.heatmap_wh; r = 3 * self.sigma; mu = self.to_hm(kp_in.float()); x, y = _grid(H, W, mu.device)
        ul = torch.trunc(mu - r); br = torch.trunc(mu + r + 1)                         # int() as adjust_target_weight
        w = vis.float() * ~((ul[..., 0] >= W) | (ul[..., 1] >= H) | (br[..., 0] < 0) | (br[..., 1] < 0))
        g = torch.exp(-((x - mu[..., 0:1, None]) ** 2 + (y - mu[..., 1:2, None]) ** 2) / (2 * self.sigma ** 2))
        return g * (w > 0.5).float()[..., None, None], w

    def decode(self, heat):
        """heat [N,K,H,W] numpy -> keypoints [N,K,2] in input pixels, max values [N,K]."""
        N, K, H, W = heat.shape; hm = np.array(heat, np.float32).reshape(N * K, H, W)
        flat = hm.reshape(N * K, -1); idx = flat.argmax(1); vals = flat[np.arange(N * K), idx]
        coords = np.stack([idx % W, idx // W], 1).astype(np.float32) * (vals > 0)[:, None]   # get_max_preds
        b = (self.kernel - 1) // 2                                                            # gaussian_blur
        for i in range(N * K):
            m0 = hm[i].max(); dr = np.zeros((H + 2 * b, W + 2 * b)); dr[b:-b, b:-b] = hm[i]
            dr = cv2.GaussianBlur(dr, (self.kernel, self.kernel), 0)[b:-b, b:-b]; m1 = dr.max()
            hm[i] = dr * (m0 / m1) if m1 != 0 else dr
        L = np.log(np.maximum(hm, 1e-10)).astype(np.float64)
        px, py = coords[:, 0].astype(int), coords[:, 1].astype(int)                           # taylor
        j = np.where((px > 1) & (px < W - 2) & (py > 1) & (py < H - 2))[0]
        if len(j):
            X, Y, Lj, a = px[j], py[j], L[j], np.arange(len(j))
            g = lambda dy, dx: Lj[a, Y + dy, X + dx]
            dx = 0.5 * (g(0, 1) - g(0, -1)); dy = 0.5 * (g(1, 0) - g(-1, 0))
            dxx = 0.25 * (g(0, 2) - 2 * g(0, 0) + g(0, -2)); dyy = 0.25 * (g(2, 0) - 2 * g(0, 0) + g(-2, 0))
            dxy = 0.25 * (g(1, 1) - g(-1, 1) - g(1, -1) + g(-1, -1))
            det = dxx * dyy - dxy ** 2; ok = det != 0; d = np.where(ok, det, 1.0)
            ox = -(dyy * dx - dxy * dy) / d; oy = -(-dxy * dx + dxx * dy) / d                  # -H^-1 @ [dx, dy]
            coords[j[ok], 0] += ox[ok].astype(np.float32); coords[j[ok], 1] += oy[ok].astype(np.float32)
        kp = (coords.reshape(N, K, 2) + 0.5) / self.ratio.astype(np.float32) - 0.5
        return kp, vals.reshape(N, K)

    def describe(self):
        return {"codec": "DARK (DarkPose): unbiased Gaussian targets + distribution-aware Taylor decoding", "input_wh": self.input_wh,
                "heatmap_wh": self.heatmap_wh, "sigma": self.sigma, "blur_kernel": self.kernel}


class UDPCodec:
    def __init__(self, input_wh, heatmap_wh, sigma=2, kernel=11):
        self.input_wh, self.heatmap_wh, self.sigma, self.kernel = tuple(input_wh), tuple(heatmap_wh), sigma, kernel
        self.stride = (np.array(input_wh, np.float64) - 1) / (np.array(heatmap_wh, np.float64) - 1)   # feat_stride

    def encode(self, kp_in, vis):
        W, H = self.heatmap_wh; r = 3 * self.sigma
        mu_ac = kp_in.float() / torch.as_tensor(self.stride, dtype=torch.float32, device=kp_in.device)
        mu = torch.trunc(mu_ac + 0.5); x, y = _grid(H, W, mu.device)                           # int(x / stride + 0.5)
        ul = mu - r; br = mu + r + 1
        w = vis.float() * ~((ul[..., 0] >= W) | (ul[..., 1] >= H) | (br[..., 0] < 0) | (br[..., 1] < 0))
        win = ((x - mu[..., 0:1, None]).abs() <= r) & ((y - mu[..., 1:2, None]).abs() <= r)
        g = torch.exp(-((x - mu_ac[..., 0:1, None]) ** 2 + (y - mu_ac[..., 1:2, None]) ** 2) / (2 * self.sigma ** 2)) * win
        return g * (w > 0.5).float()[..., None, None], w

    def decode(self, heat):
        N, K, H, W = heat.shape; hm = np.array(heat, np.float32).reshape(N * K, H, W)
        flat = hm.reshape(N * K, -1); idx = flat.argmax(1); vals = flat[np.arange(N * K), idx]
        coords = np.stack([idx % W, idx // W], 1).astype(np.float32); coords[vals <= 0] = -1      # _get_max_preds
        for i in range(N * K): cv2.GaussianBlur(hm[i], (self.kernel, self.kernel), 0, hm[i])     # post_dark_udp
        np.clip(hm, 0.001, 50, hm); np.log(hm, hm)
        pad = np.pad(hm, ((0, 0), (1, 1), (1, 1)), mode="edge").reshape(-1).astype(np.float64)
        index = (coords[:, 0] + 1 + (coords[:, 1] + 1) * (W + 2) + (W + 2) * (H + 2) * np.arange(N * K)).astype(int)
        i_, ix1, iy1, ix1y1 = pad[index], pad[index + 1], pad[index + W + 2], pad[index + W + 3]
        ix1_y1_, ix1_, iy1_ = pad[index - W - 3], pad[index - 1], pad[index - 2 - W]
        dx = 0.5 * (ix1 - ix1_); dy = 0.5 * (iy1 - iy1_)
        dxx = ix1 - 2 * i_ + ix1_; dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hess = np.stack([np.stack([dxx, dxy], -1), np.stack([dxy, dyy], -1)], -2) + np.finfo(np.float32).eps * np.eye(2)
        coords = coords - np.einsum("imn,in->im", np.linalg.inv(hess), np.stack([dx, dy], -1)).astype(np.float32)
        return coords.reshape(N, K, 2) * self.stride.astype(np.float32), vals.reshape(N, K)

    def describe(self):
        return {"codec": "UDP (ViTPose / mmpose 0.x): UDP Gaussian targets + DARK-UDP decoding", "input_wh": self.input_wh,
                "heatmap_wh": self.heatmap_wh, "sigma": self.sigma, "blur_kernel": self.kernel}


def joints_mse(heat, ch, tgt, w, n_out, half):
    """Official JointsMSELoss(use_target_weight) over all n_out channels: mean over joints of MSE(pred*w, gt*w), times 0.5
    in HRNet/DarkPose (half=True), without it in mmpose 0.x / ViTPose. Only the 7 selected channels have nonzero weight,
    so the sum runs over them and is divided by the full n_out x B x H x W."""
    B, _, h, wd = heat.shape
    sel = heat.gather(1, ch[:, :, None, None].expand(-1, -1, h, wd)).float()
    loss = (((sel - tgt) * w[:, :, None, None]) ** 2).sum() / (B * n_out * h * wd)
    return 0.5 * loss if half else loss


def select(heat, ch):
    h, w = heat.shape[-2:]; return heat.gather(1, ch[:, :, None, None].expand(-1, -1, h, w)).float().cpu().numpy()
