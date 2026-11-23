# Ported from https://github.com/zgspose/DSTA (Apache License 2.0, see LICENSE in this dir) at commit
# cb683c1e21d6bd6977bedb7965d2bfb3911399c4 (2025-04-29):
#   posetimation/zoo/DSTA/dsta_std_resnet50.py  Attention, Mlp, Block, Linear, DSTA_STD_ResNet50 (here: DSTA)
#   posetimation/zoo/backbones/Resnet.py        BasicBlock, Bottleneck, ResNet
#   posetimation/loss/criterion.py              RLELoss (here: rle_loss)
# RealNVP: DSTA imports nets/nett/RealNVP from its own mmpose copy (mmpose.models.heads.rle_regression_head, not in the
# repo); the same flow (RLE's 3 coupling pairs of 64-unit MLPs) is taken from mmpose v1.3.2 mmpose/models/utils/realnvp.py
# (Apache License 2.0, commit 5408bc76f5b848cf925a0d1857899011d8c5b497), with DSTA's prior N(0.5, I).
# DropPath: timm.layers.drop_path (Apache License 2.0), inlined.
#
# Modified for TACTUS (notice under section 4(b) of the Apache License):
#   - plain PyTorch 2.x: no mmpose / mmcv / easydict / einops / timm imports (rearrange written as reshape/permute);
#   - num_joints and num_frame are arguments (TACTUS: 36 joints; 5 frames = key frame +- 2) instead of 17 and 3;
#   - the joint-wise local-awareness groups of the spatial blocks are an argument (TACTUS: the 4 tool groups of 3 points
#     and the 6 tool-action pair groups of 4 contact points, one positional table per group size) instead of PoseTrack's
#     5 limbs of 3 joints, its 2 unused tokens and its joint reorder;
#   - forward takes frames as [B, F, 3, H, W] (the original concatenated them on channels) and returns (coord, sigma);
#     the flow term of the RLE loss (computed in the original forward when training) moved into rle_loss();
#   - init_weights() takes a backbone state dict (TACTUS: torchvision ImageNet ResNet-50) instead of a checkpoint path;
#   - the training-time PCK logging (get_accuracy) is dropped.
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training: return x
    keep = 1 - drop_prob; r = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
    return x * r.div_(keep)


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__(); self.drop_prob = drop_prob

    def forward(self, x): return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------- backbone (DSTA posetimation/zoo/backbones/Resnet.py)
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride); self.bn1 = norm_layer(planes); self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes); self.bn2 = norm_layer(planes); self.downsample = downsample; self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False); self.bn1 = norm_layer(planes, momentum=0.1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False); self.bn2 = norm_layer(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False); self.bn3 = norm_layer(planes * 4, momentum=0.1)
        self.downsample = downsample; self.stride = stride

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: residual = self.downsample(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, architecture, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self._norm_layer = norm_layer
        layers = {"resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3], "resnet50": [3, 4, 6, 3], "resnet101": [3, 4, 23, 3], "resnet152": [3, 8, 36, 3]}
        self.inplanes = 64; self.block = BasicBlock if architecture in ("resnet18", "resnet34") else Bottleneck
        self.layers = layers[architecture]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64, eps=1e-5, momentum=0.1, affine=True); self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(self.block, 64, self.layers[0])
        self.layer2 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self.make_layer(self.block, 512, self.layers[3], stride=2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                                       self._norm_layer(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes, norm_layer=self._norm_layer) for _ in range(1, blocks)]
        return nn.Sequential(*layers)


# ---------------------------------------------------------------- DSTA modules (posetimation/zoo/DSTA/dsta_std_resnet50.py)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads; head_dim = dim // num_heads; self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias); self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim); self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features; hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features); self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features); self.drop = nn.Dropout(drop)

    def forward(self, x): return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class Linear(nn.Module):
    """RLE's normalized linear layer: y = x W^T / ||x|| + b."""
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super().__init__()
        self.bias, self.norm = bias, norm; self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())
        if self.norm: y = y / torch.norm(x, dim=1, keepdim=True)
        if self.bias: y = y + self.linear.bias
        return y


class RealNVP(nn.Module):
    """RealNVP flow of RLE (mmpose v1.3.2 mmpose/models/utils/realnvp.py), prior N(prior_loc, I) (DSTA: loc 0.5)."""
    @staticmethod
    def get_scale_net():
        return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def get_trans_net():
        return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))

    @property
    def prior(self): return distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self, prior_loc=0.0):
        super().__init__()
        self.register_buffer("loc", torch.zeros(2) + prior_loc); self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))
        self.s = nn.ModuleList([self.get_scale_net() for _ in range(len(self.mask))])
        self.t = nn.ModuleList([self.get_trans_net() for _ in range(len(self.mask))])
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i]); t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        z, log_det = self.backward_p(x); return self.prior.log_prob(z) + log_det


class DSTA(nn.Module):
    """DSTA_STD_ResNet50: ResNet-50 per frame -> joint tokens (JFD) -> spatial blocks over local joint groups of the key
    frame + temporal blocks per joint over the frames -> weighted mean -> (coord, sigma) per joint (RLE)."""
    def __init__(self, num_joints, groups, num_frame=5, embed_dim=32, depth=4, drop_path_rate=0.1):
        super().__init__()
        self.num_joints, self.num_frame, self.embed_dim_ratio = num_joints, num_frame, embed_dim
        self.preact = ResNet("resnet50")
        self.group_sizes = sorted({len(g) for g in groups})
        for n in self.group_sizes:   # joint indices of all groups of size n, [G_n, n]
            self.register_buffer(f"groups_{n}", torch.tensor([g for g in groups if len(g) == n], dtype=torch.long), persistent=False)
        self.Spatial_pos_embed = nn.ParameterDict({str(n): nn.Parameter(torch.zeros(n, embed_dim)) for n in self.group_sizes})
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.Spatial_blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0.,
                                                   attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=8, mlp_ratio=2, qkv_bias=True, qk_scale=None, drop=0.,
                                           attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.Spatial_norm = norm_layer(embed_dim); self.Temporal_norm = norm_layer(embed_dim)
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.weighted_mean = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        self.head_coord = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2))
        self.head_sigma = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2))
        self.flow = RealNVP(prior_loc=0.5)                       # DSTA: MultivariateNormal(torch.zeros(2) + 0.5, torch.eye(2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_coord = Linear(2048, num_joints * embed_dim)

    def forward(self, x):
        """x [B, F, 3, H, W] (frames in time order, key frame in the middle) -> coord [B, J, 2] in [0, 1] of the input,
        sigma [B, J, 2]."""
        B, Fn = x.shape[:2]; J, C = self.num_joints, self.embed_dim_ratio
        x = x.transpose(0, 1).reshape(Fn * B, *x.shape[2:])            # original: torch.cat(x.split(3, dim=1), 0)
        if x.is_cuda: x = x.contiguous(memory_format=torch.channels_last)
        feat = self.avg_pool(self.preact(x)).reshape(Fn * B, -1)
        feat = self.fc_coord(feat).reshape(Fn * B, J, C)
        feat = torch.stack(feat.split(B, dim=0), dim=0)                 # [F, B, J, C]

        # SD: spatial blocks within each local group of the key frame
        feat_S = feat[Fn // 2]; out_S = feat_S.clone()
        for n in self.group_sizes:
            idx = getattr(self, f"groups_{n}"); G = idx.shape[0]
            t = feat_S[:, idx] + self.Spatial_pos_embed[str(n)]          # [B, G, n, C]
            t = t.reshape(B * G, n, C)
            for blk in self.Spatial_blocks: t = blk(t)
            out_S[:, idx.reshape(-1)] = self.Spatial_norm(t).reshape(B, G * n, C)

        # TD: temporal blocks per joint over the frames
        feat_T = feat.permute(1, 2, 0, 3).reshape(B * J, Fn, C) + self.Temporal_pos_embed   # 'f b p c -> (b p) f c'
        for blk in self.blocks: feat_T = blk(feat_T)
        feat_T = self.Temporal_norm(feat_T).reshape(B, J, Fn, C).permute(2, 0, 1, 3)[Fn // 2]

        # aggregation and heads
        f = self.weighted_mean(torch.stack((feat_T, out_S), dim=1)).squeeze(1)
        coord = self.head_coord(f).reshape(B, J, 2)
        sigma = self.head_sigma(f).reshape(B, J, 2).sigmoid()
        return coord, sigma

    def init_weights(self, backbone_state=None):
        """DSTA's init_weights: Conv2d N(0, 0.001), BN (1, 0), every bias 0; then the pretrained backbone (preact)."""
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1); nn.init.constant_(module.bias, 0)
            else:
                for name, _ in module.named_parameters():
                    if name == "bias": nn.init.constant_(module.bias, 0)
        if backbone_state is not None:
            sd = {k: v for k, v in backbone_state.items() if not k.startswith("fc.")}
            r = self.preact.load_state_dict(sd, strict=False)
            if r.missing_keys or r.unexpected_keys:
                raise RuntimeError(f"backbone load: missing {r.missing_keys[:5]}, unexpected {r.unexpected_keys[:5]}")


AMP = 1 / math.sqrt(2 * math.pi)


def rle_loss(coord, sigma, target, weight, flow):
    """DSTA's RLE loss (posetimation/loss/criterion.py RLELoss, size_average) with the flow term of its forward:
    nf = log(sigma) - log phi((coord - target) / sigma) weighted by w[..., :1], plus
    log Q = log(sigma / amp) + |target - coord| / (sqrt(2) sigma + 1e-9) weighted by w; summed, divided by the batch."""
    coord, sigma, target, weight = coord.float(), sigma.float(), target.float(), weight.float()
    B, J = coord.shape[:2]
    bar_mu = (coord - target) / sigma
    log_phi = flow.log_prob(bar_mu.reshape(-1, 2)).reshape(B, J, 1)
    nf_loss = (torch.log(sigma) - log_phi) * weight[:, :, :1]
    q = (torch.log(sigma / AMP) + torch.abs(target - coord) / (math.sqrt(2) * sigma + 1e-9)) * weight
    loss = nf_loss + q
    return loss.sum() / len(loss) if weight.sum() > 0 else loss.sum()
