"""TriDet (Shi et al., CVPR 2023), vendored from https://github.com/dingfengshi/TriDet (MIT).

Files are byte-identical to commit 764b8165 (libs/modeling/*.py) except meta_archs.py, losses.py and train_utils.py,
whose headers list the changes (see baselines/when/THIRD_PARTY.md). The LR schedule is TriDet's
libs/utils/lr_schedulers.py, which is identical to ActionFormer's, so it is imported from there.
"""
from .blocks import MaskedConv1D, LayerNorm, ConvBlock, SGPBlock, Scale, AffineDropPath
from .models import make_backbone, make_neck, make_meta_arch, make_generator
from . import backbones      # registers SGP / conv
from . import necks          # registers fpn / identity
from . import loc_generators # registers the point generator
from . import meta_archs     # registers TriDet
from ..actionformer.lr_schedulers import LinearWarmupCosineAnnealingLR
from .train_utils import make_optimizer, ModelEma
