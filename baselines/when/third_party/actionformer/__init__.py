"""ActionFormer (Zhang, Wu, Li, ECCV 2022), vendored from https://github.com/happyharrycn/actionformer_release (MIT).

Files are byte-identical to commit 61ea7eb9 (libs/modeling/*.py, libs/utils/lr_schedulers.py) except meta_archs.py,
losses.py and train_utils.py, whose headers list the changes (see baselines/when/THIRD_PARTY.md).
"""
from .blocks import (MaskedConv1D, MaskedMHCA, MaskedMHA, LayerNorm,
                     TransformerBlock, ConvBlock, Scale, AffineDropPath)
from .models import make_backbone, make_neck, make_meta_arch, make_generator
from . import backbones      # registers the backbones
from . import necks          # registers the necks
from . import loc_generators # registers the point generator
from . import meta_archs     # registers LocPointTransformer
from .lr_schedulers import LinearWarmupCosineAnnealingLR
from .train_utils import make_optimizer, ModelEma
