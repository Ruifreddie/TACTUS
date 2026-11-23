"""QD-DETR (Moon et al., CVPR 2023), vendored from https://github.com/wjun0830/QD-DETR (MIT).

attention.py, position_encoding.py, span_utils.py and misc.py are byte-identical to commit f8628f79; model.py,
transformer.py and matcher.py carry headers that list their changes (see baselines/when/THIRD_PARTY.md).
"""
from .model import QDDETR, SetCriterion, build_model
from .span_utils import span_cxw_to_xx, span_xx_to_cxw
