"""The four When methods behind one interface (base.Method)."""
from .mstcn import MSTCN
from .qddetr import QDDETR
from .tal import ActionFormer, TriDet

METHODS = {m.name: m for m in (ActionFormer, TriDet, QDDETR, MSTCN)}
