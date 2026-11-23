"""TactusNet: one frozen V-JEPA 2.1 encoder, four layers from coarse to fine (Which/What -> When -> How/Where -> tracks).

Architecture only: nothing here trains anything.  See README.md for what is verified and what is not.
"""
from .encoder import FakeEncoder, TokenEncoder, UniTAPTokenEncoder, VJEPAEncoder
from .labels import COMBOS, combo_id_of, split_combo, tool_action_ids
from .pyramid import TactusNet
from .schema import Annotation, ClipSpec, SemanticOut, SpatialOut, TemporalOut
from .semantic import SemanticHead, semantic_loss
from .spatial import MockSpatial, UniTAPSpatial
from .temporal import TemporalLayer
from .tracking import DryRunTracker, JobRef, ServerTracker

__all__ = ["TactusNet", "SemanticHead", "TemporalLayer", "UniTAPSpatial", "MockSpatial", "ServerTracker", "DryRunTracker", "JobRef", "VJEPAEncoder",
           "UniTAPTokenEncoder", "FakeEncoder", "ClipSpec", "Annotation", "SemanticOut", "TemporalOut", "SpatialOut", "COMBOS", "combo_id_of", "split_combo", "tool_action_ids", "semantic_loss"]
