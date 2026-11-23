"""Label conventions shared by every layer.

The combo id (0..5) is the only id that every code base agrees on.  Tool and action ids are numbered differently in
`kpt_combo` (the classification / interval code) and in the H613 contact-region export, so they are always derived from
the combo id here and never passed around on their own.
"""
from typing import Dict, List, Tuple

COMBOS: Tuple[str, ...] = ("hook|dissect", "grasper|grasp", "clipper|clip", "scissors|cut", "scissors|dissect", "grasper|dissect")
COMBO2ID: Dict[str, int] = {name: i for i, name in enumerate(COMBOS)}
TOOLS: Tuple[str, ...] = ("hook", "grasper", "clipper", "scissors")      # kpt_combo order
ACTIONS: Tuple[str, ...] = ("dissect", "grasp", "clip", "cut")           # kpt_combo order

ID_CONVENTIONS: Dict[str, Dict[str, Dict[str, int]]] = {
    # kpt_combo/loss.py::_derive_tool_action_id_from_combo
    "kpt_combo": {"tool": {"hook": 0, "grasper": 1, "clipper": 2, "scissors": 3}, "action": {"dissect": 0, "grasp": 1, "clip": 2, "cut": 3}},
    # training/unitap/third_party/affordance_h613/README.md (table of the six conditions)
    "h613": {"tool": {"hook": 0, "grasper": 1, "scissors": 2, "clipper": 3}, "action": {"dissect": 0, "grasp": 1, "cut": 2, "clip": 3}},
}


def combo_name(combo_id: int) -> str:
    if not 0 <= int(combo_id) < len(COMBOS):
        raise ValueError(f"combo_id must be in [0, {len(COMBOS)}), got {combo_id}")
    return COMBOS[int(combo_id)]


def split_combo(combo_id: int) -> Tuple[str, str]:
    tool, action = combo_name(combo_id).split("|")
    return tool, action


def combo_id_of(tool: str, action: str) -> int:
    key = f"{tool}|{action}"
    if key not in COMBO2ID:
        raise ValueError(f"{key!r} is not one of the {len(COMBOS)} supported combinations: {', '.join(COMBOS)}")
    return COMBO2ID[key]


def tool_action_ids(combo_id: int, convention: str = "kpt_combo") -> Tuple[int, int]:
    """(tool_id, action_id) of a combo under the given code base's numbering."""
    tool, action = split_combo(combo_id)
    conv = ID_CONVENTIONS[convention]
    return conv["tool"][tool], conv["action"][action]


def combo_to_tool_action_index(convention: str = "kpt_combo") -> Tuple[List[int], List[int]]:
    """Length-6 lists: for every combo id, the tool index and the action index (used to compose combo logits)."""
    pairs = [tool_action_ids(c, convention) for c in range(len(COMBOS))]
    return [p[0] for p in pairs], [p[1] for p in pairs]
