from __future__ import annotations
from uuid import UUID
from typing import Dict, List, Tuple

INTENT_TOOL_IDS: Dict[str, UUID] = {
    "find_one": UUID("11111111-1111-1111-1111-111111111111"),
    "build_set": UUID("22222222-2222-2222-2222-222222222222"),
    "out-of-scope": UUID("33333333-3333-3333-3333-333333333333"),
}

def build_intent_tools(ToolCls) -> List:
    """ToolCls = tool_suggest.models.Tool (или где он у них лежит)."""
    return [
        ToolCls(
            id=INTENT_TOOL_IDS["find_one"],
            name="intent_find_one",
            schema="Choose when user wants one specific academic paper.",
        ),
        ToolCls(
            id=INTENT_TOOL_IDS["build_set"],
            name="intent_build_set",
            schema="Choose when user wants a set/list of academic papers on a topic.",
        ),
        ToolCls(
            id=INTENT_TOOL_IDS["out-of-scope"],
            name="intent_oos",
            schema="Choose when user request is not about academic literature.",
        ),
    ]
