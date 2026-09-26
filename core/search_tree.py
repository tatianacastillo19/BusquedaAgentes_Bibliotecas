"""Bounded optional capture of search-tree states for visualization."""
from typing import Any, Dict, List, Optional


MAX_SEARCH_TREE_NODES = 200
SearchTreeTrace = List[Dict[str, Any]]


def start_tree_trace(trace: Optional[SearchTreeTrace], origin: Any) -> Optional[int]:
    if trace is None:
        return None
    trace.clear()
    return record_tree_state(trace, origin, None, 0.0)


def record_tree_state(
    trace: SearchTreeTrace,
    node: Any,
    parent_id: Optional[int],
    accumulated_cost: float,
    accepted: bool = True,
) -> Optional[int]:
    if len(trace) >= MAX_SEARCH_TREE_NODES:
        return None
    state_id = len(trace)
    trace.append({
        "id": state_id,
        "parent": parent_id,
        "node": str(node),
        "g": accumulated_cost,
        "accepted": accepted,
    })
    return state_id