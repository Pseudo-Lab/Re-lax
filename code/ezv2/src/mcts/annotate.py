
"""Shared constants and helpers for the local MCTS implementation."""

ROOT_INDEX = 0
"""Array slot used for the root node within packed `Tree.nodes`."""

NO_PARENT = -1
"""Sentinel stored in `parent_idx` when a node is the root."""

UNVISITED = -1
"""Sentinel stored in `children_idx` before a child has been expanded."""


def is_root(index: int) -> bool:
    """Returns True when `index` refers to the root node."""

    return index == ROOT_INDEX


def has_parent(index: int) -> bool:
    """Returns True when the node has a parent (i.e., is not the root)."""

    return index != NO_PARENT