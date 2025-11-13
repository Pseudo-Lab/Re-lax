from xtructure import xtructure_dataclass, FieldDescriptor, Xtructurable
import jax.numpy as jnp
from .action_space import ActionSpace, DiscreteActionSpace, ContinuousActionSpace
from .node import Node

def make_tree_class(node_class: Node, max_nodes: int) -> Xtructurable:
    """Make a tree class for the MCTS."""

    @xtructure_dataclass
    class Tree(Xtructurable):
        nodes: FieldDescriptor[node_class, max_nodes, -1]

    return Tree