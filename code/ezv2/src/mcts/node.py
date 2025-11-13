from xtructure import xtructure_dataclass, FieldDescriptor, Xtructurable
import jax.numpy as jnp
from .action_space import ActionSpace, DiscreteActionSpace, ContinuousActionSpace

EmbeddingState = Xtructurable
Node = Xtructurable

def make_embedding_state_class(embedding_type: jnp.dtype, embedding_shape: tuple[int, ...]) -> Xtructurable:
    """Make an embedding state class for the MCTS node."""

    @xtructure_dataclass
    class EmbeddingState(Xtructurable):
        embedding: FieldDescriptor[embedding_type, embedding_shape]
    return EmbeddingState

def make_node_class(embedding_state_class: EmbeddingState, action_space: ActionSpace) -> Xtructurable:
    """Make a node class for the MCTS."""

    if isinstance(action_space, DiscreteActionSpace):
        action_shape = action_space.get_shape()

        @xtructure_dataclass
        class Node(Xtructurable):
            state: FieldDescriptor[embedding_state_class] # state of the node / shape: (..., state_shape)
            visit_count: FieldDescriptor[jnp.int32, (), -1] # number of times the node has been visited / shape: ()
            parent_idx: FieldDescriptor[jnp.int32, (), -1] # index of the parent node / shape: ()
            action_from_parent_idx: FieldDescriptor[jnp.int32, (), -1] # action index taken to reach the node at this node at parent node / shape: ()
            children_idx: FieldDescriptor[jnp.int32, action_shape, -1] # indices of the child nodes / shape: (action_shape)
            value: FieldDescriptor[jnp.float32, (), -1] # value of the node / shape: ()

    elif isinstance(action_space, ContinuousActionSpace):
        sampling_number = action_space.get_sampling_number()

        @xtructure_dataclass
        class Node(Xtructurable):
            state: FieldDescriptor[embedding_state_class] # state of the node / shape: (..., state_shape)
            visit_count: FieldDescriptor[jnp.int32, (), -1] # number of times the node has been visited / shape: ()
            parent_idx: FieldDescriptor[jnp.int32, (), -1] # index of the parent node / shape: ()
            action_from_parent_idx: FieldDescriptor[jnp.int32, (), -1] # action index taken to reach the node at this node at parent node / shape: ()
            children_idx: FieldDescriptor[jnp.int32, sampling_number, -1] # indices of the child nodes / shape: (sampling_number)
            value: FieldDescriptor[jnp.float32, (), -1] # value of the node / shape: ()

    return Node
