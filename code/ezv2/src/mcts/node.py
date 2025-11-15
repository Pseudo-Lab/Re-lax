from xtructure import xtructure_dataclass, FieldDescriptor, Xtructurable
import jax.numpy as jnp
from .action_space import ActionSpace, DiscreteActionSpace, ContinuousActionSpace
from .annotate import NO_PARENT, UNVISITED

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
        children_shape = action_space.get_shape()
    elif isinstance(action_space, ContinuousActionSpace):
        sampling_number = action_space.get_sampling_number()
        children_shape = (sampling_number,)

    @xtructure_dataclass
    class Node(Xtructurable):
        state: FieldDescriptor[embedding_state_class] # state of the node / shape: (..., state_shape)
        visit_count: FieldDescriptor[jnp.int32, (), 0] # number of times the node has been visited / shape: ()
        parent_idx: FieldDescriptor[jnp.int32, (), NO_PARENT] # index of the parent node / shape: ()
        action_from_parent_idx: FieldDescriptor[jnp.int32, (), NO_PARENT] # action index taken to reach the node at this node at parent node / shape: ()
        children_idx: FieldDescriptor[jnp.int32, children_shape, UNVISITED] # indices of the child nodes / shape: (children_shape)
        children_prior_logits: FieldDescriptor[jnp.float32, children_shape, 0.0] # prior logits of the child nodes / shape: (children_shape)
        children_rewards: FieldDescriptor[jnp.float32, children_shape, 0.0] # rewards of the child nodes / shape: (children_shape)
        children_visits: FieldDescriptor[jnp.int32, children_shape, 0] # visits of the child nodes / shape: (children_shape)
        children_discounts: FieldDescriptor[jnp.float32, children_shape, 0.0] # discounts of the child nodes / shape: (children_shape)
        children_values: FieldDescriptor[jnp.float32, children_shape, 0.0] # values of the child nodes / shape: (children_shape)
        raw_value: FieldDescriptor[jnp.float32, (), 0.0] # raw value of the node / shape: ()
        value: FieldDescriptor[jnp.float32, (), 0.0] # cumulative search value of the node / shape: ()

        def with_model_outputs(
            self,
            *,
            prior_logits: jnp.ndarray | None = None,
            value: float | jnp.ndarray | None = None,
        ) -> "Node":
            """Return a new node whose priors/value reflect model predictions."""

            node = self
            if prior_logits is not None:
                logits = self._coerce_prior_logits(prior_logits)
                node = node.replace(children_prior_logits=logits)
            if value is not None:
                val = jnp.asarray(value, dtype=jnp.float32)
                node = node.replace(raw_value=val, value=val)
            return node

        @classmethod
        def _coerce_prior_logits(cls, logits: jnp.ndarray) -> jnp.ndarray:
            arr = jnp.asarray(logits, dtype=jnp.float32)
            target_shape = cls._children_shape
            expected_size = int(jnp.prod(jnp.asarray(target_shape)))
            if arr.size != expected_size:
                raise ValueError(
                    f"Prior logits size {arr.size} does not match action space ({expected_size})."
                )
            return arr.reshape(target_shape)

        @classmethod
        def action_metadata(cls):
            return getattr(cls, "_action_metadata", None)

    setattr(Node, "_embedding_state_cls", embedding_state_class)
    setattr(Node, "_children_shape", children_shape)
    setattr(Node, "_action_space", action_space)
    setattr(Node, "_action_metadata", action_space.metadata())
    return Node
