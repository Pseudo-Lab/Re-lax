from dataclasses import dataclass
from typing import Any, Callable, Generic, List, Sequence, Tuple, TypeVar

import jax.numpy as jnp
from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

from .annotate import NO_PARENT, ROOT_INDEX, UNVISITED, has_parent, is_root
from .node import Node

StateT = TypeVar("StateT")


@dataclass(frozen=True)
class MCTSCallbacks(Generic[StateT]):
    encode: Callable[[StateT], jnp.ndarray]
    decode: Callable[[jnp.ndarray], StateT]
    invalid_actions: Callable[[StateT], jnp.ndarray]
    apply_action: Callable[[StateT, int], StateT]
    is_terminal: Callable[[StateT], bool]
    transition_reward: Callable[[StateT, StateT, int], float]
    value: Callable[[StateT], float]


@dataclass(frozen=True)
class SearchTrace:
    path: Sequence[int]
    expanded_action: int | None
    leaf_value: float
    started_at_root: bool = False
    leaf_is_root: bool = False


def make_tree_class(node_class: Node, max_nodes: int, action_shape: tuple[int, ...]) -> Xtructurable:
    """Make a tree class for the MCTS."""

    @xtructure_dataclass
    class Tree(Xtructurable):
        root_invalid_actions: FieldDescriptor[jnp.uint8, action_shape]
        nodes: FieldDescriptor[node_class, (max_nodes,)]
        next_free_idx: FieldDescriptor[jnp.int32, (), 1]

        @classmethod
        def from_root(cls, callbacks: MCTSCallbacks[StateT], root_state: StateT) -> "Tree":
            tree = cls.default()
            root_embedding = callbacks.encode(root_state)
            embedding_state_cls = getattr(cls._node_class, "_embedding_state_cls")
            root_node = cls._node_class.default().replace(
                state=embedding_state_cls(embedding=root_embedding),
                parent_idx=jnp.array(NO_PARENT, dtype=jnp.int32),
                action_from_parent_idx=jnp.array(NO_PARENT, dtype=jnp.int32),
            )
            nodes = tree.nodes.at[ROOT_INDEX].set(root_node)
            invalid_mask = cls._coerce_action_mask(
                callbacks.invalid_actions(root_state),
                context="callbacks.invalid_actions(root_state)",
            )
            return tree.replace(
                nodes=nodes,
                root_invalid_actions=invalid_mask,
                next_free_idx=jnp.array(1, dtype=jnp.int32),
            )

        def run_search(
            self, callbacks: MCTSCallbacks[StateT], num_iterations: int, discount: float = 1.0
        ) -> Tuple["Tree", List[SearchTrace]]:
            tree = self
            traces: List[SearchTrace] = []
            for _ in range(num_iterations):
                tree, trace = tree._run_iteration(callbacks, discount)
                traces.append(trace)
            return tree, traces

        def ranked_actions(self) -> Sequence[int]:
            root = self.nodes[ROOT_INDEX]
            visits = jnp.asarray(root.children_visits, dtype=jnp.int32)
            return [int(idx) for idx in jnp.argsort(-visits)]

        def get_action_statistics(self) -> dict[int, dict[str, float]]:
            root = self.nodes[ROOT_INDEX]
            stats: dict[int, dict[str, float]] = {}
            for action in range(root.children_visits.shape[0]):
                visits = float(root.children_visits[action])
                total_value = float(root.children_values[action])
                stats[action] = {
                    "visits": visits,
                    "value_sum": total_value,
                    "mean_value": total_value / visits if visits else 0.0,
                    "child_index": int(root.children_idx[action]),
                }
            return stats

        def _run_iteration(
            self, callbacks: MCTSCallbacks[StateT], discount: float
        ) -> Tuple["Tree", SearchTrace]:
            path, leaf_idx, expand_action, leaf_state = self._select(callbacks)
            tree = self
            if expand_action is None or callbacks.is_terminal(leaf_state):
                leaf_value = callbacks.value(leaf_state)
            else:
                tree, child_idx, child_state = tree._expand(
                    leaf_idx, expand_action, leaf_state, callbacks, discount
                )
                path = [*path, child_idx]
                leaf_value = callbacks.value(child_state)
            tree = tree._backprop(path, float(leaf_value))
            path_root = path[0] if path else ROOT_INDEX
            leaf_idx = path[-1] if path else ROOT_INDEX
            trace = SearchTrace(
                path=tuple(path),
                expanded_action=expand_action,
                leaf_value=float(leaf_value),
                started_at_root=is_root(int(path_root)),
                leaf_is_root=is_root(int(leaf_idx)),
            )
            return tree, trace

        def _select(
            self, callbacks: MCTSCallbacks[StateT]
        ) -> Tuple[List[int], int, int | None, StateT]:
            node_idx = ROOT_INDEX
            path: List[int] = [node_idx]

            if not is_root(node_idx):
                raise ValueError("Selection phase must be rooted at ROOT_INDEX.")

            while True:
                node = self.nodes[node_idx]
                state = callbacks.decode(node.state.embedding)
                if int(node.visit_count) == 0:
                    return path, node_idx, None, state
                if callbacks.is_terminal(state):
                    return path, node_idx, None, state

                invalid_mask = self._coerce_action_mask(
                    callbacks.invalid_actions(state),
                    context="callbacks.invalid_actions(state)",
                )
                unexplored = jnp.logical_and(invalid_mask == 0, node.children_idx == UNVISITED)
                if bool(jnp.any(unexplored)):
                    action = int(jnp.argmax(unexplored.astype(jnp.int32)))
                    return path, node_idx, action, state

                best_action = self._pick_best_child(node, invalid_mask)
                child_idx = int(node.children_idx[best_action])
                if child_idx == UNVISITED:
                    return path, node_idx, best_action, state
                path.append(child_idx)
                node_idx = child_idx

        def _expand(
            self,
            parent_idx: int,
            action_idx: int,
            parent_state: StateT,
            callbacks: MCTSCallbacks[StateT],
            discount: float,
        ) -> Tuple["Tree", int, StateT]:
            child_state = callbacks.apply_action(parent_state, action_idx)
            child_embedding = callbacks.encode(child_state)
            child_idx = int(self.next_free_idx)
            if child_idx >= self._max_nodes:
                raise RuntimeError("Tree saturated; increase `max_nodes`.")

            embedding_state_cls = getattr(self._node_class, "_embedding_state_cls")
            child_node = self._node_class.default().replace(
                state=embedding_state_cls(embedding=child_embedding),
                parent_idx=jnp.array(parent_idx, dtype=jnp.int32),
                action_from_parent_idx=jnp.array(action_idx, dtype=jnp.int32),
            )

            nodes = self.nodes.at[child_idx].set(child_node)
            tree = self.replace(
                nodes=nodes,
                next_free_idx=self.next_free_idx + jnp.array(1, dtype=jnp.int32),
            )

            reward = callbacks.transition_reward(parent_state, child_state, action_idx)
            parent_node = tree.nodes[parent_idx]
            parent_node = parent_node.replace(
                children_idx=parent_node.children_idx.at[action_idx].set(child_idx),
                children_rewards=parent_node.children_rewards.at[action_idx].set(reward),
                children_discounts=parent_node.children_discounts.at[action_idx].set(discount),
                children_visits=parent_node.children_visits.at[action_idx].set(1),
                children_values=parent_node.children_values.at[action_idx].set(reward),
            )
            tree = tree._update_node(parent_idx, parent_node)
            return tree, child_idx, child_state

        def _backprop(self, path: Sequence[int], leaf_value: float) -> "Tree":
            tree = self
            value = float(leaf_value)
            for idx in reversed(path):
                node = tree.nodes[idx]
                node = node.replace(
                    visit_count=node.visit_count + jnp.array(1, dtype=jnp.int32),
                    value=node.value + jnp.array(value, dtype=jnp.float32),
                    raw_value=jnp.array(value, dtype=jnp.float32),
                )
                tree = tree._update_node(idx, node)

                parent_idx = int(node.parent_idx)
                if has_parent(parent_idx):
                    parent = tree.nodes[parent_idx]
                    action = int(node.action_from_parent_idx)
                    parent = parent.replace(
                        children_visits=parent.children_visits.at[action].add(1),
                        children_values=parent.children_values.at[action].add(-value),
                    )
                    tree = tree._update_node(parent_idx, parent)
                value = -value
            return tree

        def _pick_best_child(self, node: Node, invalid_mask: jnp.ndarray) -> int:
            visits = node.children_visits.astype(jnp.float32) + 1e-6
            values = node.children_values
            total_visits = jnp.maximum(node.visit_count.astype(jnp.float32), 1.0)
            ucb = values / visits + jnp.sqrt(total_visits) / visits
            valid_mask = jnp.where(invalid_mask == 0, 1.0, 0.0)
            ucb = jnp.where(valid_mask > 0, ucb, -jnp.inf)
            return int(jnp.argmax(ucb))

        def _update_node(self, idx: int, node: Node) -> "Tree":
            nodes = self.nodes.at[idx].set(node)
            return self.replace(nodes=nodes)

        @classmethod
        def action_metadata(cls):
            """Expose the action-space metadata baked into the node class."""

            return getattr(cls._node_class, "_action_metadata", None)

        @classmethod
        def describe_action_space(cls) -> dict[str, Any] | None:
            """Return action metadata in a logging-friendly format."""

            metadata = cls.action_metadata()
            if metadata is None:
                return None
            return {
                "kind": metadata.kind,
                "size": int(metadata.size),
                "shape": tuple(int(dim) for dim in metadata.shape),
                "nvec": tuple(int(dim) for dim in metadata.nvec) if metadata.nvec else None,
                "low": metadata.low.tolist() if metadata.low is not None else None,
                "high": metadata.high.tolist() if metadata.high is not None else None,
                "sampling_number": metadata.sampling_number,
            }

        def telemetry_snapshot(self) -> dict[str, Any]:
            """Expose high-level telemetry for logging/monitoring."""

            return {
                "max_nodes": int(self._max_nodes),
                "allocated_nodes": int(self.next_free_idx),
                "action_space": self.describe_action_space(),
            }

        def describe_trace(self, trace: SearchTrace) -> dict[str, Any]:
            """Convert a SearchTrace into telemetry-friendly data."""

            path = tuple(int(idx) for idx in trace.path)
            return {
                "path": path,
                "path_length": len(path),
                "expanded_action": trace.expanded_action,
                "leaf_value": trace.leaf_value,
                "started_at_root": trace.started_at_root,
                "leaf_is_root": trace.leaf_is_root,
            }

        @classmethod
        def _expected_action_size(cls) -> int:
            metadata = cls.action_metadata()
            if metadata is not None:
                return int(metadata.size)
            return int(jnp.prod(jnp.asarray(cls._action_shape)))

        @classmethod
        def _coerce_action_mask(
            cls, mask: jnp.ndarray, *, context: str = "action mask"
        ) -> jnp.ndarray:
            arr = jnp.asarray(mask, dtype=jnp.uint8)
            expected_size = cls._expected_action_size()
            if arr.size != expected_size:
                raise ValueError(
                    f"{context} produced size {arr.size}, expected {expected_size} "
                    f"(shape {cls._action_shape})."
                )
            return jnp.reshape(arr, cls._action_shape)

    setattr(Tree, "_node_class", node_class)
    setattr(Tree, "_action_shape", action_shape)
    setattr(Tree, "_max_nodes", max_nodes)
    return Tree