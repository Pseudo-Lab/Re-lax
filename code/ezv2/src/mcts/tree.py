from dataclasses import dataclass
from typing import Any, Callable, Generic, List, Sequence, Tuple, TypeVar
import jax

import jax.numpy as jnp
from xtructure import FieldDescriptor, Xtructurable, xtructure_dataclass

from .annotate import NO_PARENT, ROOT_INDEX, UNVISITED, has_parent, is_root
from .node import Node

StateT = TypeVar("StateT")

MAX_SEARCH_DEPTH = 128  # Maximum depth for search trace/path. Increase if game is very deep.

@dataclass(frozen=True)
class MCTSCallbacks(Generic[StateT]):
    encode: Callable[[StateT], jnp.ndarray]
    decode: Callable[[jnp.ndarray], StateT]
    invalid_actions: Callable[[StateT], jnp.ndarray]
    apply_action: Callable[[StateT, int], StateT]
    is_terminal: Callable[[StateT], bool]
    transition_reward: Callable[[StateT, StateT, int], float]
    value: Callable[[StateT], float]
    policy: Callable[[StateT], jnp.ndarray]


@xtructure_dataclass
class SearchTrace(Xtructurable):
    path: FieldDescriptor[jnp.int32, (MAX_SEARCH_DEPTH,), NO_PARENT]
    expanded_action: FieldDescriptor[jnp.int32, (), -1]
    leaf_value: FieldDescriptor[jnp.float32, (), 0.0]
    started_at_root: FieldDescriptor[jnp.bool_, (), False]
    leaf_is_root: FieldDescriptor[jnp.bool_, (), False]




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
            # Initialize root priors so first search has guidance
            root_priors = callbacks.policy(root_state)
            
            embedding_state_cls = getattr(cls._node_class, "_embedding_state_cls")
            root_node = cls._node_class.default().replace(
                state=embedding_state_cls(embedding=root_embedding),
                children_prior_logits=root_priors,
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
        ) -> Tuple["Tree", SearchTrace]: # Updated signature to return stacked trace
            
            def scan_body(tree: "Tree", _):
                next_tree, trace = tree._run_iteration(callbacks, discount)
                return next_tree, trace

            # Use jax.lax.scan to avoid unrolling the loop
            final_tree, traces_stacked = jax.lax.scan(
                scan_body, self, None, length=num_iterations
            )
            
            return final_tree, traces_stacked

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
            path_arr, path_len, leaf_idx, expand_action, leaf_state = self._select(callbacks)
            tree = self
            
            # expand_action is -1 if None
            should_expand = (expand_action != -1) & jnp.logical_not(callbacks.is_terminal(leaf_state))
            
            def _do_expand(t):
                return t._expand(
                    leaf_idx, expand_action, leaf_state, callbacks, discount
                )
            
            def _no_expand(t):
                # Return dummy child_idx and state, but they won't be used for backprop directly
                # value comes from leaf_state
                return t, -1, leaf_state # -1 child_idx
                
            tree, child_idx, child_state = jax.lax.cond(
                should_expand,
                _do_expand,
                _no_expand,
                tree
            )
            
            # If we expanded, we append child to path (conceptually).
            # But path_arr is fixed size. We update it.
            # Actually _select returns the path to the leaf.
            # If we expand, the new node is a child of leaf.
            # backprop needs the path from root to the node that was evaluated.
            # If expanded, evaluated node is child. Path includes child.
            # If not expanded, evaluated node is leaf. Path ends at leaf.
            
            final_path_len = jax.lax.select(should_expand, path_len + 1, path_len)
            final_path_arr = jax.lax.select(
                should_expand, 
                path_arr.at[path_len].set(child_idx),
                path_arr
            )
            
            # Calculate value
            leaf_value = jax.lax.cond(
                should_expand,
                lambda: callbacks.value(child_state),
                lambda: callbacks.value(leaf_state)
            )
            
            tree = tree._backprop(final_path_arr, final_path_len, leaf_value)
            
            path_root = final_path_arr[0]
            final_leaf_idx = final_path_arr[final_path_len - 1]
            
            trace = SearchTrace(
                path=final_path_arr, # Pass array directly
                expanded_action=jax.lax.select(expand_action == -1, jnp.int32(-1), expand_action),
                leaf_value=leaf_value,
                started_at_root=is_root(path_root),
                leaf_is_root=is_root(final_leaf_idx),
            )
            return tree, trace

        def _select(
            self, callbacks: MCTSCallbacks[StateT]
        ) -> Tuple[jnp.ndarray, int, int, int, StateT]:
            # State for the loop: (path, node_idx, expand_action, leaf_state, done)
            # path is trickier because it's a variable length list. 
            # In static JAX, we usually use a fixed-size array with a counter or similar.
            # Given the tree depth is bounded by max_nodes (or practically much less),
            # we can use a fixed size array for path.
            
            # MAX_SEARCH_DEPTH defined at module level
            path_arr = jnp.full((MAX_SEARCH_DEPTH,), NO_PARENT, dtype=jnp.int32)
            path_arr = path_arr.at[0].set(ROOT_INDEX)
            path_len = 1
            
            node_idx = ROOT_INDEX
            
            # Dummy initial values
            init_val = (
                path_arr, 
                path_len, 
                jnp.int32(node_idx), 
                jnp.int32(-1), # expand_action (using -1 as None)
                callbacks.decode(self.nodes[ROOT_INDEX].state.embedding), # state
                False # done
            )
            
            def cond_fun(val):
                _, _, _, _, _, done = val
                return jnp.logical_not(done)
                
            def body_fun(val):
                path_arr, path_len, node_idx, _, _, _ = val
                node = self.nodes[node_idx]
                state = callbacks.decode(node.state.embedding)
                
                # Condition 1: Node unvisited
                is_unvisited = (node.visit_count == 0)
                
                # Condition 2: Terminal state
                is_terminal = callbacks.is_terminal(state)
                
                # If either, we are done. expand_action remains None (-1).
                should_stop = jnp.logical_or(is_unvisited, is_terminal)
                
                # Else, try to select action
                invalid_mask = self._coerce_action_mask(
                    callbacks.invalid_actions(state),
                    context="callbacks.invalid_actions(state)",
                )
                unexplored = jnp.logical_and(invalid_mask == 0, node.children_idx == UNVISITED)
                has_unexplored = jnp.any(unexplored)
                
                # If has unexplored children, pick one and stop (expand_action = action)
                def _pick_unexplored(_):
                    action = jnp.argmax(unexplored.astype(jnp.int32)).astype(jnp.int32)
                    return action, True, node_idx # done, stay at current node
                
                # Else, pick best child (UCB) and descend
                def _descend(_):
                    best_action = self._pick_best_child(node, invalid_mask)
                    child_idx = node.children_idx[best_action].astype(jnp.int32)
                    
                    # Edge case: if child is theoretically unvisited but logic brought us here?
                    # (Should be covered by has_unexplored check usually)
                    is_child_unvisited = (child_idx == UNVISITED)
                    
                    return jax.lax.cond(
                        is_child_unvisited,
                        lambda: (best_action.astype(jnp.int32), True, node_idx), # Stop here, expand this action
                        lambda: (jnp.int32(-1), False, child_idx) # Continue descending
                    )

                next_action, next_done, next_node_idx = jax.lax.cond(
                    has_unexplored,
                    _pick_unexplored,
                    _descend,
                    operand=None
                )
                
                final_done = jnp.logical_or(should_stop, next_done)
                
                # Update path if we descended (next_node_idx != node_idx)
                # logic: if not done and not stop, we descended.
                descended = jnp.logical_and(jnp.logical_not(final_done), next_node_idx != node_idx)
                
                new_path_arr = jax.lax.select(
                    descended,
                    path_arr.at[path_len].set(next_node_idx),
                    path_arr
                )
                new_path_len = jax.lax.select(descended, path_len + 1, path_len)
                
                return (new_path_arr, new_path_len, next_node_idx, next_action, state, final_done)

            final_val = jax.lax.while_loop(cond_fun, body_fun, init_val)
            path_arr, path_len, leaf_idx, expand_action_idx, leaf_state, _ = final_val
            
            return path_arr, path_len, leaf_idx, expand_action_idx, leaf_state

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
            prior_logits = callbacks.policy(child_state)
            # Remove int() cast for JIT compatibility
            child_idx = self.next_free_idx 
            # We cannot check this assertion in JIT easily without host callback.
            # We can use checkify or just rely on max_nodes being sufficient.
            # if child_idx >= self._max_nodes:
            #    raise RuntimeError("Tree saturated; increase `max_nodes`.")

            embedding_state_cls = getattr(self._node_class, "_embedding_state_cls")
            child_node = self._node_class.default().replace(
                state=embedding_state_cls(embedding=child_embedding),
                children_prior_logits=prior_logits,
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
            
            # action_idx must be used as index. If tracer, at[] handles it.
            
            parent_node = parent_node.replace(
                children_idx=parent_node.children_idx.at[action_idx].set(child_idx),
                children_rewards=parent_node.children_rewards.at[action_idx].set(reward),
                children_discounts=parent_node.children_discounts.at[action_idx].set(discount),
                children_visits=parent_node.children_visits.at[action_idx].set(1),
                children_values=parent_node.children_values.at[action_idx].set(reward),
            )
            tree = tree._update_node(parent_idx, parent_node)
            return tree, child_idx, child_state

        def _backprop(self, path_arr: jnp.ndarray, path_len: int, leaf_value: float) -> "Tree":
            tree = self
            value = leaf_value
            
            should_flip = True 
            
            def body_fun(i, val):
                tree, curr_value = val
                
                idx = path_arr[i]
                is_active = (i < path_len) & (idx != NO_PARENT)
                
                # If inactive, curr_value does not matter but we propagate it.
                # Logic: update current node, then setup parent update.
                
                node = tree.nodes[idx]
                
                # Update node
                new_visit = node.visit_count + 1
                new_val = node.value + curr_value
                
                node = node.replace(
                    visit_count=jax.lax.select(is_active, new_visit, node.visit_count),
                    value=jax.lax.select(is_active, new_val, node.value),
                    raw_value=jax.lax.select(is_active, jnp.array(curr_value, dtype=jnp.float32), node.raw_value)
                )
                tree = tree._update_node(idx, node)

                # Update parent connection
                parent_idx = node.parent_idx
                has_parent_node = (parent_idx != NO_PARENT)
                
                should_update_parent = is_active & has_parent_node
                
                parent = tree.nodes[parent_idx]
                action = node.action_from_parent_idx
                
                # Parent sees value from its perspective (-curr_value)
                update_val_for_parent = -curr_value if should_flip else curr_value
                
                new_p_visits = parent.children_visits.at[action].add(1)
                new_p_values = parent.children_values.at[action].add(update_val_for_parent)
                
                parent = parent.replace(
                    children_visits=jax.lax.select(should_update_parent, new_p_visits, parent.children_visits),
                    children_values=jax.lax.select(should_update_parent, new_p_values, parent.children_values)
                )
                tree = tree._update_node(parent_idx, parent)
                
                next_value = -curr_value if should_flip else curr_value
                return tree, next_value

            # We iterate backwards from path_len-1 to 0.
            # scan over max depth, compute i = path_len - 1 - k
            
            MAX_DEPTH = path_arr.shape[0]
            
            def scan_wrapper(carry, k):
                tree, val = carry
                i = path_len - 1 - k
                is_active = (i >= 0)
                
                # Run body
                next_tree, next_val = body_fun(i, (tree, val))
                
                # Use tree_map for selective update of the Tree structure
                # This avoids potential issues with jax.lax.select on custom Pytree classes
                final_tree = jax.tree_util.tree_map(
                    lambda x, y: jax.lax.select(is_active, x, y),
                    next_tree,
                    tree
                )
                final_val = jax.lax.select(is_active, next_val, val)
                
                return (final_tree, final_val), None

            (final_tree, _), _ = jax.lax.scan(
                scan_wrapper, 
                (tree, value), 
                jnp.arange(MAX_DEPTH)
            )
            
            return final_tree

        def _pick_best_child(self, node: Node, invalid_mask: jnp.ndarray) -> int:
            # Standard PUCT constants (could be parameterized if needed)
            pb_c_init = 1.25
            pb_c_base = 19652.0

            visits = node.children_visits.astype(jnp.float32)
            total_visits = jnp.maximum(node.visit_count.astype(jnp.float32), 1.0)
            
            # Prior probabilities from policy
            prior_logits = node.children_prior_logits
            prior_probs = jax.nn.softmax(prior_logits)

            # Value estimates
            values = node.children_values
            # If visits > 0, Q = values / visits. Else Q = 0 (or potentially parent value?)
            # Using 0.0 for unvisited nodes is common if rewards are normalized or centered.
            q_values = jnp.where(visits > 0, values / visits, 0.0)

            # PUCT formula
            pb_c = pb_c_init + jnp.log((total_visits + pb_c_base + 1.0) / pb_c_base)
            u_score = pb_c * prior_probs * (jnp.sqrt(total_visits) / (visits + 1.0))
            
            ucb = q_values + u_score

            # Mask invalid actions
            valid_mask = jnp.where(invalid_mask == 0, 1.0, 0.0)
            ucb = jnp.where(valid_mask > 0, ucb, -jnp.inf)
            
            # Return JAX array (int32), not Python int
            return jnp.argmax(ucb).astype(jnp.int32)

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
