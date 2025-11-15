"""Action-selection helpers compatible with the local MCTS tree implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol

import jax
import jax.numpy as jnp
from xtructure import Xtructurable

ActionSelectionStrategy = Literal["gumbel", "gumbel_muzero", "puct", "muzero"]


class QTransformFn(Protocol):
    def __call__(self, tree: Xtructurable, node_index: int) -> jnp.ndarray: ...


def select_action(
    *,
    tree: Xtructurable,
    node_index: int,
    depth: int,
    rng_key: jax.Array,
    strategy: ActionSelectionStrategy = "gumbel",
    qtransform: Optional[QTransformFn] = None,
    **kwargs,
) -> int:
    """Selects an action for `node_index` with the requested strategy.

    Args:
        tree: Tree produced by :func:`make_tree_class`.
        node_index: Index into `tree.nodes`.
        depth: Current depth. Root nodes must pass ``0``.
        rng_key: PRNG key used for stochastic tie-breaking / noise.
        strategy: Either ``"gumbel"`` (default) or ``"puct"/"muzero"``.
        qtransform: Optional custom Q-transform. Defaults to `_default_qtransform`.
        **kwargs: Extra keyword arguments forwarded to the concrete selector.

    Returns:
        Integer action index.
    """

    q_fn = qtransform or _default_qtransform
    if strategy in ("gumbel", "gumbel_muzero"):
        return int(
            gumbel_muzero_action_selection(
                tree=tree,
                node_index=node_index,
                depth=depth,
                rng_key=rng_key,
                qtransform=q_fn,
                **kwargs,
            )
        )
    if strategy in ("puct", "muzero"):
        return int(
            muzero_action_selection(
                tree=tree,
                node_index=node_index,
                depth=depth,
                rng_key=rng_key,
                qtransform=q_fn,
                **kwargs,
            )
        )
    raise ValueError(f"Unknown action selection strategy: {strategy}")


def muzero_action_selection(
    *,
    tree: Xtructurable,
    node_index: int,
    depth: int,
    rng_key: jax.Array,
    qtransform: QTransformFn,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
) -> int:
    """Standard MuZero PUCT selection."""

    node = tree.nodes[node_index]
    visit_counts = jnp.asarray(node.children_visits, dtype=jnp.float32)
    node_visit = jnp.maximum(jnp.asarray(node.visit_count, dtype=jnp.float32), 1.0)

    pb_c = pb_c_init + jnp.log((node_visit + pb_c_base + 1.0) / pb_c_base)
    prior_logits = node.children_prior_logits
    prior_probs = jax.nn.softmax(prior_logits)

    policy_score = jnp.sqrt(node_visit) * pb_c * prior_probs / (visit_counts + 1.0)
    value_score = qtransform(tree, node_index)
    noise = 1e-7 * jax.random.uniform(rng_key, prior_probs.shape, dtype=prior_probs.dtype)
    scores = value_score + policy_score + noise

    mask = tree.root_invalid_actions if depth == 0 else None
    return masked_argmax(scores, mask)


@dataclass(frozen=True)
class GumbelMuZeroExtraData:
    """Stores the gumbel noise used at the root."""

    root_gumbel: jnp.ndarray


def gumbel_muzero_action_selection(
    *,
    tree: Xtructurable,
    node_index: int,
    depth: int,
    rng_key: jax.Array,
    qtransform: QTransformFn,
    gumbel_scale: float = 1.0,
    max_num_considered_actions: Optional[int] = None,
    extra_data: Optional[GumbelMuZeroExtraData] = None,
) -> int:
    """Full Gumbel MuZero-style selection with optional candidate pruning."""

    node = tree.nodes[node_index]
    logits = node.children_prior_logits
    qvalues = qtransform(tree, node_index)

    if extra_data is not None and extra_data.root_gumbel.shape == logits.shape:
        gumbel = extra_data.root_gumbel
    else:
        rng_key, subkey = jax.random.split(rng_key)
        gumbel = gumbel_scale * jax.random.gumbel(subkey, logits.shape, dtype=logits.dtype)

    scores = logits + qvalues + gumbel
    mask = tree.root_invalid_actions if depth == 0 else None
    scores = _apply_mask(scores, mask)

    if max_num_considered_actions is not None and max_num_considered_actions > 0:
        k = int(min(max_num_considered_actions, scores.shape[-1]))
        topk_indices = jnp.argsort(scores)[-k:]
        candidate_mask = jnp.ones_like(scores, dtype=bool).at[topk_indices].set(False)
        scores = jnp.where(candidate_mask, -jnp.inf, scores)

    return int(jnp.argmax(scores))


def masked_argmax(scores: jnp.ndarray, invalid_actions: Optional[jnp.ndarray]) -> int:
    """Returns the argmax while respecting optional invalid-action masks."""

    masked = _apply_mask(scores, invalid_actions)
    return int(jnp.argmax(masked))


def _apply_mask(scores: jnp.ndarray, invalid_actions: Optional[jnp.ndarray]) -> jnp.ndarray:
    if invalid_actions is None:
        return scores
    mask = jnp.asarray(invalid_actions, dtype=bool)
    return jnp.where(mask, -jnp.inf, scores)


def _default_qtransform(tree: Xtructurable, node_index: int) -> jnp.ndarray:
    node = tree.nodes[node_index]
    visits = jnp.maximum(jnp.asarray(node.children_visits, dtype=jnp.float32), 1.0)
    values = jnp.asarray(node.children_values, dtype=jnp.float32)
    return values / visits
