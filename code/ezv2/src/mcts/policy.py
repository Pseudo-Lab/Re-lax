"""Search policies built on top of the local MCTS tree implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Type, TypeVar

import jax
import jax.numpy as jnp
from xtructure import Xtructurable

from .annotate import ROOT_INDEX
from .tree import MCTSCallbacks, SearchTrace

TreeType = TypeVar("TreeType", bound=Xtructurable)


@dataclass
class PolicyOutput:
    """Container returned by policy helpers."""

    action: int
    action_weights: jnp.ndarray
    tree: Xtructurable
    traces: Sequence[SearchTrace]


def muzero_policy(
    tree_cls: Type[TreeType],
    callbacks: MCTSCallbacks,
    root_state,
    *,
    rng_key: jax.Array,
    num_simulations: int,
    discount: float = 1.0,
    invalid_actions: Optional[jnp.ndarray] = None,
    temperature: float = 1.0,
    dirichlet_fraction: float = 0.0,
    dirichlet_alpha: float = 0.3,
) -> PolicyOutput:
    """Runs a lightweight MuZero-style search on the local tree abstraction.

    Args:
        tree_cls: Tree class produced by :func:`make_tree_class`.
        callbacks: Environment/model-specific callbacks required by the tree.
        root_state: Domain state supplied to `tree_cls.from_root`.
        rng_key: JAX PRNG key used for noise injection and sampling.
        num_simulations: Number of tree expansions to perform.
        discount: Scalar discount propagated during backpropagation.
        invalid_actions: Optional mask (1 for invalid actions) applied at root.
        temperature: Temperature for sampling from visit-count-derived logits.
        dirichlet_fraction: Amount of Dirichlet noise to mix into root policy.
        dirichlet_alpha: Concentration parameter for the Dirichlet noise.

    Returns:
        PolicyOutput containing the selected action, action weights, and tree.
    """

    tree = tree_cls.from_root(callbacks, root_state)
    if invalid_actions is not None:
        tree = tree.replace(root_invalid_actions=_coerce_invalid_mask(tree, invalid_actions))

    tree, traces = tree.run_search(callbacks, num_iterations=num_simulations, discount=discount)
    root = tree.nodes[ROOT_INDEX]

    mask = _resolve_invalid_mask(tree, invalid_actions)
    visit_counts = jnp.asarray(root.children_visits, dtype=jnp.float32)
    action_weights = _normalize_visit_counts(visit_counts, mask)

    if dirichlet_fraction > 0.0:
        rng_key, noise_key = jax.random.split(rng_key)
        action_weights = _add_dirichlet_noise(
            noise_key,
            action_weights,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_fraction=dirichlet_fraction,
        )

    logits = _get_logits_from_probs(action_weights)
    logits = _apply_temperature(logits, temperature)

    if temperature <= 1e-6:
        action = int(jnp.argmax(logits))
    else:
        rng_key, sample_key = jax.random.split(rng_key)
        action = int(jax.random.categorical(sample_key, logits))

    action_weights = jax.nn.softmax(logits)
    return PolicyOutput(action=action, action_weights=action_weights, tree=tree, traces=traces)


def _normalize_visit_counts(counts: jnp.ndarray, invalid_mask: jnp.ndarray) -> jnp.ndarray:
    valid_mask = jnp.logical_not(invalid_mask.astype(bool))
    masked_counts = jnp.where(valid_mask, counts, 0.0)
    total = jnp.sum(masked_counts)
    num_valid = jnp.sum(valid_mask).astype(jnp.float32)
    num_actions = counts.shape[-1]

    def _nonzero_total():
        return masked_counts / total

    def _fallback():
        def _no_valid():
            return jnp.full_like(counts, 1.0 / float(num_actions))

        def _has_valid():
            return jnp.where(valid_mask, 1.0 / num_valid, 0.0)

        return jax.lax.cond(num_valid <= 0, _no_valid, _has_valid)

    return jax.lax.cond(total > 0.0, _nonzero_total, _fallback)


def _get_logits_from_probs(probs: jnp.ndarray) -> jnp.ndarray:
    tiny = jnp.finfo(probs.dtype).tiny
    return jnp.log(jnp.maximum(probs, tiny))


def _apply_temperature(logits: jnp.ndarray, temperature: float) -> jnp.ndarray:
    logits = logits - jnp.max(logits)
    tiny = jnp.finfo(logits.dtype).tiny
    denom = jnp.maximum(tiny, temperature)
    return logits / denom


def _add_dirichlet_noise(
    rng_key: jax.Array,
    probs: jnp.ndarray,
    *,
    dirichlet_alpha: float,
    dirichlet_fraction: float,
) -> jnp.ndarray:
    alpha = jnp.full(probs.shape[-1], dirichlet_alpha, dtype=probs.dtype)
    noise = jax.random.dirichlet(rng_key, alpha)
    return (1.0 - dirichlet_fraction) * probs + dirichlet_fraction * noise


def _coerce_invalid_mask(tree: Xtructurable, invalid_actions: jnp.ndarray) -> jnp.ndarray:
    mask = jnp.asarray(invalid_actions, dtype=jnp.uint8)
    mask = jnp.reshape(mask, tree.root_invalid_actions.shape)
    return mask


def _resolve_invalid_mask(tree: Xtructurable, override: Optional[jnp.ndarray]) -> jnp.ndarray:
    base_mask = tree.root_invalid_actions.astype(bool)
    if override is None:
        return base_mask
    override_mask = jnp.asarray(override, dtype=bool).reshape(base_mask.shape)
    return jnp.logical_or(base_mask, override_mask)
