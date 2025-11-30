"""Search policies built on top of the local MCTS tree implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Type, TypeVar

import jax
import jax.numpy as jnp
from xtructure import Xtructurable

from .action_selection import GumbelMuZeroExtraData
from .annotate import ROOT_INDEX
from .tree import MCTSCallbacks, SearchTrace

TreeType = TypeVar("TreeType", bound=Xtructurable)


@dataclass
class PolicyOutput:
    """Container returned by policy helpers."""

    action: int
    action_weights: jnp.ndarray
    tree: Xtructurable
    traces: SearchTrace
    gumbel_extra: Optional[GumbelMuZeroExtraData] = None

# Register PolicyOutput as a Pytree
jax.tree_util.register_pytree_node(
    PolicyOutput,
    lambda s: ((s.action, s.action_weights, s.tree, s.traces, s.gumbel_extra), None),
    lambda _, children: PolicyOutput(*children)
)



def search_visit_policy(
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
    return_gumbel_noise: bool = False,
) -> PolicyOutput:
    """Run local tree search and sample an action from visit-count weights.

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

    tree = _initialize_tree(tree_cls, callbacks, root_state, invalid_actions)
    tree, traces = tree.run_search(callbacks, num_iterations=num_simulations, discount=discount)
    root = tree.nodes[ROOT_INDEX]

    mask = _resolve_invalid_mask(tree, invalid_actions)
    visit_counts = jnp.asarray(root.children_visits, dtype=jnp.float32)
    action_weights = _normalize_visit_counts(visit_counts, mask)

    rng_key, action_weights = _maybe_apply_dirichlet_noise(
        rng_key,
        action_weights,
        dirichlet_fraction=dirichlet_fraction,
        dirichlet_alpha=dirichlet_alpha,
    )

    logits = _get_logits_from_probs(action_weights)
    logits = _apply_temperature(logits, temperature)

    action, rng_key, gumbel_extra = _sample_action_from_logits(
        logits,
        rng_key=rng_key,
        temperature=temperature,
        capture_gumbel_noise=return_gumbel_noise,
    )

    action_weights = jax.nn.softmax(logits)
    return PolicyOutput(
        action=action,
        action_weights=action_weights,
        tree=tree,
        traces=traces,
        gumbel_extra=gumbel_extra,
    )


# Backwards compatibility alias
muzero_policy = search_visit_policy
def _initialize_tree(
    tree_cls: Type[TreeType],
    callbacks: MCTSCallbacks,
    root_state,
    invalid_actions: Optional[jnp.ndarray],
) -> TreeType:
    tree = tree_cls.from_root(callbacks, root_state)
    if invalid_actions is not None:
        mask = _coerce_invalid_mask(
            tree_cls, invalid_actions, context="search_visit_policy.invalid_actions_override"
        )
        tree = tree.replace(root_invalid_actions=mask)
    return tree


def _maybe_apply_dirichlet_noise(
    rng_key: jax.Array,
    action_weights: jnp.ndarray,
    *,
    dirichlet_fraction: float,
    dirichlet_alpha: float,
) -> Tuple[jax.Array, jnp.ndarray]:
    if dirichlet_fraction <= 0.0:
        return rng_key, action_weights
    rng_key, noise_key = jax.random.split(rng_key)
    noisy = _add_dirichlet_noise(
        noise_key,
        action_weights,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_fraction=dirichlet_fraction,
    )
    return rng_key, noisy


def _sample_action_from_logits(
    logits: jnp.ndarray,
    *,
    rng_key: jax.Array,
    temperature: float,
    capture_gumbel_noise: bool,
) -> Tuple[int, jax.Array, Optional[GumbelMuZeroExtraData]]:
    
    # We use lax.cond for branching, but we must ensure return types match.
    # extra is Optional[GumbelMuZeroExtraData]. In JIT, Optional is not really supported for varying branches.
    # Both branches must return same Pytree structure.
    # So if capture_gumbel_noise is True, greedy branch must also return a dummy GumbelMuZeroExtraData or None?
    # Actually capture_gumbel_noise is static (bool), so we can use python if for that part?
    # No, capture_gumbel_noise is passed as arg. If it's static in JIT, we are fine.
    # In play_tictactoe.py, return_gumbel_noise is static_argnames? Let's check.
    # Yes, 'return_gumbel_noise' is in static_argnames.
    # But wait, _sample_action_from_logits is called with capture_gumbel_noise=return_gumbel_noise.
    # So capture_gumbel_noise is static boolean.
    
    # However, temperature is NOT static in the call from search_visit_policy (it's passed as arg).
    # So temperature check needs lax.cond.
    
    # If capture_gumbel_noise is True, we need to return extra data in both branches.
    # If False, both return None.
    
    def _sample(key):
        # Gumbel noise
        new_key, gumbel_key = jax.random.split(key)
        gumbel = jax.random.gumbel(gumbel_key, logits.shape, dtype=logits.dtype)
        
        # If temp is tiny, we just do argmax (effectively gumbel with 0 scale or pure argmax)
        # But we need to implement the branching logic correctly.
        # Argmax(logits) is equivalent to Argmax(logits + 0*gumbel)
        
        # Let's just use the Gumbel-Max trick formula:
        # action = argmax(logits + gumbel)
        # But if temp is near 0, we want argmax(logits).
        
        # Correct logic:
        # If temp > 0: sampling ~ Softmax(logits/temp)
        # If temp == 0: argmax(logits)
        
        # For standard sampling we often use Gumbel-Max trick on logits/temp?
        # search_visit_policy calls _apply_temperature BEFORE calling this.
        # So logits here are already scaled (logits/temp).
        # So we just need argmax(logits + gumbel).
        
        # Wait, if temp -> 0, logits -> infinity. _apply_temperature handles this?
        # _apply_temperature divides by max(tiny, temperature).
        # If temp is small, logits are huge. Gumbel noise becomes negligible.
        # So argmax(logits + gumbel) converges to argmax(logits).
        
        # So we actually don't strictly NEED the `if temp <= 1e-6` branch for correctness,
        # BUT we might want it for determinism or avoiding huge numbers.
        # However, huge numbers are fine in float32 usually.
        
        # The issue in the traceback was `if temperature <= 1e-6`.
        # We can replace this with `jax.lax.cond`.
        
        pass

    # Refined implementation using lax.cond
    
    def _greedy(_):
        action = jnp.argmax(logits).astype(jnp.int32)
        # Return dummy extra if needed?
        # If capture_gumbel_noise is True, we need to return something matching the structure.
        # If we are in JIT, we can't return None in one branch and object in another unless they are compatible.
        # If capture_gumbel_noise is False, both return None. Fine.
        # If True, stochastic returns object. Greedy must return object too?
        # Usually we just return None for extra in greedy? 
        # But lax.cond requires same structure.
        
        # Let's check if we can assume capture_gumbel_noise is False for now (default).
        # In play_tictactoe, it defaults to False.
        
        extra = None
        if capture_gumbel_noise:
             # Create dummy zeros
             extra = GumbelMuZeroExtraData(root_gumbel=jnp.zeros_like(logits))
             
        return action, rng_key, extra

    def _stochastic(_):
        new_rng_key, gumbel_key = jax.random.split(rng_key)
        gumbel = jax.random.gumbel(gumbel_key, logits.shape, dtype=logits.dtype)
        # Logits are already temperature scaled.
        action = jnp.argmax(logits + gumbel).astype(jnp.int32)
        
        extra = None
        if capture_gumbel_noise:
            extra = GumbelMuZeroExtraData(root_gumbel=gumbel)
            
        return action, new_rng_key, extra

    # Since temperature is a Tracer, we use lax.cond
    return jax.lax.cond(
        temperature <= 1e-6,
        _greedy,
        _stochastic,
        operand=None
    )


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


def _coerce_invalid_mask(
    tree_cls: Type[TreeType], invalid_actions: jnp.ndarray, *, context: str
) -> jnp.ndarray:
    return tree_cls._coerce_action_mask(invalid_actions, context=context)


def _resolve_invalid_mask(tree: Xtructurable, override: Optional[jnp.ndarray]) -> jnp.ndarray:
    base_mask = tree.root_invalid_actions.astype(bool)
    if override is None:
        return base_mask
    tree_cls = type(tree)
    override_mask = _coerce_invalid_mask(
        tree_cls, override, context="search_visit_policy.invalid_actions_runtime"
    ).astype(bool)
    return jnp.logical_or(base_mask, override_mask)
