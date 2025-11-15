from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

import jax.numpy as jnp
from xtructure import Xtructurable

from mcts.action_space import ActionSpace
from mcts.tree import MCTSCallbacks

StateT = TypeVar("StateT")


class SimulationModel(ABC, Generic[StateT]):
    """Abstract base class for MuZero-style simulations or real environments."""

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        """Return the discrete/continuous action space definition."""

    @property
    @abstractmethod
    def embedding_state_cls(self) -> Type[Xtructurable]:
        """Return an Xtructure class describing the latent state representation."""

    @abstractmethod
    def initial_state(self) -> StateT:
        """Return the canonical initial environment/world-model state."""

    @abstractmethod
    def encode(self, state: StateT) -> jnp.ndarray:
        """Encode a domain state into the embedding consumed by the tree."""

    @abstractmethod
    def decode(self, embedding: jnp.ndarray) -> StateT:
        """Decode an embedding produced by the model back into a domain state."""

    @abstractmethod
    def invalid_actions(self, state: StateT) -> jnp.ndarray:
        """Return a mask (1 == invalid) over actions at the supplied state."""

    @abstractmethod
    def apply_action(self, state: StateT, action: int) -> StateT:
        """Apply ``action`` to ``state`` and return the successor state."""

    @abstractmethod
    def is_terminal(self, state: StateT) -> bool:
        """Return True when the supplied state is terminal."""

    @abstractmethod
    def transition_reward(self, parent_state: StateT, child_state: StateT, action: int) -> float:
        """Return the environment reward for the transition parent->child via action."""

    @abstractmethod
    def value(self, state: StateT) -> float:
        """Heuristic/value estimate for leaf evaluation."""

    def make_callbacks(self) -> MCTSCallbacks[StateT]:
        """Produce an :class:`MCTSCallbacks` bundle compatible with the tree."""

        return MCTSCallbacks(
            encode=self.encode,
            decode=self.decode,
            invalid_actions=self.invalid_actions,
            apply_action=self.apply_action,
            is_terminal=self.is_terminal,
            transition_reward=self.transition_reward,
            value=self.value,
        )

