import abc
import math
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import jax.numpy as jnp

@dataclass(frozen=True)
class ActionSpaceMetadata:
    """Lightweight description of an action space for diagnostics."""

    kind: Literal["discrete", "continuous"]
    size: int
    shape: tuple[int, ...]
    nvec: tuple[int, ...] | None = None
    low: jnp.ndarray | None = None
    high: jnp.ndarray | None = None
    sampling_number: int | None = None


class ActionSpace(abc.ABC):
    """Action space definition for the MCTS."""

    @abc.abstractmethod
    def get_size(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def metadata(self) -> ActionSpaceMetadata:
        """Returns a descriptive summary of the action space."""
        raise NotImplementedError


class DiscreteActionSpace(ActionSpace):
    """Discrete action space definition for the MCTS."""

    def __init__(self, action_sizes: int | Sequence[int]):
        if isinstance(action_sizes, Iterable) and not isinstance(action_sizes, (str, bytes)):
            nvec = tuple(int(n) for n in action_sizes)
        else:
            nvec = (int(action_sizes),)

        if not nvec:
            raise ValueError("DiscreteActionSpace requires at least one dimension.")
        if any(n <= 0 for n in nvec):
            raise ValueError("All discrete action sizes must be positive.")

        self._nvec: tuple[int, ...] = nvec
        self._ndim = len(nvec)
        self._size = math.prod(nvec)
        self._shape: tuple[int, ...] = (self._nvec[0],) if self._ndim == 1 else self._nvec

    @property
    def nvec(self) -> tuple[int, ...]:
        return self._nvec

    def get_size(self) -> int:
        return self._size

    def get_shape(self) -> tuple[int, ...]:
        return self._shape

    def metadata(self) -> ActionSpaceMetadata:
        return ActionSpaceMetadata(
            kind="discrete",
            size=self._size,
            shape=self._shape,
            nvec=self._nvec,
        )


class ContinuousActionSpace(ActionSpace):
    """Continuous action space definition for the MCTS."""

    def __init__(
        self,
        low: float | Sequence[float],
        high: float | Sequence[float],
        shape: tuple[int, ...],
        sampling_number: int = 8,
        dtype: jnp.dtype = jnp.float32,
    ):
        if not shape:
            raise ValueError("ContinuousActionSpace requires a non-empty shape.")
        if any(dim <= 0 for dim in shape):
            raise ValueError("All continuous action dimensions must be positive.")
        if sampling_number <= 0:
            raise ValueError("`sampling_number` must be positive.")

        self._shape = tuple(int(dim) for dim in shape)
        self._dtype = dtype
        self._sampling_number = sampling_number

        low_arr = jnp.asarray(low, dtype=self._dtype)
        high_arr = jnp.asarray(high, dtype=self._dtype)

        self._low = jnp.broadcast_to(low_arr, self._shape)
        self._high = jnp.broadcast_to(high_arr, self._shape)

        if jnp.any(self._high <= self._low):
            raise ValueError("All elements of `high` must be greater than `low`.")

    @property
    def low(self) -> jnp.ndarray:
        return self._low

    @property
    def high(self) -> jnp.ndarray:
        return self._high

    def get_size(self) -> int:
        return self._sampling_number

    def get_shape(self) -> tuple[int, ...]:
        return self._shape

    def get_sampling_number(self) -> int:
        return self._sampling_number

    def metadata(self) -> ActionSpaceMetadata:
        return ActionSpaceMetadata(
            kind="continuous",
            size=self._sampling_number,
            shape=self._shape,
            low=self._low,
            high=self._high,
            sampling_number=self._sampling_number,
        )