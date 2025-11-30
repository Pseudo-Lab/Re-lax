import jax.numpy as jnp
import flashbax
from typing import NamedTuple, Generic, TypeVar

StateT = TypeVar("StateT")

class Trajectory(NamedTuple):
    """A single step in a trajectory."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    policy_logits: jnp.ndarray
    value: jnp.ndarray

class ReplayBuffer:
    """A wrapper around flashbax.trajectory_buffer."""
    
    def __init__(self, 
                 max_length: int, 
                 min_length: int, 
                 sample_batch_size: int,
                 add_batch_size: int = 1):
        self.max_length = max_length
        self.min_length = min_length
        self.sample_batch_size = sample_batch_size
        self.add_batch_size = add_batch_size
        
        # We defer initialization until the first 'init' call where we see the structure
        self._buffer_state = None
        self._buffer_fn = None

    def init(self, example_item: Trajectory):
        """Initialize the buffer state with an example item."""
        self._buffer_fn = flashbax.make_item_buffer(
            max_length=self.max_length,
            min_length=self.min_length,
            sample_batch_size=self.sample_batch_size,
            add_batches=self.add_batch_size > 1,
        )
        self._buffer_state = self._buffer_fn.init(example_item)
        return self._buffer_state

    def add(self, item: Trajectory):
        """Add an item to the buffer."""
        if self._buffer_state is None:
            raise RuntimeError("Buffer not initialized. Call init() first.")
            
        # flashbax expects batched input if add_batches=True (default in make_trajectory_buffer?)
        # Here we used make_item_buffer. 
        # If add_batch_size > 1, we assume item is batched.
        self._buffer_state = self._buffer_fn.add(self._buffer_state, item)
        return self._buffer_state

    def sample(self, rng_key):
        """Sample a batch from the buffer."""
        if self._buffer_state is None:
            raise RuntimeError("Buffer not initialized. Call init() first.")
            
        return self._buffer_fn.sample(self._buffer_state, rng_key)

    def size(self):
        """Return current size."""
        if self._buffer_state is None:
            return 0
        # Assuming Flashbax ItemBufferState structure
        return jnp.where(self._buffer_state.is_full, self.max_length, self._buffer_state.current_index)
