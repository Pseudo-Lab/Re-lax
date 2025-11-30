import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Type, Any
from xtructure import Xtructurable, FieldDescriptor, xtructure_dataclass

from ..model.efficientzero import EfficientZero
from ..mcts.action_space import ActionSpace
from .base import SimulationModel

def logits_to_scalar(logits: jnp.ndarray, support_size: int) -> float:
    """Convert categorical logits to a scalar value using expected value over support."""
    probs = jax.nn.softmax(logits)
    # Assume support is centered around 0: [-(N-1)/2, ..., (N-1)/2]
    min_val = -(support_size - 1) / 2
    support = jnp.arange(support_size, dtype=jnp.float32) + min_val
    return jnp.sum(probs * support)

class ModelEnv(SimulationModel[jnp.ndarray]):
    """
    A simulation environment that uses a learned EfficientZero model for dynamics.
    The state of this environment is the hidden representation (jnp.ndarray).
    """
    
    def __init__(
        self, 
        model: EfficientZero, 
        action_space: ActionSpace, 
        initial_observation: jnp.ndarray,
        model_state: eqx.nn.State = None
    ):
        self.model = model
        self._action_space = action_space
        self.initial_obs = initial_observation
        
        # Use provided model_state or create a new one
        if model_state is None:
            self.model_state = eqx.nn.State(model)
        else:
            self.model_state = model_state

        # Infer hidden shape from initial inference
        # We run this to get the shape and also to get the initial hidden state
        # We assume model_state is frozen (inference mode) for MCTS expansion
        hidden, _, _, _, _ = self.model.initial_inference(
            self.initial_obs, self.model_state
        )
        self._root_hidden = hidden
        self._hidden_shape = hidden.shape
        
        # Create embedding class dynamically based on shape
        @xtructure_dataclass
        class Embedding(Xtructurable):
            embedding: FieldDescriptor[jnp.float32, self._hidden_shape]
            
        self._embedding_cls = Embedding

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def embedding_state_cls(self) -> Type[Xtructurable]:
        return self._embedding_cls

    def initial_state(self) -> jnp.ndarray:
        # Returns the root hidden state
        return self._root_hidden

    def encode(self, state: jnp.ndarray) -> jnp.ndarray:
        # State is already the hidden embedding
        return state

    def decode(self, embedding: jnp.ndarray) -> jnp.ndarray:
        # Embedding is the state
        return embedding

    def invalid_actions(self, state: jnp.ndarray) -> jnp.ndarray:
        # Learned model doesn't track invalid actions internally in the hidden state usually.
        # We return all valid (zeros) or we'd need an external mask provider.
        # For now, assume all actions are valid.
        return jnp.zeros(self._action_space.get_shape(), dtype=jnp.uint8)

    def apply_action(self, state: jnp.ndarray, action: int) -> jnp.ndarray:
        # Dynamics: hidden + action -> next_hidden
        # We treat action as scalar or index.
        # The model expects action as jnp.ndarray possibly.
        # EfficientZero.recurrent_inference expects action: jnp.ndarray
        
        # Ensure action is array
        action_arr = jnp.array([action], dtype=jnp.float32) 
        # Note: Check how dynamics uses action. In efficientzero.py it makes a plane.
        
        next_hidden, _, _, _, _ = self.model.recurrent_inference(
            state, action_arr, self.model_state
        )
        return next_hidden

    def is_terminal(self, state: jnp.ndarray) -> bool:
        # Learned model doesn't explicitly predict terminal usually in EZ.
        # It predicts value/reward.
        # We rely on the search depth or value to guide.
        # Or we could threshold reward? 
        # Standard MuZero: doesn't stop at terminal in dynamics, just keeps predicting.
        return False

    def transition_reward(self, parent_state: jnp.ndarray, child_state: jnp.ndarray, action: int) -> float:
        # We need to recover the reward from the transition.
        # Since apply_action only returns state, we strictly should re-run or cache it.
        # Ideally apply_action would return (state, reward, ...), but SimulationModel splits them.
        # This implies we might re-compute dynamics.
        # Optimization: parent_state -> (next_state, reward)
        
        action_arr = jnp.array([action], dtype=jnp.float32)
        _, _, reward_logits, _, _ = self.model.recurrent_inference(
            parent_state, action_arr, self.model_state
        )
        
        num_buckets = reward_logits.shape[-1]
        return logits_to_scalar(reward_logits, num_buckets)

    def policy(self, state: jnp.ndarray) -> jnp.ndarray:
        policy_logits, _ = self.model.prediction(state)
        return policy_logits

    def value(self, state: jnp.ndarray) -> float:
        _, value_logits = self.model.prediction(state)
        num_buckets = value_logits.shape[-1]
        return logits_to_scalar(value_logits, num_buckets)

