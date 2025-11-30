import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Protocol, Union, Optional, Sequence
from abc import abstractmethod

from .modules import ResidualBlock, MLP

# --- Interfaces ---

class RepresentationNetwork(eqx.Module):
    @abstractmethod
    def __call__(self, observation: jnp.ndarray, state: eqx.nn.State) -> tuple[jnp.ndarray, eqx.nn.State]:
        ...

class DynamicsNetwork(eqx.Module):
    @abstractmethod
    def __call__(self, hidden: jnp.ndarray, action: jnp.ndarray, state: eqx.nn.State) -> tuple[jnp.ndarray, jnp.ndarray, eqx.nn.State]:
        ...

class PredictionNetwork(eqx.Module):
    @abstractmethod
    def __call__(self, hidden: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        ...

# --- Implementations: Representation ---

class ImageRepresentation(RepresentationNetwork):
    conv: eqx.nn.Conv2d
    blocks: list[ResidualBlock]
    
    def __init__(
        self, 
        in_channels: int, 
        num_blocks: int, 
        hidden_channels: int, 
        key: jax.Array
    ):
        key, subkey = jax.random.split(key)
        self.conv = eqx.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1, use_bias=False, key=subkey)
        
        self.blocks = []
        keys = jax.random.split(key, num_blocks)
        for i in range(num_blocks):
            self.blocks.append(ResidualBlock(hidden_channels, key=keys[i]))

    def __call__(self, x: jnp.ndarray, state: eqx.nn.State) -> tuple[jnp.ndarray, eqx.nn.State]:
        # x: (C, H, W)
        out = jax.nn.relu(self.conv(x))
        for block in self.blocks:
            out, state = block(out, state)
        return out, state

class VectorRepresentation(RepresentationNetwork):
    encoder: MLP
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        key: jax.Array
    ):
        # Simple MLP encoder for vector inputs
        self.encoder = MLP(in_dim, hidden_dim, hidden_sizes=[hidden_dim] * (num_layers - 1), key=key)

    def __call__(self, x: jnp.ndarray, state: eqx.nn.State) -> tuple[jnp.ndarray, eqx.nn.State]:
        # x: (D,)
        return self.encoder(x), state

# --- Implementations: Dynamics ---

class ImageDynamics(DynamicsNetwork):
    conv: eqx.nn.Conv2d
    blocks: list[ResidualBlock]
    reward_head: MLP
    action_encoding_dim: int
    
    def __init__(
        self, 
        hidden_channels: int, 
        action_encoding_dim: int, 
        num_blocks: int, 
        num_reward_buckets: int,
        key: jax.Array
    ):
        self.action_encoding_dim = action_encoding_dim
        key1, key2, key3 = jax.random.split(key, 3)
        
        # We concatenate action planes to the hidden state.
        # action_encoding_dim determines how many planes we expect.
        # For discrete actions, this could be the number of actions (one-hot planes) or 1 (scalar plane).
        # For continuous, it's the action dimension.
        self.conv = eqx.nn.Conv2d(hidden_channels + action_encoding_dim, hidden_channels, kernel_size=3, stride=1, padding=1, use_bias=False, key=key1)
        
        self.blocks = []
        keys = jax.random.split(key2, num_blocks)
        for i in range(num_blocks):
            self.blocks.append(ResidualBlock(hidden_channels, key=keys[i]))
            
        self.reward_head = MLP(hidden_channels, num_reward_buckets, hidden_sizes=[64], key=key3)

    def __call__(self, hidden: jnp.ndarray, action: jnp.ndarray, state: eqx.nn.State) -> tuple[jnp.ndarray, jnp.ndarray, eqx.nn.State]:
        # hidden: (C, H, W)
        # action: (A,) or (1,) 
        
        B, H, W = hidden.shape
        # Broadcast action to (A, H, W)
        # Action must be pre-processed to match action_encoding_dim.
        # e.g. one-hot encoded if discrete, or raw values if continuous.
        
        action_planes = jnp.broadcast_to(action[:, None, None], (action.shape[0], H, W))
        
        x = jnp.concatenate([hidden, action_planes], axis=0)
        x = jax.nn.relu(self.conv(x))
        
        for block in self.blocks:
            x, state = block(x, state)
        
        next_hidden = x
        
        # Reward prediction (GAP)
        flat = jnp.mean(next_hidden, axis=(1, 2)) 
        reward_logits = self.reward_head(flat)
        
        return next_hidden, reward_logits, state

class VectorDynamics(DynamicsNetwork):
    mlp: MLP
    reward_head: MLP
    
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        num_layers: int,
        num_reward_buckets: int,
        key: jax.Array
    ):
        k1, k2 = jax.random.split(key)
        self.mlp = MLP(hidden_dim + action_dim, hidden_dim, hidden_sizes=[hidden_dim] * num_layers, key=k1)
        self.reward_head = MLP(hidden_dim, num_reward_buckets, hidden_sizes=[64], key=k2)
        
    def __call__(self, hidden: jnp.ndarray, action: jnp.ndarray, state: eqx.nn.State) -> tuple[jnp.ndarray, jnp.ndarray, eqx.nn.State]:
        # hidden: (H,)
        # action: (A,) - already encoded (one-hot or continuous)
        x = jnp.concatenate([hidden, action], axis=0)
        next_hidden = self.mlp(x)
        reward_logits = self.reward_head(next_hidden)
        return next_hidden, reward_logits, state

# --- Implementations: Prediction ---

class DiscretePrediction(PredictionNetwork):
    policy_head: MLP
    value_head: MLP
    conv_p: Optional[eqx.nn.Conv2d]
    conv_v: Optional[eqx.nn.Conv2d]
    is_spatial: bool
    
    def __init__(
        self, 
        hidden_dim: int, 
        action_space_size: int, 
        num_value_buckets: int,
        is_spatial: bool,
        key: jax.Array
    ):
        self.is_spatial = is_spatial
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        if is_spatial:
            self.conv_p = eqx.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, key=k1)
            self.conv_v = eqx.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, key=k2)
        else:
            self.conv_p = None
            self.conv_v = None
            
        self.policy_head = MLP(hidden_dim, action_space_size, hidden_sizes=[64], key=k3)
        self.value_head = MLP(hidden_dim, num_value_buckets, hidden_sizes=[64], key=k4)

    def __call__(self, hidden: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        p_in = hidden
        v_in = hidden
        
        if self.is_spatial:
            p_in = jax.nn.relu(self.conv_p(p_in))
            p_in = jnp.mean(p_in, axis=(1, 2))
            
            v_in = jax.nn.relu(self.conv_v(v_in))
            v_in = jnp.mean(v_in, axis=(1, 2))
            
        policy_logits = self.policy_head(p_in)
        value_logits = self.value_head(v_in)
        
        return policy_logits, value_logits

class ContinuousPrediction(PredictionNetwork):
    # Outputs mean and std (or log_std) for continuous actions
    # Assuming Diagonal Gaussian policy
    
    policy_head: MLP # Outputs 2 * action_dim (mean, log_std)
    value_head: MLP
    conv_p: Optional[eqx.nn.Conv2d]
    conv_v: Optional[eqx.nn.Conv2d]
    is_spatial: bool
    
    def __init__(
        self, 
        hidden_dim: int, 
        action_dim: int, 
        num_value_buckets: int,
        is_spatial: bool,
        key: jax.Array
    ):
        self.is_spatial = is_spatial
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        if is_spatial:
            self.conv_p = eqx.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, key=k1)
            self.conv_v = eqx.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, key=k2)
        else:
            self.conv_p = None
            self.conv_v = None
            
        # Output 2 * action_dim for mean and log_std
        self.policy_head = MLP(hidden_dim, 2 * action_dim, hidden_sizes=[64], key=k3)
        self.value_head = MLP(hidden_dim, num_value_buckets, hidden_sizes=[64], key=k4)

    def __call__(self, hidden: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        p_in = hidden
        v_in = hidden
        
        if self.is_spatial:
            p_in = jax.nn.relu(self.conv_p(p_in))
            p_in = jnp.mean(p_in, axis=(1, 2))
            
            v_in = jax.nn.relu(self.conv_v(v_in))
            v_in = jnp.mean(v_in, axis=(1, 2))
            
        policy_params = self.policy_head(p_in) 
        value_logits = self.value_head(v_in)
        
        # policy_params: [mu_1, ..., mu_N, log_std_1, ..., log_std_N]
        return policy_params, value_logits


# --- Main Model ---

class EfficientZero(eqx.Module):
    representation: RepresentationNetwork
    dynamics: DynamicsNetwork
    prediction: PredictionNetwork
    projector: MLP
    predictor: MLP
    is_spatial: bool
    
    def __init__(
        self,
        representation: RepresentationNetwork,
        dynamics: DynamicsNetwork,
        prediction: PredictionNetwork,
        hidden_dim: int,
        projection_dim: int,
        is_spatial: bool,
        key: jax.Array
    ):
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
        self.is_spatial = is_spatial
        
        k1, k2 = jax.random.split(key)
        self.projector = MLP(hidden_dim, projection_dim, hidden_sizes=[512, 512], key=k1)
        self.predictor = MLP(projection_dim, projection_dim, hidden_sizes=[512], key=k2)

    def initial_inference(self, observation: jnp.ndarray, state: eqx.nn.State) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, eqx.nn.State]:
        hidden, state = self.representation(observation, state)
        policy_output, value_logits = self.prediction(hidden)
        # Reward for initial state is usually 0 or not used, returning zeros
        reward_logits = jnp.zeros_like(value_logits) 
        return hidden, value_logits, reward_logits, policy_output, state

    def recurrent_inference(self, hidden: jnp.ndarray, action: jnp.ndarray, state: eqx.nn.State) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, eqx.nn.State]:
        next_hidden, reward_logits, state = self.dynamics(hidden, action, state)
        policy_output, value_logits = self.prediction(next_hidden)
        return next_hidden, value_logits, reward_logits, policy_output, state
    
    def project(self, hidden: jnp.ndarray) -> jnp.ndarray:
        if self.is_spatial:
            flat = jnp.mean(hidden, axis=(1, 2))
        else:
            flat = hidden
        return self.projector(flat)
        
    def predict_projection(self, projection: jnp.ndarray) -> jnp.ndarray:
        return self.predictor(projection)
