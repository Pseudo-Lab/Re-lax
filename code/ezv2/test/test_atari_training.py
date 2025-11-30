import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import gymnasium as gym
from typing import List, Tuple
import numpy as np
from collections import deque
import random

import sys
import os

# Add project root to path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import (
    EfficientZero, 
    ImageRepresentation, 
    ImageDynamics, 
    DiscretePrediction
)
from src.mcts.action_space import DiscreteActionSpace

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TRAIN_STEPS = 100
BUFFER_SIZE = 100000
# Reduced dimensions for faster testing
HIDDEN_DIM = 32
PROJECTION_DIM = 64
NUM_BLOCKS = 2
NUM_VALUE_BUCKETS = 601 # Standard EZ
NUM_REWARD_BUCKETS = 601 

def make_model(key: jax.Array, action_space_size: int) -> EfficientZero:
    # MinAtar Breakout is usually 4x10x10 (C, H, W)
    # Or standard Atari is much larger. We will resize or use MinAtar for speed.
    # Let's assume standard Gym Atari wrapper output (C, 84, 84) or similar.
    
    rep = ImageRepresentation(
        in_channels=4, # Frame stack 4
        num_blocks=NUM_BLOCKS,
        hidden_channels=HIDDEN_DIM,
        key=key
    )
    
    dyn = ImageDynamics(
        hidden_channels=HIDDEN_DIM,
        action_encoding_dim=1, # Scalar plane for action
        num_blocks=NUM_BLOCKS,
        num_reward_buckets=NUM_REWARD_BUCKETS,
        key=key
    )
    
    pred = DiscretePrediction(
        hidden_dim=HIDDEN_DIM,
        action_space_size=action_space_size,
        num_value_buckets=NUM_VALUE_BUCKETS,
        is_spatial=True,
        key=key
    )
    
    return EfficientZero(
        representation=rep,
        dynamics=dyn,
        prediction=pred,
        hidden_dim=HIDDEN_DIM,
        projection_dim=PROJECTION_DIM,
        is_spatial=True,
        key=key
    )

# Simple Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = map(np.stack, zip(*batch))
        return obs, action, reward, next_obs, done
    
    def __len__(self):
        return len(self.buffer)

@eqx.filter_jit
def train_step(
    model: EfficientZero, 
    opt_state: optax.OptState, 
    optimizer: optax.GradientTransformation,
    obs: jnp.ndarray, 
    action: jnp.ndarray, 
    target_reward: jnp.ndarray,
    target_value: jnp.ndarray,
    target_policy: jnp.ndarray
) -> Tuple[EfficientZero, optax.OptState, float]:
    
    # Simplified loss function for demonstration:
    # 1. Initial inference
    # 2. Recurrent inference (1 step)
    # 3. Policy & Value & Reward loss
    
    # Note: Real EZ loss is much more complex (unroll steps, consistency loss, etc.)
    # This is a sanity check "can it fit 1-step dynamics" test.
    
    def loss_fn(m: EfficientZero):
        state = eqx.nn.State(m)
        
        # 1. Initial Step
        hidden, value_logits, _, policy_logits, state = m.initial_inference(obs, state)
        
        # 2. Dynamics Step (Recurrent)
        # Action needs to be prepared for dynamics. 
        # ImageDynamics expects (1,) plane if dim=1, or (A,) if one-hot.
        # We configured action_encoding_dim=1, so we pass scalar action.
        # action shape in batch: (B,) -> (B, 1) needed? 
        # vmap handles batch dim. So we just pass action.
        
        # We need to vmap the model call over batch
        # But filter_jit handles the function level. 
        # We need to use vmap inside or assume input has batch dim and model handles it?
        # Equinox models usually handle single sample. We need vmap.
        
        return 0.0 # Placeholder
        
    # We need to structure this properly with vmap.
    # Let's redefine loss_fn outside.
    pass

# --- Redefining logic with vmap ---

def single_sample_loss(
    model: EfficientZero, 
    obs: jnp.ndarray, 
    action: jnp.ndarray, 
    next_obs: jnp.ndarray, # For consistency check (optional)
    reward: float,
    done: float
):
    # We'll just check if we can predict reward and next value from current state + action
    # This is a "1-step lookahead" training test.
    
    state = eqx.nn.State(model)
    
    # A. Initial Inference
    hidden, val_logits, _, pol_logits, state = model.initial_inference(obs, state)
    
    # B. Dynamics Inference
    # Prepare action: scalar -> (1,)
    action_arr = jnp.array([action], dtype=jnp.float32) 
    
    next_hidden, pred_next_val_logits, pred_reward_logits, pred_next_pol_logits, state = \
        model.recurrent_inference(hidden, action_arr, state)
        
    # Loss components (Simplified):
    # 1. Policy Loss (vs random target or just checks it runs) - skipping for dummy test
    # 2. Value Loss (vs bootstrapped or dummy target)
    # 3. Reward Loss (vs actual reward)
    
    # Convert scalar reward to bucket target (simplified: just MSE on scalar for test?)
    # EfficientZero uses categorical buckets.
    # For this test, let's just try to regress the scalar reward to prove dynamics works.
    # We need to decode logits to scalar or encode scalar to logits.
    # Let's assume we implemented `scalar_to_support` helper, but for now:
    # We'll just do a soft cross-entropy against a one-hot target for the nearest bucket.
    
    # Reward Loss
    reward_idx = jnp.clip(jnp.round(reward) + NUM_REWARD_BUCKETS // 2, 0, NUM_REWARD_BUCKETS - 1).astype(jnp.int32)
    reward_target = jax.nn.one_hot(reward_idx, NUM_REWARD_BUCKETS)
    reward_loss = optax.softmax_cross_entropy(pred_reward_logits, reward_target)
    
    return reward_loss

@eqx.filter_jit
def train_step_vmapped(
    model: EfficientZero, 
    opt_state: optax.OptState, 
    optimizer: optax.GradientTransformation,
    obs_batch: jnp.ndarray, 
    action_batch: jnp.ndarray, 
    reward_batch: jnp.ndarray,
    next_obs_batch: jnp.ndarray,
    done_batch: jnp.ndarray
):
    def loss_fn(m):
        losses = jax.vmap(
            single_sample_loss, 
            in_axes=(None, 0, 0, 0, 0, 0),
            axis_name="batch"
        )(
            m, obs_batch, action_batch, next_obs_batch, reward_batch, done_batch
        )
        return jnp.mean(losses)
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def main():
    print("Initializing environment (BreakoutNoFrameskip-v4 via gym)...")
    # Requires 'pip install gymnasium[atari] autorom[accept-rom-license]'
    try:
        # Try new ALE/ Gymnasium style first, fallback to old if needed
        # Explicitly import ale_py to register environments if needed
        import ale_py
        gym.register_envs(ale_py)
        
        env_id = "ALE/Breakout-v5"
        print(f"Attempting to load {env_id}...")
        # Disable internal frameskip to allow wrapper to handle it
        env = gym.make(env_id, frameskip=1)
        env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)
        env = gym.wrappers.FrameStackObservation(env, 4) # Renamed in recent Gym versions
    except Exception as e1:
        print(f"Failed {env_id}: {e1}")
        try:
            # Fallback to legacy ID
            env_id = "BreakoutNoFrameskip-v4"
            print(f"Attempting to load {env_id}...")
            env = gym.make(env_id)
            env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)
            env = gym.wrappers.FrameStackObservation(env, 4)
        except Exception as e2:
            print(f"Failed {env_id}: {e2}")
            print("Using dummy environment loop for verification.")
            return

    action_dim = env.action_space.n
    print(f"Action Space: {action_dim}, Observation Shape: {env.observation_space.shape}")
    
    # Initialize Model
    key = jax.random.PRNGKey(42)
    model = make_model(key, action_dim)
    
    # Optimizer
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Replay Buffer
    rb = ReplayBuffer(BUFFER_SIZE)
    
    # Collect Data
    print("Collecting random data...")
    obs, _ = env.reset()
    for _ in range(BUFFER_SIZE):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Obs is LazyFrames, convert to array
        # Shape comes as (4, 84, 84) from wrapper? 
        # Gym FrameStack usually gives (4, 84, 84) if using grayscale.
        
        rb.push(np.array(obs), action, reward, np.array(next_obs), done)
        
        obs = next_obs
        if done:
            obs, _ = env.reset()
            
    print(f"Buffer size: {len(rb)}")
    
    # Train Loop
    print("Starting training loop (1-step dynamics check)...")
    obs_b, act_b, rew_b, next_obs_b, done_b = rb.sample(BATCH_SIZE)
    
    # Convert to JAX arrays
    # Ensure channel first/last logic matches model
    # Model expects (C, H, W). Gym FrameStack is (C, H, W) usually with AtariPreprocessing?
    # Actually Gym AtariPreprocessing output is (84, 84, 1) usually?
    # FrameStack adds dimension: (Stack, 84, 84) if grayscale=True? 
    # Let's check shape.
    print(f"Sampled obs shape: {obs_b.shape}") 
    
    # If shape is (B, 4, 84, 84), it matches our model Conv2d (C, H, W).
    
    obs_b = jnp.array(obs_b, dtype=jnp.float32)
    act_b = jnp.array(act_b, dtype=jnp.float32)
    rew_b = jnp.array(rew_b, dtype=jnp.float32)
    next_obs_b = jnp.array(next_obs_b, dtype=jnp.float32)
    done_b = jnp.array(done_b, dtype=jnp.float32)
    
    for i in range(TRAIN_STEPS):
        model, opt_state, loss = train_step_vmapped(
            model, opt_state, optimizer, obs_b, act_b, rew_b, next_obs_b, done_b
        )
        obs_b, act_b, rew_b, next_obs_b, done_b = rb.sample(BATCH_SIZE)
        obs_b = jnp.array(obs_b, dtype=jnp.float32)
        act_b = jnp.array(act_b, dtype=jnp.float32)
        rew_b = jnp.array(rew_b, dtype=jnp.float32)
        next_obs_b = jnp.array(next_obs_b, dtype=jnp.float32)
        done_b = jnp.array(done_b, dtype=jnp.float32)
        if i % 10 == 0:
            print(f"Step {i}: Loss = {loss:.4f}")
            
    print("Training finished successfully.")

if __name__ == "__main__":
    main()

