import sys
import pathlib
import pytest
import jax
import jax.numpy as jnp

# Add src to path
TEST_PATH = pathlib.Path(__file__).resolve().parent
SRC_PATH = TEST_PATH.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.replaybuffer import ReplayBuffer, Trajectory

def test_replay_buffer():
    # Define shapes
    obs_shape = (4,)
    action_shape = ()
    
    dummy_traj = Trajectory(
        observation=jnp.zeros(obs_shape),
        action=jnp.zeros(action_shape, dtype=jnp.int32),
        reward=jnp.array(0.0),
        discount=jnp.array(1.0),
        policy_logits=jnp.zeros((10,)),
        value=jnp.array(0.0)
    )

    # Initialize buffer
    buffer = ReplayBuffer(max_length=100, min_length=2, sample_batch_size=2)
    state = buffer.init(dummy_traj)

    # Add data
    rng = jax.random.PRNGKey(0)
    
    # Add 10 items
    for i in range(10):
        item = Trajectory(
            observation=jnp.ones(obs_shape) * i,
            action=jnp.array(i, dtype=jnp.int32),
            reward=jnp.array(float(i)),
            discount=jnp.array(1.0),
            policy_logits=jnp.zeros((10,)),
            value=jnp.array(0.0)
        )
        state = buffer.add(item)
    
    # Check size
    assert int(buffer.size()) == 10

    # Sample
    batch = buffer.sample(rng)
    assert batch.experience.observation.shape == (2, 4)
    assert batch.experience.action.shape == (2,)
