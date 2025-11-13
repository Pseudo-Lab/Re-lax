import jax
import jax.numpy as jnp
import equinox as eqx

class Model(eqx.Module):
    def __init__(self, embedding_shape: tuple[int, ...]):
        pass

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x