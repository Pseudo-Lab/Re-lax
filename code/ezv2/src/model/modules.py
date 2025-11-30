import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Sequence

class ResidualBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    bn2: eqx.nn.BatchNorm
    
    def __init__(
        self, 
        channels: int, 
        key: jax.Array,
        kernel_size: int = 3,
        stride: int = 1
    ):
        key1, key2 = jax.random.split(key)
        padding = (kernel_size - 1) // 2
        self.conv1 = eqx.nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding, use_bias=False, key=key1)
        self.bn1 = eqx.nn.BatchNorm(channels, axis_name="batch", mode="batch")
        self.conv2 = eqx.nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding, use_bias=False, key=key2)
        self.bn2 = eqx.nn.BatchNorm(channels, axis_name="batch", mode="batch")

    def __call__(self, x: jnp.ndarray, state: eqx.nn.State) -> tuple[jnp.ndarray, eqx.nn.State]:
        out, state = self.bn1(self.conv1(x), state)
        out = jax.nn.relu(out)
        out, state = self.bn2(self.conv2(out), state)
        return jax.nn.relu(x + out), state

class MLP(eqx.Module):
    layers: list[eqx.nn.Linear]
    
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_sizes: Sequence[int],
        key: jax.Array
    ):
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        sizes = [int(in_size)] + [int(h) for h in hidden_sizes] + [int(out_size)]
        
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(
                eqx.nn.Linear(sizes[i], sizes[i+1], key=keys[i])
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

