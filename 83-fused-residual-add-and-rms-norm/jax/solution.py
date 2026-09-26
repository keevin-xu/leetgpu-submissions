# LeetGPU Solution
# Challenge: #83 · Fused Residual Add and RMS Norm
# Language: JAX (jax)

import jax
import jax.numpy as jnp


# x, residual, weight are tensors on device
@jax.jit
def solve(
    x: jax.Array,
    residual: jax.Array,
    weight: jax.Array,
    N: int,
    C: int,
    eps: float,
) -> jax.Array:
    # return output tensor directly
    pass
