"""Shared observation encoders used across network modules."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class ObsEncoderFlatSymbolic(nn.Module):
    """Flat symbolic observation encoder.

    Input Shape: (*batch_dims, obs_dim)
    Output Shape: (*batch_dims, obs_emb_dim)
    """

    obs_emb_dim: int

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        x = observations.astype(jnp.float32)
        x = nn.LayerNorm()(x)
        x = nn.Dense(
            self.obs_emb_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.gelu(x)
        x = nn.Dense(
            self.obs_emb_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.gelu(x)
        return x
