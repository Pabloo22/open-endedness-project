import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

SUPPORTED_RND_ENCODER_MODES = ("flat_symbolic",)


class RNDEncoderFlatSymbolic(nn.Module):
    obs_emb_dim: int

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        # observations: [num_samples, obs_dim] -> [num_samples, obs_emb_dim]
        if observations.ndim != 2:
            msg = (
                "RNDEncoderFlatSymbolic expects observations with shape "
                f"[num_samples, obs_dim], got rank {observations.ndim}."
            )
            raise ValueError(msg)

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


class Target_and_Predictor(nn.Module):
    encoder_mode: str
    output_embedding_dim: int
    # encoder
    obs_emb_dim: int
    # mlp head
    head_activation: str
    mlp_dim: int

    def setup(self):
        if self.encoder_mode == "flat_symbolic":
            self.input_encoder = RNDEncoderFlatSymbolic(obs_emb_dim=self.obs_emb_dim)
        else:
            msg = (
                f"Unsupported rnd encoder_mode={self.encoder_mode!r}. "
                f"Expected one of {SUPPORTED_RND_ENCODER_MODES}."
            )
            raise ValueError(msg)

        if self.head_activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

        self.linear1 = nn.Dense(
            self.mlp_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.linear2 = nn.Dense(
            self.mlp_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.linear_out = nn.Dense(
            self.output_embedding_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )

    def __call__(self, observations: jax.Array) -> jax.Array:
        # observations: [num_samples, obs_dim]
        encoded_input = self.input_encoder(observations=observations)

        outputs = self.linear1(encoded_input)
        outputs = self.activation_fn(outputs)
        outputs = self.linear2(outputs)
        outputs = self.activation_fn(outputs)
        # output_logits: [num_samples, output_embedding_dim]
        output_logits = self.linear_out(outputs)

        return output_logits
