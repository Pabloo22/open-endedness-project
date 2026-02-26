"""Alpha-conditioned actor-critic network for the main algorithm."""

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from crew.networks.encoders import ObsEncoderFlatSymbolic
from crew.networks.transformer_xl_base import Transformer_XL


class ActorCriticTransformer(nn.Module):
    num_actions: int
    num_reward_functions: int
    # observation encoder
    obs_emb_dim: int
    # transformer
    hidden_dim: int
    num_attn_heads: int
    qkv_features: int
    num_layers_in_transformer: int
    gating: bool
    gating_bias: float
    # actor and critic heads
    head_activation: str
    head_hidden_dim: int
    # alpha injection toggles
    inject_alpha_at_trunk: bool
    inject_alpha_at_actor_head: bool
    inject_alpha_at_critic_head: bool

    def setup(self):
        self.input_encoder = ObsEncoderFlatSymbolic(obs_emb_dim=self.obs_emb_dim)
        self.transformer = Transformer_XL(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_attn_heads,
            qkv_features=self.qkv_features,
            num_layers=self.num_layers_in_transformer,
            gating=self.gating,
            gating_bias=self.gating_bias,
        )

        if self.head_activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

        self.actor_linear1 = nn.Dense(
            self.head_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.actor_linear2 = nn.Dense(
            self.head_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.actor_out = nn.Dense(
            self.num_actions,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )

        self.critic_linear1 = nn.Dense(
            self.head_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.critic_linear2 = nn.Dense(
            self.head_hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.critic_out = nn.Dense(
            self.num_reward_functions,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )

    def __call__(
        self,
        memories: jax.Array,
        observations: jax.Array,
        mask: jax.Array,
        alpha_batch: jax.Array,
    ) -> tuple[distrax.Categorical, jax.Array]:
        """Forward pass without memory cache output (eval stepping mode)."""
        # Expected shapes:
        # - observations: [B, 1, obs_dim]
        # - alpha_batch: [B, R]
        encoded_obs = self.input_encoder(observations=observations[:, 0, :])  # [B, obs_emb]

        if self.inject_alpha_at_trunk:
            transformer_input = jnp.concatenate((encoded_obs, alpha_batch), axis=-1)
        else:
            transformer_input = encoded_obs
        features = self.transformer(memories, transformer_input, mask)  # [B, hidden_dim]

        if self.inject_alpha_at_actor_head:
            actor_input = jnp.concatenate((features, alpha_batch), axis=-1)
        else:
            actor_input = features
        actor = self.actor_linear1(actor_input)
        actor = self.activation_fn(actor)
        actor = self.actor_linear2(actor)
        actor = self.activation_fn(actor)
        actor_logits = self.actor_out(actor)
        pi = distrax.Categorical(logits=actor_logits)

        if self.inject_alpha_at_critic_head:
            critic_input = jnp.concatenate((features, alpha_batch), axis=-1)
        else:
            critic_input = features
        critic = self.critic_linear1(critic_input)
        critic = self.activation_fn(critic)
        critic = self.critic_linear2(critic)
        critic = self.activation_fn(critic)
        values = self.critic_out(critic)  # [B, R]

        return pi, values

    def model_forward_eval(
        self,
        memories: jax.Array,
        observations: jax.Array,
        mask: jax.Array,
        alpha_batch: jax.Array,
    ) -> tuple[distrax.Categorical, jax.Array, jax.Array]:
        """Eval forward pass used during rollout collection."""
        # Expected shapes:
        # - observations: [B, 1, obs_dim]
        # - alpha_batch: [B, R]
        encoded_obs = self.input_encoder(observations=observations[:, 0, :])  # [B, obs_emb]

        if self.inject_alpha_at_trunk:
            transformer_input = jnp.concatenate((encoded_obs, alpha_batch), axis=-1)
        else:
            transformer_input = encoded_obs
        features, memory_out = self.transformer.forward_eval(memories, transformer_input, mask)

        if self.inject_alpha_at_actor_head:
            actor_input = jnp.concatenate((features, alpha_batch), axis=-1)
        else:
            actor_input = features
        actor = self.actor_linear1(actor_input)
        actor = self.activation_fn(actor)
        actor = self.actor_linear2(actor)
        actor = self.activation_fn(actor)
        actor_logits = self.actor_out(actor)
        pi = distrax.Categorical(logits=actor_logits)

        if self.inject_alpha_at_critic_head:
            critic_input = jnp.concatenate((features, alpha_batch), axis=-1)
        else:
            critic_input = features
        critic = self.critic_linear1(critic_input)
        critic = self.activation_fn(critic)
        critic = self.critic_linear2(critic)
        critic = self.activation_fn(critic)
        values = self.critic_out(critic)  # [B, R]

        return pi, values, memory_out

    def model_forward_train(
        self,
        memories: jax.Array,
        observations: jax.Array,
        mask: jax.Array,
        alpha_batch: jax.Array,
    ) -> tuple[distrax.Categorical, jax.Array]:
        """Train forward pass used during PPO minibatch updates."""
        # Expected shapes:
        # - observations: [B, T, obs_dim]
        # - alpha_batch: [B, R]
        encoded_obs = self.input_encoder(observations=observations)  # [B, T, obs_emb]

        if self.inject_alpha_at_trunk or self.inject_alpha_at_actor_head or self.inject_alpha_at_critic_head:
            alpha_over_time = jnp.broadcast_to(alpha_batch[:, None, :], (alpha_batch.shape[0], encoded_obs.shape[1], alpha_batch.shape[1]))  # [B, T, R]

        if self.inject_alpha_at_trunk:
            transformer_input = jnp.concatenate((encoded_obs, alpha_over_time), axis=-1)  # [B, T, obs_emb + R]
        else:
            transformer_input = encoded_obs
        features = self.transformer.forward_train(memories, transformer_input, mask)  # [B, T, hidden_dim]

        if self.inject_alpha_at_actor_head:
            actor_input = jnp.concatenate((features, alpha_over_time), axis=-1)
        else:
            actor_input = features
        actor = self.actor_linear1(actor_input)
        actor = self.activation_fn(actor)
        actor = self.actor_linear2(actor)
        actor = self.activation_fn(actor)
        actor_logits = self.actor_out(actor)
        pi = distrax.Categorical(logits=actor_logits)

        if self.inject_alpha_at_critic_head:
            critic_input = jnp.concatenate((features, alpha_over_time), axis=-1)
        else:
            critic_input = features
        critic = self.critic_linear1(critic_input)
        critic = self.activation_fn(critic)
        critic = self.critic_linear2(critic)
        critic = self.activation_fn(critic)
        values = self.critic_out(critic)  # [B, T, R]

        return pi, values
