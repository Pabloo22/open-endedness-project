"""Shared observation encoders used across network modules."""

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, normal, orthogonal


CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID = "Craftax-Classic-Symbolic-v1"
CRAFTAX_SYMBOLIC_ENV_ID = "Craftax-Symbolic-v1"
SUPPORTED_STRUCTURED_ENCODER_ENV_IDS = (
    CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID,
    CRAFTAX_SYMBOLIC_ENV_ID,
)
SUPPORTED_ENCODER_MODES = ("flat_symbolic", "craftax_structured")

CRAFTAX_CLASSIC_HEIGHT = 7
CRAFTAX_CLASSIC_WIDTH = 9
CRAFTAX_CLASSIC_BLOCK_TOKENS = 17
CRAFTAX_CLASSIC_ACTOR_CHANNELS = 4
CRAFTAX_CLASSIC_EXTRA_FEATURES_DIM = 22
CRAFTAX_CLASSIC_MAP_CHANNELS = CRAFTAX_CLASSIC_BLOCK_TOKENS + CRAFTAX_CLASSIC_ACTOR_CHANNELS
CRAFTAX_CLASSIC_TOTAL_OBS_DIM = (
    CRAFTAX_CLASSIC_HEIGHT * CRAFTAX_CLASSIC_WIDTH * CRAFTAX_CLASSIC_MAP_CHANNELS + CRAFTAX_CLASSIC_EXTRA_FEATURES_DIM
)

CRAFTAX_HEIGHT = 9
CRAFTAX_WIDTH = 11
CRAFTAX_BLOCK_TOKENS = 37
CRAFTAX_ITEM_TOKENS = 5
CRAFTAX_ACTOR_CHANNELS = 40
CRAFTAX_EXTRA_FEATURES_DIM = 51
CRAFTAX_MAP_CHANNELS = CRAFTAX_BLOCK_TOKENS + CRAFTAX_ITEM_TOKENS + CRAFTAX_ACTOR_CHANNELS + 1
CRAFTAX_TOTAL_OBS_DIM = CRAFTAX_HEIGHT * CRAFTAX_WIDTH * CRAFTAX_MAP_CHANNELS + CRAFTAX_EXTRA_FEATURES_DIM


class StructuredCraftaxObservation(NamedTuple):
    """Structured symbolic observation derived from the flat Craftax vector."""

    block_ids: jax.Array
    item_ids: jax.Array | None
    actor_multihot: jax.Array
    visibility: jax.Array | None
    extra_features: jax.Array


def split_flat_craftax_symbolic_observation(
    observations: jax.Array,
    env_id: str,
) -> StructuredCraftaxObservation:
    """Split flat symbolic Craftax observations into structured spatial factors."""
    if env_id == CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID:
        flat_map_dim = CRAFTAX_CLASSIC_HEIGHT * CRAFTAX_CLASSIC_WIDTH * CRAFTAX_CLASSIC_MAP_CHANNELS
        flat_map = observations[:, :flat_map_dim]
        extra_features = observations[:, flat_map_dim:]
        map_channels = flat_map.reshape(
            (-1, CRAFTAX_CLASSIC_HEIGHT, CRAFTAX_CLASSIC_WIDTH, CRAFTAX_CLASSIC_MAP_CHANNELS)
        )

        block_channels = map_channels[..., :CRAFTAX_CLASSIC_BLOCK_TOKENS]
        actor_multihot = map_channels[..., CRAFTAX_CLASSIC_BLOCK_TOKENS:].astype(jnp.float32)

        block_ids = jnp.argmax(block_channels, axis=-1).astype(jnp.int32) + 1
        block_ids = jnp.where(jnp.sum(block_channels, axis=-1) > 0.0, block_ids, jnp.zeros_like(block_ids))

        return StructuredCraftaxObservation(
            block_ids=block_ids,
            item_ids=None,
            actor_multihot=actor_multihot,
            visibility=None,
            extra_features=extra_features.astype(jnp.float32),
        )

    if env_id == CRAFTAX_SYMBOLIC_ENV_ID:
        flat_map_dim = CRAFTAX_HEIGHT * CRAFTAX_WIDTH * CRAFTAX_MAP_CHANNELS
        flat_map = observations[:, :flat_map_dim]
        extra_features = observations[:, flat_map_dim:]
        map_channels = flat_map.reshape((-1, CRAFTAX_HEIGHT, CRAFTAX_WIDTH, CRAFTAX_MAP_CHANNELS))

        block_channels = map_channels[..., :CRAFTAX_BLOCK_TOKENS]
        item_channels = map_channels[..., CRAFTAX_BLOCK_TOKENS : CRAFTAX_BLOCK_TOKENS + CRAFTAX_ITEM_TOKENS]
        actor_multihot = map_channels[
            ...,
            CRAFTAX_BLOCK_TOKENS + CRAFTAX_ITEM_TOKENS : CRAFTAX_BLOCK_TOKENS + CRAFTAX_ITEM_TOKENS + CRAFTAX_ACTOR_CHANNELS,
        ].astype(jnp.float32)
        visibility = map_channels[..., -1].astype(jnp.int32)
        visible_mask = visibility.astype(bool)

        block_ids = jnp.argmax(block_channels, axis=-1).astype(jnp.int32) + 1
        block_ids = jnp.where(
            jnp.logical_and(jnp.sum(block_channels, axis=-1) > 0.0, visible_mask),
            block_ids,
            jnp.zeros_like(block_ids),
        )

        item_ids = jnp.argmax(item_channels, axis=-1).astype(jnp.int32) + 1
        item_ids = jnp.where(
            jnp.logical_and(jnp.sum(item_channels, axis=-1) > 0.0, visible_mask),
            item_ids,
            jnp.zeros_like(item_ids),
        )

        actor_multihot = actor_multihot * visible_mask[..., None].astype(jnp.float32)
        return StructuredCraftaxObservation(
            block_ids=block_ids,
            item_ids=item_ids,
            actor_multihot=actor_multihot,
            visibility=visibility,
            extra_features=extra_features.astype(jnp.float32),
        )

    msg = (
        "Structured Craftax observation encoder only supports "
        f"{SUPPORTED_STRUCTURED_ENCODER_ENV_IDS}. Received env_id={env_id!r}."
    )
    raise ValueError(msg)


class ObsEncoderFlatSymbolic(nn.Module):
    """Flat symbolic observation encoder.

    Input Shape: (*batch_dims, obs_dim)
    Output Shape: (*batch_dims, obs_emb_dim)
    """

    obs_emb_dim: int

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        x = observations.astype(jnp.float32)
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


class ObsEncoderCraftaxStructured(nn.Module):
    """Craftax-specific structured symbolic observation encoder.

    Input Shape: (*batch_dims, obs_dim)
    Output Shape: (*batch_dims, obs_emb_dim)
    """

    env_id: str
    obs_emb_dim: int

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        batch_shape = observations.shape[:-1]
        # Flatten arbitrary leading batch dims, encode once, then restore them before returning.
        observations = observations.astype(jnp.float32).reshape((-1, observations.shape[-1]))
        structured = split_flat_craftax_symbolic_observation(observations, self.env_id)

        if self.env_id == CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID:
            num_block_tokens = CRAFTAX_CLASSIC_BLOCK_TOKENS
            num_actor_channels = CRAFTAX_CLASSIC_ACTOR_CHANNELS
        else:
            num_block_tokens = CRAFTAX_BLOCK_TOKENS
            num_actor_channels = CRAFTAX_ACTOR_CHANNELS

        block_embedding = nn.Embed(
            num_embeddings=num_block_tokens + 1,
            features=16,
            embedding_init=normal(stddev=0.02),
            name="block_embedding",
        )(structured.block_ids)

        actor_embedding_table = self.param(
            "actor_embedding_table",
            normal(stddev=0.02),
            (num_actor_channels, 16),
        )
        actor_embedding = structured.actor_multihot @ actor_embedding_table
        actor_is_present = jnp.any(structured.actor_multihot > 0.0, axis=-1, keepdims=True)
        no_actor_embedding = self.param("no_actor_embedding", normal(stddev=0.02), (16,))
        actor_embedding = actor_embedding + (1.0 - actor_is_present.astype(jnp.float32)) * no_actor_embedding.reshape(
            (1, 1, 1, -1)
        )

        if self.env_id == CRAFTAX_SYMBOLIC_ENV_ID:
            item_embedding = nn.Embed(
                num_embeddings=CRAFTAX_ITEM_TOKENS + 1,
                features=8,
                embedding_init=normal(stddev=0.02),
                name="item_embedding",
            )(structured.item_ids)
            visibility_embedding = nn.Embed(
                num_embeddings=2,
                features=4,
                embedding_init=normal(stddev=0.02),
                name="visibility_embedding",
            )(structured.visibility)
            cell_features = jnp.concatenate(
                [block_embedding, item_embedding, actor_embedding, visibility_embedding],
                axis=-1,
            )
        else:
            cell_features = jnp.concatenate([block_embedding, actor_embedding], axis=-1)

        cell_features = nn.Dense(
            32,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(cell_features)
        cell_features = nn.gelu(cell_features)

        spatial_features = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(cell_features)
        spatial_features = nn.gelu(spatial_features)
        spatial_features = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(spatial_features)
        spatial_features = nn.gelu(spatial_features)
        spatial_features = spatial_features.reshape((spatial_features.shape[0], -1))

        extra_features = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(structured.extra_features)
        extra_features = nn.gelu(extra_features)

        fused_features = jnp.concatenate([spatial_features, extra_features], axis=-1)
        fused_features = nn.Dense(
            self.obs_emb_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(fused_features)
        fused_features = nn.gelu(fused_features)
        return fused_features.reshape((*batch_shape, self.obs_emb_dim))


def build_observation_encoder(
    *,
    encoder_mode: str,
    env_id: str,
    obs_emb_dim: int,
) -> nn.Module:
    """Construct the requested observation encoder."""
    if encoder_mode == "flat_symbolic":
        return ObsEncoderFlatSymbolic(obs_emb_dim=obs_emb_dim)
    if encoder_mode == "craftax_structured":
        if env_id not in SUPPORTED_STRUCTURED_ENCODER_ENV_IDS:
            msg = (
                "Structured Craftax observation encoder only supports "
                f"{SUPPORTED_STRUCTURED_ENCODER_ENV_IDS}. Received env_id={env_id!r}."
            )
            raise ValueError(msg)
        return ObsEncoderCraftaxStructured(
            env_id=env_id,
            obs_emb_dim=obs_emb_dim,
        )

    msg = f"Unsupported encoder_mode {encoder_mode!r}. Supported values: {SUPPORTED_ENCODER_MODES}."
    raise ValueError(msg)
