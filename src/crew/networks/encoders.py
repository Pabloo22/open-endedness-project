"""Shared observation encoders used across network modules."""

from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CraftaxSymbolicObservationSpec:
    """Static schema for one flat symbolic Craftax observation."""

    env_id: str
    height: int
    width: int
    num_block_tokens: int
    num_item_tokens: int
    num_actor_channels: int
    has_items: bool
    has_visibility: bool
    extra_features_dim: int

    @property
    def map_channels(self) -> int:
        return self.num_block_tokens + self.num_item_tokens + self.num_actor_channels + int(self.has_visibility)

    @property
    def flat_map_dim(self) -> int:
        return self.height * self.width * self.map_channels

    @property
    def total_obs_dim(self) -> int:
        return self.flat_map_dim + self.extra_features_dim


class StructuredCraftaxObservation(NamedTuple):
    """Structured symbolic observation derived from the flat Craftax vector."""

    block_ids: jax.Array
    item_ids: jax.Array | None
    actor_multihot: jax.Array
    visibility: jax.Array | None
    extra_features: jax.Array


def get_craftax_symbolic_observation_spec(env_id: str) -> CraftaxSymbolicObservationSpec:
    """Return the static symbolic observation schema for a supported Craftax env."""
    if env_id == CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID:
        return CraftaxSymbolicObservationSpec(
            env_id=env_id,
            height=7,
            width=9,
            num_block_tokens=17,
            num_item_tokens=0,
            num_actor_channels=4,
            has_items=False,
            has_visibility=False,
            extra_features_dim=22,
        )
    if env_id == CRAFTAX_SYMBOLIC_ENV_ID:
        return CraftaxSymbolicObservationSpec(
            env_id=env_id,
            height=9,
            width=11,
            num_block_tokens=37,
            num_item_tokens=5,
            num_actor_channels=40,
            has_items=True,
            has_visibility=True,
            extra_features_dim=51,
        )

    msg = (
        "Structured Craftax observation encoder only supports "
        f"{SUPPORTED_STRUCTURED_ENCODER_ENV_IDS}. Received env_id={env_id!r}."
    )
    raise ValueError(msg)


def _reshape_flat_observations(observations: jax.Array) -> tuple[jax.Array, Sequence[int]]:
    """Flatten leading batch dims into one axis for encoder internals."""
    batch_shape = observations.shape[:-1]
    flattened = observations.astype(jnp.float32).reshape((-1, observations.shape[-1]))
    return flattened, batch_shape


def _restore_batch_shape(observations: jax.Array, batch_shape: Sequence[int]) -> jax.Array:
    """Restore flattened encoder outputs back to the original leading dims."""
    return observations.reshape((*batch_shape, observations.shape[-1]))


def _extract_token_ids(channel_bank: jax.Array, visible_mask: jax.Array | None) -> jax.Array:
    """Convert one-hot channels into token ids with 0 reserved for unseen cells."""
    token_ids = jnp.argmax(channel_bank, axis=-1).astype(jnp.int32) + 1
    is_present = jnp.sum(channel_bank, axis=-1) > 0.0
    if visible_mask is None:
        is_observed = is_present
    else:
        is_observed = jnp.logical_and(is_present, visible_mask)
    return jnp.where(is_observed, token_ids, jnp.zeros_like(token_ids))


def split_flat_craftax_symbolic_observation(
    observations: jax.Array,
    obs_spec: CraftaxSymbolicObservationSpec,
) -> StructuredCraftaxObservation:
    """Split flat symbolic Craftax observations into structured spatial factors.

    Shapes:
    - observations: [N, obs_dim]
    - block_ids: [N, H, W]
    - item_ids: [N, H, W] for full Craftax, otherwise ``None``
    - actor_multihot: [N, H, W, A]
    - visibility: [N, H, W] for full Craftax, otherwise ``None``
    - extra_features: [N, extra_dim]
    """
    if observations.ndim != 2:
        msg = f"Expected observations with shape [N, obs_dim]. Received {observations.shape}."
        raise ValueError(msg)
    if observations.shape[-1] != obs_spec.total_obs_dim:
        msg = (
            "Observation dim does not match the Craftax symbolic schema. "
            f"Expected {obs_spec.total_obs_dim}, received {observations.shape[-1]} "
            f"for env_id={obs_spec.env_id!r}."
        )
        raise ValueError(msg)

    flat_map = observations[:, : obs_spec.flat_map_dim]
    extra_features = observations[:, obs_spec.flat_map_dim :]
    map_channels = flat_map.reshape((-1, obs_spec.height, obs_spec.width, obs_spec.map_channels))

    channel_start = 0
    block_channels = map_channels[..., channel_start : channel_start + obs_spec.num_block_tokens]
    channel_start += obs_spec.num_block_tokens

    item_channels = None
    if obs_spec.has_items:
        item_channels = map_channels[..., channel_start : channel_start + obs_spec.num_item_tokens]
        channel_start += obs_spec.num_item_tokens

    actor_multihot = map_channels[..., channel_start : channel_start + obs_spec.num_actor_channels].astype(jnp.float32)
    channel_start += obs_spec.num_actor_channels

    visibility = None
    visible_mask = None
    if obs_spec.has_visibility:
        visibility = map_channels[..., channel_start].astype(jnp.int32)
        visible_mask = visibility.astype(bool)

    block_ids = _extract_token_ids(block_channels, visible_mask=visible_mask)
    item_ids = None
    if item_channels is not None:
        item_ids = _extract_token_ids(item_channels, visible_mask=visible_mask)

    if visible_mask is not None:
        actor_multihot = actor_multihot * visible_mask[..., None].astype(jnp.float32)

    return StructuredCraftaxObservation(
        block_ids=block_ids,
        item_ids=item_ids,
        actor_multihot=actor_multihot,
        visibility=visibility,
        extra_features=extra_features.astype(jnp.float32),
    )


def _dense(features: int) -> nn.Dense:
    return nn.Dense(
        features,
        kernel_init=orthogonal(np.sqrt(2)),
        bias_init=constant(0.0),
    )


def _conv(features: int) -> nn.Conv:
    return nn.Conv(
        features=features,
        kernel_size=(3, 3),
        padding="SAME",
        kernel_init=orthogonal(np.sqrt(2)),
        bias_init=constant(0.0),
    )


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
        x = _dense(self.obs_emb_dim)(x)
        x = nn.gelu(x)
        x = _dense(self.obs_emb_dim)(x)
        x = nn.gelu(x)
        return x


class ObsEncoderCraftaxStructured(nn.Module):
    """Craftax-specific structured symbolic observation encoder.

    Input Shape: (*batch_dims, obs_dim)
    Output Shape: (*batch_dims, obs_emb_dim)
    """

    obs_emb_dim: int
    obs_spec: CraftaxSymbolicObservationSpec

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        observations_flat, batch_shape = _reshape_flat_observations(observations)
        structured = split_flat_craftax_symbolic_observation(observations_flat, self.obs_spec)

        # Shapes:
        # - block_ids: [N, H, W]
        # - actor_multihot: [N, H, W, A]
        # - extra_features: [N, E]
        block_embedding = nn.Embed(
            num_embeddings=self.obs_spec.num_block_tokens + 1,
            features=16,
            embedding_init=normal(stddev=0.02),
            name="block_embedding",
        )(structured.block_ids)
        cell_features = [block_embedding]

        if structured.item_ids is not None:
            item_embedding = nn.Embed(
                num_embeddings=self.obs_spec.num_item_tokens + 1,
                features=8,
                embedding_init=normal(stddev=0.02),
                name="item_embedding",
            )(structured.item_ids)
            cell_features.append(item_embedding)

        actor_embedding_table = self.param(
            "actor_embedding_table",
            normal(stddev=0.02),
            (self.obs_spec.num_actor_channels, 16),
        )
        actor_embedding = structured.actor_multihot @ actor_embedding_table
        actor_is_present = jnp.any(structured.actor_multihot > 0.0, axis=-1, keepdims=True)
        no_actor_embedding = self.param("no_actor_embedding", normal(stddev=0.02), (16,))
        actor_embedding = actor_embedding + (1.0 - actor_is_present.astype(jnp.float32)) * no_actor_embedding.reshape(
            (1, 1, 1, -1)
        )
        cell_features.append(actor_embedding)

        if structured.visibility is not None:
            visibility_embedding = nn.Embed(
                num_embeddings=2,
                features=4,
                embedding_init=normal(stddev=0.02),
                name="visibility_embedding",
            )(structured.visibility)
            cell_features.append(visibility_embedding)

        cell_features = jnp.concatenate(cell_features, axis=-1)
        cell_features = _dense(32)(cell_features)
        cell_features = nn.gelu(cell_features)

        spatial_features = _conv(32)(cell_features)
        spatial_features = nn.gelu(spatial_features)
        spatial_features = _conv(32)(spatial_features)
        spatial_features = nn.gelu(spatial_features)
        spatial_features = spatial_features.reshape((spatial_features.shape[0], -1))

        extra_features = nn.LayerNorm()(structured.extra_features)
        extra_features = _dense(64)(extra_features)
        extra_features = nn.gelu(extra_features)

        fused_features = jnp.concatenate([spatial_features, extra_features], axis=-1)
        fused_features = _dense(self.obs_emb_dim)(fused_features)
        fused_features = nn.gelu(fused_features)
        fused_features = _dense(self.obs_emb_dim)(fused_features)
        fused_features = nn.gelu(fused_features)
        return _restore_batch_shape(fused_features, batch_shape)


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
        return ObsEncoderCraftaxStructured(
            obs_emb_dim=obs_emb_dim,
            obs_spec=get_craftax_symbolic_observation_spec(env_id),
        )

    msg = f"Unsupported encoder_mode {encoder_mode!r}. Supported values: {SUPPORTED_ENCODER_MODES}."
    raise ValueError(msg)
