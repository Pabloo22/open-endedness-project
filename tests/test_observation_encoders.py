import unittest

import jax
import jax.numpy as jnp
import numpy as np

from curemix.main_algo.actor_critic import ActorCriticTransformer
from curemix.main_algo.intrinsic_modules.icm import ICMNet
from curemix.main_algo.intrinsic_modules.ngu import NGUEmbeddingNetwork
from curemix.main_algo.intrinsic_modules.rnd import RNDTargetAndPredictor
from curemix.networks.encoders import (
    CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID,
    CRAFTAX_SYMBOLIC_ENV_ID,
    build_observation_encoder,
    extract_inventory_from_flat_craftax_symbolic_observation,
    split_flat_craftax_symbolic_observation,
)


def _layout(env_id: str) -> dict[str, int | bool]:
    if env_id == CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID:
        return {
            "height": 7,
            "width": 9,
            "num_block_tokens": 17,
            "num_item_tokens": 0,
            "num_actor_channels": 4,
            "map_channels": 21,
            "extra_features_dim": 22,
            "inventory_dim": 12,
            "total_obs_dim": 1345,
            "has_items": False,
            "has_visibility": False,
        }
    return {
        "height": 9,
        "width": 11,
        "num_block_tokens": 37,
        "num_item_tokens": 5,
        "num_actor_channels": 40,
        "map_channels": 83,
        "extra_features_dim": 51,
        "inventory_dim": 16,
        "total_obs_dim": 8268,
        "has_items": True,
        "has_visibility": True,
    }


def _build_flat_obs(
    *,
    env_id: str,
    visible_cells: list[tuple[int, int, int, int | None, tuple[int, ...]]],
    extra_features: np.ndarray | None = None,
) -> jax.Array:
    layout = _layout(env_id)
    map_tensor = np.zeros((layout["height"], layout["width"], layout["map_channels"]), dtype=np.float32)

    for row, col, block_channel, item_channel, actor_channels in visible_cells:
        map_tensor[row, col, block_channel] = 1.0

        channel_offset = layout["num_block_tokens"]
        if layout["has_items"]:
            if item_channel is None:
                raise ValueError("Full Craftax visible cells must specify an item channel, including ItemType.NONE.")
            map_tensor[row, col, channel_offset + item_channel] = 1.0
            channel_offset += layout["num_item_tokens"]

        for actor_channel in actor_channels:
            map_tensor[row, col, channel_offset + actor_channel] = 1.0
        channel_offset += layout["num_actor_channels"]

        if layout["has_visibility"]:
            map_tensor[row, col, channel_offset] = 1.0

    if extra_features is None:
        extra_features = np.arange(layout["extra_features_dim"], dtype=np.float32)

    flat_obs = np.concatenate([map_tensor.reshape((-1,)), extra_features], axis=0)
    return jnp.asarray(flat_obs[None, :], dtype=jnp.float32)


class TestCraftaxStructuredObservationSplit(unittest.TestCase):
    def test_split_full_craftax_symbolic_observation(self):
        observations = _build_flat_obs(
            env_id=CRAFTAX_SYMBOLIC_ENV_ID,
            visible_cells=[
                (2, 3, 6, 2, (1, 17)),
                (1, 1, 4, 0, ()),
            ],
        )
        structured = split_flat_craftax_symbolic_observation(observations, CRAFTAX_SYMBOLIC_ENV_ID)

        self.assertEqual(structured.block_ids.shape, (1, 9, 11))
        self.assertEqual(structured.item_ids.shape, (1, 9, 11))
        self.assertEqual(structured.actor_multihot.shape, (1, 9, 11, 40))
        self.assertEqual(structured.visibility.shape, (1, 9, 11))
        self.assertEqual(structured.extra_features.shape, (1, 51))

        self.assertEqual(int(structured.block_ids[0, 2, 3]), 7)
        self.assertEqual(int(structured.item_ids[0, 2, 3]), 3)
        self.assertEqual(int(structured.visibility[0, 2, 3]), 1)
        self.assertEqual(float(structured.actor_multihot[0, 2, 3, 1]), 1.0)
        self.assertEqual(float(structured.actor_multihot[0, 2, 3, 17]), 1.0)

        # Visible cell with ItemType.NONE should not be confused with an unseen cell.
        self.assertEqual(int(structured.item_ids[0, 1, 1]), 1)
        self.assertEqual(int(structured.visibility[0, 1, 1]), 1)

        # Unseen cells stay at the reserved token id 0 and carry no actor signal.
        self.assertEqual(int(structured.block_ids[0, 0, 0]), 0)
        self.assertEqual(int(structured.item_ids[0, 0, 0]), 0)
        self.assertEqual(float(jnp.sum(structured.actor_multihot[0, 0, 0])), 0.0)
        np.testing.assert_allclose(
            np.asarray(structured.extra_features[0]),
            np.arange(51, dtype=np.float32),
        )

    def test_split_classic_craftax_symbolic_observation(self):
        observations = _build_flat_obs(
            env_id=CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID,
            visible_cells=[(3, 4, 5, None, (2,))],
        )
        structured = split_flat_craftax_symbolic_observation(observations, CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID)

        self.assertEqual(structured.block_ids.shape, (1, 7, 9))
        self.assertIsNone(structured.item_ids)
        self.assertEqual(structured.actor_multihot.shape, (1, 7, 9, 4))
        self.assertIsNone(structured.visibility)
        self.assertEqual(structured.extra_features.shape, (1, 22))

        self.assertEqual(int(structured.block_ids[0, 3, 4]), 6)
        self.assertEqual(float(structured.actor_multihot[0, 3, 4, 2]), 1.0)
        np.testing.assert_allclose(
            np.asarray(structured.extra_features[0]),
            np.arange(22, dtype=np.float32),
        )


class TestCraftaxStructuredEncoder(unittest.TestCase):
    def test_structured_encoder_preserves_leading_batch_dims_and_jits(self):
        for env_id in (CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID, CRAFTAX_SYMBOLIC_ENV_ID):
            with self.subTest(env_id=env_id):
                layout = _layout(env_id)
                encoder = build_observation_encoder(
                    encoder_mode="craftax_structured",
                    env_id=env_id,
                    obs_emb_dim=32,
                )
                observations = jnp.zeros((2, 3, layout["total_obs_dim"]), dtype=jnp.float32)
                params = encoder.init(jax.random.key(0), observations)

                outputs = encoder.apply(params, observations)
                self.assertEqual(outputs.shape, (2, 3, 32))

                jitted_apply = jax.jit(lambda x: encoder.apply(params, x))
                jitted_outputs = jitted_apply(observations)
                self.assertEqual(jitted_outputs.shape, (2, 3, 32))


class TestInventoryOnlyEncoder(unittest.TestCase):
    def test_inventory_slice_matches_the_first_extra_features(self):
        layout = _layout(CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID)
        observations = _build_flat_obs(env_id=CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID, visible_cells=[])
        inventory = extract_inventory_from_flat_craftax_symbolic_observation(observations, CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID)
        np.testing.assert_allclose(
            np.asarray(inventory[0]),
            np.arange(layout["inventory_dim"], dtype=np.float32),
        )

    def test_inventory_only_encoder_preserves_leading_batch_dims_and_jits(self):
        layout = _layout(CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID)
        encoder = build_observation_encoder(
            encoder_mode="inventory_only",
            env_id=CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID,
            obs_emb_dim=32,
        )
        observations = jnp.zeros((2, 3, layout["total_obs_dim"]), dtype=jnp.float32)
        params = encoder.init(jax.random.key(5), observations)

        outputs = encoder.apply(params, observations)
        self.assertEqual(outputs.shape, (2, 3, 32))

        jitted_apply = jax.jit(lambda x: encoder.apply(params, x))
        jitted_outputs = jitted_apply(observations)
        self.assertEqual(jitted_outputs.shape, (2, 3, 32))

    def test_inventory_only_encoder_is_not_supported_for_full_craftax(self):
        with self.assertRaises(ValueError):
            build_observation_encoder(
                encoder_mode="inventory_only",
                env_id=CRAFTAX_SYMBOLIC_ENV_ID,
                obs_emb_dim=32,
            )


class TestStructuredEncoderIntegration(unittest.TestCase):
    def test_actor_critic_init_and_forward_with_structured_encoder(self):
        layout = _layout(CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID)
        network = ActorCriticTransformer(
            num_actions=17,
            num_reward_functions=2,
            env_id=CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID,
            encoder_mode="craftax_structured",
            obs_emb_dim=32,
            hidden_dim=16,
            num_attn_heads=2,
            qkv_features=16,
            num_layers_in_transformer=1,
            gating=True,
            gating_bias=2.0,
            head_activation="relu",
            head_hidden_dim=16,
            inject_alpha_at_trunk=True,
            inject_alpha_at_actor_head=True,
            inject_alpha_at_critic_head=True,
        )
        init_memory = jnp.zeros((2, 4, 1, 16), dtype=jnp.float32)
        init_obs = jnp.zeros((2, 1, layout["total_obs_dim"]), dtype=jnp.float32)
        init_mask = jnp.zeros((2, 2, 1, 5), dtype=jnp.bool_)
        init_alpha = jnp.zeros((2, 2), dtype=jnp.float32)

        params = network.init(jax.random.key(1), init_memory, init_obs, init_mask, init_alpha)
        pi, values = network.apply(params, init_memory, init_obs, init_mask, init_alpha)

        self.assertEqual(pi.logits.shape, (2, 17))
        self.assertEqual(values.shape, (2, 2))

    def test_intrinsic_networks_init_with_structured_encoder(self):
        layout = _layout(CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID)
        observations = jnp.zeros((2, layout["total_obs_dim"]), dtype=jnp.float32)
        next_observations = jnp.zeros((2, layout["total_obs_dim"]), dtype=jnp.float32)
        actions = jnp.zeros((2,), dtype=jnp.int32)

        rnd_network = RNDTargetAndPredictor(
            encoder_mode="craftax_structured",
            env_id=CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID,
            output_embedding_dim=16,
            obs_emb_dim=32,
            head_activation="relu",
            head_hidden_dim=16,
        )
        rnd_params = rnd_network.init(jax.random.key(2), observations)
        rnd_outputs = rnd_network.apply(rnd_params, observations)
        self.assertEqual(rnd_outputs.shape, (2, 16))

        ngu_network = NGUEmbeddingNetwork(
            encoder_mode="craftax_structured",
            env_id=CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID,
            output_embedding_dim=16,
            obs_emb_dim=32,
            head_activation="relu",
            head_hidden_dim=16,
        )
        ngu_params = ngu_network.init(jax.random.key(3), observations)
        ngu_outputs = ngu_network.apply(ngu_params, observations)
        self.assertEqual(ngu_outputs.shape, (2, 16))

        icm_network = ICMNet(
            env_id=CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID,
            encoder_mode="craftax_structured",
            obs_emb_dim=16,
            action_dim=17,
            forward_hidden_dims=[16],
            inverse_hidden_dims=[16],
            activation_fn="relu",
        )
        icm_params = icm_network.init(
            jax.random.key(4),
            observations,
            next_observations,
            actions,
            method=ICMNet.init_all,
        )
        z_hat_next, a_hat = icm_network.apply(
            icm_params,
            observations,
            next_observations,
            actions,
            method=ICMNet.init_all,
        )
        self.assertEqual(z_hat_next.shape, (2, 16))
        self.assertEqual(a_hat.shape, (2, 17))

    def test_intrinsic_networks_init_with_inventory_only_encoder(self):
        layout = _layout(CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID)
        observations = jnp.zeros((2, layout["total_obs_dim"]), dtype=jnp.float32)
        next_observations = jnp.zeros((2, layout["total_obs_dim"]), dtype=jnp.float32)
        actions = jnp.zeros((2,), dtype=jnp.int32)

        rnd_network = RNDTargetAndPredictor(
            encoder_mode="inventory_only",
            env_id=CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID,
            output_embedding_dim=16,
            obs_emb_dim=32,
            head_activation="relu",
            head_hidden_dim=16,
        )
        rnd_params = rnd_network.init(jax.random.key(6), observations)
        rnd_outputs = rnd_network.apply(rnd_params, observations)
        self.assertEqual(rnd_outputs.shape, (2, 16))

        ngu_network = NGUEmbeddingNetwork(
            encoder_mode="inventory_only",
            env_id=CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID,
            output_embedding_dim=16,
            obs_emb_dim=32,
            head_activation="relu",
            head_hidden_dim=16,
        )
        ngu_params = ngu_network.init(jax.random.key(7), observations)
        ngu_outputs = ngu_network.apply(ngu_params, observations)
        self.assertEqual(ngu_outputs.shape, (2, 16))

        icm_network = ICMNet(
            env_id=CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID,
            encoder_mode="inventory_only",
            obs_emb_dim=16,
            action_dim=17,
            forward_hidden_dims=[16],
            inverse_hidden_dims=[16],
            activation_fn="relu",
        )
        icm_params = icm_network.init(
            jax.random.key(8),
            observations,
            next_observations,
            actions,
            method=ICMNet.init_all,
        )
        z_hat_next, a_hat = icm_network.apply(
            icm_params,
            observations,
            next_observations,
            actions,
            method=ICMNet.init_all,
        )
        self.assertEqual(z_hat_next.shape, (2, 16))
        self.assertEqual(a_hat.shape, (2, 17))


if __name__ == "__main__":
    unittest.main()
