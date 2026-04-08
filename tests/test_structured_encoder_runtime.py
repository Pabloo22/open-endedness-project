import unittest

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name

from curemix.main_algo.config import CurriculumConfig, ICMConfig, NGUConfig, RNDConfig, TrainConfig
from curemix.main_algo.intrinsic_modules.registry import get_intrinsic_module
from curemix.main_algo.main_loop import full_training
from curemix.main_algo.setups import set_up_for_training
from curemix.main_algo.types import IntrinsicModulesUpdateData, TransitionDataBase
from curemix.networks.encoders import (
    CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID,
    CRAFTAX_SYMBOLIC_ENV_ID,
    split_flat_craftax_symbolic_observation,
)


def _layout(env_id: str) -> dict[str, int]:
    if env_id == CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID:
        return {
            "num_block_tokens": 17,
            "num_item_tokens": 0,
            "flat_map_dim": 1323,
            "total_obs_dim": 1345,
        }
    return {
        "num_block_tokens": 37,
        "num_item_tokens": 5,
        "flat_map_dim": 8217,
        "total_obs_dim": 8268,
    }


def _roundtrip_flat_observations(
    observations: jax.Array,
    env_id: str,
) -> jax.Array:
    layout = _layout(env_id)
    structured = split_flat_craftax_symbolic_observation(observations, env_id)

    block_ids = structured.block_ids
    block_one_hot = jax.nn.one_hot(
        jnp.maximum(block_ids - 1, 0),
        num_classes=layout["num_block_tokens"],
        dtype=jnp.float32,
    ) * (block_ids > 0)[..., None].astype(jnp.float32)

    map_parts = [block_one_hot]
    if structured.item_ids is not None:
        item_ids = structured.item_ids
        item_one_hot = jax.nn.one_hot(
            jnp.maximum(item_ids - 1, 0),
            num_classes=layout["num_item_tokens"],
            dtype=jnp.float32,
        ) * (item_ids > 0)[..., None].astype(jnp.float32)
        map_parts.append(item_one_hot)

    map_parts.append(structured.actor_multihot.astype(jnp.float32))
    if structured.visibility is not None:
        map_parts.append(structured.visibility[..., None].astype(jnp.float32))

    reconstructed_map = jnp.concatenate(map_parts, axis=-1).reshape((observations.shape[0], layout["flat_map_dim"]))
    return jnp.concatenate([reconstructed_map, structured.extra_features], axis=-1)


def _sample_reset_observations(env_id: str, num_samples: int, seed_offset: int = 0) -> jax.Array:
    env = make_craftax_env_from_name(env_id, auto_reset=False)
    env_params = env.default_params
    observations = []
    for sample_idx in range(num_samples):
        obs, _ = env.reset(jax.random.key(seed_offset + sample_idx), env_params)
        observations.append(np.asarray(obs, dtype=np.float32))
    return jnp.asarray(np.stack(observations, axis=0), dtype=jnp.float32)


def _base_small_config_kwargs(env_id: str) -> dict:
    return {
        "env_id": env_id,
        "encoder_mode": "craftax_structured",
        "total_timesteps": 16,
        "num_envs_per_batch": 4,
        "num_steps_per_env": 4,
        "num_steps_per_update": 4,
        "num_minibatches": 1,
        "past_context_length": 4,
        "subsequence_length_in_loss_calculation": 4,
        "num_transformer_blocks": 1,
        "transformer_hidden_states_dim": 16,
        "qkv_features": 16,
        "num_attn_heads": 2,
        "head_hidden_dim": 16,
        "obs_emb_dim": 32,
        "episode_max_steps": 4,
        "eval_every_n_batches": 1,
        "eval_num_envs": 1,
        "eval_num_episodes": 1,
        "enable_wandb": False,
        "curriculum": CurriculumConfig(
            replay_buffer_num_batches=4,
            predictor_lr=1e-4,
            predictor_update_epochs=1,
            predictor_num_minibatches=4,
            predictor_hidden_dim=16,
            importance_num_candidates_multiplier=2,
            min_batches_for_predictor_sampling=1,
        ),
    }


def _build_intrinsic_module_config(module_name: str, env_id: str = CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID) -> TrainConfig:
    kwargs = _base_small_config_kwargs(env_id)
    kwargs["selected_intrinsic_modules"] = (module_name,)
    kwargs["rnd"] = RNDConfig(
        encoder_mode="craftax_structured",
        output_embedding_dim=16,
        head_hidden_dim=16,
        predictor_update_epochs=1,
        predictor_num_minibatches=4,
        num_chunks_in_rewards_computation=4,
    )
    kwargs["ngu"] = NGUConfig(
        encoder_mode="craftax_structured",
        output_embedding_dim=16,
        head_hidden_dim=16,
        episodic_memory_capacity=4,
        num_neighbors=2,
        embedding_num_minibatches=4,
        num_chunks_in_rewards_computation=4,
    )
    kwargs["icm"] = ICMConfig(
        encoder_mode="craftax_structured",
        forward_hidden_dims=[16],
        inverse_hidden_dims=[16],
        obs_emb_dim=16,
        update_epochs=1,
        num_minibatches=4,
        num_chunks_in_rewards_computation=4,
    )
    return TrainConfig(**kwargs)


def _build_inventory_only_intrinsic_config(
    module_name: str,
    env_id: str = CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID,
) -> TrainConfig:
    kwargs = _base_small_config_kwargs(env_id)
    kwargs["encoder_mode"] = "flat_symbolic"
    kwargs["selected_intrinsic_modules"] = (module_name,)
    kwargs["rnd"] = RNDConfig(
        encoder_mode="inventory_only",
        output_embedding_dim=16,
        head_hidden_dim=16,
        predictor_update_epochs=1,
        predictor_num_minibatches=4,
        num_chunks_in_rewards_computation=4,
    )
    kwargs["ngu"] = NGUConfig(
        encoder_mode="inventory_only",
        output_embedding_dim=16,
        head_hidden_dim=16,
        episodic_memory_capacity=4,
        num_neighbors=2,
        embedding_num_minibatches=4,
        num_chunks_in_rewards_computation=4,
    )
    kwargs["icm"] = ICMConfig(
        encoder_mode="inventory_only",
        forward_hidden_dims=[16],
        inverse_hidden_dims=[16],
        obs_emb_dim=16,
        update_epochs=1,
        num_minibatches=4,
        num_chunks_in_rewards_computation=4,
    )
    return TrainConfig(**kwargs)


def _make_intrinsic_transitions(config: TrainConfig) -> tuple[TransitionDataBase, IntrinsicModulesUpdateData]:
    batch_size = config.num_envs_per_batch
    seq_len = config.num_steps_per_update
    num_samples = seq_len * batch_size
    obs = _sample_reset_observations(config.env_id, num_samples, seed_offset=0).reshape((seq_len, batch_size, -1))
    next_obs = _sample_reset_observations(config.env_id, num_samples, seed_offset=100).reshape((seq_len, batch_size, -1))

    env = make_craftax_env_from_name(config.env_id, auto_reset=False)
    action_dim = env.action_space(env.default_params).n
    action = jnp.asarray(np.arange(num_samples).reshape((seq_len, batch_size)) % action_dim, dtype=jnp.int32)
    done = jnp.zeros((seq_len, batch_size), dtype=jnp.bool_)
    done = done.at[-1, 0].set(True)

    rollout = TransitionDataBase(
        obs=obs,
        next_obs=next_obs,
        action=action,
        done=done,
        reward=jnp.zeros((seq_len, batch_size), dtype=jnp.float32),
        value=jnp.zeros((seq_len, batch_size, config.num_reward_functions), dtype=jnp.float32),
        log_prob=jnp.zeros((seq_len, batch_size), dtype=jnp.float32),
    )
    update = IntrinsicModulesUpdateData(
        obs=obs,
        next_obs=next_obs,
        action=action,
        done=done,
    )
    return rollout, update


class TestStructuredEncoderRealEnvRoundtrip(unittest.TestCase):
    def test_splitter_roundtrips_real_env_observations(self):
        for env_id in (CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID, CRAFTAX_SYMBOLIC_ENV_ID):
            with self.subTest(env_id=env_id):
                observations = _sample_reset_observations(env_id, num_samples=3)
                reconstructed = _roundtrip_flat_observations(observations, env_id)
                np.testing.assert_allclose(
                    np.asarray(reconstructed),
                    np.asarray(observations),
                    rtol=0.0,
                    atol=1e-6,
                )


class TestStructuredEncoderIntrinsicModuleSmoke(unittest.TestCase):
    def test_intrinsic_modules_compute_rewards_and_update_with_structured_encoder(self):
        expected_metric_keys = {
            "rnd": ("intrinsic_modules/rnd/predictor_loss",),
            "ngu": ("intrinsic_modules/ngu/embedding_loss",),
            "icm": (
                "intrinsic_modules/icm/loss",
                "intrinsic_modules/icm/forward_loss",
                "intrinsic_modules/icm/inverse_loss",
            ),
        }

        for module_name in ("rnd", "ngu", "icm"):
            with self.subTest(module_name=module_name):
                config = _build_intrinsic_module_config(module_name)
                obs_shape = (_layout(config.env_id)["total_obs_dim"],)
                rollout, update_data = _make_intrinsic_transitions(config)
                module = get_intrinsic_module(module_name)

                rng = jax.random.key(0)
                state = module.init_state(rng, obs_shape, config)
                rewards = module.compute_rewards(rng, state, rollout, config)
                updated_state, metrics = module.update(rng, state, update_data, config)

                del updated_state
                self.assertEqual(rewards.shape, (config.num_steps_per_update, config.num_envs_per_batch))
                self.assertTrue(bool(jnp.all(jnp.isfinite(rewards))))
                self.assertEqual(set(metrics.keys()), set(expected_metric_keys[module_name]))
                for metric_value in metrics.values():
                    self.assertTrue(bool(jnp.all(jnp.isfinite(metric_value))))

    def test_intrinsic_modules_compute_rewards_and_update_with_inventory_only_encoder(self):
        for module_name in ("rnd", "ngu", "icm"):
            with self.subTest(module_name=module_name):
                config = _build_inventory_only_intrinsic_config(module_name)
                obs_shape = (_layout(config.env_id)["total_obs_dim"],)
                rollout, update_data = _make_intrinsic_transitions(config)
                module = get_intrinsic_module(module_name)

                rng = jax.random.key(1)
                state = module.init_state(rng, obs_shape, config)
                rewards = module.compute_rewards(rng, state, rollout, config)
                updated_state, metrics = module.update(rng, state, update_data, config)

                del updated_state
                self.assertEqual(rewards.shape, (config.num_steps_per_update, config.num_envs_per_batch))
                self.assertTrue(bool(jnp.all(jnp.isfinite(rewards))))
                for metric_value in metrics.values():
                    self.assertTrue(bool(jnp.all(jnp.isfinite(metric_value))))


class TestStructuredEncoderTrainingSmoke(unittest.TestCase):
    def test_full_training_runs_with_structured_encoder_for_both_symbolic_envs(self):
        for env_id in (CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID, CRAFTAX_SYMBOLIC_ENV_ID):
            with self.subTest(env_id=env_id):
                config = TrainConfig(
                    **_base_small_config_kwargs(env_id),
                    training_mode="baseline",
                    selected_intrinsic_modules=(),
                    baseline_fixed_training_alpha=(1.0,),
                )
                (
                    rng,
                    env,
                    env_params,
                    agent_train_state,
                    reward_normalization_stats,
                    intrinsic_modules,
                    intrinsic_states,
                    curriculum_state,
                ) = set_up_for_training(config)

                out = jax.block_until_ready(
                    full_training(
                        rng=rng,
                        agent_train_state=agent_train_state,
                        reward_normalization_stats=reward_normalization_stats,
                        intrinsic_states=intrinsic_states,
                        curriculum_state=curriculum_state,
                        env=env,
                        env_params=env_params,
                        intrinsic_modules=intrinsic_modules,
                        config=config,
                    )
                )
                self.assertIn("metrics", out)
                self.assertIn("eval/returns", out["metrics"])
                self.assertEqual(out["metrics"]["run/batch_idx"].shape[0], config.num_batches_of_envs)

    def test_full_training_runs_with_inventory_only_actor_encoder_for_classic_craftax(self):
        kwargs = _base_small_config_kwargs(CRAFTAX_CLASSIC_SYMBOLIC_ENV_ID)
        kwargs["encoder_mode"] = "inventory_only"
        config = TrainConfig(
            **kwargs,
            training_mode="baseline",
            selected_intrinsic_modules=(),
            baseline_fixed_training_alpha=(1.0,),
        )
        (
            rng,
            env,
            env_params,
            agent_train_state,
            reward_normalization_stats,
            intrinsic_modules,
            intrinsic_states,
            curriculum_state,
        ) = set_up_for_training(config)

        out = jax.block_until_ready(
            full_training(
                rng=rng,
                agent_train_state=agent_train_state,
                reward_normalization_stats=reward_normalization_stats,
                intrinsic_states=intrinsic_states,
                curriculum_state=curriculum_state,
                env=env,
                env_params=env_params,
                intrinsic_modules=intrinsic_modules,
                config=config,
            )
        )
        self.assertIn("metrics", out)
        self.assertIn("eval/returns", out["metrics"])


if __name__ == "__main__":
    unittest.main()
