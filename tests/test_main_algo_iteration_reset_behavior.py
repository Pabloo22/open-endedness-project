import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from curemix.main_algo.main_loop import train_one_iteration
from curemix.main_algo.reward_normalization import init_reward_normalization_stats
from curemix.main_algo.types import CurriculumState, LpEstimationData


class _DummyResetOnlyEnv:
    def reset(self, key, params):
        del key, params
        return jnp.asarray([0.0, 0.0], dtype=jnp.float32), jnp.asarray(0, dtype=jnp.int32)


def _build_test_config(reset_running_return: bool) -> SimpleNamespace:
    return SimpleNamespace(
        num_envs_per_batch=2,
        num_reward_functions=2,
        num_updates_per_batch=1,
        num_steps_per_update=1,
        past_context_length=2,
        num_transformer_blocks=1,
        transformer_hidden_states_dim=4,
        num_attn_heads=2,
        reset_normalization_running_forward_return_on_new_alpha=reset_running_return,
        is_episodic_per_reward_function=jnp.asarray([True, False], dtype=jnp.bool_),
        gamma_per_reward_function=jnp.asarray([0.99, 0.99], dtype=jnp.float32),
        reward_norm_eps=1e-8,
        curriculum=SimpleNamespace(
            lp_norm_ema_beta=0.05,
            score_lp_mode="alp",
            score_lambda=0.5,
        ),
    )


class TestTrainOneIterationResetBehavior(unittest.TestCase):
    def _run_once_and_capture_scan_init_stats(self, reset_running_return: bool):
        config = _build_test_config(reset_running_return=reset_running_return)
        reward_stats = init_reward_normalization_stats(
            num_envs=config.num_envs_per_batch,
            num_reward_functions=config.num_reward_functions,
        ).replace(
            running_forward_return=jnp.asarray([[3.0, 7.0], [5.0, 11.0]], dtype=jnp.float32),
            previous_done=jnp.zeros((config.num_envs_per_batch, config.num_reward_functions), dtype=jnp.bool_),
        )

        curriculum_state = CurriculumState(
            alpha_score_replay_buffer=object(),
            score_predictor_train_state=object(),
            lp_normalization_stats=object(),
            num_batches_seen=jnp.array(0, dtype=jnp.int32),
        )

        captured: dict[str, object] = {}

        def fake_scan(fn, init, xs, length):  # noqa: ARG001
            captured["scan_init_reward_stats"] = init[0].reward_normalization_stats
            inner_metrics = {"dummy_metric": jnp.zeros((config.num_updates_per_batch,), dtype=jnp.float32)}
            lp_data = LpEstimationData(
                raw_rewards=jnp.zeros(
                    (config.num_updates_per_batch, 1, config.num_envs_per_batch, config.num_reward_functions),
                    dtype=jnp.float32,
                ),
                done_masks=jnp.zeros(
                    (config.num_updates_per_batch, 1, config.num_envs_per_batch, config.num_reward_functions),
                    dtype=jnp.bool_,
                ),
            )
            return init, (inner_metrics, lp_data)

        with (
            mock.patch(
                "crew.main_algo.main_loop.sample_alpha_batch",
                return_value=(
                    jax.random.key(1),
                    jnp.asarray([[0.6, 0.4], [0.2, 0.8]], dtype=jnp.float32),
                    {"curriculum/pred_score_mean": jnp.array(0.0, dtype=jnp.float32)},
                ),
            ),
            mock.patch("crew.main_algo.main_loop.jax.lax.scan", side_effect=fake_scan),
            mock.patch(
                "crew.main_algo.main_loop.update_lp_normalization_stats_from_data",
                return_value=SimpleNamespace(var=jnp.ones((config.num_reward_functions,), dtype=jnp.float32)),
            ),
            mock.patch(
                "crew.main_algo.main_loop.estimate_lp_per_reward_function",
                return_value=(
                    jnp.zeros((config.num_envs_per_batch, config.num_reward_functions), dtype=jnp.float32),
                    jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32),
                ),
            ),
            mock.patch(
                "crew.main_algo.main_loop.compute_scores",
                return_value=(
                    jnp.zeros((config.num_envs_per_batch,), dtype=jnp.float32),
                    {"curriculum/score_mean": jnp.array(0.0, dtype=jnp.float32)},
                ),
            ),
            mock.patch("crew.main_algo.main_loop.add_alpha_score_batch", return_value=object()),
            mock.patch(
                "crew.main_algo.main_loop.train_score_predictor_on_buffer",
                return_value=(jax.random.key(3), object(), {"curriculum/predictor_loss": jnp.array(0.0)}),
            ),
        ):
            train_one_iteration(
                rng=jax.random.key(0),
                agent_train_state=object(),
                reward_normalization_stats=reward_stats,
                intrinsic_states=(),
                curriculum_state=curriculum_state,
                env=_DummyResetOnlyEnv(),
                env_params=object(),
                intrinsic_modules=(),
                config=config,
            )

        return captured["scan_init_reward_stats"]

    def test_running_forward_return_reset_is_controlled_by_config(self):
        stats_without_reset = self._run_once_and_capture_scan_init_stats(reset_running_return=False)
        np.testing.assert_allclose(
            np.asarray(stats_without_reset.running_forward_return),
            np.asarray([[3.0, 7.0], [5.0, 11.0]], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            np.asarray(stats_without_reset.previous_done),
            np.asarray([[True, False], [True, False]], dtype=bool),
        )

        stats_with_reset = self._run_once_and_capture_scan_init_stats(reset_running_return=True)
        np.testing.assert_allclose(
            np.asarray(stats_with_reset.running_forward_return),
            np.zeros((2, 2), dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            np.asarray(stats_with_reset.previous_done),
            np.asarray([[False, False], [False, False]], dtype=bool),
        )

    def test_reward_side_intrinsic_states_stay_frozen_while_learner_states_advance(self):
        config = _build_test_config(reset_running_return=False)
        config.num_updates_per_batch = 2
        reward_stats = init_reward_normalization_stats(
            num_envs=config.num_envs_per_batch,
            num_reward_functions=config.num_reward_functions,
        )
        curriculum_state = CurriculumState(
            alpha_score_replay_buffer=object(),
            score_predictor_train_state=object(),
            lp_normalization_stats=object(),
            num_batches_seen=jnp.array(0, dtype=jnp.int32),
        )

        reward_state_calls: list[tuple[int, ...]] = []
        learner_state_calls: list[tuple[int, ...]] = []

        def fake_inner_update(
            runner_state,
            env,
            env_params,
            alpha_batch,
            reward_intrinsic_states,
            intrinsic_states_to_update,
            intrinsic_modules,
            config,
        ):  # noqa: ARG001
            del env, env_params, alpha_batch, intrinsic_modules
            reward_state_calls.append(reward_intrinsic_states)
            learner_state_calls.append(intrinsic_states_to_update)
            next_intrinsic_states = tuple(state + 1 for state in intrinsic_states_to_update)
            metrics = {
                "dummy_metric": jnp.array(1.0, dtype=jnp.float32),
                "intrinsic_modules/rnd/predictor_loss": jnp.array(2.0, dtype=jnp.float32),
            }
            lp_data = LpEstimationData(
                raw_rewards=jnp.zeros((1, config.num_envs_per_batch, config.num_reward_functions), dtype=jnp.float32),
                done_masks=jnp.zeros((1, config.num_envs_per_batch, config.num_reward_functions), dtype=jnp.bool_),
            )
            return runner_state, next_intrinsic_states, metrics, lp_data

        def fake_scan(fn, init, xs, length):  # noqa: ARG001
            carry = init
            metrics_per_update = []
            lp_data_per_update = []
            for _ in range(length):
                carry, (metrics, lp_data) = fn(carry, None)
                metrics_per_update.append(metrics)
                lp_data_per_update.append(lp_data)
            stacked_metrics = jax.tree_util.tree_map(
                lambda *values: jnp.stack(values, axis=0),
                *metrics_per_update,
            )
            stacked_lp_data = jax.tree_util.tree_map(
                lambda *values: jnp.stack(values, axis=0),
                *lp_data_per_update,
            )
            return carry, (stacked_metrics, stacked_lp_data)

        with (
            mock.patch(
                "crew.main_algo.main_loop.sample_alpha_batch",
                return_value=(
                    jax.random.key(1),
                    jnp.asarray([[0.6, 0.4], [0.2, 0.8]], dtype=jnp.float32),
                    {"curriculum/pred_score_mean": jnp.array(0.0, dtype=jnp.float32)},
                ),
            ),
            mock.patch("crew.main_algo.main_loop.jax.lax.scan", side_effect=fake_scan),
            mock.patch(
                "crew.main_algo.main_loop.collect_data_and_update_agent_and_intrinsic_modules",
                side_effect=fake_inner_update,
            ),
            mock.patch(
                "crew.main_algo.main_loop.update_lp_normalization_stats_from_data",
                return_value=SimpleNamespace(var=jnp.ones((config.num_reward_functions,), dtype=jnp.float32)),
            ),
            mock.patch(
                "crew.main_algo.main_loop.estimate_lp_per_reward_function",
                return_value=(
                    jnp.zeros((config.num_envs_per_batch, config.num_reward_functions), dtype=jnp.float32),
                    jnp.zeros((config.num_envs_per_batch,), dtype=jnp.int32),
                ),
            ),
            mock.patch(
                "crew.main_algo.main_loop.compute_scores",
                return_value=(
                    jnp.zeros((config.num_envs_per_batch,), dtype=jnp.float32),
                    {"curriculum/score_mean": jnp.array(0.0, dtype=jnp.float32)},
                ),
            ),
            mock.patch("crew.main_algo.main_loop.add_alpha_score_batch", return_value=object()),
            mock.patch(
                "crew.main_algo.main_loop.train_score_predictor_on_buffer",
                return_value=(jax.random.key(3), object(), {"curriculum/predictor_loss": jnp.array(0.0)}),
            ),
        ):
            _, _, _, intrinsic_states, _, _ = train_one_iteration(
                rng=jax.random.key(0),
                agent_train_state=object(),
                reward_normalization_stats=reward_stats,
                intrinsic_states=(10, 20),
                curriculum_state=curriculum_state,
                env=_DummyResetOnlyEnv(),
                env_params=object(),
                intrinsic_modules=(),
                config=config,
            )

        self.assertEqual(reward_state_calls, [(10, 20), (10, 20)])
        self.assertEqual(learner_state_calls, [(10, 20), (11, 21)])
        self.assertEqual(intrinsic_states, (12, 22))


if __name__ == "__main__":
    unittest.main()
