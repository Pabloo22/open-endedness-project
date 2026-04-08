import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp

from curemix.main_algo.baseline_main_loop import full_training_baseline


class _DummyResetOnlyEnv:
    def reset(self, key, params):
        del key, params
        return jnp.asarray([0.0, 0.0], dtype=jnp.float32), jnp.asarray(0, dtype=jnp.int32)


def _fake_train_one_iteration_baseline(
    runner_state,
    intrinsic_states,
    alpha_batch,
    env,
    env_params,
    intrinsic_modules,
    config,
):
    del alpha_batch, env, env_params, intrinsic_modules, config
    metrics = {"train/loss": jnp.asarray(1.0, dtype=jnp.float32)}
    return runner_state, intrinsic_states, metrics


def _fake_evaluate_policy_on_alphas(
    rng,
    agent_train_state,
    env,
    env_params,
    evaluation_alphas,
    num_eval_envs,
    num_eval_episodes,
    achievement_names,
    config,
):
    del (
        agent_train_state,
        env,
        env_params,
        evaluation_alphas,
        num_eval_envs,
        num_eval_episodes,
        achievement_names,
        config,
    )
    eval_metrics = {
        "eval/returns": jnp.asarray([[[1.0]]], dtype=jnp.float32),
        "eval/lengths": jnp.asarray([[[2]]], dtype=jnp.int32),
        "eval/achievements": jnp.asarray([[[[True]]]], dtype=jnp.bool_),
    }
    return rng, eval_metrics


def _build_test_config(
    is_timing_run: bool,
    num_batches_of_envs: int = 1,
    num_updates_per_batch: int = 2,
    eval_every_n_batches: int = 1,
):
    return SimpleNamespace(
        num_envs_per_batch=2,
        num_steps_per_update=4,
        num_batches_of_envs=num_batches_of_envs,
        num_updates_per_batch=num_updates_per_batch,
        eval_every_n_batches=eval_every_n_batches,
        eval_num_envs=1,
        eval_num_episodes=1,
        evaluation_alphas_array=jnp.asarray([[1.0]], dtype=jnp.float32),
        num_reward_functions=1,
        baseline_fixed_training_alpha=(1.0,),
        is_timing_run=is_timing_run,
        past_context_length=4,
        num_transformer_blocks=1,
        transformer_hidden_states_dim=8,
        num_attn_heads=2,
    )


class TestMainAlgoBaselineMainLoopTiming(unittest.TestCase):
    def test_non_timing_mode_does_not_block_or_print_timing(self):
        config = _build_test_config(is_timing_run=False)
        env = _DummyResetOnlyEnv()

        with (
            mock.patch("crew.main_algo.baseline_main_loop.jax.jit", side_effect=lambda fn: fn),
            mock.patch(
                "crew.main_algo.baseline_main_loop.jax.block_until_ready",
                side_effect=lambda x: x,
            ) as block_until_ready_mock,
            mock.patch(
                "crew.main_algo.baseline_main_loop.train_one_iteration_baseline",
                side_effect=_fake_train_one_iteration_baseline,
            ),
            mock.patch(
                "crew.main_algo.baseline_main_loop.evaluate_policy_on_alphas",
                side_effect=_fake_evaluate_policy_on_alphas,
            ) as evaluate_mock,
            mock.patch("crew.main_algo.baseline_main_loop.infer_achievement_names", return_value=("Achievements/a",)),
            mock.patch("crew.main_algo.baseline_main_loop.init_wandb_run", return_value=None),
            mock.patch("crew.main_algo.baseline_main_loop.finish_wandb_run"),
            mock.patch("crew.main_algo.baseline_main_loop.log_outer_batch_to_wandb"),
            mock.patch("crew.main_algo.baseline_main_loop.time.perf_counter", side_effect=[10.0, 11.0, 12.0]),
            mock.patch("builtins.print") as print_mock,
        ):
            output = full_training_baseline(
                rng=jax.random.key(0),
                agent_train_state=object(),
                reward_normalization_stats=object(),
                intrinsic_states=(),
                env=env,
                env_params=object(),
                intrinsic_modules=(),
                config=config,
            )

        self.assertEqual(block_until_ready_mock.call_count, 0)
        self.assertEqual(print_mock.call_count, 0)
        self.assertEqual(evaluate_mock.call_count, 1)
        self.assertEqual(output["metrics"]["run/total_env_steps"].shape, (2,))

    def test_timing_mode_blocks_and_prints_stage_times(self):
        config = _build_test_config(is_timing_run=True)
        env = _DummyResetOnlyEnv()

        with (
            mock.patch("crew.main_algo.baseline_main_loop.jax.jit", side_effect=lambda fn: fn),
            mock.patch(
                "crew.main_algo.baseline_main_loop.jax.block_until_ready",
                side_effect=lambda x: x,
            ) as block_until_ready_mock,
            mock.patch(
                "crew.main_algo.baseline_main_loop.train_one_iteration_baseline",
                side_effect=_fake_train_one_iteration_baseline,
            ),
            mock.patch(
                "crew.main_algo.baseline_main_loop.evaluate_policy_on_alphas",
                side_effect=_fake_evaluate_policy_on_alphas,
            ),
            mock.patch("crew.main_algo.baseline_main_loop.infer_achievement_names", return_value=("Achievements/a",)),
            mock.patch("crew.main_algo.baseline_main_loop.init_wandb_run", return_value=None),
            mock.patch("crew.main_algo.baseline_main_loop.finish_wandb_run"),
            mock.patch("crew.main_algo.baseline_main_loop.log_outer_batch_to_wandb"),
            mock.patch(
                "crew.main_algo.baseline_main_loop.time.perf_counter",
                side_effect=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
            ),
            mock.patch("builtins.print") as print_mock,
        ):
            full_training_baseline(
                rng=jax.random.key(0),
                agent_train_state=object(),
                reward_normalization_stats=object(),
                intrinsic_states=(),
                env=env,
                env_params=object(),
                intrinsic_modules=(),
                config=config,
            )

        self.assertEqual(block_until_ready_mock.call_count, 3)
        printed_messages = [" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list]
        self.assertEqual(len(printed_messages), 5)
        self.assertTrue(any("train_one_iteration=" in message for message in printed_messages))
        self.assertTrue(any("evaluation=" in message for message in printed_messages))
        self.assertTrue(any("logging=" in message for message in printed_messages))

    def test_eval_cadence_matches_curriculum_env_step_frequency(self):
        config = _build_test_config(
            is_timing_run=False,
            num_batches_of_envs=4,
            num_updates_per_batch=3,
            eval_every_n_batches=2,
        )
        env = _DummyResetOnlyEnv()

        with (
            mock.patch("crew.main_algo.baseline_main_loop.jax.jit", side_effect=lambda fn: fn),
            mock.patch(
                "crew.main_algo.baseline_main_loop.train_one_iteration_baseline",
                side_effect=_fake_train_one_iteration_baseline,
            ),
            mock.patch(
                "crew.main_algo.baseline_main_loop.evaluate_policy_on_alphas",
                side_effect=_fake_evaluate_policy_on_alphas,
            ) as evaluate_mock,
            mock.patch("crew.main_algo.baseline_main_loop.infer_achievement_names", return_value=("Achievements/a",)),
            mock.patch("crew.main_algo.baseline_main_loop.init_wandb_run", return_value=None),
            mock.patch("crew.main_algo.baseline_main_loop.finish_wandb_run"),
            mock.patch("crew.main_algo.baseline_main_loop.log_outer_batch_to_wandb"),
            mock.patch(
                "crew.main_algo.baseline_main_loop.time.perf_counter",
                side_effect=[float(x) for x in range(100)],
            ),
        ):
            full_training_baseline(
                rng=jax.random.key(0),
                agent_train_state=object(),
                reward_normalization_stats=object(),
                intrinsic_states=(),
                env=env,
                env_params=object(),
                intrinsic_modules=(),
                config=config,
            )

        # With U=3 and E=2, eval updates are 3 and 9 within total updates=12.
        self.assertEqual(evaluate_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
