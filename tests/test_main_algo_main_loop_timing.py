import unittest
from types import SimpleNamespace
from unittest import mock

import jax.numpy as jnp

from crew.main_algo.main_loop import full_training


def _fake_train_one_iteration(
    rng,
    agent_train_state,
    reward_normalization_stats,
    intrinsic_states,
    curriculum_state,
    env,
    env_params,
    intrinsic_modules,
    config,
):
    metrics = {"train/loss": jnp.asarray(1.0, dtype=jnp.float32)}
    return (
        rng + 1,
        agent_train_state,
        reward_normalization_stats,
        intrinsic_states,
        curriculum_state,
        metrics,
    )


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
    eval_metrics = {
        "eval/returns": jnp.asarray([[[1.0]]], dtype=jnp.float32),
        "eval/lengths": jnp.asarray([[[2]]], dtype=jnp.int32),
        "eval/achievements": jnp.asarray([[[[True]]]], dtype=jnp.bool_),
    }
    return rng + 1, eval_metrics


def _build_test_config(is_timing_run: bool) -> SimpleNamespace:
    return SimpleNamespace(
        num_envs_per_batch=2,
        num_steps_per_env=4,
        num_batches_of_envs=1,
        eval_every_n_batches=1,
        eval_num_envs=1,
        eval_num_episodes=1,
        evaluation_alphas_array=jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
        video_num_episodes=0,
        is_timing_run=is_timing_run,
    )


class TestMainAlgoMainLoopTiming(unittest.TestCase):
    def test_non_timing_mode_does_not_block_or_print_timing(self):
        config = _build_test_config(is_timing_run=False)

        with (
            mock.patch("crew.main_algo.main_loop.jax.jit", side_effect=lambda fn: fn),
            mock.patch(
                "crew.main_algo.main_loop.jax.block_until_ready",
                side_effect=lambda x: x,
            ) as block_until_ready_mock,
            mock.patch("crew.main_algo.main_loop.train_one_iteration", side_effect=_fake_train_one_iteration),
            mock.patch(
                "crew.main_algo.main_loop.evaluate_policy_on_alphas", side_effect=_fake_evaluate_policy_on_alphas
            ),
            mock.patch("crew.main_algo.main_loop.infer_achievement_names", return_value=("Achievements/a",)),
            mock.patch("crew.main_algo.main_loop.init_wandb_run", return_value=None),
            mock.patch("crew.main_algo.main_loop.finish_wandb_run"),
            mock.patch("crew.main_algo.main_loop.log_outer_batch_to_wandb"),
            mock.patch("crew.main_algo.main_loop.time.perf_counter", side_effect=[10.0, 12.0]),
            mock.patch("builtins.print") as print_mock,
        ):
            output = full_training(
                rng=jnp.asarray(0, dtype=jnp.int32),
                agent_train_state=object(),
                reward_normalization_stats=object(),
                intrinsic_states=object(),
                curriculum_state=object(),
                env=object(),
                env_params=object(),
                intrinsic_modules=(),
                config=config,
            )

        self.assertEqual(block_until_ready_mock.call_count, 0)
        self.assertEqual(print_mock.call_count, 0)
        self.assertAlmostEqual(
            float(output["metrics"]["time/cumulative_wall_clock_sec"][0]),
            2.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(output["metrics"]["time/env_steps_per_sec"][0]),
            4.0,
            places=6,
        )

    def test_timing_mode_blocks_and_prints_stage_times(self):
        config = _build_test_config(is_timing_run=True)

        with (
            mock.patch("crew.main_algo.main_loop.jax.jit", side_effect=lambda fn: fn),
            mock.patch(
                "crew.main_algo.main_loop.jax.block_until_ready",
                side_effect=lambda x: x,
            ) as block_until_ready_mock,
            mock.patch("crew.main_algo.main_loop.train_one_iteration", side_effect=_fake_train_one_iteration),
            mock.patch(
                "crew.main_algo.main_loop.evaluate_policy_on_alphas", side_effect=_fake_evaluate_policy_on_alphas
            ),
            mock.patch("crew.main_algo.main_loop.infer_achievement_names", return_value=("Achievements/a",)),
            mock.patch("crew.main_algo.main_loop.init_wandb_run", return_value=None),
            mock.patch("crew.main_algo.main_loop.finish_wandb_run"),
            mock.patch("crew.main_algo.main_loop.log_outer_batch_to_wandb"),
            mock.patch(
                "crew.main_algo.main_loop.time.perf_counter",
                side_effect=[10.0, 12.0, 20.0, 23.0, 30.0, 31.0, 40.0, 41.0],
            ),
            mock.patch("builtins.print") as print_mock,
        ):
            full_training(
                rng=jnp.asarray(0, dtype=jnp.int32),
                agent_train_state=object(),
                reward_normalization_stats=object(),
                intrinsic_states=object(),
                curriculum_state=object(),
                env=object(),
                env_params=object(),
                intrinsic_modules=(),
                config=config,
            )

        self.assertEqual(block_until_ready_mock.call_count, 2)
        printed_messages = [" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list]
        self.assertEqual(len(printed_messages), 3)
        self.assertTrue(any("train_one_iteration=" in message for message in printed_messages))
        self.assertTrue(any("evaluation=" in message for message in printed_messages))
        self.assertTrue(any("logging=" in message for message in printed_messages))


if __name__ == "__main__":
    unittest.main()
