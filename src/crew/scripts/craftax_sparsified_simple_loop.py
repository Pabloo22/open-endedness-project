from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial

from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP, Achievement
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
from crew.main_algo.intrinsic_modules.api import IntrinsicModule
from crew.main_algo.types import IntrinsicModulesUpdateData, TransitionDataBase
from crew.shared_code.wrappers import AutoResetEnvWrapper

# --- 1. Custom Wrapper Definitions ---


@struct.dataclass
class SparseWrapperState:
    env_state: Any
    episode_return: jnp.ndarray
    prev_obs: jnp.ndarray
    intrinsic_states: tuple  # one IntrinsicModuleState per module (empty tuple if none)


class SparseIntrinsicCraftaxWrapper:
    """Craftax wrapper that sparsifies extrinsic rewards and combines them with
    an arbitrary number of intrinsic reward modules.

    Args:
        env: The base environment (already wrapped with AutoResetEnvWrapper).
        modules: List of intrinsic reward modules conforming to IntrinsicModule.
            Pass an empty list (default) to disable all intrinsic rewards.
        alpha: Weight vector of length ``len(modules) + 1``.
            ``alpha[0]`` scales the (sparse) extrinsic reward;
            ``alpha[1:]`` scale each intrinsic module in order.
            Defaults to ``[1.0]`` when *modules* is empty, or equal weights
            ``1 / (n + 1)`` for each component when *n* modules are provided.
        blocked_achievement_ids: Optional list of ``Achievement`` integer values
            whose reward contribution is removed before accumulation.  All other
            reward signal (health reward, non-blocked achievements) is unchanged.
            Defaults to ``None`` (no achievements blocked).
    """

    def __init__(
        self,
        env: CraftaxSymbolicEnv,
        modules: list[IntrinsicModule] | None = None,
        alpha: list[float] | None = None,
        blocked_achievement_ids: list[int] | None = None,
    ):
        self._env = env
        self._modules: list[IntrinsicModule] = modules if modules is not None else []
        n = len(self._modules)

        num_achievements = len(Achievement)
        blocked_ids = (
            blocked_achievement_ids if blocked_achievement_ids is not None else []
        )
        blocked_mask = jnp.zeros(num_achievements, dtype=jnp.bool_)
        if blocked_ids:
            blocked_mask = blocked_mask.at[jnp.array(blocked_ids, dtype=jnp.int32)].set(
                True
            )
        self._blocked_mask: jnp.ndarray = blocked_mask
        # Pre-compute the reward that each blocked achievement would contribute.
        self._blocked_reward_map: jnp.ndarray = (
            jnp.array(ACHIEVEMENT_REWARD_MAP, dtype=jnp.float32) * blocked_mask
        )

        if alpha is None:
            if n == 0:
                self._alpha = jnp.array([1.0], dtype=jnp.float32)
            else:
                self._alpha = jnp.ones(n + 1, dtype=jnp.float32) / (n + 1)
        else:
            if len(alpha) != n + 1:
                raise ValueError(
                    f"len(alpha) must equal len(modules) + 1, "
                    f"got {len(alpha)} and {n + 1}."
                )
            self._alpha = jnp.array(alpha, dtype=jnp.float32)

    @property
    def default_params(self):
        return self._env.default_params

    def action_space(self, params):
        return self._env.action_space(params)

    def observation_space(self, params):
        return self._env.observation_space(params)

    def reset(self, key, params=None):
        if params is None:
            params = self.default_params

        obs, env_state = self._env.reset(key, params)

        intrinsic_states = []
        for module in self._modules:
            key, rng = jax.random.split(key)
            intrinsic_states.append(module.init_state(rng, obs.shape, config=None))

        wrapper_state = SparseWrapperState(
            env_state=env_state,
            episode_return=jnp.float32(0.0),
            prev_obs=obs,
            intrinsic_states=tuple(intrinsic_states),
        )
        return obs, wrapper_state

    def step(self, key, state: SparseWrapperState, action, params=None):
        if params is None:
            params = self.default_params

        next_obs, next_env_state, dense_reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        # Reconstruct correct achievements and remove blocked reward.
        post_step_achievements = self._get_post_step_achievements(info, next_env_state)
        newly_unlocked = post_step_achievements & ~state.env_state.achievements
        dense_reward = self._adjust_dense_reward(dense_reward, newly_unlocked)

        new_episode_return = state.episode_return + dense_reward

        # Sparsify: output total accumulated reward ONLY when the episode ends.
        sparse_extrinsic = jax.lax.select(done, new_episode_return, jnp.float32(0.0))

        # Build a single-step transition for intrinsic modules.
        transition = TransitionDataBase(
            obs=state.prev_obs,
            next_obs=next_obs,
            action=action,
            done=done,
            reward=dense_reward,
            value=jnp.zeros((), dtype=jnp.float32),  # unused by most modules
            log_prob=jnp.zeros((), dtype=jnp.float32),  # unused by most modules
        )

        key, total_intrinsic, new_intrinsic_states = self._apply_all_intrinsic_modules(
            key, state.intrinsic_states, transition, done, info
        )
        total_reward = self._alpha[0] * sparse_extrinsic + total_intrinsic

        # Reset accumulated return at episode boundaries.
        next_episode_return = jax.lax.select(done, jnp.float32(0.0), new_episode_return)

        next_wrapper_state = SparseWrapperState(
            env_state=next_env_state,
            episode_return=next_episode_return,
            prev_obs=next_obs,
            intrinsic_states=tuple(new_intrinsic_states),
        )

        info["dense_reward"] = dense_reward
        info["sparse_extrinsic_reward"] = sparse_extrinsic

        return next_obs, next_wrapper_state, total_reward, done, info

    def _get_post_step_achievements(self, info, next_env_state) -> jnp.ndarray:
        """Reconstruct post-step achievements before auto-reset clobbers them.

        ``AutoResetEnvWrapper`` zeroes ``env_state.achievements`` on episode
        end, but ``log_achievements_to_info`` preserves them in
        ``info["Achievements/..."] * 100``.  Taking the OR of both sources
        gives the correct achievements regardless of whether the episode just
        ended.
        """
        info_achievements = (
            jnp.stack(
                [
                    info[f"Achievements/{achievement.name.lower()}"]
                    for achievement in Achievement
                ]
            )
            / 100.0
        ).astype(jnp.bool_)
        return info_achievements | next_env_state.achievements

    def _adjust_dense_reward(
        self, dense_reward: jnp.ndarray, newly_unlocked: jnp.ndarray
    ) -> jnp.ndarray:
        """Subtract the reward contribution of blocked achievements."""
        blocked_reward_this_step = (
            newly_unlocked.astype(jnp.float32) * self._blocked_reward_map
        ).sum()
        return dense_reward - blocked_reward_this_step

    def _apply_intrinsic_module(
        self,
        key: jnp.ndarray,
        module: IntrinsicModule,
        module_state: Any,
        transition: TransitionDataBase,
        done: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, Any]:
        """Run a single intrinsic module: compute reward then update its state.

        Returns:
            key: Updated PRNG key.
            intrinsic_reward: Scalar reward (zeroed at episode boundaries).
            new_module_state: Updated module state after the online update.
        """
        key, rng_compute = jax.random.split(key)
        raw_intrinsic = module.compute_rewards(
            rng_compute, module_state, transition, config=None
        )
        # Zero out intrinsic reward at episode boundaries.
        intrinsic_reward = jax.lax.select(
            done, jnp.float32(0.0), raw_intrinsic.astype(jnp.float32)
        )

        key, rng_update = jax.random.split(key)
        new_module_state, _ = module.update(
            rng_update,
            module_state,
            IntrinsicModulesUpdateData(
                obs=transition.obs,
                next_obs=transition.next_obs,
                action=transition.action,
                done=transition.done,
            ),
            config=None,
        )
        return key, intrinsic_reward, new_module_state

    def _apply_all_intrinsic_modules(
        self,
        key: jnp.ndarray,
        intrinsic_states: tuple,
        transition: TransitionDataBase,
        done: jnp.ndarray,
        info: dict,
    ) -> tuple[jnp.ndarray, jnp.ndarray, list[Any]]:
        """Apply every intrinsic module and return the weighted reward sum.

        Returns:
            key: Updated PRNG key.
            total_intrinsic: Weighted sum of all intrinsic rewards (scalar).
            new_intrinsic_states: List of updated module states.
        """
        total_intrinsic = jnp.float32(0.0)
        new_intrinsic_states: list[Any] = []

        for i, (module, module_state) in enumerate(
            zip(self._modules, intrinsic_states)
        ):
            key, intrinsic_reward, new_module_state = self._apply_intrinsic_module(
                key, module, module_state, transition, done
            )
            total_intrinsic = total_intrinsic + self._alpha[i + 1] * intrinsic_reward
            info[f"intrinsic_reward_{module.name}"] = intrinsic_reward
            new_intrinsic_states.append(new_module_state)

        return key, total_intrinsic, new_intrinsic_states


# --- 2. Configuration ---


@dataclass
class Config:
    experiment: str = "random"
    seed: int = 0
    total_timesteps: int = 1_000_000
    num_parallel_envs: int = 128
    num_steps_per_env: int = 1024

    def __post_init__(self):
        self.num_batches_of_envs = math.ceil(
            self.total_timesteps / (self.num_parallel_envs * self.num_steps_per_env)
        )


# --- 3. JAX Scanning / Vmapping ---


def step_envs_random(
    runner_state, _, env: CraftaxSymbolicEnv, env_params, config: Config
):
    rng, state = runner_state
    rng, step_rng, action_rng = jax.random.split(rng, 3)
    step_rngs = jax.random.split(step_rng, config.num_parallel_envs)

    num_actions = env.action_space(env_params).n
    actions = jax.random.randint(
        action_rng, (config.num_parallel_envs,), minval=0, maxval=num_actions
    )

    # vmap seamlessly handles SparseWrapperState because it's a registered PyTree
    _, next_state, *_ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        step_rngs, state, actions, env_params
    )

    # Optional: You could accumulate `info["intrinsic_reward"]` here for logging

    return (rng, next_state), None


def collect_data_random(runner_state, env, env_params, config):
    runner_state, _ = jax.lax.scan(
        Partial(step_envs_random, env=env, env_params=env_params, config=config),
        runner_state,
        None,
        config.num_steps_per_env,
    )

    return runner_state


# --- 4. Main Experiment Runner ---


def speed_experiment_random_policy(config: Config):
    # Setup base env, auto-reset, and sparse wrapper (no intrinsic modules by default).
    env = CraftaxSymbolicEnv()
    env = AutoResetEnvWrapper(env)
    env = SparseIntrinsicCraftaxWrapper(env, modules=[], alpha=[1.0])
    env_params = env.default_params

    rng = jax.random.PRNGKey(config.seed)
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, config.num_parallel_envs)

    obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)
    runner_state = (rng, state)

    # Compile + warmup once.
    compile_t0 = time.perf_counter()
    jitted_collect_data_random = jax.jit(
        Partial(collect_data_random, env=env, env_params=env_params, config=config)
    )
    runner_state = jitted_collect_data_random(runner_state)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), runner_state)
    compile_time = time.perf_counter() - compile_t0

    run_t0 = time.perf_counter()
    for _ in range(config.num_batches_of_envs):
        runner_state = jitted_collect_data_random(runner_state)

    jax.tree_util.tree_map(lambda x: x.block_until_ready(), runner_state)
    run_time = time.perf_counter() - run_t0

    executed_timesteps = (
        config.num_batches_of_envs * config.num_parallel_envs * config.num_steps_per_env
    )

    print("=== Craftax (Sparse + Intrinsic) Random Policy Data Collection Speed ===")
    print(f"Device: {jax.default_backend()}")
    print(f"total_timesteps: {executed_timesteps}")
    print(f"num_parallel_envs: {config.num_parallel_envs}")
    print(f"num_steps_per_env: {config.num_steps_per_env}")
    print(f"num_batches_of_envs: {config.num_batches_of_envs}")
    print(f"compile_time_s: {compile_time:.3f}")
    print(f"run_time_s: {run_time:.3f}")
    print(f"steps_per_second: {executed_timesteps / run_time:,.1f}")
    print(f"mean_outer_loop_time_s: {run_time / config.num_batches_of_envs:.6f}")


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Craftax data collection speed benchmark."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["random"],
        default="random",
    )
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--num-parallel-envs", type=int, default=64)
    parser.add_argument("--num-steps-per-env", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    return Config(
        experiment=args.experiment,
        total_timesteps=args.total_timesteps,
        num_parallel_envs=args.num_parallel_envs,
        num_steps_per_env=args.num_steps_per_env,
        seed=args.seed,
    )


if __name__ == "__main__":
    config_ = _parse_args()
    if config_.experiment == "random":
        speed_experiment_random_policy(config_)
