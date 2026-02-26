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
    """

    def __init__(
        self,
        env,
        modules: list[IntrinsicModule] | None = None,
        alpha: list[float] | None = None,
    ):
        self._env = env
        self._modules: list[IntrinsicModule] = modules if modules is not None else []
        n = len(self._modules)

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

        total_reward = self._alpha[0] * sparse_extrinsic
        new_intrinsic_states: list[Any] = []

        for i, (module, module_state) in enumerate(
            zip(self._modules, state.intrinsic_states)
        ):
            key, rng = jax.random.split(key)
            raw_intrinsic = module.compute_rewards(
                rng, module_state, transition, config=None
            )
            # Zero out intrinsic reward at episode boundaries.
            intrinsic_reward = jax.lax.select(
                done, jnp.float32(0.0), raw_intrinsic.astype(jnp.float32)
            )
            total_reward = total_reward + self._alpha[i + 1] * intrinsic_reward
            info[f"intrinsic_reward_{module.name}"] = intrinsic_reward
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
            new_intrinsic_states.append(new_module_state)

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


def step_envs_random(runner_state, _, env, env_params, config):
    rng, state = runner_state
    rng, step_rng, action_rng = jax.random.split(rng, 3)
    step_rngs = jax.random.split(step_rng, config.num_parallel_envs)

    num_actions = env.action_space(env_params).n
    actions = jax.random.randint(
        action_rng, (config.num_parallel_envs,), minval=0, maxval=num_actions
    )

    # vmap seamlessly handles SparseWrapperState because it's a registered PyTree
    obs, next_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
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
