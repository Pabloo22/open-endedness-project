from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
from jax.tree_util import Partial

from crew.shared_code.wrappers import AutoResetEnvWrapper


@dataclass
class Config:
    experiment: str = "random"
    seed: int = 0
    total_timesteps: int = 1_000_000
    num_parallel_envs: int = 128
    num_steps_per_env: int = 1024

    def __post_init__(self):
        self.num_batches_of_envs = math.ceil(self.total_timesteps / (self.num_parallel_envs * self.num_steps_per_env))


def step_envs_random(runner_state, _, env, env_params, config):
    rng, state = runner_state
    rng, step_rng, action_rng = jax.random.split(rng, 3)
    step_rngs = jax.random.split(step_rng, config.num_parallel_envs)

    num_actions = env.action_space(env_params).n
    actions = jax.random.randint(action_rng, (config.num_parallel_envs,), minval=0, maxval=num_actions)

    obs, next_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(step_rngs, state, actions, env_params)
    return (rng, next_state), None


def collect_data_random(runner_state, env, env_params, config):
    rng, _ = runner_state
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, config.num_parallel_envs)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)
    runner_state = (rng, state)

    runner_state, _ = jax.lax.scan(
        Partial(step_envs_random, env=env, env_params=env_params, config=config),
        runner_state,
        None,
        config.num_steps_per_env,
    )

    return runner_state


def speed_experiment_random_policy(config: Config):
    env = CraftaxSymbolicEnv()
    env = AutoResetEnvWrapper(env)
    env_params = env.default_params

    rng = jax.random.PRNGKey(config.seed)
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, config.num_parallel_envs)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)
    runner_state = (rng, state)

    # Compile + warmup once.
    compile_t0 = time.perf_counter()
    jitted_collect_data_random = jax.jit(Partial(collect_data_random, env=env, env_params=env_params, config=config))
    runner_state = jitted_collect_data_random(runner_state)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), runner_state)
    compile_time = time.perf_counter() - compile_t0

    run_t0 = time.perf_counter()
    for _ in range(config.num_batches_of_envs):
        runner_state = jitted_collect_data_random(runner_state)

    jax.tree_util.tree_map(lambda x: x.block_until_ready(), runner_state)
    run_time = time.perf_counter() - run_t0

    executed_timesteps = config.num_batches_of_envs * config.num_parallel_envs * config.num_steps_per_env
    print("=== Craftax Random Policy Data Collection Speed ===")
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
    parser = argparse.ArgumentParser(description="Craftax data collection speed benchmark.")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=("random"),
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
    config = _parse_args()
    if config.experiment == "random":
        speed_experiment_random_policy(config)
