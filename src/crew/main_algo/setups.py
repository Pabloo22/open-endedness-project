from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from craftax.craftax_env import make_craftax_env_from_name
from flax.training.train_state import TrainState

from crew.main_algo.actor_critic import ActorCriticTransformer
from crew.main_algo.config import TrainConfig
from crew.main_algo.curriculum.lp_normalization import init_lp_normalization_stats
from crew.main_algo.curriculum.replay_buffer import init_alpha_score_replay_buffer
from crew.main_algo.curriculum.score_predictor import init_score_predictor_train_state
from crew.main_algo.intrinsic_modules.api import IntrinsicModule
from crew.main_algo.intrinsic_modules.registry import get_selected_intrinsic_modules
from crew.main_algo.reward_normalization import init_reward_normalization_stats
from crew.main_algo.types import (
    CurriculumState,
    IntrinsicStates,
    RewardNormalizationStats,
)
from crew.main_algo.wrappers import (
    FixedResetKeyEnvWrapper,
    OptimisticResetVecEnvWrapper,
    SparseCraftaxWrapper,
)


def _resolve_optimistic_reset_ratio(num_envs: int, ratio_limit: int) -> int:
    upper_bound = min(num_envs, ratio_limit)
    for candidate in range(upper_bound, 0, -1):
        if num_envs % candidate == 0:
            return candidate
    return 1


def setup_actor_critic_train_state(
    rng: jax.Array,
    env: Any,
    env_params: Any,
    config: TrainConfig,
    lr_schedule: Callable | None = None,
) -> tuple[jax.Array, TrainState]:
    """Initialize alpha-conditioned actor-critic train state."""
    num_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape

    network = ActorCriticTransformer(
        num_actions=num_actions,
        num_reward_functions=config.num_reward_functions,
        # observation encoder
        env_id=config.env_id,
        encoder_mode=config.encoder_mode,
        obs_emb_dim=config.obs_emb_dim,
        # transformer
        hidden_dim=config.transformer_hidden_states_dim,
        num_attn_heads=config.num_attn_heads,
        qkv_features=config.qkv_features,
        num_layers_in_transformer=config.num_transformer_blocks,
        gating=config.gating,
        gating_bias=config.gating_bias,
        # actor and critic heads
        head_activation=config.head_activation,
        head_hidden_dim=config.head_hidden_dim,
        # alpha injection
        inject_alpha_at_trunk=config.inject_alpha_at_trunk,
        inject_alpha_at_actor_head=config.inject_alpha_at_actor_head,
        inject_alpha_at_critic_head=config.inject_alpha_at_critic_head,
    )

    init_obs = jnp.zeros((2, 1, *obs_shape), dtype=jnp.float32)
    init_alpha = jnp.zeros((2, config.num_reward_functions), dtype=jnp.float32)
    init_memory = jnp.zeros(
        (
            2,
            config.past_context_length,
            config.num_transformer_blocks,
            config.transformer_hidden_states_dim,
        ),
        dtype=jnp.float32,
    )
    init_mask = jnp.zeros(
        (2, config.num_attn_heads, 1, config.past_context_length + 1),
        dtype=jnp.bool_,
    )

    rng, init_rng = jax.random.split(rng)
    network_params = network.init(
        init_rng,
        init_memory,
        init_obs,
        init_mask,
        init_alpha,
    )

    if lr_schedule is not None:
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=lr_schedule,
                eps=config.adam_eps,
            ),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=config.lr,
                eps=config.adam_eps,
            ),
        )

    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
    return rng, train_state


def setup_intrinsic_module_states(
    rng: jax.Array,
    intrinsic_modules: tuple[IntrinsicModule, ...],
    env: Any,
    env_params: Any,
    config: TrainConfig,
) -> tuple[jax.Array, IntrinsicStates]:
    """Initialize the selected intrinsic module states in module-order."""
    obs_shape = env.observation_space(env_params).shape
    intrinsic_states = []
    for module in intrinsic_modules:
        rng, module_rng = jax.random.split(rng)
        intrinsic_states.append(module.init_state(module_rng, obs_shape, config))
    return rng, tuple(intrinsic_states)


def initialize_curriculum_state(
    rng: jax.Array,
    config: TrainConfig,
) -> tuple[jax.Array, CurriculumState]:
    """Create initial curriculum state (score buffer and related metadata)."""
    alpha_score_replay_buffer = init_alpha_score_replay_buffer(config=config)
    rng, predictor_init_rng = jax.random.split(rng)
    score_predictor_train_state = init_score_predictor_train_state(
        rng=predictor_init_rng,
        config=config,
    )
    lp_normalization_stats = init_lp_normalization_stats(
        num_reward_functions=config.num_reward_functions,
    )
    curriculum_state = CurriculumState(
        alpha_score_replay_buffer=alpha_score_replay_buffer,
        score_predictor_train_state=score_predictor_train_state,
        lp_normalization_stats=lp_normalization_stats,
        num_batches_seen=jnp.array(0, dtype=jnp.int32),
    )
    return rng, curriculum_state


def set_up_for_training(
    config: TrainConfig,
) -> tuple[
    jax.Array,
    Any,
    Any,
    TrainState,
    RewardNormalizationStats,
    tuple[IntrinsicModule, ...],
    IntrinsicStates,
    CurriculumState,
]:
    """Set up env, agent, intrinsic modules, and curriculum."""
    rng = jax.random.key(config.train_seed)

    # Environment setup.
    base_env = make_craftax_env_from_name(config.env_id, auto_reset=False)
    env_params = base_env.default_params
    if config.episode_max_steps is not None:
        env_params = env_params.replace(max_timesteps=config.episode_max_steps)
    base_env = SparseCraftaxWrapper(
        base_env,
        blocked_achievement_ids=config.achievement_ids_to_block,
        remove_health_reward=config.remove_health_reward,
    )
    if not config.procedural_generation:
        base_env = FixedResetKeyEnvWrapper(
            base_env,
            fixed_reset_seed=config.fixed_reset_seed,
        )
    reset_ratio = _resolve_optimistic_reset_ratio(
        num_envs=config.num_envs_per_batch,
        ratio_limit=config.optimistic_reset_ratio_limit,
    )
    env = OptimisticResetVecEnvWrapper(
        base_env,
        num_envs=config.num_envs_per_batch,
        reset_ratio=reset_ratio,
    )

    # Agent setup.
    if config.anneal_lr:

        def linear_schedule(count):
            total_param_updates_per_batch = config.num_minibatches * config.update_epochs * config.num_updates_per_batch
            frac = 1.0 - (count // total_param_updates_per_batch) / config.num_batches_of_envs
            return config.lr * frac

        rng, agent_train_state = setup_actor_critic_train_state(
            rng=rng,
            env=env,
            env_params=env_params,
            config=config,
            lr_schedule=linear_schedule,
        )
    else:
        rng, agent_train_state = setup_actor_critic_train_state(
            rng=rng,
            env=env,
            env_params=env_params,
            config=config,
        )

    # Intrinsic module setup.
    intrinsic_modules = get_selected_intrinsic_modules(config.selected_intrinsic_modules)
    rng, intrinsic_states = setup_intrinsic_module_states(
        rng=rng,
        intrinsic_modules=intrinsic_modules,
        env=env,
        env_params=env_params,
        config=config,
    )

    reward_normalization_stats = init_reward_normalization_stats(
        num_envs=config.num_envs_per_batch,
        num_reward_functions=config.num_reward_functions,
    )
    rng, curriculum_state = initialize_curriculum_state(
        rng=rng,
        config=config,
    )
    return (
        rng,
        env,
        env_params,
        agent_train_state,
        reward_normalization_stats,
        intrinsic_modules,
        intrinsic_states,
        curriculum_state,
    )
