"""Alpha sampling utilities for curriculum-guided training."""

from typing import Any

import jax
import jax.numpy as jnp

from crew.main_algo.types import CurriculumState


def uniform_sampling(
    rng: jax.Array,
    num_envs_per_batch: int,
    num_reward_functions: int,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Sample alphas uniformly over simplex."""
    rng, alpha_rng = jax.random.split(rng)
    alpha_batch = jax.random.dirichlet(
        alpha_rng,
        alpha=jnp.ones((num_reward_functions,), dtype=jnp.float32),
        shape=(num_envs_per_batch,),
    )
    diagnostics = {
        "curriculum/sampler_used_predictor": jnp.array(0.0, dtype=jnp.float32),
        "curriculum/sampler_candidate_pool_size": jnp.array(0.0, dtype=jnp.float32),
        "curriculum/sampler_pred_score_mean": jnp.array(0.0, dtype=jnp.float32),
        "curriculum/sampler_pred_score_std": jnp.array(0.0, dtype=jnp.float32),
    }
    return rng, alpha_batch, diagnostics


def predictor_based_importance_sampling(
    rng: jax.Array,
    curriculum_state: CurriculumState,
    config: Any,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Sample alphas by reweighting Dirichlet candidates using predictor outputs.

    Shapes:
    - output alpha_batch: [B, R]
    """
    num_candidates = config.curriculum.importance_num_candidates_multiplier * config.num_envs_per_batch
    rng, candidates_rng, resample_rng = jax.random.split(rng, 3)
    alpha_candidates = jax.random.dirichlet(
        candidates_rng,
        alpha=jnp.ones((config.num_reward_functions,), dtype=jnp.float32),
        shape=(num_candidates,),
    )  # [C, R]

    predicted_scores = curriculum_state.score_predictor_train_state.apply_fn(
        curriculum_state.score_predictor_train_state.params,
        alpha_candidates,
    )  # [C]
    sampling_weights = jnp.clip(predicted_scores, min=0.0) + jnp.array(config.curriculum.sampling_weights_eps, dtype=predicted_scores.dtype)  # [C]
    sampling_probabilities = sampling_weights / jnp.sum(sampling_weights)  # [C]

    sampled_candidate_indices = jax.random.choice(
        resample_rng,
        a=num_candidates,
        shape=(config.num_envs_per_batch,),
        replace=True,
        p=sampling_probabilities,
    )  # [B]
    alpha_batch = jnp.take(alpha_candidates, sampled_candidate_indices, axis=0)  # [B, R]

    diagnostics = {
        "curriculum/sampler_used_predictor": jnp.array(1.0, dtype=jnp.float32),
        "curriculum/sampler_candidate_pool_size": jnp.asarray(num_candidates, dtype=jnp.float32),
        "curriculum/sampler_pred_score_mean": jnp.mean(predicted_scores),
        "curriculum/sampler_pred_score_std": jnp.std(predicted_scores),
    }
    return rng, alpha_batch, diagnostics


def sample_alpha_batch(
    rng: jax.Array,
    curriculum_state: CurriculumState,
    config: Any,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Sample alphas from either warmup uniform or predictor-guided path.

    Shapes:
    - output alpha_batch: [B, R]
    """
    min_batches_for_predictor_sampling = jnp.asarray(
        config.curriculum.min_batches_for_predictor_sampling,
        dtype=curriculum_state.num_batches_seen.dtype,
    )
    use_predictor_sampling = curriculum_state.num_batches_seen >= min_batches_for_predictor_sampling

    def predictor_branch(branch_rng: jax.Array) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        return predictor_based_importance_sampling(
            rng=branch_rng,
            curriculum_state=curriculum_state,
            config=config,
        )

    def uniform_branch(branch_rng: jax.Array) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        return uniform_sampling(
            rng=branch_rng,
            num_envs_per_batch=config.num_envs_per_batch,
            num_reward_functions=config.num_reward_functions,
        )

    return jax.lax.cond(
        use_predictor_sampling,
        predictor_branch,
        uniform_branch,
        rng,
    )
