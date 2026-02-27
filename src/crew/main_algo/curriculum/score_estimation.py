import jax
import jax.numpy as jnp


def compute_scores(
    alpha_batch: jax.Array,
    lp_per_reward_function: jax.Array,
    score_lp_mode: str,
    score_lambda: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute curriculum scores from alpha and LP tensors.

    Shapes:
    - alpha_batch: [B, R]
    - lp_per_reward_function: [B, R]
    - returns:
      - scores: [B]
      - metrics scalars
    """
    if score_lp_mode == "alp":
        lp_used = jnp.abs(lp_per_reward_function)
    elif score_lp_mode == "lp":
        lp_used = jnp.clip(lp_per_reward_function, min=0.0)
    else:
        msg = f"Unsupported score_lp_mode={score_lp_mode!r}. Expected one of ('alp', 'lp')."
        raise ValueError(msg)

    # Alpha-weighted total progress term across reward functions.
    lp_total = jnp.sum(alpha_batch * lp_used, axis=1)  # [B]
    extrinsic_lp = lp_used[:, 0]  # [B]
    lambda_t = jnp.asarray(score_lambda, dtype=lp_total.dtype)
    scores = lambda_t * lp_total + (jnp.array(1.0, dtype=lp_total.dtype) - lambda_t) * extrinsic_lp

    metrics = {
        "curriculum/score_mean": jnp.mean(scores),
    }
    return scores, metrics
