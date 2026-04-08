"""Record agent episode videos and log them to W&B.

This module runs a trained policy through complete episodes outside the JIT'd
eval pipeline, rendering each step via Craftax's pixel renderer. The resulting
videos are logged to W&B as ``wandb.Video`` objects.

Usage from the training loop::

    from crew.main_algo.video import record_and_log_videos
    record_and_log_videos(
        run=wandb_run,
        train_state=agent_train_state,
        env=eval_env,
        env_params=env_params,
        config=config,
        alpha=alpha_vector,
        num_episodes=2,
        step=total_env_steps,
    )
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from curemix.main_algo.actor_critic import ActorCriticTransformer


def _get_renderer(env_id: str):
    """Return the (render_fn, block_pixel_size) pair for the given env."""
    if "Classic" in env_id:
        from craftax.craftax_classic.renderer import BLOCK_PIXEL_SIZE_IMG, render_craftax_pixels
    else:
        from craftax.craftax.renderer import BLOCK_PIXEL_SIZE_IMG, render_craftax_pixels
    return render_craftax_pixels, BLOCK_PIXEL_SIZE_IMG


def record_episodes(
    rng: jax.Array,
    train_state: Any,
    env: Any,
    env_params: Any,
    config: Any,
    alpha: jax.Array,
    num_episodes: int = 1,
    max_steps_per_episode: int | None = None,
) -> list[np.ndarray]:
    """Run ``num_episodes`` and return a list of uint8 frame arrays [T, H, W, C].

    This function is NOT jit-compiled — it steps one action at a time in Python
    so it can call the Craftax pixel renderer on each ``env_state``.
    """
    render_fn, block_pixel_size = _get_renderer(config.env_id)
    if max_steps_per_episode is None:
        max_steps_per_episode = config.episode_max_steps or 4096

    all_episode_frames: list[np.ndarray] = []

    for _ in range(num_episodes):
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng, env_params)

        # Transformer memory state for a single env.
        memories = jnp.zeros(
            (1, config.past_context_length, config.num_transformer_blocks, config.transformer_hidden_states_dim),
            dtype=jnp.float32,
        )
        memories_mask = jnp.zeros(
            (1, config.num_attn_heads, 1, config.past_context_length + 1),
            dtype=jnp.bool_,
        )
        memories_mask_idx = jnp.full((1,), config.past_context_length + 1, dtype=jnp.int32)
        done_prev = False

        frames: list[np.ndarray] = []

        for _step in range(max_steps_per_episode):
            # Render current state.
            frame = render_fn(env_state, block_pixel_size)
            frame_np = np.asarray(frame, dtype=np.uint8)
            frames.append(frame_np)

            # Update attention mask.
            if done_prev:
                memories_mask_idx = jnp.full((1,), config.past_context_length, dtype=jnp.int32)
                memories_mask = jnp.zeros_like(memories_mask)
            else:
                memories_mask_idx = jnp.clip(memories_mask_idx - 1, 0, config.past_context_length)

            memories_mask_idx_one_hot = jax.nn.one_hot(
                memories_mask_idx, config.past_context_length + 1, dtype=jnp.bool_
            )
            memories_mask_idx_one_hot = jnp.repeat(
                memories_mask_idx_one_hot[:, None, None, :], config.num_attn_heads, axis=1
            )
            memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_one_hot)

            # Sample action.
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            alpha_batch = alpha[None, :]
            pi, _, memories_out = train_state.apply_fn(
                train_state.params,
                memories,
                obs[None, None, :],
                memories_mask,
                alpha_batch,
                method=ActorCriticTransformer.model_forward_eval,
            )
            action = pi.sample(seed=action_rng).squeeze(axis=0)
            memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

            # Step environment.
            obs, env_state, _reward, done, _info = env.step(step_rng, env_state, action, env_params)
            done_prev = bool(done)

            if done_prev:
                # Render the terminal frame too.
                frame = render_fn(env_state, block_pixel_size)
                frames.append(np.asarray(frame, dtype=np.uint8))
                break

        all_episode_frames.append(np.stack(frames, axis=0))  # [T, H, W, C]

    return all_episode_frames


def record_and_log_videos(
    run: Any,
    train_state: Any,
    env: Any,
    env_params: Any,
    config: Any,
    alpha: jax.Array,
    num_episodes: int = 1,
    step: int | None = None,
    fps: int = 10,
    rng: jax.Array | None = None,
) -> None:
    """Record episode videos and log them to the active W&B run.

    Args:
        run: The active ``wandb.Run`` (or ``None`` to skip).
        train_state: The agent's ``TrainState``.
        env: A single (non-vectorized) environment (e.g. ``AutoResetEnvWrapper``).
        env_params: Environment parameters.
        config: ``TrainConfig``.
        alpha: Reward weighting vector [R].
        num_episodes: How many episodes to record.
        step: W&B logging step (typically ``total_env_steps``).
        fps: Video frame rate.
        rng: Optional RNG key; one is created if not provided.
    """
    if run is None:
        return

    import wandb

    if rng is None:
        rng = jax.random.PRNGKey(0)

    episode_frames = record_episodes(
        rng=rng,
        train_state=train_state,
        env=env,
        env_params=env_params,
        config=config,
        alpha=alpha,
        num_episodes=num_episodes,
    )

    payload: dict[str, Any] = {}
    for ep_idx, frames in enumerate(episode_frames):
        # wandb.Video expects [T, C, H, W] uint8
        video_array = np.transpose(frames, (0, 3, 1, 2))
        payload[f"video/episode_{ep_idx}"] = wandb.Video(video_array, fps=fps, format="mp4")

    wandb.log(payload, step=step, commit=False)
