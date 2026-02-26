"""Configuration objects for the main algorithm training stack."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import jax.numpy as jnp


@dataclass
class RNDConfig:
    encoder_mode: str = "flat_symbolic"
    output_embedding_dim: int = 256
    head_activation: str = "relu"
    head_hidden_dim: int = 256

    predictor_network_lr: float = 1e-4
    predictor_update_epochs: int = 1
    predictor_num_minibatches: int = 64
    num_chunks_in_rewards_computation: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95

    SUPPORTED_ENCODER_MODES: ClassVar[tuple[str, ...]] = ("flat_symbolic",)
    SUPPORTED_HEAD_ACTIVATIONS: ClassVar[tuple[str, ...]] = ("relu", "tanh")


@dataclass
class CurriculumConfig:
    score_lp_mode: str = "alp"
    score_lambda: float = 0.5
    replay_buffer_num_batches: int = 5
    predictor_lr: float = 1e-4
    predictor_update_epochs: int = 1
    predictor_num_minibatches: int = 16
    predictor_hidden_dim: int = 128
    predictor_activation: str = "relu"
    importance_num_candidates_multiplier: int = 10
    min_batches_for_predictor_sampling: int = 1
    sampling_weights_eps: float = 1e-8

    SUPPORTED_SCORE_LP_MODES: ClassVar[tuple[str, ...]] = ("alp", "lp")
    SUPPORTED_PREDICTOR_ACTIVATIONS: ClassVar[tuple[str, ...]] = ("relu", "tanh")


@dataclass
class TrainConfig:
    train_seed: int = 42
    env_id: str = "Craftax-Classic-Symbolic-v1"
    episode_max_steps: int | None = 1000

    # training
    num_envs_per_batch: int = 2048
    num_steps_per_env: int = 5120
    num_steps_per_update: int = 256
    total_timesteps: int = 1_000_000_000
    num_batches_of_envs: int = field(init=False)
    num_updates_per_batch: int = field(init=False)
    artifacts_root: str = str(Path(__file__).resolve().parents[3] / "artifacts")

    update_epochs: int = 1
    num_minibatches: int = 16

    adam_eps: float = 1e-5
    lr: float = 2e-4
    anneal_lr: bool = False
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # reward-function preprocessing
    reward_norm_eps: float = 1e-8
    adv_norm_eps: float = 1e-8
    reward_norm_clip: float | None = None

    # encoder
    obs_emb_dim: int = 256

    # Transformer XL specific
    past_context_length: int = 128
    subsequence_length_in_loss_calculation: int = 64
    num_attn_heads: int = 4
    num_transformer_blocks: int = 2
    transformer_hidden_states_dim: int = 192
    qkv_features: int = 192
    gating: bool = True
    gating_bias: float = 2.0

    # actor and critic head MLPs
    head_activation: str = "relu"
    head_hidden_dim: int = 256

    # alpha conditioning
    inject_alpha_at_trunk: bool = True
    inject_alpha_at_actor_head: bool = True
    inject_alpha_at_critic_head: bool = True

    # module selection and module-specific nested config
    selected_intrinsic_modules: tuple[str, ...] = ("rnd",)
    num_reward_functions: int = field(init=False)
    is_episodic_per_reward_function: jnp.ndarray = field(init=False)
    gamma_per_reward_function: jnp.ndarray = field(init=False)
    gae_lambda_per_reward_function: jnp.ndarray = field(init=False)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    rnd: RNDConfig = field(default_factory=RNDConfig)

    # eval
    eval_num_envs: int = 1024
    eval_num_episodes: int = 20

    SUPPORTED_ENV_IDS: ClassVar[tuple[str, ...]] = (
        "Craftax-Classic-Symbolic-v1",
        "Craftax-Symbolic-v1",
    )
    SUPPORTED_HEAD_ACTIVATIONS: ClassVar[tuple[str, ...]] = ("relu", "tanh")

    def __post_init__(self):
        if self.env_id not in self.SUPPORTED_ENV_IDS:
            msg = f"env_id must be one of {self.SUPPORTED_ENV_IDS}. Received env_id={self.env_id!r}."
            raise ValueError(msg)

        if self.head_activation not in self.SUPPORTED_HEAD_ACTIVATIONS:
            msg = f"head_activation must be one of {self.SUPPORTED_HEAD_ACTIVATIONS}. Received {self.head_activation!r}."
            raise ValueError(msg)

        self._validate_selected_intrinsic_modules()
        self._validate_training_layout()
        self._validate_selected_module_configs()
        self._validate_curriculum_config()
        self.num_reward_functions = 1 + len(self.selected_intrinsic_modules)
        self.is_episodic_per_reward_function = self._build_is_episodic_per_reward_function()
        self.gamma_per_reward_function = self._build_gamma_per_reward_function()
        self.gae_lambda_per_reward_function = self._build_gae_lambda_per_reward_function()

        self.num_batches_of_envs = math.ceil(self.total_timesteps / (self.num_envs_per_batch * self.num_steps_per_env))
        self.num_updates_per_batch = self.num_steps_per_env // self.num_steps_per_update

    def _validate_selected_intrinsic_modules(self):
        from crew.main_algo.intrinsic_modules.registry import (
            get_registered_intrinsic_module_names,
        )

        registered_names = get_registered_intrinsic_module_names()
        if not self.selected_intrinsic_modules:
            msg = "selected_intrinsic_modules cannot be empty for this algorithm."
            raise ValueError(msg)
        if len(self.selected_intrinsic_modules) != len(set(self.selected_intrinsic_modules)):
            msg = f"selected_intrinsic_modules must not contain duplicates. Received {self.selected_intrinsic_modules!r}."
            raise ValueError(msg)

        for module_name in self.selected_intrinsic_modules:
            if not module_name:
                msg = "selected_intrinsic_modules cannot contain empty names."
                raise ValueError(msg)
            if module_name not in registered_names:
                msg = f"Unsupported intrinsic module name {module_name!r}. Supported names: {registered_names}."
                raise ValueError(msg)

    def _validate_training_layout(self):
        if self.num_envs_per_batch <= 0 or self.num_steps_per_env <= 0 or self.num_steps_per_update <= 0 or self.num_minibatches <= 0 or self.subsequence_length_in_loss_calculation <= 0:
            msg = "num_envs_per_batch, num_steps_per_env, num_steps_per_update, num_minibatches, and subsequence_length_in_loss_calculation must be > 0."
            raise ValueError(msg)

        if self.num_envs_per_batch % self.num_minibatches != 0:
            msg = f"num_envs_per_batch ({self.num_envs_per_batch}) must be divisible by num_minibatches ({self.num_minibatches})."
            raise ValueError(msg)

        if self.num_steps_per_env % self.num_steps_per_update != 0:
            msg = f"num_steps_per_env ({self.num_steps_per_env}) must be divisible by num_steps_per_update ({self.num_steps_per_update})."
            raise ValueError(msg)

        if self.num_steps_per_update % self.subsequence_length_in_loss_calculation != 0:
            msg = "num_steps_per_update must be divisible by subsequence_length_in_loss_calculation."
            raise ValueError(msg)
        if self.reward_norm_eps <= 0 or self.adv_norm_eps <= 0:
            msg = "reward_norm_eps and adv_norm_eps must be > 0."
            raise ValueError(msg)
        if self.reward_norm_clip is not None and self.reward_norm_clip <= 0:
            msg = "reward_norm_clip must be > 0 when provided."
            raise ValueError(msg)

    def _validate_selected_module_configs(self):
        # RND-specific static checks; rollout-dependent checks are deferred until phase 2.
        if "rnd" in self.selected_intrinsic_modules:
            if self.rnd.encoder_mode not in RNDConfig.SUPPORTED_ENCODER_MODES:
                msg = f"rnd.encoder_mode must be one of {RNDConfig.SUPPORTED_ENCODER_MODES}. Received {self.rnd.encoder_mode!r}."
                raise ValueError(msg)
            if self.rnd.head_activation not in RNDConfig.SUPPORTED_HEAD_ACTIVATIONS:
                msg = f"rnd.head_activation must be one of {RNDConfig.SUPPORTED_HEAD_ACTIVATIONS}. Received {self.rnd.head_activation!r}."
                raise ValueError(msg)
            if self.rnd.predictor_num_minibatches <= 0 or self.rnd.num_chunks_in_rewards_computation <= 0:
                msg = "rnd.predictor_num_minibatches and rnd.num_chunks_in_rewards_computation must be > 0."
                raise ValueError(msg)

            total_steps_per_update = self.num_envs_per_batch * self.num_steps_per_update
            if total_steps_per_update % self.rnd.predictor_num_minibatches != 0:
                msg = f"Total collected steps per update (num_envs_per_batch * num_steps_per_update) must be divisible by rnd.predictor_num_minibatches ({self.rnd.predictor_num_minibatches})."
                raise ValueError(msg)
            if total_steps_per_update % self.rnd.num_chunks_in_rewards_computation != 0:
                msg = (
                    "Total collected steps per update "
                    "(num_envs_per_batch * num_steps_per_update) must be divisible by "
                    "rnd.num_chunks_in_rewards_computation "
                    f"({self.rnd.num_chunks_in_rewards_computation})."
                )
                raise ValueError(msg)

    def _validate_curriculum_config(self):
        if self.curriculum.score_lp_mode not in CurriculumConfig.SUPPORTED_SCORE_LP_MODES:
            msg = f"curriculum.score_lp_mode must be one of {CurriculumConfig.SUPPORTED_SCORE_LP_MODES}. Received {self.curriculum.score_lp_mode!r}."
            raise ValueError(msg)
        if not (0.0 <= self.curriculum.score_lambda <= 1.0):
            msg = f"curriculum.score_lambda must be in [0, 1]. Received {self.curriculum.score_lambda}."
            raise ValueError(msg)
        if self.curriculum.replay_buffer_num_batches <= 0:
            msg = f"curriculum.replay_buffer_num_batches must be > 0. Received {self.curriculum.replay_buffer_num_batches}."
            raise ValueError(msg)
        if self.curriculum.predictor_activation not in CurriculumConfig.SUPPORTED_PREDICTOR_ACTIVATIONS:
            msg = f"curriculum.predictor_activation must be one of {CurriculumConfig.SUPPORTED_PREDICTOR_ACTIVATIONS}. Received {self.curriculum.predictor_activation!r}."
            raise ValueError(msg)
        if (
            self.curriculum.predictor_lr <= 0
            or self.curriculum.predictor_update_epochs <= 0
            or self.curriculum.predictor_num_minibatches <= 0
            or self.curriculum.predictor_hidden_dim <= 0
            or self.curriculum.importance_num_candidates_multiplier <= 0
            or self.curriculum.sampling_weights_eps <= 0
        ):
            msg = (
                "curriculum.predictor_lr, curriculum.predictor_update_epochs, "
                "curriculum.predictor_num_minibatches, curriculum.predictor_hidden_dim, "
                "curriculum.importance_num_candidates_multiplier, and curriculum.sampling_weights_eps must be > 0."
            )
            raise ValueError(msg)
        if self.curriculum.min_batches_for_predictor_sampling < 0:
            msg = f"curriculum.min_batches_for_predictor_sampling must be >= 0. Received {self.curriculum.min_batches_for_predictor_sampling}."
            raise ValueError(msg)

        buffer_capacity = self.curriculum.replay_buffer_num_batches * self.num_envs_per_batch
        if buffer_capacity % self.curriculum.predictor_num_minibatches != 0:
            msg = (
                "Total replay-buffer capacity "
                "(curriculum.replay_buffer_num_batches * num_envs_per_batch) "
                "must be divisible by curriculum.predictor_num_minibatches. "
                f"Received capacity={buffer_capacity} and "
                f"predictor_num_minibatches={self.curriculum.predictor_num_minibatches}."
            )
            raise ValueError(msg)

    def _build_gamma_per_reward_function(self) -> jnp.ndarray:
        """Build per-reward-function gamma values [R]."""
        gamma_values = [self.gamma]
        for module_name in self.selected_intrinsic_modules:
            if module_name == "rnd":
                gamma_values.append(self.rnd.gamma)
            else:
                msg = f"Unsupported intrinsic module {module_name!r} for gamma construction."
                raise ValueError(msg)
        return jnp.asarray(gamma_values, dtype=jnp.float32)

    def _build_is_episodic_per_reward_function(self) -> jnp.ndarray:
        """Build per-reward-function episodic flags [R]."""
        from crew.main_algo.intrinsic_modules.registry import get_intrinsic_module

        episodic_flags = [True]  # Extrinsic reward stream.
        for module_name in self.selected_intrinsic_modules:
            module = get_intrinsic_module(module_name)
            episodic_flags.append(bool(module.is_episodic))

        episodic_flags_array = jnp.asarray(episodic_flags, dtype=jnp.bool_)
        if episodic_flags_array.shape != (self.num_reward_functions,):
            msg = f"is_episodic_per_reward_function must have shape ({self.num_reward_functions},). Received {episodic_flags_array.shape}."
            raise ValueError(msg)
        return episodic_flags_array

    def _build_gae_lambda_per_reward_function(self) -> jnp.ndarray:
        """Build per-reward-function gae_lambda values [R]."""
        gae_lambda_values = [self.gae_lambda]
        for module_name in self.selected_intrinsic_modules:
            if module_name == "rnd":
                gae_lambda_values.append(self.rnd.gae_lambda)
            else:
                msg = f"Unsupported intrinsic module {module_name!r} for gae_lambda construction."
                raise ValueError(msg)
        return jnp.asarray(gae_lambda_values, dtype=jnp.float32)
