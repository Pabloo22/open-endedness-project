"""Configuration objects for the main algorithm training stack."""

import math
from collections.abc import Sequence
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
class ICMConfig:
    # NN configuration
    activation_fn: str = "relu"
    forward_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    inverse_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    
    # hyperparams
    lr: float = 1e-4
    reward_eta: float = 0.01
    beta: float = 0.2
    update_epochs: int = 1
    num_minibatches: int = 64
    num_chunks_in_rewards_computation: int = 64
    eps: float = 1e-8
    
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
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
    lp_norm_ema_beta: float = 0.05

    SUPPORTED_SCORE_LP_MODES: ClassVar[tuple[str, ...]] = ("alp", "lp")
    SUPPORTED_PREDICTOR_ACTIVATIONS: ClassVar[tuple[str, ...]] = ("relu", "tanh")


@dataclass
class TrainConfig:
    train_seed: int = 42
    env_id: str = "Craftax-Classic-Symbolic-v1"
    achievement_ids_to_block: Sequence[int] = ()
    remove_health_reward: bool = False
    episode_max_steps: int | None = 3000
    training_mode: str = "curriculum"

    # training
    num_envs_per_batch: int = 1024
    num_steps_per_env: int = 4096
    num_steps_per_update: int = 256
    total_timesteps: int = 1_000_000_000
    num_batches_of_envs: int = field(init=False)
    num_updates_per_batch: int = field(init=False)
    artifacts_root: str = str(Path(__file__).resolve().parents[3] / "artifacts")

    update_epochs: int = 1
    num_minibatches: int = 16
    optimistic_reset_ratio_limit: int = 16

    adam_eps: float = 1e-5
    lr: float = 2e-4
    anneal_lr: bool = False
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # reward-function preprocessing
    reward_norm_eps: float = 1e-8
    adv_norm_eps: float = 1e-8
    reward_norm_clip: float | None = None
    reset_normalization_running_forward_return_on_new_alpha: bool = False

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
    use_weighted_value_loss: bool = True

    # module selection and module-specific nested config
    selected_intrinsic_modules: tuple[str, ...] = ("rnd",)
    baseline_fixed_training_alpha: tuple[float, ...] | None = None
    num_reward_functions: int = field(init=False)
    is_episodic_per_reward_function: jnp.ndarray = field(init=False)
    gamma_per_reward_function: jnp.ndarray = field(init=False)
    gae_lambda_per_reward_function: jnp.ndarray = field(init=False)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    rnd: RNDConfig = field(default_factory=RNDConfig)
    icm: ICMConfig = field(default_factory=ICMConfig)

    # eval
    eval_every_n_batches: int = 2
    eval_num_envs: int = 1024
    eval_num_episodes: int = 2
    evaluation_alphas: tuple[tuple[float, ...], ...] | None = None
    evaluation_alpha_labels: tuple[str, ...] = field(init=False)
    evaluation_alphas_array: jnp.ndarray = field(init=False)

    # logging
    enable_wandb: bool = True
    wandb_project: str = "open_end_proj"
    wandb_run_name: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_group: str | None = None
    wandb_entity: str | None = None
    is_timing_run: bool = False

    SUPPORTED_ENV_IDS: ClassVar[tuple[str, ...]] = (
        "Craftax-Classic-Symbolic-v1",
        "Craftax-Symbolic-v1",
    )
    SUPPORTED_HEAD_ACTIVATIONS: ClassVar[tuple[str, ...]] = ("relu", "tanh")
    SUPPORTED_TRAINING_MODES: ClassVar[tuple[str, ...]] = ("curriculum", "baseline")

    def __post_init__(self):
        if self.training_mode not in self.SUPPORTED_TRAINING_MODES:
            msg = f"training_mode must be one of {self.SUPPORTED_TRAINING_MODES}. " f"Received {self.training_mode!r}."
            raise ValueError(msg)

        if self.env_id not in self.SUPPORTED_ENV_IDS:
            msg = f"env_id must be one of {self.SUPPORTED_ENV_IDS}. Received env_id={self.env_id!r}."
            raise ValueError(msg)

        if self.head_activation not in self.SUPPORTED_HEAD_ACTIVATIONS:
            msg = (
                f"head_activation must be one of {self.SUPPORTED_HEAD_ACTIVATIONS}. Received {self.head_activation!r}."
            )
            raise ValueError(msg)

        self._validate_selected_intrinsic_modules()
        self._validate_training_layout()
        self._validate_selected_module_configs()
        self.num_reward_functions = 1 + len(self.selected_intrinsic_modules)
        self._apply_mode_specific_overrides()
        self.baseline_fixed_training_alpha = self._resolve_baseline_fixed_training_alpha()
        self.is_episodic_per_reward_function = self._build_is_episodic_per_reward_function()
        self.gamma_per_reward_function = self._build_gamma_per_reward_function()
        self.gae_lambda_per_reward_function = self._build_gae_lambda_per_reward_function()

        self.num_batches_of_envs = math.ceil(self.total_timesteps / (self.num_envs_per_batch * self.num_steps_per_env))
        self.num_updates_per_batch = self.num_steps_per_env // self.num_steps_per_update
        if self.training_mode == "curriculum":
            self._validate_curriculum_config()
        self._validate_eval_config()
        self._validate_wandb_config()
        self.evaluation_alphas_array = self._build_evaluation_alphas_array()
        self.evaluation_alpha_labels = self._build_evaluation_alpha_labels()

    def _validate_selected_intrinsic_modules(self):
        from crew.main_algo.intrinsic_modules.registry import (
            get_registered_intrinsic_module_names,
        )

        registered_names = get_registered_intrinsic_module_names()
        if self.training_mode == "curriculum" and not self.selected_intrinsic_modules:
            msg = "selected_intrinsic_modules cannot be empty for this algorithm."
            raise ValueError(msg)
        if len(self.selected_intrinsic_modules) != len(set(self.selected_intrinsic_modules)):
            msg = (
                f"selected_intrinsic_modules must not contain duplicates. Received {self.selected_intrinsic_modules!r}."
            )
            raise ValueError(msg)

        for module_name in self.selected_intrinsic_modules:
            if not module_name:
                msg = "selected_intrinsic_modules cannot contain empty names."
                raise ValueError(msg)
            if module_name not in registered_names:
                msg = f"Unsupported intrinsic module name {module_name!r}. Supported names: {registered_names}."
                raise ValueError(msg)

    def _validate_training_layout(self):
        if (
            self.num_envs_per_batch <= 0
            or self.num_steps_per_env <= 0
            or self.num_steps_per_update <= 0
            or self.num_minibatches <= 0
            or self.optimistic_reset_ratio_limit <= 0
            or self.subsequence_length_in_loss_calculation <= 0
        ):
            msg = (
                "num_envs_per_batch, num_steps_per_env, num_steps_per_update, num_minibatches, "
                "optimistic_reset_ratio_limit, and subsequence_length_in_loss_calculation must be > 0."
            )
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

        if "icm" in self.selected_intrinsic_modules:
            if self.icm.beta < 0.0 or self.icm.beta > 1.0:
                msg = f"icm.beta must be in [0, 1]. Received {self.icm.beta}."
                raise ValueError(msg)
            if self.icm.reward_eta <= 0.0:
                msg = f"icm.reward_eta must be > 0. Received {self.icm.reward_eta}."
                raise ValueError(msg)
            if self.icm.activation_fn not in ICMConfig.SUPPORTED_HEAD_ACTIVATIONS:
                msg = f"icm.activation_fn must be one of {ICMConfig.SUPPORTED_HEAD_ACTIVATIONS}. Received {self.icm.activation_fn!r}."
                raise ValueError(msg)
            if self.icm.num_minibatches <= 0 or self.icm.update_epochs <= 0:
                msg = "icm.num_minibatches and icm.update_epochs must be > 0."
                raise ValueError(msg)
            
            total_steps_per_update = self.num_envs_per_batch * self.num_steps_per_update
            if total_steps_per_update % self.icm.num_minibatches != 0:
                msg = f"Total collected steps per update (num_envs_per_batch * num_steps_per_update) must be divisible by icm.num_minibatches ({self.icm.num_minibatches})."
                raise ValueError(msg)
            if total_steps_per_update % self.icm.num_chunks_in_rewards_computation != 0:
                msg = (
                    "Total collected steps per update "
                    "(num_envs_per_batch * num_steps_per_update) must be divisible by "
                    "icm.num_chunks_in_rewards_computation "
                    f"({self.icm.num_chunks_in_rewards_computation})."
                )
                raise ValueError(msg)
            
            
        
    def _apply_mode_specific_overrides(self):
        if self.training_mode == "baseline":
            # Baseline policy/value are not alpha-conditioned.
            self.inject_alpha_at_trunk = False
            self.inject_alpha_at_actor_head = False
            self.inject_alpha_at_critic_head = False

    def _resolve_baseline_fixed_training_alpha(self) -> tuple[float, ...] | None:
        if self.training_mode != "baseline":
            return self.baseline_fixed_training_alpha

        if self.baseline_fixed_training_alpha is None:
            return tuple(1.0 if reward_fn_idx == 0 else 0.0 for reward_fn_idx in range(self.num_reward_functions))

        baseline_alpha = jnp.asarray(self.baseline_fixed_training_alpha, dtype=jnp.float32)
        if baseline_alpha.ndim != 1:
            msg = (
                "baseline_fixed_training_alpha must be a 1D tuple-like object with shape [R]. "
                f"Received shape {baseline_alpha.shape}."
            )
            raise ValueError(msg)
        if baseline_alpha.shape[0] != self.num_reward_functions:
            msg = (
                "baseline_fixed_training_alpha must have one coefficient per reward function. "
                f"Expected length {self.num_reward_functions}, received {baseline_alpha.shape[0]}."
            )
            raise ValueError(msg)
        if not bool(jnp.all(jnp.isfinite(baseline_alpha))):
            msg = "baseline_fixed_training_alpha must be finite."
            raise ValueError(msg)
        if bool(jnp.any(baseline_alpha < 0.0)):
            msg = "baseline_fixed_training_alpha must be non-negative."
            raise ValueError(msg)
        alpha_sum = jnp.sum(baseline_alpha)
        if not bool(jnp.allclose(alpha_sum, jnp.array(1.0, dtype=baseline_alpha.dtype), atol=1e-6)):
            msg = "baseline_fixed_training_alpha must sum to 1. " f"Received sum {float(alpha_sum)}."
            raise ValueError(msg)
        return tuple(self.baseline_fixed_training_alpha)

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
        if not (0.0 < self.curriculum.lp_norm_ema_beta <= 1.0):
            msg = "curriculum.lp_norm_ema_beta must be in (0, 1]. " f"Received {self.curriculum.lp_norm_ema_beta}."
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
            elif module_name == "icm":
                gamma_values.append(self.icm.gamma)
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
            elif module_name == "icm":
                gae_lambda_values.append(self.icm.gae_lambda)
            else:
                msg = f"Unsupported intrinsic module {module_name!r} for gae_lambda construction."
                raise ValueError(msg)
        return jnp.asarray(gae_lambda_values, dtype=jnp.float32)

    def _validate_eval_config(self):
        if self.eval_every_n_batches <= 0:
            msg = f"eval_every_n_batches must be > 0. Received {self.eval_every_n_batches}."
            raise ValueError(msg)
        if self.eval_num_envs <= 0:
            msg = f"eval_num_envs must be > 0. Received {self.eval_num_envs}."
            raise ValueError(msg)
        if self.eval_num_episodes <= 0:
            msg = f"eval_num_episodes must be > 0. Received {self.eval_num_episodes}."
            raise ValueError(msg)

    def _validate_wandb_config(self):
        if not self.wandb_project.strip():
            msg = "wandb_project must be a non-empty string."
            raise ValueError(msg)
        if self.wandb_run_name is not None and not self.wandb_run_name.strip():
            msg = "wandb_run_name must be non-empty when provided."
            raise ValueError(msg)
        if self.wandb_group is not None and not self.wandb_group.strip():
            msg = "wandb_group must be non-empty when provided."
            raise ValueError(msg)
        if self.wandb_entity is not None and not self.wandb_entity.strip():
            msg = "wandb_entity must be non-empty when provided."
            raise ValueError(msg)
        for tag in self.wandb_tags:
            if not tag.strip():
                msg = "wandb_tags cannot contain empty strings."
                raise ValueError(msg)

    def _build_evaluation_alphas_array(self) -> jnp.ndarray:
        """Build fixed evaluation alpha matrix with shape [A, R]."""
        if self.training_mode == "baseline":
            if self.baseline_fixed_training_alpha is None:  # pragma: no cover - __post_init__ sets this in baseline.
                msg = "baseline_fixed_training_alpha must be set in baseline mode."
                raise ValueError(msg)
            raw_evaluation_alphas: tuple[tuple[float, ...], ...] = (self.baseline_fixed_training_alpha,)
        elif self.evaluation_alphas is None:
            extrinsic_only_alpha = tuple(
                1.0 if reward_fn_idx == 0 else 0.0 for reward_fn_idx in range(self.num_reward_functions)
            )
            raw_evaluation_alphas: tuple[tuple[float, ...], ...] = (extrinsic_only_alpha,)
        else:
            raw_evaluation_alphas = self.evaluation_alphas

        evaluation_alphas_array = jnp.asarray(raw_evaluation_alphas, dtype=jnp.float32)
        if evaluation_alphas_array.ndim != 2:
            msg = (
                "evaluation_alphas must be a 2D tuple-like object with shape [A, R]. "
                f"Received shape {evaluation_alphas_array.shape}."
            )
            raise ValueError(msg)
        if evaluation_alphas_array.shape[0] <= 0:
            msg = "evaluation_alphas must contain at least one alpha vector."
            raise ValueError(msg)
        if evaluation_alphas_array.shape[1] != self.num_reward_functions:
            msg = (
                "Each evaluation alpha must have one coefficient per reward function. "
                f"Expected second dimension {self.num_reward_functions}, received {evaluation_alphas_array.shape[1]}."
            )
            raise ValueError(msg)
        if not bool(jnp.all(jnp.isfinite(evaluation_alphas_array))):
            msg = "evaluation_alphas must be finite."
            raise ValueError(msg)
        if bool(jnp.any(evaluation_alphas_array < 0.0)):
            msg = "evaluation_alphas must be non-negative."
            raise ValueError(msg)

        row_sums = jnp.sum(evaluation_alphas_array, axis=1)
        if not bool(jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-6)):
            msg = "Each evaluation alpha must sum to 1. " f"Received sums {row_sums}."
            raise ValueError(msg)
        return evaluation_alphas_array

    def _build_evaluation_alpha_labels(self) -> tuple[str, ...]:
        reward_label_prefixes = (
            "ext",
            *(
                "".join(character for character in module_name.lower() if character.isalnum()) or f"r{module_idx + 1}"
                for module_idx, module_name in enumerate(self.selected_intrinsic_modules)
            ),
        )
        labels = []
        for alpha in self.evaluation_alphas_array:
            alpha_tokens = []
            for reward_prefix, reward_weight in zip(reward_label_prefixes, alpha, strict=True):
                decile_weight = int(round(float(reward_weight) * 10.0))
                decile_weight = max(0, min(10, decile_weight))
                alpha_tokens.append(f"{reward_prefix}{decile_weight:02d}")
            labels.append("_".join(alpha_tokens))

        unique_labels: list[str] = []
        label_occurrences: dict[str, int] = {}
        for label in labels:
            occurrence_count = label_occurrences.get(label, 0)
            label_occurrences[label] = occurrence_count + 1
            if occurrence_count == 0:
                unique_labels.append(label)
            else:
                unique_labels.append(f"{label}_{occurrence_count + 1}")
        return tuple(unique_labels)
