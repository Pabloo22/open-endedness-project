import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


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

    # encoders
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

    # RND specific
    rnd_encoder_mode: str = "flat_symbolic"
    rnd_output_embedding_dim: int = 256  # 64
    rnd_head_activation: str = "relu"
    rnd_head_hidden_dim: int = 256

    predictor_network_lr: float = 1e-4  # 1e-5
    rnd_predictor_update_epochs: int = 1
    rnd_predictor_num_minibatches: int = 64
    num_chunks_in_rnd_rewards_computation: int = 64

    gamma_intrinsic: float = 0.99
    gae_lambda_intrinsic: float = 0.95

    extrinsic_coef: float = 1.0
    intrinsic_coef: float = 0.1

    # eval
    eval_num_envs: int = 1024
    eval_num_episodes: int = 20

    SUPPORTED_ENV_IDS: ClassVar[tuple[str, ...]] = (
        "Craftax-Classic-Symbolic-v1",
        "Craftax-Symbolic-v1",
    )
    SUPPORTED_RND_ENCODER_MODES: ClassVar[tuple[str, ...]] = ("flat_symbolic",)

    def __post_init__(self):
        if self.env_id not in self.SUPPORTED_ENV_IDS:
            msg = f"env_id must be one of {self.SUPPORTED_ENV_IDS}. Received env_id={self.env_id!r}."
            raise ValueError(msg)
        if self.rnd_encoder_mode not in self.SUPPORTED_RND_ENCODER_MODES:
            msg = f"rnd_encoder_mode must be one of {self.SUPPORTED_RND_ENCODER_MODES}. Received rnd_encoder_mode={self.rnd_encoder_mode!r}."
            raise ValueError(msg)

        if self.num_envs_per_batch <= 1:
            msg = f"num_envs_per_batch must be > 1. Received {self.num_envs_per_batch}."
            raise ValueError(msg)
        if self.num_envs_per_batch % self.num_minibatches != 0:
            msg = f"num_envs_per_batch ({self.num_envs_per_batch}) must be divisible by num_minibatches ({self.num_minibatches})."
            raise ValueError(msg)

        self.num_batches_of_envs = math.ceil(
            self.total_timesteps / (self.num_envs_per_batch * self.num_steps_per_env)
        )
        self.num_updates_per_batch = self.num_steps_per_env // self.num_steps_per_update
        # checks
        if self.num_steps_per_env % self.num_steps_per_update != 0:
            msg = f"num_steps_per_env ({self.num_steps_per_env}) must be divisible by num_steps_per_update ({self.num_steps_per_update})"
            raise ValueError(msg)

        if self.num_steps_per_update % self.subsequence_length_in_loss_calculation != 0:
            msg = "num_steps_per_update must be divisible by subsequence_length_in_loss_calculation "
            raise ValueError(msg)

        total_steps_per_update = self.num_envs_per_batch * self.num_steps_per_update
        if total_steps_per_update % self.rnd_predictor_num_minibatches != 0:
            msg = "Total collected steps per update (num_envs_per_batch x num_steps_per_update) must be divisible by rnd_predictor_num_minibatches."
            raise ValueError(msg)
        if total_steps_per_update % self.num_chunks_in_rnd_rewards_computation != 0:
            msg = "Total collected steps must be divisible by num_chunks_in_rnd_rewards_computation."
            raise ValueError(msg)
