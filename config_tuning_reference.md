# Config Fields Tuning Reference

This file documents the fields in `src/curemix/main_algo/config.py`

Column meanings:
- **Field**: config key.
- **Explanation**: what it controls in this codebase.
- **When to try new values**: when to keep fixed vs prioritize a sweep.
- **Reasonable values to try (Default)**: practical options/ranges plus current default.

Note that changing the value of one field can affect what the best values for other fields are, thus we should be fairly consistent while doing tuning.

The specifications for when to try new values and reasonable values to try should only be considered as initial suggestions.

For choosing the values to fix on the base agent training, you can also look into the values used by the transformerXL original jax implementation

## `TrainConfig`

### Environment and Experiment Setup

| Field | Explanation | When to try new values | Reasonable values to try (Default) |
|---|---|---|---|
| `train_seed` | Global RNG seed | Keep fixed for debugging; sweep multiple seeds for final comparisons. | any 3-5+ seeds for final experiments |
| `env_id` | Craftax environment variant. `"Craftax-Classic-Symbolic-v1"`, `"Craftax-Symbolic-v1"` | Use `"Craftax-Classic-Symbolic-v1"` for debugging and hyp tuning. It is faster and simpler. Use  `"Craftax-Symbolic-v1"` for final experiments | - |
| `achievement_ids_to_block` | Removes selected extrinsic achievement rewards via wrapper. | Helps try different setups during both development and final evaluations | - |
| `remove_health_reward` | Removes health-delta reward from extrinsic signal. | - | - |
| `episode_max_steps` | Overrides per-episode horizon (`None` keeps env default). Shorter horizons allow better LP estimation but simplify the env and can potentially make certain achievements too hard. Note episodes are usually much shorter because agent dies. | Priority: high. Can try different values during development. starting with lower. | `None`, `1000-3000-5000-10000`. Default: `3000`. |
| `training_mode` | Chooses curriculum alpha sampling vs fixed-alpha baseline pipeline.  `"curriculum"`, `"baseline"`| - | - |

### Rollout Layout and Compute Budget

| Field | Explanation | When to try new values | Reasonable values to try (Default) |
|---|---|---|---|
| `num_envs_per_batch` | Parallel envs count. Drives throughput and gradient batch stats. | Chosen based on memory limits. Should remain fixed during evaluations. | `128-4096` . Default: `1024`. |
| `num_steps_per_env` | Steps per env before alpha resampling. affects LP estimation stability and frequency of intrinsic reward updates. At least two episode should fit in each env with very high probability | Priority: high. try values during hyp tuning. | `2048-8192` . Default: `4096`. |
| `num_steps_per_update` | Rollout length per PPO update inside one batch. | fixed | `256` (optionally try 128 and 512) (must divide `num_steps_per_env`; divisible by `subsequence_length_in_loss_calculation`). |
| `total_timesteps` | Total environment steps target. | - | Development `any`; full runs `1e7-5e9+`. Default: `1_000_000_000`. |
| `num_batches_of_envs` | Derived outer-loop iteration count. | Never tune directly. |-|
| `num_updates_per_batch` | Derived PPO updates per outer batch. | Never tune directl | - |
| `artifacts_root` | Root folder for saved training artifacts/checkpoints. | Change when organizing storage locations. | Any writable path. |

### PPO Optimization

| Field | Explanation | When to try new values | Reasonable values to try (Default) |
|---|---|---|---|
| `update_epochs` | PPO epochs over each collected rollout window. | fixed/ low priority | `1-4`. Default: `1`. |
| `num_minibatches` | Number of env-axis minibatches per PPO epoch. | fixed/ low priority | `4-64` (must divide `num_envs_per_batch`). Default: `16`. |
| `adam_eps` | Adam epsilon. | fixed/ low priority | `1e-8` or `1e-5`. Default: `1e-5`. |
| `lr` | Actor-critic learning rate. | priority: medium. Usually very important but we already have a reasonable value. | `5e-3` to `5e-4`. Default: `2e-4`. |
| `anneal_lr` | Enables linear LR annealing across outer batches. | fixed |  Default: `False`. |
| `clip_eps` | PPO clipping parameter (policy and value clipping). | priority: medium. important but we already have a reasonable value. | `0.1-0.3`. Default: `0.2`. |
| `gamma` | Extrinsic reward discount factor. | priority: low/medium | `0.99, 0.995, 0.999`. Default: `0.99`. |
| `gae_lambda` | Extrinsic GAE lambda. | priority: low | `0.9-0.99`. Default: `0.95`. |
| `ent_coef` | Entropy bonus weight in PPO loss. | Priority: medium/high.  Usually very important but we already have a reasonable value. | `0.0-0.05`. Default: `0.01`. |
| `vf_coef` | Critic loss weight in total PPO loss. | Priority: fixed/low | `0.25-1.0`. Default: `0.5`. |
| `max_grad_norm` | Global grad clipping threshold. | Priority: fixed/low | `0.3-1.0`. Default: `0.5`. |

### Reward and Advantage Preprocessing

| Field | Explanation | When to try new values | Reasonable values to try (Default) |
|---|---|---|---|
| `reward_norm_eps` | Epsilon in reward normalization denominator. | fixed | `1e-8` to `1e-5`. Default: `1e-8`. |
| `adv_norm_eps` | Epsilon in advantage normalization denominator. | fixed | `1e-8` to `1e-5`. Default: `1e-8`. |
| `reward_norm_clip` | Optional clipping after reward normalization. | Priority: fixed/low | `None`, or `3-20`. Default: `None`. |
| `reset_normalization_running_forward_return_on_new_alpha` | Resets running forward-return state when sampling a new alpha batch. | Priority: low | `False`, `True`. Default: `False`. |

### Encoder and Transformer Architecture

| Field | Explanation | When to try new values | Reasonable values to try (Default) |
|---|---|---|---|
| `obs_emb_dim` | Observation encoder output width. | Priority: medium/low. | `128-1024` |
| `past_context_length` | Transformer-XL memory context length. Affect length of attended history. Big impact on speed. | Priority: fixed/low | `64-256`Default: `128`. |
| `subsequence_length_in_loss_calculation` | Affects gradient propagation through history. Big impact on speed. |  Priority: fixed/low | `32-128` (must divide `num_steps_per_update`). Default: `64`. |
| `num_attn_heads` | Number of attention heads. | Priority: fixed/low | `4`, `8`. Default: `4`. |
| `num_transformer_blocks` | Number of transformer layers. Affects network size and how much into the past the network can see. Big impact on speed. | Priority: fixed/low | `1-6`. Default: `2`. |
| `transformer_hidden_states_dim` | Transformer hidden width. Affects network size and speed. | Priority: fixed/low | `128-512`. Default: `192`. |
| `qkv_features` | Attention q/k/v projection dimension. Must divide by heads. | Priority: fixed/low | same value as transformer_hidden_states_dim|
| `gating` | Enables gated residual formulation in transformer layers. | fixed. | `True`. |
| `gating_bias` | Initial gating bias parameter. | fixed | `2.0`. |

### Actor and Critic Heads

| Field | Explanation | When to try new values | Reasonable values to try (Default) |
|---|---|---|---|
| `head_activation` | Activation used in actor/critic MLP heads. | fixed | `"relu"`, `"tanh"`. Default: `"relu"`. |
| `head_hidden_dim` | Hidden width of actor/critic head MLPs. | Priority: fixed/low | `128-512`. Default: `256`. |

### Alpha Conditioning and Multi-Reward Setup

| Field | Explanation | When to try new values | Reasonable values to try (Default) |
|---|---|---|---|
| `inject_alpha_at_trunk` | Concatenates alpha at transformer trunk input. | priority: medium | `True`, `False`. Default: `True` (auto-forced `False` in baseline mode). |
| `inject_alpha_at_actor_head` | Concatenates alpha at actor head input. | fixed. | `True`, `False`. Default: `True` (auto-forced `False` in baseline mode). |
| `inject_alpha_at_critic_head` | Concatenates alpha at critic head input. | fixed. | `True`, `False`. Default: `True` (auto-forced `False` in baseline mode). |
| `use_weighted_value_loss` | Weights critic-loss per reward head by alpha if `True`. | can be a good ablation. | `True`, `False`. Default: `True`. |
| `selected_intrinsic_modules` | sets the intrinsic modules to use | Priority high. key to our experimetns | Default: `("rnd",)`. |
| `baseline_fixed_training_alpha` | Fixed alpha vector in baseline mode (non-negative, sums to 1). | High-priority in baseline experiments; irrelevant in curriculum mode. | Can look for reasonable values in papers. Default: `None` (resolved to extrinsic-only in baseline). |
| `num_reward_functions` | Number of reward heads (`1 + intrinsic modules`). Auto-derived from `selected_intrinsic_modules`. | Never tune directly. | - |
| `is_episodic_per_reward_function` | Per-reward episodic flags used by LP/normalization logic. Auto-derived (`extrinsic=True`, module-dependent for intrinsic). | Never tune directly. | - |
| `gamma_per_reward_function` | Per-reward gamma vector for GAE and LP estimators.  Auto-derived from `gamma` and module-specific gammas. | Never tune directly | - |
| `gae_lambda_per_reward_function` | Per-reward GAE-lambda vector. Auto-derived from `gae_lambda` and intrinsic modules config. |  Never tune directly | - |
| `curriculum` | Nested curriculum hyperparameter object. `CurriculumConfig(...)` | - | - |
| `rnd` | Nested RND module hyperparameter object. `RNDConfig(...)` | - | - |

- an ablation could also test turning off all the alpha conditioning in the network.

### Evaluation

| Field | Explanation | When to try new values | Reasonable values to try (Default) |
|---|---|---|---|
| `eval_every_n_batches` | Evaluation frequency in outer-batch units | - | `1-10`. |
| `eval_num_envs` | Parallel envs used per evaluation alpha. | Choose according to eval_variance, memory availability, eval_num_episodes, and speed. | `128-4096`. Default: `1024`. |
| `eval_num_episodes` | Episodes per eval env per alpha. | - | - |
| `evaluation_alphas` | alphas on which evaluation is performed | Priority: high. for analysis/interpretability of policy under reward mixtures during training (we can always run the final version on new ones though). | - |
| `evaluation_alpha_labels` | Human-readable labels built from eval alpha weights. Auto-derived. | Never tune directly. | - |
| `evaluation_alphas_array` | Validated array form of eval alphas. Auto-derived. | Never tune directly. | - |

### Logging and Diagnostics

| Field | Explanation | When to try new values | Reasonable values to try (Default) |
|---|---|---|---|
| `enable_wandb` | Enables Weights & Biases logging. | - | - |
| `wandb_project` | W&B project name. | - | - |
| `wandb_run_name` | Optional explicit run name. Reasonable names are auto-derived otherwise | - | - |
| `wandb_tags` | Optional run tags. | Very useful for organizing runs | - |
| `wandb_group` | Optional run group name. | - | - |
| `wandb_entity` | Optional W&B entity/team. | - | - |
| `is_timing_run` | Enables extra synchronization/prints for runtime profiling. | Set `True` only for profiling; keep `False` for normal runs. | Default: `False`. |

## `CurriculumConfig`

| Field | Explanation | When to try new values | Reasonable values to try (Default) |
|---|---|---|---|
| `score_lp_mode` |whether to use absolute learning progress, or normal learnign progress | Priority: high | `"alp"`, `"lp"`. Default: `"alp"`. |
| `score_lambda` | Mix between alpha-weighted LP and extrinsic LP in score. | Priority: high. Can be part of ablations. | `0.0-1.0` (including extremes). Default: `0.5`. |
| `replay_buffer_num_batches` | Replay buffer capacity in outer-batch units for `(alpha, score)` targets. | Priority: low | `1-10`. Default: `5 or 3`. |
| `predictor_lr` | Score predictor optimizer learning rate. | Priority: medium | `1e-5` to `5e-4`. Default: `1e-4`. |
| `predictor_update_epochs` | Predictor training epochs per outer batch. | Priority: low | `1-5`. Default: `1`. |
| `predictor_num_minibatches` | Predictor minibatches per epoch over full replay buffer capacity. | Priority: low | `4-64` (must divide capacity). Default: `16`. |
| `predictor_hidden_dim` | Hidden width of predictor MLP. | Priority: low | `64-512`. Default: `128`. |
| `predictor_activation` | Predictor activation function. | fixed | `"relu"`, `"tanh"`. Default: `"relu"`. |
| `importance_num_candidates_multiplier` | Candidate alpha pool multiplier before importance resampling. | ? | ? |
| `min_batches_for_predictor_sampling` | Number of warmup batches using uniform alpha sampling. | Priority: low/medium | `0-10`. Default: `2`. |
| `sampling_weights_eps` | Positive floor added to sampling weights. | fixed | `1e-8`. |
| `lp_norm_ema_beta` | EMA coefficient for LP normalization stats. | Priority: high | `0.01-0.2` (valid `(0,1]`). Default: `0.05`. |

## `RNDConfig`

| Field | Explanation | When to try new values | Reasonable values to try (Default) |
|---|---|---|---|
| `encoder_mode` | RND encoder mode. | Keep fixed (only one supported mode currently). | - |
| `output_embedding_dim` | Output embedding width for target/predictor distance. | Priority:low/medium | `64-512`. Default: `256`. |
| `head_activation` | Activation in RND target/predictor heads. | fixed | `"relu"`, `"tanh"`. Default: `"relu"`. |
| `head_hidden_dim` | Hidden width in RND target/predictor heads. | Priority:fixed/low | `128-512`. Default: `256`. |
| `predictor_network_lr` | RND predictor optimizer learning rate. | priority: medium. | `1e-5` to `5e-4`. Default: `1e-4`. |
| `predictor_update_epochs` | RND predictor epochs after each outer batch. | Priority: low | `1-5`. Default: `1`. |
| `predictor_num_minibatches` | Minibatches for RND predictor update. | Priority: low | `8-128` (must divide `num_envs_per_batch * num_steps_per_update`). Default: `64`. |
| `num_chunks_in_rewards_computation` | Chunking factor for RND reward computation to manage memory/throughput. Does not affect results. | - | - |
| `gamma` | Discount factor for RND reward stream (non-episodic done mask). | priority: medium/high. | `0.99,0.995,0.999`. Default: `0.99`. |
| `gae_lambda` | GAE lambda for RND reward stream. | Priority: fixed/low | `0.9-0.99`. Default: `0.95`. |

