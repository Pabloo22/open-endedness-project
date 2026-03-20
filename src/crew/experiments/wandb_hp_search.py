"""Run hyperparameter tuning for the training stack with W&B random search.

This module wraps the existing training entrypoint in a Weights & Biases sweep
agent so that W&B can randomly sample hyperparameters and launch repeated
training trials.

Typical workflow:
1. Adjust the search space in ``build_default_sweep_config`` to the config
    fields and value ranges you want to tune.
2. Start a new sweep and agent with a command such as
    ``poetry run python -m crew.experiments.wandb_random_search --count 100 --tuning-phase generic``.
3. If you only want to create the sweep and inspect it in W&B before running
    trials, use ``--create-only``.
4. If you already have a sweep id, reuse it with ``--sweep-id <id>``.

Each trial starts from the base configuration built by
``build_base_tuning_config`` and then applies the sampled W&B overrides before
rebuilding ``TrainConfig`` so validation still runs. Trials are executed
through ``run_main_algo_training``.

The script deliberately disables the inner training loop's W&B logging to avoid
nested runs. Instead, each sweep trial owns a single W&B run and logs a compact
final summary, including the main sweep objective
``tuning/objective_eval_return_mean`` and several final training and evaluation
diagnostics. Use ``--save-results`` if you also want each trial to persist the
full checkpoint and training artifacts.

Commdands run so far:

poetry run python -m crew.experiments.wandb_hp_search --count 100 --tuning-phase generic``

poetry run python -m crew.experiments.wandb_hp_search --tuning-phase generic --method grid --enable-inner-wandb

nohup poetry run python -m crew.experiments.wandb_hp_search --tuning-phase intrinsic --intrinsic-modules rnd \
    --enable-inner-wandb --count 100 >& nohup.out &

with `get_best_generic_params`:
nohup poetry run python -m crew.experiments.wandb_hp_search --tuning-phase generic --method grid  \
    --enable-inner-wandb --fixed-override video_num_episodes=5 --total-timesteps 1_000_000_000 >& nohup.out &

with `get_best_lightweight_generic_params`:
nohup poetry run python -m crew.experiments.wandb_hp_search --tuning-phase intrinsic --intrinsic-modules rnd \
    --enable-inner-wandb --fixed-override video_num_episodes=1 --count 100 >& nohup.out &

with `get_best_generic_params`:
nohup poetry run python -m crew.experiments.wandb_hp_search --tuning-phase generic --method grid  \
    --enable-inner-wandb --fixed-override video_num_episodes=5 --total-timesteps 1_000_000_000 \
    --sweep-id pp976ug7 >& nohup.out &
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import wandb

from crew.experiments.run_training import run_main_algo_training
from crew.experiments.tuning_configs import (
    DEFAULT_BASELINE_INTRINSIC_ALPHA as _DEFAULT_BASELINE_INTRINSIC_ALPHA,
    DEFAULT_INTRINSIC_MODULES,
    get_curriculum_base_config_for_modules,
    get_curriculum_search_space,
    get_generic_base_config,
    get_generic_search_space,
    get_intrinsic_base_config,
    get_intrinsic_search_space,
)
from crew.main_algo.config import TrainConfig
from crew.main_algo.logging import build_wandb_run_name


DEFAULT_OBJECTIVE_METRIC = "tuning/objective_eval_return_mean"
DEFAULT_BASELINE_INTRINSIC_ALPHA = _DEFAULT_BASELINE_INTRINSIC_ALPHA

TUNING_PHASE_GENERIC = "generic"
TUNING_PHASE_INTRINSIC = "intrinsic"
TUNING_PHASE_CURRICULUM = "curriculum"
SUPPORTED_TUNING_PHASES = (
    TUNING_PHASE_GENERIC,
    TUNING_PHASE_INTRINSIC,
    TUNING_PHASE_CURRICULUM,
)

SWEEP_METHOD_RANDOM = "random"
SWEEP_METHOD_GRID = "grid"
SUPPORTED_SWEEP_METHODS = (SWEEP_METHOD_RANDOM, SWEEP_METHOD_GRID)


def main() -> None:
    """Create or reuse a W&B sweep and launch the requested number of trials."""
    args = parse_args()
    intrinsic_modules = tuple(args.intrinsic_modules)
    base_config = build_base_tuning_config(
        tuning_phase=args.tuning_phase,
        intrinsic_modules=intrinsic_modules,
        project=args.project,
        entity=args.entity,
        group=args.group,
        train_seed=args.train_seed,
        total_timesteps=args.total_timesteps,
        enable_inner_wandb=args.enable_inner_wandb,
    )
    fixed_overrides = parse_fixed_overrides(args.fixed_override)
    if fixed_overrides:
        base_config = build_trial_config_from_overrides(base_config=base_config, overrides=fixed_overrides)

    sweep_config = build_default_sweep_config(
        tuning_phase=args.tuning_phase,
        intrinsic_modules=intrinsic_modules,
        method=args.method,
    )

    sweep_id = args.sweep_id
    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)
        print(f"Created W&B sweep: {sweep_id}")
        if args.create_only:
            return
    else:
        print(f"Using existing W&B sweep: {sweep_id}")

    count = None if args.method == SWEEP_METHOD_GRID else args.count
    wandb.agent(
        sweep_id,
        function=lambda: run_single_trial(base_config=base_config, save_results=args.save_results),
        count=count,
        project=args.project,
        entity=args.entity,
    )


def build_default_sweep_config(
    *,
    tuning_phase: str,
    intrinsic_modules: tuple[str, ...] = DEFAULT_INTRINSIC_MODULES,
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
    method: str = SWEEP_METHOD_RANDOM,
) -> dict[str, Any]:
    """Return the default W&B sweep definition for the selected tuning phase.

    Args:
        method: W&B sweep method. Either ``"random"`` (default) or ``"grid"``.
            Grid search requires all search-space parameters to use ``values``
            rather than continuous ranges.
    """
    if method not in SUPPORTED_SWEEP_METHODS:
        msg = f"Unsupported sweep method {method!r}. Expected one of {SUPPORTED_SWEEP_METHODS}."
        raise ValueError(msg)
    parameters = build_phase_search_space(tuning_phase=tuning_phase, intrinsic_modules=intrinsic_modules)

    return {
        "method": method,
        "metric": {"name": objective_metric, "goal": "maximize"},
        "parameters": parameters,
    }


def build_phase_search_space(*, tuning_phase: str, intrinsic_modules: tuple[str, ...]) -> dict[str, Any]:
    """Return the sweep parameter space for one tuning phase."""
    if tuning_phase == TUNING_PHASE_GENERIC:
        return get_generic_search_space()
    if tuning_phase == TUNING_PHASE_INTRINSIC:
        return get_intrinsic_search_space(_require_single_intrinsic_module(tuning_phase, intrinsic_modules))
    if tuning_phase == TUNING_PHASE_CURRICULUM:
        return get_curriculum_search_space()

    msg = f"Unsupported tuning_phase {tuning_phase!r}. Expected one of {SUPPORTED_TUNING_PHASES}."
    raise ValueError(msg)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for creating or running the random-search sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="Number of random-search trials to run.")
    parser.add_argument("--project", type=str, default="open_end_proj", help="W&B project name.")
    parser.add_argument("--entity", type=str, default=None, help="Optional W&B entity/team.")
    parser.add_argument("--group", type=str, default=None, help="Optional W&B group shared across trials.")
    parser.add_argument(
        "--tuning-phase",
        choices=SUPPORTED_TUNING_PHASES,
        required=True,
        help=(
            "Which tuning phase to run. "
            "generic tunes shared PPO/model hyperparameters, intrinsic tunes one intrinsic module in baseline mode, "
            "and curriculum tunes curriculum-only hyperparameters while using all selected reward functions."
        ),
    )
    parser.add_argument(
        "--intrinsic-modules",
        nargs="+",
        default=list(DEFAULT_INTRINSIC_MODULES),
        help=(
            "Intrinsic modules to include. Use one module for the intrinsic phase. "
            "Use one or more modules for the curriculum phase, for example --intrinsic-modules rnd icm."
        ),
    )
    parser.add_argument(
        "--train-seed", type=int, default=0, help="Base seed used for setting the seed of each trial."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000_000,
        help="Total environment steps per trial.",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Existing W&B sweep id. If omitted, the script creates a new random sweep.",
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Create the sweep and print its id without launching an agent.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Persist the full checkpoint/result artifact for each trial.",
    )
    parser.add_argument(
        "--method",
        choices=list(SUPPORTED_SWEEP_METHODS),
        default=SWEEP_METHOD_RANDOM,
        help="W&B sweep search method. Use 'grid' for an exhaustive grid search over discrete 'values' parameters.",
    )
    parser.add_argument(
        "--fixed-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Apply a fixed override before the sweep starts. "
            "Use dotted keys and JSON or Python literals, for example "
            "--fixed-override num_envs_per_batch=128 or "
            "--fixed-override baseline_fixed_training_alpha='[0.8, 0.2]'."
        ),
    )
    parser.add_argument(
        "--enable-inner-wandb",
        action="store_true",
        help="Enable full W&B logging for the inner training runs. Disabled by default to avoid nested runs.",
    )
    return parser.parse_args()


def build_base_tuning_config(
    *,
    tuning_phase: str,
    intrinsic_modules: tuple[str, ...] = DEFAULT_INTRINSIC_MODULES,
    entity: str | None,
    group: str | None,
    train_seed: int,
    total_timesteps: int,
    enable_inner_wandb: bool = False,
    project: str = "openendedness-2026",
) -> TrainConfig:
    """Build the fixed base training config used as the starting point for every trial."""
    shared_runtime_kwargs = {
        "train_seed": train_seed,
        "total_timesteps": total_timesteps,
        "enable_wandb": enable_inner_wandb,
        "wandb_project": project,
        "wandb_entity": entity,
        "wandb_group": group,
    }

    if tuning_phase == TUNING_PHASE_GENERIC:
        base_dict = get_generic_base_config()
    elif tuning_phase == TUNING_PHASE_INTRINSIC:
        intrinsic_module = _require_single_intrinsic_module(tuning_phase, intrinsic_modules)
        base_dict = get_intrinsic_base_config(intrinsic_module)
    elif tuning_phase == TUNING_PHASE_CURRICULUM:
        base_dict = get_curriculum_base_config_for_modules(_require_intrinsic_modules(intrinsic_modules))
    else:
        msg = f"Unsupported tuning phase {tuning_phase!r}. Expected one of {SUPPORTED_TUNING_PHASES}."
        raise ValueError(msg)

    raw_config = {**base_dict, **shared_runtime_kwargs}
    flat_kwargs = {k: v for k, v in raw_config.items() if "." not in k}
    dot_kwargs = {k: v for k, v in raw_config.items() if "." in k}

    config = TrainConfig(**flat_kwargs)
    for dotted_path, value in dot_kwargs.items():
        _set_nested_attr(config, dotted_path, value)

    return TrainConfig(**_dataclass_init_kwargs(config))


def parse_fixed_overrides(raw_overrides: list[str]) -> dict[str, Any]:
    """Parse repeated KEY=VALUE overrides from the CLI."""
    parsed: dict[str, Any] = {}
    for raw_override in raw_overrides:
        key, separator, raw_value = raw_override.partition("=")
        if not separator or not key:
            msg = f"Invalid override {raw_override!r}. Expected KEY=VALUE."
            raise ValueError(msg)
        parsed[key] = _parse_override_value(raw_value)
    return parsed


def _parse_override_value(raw_value: str) -> Any:
    """Parse one CLI override value using JSON first, then Python literals, then raw string."""
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            return raw_value


def _require_intrinsic_modules(intrinsic_modules: tuple[str, ...]) -> tuple[str, ...]:
    if not intrinsic_modules:
        msg = "At least one intrinsic module must be provided for intrinsic and curriculum tuning phases."
        raise ValueError(msg)
    return intrinsic_modules


def _require_single_intrinsic_module(tuning_phase: str, intrinsic_modules: tuple[str, ...]) -> str:
    resolved_intrinsic_modules = _require_intrinsic_modules(intrinsic_modules)
    if len(resolved_intrinsic_modules) != 1:
        msg = f"{tuning_phase} phase expects exactly one intrinsic module. " f"Received {resolved_intrinsic_modules!r}."
        raise ValueError(msg)
    return resolved_intrinsic_modules[0]


def build_trial_config_from_overrides(
    base_config: TrainConfig,
    overrides: Mapping[str, Any],
) -> TrainConfig:
    """Apply sampled sweep overrides to a base config and return a revalidated TrainConfig."""
    mutable_config = _clone_train_config(base_config)
    applied_override_keys: set[str] = set()

    for key, value in overrides.items():
        if key.startswith("_"):
            continue
        _set_nested_attr(mutable_config, key, value)
        applied_override_keys.add(key)

    if "transformer_hidden_states_dim" in applied_override_keys and "qkv_features" not in applied_override_keys:
        mutable_config.qkv_features = mutable_config.transformer_hidden_states_dim

    return TrainConfig(**cast(Any, _dataclass_init_kwargs(mutable_config)))


def extract_trial_summary(train_info: dict[str, Any]) -> dict[str, Any]:
    """Extract the final scalar metrics logged back to the W&B sweep run."""
    metrics = train_info["metrics"]
    eval_returns = _last_metric_slice(metrics["eval/returns"], dtype=np.float32)
    eval_lengths = _last_metric_slice(metrics["eval/lengths"], dtype=np.float32)
    eval_achievements = _last_metric_slice(metrics["eval/achievements"], dtype=np.float32)

    summary: dict[str, Any] = {
        DEFAULT_OBJECTIVE_METRIC: float(eval_returns.mean()),
        "tuning/objective_eval_return_alpha0_mean": float(eval_returns[0].mean()),
        "tuning/final_eval_length_mean": float(eval_lengths.mean()),
        "tuning/final_eval_num_achievements_mean": float(eval_achievements.sum(axis=-1).mean()),
        "tuning/final_total_env_steps": int(_last_scalar_metric(metrics["run/total_env_steps"])),
        "tuning/final_total_loss": float(_last_scalar_metric(metrics["ppo/total_loss"])),
        "tuning/final_actor_loss": float(_last_scalar_metric(metrics["ppo/actor_loss"])),
        "tuning/final_entropy": float(_last_scalar_metric(metrics["ppo/entropy"])),
        "tuning/final_approx_kl": float(_last_scalar_metric(metrics["ppo/approx_kl"])),
    }

    if "curriculum/score_mean" in metrics:
        summary["tuning/final_curriculum_score_mean"] = float(_last_scalar_metric(metrics["curriculum/score_mean"]))
    if "intrinsic_modules/rnd/predictor_loss" in metrics:
        summary["tuning/final_rnd_predictor_loss"] = float(
            _last_scalar_metric(metrics["intrinsic_modules/rnd/predictor_loss"])
        )

    return summary


def _last_metric_slice(metric_value: Any, *, dtype: Any) -> np.ndarray:
    """Return the last recorded slice from a metric history as a NumPy array."""
    metric_array = np.asarray(metric_value, dtype=dtype)
    return metric_array[-1] if metric_array.ndim > 0 else metric_array


def _last_scalar_metric(metric_value: Any) -> float | int:
    """Return the final recorded value from a scalar metric history."""
    metric_array = np.asarray(metric_value)
    return metric_array.item() if metric_array.ndim == 0 else metric_array.reshape(-1)[-1].item()


def _dataclass_init_kwargs(instance: Any) -> dict[str, Any]:
    """Return a deep-copied mapping of init-enabled dataclass fields."""
    return {
        field.name: copy.deepcopy(getattr(instance, field.name)) for field in dataclasses.fields(instance) if field.init
    }


def _clone_train_config(config: TrainConfig) -> TrainConfig:
    """Clone a TrainConfig by rebuilding it from its constructor fields."""
    return TrainConfig(**cast(Any, _dataclass_init_kwargs(config)))


def _derive_trial_seed(base_seed: int, run_id: str) -> int:
    """Derive a deterministic per-trial seed from the base seed and W&B run id."""
    seed_material = f"{base_seed}:{run_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(seed_material).digest()[:4], byteorder="big", signed=False)


def _set_nested_attr(instance: Any, dotted_path: str, value: Any) -> None:
    """Set an attribute on an object, supporting dotted paths for nested config fields."""
    path_parts = dotted_path.split(".")
    target = instance
    for path_part in path_parts[:-1]:
        if not hasattr(target, path_part):
            raise KeyError(f"Unsupported config path {dotted_path!r}.")
        target = getattr(target, path_part)

    final_part = path_parts[-1]
    if not hasattr(target, final_part):
        raise KeyError(f"Unsupported config path {dotted_path!r}.")
    setattr(target, final_part, value)


def _serialize_for_wandb(value: Any) -> Any:
    """Convert dataclass-backed config values into W&B-friendly plain Python containers."""
    if dataclasses.is_dataclass(value):
        return {
            field.name: _serialize_for_wandb(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.init
        }
    if isinstance(value, (tuple, list)):
        return [_serialize_for_wandb(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_for_wandb(item) for key, item in value.items()}
    return value


def run_single_trial(base_config: TrainConfig, save_results: bool) -> None:
    """Execute one sweep trial from sampled W&B overrides through final summary logging."""
    run = wandb.init()
    if run is None:
        raise RuntimeError("wandb.init() did not return a run.")

    try:
        overrides = dict(run.config)
        overrides.setdefault("train_seed", _derive_trial_seed(base_config.train_seed, run.id))

        trial_config = build_trial_config_from_overrides(base_config=base_config, overrides=overrides)
        run.config.update(
            {
                "train_seed": trial_config.train_seed,
                "resolved_config": _serialize_for_wandb(trial_config),
            },
            allow_val_change=True,
        )
        run.name = build_wandb_run_name(trial_config)

        train_info = run_main_algo_training(config=trial_config, save_results=save_results)
        summary = extract_trial_summary(train_info=train_info)
        summary["tuning/status"] = "completed"

        wandb.log(summary)
        run.summary.update(summary)
    except Exception as exc:
        run.summary["tuning/status"] = "failed"
        run.summary["tuning/error"] = repr(exc)
        raise
    finally:
        run.finish()


if __name__ == "__main__":
    main()
