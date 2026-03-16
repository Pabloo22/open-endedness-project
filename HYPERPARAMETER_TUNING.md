# Hyperparameter Tuning Guide

This guide documents the current workflow used by `src/crew/experiments/wandb_random_search.py` and the `src/crew/experiments/tuning_configs/` package.

## 1) What this script does

`wandb_random_search.py` wraps the training entrypoint (`run_main_algo_training`) in a W&B sweep agent.

For each trial it:
- Starts from a **base** `TrainConfig` (depends on `--tuning-phase`).
- Applies sampled W&B overrides (supports dotted keys like `rnd.predictor_network_lr`).
- Rebuilds `TrainConfig` so validation still runs.
- Runs one training job.
- Logs compact final metrics (including `tuning/objective_eval_return_mean`) to the trial run summary.

Important implementation detail: base tuning presets set `enable_wandb=False`, so inner training-loop logging is disabled. The sweep trial run itself is the single W&B run to avoid nested runs.

---

## 2) Typical commands

Create + run random search (100 trials):

```bash
poetry run python -m crew.experiments.wandb_random_search \
  --tuning-phase generic \
  --count 100
```

Create sweep only (no agent/trials):

```bash
poetry run python -m crew.experiments.wandb_random_search \
  --tuning-phase intrinsic \
  --intrinsic-modules rnd \
  --create-only
```

Reuse an existing sweep id:

```bash
poetry run python -m crew.experiments.wandb_random_search \
  --tuning-phase curriculum \
  --intrinsic-modules rnd \
  --sweep-id <entity/project/sweep_id>
```

Run grid search:

```bash
poetry run python -m crew.experiments.wandb_random_search \
  --tuning-phase generic \
  --method grid
```

With fixed overrides before sweep sampling:

```bash
poetry run python -m crew.experiments.wandb_random_search \
  --tuning-phase intrinsic \
  --intrinsic-modules rnd \
  --fixed-override num_envs_per_batch=512 \
  --fixed-override baseline_fixed_training_alpha='[0.8, 0.2]'
```

---

## 3) Tuning phases (current behavior)

### `generic`
- Uses `get_generic_base_config()`.
- Uses `get_generic_search_space()`.
- Tunes shared PPO/model hyperparameters.

### `intrinsic`
- Uses `get_intrinsic_base_config(<module>)`.
- Uses `get_intrinsic_search_space(<module>)`.
- Requires **exactly one** intrinsic module in `--intrinsic-modules`.
- Uses baseline training mode with fixed alpha vector including extrinsic + that intrinsic reward.

### `curriculum`
- Uses `get_curriculum_base_config_for_modules(<modules_tuple>)`.
- Uses `get_curriculum_search_space()`.
- Requires at least one intrinsic module and a matching active curriculum preset.

---

## 4) How the tuning config package is organized

`src/crew/experiments/tuning_configs/` contains:
- Versioned presets/search spaces per phase (`_generic_phase.py`, `_rnd_phase.py`, `_icm_phase.py`, `_ngu_phase.py`, `_curriculum_phase.py`).
- Active aliases in `_active_configs.py`.
- Public exports in `__init__.py`.

Recommended pattern (already used in files):
- Add new versioned constants/functions (`*_V2`, `*_V3`) instead of rewriting old versions.
- Move `ACTIVE_*` aliases to the new version when ready.

---

## 5) Add a new intrinsic reward function (end-to-end)

Example reward name below: `icm`.

### Step A — Add module implementation and registry wiring

1. Implement module API under `src/crew/main_algo/intrinsic_modules/`.
   - Follow `IntrinsicModule` protocol in `intrinsic_modules/api.py`.
2. Register the module in `intrinsic_modules/registry.py`:
   - Import your module class.
   - Add entry to `_REGISTRY`, e.g. `"icm": ICMIntrinsicModule()`.

Without this, `TrainConfig.selected_intrinsic_modules` validation fails.

### Step B — Add config support in `TrainConfig`

If your module needs nested config fields (like `rnd.*`), add:
- A dataclass for module hyperparameters in `src/crew/main_algo/config.py`.
- A new field in `TrainConfig` (e.g. `icm: ICMConfig = field(default_factory=ICMConfig)`).
- Validation logic in `_validate_selected_module_configs`.
- Module-specific discount/lambda behavior if needed when building per-reward vectors.

### Step C — Add phase tuning file

Create a new file in `src/crew/experiments/tuning_configs/` named:
- `_icm_phase.py` (pattern: `_{reward_name}_phase.py`)

Add at least:
- `get_icm_base_config_v1()` (module-specific base overrides).
- `get_icm_search_space_v1()` (module-specific sweep params, using dotted keys like `icm.some_param`).

### Step D — Activate the new module in tuning presets

Update `src/crew/experiments/tuning_configs/_active_configs.py`:
- Import your new phase functions.
- Add active intrinsic base config entry in `ACTIVE_INTRINSIC_BASE_CONFIGS`.
- Add active intrinsic search space entry in `ACTIVE_INTRINSIC_SEARCH_SPACES`.
- If curriculum with this module is supported, add tuple key(s) in `ACTIVE_CURRICULUM_BASE_CONFIGS`.

Notes:
- Keys for curriculum presets use a normalized sorted tuple internally.
- If no matching curriculum preset exists, curriculum tuning raises `ValueError`.

### Step E — Export from `tuning_configs/__init__.py`

Update `src/crew/experiments/tuning_configs/__init__.py`:
- Import new `get_*_base_config_v1` / `get_*_search_space_v1` symbols.
- Add them to `__all__` if meant to be public.

### Step F — Verify CLI usage

Once module is registered and active configs are wired:
- Intrinsic phase:

```bash
poetry run python -m crew.experiments.wandb_random_search \
  --tuning-phase intrinsic \
  --intrinsic-modules icm
```

- Curriculum with multiple modules:

```bash
poetry run python -m crew.experiments.wandb_random_search \
  --tuning-phase curriculum \
  --intrinsic-modules rnd icm
```

---

## 6) `wandb_random_search.py` argparse parameters

| Arg | Type | Default | Required | Meaning / notes |
|---|---|---:|---|---|
| `--count` | `int` | `10` | No | Number of trials for random method. Ignored for `--method grid`. |
| `--project` | `str` | `open_end_proj` | No | W&B project used for sweep + trial runs. |
| `--entity` | `str \| None` | `None` | No | Optional W&B entity/team. |
| `--group` | `str \| None` | `None` | No | Optional W&B group label shared across trials. |
| `--tuning-phase` | choice | — | **Yes** | One of `generic`, `intrinsic`, `curriculum`. Selects base config + search space builders. |
| `--intrinsic-modules` | `list[str]` (`nargs='+'`) | `DEFAULT_INTRINSIC_MODULES` | No | Modules included in trial config. Intrinsic phase requires exactly one. Curriculum uses one or more (with matching preset). |
| `--train-seed` | `int` | `0` | No | Base seed before per-run deterministic derivation from W&B run id. |
| `--total-timesteps` | `int` | `100000000` | No | Total env steps for each trial. |
| `--sweep-id` | `str \| None` | `None` | No | Reuse existing sweep instead of creating a new one. |
| `--create-only` | flag | `False` | No | Create sweep and exit without launching agent/trials. |
| `--save-results` | flag | `False` | No | Save full artifact/checkpoint from each trial. Can use significant disk space. |
| `--method` | choice | `random` | No | `random` or `grid`. For grid, all parameters should be discrete `values`. |
| `--fixed-override` | repeatable `KEY=VALUE` | `[]` | No | Apply fixed overrides before sweep sampling (supports dotted keys and JSON/Python literals). |

---

## 7) Important gotchas and current project state

1. **Only registered intrinsic modules are valid.**
   `TrainConfig` validates module names against `intrinsic_modules/registry.py`.

2. **`intrinsic` tuning phase currently enforces one module.**
   Passing multiple names raises an error by design.

3. **`curriculum` tuning requires active preset for the exact module tuple.**
   If your tuple is not in `ACTIVE_CURRICULUM_BASE_CONFIGS`, it fails fast.

4. **Changing `transformer_hidden_states_dim` auto-updates `qkv_features` only if you did not override `qkv_features`.**

5. **`fixed-override` parsing order:** JSON → Python literal (`ast.literal_eval`) → raw string.
   - Example: `baseline_fixed_training_alpha='[0.8, 0.2]'` becomes a list.

6. **Training config validation still applies after overrides.**
   Invalid divisibility or shape constraints fail at `TrainConfig` construction.

7. **Current active tuning maps now include `rnd`, `icm`, and `ngu` placeholders.**
   - `ACTIVE_INTRINSIC_BASE_CONFIGS` has entries for all three modules.
   - `ACTIVE_INTRINSIC_SEARCH_SPACES` has entries for all three modules.
   - `ACTIVE_CURRICULUM_BASE_CONFIGS` currently defines placeholder configs for single, pair, and triple module tuples.

8. **Intrinsic phase base configs use `training_mode="baseline"` (not `"intrinsic"`).**
   This matches `TrainConfig` supported modes and is required for valid config construction.

9. **`icm` and `ngu` tuning presets exist, but runtime still depends on module registration.**
   If `icm`/`ngu` are not registered in `src/crew/main_algo/intrinsic_modules/registry.py`, trials using them will fail validation.

10. **Sweep objective logged by this script:**
   - Primary objective: `tuning/objective_eval_return_mean`
   - Also logs additional final metrics (`ppo/*`, `eval/*`, curriculum/RND-specific when present).

---

## 9) Snapshot of active tuning maps (today)

Current active aliases in `_active_configs.py`:

- `ACTIVE_INTRINSIC_BASE_CONFIGS`: `rnd`, `icm`, `ngu`
- `ACTIVE_INTRINSIC_SEARCH_SPACES`: `rnd`, `icm`, `ngu`
- `ACTIVE_CURRICULUM_BASE_CONFIGS` keys:
  - `("rnd",)`
  - `("icm",)`
  - `("ngu",)`
  - `("rnd", "icm")`
  - `("rnd", "ngu")`
  - `("icm", "ngu")`
  - `("rnd", "icm", "ngu")`

The curriculum entries are generated with a helper that sets:
- `training_mode="curriculum"`
- `selected_intrinsic_modules=<tuple key>`
- `evaluation_alphas=(mixed_alpha, extrinsic_only_alpha)`

---

## 10) Suggested workflow for safe iteration

1. Start with `--create-only` to inspect sweep config in W&B UI.
2. Run a very small smoke sweep (`--count 1` or `2`).
3. Confirm final summary metrics and run naming look correct.
4. Scale up count.
5. Enable `--save-results` only when you actually need checkpoints/artifacts from each trial.
