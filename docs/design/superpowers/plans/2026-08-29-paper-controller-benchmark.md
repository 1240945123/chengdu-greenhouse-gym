# Chengdu Paper Controller Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resource-safe, paper-ready controller benchmark that repairs rollout failure semantics, screens four additional RL methods, and compares frozen classical and RL controllers on 60-day Chengdu weather holdouts.

**Architecture:** A frozen YAML protocol feeds small modules for rollout recording, algorithm registration, screening, training, evaluation, statistics, and reporting. Every stage writes hash-addressed atomic artifacts under a new result root, and the pipeline refuses to advance when trajectories are incomplete or selections are missing.

**Tech Stack:** Python 3.12, Gymnasium, CasADi, pandas, NumPy, SciPy, Stable-Baselines3 2.8, sb3-contrib RecurrentPPO, PyTorch, pytest, YAML, Git.

---

## File Map

**Create**

- `configs/benchmarks/chengdu_paper_benchmark_v1.yml`: frozen data roles, horizons, budgets, algorithms, margins, and resource limits.
- `experiments/controllers/paper_benchmark_protocol.py`: validate and hash the protocol.
- `experiments/controllers/paper_algorithm_registry.py`: uniform SB3 and RecurrentPPO construction, save, resume, and inference metadata.
- `experiments/controllers/paper_screening.py`: screening matrix, validation aggregation, non-inferiority gates, and promotion.
- `experiments/controllers/paper_training.py`: generic checkpoint training with atomic metadata and replay-buffer handling.
- `experiments/controllers/paper_evaluation.py`: classical/RL rollout jobs, 60-day holdout roles, validation safety reference, and immutable trajectory reuse.
- `experiments/controllers/paper_statistics.py`: paired summaries, hierarchical descriptive bootstrap, effect sizes, and Holm correction.
- `experiments/controllers/paper_report.py`: paper CSV, Markdown tables, figures, and claim audit.
- `experiments/controllers/run_paper_benchmark.py`: resumable stage orchestration with three-worker cap.
- `tests/test_paper_benchmark_protocol.py`
- `tests/test_paper_algorithm_registry.py`
- `tests/test_paper_screening.py`
- `tests/test_paper_training.py`
- `tests/test_paper_evaluation.py`
- `tests/test_paper_statistics.py`
- `tests/test_paper_report.py`
- `tests/test_paper_pipeline.py`

**Modify**

- `.gitignore`: exclude local runtimes, caches, generated outputs, and model artifacts.
- `experiments/controllers/evaluation_failures.py`: make failed-step rows self-contained even when failure occurs on the first step.
- `experiments/controllers/run_v5_controller_smoke.py`: append failure transitions before breaking and summarize explicit terminal status.
- `tests/test_controller_evaluation_failures.py`: cover first-step and later-step failures.
- `tests/test_v5_controller_smoke.py`: cover classical and saved-RL failure recording.
- `README.md`: document paper benchmark commands, evidence roles, and result labels.

## Frozen Protocol Values

- Training weather: all valid 14-day windows in 2023-2024 spring/autumn.
- Validation weather: 2025 spring starts `59, 73, 87, 101, 115, 129, 143, 157`.
- Primary holdout: 2025 autumn 60-day starts `226` and `286`.
- Robustness: 60 days from each configured 2023-2025 spring/autumn season start.
- Screening algorithms: `td3`, `ddpg`, `a2c`, `recurrent_ppo`.
- Screening seeds/checkpoints: seeds `0, 1`; checkpoints `20_480, 102_400`.
- Full algorithms: `ppo`, `sac`, `residual_ppo` plus promoted algorithms.
- Full seeds/checkpoints: seeds `0, 1, 2`; checkpoints `20_480, 102_400, 307_200, 614_400`.
- Safety margins versus paired PID: mean active fraction `+0.05`; any-window active fraction `+0.10`.
- Climate margins versus PID: temperature-band MAE `+0.5 C`, RH-band MAE `+3.0 percentage points`, joint comfort `-0.05`.
- Resource cap: at most three workers; one PyTorch intra-op and inter-op thread per worker.

### Task 1: Establish a Trackable Source Baseline

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Write the repository-ignore test**

Add to `tests/repository_layout.py`:

```python
def test_local_runtime_and_generated_roots_are_ignored():
    ignored = set(Path(".gitignore").read_text(encoding="utf-8").splitlines())
    assert {".venv/", ".deps/", ".pip-cache/", ".pytest_cache/", ".tmp/", ".wandb/", "outputs/"} <= ignored
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/repository_layout.py::test_local_runtime_and_generated_roots_are_ignored -q`

Expected: FAIL because at least `.venv/` is absent.

- [ ] **Step 3: Add exact ignore entries**

Append:

```gitignore
.cache/
.deps/
.pip-cache/
.pytest_cache/
.tmp/
.venv/
.wandb/
outputs/
*.zip
*.pkl
*.pt
*.pth
```

- [ ] **Step 4: Verify ignores and create the source baseline commit**

Run: `.venv/Scripts/python.exe -m pytest tests/repository_layout.py -q`

Expected: PASS.

Run: `git status --short --ignored`

Expected: local runtime/data/result roots are marked `!!`, while source roots remain untracked.

Run:

```bash
git add .gitignore CITATION.cff LICENSE README.md pyproject.toml RL common configs docs experiments glassgym images post_processing processing run_scripts scripts tests visualisations
git commit -m "chore: establish project source baseline"
```

### Task 2: Add the Immutable Paper Protocol

**Files:**
- Create: `configs/benchmarks/chengdu_paper_benchmark_v1.yml`
- Create: `experiments/controllers/paper_benchmark_protocol.py`
- Create: `tests/test_paper_benchmark_protocol.py`

- [ ] **Step 1: Write failing protocol tests**

```python
def test_protocol_freezes_non_overlapping_sixty_day_holdouts():
    protocol = load_paper_protocol()
    assert protocol.holdouts == ((2025, 226, 60), (2025, 286, 60))
    assert protocol.screening_seeds == (0, 1)
    assert protocol.full_seeds == (0, 1, 2)
    assert protocol.max_workers == 3

def test_protocol_rejects_holdout_overlap(tmp_path):
    source = Path("configs/benchmarks/chengdu_paper_benchmark_v1.yml")
    path = tmp_path / "overlap.yml"
    path.write_text(
        source.read_text(encoding="utf-8").replace(
            "- [2025, 286, 60]", "- [2025, 285, 60]"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="holdout windows overlap"):
        load_paper_protocol(path)

def test_protocol_rejects_training_or_validation_weather_in_primary_holdout():
    protocol = load_paper_protocol()
    assert all(year < 2025 for year, _day in protocol.training_scenarios)
    assert not scenarios_overlap(protocol.validation_scenarios, protocol.holdouts)
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_benchmark_protocol.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the protocol model and validation**

Use immutable dataclasses:

```python
@dataclass(frozen=True)
class NonInferiorityMargins:
    safety_mean: float
    safety_window: float
    temperature_mae_c: float
    humidity_mae_pct: float
    comfort_fraction: float

@dataclass(frozen=True)
class PaperBenchmarkProtocol:
    training_scenarios: tuple[tuple[int, int], ...]
    validation_scenarios: tuple[tuple[int, int], ...]
    holdouts: tuple[tuple[int, int, int], ...]
    robustness_scenarios: tuple[tuple[str, int, int, int, str], ...]
    screening_algorithms: tuple[str, ...]
    screening_seeds: tuple[int, ...]
    screening_checkpoints: tuple[int, ...]
    full_seeds: tuple[int, ...]
    full_checkpoints: tuple[int, ...]
    minimum_full_checkpoint: int
    algorithm_hyperparameters: dict[str, dict[str, object]]
    margins: NonInferiorityMargins
    max_workers: int
    source_path: Path
    source_sha256: str
```

Reject overlapping holdouts, validation/holdout overlap, training years at or after validation year, duplicate seeds, nonpositive budgets, worker counts outside `1..3`, and missing algorithm hyperparameters. Serialize a resolved manifest with the source SHA256 and explicit evidence-role labels.

- [ ] **Step 4: Run protocol tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_benchmark_protocol.py tests/test_v5_full_training_protocol.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/benchmarks/chengdu_paper_benchmark_v1.yml experiments/controllers/paper_benchmark_protocol.py tests/test_paper_benchmark_protocol.py
git commit -m "feat: freeze paper benchmark protocol"
```

### Task 3: Record Every Failed Rollout Step

**Files:**
- Modify: `experiments/controllers/evaluation_failures.py`
- Modify: `experiments/controllers/run_v5_controller_smoke.py`
- Modify: `tests/test_controller_evaluation_failures.py`
- Modify: `tests/test_v5_controller_smoke.py`

- [ ] **Step 1: Write failing first-step and evaluator tests**

```python
FAILURE_INFO = {
    "reward": -999.0,
    "failure": True,
    "failure_kind": "numerical_integration",
    "failure_phase": "integration",
    "failure_exception": "RuntimeError",
    "failure_message": "mxstep",
    "failure_timestep": 3,
}

VALID_INFO = {"failure": False}

def make_minimal_normal_row(*, timestep, hour, reward, terminated,
                            truncated, inference_seconds, **_kwargs):
    return {
        "timestep": timestep, "hour_of_day": hour, "reward": reward,
        "air_temperature": 24.0, "relative_humidity": 75.0,
        "temperature_band_error": 0.0,
        "relative_humidity_band_error": 0.0, "joint_comfort": True,
        "proposed_uRoofVent": 0.0, "proposed_uFan": 0.0,
        "executed_uRoofVent": 0.0, "executed_uFan": 0.0,
        "safety_intervened": False, "strict_safety_intervened": False,
        "material_safety_intervened": False,
        "active_safety_intervened": False,
        "material_projection_max": 0.0, "active_projection_max": 0.0,
        "safety_intervention_reasons": "", "residual_fallback": False,
        "controller_inference_seconds": inference_seconds,
        "c_buffer_mg_m2": 1.0, "c_leaf_mg_m2": 1.0,
        "c_stem_mg_m2": 1.0, "c_fruit_mg_m2": 1.0,
        "terminated": terminated, "truncated": truncated,
        "numerical_failure": False,
        "terminal_status": "complete" if terminated else "running",
    }

def test_failure_row_can_be_first_transition():
    rows = []
    append_environment_failure_row(rows, FAILURE_INFO, timestep=0, previous_row=None)
    assert len(rows) == 1
    assert rows[0]["timestep"] == 0
    assert rows[0]["reward"] == pytest.approx(-999.0)
    assert rows[0]["truncated"] is True
    assert rows[0]["environment_failure"] is True

def test_classical_evaluator_appends_failure_before_break(monkeypatch):
    class FakeActionSpace:
        shape = (2,)

    class FakePolicy:
        def predict_action(self, _env):
            return np.zeros(2, dtype=np.float32)

    class FakeEnv:
        N = 4
        hour_of_day = 0.0
        action_space = FakeActionSpace()
        x = np.ones(28)

        def reset(self, seed=0):
            return np.zeros(2), {}

        def step(self, _action):
            self.hour_of_day += 0.25
            if self.hour_of_day == 1.0:
                return np.zeros(2), -999.0, False, True, FAILURE_INFO
            return np.zeros(2), -1.0, False, False, VALID_INFO

    monkeypatch.setattr(smoke, "build_v5_hybrid_environment", lambda **_kw: FakeEnv())
    monkeypatch.setattr(smoke, "_record_step", make_minimal_normal_row)
    metrics, trajectory = smoke.evaluate_classical_policy(
        algorithm="baseline",
        policy=FakePolicy(),
        growth_year=2025,
        start_day=226,
        episode_days=1,
    )
    assert len(trajectory) == 4
    assert trajectory.iloc[-1]["terminal_status"] == "numerical_failure"
    assert metrics["numerical_failure"] is True
    assert metrics["completed_episode"] is False
```

Add the equivalent saved-RL evaluator test.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_controller_evaluation_failures.py tests/test_v5_controller_smoke.py -q`

Expected: FAIL because evaluators break before appending failures.

- [ ] **Step 3: Implement self-contained failure records**

Change the helper signature to:

```python
def append_environment_failure_row(
    rows: list[dict[str, Any]],
    info: dict[str, Any],
    *,
    timestep: int,
    previous_row: dict[str, Any] | None = None,
) -> None:
    base = {
        "air_temperature": float("nan"),
        "relative_humidity": float("nan"),
        "temperature_band_error": float("nan"),
        "relative_humidity_band_error": float("nan"),
        "joint_comfort": False,
        "proposed_uRoofVent": float("nan"),
        "proposed_uFan": float("nan"),
        "executed_uRoofVent": float("nan"),
        "executed_uFan": float("nan"),
        "active_safety_intervened": False,
        "active_projection_max": float("nan"),
        "residual_fallback": False,
        "controller_inference_seconds": float("nan"),
        "c_buffer_mg_m2": float("nan"),
        "c_leaf_mg_m2": float("nan"),
        "c_stem_mg_m2": float("nan"),
        "c_fruit_mg_m2": float("nan"),
    }
    base.update(previous_row or (rows[-1] if rows else {}))
    base.update({
        "timestep": int(timestep),
        "reward": float(info["reward"]),
        "terminated": False,
        "truncated": True,
        "numerical_failure": True,
        "environment_failure": True,
        "terminal_status": "numerical_failure",
        "environment_failure_kind": str(info.get("failure_kind", "unknown")),
        "environment_failure_phase": str(info.get("failure_phase", "unknown")),
        "environment_failure_exception": str(info.get("failure_exception", "unknown")),
        "environment_failure_message": str(info.get("failure_message", "")),
        "environment_failure_timestep": int(info.get("failure_timestep", timestep)),
    })
    rows.append(base)
```

In both evaluator loops, call this helper before `break`. Add `terminal_status` to normal rows (`running`, `complete`, or `timeout`) and derive completion from expected length plus final status, not only `terminated`.
Update `summarize_smoke_trajectory` to return failure metrics without indexing
missing or empty columns. A first-step failure therefore produces one invalid,
non-finite failure row and a valid summary object instead of a secondary
`KeyError`.

- [ ] **Step 4: Run focused and solver-semantic tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_controller_evaluation_failures.py tests/test_v5_controller_smoke.py tests/test_solver_failure_semantics_v2.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/controllers/evaluation_failures.py experiments/controllers/run_v5_controller_smoke.py tests/test_controller_evaluation_failures.py tests/test_v5_controller_smoke.py
git commit -m "fix: preserve failed greenhouse rollout steps"
```

### Task 4: Prove 60-Day Solver Stability Before Training

**Files:**
- Create: `experiments/controllers/paper_evaluation.py`
- Create: `tests/test_paper_evaluation.py`

- [ ] **Step 1: Write failing trajectory-validation tests**

```python
def test_validate_rollout_requires_exact_expected_steps():
    result = validate_rollout(pd.DataFrame([COMPLETE_ROW] * 5759), expected_steps=5760)
    assert result.valid is False
    assert result.reason == "step_count_mismatch"

def test_validate_rollout_rejects_hidden_failure():
    frame = pd.DataFrame([COMPLETE_ROW] * 5759 + [FAILURE_ROW])
    result = validate_rollout(frame, expected_steps=5760)
    assert result.valid is False
    assert result.reason == "numerical_failure"
```

- [ ] **Step 2: Run and verify missing-module failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_evaluation.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement rollout identity and preflight**

```python
@dataclass(frozen=True)
class RolloutValidation:
    valid: bool
    reason: str
    expected_steps: int
    observed_steps: int

def validate_rollout(frame: pd.DataFrame, *, expected_steps: int) -> RolloutValidation:
    if len(frame) != expected_steps:
        return RolloutValidation(False, "step_count_mismatch", expected_steps, len(frame))
    if frame["numerical_failure"].astype(bool).any():
        return RolloutValidation(False, "numerical_failure", expected_steps, len(frame))
    if not np.isfinite(frame.select_dtypes(include=[np.number])).all().all():
        return RolloutValidation(False, "nonfinite", expected_steps, len(frame))
    if str(frame.iloc[-1]["terminal_status"]) != "complete":
        return RolloutValidation(False, "terminal_status", expected_steps, len(frame))
    return RolloutValidation(True, "complete", expected_steps, len(frame))
```

Add `run_classical_preflight()` for baseline and PID over all six fixed 60-day scenarios. Write each trajectory atomically and emit `preflight_summary.csv`; raise `RuntimeError` if any row is invalid.

- [ ] **Step 4: Run unit tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_evaluation.py -q`

Expected: PASS.

- [ ] **Step 5: Run the real 60-day preflight with one worker**

Run: `.venv/Scripts/python.exe -m experiments.controllers.paper_evaluation preflight --max-workers 1`

Expected: 12 complete rows, each with `steps=5760`, no numerical failures, and process exit 0. If this fails, stop here and diagnose the first preserved solver failure; do not start RL screening.

- [ ] **Step 6: Commit**

```bash
git add experiments/controllers/paper_evaluation.py tests/test_paper_evaluation.py
git commit -m "feat: gate training on sixty day stability"
```

### Task 5: Add a Uniform RL Algorithm Registry

**Files:**
- Create: `experiments/controllers/paper_algorithm_registry.py`
- Create: `tests/test_paper_algorithm_registry.py`

- [ ] **Step 1: Write failing registry tests**

```python
@pytest.mark.parametrize("name", ["ppo", "sac", "td3", "ddpg", "a2c", "recurrent_ppo"])
def test_registry_builds_supported_algorithm(name, tiny_vec_env):
    model = build_model(name, tiny_vec_env, seed=0, hyperparameters=tiny_hparams(name))
    assert model.device.type == "cpu"

def test_registry_declares_replay_and_recurrent_capabilities():
    assert get_algorithm_spec("sac").uses_replay_buffer
    assert get_algorithm_spec("td3").uses_replay_buffer
    assert get_algorithm_spec("ddpg").uses_replay_buffer
    assert get_algorithm_spec("recurrent_ppo").recurrent
    assert get_algorithm_spec("recurrent_ppo").policy == "MlpLstmPolicy"
```

Define `tiny_vec_env` as a `DummyVecEnv` containing a two-observation,
two-action `gym.Env` with eight-step episodes. Define `tiny_hparams(name)` in
the test file with `n_steps=8`, `batch_size=4` for on-policy methods and
`buffer_size=64`, `learning_starts=1`, `batch_size=4`, `train_freq=1`,
`gradient_steps=1` for replay methods. This keeps every parameterized case
below one second while exercising the real library classes.

Use these exact helpers:

```python
class TinyContinuousEnv(gym.Env):
    observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self):
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        terminated = self.steps == 8
        return np.zeros(2, dtype=np.float32), -float(np.square(action).sum()), terminated, False, {}

@pytest.fixture
def tiny_vec_env():
    env = DummyVecEnv([TinyContinuousEnv])
    yield env
    env.close()

def tiny_hparams(name):
    if name in {"ppo", "recurrent_ppo"}:
        return {"learning_rate": 1e-3, "n_steps": 8, "batch_size": 4,
                "n_epochs": 1, "gamma": 0.99}
    if name == "a2c":
        return {"learning_rate": 1e-3, "n_steps": 8, "gamma": 0.99}
    return {"learning_rate": 1e-3, "buffer_size": 64,
            "learning_starts": 1, "batch_size": 4, "tau": 0.01,
            "gamma": 0.99, "train_freq": 1, "gradient_steps": 1}
```

- [ ] **Step 2: Run and verify missing-module failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_algorithm_registry.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the registry**

```python
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from sb3_contrib import RecurrentPPO

@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    model_class: type
    policy: str
    uses_replay_buffer: bool = False
    recurrent: bool = False

REGISTRY = {
    "ppo": AlgorithmSpec("ppo", PPO, "MlpPolicy"),
    "sac": AlgorithmSpec("sac", SAC, "MlpPolicy", uses_replay_buffer=True),
    "td3": AlgorithmSpec("td3", TD3, "MlpPolicy", uses_replay_buffer=True),
    "ddpg": AlgorithmSpec("ddpg", DDPG, "MlpPolicy", uses_replay_buffer=True),
    "a2c": AlgorithmSpec("a2c", A2C, "MlpPolicy"),
    "recurrent_ppo": AlgorithmSpec("recurrent_ppo", RecurrentPPO, "MlpLstmPolicy", recurrent=True),
}
```

Expose `build_model`, `load_model`, `save_training_state`, and `load_training_state`. Replay algorithms save/load `replay_buffer.pkl`. Recurrent inference carries `(state, episode_start)` between steps and resets both at episode boundaries.

- [ ] **Step 4: Run registry tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_algorithm_registry.py -q`

Expected: PASS for all six algorithms.

- [ ] **Step 5: Commit**

```bash
git add experiments/controllers/paper_algorithm_registry.py tests/test_paper_algorithm_registry.py
git commit -m "feat: register continuous control rl algorithms"
```

### Task 6: Implement Generic Checkpoint Training and Resume

**Files:**
- Create: `experiments/controllers/paper_training.py`
- Create: `tests/test_paper_training.py`

- [ ] **Step 1: Write failing checkpoint tests**

```python
def test_checkpoint_is_reused_only_when_identity_matches(tmp_path, tiny_protocol):
    training_job = TrainingJob("td3", 0)
    first = train_checkpoints(training_job, protocol=tiny_protocol,
                              checkpoints=(64,), output_root=tmp_path)
    second = train_checkpoints(training_job, protocol=tiny_protocol,
                               checkpoints=(64,), output_root=tmp_path)
    assert first[0]["reused"] is False
    assert second[0]["reused"] is True

def test_replay_buffer_and_normalizer_are_required_for_resume(tmp_path, tiny_protocol):
    training_job = TrainingJob("ddpg", 0)
    train_checkpoints(training_job, protocol=tiny_protocol,
                      checkpoints=(64,), output_root=tmp_path)
    replay = next(tmp_path.rglob("replay_buffer.pkl"))
    replay.unlink()
    identity = checkpoint_identity(training_job, tiny_protocol, checkpoint=64)
    assert checkpoint_is_reusable(replay.parent, identity) is False
```

Define `tiny_protocol` in this test file by loading the real protocol and using
`dataclasses.replace` to set one 1-day training scenario, checkpoints `(64,)`,
and tiny hyperparameters from Task 5. The production loader remains unchanged.

```python
@pytest.fixture
def tiny_protocol():
    protocol = load_paper_protocol()
    tiny_algorithms = {
        name: tiny_hparams(name)
        for name in ("ppo", "sac", "td3", "ddpg", "a2c", "recurrent_ppo")
    }
    return dataclasses.replace(
        protocol,
        training_scenarios=((2023, 59),),
        screening_checkpoints=(64,),
        full_checkpoints=(64,),
        minimum_full_checkpoint=64,
        algorithm_hyperparameters=tiny_algorithms,
    )
```

- [ ] **Step 2: Run and verify missing-module failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_training.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement checkpoint identity and training**

Define `TrainingJob(algorithm, seed, residual_pid_scale=None)`. Build an identity from algorithm, seed, cumulative checkpoint, episode days, scenario list, hyperparameters, protocol SHA256, and environment model hashes. Save `model.zip`, `vecnormalize.pkl`, optional `replay_buffer.pkl`, and atomic `training_metadata.json`.

Use:

```python
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
model.learn(total_timesteps=target - model.num_timesteps,
            reset_num_timesteps=False,
            callback=AtomicProgressCallback(
                output_dir=checkpoint_root,
                interval_timesteps=32,
                unit_id=job.unit_id,
                target_checkpoint=target,
            ))
```

Reject a checkpoint if any required file is absent or any identity field differs. Never silently load a checkpoint from another protocol.

- [ ] **Step 4: Run training tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_training.py -q`

Expected: PASS, including TD3/DDPG replay resume and RecurrentPPO resume.

- [ ] **Step 5: Commit**

```bash
git add experiments/controllers/paper_training.py tests/test_paper_training.py
git commit -m "feat: add resumable paper benchmark training"
```

### Task 7: Implement PID-Relative Screening and Promotion

**Files:**
- Create: `experiments/controllers/paper_screening.py`
- Create: `tests/test_paper_screening.py`

- [ ] **Step 1: Write failing non-inferiority and promotion tests**

```python
def test_safety_gate_uses_paired_pid_windows():
    candidate = metrics(safety=[0.10, 0.20])
    pid = metrics(safety=[0.06, 0.12])
    result = assess_noninferiority(candidate, pid, MARGINS)
    assert result.safety_mean_pass
    assert result.safety_window_pass

def test_algorithm_ranking_uses_seed_mean_not_best_seed():
    rows = screening_rows({"td3": [-10, -100], "ddpg": [-40, -40]})
    ranking = rank_screening_algorithms(rows, pid_rows(), reference_rows())
    assert ranking.iloc[0]["algorithm"] == "ddpg"

def test_promotes_top_two_and_qualified_reference_beater():
    ranking = pd.DataFrame([
        {"algorithm": "td3", "feasible": True, "rank": 1,
         "reward_improvement_vs_best_reference": 0.01, "positive_pair_fraction": 0.70},
        {"algorithm": "recurrent_ppo", "feasible": True, "rank": 2,
         "reward_improvement_vs_best_reference": 0.01, "positive_pair_fraction": 0.70},
        {"algorithm": "a2c", "feasible": True, "rank": 3,
         "reward_improvement_vs_best_reference": 0.03, "positive_pair_fraction": 0.80},
    ])
    promoted = select_promotions(ranking)
    assert promoted == ("td3", "recurrent_ppo", "a2c")
```

- [ ] **Step 2: Run and verify missing-module failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_screening.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement gates and deterministic ranking**

For each candidate, join PID by `(growth_year, start_day)`. Require all health gates, mean/window safety margins, temperature/RH/comfort margins, and complete scenario coverage. Aggregate each algorithm across both screening seeds. Sort by mean reward descending; for algorithms within 1% of the top reward, use safety, action variation, and inference time ascending. Emit explicit pass/fail columns and reasons.

Select the top two feasible new algorithms. Add another only when reward beats the better PPO/SAC reference by at least 2%, at least 75% of paired seed-window differences are positive, and all non-inferiority gates pass.

- [ ] **Step 4: Run screening tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_screening.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/controllers/paper_screening.py tests/test_paper_screening.py
git commit -m "feat: add preregistered rl screening"
```

### Task 8: Evaluate Validation, Holdout, and Robustness Roles

**Files:**
- Modify: `experiments/controllers/paper_evaluation.py`
- Modify: `tests/test_paper_evaluation.py`

- [ ] **Step 1: Write failing evidence-role and reuse tests**

```python
def make_complete_row(timestep):
    return {
        "timestep": timestep, "reward": -1.0,
        "air_temperature": 24.0, "relative_humidity": 75.0,
        "temperature_band_error": 0.0,
        "relative_humidity_band_error": 0.0, "joint_comfort": True,
        "executed_uRoofVent": 0.0, "executed_uFan": 0.0,
        "active_safety_intervened": False, "active_projection_max": 0.0,
        "residual_fallback": False, "controller_inference_seconds": 0.001,
        "c_buffer_mg_m2": 1.0, "c_leaf_mg_m2": 1.0,
        "c_stem_mg_m2": 1.0, "c_fruit_mg_m2": 1.0,
        "terminated": False, "truncated": False,
        "numerical_failure": False, "terminal_status": "running",
    }

def test_primary_holdout_contains_two_frozen_sixty_day_windows(protocol):
    jobs = build_evaluation_jobs(protocol, role="primary_holdout", algorithms=("pid", "ppo"))
    assert {(j.start_day, j.episode_days) for j in jobs} == {(226, 60), (286, 60)}

def test_holdout_refuses_missing_frozen_selection(tmp_path):
    with pytest.raises(ValueError, match="missing frozen selection for ppo"):
        run_evaluation(
            protocol=load_paper_protocol(),
            output_root=tmp_path,
            role="primary_holdout",
            algorithms=("ppo",),
            selections={},
            max_workers=1,
        )

def test_incomplete_cached_trajectory_is_deleted_and_rerun(tmp_path, monkeypatch):
    job = EvaluationJob("primary_holdout", "2025_autumn_a", "pid", 0,
                        2025, 226, 60, None)
    cached = pd.DataFrame([make_complete_row(i) for i in range(5759)])
    write_trajectory_atomic(trajectory_path(tmp_path, job), cached)
    write_json_atomic(identity_path(tmp_path, job), evaluation_identity(job))
    replacement = pd.DataFrame([make_complete_row(i) for i in range(5760)])
    replacement.loc[5759, "terminal_status"] = "complete"
    monkeypatch.setattr(paper_evaluation, "execute_rollout", lambda _job: replacement)
    result = evaluate_job(job, output_root=tmp_path,
                          frozen_selections={}, force=False)
    assert result["steps"] == 5760
    assert result["artifact_reused"] is False
```

- [ ] **Step 2: Run and verify failures**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_evaluation.py -q`

Expected: FAIL with missing evidence-role validation and cache identity checks.

- [ ] **Step 3: Implement evaluation jobs and atomic persistence**

Use:

```python
@dataclass(frozen=True)
class EvaluationJob:
    role: str
    season_id: str
    algorithm: str
    seed: int
    growth_year: int
    start_day: int
    episode_days: int
    selection_sha256: str | None
```

Classical algorithms use seed 0. RL uses all three full seeds in final tables; the selected checkpoint is frozen per algorithm and seed using validation data. Store one trajectory and one sidecar identity JSON per job. A cache hit requires identity equality and `validate_rollout(frame, expected_steps=job.episode_days * 96).valid`.

Generate validation PID metrics before screening, primary holdout metrics only after `frozen_selections.json`, and robustness metrics only after primary artifacts are immutable.

- [ ] **Step 4: Run evaluation tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_evaluation.py tests/test_v5_full_evaluation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/controllers/paper_evaluation.py tests/test_paper_evaluation.py
git commit -m "feat: enforce paper benchmark evidence roles"
```

### Task 9: Build Statistical Comparisons Without Best-Seed Bias

**Files:**
- Create: `experiments/controllers/paper_statistics.py`
- Create: `tests/test_paper_statistics.py`

- [ ] **Step 1: Write failing statistics tests**

```python
def test_summary_uses_all_rl_seeds():
    summary = summarize_algorithm_metrics(metric_frame())
    assert summary.loc["ppo", "runs"] == 6
    assert summary.loc["ppo", "reward_mean"] == pytest.approx(np.mean(PPO_REWARDS))

def test_paired_differences_join_same_weather_window():
    paired = paired_against_reference(metric_frame(), reference="pid")
    assert set(paired["start_day"]) == {226, 286}

def test_holm_adjustment_is_monotonic_and_bounded():
    adjusted = holm_adjust([0.01, 0.04, 0.20])
    assert np.all((0 <= adjusted) & (adjusted <= 1))
    assert list(adjusted) == sorted(adjusted)
```

Define the fixture with explicit values:

```python
PPO_REWARDS = np.array([-90.0, -88.0, -92.0, -91.0, -87.0, -89.0])

def metric_frame():
    rows = []
    for start_day, pid_reward in ((226, -100.0), (286, -98.0)):
        rows.append({"algorithm": "pid", "seed": 0, "start_day": start_day,
                     "cumulative_reward": pid_reward, "completed_episode": True})
    for index, (seed, start_day) in enumerate(itertools.product(range(3), (226, 286))):
        rows.append({"algorithm": "ppo", "seed": seed, "start_day": start_day,
                     "cumulative_reward": PPO_REWARDS[index], "completed_episode": True})
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Run and verify missing-module failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_statistics.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement summaries and paired analysis**

Compute mean, sample SD, median, IQR, percentile 95% descriptive bootstrap CI, paired absolute/percentage differences, and Hedges' corrected standardized effect. Resample weather windows first and RL seeds second with a frozen seed and 10,000 draws. Label inference `descriptive_limited_two_weather_windows`; do not convert bootstrap intervals into strong population claims.

Implement Holm adjustment directly:

```python
def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = np.minimum(1.0, values[order] * (len(values) - np.arange(len(values))))
    ranked = np.maximum.accumulate(ranked)
    result = np.empty_like(ranked)
    result[order] = ranked
    return result
```

- [ ] **Step 4: Run statistics tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_statistics.py -q`

Expected: PASS with deterministic bootstrap outputs.

- [ ] **Step 5: Commit**

```bash
git add experiments/controllers/paper_statistics.py tests/test_paper_statistics.py
git commit -m "feat: add all-seed paired benchmark statistics"
```

### Task 10: Generate Paper-Ready Reports and Claim Audits

**Files:**
- Create: `experiments/controllers/paper_report.py`
- Create: `tests/test_paper_report.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing report tests**

```python
def test_report_labels_crop_outputs_as_simulated(tmp_path):
    report = build_report(report_metric_frame(), output_dir=tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "simulated crop indicator" in text
    assert "target harvest accuracy" not in text

def test_report_separates_primary_and_descriptive_roles(tmp_path):
    build_report(report_metric_frame(), output_dir=tmp_path)
    table = pd.read_csv(tmp_path / "paper_main_table.csv")
    assert set(table["evidence_role"]) == {"primary_holdout"}
```

Define the report fixture explicitly:

```python
def report_metric_frame():
    return pd.DataFrame([
        {"evidence_role": "primary_holdout", "algorithm": "pid", "seed": 0,
         "start_day": 226, "cumulative_reward": -100.0,
         "completed_episode": True, "fruit_state_change_mg_m2": 10.0},
        *[
            {"evidence_role": "primary_holdout", "algorithm": "ppo", "seed": seed,
             "start_day": 226, "cumulative_reward": reward,
             "completed_episode": True, "fruit_state_change_mg_m2": 11.0}
            for seed, reward in enumerate((-90.0, -91.0, -89.0))
        ],
        {"evidence_role": "descriptive_training_weather", "algorithm": "ppo",
         "seed": 0, "start_day": 59, "cumulative_reward": -95.0,
         "completed_episode": True, "fruit_state_change_mg_m2": 9.0},
    ])
```

- [ ] **Step 2: Run and verify missing-module failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_report.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement reports**

Generate:

- `paper_main_table.csv`: all-seed primary metrics.
- `paper_pairwise_pid.csv` and `paper_pairwise_mpc.csv`.
- `paper_robustness_table.csv`: explicitly descriptive rows.
- `screening_appendix.csv`: every screened seed and algorithm.
- `reasonableness_report.md`: failures, evidence roles, superiority checks, and claim limits.
- `figures/reward_by_algorithm.png`, `figures/climate_tradeoff.png`, `figures/safety_tradeoff.png`, and best-seed illustrative trajectories.

The report may state RL superiority only when both holdout windows have positive paired reward, all primary rollouts complete, and safety/climate non-inferiority passes. Otherwise use metric-specific language.

- [ ] **Step 4: Run report tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_report.py -q`

Expected: PASS.

- [ ] **Step 5: Document commands and commit**

Add README commands for `preflight`, `screen`, `train`, `holdout`, `robustness`, `report`, `status`, `pause`, and `resume`.

```bash
git add experiments/controllers/paper_report.py tests/test_paper_report.py README.md
git commit -m "feat: generate paper benchmark reports"
```

### Task 11: Add a Resource-Safe Resumable Pipeline

**Files:**
- Create: `experiments/controllers/run_paper_benchmark.py`
- Create: `tests/test_paper_pipeline.py`

- [ ] **Step 1: Write failing stage and resource tests**

```python
def test_stage_order_keeps_holdout_after_freeze():
    assert pipeline_stage_names() == (
        "source_preflight", "classical_validation", "rl_screening",
        "promotion_freeze", "full_training", "final_validation",
        "selection_freeze", "primary_holdout", "robustness", "report",
    )

def test_worker_count_is_capped_at_three():
    assert resolve_workers(requested=12, jobs=20, protocol_limit=3) == 3

def test_resume_skips_only_hash_matching_complete_stage(tmp_path):
    write_stage_status(tmp_path, "source_preflight", complete=True, matching_hash=True)
    assert stages_to_run(tmp_path)[0] == "classical_validation"
```

- [ ] **Step 2: Run and verify missing-module failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_pipeline.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement atomic stage orchestration**

Each stage writes `pipeline_status.json` with start/end timestamps, protocol SHA256, completed unit IDs, failed unit IDs, and traceback. A `PAUSE` file prevents new units from starting while allowing current units to checkpoint. Resume removes no artifacts and validates each completed unit before reuse.

Use `ProcessPoolExecutor(max_workers=min(requested, protocol.max_workers, jobs))`; worker initialization sets PyTorch threads to one. The launcher sets below-normal priority and bounded CPU affinity on Windows, but failure to set priority is recorded as a warning rather than changing experiment results.

- [ ] **Step 4: Run pipeline tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_paper_pipeline.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/controllers/run_paper_benchmark.py tests/test_paper_pipeline.py
git commit -m "feat: orchestrate resource safe paper benchmark"
```

### Task 12: Verify the Integrated System Before Screening

**Files:**
- No planned modifications. If a command fails, stop and create a separate
  failing-test task for that exact defect before changing source.

- [ ] **Step 1: Run the paper benchmark unit and integration tests**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_paper_benchmark_protocol.py tests/test_controller_evaluation_failures.py tests/test_v5_controller_smoke.py tests/test_paper_algorithm_registry.py tests/test_paper_training.py tests/test_paper_screening.py tests/test_paper_evaluation.py tests/test_paper_statistics.py tests/test_paper_report.py tests/test_paper_pipeline.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 2: Run the existing V5 regression suite**

Run:

```bash
.venv/Scripts/python.exe -m pytest tests/test_solver_failure_semantics_v2.py tests/test_v5_hybrid_environment.py tests/test_v5_full_training_protocol.py tests/test_v5_full_training.py tests/test_v5_full_evaluation.py tests/test_v5_full_pipeline.py -q
```

Expected: all tests PASS.

- [ ] **Step 3: Run short end-to-end algorithm smoke jobs**

Run:

```bash
.venv/Scripts/python.exe -m experiments.controllers.run_paper_benchmark smoke --algorithms td3 ddpg a2c recurrent_ppo --timesteps 256 --max-workers 1
```

Expected: each algorithm trains, saves, reloads, evaluates a complete short rollout, and emits finite metrics.

- [ ] **Step 4: Re-run and verify artifact reuse**

Run the same command again.

Expected: all four units report `reused=true`; no model file timestamp changes.

- [ ] **Step 5: Run the real preflight gate**

Run:

```bash
.venv/Scripts/python.exe -m experiments.controllers.run_paper_benchmark preflight --max-workers 1
```

Expected: all baseline/PID 60-day scenarios complete with 5,760 steps. Do not launch screening if any scenario fails.

- [ ] **Step 6: Commit verification evidence**

Write the command versions, pass counts, and preflight summary SHA256 to `docs/experiments/paper_benchmark_preflight.md`, then:

```bash
git add docs/experiments/paper_benchmark_preflight.md
git commit -m "docs: record paper benchmark preflight"
```

## Long-Run Launch Gate

Only after Task 12 passes:

```bash
.venv/Scripts/python.exe -m experiments.controllers.run_paper_benchmark screen --max-workers 3
```

Review and freeze `screening/promotions.json` before running:

```bash
.venv/Scripts/python.exe -m experiments.controllers.run_paper_benchmark train --max-workers 3
.venv/Scripts/python.exe -m experiments.controllers.run_paper_benchmark holdout --max-workers 3
.venv/Scripts/python.exe -m experiments.controllers.run_paper_benchmark robustness --max-workers 3
.venv/Scripts/python.exe -m experiments.controllers.run_paper_benchmark report --max-workers 1
```

Do not claim completion until the primary holdout trajectories validate, the all-seed statistics regenerate from stored trajectories, and `reasonableness_report.md` passes its claim audit.
