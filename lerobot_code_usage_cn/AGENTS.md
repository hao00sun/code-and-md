This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                        # All tests
uv run pytest tests/policies/test_act.py -svv                # Single test file
uv run pytest tests/policies/ -svv -k "test_name"            # Single test by pattern
DEVICE=cuda make test-end-to-end                             # All E2E tests
DEVICE=cuda make test-act-ete-train                          # Individual E2E (also: diffusion, tdmpc, smolvla; -train / -eval variants)
pre-commit run --all-files                                  # Lint + format (ruff, typos, bandit, etc.)
mypy --config-file=pyproject.toml src/lerobot/envs/         # Type-check a specific strict module
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions (`FeatureType`: STATE, VISUAL, ENV, ACTION, REWARD, LANGUAGE; `NormalizationMode`).

### Draccus config/CLI pattern

The entire CLI is auto-generated from dataclass configs by draccus. `TrainPipelineConfig` is the root config — every field on it and its sub-configs becomes a CLI flag using dotted paths (e.g., `--policy.type=act`, `--dataset.repo_id=lerobot/aloha`). Policy configs use `@register_subclass("name")` so `--policy.type=act` resolves to `ACTConfig`. The config is both the CLI interface and the serialization format — training saves `train_config.json` in each checkpoint, and you can resume with `--config_path=<checkpoint>/train_config.json --resume=true`. Use `--policy.path=<repo_or_dir>` to load pretrained weights (new config, pretrained weights) vs `--config_path` (full config restore, typically for resuming).

### Policy structure convention

Each policy follows a strict naming convention that drives the dynamic import system in `factory.py`:

- `configuration_<name>.py` — `<Name>Config` class (extends `PreTrainedConfig`, registers with draccus)
- `modeling_<name>.py` — `<Name>Policy` class (extends `PreTrainedPolicy`)
- `processor_<name>.py` — `make_<name>_pre_post_processors()` factory function

Every policy must define `config_class` and `name` class attributes. Third-party policies can be discovered by draccus's registry if they follow the same convention.

### Processor pipeline

Two `PolicyProcessorPipeline` instances wrap every policy:

- **Preprocessor** (`policy_preprocessor.json`): dataset batch → policy input tensor dict (normalization, image transforms, tokenization)
- **Postprocessor** (`policy_postprocessor.json`): policy output tensor dict → action (unnormalization, action chunking, relative→absolute conversion)

Each pipeline is a chain of `ProcessorStep` subclasses. The steps are serialized alongside checkpoints. When loading a pretrained policy, the processor pipelines are reconstructed from their JSON configs and re-wired (e.g., `AbsoluteActionsProcessorStep` needs a live reference to `RelativeActionsProcessorStep`).

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
