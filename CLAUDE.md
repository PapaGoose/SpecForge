# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpecForge is a framework for training speculative decoding draft models (EAGLE3, P-EAGLE, DFlash, Domino, DSpark), developed by the SGLang team. Trained models integrate directly with the [SGLang](https://github.com/sgl-project/sglang) serving framework.

This is a fork of `sgl-project/SpecForge`. Fork-local additions on top of upstream:

- **ClearML tracker** — `ClearmlTracker` in `specforge/tracker.py`, selected with `tracking.report_to: clearml`.
- **Periodic S3 upload during capture** — `scripts/prepare_hidden_states.py --clearml-upload --s3-output-uri ...` uploads each completed file group through ClearML `StorageManager` and deletes it locally, so long captures do not fill the disk.
- **`configs/llama3-8B-eagle3.json`** is modified for the fork's target model (hidden_size 5120, 40 attention heads, vocab_size 125800) and no longer matches stock Llama-3-8B.
- **`examples/configs/local/`** — fork-local run recipes. Upstream tests glob `examples/configs/*.yaml` non-recursively, so recipes here stay out of them.

## Installation

```bash
uv venv -p 3.11
source .venv/bin/activate
uv pip install -v . --prerelease=allow
```

```bash
uv pip install -v ".[fa]" --prerelease=allow
```

## Common Commands

**Run all tests:**
```bash
python -m unittest discover -s ./tests -p "test_*.py" -v
```

**Run a single test file:**
```bash
python -m unittest tests.test_config.test_recipe_readme -v
```

**Lint (pre-commit hooks):**
```bash
pre-commit run --all-files
```

**Train (the only public entry point):**
```bash
specforge train --config examples/configs/local/llama3-8b-eagle3-split-managed-local.yaml
```

**Inspect the process plan without starting workers:**
```bash
specforge train --config <config.yaml> --plan
```

**Standalone hidden-state capture (offline path), with S3 offload:**
```bash
python scripts/prepare_hidden_states.py --clearml-upload --s3-output-uri s3://bucket/path/to/run
```

**Export a trained checkpoint:**
```bash
specforge export --to sglang --checkpoint <ckpt> --draft-config <json> --output-dir <dir>
```

## Architecture

There is **one** training entry point, `specforge train`, driven by a typed YAML config (`specforge/config/schema.py`). The config selects an algorithm and a topology; it never selects a second trainer. `specforge/cli.py` parses the command and owns the distributed process lifecycle; `specforge/launch_plan.py` decides whether this process is a supervisor or a worker and self-launches the rest.

### Topologies

`specforge/launch.py` exposes exactly four builders:

- `build_offline_runtime` — colocated offline, reads precomputed feature files
- `build_disagg_offline_runtime` — offline over a shared dir or Mooncake
- `build_disagg_online_producer` — drives prompts against a patched SGLang capture server
- `build_disagg_online_consumer` — the trainer side of an online run

All trainer-bearing builders converge on `Trainer -> FeatureDataLoader -> TrainerController -> TrainerCore`. Only the reference source and the feature-store backend change.

**Online training is disaggregated only.** The target model runs in an external patched SGLang server, not in the trainer process. `deployment.disaggregated.managed_local` lets one command own the whole single-node stack: a loopback Mooncake master, the capture server(s) on the target GPUs (`capture_servers[].cuda_visible_devices` / `tp_size`), and the trainer workers on the draft GPUs (`trainer_cuda_visible_devices` / `deployment.trainer.nproc_per_node`). This is the supported replacement for the old in-process split-GPU mode.

### Package Structure (`specforge/`)

- **`cli.py`** — the `specforge` command: `train`, `export`, `benchmark`
- **`launch.py` / `launch_plan.py`** — topology builders and the supervisor/worker process plan
- **`config/`** — the typed run config (`Config`, `ModelConfig`, `TrainingConfig`, `DeploymentConfig`, …); strict, so unknown keys are errors
- **`application/`** — resolves a `Config` into a bound run (algorithm + topology)
- **`algorithms/`** — per-method models and providers: `eagle3/`, `peagle/`, `dflash/`, `domino/`, `dspark/`, plus `common/` and the registry
- **`runtime/`** — substrate only
  - `control_plane/` — prompt lifecycle, metadata ledger, distributed ack authority
  - `data_plane/` — feature stores, streaming ref channels, `RefDistributor`, `FeatureDataLoader`
  - `contracts.py` — the metadata/tensor boundary (`assert_no_tensors`)
- **`training/`** — `Trainer`, `TrainerController`, `TrainerCore`, FSDP backend, checkpointing, strategies, tracking adapter, model assembly
- **`inference/`** — rollout workers and server capture
- **`offline_capture/`** — standalone SGLang capture used by `scripts/prepare_hidden_states.py`, including the sglang patch
- **`modeling/`** — draft models, target head, `auto.py` dispatch
- **`data/`** — dataset building, prompt templates, parsing
- **`layers/`**, **`optimizer.py`**, **`lr_scheduler.py`** — model and optimization primitives
- **`distributed.py`** — process groups: TP, DP, draft DP, FSDP, sequence parallel (Ulysses/Ring); device-type aware (cuda/npu/cpu)
- **`tracker.py`** — W&B, TensorBoard, SwanLab, MLflow, ClearML, wired through `training/tracking.py`
- **`eval/`**, **`export/`**, **`benchmarks/`** — evaluation, checkpoint export, server benchmarking

### Checkpoints

`specforge/training/checkpoint.py` owns the layout: `{run_id}-step{N}/` under `output_dir`, with a rank-0 shared `training_state.pt` plus `training_state_rank{r}.pt` per rank, and `{run_id}-latest` / `{run_id}-best` pointers. Resume is step-granular via `resolve_resume_dir`; multi-rank runs need a shared filesystem.

### EAGLE3 Training Flow

1. Capture hidden states from the target model at 3 auxiliary layers (layer 1, `num_layers//2`, `num_layers-4`)
2. Concatenate and project: `(batch, seq_len, 3×hidden_size)` → `(batch, seq_len, hidden_size)`
3. Concatenate projected hidden states with embedding output: `(batch, seq_len, hidden_size×2)`
4. Train the draft model with test-time training (TTT)

## Adding Configuration

New fields on any typed config model must also be documented in `examples/configs/README.md` as `` `section.field` `` — `tests/test_config/test_recipe_readme.py` asserts the reference stays in sync with the schema. `tests/test_config/test_unified_feature_reachability.py` hardcodes the number of files in `examples/configs/*.yaml`, which is why fork-local recipes live in `examples/configs/local/`.

## Code Quality

- **Formatter**: `black` (code), `isort` (imports)
- **Linter**: `ruff` (F401), `autoflake`
- **C++/CUDA**: `clang-format`
- All checks enforced via pre-commit and GitHub Actions CI on PRs
