# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpecForge is a framework for training speculative decoding models (EAGLE3 and DFlash), developed by the SGLang team. Trained models integrate directly with the [SGLang](https://github.com/sgl-project/sglang) serving framework.

## Installation

```bash
uv venv -p 3.11
source .venv/bin/activate
uv pip install -v . --prerelease=allow

# With flash-attention support
uv pip install -v ".[fa]" --prerelease=allow
```

## Common Commands

**Run all tests:**
```bash
python -m unittest discover -s ./tests -p "test_*.py" -v
```

**Run a single test file:**
```bash
python -m unittest tests/test_data/test_parsers.py -v
```

**Lint (pre-commit hooks):**
```bash
pre-commit run --all-files
```

**Training (online EAGLE3, multi-GPU):**
```bash
torchrun --nproc_per_node=8 scripts/train_eagle3.py \
  --target-model-path <path> \
  --output-dir <output> \
  [--tensor-parallel-size 2] \
  [--use-fsdp]
```

**Data preparation:**
```bash
python scripts/prepare_data.py
python scripts/prepare_hidden_states.py
```

**Benchmarking:**
```bash
python benchmarks/bench_eagle3.py --model <path> --dataset mtbench
```

## Architecture

### Package Structure (`specforge/`)

- **`core/`** — Training loop logic
  - `eagle3.py` — `OnlineEagle3Model` and `OfflineEagle3Model` using test-time training (TTT)
  - `dflash.py` — DFlash training model
  - `loss.py` — Custom loss functions (`LogSoftmaxLoss`)
  - `eagle3_adapters.py` — Attention backend adapters (SDPA, USP/sequence-parallel)

- **`modeling/`** — Model definitions
  - `auto.py` — `AutoEagle3DraftModel` and `AutoDistributedTargetModel` (dispatch by config type)
  - `draft/` — Draft model implementations (currently `LlamaForCausalLMEagle3`)
  - `target/` — Target model wrappers
    - `custom_backend/` — Custom tensor-parallel implementations: Llama, Llama4, Qwen2, Qwen3, Qwen3MoE, Phi3, GptOss
    - `sglang_backend/` — SGLang server-based target model
    - `eagle3_target_model.py` — `Eagle3TargetModel` wrapper + `TargetHead` for hidden state projection

- **`data/`** — Dataset handling
  - `preprocessing.py` — `build_eagle3_dataset`, `build_offline_eagle3_dataset`
  - `parse.py` — Data parsing utilities
  - `template.py` — Prompt templates per model family

- **`layers/`** — Custom neural network layers with optimization
  - `ring/` — Ring attention for sequence parallelism

- **`distributed.py`** — Process group management: TP, DP, FSDP, sequence parallel (Ulysses/Ring)
- **`tracker.py`** — Experiment tracking: W&B, TensorBoard, SwanLab, MLflow, ClearML
- **`args.py`** — `TrackerArgs`, `SGLangBackendArgs` dataclasses
- **`optimizer.py`** — `BF16Optimizer`
- **`utils.py`** — General utilities

### EAGLE3 Training Flow

1. Extract hidden states from the target model at 3 auxiliary layers (layer 1, `num_layers//2`, `num_layers-4`)
2. Concatenate and project: `(batch, seq_len, 3×hidden_size)` → `(batch, seq_len, hidden_size)`
3. Concatenate projected hidden states with embedding output: `(batch, seq_len, hidden_size×2)`
4. Train draft model with test-time training (TTT)

### Distributed Training Dimensions

- **TP (Tensor Parallel)**: splits model weights across GPUs
- **DP (Data Parallel)**: replicates model, splits data
- **FSDP**: shards parameters, gradients, and optimizer states
- **Sequence Parallel**: Ring or Ulysses layouts via `yunchang`

### Adding a New Model

To add support for a new target model architecture:
1. Add a custom backend in `specforge/modeling/target/custom_backend/`
2. Register it in `specforge/modeling/auto.py` (`AutoDistributedTargetModel._model_mapping`)
3. Add draft model support in `specforge/modeling/draft/` if needed
4. Register in `AutoEagle3DraftModel._model_mapping`

## Code Quality

- **Formatter**: `black` (code), `isort` (imports)
- **Linter**: `ruff` (F401), `autoflake`
- **C++/CUDA**: `clang-format`
- All checks enforced via pre-commit and GitHub Actions CI on PRs
