# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Code for the paper "Much Ado About Noising: Dispelling the Myths of Generative Robotic Control". The `mip` package is a PyTorch behavior-cloning framework built on the stochastic interpolant view: one network interface, many training objectives (flow matching, regression, MIP = Minimum Iterative Policy, and flow-map / self-distillation variants), evaluated on Robomimic, Franka Kitchen and PushT.

## Commands

The project uses `uv` (Python 3.12, `uv.lock` is committed).

```bash
uv sync --extra dev                 # install with ruff + pytest

uv run pytest                       # unit tests live in tests/*_test.py
uv run pytest tests/losses_test.py
uv run pytest tests/agent_test.py::TestTrainingAgent::test_agent_different_loss_types

uv run ruff format .                # CI runs `ruff format --check .`
uv run ruff check --fix .           # CI runs `ruff check .`
```

CI (`.github/workflows/python-ci.yaml`) is exactly: `ruff format --check .`, `ruff check .`, `pytest`. The pre-commit hooks run the same two ruff commands through `uv run`.

### Training and evaluation

```bash
# Fast end-to-end check: 10 gradient steps, batch 2, 2 envs, wandb off, no auto-resume
uv run examples/train_robomimic.py -cn exps/debug.yaml task=lift_ph_state network=mlp

# Full runs (one script per benchmark)
uv run examples/train_robomimic.py task=lift_ph_state network=chiunet optimization.loss_type=flow
uv run examples/train_pusht.py task=pusht_state
uv run examples/train_kitchen.py task=kitchen_state      # needs: uv pip install "mujoco==3.1.6"

# Evaluate a checkpoint
uv run examples/train_robomimic.py task=lift_ph_state_abs mode=eval optimization.model_path=/path/to/ckpt.pt

# Sweep (Hydra multirun; launcher=cluster submits through submitit/SLURM)
uv run examples/train_robomimic.py task=lift_ph_state,can_ph_state --multirun
```

`examples/test_training/test_{robomimic,pusht,kitchen}.py` are standalone smoke-test scripts, not pytest tests: each one launches every task × network × {flow, mip, regression} combination through the debug config as a subprocess. Run with `uv run examples/test_training/test_robomimic.py`; they are slow and need the datasets and simulators.

### Docs

The documentation site is VitePress under `docs/` (`cd docs && npm ci && npm run docs:dev`), deployed to GitHub Pages on every push to `main`.

## Architecture

### How an objective is assembled

`TrainingAgent` (`mip/agent.py`) builds everything from the config. The single string `optimization.loss_type` selects three things that must stay in sync:

- `losses.get_loss_fn` — the training loss
- `samplers.get_sampler` — the matching inference procedure
- `samplers.get_default_step_list` — which `num_steps` values are evaluated (9, 3, 1 for flow-type objectives; 1 for `regression`, `mip`, `tsd`). It raises on unknown names, so a new objective must be registered here as well as in the other two.

Valid values: `flow`, `regression`, `straight_flow`, `tsd`, `mip`, `ctm`, `psd`, `lsd`, `esd`, `mf`, `lmd`. The math for each is in `docs/getting-started/design.md`.

Fixed signatures:

```python
loss_fn(config.optimization, flow_map, encoder, interpolant, act, obs, delta_t) -> (loss, info_dict)
sampler(config.optimization, flow_map, encoder, act_0, obs) -> act
network.forward(x, s, t, condition) -> (out, scalar)   # x: (B, Ta, act_dim); s, t: (B,)
encoder(obs, mask) -> condition                        # called as encoder(obs, None)
```

### Conventions that span files

- **Time runs noise → data.** `Interpolant` gives `I_t = α(t)·a_0 + β(t)·a_1` with `a_0` Gaussian noise at `t=0` and `a_1` the action chunk at `t=1`.
- **`FlowMap` wraps the network** as `X_{s,t}(x) = x + (t − s)·net(x, s, t, cond)`. `get_velocity(t, x, cond)` is the same network called with `s = t`. The network's second (scalar) output is discarded.
- **For `regression`, `mip` and `tsd` the output of `get_velocity` is the action prediction itself**, not a velocity to integrate. MIP is two such calls: one from a zero input at `t=0`, one from that prediction at `t = optimization.t_two_step`.
- **`delta_t` is a curriculum, not a step size.** `WarmupAnnealingScheduler` ramps it from 0 to 1 over training and the flow-map losses (`ctm`, `psd`, `lsd`, `esd`, `mf`, `lmd`) use it to cap the gap `t − s`. The other losses use it only as a shape/device template.
- **`lsd`, `esd`, `mf`, `lmd` differentiate through the network** with `torch.func.jvp` + `vmap` (`FlowMap.jvp_*`). These do not work with the transformer networks; use a UNet.
- **`ctm` and `lmd` call `flow_map.get_reference_velocity`**, which needs a `reference_net`. `TrainingAgent` builds `FlowMap(net)` without one; only `tests/losses_test.py` passes it.
- **`act_0` passed to `agent.sample` only supplies shape and device.** The start point comes from `optimization.sample_mode`: `zero` (the YAML default) starts from zeros and makes sampling deterministic, `stochastic` starts from Gaussian noise.
- **Networks are conditioned on encoder output, never raw observations.** `get_network` always builds the network with `obs_dim = network.emb_dim`, matching the `(B, obs_steps, emb_dim)` output of the `mlp`, `per_step_mlp` and image encoders (`identity` passes observations through unchanged). `obs_type: image` forces `MultiImageObsEncoder`; state and keypoint tasks use `network.encoder_type`.
- **Evaluation uses the EMA weights.** Observations are normalized with `dataset.normalizer`, the sampled chunk is unnormalized, and steps `[obs_steps − 1 : obs_steps − 1 + act_steps]` of the `horizon`-long prediction are executed through `MultiStepWrapper`.

### Configuration

Hydra composes `examples/configs/main.yaml` from the groups `task`, `network`, `optimization`, `log` and `launcher`. Things that are easy to get wrong:

- **The YAML files are the real defaults.** The dataclasses in `mip/config.py` are type hints for the training scripts (which receive a plain `DictConfig`) and are instantiated directly only by unit tests. Their defaults differ from the YAML (for example `loss_scale` 100 vs 0.1, `sample_mode` `stochastic` vs `zero`, `use_compile` true vs false). A new option goes in both places. This is also why code probes optional keys with `getattr(cfg, key, default)` / `hasattr`.
- **The config is mutated at startup.** `train_robomimic.py` overwrites `task.obs_dim` from the first env observation (or with `emb_dim` for images) and, for `network=chiunet`, rounds `task.horizon` up to a power of two.
- **Datasets download from HuggingFace** (`ChaoyiPan/mip-dataset`) when `task.dataset_repo` and `task.dataset_filename` are present, and that branch is checked before `task.dataset_path`. The Robomimic env is also built from the dataset file's metadata, so env and dataset always share one file.
- **Network YAMLs inherit `network/_base.yaml`** (encoder settings); Robomimic and PushT task YAMLs inherit `robomimic_base.yaml` / `pusht_base.yaml`.

### Robomimic action spaces

Task files are named `{env}_{ph|mh}_{state|image}[_{variant}]`. For state tasks the variant picks the action space through `task.abs_action` + `task.action_type`, and it changes the dataset file, the env controller and the eval post-processing together:

| Suffix | Dataset file | `act_dim` (single / dual arm) | Behaviour |
|---|---|---|---|
| none, `_abs` | `low_dim_abs.hdf5` | 10 / 20 | Absolute EEF pose with 6D rotation; controller switched to absolute. The paper's setting. |
| `_rel` | `low_dim_abs.hdf5` | 10 / 20 | Absolute actions re-expressed relative to the current EEF pose (`mip/action_utils.py`), converted back to absolute in the eval loop before stepping. |
| `_delta` | `low_dim.hdf5` | 7 / 14 | Raw Robomimic delta actions, `abs_action: false`. |
| `_delta_legacy` | `low_dim.hdf5` | 10 / 20 | Delta dataset pushed through the rot6d conversion while keeping the delta controller. Required to evaluate the released checkpoints. |

### Checkpoints and auto-resume

- Per run: `{log.log_dir}/models/model_{latest,best}.pt` and `{log.log_dir}/metrics.jsonl` (default `logs/`).
- Global: `checkpoints/{env_name}_{env_type}_{obs_type}_{loss_type}_{network_type}_{emb_dim}_seed{seed}_success{pct}.pt`, relative to the working directory. Only the highest-success file per name is kept. "Best" is judged on `mean_success_{N}` for the first entry of the step list (9 steps for flow-type objectives).
- `optimization.auto_resume` defaults to true and loads the matching global checkpoint, with optimizer and step count, at startup. The name does not encode the action-space variant or any other hyperparameter, so runs differing only in e.g. `_abs` vs `_rel` or learning rate resume from each other. Pass `optimization.auto_resume=false` or a different seed when that matters.

### Three parallel training scripts

`examples/train_{robomimic,pusht,kitchen}.py` are near-copies, each with its own `train` and eval loop, paired with `mip/datasets/{name}_dataset.make_dataset` and `mip/envs/{name}` `make_vec_env`. A change to the loop usually has to be mirrored in all three. `docs/development/add_new_task.md` describes porting a new benchmark this way; `add_new_method.md` and `add_new_architecture.md` cover objectives and networks.

`train_robomimic.py` sets `MUJOCO_GL=osmesa` at import time, which overrides whatever is exported in the shell.

### Compilation and CUDA graphs

`optimization.use_compile` compiles the whole update (forward, backward, clipping, optimizer step, EMA) and the sampler. `optimization.use_cudagraphs` additionally wraps the update in `CudaGraphModule` and keeps detached parameter copies that are re-synced before each `sample` call. CUDA graphs need static shapes: state observations only, `drop_last=True` in the dataloader, and in-place sampling inside losses (`torch.empty_like(x).uniform_()` / `.normal_()` rather than `torch.rand*`) — keep that pattern in new losses.

### Outside the `mip` package

- `toyexp/toyexp/` — self-contained toy experiments with their own README, configs and dependencies.
- `examples/process_dataset/` — scripts that produced the HuggingFace datasets (including the absolute-action conversion).
- `examples/pi0_pytorch.diff` — reference diff showing how to port the MIP loss and sampler into another flow-policy codebase.
- `mip/envs/kitchen/kitchen_thirdparty/` — vendored Franka Kitchen environment and assets.
