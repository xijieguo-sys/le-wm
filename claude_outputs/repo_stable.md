# Phase 3b: Shared Framework (`stable_pretraining`, `stable_worldmodel`) Reuse Map

Goal: enumerate exactly which pieces of `stable_pretraining` (`spt`) and
`stable_worldmodel` (`swm`) the current `repos/le-wm/` codebase already
imports, and decide which of them the HWM-style hierarchical extension can
reuse vs. has to write fresh inside `repos/le-wm/`.

All file:line references resolved against
`/oscar/home/xguo84/final_project/repos/stable-pretraining/` and
`/oscar/home/xguo84/final_project/repos/stable-worldmodel/`. Le-wm
imports collected by grepping `repos/le-wm/` for `stable_pretraining` /
`stable_worldmodel` / `from spt` / `from swm`. Hits in le-wm:

- `train.py:7,8,54,67,71,82,135,136,148,171`
- `eval.py:10,15,41,42,58,88,93,95,100,103`
- `utils.py:4`
- `config/eval/solver/{cem,adam}.yaml:1`

---

## 1. Shared backbone architectures le-wm depends on

### 1.1 `spt.backbone.utils.vit_hf` — the LeWM encoder factory
`stable_pretraining/backbone/utils.py:60-159`. Called from
`le-wm/train.py:82-88` as
```python
encoder = spt.backbone.utils.vit_hf(cfg.encoder_scale, patch_size=cfg.patch_size,
        image_size=cfg.img_size, pretrained=False, use_mask_token=False)
```
Returns a HuggingFace `transformers.ViTModel` (`add_pooling_layer=False`,
`use_mask_token=use_mask_token`); when `pretrained=True` it pulls
`google/vit-{size}-patch{patch_size}-{image_size}` from HF Hub, otherwise
builds from a `ViTConfig` whose dims are taken from a fixed size table at
`utils.py:107-129`. `tiny` ⇒ hidden_size=192, 12 layers, 3 heads — the LeWM
defaults. The factory always sets `model.config.interpolate_pos_encoding =
True` (`utils.py:157`), which is why `JEPA.encode` passes the
`interpolate_pos_encoding=True` kwarg at `le-wm/jepa.py:37`. **Reuse as-is**
for the high-level world model: when the HWM extension freezes the
low-level encoder, it just calls `requires_grad_(False)`/`eval()` on the
same `ViTModel`. No new factory needed.

### 1.2 Other `spt.backbone.*` / `spt.models.*`
Le-wm uses **only** `vit_hf`. The local `module.py` provides its own
transformer pieces (`Block`, `ConditionalBlock`, `Embedder`, `MLP`,
`ARPredictor`, local `SIGReg`) — there is no dependency on `spt.backbone.vit`,
`spt.backbone.aggregator`, `spt.backbone.mlp`, etc. `spt.TeacherStudentWrapper`
(also in `backbone/utils.py:336`) is **not** used (LeWM has no EMA target).
Implication: the high-level predictor can be built from the existing
`module.ARPredictor` / `module.Block` directly without touching `spt`.

### 1.3 `swm.World` (`stable_worldmodel/world/world.py:66-150`)
Constructed at `le-wm/eval.py:58` as `swm.World(**cfg.world,
image_shape=(224,224))`. Wraps an `EnvPool` of `num_envs` envs through
`MegaWrapper` and exposes `set_policy(policy)`, `evaluate(...)`. Eval mode
goes through `World.evaluate(dataset=..., episodes_idx=..., start_steps=...,
goal_offset=..., eval_budget=...)` — `world/world.py:163-230`, dispatching
to `_evaluate_from_dataset` at `world/world.py:440`. The hierarchical
policy lives **inside** the policy abstraction (`set_policy(policy)` only
needs an object with `get_action(info_dict)`), so `World` is unchanged —
**reuse as-is**.

NOTE: `le-wm/eval.py:142` calls `world.evaluate_from_dataset(...)` (not
`world.evaluate(...)`) — this name does not exist in
`world/world.py:163-230` (only `evaluate`/`_evaluate_from_dataset` do).
Either le-wm runs against a different swm version than the one in this
worktree, or the call site is outdated. Worth flagging when wiring up the
hierarchical eval — the same call would carry over verbatim.

### 1.4 `swm.PlanConfig` (`stable_worldmodel/policy.py:16-38`)
Frozen dataclass: `horizon`, `receding_horizon`, `history_len=1`,
`action_block=1`, `warm_start=True`. `plan_len = horizon * action_block`.
Instantiated at `eval.py:93` with `cfg.plan_config`. The CEM solver reads
`config.action_block` and `config.horizon` via `CEMSolver.action_dim` /
`CEMSolver.horizon` (`solver/cem.py:74-82`). For hierarchy we want **two**
configs (one per CEM level) — the dataclass is frozen but trivially
re-instantiable, so just build two `PlanConfig` objects. **Reuse as-is**.

### 1.5 `swm.policy.AutoCostModel` (`stable_worldmodel/policy.py:556-574`)
Loads a torch-pickled checkpoint (`spt.Module` written by
`utils.ModelObjectCallBack` at `le-wm/utils.py:53-56`), recursively scans
its children, and returns the first sub-module exposing a `get_cost`
attribute. In our codebase that's the `JEPA` instance
(`jepa.py:128-153`). `_load_model_with_attribute` at
`policy.py:479-532` does the disk lookup. For hierarchy we need to load
**two** checkpoints (low-level LeWM + high-level WM) — `AutoCostModel`
works for each independently, so **reuse twice**, once per level. The
high-level `JEPA`-shaped module just has to expose its own `get_cost`.

### 1.6 `swm.policy.WorldModelPolicy`
(`stable_worldmodel/policy.py:329-476`).
Constructor: `solver: Solver`, `config: PlanConfig`, `process: dict[k,
Transformable]`, `transform: dict[k, Callable]`. `set_env(env)` calls
`solver.configure(action_space, n_envs, config)` and allocates a
per-env action buffer of size `config.receding_horizon *
config.action_block` (`policy.py:362-377`). `get_action(info_dict)`
preprocesses obs through `_prepare_info` (normalizers + image transforms,
`policy.py:121-183`), figures out which envs need replanning, calls
`outputs = self.solver(sliced, init_action=...)`, slices off the first
`receding_horizon` actions to execute, optionally warm-starts the next
solve from the rest, and dequeues one primitive action per env per call.
**Reuse as the low-level (inner) policy**. The hierarchical wrapper sits
**outside** this — it owns the high-level CEM, computes a subgoal, then
delegates to a `WorldModelPolicy`-shaped inner policy whose model has been
re-cost-ed against the subgoal latent (see §3.2). Subclassing
`WorldModelPolicy` and overriding `get_action` is the natural insertion
point.

### 1.7 `swm.policy.RandomPolicy` (`policy.py:186-219`)
Used at `le-wm/eval.py:100` as the no-model baseline. Not relevant to the
hierarchical extension other than as a regression check.

### 1.8 `swm.data.HDF5Dataset`
(`stable_worldmodel/data/formats/hdf5.py:24-110`, re-exported via
`swm.data` at `data/__init__.py:28`). Constructor takes `name`,
`frameskip`, `num_steps`, `transform`, `keys_to_load`, `keys_to_cache`,
`keys_to_merge`, `cache_dir`, `path`. `_load_slice(ep_idx, start, end)`
returns a per-key dict; pixel arrays auto-permute to (T, C, H, W);
`'action'` is **not** decimated by `frameskip` (line 86-87) so the LeWM
"action block" of size 5 falls out for free. Inherits from
`Dataset` at `data/dataset.py:21-106`, which exposes
`column_names`, `get_col_data`, `get_dim`, `get_row_data`, and the
`(num_steps × frameskip)`-windowed `__getitem__`. Used at
`train.py:54` and `eval.py:42`. For HWM we need **waypoint sampling
across longer windows of variable stride** — `HDF5Dataset.__getitem__`
gives only fixed `(num_steps, frameskip)` windows. We will likely write a
thin wrapper `Dataset` (in a new `repos/le-wm/data.py`) that holds an
`HDF5Dataset` and resamples N waypoint indices per `__getitem__`. Reuse
the **reader** (the `_load_slice` plumbing, plus `get_col_data` for
normalizer fitting), **wrap** for waypoint logic.

### 1.9 `swm.data.utils.get_cache_dir`
`data/utils.py:14-27`. Resolves `STABLEWM_HOME` env or `~/.stable_worldmodel`,
optional `sub_folder`. Used at `train.py:148`, `eval.py:41,103`. **Reuse
as-is** for placing high-level checkpoints / waypoint-cache files under
the same cache root.

---

## 2. Shared utilities (losses, encoders, schedulers)

### 2.1 `spt.data.transforms.{Compose, ToImage, Resize, WrapTorchTransform}`
All in `stable_pretraining/data/transforms.py`:
- `Compose` (line 979-989) — sequential dict-in/dict-out application.
- `ToImage` (line 86-108) — wraps `to_image + ToDtype + Normalize` with
  source/target dict keys; called from `le-wm/utils.py:9` to build the
  pixel preprocessor.
- `Resize` (line 582-602) — `v2.Resize` with source/target keys; used at
  `le-wm/utils.py:11`.
- `WrapTorchTransform` (line 166-178) — wraps any callable, applies it to
  `source` key, writes to `target`; used at `le-wm/utils.py:25` to apply
  per-column normalizers (mean/std fit on the dataset).

These are agnostic to flat vs hierarchical training — the high-level
training loop encodes the same pixel observations with the same encoder,
so the same pipeline transfers verbatim. **Reuse as-is.**

### 2.2 `spt.data.dataset_stats.ImageNet`
`stable_pretraining/data/dataset_stats.py:9` — just a dict of ImageNet
mean/std. Used at `le-wm/utils.py:8` and `le-wm/eval.py:23` (as
`spt.data.dataset_stats.ImageNet`). **Reuse as-is.**

### 2.3 `LinearWarmupCosineAnnealingLR`
`stable_pretraining/optim/lr_scheduler.py:374-440`. Standard linear-warmup +
cosine-annealing scheduler taking `warmup_steps`, `max_steps`,
`warmup_start_lr`, `eta_min`. Recognised by `spt.Module.configure_optimizers`
via the registry at `lr_scheduler.py:73`, so `train.py:130`'s
`{"scheduler": {"type": "LinearWarmupCosineAnnealingLR"}}` resolves
correctly. **Reuse for high-level training**; the hyperparameters can be
copied from `train/lewm.yaml`.

### 2.4 `spt.data.random_split`
`stable_pretraining/data/utils.py:180-235`. Same semantics as
`torch.utils.data.random_split` but accepts fractions and rebalances
remainders. Used at `le-wm/train.py:71-73`. **Reuse as-is** for the
high-level training script's train/val split.

### 2.5 `spt.Module` (`stable_pretraining/module.py:21-660`)
PyTorch Lightning subclass with **manual** optimization
(`automatic_optimization=False`, line 103). Custom `forward(batch, stage)`
is bound from the user-supplied `forward=` kwarg (line 138-139). Supports
multi-optimizer config via `self.optim={name: {modules, optimizer,
scheduler, interval, frequency}}` with regex param matching
(line 32-67); a single-optim form (the form le-wm uses) is also accepted.
`training_step` (line 228) and `validation_step` (line 366) both delegate
to the user `forward`. Configures optimizers + schedulers in
`configure_optimizers` (line 528). `le-wm/train.py:136-141` instantiates
it as `spt.Module(model=..., sigreg=..., forward=partial(lejepa_forward,
cfg=cfg), optim=optimizers)`. **Reuse as-is** for the high-level training
script — `train_highlevel.py` will write a parallel `forward` (HWM Eq.1
L1 on next-waypoint latent), wire `model=HighLevelWorldModel` and let the
existing manual-optim plumbing handle gradient updates and scheduling.

### 2.6 `spt.Manager` (`stable_pretraining/manager.py:211-330+`)
`submitit.helpers.Checkpointable` subclass that owns a `pl.Trainer`,
`spt.Module`, and `pl.LightningDataModule`. Handles:
- Lightning `trainer.fit(...)` / `validate(...)` orchestration in
  `__call__`.
- Resume logic via `ckpt_path` (line 245-248) with sidecar metadata for
  wandb/trackio run-id continuity.
- `cache_dir` mode: stores a `_run_dir` so checkpoints live under the
  swm cache.
Used at `le-wm/train.py:171-178` as `manager = spt.Manager(trainer=...,
module=world_model, data=data_module, ckpt_path=...); manager()`. **Reuse
as-is** for high-level training; same incantation, different
`module`/`data`/`ckpt_path`.

### 2.7 `spt.data.DataModule` (`stable_pretraining/data/module.py:39+`)
Thin `pl.LightningDataModule` taking `train`/`val`/`test`/`predict`
DataLoader configs (or instantiated DataLoaders) and exposing the standard
hooks. Used at `le-wm/train.py:135`. **Reuse as-is.**

### 2.8 `swm`-side losses / regularizers
`stable_worldmodel.wm.loss` (`wm/loss.py:1-133`) provides `SIGReg`,
`VCReg`, `PLDMLoss`, `TemporalStraighteningLoss`. Le-wm does **not**
import this module — it uses its **own** `SIGReg` defined locally at
`le-wm/module.py:10`, imported via `from module import ARPredictor,
Embedder, MLP, SIGReg` (`train.py:14`). The two SIGReg implementations
look interchangeable (same Epps–Pulley sketch, same default knots/projs).
For HWM training there is **no new SIGReg**: the high-level model
inherits the latent space from the frozen LeWM encoder and is supervised
by L1 only (see Phase 1 §2.3, Phase 2 §4). Nothing in `swm.wm.loss`
needs to be added; if we ever want auxiliary regularizers (`VCReg`,
temporal-straightening), they are available off-the-shelf — but
**not required** for the planned design.

### 2.9 `swm.wm.lewm.LeWM` (`stable_worldmodel/wm/lewm/lewm.py:7-149`)
A near-duplicate of `repos/le-wm/jepa.py:JEPA` that lives **inside**
`stable-worldmodel`. Same `encode/predict/rollout/criterion/get_cost`
contract, with one notable difference: the swm-side `rollout` caches
`info['emb']` across calls (line 73-76), and `get_cost` caches
`info['goal_emb']` (line 126-139), making repeated CEM iterations cheaper.
The le-wm copy re-encodes every iteration. Useful as a cross-check that
our hierarchical `get_cost` interface matches a pre-existing pattern, but
the host project's hard rule is "ONLY modify files under `repos/le-wm/`",
so we must duplicate that caching pattern locally rather than swap to the
swm class.

---

## 3. Planning / latent-space utilities (most important section)

### 3.1 `swm.solver.CEMSolver` (`stable_worldmodel/solver/cem.py:15-241`)
Constructor (line 29-51) takes exactly the args called out in the task
spec: `model: Costable`, `batch_size=1`, `num_samples=300`,
`var_scale=1.0`, `n_steps=30`, `topk=30`, `device='cpu'`, `seed=1234`.
Pulls `dtype` from `next(model.parameters()).dtype` (line 49) — falls
back to `float32` if `model` is dtype-less.

`configure(action_space, n_envs, config: PlanConfig)` (line 53-66)
caches `config.action_block` to derive
`action_dim = action_space.shape[1:].prod() * config.action_block`
(line 75-76 — yes it multiplies by `action_block`). `horizon` reads
`config.horizon`. Action distribution lives at `(n_envs, horizon,
action_dim)` (line 95-112).

The **call interface** is exactly the contract assumed by the host plan:
inside the optimization loop at line 169-227, candidates of shape
`(current_bs, num_samples, horizon, action_dim)` are passed to
`self.model.get_cost(expanded_infos, candidates)` (line 190); costs must
come back as `(current_bs, num_samples)` — asserted at line 195-201.
Top-k drives mean/std updates (line 205-223), iterated `n_steps` times.
Output dict has `actions` (the final mean), `mean`, `var`, `costs`.

Two CEMSolver instances can coexist in the same process trivially: the
class holds only solver state plus a `torch.Generator`. **Reuse twice**
— one with a low-level `JEPA` model (action-block dim = 5×2 = 10 on
PushT) and one with a `HighLevelWorldModel`-shaped model whose
`get_cost(info, l_candidates)` rolls out the high-level predictor and
returns the L1 cost to `z_g`. Mirrors HWM Tab. 10 verbatim.

Caveat: the configure-time `action_dim = action_space_dim * action_block`
(line 76) is hardwired to the env's action space. For the **high-level**
solver, candidates live in macro-action space `R^{d_l}`, **not** in the
env's action space. The cleanest patch is to give the high-level
`HighLevelWorldModel.get_cost` a candidate tensor whose trailing dim is
already `d_l` and either (a) configure the high-level CEM with a custom
`action_space=Box(low,high,shape=(n_envs, d_l))` we synthesize, with
`PlanConfig(action_block=1)`, so that
`solver.action_dim == d_l`, or (b) subclass `CEMSolver` to bypass
`configure`'s coupling to the env action space. (a) is preferred — keeps
the change inside le-wm.

### 3.2 `swm.policy.WorldModelPolicy` slot for hierarchy
Already covered in §1.6. The cleanest insertion is a sibling class
`HierarchicalWorldModelPolicy(BasePolicy)` (in a new
`repos/le-wm/policy.py`) that:
- holds `low_solver: CEMSolver`, `high_solver: CEMSolver`,
  `low_model`, `high_model`, `low_config`, `high_config`;
- implements `set_env(env)`: calls `low_solver.configure(env.action_space,
  n_envs, low_config)` and `high_solver.configure(<synthetic d_l-dim
  Box>, n_envs, high_config)`;
- implements `get_action(info_dict)`: re-uses
  `WorldModelPolicy._prepare_info` (so we keep the same image transforms
  / normalizers — pull it up by inheriting from `BasePolicy`), runs the
  high-level CEM to produce the first subgoal latent z̃_1 = P^(2)(l*_1;
  z_1), then runs the low-level CEM with a re-targeted info_dict (replace
  `goal_emb` with z̃_1 instead of z_g) and dequeues `receding_horizon`
  primitive actions.

The action buffer / replan logic at `policy.py:401-468` is identical for
hierarchy (we still execute K = receding_horizon low-level steps before
re-planning), so **subclassing `WorldModelPolicy`** and overriding only
the inner solve call is feasible and keeps the diff small.

### 3.3 `swm.policy.AutoCostModel`
Already covered in §1.5. Per le-wm's existing flow
(`eval.py:88,model = swm.policy.AutoCostModel(cfg.policy)`), checkpoints
are looked up by `_load_model_with_attribute`
(`policy.py:479-532`) which takes a run name → checkpoint path under the
swm cache, `torch.load(weights_only=False)` it, then DFS for a child
exposing `get_cost`. For hierarchy we will load **both** the LeWM
checkpoint (for the inner level) and the high-level WM checkpoint (for
the outer level) via two `AutoCostModel(...)` calls. **Reuse twice.**

### 3.4 Latent-space cost helpers / distance functions / MPC schedulers
- **Cost helpers**: There is no shared `swm.cost.*` namespace. The cost
  function lives on the model (`get_cost`/`criterion`), and each `wm/*`
  model defines its own (`wm/lewm/lewm.py:104-145`,
  `wm/pldm/pldm.py:112-146`, `wm/prejepa/prejepa.py:364-469`). All use
  MSE on the last predicted latent vs goal latent. HWM wants L1 on the
  last predicted latent vs subgoal latent — small change in the local
  `JEPA.criterion` (le-wm/jepa.py:120). **Wrap locally.**
- **Distance functions**: not abstracted in swm — `JEPA.criterion` calls
  `F.mse_loss` directly. Adding `distance: 'mse'|'l1'` is a 3-line edit
  to `repos/le-wm/jepa.py:112-126`.
- **MPC schedulers**: handled implicitly by `WorldModelPolicy`'s
  `_action_buffer + receding_horizon + warm_start` (`policy.py:362-468`).
  Re-used as-is.

### 3.5 MPPI availability
Yes: `swm.solver.MPPISolver` (`stable_worldmodel/solver/__init__.py:5`,
implementation in `stable_worldmodel/solver/mppi.py:15-200+`) takes the
same constructor shape as CEM with an extra `temperature`. Same
`model.get_cost(info, candidates)` interface (mppi.py uses identical
`model.get_cost` call). Also available: `ICEMSolver`,
`PredictiveSamplingSolver`, `LagrangianSolver`, `PGDSolver`,
`GradientSolver`. So if we ever want to mirror HWM's MPPI-on-Diverse-Maze
configuration, it's literally a `_target_:
stable_worldmodel.solver.MPPISolver` swap in a yaml. For Push-T / Cube
both HWM and the planned extension stay on CEM, so MPPI is **available
but not needed**.

---

## 4. Reuse-vs-rewrite checklist

| Component we need | Available in `spt`/`swm`? | Decision (reuse / wrap / rewrite in `le-wm`) |
|---|---|---|
| Low-level encoder (E) | Yes — `spt.backbone.utils.vit_hf` (`backbone/utils.py:60`) returns HF `ViTModel`; loaded LeWM checkpoint already wraps it. | **Reuse** the trained instance (load via `swm.policy.AutoCostModel`, then `requires_grad_(False).eval()`). No new code. |
| Low-level predictor P^(1) | Yes — already lives in `repos/le-wm/jepa.py:JEPA` + `module.py:ARPredictor` (instantiated in `train.py:94`). | **Reuse** the trained instance, frozen. |
| Macro-action encoder A_ψ (HWM) | No — `spt.backbone.*` has no chunk-encoder; `module.Embedder` (le-wm) is per-step linear, not chunk-CLS-transformer. | **Rewrite** in `repos/le-wm/module.py` as new class `MacroActionEncoder` (small transformer + CLS token + MLP head). |
| High-level predictor P^(2) | Architecturally yes — `module.ARPredictor` accepts arbitrary `(emb, act_emb)` sequences. | **Reuse** `module.ARPredictor` class, instantiate a fresh copy with macro-action conditioning dim = `d_l`. |
| Waypoint dataset | Partially — `swm.data.HDF5Dataset` (`data/formats/hdf5.py:24`) gives episode reads; `Dataset.__getitem__` only does fixed-stride windows. | **Wrap** `HDF5Dataset` with a new `WaypointSubtrajectoryDataset` in a new `repos/le-wm/data.py` that resamples `N` waypoint indices per item and emits padded inter-action chunks. |
| Low-level CEM | Yes — `swm.solver.CEMSolver` (`solver/cem.py:15`) drives `model.get_cost` directly. | **Reuse as-is.** Same yaml as today (`config/eval/solver/cem.yaml`). |
| High-level CEM | Yes — same `swm.solver.CEMSolver`, **second instance**. | **Reuse**, with a synthetic `Box` action space of dim `d_l` and a `PlanConfig(action_block=1, horizon=H, receding_horizon=1)` so that `solver.action_dim == d_l`. New yaml `config/eval/solver/hcem.yaml` instantiating two CEMSolvers. |
| Hierarchical planner outer loop | No — neither `swm.policy` nor `swm.solver` has a two-level planner. | **Rewrite** as `repos/le-wm/planner.py:HierarchicalPlanner` orchestrating high+low CEM and exposing `plan(obs, goal_obs)`. |
| Hierarchical policy | Partially — `swm.policy.WorldModelPolicy` (`policy.py:329`) is single-solver. Its `_prepare_info`, action buffer, replan logic, and `set_env(env)` are level-agnostic and worth keeping. | **Wrap (subclass)** `WorldModelPolicy` in `repos/le-wm/policy.py:HierarchicalWorldModelPolicy`; override `get_action` to drive the two-level planner; reuse `_prepare_info` + `_action_buffer` machinery verbatim. |
| Hierarchical eval entry point | Partially — `eval.py` and `swm.World.evaluate(...)` already wire policy → world → metrics. | **Wrap**: keep `repos/le-wm/eval.py`'s structure; add a config-driven branch (`if cfg.solver._target_ resolves to HierarchicalWorldModelPolicy`) that loads two checkpoints via `swm.policy.AutoCostModel` and instantiates the hierarchical policy. New configs `config/eval/pusht_hwm.yaml`, `config/eval/cube_hwm.yaml`, `config/eval/solver/hcem.yaml`. |
| Distance function (L1 vs L2) | No abstraction — hard-coded `F.mse_loss` in `jepa.py:120`. | **Wrap locally**: add `distance: 'mse'\|'l1'` arg to `JEPA.criterion`. ~3 LOC. Default `'mse'` preserves flat behaviour. |
| Loss / regularizer for HWM | No new SIGReg needed (latent space inherited from frozen encoder). | **Reuse-by-omission.** Just teacher-forced L1 in the new `train_highlevel.py`. |
| Training scaffolding (Lightning module, manager, datamodule, scheduler, transforms) | Yes — `spt.Module`, `spt.Manager`, `spt.data.DataModule`, `spt.data.transforms.{Compose,ToImage,Resize,WrapTorchTransform}`, `spt.optim.lr_scheduler.LinearWarmupCosineAnnealingLR`, `spt.data.random_split`. | **Reuse all** in the new `train_highlevel.py`; the only thing different from `train.py` is the `forward` callable and the dataset. |
| MPPI solver (HWM Diverse-Maze) | Yes — `swm.solver.MPPISolver` (`solver/mppi.py:15`). | **Available but not used** for Push-T/Cube; nothing to do. |

Bottom line: of the ten components the host project wants to add, **six
are reuse-or-wrap of existing `spt`/`swm` symbols** (low-level encoder,
low-level predictor, low-level CEM, high-level CEM, hierarchical policy
shell, eval entry point); **four are genuinely new code in
`repos/le-wm/`** (`MacroActionEncoder`, `HighLevelWorldModel` wrapper +
its `get_cost`, `WaypointSubtrajectoryDataset`, `HierarchicalPlanner`).
This matches the Phase-2 deliverable list and confirms the
"keep changes minimal and local to LeWorldModel" engineering constraint
is achievable.
