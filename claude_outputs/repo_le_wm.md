# Phase 3: LeWorldModel Repository Analysis

A code-level map of `repos/le-wm/`, written so the Phase-5 implementation knows
exactly where the HWM-style hierarchical components plug in. References use
`file:line` for the exact spot. All paths below are absolute under
`/oscar/home/xguo84/final_project/repos/le-wm/`.

Top-level layout (only files we care about):

- `train.py`, `eval.py` — Hydra entry points.
- `jepa.py` — `JEPA` (low-level world-model wrapper).
- `module.py` — `SIGReg`, `Block`, `ConditionalBlock`, `Transformer`, `ARPredictor`,
  `Embedder`, `MLP`, `Attention`, `FeedForward`.
- `utils.py` — image/normalizer transforms + `ModelObjectCallBack`.
- `config/train/lewm.yaml`, `config/train/data/{pusht,ogb,dmc,tworoom}.yaml`,
  `config/train/launcher/local.yaml`.
- `config/eval/{pusht,cube,reacher,tworoom}.yaml`,
  `config/eval/solver/{cem,adam}.yaml`, `config/eval/launcher/local.yaml`.

There is **no** `data.py`, no `planner.py`, no policy file, no test directory in
le-wm itself — those are the additive surfaces Phase 4/5 will use.

---

## 1. Dataset loading and HDF5Dataset usage

The training entry point builds `swm.data.HDF5Dataset` once with
`transform=None`, then composes a transform list and assigns it after
inferring per-column normalizers.

- Construction: `train.py:54` calls `swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)`.
- Per-column transforms: `train.py:55-67`: `get_img_preprocessor` for `pixels`,
  then a `WrapTorchTransform`-based normalizer for every non-pixel key in
  `cfg.data.dataset.keys_to_load` (`utils.py:14-26`). The normalizer is fit on
  `dataset.get_col_data(col)`, with NaNs dropped, and also writes
  `cfg.wm.{col}_dim` so downstream model heads see the right dim
  (`train.py:65`).
- Frame-skip is handled at the dataset config level, not in code: see
  `config/train/data/pusht.yaml:3` (`frameskip: 5`) and
  `config/train/data/ogb.yaml:4`. The dataset emits `num_steps =
  history_size + num_preds` frames per sample (`pusht.yaml:2`,
  `ogb.yaml:3`); frame-skip 5 means each LeWM frame already corresponds to 5 env
  steps and each "action" entry is the 5-stride action block.
- Train/val split: `train.py:71-73` uses `spt.data.random_split` with a fixed
  generator seeded by `cfg.seed`.
- Loaders: `train.py:75-76`, batch size 128 from `lewm.yaml:24-28`.
- `keys_to_load` (e.g. `pusht.yaml:5-9`) controls which HDF5 columns are pulled
  per item; `keys_to_cache` is the eval-only persistence list
  (`pusht.yaml:10-13`).

For waypoints we keep `HDF5Dataset` as the raw source and add a wrapping
transform/dataset that re-samples within each item; we **do not** modify
`stable-worldmodel`.

## 2. Hydra config structure

- Train top-level: `config/train/lewm.yaml`. Defaults chain
  `lewm.yaml:1-3`: `_self_` then `data: pusht`. Launcher is composed via
  `config/train/launcher/local.yaml` which uses `# @package _global_` and
  overrides `hydra/launcher` — but it is only loaded when invoked with
  `+launcher=local` (the file itself isn't in `defaults`).
- Eval top-level: `config/eval/pusht.yaml:1-4` lists
  `defaults: [launcher: local, solver: cem, _self_]`. `cube.yaml:1-4` is
  identical except for env-specific settings.
- Solver group: `config/eval/solver/cem.yaml` and `adam.yaml`. The CEM file
  (`cem.yaml:1`) instantiates `stable_worldmodel.solver.CEMSolver` with
  `model: ???` set at runtime by `eval.py:94`
  (`hydra.utils.instantiate(cfg.solver, model=model)`).
- Data group (train only): `config/train/data/{pusht,ogb,dmc,tworoom}.yaml`
  — these populate `cfg.data.dataset.*`.
- Override style: hyphenated CLI flags as standard Hydra (e.g.
  `python eval.py --config-name pusht policy=<run_id>`). The `eval.py:49`
  decorator hard-codes `pusht` as default; `--config-name cube` swaps in
  `cube.yaml`.
- Hierarchical eval will need to add **two** new groups:
  `config/eval/solver/hcem.yaml` (or similar, since the current solver group is
  flat-only) and a top-level `pusht_hwm.yaml`/`cube_hwm.yaml` that pulls it in.

## 3. Model classes

`module.py` has the building blocks; `jepa.py` ties them together.

- `SIGReg` (`module.py:10-36`): registers `t`, `phi`, `weights` buffers; takes
  `proj` of shape `(T, B, D)` (note transpose in `train.py:41`) and returns the
  Epps–Pulley statistic averaged over projections. Defaults `num_proj=1024`,
  `knots=17`.
- `Attention` (`module.py:56-85`): standard pre-norm SDPA, `is_causal=True` by
  default — used both inside `Block` and `ConditionalBlock`.
- `Block` (`module.py:114-128`): plain pre-norm transformer block with
  `elementwise_affine=False` LayerNorms (because AdaLN modulates them).
- `ConditionalBlock` (`module.py:88-111`): same shape as `Block` but consumes
  conditioning `c` via `adaLN_modulation` projecting to `6*dim` and producing
  `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`. The
  output projection is **zero-initialized** (`module.py:102-103`) — that's the
  AdaLN-zero trick.
- `Transformer` (`module.py:131-187`): wraps a stack of `block_class` instances
  with optional input/cond/output projections. Routes `c` into
  `ConditionalBlock` blocks but skips it for plain `Block`s
  (`module.py:181-182`).
- `Embedder` (`module.py:189-214`): the **low-level action embedder**. Conv1d
  patch_embed (kernel 1, basically a per-step linear) + 2-layer MLP, mapping
  `(B, T, input_dim) -> (B, T, emb_dim)`. Importantly this acts on a single LeWM
  frame's action block (concatenated 5 primitive actions in PushT), not on a
  multi-frame chunk. **This is NOT HWM's macro-action encoder** — it doesn't
  compress across the temporal dimension and it doesn't have a CLS token.
- `MLP` (`module.py:217-241`): used for `projector` and `pred_proj` with
  BatchNorm1d (`train.py:106-115`).
- `ARPredictor` (`module.py:244-285`): `pos_embedding` of shape
  `(1, num_frames, input_dim)` + `Transformer(block_class=ConditionalBlock)`.
  Forward signature is `forward(self, x, c)` with `x: (B, T, d)`,
  `c: (B, T, act_dim)` — exactly the shape we need to swap for
  `(B, K, d_l)` macro-action conditioning.
- `JEPA` (`jepa.py:11-27`): holds `encoder`, `predictor`, `action_encoder`,
  `projector`, `pred_proj`. `JEPA.action_encoder` *is* `Embedder` — keep this
  named `low_level_action_embedder` in the high-level write-up to avoid
  confusion with HWM's `A_ψ`.
- `JEPA.encode` (`jepa.py:29-45`): runs encoder + projector to get `info["emb"]`
  and (if action present) `info["act_emb"]` via `action_encoder`.
- `JEPA.predict` (`jepa.py:47-55`): wraps `predictor` + `pred_proj`.
- `JEPA.rollout` (`jepa.py:61-110`): autoregressive rollout under a candidate
  action tensor of shape `(B, S, T, action_dim)` where S is sample count;
  history truncates at `history_size` (default 3, `lewm.yaml:47`). Stores result
  in `info["predicted_emb"]` of shape `(B, S, T, D)`.
- `JEPA.criterion` (`jepa.py:112-126`): MSE on the **terminal** step
  (`pred_emb[..., -1:, :]` vs `goal_emb[..., -1:, :]`), reduction='none' summed
  over remaining dims, returning per-sample cost `(B, S)`.
- `JEPA.get_cost` (`jepa.py:128-153`): the public hook used by
  `swm.policy.AutoCostModel`. Encodes goal pixels, runs `rollout`, then
  `criterion`.

## 4. Training script

`train.py` is the only training entry point.

- Forward: `lejepa_forward` (`train.py:18-46`). Reads
  `ctx_len=cfg.wm.history_size` (3) and `n_preds=cfg.wm.num_preds` (1) from
  `lewm.yaml:46-49`. Encodes the full clip, slices `ctx_emb = emb[:, :ctx_len]`
  and `tgt_emb = emb[:, n_preds:]`, predicts `pred_emb`, computes
  `pred_loss = MSE(pred_emb, tgt_emb)` (`train.py:40`). SIGReg is computed on
  the **transposed** embedding tensor `emb.transpose(0, 1)` -> shape `(T, B, D)`
  per `module.py:25-36`'s contract (`train.py:41`). Total loss is
  `pred_loss + lambd * sigreg_loss` with `lambd = cfg.loss.sigreg.weight = 0.09`
  (`lewm.yaml:60-61`).
- Model assembly: `train.py:82-124`. ViT via
  `spt.backbone.utils.vit_hf(cfg.encoder_scale, ...)` with `pretrained=False`,
  patch 14, image 224 (matches paper). `effective_act_dim =
  frameskip * action_dim` (`train.py:92`). `predictor`, `action_encoder`,
  `projector`, `pred_proj` are constructed individually then wrapped in `JEPA`.
- Optimizer: `train.py:126-133`. AdamW (`lewm.yaml:30-33`, lr 5e-5, wd 1e-3) +
  `LinearWarmupCosineAnnealingLR`, scheduled per epoch.
- Orchestration: `spt.Module(model=world_model, sigreg=SIGReg(...),
  forward=partial(lejepa_forward, cfg=cfg), optim=optimizers)` at
  `train.py:136-141`. The `spt.Manager` (`train.py:171-178`) wires up trainer +
  module + datamodule and writes its checkpoint to
  `<run_dir>/<cfg.output_model_name>_weights.ckpt`.
- Checkpointing: `ModelObjectCallBack` from `utils.py:28-57` saves a *pickle of
  the bare PyTorch model* (`torch.save(pl_module.model, path)`,
  `utils.py:55`) every epoch — this is what `swm.policy.AutoCostModel` later
  loads at eval time. Filename pattern:
  `<output_model_name>_epoch_<N>_object.ckpt`.

For high-level training, the cleanest pattern is a sibling
`train_highlevel.py` that mirrors this layout but loads a frozen `JEPA`
checkpoint and only optimizes the new `MacroActionEncoder` + a fresh
high-level `ARPredictor`.

## 5. Planning / evaluation code

`eval.py` is small and linear — easy to extend.

- `cfg` is loaded with `--config-name pusht` by default (`eval.py:49`).
- World env: `swm.World(**cfg.world, image_shape=(224, 224))` at
  `eval.py:58`.
- Image transforms: `img_transform(cfg)` at `eval.py:17-26` — Compose of
  `ToImage`, `ToDtype(float32, scale=True)`, `Normalize(ImageNet)`,
  `Resize(cfg.eval.img_size)`.
- Dataset is loaded purely for sampling start/goal frames; per-column
  StandardScalers fit on it (`eval.py:71-83`) and stored in `process` (with the
  `goal_*` mirroring trick at `eval.py:81-82`).
- Model load (`eval.py:88-92`):
  `model = swm.policy.AutoCostModel(cfg.policy)` -> the function at
  `stable_worldmodel/policy.py:556-574` scans the saved object for an attribute
  named `get_cost` and returns that module. Then `.to('cuda').eval()`,
  `requires_grad_(False)`, `interpolate_pos_encoding=True`.
- Plan/solver assembly (`eval.py:93-97`):
  - `swm.PlanConfig(**cfg.plan_config)` from `pusht.yaml:23-26`
    (`horizon=5`, `receding_horizon=5`, `action_block=5`).
  - `solver = hydra.utils.instantiate(cfg.solver, model=model)` — this is
    where we will branch: today there's exactly one solver, and the policy
    accepts exactly one solver.
  - `policy = swm.policy.WorldModelPolicy(solver=solver, config=config,
    process=process, transform=transform)`.
- Episode loop: `world.set_policy(policy)` then
  `world.evaluate_from_dataset(...)` at `eval.py:142-150` runs N=cfg.eval.num_eval
  episodes from sampled (start_step, goal=start_step+goal_offset) pairs;
  callables apply env-specific state-setting (`pusht.yaml:35-46`,
  `cube.yaml:40-58`).
- The constraint at `eval.py:52-54` (`horizon * action_block <= eval_budget`)
  means PushT default config plans 5 LeWM blocks × 5 frame-skip = 25 env
  steps per plan, equal to `eval_budget=50`'s headroom.

## 6. Existing CEM planner

- `config/eval/solver/cem.yaml:1-9` instantiates
  `stable_worldmodel.solver.CEMSolver` with `model=???` (filled in at runtime),
  `num_samples=300`, `n_steps=30`, `topk=30`, `var_scale=1.0`. These match the
  LeWM paper's PushT planning hyperparameters.
- The cost path is:
  1. `CEMSolver.solve` (`stable_worldmodel/solver/cem.py:115`) calls
     `model.get_cost(info_dict, action_candidates)` — the protocol any
     "cost model" must satisfy.
  2. That dispatches into `JEPA.get_cost` (`jepa.py:128-153`), which encodes
     the goal pixels, then calls `JEPA.rollout` (`jepa.py:61-110`) to fill
     `info["predicted_emb"]`, then `JEPA.criterion` (`jepa.py:112-126`).
  3. `JEPA.criterion` is currently **terminal MSE** between the predicted
     final latent and the goal latent (`jepa.py:120-124`). Per-sample cost
     `(B, S)` is returned.
- The key insight: because CEM only needs an object with `get_cost`, the
  hierarchical low-level pass can reuse `JEPA.get_cost` verbatim by passing it
  a **subgoal latent** in `info_dict["goal_emb"]` instead of computed goal
  pixels — but that requires either a small `criterion` refactor (accept a
  precomputed goal_emb and skip re-encoding) or a tiny wrapper module exposing
  `get_cost(info, actions)` that calls `model.rollout` then a custom
  L1-to-subgoal criterion. The latter is cleaner and zero-touch on `jepa.py`.

## 7. Where a high-level waypoint dataset should plug in

The clean insertion point is inside `train.py`'s data section, between the
`HDF5Dataset` construction (`train.py:54`) and the `transform` assignment
(`train.py:67-68`). Right now `transform = spt.data.transforms.Compose(*transforms)`
is a Compose of per-column ops that operate on a single dataset item — i.e. on
a clip of `num_steps = history_size + num_preds = 4` frames at frame-skip 5.

What changes for waypoints:

- The current data config (`config/train/data/pusht.yaml:2`) sets
  `num_steps = ${eval:'${wm.num_preds} + ${wm.history_size}'}` (= 4) and
  `frameskip: 5`. The waypoint loader needs to pull a *longer* sub-trajectory
  per item — say 5 waypoints × max-stride 14 LeWM blocks = up to 70 LeWM
  frames — then pick N indices with variable spacing. This means the
  underlying HDF5 sample window has to be wider, so a new
  `config/train/data/pusht_waypoints.yaml` is needed (extending `pusht.yaml`
  with a larger `num_steps` and adding waypoint-sampler hyperparameters), and
  the high-level training script (a sibling `train_highlevel.py`) will compose
  a new `WaypointSubtrajectoryDataset` transform that:
  - draws N waypoint indices `1 = t_1 < ... < t_N` per item under a
    min/max-gap constraint,
  - groups the action blocks between consecutive waypoints into padded chunks
    `(N-1, L_max, action_block_dim)` plus a boolean mask,
  - returns the waypoint frames as a `(N, C, H, W)` stack alongside the
    chunked actions.
- The transform shouldn't live in `stable-worldmodel` (read-only). It should
  go in a new `data.py` (or a `waypoint_transform` section of `utils.py`) and
  be added to the `transforms` list at the equivalent of `train.py:55-67`.
- Since the LeWM encoder is frozen at high-level training time, we can also
  pre-encode the entire training set's frames once and cache `(z_t, action)`
  triples to disk; this avoids running ViT in the high-level loop. That cache
  fits the existing `keys_to_cache` mechanism conceptually but requires writing
  a new cache key (latents) — a Phase-5 optimization, not needed for v1.

## 8. Where a high-level latent predictor should plug in

The change is a one-line architectural swap:

- Today: `predictor = ARPredictor(num_frames=cfg.wm.history_size,
  input_dim=embed_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
  **cfg.predictor)` at `train.py:94-100`. `forward(x, c)`: `x` is a sequence
  of latents `(B, T, D)`; `c` is per-step action embeddings `(B, T, D)` from
  `Embedder` (`module.py:189-214`).
- Tomorrow: same `ARPredictor` class instantiated identically, but the
  conditioning tensor `c` is the per-segment macro-action embedding
  `(B, K, d_l)` (lifted to `hidden_dim` via the existing `cond_proj` inside
  `Transformer` at `module.py:156-160`). No change to `ARPredictor`,
  `ConditionalBlock`, or `Transformer` is required — they already accept
  arbitrary `c` of dim `input_dim` (which is `embed_dim` for the predictor)
  and the `cond_proj` linear handles the lift if `d_l != embed_dim`. So
  effectively the high-level predictor is a fresh `ARPredictor(...)` (with
  `num_frames=N` instead of `history_size=3`) plus a new `MacroActionEncoder`
  whose output dim is whatever we want as the conditioning input dim.
- One real decision: do we (a) feed `c` of dim `d_l` directly and rely on
  `cond_proj`, or (b) pre-project `d_l -> embed_dim` inside the
  `MacroActionEncoder` head? Option (b) is cleaner for symmetry with the
  low-level path (where `Embedder` already outputs `embed_dim`). Either works
  and neither requires touching `module.py` core.
- The `HighLevelWorldModel` wrapper class can mirror `JEPA` (frozen
  `encoder`, `projector` / `pred_proj` reused as-is or fresh; new
  `predictor`; new `action_encoder = MacroActionEncoder`) so `get_cost` still
  has the exact signature `swm.policy.AutoCostModel` expects.

## 9. Where a hierarchical planner should plug in

Hierarchy is an **inference-time** abstraction; the cleanest plug-in is at
`eval.py:93-97`:

```python
config = swm.PlanConfig(**cfg.plan_config)
solver = hydra.utils.instantiate(cfg.solver, model=model)
policy = swm.policy.WorldModelPolicy(solver=solver, config=config, ...)
```

CEM lives in `stable_worldmodel.solver` (read-only), so the hierarchical
wrapper has to live in le-wm. Two integration shapes both work:

1. **New solver class in le-wm** (`planner.py`) named e.g.
   `HierarchicalCEMSolver` whose `solve(...)` runs (a) high-level CEM to get
   a subgoal latent, (b) low-level CEM toward that subgoal. Internally it
   instantiates two `stable_worldmodel.solver.CEMSolver` objects, each with
   its own cost callable. `eval.py:94`'s
   `hydra.utils.instantiate(cfg.solver, model=model)` then targets the new
   class via `config/eval/solver/hcem.yaml`. `eval.py` itself only needs to
   change so that `model` is a tuple/dict of `(low_level_model,
   high_level_model)` rather than a single module — most easily done by
   adding two `cfg.policy_low`/`cfg.policy_high` fields and loading two
   `AutoCostModel`s, and passing both into the hierarchical solver's
   constructor.
2. **New policy class in le-wm** (`HierarchicalWorldModelPolicy`) that
   subsumes both solvers internally and replaces the
   `swm.policy.WorldModelPolicy(...)` line. This keeps the solver group
   simpler (one CEM target each) but means duplicating `WorldModelPolicy`'s
   action-buffer/MPC plumbing (the deque at
   `stable-worldmodel/.../policy.py:357`).

Option 1 is the smaller, more localized diff and re-uses the existing
`WorldModelPolicy` MPC loop unchanged — recommended. Total `eval.py` diff is
~10 lines: load two checkpoints, build the hierarchical solver from them,
hand it to `WorldModelPolicy`. New configs add a fourth-level group
`hcem.yaml` plus top-level `pusht_hwm.yaml`/`cube_hwm.yaml`.

The cost-function detail from §6 matters here: the low-level inner CEM in the
hierarchy should score against a precomputed *subgoal* latent (not re-encode
goal pixels). That argues for a thin "cost adapter" object inside `planner.py`
that exposes `get_cost(info_dict, actions)` over a `JEPA` model but skips
re-encoding when `info_dict["goal_emb"]` is already populated, and (per HWM)
uses L1 instead of MSE. This keeps `jepa.py` itself unchanged.

---

## Integration cheat-sheet

Files that gain new code (additive, no edits to existing logic):

- `repos/le-wm/module.py` — add `MacroActionEncoder` next to `Embedder`. Pure
  addition; nothing else in `module.py` changes.
- `repos/le-wm/hwm.py` (new) — `HighLevelWorldModel` mirroring the public
  surface of `JEPA` (`encode`/`predict`/`rollout`/`criterion`/`get_cost`),
  built from a frozen `JEPA` plus a fresh `ARPredictor` and the new
  `MacroActionEncoder`.
- `repos/le-wm/data.py` (new) — `WaypointSubtrajectoryDataset` /
  `WaypointSampler` transform compatible with
  `spt.data.transforms.Compose`.
- `repos/le-wm/planner.py` (new) — `HierarchicalCEMSolver` (wraps two
  `stable_worldmodel.solver.CEMSolver` instances) plus a small
  `SubgoalCostAdapter` that exposes `get_cost(info, actions)` and
  uses L1 against a precomputed subgoal latent.
- `repos/le-wm/train_highlevel.py` (new) — sibling to `train.py`, loads a
  frozen `JEPA` checkpoint, instantiates the new modules, runs the L1
  teacher-forced loss on waypoint triples.
- `repos/le-wm/config/train/hwm.yaml`,
  `repos/le-wm/config/train/data/{pusht,ogb}_waypoints.yaml`.
- `repos/le-wm/config/eval/solver/hcem.yaml`,
  `repos/le-wm/config/eval/{pusht,cube}_hwm.yaml`.

Files that need a small edit:

- `repos/le-wm/eval.py` (~10 lines) — branch on whether `cfg.solver._target_`
  is hierarchical: if so, load two `AutoCostModel`s (`policy_low`,
  `policy_high`) and pass both into the hierarchical solver before
  `WorldModelPolicy` wrapping.

Files explicitly NOT touched (preserves flat-LeWM behaviour):

- `repos/le-wm/train.py` (low-level training) — unchanged.
- `repos/le-wm/jepa.py` — unchanged. (The L1-vs-MSE knob and the
  pre-encoded-goal shortcut both live in `planner.py`'s cost adapter, not in
  `JEPA.criterion`.)
- `repos/le-wm/utils.py` — unchanged.
- Anything in `repos/HWM_PLDM/`, `repos/stable-pretraining/`,
  `repos/stable-worldmodel/`.
