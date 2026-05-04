# Phase 3: Merged Repository Analysis

Synthesis of three parallel sub-agent reports. Use this document as the single
reference for Phase 4 (architecture) and Phase 5 (implementation); the
per-repo files contain more depth if you need it.

Source documents
- `claude_outputs/repo_le_wm.md`     — `repos/le-wm/` (primary)
- `claude_outputs/repo_hwm.md`       — `repos/HWM_PLDM/`  (reference)
- `claude_outputs/repo_stable.md`    — `repos/stable-pretraining/` + `repos/stable-worldmodel/`

All paths absolute under `/oscar/home/xguo84/final_project/repos/`.

---

## 1. Cross-cutting conclusions (the headlines)

### 1.1 The integration is genuinely additive
- All four pieces of the framework that the host plan wanted to lean on really
  do exist and are reusable as-is: `spt.backbone.utils.vit_hf`,
  `spt.Module`/`spt.Manager`, `swm.data.HDF5Dataset`,
  `swm.solver.CEMSolver`, `swm.policy.WorldModelPolicy`,
  `swm.policy.AutoCostModel`, `swm.World`. None of them need to be modified.
- LeWM's existing classes are also already shaped right for hierarchy:
  `module.ARPredictor` accepts arbitrary `(x, c)` conditioning with an
  internal `cond_proj` that lifts `d_l → hidden_dim` (`module.py:156-160`),
  so the high-level predictor is a **fresh `ARPredictor(...)` instantiation**
  with no class change. `JEPA.get_cost` already matches the contract
  `swm.solver.CEMSolver` calls into (`solver/cem.py:190`), so the same
  contract works for the high-level world model.
- Six of ten new components are reuse-or-wrap; only **four are genuinely new
  code** in `repos/le-wm/`: `MacroActionEncoder`, `HighLevelWorldModel`
  wrapper exposing `get_cost`, `WaypointSubtrajectoryDataset`, and the
  `HierarchicalPlanner`/`HierarchicalWorldModelPolicy`.

### 1.2 The HWM_PLDM reference is a partial implementation
- It is the **PLDM-only minimal release**. Several of the paper's design
  choices are not in this code at all: variable-stride waypoint sampling,
  transformer-with-CLS macro-action encoder, L1 costs, Push-T/DINO-WM, CEM
  optimisation. The repo uses fixed-stride sampling, an MLP `A_ψ`, MSE costs,
  and MPPI.
- We must port the **recipe**, not the code. Specifically: the paper
  prescribes the right behaviours, the repo prescribes a working but
  PLDM-shaped instance of them. When the two disagree, follow the paper.
- Genuinely useful patterns from the repo (idea-level only):
  - "first subgoal = `l2_pred_obs[1]`" pattern
    (`HWM_PLDM/pldm/planning/planners/two_lvl_planner.py:73-75`).
  - Joint training of `A_ψ` and the high-level predictor via the next-state
    L1 loss alone — no separate `A_ψ` loss
    (`pldm/models/predictors/sequence_predictor.py:269-289`).
  - Optional Stage-2 final-mile flat fallback if hierarchy stalls near the
    goal (`pldm/planning/mpc.py:225-269`) — keep as a config-gated escape
    hatch, default off.

### 1.3 The "two action encoders" naming gotcha is real
LeWM's `JEPA.action_encoder` is **not** HWM's `A_ψ`.
- LeWM's `Embedder` (`module.py:189-214`) is a per-step Conv1d+MLP that
  embeds a single LeWM frame's 5-frameskip action block (10-D for PushT)
  into the predictor's conditioning dim. It does not compress across time
  and has no CLS token.
- HWM's `A_ψ` is supposed to be a transformer over a *variable-length chunk
  of LeWM action blocks*, with a CLS token, producing a single
  `d_l`-dimensional macro-action.
- Phase 5 must keep both, side-by-side: `Embedder` stays as
  `low_level_action_embedder`; new `MacroActionEncoder` lives next to it in
  `module.py`. **Do not refactor `Embedder`.**

### 1.4 Three latent-shape decisions are already settled
- LeWM latent: 192-d single CLS-token vector per frame, post-projector
  (`jepa.py:29-45`, projector is `MLP+BatchNorm` at `train.py:104-109`).
  HWM_PLDM's `(C, H, W)` conv feature map and its spatial action injection
  (`Expander2D`) do not apply.
- High-level latent space = same 192-d space (frozen encoder reused). No
  separate "L2 backbone" needed; the identity-encoder pattern from
  `HWM_PLDM/pldm/models/encoders/encoders.py:424-455` is unnecessary because
  LeWM's encoder already produces a single shared latent.
- Action conditioning at every layer is AdaLN
  (`module.ConditionalBlock`, `module.py:88-111`). This is preserved at the
  high level — the macro-action embedding is just a different `c` tensor of
  shape `(B, K, d_l)` lifted to `hidden_dim` by the existing `cond_proj`.
  Spatial-broadcast / `Expander2D` paths are not ported.

### 1.5 One existing-code edit is sufficient
The only flat-LeWM file that needs to change is `eval.py` (~10 lines: load
two `AutoCostModel` checkpoints, branch on `cfg.solver._target_`,
instantiate the hierarchical policy). Everything else is **new files** under
`repos/le-wm/`. This satisfies the Phase-2 constraint "preserve flat LeWM
behaviour identical when hierarchy is off". A small optional `distance:
'mse'|'l1'` arg on `JEPA.criterion` (3 LOC) is also possible but not
strictly required — the cost adapter inside `planner.py` can apply L1 over
the predicted latent without touching `jepa.py` at all.

### 1.6 One stale-call gotcha pre-existing in `eval.py`
`eval.py:142` calls `world.evaluate_from_dataset(...)`, but the worktree
`stable-worldmodel` only has `World.evaluate(...)` (with an internal
`_evaluate_from_dataset`) at `world/world.py:163-440`. Either le-wm runs
against a different swm version, or the call site is outdated. Either way
the hierarchical eval should match whatever convention the existing flat
eval uses (or fix it once and apply to both paths). Flag for Phase 5.

---

## 2. Per-repo summary

### 2.1 `repos/le-wm/` (primary)

Files that matter:
- Entry points: `train.py`, `eval.py`. Both Hydra-driven.
- Model code: `jepa.py` (the `JEPA` low-level world-model wrapper),
  `module.py` (`SIGReg`, `Block`, `ConditionalBlock`, `Transformer`,
  `ARPredictor`, `Embedder`, `MLP`).
- Utilities: `utils.py` (image transforms, `ModelObjectCallBack`).
- Configs: `config/train/lewm.yaml`, `config/train/data/*.yaml`,
  `config/eval/{pusht,cube,reacher,tworoom}.yaml`,
  `config/eval/solver/{cem,adam}.yaml`.

Existing data flow: `swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)`
constructed at `train.py:54`, then a `spt.data.transforms.Compose` of an
image preprocessor + per-column normalizers is assigned. Frame-skip 5 is
declared in `config/train/data/pusht.yaml:3` — the dataset emits
`num_steps = history_size + num_preds = 4` frames per item, each with the
matching action block.

Existing planning flow: `eval.py:88-95` loads the model with
`swm.policy.AutoCostModel(cfg.policy)` (which scans the pickled checkpoint
for a `get_cost` attribute and returns the matching submodule —
`swm/policy.py:556-574`), instantiates a single `solver` from `cfg.solver`
via `hydra.utils.instantiate`, and wraps it in `swm.policy.WorldModelPolicy`.
`world.evaluate_from_dataset(...)` runs `cfg.eval.num_eval` episodes from
sampled `(start_step, goal=start_step+goal_offset)` pairs.

Existing CEM flow: `config/eval/solver/cem.yaml` instantiates
`stable_worldmodel.solver.CEMSolver(model=..., num_samples=300, n_steps=30,
topk=30, var_scale=1.0)`. The optimization loop calls
`model.get_cost(info_dict, action_candidates)` (`solver/cem.py:190`); cost
must come back as `(current_bs, num_samples)`. `JEPA.get_cost` re-encodes
the goal pixels every CEM call — the swm-side `LeWM` class
(`stable_worldmodel/wm/lewm/lewm.py:7-149`) caches `info['goal_emb']` for
speed, but le-wm's local `JEPA` does not. Worth caching locally inside the
hierarchical cost adapter.

### 2.2 `repos/HWM_PLDM/` (reference, READ-ONLY)

Five high-impact divergences from the paper, all of which we must override
when porting to LeWM:

| Concern | What HWM_PLDM does | What we must do for LeWM |
|---|---|---|
| Waypoint sampling | Fixed stride `l2_step_skip=10`, fixed N=6 (`d4rl.py:226-246`) | Variable-stride sampler matching paper (Push-T 25–70 env steps, N=5) |
| `A_ψ` architecture | 32-32 MLP over flattened fixed-length chunk (`misc.py:134-180`) | Transformer + CLS + MLP head (paper) |
| Cost distance | MSE in both training and planning (`prediction.py:83`, `objectives_v2.py:102,178`) | L1 (paper Eqs. 1, 2) |
| Optimiser | MPPI (`mppi_planner.py`, `mppi_torch.py`) | CEM (LeWM's existing baseline) |
| Subgoal advancement | Not implemented — re-runs L2 every `replan_every` | Default same; `gating` flag optional, default off |

Useful design patterns to keep:
- First subgoal extraction: `pred_obs[1]` (index 1, not 0) is the predicted
  next-waypoint latent; pass to L1 with `repr_input=True`
  (`two_lvl_planner.py:73-75`).
- Joint training of `A_ψ` and high-level predictor via end-to-end
  next-latent loss only (no separate `A_ψ` regulariser at training time;
  light L2 reg `z_reg_coeff=0.1` only applies at MPPI sampling time, which
  doesn't apply to our CEM path).
- Stage-1 / Stage-2 split with a flat planner on the last 15 env steps
  (`mpc.py:225-269`) — config-gate as optional fallback.

Hyperparameters that are PLDM-tuned (do not copy):
- MPPI `noise_sigma=10` for `z_dim=8` after `LayerNorm` — different optimiser, different action norm.
- `num_samples=2000–4000` at L2 over horizons of 18 macro-steps —
  PLDM-specific predictor cost; LeWM-CEM hyperparameters should follow
  HWM Tab. 10 (Push-T `d=50`: high-level 1500 samples, 40 iters, pred H=4;
  low-level 900 samples, 20 iters, pred h=5, replan k=5).
- `replan_every=4` env steps × `action_repeat=4` env-wrapper = 16 raw env
  steps. LeWM has no `action_repeat`; replan cadence should be expressed in
  LeWM action blocks (`k=5` LeWM blocks = 25 env steps).

### 2.3 `repos/stable-pretraining/` + `repos/stable-worldmodel/` (frameworks)

Reusable as-is:
- `spt.backbone.utils.vit_hf` (`stable-pretraining/backbone/utils.py:60-159`) — the LeWM encoder factory; HF `ViTModel` with `interpolate_pos_encoding=True`.
- `spt.Module` (`stable-pretraining/module.py:21-660`) — Lightning subclass
  with manual optimization; binds a user `forward(batch, stage)` and routes
  multi-optim configs.
- `spt.Manager` (`stable-pretraining/manager.py:211-330+`) —
  submitit/Lightning orchestrator with resume + cache-dir logic.
- `spt.data.DataModule`, `spt.data.random_split`,
  `spt.data.transforms.{Compose, ToImage, Resize, WrapTorchTransform}`,
  `spt.data.dataset_stats.ImageNet`,
  `spt.optim.lr_scheduler.LinearWarmupCosineAnnealingLR`.
- `swm.World`, `swm.PlanConfig`, `swm.policy.WorldModelPolicy`,
  `swm.policy.AutoCostModel`, `swm.policy.RandomPolicy`,
  `swm.data.HDF5Dataset`, `swm.data.utils.get_cache_dir`.
- `swm.solver.CEMSolver` — twice, once per level. Same constructor shape;
  `model.get_cost(info, candidates)` is the only contract
  (`solver/cem.py:190`). Two instances coexist trivially in one process.

One CEMSolver coupling to handle: `CEMSolver.configure` derives
`action_dim = action_space.shape[1:].prod() * config.action_block`
(`solver/cem.py:75-76`) from the env's action space. The high-level CEM
operates in macro-action space `R^{d_l}`, **not** env action space. Cleanest
fix is to feed the high-level solver a synthetic `gym.spaces.Box` of dim
`d_l` plus `PlanConfig(action_block=1, horizon=H, receding_horizon=1)` so
that `solver.action_dim == d_l`. No subclass needed.

Distance abstraction (L1 vs L2): not provided by `swm`; cost lives on each
model's `get_cost`/`criterion`. All shipped models (`wm/lewm/lewm.py`,
`wm/pldm/pldm.py`, `wm/prejepa/prejepa.py`) inline `F.mse_loss`. The L1
switch is local to LeWM either via a new `distance` arg on
`JEPA.criterion` (3 LOC) or via a thin cost adapter inside `planner.py`
(zero touch on `jepa.py`).

MPPI is available (`swm.solver.MPPISolver`, `solver/mppi.py:15-200+`) with
the same contract; a yaml swap would be enough to mirror HWM's Diverse-Maze
configuration. Not needed for Push-T / Cube — keep CEM.

There is also a near-duplicate `swm.wm.lewm.LeWM` class
(`stable-worldmodel/wm/lewm/lewm.py:7-149`) that mirrors `repos/le-wm/jepa.py:JEPA`
but caches `info['emb']` and `info['goal_emb']` across calls
(`lewm.py:73-76, 126-139`). The host project's hard rule is "ONLY modify
files under `repos/le-wm/`", so we duplicate the caching pattern locally
rather than swap to the swm class. Useful as a cross-check that our
`get_cost` interface matches a pre-existing pattern.

---

## 3. Master reuse-vs-rewrite table

| Component we need | Available in `spt`/`swm`? | Decision |
|---|---|---|
| Low-level encoder (`E`) | Yes — `spt.backbone.utils.vit_hf` | **Reuse trained instance**; freeze via `requires_grad_(False).eval()`. |
| Low-level predictor (`P^(1)`) | Yes — `repos/le-wm/jepa.py:JEPA` + `module.py:ARPredictor` | **Reuse trained instance**, frozen. |
| Macro-action encoder (`A_ψ`) | No — `module.Embedder` is per-step linear, not chunk-CLS-transformer | **New class** in `repos/le-wm/module.py`: small transformer + CLS + MLP head. Variable-length via padding mask. |
| High-level predictor (`P^(2)`) | Architecturally yes — `module.ARPredictor` already takes arbitrary `(x, c)` | **Reuse class**, fresh instance with macro-action conditioning dim `d_l`. |
| Waypoint dataset | Partial — `swm.data.HDF5Dataset` reads episodes; only fixed-stride windows | **Wrap** with new `WaypointSubtrajectoryDataset` in `repos/le-wm/data.py`. Variable-stride sampler emits `(waypoint_pixels, inter_action_chunks, inter_action_mask)`. |
| Low-level CEM | Yes — `swm.solver.CEMSolver` | **Reuse as-is** with current `config/eval/solver/cem.yaml`. |
| High-level CEM | Yes — same `swm.solver.CEMSolver`, second instance | **Reuse**, with synthetic `gym.spaces.Box(shape=(d_l,))` and `PlanConfig(action_block=1, horizon=H, receding_horizon=1)`. New `config/eval/solver/hcem.yaml`. |
| Hierarchical planner outer loop | No two-level planner anywhere in `swm` | **New** `repos/le-wm/planner.py:HierarchicalPlanner` (or `HierarchicalCEMSolver`) wrapping two `CEMSolver` instances. |
| Hierarchical policy | Partial — `swm.policy.WorldModelPolicy` is single-solver but its `_prepare_info` / action-buffer / replan logic is level-agnostic | **Subclass** `WorldModelPolicy` as `HierarchicalWorldModelPolicy` in `repos/le-wm/policy.py`; override `get_action`. |
| Hierarchical eval entry point | Partial — `eval.py` already wires policy → world → metrics | **Edit `eval.py`** (~10 lines): load two `AutoCostModel`s, branch on hierarchical vs flat solver target. New configs `config/eval/{pusht,cube}_hwm.yaml`. |
| Distance function (L1 vs MSE) | No abstraction — `F.mse_loss` inlined in `jepa.py:120` | **Wrap locally**: prefer a thin cost adapter inside `planner.py` (zero touch on `jepa.py`). Optional 3-LOC `distance` arg on `JEPA.criterion` if convenient. |
| HWM training loss | No — repo uses MSE (`prediction.py:83`); paper specifies L1 | **New** `lejepa_highlevel_forward` in `train_highlevel.py`: teacher-forced L1 on next-waypoint latent (HWM Eq. 1). |
| Loss / regularizer for HWM | No new SIGReg needed — latent space inherited from frozen encoder | **Reuse-by-omission.** No regulariser at training time. If `‖l‖` collapses, add a tiny L2 reg on macro-actions (Phase-5 escape hatch). |
| Training scaffolding | Yes — `spt.Module`, `spt.Manager`, `spt.data.DataModule`, `spt.data.transforms.*`, `LinearWarmupCosineAnnealingLR`, `random_split` | **Reuse all** in new `train_highlevel.py`. Only the `forward` callable and the dataset differ from `train.py`. |
| MPPI solver (HWM Diverse-Maze) | Yes — `swm.solver.MPPISolver` | **Available but unused** for Push-T/Cube. |

---

## 4. File-level integration plan

### 4.1 New files (additive)
- `repos/le-wm/module.py` — extend with `MacroActionEncoder`. Pure addition;
  existing classes untouched.
- `repos/le-wm/hwm.py` *(or extend `jepa.py`)* — `HighLevelWorldModel`
  exposing `encode/predict/rollout/criterion/get_cost` with the same shape
  as `JEPA`. Wraps a frozen `JEPA` (encoder + projectors reused) plus a
  fresh `ARPredictor` + `MacroActionEncoder`.
- `repos/le-wm/data.py` — `WaypointSubtrajectoryDataset` (variable-stride
  waypoint sampler). Compatible with `spt.data.transforms.Compose`.
- `repos/le-wm/planner.py` — `HierarchicalCEMSolver` (or
  `HierarchicalPlanner`) wrapping two `swm.solver.CEMSolver` instances +
  `SubgoalCostAdapter` exposing `get_cost(info, actions)` with L1 against a
  precomputed subgoal latent.
- `repos/le-wm/policy.py` — `HierarchicalWorldModelPolicy` subclassing
  `swm.policy.WorldModelPolicy` (overrides `get_action`; reuses
  `_prepare_info` + action buffer machinery).
- `repos/le-wm/train_highlevel.py` — sibling to `train.py`. Loads frozen
  `JEPA` checkpoint, instantiates `MacroActionEncoder` + new high-level
  `ARPredictor`, runs HWM Eq.1 L1 teacher-forced training.
- `repos/le-wm/config/train/hwm.yaml`,
  `repos/le-wm/config/train/data/{pusht,ogb}_waypoints.yaml`.
- `repos/le-wm/config/eval/solver/hcem.yaml`,
  `repos/le-wm/config/eval/{pusht,cube}_hwm.yaml`.
- `repos/le-wm/tests/test_hierarchical.py` (Phase-5 deliverable).

### 4.2 Modified files (minimal)
- `repos/le-wm/eval.py` — single ~10-line branch on
  `cfg.solver._target_`: if hierarchical, load two `AutoCostModel`s
  (`policy_low`, `policy_high`) and pass both into the hierarchical
  solver/policy. Pre-existing `world.evaluate_from_dataset(...)` call site
  inherits whatever convention the flat path uses (see §1.6 stale-call
  flag — fix it once for both paths if needed).
- *(Optional)* `repos/le-wm/jepa.py` — 3-LOC `distance: 'mse'|'l1'` arg on
  `JEPA.criterion`. Skip if cost adapter inside `planner.py` is preferred.

### 4.3 Untouched (preserves flat-LeWM behaviour exactly)
- `repos/le-wm/train.py` (low-level training).
- `repos/le-wm/utils.py`.
- All existing files under `repos/HWM_PLDM/`, `repos/stable-pretraining/`,
  `repos/stable-worldmodel/` — these are READ-ONLY references per
  CLAUDE.md hard rules.

---

## 5. Consolidated gotchas list

Phase 5 must not miss these:

1. **`Embedder` ≠ `MacroActionEncoder`** — keep both. Different temporal
   scales, different architectures, different roles.
2. **First subgoal index = 1, not 0** in the high-level rollout
   (`pred_obs[0]` is the current latent, `pred_obs[1]` is the predicted
   next-waypoint latent).
3. **Cost is L1, not MSE** for high-level training and both planning
   levels (paper, not the HWM_PLDM repo).
4. **CEM, not MPPI** — keep LeWM's existing optimiser. Don't substitute
   from the reference repo.
5. **CEMSolver assumes env action space**: high-level CEM needs a synthetic
   `gym.spaces.Box(shape=(d_l,))` action space and `PlanConfig(action_block=1)`
   so `solver.action_dim == d_l`.
6. **Frame-skip arithmetic**: LeWM already groups 5 primitive actions per
   block. Set `macro_step_skip` in *LeWM blocks*, not raw env steps. Range
   5–14 LeWM blocks = 25–70 env steps (matches paper Push-T). Don't copy
   HWM_PLDM's `step_skip=10` raw env steps.
7. **Latent shape is single-vector**: 192-d post-projector. None of the
   `(C, H, W)` conv-spatial / `Expander2D` / `RunningCost` reshape paths
   from HWM_PLDM carry over.
8. **Joint training** of `A_ψ` and `P^(2)` via the next-latent L1 loss
   only. No separate `A_ψ` loss. If `‖l‖` collapses empirically, add a
   small L2 reg (Phase-5 escape hatch, default off).
9. **Hyperparameters in `large_diverse_25maps_l2.yaml` are PLDM-tuned for
   MPPI on z_dim=8 conv latents**. Don't copy `noise_sigma`,
   `num_samples`, etc. directly. Follow HWM Tab. 10 (Push-T `d=50`):
   high-level 1500 samples / 40 iters / pred H=4; low-level 900 samples
   / 20 iters / pred h=5; replan k=5 LeWM blocks.
10. **Subgoal gating / advancement is not in HWM_PLDM** even though the
    paper hints at it. Implement only behind a config flag, default off,
    so we can reproduce the paper baseline first.
11. **No new SIGReg at the high level**. The latent space is inherited
    from the frozen encoder; matching it via L1 is sufficient.
12. **Stale `world.evaluate_from_dataset` call** at `eval.py:142` —
    pre-existing in flat LeWM. Either fix once or match the convention
    that the flat path uses; do not silently diverge in the hierarchical
    branch.
13. **Checkpoints are pickled bare PyTorch model objects** (`utils.py:55`,
    `torch.save(model, path)`); `swm.policy.AutoCostModel` scans for a
    `get_cost` attribute (`policy.py:556-574`). The new
    `HighLevelWorldModel` only needs to expose `get_cost` to "just work"
    with the existing eval loader — no special checkpoint format required.

---

## 6. Bottom line for Phase 4

The architecture phase can take all of the above as fixed. The two design
decisions that still need to be made deliberately in Phase 4 are:

- **`d_l` (latent macro-action dim) per environment.** Start at 10 for
  PushT (matches paper's `5×2` concatenation) and 4 for Cube (HWM Fig. 7
  optimum); commit to a sweep range in Phase 4.
- **Whether the high-level predictor is same-size or scaled-up vs LeWM's
  6-layer 192-dim predictor.** HWM scales up (~25M → ~75M for
  DINO-WM/Push-T) but the host engineering constraints favour starting
  same-size to keep training time short, then sizing up only if
  long-horizon prediction error doesn't beat low-level autoregression
  beyond ~1.5 s (HWM Fig. 6).

Everything else — module boundaries, training loop, eval wiring, hyperparameter
sources, config layout — is settled by the analysis above and ready to be
reflected into Phase 4's architecture diagram.
