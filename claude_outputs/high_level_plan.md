# Phase 2: High-Level Integration Plan

Plan for adding HWM-style hierarchical planning on top of the existing LeWorldModel
codebase in `repos/le-wm/`. Module-level only — no code yet. References use the
file names that already exist (`jepa.py`, `module.py`, `train.py`, `eval.py`,
`config/train/lewm.yaml`, `config/eval/pusht.yaml`, etc.). Detailed file edits are
deferred to Phase 4/5.

The guiding principle, from `CLAUDE.md` and the engineering constraints, is:
**flat LeWM behaviour must remain identical when hierarchy is off**, and changes
must stay local to `repos/le-wm/`. The reference repos
(`HWM_PLDM/`, `stable-pretraining/`, `stable-worldmodel/`) stay read-only.

---

## 1. Naming and the "two action encoders" gotcha

LeWM already has a `JEPA.action_encoder` (`module.Embedder`) that lifts a
5-frameskip block of primitive actions into the predictor's embedding dim for AdaLN
conditioning. This is **not** what HWM calls an action encoder.

To keep the diff readable we use these names throughout the rest of the project:

- `low_level_action_embedder` — the existing `Embedder`. Unchanged.
- `macro_action_encoder` — **new** transformer-based module (HWM's `A_ψ`). Compresses
  a sequence of LeWM action blocks across multiple low-level steps into one
  latent macro-action `l ∈ R^{d_l}`.
- `low_level_wm` — the existing `JEPA` (`encoder` + `predictor`). Frozen at
  hierarchy time.
- `high_level_wm` — **new** `JEPA`-shaped object reusing the *same* frozen encoder
  and the *same* `ARPredictor` architecture, but conditioned on macro-actions and
  trained at a longer temporal stride.

---

## 2. New modules to add (under `repos/le-wm/`)

### 2.1 `MacroActionEncoder` (in `module.py`)
- Small transformer (a couple of `Block`s) plus a learnable CLS token and an MLP
  head, mirroring HWM's `A_ψ`.
- Input: a chunk of LeWM action blocks `a_{t_k : t_{k+1}}` of variable length
  (right-padded + attention mask). For PushT, each LeWM action block is already 10-D
  (5 primitive 2-D actions); HWM uses `d_l = 10` here, so input dim = 10 ×
  chunk_length, output dim = `d_l`.
- Output: `l_{t_k} ∈ R^{d_l}` (CLS token through MLP head).
- New, but stays in `module.py` to keep the model-zoo location stable.

### 2.2 `HighLevelWorldModel` (new file `hwm.py`, or new section of `jepa.py`)
- Wraps the **frozen LeWM encoder** plus a fresh `ARPredictor` (same class as the
  low-level predictor), conditioned on macro-actions instead of low-level action
  embeddings.
- Public API mirrors `JEPA.encode` / `JEPA.predict` / `JEPA.rollout` / `JEPA.criterion`
  so it can plug into the existing `swm.policy.WorldModelPolicy` cost interface.
- Internally calls `macro_action_encoder` to produce conditioning instead of
  `low_level_action_embedder`.
- Optionally scaled up (HWM doubles capacity for Push-T). Phase-4 decision; default
  to the same depth/dim as low-level so we keep parameter count bounded.
- Whether the encoder is frozen or fine-tuned is a config switch (`freeze_encoder:
  true` by default — HWM uses frozen encoder for PointMaze, fine-tuned only on the
  Franka real-robot case which we are not targeting).

### 2.3 `HierarchicalPlanner` (new file `planner.py`)
- Wraps two CEM passes around `low_level_wm` and `high_level_wm`. Conceptually:
  ```
  1. encode z_1, z_g
  2. high_level CEM on l_{1:H} minimizing ||z_g - P^(2)(l_{1:H}; z_1)||_1
     -> first subgoal z̃_1 = P^(2)(l*_1; z_1)
  3. low_level  CEM on a_{1:h} minimizing ||z̃_1 - P^(1)(a_{1:h}; z_1)||_1
  4. execute first k actions, re-encode, replan
  ```
- Reuses `stable_worldmodel.solver.CEMSolver` twice (do **not** re-implement CEM).
  The two solvers differ only in: action dim (`d_l` vs primitive action dim),
  horizon (`H` vs `h`), num samples / iters (separate config blocks), and the
  cost function passed in.
- Cost functions are thin wrappers around the existing `JEPA.get_cost` /
  `JEPA.criterion`; the only change is L1 (HWM Eq. 1) vs LeWM's MSE — make this a
  config knob (`distance: l1 | l2`, default `l1` for hierarchy to match HWM, `l2`
  for flat to match LeWM).
- Exposes a `plan(obs, goal_obs)` method that returns the next `k` low-level
  actions to execute. This is what the policy wrapper calls each MPC step.

### 2.4 `HierarchicalPolicy` (extension of the existing policy hookup in `eval.py`)
- A `swm.policy.WorldModelPolicy`-shaped object that holds a `HierarchicalPlanner`
  instead of a single solver. Selected by config (see §6).
- Lives either in a new `policy.py` in `repos/le-wm/` or as a small `hydra.utils.instantiate`
  target in the eval config — whichever keeps `eval.py` lean.

### 2.5 `WaypointSubtrajectoryDataset` (new transform under `train.py` data path)
- Wraps `swm.data.HDF5Dataset` and emits, per training sample, a sequence of N
  waypoint indices `1 = t_1 < t_2 < ... < t_N` together with the action chunks
  between them.
- Two sampling modes (config knob):
  - **Variable-stride** (HWM default): sample N indices per trajectory subject to a
    min/max gap. PushT setting: trajectory segments 25–70 env steps, N = 5
    waypoints — translates to 5–14 LeWM blocks given frame-skip 5.
  - **Fixed-stride** (HWM PointMaze default): every 10 env steps, N = 3–6.
- Returns:
  - `waypoint_pixels`: (B, N, C, H, W) — for encoding waypoint latents
  - `inter_actions`: (B, N − 1, L_max, action_block_dim) padded action chunks
  - `inter_action_mask`: (B, N − 1, L_max) bool — for the variable-length encoder
- Lives in a new file `data.py` (or a section of `utils.py`) under
  `repos/le-wm/`. We deliberately do not edit `stable-worldmodel`.

---

## 3. Existing modules to modify (minimally)

The whole point is to keep flat LeWM untouched. Modifications are additive and
gated by config flags.

### 3.1 `jepa.py`
- **No change** to `JEPA.encode`, `JEPA.predict`, `JEPA.rollout`, `JEPA.criterion`,
  `JEPA.get_cost`. The class continues to be the low-level model.
- Add a small `freeze()` helper (or just call `requires_grad_(False)` from outside)
  so the high-level training script can freeze a loaded LeWM checkpoint cleanly.
- Optional: refactor `criterion` to accept a `distance` argument (`'mse'` or
  `'l1'`) defaulting to `'mse'`. This costs ≈3 lines and lets the hierarchical
  planner reuse it directly. Default keeps flat behaviour unchanged.

### 3.2 `module.py`
- Add `MacroActionEncoder` (see §2.1). Existing classes untouched.

### 3.3 `train.py`
- Untouched for low-level (flat-LeWM) training.
- Add a parallel entry point `train_highlevel.py` (preferred — keeps `train.py`
  clean) that:
  - loads a frozen LeWM checkpoint (encoder + low-level predictor frozen),
  - instantiates `MacroActionEncoder` + new high-level `ARPredictor`,
  - uses the new waypoint dataset,
  - runs the loss in §4.
  An alternative is a `cfg.wm.type: highlevel` switch inside `train.py`, but the
  forward step is different enough (waypoint sampling, no SIGReg, frozen encoder)
  that a sibling script is cleaner.

### 3.4 `eval.py`
- Add a config-driven branch: if `cfg.solver._target_` resolves to a hierarchical
  planner, instantiate it with both low-level and high-level checkpoints; otherwise
  fall through to today's single-level `WorldModelPolicy`. Effective diff is
  ~10 lines.

### 3.5 `config/`
- New file `config/train/hwm.yaml` (high-level training).
- New `config/eval/solver/hcem.yaml` for the hierarchical solver (high+low CEM).
- New `config/eval/pusht_hwm.yaml`, `config/eval/cube_hwm.yaml` to point eval at
  the hierarchical policy + appropriate horizons. Existing `pusht.yaml` /
  `cube.yaml` stay as-is.
- `config/train/data/` gains a `pusht_waypoints.yaml` and `ogb_waypoints.yaml`
  describing the waypoint sampler. These extend the existing `pusht.yaml`
  /`ogb.yaml` rather than replace them.

---

## 4. Training-objective changes

We **do not change** the LeWM training objective. Phase-1 LeWM training (single
`L_pred + λ · SIGReg` from `train.py`) runs unchanged on the existing data, and
the resulting checkpoint is the input to high-level training.

For **high-level training** we add a fresh forward step (HWM Eq. 1):

- Frozen: encoder, low-level predictor, `low_level_action_embedder`,
  `projector`, `pred_proj`. (`requires_grad_(False)`, eval mode.)
- Trainable: `macro_action_encoder` (`A_ψ`), `high_level_predictor` (`P^(2)`).
- Per sample with waypoints `(t_1, ..., t_N)`:
  1. encode every waypoint frame with the frozen encoder + projector to get
     `z_{t_1}, ..., z_{t_N}` (precomputable / cacheable since encoder is frozen).
  2. for each k = 1..N−1, compute `l_{t_k} = A_ψ(a_{t_k:t_{k+1}})`.
  3. teacher-forced rollout under the high-level predictor:
     `ẑ_{t_{k+1}} = P^(2)((l_{t_i}, z_{t_i})_{i ≤ k})`.
  4. **Loss**: `L_tf = (1/N) Σ_k ‖ẑ_{t_{k+1}} − z_{t_{k+1}}‖_1` (HWM Eq. 1).
- Optionally, multi-step autoregressive rollout L1 (HWM Eq. 3) with weight
  `γ_roll`. Default to `γ_roll = 0` for Push-T (matches HWM Tab. 7) and consider
  enabling for Cube only after the simpler version trains stably.
- **No new SIGReg.** The latent space is inherited from the already-regularized
  encoder; matching to that space via L1 is sufficient. This preserves LeWM's
  "two-term, one hyperparameter" simplicity for the parts we leave alone.
- Macro-action regularization: keep an eye on `‖l‖` collapsing. HWM does not add
  an explicit regularizer; if collapse appears empirically, fall back to a small
  L2 on macro-actions (Phase 5 decision, not committed up-front).

Hyperparameters to expose in `hwm.yaml`:
- `d_l` (latent macro-action dim): start at 10 for PushT (HWM default), 4 for
  Cube. Sweep within {3, 4, 6, 8, 10, 16}.
- `N` (waypoints per trajectory): 5 for PushT, 3 for Cube initially.
- waypoint min/max stride (env steps): (25, 70) for PushT, (12, 60) for Cube.
- `γ_tf = 1.0`, `γ_roll = 0.0` initially.
- AdamW, `lr ≈ 1e-4`, `weight_decay = 1e-3`, ~500 epochs (matches HWM Tab. 7).

---

## 5. Planning / rollout-loop changes

The flat planning loop (`eval.py` + `swm.policy.WorldModelPolicy` +
`stable_worldmodel.solver.CEMSolver`) is preserved verbatim. Hierarchy is a
**new branch** inside `eval.py` selected by config.

Hierarchical MPC step:

1. **Encode current obs and goal** with the frozen LeWM encoder:
   `z_1 = enc(o_1)`, `z_g = enc(o_g)`.
2. **High-level CEM** (in macro-action space):
   - sample `N_h` candidate macro-action sequences `l̃_{1:H} ~ N(μ_h, Σ_h)`,
     `l̃_{1:H} ∈ R^{H × d_l}`,
   - for each, autoregressively roll out `P^(2)` from `z_1` and score
     `‖z_g − P^(2)(l̃_{1:H}; z_1)‖_1`,
   - update sampling distribution from top-`K_h` elites,
   - repeat for `T_h` iterations.
   - First subgoal: `z̃_1 = P^(2)(l*_1; z_1)`.
3. **Low-level CEM** (in primitive action-block space):
   - identical to the current LeWM CEM, but cost is
     `‖z̃_1 − P^(1)(â_{1:h}; z_1)‖_1` (subgoal-matching) instead of
     `‖z_g − P^(1)(â_{1:h}; z_1)‖_2^2`.
   - existing horizon (`h = 5` LeWM blocks) is exactly what HWM uses on Push-T.
4. **MPC**: execute first `k` low-level actions (`k = 5` LeWM blocks → entire
   low-level horizon → matches LeWM's existing `receding_horizon: 5` and HWM's
   Push-T `k = 5`), re-encode, repeat from step 1.
5. **Subgoal-reached gating** (optional, lifted from HWM): when
   `‖z̃_1 − z_t‖_1` falls below a threshold, advance to the next subgoal `z̃_2`
   from the same high-level plan rather than re-running high-level CEM. Keeps
   replanning cost down. Default off; gated by config.

Suggested CEM hyperparameters (mirror HWM Tab. 10 for Push-T, d=50):
- High-level: 1500 samples, 40 iters, 10 elites, pred `H = 4` macro-steps, var EMA 0.9.
- Low-level: 900 samples, 20 iters, 10 elites, pred `h = 5`, var EMA 0.8, k = 5.
- Replan every k = 5 LeWM blocks (= 25 env steps). For non-zero subgoal-gating,
  threshold on `‖z̃_1 − z_t‖_1` percentile from training.

Compute claim from HWM (3× less planning compute at matched success on Push-T) is
the headline efficiency metric we should be able to reproduce.

---

## 6. Evaluation strategy

We evaluate three things, in order. Each step also serves as a sanity gate before
moving on.

### 6.1 Sanity: flat LeWM unchanged
- Run today's `python eval.py --config-name pusht` with the unmodified LeWM
  checkpoint after the hierarchy patches are merged.
- **Pass criterion**: PushT success rate within ±2 pts of the pre-hierarchy
  baseline (paper reports 90 ± 1.4). Same for Cube within ±3 pts of 74.
- This protects against accidental regressions from refactors to `jepa.py` or
  `eval.py` / config plumbing.

### 6.2 High-level world model trains and predicts
Two non-control checks, before touching planning:
- **Latent prediction error vs horizon**: replicate HWM Fig. 6 — measure L1 error
  between predicted future latent and ground-truth latent on a held-out validation
  set, separately for low-level autoregressive rollouts and one-step high-level
  predictions, at horizons {0.5, 1.0, 1.5, 2.0} s. Expectation:
  high-level beats low-level beyond ~1.5 s.
- **Subgoal decoding sanity** (optional, qualitative): use the existing decoder
  (Sec. 5.1 of LeWM paper, `App. D`) to decode `z̃_1` predicted by the high-level
  model and check it visually resembles a plausible mid-trajectory frame — purely
  diagnostic, not a metric.

### 6.3 Hierarchical control beats flat at long horizons
This is the main result. Two environments matching the project goal: Push-T and
OGBench-Cube.

- **Push-T long-horizon sweep** — replicate HWM Tab. 2 with LeWM as the
  low-level backbone:
  - Goal offset `d ∈ {25, 50, 75}` env steps (existing config uses 25).
  - Compare flat LeWM vs hierarchical LeWM at each `d`.
  - **Pass criterion**: hierarchical ≥ flat at `d = 25`, hierarchical clearly
    better at `d = 50` and `d = 75`. Target a similar relative gain to HWM's
    DINO-WM numbers (84 → 89 at d=25, 55 → 78 at d=50, 17 → 61 at d=75).
- **OGBench-Cube** — keep the existing `cube.yaml` eval setup, run hierarchical
  vs flat. **Pass criterion**: hierarchical ≥ flat. Cube is the harder visual
  case so we are looking for "no regression and ideally a small gain".
- **Compute trade-off** (HWM Fig. 5): sweep CEM `num_samples` for the flat planner
  and for the hierarchical planner, plot success vs wall-clock per plan. Goal:
  show hierarchical reaches the same success rate as flat with ≈3× less compute.

### 6.4 Ablations (optional, time-permitting; mirror HWM §4)
- `d_l` sweep on PushT in `{3, 4, 6, 8, 10, 16}`.
- delta-pose vs latent macro-action (HWM §4.1) — concat-of-primitive-actions
  baseline.
- `N` (waypoints per trajectory) sweep ∈ {3, 5, 7}.
- frozen vs fine-tuned encoder for the high-level model.

### 6.5 Determinism / engineering
- Unit tests in a new `tests/test_hierarchical.py`:
  - `MacroActionEncoder` shape contract, padding-mask correctness.
  - `HighLevelWorldModel.encode` output equals `JEPA.encode` output on the same
    obs (proves shared latent space).
  - `HierarchicalPlanner.plan` returns an action of the right shape and matches
    flat behaviour when the high-level model is replaced by an identity rollout
    (regression guard).
- A short smoke run in CI: 1-epoch high-level training on a tiny PushT subset +
  10-episode hierarchical eval, just to catch wiring issues.

---

## 7. Summary of deliverables (Phase 5 will implement these)

New files
- `module.py` (extended): `MacroActionEncoder`.
- `hwm.py` (or `jepa.py` extended): `HighLevelWorldModel`.
- `planner.py`: `HierarchicalPlanner` (+ optional `HierarchicalPolicy`).
- `data.py`: `WaypointSubtrajectoryDataset` / waypoint sampling transform.
- `train_highlevel.py`: high-level training entry point.
- `config/train/hwm.yaml`, `config/train/data/pusht_waypoints.yaml`, `…/ogb_waypoints.yaml`.
- `config/eval/solver/hcem.yaml`, `config/eval/pusht_hwm.yaml`, `config/eval/cube_hwm.yaml`.
- `tests/test_hierarchical.py`.

Modified files (minimal)
- `jepa.py`: optional `distance` argument on `criterion`; nothing else.
- `eval.py`: branch on hierarchical vs flat solver instantiation.

Not touched
- LeWM training loop (`train.py`).
- LeWM training objective / SIGReg.
- Anything in `repos/HWM_PLDM/`, `repos/stable-pretraining/`, `repos/stable-worldmodel/`.
