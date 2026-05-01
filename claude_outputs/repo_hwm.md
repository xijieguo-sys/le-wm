# Phase 3: HWM_PLDM Reference Repo Analysis

Focused walk of `repos/HWM_PLDM/` to extract concretely *how* the public PLDM
implementation realises HWM-style hierarchy, mapped against LeWM's intended
integration. All paths absolute. Citations are `file:line`.

> Caveat up front: `HWM_PLDM` is the **PLDM-only** minimal release of the HWM
> paper. Several pieces the paper (and our `paper_context.md`) describe — Push-T
> with DINO-WM, variable-length waypoints, an `A_ψ` *transformer* with CLS, L1
> matching costs at both training and planning time — are **not** present here.
> The repo implements the same hierarchy *recipe* but with PLDM-specific design
> choices (fixed stride, MLP `A_ψ`, MSE costs, conv predictors). I flag every
> spot where this minimal reference diverges from the paper, so we don't
> port-by-default the wrong thing into LeWM.

---

## 1. High-level training data: waypoint indices and action sub-sequences

The high-level training sample is built **per `__getitem__`** in
`D4RLDataset` at `/oscar/home/xguo84/final_project/repos/HWM_PLDM/pldm_envs/diverse_maze/d4rl.py:226-246`.
There is **no separate sampler / dataset class** for the high level — the same
`Dataset` returns both an L1 sub-trajectory and an L2 sub-trajectory together
when `l2_n_steps > 0` is set (config switch).

Key facts:
- Waypoint stride is **fixed**, set once via config: `l2_step_skip` (10 in the
  Push-T-style maze config, `pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml:3`).
- Number of waypoints `N` is also fixed: `l2_n_steps = 6` (yaml line 4) → each
  L2 sample covers `l2_n_steps_total = N*step_skip + 1 = 61` env steps
  (`pldm_envs/diverse_maze/d4rl.py:22-25`).
- States at waypoints are sampled by **strided slicing**:
  `_load_data_from_start_idx(..., skip_frame=l2_step_skip)` at line 232.
  No re-encoding inside the dataset — raw images come out, the encoder runs
  later inside the model.
- Inter-waypoint actions are loaded contiguously (no skip) over the full
  60-step span and then split into chunks via tensor `.split(step_skip)`:
  `chunks = l2_actions1.split(self.config.l2_step_skip)`
  `l2_actions = torch.stack(chunks, dim=0)`  (lines 236-238)
  → shape `(N, step_skip, action_dim) = (6, 10, 2)`.
- `chunked_actions: true` is mandatory for L2 — the alternative branch is
  `raise NotImplementedError` (line 240).

Wiring into PyTorch:
- The dataset produces a `D4RLSample` NamedTuple with `l2_states` (7, ...),
  `l2_actions` (6, 10, 2), `l2_proprio_pos`, `l2_proprio_vel`, `l2_locations`
  (`pldm_envs/diverse_maze/enums.py:8-32`).
- `DataLoader` is plain torch with collate-by-stack; the L2 path is gated by
  `data.d4rl_config.l2_n_steps > 0` (`pldm/data/utils.py:159, 190`).
- No padding mask, no variable length: every sample has exactly the same
  shapes.

> **Divergence from paper.** The paper sampling rule ("for each trajectory,
> sample `N` waypoint indices `1 = t_1 < … < t_N < T`"; Push-T 25–70 step
> spans, Franka middle waypoint sampled uniformly) is **not** in this repo —
> the minimal release uses a single fixed stride. Our LeWM port should
> implement the variable-stride sampler ourselves; we cannot lift it.

---

## 2. Waypoint / subgoal representation, training vs planning

### 2.1 Training time
Waypoints live as **raw observations** until they hit the encoder inside the
model forward. `HJEPA.forward_posterior`
(`pldm/models/hjepa.py:158-187`) does:
1. encode the L2-strided frames `l2_states` through the **frozen L1 backbone**
   in `encode_only=True` mode (lines 162-171),
2. take the L1 encoder's `obs_component` (the spatial conv feature map, see
   §5) as the L2 backbone input,
3. for L2 the backbone is `identity_encoder`
   (`pldm/models/encoders/encoders.py:424-455`) — it just passes through and
   re-attaches proprio. So the L2 latent space *is* the L1 conv feature space.

Shape at training:
- L1 obs encoding: a tuple, e.g. `(channels, H', W')` like `(16, H, W)` from
  `d4rl_a` (`pldm/models/encoders/encoders.py:37-43`). NOT a single vector.
- L2 input: same shape, plus optional proprio channels concatenated
  (`encoders.py:443-455`).

### 2.2 Planning time — first subgoal extraction
`TwoLvlPlanner.plan` (`pldm/planning/planners/two_lvl_planner.py:25-85`):
```python
l2_result = self.l2_planner.plan(current_state=backbone_output, plan_size=plan_size, repr_input=True)
...
self.l1_planner.reset_targets(l2_result.pred_obs[1].detach(), repr_input=True)
```
- The first subgoal `z̃_1` is `l2_result.pred_obs[1]` — index `1` in the
  predicted-obs tensor of the L2 rollout (= one macro-step ahead from the
  current latent), **as a predicted latent**, not a decoded image.
- `repr_input=True` means the L1 cost objective receives the latent directly
  via `set_target` (`pldm/planning/objectives_v2.py:48-60`).
- **No image decoder is in this loop.** Subgoal-as-image only appears in
  visualisation utilities (`pldm/planning/plotting.py`), not in the control
  loop.

> **Divergence from paper.** The paper describes L2 producing the next-
> waypoint *latent* using a shared latent space — same here. But because PLDM's
> "latent" is a 16×H×W conv feature map, "matching the subgoal" is a per-cell
> MSE rather than a `‖·‖_1` over a single 192-d vector as in LeWM. We have to
> re-derive the cost shape for LeWM.

---

## 3. Macro-action encoder `A_ψ`

In this repo `A_ψ` is **not** the transformer-with-CLS the paper describes —
it is the **`posterior_model`** of the L2 sequence predictor.

Architecture (per yaml `large_diverse_25maps_l2.yaml:97-105` and code):
- `posterior_input_type: 'actions'`
- `posterior_input_dim: 20` = `l1_action_dim * step_skip = 2 * 10`
  (computed at `pldm/models/jepa.py:90-94`).
- `posterior_arch: '32-32'`  → 2-hidden-layer MLP, hidden 32, hidden 32.
  Constructed by `PosteriorContinuous(MLP(arch='32-32', input_dim=20,
  output_shape=2*z_dim))` (`pldm/models/misc.py:134-180`).
- `z_dim: 8` → output `l_t ∈ R^8` (yaml line 95). For PLDM, `d_l = 8`.
  Output is split into `(mu, std)`; planning uses `mu` only
  (`misc.py:177` and `sequence_predictor.py:280`).
- `mu_ln = LayerNorm(z_dim)` is applied to `mu` (`misc.py:148, 178`) — this is
  the only inductive bias keeping the macro-action latent in a bounded range.

Variable-length handling: **none.** Chunk size is fixed to `step_skip=10`,
input is just flattened: `posterior_input = actions[i].view(bs, -1)`
(`pldm/models/predictors/sequence_predictor.py:269-275`). No padding mask,
no transformer, no CLS pooling.

How it's trained: **jointly** with `P^(2)`. `compute_posterior=True` is
passed inside `JEPA.forward_posterior` for the L2 path
(`pldm/models/jepa.py:242-250`), and the resulting macro-action `posterior` is
fed as the `predictor_input` to the conv predictor at
`sequence_predictor.py:285-289`. There is no separate loss on `A_ψ`; it's
trained end-to-end via the L2 next-state prediction loss (see §4 below).

There's also an `AnalyticalPosterior` (`misc.py:207-233`) that just sums
primitive actions in the chunk — used for ablation, not the headline run.

> **Divergence from paper.** Paper's `A_ψ` is a transformer with a CLS token +
> MLP head. Repo's is an MLP over a flattened, fixed-length action vector.
> For LeWM we should follow the paper (transformer + CLS) because we want
> variable-length chunks; lifting the MLP from here would silently bake the
> fixed-stride assumption into our model.

---

## 4. Hierarchical and low-level planning

The two-level outer loop is `TwoLvlPlanner`
(`pldm/planning/planners/two_lvl_planner.py:11-85`). The MPC driver is
`MPCEvaluator._perform_mpc` (`pldm/planning/mpc.py:348-616`).

### 4.1 Optimiser
- **MPPI**, not CEM, in this repo. `MPPIPlanner` wraps a per-env list of
  `MPPI` controllers (`pldm/planning/planners/mppi_planner.py:273-380`,
  `mppi_torch.py:18+`). Importance-weighted update of a nominal control
  sequence; no elite truncation.
- `PlannerType.SGD` and `PlannerType.BeamSearch` exist in the enum
  (`pldm/planning/planners/enums.py:6-9`) but the diverse-maze yaml uses
  `PlannerType.MPPI` for both levels.
- Single-iteration planning (no explicit `n_iters` like CEM): MPPI updates the
  nominal trajectory once per `command()` call. Re-optimisation comes via the
  outer `replan_every` loop, not inner CEM iterations.

### 4.2 Cost functions
Both levels use **MSE**, not L1, in this implementation
(`pldm/planning/objectives_v2.py:102, 178`,
`pldm/planning/planners/mppi_planner.py:268`):
```python
diff = (encodings - target).pow(2)
```
- L2 cost target: the **final goal latent** `z_g`, set once per MPC call via
  `planner.reset_targets(targets_t, ...)` at `mpc.py:378`.
- L1 cost target: the **first L2-predicted subgoal** `z̃_1`, set per replan
  via `self.l1_planner.reset_targets(l2_result.pred_obs[1].detach(),
  repr_input=True)` (`two_lvl_planner.py:73-75`).
- `sum_all_diffs: true` for both levels in the yaml — the cost is summed
  across the whole rollout with a linear weight schedule
  (`objectives_v2.py:104-118`). When false, only the last 3 timesteps are
  scored (see `sum_last_n` default).
- `sum_last_n: 3` for L2 (yaml line 224) — keeps the bulk of the L2 cost on
  the last few predicted latents close to the goal.

### 4.3 First subgoal `z̃_1`
- Computed inside `TwoLvlPlanner.plan` as `l2_result.pred_obs[1]` — index 1
  of the L2 prediction tensor (`pred_obs` has shape `(plan_size+1, B, ...)`,
  index 0 is the current state).
- Passed to L1 by `l1_planner.reset_targets(..., repr_input=True)`.
- Then L1 plans for exactly `plan_size = self.l2_step_skip = 10` env steps
  (`two_lvl_planner.py:79`).

### 4.4 Hyperparameters per level (PLDM Push-T-style maze, hard split)

From `large_diverse_25maps_l2.yaml:178-281`:

| level | num_samples | noise_sigma | min_step | max_step | plan length | action dim |
|---|---|---|---|---|---|---|
| L1  | 500–1000 (per difficulty) | 5  | 0    | 1   | up to 128 prim. actions   | 2 (primitive) |
| L2  | 2000–4000                 | 10 | -2.5 | 2.5 | 3–18 macro-steps          | 8 (latent z)  |

Other knobs:
- `replan_every: 4` env steps (`yaml:193`); with `action_repeat: 4` this means
  L2 re-runs every 4 outer loop iterations = 16 raw env steps. L2 covers up
  to `max_plan_length_l2 * step_skip = 18 * 10 = 180` env steps.
- `var_samples: 0`, `lambda_: 0.0025` (yaml:214, 230) — no ensemble variance
  cost, fairly low MPPI temperature.
- L2 `z_reg_coeff: 0.1` (yaml:231) — light L2 reg on sampled macro-actions to
  keep them near zero. This is the *only* macro-action regulariser.

### 4.5 Subgoal advancement / gating
**Not implemented.** I grepped for `subgoal|threshold|advance|gating` in the
planning tree; only matches are unrelated config fields
(`mppi_torch.py:191`, `planners/enums.py:55`). Every `replan_every` step the
two-level planner runs L2 from scratch and takes `pred_obs[1]` again as the
subgoal. The "advance to next subgoal once `‖z̃_1 − z_t‖` is small" idea from
the paper is not in the code path.

There **is** a Stage-1 / Stage-2 split (`mpc.py:225-269`): hierarchical MPC
runs for `n_steps - final_trans_steps` env steps, then **flat L1 planning**
takes over for the last `final_trans_steps` (15 in yaml line 182). This is a
final-mile fallback, not subgoal advancement.

### 4.6 MPC outer loop tying both levels
`mpc.py:434-477` — the same `if i % replan_every == 0:` branch calls
`planner.plan(...)`; for the bilevel planner that single call internally runs
L2 then L1 (`two_lvl_planner.plan`). Then for each of the next
`replan_every` env steps, `planning_result.actions[:, i % replan_every:]` is
indexed into and applied (`mpc.py:519-523`). After Stage 1, Stage 2 reuses
the same loop with a flat L1 planner.

---

## 5. Adapter points for porting to LeWM

### 5.1 Latent shape mismatch (most disruptive)
- LeWM: single 192-d CLS token per frame, post-projector
  (paper context §1.2, §1.4).
- HWM_PLDM: 4-D conv feature map per frame, e.g. `(channels=16, H, W)` from
  `MeNet6` with `d4rl_a`. Action conditioning is broadcast spatially via
  `Expander2D` and concatenated as channels (`conv_predictors.py:484-499`).
- Consequence: every cost/objective in this repo is `(state - target).pow(2)`
  reduced over channels and spatial dims. For LeWM we just take a 1-D `‖·‖`
  over the 192-d vector. Mechanically simpler, but it means we cannot reuse
  any of `objectives_v2.py` or the conv-spatial paths in `mppi_planner.py`
  (the `pred_encoder=…` branch at `mppi_planner.py:228-256` is purely there
  to handle the L1→L2 spatial reshape and is unnecessary for us).
- Action conditioning: PLDM = channel-broadcast over spatial map. LeWM =
  AdaLN per transformer block. We must keep LeWM's AdaLN; do not import the
  `Expander2D` path.

### 5.2 Frame-skip / action-chunk semantics
- `HWM_PLDM` operates on **primitive** env actions (`action_dim: 2`,
  `step_skip: 10`). Macro-action chunk = 10 primitive actions.
- LeWM has frame-skip 5 baked in: a single LeWM "action" is already a 10-D
  vector (5 × 2 primitive). So a "step_skip" of 10 in LeWM units is **50 raw
  env steps** — twice what HWM_PLDM uses, and longer than HWM-paper's Push-T
  segments (25–70 env steps).
- Recommended re-alignment: pick `macro_step_skip` in LeWM **action blocks**.
  A natural starting point is `macro_step_skip = 5` LeWM blocks = 25 env
  steps (matches LeWM's existing low-level horizon `H = 5` blocks and HWM
  paper's Push-T low-level `h = 5, k = 5`). Variable-stride sampler in the
  range 5–14 LeWM blocks (i.e. 25–70 env steps) reproduces the paper.
- The `posterior_input_dim = l1_action_dim * step_skip` shortcut from
  `jepa.py:94` only works because the chunk size is fixed. With variable
  chunks we need a transformer + padding mask, which the repo does not have.

### 5.3 Predictor architecture
- HWM_PLDM L2 predictor: `ConvPredictor` with subclass `l2_d4rl_e_p` over
  spatial latents, residual conv blocks (`conv_predictors.py:443-532`,
  shape descriptors at `conv_predictors.py:24-30`).
- LeWM low-level predictor: 6-layer transformer, AdaLN, history `N = 3`, no
  spatial dim (`paper_context.md` §1.3). The HWM paper itself uses a
  scaled-up transformer (Push-T DINO-WM: 25M → 75M, 6→10 layers, 384→768
  dim).
- Implication for our high-level predictor: re-use **LeWM's `ARPredictor`**
  class, conditioned on macro-actions instead of primitive actions, exactly
  as the paper does. Do NOT lift any `Conv*Predictor` from this repo.
- AdaLN history: keep LeWM's `N=3`; the paper does not contradict this and
  the ConvPredictor history-of-1 path here would break LeWM's autoregressive
  stability properties.

### 5.4 Backbone-specific utilities to NOT import
Hard-coupled to PLDM/D4RL and should stay in `HWM_PLDM`:
- `pldm.models.encoders.encoders` (MeNet6, IdentityEncoder, FuseXYEncoder,
  conv layer configs at `encoders.py:11-107`) — all conv-spatial.
- `pldm.models.predictors.conv_predictors` and `predictors.predictors`
  (`build_predictor` defaults to RSSM/MLP/Conv variants).
- `pldm.planning.planners.mppi_planner.RunningCost` (`mppi_planner.py:208-270`)
  — proprio/spatial cost reshaping is PLDM-specific. Re-implement the cost
  in LeWM (a one-liner: `((pred - target) ** p).mean(-1)` for `p ∈ {1, 2}`).
- `pldm.planning.planners.mppi_torch.MPPI` — usable in principle, but LeWM
  already has a CEM path via `stable_worldmodel.solver.CEMSolver`. Re-using
  CEM keeps us aligned with LeWM's evaluation; switching to MPPI would be a
  separate decision.
- `pldm_envs.utils.normalizer` and the action-chunk pipeline through
  `chunked_actions` — couples to the diverse-maze data layout.

What is reusable as *inspiration* (small, idea-only):
- The "set L1 target = `l2_pred_obs[1]`" pattern (`two_lvl_planner.py:73`).
- The `posterior_input_type='actions'` plumbing for joint training of
  `A_ψ` + `P^(2)` (`sequence_predictor.py:269-289`,
  `prediction.py:39-89`).
- The Stage-2 final-mile flat planner (`mpc.py:225-269`) — useful trick if
  hierarchical control struggles in the goal vicinity.

### 5.5 Cost-function distance (L1 vs MSE)
The paper specifies L1 at both training and planning time. The PLDM
implementation uses MSE everywhere (`prediction.py:83`,
`objectives_v2.py:102, 178`). LeWM's existing flat planner also uses MSE
(`paper_context.md` §1.7, Eq. 4). The right thing for the LeWM port is:
- Match HWM paper: use **L1** for both high-level training (HWM Eq. 1) and
  hierarchical planning costs.
- Keep LeWM's flat MSE path untouched — make `distance` a config knob in
  `JEPA.criterion` (already in our high-level plan §3.1).

### 5.6 Hyperparameters that need re-tuning (don't copy-paste)
- `noise_sigma=10` for L2 in this repo is calibrated to a `z_dim=8`,
  layer-normed macro-action whose typical norm is O(1). LeWM-Push-T `d_l=10`
  starting point (paper `5×2` concatenation, `paper_context.md` §2.2)
  needs its own sigma sweep; using 10 blindly will likely over-explore.
- `num_samples=2000–4000` (L2) at horizons of 18 macro-steps — viable
  because PLDM's predictor is small (~few M params). LeWM's 6-layer
  transformer is similar; should be fine on one GPU but worth a budget
  sanity check.
- `replan_every=4` in this repo accounts for `action_repeat=4` env wrapper.
  LeWM has *no* `action_repeat` wrapper; replan cadence should be expressed
  directly in LeWM action blocks (e.g., `k=5`).
- `final_trans_steps=15`: not a knob worth porting yet — only relevant if
  hierarchical control under-performs flat in the last few frames.

---

## Mapping HWM_PLDM → LeWM

| HWM_PLDM module / artifact | LeWM equivalent (target) | What changes |
|---|---|---|
| `pldm.models.jepa.JEPA` (level1)  | existing `JEPA` in `repos/le-wm/jepa.py` | Freeze and treat as `low_level_wm`. No code change beyond a `freeze()` helper / `requires_grad_(False)` call. |
| `pldm.models.jepa.JEPA` (level2 with `IdentityEncoder` backbone) | new `HighLevelWorldModel` (`hwm.py` or extension of `jepa.py`) | Reuse LeWM encoder (frozen). High-level predictor = a fresh `ARPredictor`. No identity-encoder layer needed since LeWM already produces a single shared latent. |
| `MeNet6` / `ConvPredictor` (`encoders.py:131`, `conv_predictors.py:443`) | LeWM's ViT-Tiny encoder + `ARPredictor` | Replace entirely; latent is 192-d vector, predictor is transformer + AdaLN. |
| `PosteriorContinuous` MLP as `A_ψ` (`misc.py:134-180`) | new `MacroActionEncoder` in `module.py` | Switch to **transformer + CLS + MLP head** (matches paper, allows variable-length chunks with padding mask). Output dim = `d_l`. |
| `D4RLDataset.__getitem__` chunking (`d4rl.py:226-260`) | new `WaypointSubtrajectoryDataset` (`data.py` or extension of `train.py` data path) | Variable-stride waypoint sampler returning `(waypoint_pixels, inter_action_chunks, inter_action_mask)`. |
| `chunked_actions: true` config switch (`l2_step_skip`, `l2_n_steps`) | hydra config under `config/train/data/pusht_waypoints.yaml` | Hyperparams per env: `min_stride`, `max_stride`, `N`, `chunk_pad`. |
| `compute_posterior=True` in `sequence_predictor` (`sequence_predictor.py:269-289`) | high-level training step in `train_highlevel.py` | Joint update of `A_ψ` + `high_level_predictor` via L1 next-latent loss. Encoder/L1 predictor frozen. |
| `PredictionObjective(.).pow(2).mean()` (`prediction.py:83`) | L1 loss in high-level trainer (HWM Eq. 1) | Switch `pow(2)` → `abs()`. Match paper, not this repo. |
| `TwoLvlPlanner.plan` (`two_lvl_planner.py:25-85`) | new `HierarchicalPlanner` in `planner.py` | Swap MPPI for `stable_worldmodel.solver.CEMSolver` (twice). Keep "first subgoal = `pred_obs[1]`" pattern. Cost = L1 (paper) instead of MSE (this repo). |
| `MPPIPlanner` (`mppi_planner.py`) | existing `CEMSolver` + thin cost wrappers | Two CEM passes, one per level. No port of MPPI-specific code. |
| `MPCEvaluator._perform_mpc` Stage-1/Stage-2 (`mpc.py:225-269`) | optional final-mile fallback in `HierarchicalPlanner` | Implement only if empirical control quality near goal regresses; not a default. |
| `replan_every: 4` + `action_repeat: 4` (`yaml:193, 179`) | `cfg.solver.replan_every` in LeWM action-block units | LeWM has no action_repeat wrapper; just set `replan_every = 5` LeWM blocks (= 25 env steps) to mirror paper's `k=5`. |

---

## Gotchas the implementation phase must not miss

- **Latent shape**: HWM_PLDM's "latent" is a `(C, H, W)` conv feature map.
  LeWM's is a 192-d CLS vector. Costs, action injection (AdaLN vs spatial
  broadcast), and the "shared latent" assumption are all simpler in LeWM.
  Anything written in `pldm/models/predictors/conv_predictors.py` does **not**
  carry over.
- **Macro-action encoder is not a transformer in this repo.** `A_ψ` is a 32-32
  MLP (`misc.py:134-180`) over a flat fixed-length chunk. Implementing the
  paper's transformer-with-CLS in LeWM requires variable-length support
  (padding + attention mask) which we have to write ourselves.
- **Waypoint stride is fixed in this repo.** `l2_step_skip=10`,
  `l2_n_steps=6`. The paper's variable-stride sampler is *not* implemented
  here. Don't assume the dataset code is reusable; we need to write our own.
- **Cost is MSE here, paper says L1.** Both `prediction.py:83` and the MPC
  cost in `objectives_v2.py:102` use `.pow(2)`. Paper Eq. 1 and Eq. 2 use
  `‖·‖_1`. Implement L1 per the paper; expose `distance` as a config knob.
- **Optimiser is MPPI, not CEM.** This repo never uses CEM in the planning
  path. LeWM does — we should keep CEM (matches LeWM's existing eval
  baseline) and not silently substitute MPPI.
- **No subgoal advancement / gating.** The paper mentions it as a possibility
  but the repo just re-runs L2 every `replan_every`. If we add gating in
  LeWM, it's a Phase-5 addition with no reference implementation — guard it
  behind a config flag and default off, otherwise we cannot reproduce the
  paper baseline.
- **First subgoal index = 1, not 0.** `pred_obs[0]` is the current state;
  `pred_obs[1]` is the predicted next-waypoint latent
  (`two_lvl_planner.py:73-75`). Easy off-by-one to get wrong.
- **Hyperparameters in `large_diverse_25maps_l2.yaml` are PLDM-tuned**:
  `noise_sigma=10` for `z_dim=8` MPPI macro-actions, `num_samples` 2000–4000.
  Don't copy these for LeWM CEM; the units and the optimiser are both
  different.
- **Frame-skip arithmetic**: PLDM `step_skip=10` env steps × `action_repeat=4`
  in the wrapper means each L2 macro-action covers 40 raw env steps. LeWM
  already groups 5 primitive actions per LeWM block; if we say
  `macro_step_skip=5`, that's 25 env steps. Pick `macro_step_skip` in LeWM
  *blocks* and verify the env-step total matches the paper's Push-T 25–70
  range.
- **Joint training of `A_ψ` and `P^(2)`.** Joint, end-to-end on the
  next-latent prediction loss only — no separate `A_ψ` loss in this repo. Our
  high-level trainer should do the same; if `‖l‖` collapses we add a small
  L2 reg (the repo's `z_reg_coeff: 0.1` is at *planning* time, not training,
  and applies only to MPPI-sampled actions).
- **Stage-2 final-mile flat planner (`mpc.py:225-269`) exists in this repo
  but is rarely discussed in the paper.** Treat as an inspiration, not a
  required component. Default off in our LeWM port, enable only if eval
  shows the hierarchical planner stalls near the goal.
