# Phase 1: Paper Context

Notes from reading `papers/LeWorldModel.pdf` (Maes et al., 2026) and
`papers/HierarchialWorldModel.pdf` (Zhang et al., 2026, "HWM"). All section/figure
references point back into those PDFs.

---

## 1. LeWorldModel (LeWM) Architecture Summary

### 1.1 Overall pipeline
- JEPA-style **end-to-end** world model trained from raw pixels — no pretrained encoder, no
  EMA target, no stop-gradient, no reconstruction.
- Two modules, trained jointly:
  - `Encoder`  z_t = enc_θ(o_t)
  - `Predictor` ẑ_{t+1} = pred_φ(z_t, a_t)
- Receives offline, reward-free trajectories of pixel obs `o_{1:T}` and actions `a_{1:T}`.
- Total trainable parameters ≈ 15M (≈ 5M encoder + 10M predictor); single-GPU trainable.

### 1.2 Encoder
- Vision Transformer **Tiny** (ViT-Tiny, HuggingFace), patch size 14, 12 layers,
  3 attention heads, hidden dim 192, ~5M params.
- Output `z_t` is the `[CLS]` token of the last encoder layer, then passed through a
  **1-layer MLP + BatchNorm** projector. The projector is required because the final
  ViT LayerNorm would otherwise interfere with the SIGReg anti-collapse objective.
- Embedding dim D = 192.

### 1.3 Predictor
- Transformer with 6 layers, 16 attention heads, 10% dropout (~10M params, ViT-S-class).
- **Action conditioning via AdaLN** (Adaptive LayerNorm) at each layer, with AdaLN params
  zero-initialized so action conditioning ramps up gradually.
- **History length** N = 3 (PushT, OGBench-Cube), 1 (TwoRoom). Causal temporal mask;
  predictor is autoregressive — given history of latents and actions it predicts next
  latent.
- A second projector identical to the encoder's is applied on the predictor output.

### 1.4 Latent space structure
- Compact (D=192), continuous, single-vector-per-frame (no per-patch tokens at planning
  time — this is what gives LeWM ~200× fewer planning tokens than DINO-WM).
- Trained to be **isotropic Gaussian-distributed** via SIGReg.
- Empirically the space encodes physical structure: linear/MLP probes recover agent
  position, block position, block angle on PushT (Tab. 1); t-SNE preserves spatial
  layout (Fig. 9); a separately trained decoder recovers the visual scene from a
  single 192-d embedding (Fig. 8).
- Trajectories become temporally "straight" in this space as an emergent property.

### 1.5 Training objective
Two terms, single tunable weight λ:
- **Prediction loss** (teacher forcing, MSE on next embedding):
  L_pred = ‖ẑ_{t+1} − z_{t+1}‖²₂ , ẑ_{t+1} = pred_φ(z_t, a_t)
- **SIGReg** (Sketched Isotropic Gaussian Regularizer): project Z onto M random
  unit-norm directions u^(m) ∈ S^{D−1}, apply the univariate Epps–Pulley normality
  test on each projection, average. By Cramér–Wold, matching all 1-D marginals to
  N(0,1) ⇒ matching the joint to N(0,I).
  - Defaults: M = 1024, λ = 0.1. Performance is largely insensitive to M and to the
    Epps–Pulley quadrature, so λ is the only effective hyperparameter and is found
    via log-time bisection.
- **Total**: L_LeWM = L_pred + λ · SIGReg(Z) (Eq. 3).
- No EMA, no stop-grad, no auxiliary heads.

### 1.6 Data / preprocessing
- **Frame-skip 5** by default: 5 consecutive primitive actions are grouped into a
  single "action block" `a_t`, and frames are subsampled accordingly.
  Each LeWM "step" therefore = 5 environment steps.
- Batch size 128, sub-trajectories of 4 frames × 4 action blocks = 20 env steps per
  sample. Frames are 224 × 224.
- Trained on PushT and OGBench-Cube for 10 epochs each.

### 1.7 Planning / imagination (latent MPC)
- Performed at inference using the **frozen** trained world model.
- Given initial obs o_1 and goal obs o_g, encode z_1 = enc(o_1), z_g = enc(o_g).
- Roll out predictor autoregressively for H steps under a candidate action sequence
  a_{1:H} (in action blocks).
- Cost is **terminal goal-matching in latent space**:
  C(ẑ_H) = ‖ẑ_H − z_g‖²₂ (Eq. 4).
- Optimise via **CEM** (Alg. 2, App. B):
  - 300 candidate sequences per iteration.
  - 30 iterations on PushT, 10 on the easier envs.
  - Top **30 elites** update sampling N(μ, Σ); init Σ = I, init variance 1.
  - Planning horizon H = 5 LeWM steps = 25 environment steps.
- **Receding-horizon MPC**: optimise H actions, execute the entire optimised sequence
  (K = H), then re-encode the new observation and re-plan. This follows the DINO-WM
  setup [18].
- Full planning completes in < 1 s/run, ~48× faster than DINO-WM (Fig. 3).

### 1.8 Limitations called out by the authors (relevant for us)
- Planning is restricted to **short horizons** because of compounding rollout error in
  the autoregressive predictor.
- The conclusion explicitly names *"Hierarchical world modeling … to address
  long-horizon reasoning and planning"* as the main future direction. This is exactly
  the gap HWM fills.

---

## 2. HWM (Hierarchical Planning with Latent World Models) Summary

### 2.1 Setup and core idea
- Offline MDP, goal-reaching tasks specified by a single goal observation s_g.
- **Top-down hierarchical MPC** purely at inference time. No policies, no skills,
  no rewards, no inverse models. Reuses pretrained low-level world models as-is.
- Two world models share the **same latent space** (encoder E):
  - Low-level **P^(1)(z_{t+1} | z_t, a_t)**: short horizon, primitive actions.
  - High-level **P^(2)(z_{t+h} | z_t, l_t)**: long horizon, **latent macro-actions** l_t.
- This shared-latent design is the key trick — high-level predictions can be used
  *directly* as subgoals for the low-level planner via latent matching.

### 2.2 Macro-actions and the action encoder
- A learned **action encoder A_ψ** (transformer with a CLS token + MLP head) compresses
  a variable-length chunk of primitive actions a_{t_k : t_{k+1}} into a single latent
  macro-action l_{t_k} ∈ R^d_l.
- Variable-length is intentional: each high-level transition can correspond to a
  different number of low-level steps (no fixed stride assumption, except for some
  experiments).
- **Latent action dim is critical** (Fig. 7, §4.3): too small → high-level can't
  represent useful trajectories; too large → high-level proposes subgoals the
  low-level planner can't actually reach.
  - Sweet spot: **d_l = 4** for Franka; d_l = 8 for PointMaze; d_l = 10 for
    Push-T (concatenation of 5 primitive 2-D actions).

### 2.3 High-level world model
- Same backbone class as the low-level model but **scaled up** and conditioned on
  macro-actions instead of primitives.
  - For DINO-WM Push-T: 25M → 75M (layers 6→10, dim 384→768, MLP dim 2048→3072,
    heads 16→12).
- For each trajectory, sample N waypoint indices 1 = t_1 < … < t_N < T. Build the
  k-th high-level transition as (s_{t_k}, a_{t_k:t_{k+1}}, s_{t_{k+1}}).
  - Push-T: trajectory segments 25–70 timesteps long, **N = 5 waypoints**.
  - Franka: segments 0.33–4 s, **N = 3 waypoints**, middle waypoint sampled uniformly.
- Encode states with the **same encoder E** as the low-level model; encode action
  chunks via A_ψ. Feed interleaved (l_{t_k}, z_{t_k}) into P^(2) and predict next
  waypoint latent.
- Loss: teacher-forced L1 on next waypoint latent (Eq. 1):
  L_tf = (1/N) Σ ‖ẑ_{t_{k+1}} − z_{t_{k+1}}‖₁
- Some setups also include a multi-step autoregressive rollout L1 loss.
- Encoder E and the low-level P^(1) are kept **frozen** when training high-level
  in the PointMaze setup; on DROID/Push-T they may be re-used as-is.

### 2.4 Hierarchical planning at inference (top-down)
- **High-level CEM**: optimise macro-action sequence l̃_{1:H}
  E_2(l̃_{1:H}; z_1, z_g) = ‖z_g − P^(2)(l̃_{1:H}; z_1)‖₁
  l*_{1:H} = argmin E_2.
  Unrolling P^(2) under l*_{1:H} yields subgoals z̃_i = P^(2)(l*_{1:i}; z_1).
- **Low-level CEM**: starting from z_1, optimise a primitive action sequence a_{1:h}
  to reach the *first* subgoal z̃_1:
  E_1(â_{1:h}; z_1, z̃_1) = ‖z̃_1 − P^(1)(â_{1:h}; z_1)‖₁
  a*_{1:h} = argmin E_1.
- Execute first action(s), re-encode, re-plan every k env steps (MPC).
- Both levels run **CEM** (or MPPI for Diverse Maze) in parallel on GPU.
- Concrete CEM hyperparameters (Push-T, d = 50, App. C, Tab. 10):
  - High-level: 1500 samples, 40 iters, pred H = 4 macro-steps.
  - Low-level: 900 samples, 20 iters, pred h = 5, replan k = 5.

### 2.5 Why hierarchy helps (Sec. 4 analyses)
- **Long-horizon prediction is more accurate** at the high level: a one-step
  high-level prediction beats a 16-step autoregressive rollout of the low-level model
  beyond ~1.5 s lookahead (Fig. 6) — fewer rollout steps ⇒ less compounded error.
- **Search space is smaller**: optimising H ≪ T macro-actions of dim d_l vs T
  primitives.
- **Non-greedy behaviour** falls out naturally: the high-level can propose subgoals
  that temporarily move *away* from the goal. Flat planners on Franka pick-&-place
  scored 0%; HWM scored 70% from a single goal image (Tab. 1).

### 2.6 Empirical highlights relevant to our scope (Push-T, Cube)
- **Push-T (DINO-WM backbone)**: at d = 75, single-level 17% → hierarchical 61%
  (Tab. 2). Hierarchical also matches single-level success with **3× less planning
  compute** (Fig. 5).
- HWM is described as a **plug-in inference-time abstraction** that works across
  VJEPA2-AC, DINO-WM, and PLDM — i.e. the same recipe should apply to LeWM.

---

## 3. Initial thoughts on plugging HWM into LeWorldModel

LeWM and HWM line up cleanly:

1. **LeWM = ready-made low-level model.** LeWM's (encoder E, predictor pred_φ) is
   exactly P^(1) in HWM terms. It already produces a single shared latent z_t per
   frame — the *shared-latent* requirement that HWM relies on for subgoal transfer
   is satisfied by construction. We do **not** need to retrain it. The first
   integration step is to freeze the trained LeWM checkpoint and treat it as P^(1).

2. **Frame-skip needs accounting for.** LeWM already groups 5 primitive actions into
   one action block, so what HWM calls a "primitive action" a_t is, in LeWM, a
   5-step block. That makes LeWM's effective horizon (H = 5 LeWM steps = 25 env
   steps) the natural inner loop for HWM's low-level planner (HWM's Push-T low-level
   planner uses pred h = 5, k = 5 — same numbers). Macro-actions then span multiple
   LeWM blocks, e.g. 5–14 LeWM blocks ≈ 25–70 env steps for Push-T, matching HWM.

3. **New components to add (kept local to `repos/le-wm/`):**
   - `ActionEncoder A_ψ`: small transformer over a chunk of LeWM action blocks →
     latent macro-action l ∈ R^{d_l}. Start with d_l = 10 for Push-T (matches HWM's
     5×2 setup) and tune by sweep; for OGBench-Cube pick a small d_l (≈ 4) to stay
     in HWM's "reachable subgoal" regime.
   - `HighLevelPredictor P^(2)`: same architecture as LeWM's predictor (transformer
     w/ AdaLN), but conditioned on macro-actions instead of primitive actions, and
     trained on waypoint-spaced latents from frozen LeWM. We can scale it up
     (HWM doubles capacity for Push-T) but should first try same-size to keep diffs
     small.
   - `HierarchicalPlanner`: wraps two CEM passes around the existing LeWM CEM
     utility — high-level CEM proposes subgoal l*_{1:H}, low-level CEM (essentially
     today's planner) optimises primitives toward subgoal z̃_1.

4. **Dataset re-use.** The same offline trajectories LeWM trains on already cover
   HWM's needs. We only need a new sampler that, per trajectory, draws N waypoint
   indices (variable spacing) and emits the (z_{t_k}, a_{t_k:t_{k+1}}, z_{t_{k+1}})
   triples HWM trains on. The encoder is frozen, so latents can be precomputed.

5. **Loss.** High-level training is just teacher-forced L1 (HWM Eq. 1) on next
   waypoint latent. We do **not** need a fresh SIGReg on P^(2) — the latent space is
   inherited from LeWM's encoder, which is already SIGReg-regularized. This keeps
   the change minimal and preserves LeWM's "two-term, one hyperparameter" property
   for the parts we leave alone.

6. **What stays untouched.** LeWM training pipeline, encoder, low-level predictor,
   SIGReg, the existing single-level CEM/MPC code path. Hierarchy is an additive
   wrapper that can be toggled by config, leaving flat-LeWM behaviour identical when
   off — this matches the "preserve existing flat behaviour" engineering constraint.

7. **Evaluation.** Push-T and OGBench-Cube are the right targets: HWM already
   reports Push-T numbers (DINO-WM backbone) and LeWM beats DINO-WM on Push-T as a
   flat baseline (90% vs 13% at HWM's typical horizons). Reproducing HWM-style
   long-horizon Push-T (d ∈ {25, 50, 75}) on top of LeWM is the most direct way to
   show the hierarchy actually buys long-horizon performance. OGBench-Cube is the
   harder visual case where we'd want to verify hierarchy doesn't regress.

8. **Watch-outs flagged by the papers:**
   - HWM needs d_l tuned per environment (Fig. 7); plan to sweep.
   - LeWM's SIGReg can struggle in very low-intrinsic-dimensionality envs
     (TwoRoom). Push-T and Cube are richer, so this should be fine.
   - Compounding rollout error in P^(1) is exactly what hierarchy is supposed to
     mitigate, so success metric should include performance at *long* horizons
     (d ≥ 50), not just short.
