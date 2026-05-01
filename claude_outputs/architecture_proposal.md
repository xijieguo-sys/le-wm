# Phase 4: Architecture Proposal — Hierarchical LeWorldModel

Blueprint for Phase 5. Every decision below is grounded in code citations
into `repos/le-wm/`, `repos/stable-pretraining/`, `repos/stable-worldmodel/`
or in the HWM paper. The HWM_PLDM repo is treated as an idea-level reference
only — its concrete choices (MPPI, MSE, MLP `A_ψ`, fixed stride, conv-spatial
latents) are PLDM-specific and **do not** carry over.

Primary target: **Push-T**. Secondary target: **OGBench-Cube** (only if time
permits). All file paths are absolute under
`/oscar/home/xguo84/final_project/repos/le-wm/` unless noted.

---

## 0. Naming convention used throughout

| Symbol | Code name | Lives in | Role |
|---|---|---|---|
| `E` | `JEPA.encoder` (frozen) + `JEPA.projector` (frozen) | `jepa.py` | Pixel → latent `z ∈ R^192` |
| `P^(1)` | `JEPA.predictor` + `JEPA.pred_proj` (frozen) | `jepa.py`, `module.py` | Low-level dynamics |
| `Embedder` | `JEPA.action_encoder` (frozen) | `module.py:189` | Per-block action → 192-D AdaLN cond |
| `A_ψ` | **NEW** `MacroActionEncoder` | `module.py` (extension) | Variable-length action chunk → `l ∈ R^{d_l}` |
| `MacroEmbedder` | **NEW** `nn.Linear(d_l, 192)` | inside `HighLevelWorldModel` | `l → 192-D` AdaLN cond (predictor expects 192-D `c`) |
| `P^(2)` | **NEW** `ARPredictor` instance | `hwm.py` | High-level dynamics on macro-actions |
| `HighLevelWorldModel` | **NEW** `JEPA`-shaped wrapper | `hwm.py` | Mirrors `JEPA` API for `swm.policy.AutoCostModel` |
| `MacroPrior` | **NEW** `(μ_l, σ_l)` running stats | buffers on `HighLevelWorldModel` | Empirical distribution of `A_ψ` outputs from training; CEM init + soft penalty |
| Hierarchical solver | **NEW** `HierarchicalCEMSolver` | `planner.py` | Drop-in for `CEMSolver`; wraps two CEM instances and the cost adapters |

The "two action encoders" gotcha (analysis §1.3, plan §1) is preserved by
construction: `Embedder` stays inside `JEPA`; `MacroActionEncoder` is a fresh
class. Phase 5 must not refactor `Embedder`.

**No new policy class.** `HierarchicalCEMSolver` matches the existing
`swm.solver.CEMSolver` contract (`configure(*, action_space, n_envs, config)`
and `solve(info_dict, init_action=None) → dict`, verified at
`stable-worldmodel/stable_worldmodel/solver/cem.py:53–117`), so the existing
`swm.policy.WorldModelPolicy` drives it unchanged.

---

## 1. New modules / classes

### 1.1 `MacroActionEncoder` — `module.py` (additive)

HWM's `A_ψ`: a small bidirectional transformer with a learnable CLS token and
an MLP head that compresses a variable-length chunk of LeWM action blocks
into a single macro-action latent.

- **Input**: `actions: (B, L, A_block)` and `mask: (B, L)` bool, where
  `A_block = frameskip × action_dim` (10 for Push-T, 4 for Cube — per
  `train.py:92`, `effective_act_dim = frameskip × wm.action_dim`).
  `L` is the chunk length in LeWM action blocks (variable across samples,
  right-padded; `L_max ≤ macro_max_blocks`).
- **Output**: `l: (B, d_l)` — CLS token through MLP head.
- **Architecture**:
  - Token embedding: `nn.Linear(A_block, d_token)` lifts each block (no
    Conv1d-over-time; the temporal axis is too short to benefit, and a
    plain Linear is what HWM describes).
  - Learnable positional embedding `(1, L_max + 1, d_token)`; CLS is index 0.
  - 2 standard `Block`s (reuse `module.Block`), `heads = 4`, `dim_head = 32`,
    `mlp_dim = 4 * d_token`, dropout 0.1. `d_token = 64` initially.
  - **Padding mask**: `Block.attn` currently uses `is_causal=True` with no
    key-padding mask (`module.py:75–85`). For variable-length chunks we
    cannot just reuse `Block` unchanged. Two options:
    (a) Right-pad to `L_max` and **drop padded queries** before the MLP
        head — equivalent in effect to masking, since CLS attends to all
        padded keys but we only read CLS.
        *Problem*: padded values still influence CLS via attention.
    (b) Add a sibling `MaskedBlock` that accepts an attention mask,
        OR call `F.scaled_dot_product_attention(..., attn_mask=...)`
        directly inside `MacroActionEncoder` (skip `Block`).
    **Decision**: option (b). Inline a 2-layer attention stack inside
    `MacroActionEncoder` so we control the mask. It's ~30 LOC and avoids
    surgery on the shared `Block`.
  - MLP head: `nn.Linear(d_token, mlp_head_dim) → SiLU → nn.Linear(mlp_head_dim, d_l)`.
- **Hyperparameters in config**: `d_token`, `d_l`, `n_layers=2`, `n_heads=4`,
  `mlp_head_dim=128`, `max_blocks` (= `L_max`).
- **Init**: small (`std=0.02` on Linear, zero-init MLP head's last layer).
  Don't zero-init the attention output — we want gradient signal from
  the start.

### 1.2 `HighLevelWorldModel` — new file `hwm.py`

Wraps a frozen `JEPA` (encoder + projector only) plus a fresh
`MacroActionEncoder`, a fresh `ARPredictor`, and a fresh `pred_proj` to
mirror the `JEPA` public API. Exposes the same
`encode / predict / rollout / criterion / get_cost` contract so it plugs
into `swm.policy.AutoCostModel` (`stable-worldmodel/wm/lewm/lewm.py:7–149`
is a working precedent — `AutoCostModel` finds it by scanning for a
`get_cost` attribute, `stable-worldmodel/policy.py:556–574`).

```
class HighLevelWorldModel(nn.Module):
    encoder         # ← shared frozen JEPA.encoder
    projector       # ← shared frozen JEPA.projector
    macro_encoder   # ← new MacroActionEncoder (A_ψ)              trainable
    macro_embedder  # ← new nn.Linear(d_l, embed_dim=192)          trainable
    predictor       # ← new ARPredictor (num_frames=history_size=3) trainable
    pred_proj       # ← new MLP head (same shape as JEPA.pred_proj) trainable

    # Buffers (not trained, persisted with state_dict — survive pickle):
    macro_mean     : (d_l,)  # μ_l: running mean of A_ψ outputs at training
    macro_std      : (d_l,)  # σ_l: running std of A_ψ outputs at training
    history_size   : int     # =3, mirrors LeWM (cfg.wm.history_size)
    d_l            : int

    encode(info)       # frozen encoder + projector; sets info['emb'].
                       #   For training, also reads info['actions_chunk']
                       #   + info['actions_mask'] and sets info['macro_emb'].
                       #   For planning (CEM), action_candidates ARE
                       #   macro-actions sampled in R^{d_l} and bypass A_ψ —
                       #   see rollout() below.
    predict(emb, mac)  # ARPredictor(emb, mac) → hidden; pred_proj(hidden)
                       # → next_emb of shape (B, T, embed_dim).
    rollout(info, l_cands)
                       # Sliding-window autoregressive rollout with
                       # history_size=3 (mirrors JEPA.rollout's loop at
                       # jepa.py:88–97). l_cands ∈ R^{B × S × H × d_l};
                       # each step lifts l → MacroEmbedder → 192-D, slides
                       # the (z, e_l) context window of length 3.
                       # At step t, context length is L = min(t+1, HS) —
                       # the model is trained to handle any L ∈ {1,...,HS}
                       # via prefix-length sampling (§5.2 step 4).
    criterion(info)        # L1 against goal_emb at last step (HWM Eq. 2).
    get_cost(info, l_cands) # contract for swm.solver.CEMSolver.
```

**Key mechanical detail (`MacroEmbedder`).** In `train.py:91–100`,
`embed_dim = 192` and `hidden_dim = 192` (ViT-Tiny).
`Transformer.cond_proj` becomes `nn.Identity` (`module.py:156–160`). That
means `ARPredictor` requires its conditioning tensor to already be in 192-D.
The existing `Embedder` (`module.py:189–214`) does this lift via its `emb`
linear. For HWM, since macro-actions are `d_l`-D (`d_l = 10` for Push-T per
HWM Tab. 10), we need an explicit small
`MacroEmbedder = nn.Linear(d_l, embed_dim=192)` before passing to the
predictor. **If this projection is missed, `ARPredictor` will silently
consume a `d_l`-shaped tensor through a no-op `cond_proj` and AdaLN will
see the wrong shape.** Phase 5 must include a shape assertion at the
boundary (§7.6 test 3 covers this).

**Sliding-window predictor with prefix-length training (mirrors LeWM).**
Set `num_frames = history_size = 3` on the high-level `ARPredictor`. This
matches LeWM's training convention (`cfg.wm.history_size: 3` in
`config/train/lewm.yaml:47`) and the existing `JEPA.rollout` loop
(`jepa.py:88–97`: `emb_trunc = emb[:, -HS:]`, `act_trunc = act_emb[:, -HS:]`,
`pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]`).
`ARPredictor.pos_embedding` has shape `(1, num_frames, input_dim)`
(`module.py:262`); since `forward` does `x = x + self.pos_embedding[:, :T]`
(`module.py:281–282`), the predictor mechanically accepts any
`T ≤ num_frames`.

Critically, **inference will encounter context lengths shorter than
`HS`**: the high-level rollout starts from a single latent `z_1` and
unrolls `H = 4` macro-steps, so the first two prediction steps see
context lengths 1 and 2. With `H = 4` that's *50%* of the rollout out
of distribution if we train on length-3 contexts only. To fix this,
high-level training samples a **prefix length per item**
`L ~ Uniform{1, …, HS}` and feeds the last `L` of context to the
predictor (§5.2 step 4). The model is therefore trained to handle every
`L ∈ {1, 2, 3}` natively; inference at step `t` uses
`L = min(t + 1, HS)` and reuses `JEPA.rollout`'s `[:, -HS:]` pattern
unchanged (§6.2).

**Encoder freezing.** In `__init__`, copy references to `JEPA.encoder` and
`JEPA.projector` from a passed-in `low_level: JEPA`. Apply
`requires_grad_(False)` and `.eval()` on the shared modules. Construction
order: load `JEPA` checkpoint → freeze → wrap in `HighLevelWorldModel`.

**Fresh trainable `pred_proj` (decision).** We instantiate a *new*
trainable `pred_proj` rather than sharing `JEPA.pred_proj`. Rationale:

- The "shared latent space" invariant that hierarchy relies on is enforced
  by the **loss** (L1 against `(encoder + projector)(o_{t_{k+1}})`, all
  frozen), not by sharing the predictor's output projector.
- The frozen low-level `pred_proj` was trained for one-step low-level
  prediction in 192-D space; constraining `P^(2)`'s last layer to that
  exact mapping limits capacity for no semantic reason — its job is
  different (long-horizon prediction over macro-actions).
- Fresh trainable `pred_proj` gives `P^(2)` an independent decoding path
  while the L1 supervision still anchors its outputs to the *same* latent
  space the encoder/projector defines.

Sharing remains available as an ablation, behind a config switch
`share_pred_proj: false` (default). When `true`, the wrapper aliases
`self.pred_proj = low_level.pred_proj` and applies `requires_grad_(False)`.

**Macro-action prior buffers (`macro_mean`, `macro_std`).** Registered as
non-persistent? **Persistent** — they need to survive
`torch.save(model, path)` (`utils.py:55`) so the planner can use them at
eval time. Maintained as exponential-moving-average buffers updated each
training batch from `A_ψ`'s outputs (§5.2 step 8). At plan time,
`HierarchicalCEMSolver` reads them off the loaded checkpoint to seed
high-level CEM (§6.2 step 3) and to weight the prior penalty (§1.4).

### 1.3 `HierarchicalCEMSolver` — new file `planner.py`

A **drop-in replacement** for `swm.solver.CEMSolver`. Matches the existing
solver contract verified at `stable-worldmodel/stable_worldmodel/solver/cem.py`:

- `configure(self, *, action_space, n_envs, config) → None` — keyword-only
  args; sets `_action_dim = int(np.prod(action_space.shape[1:]))`
  (cem.py:53–61).
- `__call__(...) → solve(...)` (cem.py:87–89).
- `solve(self, info_dict, init_action=None) → dict` — returns
  `{'actions': tensor[n_envs, horizon, action_dim], 'costs': ..., 'mean': ...,
  'var': ...}` (cem.py:114–...).

Because the contract matches, the existing `swm.policy.WorldModelPolicy`
drives the hierarchical solver unchanged. **No new policy class.**

```
class HierarchicalCEMSolver:
    def __init__(self,
                 model_low,            # frozen JEPA loaded by AutoCostModel
                 model_high,           # HighLevelWorldModel loaded by AutoCostModel
                 high_cfg, low_cfg,    # nested kwargs for the two CEMSolver instances
                 high_plan_cfg,        # PlanConfig kwargs for the high level
                 d_l,                  # macro-action dim (read from model_high.d_l)
                 replan_high_every=1,
                 advance_subgoal=False,
                 subgoal_threshold=None,
                 prior_weight=0.1,     # weight on the macro-action prior penalty
                 device='cuda',
                 seed=42):
        # Build the two underlying CEMSolver instances.
        self.solver_low  = CEMSolver(model=SubgoalCostAdapter(model_low),
                                     **low_cfg, device=device, seed=seed)
        self.solver_high = CEMSolver(
            model=HighLevelCostAdapter(model_high, prior_weight=prior_weight),
            **high_cfg, device=device, seed=seed + 1)
        self.model_high = model_high  # for the subgoal extraction step
        self.d_l = d_l
        self.replan_high_every = replan_high_every
        self.high_plan_cfg = high_plan_cfg
        self._cached_subgoal_seq = None  # the full predicted z̃_{1:H}
        self._cached_subgoal = None      # z̃_1 (current low-level target)
        self._steps_since_high = -1      # forces high replan on first call
        self._n_envs = None

    def configure(self, *, action_space, n_envs, config):
        # 1. Low-level solver: pass through the env's real action_space and
        #    the user's plan_config (PlanConfig).
        self.solver_low.configure(
            action_space=action_space, n_envs=n_envs, config=config)

        # 2. High-level solver: synthetic action_space of shape (n_envs, d_l).
        #    cem.py:60 does action_space.shape[1:].prod() — so .shape MUST be
        #    (n_envs, d_l), NOT (d_l,). Otherwise prod(()) == 1.0 silently.
        synth_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_envs, self.d_l), dtype=np.float32)
        synth_plan = swm.PlanConfig(**self.high_plan_cfg)  # action_block=1, etc.
        self.solver_high.configure(
            action_space=synth_space, n_envs=n_envs, config=synth_plan)

        self._n_envs = n_envs

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, info_dict, init_action=None):
        # 1. Bring info to the low-level model's device, encode current and
        #    goal latents ONCE per solve() call. (WorldModelPolicy._prepare_info
        #    does NOT populate emb/goal_emb — verified at policy.py:121–183.
        #    The adapters read these cached latents to skip re-encoding.)
        info_low = self._encode_and_cache(info_dict)

        # 2. Decide whether to replan high (cadence + advance_subgoal logic).
        do_high = (self._cached_subgoal is None
                   or self._steps_since_high % self.replan_high_every == 0
                   or self._should_advance(info_low))

        if do_high:
            # info_high holds z_1 and z_g (no pixels needed past this point).
            info_high = {
                'emb':      info_low['emb'],
                'goal_emb': info_low['goal_emb'],
            }
            # CEM init from MacroPrior μ_l, σ_l (lifted to the right shape).
            init_high = self.model_high.macro_mean.expand(
                self._n_envs, self.solver_high.horizon, self.d_l).clone()
            high_out = self.solver_high.solve(info_high, init_action=init_high)
            # high_out['actions']: (n_envs, H, d_l) — best macro-action seq.
            # Roll it through P^(2) to materialise z̃_{1:H}.
            self._cached_subgoal_seq = self._rollout_subgoals(
                z_init=info_low['emb'], l_seq=high_out['actions'])
            # Pick z̃_1 (index 1 of the rollout; index 0 is z_1 itself —
            # repo_analysis §1.2 gotcha #2).
            self._cached_subgoal = self._cached_subgoal_seq[:, 1]
            self._steps_since_high = 0
        else:
            self._steps_since_high += 1
            # advance_subgoal advances the index into self._cached_subgoal_seq.

        # 3. Inject the current subgoal into info_low (overrides goal_emb).
        info_low_for_low = {**info_low, 'goal_emb': self._cached_subgoal}
        return self.solver_low.solve(info_low_for_low, init_action=init_action)
```

Three notes on the contract:

- **`init_action`**: the existing `WorldModelPolicy.get_action` may pass an
  init-action seed (from the previous-step warm start). We forward it
  unchanged to `solver_low`. The high-level solver gets its own warm start
  from the macro prior `μ_l` (CEM `init_action_distrib` accepts an actions
  tensor, cem.py:91–112).
- **Return shape**: `{'actions': (n_envs, low_horizon, low_action_dim), ...}`.
  `WorldModelPolicy` reads `actions` and ignores the rest, matching how it
  consumes `CEMSolver`'s output today.
- **Device handling**: `info_dict` arrives on CPU (numpy) per
  `_prepare_info`. `self._encode_and_cache` moves pixels to the model
  device, encodes, and stores latents back into the dict.

**`_encode_and_cache(info_dict)`** — explicit:

```
def _encode_and_cache(self, info_dict):
    info = {k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in info_dict.items()}              # avoid in-place mutation
    info = move_tensors_to(info, self.solver_low.device)

    # Encode current pixels.  JEPA.encode mutates in place; we just want emb.
    enc_in = {'pixels': info['pixels']}
    info['emb'] = self.solver_low.model.wrapped.encode(enc_in)['emb']

    # Encode goal pixels ONCE; cache by goal pixel data_ptr() hash so a
    # repeated solve() call on the same goal skips re-encode (mirrors
    # stable-worldmodel/wm/lewm/lewm.py:73–76).
    goal_key = info['goal'].data_ptr()
    if self._goal_cache_key != goal_key:
        info['goal_emb'] = self.solver_low.model.wrapped.encode(
            {'pixels': info['goal']})['emb']
        self._goal_cache = info['goal_emb']
        self._goal_cache_key = goal_key
    else:
        info['goal_emb'] = self._goal_cache
    return info
```

(`solver_low.model.wrapped` is the underlying frozen `JEPA`; the adapter
exposes it as `.wrapped` so the solver can borrow its encoder without
reaching past abstraction layers.)

**`replan_high_every`**: number of `solve()` calls between high-level
re-plans. Default `1` (re-run high-level every step, matching HWM_PLDM's
`two_lvl_planner`). Raise to ≥2 once high-level wall-clock dominates.

**`advance_subgoal` (gated, default OFF)**: if `True`, when
`||z̃_i − z_t||₁ < subgoal_threshold`, advance the cached subgoal index to
`z̃_{i+1}` instead of re-running the high-level CEM. This is the "subgoal
gating" the paper hints at; HWM_PLDM doesn't implement it; we ship it
disabled to reproduce the HWM baseline first.

### 1.4 Cost adapters — inside `planner.py`

Two thin `nn.Module` adapters expose a `get_cost(info, action_candidates)`
matching the `CEMSolver` contract (cem.py:190 calls
`self.model.get_cost(info, candidates)`), each tailored to one level.
Both adapters expose a `.wrapped` attribute pointing at the underlying
world model so `HierarchicalCEMSolver._encode_and_cache` can borrow the
shared frozen encoder without reaching past abstractions.

#### Shape contract — `goal_emb` after CEM expansion

`CEMSolver.solve` expands every tensor in `info_dict` over a sample axis
before calling `get_cost` (`stable-worldmodel/stable_worldmodel/solver/cem.py:144–159`):

```python
v_batch = v_batch.unsqueeze(1).expand(
    current_bs, self.num_samples, *v_batch.shape[1:])
```

So an `info['goal_emb']` we inject at shape `(n_envs, D)` arrives at the
adapter as `(B, S, D)` — already matching the predicted last-step latent.
The adapters MUST NOT call `unsqueeze(1).expand_as(pred)` on it; that
would produce `(B, 1, S, D)` and `expand_as((B, S, D))` would raise on
ndim mismatch.

To stay robust against future callers that might inject `(B, S, D)`
directly (no CEM expansion needed) or `(B, T, D)` / `(B, S, T, D)` via a
different code path, both adapters route through one helper:

```
def _match_goal_shape(goal, pred):
    # pred is (B, S, D); normalise goal into the same shape.
    if goal.ndim == pred.ndim - 1:        # (B, D)  → unsqueeze sample axis
        goal = goal.unsqueeze(1)
    elif goal.ndim == pred.ndim + 1:      # (B, S, T, D) → take last time step
        goal = goal[..., -1, :]
    elif goal.ndim != pred.ndim:
        raise ValueError(
            f'unexpected goal_emb shape {tuple(goal.shape)} '
            f'vs pred {tuple(pred.shape)}')
    return goal.expand_as(pred)
```

Both adapters call this helper exactly once per `get_cost` invocation,
right before the L1 difference. §7.6 test 4 verifies the contract on
all three input shapes.

#### 1.4.1 `SubgoalCostAdapter(model_low)` — wraps frozen `JEPA`

Differs from `JEPA.get_cost` (jepa.py:128–153) in two ways:

- **No goal re-encoding.** `JEPA.get_cost` calls `self.encode(goal)` every
  CEM iteration, re-encoding goal pixels each time. The low-level subgoal
  is a *latent* `z̃_i` precomputed by the hierarchical solver — there are
  no goal pixels here. The adapter reads `info['goal_emb']` (which the
  solver set to the subgoal latent) directly.
- **L1, not MSE.** `JEPA.criterion` (jepa.py:112–126) uses `F.mse_loss`;
  paper Eq. 2 specifies L1.

```
class SubgoalCostAdapter(nn.Module):
    def __init__(self, model_low: JEPA):
        super().__init__()
        self.wrapped = model_low

    def get_cost(self, info, action_candidates):
        # info['emb'] and info['goal_emb'] (= z̃_i) MUST already be set by
        # HierarchicalCEMSolver._encode_and_cache. Assert this — silent
        # absence would trigger JEPA.encode and re-encode the wrong pixels.
        assert 'emb' in info and 'goal_emb' in info, \
            'SubgoalCostAdapter requires pre-cached emb/goal_emb'

        info = {k: v for k, v in info.items()}        # shallow copy; rollout
                                                       # mutates 'predicted_emb'
        info = self.wrapped.rollout(info, action_candidates)  # jepa.py:61–110
        pred = info['predicted_emb'][..., -1, :]       # (B, S, D)
        goal = _match_goal_shape(info['goal_emb'], pred)
        return (pred - goal.detach()).abs().sum(-1)    # (B, S)
```

This is **zero edits to `jepa.py`**, satisfying the "preserve flat LeWM
behaviour" constraint (analysis §1.5). The optional `distance` arg on
`JEPA.criterion` is therefore **dropped** from the architecture.

#### 1.4.2 `HighLevelCostAdapter(model_high, prior_weight=0.1)` — wraps `HighLevelWorldModel`

```
class HighLevelCostAdapter(nn.Module):
    def __init__(self, model_high: HighLevelWorldModel, prior_weight=0.1):
        super().__init__()
        self.wrapped = model_high
        self.prior_weight = prior_weight

    def get_cost(self, info, l_candidates):
        # info['emb'] and info['goal_emb'] are set by the solver (no pixels).
        assert 'emb' in info and 'goal_emb' in info
        info = {k: v for k, v in info.items()}
        info = self.wrapped.rollout(info, l_candidates)   # sliding-window
        pred = info['predicted_emb'][..., -1, :]          # (B, S, D)
        goal = _match_goal_shape(info['goal_emb'], pred)
        cost_l1 = (pred - goal.detach()).abs().sum(-1)    # (B, S)

        # Macro-action prior penalty: pull samples toward the empirical
        # distribution observed at training time. ((l - μ) / σ)^2 averaged
        # over (H, d_l). Shape arithmetic: l_candidates (B, S, H, d_l).
        mu  = self.wrapped.macro_mean.view(1, 1, 1, -1)    # (1,1,1,d_l)
        std = self.wrapped.macro_std.view(1, 1, 1, -1).clamp_min(1e-3)
        prior = ((l_candidates - mu) / std).square().mean(dim=(-1, -2))  # (B,S)
        return cost_l1 + self.prior_weight * prior
```

`prior_weight = 0.1` mirrors HWM_PLDM's `z_reg_coeff` ballpark; a sweep
in `{0.01, 0.1, 0.3, 1.0}` is a Phase-5 ablation if Push-T results are
sensitive.

### 1.5 `WaypointSubtrajectoryDataset` — new file `data.py`

Wraps `swm.data.HDF5Dataset`. Per item, samples N waypoint indices from a
single trajectory and returns:

```
{
    "waypoint_pixels":  (N, C, H, W)   # frames at t_1, ..., t_N
    "actions_chunk":    (N-1, L_max, A_block)  # right-padded
    "actions_mask":     (N-1, L_max)   bool
    "stats" (optional): for sanity logging
}
```

Sampling modes (config-controlled):

- **Variable-stride** (default; HWM paper recipe). Per trajectory of length
  `T_ep` LeWM blocks: pick `t_1 = 0`; iteratively pick
  `t_{k+1} ~ Uniform(t_k + min_blocks, t_k + max_blocks)` clipped to
  `T_ep − 1`. Stop at `t_N` where `N` is `min(N_target, blocks_remaining)`.
  - Push-T: `min_blocks=5, max_blocks=14, N_target=5`. (5–14 LeWM blocks at
    frameskip 5 = 25–70 env steps, matches paper.)
  - Cube: `min_blocks=3, max_blocks=12, N_target=3` initially (paper uses
    Franka-like 3 waypoints over 0.33–4 s).
- **Fixed-stride** (HWM_PLDM-style fallback): `t_k = k × stride`. Useful
  for sanity checks only.

**Variable `L_max`**: pick a global `L_max = max_blocks` (= 14 for Push-T)
so chunks fit a fixed tensor; pad shorter ones. The action chunk for the
*k*-th transition has effective length `t_{k+1} − t_k`.

**Encoder caching (optimization, do later)**: since the encoder is frozen,
waypoint latents `z_{t_k}` could be precomputed and cached on disk. Phase 5
should ship this **without** caching first (encode in the forward pass) and
add the cache only if epoch time is dominated by encoding.

---

## 2. Existing modules / classes — reuse policy

| Existing | Decision | Why |
|---|---|---|
| `JEPA` (`jepa.py`) | **Reuse instance, frozen** | `repos/le-wm/jepa.py:11–153`; serves as low-level model. Shared `encoder`, `projector`, `pred_proj` propagate to high-level path. |
| `JEPA.get_cost` | **Bypass via `SubgoalCostAdapter`** for the low-level path | `get_cost` re-encodes `info['goal']` pixels (`jepa.py:138–148`); the low-level subgoal is a latent, not pixels. Adapter reads pre-cached `info['goal_emb']` directly (§1.4.1). |
| `JEPA.encode`, `JEPA.rollout` | **Reuse from inside the adapters and the solver's `_encode_and_cache`** | Both methods are correct; we just want to control *when* they run (once per `solve` call, not once per CEM sample). |
| `JEPA.pred_proj` | **Do NOT share by default** | Fresh trainable `pred_proj` for `HighLevelWorldModel` (§1.2). Sharing available as `share_pred_proj: true` ablation. |
| `module.ARPredictor` | **Reuse class, fresh instance** for `P^(2)` with `num_frames=3` | `module.py:244–285` accepts arbitrary `(x, c)`. Sliding-window context length matches LeWM (`cfg.wm.history_size`). |
| `module.Block` | **Reuse for `MacroActionEncoder` body? No** | `Block.attn` has no key-padding mask. Inline a masked attention stack instead (§1.1). |
| `module.MLP` | **Reuse** for `MacroActionEncoder` MLP head and the high-level `pred_proj` | `module.py:217–241`. |
| `swm.solver.CEMSolver` | **Reuse twice, one per level**, behind `HierarchicalCEMSolver` | Same `model.get_cost(info, candidates)` contract (`stable-worldmodel/stable_worldmodel/solver/cem.py:190`). High-level instance gets a synthetic `(n_envs, d_l)` action space (§1.3). |
| `swm.policy.WorldModelPolicy` | **Reuse as-is**, NOT subclassed | Drives `HierarchicalCEMSolver` because the latter matches the `configure(*, action_space, n_envs, config) + solve(info_dict, init_action) → dict` contract (§1.3). No new policy file. |
| `swm.policy.AutoCostModel` | **Reuse twice**, one to load `JEPA`, one to load `HighLevelWorldModel` | Loader scans for a `get_cost` attribute (`stable-worldmodel/policy.py:556–574`); both classes expose it. No checkpoint format change needed (analysis §2.4). |
| `swm.data.HDF5Dataset` | **Reuse, wrap** | Episodes already accessible by `episode_idx` + `step_idx`; we add a sampler on top. |
| `spt.Module`, `spt.Manager`, `spt.data.DataModule`, `spt.data.transforms.Compose`, `spt.optim.lr_scheduler.LinearWarmupCosineAnnealingLR` | **Reuse all** | High-level training reuses the same scaffolding as `train.py`. |
| `swm.World` (env wrapper) | **Reuse** | Hierarchy is purely an inference-time abstraction; the env interaction layer is unchanged. |
| `swm.solver.MPPISolver` | **Available, unused** | We stay on CEM for Push-T/Cube. |

**No edits** to anything outside `repos/le-wm/`.

---

## 3. Config changes

All under `repos/le-wm/config/`. Existing files stay untouched; new files
are additive and selected via Hydra overrides.

### 3.1 New training configs

`config/train/hwm.yaml` — sibling of `lewm.yaml`. Drives `train_highlevel.py`.

```yaml
defaults:
  - _self_
  - data: pusht_waypoints   # new

output_model_name: hwm
subdir: ${hydra:job.id}
seed: 3072
img_size: 224
patch_size: 14
encoder_scale: tiny

# checkpoint to load for the frozen low-level world model
low_level_ckpt: ???   # path to a JEPA pickle (utils.py:55)

trainer:
  max_epochs: 500       # HWM Tab. 7 (Push-T)
  devices: auto
  accelerator: gpu
  precision: bf16
  gradient_clip_val: 1.0

loader:
  batch_size: 64        # smaller than flat training: each item is N=5 frames
  num_workers: 6
  persistent_workers: True

optimizer:
  type: AdamW
  lr: 1e-4
  weight_decay: 1e-3

wm:
  type: hwm
  d_l: 10               # Push-T: 5 primitive 2-D actions concatenated
  history_size: 3       # K=3 high-level latents in P^(2)'s context.
                        # MUST equal predictor.num_frames (passed below).
  num_preds: 1
  embed_dim: 192        # MUST match the frozen encoder's projector out-dim
  share_pred_proj: false  # default: fresh trainable pred_proj (§1.2)

macro_encoder:
  d_token: 64
  n_layers: 2
  n_heads: 4
  mlp_head_dim: 128
  max_blocks: 14        # L_max for Push-T (matches max waypoint stride)

macro_prior:
  ema_momentum: 0.99    # EMA coefficient for μ_l, σ_l updates (§5.2 step 8)
  init_std: 1.0         # σ_l initial value before any updates

predictor:                # high-level ARPredictor; same shape as flat
  num_frames: 3           # MUST equal wm.history_size (sliding window)
  depth: 6
  heads: 16
  mlp_dim: 2048
  dim_head: 64
  dropout: 0.1
  emb_dropout: 0.0

loss:
  distance: l1            # HWM Eq. 1
  teacher_force_weight: 1.0
  rollout_weight: 0.0     # Push-T default per HWM Tab. 7
```

`config/train/data/pusht_waypoints.yaml` — sibling of `pusht.yaml`:

```yaml
defaults:
  - pusht
  - _self_

waypoint_sampler:
  mode: variable
  n_target: 5
  min_blocks: 5
  max_blocks: 14
  seed: ${seed}
```

`config/train/data/ogb_waypoints.yaml` — sibling of `ogb.yaml` with
`n_target: 3, min_blocks: 3, max_blocks: 12`. Only used if/when we
target Cube training.

### 3.2 New eval configs

`config/eval/solver/hcem.yaml`:

```yaml
# eval.py is run from repos/le-wm/, so flat module path (no le_wm package).
_target_: planner.HierarchicalCEMSolver
model_low:  ???            # injected by eval.py from cfg.policy
model_high: ???            # injected by eval.py from cfg.policy_high
d_l: ???                   # read from model_high.d_l at construction
replan_high_every: 1
advance_subgoal: false
subgoal_threshold: null
prior_weight: 0.1
device: cuda
seed: ${seed}

# PlanConfig kwargs for the high-level CEMSolver (action_block=1 because
# macro-actions are not blocked).
high_plan_cfg:
  horizon: 4               # H = 4 macro-steps (HWM Tab. 10)
  receding_horizon: 1
  action_block: 1

high_cfg:
  num_samples: 1500        # HWM Tab. 10, Push-T d=50
  n_steps: 40
  topk: 150                # 10% elites
  var_scale: 1.0
  batch_size: 1

low_cfg:
  num_samples: 900         # HWM Tab. 10
  n_steps: 20
  topk: 90
  var_scale: 1.0
  batch_size: 1
```

`config/eval/pusht_hwm.yaml` — sibling of `pusht.yaml`:

```yaml
defaults:
  - launcher: local
  - solver: hcem
  - _self_

world:
  env_name: swm/PushT-v1
  num_envs: ${eval.num_eval}
  max_episode_steps: ???
  history_size: 1
  frame_skip: 1

dataset:
  stats: ${eval.dataset_name}
  keys_to_cache: [action, proprio, state]

seed: 42

# TWO ckpt names — eval.py loads both via swm.policy.AutoCostModel and
# wires them into the solver as model_low and model_high. high_plan_cfg
# lives inside config/eval/solver/hcem.yaml.
policy:      ???        # low-level JEPA ckpt
policy_high: ???        # high-level HWM ckpt

plan_config:             # passed to WorldModelPolicy → solver.configure as `config`
  horizon: 5             # low-level pred horizon (LeWM blocks)
  receding_horizon: 5    # = k = 5 LeWM blocks (HWM Tab. 10 Push-T)
  action_block: 5

eval:
  num_eval: 50
  goal_offset_steps: 25  # vary in {25, 50, 75} for the long-horizon sweep
  eval_budget: 50
  img_size: 224
  dataset_name: pusht_expert_train
  callables:
    - method: _set_state
      args: {state: {value: state}}
    - method: _set_goal_state
      args: {goal_state: {value: goal_state}}

output:
  filename: pusht_hwm_results.txt
```

`config/eval/cube_hwm.yaml` — analogous, retargeted hyperparameters.

### 3.3 Modified config: none

`config/eval/pusht.yaml`, `config/eval/cube.yaml`, etc. stay byte-identical.
Existing flat behaviour is preserved.

---

## 4. Dataset changes

Push-T and Cube data already exists in HDF5 form; we do not touch the
underlying datasets. The wrapper `WaypointSubtrajectoryDataset` (§1.5) is
the only new dataset code.

Two subtle invariants Phase 5 must respect, both grounded in `jepa.py`:

1. **Action units = LeWM blocks of `frameskip` primitives**.
   `train.py:92`: `effective_act_dim = frameskip × wm.action_dim`.
   The HDF5 dataset's `action` column is already grouped this way for
   Push-T (10-D rows). Waypoint indices are therefore in *LeWM blocks*,
   not raw env steps. Conversions: 1 block = 5 env steps (Push-T), 1 block
   = 5 env steps (Cube; same `frameskip`).

2. **Episode boundaries**. `eval.py:113–120` keys on `episode_idx` (or
   `ep_idx`) + `step_idx`. The waypoint sampler must respect episode
   boundaries — sample within a single episode, never spanning two.
   `swm.data.HDF5Dataset.get_col_data('episode_idx')` is the existing
   accessor.

A small risk: variable-stride sampling produces variable `L` per row,
which doesn't batch directly. The dataset emits already-padded
`(N-1, L_max, A_block)` and the corresponding mask. `spt.data.transforms`
won't interfere — they touch `pixels` and per-column scalars, not the
chunk tensor.

`spt.data.transforms.Compose` is reused as in `train.py` for `pixels`
preprocessing; the sampler runs *before* transforms (waypoint *indices*
are independent of pixel processing).

---

## 5. Training flow — high-level world model

Driver: new file `train_highlevel.py`, structurally a sibling of `train.py`.
We deliberately keep `train.py` untouched.

### 5.1 Construction

1. Load frozen `JEPA` from `cfg.low_level_ckpt`. Apply
   `requires_grad_(False)` and `.eval()` on encoder, projector,
   `action_encoder` (the low-level `Embedder`), `predictor`, and
   `pred_proj`. The low-level `predictor` is not used during high-level
   training itself, but freezing keeps `state_dict()` shapes stable in
   case we later jointly fine-tune.
2. Build `MacroActionEncoder` from `cfg.macro_encoder`.
3. Build `MacroEmbedder = nn.Linear(d_l, embed_dim=192)`. Init small
   (std 0.02). Trainable.
4. Build a fresh `ARPredictor` for `P^(2)` from `cfg.predictor` with
   `num_frames = cfg.wm.history_size = 3` (sliding window, §1.2).
5. Build a fresh trainable `pred_proj = MLP(input_dim=hidden_dim,
   output_dim=embed_dim, hidden_dim=2048, norm_fn=BatchNorm1d)` matching
   `JEPA.pred_proj`'s shape (`train.py:111–116`). Default trainable;
   share-with-frozen as a `cfg.wm.share_pred_proj` ablation.
6. Register macro-prior buffers on `HighLevelWorldModel`:
   `register_buffer('macro_mean', torch.zeros(d_l))`,
   `register_buffer('macro_std', torch.full((d_l,), cfg.macro_prior.init_std))`.
   Persistent so they survive `torch.save(model, path)`.
7. Wrap in `HighLevelWorldModel`. Move to GPU.
8. Wrap in `spt.Module(model=hwm, forward=hwm_forward, optim=...)` to
   reuse Lightning + scheduler scaffolding.

### 5.2 Per-batch forward (`hwm_forward`)

Input batch from `WaypointSubtrajectoryDataset`:

```
batch = {
    "waypoint_pixels": (B, N, C, H, W),
    "actions_chunk":   (B, N-1, L_max, A_block),
    "actions_mask":    (B, N-1, L_max) bool,
}
```

Sliding-window training over `N` waypoints with **variable prefix
length** `L ∈ {1, …, HS}` and `HS = history_size = 3`. Training the
predictor on every `L` is necessary because inference rollouts begin
with a single latent and only reach steady-state context length `HS`
after `HS − 1` steps — for `H = 4` macro-steps that's half the rollout.
Mechanically, `ARPredictor.forward` already supports any `T ≤ num_frames`
via `pos_embedding[:, :T]` (`module.py:281–282`); the issue is *training
distribution*, not architecture.

Two equivalent ways to schedule the windows; we use the first:

- **Random prefix per item** (chosen): for each batch item, sample
  `L ~ Uniform{1, …, HS}` and a *target index* `t ∈ {L, …, N − 1}`
  uniformly. Predict `z_W[:, t]` from the last `L` context entries
  ending at `t − 1`. Each gradient step sees a balanced mix of context
  lengths.
- **All windows per item**: enumerate every `(L, t)` pair with
  `L ∈ {1, …, HS}` and `t ∈ {L, …, N − 1}`. Higher signal per item but
  more compute and a fixed ratio of `L=1` windows.

Steps:

1. **Encode waypoints with frozen encoder.**
   Reshape to `(B*N, C, H, W)`, run `encoder` + `projector`, reshape
   back to `z_W = (B, N, embed_dim)`. Detach (encoder frozen).
2. **Encode action chunks.** Flatten `(B, N-1, L_max, A_block) →
   (B*(N-1), L_max, A_block)` and the matching mask. Run
   `MacroActionEncoder` to get `l = (B*(N-1), d_l)`, reshape to
   `(B, N-1, d_l)`.
3. **Lift to AdaLN dim.** `e_l = MacroEmbedder(l) → (B, N-1, embed_dim)`.
4. **Sample a prefix-length window per item.**
   Per batch index `b`:
   - Draw `L_b ~ Uniform{1, …, HS}` (e.g. via a single `torch.randint`).
   - Draw a target index `t_b ~ Uniform{L_b, …, N − 1}`.
   - Take `z_ctx_b = z_W[b, t_b - L_b : t_b]`            shape `(L_b, D)`
   - Take `e_ctx_b = e_l[b, t_b - L_b : t_b]`            shape `(L_b, D)`
   - Take `z_tgt_b = z_W[b, t_b]`                        shape `(D,)`
   Because `L_b` varies across the batch, stack with **right-aligned**
   left-padding of zeros up to length `HS` and pass the predictor a
   length parameter (it slices `pos_embedding[:, :L_b]` per item — see
   step 5). The padded slots are never read because the predictor
   forward processes only the first `L_b` positions.

   Implementation note: PyTorch attention with key-padding masks
   handles batched variable lengths cleanly, but `module.ConditionalBlock`
   does not currently take a key-padding mask
   (`module.py:88–111`, `Attention.forward` only knows `is_causal`).
   For Phase 5, implement the prefix sampling as **per-length
   batching**: group items in the batch by their sampled `L_b`,
   forward each group separately (all items in a group share the same
   `L`, so the standard predictor call works), and concatenate
   gradients. This is ~15 LOC in `hwm_forward`, avoids touching
   `ConditionalBlock`, and amortises across `HS = 3` calls per
   training step.
5. **Predict per group.** For each `L`-group:
   `ẑ = self.predict(z_ctx_L, e_ctx_L)[:, -1]`, shape `(B_L, D)` where
   `B_L` is the number of items that drew length `L`. Internally:
   `pred_proj(ARPredictor(z_ctx_L, e_ctx_L))[:, -1]`.
   Concatenate `ẑ` and `z_tgt` across groups → `ẑ_all (B, D)`,
   `z_tgt_all (B, D)`.
6. **L1 loss (HWM Eq. 1).**
   `L_tf = || ẑ_all − z_tgt_all.detach() ||_1.mean()`.
   `.detach()` on the target makes it explicit we never backprop through
   the frozen encoder.
7. **No SIGReg** at the high level (latent space inherited from frozen
   encoder; analysis §1.4, plan §4).
8. **Update macro-prior EMA buffers** (no_grad):
   ```
   l_flat = l.reshape(-1, d_l).detach()  # (B*(N-1), d_l)
   batch_mean = l_flat.mean(0)
   batch_std  = l_flat.std(0)
   m = cfg.macro_prior.ema_momentum
   self.macro_mean.lerp_(batch_mean, 1 - m)
   self.macro_std.lerp_(batch_std,  1 - m)
   ```
   Tracks the empirical distribution of `A_ψ` outputs. Used by
   `HighLevelCostAdapter` (prior penalty) and `HierarchicalCEMSolver`
   (CEM init).
9. **Optional rollout loss** (`cfg.loss.rollout_weight > 0`, default 0):
   autoregressively unroll `P^(2)` from `z_W[:, :HS]` under
   `e_l[:, HS:]`, L1 against the corresponding ground-truth waypoints.
   Default off for Push-T per HWM Tab. 7; consider enabling for Cube
   only if teacher forcing alone underperforms.
10. **Logging**: per-epoch `L_tf`, `||l||₂` mean/std,
    `macro_mean.norm()`, `macro_std.mean()`, plus the horizon-vs-error
    diagnostic (§5.5).

### 5.3 Optimizer

AdamW, lr 1e-4, weight decay 1e-3, `LinearWarmupCosineAnnealingLR`. Same
shape as `train.py:30–34`. **Build the optimizer's `params` from
`filter(lambda p: p.requires_grad, model.parameters())`** so the frozen
encoder/projector/low-level-predictor never appear in any optimizer
group. Trainable set: `macro_encoder`, `macro_embedder`, `P^(2)` (new
`ARPredictor`), and `pred_proj` (new MLP). ~10–15M parameters total.

### 5.4 Checkpointing

Use `utils.ModelObjectCallBack` (same as `train.py:159`). The pickled
object will be a `HighLevelWorldModel` — `swm.policy.AutoCostModel` finds
its `get_cost` attribute by attribute scan
(`stable-worldmodel/policy.py:556–574`), so no custom format is needed.

### 5.5 Validation metrics during training

- L1 teacher-forced loss on a held-out validation split (use
  `spt.data.random_split` matching `train.py`).
- One-step prediction error vs k-step low-level autoregressive
  prediction error, sampled at evaluation horizons {0.5, 1.0, 1.5, 2.0}
  seconds. Logging only — replicates HWM Fig. 6 and is the diagnostic
  we'll trust before ever running planning eval.

---

## 6. Planning flow — hierarchical CEM at inference

Driver: extended `eval.py` (~10 LOC change, see §7). The outer MPC loop is
the existing `swm.policy.WorldModelPolicy` — unchanged. All hierarchy lives
inside `HierarchicalCEMSolver`.

### 6.1 Outer MPC loop (per env, per timestep)

`WorldModelPolicy._prepare_info` (`stable-worldmodel/stable_worldmodel/policy.py:121–183`)
preprocesses pixels via `process` / `transform` and returns a dict with
`pixels`, `goal`, etc. **It does NOT call `encode` on anything** — verified
by reading the method body. So `info['emb']` / `info['goal_emb']` are
*not* present when control reaches `solver.solve(info_dict)`.

The hierarchical solver therefore encodes both latents itself in
`_encode_and_cache(info_dict)` (§1.3). Goal pixels are encoded once per
unique goal (memoised by `data_ptr()`); the current-obs pixels are
encoded once per `solve()` call.

### 6.2 Inside `HierarchicalCEMSolver.solve(info_dict, init_action=None)`

1. **Encode and cache.** `info = self._encode_and_cache(info_dict)`. Now
   `info['emb']` (= `z_1`) and `info['goal_emb']` (= `z_g`) are tensors
   on the model device. `info_dict` itself is not mutated.
2. **Decide whether to replan high.** Yes if any of:
   - `self._cached_subgoal is None` (first call ever, or after reset),
   - `self._steps_since_high % self.replan_high_every == 0`,
   - `self.advance_subgoal` is True and the current cached subgoal index
     has been reached (`||z̃_i − z_1||_1 < subgoal_threshold`) AND we
     have a fresh `z̃_{i+1}` in `self._cached_subgoal_seq`.
3. **High-level CEM** (when replanning):
   - `info_high = {'emb': info['emb'], 'goal_emb': info['goal_emb']}`.
     No pixels — the high-level adapter never touches them.
   - **Init from prior**: `init_action = model_high.macro_mean.expand(
     n_envs, H_high, d_l).clone()`. CEMSolver accepts an actions tensor
     in `init_action_distrib` (cem.py:91–112), so the high-level CEM
     starts with `μ = μ_l` instead of zero. The `var_scale` is multiplied
     by `model_high.macro_std.mean()` (or set per-dimension via a small
     adapter — Phase-5 detail) so the initial sampling spread is
     plausible in macro-action space.
   - **Cost**: `HighLevelCostAdapter.get_cost(info_high, l_cands)` — L1
     between rolled-out final latent and `z_g` plus the prior penalty
     (§1.4.2).
   - `high_out = self.solver_high.solve(info_high, init_action=init_high)`
     returns `{'actions': l*_{1:H}}` of shape `(n_envs, H, d_l)`.
   - **Materialise subgoals**: roll `l*_{1:H}` through
     `model_high.rollout({'emb': z_1}, l*[:, None, :, :])` (add a
     singleton sample axis to match `JEPA.rollout`'s `(B, S, T, ...)`
     contract) → `predicted_emb` of shape
     `(n_envs, 1, H+something, D)`. Cache the sequence; index 1 is
     `z̃_1`, index `i` is `z̃_i` (analysis §1.2 gotcha #2: index 0 is
     `z_1` itself, the start state).
     The rollout loop reuses `JEPA.rollout`'s `[:, -HS:]` slicing
     verbatim (`jepa.py:91`); at inference step `t` (0-indexed) it sees
     a context of length `L = min(t + 1, HS)`, exactly the distribution
     the predictor was trained on per §5.2 step 4.
   - `self._cached_subgoal = z̃_1`. `self._steps_since_high = 0`.
4. **Low-level CEM** (every step):
   - `info_low = {**info, 'goal_emb': self._cached_subgoal}` — overrides
     the cached `z_g` with the subgoal latent. The low-level adapter
     reads `goal_emb` directly (§1.4.1) and never re-encodes goal pixels.
   - `low_out = self.solver_low.solve(info_low, init_action=init_action)`.
     Sampling per HWM Tab. 10: 900 samples, 20 iters, h=5 LeWM blocks =
     25 env steps. Receding horizon `k = 5` blocks per
     `plan_config.receding_horizon` in `pusht_hwm.yaml`.
   - Return `low_out` directly to the caller. `WorldModelPolicy` reads
     `low_out['actions']` and ignores the rest.
5. **Step counter**. `self._steps_since_high += 1`. (Set in step 2's
   else branch when we *don't* replan high.)

### 6.3 Action-space arithmetic (re-emphasised)

LeWM groups 5 primitive env actions into one block. So:

| Quantity | Value (Push-T) | Notes |
|---|---|---|
| `frameskip` | 5 | `config/train/data/pusht.yaml:3` |
| Low-level action dim per block | 10 | 5 × 2-D primitives |
| `d_l` | 10 | starting point; sweep {3, 4, 6, 8, 10, 16} |
| Low-level pred horizon `h` | 5 LeWM blocks = 25 env steps | matches `plan_config.horizon` in `pusht.yaml` |
| Low-level replan cadence `k` | 5 LeWM blocks = 25 env steps | matches `receding_horizon` |
| High-level pred horizon `H` | 4 macro-steps | HWM Tab. 10 |
| Macro stride per high-level step | 5–14 LeWM blocks = 25–70 env steps | variable; matches paper |
| Goal offset eval distance `d` | {25, 50, 75} env steps | sweep |

`HWM_PLDM`'s `step_skip=10`, `replan_every=4`, `action_repeat=4` are
PLDM-specific and **must not be copied verbatim** (analysis §2.2).

### 6.4 Numerical detail: encoding cadence

The encoding budget per `solve()` call:

- **Current obs `o_1`**: encoded once per `solve()` call inside
  `_encode_and_cache`. `WorldModelPolicy` calls `solve` once per
  receding-horizon execution (every 5 LeWM blocks for Push-T), so this
  is amortised across 25 env steps.
- **Goal pixels `o_g`**: encoded *once* per unique goal, memoised in
  `_encode_and_cache` by `info['goal'].data_ptr()`. The episode goal
  doesn't change mid-episode, so this is one encode per episode.
- **CEM iterations**: zero encodes. Both adapters read pre-cached
  `info['emb']` and `info['goal_emb']` (or the cached subgoal latent).
  This is the load-bearing optimisation — without it, 1500 high-level
  samples × 40 iters = 60K encodes per replan.

The swm-side `LeWM` class (`stable-worldmodel/wm/lewm/lewm.py:73–76`)
follows the same pattern; we replicate it locally rather than depend on
the swm version (per the "ONLY modify files under repos/le-wm/" rule).

---

## 7. Evaluation plan

Three phases, each gated by a sanity criterion before the next runs.

### 7.1 Phase A — flat regression test (sanity gate, < 1 hour)

Goal: prove hierarchy patches preserve flat behaviour exactly.

- `python eval.py --config-name pusht policy=<existing_lewm_ckpt>`
  must reproduce the pre-hierarchy success rate within ±2 pts (LeWM
  paper reports 90 ± 1.4 on Push-T `d=25`).
- `python eval.py --config-name cube policy=<existing_lewm_ckpt>`
  within ±3 pts of 74.

If this regresses, the hierarchy patches changed something in
`eval.py`'s common path. Halt and fix before proceeding.

### 7.2 Phase B — high-level model trains & predicts (no planning yet)

Goal: confirm `P^(2)` training is healthy and the latent space is
*shared* with `P^(1)` as intended.

1. **Training loss curves**. Teacher-forced L1 should decrease and
   plateau within ~200 epochs (compare to paper's ~500-epoch budget).
2. **Macro-action norm** `||l||₂` should stabilise around an O(1) scale.
   Collapse → trigger Phase 5 escape hatch (small L2 reg on `l`,
   analysis §1.2 / plan §4).
3. **Latent-prediction error vs horizon** (replicate HWM Fig. 6).
   Held-out validation episodes:
   - For each rollout horizon `T` ∈ {0.5, 1.0, 1.5, 2.0} s, compare
     L1 error of (a) one-step `P^(2)` prediction at the closest
     waypoint and (b) `T/(frameskip * dt_env)`-step autoregressive
     `P^(1)` rollout against ground-truth latent.
   - **Pass criterion**: `P^(2)` ≤ `P^(1)` for `T ≥ 1.5` s. If not,
     hierarchy will not help downstream — go back and tune
     `d_l`/architecture before planning eval.

### 7.3 Phase C — hierarchical control (the headline)

Push-T long-horizon sweep, replicating HWM Tab. 2 with LeWM as the
backbone:

| Goal offset `d` (env steps) | Flat (LeWM) | Hierarchical (HWM-LeWM) | Pass criterion |
|---|---|---|---|
| 25 | existing 90 ± 1.4 | ≥ 88 | ≥ flat (no regression) |
| 50 | existing # tbd | ≥ flat + 5 pts | clear gain |
| 75 | existing # tbd | ≥ flat + 10 pts | strong gain |

50 trials per cell, fixed seed list. Use LeWM's existing
`pusht_expert_train` dataset (`config/eval/pusht.yaml:34`).

OGBench-Cube — single point check at `d = 25`. Pass criterion:
hierarchical ≥ flat. (Cube is the visually harder env — we want
"no regression and ideally a small gain"; full long-horizon sweep is
out of scope unless time permits.)

### 7.4 Compute trade-off (HWM Fig. 5)

Sweep `num_samples` ∈ {300, 600, 900, 1500, 3000} for the flat planner
and `(num_samples_high, num_samples_low) ∈ {(300, 300), (900, 900),
(1500, 900)}` for hierarchy. Plot success vs wall-clock per plan.
**Goal**: hierarchical reaches the same success as flat with ~3× less
compute (paper claim).

### 7.5 Ablations (time permitting)

Strictly ranked by ROI:

1. `d_l` sweep on Push-T `d=50`: {3, 4, 6, 8, 10, 16}. Highest-value
   ablation (HWM Fig. 7).
2. `N` (waypoints per training trajectory) sweep ∈ {3, 5, 7}.
3. Variable-stride vs fixed-stride waypoint sampling.
4. `advance_subgoal: true` vs default off.
5. Low-level cost L1 vs MSE (the design says L1; ablation justifies it).

### 7.6 Engineering tests (Phase 5 deliverable)

`tests/test_hierarchical.py`, ~9 tests:

1. **`MacroActionEncoder` shape + padding-mask invariance**: padded chunks
   produce the same `l` regardless of padded values. Also asserts shape
   `(B, d_l)`.
2. **Shared encoder**: `HighLevelWorldModel.encode({'pixels': x})['emb']`
   equals `JEPA.encode({'pixels': x})['emb']` on the same pixel batch
   (proves the shared latent space — *load-bearing invariant of the
   whole approach*).
3. **`MacroEmbedder` shape boundary**: `HighLevelWorldModel.predict` with
   a `(B, T, d_l)` raw macro-action would AdaLN with the wrong shape;
   verify the `MacroEmbedder` lifts to `(B, T, 192)` *before*
   `ARPredictor` is reached.
4. **`_match_goal_shape` contract**: feed `goal` as each of `(B, D)`,
   `(B, S, D)`, `(B, S, T, D)` against a fixed `pred (B, S, D)` and
   assert the helper returns `(B, S, D)` with the right values
   (broadcast for `(B, D)`, identity for `(B, S, D)`, last-time-step
   slice for `(B, S, T, D)`). Then call each adapter's `get_cost` end
   to end with synthetic info that mimics CEM expansion (`goal_emb`
   pre-expanded to `(B, S, D)`) and assert cost shape `(B, S)`.
5. **`HierarchicalCEMSolver.configure` synthetic action space**: after
   configure with `n_envs=4, d_l=10`, assert `solver_high._action_dim ==
   10` and `solver_low._action_dim == env_action_dim`. Specifically
   tests the `(n_envs, d_l)` shape requirement (not `(d_l,)`).
6. **`_encode_and_cache` does not mutate input**: pass an `info_dict`,
   call `solve`, assert the original dict still lacks `emb`/`goal_emb`.
   Catches regressions on the in-place-mutation issue (§8 risk #3).
7. **Prefix-length training coverage**: instrument `hwm_forward` with a
   counter on the sampled `L`. Run 1000 dummy forward steps and assert
   each `L ∈ {1, …, HS}` was sampled at frequency `≥ 1/HS − ε`. Catches
   silent regressions to fixed-length-3-only training (which would
   reintroduce the inference-time train/eval mismatch).
8. **Identity high-level rollout**: replace `P^(2)` with an identity
   that copies `z_g` into the subgoal cache; assert the hierarchical
   planner's actions match the flat planner with L1 cost on Push-T.
   Catches subtle adapter wiring bugs.
9. **End-to-end smoke**: 1 high-level training step on a 2-episode
   tiny dataset + 1 hierarchical eval episode, CPU-friendly tiny
   config. CI-runnable.

---

## 8. Risks and likely bugs

In rough order of severity. Each item names what to watch and the
specific mitigation.

1. **Stale `world.evaluate_from_dataset` call** (`eval.py:142`) ⟶
   *pre-existing bug*. The repo `stable-worldmodel/world/world.py:163–440`
   exposes `World.evaluate(...)` with an internal
   `_evaluate_from_dataset`, not a public method by that name (analysis
   §1.6). Either le-wm runs against a different swm version locally, or
   the call site is outdated. **Mitigation**: in Phase 5, run
   `python eval.py --config-name pusht` once on a clean checkout
   *before* writing any hierarchical code. If it fails, fix the call
   convention (rename to `world.evaluate(...)` and pass the dataset
   args via the API the swm version actually exposes), and apply the
   fix to the hierarchical eval too. Do not silently diverge between
   flat and hierarchical eval entry points.

2. **`embed_dim == hidden_dim == 192` collapses `cond_proj` to identity**.
   `train.py:90–91`: `hidden_dim = 192` (ViT-Tiny), `embed_dim = 192`.
   `module.py:156–160`: `cond_proj = nn.Linear(input_dim, hidden_dim)
   if input_dim != hidden_dim else nn.Identity()`. The existing
   low-level path works because `Embedder` outputs 192-D
   (`train.py:102`). The high-level path **must** insert a
   `MacroEmbedder = Linear(d_l, 192)` before the predictor, because
   macro-actions are `d_l`-D. **If this projection is missed,
   `ARPredictor` will silently consume a `d_l`-shaped tensor through
   a no-op `cond_proj`, and AdaLN will see the wrong shape**. Phase 5
   must include a shape assertion at the boundary.

3. **`info` is mutated in-place by `JEPA.encode` and `JEPA.rollout`**
   (`jepa.py:29–45`, `jepa.py:78–110`). The hierarchical planner runs
   *multiple* `encode`/`rollout` calls per `solve()` (high-level
   rollout for subgoal materialisation, then low-level rollouts inside
   CEM). Shared dicts would overwrite `info['emb']`/`'predicted_emb'`
   between calls. **Mitigation**: `_encode_and_cache` constructs a
   shallow-copied dict and clones tensors before encoding; cost
   adapters do `info = {k: v for k, v in info.items()}` before calling
   `rollout`. Test 6 in §7.6 is the regression guard.

4. **Synthetic action space shape MUST be `(n_envs, d_l)`, not `(d_l,)`**.
   `CEMSolver.configure` derives `_action_dim = int(np.prod(
   action_space.shape[1:]))` (`stable-worldmodel/stable_worldmodel/solver/cem.py:60`).
   With `shape=(d_l,)`, `shape[1:]` is `()` and `np.prod(()) == 1.0`
   — CEM would silently sample 1-D actions. **Mitigation**: build
   `gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_envs, self.d_l))`
   inside `configure`, where `n_envs` is the value just received.
   Test 5 in §7.6 catches regressions.

5. **`d_l` is critical and environment-dependent** (HWM Fig. 7;
   analysis §2.2). Push-T = 10, Cube = 4 are starting points, not
   final values. **Mitigation**: ship the `d_l` sweep as the first
   ablation (§7.5). Picking `d_l` too small → high-level can't
   represent useful trajectories; too large → high-level proposes
   subgoals the low-level can't reach.

6. **High-level predictor under-capacity**. HWM scales up the high-level
   predictor for DINO-WM Push-T (25M → 75M, layers 6→10, dim 384→768).
   We start with **same-size** as the low-level (6 layers, 192 dim)
   to keep diff small and training fast. **Mitigation**: Phase B's
   horizon-vs-error diagnostic (§7.2 step 3) is the gate. If `P^(2)`
   does not beat `P^(1)`-rollout at `T ≥ 1.5` s, scale up via
   `cfg.predictor.depth: 10, hidden_dim: 384, mlp_dim: 3072` (one
   yaml change; no code change because `ARPredictor` already
   parameterises this).

7. **Macro-action collapse** (analysis §1.2 gotcha #8, plan §4).
   With no explicit regulariser, `‖l‖` could collapse during training
   — `A_ψ` outputs near-zero, and `P^(2)` learns to rely on the
   teacher-forced `z` alone. **Mitigation**: the macro-prior buffers
   (`macro_mean`, `macro_std`) are themselves the canary — if
   `macro_std.mean()` drops below ~0.1 within a few epochs, collapse is
   happening. Add a small `+ 1e-3 * mean(‖l‖²)` regulariser (escape
   hatch, default off). Note: the `prior_weight` cost penalty in
   `HighLevelCostAdapter` does NOT prevent training-time collapse — it
   only shapes the CEM sampling distribution at plan time.

8. **Variable-length action chunks vs `Block`'s lack of padding mask**
   (§1.1). Reusing the existing `module.Block` would let padded values
   pollute the CLS token via attention. **Mitigation**: inline a
   masked attention layer inside `MacroActionEncoder` (§1.1 option b).
   Add a unit test that perturbing padded values does not change `l`
   (§7.6 test 1).

9. **Action chunk dim assumption**. `WaypointSubtrajectoryDataset`
   assumes the HDF5 `action` column is already grouped at `frameskip`
   (i.e. one row per LeWM block, dim `effective_act_dim = 10` for
   Push-T). Per `train.py:92` this is what the trainer assumes too. If
   the HDF5 stores raw env-step actions (dim 2 for Push-T), the
   sampler will silently emit chunks at the wrong granularity.
   **Mitigation**: add an assertion at dataset construction time:
   `dataset.get_col_data('action').shape[-1] == effective_act_dim`.

10. **Goal-pixel re-encoding cost in the high-level CEM loop**.
    A naive `HighLevelCostAdapter.get_cost` would re-encode goal pixels
    every CEM iteration. With 1500 samples × 40 iters = 60K rollouts per
    replan, even a cheap encode would dominate. **Mitigation**: encode
    `z_g` once in `HierarchicalCEMSolver._encode_and_cache` and pass it
    in `info_high['goal_emb']`; the adapter reads it directly and never
    encodes (§1.4.2). Same pattern as
    `stable-worldmodel/wm/lewm/lewm.py:73–76`.

11. **Subgoal feasibility**. The high-level predictor may propose
    subgoals that don't correspond to any reachable physical state
    (off-manifold latents). Empirically HWM observes this and points to
    the `d_l` choice as the regulariser (Fig. 7). **Mitigation**:
    early diagnostic in Phase B step 2 (decode `z̃_1` with the LeWM
    decoder if available, eyeball it on Push-T). Not a hard pass/fail
    gate; informs the `d_l` sweep.

12. **`receding_horizon` mismatch between levels**. The low-level
    config wants `receding_horizon = 5` (LeWM blocks); the high-level
    config wants `receding_horizon = 1` (one macro-step). Mixing them
    up will either (a) replan the low level too rarely or (b) replan
    the high level inside `WorldModelPolicy._prepare_info` after every
    primitive action. **Mitigation**: separate `plan_config` and
    `high_plan_config` blocks in `pusht_hwm.yaml` (§3.2); never share
    one between the two solvers.

13. **`spt.Module` multi-optimizer surprise**. `train.py:126–133`
    declares one optimizer group; `train_highlevel.py` only optimises
    `macro_encoder + macro_embedder + P^(2)`. The frozen modules must
    not appear in the optimizer's `params`. **Mitigation**: filter
    `model.named_parameters()` by `requires_grad=True` when building
    the optimizer (one line). Verify zero gradient flows through the
    encoder via a one-step-then-check-grad test.

14. **Determinism / numerical drift across CEM iters**. CEM resamples
    via `torch.normal`; the seed is set once at solver construction
    (`config/eval/solver/cem.yaml:9`). For two solvers in the same
    process, ensure they use independent `torch.Generator`s seeded
    deterministically (the existing `CEMSolver` already supports a
    `seed` arg). `HierarchicalCEMSolver.__init__` passes `seed` to
    `solver_low` and `seed + 1` to `solver_high`.

15. **`ARPredictor.pos_embedding` is fixed-length** (`module.py:262`,
    `nn.Parameter(torch.randn(1, num_frames, input_dim))`). Setting
    `num_frames=3` per §1.2 and feeding context windows of length 3
    via §5.2's sliding-window training is mandatory. A misconfigured
    `predictor.num_frames` (e.g., set to `N − 1 = 4`) would not fit
    LeWM's history convention and would silently break the inference
    rollout that reuses the `JEPA.rollout` machinery.

16. **Macro-prior staleness if low-level checkpoint changes**. The
    macro-prior buffers are computed during high-level training under
    a *specific* frozen low-level checkpoint. If we later swap the
    low-level checkpoint at eval time, the buffers are mis-calibrated.
    **Mitigation**: assert at planner construction that the
    `cfg.low_level_ckpt` recorded inside `model_high` matches the
    `cfg.policy` checkpoint. Phase-5 detail; not blocking for the
    initial single-checkpoint experiments.

---

## 9. Bottom line

- **3 new code files** under `repos/le-wm/`: `hwm.py`, `data.py`,
  `planner.py`. (`policy.py` removed — `HierarchicalCEMSolver` matches
  the `CEMSolver` contract so the existing `swm.policy.WorldModelPolicy`
  drives it directly.)
- **2 modified files**: `module.py` (add `MacroActionEncoder`) and
  `eval.py` (~10-line addition: load `model_high` via a second
  `swm.policy.AutoCostModel` and inject it into the solver alongside
  `model_low` when `cfg.solver._target_` is the hierarchical solver).
- **0 files modified outside `repos/le-wm/`**.
- **1 new training entry point**: `train_highlevel.py`.
- **6 new YAML configs**: `train/hwm.yaml`,
  `train/data/{pusht,ogb}_waypoints.yaml`, `eval/solver/hcem.yaml`,
  `eval/{pusht,cube}_hwm.yaml`. All Hydra `_target_`s use **flat module
  paths** (`planner.HierarchicalCEMSolver`, `hwm.HighLevelWorldModel`,
  `data.WaypointSubtrajectoryDataset`) since `eval.py` and
  `train_highlevel.py` run from `repos/le-wm/` and there is no
  `le_wm` package.
- **1 test file**: `tests/test_hierarchical.py`.

Three design decisions taken deliberately and documented above:

1. **Same-size high-level predictor** as the low-level one (6 layers,
   192 dim) with `num_frames=3` to match LeWM's `history_size`. Sliding
   windows at training time mirror LeWM's training convention exactly;
   inference reuses `JEPA.rollout`'s loop verbatim. Diagnostic-driven
   escape hatch to scale up only if Phase B shows the long-horizon
   advantage doesn't materialise.
2. **L1 cost via cost adapters in `planner.py`**, not via a `distance`
   arg on `JEPA.criterion`. Zero edits to `jepa.py`.
3. **Fresh trainable `pred_proj`** for the high-level world model
   (default), with sharing available as a `share_pred_proj: true`
   ablation. Shared-latent invariant is enforced by the L1 supervision
   against frozen `(encoder + projector)` outputs, not by sharing the
   predictor's output projector.

Three pieces of new machinery beyond the basic HWM recipe:

1. **Macro-prior buffers** (`macro_mean`, `macro_std`) computed as EMAs
   over `A_ψ` outputs at training time, persisted on the checkpoint,
   and used at plan time to (a) initialise the high-level CEM mean and
   (b) add a soft `((l − μ)/σ)²` prior penalty on macro-action samples.
   Replaces vanilla `N(0, I)` initialisation, which is wasteful in
   unbounded `R^{d_l}`.
2. **Explicit pre-CEM encoding cache** (`_encode_and_cache`) that
   encodes current pixels once per `solve()` and goal pixels once per
   unique goal. Both adapters then read pre-cached latents — zero
   encodes inside the CEM loops. The optimisation is required (60K
   high-level rollouts per replan would otherwise dominate runtime).
3. **Drop-in solver contract** matching
   `configure(*, action_space, n_envs, config)` and
   `solve(info_dict, init_action=None) → dict` (verified at
   `stable-worldmodel/stable_worldmodel/solver/cem.py:53–117`). Lets us
   reuse `WorldModelPolicy` unchanged and saves a policy class.

This blueprint is grounded in the actual code surface (every claim has
a citation into `repos/le-wm/`, `repos/stable-worldmodel/`, or the HWM
paper). Phase 5 can lift it directly into implementation.
