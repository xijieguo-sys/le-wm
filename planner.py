"""Hierarchical CEM planner and cost adapters.

HWM paper Sec. 3.3 -- top-down hierarchical MPC. Outer high-level CEM
proposes macro-actions toward the final goal; inner low-level CEM optimises
primitive actions toward the first predicted subgoal.

Drop-in replacement for `swm.solver.CEMSolver`. Matches the contract verified
at `stable-worldmodel/stable_worldmodel/solver/cem.py:53-117`:
    configure(*, action_space, n_envs, config) -> None
    __call__(...) -> solve(info_dict, init_action=None) -> dict
so the existing `swm.policy.WorldModelPolicy` drives it unchanged.
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

import stable_worldmodel as swm
from stable_worldmodel.solver import CEMSolver


class VarEMACEMSolver(CEMSolver):
    """CEMSolver with EMA smoothing on the per-iter sampling spread.

    HWM paper Tab. 10's "Var EMA" column. The vanilla CEMSolver refits
    `batch_var` to `topk_candidates.std(dim=1)` each iteration; this
    subclass blends the new std with the previous one:

        batch_var = var_ema * batch_var + (1 - var_ema) * elite_std

    With var_ema = 0 this reduces to identical behavior. (`batch_var` is
    used as `randn * batch_var + batch_mean`, so it is in fact a *std*
    despite the upstream variable name.)
    """

    def __init__(self, *args, var_ema: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.var_ema = float(var_ema)

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        # Near-copy of swm CEMSolver.solve (installed package). Only the
        # elite-refit step is modified to apply var_ema smoothing on the
        # sampling spread.
        start_time = time.time()
        outputs = {'costs': [], 'mean': [], 'var': []}

        mean, var = self.init_action_distrib(init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        total_envs = self.n_envs

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx

            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            expanded_infos = {}
            for k, v in info_dict.items():
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    v_batch = v_batch.unsqueeze(1)
                    v_batch = v_batch.expand(
                        current_bs, self.num_samples, *v_batch.shape[2:]
                    )
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(
                        v_batch[:, None, ...], self.num_samples, axis=1
                    )
                expanded_infos[k] = v_batch

            final_batch_cost = None
            for step in range(self.n_steps):
                candidates = torch.randn(
                    current_bs, self.num_samples,
                    self.horizon, self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                )
                candidates = (
                    candidates * batch_var.unsqueeze(1)
                    + batch_mean.unsqueeze(1)
                )
                candidates[:, 0] = batch_mean
                current_info = expanded_infos.copy()
                costs = self.model.get_cost(current_info, candidates)

                assert isinstance(costs, torch.Tensor), (
                    f'Expected cost to be a torch.Tensor, got {type(costs)}'
                )
                assert (
                    costs.ndim == 2
                    and costs.shape[0] == current_bs
                    and costs.shape[1] == self.num_samples
                ), (
                    f'Expected cost shape ({current_bs}, {self.num_samples}), '
                    f'got {costs.shape}'
                )

                topk_vals, topk_inds = torch.topk(
                    costs, k=self.topk, dim=1, largest=False
                )
                batch_indices = (
                    torch.arange(current_bs, device=self.device)
                    .unsqueeze(1)
                    .expand(-1, self.topk)
                )
                topk_candidates = candidates[batch_indices, topk_inds]

                # Refit. Mean updates fully to the elite mean; var (= std)
                # is EMA-smoothed against the elite std per Tab. 10.
                elite_std = topk_candidates.std(dim=1)
                batch_mean = topk_candidates.mean(dim=1)
                if self.var_ema > 0.0:
                    batch_var = (
                        self.var_ema * batch_var
                        + (1.0 - self.var_ema) * elite_std
                    )
                else:
                    batch_var = elite_std

                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs['costs'].extend(final_batch_cost)

        outputs['actions'] = mean.detach().cpu()
        outputs['mean'] = [mean.detach().cpu()]
        outputs['var'] = [var.detach().cpu()]
        print(f'CEM solve time: {time.time() - start_time:.4f} seconds')
        return outputs


def _match_goal_shape(goal: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Align `goal` to `pred`'s shape.

    `pred` is (B, S, D). `goal` may arrive as (B, D) (pre-CEM injection),
    (B, S, D) (post-CEM expansion -- the common case), or (B, S, T, D)
    (full predicted-emb shape). Returns a tensor expanded/sliced to (B, S, D).
    """
    if goal.dim() == pred.dim() - 1:
        # (B, D) -> (B, 1, D)
        goal = goal.unsqueeze(1)
    elif goal.dim() == pred.dim() + 1:
        # (B, S, T, D) -> (B, S, D)
        goal = goal[..., -1, :]
    elif goal.dim() != pred.dim():
        raise ValueError(
            f'unexpected goal_emb shape {tuple(goal.shape)} vs pred {tuple(pred.shape)}'
        )
    return goal.expand_as(pred)


def _rollout_from_emb(
    predict_fn,
    action_encoder,
    z_init: torch.Tensor,
    action_candidates: torch.Tensor,
    history_size: int,
) -> torch.Tensor:
    """Sliding-window rollout starting from a pre-cached latent.

    Mirrors `JEPA.rollout` (`jepa.py:61-110`) but skips the initial pixel
    encode -- we already have z_1 cached. Equivalent number of predict()
    calls; only the structure differs.

    z_init: (B, S, D) or (B, D) -- single initial latent
    action_candidates: (B, S, T, A_block)
    Returns predicted_emb of shape (B, S, T+1, D); last index is z_{T+1}.
    """
    B, S, T, _ = action_candidates.shape
    if z_init.dim() == 2:
        z_init = z_init.unsqueeze(1).expand(B, S, z_init.size(-1))
    z_init = z_init.contiguous()

    emb = z_init.unsqueeze(2).reshape(B * S, 1, -1)  # (BS, 1, D)
    action_flat = action_candidates.reshape(B * S, T, action_candidates.size(-1))
    act_emb_full = action_encoder(action_flat)  # (BS, T, D_act)

    HS = history_size
    for t in range(T):
        L = min(emb.size(1), HS)
        ctx_emb = emb[:, -L:]
        a_start = max(0, t + 1 - L)
        ctx_act = act_emb_full[:, a_start : t + 1]
        pred = predict_fn(ctx_emb, ctx_act)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred], dim=1)

    return emb.reshape(B, S, 1 + T, -1)


# ---------------------------------------------------------------------------
# Cost adapters -- thin nn.Modules exposing the CEMSolver `get_cost` contract.
# ---------------------------------------------------------------------------


class SubgoalCostAdapter(nn.Module):
    """Wraps the frozen low-level JEPA for subgoal-following CEM.

    Differs from `JEPA.get_cost` in two ways (HWM paper Eq. 2):
    - reads pre-cached info['goal_emb'] (set by HierarchicalCEMSolver to the
      latent subgoal z̃_i) instead of re-encoding goal pixels every CEM iter.
    - uses L1 cost, not MSE.
    """

    def __init__(self, model_low: nn.Module, history_size: int = 3):
        super().__init__()
        self.wrapped = model_low
        self.history_size = int(history_size)

    def get_cost(self, info: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        assert 'emb' in info and 'goal_emb' in info, (
            'SubgoalCostAdapter requires pre-cached emb/goal_emb -- the '
            'hierarchical solver populates these via _encode_and_cache.'
        )
        z_init = info['emb']
        # CEM expansion produces (B, S, D) for both emb and goal_emb. _rollout_from_emb
        # accepts (B, D) or (B, S, D); reduce a 4-D shape (B, S, T, D) to last frame.
        if z_init.dim() == 4:
            z_init = z_init[..., -1, :]
        rollout = _rollout_from_emb(
            self.wrapped.predict,
            self.wrapped.action_encoder,
            z_init,
            action_candidates,
            self.history_size,
        )
        pred = rollout[..., -1, :]  # (B, S, D)
        goal = _match_goal_shape(info['goal_emb'], pred)
        return (pred - goal.detach()).abs().sum(-1)  # (B, S)


class HighLevelCostAdapter(nn.Module):
    """Wraps HighLevelWorldModel for the high-level CEM.

    Cost is L1 against the final goal latent z_g plus a soft prior penalty
    on macro-action samples ((l - μ_l) / σ_l)^2 averaged over (H, d_l).
    The prior pulls samples toward the empirical distribution of A_psi
    outputs observed at training time (see HighLevelWorldModel buffers).
    """

    def __init__(self, model_high: nn.Module, prior_weight: float = 0.0):
        # Default 0.0 = paper-faithful (no macro-action cost penalty). Set
        # >0 to opt into the prior penalty as an ablation.
        super().__init__()
        self.wrapped = model_high
        self.prior_weight = float(prior_weight)

    def get_cost(self, info: dict, l_candidates: torch.Tensor) -> torch.Tensor:
        assert 'emb' in info and 'goal_emb' in info, (
            'HighLevelCostAdapter requires pre-cached emb/goal_emb.'
        )
        # HighLevelWorldModel.rollout already handles shape and CEM expansion.
        info_local = {k: v for k, v in info.items()}
        info_local = self.wrapped.rollout(info_local, l_candidates)
        pred = info_local['predicted_emb'][..., -1, :]  # (B, S, D)
        goal = _match_goal_shape(info_local['goal_emb'], pred)
        cost_l1 = (pred - goal.detach()).abs().sum(-1)  # (B, S)

        if self.prior_weight > 0.0:
            mu = self.wrapped.macro_mean.view(1, 1, 1, -1)
            std = self.wrapped.macro_std.view(1, 1, 1, -1).clamp_min(1e-3)
            # (B, S, H, d_l) -> (B, S) via mean over (H, d_l)
            prior = ((l_candidates - mu) / std).square().mean(dim=(-1, -2))
            return cost_l1 + self.prior_weight * prior
        return cost_l1


# ---------------------------------------------------------------------------
# Hierarchical solver
# ---------------------------------------------------------------------------


class HierarchicalCEMSolver:
    """Two-level CEM planner. Drop-in replacement for swm.solver.CEMSolver.

    Outer (high-level) CEM proposes macro-actions toward the final goal;
    inner (low-level) CEM optimises primitive actions toward the first
    predicted subgoal latent. See architecture proposal §6.2.
    """

    def __init__(
        self,
        model_low,
        model_high,
        high_cfg: dict,
        low_cfg: dict,
        high_plan_cfg: dict,
        d_l: int | None = None,
        replan_high_every: int = 1,
        advance_subgoal: bool = False,
        subgoal_threshold: float | None = None,
        prior_weight: float = 0.0,          # paper-faithful default; cost penalty off
        use_macro_prior_init: bool = False, # paper-faithful default; CEM init at zeros
        history_size: int = 3,
        device: str | torch.device = 'cuda',
        seed: int = 1234,
    ) -> None:
        self.device = device
        self.history_size = int(history_size)
        self.replan_high_every = int(replan_high_every)
        self.advance_subgoal = bool(advance_subgoal)
        self.subgoal_threshold = subgoal_threshold
        # When False (paper-faithful default): high-level CEM inits with
        # zeros mean and yaml's var_scale. When True: inits from
        # macro_mean / macro_std.mean() learned during training.
        self.use_macro_prior_init = bool(use_macro_prior_init)

        # d_l can be read off the loaded high-level model checkpoint.
        if d_l is None:
            d_l = int(getattr(model_high, 'd_l'))
        self.d_l = int(d_l)

        # Underlying CEM instances. We use VarEMACEMSolver (paper Tab. 10's
        # Var EMA column); when var_ema is 0 or absent it behaves identically
        # to vanilla CEMSolver, so old configs without var_ema still work.
        self.solver_low = VarEMACEMSolver(
            model=SubgoalCostAdapter(model_low, history_size=self.history_size),
            device=device,
            seed=seed,
            **low_cfg,
        )
        self.solver_high = VarEMACEMSolver(
            model=HighLevelCostAdapter(model_high, prior_weight=prior_weight),
            device=device,
            seed=seed + 1,
            **high_cfg,
        )

        # Direct refs for subgoal materialisation and goal caching.
        self.model_low = model_low
        self.model_high = model_high

        # PlanConfig kwargs for the high-level CEMSolver. action_block=1
        # because macro-actions aren't blocked.
        self.high_plan_cfg = dict(high_plan_cfg)

        # MPC bookkeeping.
        self._cached_subgoal = None       # (n_envs, D) -- z̃_i (current low-level target)
        self._cached_subgoal_seq = None   # (n_envs, H+1, D) -- z̃_{0:H} (z̃_0 = z_1)
        self._cached_subgoal_idx = 1      # which subgoal in the sequence we're chasing
        self._steps_since_high = -1       # forces high replan on first call
        self._n_envs = None

        # Goal-pixel cache (per-episode goal). Key: a small content
        # fingerprint (shape + sum of a strided view) -- using data_ptr()
        # would miss every call because WorldModelPolicy._prepare_info
        # reconstructs the goal tensor each solve(). Same caching idea
        # as `stable-worldmodel/wm/lewm/lewm.py:73-76`.
        self._goal_cache = None
        self._goal_cache_fp = None       # (shape_tuple, fingerprint_float)

        self._configured = False

    # ----- API surface required by WorldModelPolicy / CEMSolver contract --

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        # Low-level: passthrough.
        self.solver_low.configure(
            action_space=action_space, n_envs=n_envs, config=config
        )

        # High-level: synthetic action_space of shape (n_envs, d_l).
        # CEMSolver.configure (cem.py:60) does `np.prod(shape[1:])` -- with
        # shape=(d_l,) we'd silently get 1; the (n_envs, d_l) form is required.
        synth_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_envs, self.d_l),
            dtype=np.float32,
        )
        synth_plan = swm.PlanConfig(**self.high_plan_cfg)
        self.solver_high.configure(
            action_space=synth_space, n_envs=n_envs, config=synth_plan
        )

        self._n_envs = int(n_envs)
        self._configured = True

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        # WorldModelPolicy reads this for its action buffer arithmetic; defer
        # to the low-level solver, which is what controls primitive actions.
        return self.solver_low.action_dim

    @property
    def horizon(self) -> int:
        return self.solver_low.horizon

    @property
    def dtype(self) -> torch.dtype:
        return self.solver_low.dtype

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    # ----- core solve --------------------------------------------------

    def solve(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
    ) -> dict:
        start_time = time.time()
        # 1. Encode current obs and (cached) goal -> info_low contains 'emb'/'goal_emb'.
        info_low = self._encode_and_cache(info_dict)

        n_envs = int(info_low['emb'].shape[0])

        # 2. Decide whether to replan high.
        do_high = (
            self._cached_subgoal is None
            or self._steps_since_high < 0
            or self._steps_since_high % self.replan_high_every == 0
            or self._should_advance_subgoal(info_low)
        )

        if do_high:
            self._replan_high(info_low, n_envs)
            self._steps_since_high = 0
        else:
            self._steps_since_high += 1

        # 3. Inject cached subgoal latent as goal_emb for the low-level solver.
        #    A shallow copy keeps the original info_dict pristine.
        info_for_low = {k: v for k, v in info_low.items()}
        info_for_low['goal_emb'] = self._cached_subgoal

        # 4. Low-level solve -- returns {'actions': (n_envs, h, action_dim), ...}.
        low_out = self.solver_low.solve(info_for_low, init_action=init_action)
        low_out['hier_solve_time'] = time.time() - start_time
        return low_out

    # ----- helpers ------------------------------------------------------

    @torch.inference_mode()
    def _encode_and_cache(self, info_dict: dict) -> dict:
        """Encode current pixels and (memoised) goal pixels into latents.

        WorldModelPolicy._prepare_info (`stable-worldmodel/policy.py:121-183`)
        only runs preprocess/transform; it does NOT populate emb/goal_emb.
        That responsibility lives here so the cost adapters can skip
        redundant encodes inside the CEM loops (60K+ rollouts/replan).
        """
        info = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in info_dict.items()}

        device = self.device
        # Move tensors to model device.
        for k, v in info.items():
            if torch.is_tensor(v):
                info[k] = v.to(device)

        # --- Encode current pixels -> info['emb'] --------------------------
        # JEPA.encode expects (B, T, C, H, W). info['pixels'] arrives shape
        # depending on the env wrapper; reshape if needed.
        pix = info['pixels']
        added_t = False
        if pix.dim() == 4:                # (B, C, H, W) -- add time axis
            pix = pix.unsqueeze(1)
            added_t = True
        enc_in = {'pixels': pix}
        enc_out = self.model_low.encode(enc_in)
        emb = enc_out['emb']             # (B, T, D)
        if added_t:
            emb = emb.squeeze(1)         # (B, D)
        info['emb'] = emb

        # --- Encode goal pixels (memoised) -> info['goal_emb'] -------------
        # Cache by content fingerprint: shape + sum of a strided slice.
        # Episode goals are stable so the fingerprint is stable; a fresh
        # tensor with the same content yields a hit (data_ptr would miss).
        goal = info['goal']
        if torch.is_tensor(goal):
            # Strided slice picks ~1024 floats spread across the tensor;
            # sum() is one CUDA reduction in microseconds.
            stride = max(1, goal.numel() // 1024)
            fp = (tuple(goal.shape), float(goal.flatten()[::stride].sum().item()))
        else:
            fp = None
        if self._goal_cache is None or self._goal_cache_fp != fp:
            gpix = goal
            g_added = False
            if gpix.dim() == 4:
                gpix = gpix.unsqueeze(1)
                g_added = True
            g_enc = self.model_low.encode({'pixels': gpix})
            g_emb = g_enc['emb']
            if g_added:
                g_emb = g_emb.squeeze(1)
            self._goal_cache = g_emb
            self._goal_cache_fp = fp
        info['goal_emb'] = self._goal_cache

        return info

    @torch.inference_mode()
    def _replan_high(self, info_low: dict, n_envs: int) -> None:
        """Run the high-level CEM and materialise the subgoal sequence."""
        info_high = {
            'emb': info_low['emb'],
            'goal_emb': info_low['goal_emb'],
        }

        if self.use_macro_prior_init:
            # OPT-IN: init high-level CEM from the trained macro-action prior
            # buffers (μ_l mean, σ_l.mean() var). Paper does NOT do this; this
            # is an extension we expose for ablation. Setting use_macro_prior_init
            # to True uses both μ_l and σ_l for the FIRST CEM iteration; later
            # iterations refit from elites (with optional var_ema smoothing).
            H_high = self.solver_high.horizon
            mu = self.model_high.macro_mean.detach().to(self.device)
            init_high = mu.view(1, 1, -1).expand(n_envs, H_high, self.d_l).contiguous()
            sigma_scale = float(self.model_high.macro_std.mean().detach().cpu().item())
            sigma_scale = max(sigma_scale, 1e-3)
            prev_var_scale = self.solver_high.var_scale
            self.solver_high.var_scale = sigma_scale
            try:
                high_out = self.solver_high.solve(info_high, init_action=init_high)
            finally:
                self.solver_high.var_scale = prev_var_scale
        else:
            # Paper-faithful: zeros mean, yaml var_scale. CEMSolver fills its
            # mean tensor with zeros internally when init_action is None.
            high_out = self.solver_high.solve(info_high)

        # high_out['actions']: (n_envs, H_high, d_l).
        l_seq = high_out['actions'].to(self.device)

        # Roll the chosen macro-action sequence through P^(2) to materialise
        # the predicted subgoal latents z̃_{0:H_high}. Add a singleton sample
        # axis to match HighLevelWorldModel.rollout's (B, S, T, d_l) contract.
        info_for_rollout = {'emb': info_high['emb']}
        l_seq_unsq = l_seq.unsqueeze(1)  # (n_envs, 1, H_high, d_l)
        rollout = self.model_high.rollout(info_for_rollout, l_seq_unsq)
        # predicted_emb: (n_envs, 1, H_high+1, D)  [index 0 is z_1; index k is z̃_k]
        seq = rollout['predicted_emb'].squeeze(1)  # (n_envs, H_high+1, D)

        self._cached_subgoal_seq = seq
        self._cached_subgoal_idx = 1            # chase z̃_1 first
        self._cached_subgoal = seq[:, 1]        # (n_envs, D)

    def _should_advance_subgoal(self, info_low: dict) -> bool:
        """Subgoal-gating logic (default OFF; opt-in via advance_subgoal)."""
        if not self.advance_subgoal or self.subgoal_threshold is None:
            return False
        if self._cached_subgoal is None or self._cached_subgoal_seq is None:
            return False
        # If we're already at the end of the cached sequence, we MUST replan.
        if self._cached_subgoal_idx + 1 >= self._cached_subgoal_seq.size(1):
            return True
        # Otherwise, compare current latent to current subgoal.
        z_now = info_low['emb']
        if z_now.dim() == 3:  # (B, T, D) -- take last
            z_now = z_now[:, -1]
        dist = (z_now - self._cached_subgoal).abs().sum(-1)  # (n_envs,)
        if (dist < self.subgoal_threshold).all():
            self._cached_subgoal_idx += 1
            self._cached_subgoal = self._cached_subgoal_seq[:, self._cached_subgoal_idx]
            return False  # advanced; no full high-level replan needed
        return False
