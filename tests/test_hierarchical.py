"""Engineering tests for the hierarchical pipeline (architecture proposal §7.6).

Run from the repo root:

    python -m unittest tests.test_hierarchical -v

These tests use small models (2-layer predictor, 8x8 pixels) so they finish
in <30 s on CPU. The end-to-end smoke test is the slowest; the rest are
sub-second contract checks.
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Ensure the repo root is on sys.path so `from module import ...` resolves
# whether this file is invoked via `python -m unittest tests.test_hierarchical`
# or directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import stable_pretraining as spt
import stable_worldmodel as swm

from data import WaypointSubtrajectoryDataset
from hwm import HighLevelWorldModel
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, MacroActionEncoder
from planner import (
    HierarchicalCEMSolver,
    HighLevelCostAdapter,
    SubgoalCostAdapter,
    _match_goal_shape,
)


# ---------- helpers --------------------------------------------------------


def _build_tiny_jepa(embed_dim: int = 192, action_dim: int = 10):
    """Build a JEPA with a real ViT encoder but a 2-layer predictor."""
    encoder = spt.backbone.utils.vit_hf(
        'tiny', patch_size=14, image_size=224,
        pretrained=False, use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    projector = MLP(hidden_dim, 2048, embed_dim, norm_fn=nn.BatchNorm1d)
    pred_proj = MLP(hidden_dim, 2048, embed_dim, norm_fn=nn.BatchNorm1d)
    predictor = ARPredictor(
        num_frames=3, input_dim=embed_dim, hidden_dim=hidden_dim,
        output_dim=hidden_dim, depth=2, heads=4, mlp_dim=512, dim_head=64,
        dropout=0.0, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=action_dim, emb_dim=embed_dim)
    return JEPA(encoder=encoder, predictor=predictor, action_encoder=action_encoder,
                projector=projector, pred_proj=pred_proj)


def _build_tiny_hwm(jepa: JEPA, d_l: int = 10, action_block: int = 10):
    macro_encoder = MacroActionEncoder(input_dim=action_block, d_l=d_l, max_blocks=14)
    macro_embedder = nn.Linear(d_l, 192)
    predictor = ARPredictor(
        num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
        depth=2, heads=4, mlp_dim=512, dim_head=64,
        dropout=0.0, emb_dropout=0.0,
    )
    pred_proj = MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d)
    return HighLevelWorldModel(
        encoder=jepa.encoder, projector=jepa.projector,
        macro_encoder=macro_encoder, macro_embedder=macro_embedder,
        predictor=predictor, pred_proj=pred_proj,
        d_l=d_l, history_size=3,
    )


# ---------- tests ----------------------------------------------------------


class Test01PaddingMask(unittest.TestCase):
    """MacroActionEncoder shape + padding-mask invariance."""

    def test_padding_invariance(self):
        torch.manual_seed(0)
        m = MacroActionEncoder(input_dim=10, d_l=10, max_blocks=14).eval()
        B, L = 4, 7
        actions = torch.randn(B, L, 10)
        mask = torch.tensor([
            [True] * 5 + [False] * 2,
            [True] * 7,
            [True] * 1 + [False] * 6,
            [True] * 4 + [False] * 3,
        ])
        out1 = m(actions, mask)
        # Replace padded values with random garbage; output must be unchanged.
        actions2 = actions.clone()
        actions2[~mask] = torch.randn_like(actions2[~mask]) * 100
        out2 = m(actions2, mask)
        self.assertEqual(tuple(out1.shape), (B, 10))
        self.assertLess((out1 - out2).abs().max().item(), 1e-6)


class Test02SharedEncoder(unittest.TestCase):
    """HighLevelWorldModel and JEPA produce IDENTICAL latents on the same pixels."""

    def test_shared_encoder(self):
        torch.manual_seed(0)
        jepa = _build_tiny_jepa().eval()
        hwm = _build_tiny_hwm(jepa).eval()

        pixels = torch.rand(2, 3, 3, 224, 224)
        z_jepa = jepa.encode({'pixels': pixels.clone()})['emb']
        z_hwm = hwm.encode({'pixels': pixels.clone()})['emb']

        self.assertEqual(z_jepa.shape, z_hwm.shape)
        # Equal because the encoder + projector references are shared (same nn.Module).
        self.assertLess((z_jepa - z_hwm).abs().max().item(), 1e-6)


class Test03MacroEmbedderBoundary(unittest.TestCase):
    """`predict` must consume embed_dim-D conditioning, not d_l-D."""

    def test_macro_embedder_lifts(self):
        torch.manual_seed(0)
        jepa = _build_tiny_jepa()
        hwm = _build_tiny_hwm(jepa, d_l=10)
        hwm.eval()

        # Hand-roll a forward at the boundary: macro encoder -> macro_embedder ->
        # predict. The predict output's last dim must equal hwm's embed_dim (192).
        B = 2
        l = torch.randn(B, 4, 10)               # raw macro-actions (d_l=10)
        e = hwm.macro_embedder(l)               # lifted to 192
        self.assertEqual(e.shape[-1], 192)

        # Feed (z_ctx, e_ctx) of length 3 (= history_size) to predict.
        z = torch.randn(B, 3, 192)
        e3 = e[:, :3]
        out = hwm.predict(z, e3)
        self.assertEqual(out.shape, (B, 3, 192))


class Test04MatchGoalShape(unittest.TestCase):
    """_match_goal_shape contract on (B, D), (B, S, D), (B, S, T, D)."""

    def test_all_shapes(self):
        pred = torch.zeros(3, 5, 192)
        # (B, D)
        out = _match_goal_shape(torch.ones(3, 192), pred)
        self.assertEqual(tuple(out.shape), (3, 5, 192))
        self.assertTrue((out == 1).all())
        # (B, S, D)
        out = _match_goal_shape(torch.ones(3, 5, 192) * 2, pred)
        self.assertEqual(tuple(out.shape), (3, 5, 192))
        self.assertTrue((out == 2).all())
        # (B, S, T, D) -- last time step taken
        g = torch.zeros(3, 5, 4, 192)
        g[..., -1, :] = 7
        out = _match_goal_shape(g, pred)
        self.assertEqual(tuple(out.shape), (3, 5, 192))
        self.assertTrue((out == 7).all())

        # Bad shape raises
        with self.assertRaises(ValueError):
            _match_goal_shape(torch.zeros(2), pred)

    def test_adapter_get_cost_post_cem_expansion(self):
        """End-to-end: pass a (B, S, D) goal_emb (mimicking CEM expansion)
        through SubgoalCostAdapter and assert (B, S) cost output."""
        torch.manual_seed(0)
        jepa = _build_tiny_jepa().eval()
        adapter = SubgoalCostAdapter(jepa, history_size=3)

        B, S, T = 2, 4, 5
        info = {
            'emb': torch.randn(B, S, 192),
            'goal_emb': torch.randn(B, S, 192),
        }
        actions = torch.randn(B, S, T, 10)
        cost = adapter.get_cost(info, actions)
        self.assertEqual(tuple(cost.shape), (B, S))


class Test05ConfigureSyntheticActionSpace(unittest.TestCase):
    """HierarchicalCEMSolver.configure must give solver_high._action_dim == d_l."""

    def test_configure_action_dims(self):
        torch.manual_seed(0)
        jepa = _build_tiny_jepa().eval()
        hwm = _build_tiny_hwm(jepa).eval()
        solver = HierarchicalCEMSolver(
            model_low=jepa, model_high=hwm,
            high_cfg=dict(num_samples=4, n_steps=1, topk=2, var_scale=1.0, batch_size=1),
            low_cfg=dict(num_samples=4, n_steps=1, topk=2, var_scale=1.0, batch_size=1),
            high_plan_cfg=dict(horizon=2, receding_horizon=1, action_block=1),
            d_l=10, device='cpu', seed=42,
        )
        n_envs = 3
        env_space = gym.spaces.Box(low=-1, high=1, shape=(n_envs, 2), dtype=np.float32)
        plan = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
        solver.configure(action_space=env_space, n_envs=n_envs, config=plan)

        # action_block * action_dim per cem.py:76
        self.assertEqual(solver.solver_low.action_dim, 10)   # 2 * 5 = 10
        self.assertEqual(solver.solver_high.action_dim, 10)  # d_l * 1 = 10


class Test06EncodeAndCacheNoMutation(unittest.TestCase):
    """_encode_and_cache must NOT mutate the caller's info_dict."""

    def test_input_unchanged(self):
        torch.manual_seed(0)
        jepa = _build_tiny_jepa().eval()
        hwm = _build_tiny_hwm(jepa).eval()
        solver = HierarchicalCEMSolver(
            model_low=jepa, model_high=hwm,
            high_cfg=dict(num_samples=4, n_steps=1, topk=2, var_scale=1.0, batch_size=1),
            low_cfg=dict(num_samples=4, n_steps=1, topk=2, var_scale=1.0, batch_size=1),
            high_plan_cfg=dict(horizon=2, receding_horizon=1, action_block=1),
            d_l=10, device='cpu', seed=42,
        )
        info_in = {
            'pixels': torch.rand(2, 3, 224, 224),
            'goal':   torch.rand(2, 3, 224, 224),
        }
        keys_before = set(info_in.keys())
        _ = solver._encode_and_cache(info_in)
        keys_after = set(info_in.keys())
        self.assertEqual(keys_before, keys_after)
        self.assertNotIn('emb', info_in)
        self.assertNotIn('goal_emb', info_in)


class Test07PrefixLengthCoverage(unittest.TestCase):
    """hwm_forward must use every L in {1,...,HS} during training. Catches
    regressions that would hardcode L=HS-only contexts and reintroduce the
    inference-time train/eval mismatch (architecture proposal §5.2 step 4)."""

    def test_forward_hits_all_lengths(self):
        from train_highlevel import hwm_forward
        from omegaconf import OmegaConf
        from functools import partial

        torch.manual_seed(0)
        jepa = _build_tiny_jepa().eval()
        hwm = _build_tiny_hwm(jepa).eval()

        # Instrument predict() so we can see which context lengths actually
        # reach the predictor. We monkey-patch the predict method to record
        # the second axis of its first arg before delegating.
        seen_lengths = []
        original_predict = hwm.predict

        def instrumented_predict(emb, macro_emb):
            seen_lengths.append(emb.size(1))
            return original_predict(emb, macro_emb)

        hwm.predict = instrumented_predict

        cfg = OmegaConf.create({
            'wm': {'history_size': 3, 'd_l': 10},
            'macro_prior': {'ema_momentum': 0.99},
        })

        # Stand-in for the spt.Module wrapper; hwm_forward only reads
        # self.model and self.log_dict.
        class _StubSelf:
            def __init__(self, m):
                self.model = m
            def log_dict(self, *args, **kwargs):
                pass
        stub = _StubSelf(hwm)

        torch.manual_seed(7)
        # Run several batches with varying random state so all 3 lengths
        # surface. With B=24, expected ~8 items per length each batch.
        for _ in range(8):
            batch = {
                'pixels':         torch.rand(24, 5, 3, 224, 224),
                'actions_chunk':  torch.randn(24, 4, 14, 10),
                'actions_mask':   torch.ones(24, 4, 14, dtype=torch.bool),
            }
            hwm_forward(stub, batch, 'train', cfg)

        # The instrumented predict should have been called with each
        # context length 1, 2, and 3 across the 8 batches.
        unique = set(seen_lengths)
        self.assertEqual(unique, {1, 2, 3},
                         f'expected all of {{1,2,3}} to appear; got {sorted(unique)}')


class Test08IdentityRollout(unittest.TestCase):
    """Identity high-level: subgoal == final goal -> hierarchical with H=1
    behaves like flat (sanity that the adapter wiring is correct)."""

    def test_identity_subgoal_wiring(self):
        """Patch `model_high.rollout` to a deterministic identity (every
        predicted latent equals z_init). After a solve(), the cached subgoal
        z̃_1 must therefore equal z_init -- proves the rollout output flows
        into the subgoal cache correctly. Catches subtle wiring bugs in
        _replan_high (subgoal index, sample axis squeeze, etc.)."""
        torch.manual_seed(0)
        jepa = _build_tiny_jepa().eval()
        hwm = _build_tiny_hwm(jepa).eval()

        # Identity rollout: read 'emb' (z_init) and return it broadcast across
        # all output positions. _replan_high passes info_for_rollout =
        # {'emb': info_high['emb']} (no goal_emb), so we must not depend on it.
        def identity_rollout(info, l_cands):
            B, S, T = l_cands.shape[:3]
            z = info['emb']                          # (B, D)
            if z.dim() == 2:
                z = z.unsqueeze(1).expand(B, S, z.size(-1))
            pred = z.unsqueeze(2).expand(B, S, T + 1, z.size(-1)).contiguous()
            info['predicted_emb'] = pred
            return info

        hwm.rollout = identity_rollout

        solver = HierarchicalCEMSolver(
            model_low=jepa, model_high=hwm,
            high_cfg=dict(num_samples=4, n_steps=1, topk=2, var_scale=1.0, batch_size=1),
            low_cfg=dict(num_samples=4, n_steps=1, topk=2, var_scale=1.0, batch_size=1),
            high_plan_cfg=dict(horizon=1, receding_horizon=1, action_block=1),
            d_l=10, device='cpu', seed=42,
        )
        n_envs = 2
        env_space = gym.spaces.Box(low=-1, high=1, shape=(n_envs, 2), dtype=np.float32)
        plan = swm.PlanConfig(horizon=3, receding_horizon=3, action_block=5)
        solver.configure(action_space=env_space, n_envs=n_envs, config=plan)

        info = {
            'pixels': torch.rand(n_envs, 3, 224, 224),
            'goal':   torch.rand(n_envs, 3, 224, 224),
        }
        out = solver.solve(info)
        self.assertEqual(out['actions'].shape, (n_envs, 3, 10))
        # Identity rollout returns z_init at every position, so the cached
        # subgoal (index 1) must equal z_init = info_low['emb'].
        # info_low['emb'] is what _encode_and_cache produced from info['pixels'];
        # we can recover it by calling _encode_and_cache again.
        info_check = solver._encode_and_cache(info)
        z_init = info_check['emb']                    # (n_envs, D)
        cached = solver._cached_subgoal               # (n_envs, D)
        self.assertLess((cached - z_init).abs().max().item(), 1e-5)


class Test09EndToEndSmoke(unittest.TestCase):
    """End-to-end: hwm_forward over a stub dataset + a hierarchical solve."""

    def test_smoke(self):
        torch.manual_seed(0)
        jepa = _build_tiny_jepa().eval()
        hwm = _build_tiny_hwm(jepa).eval()

        # ---- hwm_forward equivalent: 1 training-step worth of work --------
        B, N, L_max = 4, 5, 14
        # Random pixels (skipping real WaypointSubtrajectoryDataset for speed).
        batch = {
            'pixels': torch.rand(B, N, 3, 224, 224),
            'actions_chunk': torch.randn(B, N - 1, L_max, 10),
            'actions_mask': torch.ones(B, N - 1, L_max, dtype=torch.bool),
        }
        out = hwm.encode(batch)
        z_W, e_l = out['emb'], out['macro_emb']

        HS = 3
        L_b = torch.randint(1, HS + 1, (B,))
        t_b = L_b + (torch.rand(B) * (N - 1 - L_b + 1)).long()

        preds, tgts = [], []
        for L in range(1, HS + 1):
            sel = (L_b == L).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            t_sel = t_b[sel]
            positions = (t_sel.unsqueeze(1) - L) + torch.arange(L)
            z_ctx = z_W[sel.unsqueeze(1), positions]
            e_ctx = e_l[sel.unsqueeze(1), positions]
            preds.append(hwm.predict(z_ctx, e_ctx)[:, -1])
            tgts.append(z_W[sel, t_sel])
        L_tf = (torch.cat(preds) - torch.cat(tgts).detach()).abs().sum(-1).mean()
        self.assertGreater(L_tf.item(), 0.0)

        # ---- planning: one solve() ---------------------------------------
        solver = HierarchicalCEMSolver(
            model_low=jepa, model_high=hwm,
            high_cfg=dict(num_samples=4, n_steps=1, topk=2, var_scale=1.0, batch_size=1),
            low_cfg=dict(num_samples=4, n_steps=1, topk=2, var_scale=1.0, batch_size=1),
            high_plan_cfg=dict(horizon=2, receding_horizon=1, action_block=1),
            d_l=10, device='cpu', seed=42,
        )
        n_envs = 2
        env_space = gym.spaces.Box(low=-1, high=1, shape=(n_envs, 2), dtype=np.float32)
        plan = swm.PlanConfig(horizon=3, receding_horizon=3, action_block=5)
        solver.configure(action_space=env_space, n_envs=n_envs, config=plan)
        info = {
            'pixels': torch.rand(n_envs, 3, 224, 224),
            'goal':   torch.rand(n_envs, 3, 224, 224),
        }
        out = solver.solve(info)
        self.assertEqual(out['actions'].shape, (n_envs, 3, 10))


if __name__ == '__main__':
    unittest.main(verbosity=2)
