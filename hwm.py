"""High-level world model for hierarchical planning.

HWM paper Sec. 3.1 -- two world models share a single encoder. The high-level
model P^(2) operates on macro-action latents l in R^{d_l} and predicts the
next waypoint latent z_{t_{k+1}} given (z_{t_k}, l_{t_k}).

This module mirrors `JEPA` (jepa.py) so that `swm.policy.AutoCostModel`
loads it transparently (loader scans for a `get_cost` attribute, see
`stable-worldmodel/policy.py:556-574`).
"""

import torch
from einops import rearrange
from torch import nn


class HighLevelWorldModel(nn.Module):
    """JEPA-shaped wrapper for the high-level dynamics P^(2).

    Frozen (shared with the low-level JEPA): encoder, projector.
    Trainable: macro_encoder (A_psi), macro_embedder (d_l -> 192),
               predictor (ARPredictor with num_frames=history_size),
               pred_proj.

    Persistent buffers: macro_mean, macro_std -- empirical mean/std of
    A_psi outputs at training time, used by HierarchicalCEMSolver to
    initialise the high-level CEM and by HighLevelCostAdapter as a soft
    prior penalty.
    """

    def __init__(
        self,
        encoder,
        projector,
        macro_encoder,
        macro_embedder,
        predictor,
        pred_proj,
        d_l,
        history_size,
    ):
        super().__init__()

        self.encoder = encoder
        self.projector = projector
        self.macro_encoder = macro_encoder  # A_psi
        self.macro_embedder = macro_embedder  # Linear(d_l, embed_dim) for AdaLN
        self.predictor = predictor
        self.pred_proj = pred_proj

        self.d_l = int(d_l)
        self.history_size = int(history_size)

        # Macro-action prior (HWM paper does not name this; we add it because
        # CEM in unbounded R^{d_l} starting at N(0, I) wastes most samples).
        self.register_buffer("macro_mean", torch.zeros(self.d_l))
        self.register_buffer("macro_std", torch.ones(self.d_l))

        # Freeze the shared modules (idempotent).
        self.encoder.requires_grad_(False).eval()
        self.projector.requires_grad_(False).eval()

    def freeze_shared(self):
        """Re-apply the freeze (call after loading a checkpoint)."""
        self.encoder.requires_grad_(False).eval()
        self.projector.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        """Keep the frozen modules in eval() even when the wrapper is trained.

        Default nn.Module.train() recurses into children, which would
        re-enable dropout/BN-update in the encoder and projector. The shared
        latent invariant requires deterministic frozen latents.
        """
        super().train(mode)
        self.encoder.eval()
        self.projector.eval()
        return self

    # ----------------------------------------------------------------------
    # JEPA-shaped public API
    # ----------------------------------------------------------------------

    def encode(self, info):
        """Encode pixels (and optionally action chunks) into latents.

        Mirrors JEPA.encode:
        - reads info['pixels'] -> writes info['emb']
        - if info contains 'actions_chunk' + 'actions_mask', writes
          info['macro_emb'] of shape (B, K, embed_dim) -- the AdaLN
          conditioning for P^(2). Used at TRAINING TIME only.

        At PLANNING TIME the macro-action candidates come from CEM directly
        in R^{d_l}; the rollout calls macro_embedder on them and bypasses
        the macro_encoder. See `rollout()`.
        """
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        with torch.no_grad():
            output = self.encoder(pixels, interpolate_pos_encoding=True)
            cls = output.last_hidden_state[:, 0]
            emb = self.projector(cls)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "actions_chunk" in info and "actions_mask" in info:
            chunks = info["actions_chunk"]  # (B, K, L_max, A_block)
            mask = info["actions_mask"]  # (B, K, L_max) bool
            B, K, L, A = chunks.shape
            l = self.macro_encoder(
                chunks.reshape(B * K, L, A), mask.reshape(B * K, L)
            )  # (B*K, d_l)
            info["macro_l"] = l.reshape(B, K, self.d_l)
            info["macro_emb"] = self.macro_embedder(info["macro_l"])

        return info

    def predict(self, emb, macro_emb):
        """Predict next-step latent given context latents and macro embeddings.

        emb:       (B, T, embed_dim)
        macro_emb: (B, T, embed_dim) -- AdaLN conditioning, already lifted
                   from R^{d_l} via macro_embedder
        returns:   (B, T, embed_dim)
        """
        preds = self.predictor(emb, macro_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    # ----------------------------------------------------------------------
    # Inference-only -- mirrors JEPA.rollout's signature
    # ----------------------------------------------------------------------

    @torch.inference_mode()
    def rollout(self, info, action_sequence, history_size: int = None):
        """Sliding-window rollout in macro-action space.

        Mirrors JEPA.rollout (jepa.py:61-110) but conditions on macro-action
        candidates l_cands instead of primitive action blocks. The macro_encoder
        is BYPASSED -- l_cands come straight from CEM in R^{d_l}.

        Inputs:
            info: dict with 'emb' (B, S, T_init, D) already populated by the
                  caller (typically HierarchicalCEMSolver._encode_and_cache).
                  CEMSolver expands info tensors over the sample dim, so emb
                  arrives as (B, S, T_init, D) or (B, S, D); we reshape to
                  the (B, S, T_init, D) form expected by the loop.
            action_sequence: (B, S, T, d_l) -- macro-action candidates.

        Returns:
            info with 'predicted_emb' set to (B, S, T_init + T, D).
        """
        HS = history_size if history_size is not None else self.history_size

        B, S, T = action_sequence.shape[:3]

        # Lift macro-actions -> embed_dim AdaLN conditioning.
        # action_sequence (B, S, T, d_l) -> macro_emb_seq (B, S, T, embed_dim)
        flat_l = action_sequence.reshape(B * S * T, self.d_l)
        flat_e = self.macro_embedder(flat_l)
        macro_emb_seq = flat_e.reshape(B, S, T, -1)

        # Bring the initial latent into (B, S, T_init, D). Accepted inputs:
        #   (B, D)        -- single latent, no time axis    -> (B, S, 1, D)
        #   (B, T, D)     -- history of latents, no S axis  -> (B, S, T, D)
        #   (B, S, D)     -- CEM-expanded single latent     -> (B, S, 1, D)
        #   (B, S, T, D)  -- CEM-expanded history           -> as-is
        emb = info["emb"]
        if emb.dim() == 2:                          # (B, D)
            emb = emb.unsqueeze(1).unsqueeze(1)     # (B, 1, 1, D)
            emb = emb.expand(B, S, 1, emb.size(-1))
        elif emb.dim() == 3 and emb.size(0) == B and emb.size(1) != S:  # (B, T, D)
            emb = emb.unsqueeze(1).expand(B, S, emb.size(1), emb.size(2))
        elif emb.dim() == 3 and emb.size(1) == S:   # (B, S, D)
            emb = emb.unsqueeze(2)                  # (B, S, 1, D)
        # else: (B, S, T, D) -- as-is
        emb = emb.contiguous()

        # Flatten (B, S) -> (BS) for the rollout loop, mirroring JEPA.rollout.
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        macro_seq = rearrange(macro_emb_seq, "b s ... -> (b s) ...")

        # Autoregressive rollout. At step t, context length is L = min(t+1, HS)
        # because [:, -HS:] takes whatever is available; the predictor was
        # trained on every L in {1,...,HS} via prefix-length sampling.
        for t in range(T):
            emb_trunc = emb[:, -HS:]  # (BS, L, D)
            mac_trunc = macro_seq[:, max(0, t + 1 - HS) : t + 1]  # (BS, L, D)
            # Align lengths if context grew faster than macro window.
            L = min(emb_trunc.size(1), mac_trunc.size(1))
            emb_trunc = emb_trunc[:, -L:]
            mac_trunc = mac_trunc[:, -L:]

            pred_emb = self.predict(emb_trunc, mac_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        return info

    def criterion(self, info_dict: dict):
        """L1 cost between the predicted final latent and the goal latent.

        HWM paper Eq. 2 (planning cost). pred_emb shape (B, S, T_total, D);
        goal_emb shape varies (B,D) / (B,S,D) / (B,S,T,D) depending on caller.
        Robust shape handling is delegated to the cost adapters in planner.py.
        Here we only handle the simple case where goal_emb is already (B,S,D).
        """
        pred_emb = info_dict["predicted_emb"][..., -1, :]  # (B, S, D)
        goal_emb = info_dict["goal_emb"]
        if goal_emb.dim() == pred_emb.dim() - 1:
            goal_emb = goal_emb.unsqueeze(1).expand_as(pred_emb)
        elif goal_emb.dim() == pred_emb.dim() + 1:
            goal_emb = goal_emb[..., -1, :]
        cost = (pred_emb - goal_emb.detach()).abs().sum(-1)  # (B, S)
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """CEMSolver contract: cost of macro-action candidates.

        action_candidates: (B, S, T, d_l).
        info_dict must already contain 'emb' (current latent) and 'goal_emb'
        (final goal latent). The hierarchical solver's _encode_and_cache
        guarantees this; calling get_cost on a bare info_dict with raw pixels
        will fail intentionally.
        """
        assert (
            "emb" in info_dict and "goal_emb" in info_dict
        ), "HighLevelWorldModel.get_cost requires pre-cached 'emb' and 'goal_emb'"
        # Shallow-copy so we don't mutate the caller's dict (rollout writes
        # 'predicted_emb' and we may .to(device) tensors that aren't on it).
        info_local = {k: v for k, v in info_dict.items()}
        device = next(self.parameters()).device
        for k in list(info_local.keys()):
            if torch.is_tensor(info_local[k]):
                info_local[k] = info_local[k].to(device)
        info_local = self.rollout(info_local, action_candidates)
        return self.criterion(info_local)

    # ----------------------------------------------------------------------
    # Macro-action prior bookkeeping (used by hwm_forward in train_highlevel.py)
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def update_macro_prior(self, l_batch: torch.Tensor, momentum: float = 0.99):
        """EMA-update macro_mean and macro_std from a batch of A_psi outputs.

        l_batch: (..., d_l) -- any leading shape; flattens for stats.
        """
        flat = l_batch.detach().reshape(-1, self.d_l)
        if flat.size(0) < 2:
            return
        batch_mean = flat.mean(0)
        batch_std = flat.std(0).clamp_min(1e-6)
        self.macro_mean.mul_(momentum).add_(batch_mean, alpha=1 - momentum)
        self.macro_std.mul_(momentum).add_(batch_std, alpha=1 - momentum)
