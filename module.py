import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class MacroActionEncoder(nn.Module):
    """A_psi: variable-length action chunk -> macro-action latent l in R^{d_l}.

    HWM paper Sec. 3.2 -- abstract action encoder. Compresses a chunk of
    a_{t_k:t_{k+1}} primitive-action blocks into a single latent macro-action
    via a transformer with a learnable CLS token + MLP head.

    We do NOT reuse module.Block because Block.attn has no key-padding mask
    and our chunks are right-padded to a fixed L_max. Inlining a 2-layer
    masked self-attention stack keeps module.Block unchanged.
    """

    def __init__(
        self,
        input_dim,
        d_l,
        d_token=64,
        n_layers=2,
        n_heads=4,
        mlp_head_dim=128,
        max_blocks=14,
        dropout=0.1,
    ):
        super().__init__()
        self.d_l = d_l
        self.d_token = d_token
        self.n_heads = n_heads
        self.head_dim = d_token // n_heads
        self.dropout_p = float(dropout)
        assert d_token % n_heads == 0, "d_token must be divisible by n_heads"

        self.token_embed = nn.Linear(input_dim, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        # +1 slot for the CLS token at index 0.
        self.pos_embedding = nn.Parameter(torch.randn(1, max_blocks + 1, d_token) * 0.02)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.ModuleList([nn.LayerNorm(d_token) for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_token) for _ in range(n_layers)])
        self.qkv = nn.ModuleList(
            [nn.Linear(d_token, 3 * d_token, bias=False) for _ in range(n_layers)]
        )
        self.proj = nn.ModuleList(
            [nn.Linear(d_token, d_token) for _ in range(n_layers)]
        )
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_token, 4 * d_token),
                    nn.GELU(),
                    nn.Linear(4 * d_token, d_token),
                    nn.Dropout(dropout),
                )
                for _ in range(n_layers)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, mlp_head_dim),
            nn.SiLU(),
            nn.Linear(mlp_head_dim, d_l),
        )
        # Zero-init head's last layer so initial macro-actions start near zero
        # (mirrors AdaLN-zero pattern in ConditionalBlock).
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _attend(self, x, key_pad_mask, layer_idx):
        """Single masked self-attention layer (no causal mask -- bidirectional).

        x: (B, L+1, d_token)
        key_pad_mask: (B, L+1) bool -- True where the slot is REAL (CLS + valid blocks).
        """
        B, T, D = x.shape
        h = self.norm1[layer_idx](x)
        qkv = self.qkv[layer_idx](h).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.n_heads) for t in qkv)

        # attn_mask: (B, 1, 1, T) -> -inf at padded keys so softmax ignores them.
        # F.scaled_dot_product_attention accepts a float bias-style mask.
        # Need to broadcast key_pad_mask to (B, 1, 1, T).
        attn_bias = torch.zeros(B, 1, 1, T, dtype=x.dtype, device=x.device)
        attn_bias.masked_fill_(~key_pad_mask[:, None, None, :], float("-inf"))

        drop = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=drop, is_causal=False
        )
        out = rearrange(out, "b h t d -> b t (h d)")
        x = x + self.proj[layer_idx](out)

        h2 = self.norm2[layer_idx](x)
        x = x + self.mlp[layer_idx](h2)
        return x

    def forward(self, actions, mask):
        """
        actions: (B, L, input_dim)  -- right-padded LeWM action blocks
        mask:    (B, L) bool        -- True for REAL blocks, False for padded
        returns: (B, d_l)
        """
        B, L, _ = actions.shape
        x = self.token_embed(actions.float())  # (B, L, d_token)
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, L+1, d_token); CLS at index 0
        x = x + self.pos_embedding[:, : L + 1]
        x = self.dropout(x)

        # CLS slot is always valid; prepend True to the mask.
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        key_pad_mask = torch.cat([cls_mask, mask], dim=1)  # (B, L+1)

        for layer_idx in range(len(self.qkv)):
            x = self._attend(x, key_pad_mask, layer_idx)

        cls_out = x[:, 0]  # (B, d_token)
        return self.head(cls_out)  # (B, d_l)
