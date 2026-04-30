"""High-level world model training entry point.

HWM paper Sec. 3.1, Eq. 1 -- teacher-forced L1 on next-waypoint latent. The
encoder is shared with the (already trained) low-level JEPA and frozen here;
we only optimise the macro-action encoder, the macro embedder, the new
ARPredictor, and a fresh pred_proj (architecture proposal §5.1).

Sliding-window training over prefix lengths L in {1, ..., HS} (architecture
proposal §5.2 step 4). At inference the high-level rollout begins with a
single latent and reaches steady-state context length HS only after HS-1
steps; without prefix-length training, half the rollout would be OOD.
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from data import WaypointSubtrajectoryDataset
from hwm import HighLevelWorldModel
from module import ARPredictor, MLP, MacroActionEncoder
from utils import get_img_preprocessor, ModelObjectCallBack


def _build_action_norm(dataset):
    """Per-dimension StandardScaler-equivalent for the 'action' column.

    Mirrors utils.get_column_normalizer's body but returns the raw
    ((x - mean) / std) callable instead of a dict-keyed transform, since
    WaypointSubtrajectoryDataset normalises actions on its own loaded slice.
    """
    col_data = dataset.get_col_data('action')
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone().clamp_min(1e-6)

    def norm_fn(x: torch.Tensor) -> torch.Tensor:
        return ((x - mean) / std).float()

    return norm_fn


def hwm_forward(self, batch, stage, cfg):
    """Per-batch forward for high-level training.

    HWM paper Eq. 1: L_tf = (1/N) sum_k |z_hat_{t_{k+1}} - z_{t_{k+1}}|_1.
    Implemented via prefix-length sliding-window sampling so the predictor
    sees every context length L in {1, ..., HS} during training and the
    inference rollout's first HS-1 steps are not OOD (architecture §5.2).
    """
    HS = cfg.wm.history_size

    # 1. Encode pixels (frozen encoder) and action chunks (trainable A_psi).
    output = self.model.encode({
        'pixels': batch['pixels'],
        'actions_chunk': batch['actions_chunk'],
        'actions_mask': batch['actions_mask'],
    })
    z_W = output['emb']        # (B, N, D) -- frozen-encoder waypoint latents
    e_l = output['macro_emb']  # (B, N-1, D) -- AdaLN conditioning
    l_raw = output['macro_l']  # (B, N-1, d_l) -- raw macro-actions

    B, N, D = z_W.shape

    # 2. Per-item prefix-length sampling.
    #    L_b ~ Uniform{1, ..., HS}; t_b ~ Uniform{L_b, ..., N-1}.
    device = z_W.device
    L_b = torch.randint(1, HS + 1, (B,), device=device)  # in [1, HS]
    # Per-item upper bound for t_b is N-1 (inclusive); lower bound is L_b.
    t_low = L_b
    t_high = torch.full((B,), N - 1, device=device)
    span = (t_high - t_low + 1).clamp_min(1)
    t_b = t_low + (torch.rand(B, device=device) * span).long().clamp_max(span - 1)

    # 3. Per-length batching. Group items by L, forward each group separately.
    pred_chunks, tgt_chunks = [], []
    for L in range(1, HS + 1):
        sel = (L_b == L).nonzero(as_tuple=True)[0]
        # Skip empty groups; also skip G=1 -- pred_proj's BatchNorm1d would
        # crash with "Expected more than 1 value per channel". Real training
        # batch=64 with HS=3 gives expected G ≈ 21 per group; G=1 only
        # arises with very small val batches.
        if sel.numel() < 2:
            continue
        b_sel = sel                                 # (G,)
        t_sel = t_b[sel]                            # (G,)
        # positions to gather: (G, L)
        positions = (t_sel.unsqueeze(1) - L) + torch.arange(L, device=device)
        z_ctx = z_W[b_sel.unsqueeze(1), positions]   # (G, L, D)
        e_ctx = e_l[b_sel.unsqueeze(1), positions]   # (G, L, D)
        z_tgt = z_W[b_sel, t_sel]                    # (G, D)

        pred = self.model.predict(z_ctx, e_ctx)[:, -1]  # (G, D)
        pred_chunks.append(pred)
        tgt_chunks.append(z_tgt)

    pred_all = torch.cat(pred_chunks, dim=0)
    tgt_all = torch.cat(tgt_chunks, dim=0)
    # HWM Eq. 1 is the L1 norm summed over D; we mean over D as well so the
    # loss magnitude (~1) matches LeWM's MSE under shared lr/grad-clip.
    # Argmin is unchanged; CEM cost adapters keep the sum form for ranking.
    L_tf = (pred_all - tgt_all.detach()).abs().mean()

    # 4. EMA update for the macro-action prior buffers (used at planning
    #    time by HierarchicalCEMSolver to seed CEM and weight the prior
    #    penalty -- see planner.HighLevelCostAdapter.get_cost).
    self.model.update_macro_prior(l_raw, momentum=cfg.macro_prior.ema_momentum)

    output['L_tf'] = L_tf
    output['loss'] = L_tf
    output['macro_norm'] = l_raw.norm(dim=-1).mean()
    output['macro_std_mean'] = self.model.macro_std.mean()

    log_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items()
        if k in ('loss', 'L_tf', 'macro_norm', 'macro_std_mean')
    }
    self.log_dict(log_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='./config/train', config_name='hwm')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    base = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    # Pixel preprocessor (matches train.py:55) and action normaliser.
    pixel_transform = get_img_preprocessor(
        source='pixels', target='pixels', img_size=cfg.img_size,
    )
    action_norm = _build_action_norm(base)

    # Compose's transforms expect a dict; we built `pixel_transform` which is
    # itself a Compose over the 'pixels' key. We pass it directly to the
    # waypoint dataset, which calls it on the (N, C, H, W) waypoint pixels.
    def pixel_only(x):
        return pixel_transform({'pixels': x})['pixels']

    waypoints = WaypointSubtrajectoryDataset(
        base=base,
        n_target=cfg.data.waypoint_sampler.n_target,
        min_blocks=cfg.data.waypoint_sampler.min_blocks,
        max_blocks=cfg.data.waypoint_sampler.max_blocks,
        mode=cfg.data.waypoint_sampler.mode,
        stride=cfg.data.waypoint_sampler.get('stride'),
        action_normalizer=action_norm,
        pixel_transform=pixel_only,
        seed=cfg.seed,
    )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        waypoints,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False,
    )

    ##############################
    ##       model / optim      ##
    ##############################

    # Load frozen low-level JEPA. utils.ModelObjectCallBack pickles the
    # bare module via torch.save(model, path); torch.load(weights_only=False)
    # reconstructs it.
    low_level = torch.load(cfg.low_level_ckpt, map_location='cpu', weights_only=False)
    if hasattr(low_level, 'model'):
        # In case the ckpt was saved as a spt.Module wrapper; unwrap.
        low_level = low_level.model
    low_level.requires_grad_(False).eval()

    encoder = low_level.encoder
    projector = low_level.projector
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get('embed_dim', hidden_dim)

    # Action chunk dim per LeWM block (e.g., 10 for Push-T frameskip 5 x 2-D).
    a_block_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    macro_encoder = MacroActionEncoder(
        input_dim=a_block_dim,
        d_l=cfg.wm.d_l,
        d_token=cfg.macro_encoder.d_token,
        n_layers=cfg.macro_encoder.n_layers,
        n_heads=cfg.macro_encoder.n_heads,
        mlp_head_dim=cfg.macro_encoder.mlp_head_dim,
        max_blocks=cfg.macro_encoder.max_blocks,
        dropout=cfg.macro_encoder.get('dropout', 0.1),
    )

    macro_embedder = torch.nn.Linear(cfg.wm.d_l, embed_dim)
    torch.nn.init.normal_(macro_embedder.weight, std=0.02)
    torch.nn.init.zeros_(macro_embedder.bias)

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=cfg.predictor.depth,
        heads=cfg.predictor.heads,
        mlp_dim=cfg.predictor.mlp_dim,
        dim_head=cfg.predictor.dim_head,
        dropout=cfg.predictor.dropout,
        emb_dropout=cfg.predictor.emb_dropout,
    )

    pred_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    if cfg.wm.get('share_pred_proj', False):
        # Ablation: reuse and freeze the low-level pred_proj.
        pred_proj = low_level.pred_proj
        pred_proj.requires_grad_(False).eval()

    world_model = HighLevelWorldModel(
        encoder=encoder,
        projector=projector,
        macro_encoder=macro_encoder,
        macro_embedder=macro_embedder,
        predictor=predictor,
        pred_proj=pred_proj,
        d_l=cfg.wm.d_l,
        history_size=cfg.wm.history_size,
    )

    # Optimiser only sees trainable parameters; encoder/projector are frozen.
    # spt.Module's optim spec selects by `modules` -- 'model' wraps everything
    # but Lightning's optimizer construction will pick up only requires_grad
    # params via filter at the param-group level. We make this explicit.
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {'type': 'LinearWarmupCosineAnnealingLR'},
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=world_model,
        forward=partial(hwm_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        ckpt_path=run_dir / f'{cfg.output_model_name}_weights.ckpt',
    )

    manager()
    return


if __name__ == '__main__':
    run()
