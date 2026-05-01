# Phase 6: Code Review

Skeptical review of every file created or modified in Phase 5. Bugs are
ranked by severity (critical → cosmetic). After the table, "Fix" sections
record the actual edits applied.

---

## Bug Index

| # | File | Lines | Severity | Description |
|---|------|-------|----------|-------------|
| 1 | `train_highlevel.py` | 92 | **HIGH** | L_tf magnitude is ~D× larger than LeWM's MSE, destabilising training under shared lr/grad-clip. Use `.abs().mean()` over (B, D), not `.sum(-1).mean()`. Same fix needed in cost adapters for monotonicity preservation. |
| 2 | `planner.py` (`_encode_and_cache`) | 354–366 | **HIGH** | Goal-pixel cache key is `data_ptr()`. `WorldModelPolicy._prepare_info` rebuilds the goal tensor every call, so `data_ptr` changes and cache always misses → goal is re-encoded every `solve()`. Replace with content-equality cache. |
| 3 | `planner.py` (`_replan_high`) | 393–397 | **MEDIUM** | High-level CEM init uses `μ_l` for the mean but ignores `σ_l` for the variance — `var_scale` stays at the yaml default (1.0). Architecture proposal §6.2 explicitly requested per-dim σ_l scaling. |
| 4 | `module.py` (`MacroActionEncoder._attend`) | 372–375 | **MEDIUM** | Attention dropout is hardcoded to `0.0` despite `dropout` arg being captured at construction. The `drop` local is dead code. |
| 5 | `hwm.py` | 13, 17 | **LOW** | Unused imports: `torch.nn.functional as F`, `from jepa import detach_clone`. |
| 6 | `eval.py` | 88–112 (mod) | **LOW** | After loading `model_high`, its encoder/projector are duplicates of `model_low`'s — wasted memory. Suggest sharing references at load time. |
| 7 | `data.py` (`__getitem__`) | 122 | **LOW** | `np.random.default_rng()` per call has no seed; non-reproducible across runs. Use a per-call rng seeded from `torch.utils.data.get_worker_info()` for reproducibility. |
| 8 | `train_highlevel.py` (`hwm_forward`) | n/a | **LOW** | Per-length groups can have G=1 items at small batch sizes, which crashes `pred_proj`'s `BatchNorm1d`. Real cfg has B=64 so safe in practice, but the smoke test discovered this — add an assert. |
| 9 | `data.py` (`_sample_waypoints`) | 122 | **LOW** | Non-determinism inside the loader makes the prefix-length test (`Test07`) verify the *sampler distribution* but not the actual `hwm_forward` plumbing. Architecture spec §7.6 wanted forward-instrumentation. |
| 10 | `hwm.py` (`get_cost`) | 234–237 | **LOW** | Mutates user's `info_dict` in place via `.to(device)`. Hierarchical path bypasses this method (uses adapters instead), but defensive coding requires a shallow copy first. |
| 11 | `hwm.py` (`rollout`) | 176 | **COSMETIC** | Branch disambiguation between `(B, T, D)` and `(B, S, D)` is fragile when `T == S`. Not exercised in any code path; flag for future refactor. |
| 12 | `train_highlevel.py` | 209 | **COSMETIC** | Optimizer config uses `'modules': 'model'` which selects all params including frozen encoder/projector. Empirically harmless (None-grad params skipped) but wasteful. |

---

## Fixes

### Fix #1 — Loss magnitude (HIGH)

**Diagnosis.** HWM Eq. 1 is `(1/N) Σ_k ‖ẑ - z‖₁`. Implementing this literally
as `.abs().sum(-1).mean()` produces a loss ~`D × mean_abs_diff`. With
`D = 192`, the L_tf magnitude at random init is ~58 (matches the smoke
test); LeWM's MSE under the same setup is ~1. Same lr, AdamW, and
`gradient_clip_val: 1.0` produce **clipped gradients** for HWM —
effectively cutting the lr by ~D.

For CEM the *ranking* is monotone in scale, so cost can stay summed; for
training the gradient magnitude matters. We change the **training loss**
to `.abs().mean()` over `(B, D)`. Cost adapters keep the sum form (CEM
ranks unchanged, but document the difference).

**Edit:** `train_highlevel.py` `hwm_forward`.

### Fix #2 — Goal-pixel cache miss (HIGH)

**Diagnosis.** `WorldModelPolicy._prepare_info` (`stable-worldmodel/policy.py:121–183`)
calls `torch.stack([transform[k](...) for x in v])` for each pixel-keyed
field. That allocates a fresh tensor every `solve()` call. So
`info['goal'].data_ptr()` differs every call → my `_goal_cache_key`
comparison always misses → goal re-encoded every solve.

The miss isn't catastrophic (one encode per `solve()`, not per CEM iter),
but the comment says we cache "once per unique goal" — false advertising.

**Fix.** Compare goal tensors by content via `torch.equal` (or a small
shape-then-bytes check). Goal is small (typically `(n_envs, C, 224, 224)`
≈ 30 MB at fp32), and we run this once per `solve()` (every 25 env steps
on Push-T), so the comparison cost is microseconds and well amortised.

**Edit:** `planner.py` `_encode_and_cache`.

### Fix #3 — Macro-prior σ_l for CEM init variance (MEDIUM)

**Diagnosis.** Architecture proposal §6.2 step 3:
> The `var_scale` is multiplied by `model_high.macro_std.mean()` ... so
> the initial sampling spread is plausible in macro-action space.

Currently `solver_high.var_scale = high_cfg.var_scale = 1.0` (set at
`__init__` time, never updated). For a trained σ_l ≈ 0.3, CEM samples are
3× too noisy on the first iteration, slowing convergence.

**Fix.** Override `solver_high.var_scale` per-call before `solve()` to
`macro_std.mean().item()`. Cleanly contained inside `_replan_high`.
(A more principled fix would override `init_action_distrib` to set
per-dim `var = σ_l²`, but that requires subclassing CEMSolver. Punt to
an ablation if the scalar override doesn't suffice.)

**Edit:** `planner.py` `_replan_high`.

### Fix #4 — MacroActionEncoder attention dropout (MEDIUM)

**Diagnosis.** `_attend` constructs `drop = self.qkv[layer_idx].weight.new_tensor(0.0)`
but never passes it to `F.scaled_dot_product_attention`. The `dropout_p=0.0`
kwarg is hardcoded, so attention dropout is silently disabled regardless
of the constructor's `dropout` arg.

**Fix.** Store `self.dropout_p` and use it in `F.scaled_dot_product_attention`
when `self.training`. Remove the dead `drop` local.

**Edit:** `module.py` `MacroActionEncoder`.

### Fix #5 — Unused imports (LOW)

**Edit:** Remove `import torch.nn.functional as F` and
`from jepa import detach_clone` from `hwm.py`. Neither is referenced.

### Fix #6 — Encoder duplication at load time (LOW)

**Diagnosis.** `model_low` and `model_high` are loaded via two separate
`AutoCostModel` calls; each pickle includes its own copy of the ViT-Tiny
encoder (~5M params). Memory waste ~40 MB; functionally fine since the
encoders are bit-identical (HWM's encoder was the frozen-shared reference
to JEPA's at training time). At inference, sharing the reference also
guarantees the shared-latent invariant against any future drift.

**Fix.** After both checkpoints are loaded in `eval.py`, alias
`model_high.encoder = model_low.encoder` and
`model_high.projector = model_low.projector`. Document why.

**Edit:** `eval.py` hierarchical branch.

### Fix #7 — Worker-deterministic dataset rng (LOW)

**Diagnosis.** `np.random.default_rng()` per call uses OS entropy → fully
non-deterministic. Repro across runs is impossible. PyTorch best practice
is to seed via `worker_init_fn` so each DataLoader worker gets a unique,
reproducible seed. Without it, results vary even with `torch.manual_seed`.

**Fix.** Use `torch.utils.data.get_worker_info()` to derive a per-worker
seed; combine with the dataset's `_init_seed` and the call index for a
unique-but-reproducible rng per `__getitem__`.

**Edit:** `data.py` `__getitem__`.

### Fix #8 — Per-length group size assertion (LOW)

**Diagnosis.** With `B < 2 * HS`, a per-length group can have G=1 item.
`pred_proj`'s `BatchNorm1d` then fails with
`"Expected more than 1 value per channel when training, got input size [1, 2048]"`.
The smoke-test caught it at B=4. Real training uses B=64, so safe;
mid-training validation could trip it if val_batch_size is small.

**Fix.** Skip per-length groups with size <2 during training (warn but
continue — they contribute no learning signal). Equivalent to "don't
optimize this length this step", which is fine since the next step
hits a different sample.

**Edit:** `train_highlevel.py` `hwm_forward`.

### Fix #9 — Test07 forward instrumentation (LOW)

**Diagnosis.** `Test07PrefixLengthCoverage` re-implements the sampler and
checks the distribution — it doesn't actually call `hwm_forward` and
verify that the forward consumed all lengths. A regression that hardcodes
`L=3` in `hwm_forward` would not be caught.

**Fix.** Replace the test body with a call into `hwm_forward` (with a
counter monkeypatched onto the predictor) over many synthetic batches
to assert each L group fires.

**Edit:** `tests/test_hierarchical.py`.

### Fix #10 — `HighLevelWorldModel.get_cost` mutation (LOW)

**Diagnosis.** `info_dict[k] = info_dict[k].to(device)` mutates the
caller's dict. Hierarchical adapters bypass this method, but defensive
copying is cheap.

**Fix.** Shallow-copy `info_dict` before any modification.

**Edit:** `hwm.py` `get_cost`.

### Fix #11 — Rollout shape branching fragility (COSMETIC)

Documented in the file comment; not exercised by any code path. Skipped.

### Fix #12 — Optimizer params include frozen modules (COSMETIC)

Lightning + AdamW silently skip None-grad params; no functional bug.
Mirrors the existing `train.py` pattern. Skipped to minimise diff.

---

## Test result after fixes

All 10 unit tests must continue to pass. Run via:

```
python -m unittest tests.test_hierarchical -v
```
