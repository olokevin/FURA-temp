# SVD-LLM-v2 Compressed Full-FT on MATH-10K Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `svd_llm_v2` / `svd_llm_v2_combined` compression into `ref/LIFT/src/finetune_sft.py` via the existing `compress_integration.py`, then deliver six launcher scripts for the MATH-10K → gsm8k experiment (Llama-3.2-1B / 3B baselines plus 3B compressed to 2B and 1B via two SVD modes).

**Architecture:** Extend `compress_integration.py` to recognize two new `calib_mode` values (`svd_v2`, `svd_v2_combined`) and a new `--compression_ratio` CLI arg. Add a compression hook to `finetune_sft.py` that calls `decompose_with_loader` after model construction and a materialize step before `save_hf_format` so eval consumes a vanilla HF checkpoint. Six small shell scripts use the same hyperparameters (lr=1e-4, 3 epochs, seed=43) and write to `/data/yequan/compress_train/sft/math/<MODEL>/<run_name>`.

**Tech Stack:** Python 3.13, PyTorch, HuggingFace `transformers`, `accelerate`, the in-repo `compress` package under `src/compress/`, bash. Tests via `unittest`.

---

## Spec reference

Design: `docs/superpowers/specs/2026-05-11-svd-compress-full-ft-math10k-design.md` (commit `b80b20b`).

## File structure

- **`compress_integration.py`** (modify): public surface for LIFT/run_sft callers. New entries in `CALIB_MODE_TO_TRAIN_MODE`; new `--compression_ratio` arg; updated validator branch for SVD modes; new helpers `apply_calibrated_svd` and `materialize_svd_to_linear`.
- **`ref/LIFT/src/finetune_sft.py`** (modify): import-path fix; register calib args; call `apply_calibrated_svd` before optimizer; materialize SVD layers before `save_hf_format`.
- **`tests/test_compress_integration_build_config.py`** (modify): three new cases for the SVD path.
- **`tests/test_compress_integration_materialize.py`** (modify): one new case for SVD materialize.
- **`ref/LIFT/bash_scripts/finetune_math_full_a1_llama32_1b.sh`** (new)
- **`ref/LIFT/bash_scripts/finetune_math_full_a2_llama32_3b.sh`** (new)
- **`ref/LIFT/bash_scripts/finetune_math_full_b1_svd_v2_2b.sh`** (new)
- **`ref/LIFT/bash_scripts/finetune_math_full_b2_svd_v2_combined_2b.sh`** (new)
- **`ref/LIFT/bash_scripts/finetune_math_full_c1_svd_v2_1b.sh`** (new)
- **`ref/LIFT/bash_scripts/finetune_math_full_c2_svd_v2_combined_1b.sh`** (new)

## Key project facts the implementer needs

- `compress_integration.py` already wires BTT calibrated modes; we are mirroring that pattern for SVD modes.
- `compress.svd.svd_linear.SVDCompressedLinear` initialises `U_r` and `V_r` as `nn.Parameter(..., requires_grad=False)`. **We MUST flip both to `requires_grad=True` after `decompose_with_loader` returns**, because the SVD modes in `decomposition.py:289` call `model.requires_grad_(True)` — but `SVDCompressedLinear` overrides `requires_grad` per-Parameter at construction. Verify by inspection; if `model.requires_grad_(True)` already cascades through `SVDCompressedLinear.parameters()` (it should, since `Parameter.requires_grad` is settable), no extra code needed. The test in Task 5 catches this either way.
- `compress.decomposition.decompose_with_loader` returns the in-place mutated model AND ignores `decomp_mode`/`train_position` for non-BTT modes (those are BTT-only) — so passing them through `build_decomposition_config` is harmless but the SVD branch should set them to safe defaults.
- `compress_integration.build_calib_loader(args, tokenizer=..., training_dataset=..., training_collate_fn=...)` already handles `calib_source=training_data`; we reuse it as-is.
- `ref/LIFT/src/finetune_sft.py` uses `save_hf_format(last_model, tokenizer, args, sub_folder="last")` at line 935. That function (in `ref/LIFT/src/utils/model_utils.py`) calls `torch.save(state_dict(), ...)` and `config.to_json_file(...)`. **For SVD-mode runs we must replace every `SVDCompressedLinear` with `nn.Linear(materialized_dense)` BEFORE this save call**, otherwise the saved state_dict will contain keys like `model.layers.0.self_attn.q_proj.U_r` that the un-compressed Llama config cannot load at eval time.
- Models cached: `/data/ryan/huggingface_cache/hub/models--meta-llama--Llama-3.2-{1B,3B}`. Use the literal HF id `meta-llama/Llama-3.2-1B` / `meta-llama/Llama-3.2-3B`; `HF_HOME` is set per-script to that cache.
- MATH-10K: `/data/ruijiezhang/llm-adapter_bp/LLM-Adapters/ft-training_set/math_10k.json`.
- Output root: `/data/yequan/compress_train/sft/math/<MODEL>/<run_name>`.
- Wandb project: `compress_sft_math_llama3.2`.
- Eval: `EVAL_DATASETS="gsm8k" bash ./bash_scripts/eval_math.sh CKPT=$OUTPUT/last base_model=$MODEL ...`.

---

## Task 1: Extend `CALIB_MODE_TO_TRAIN_MODE` map and add `--compression_ratio` arg

**Files:**
- Modify: `compress_integration.py:40-46` (map), `compress_integration.py:58-95` (`add_calibrated_btt_args`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compress_integration_build_config.py`:

```python
    def test_calib_mode_to_train_mode_mapping_includes_svd(self):
        for calib, train in [("svd_v2", "svd_llm_v2"),
                              ("svd_v2_combined", "svd_llm_v2_combined")]:
            args = _parse(["--calib-mode", calib, "--calib-source", "training_data",
                           "--compression-ratio", "0.5"])
            cfg = ci.build_decomposition_config(args, model=ToyModel())
            self.assertEqual(cfg.train_mode, train)
            self.assertAlmostEqual(cfg.compression_ratio, 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m unittest tests.test_compress_integration_build_config.TestBuildConfig.test_calib_mode_to_train_mode_mapping_includes_svd -v
```

Expected: FAIL with `argument --calib-mode: invalid choice: 'svd_v2'` (or similar — the choices list does not yet include `svd_v2`).

- [ ] **Step 3: Extend the map**

In `compress_integration.py`, replace the existing `CALIB_MODE_TO_TRAIN_MODE` definition (around line 40) with:

```python
CALIB_MODE_TO_TRAIN_MODE = {
    "v2": "btt_llm_v2",
    "v2_bp": "btt_llm_v2_bp",
    "v2_combined": "btt_llm_v2_combined",
    "twosteps": "btt_twosteps",
    "svd_v2": "svd_llm_v2",
    "svd_v2_combined": "svd_llm_v2_combined",
}
```

- [ ] **Step 4: Add the `--compression-ratio` arg**

At the end of `add_calibrated_btt_args` (right after the existing `--calib-batch-size` argument), insert:

```python
    parser.add_argument(
        _arg_name("compression-ratio", hyphen_style=hyphen_style),
        type=float, default=1.0,
        help=(
            "Compression ratio for SVD calib modes (svd_v2 / svd_v2_combined). "
            "Fraction of compressible-layer params to retain. Must be in (0, 1]. "
            "Ignored for BTT calib modes (use --blocktt-rank instead)."
        ),
    )
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m unittest tests.test_compress_integration_build_config.TestBuildConfig.test_calib_mode_to_train_mode_mapping_includes_svd -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add compress_integration.py tests/test_compress_integration_build_config.py
git commit -m "$(cat <<'EOF'
compress: add svd_v2 / svd_v2_combined to CALIB_MODE_TO_TRAIN_MODE

Adds the two SVD-LLM-v2 calib modes to the public map and a new
--compression-ratio CLI arg used by the SVD path (BTT path keeps
using --blocktt-rank). No behavior change yet — build_decomposition_config
and validate_calibrated_btt_args still gate on BTT only.

Co-Authored-By: Claude Opus 4 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Route SVD modes through `build_decomposition_config`

**Files:**
- Modify: `compress_integration.py:210-246` (`build_decomposition_config`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compress_integration_build_config.py`:

```python
    def test_svd_v2_uses_compression_ratio_arg(self):
        args = _parse(["--calib-mode", "svd_v2", "--calib-source", "training_data",
                       "--compression-ratio", "0.25"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        self.assertEqual(cfg.train_mode, "svd_llm_v2")
        self.assertAlmostEqual(cfg.compression_ratio, 0.25)

    def test_svd_v2_ignores_blocktt_rank(self):
        # blocktt-rank should be ignored on the SVD path; the ratio comes from
        # --compression-ratio only.
        args = _parse(["--calib-mode", "svd_v2_combined", "--calib-source", "training_data",
                       "--compression-ratio", "0.6", "--blocktt-rank", "0.99"])
        cfg = ci.build_decomposition_config(args, model=ToyModel())
        self.assertEqual(cfg.train_mode, "svd_llm_v2_combined")
        self.assertAlmostEqual(cfg.compression_ratio, 0.6)

    def test_svd_v2_rejects_bad_ratio(self):
        for bad in ("0", "1.5", "-0.1"):
            args = _parse(["--calib-mode", "svd_v2", "--calib-source", "training_data",
                           "--compression-ratio", bad])
            with self.assertRaises(ValueError):
                ci.build_decomposition_config(args, model=ToyModel())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m unittest tests.test_compress_integration_build_config -v
```

Expected: at least the three new tests FAIL (because `build_decomposition_config` still calls `_resolve_ratio_from_rank` for SVD modes, which would give 0.99 on the second test and 1.0 on the first).

- [ ] **Step 3: Branch on SVD modes inside `build_decomposition_config`**

Replace the existing body of `build_decomposition_config` (line 210-246) with:

```python
def build_decomposition_config(args, *, hyphen_style: bool = True, model=None) -> DecompositionConfig:
    """Translate CLI args to a DecompositionConfig. `model` is required so that
    --trainable-type can be inverted into a skip_layers list."""
    del hyphen_style  # kept for signature stability; routed via _arg_name upstream
    if model is None:
        raise ValueError("build_decomposition_config requires model= for skip_layers inversion")

    calib_mode = getattr(args, "calib_mode", None)
    if calib_mode is None:
        raise ValueError(
            "build_decomposition_config requires args.calib_mode; did you call "
            "add_calibrated_btt_args() on the parser?"
        )
    if calib_mode not in CALIB_MODE_TO_TRAIN_MODE:
        raise ValueError(
            f"calib_mode must be one of {sorted(CALIB_MODE_TO_TRAIN_MODE.keys())}; "
            f"got {calib_mode!r}"
        )
    train_mode = CALIB_MODE_TO_TRAIN_MODE[calib_mode]

    if calib_mode.startswith("svd_"):
        ratio_raw = getattr(args, "compression_ratio", 1.0)
        try:
            ratio = float(ratio_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"--compression-ratio must be a float; got {ratio_raw!r}"
            ) from exc
        if not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"--compression-ratio must be in (0, 1] for SVD calib modes; got {ratio}"
            )
    else:
        ratio = _resolve_ratio_from_rank(getattr(args, "blocktt_rank", "full"))

    trainable_type = getattr(args, "trainable_type", "all")
    skip_layers = _build_skip_layers(model, trainable_type)

    return DecompositionConfig(
        train_mode=train_mode,
        compression_ratio=ratio,
        calib_source=getattr(args, "calib_source", "c4"),
        calib_traces_path=getattr(args, "calib_traces_path", None),
        calib_num_seqs=getattr(args, "calib_num_seqs", 128),
        calib_max_length=getattr(args, "calib_max_length", 2048),
        calib_seed=getattr(args, "calib_seed", 3),
        skip_layers=skip_layers,
        decomp_mode=getattr(args, "decomp_mode", "square"),
        train_position=getattr(args, "train_position", "both"),
        s_merged_to=getattr(args, "s_merged_to", None),
        factorize_by_head=bool(getattr(args, "blocktt_factorize_by_head", True)),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m unittest tests.test_compress_integration_build_config -v
```

Expected: ALL tests PASS (the new three plus the existing six).

- [ ] **Step 5: Commit**

```bash
git add compress_integration.py tests/test_compress_integration_build_config.py
git commit -m "$(cat <<'EOF'
compress: route svd_v2 modes through --compression-ratio

build_decomposition_config now branches on calib_mode.startswith('svd_')
and pulls the compression ratio from args.compression_ratio. BTT path
unchanged (still uses --blocktt-rank). Rejects ratios outside (0, 1].

Co-Authored-By: Claude Opus 4 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update `validate_calibrated_btt_args` for SVD modes

**Files:**
- Modify: `compress_integration.py:99-151` (`validate_calibrated_btt_args`).

Today the validator hard-rejects any `train_mode != "blocktt"` and the integer-rank-rejection rule only matters for BTT. For SVD modes, the only validation we need is that `--compression-ratio` is in (0, 1], which `build_decomposition_config` already enforces. We must therefore *exempt* SVD modes from the `train_mode == "blocktt"` rule.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compress_integration_validate.py` (look at the existing file to copy the import-and-namespace pattern; if the file does not have a `_parse` helper, mimic the one in `test_compress_integration_build_config.py`):

```python
    def test_validate_svd_v2_allows_non_blocktt_train_mode(self):
        # SVD calib modes are used with --train-mode full (or absent). The
        # validator must not reject them.
        ns = argparse.Namespace(
            calib_mode="svd_v2",
            calib_source="training_data",
            calib_traces_path=None,
            blocktt_rank="full",
            train_mode="full",
            compression_ratio=0.5,
        )
        ci.validate_calibrated_btt_args(ns, argv=[])

    def test_validate_svd_v2_combined_allows_non_blocktt(self):
        ns = argparse.Namespace(
            calib_mode="svd_v2_combined",
            calib_source="training_data",
            calib_traces_path=None,
            blocktt_rank="full",
            train_mode="full",
            compression_ratio=0.3,
        )
        ci.validate_calibrated_btt_args(ns, argv=[])
```

(If `test_compress_integration_validate.py` does not exist, create it with the standard boilerplate from other tests/test_compress_integration_*.py files plus an `import argparse` and an `import compress_integration as ci`.)

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m unittest tests.test_compress_integration_validate -v
```

Expected: FAIL with "--calib-mode only valid with --train-mode blocktt".

- [ ] **Step 3: Exempt SVD modes**

In `compress_integration.py`, change the first rule inside `validate_calibrated_btt_args` (around line 110-116) from:

```python
    # 1. --calib-mode != none requires --train-mode blocktt (if parser has train_mode)
    if calib_mode != "none" and hasattr(args, "train_mode"):
        if args.train_mode != "blocktt":
            raise ValueError(
                "--calib-mode only valid with --train-mode blocktt "
                f"(got --train-mode={args.train_mode!r})"
            )
```

to:

```python
    # 1. BTT calib modes require --train-mode blocktt; SVD calib modes are
    #    applied inside full-FT scripts so they have no train_mode constraint.
    if calib_mode != "none" and not calib_mode.startswith("svd_") \
            and hasattr(args, "train_mode"):
        if args.train_mode != "blocktt":
            raise ValueError(
                "--calib-mode only valid with --train-mode blocktt "
                f"(got --train-mode={args.train_mode!r})"
            )
```

Also exempt SVD modes from the integer-rank-rejection rule (around line 126):

```python
    # 3. Integer --blocktt-rank rejected on the BTT calibrated path; float must be in (0, 1]
    if calib_mode != "none" and not calib_mode.startswith("svd_"):
        rank_raw = getattr(args, "blocktt_rank", "full")
        # ... (rest of the existing block unchanged)
```

Wrap the existing block in this `not calib_mode.startswith("svd_")` guard. The SVD path ignores `--blocktt-rank` entirely (Task 2's `build_decomposition_config` branch).

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m unittest tests.test_compress_integration_validate -v
```

Expected: PASS (the new two cases plus any pre-existing ones).

- [ ] **Step 5: Commit**

```bash
git add compress_integration.py tests/test_compress_integration_validate.py
git commit -m "$(cat <<'EOF'
compress: exempt svd_v2 calib modes from BTT-only validator rules

SVD calib modes run inside full-FT scripts and ignore --blocktt-rank,
so skip the train_mode==blocktt check and the integer-rank rejection.
BTT modes unchanged.

Co-Authored-By: Claude Opus 4 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add `apply_calibrated_svd` helper

**Files:**
- Modify: `compress_integration.py` (append after `apply_calibrated_btt`, around line 372).

- [ ] **Step 1: Write the failing test**

Create `tests/test_compress_integration_apply_svd.py`:

```python
"""Test apply_calibrated_svd end-to-end on a tiny model."""
import argparse
import os, sys
import unittest
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci
from compress.svd.svd_linear import SVDCompressedLinear


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # one transformer-like block with the standard target leaf names
        self.gate_proj = nn.Linear(16, 32)
        self.up_proj = nn.Linear(16, 32)
        self.down_proj = nn.Linear(32, 16)
        self.q_proj = nn.Linear(16, 16)
        self.k_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)
        self.o_proj = nn.Linear(16, 16)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock()])
        self.lm_head = nn.Linear(16, 32)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = torch.randn(input_ids.shape[0], 16, device=input_ids.device,
                        dtype=next(self.parameters()).dtype)
        b = self.layers[0]
        h = b.gate_proj(x) * b.up_proj(x)
        h = b.down_proj(h)
        logits = self.lm_head(h)
        loss = logits.float().sum() * 0.0  # deterministic 0-loss for backward calib
        return type("Out", (), {"loss": loss, "logits": logits})()


def _parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--train-mode", default="full")
    p.add_argument("--blocktt-rank", default="full")
    p.add_argument("--trainable-type", default="all")
    p.add_argument("--decomp-mode", default="square")
    p.add_argument("--train-position", default="small")
    p.add_argument("--s-merged-to", default=None)
    p.add_argument("--blocktt-factorize-by-head", action="store_true", default=True)
    ci.add_calibrated_btt_args(p, hyphen_style=True)
    return p.parse_args(argv)


class _ToyCalibLoader:
    """Yields one batch of input_ids; exercises forward pass for activation
    covariance collection."""
    def __init__(self, batch_size=2, seq_len=4):
        self.batches = [
            {"input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long)}
        ]
    def __iter__(self):
        return iter(self.batches)
    def __len__(self):
        return len(self.batches)


class TestApplyCalibratedSVD(unittest.TestCase):
    def test_apply_svd_v2_replaces_nn_linear_and_keeps_grads(self):
        model = TinyModel()
        args = _parse(["--calib-mode", "svd_v2", "--calib-source", "training_data",
                       "--compression-ratio", "0.5"])
        loader = _ToyCalibLoader()
        out = ci.apply_calibrated_svd(model, args, calib_loader=loader, device="cpu")

        # Every transformer-block linear should now be SVDCompressedLinear
        block = out.layers[0]
        for name in ("gate_proj", "up_proj", "down_proj", "q_proj",
                     "k_proj", "v_proj", "o_proj"):
            mod = getattr(block, name)
            self.assertIsInstance(mod, SVDCompressedLinear,
                                  f"{name} should be SVDCompressedLinear")
            # Both factors must be trainable for full-FT after compression
            self.assertTrue(mod.U_r.requires_grad,
                            f"{name}.U_r should require grad")
            self.assertTrue(mod.V_r.requires_grad,
                            f"{name}.V_r should require grad")

        # lm_head is skipped, remains an nn.Linear with grad enabled
        self.assertIsInstance(out.lm_head, nn.Linear)
        self.assertTrue(out.lm_head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m unittest tests.test_compress_integration_apply_svd -v
```

Expected: FAIL with `AttributeError: module 'compress_integration' has no attribute 'apply_calibrated_svd'`.

- [ ] **Step 3: Implement `apply_calibrated_svd`**

Add after `apply_calibrated_btt` in `compress_integration.py`:

```python
def apply_calibrated_svd(
    model, args, *, calib_loader, device: Optional[str] = None,
    hyphen_style: bool = True,
) -> nn.Module:
    """Apply an SVD-LLM-v2 compression in-place. Used by full-FT scripts that
    want to initialize weights on the low-rank manifold before running
    standard supervised fine-tuning. After compression, every
    SVDCompressedLinear's U_r/V_r factors are flipped to requires_grad=True
    so the optimizer trains the factored weights end-to-end."""
    cfg = build_decomposition_config(args, hyphen_style=hyphen_style, model=model)
    if not cfg.train_mode.startswith("svd_"):
        raise ValueError(
            f"apply_calibrated_svd called with non-SVD train_mode={cfg.train_mode!r}; "
            "use apply_calibrated_btt for BTT modes."
        )
    model = decompose_with_loader(
        model, cfg, calib_loader=calib_loader, device=device,
        return_trainability_stats=False,
    )
    # SVDCompressedLinear initialises U_r/V_r with requires_grad=False (it is
    # used by frozen-compress callers too). For full-FT after compression we
    # must flip them on explicitly — the model-level requires_grad_(True) call
    # inside decompose_with_loader does cascade through Parameters, but we
    # re-assert it here for clarity and as a guard against future changes.
    from compress.svd.svd_linear import SVDCompressedLinear
    for module in model.modules():
        if isinstance(module, SVDCompressedLinear):
            module.U_r.requires_grad = True
            module.V_r.requires_grad = True
            if module.bias is not None:
                module.bias.requires_grad = True
    return model
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m unittest tests.test_compress_integration_apply_svd -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add compress_integration.py tests/test_compress_integration_apply_svd.py
git commit -m "$(cat <<'EOF'
compress: add apply_calibrated_svd helper for full-FT scripts

Mirrors apply_calibrated_btt but for svd_v2 / svd_v2_combined modes.
Calls decompose_with_loader, then flips U_r/V_r to requires_grad=True
so the full-FT optimizer trains the factored weights end-to-end.

Co-Authored-By: Claude Opus 4 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add `materialize_svd_to_linear` helper

**Files:**
- Modify: `compress_integration.py` (append after `materialize_calibrated_btt_to_linear`, around line 425).

- [ ] **Step 1: Write the failing test**

Create `tests/test_compress_integration_materialize_svd.py`:

```python
"""Test materialize_svd_to_linear converts SVDCompressedLinear back to nn.Linear
with weights that reproduce the factored forward."""
import os, sys
import unittest
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compress_integration as ci
from compress.svd.svd_linear import SVDCompressedLinear


def _make_svd_linear(d_in=8, d_out=16, rank=4, dtype=torch.float32):
    U = torch.randn(rank, d_out, dtype=dtype)
    V = torch.randn(d_in, rank, dtype=dtype)
    return SVDCompressedLinear(U, V, bias=torch.randn(d_out, dtype=dtype))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _make_svd_linear()
        self.head = nn.Linear(16, 16)


class TestMaterializeSVD(unittest.TestCase):
    def test_materialize_preserves_forward(self):
        model = TinyModel()
        x = torch.randn(2, 8)
        with torch.no_grad():
            y_before = model.layer(x)

        ci.materialize_svd_to_linear(model)

        self.assertIsInstance(model.layer, nn.Linear)
        self.assertNotIsInstance(model.layer, SVDCompressedLinear)
        self.assertEqual(model.layer.weight.shape, (16, 8))
        self.assertIsNotNone(model.layer.bias)

        with torch.no_grad():
            y_after = model.layer(x)
        self.assertTrue(torch.allclose(y_before, y_after, atol=1e-5),
                        f"max diff {(y_before - y_after).abs().max().item()}")

    def test_materialize_leaves_unrelated_linear_alone(self):
        model = TinyModel()
        head_weight_before = model.head.weight.detach().clone()
        ci.materialize_svd_to_linear(model)
        self.assertTrue(torch.equal(model.head.weight, head_weight_before))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m unittest tests.test_compress_integration_materialize_svd -v
```

Expected: FAIL with `AttributeError: module 'compress_integration' has no attribute 'materialize_svd_to_linear'`.

- [ ] **Step 3: Implement `materialize_svd_to_linear`**

Add after `materialize_calibrated_btt_to_linear` in `compress_integration.py`:

```python
@torch.no_grad()
def materialize_svd_to_linear(model: nn.Module) -> nn.Module:
    """In-place: replace every SVDCompressedLinear in `model` with an
    nn.Linear whose weight is the dense product `(V_r @ U_r).T`. Mirrors
    `materialize_calibrated_btt_to_linear` but for the SVD calib path.

    The factored forward in SVDCompressedLinear is `(x @ V_r) @ U_r + b`,
    which equals `x @ (V_r @ U_r) + b`. The standard nn.Linear computes
    `x @ W.T + b`, so the materialized weight is `(V_r @ U_r).T`.

    Trade-off: the materialized checkpoint is dense (same on-disk size as
    the uncompressed base model). This is intentional: it lets eval_math.sh
    load the model as a vanilla HF checkpoint with no custom modules. The
    *training* still happened on the low-rank manifold, which is the
    experimental quantity of interest.
    """
    from compress.svd.svd_linear import SVDCompressedLinear

    replacements = [(n, m) for n, m in model.named_modules()
                    if isinstance(m, SVDCompressedLinear)]
    for name, svd in replacements:
        # V_r: (d_in, rank), U_r: (rank, d_out)
        # Dense weight for nn.Linear is (d_out, d_in) = (V_r @ U_r).T
        dense_weight = (svd.V_r @ svd.U_r).t().contiguous()
        d_out, d_in = dense_weight.shape
        linear = nn.Linear(
            d_in,
            d_out,
            bias=svd.bias is not None,
            device=dense_weight.device,
            dtype=dense_weight.dtype,
        )
        linear.weight.data.copy_(dense_weight)
        if svd.bias is not None:
            linear.bias.data.copy_(svd.bias.data)

        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], linear)
    return model
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m unittest tests.test_compress_integration_materialize_svd -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add compress_integration.py tests/test_compress_integration_materialize_svd.py
git commit -m "$(cat <<'EOF'
compress: add materialize_svd_to_linear for vanilla-HF checkpoints

After SVD full-FT, every SVDCompressedLinear is converted to an
nn.Linear holding the dense product (V_r @ U_r).T so eval_math.sh
can load the saved checkpoint without custom module registration.
Training stayed on the low-rank manifold; only the save step
materializes dense.

Co-Authored-By: Claude Opus 4 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Wire compression into `ref/LIFT/src/finetune_sft.py`

**Files:**
- Modify: `ref/LIFT/src/finetune_sft.py` (imports near line 11; arg registration near line 540; compression hook between dataset construction and optimizer; materialize hook before `save_hf_format`).

- [ ] **Step 1: Add path fix + import**

At the very top of `ref/LIFT/src/finetune_sft.py`, just after the existing two `sys.path.insert` blocks (line 10), add the third block that pulls in the repo-root-level `compress_integration.py`:

```python
# Ensure repo root (with compress_integration.py and src/compress) is importable
_LIFT_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
if _LIFT_REPO_ROOT not in sys.path:
    sys.path.insert(0, _LIFT_REPO_ROOT)
_LIFT_SRC = os.path.join(_LIFT_REPO_ROOT, "src")
if os.path.isdir(_LIFT_SRC) and _LIFT_SRC not in sys.path:
    sys.path.insert(0, _LIFT_SRC)
```

Then below the existing `from utils.data_utils import ...` line (around line 52), add:

```python
from compress_integration import (
    add_calibrated_btt_args,
    validate_calibrated_btt_args,
    build_calib_loader,
    apply_calibrated_svd,
    materialize_svd_to_linear,
)
```

- [ ] **Step 2: Register calib args on the parser**

Inside `parse_args()` (defined around line 268), find the very last `parser.add_argument(...)` call (right before `args = parser.parse_args()` near line 540) and append:

```python
    add_calibrated_btt_args(parser, hyphen_style=False)
```

Use `hyphen_style=False` because LIFT's other args use underscored names (e.g. `--model_name_or_path`). After parsing, the args namespace will expose `args.calib_mode`, `args.calib_source`, `args.compression_ratio`, etc.

Then after `args = parser.parse_args()` and before `return args`, add:

```python
    validate_calibrated_btt_args(args, argv=sys.argv[1:], hyphen_style=False)
```

- [ ] **Step 3: Add compression hook in `main()`**

In `main()`, find the block (around line 685) where `train_dataloader` is constructed:

```python
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
    )
```

Immediately AFTER this DataLoader construction (and BEFORE the `optimizer_grouped_parameters = ...` line near 695), insert:

```python
    # --- SVD compression hook (svd_v2 / svd_v2_combined) ---
    if getattr(args, "calib_mode", "none").startswith("svd_"):
        _calib_collate = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        _calib_loader = build_calib_loader(
            args,
            tokenizer=tokenizer,
            training_dataset=train_dataset,
            training_collate_fn=_calib_collate,
            hyphen_style=False,
        )
        print(f"[svd-compress] applying calib_mode={args.calib_mode} "
              f"compression_ratio={args.compression_ratio}")
        model = apply_calibrated_svd(
            model, args, calib_loader=_calib_loader,
            device=str(accelerator.device), hyphen_style=False,
        )
        # Re-count trainable params after compression for SysMon (constructed later)
        _trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _total_after = sum(p.numel() for p in model.parameters())
        print(f"[svd-compress] post-compression params: "
              f"trainable={_trainable_after}, total={_total_after}")
```

- [ ] **Step 4: Add materialize hook before save**

In `main()`, find the save block (around line 934-935):

```python
        last_model = accelerator.unwrap_model(model)
        save_hf_format(last_model, tokenizer, args, sub_folder="last")
```

Immediately BEFORE `save_hf_format(last_model, ...)`, insert:

```python
        if getattr(args, "calib_mode", "none").startswith("svd_"):
            materialize_svd_to_linear(last_model)
            print_rank_0("[svd-compress] materialized SVDCompressedLinear -> nn.Linear before save",
                         args.global_rank)
```

Also, in the `best_model` branch (line 938 onwards), insert the same materialize call before `save_hf_format(best_model, ...)`:

```python
        if best_model is not None:
            if getattr(args, "calib_mode", "none").startswith("svd_"):
                materialize_svd_to_linear(best_model)
                print_rank_0("[svd-compress] materialized best-model SVDCompressedLinear -> nn.Linear before save",
                             args.global_rank)
            save_hf_format(best_model, tokenizer, args, sub_folder="best")
```

- [ ] **Step 5: Syntax check**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m py_compile ref/LIFT/src/finetune_sft.py compress_integration.py
```

Expected: no output (success).

- [ ] **Step 6: Smoke test parser**

```bash
cd /home/yequan/Project/lora/lora-without-regret/ref/LIFT
python -c "
import sys
sys.argv = ['finetune_sft.py',
            '--model_name_or_path', 'meta-llama/Llama-3.2-1B',
            '--data_path', '/tmp/dummy.json',
            '--calib_mode', 'svd_v2',
            '--compression_ratio', '0.5',
            '--output_dir', '/tmp/out']
import src.finetune_sft as m
args = m.parse_args()
print('calib_mode =', args.calib_mode)
print('compression_ratio =', args.compression_ratio)
print('calib_source =', args.calib_source)
"
```

Expected stdout:
```
calib_mode = svd_v2
compression_ratio = 0.5
calib_source = c4
```

- [ ] **Step 7: Commit**

```bash
git add ref/LIFT/src/finetune_sft.py
git commit -m "$(cat <<'EOF'
lift: wire svd_v2 / svd_v2_combined into finetune_sft.py

Adds path-fixup imports, registers calib args via
add_calibrated_btt_args(hyphen_style=False), inserts apply_calibrated_svd
between train_dataset construction and the optimizer, and materializes
SVDCompressedLinear -> nn.Linear before save_hf_format so eval consumes
a vanilla HF checkpoint. BTT path and non-calib full-FT runs untouched.

Co-Authored-By: Claude Opus 4 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Create Group A shell scripts (baselines)

**Files:**
- Create: `ref/LIFT/bash_scripts/finetune_math_full_a1_llama32_1b.sh`
- Create: `ref/LIFT/bash_scripts/finetune_math_full_a2_llama32_3b.sh`

- [ ] **Step 1: Write A1 (Llama-3.2-1B baseline)**

Create `ref/LIFT/bash_scripts/finetune_math_full_a1_llama32_1b.sh`:

```bash
#!/bin/bash
# Group A1: Llama-3.2-1B baseline full FT on MATH-10K, eval gsm8k only.
# Launch with: DEVICE=N bash bash_scripts/finetune_math_full_a1_llama32_1b.sh

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/ryan/huggingface_cache}"

if [ -n "${DEVICE:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$DEVICE"
fi

SRC_DIR="${SRC_DIR:-/home/yequan/Project/lora/lora-without-regret/ref/LIFT}"
DATA_DIR="${DATA_DIR:-/data/ruijiezhang/llm-adapter_bp/LLM-Adapters}"
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-/data/yequan/compress_train/sft}"

MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"
lr="${lr:-1e-4}"
seed="${seed:-43}"
model_tag="${MODEL##*/}"
wandb_project="${wandb_project:-compress_sft_math_llama3.2}"
wandb_run_id="${wandb_run_id:-$(python -c 'import wandb; print(wandb.util.generate_id())')}"

export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/math/${MODEL}/full-lr_${lr}-seed_${seed}}"
run_name="${run_name:-$(basename "$OUTPUT")}"
mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_sft.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BS:-2} \
    --per_device_eval_batch_size 16 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps ${GRAD_ACC_STEPS:-8} \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --instruction_type single \
    ${GRAD_CKPT:+--gradient_checkpointing} \
    --load_last_model \
    --data_path ${DATA_DIR}/ft-training_set/math_10k.json \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

EVAL_DATASETS="gsm8k" bash ./bash_scripts/eval_math.sh \
    CKPT="$OUTPUT/last" \
    base_model="${MODEL}" \
    wandb_project="${wandb_project}" \
    wandb_run_name="${run_name}" \
    wandb_run_id="${wandb_run_id}"
```

- [ ] **Step 2: Write A2 (Llama-3.2-3B baseline)**

Copy A1 to `finetune_math_full_a2_llama32_3b.sh` and change the `MODEL` line to:

```bash
MODEL="${MODEL:-meta-llama/Llama-3.2-3B}"
```

All other content identical to A1.

- [ ] **Step 3: Make executable + shellcheck syntax**

```bash
cd /home/yequan/Project/lora/lora-without-regret
chmod +x ref/LIFT/bash_scripts/finetune_math_full_a1_llama32_1b.sh
chmod +x ref/LIFT/bash_scripts/finetune_math_full_a2_llama32_3b.sh
bash -n ref/LIFT/bash_scripts/finetune_math_full_a1_llama32_1b.sh
bash -n ref/LIFT/bash_scripts/finetune_math_full_a2_llama32_3b.sh
echo $?
```

Expected: no output from `bash -n`, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add ref/LIFT/bash_scripts/finetune_math_full_a1_llama32_1b.sh ref/LIFT/bash_scripts/finetune_math_full_a2_llama32_3b.sh
git commit -m "$(cat <<'EOF'
lift: add Group A (1B/3B baseline) MATH-10K launchers

Two no-compression full-FT scripts targeting meta-llama/Llama-3.2-1B
and Llama-3.2-3B at lr=1e-4, 3 epochs, seed=43. Output to
/data/yequan/compress_train/sft/math/<MODEL>/full-lr_1e-4-seed_43.
Eval limited to gsm8k via EVAL_DATASETS env. Wandb project
compress_sft_math_llama3.2.

Co-Authored-By: Claude Opus 4 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Create Group B shell scripts (3B → 2B compressed)

**Files:**
- Create: `ref/LIFT/bash_scripts/finetune_math_full_b1_svd_v2_2b.sh`
- Create: `ref/LIFT/bash_scripts/finetune_math_full_b2_svd_v2_combined_2b.sh`

- [ ] **Step 1: Write B1 (svd_v2, target 2B)**

Create `ref/LIFT/bash_scripts/finetune_math_full_b1_svd_v2_2b.sh`:

```bash
#!/bin/bash
# Group B1: Llama-3.2-3B -> ~2B via svd_llm_v2, then full FT on MATH-10K, eval gsm8k.
# Launch with: DEVICE=N bash bash_scripts/finetune_math_full_b1_svd_v2_2b.sh

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/ryan/huggingface_cache}"

if [ -n "${DEVICE:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$DEVICE"
fi

SRC_DIR="${SRC_DIR:-/home/yequan/Project/lora/lora-without-regret/ref/LIFT}"
DATA_DIR="${DATA_DIR:-/data/ruijiezhang/llm-adapter_bp/LLM-Adapters}"
OUTPUT_SRC_DIR="${OUTPUT_SRC_DIR:-/data/yequan/compress_train/sft}"

MODEL="${MODEL:-meta-llama/Llama-3.2-3B}"
lr="${lr:-1e-4}"
seed="${seed:-43}"
calib_mode="${calib_mode:-svd_v2}"
compression_ratio="${compression_ratio:-0.589}"
target_tag="${target_tag:-2B}"

calib_source="${calib_source:-training_data}"
calib_num_seqs="${calib_num_seqs:-128}"
calib_max_length="${calib_max_length:-2048}"
calib_batch_size="${calib_batch_size:-4}"
calib_seed="${calib_seed:-3}"

model_tag="${MODEL##*/}"
wandb_project="${wandb_project:-compress_sft_math_llama3.2}"
wandb_run_id="${wandb_run_id:-$(python -c 'import wandb; print(wandb.util.generate_id())')}"

export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo $MODEL

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/math/${MODEL}/full-calib_${calib_mode}-target_${target_tag}-lr_${lr}-seed_${seed}}"
run_name="${run_name:-$(basename "$OUTPUT")}"
mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_sft.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BS:-2} \
    --per_device_eval_batch_size 16 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps ${GRAD_ACC_STEPS:-8} \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --instruction_type single \
    ${GRAD_CKPT:+--gradient_checkpointing} \
    --load_last_model \
    --data_path ${DATA_DIR}/ft-training_set/math_10k.json \
    --calib_mode ${calib_mode} \
    --calib_source ${calib_source} \
    --calib_num_seqs ${calib_num_seqs} \
    --calib_max_length ${calib_max_length} \
    --calib_batch_size ${calib_batch_size} \
    --calib_seed ${calib_seed} \
    --compression_ratio ${compression_ratio} \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

EVAL_DATASETS="gsm8k" bash ./bash_scripts/eval_math.sh \
    CKPT="$OUTPUT/last" \
    base_model="${MODEL}" \
    wandb_project="${wandb_project}" \
    wandb_run_name="${run_name}" \
    wandb_run_id="${wandb_run_id}"
```

- [ ] **Step 2: Write B2 (svd_v2_combined, target 2B)**

Copy B1 to `finetune_math_full_b2_svd_v2_combined_2b.sh` and change the `calib_mode` default line:

```bash
calib_mode="${calib_mode:-svd_v2_combined}"
```

All other content identical to B1.

- [ ] **Step 3: Chmod + syntax check**

```bash
cd /home/yequan/Project/lora/lora-without-regret
chmod +x ref/LIFT/bash_scripts/finetune_math_full_b1_svd_v2_2b.sh
chmod +x ref/LIFT/bash_scripts/finetune_math_full_b2_svd_v2_combined_2b.sh
bash -n ref/LIFT/bash_scripts/finetune_math_full_b1_svd_v2_2b.sh
bash -n ref/LIFT/bash_scripts/finetune_math_full_b2_svd_v2_combined_2b.sh
echo $?
```

Expected: exit 0, no output.

- [ ] **Step 4: Commit**

```bash
git add ref/LIFT/bash_scripts/finetune_math_full_b1_svd_v2_2b.sh ref/LIFT/bash_scripts/finetune_math_full_b2_svd_v2_combined_2b.sh
git commit -m "$(cat <<'EOF'
lift: add Group B (3B->2B compressed) MATH-10K launchers

Two SVD-LLM-v2 launchers compressing Llama-3.2-3B to ~2B total params
(compression_ratio=0.589 on the compressible-linear subset) via
svd_v2 and svd_v2_combined, then full FT at lr=1e-4 for 3 epochs.
Training-data calibration (128 seqs, max_len 2048, batch 4).
Eval gsm8k only.

Co-Authored-By: Claude Opus 4 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Create Group C shell scripts (3B → 1B compressed)

**Files:**
- Create: `ref/LIFT/bash_scripts/finetune_math_full_c1_svd_v2_1b.sh`
- Create: `ref/LIFT/bash_scripts/finetune_math_full_c2_svd_v2_combined_1b.sh`

- [ ] **Step 1: Write C1 (svd_v2, target 1B)**

Copy `finetune_math_full_b1_svd_v2_2b.sh` to `finetune_math_full_c1_svd_v2_1b.sh` and change two defaults:

```bash
compression_ratio="${compression_ratio:-0.250}"
target_tag="${target_tag:-1B}"
```

(`calib_mode` stays `svd_v2`.) All other content identical to B1.

- [ ] **Step 2: Write C2 (svd_v2_combined, target 1B)**

Copy `finetune_math_full_b2_svd_v2_combined_2b.sh` to `finetune_math_full_c2_svd_v2_combined_1b.sh` and change:

```bash
compression_ratio="${compression_ratio:-0.250}"
target_tag="${target_tag:-1B}"
```

(`calib_mode` stays `svd_v2_combined`.) All other content identical to B2.

- [ ] **Step 3: Chmod + syntax check**

```bash
cd /home/yequan/Project/lora/lora-without-regret
chmod +x ref/LIFT/bash_scripts/finetune_math_full_c1_svd_v2_1b.sh
chmod +x ref/LIFT/bash_scripts/finetune_math_full_c2_svd_v2_combined_1b.sh
bash -n ref/LIFT/bash_scripts/finetune_math_full_c1_svd_v2_1b.sh
bash -n ref/LIFT/bash_scripts/finetune_math_full_c2_svd_v2_combined_1b.sh
echo $?
```

Expected: exit 0, no output.

- [ ] **Step 4: Commit**

```bash
git add ref/LIFT/bash_scripts/finetune_math_full_c1_svd_v2_1b.sh ref/LIFT/bash_scripts/finetune_math_full_c2_svd_v2_combined_1b.sh
git commit -m "$(cat <<'EOF'
lift: add Group C (3B->1B compressed) MATH-10K launchers

Two SVD-LLM-v2 launchers compressing Llama-3.2-3B to ~1B total params
(compression_ratio=0.250) via svd_v2 and svd_v2_combined, then full
FT at lr=1e-4 for 3 epochs. Same calibration settings as Group B.
Eval gsm8k only.

Co-Authored-By: Claude Opus 4 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Smoke test the uncompressed full-FT path (regression guard)

This task verifies that adding the calib-args plumbing to `finetune_sft.py` did NOT break the no-compression code path.

**Files:** None. Runs an existing script with `MAX_STEPS` override.

- [ ] **Step 1: Pick an idle GPU**

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
```

Pick the GPU with the most free memory (call it `$DEV`).

- [ ] **Step 2: Run A1 with 5 optimizer steps**

```bash
cd /home/yequan/Project/lora/lora-without-regret
DEVICE=$DEV \
OUTPUT=/tmp/svd_smoke/a1_uncompressed \
WANDB_MODE=disabled \
PER_DEVICE_TRAIN_BS=2 GRAD_ACC_STEPS=1 \
timeout 600 bash ref/LIFT/bash_scripts/finetune_math_full_a1_llama32_1b.sh \
    2>&1 | tee /tmp/svd_smoke/a1.log

# Inject max_steps via a quick edit hack: kill after step 5 by setting a low
# epoch count or use MAX_STEPS env (note: only some LIFT scripts honor this;
# if a1 does not, append --max_steps=5 to the accelerate command manually for
# the smoke test).
```

If the A1 script does not honor `MAX_STEPS`, prefer this direct invocation:

```bash
cd /home/yequan/Project/lora/lora-without-regret/ref/LIFT
mkdir -p /tmp/svd_smoke/a1_uncompressed
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$DEV HF_HOME=/data/ryan/huggingface_cache \
timeout 600 accelerate launch --num_machines 1 --num_processes 1 --mixed_precision bf16 \
    src/finetune_sft.py \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 16 \
    --logging_steps 1 --max_seq_len 512 \
    --learning_rate 1e-4 --weight_decay 0. \
    --num_train_epochs 1 --max_steps 5 \
    --gradient_accumulation_steps 1 --lr_scheduler_type linear --num_warmup_steps 0 \
    --seed 43 --instruction_type single --load_last_model \
    --data_path /data/ruijiezhang/llm-adapter_bp/LLM-Adapters/ft-training_set/math_10k.json \
    --output_dir /tmp/svd_smoke/a1_uncompressed \
    --no_wandb 2>&1 | tee /tmp/svd_smoke/a1.log
```

- [ ] **Step 3: Verify it ran**

```bash
ls /tmp/svd_smoke/a1_uncompressed/last/
grep -E 'Saved last-step|Error|Traceback' /tmp/svd_smoke/a1.log | tail -20
```

Expected: `last/` contains `config.json` and a weights file (`pytorch_model.bin` or `model.safetensors`). The log shows `Saved last-step checkpoint to /tmp/svd_smoke/a1_uncompressed/last`. No `Traceback`.

- [ ] **Step 4: Clean up**

```bash
rm -rf /tmp/svd_smoke/a1_uncompressed
```

No commit (this task is a smoke test, not code).

---

## Task 11: Smoke test the SVD-v2 compressed path end-to-end

**Files:** None. Runs the B1 script via a direct invocation with `--max_steps`.

- [ ] **Step 1: Pick an idle GPU**

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
```

- [ ] **Step 2: Run B1 with 5 optimizer steps**

```bash
cd /home/yequan/Project/lora/lora-without-regret/ref/LIFT
mkdir -p /tmp/svd_smoke/b1_svd_v2
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$DEV HF_HOME=/data/ryan/huggingface_cache \
timeout 1800 accelerate launch --num_machines 1 --num_processes 1 --mixed_precision bf16 \
    src/finetune_sft.py \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 16 \
    --logging_steps 1 --max_seq_len 512 \
    --learning_rate 1e-4 --weight_decay 0. \
    --num_train_epochs 1 --max_steps 5 \
    --gradient_accumulation_steps 1 --lr_scheduler_type linear --num_warmup_steps 0 \
    --seed 43 --instruction_type single --load_last_model \
    --data_path /data/ruijiezhang/llm-adapter_bp/LLM-Adapters/ft-training_set/math_10k.json \
    --calib_mode svd_v2 --calib_source training_data \
    --calib_num_seqs 16 --calib_max_length 512 --calib_batch_size 2 --calib_seed 3 \
    --compression_ratio 0.589 \
    --output_dir /tmp/svd_smoke/b1_svd_v2 \
    --no_wandb 2>&1 | tee /tmp/svd_smoke/b1.log
```

Note: `calib_num_seqs=16`, `calib_max_length=512` for smoke speed.

- [ ] **Step 3: Verify the compression actually happened**

```bash
grep -E '\[svd-compress\]|SVD-LLM-V2|compressing' /tmp/svd_smoke/b1.log | head -20
grep -E 'Saved last-step|Traceback' /tmp/svd_smoke/b1.log | tail -10
ls /tmp/svd_smoke/b1_svd_v2/last/
```

Expected: log shows
```
[svd-compress] applying calib_mode=svd_v2 compression_ratio=0.589
SVD-LLM-V2: compressing N layers, target ... params (58%)
... lots of per-layer "name: (d_out,d_in) -> rank N, params ..." lines ...
[svd-compress] post-compression params: trainable=... total=...
[svd-compress] materialized SVDCompressedLinear -> nn.Linear before save
Saved last-step checkpoint to /tmp/svd_smoke/b1_svd_v2/last
```

`last/` should contain `config.json` and a weights file. No `Traceback`.

- [ ] **Step 4: Verify the saved checkpoint loads as a vanilla Llama**

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoConfig
import torch
m = AutoModelForCausalLM.from_pretrained('/tmp/svd_smoke/b1_svd_v2/last',
                                         dtype=torch.bfloat16)
print('loaded model:', type(m).__name__)
print('first linear weight shape:', m.model.layers[0].self_attn.q_proj.weight.shape)
print('total params:', sum(p.numel() for p in m.parameters()))
"
```

Expected: loads without error. Linear shapes match the uncompressed Llama-3.2-3B (because we materialized dense). Total params ~3.21B.

- [ ] **Step 5: Smoke-eval gsm8k on 5 examples (cheap sanity)**

```bash
cd /home/yequan/Project/lora/lora-without-regret/ref/LIFT
# Limit eval data: gsm8k test.json has ~1300 examples; sed off the first 5
mkdir -p /tmp/svd_smoke/gsm8k_smoke
head -7 /data/ruijiezhang/llm-adapter_bp/LLM-Adapters/dataset/gsm8k/test.json > /tmp/svd_smoke/gsm8k_smoke/test.json 2>/dev/null || true
# If gsm8k test.json is a single JSON array (not JSONL), we just run on all
# of it with --per_device_eval_batch_size 1 + a per-script step cap (none
# exists today; this is informational). Skip the head trick and just run.

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$DEV \
EVAL_DATASETS="gsm8k" \
timeout 1800 bash ./bash_scripts/eval_math.sh \
    CKPT=/tmp/svd_smoke/b1_svd_v2/last \
    base_model=meta-llama/Llama-3.2-3B \
    2>&1 | tee /tmp/svd_smoke/b1_eval.log
```

Expected: eval runs, prints an `accuracy` or final metric line, and writes `/tmp/svd_smoke/b1_svd_v2/last/math/gsm8k/`. We do NOT require any specific accuracy here — just no crashes.

If you do not want to wait for the full gsm8k pass (~10-30 min on a 3B), it is acceptable to skip Step 5 and rely on Step 4 (model loads) as the eval-path proof.

- [ ] **Step 6: Clean up**

```bash
rm -rf /tmp/svd_smoke
```

No commit.

---

## Task 12: Smoke test the SVD-v2-combined path

**Files:** None. Runs the B2 script logic via a direct invocation.

Same shape as Task 11, but with `--calib_mode svd_v2_combined`. The combined mode runs an extra backward pass through the calibration loader, so memory pressure is higher; check it does not OOM on 3B.

- [ ] **Step 1: Run B2 with 5 optimizer steps**

```bash
cd /home/yequan/Project/lora/lora-without-regret/ref/LIFT
mkdir -p /tmp/svd_smoke_combined
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$DEV HF_HOME=/data/ryan/huggingface_cache \
timeout 1800 accelerate launch --num_machines 1 --num_processes 1 --mixed_precision bf16 \
    src/finetune_sft.py \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 16 \
    --logging_steps 1 --max_seq_len 512 \
    --learning_rate 1e-4 --weight_decay 0. \
    --num_train_epochs 1 --max_steps 5 \
    --gradient_accumulation_steps 1 --lr_scheduler_type linear --num_warmup_steps 0 \
    --seed 43 --instruction_type single --load_last_model \
    --data_path /data/ruijiezhang/llm-adapter_bp/LLM-Adapters/ft-training_set/math_10k.json \
    --calib_mode svd_v2_combined --calib_source training_data \
    --calib_num_seqs 16 --calib_max_length 512 --calib_batch_size 2 --calib_seed 3 \
    --compression_ratio 0.589 \
    --output_dir /tmp/svd_smoke_combined \
    --no_wandb 2>&1 | tee /tmp/svd_smoke_combined.log
```

- [ ] **Step 2: Verify**

```bash
grep -E '\[svd-compress\]|svd_llm_v2_combined|backward' /tmp/svd_smoke_combined.log | head -20
grep -E 'Saved last-step|Traceback|OutOfMemory' /tmp/svd_smoke_combined.log | tail -10
ls /tmp/svd_smoke_combined/last/
```

Expected: log shows the combined compression path, includes backward-covariance collection messages, ends with the saved-checkpoint line. No OOM and no Traceback.

- [ ] **Step 3: Clean up**

```bash
rm -rf /tmp/svd_smoke_combined /tmp/svd_smoke_combined.log
```

No commit.

---

## Task 13: Run the full unit-test suite

**Files:** None.

- [ ] **Step 1: Run all compress_integration tests**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m unittest discover -s tests -p 'test_compress_integration_*.py' -v
```

Expected: all tests pass.

- [ ] **Step 2: Run the existing repo regression suite (cheap subset)**

```bash
python -m unittest tests.test_btt_pipeline_compat -v
python -m unittest tests.test_svd_pipeline_compat -v
```

Expected: all pass (these are the project's compatibility shims; our changes should not break them).

No commit (this task gates the launch).

---

## Self-review checklist (for the executing agent)

After Task 13, before considering the plan complete, verify:

- [ ] All six shell scripts exist under `ref/LIFT/bash_scripts/`, are executable, and pass `bash -n`.
- [ ] `compress_integration.py` has new helpers `apply_calibrated_svd` and `materialize_svd_to_linear` reachable from `from compress_integration import ...`.
- [ ] `ref/LIFT/src/finetune_sft.py` registers calib args, calls `apply_calibrated_svd` before the optimizer, and calls `materialize_svd_to_linear` before `save_hf_format`.
- [ ] `git log --oneline -10` shows commits for each Task 1–9.
- [ ] Smoke tests in Tasks 10–12 all completed without errors.
- [ ] `python -m unittest discover -s tests -p 'test_compress_integration_*.py'` passes.

## What the user will do after this plan finishes

Manually launch the six scripts on chosen GPUs, e.g.:
```bash
DEVICE=0 bash ref/LIFT/bash_scripts/finetune_math_full_a1_llama32_1b.sh &
DEVICE=1 bash ref/LIFT/bash_scripts/finetune_math_full_a2_llama32_3b.sh &
DEVICE=2 bash ref/LIFT/bash_scripts/finetune_math_full_b1_svd_v2_2b.sh &
DEVICE=3 bash ref/LIFT/bash_scripts/finetune_math_full_b2_svd_v2_combined_2b.sh &
DEVICE=4 bash ref/LIFT/bash_scripts/finetune_math_full_c1_svd_v2_1b.sh &
DEVICE=5 bash ref/LIFT/bash_scripts/finetune_math_full_c2_svd_v2_combined_1b.sh &
```

(The plan does not include launching the real runs — only preparing the scripts and proving via smoke tests that they work.)
