# Noise-Scale CBS Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `NoiseScaleProbe` utility that estimates McCandlish `B_simple = tr(Σ) / ‖G‖²` on trainable parameters during `commonsense_170k` finetune, wire it into the LoRA and BlockTT training scripts behind a CLI flag, and provide standalone launchers and a comparison plot script.

**Architecture:** Standalone module under `ref/LIFT/src/utils/noise_scale.py` exposes a `NoiseScaleProbe` class. Both `finetune_lora.py` and `finetune_blocktt.py` add three CLI flags and four call-sites (construct, call inside training loop, close after loop). The probe runs every N optimizer steps (default 200) by drawing 16 examples from a private dataloader cursor and computing per-example bf16 → fp32 backward passes; statistics are computed online via the variance identity (no flat vectors materialized). Results land in `<output_dir>/noise_scale.jsonl` and (best-effort) wandb. A standalone `plot_noise_scale.py` produces a 5-subplot comparison figure across runs.

**Tech Stack:** PyTorch, HuggingFace Accelerate, transformers, pandas (plot script), matplotlib (plot script), wandb (optional sink), Python 3.10+, `unittest`.

**Spec:** [`docs/superpowers/specs/2026-05-02-noise-scale-cbs-design.md`](../specs/2026-05-02-noise-scale-cbs-design.md)

---

## File Structure

**New files:**

| Path | Responsibility |
|---|---|
| `ref/LIFT/src/utils/noise_scale.py` | `NoiseScaleProbe` class: estimator, per-example backward, JSONL/wandb sinks |
| `ref/LIFT/src/tests/test_noise_scale_estimator.py` | Pure-math unit tests for the estimator helpers |
| `ref/LIFT/src/tests/test_noise_scale_probe_compat.py` | Integration test: probe is non-destructive on a tiny model |
| `ref/LIFT/src/tests/test_noise_scale_cli.py` | argparse smoke test for the new flags |
| `ref/LIFT/scripts/plot_noise_scale.py` | Standalone multi-run comparison plot script |
| `ref/LIFT/bash_scripts/finetune_commonsense_lora_cbs.sh` | Launcher: LoRA, 1 epoch, probing on |
| `ref/LIFT/bash_scripts/finetune_commonsense_blocktt_cbs.sh` | Launcher: BlockTT (fura defaults), 1 epoch, probing on |

**Modified files:**

| Path | Change |
|---|---|
| `ref/LIFT/src/finetune_lora.py` | Add 3 CLI flags + 3 small code blocks (construct, call, close) |
| `ref/LIFT/src/finetune_blocktt.py` | Same four edits at structurally analogous points |

**Unchanged (explicitly):** `ref/LIFT/bash_scripts/finetune_commonsense_lora.sh`, `ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh`.

---

## Test conventions used by this repo

The existing `ref/LIFT/src/tests/test_finetune_svd.py` shows the convention:

- Tests live under `ref/LIFT/src/tests/`.
- Tests import siblings from `ref/LIFT/src/` by inserting `_LIFT_SRC` (i.e. `ref/LIFT/src/`) into `sys.path` at the top of the file.
- Tests are run from the repo root with `python -m unittest ref.LIFT.src.tests.<module>` **or** from `ref/LIFT/src/` directly with `python -m unittest tests.<module>`.

For consistency, all new tests in this plan follow the same `sys.path`-injection pattern. The exact run command is given in each task.

---

## Task 1: Estimator helpers — pure functions

**Files:**
- Create: `ref/LIFT/src/utils/noise_scale.py` (initial scaffold + helpers only)
- Test: `ref/LIFT/src/tests/test_noise_scale_estimator.py`

This task implements the math first, with no model and no I/O. The helpers we build here are called by the probe's `maybe_run` method in later tasks.

- [ ] **Step 1.1: Write failing tests for `compute_noise_scale_stats`**

Create `ref/LIFT/src/tests/test_noise_scale_estimator.py`:

```python
"""Pure-math unit tests for noise_scale estimator helpers."""
import math
import os
import sys
import unittest

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_LIFT_SRC = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _LIFT_SRC not in sys.path:
    sys.path.insert(0, _LIFT_SRC)

from utils.noise_scale import compute_noise_scale_stats


class TestComputeNoiseScaleStats(unittest.TestCase):
    """The estimator implements McCandlish et al. Appendix A.1.

    Inputs to compute_noise_scale_stats:
      - sum_g_sq_per_example: float = sum_i ||g_i||^2          (scalar)
      - sum_g_norm_sq:        float = ||sum_i g_i||^2           (scalar)
      - batch_size:           int   = B (number of per-example grads)

    Outputs (dict):
      - tr_sigma:        (1/(B-1)) * (sum_g_sq_per_example - sum_g_norm_sq / B)
      - g_norm_sq:       sum_g_norm_sq / B^2 - tr_sigma / B
      - g_norm_sq_naive: sum_g_norm_sq / B^2
      - b_simple:        tr_sigma / max(g_norm_sq, 1e-12)
      - b_simple_valid:  bool, True iff g_norm_sq > 0
    """

    def test_identical_gradients_zero_variance(self):
        # All g_i identical => tr_sigma == 0, b_simple == 0.
        B = 16
        g = torch.tensor([1.0, 2.0, 3.0])
        g_norm_sq_single = float((g * g).sum())
        sum_g_sq_per_example = B * g_norm_sq_single
        sum_g = B * g
        sum_g_norm_sq = float((sum_g * sum_g).sum())

        stats = compute_noise_scale_stats(
            sum_g_sq_per_example=sum_g_sq_per_example,
            sum_g_norm_sq=sum_g_norm_sq,
            batch_size=B,
        )
        self.assertAlmostEqual(stats["tr_sigma"], 0.0, places=5)
        self.assertAlmostEqual(stats["g_norm_sq"], g_norm_sq_single, places=5)
        self.assertAlmostEqual(stats["g_norm_sq_naive"], g_norm_sq_single, places=5)
        self.assertAlmostEqual(stats["b_simple"], 0.0, places=5)
        self.assertTrue(stats["b_simple_valid"])

    def test_gaussian_recovery(self):
        # Sample g_i = G + eps_i, eps_i ~ N(0, sigma^2 I_d).
        # Verify estimator recovers tr(Sigma) ~= d * sigma^2 and ||G||^2.
        torch.manual_seed(0)
        B, d = 16, 64
        sigma = 0.5
        G = torch.randn(d) * 2.0
        true_tr_sigma = d * sigma * sigma
        true_g_norm_sq = float((G * G).sum())

        ests_tr, ests_gnsq = [], []
        for _ in range(1000):
            eps = torch.randn(B, d) * sigma
            g = G.unsqueeze(0) + eps  # [B, d]
            sum_g_sq_per_example = float((g * g).sum())
            sum_g = g.sum(dim=0)
            sum_g_norm_sq = float((sum_g * sum_g).sum())
            stats = compute_noise_scale_stats(
                sum_g_sq_per_example=sum_g_sq_per_example,
                sum_g_norm_sq=sum_g_norm_sq,
                batch_size=B,
            )
            ests_tr.append(stats["tr_sigma"])
            ests_gnsq.append(stats["g_norm_sq"])

        mean_tr = sum(ests_tr) / len(ests_tr)
        mean_gnsq = sum(ests_gnsq) / len(ests_gnsq)
        # Loose tolerances: 1000 trials, but variance is non-trivial.
        self.assertLess(abs(mean_tr - true_tr_sigma) / true_tr_sigma, 0.05)
        self.assertLess(abs(mean_gnsq - true_g_norm_sq) / true_g_norm_sq, 0.05)

    def test_negative_g_norm_sq_flagged(self):
        # Construct a case where sum_g_norm_sq / B^2 < tr_sigma / B
        # => g_norm_sq < 0; b_simple_valid must be False; b_simple finite (clipped).
        B = 16
        # Make grads have huge variance and mean close to zero.
        sum_g_sq_per_example = 1000.0  # high per-example energy
        sum_g_norm_sq = 0.0001          # but mean grad almost zero
        stats = compute_noise_scale_stats(
            sum_g_sq_per_example=sum_g_sq_per_example,
            sum_g_norm_sq=sum_g_norm_sq,
            batch_size=B,
        )
        self.assertGreater(stats["tr_sigma"], 0.0)
        self.assertLess(stats["g_norm_sq"], 0.0)
        self.assertFalse(stats["b_simple_valid"])
        self.assertTrue(math.isfinite(stats["b_simple"]))
        # Naive should still be the raw mean energy.
        self.assertAlmostEqual(stats["g_norm_sq_naive"], sum_g_norm_sq / (B * B), places=8)

    def test_batch_size_one_rejected(self):
        with self.assertRaises(ValueError):
            compute_noise_scale_stats(
                sum_g_sq_per_example=1.0,
                sum_g_norm_sq=1.0,
                batch_size=1,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 1.2: Run tests, verify they fail with import error**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m unittest ref.LIFT.src.tests.test_noise_scale_estimator -v
```

Expected: `ModuleNotFoundError: No module named 'utils.noise_scale'` or `ImportError: cannot import name 'compute_noise_scale_stats'`.

- [ ] **Step 1.3: Implement `compute_noise_scale_stats`**

Create `ref/LIFT/src/utils/noise_scale.py`:

```python
"""McCandlish noise-scale probe utility for LIFT finetune scripts.

Implements B_simple = tr(Sigma) / ||G||^2 from McCandlish et al.,
"An Empirical Model of Large-Batch Training" (arXiv:1812.06162).

Exports:
    NoiseScaleProbe        -- main probe class (added in a later task)
    compute_noise_scale_stats  -- pure-math estimator helper
"""
from __future__ import annotations

from typing import Dict


_EPS = 1e-12


def compute_noise_scale_stats(
    sum_g_sq_per_example: float,
    sum_g_norm_sq: float,
    batch_size: int,
) -> Dict[str, float | bool]:
    """Compute single-batch noise-scale estimators from accumulated scalars.

    Args:
        sum_g_sq_per_example: sum_i ||g_i||^2  over i = 1..B
        sum_g_norm_sq:        ||sum_i g_i||^2
        batch_size:           B (>= 2)

    Returns dict with keys:
        tr_sigma, g_norm_sq, g_norm_sq_naive, b_simple, b_simple_valid
    """
    if batch_size < 2:
        raise ValueError(f"batch_size must be >= 2, got {batch_size}")
    B = float(batch_size)

    # ||g_bar||^2 = ||sum_i g_i||^2 / B^2
    g_norm_sq_naive = sum_g_norm_sq / (B * B)
    # Sum of squared deviations: sum_i ||g_i - g_bar||^2 = sum_i ||g_i||^2 - B * ||g_bar||^2
    sum_sq_dev = sum_g_sq_per_example - B * g_norm_sq_naive
    # tr(Sigma) unbiased estimator
    tr_sigma = sum_sq_dev / (B - 1.0)
    # Bias-corrected ||G||^2
    g_norm_sq = g_norm_sq_naive - tr_sigma / B
    valid = g_norm_sq > 0.0
    b_simple = tr_sigma / max(g_norm_sq, _EPS)

    return {
        "tr_sigma": float(tr_sigma),
        "g_norm_sq": float(g_norm_sq),
        "g_norm_sq_naive": float(g_norm_sq_naive),
        "b_simple": float(b_simple),
        "b_simple_valid": bool(valid),
    }
```

- [ ] **Step 1.4: Run tests, verify they pass**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m unittest ref.LIFT.src.tests.test_noise_scale_estimator -v
```

Expected: 4 tests pass.

- [ ] **Step 1.5: Commit**

```bash
git add ref/LIFT/src/utils/noise_scale.py ref/LIFT/src/tests/test_noise_scale_estimator.py
git commit -m "noise-scale: add pure estimator helper with unit tests"
```

---

## Task 2: `NoiseScaleProbe` class skeleton — construction and lifecycle

**Files:**
- Modify: `ref/LIFT/src/utils/noise_scale.py`
- Test: `ref/LIFT/src/tests/test_noise_scale_probe_compat.py` (new file, partial)

This task adds the probe class with construction, `close()`, and the JSONL writer — but **not yet** the per-example backward. We test the lifecycle separately so the more complex test in Task 3 has a smaller surface to cover.

- [ ] **Step 2.1: Write failing test for probe construction and JSONL writer**

Create `ref/LIFT/src/tests/test_noise_scale_probe_compat.py`:

```python
"""Integration tests for NoiseScaleProbe.

Phase 1 (this file at end of Task 2): construction, JSONL lifecycle.
Phase 2 (Task 3): full bit-identity training-trajectory test.
"""
import json
import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_LIFT_SRC = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _LIFT_SRC not in sys.path:
    sys.path.insert(0, _LIFT_SRC)

from utils.noise_scale import NoiseScaleProbe


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n=64, d=8):
        torch.manual_seed(123)
        self.x = torch.randn(n, d)
        self.y = torch.randint(0, 2, (n,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"input": self.x[i], "target": self.y[i]}


def _collate(batch):
    return {
        "input": torch.stack([b["input"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
    }


class _FakeAccelerator:
    """Minimal shim so the probe's Accelerator-typed kwarg is satisfied in tests."""
    def __init__(self):
        self.is_main_process = True
        self.num_processes = 1
        self.device = torch.device("cpu")
        self.logged = []

    def log(self, values, step=None):
        self.logged.append((dict(values), step))


class TestNoiseScaleProbeLifecycle(unittest.TestCase):
    def test_construction_and_close_writes_no_rows(self):
        # If maybe_run is never called, JSONL is empty (file may exist but no lines).
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "noise_scale.jsonl")
            probe = NoiseScaleProbe(
                model=nn.Linear(8, 2),
                accelerator=_FakeAccelerator(),
                probe_dataset=_TinyDataset(),
                collate_fn=_collate,
                probe_batch_size=4,
                every_n_steps=10,
                seed=0,
                output_path=path,
                log_to_wandb=False,
            )
            probe.close()
            # File created on first probe call only; no probe call here, so file may not exist.
            # Just assert close is idempotent and no crash.
            probe.close()

    def test_multi_gpu_construction_rejected(self):
        acc = _FakeAccelerator()
        acc.num_processes = 2
        with self.assertRaises(RuntimeError):
            NoiseScaleProbe(
                model=nn.Linear(8, 2),
                accelerator=acc,
                probe_dataset=_TinyDataset(),
                collate_fn=_collate,
                probe_batch_size=4,
                every_n_steps=10,
            )

    def test_batch_size_one_rejected(self):
        with self.assertRaises(ValueError):
            NoiseScaleProbe(
                model=nn.Linear(8, 2),
                accelerator=_FakeAccelerator(),
                probe_dataset=_TinyDataset(),
                collate_fn=_collate,
                probe_batch_size=1,
                every_n_steps=10,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2.2: Run tests, verify they fail**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m unittest ref.LIFT.src.tests.test_noise_scale_probe_compat -v
```

Expected: `ImportError: cannot import name 'NoiseScaleProbe'`.

- [ ] **Step 2.3: Implement the probe class skeleton (no maybe_run yet)**

Append to `ref/LIFT/src/utils/noise_scale.py`:

```python
import json
import os
import time
from typing import Any, Callable, Dict, Iterator, Optional

import torch
from torch.utils.data import DataLoader, Dataset


class NoiseScaleProbe:
    """McCandlish noise-scale probe over trainable parameters.

    Single-GPU only. Non-destructive: saves and restores param.grad around
    each probe so the surrounding training trajectory is unaffected.

    The probe consumes its own dataloader cursor (independent shuffle seed),
    so probe examples are disjoint from what the training loop sees.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        accelerator: Any,
        probe_dataset: Dataset,
        collate_fn: Callable,
        probe_batch_size: int = 16,
        every_n_steps: int = 200,
        seed: int = 1234,
        output_path: Optional[str] = None,
        log_to_wandb: bool = True,
        device: Optional[torch.device] = None,
    ):
        if probe_batch_size < 2:
            raise ValueError(f"probe_batch_size must be >= 2, got {probe_batch_size}")
        if getattr(accelerator, "num_processes", 1) != 1:
            raise RuntimeError(
                "NoiseScaleProbe is single-GPU only "
                f"(accelerator.num_processes = {accelerator.num_processes})."
            )
        self._model = model
        self._accelerator = accelerator
        self._probe_dataset = probe_dataset
        self._collate_fn = collate_fn
        self._probe_batch_size = int(probe_batch_size)
        self._every_n_steps = int(every_n_steps)
        self._seed = int(seed)
        self._output_path = str(output_path) if output_path is not None else None
        self._log_to_wandb = bool(log_to_wandb)
        self._device = device if device is not None else getattr(accelerator, "device", torch.device("cpu"))

        self._loader: Optional[DataLoader] = None
        self._iter: Optional[Iterator] = None
        self._sum_g: Dict[torch.nn.Parameter, torch.Tensor] = {}
        self._jsonl_fh = None

    # ------------------------------------------------------------------ private

    def _ensure_loader(self) -> None:
        if self._loader is not None:
            return
        gen = torch.Generator()
        gen.manual_seed(self._seed)
        self._loader = DataLoader(
            self._probe_dataset,
            batch_size=self._probe_batch_size,
            shuffle=True,
            generator=gen,
            num_workers=0,
            collate_fn=self._collate_fn,
            drop_last=True,
        )
        self._iter = iter(self._loader)

    def _next_batch(self):
        self._ensure_loader()
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            return next(self._iter)

    def _ensure_jsonl(self) -> None:
        if self._jsonl_fh is not None or self._output_path is None:
            return
        os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)
        self._jsonl_fh = open(self._output_path, "a", buffering=1)  # line-buffered

    def _write_row(self, row: Dict[str, Any]) -> None:
        if self._output_path is not None:
            self._ensure_jsonl()
            self._jsonl_fh.write(json.dumps(row) + "\n")
            self._jsonl_fh.flush()
        if self._log_to_wandb:
            self._maybe_log_wandb(row)

    def _maybe_log_wandb(self, row: Dict[str, Any]) -> None:
        # Best-effort. If wandb isn't initialized OR the accelerator's log fails, swallow.
        try:
            payload = {
                "noise_scale/b_simple": row.get("b_simple"),
                "noise_scale/tr_sigma": row.get("tr_sigma"),
                "noise_scale/g_norm_sq": row.get("g_norm_sq"),
                "noise_scale/g_norm_sq_naive": row.get("g_norm_sq_naive"),
                "noise_scale/loss_at_probe": row.get("loss_at_probe"),
                "noise_scale/probe_wall_time_s": row.get("wall_time_s"),
            }
            payload = {k: v for k, v in payload.items() if v is not None}
            if payload and hasattr(self._accelerator, "log"):
                self._accelerator.log(payload, step=row.get("step"))
        except Exception:
            pass

    # ------------------------------------------------------------------- public

    def maybe_run(self, step: int, total_steps: int) -> Optional[Dict[str, Any]]:
        # Implemented in Task 3.
        raise NotImplementedError("maybe_run added in Task 3")

    def close(self) -> None:
        if self._jsonl_fh is not None:
            try:
                self._jsonl_fh.close()
            finally:
                self._jsonl_fh = None
```

- [ ] **Step 2.4: Run tests, verify they pass**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m unittest ref.LIFT.src.tests.test_noise_scale_probe_compat -v
```

Expected: 3 tests pass.

- [ ] **Step 2.5: Commit**

```bash
git add ref/LIFT/src/utils/noise_scale.py ref/LIFT/src/tests/test_noise_scale_probe_compat.py
git commit -m "noise-scale: add NoiseScaleProbe lifecycle (construct, close, JSONL)"
```

---

## Task 3: `maybe_run` — per-example backward and bit-identity guarantee

**Files:**
- Modify: `ref/LIFT/src/utils/noise_scale.py` (add `maybe_run`)
- Modify: `ref/LIFT/src/tests/test_noise_scale_probe_compat.py` (add bit-identity test)

This task is the heart of the feature. The probe runs a per-example backward pass, accumulates `sum_g_sq_per_example` and `sum_g[p]` in fp32, computes statistics via the helper from Task 1, and restores grads bit-exact.

The test is a strong correctness gate: weight tensors after N optimizer steps with probing **enabled** must equal weight tensors with probing **disabled** when the seed is identical and the probe has its own dataloader cursor.

- [ ] **Step 3.1: Add the bit-identity test**

Append to `ref/LIFT/src/tests/test_noise_scale_probe_compat.py`:

```python
class _TinyMLP(nn.Module):
    def __init__(self, d_in=8, d_hidden=16, d_out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, input, target):
        logits = self.net(input)
        loss = nn.functional.cross_entropy(logits, target)
        # Mimic HuggingFace-style outputs object with a `.loss` attr.
        class _O:
            pass
        out = _O()
        out.loss = loss
        out.logits = logits
        return out


def _train_n_steps(seed, n_steps, probe_every=None, probe_batch_size=4, jsonl_path=None):
    torch.manual_seed(seed)
    model = _TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    dataset = _TinyDataset(n=128, d=8)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_collate,
    )

    probe = None
    if probe_every is not None:
        probe = NoiseScaleProbe(
            model=model,
            accelerator=_FakeAccelerator(),
            probe_dataset=dataset,
            collate_fn=_collate,
            probe_batch_size=probe_batch_size,
            every_n_steps=probe_every,
            seed=seed + 7919,
            output_path=jsonl_path,
            log_to_wandb=False,
        )

    step = 0
    it = iter(loader)
    while step < n_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
        step += 1
        if probe is not None:
            probe.maybe_run(step, n_steps)

    if probe is not None:
        probe.close()
    return {n: p.detach().clone() for n, p in model.named_parameters()}


from torch.utils.data import DataLoader  # for _train_n_steps above


class TestNoiseScaleProbeBitIdentity(unittest.TestCase):
    def test_probe_does_not_perturb_training(self):
        weights_off = _train_n_steps(seed=42, n_steps=20, probe_every=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "noise_scale.jsonl")
            weights_on = _train_n_steps(
                seed=42, n_steps=20, probe_every=5,
                probe_batch_size=4, jsonl_path=path,
            )
            # JSONL should have floor(20 / 5) = 4 rows (steps 5, 10, 15, 20).
            with open(path) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(rows), 4)
            for r in rows:
                for key in (
                    "step", "probe_batch_size", "tr_sigma", "g_norm_sq",
                    "g_norm_sq_naive", "b_simple", "b_simple_valid",
                    "loss_at_probe", "num_trainable_params", "wall_time_s",
                    "epoch_frac",
                ):
                    self.assertIn(key, r)

        for name in weights_off:
            self.assertTrue(
                torch.equal(weights_off[name], weights_on[name]),
                msg=f"probe perturbed parameter {name}",
            )
```

(Note: the `from torch.utils.data import DataLoader` import inside the file body is intentional — it sits next to the code that uses it. If your linter complains, move it to the top of the file alongside the other imports.)

- [ ] **Step 3.2: Run tests, verify they fail**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m unittest ref.LIFT.src.tests.test_noise_scale_probe_compat -v
```

Expected: `TestNoiseScaleProbeBitIdentity.test_probe_does_not_perturb_training` fails with `NotImplementedError: maybe_run added in Task 3`.

- [ ] **Step 3.3: Implement `maybe_run` in `noise_scale.py`**

Replace the `maybe_run` stub in `ref/LIFT/src/utils/noise_scale.py` with the full implementation. Add at the top of the file (just after existing imports):

```python
from utils.noise_scale import compute_noise_scale_stats  # NOTE: adjust if you put helper in same module
```

Wait — `compute_noise_scale_stats` is already defined in this same module (Task 1), so no extra import needed. Use it directly.

Replace the `maybe_run` method with:

```python
    def maybe_run(self, step: int, total_steps: int) -> Optional[Dict[str, Any]]:
        if step <= 0 or (step % self._every_n_steps) != 0:
            return None
        if not getattr(self._accelerator, "is_main_process", True):
            return None

        was_training = self._model.training
        self._model.train()  # keep dropout/etc. enabled so noise reflects training-time stochasticity

        # Identify trainable params once per call (cheap).
        trainable = [p for p in self._model.parameters() if p.requires_grad]
        if not trainable:
            return None

        # Save current grads to restore later.
        saved_grads: Dict[torch.nn.Parameter, Optional[torch.Tensor]] = {}
        for p in trainable:
            saved_grads[p] = p.grad.detach().clone() if p.grad is not None else None

        # Lazy-allocate fp32 accumulator buffers, one per trainable param.
        if not self._sum_g:
            for p in trainable:
                self._sum_g[p] = torch.zeros_like(p, dtype=torch.float32, device=p.device)
        else:
            for p in trainable:
                self._sum_g[p].zero_()

        sum_g_sq_per_example = 0.0
        loss_sum = 0.0
        wall_t0 = time.time()
        error: Optional[str] = None

        try:
            batch = self._next_batch()
            B = self._infer_batch_size(batch)
            if B < 2:
                error = "probe_batch_too_small"
            else:
                # Truncate to configured probe_batch_size if larger.
                B = min(B, self._probe_batch_size)
                for i in range(B):
                    sub = self._slice_batch(batch, i, i + 1)
                    self._zero_param_grads(trainable)
                    out = self._model(**sub)
                    loss = out.loss if hasattr(out, "loss") else out["loss"]
                    loss.backward()
                    loss_sum += float(loss.detach().item())

                    for p in trainable:
                        if p.grad is None:
                            continue
                        g32 = p.grad.detach().to(torch.float32)
                        if not torch.isfinite(g32).all():
                            error = "non_finite_grad"
                            break
                        sum_g_sq_per_example += float((g32 * g32).sum().item())
                        self._sum_g[p].add_(g32)
                    if error is not None:
                        break
        except torch.cuda.OutOfMemoryError:
            error = "oom"
        except Exception as e:  # noqa: BLE001 - probe must never crash training
            error = f"unexpected:{type(e).__name__}"

        # Compute stats (only if no error).
        row: Dict[str, Any] = {
            "step": int(step),
            "epoch_frac": (float(step) / float(total_steps)) if total_steps else None,
            "probe_batch_size": int(B if error is None else 0),
            "num_trainable_params": int(sum(p.numel() for p in trainable)),
            "wall_time_s": float(time.time() - wall_t0),
        }

        if error is None:
            sum_g_norm_sq = 0.0
            for p in trainable:
                buf = self._sum_g[p]
                sum_g_norm_sq += float((buf * buf).sum().item())
            stats = compute_noise_scale_stats(
                sum_g_sq_per_example=sum_g_sq_per_example,
                sum_g_norm_sq=sum_g_norm_sq,
                batch_size=B,
            )
            row.update(stats)
            row["loss_at_probe"] = loss_sum / B
            row["error"] = None
        else:
            row.update({
                "tr_sigma": None,
                "g_norm_sq": None,
                "g_norm_sq_naive": None,
                "b_simple": None,
                "b_simple_valid": False,
                "loss_at_probe": None,
                "error": error,
            })

        # Restore grads to bit-exact saved state.
        for p in trainable:
            saved = saved_grads.get(p, None)
            if saved is None:
                p.grad = None
            else:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.copy_(saved)

        if not was_training:
            self._model.eval()

        self._write_row(row)
        return row

    # ------------------------------------------------------------- batch helpers

    @staticmethod
    def _zero_param_grads(params) -> None:
        for p in params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @staticmethod
    def _infer_batch_size(batch) -> int:
        if isinstance(batch, dict):
            for v in batch.values():
                if torch.is_tensor(v):
                    return int(v.shape[0])
            raise ValueError("Probe batch dict has no tensor values")
        if torch.is_tensor(batch):
            return int(batch.shape[0])
        raise TypeError(f"Unsupported batch type for probe: {type(batch)}")

    @staticmethod
    def _slice_batch(batch, start: int, end: int):
        if isinstance(batch, dict):
            return {
                k: (v[start:end] if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
        if torch.is_tensor(batch):
            return batch[start:end]
        raise TypeError(f"Unsupported batch type for probe: {type(batch)}")
```

- [ ] **Step 3.4: Run tests, verify they pass**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m unittest ref.LIFT.src.tests.test_noise_scale_probe_compat -v
python -m unittest ref.LIFT.src.tests.test_noise_scale_estimator -v
```

Expected: All tests pass (3 lifecycle + 1 bit-identity = 4 in compat file; 4 in estimator file).

- [ ] **Step 3.5: Commit**

```bash
git add ref/LIFT/src/utils/noise_scale.py ref/LIFT/src/tests/test_noise_scale_probe_compat.py
git commit -m "noise-scale: implement maybe_run with bit-identity guarantee"
```

---

## Task 4: Wire probe into `finetune_lora.py`

**Files:**
- Modify: `ref/LIFT/src/finetune_lora.py`

The four edits from the spec, applied verbatim. Existing tests under `ref/LIFT/src/tests/` should continue to pass (the SVD test is unrelated; LoRA has no existing test).

- [ ] **Step 4.1: Locate the argparse block and add three flags**

Read `ref/LIFT/src/finetune_lora.py` and find the argparse section (search for `parser = argparse.ArgumentParser` or `parser.add_argument("--gradient_accumulation_steps"`).

Add the following three arguments alongside the existing flags (anywhere after the parser is created, before `args = parser.parse_args()`). A natural place is right after the `--gradient_accumulation_steps` argument:

```python
    parser.add_argument(
        "--track_noise_scale",
        action="store_true",
        help="Enable McCandlish noise-scale probe (writes <output_dir>/noise_scale.jsonl).",
    )
    parser.add_argument(
        "--noise_scale_every_n_steps",
        type=int,
        default=200,
        help="Probe cadence in optimizer steps.",
    )
    parser.add_argument(
        "--noise_scale_probe_batch_size",
        type=int,
        default=16,
        help="Per-probe batch size B (>= 2).",
    )
```

- [ ] **Step 4.2: Construct the probe before the training loop**

Insert immediately **after** line 612 (the `sysmon.base_params = ...` line) and **before** the `# Training function` comment / `def train_epoch(epoch):` at around line 614–615:

```python
    # ----------------------------------------------------------- noise-scale probe
    noise_probe = None
    if args.track_noise_scale and accelerator.is_main_process:
        from utils.noise_scale import NoiseScaleProbe
        noise_probe = NoiseScaleProbe(
            model=model,
            accelerator=accelerator,
            probe_dataset=train_dataset,
            collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
            probe_batch_size=args.noise_scale_probe_batch_size,
            every_n_steps=args.noise_scale_every_n_steps,
            seed=args.seed + 7919,
            output_path=os.path.join(args.output_dir, "noise_scale.jsonl"),
        )
```

(The collate_fn argument is reconstructed from `DataCollatorForSupervisedDataset(tokenizer=tokenizer)` — the same expression already used at line 538 — because the `train_dataloader.collate_fn` was created before `accelerator.prepare(...)` and may have been wrapped.)

- [ ] **Step 4.3: Call `maybe_run` inside the training loop**

Find the block (around line 626–637):

```python
            if accelerator.sync_gradients:
                _t0 = time.time()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                sysmon.record_step(time.time() - _t0)
                progress_bar.update(1)
                args.completed_steps += 1
                if args.max_steps > 0 and args.completed_steps >= args.max_steps:
                    return
```

Insert the probe call between `args.completed_steps += 1` and the `if args.max_steps > 0 ...` line:

```python
                args.completed_steps += 1
                if noise_probe is not None:
                    noise_probe.maybe_run(args.completed_steps, max_train_steps)
                if args.max_steps > 0 and args.completed_steps >= args.max_steps:
                    return
```

(`max_train_steps` is the local variable computed earlier; it's in scope inside the closure `train_epoch`.)

- [ ] **Step 4.4: Close the probe after training ends**

Find the location after the training loop (after the `for epoch in range(args.num_train_epochs):` loop) — search for where `train_epoch(epoch)` is called. Add a `noise_probe.close()` immediately after that loop:

```python
    for epoch in range(args.num_train_epochs):
        train_epoch(epoch)
        if args.max_steps > 0 and args.completed_steps >= args.max_steps:
            break

    if noise_probe is not None:
        noise_probe.close()
```

(If the existing structure has different control flow around the for-loop, adapt minimally — the constraint is that `close()` runs exactly once after training finishes.)

- [ ] **Step 4.5: Syntax-check the edited script**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m py_compile ref/LIFT/src/finetune_lora.py
```

Expected: silent success (no output, exit 0).

- [ ] **Step 4.6: Verify no-flag run is bit-identical via static reading**

Read `ref/LIFT/src/finetune_lora.py` again and confirm: the new `noise_probe` is `None` when `--track_noise_scale` is not passed, and every probe interaction is gated on `if noise_probe is not None:`. The only non-gated change is the addition of three argparse flags (which adds attributes to `args` but doesn't change runtime behavior when not passed).

- [ ] **Step 4.7: Commit**

```bash
git add ref/LIFT/src/finetune_lora.py
git commit -m "finetune_lora: wire noise-scale probe behind --track_noise_scale flag"
```

---

## Task 5: Wire probe into `finetune_blocktt.py`

**Files:**
- Modify: `ref/LIFT/src/finetune_blocktt.py`

Structurally analogous to Task 4. Specific line numbers below were taken at design time and may have drifted slightly; use them as anchors.

- [ ] **Step 5.1: Add the three argparse flags**

Find the argparse section in `ref/LIFT/src/finetune_blocktt.py`. Add the same three flags as in Task 4.1 (verbatim):

```python
    parser.add_argument(
        "--track_noise_scale",
        action="store_true",
        help="Enable McCandlish noise-scale probe (writes <output_dir>/noise_scale.jsonl).",
    )
    parser.add_argument(
        "--noise_scale_every_n_steps",
        type=int,
        default=200,
        help="Probe cadence in optimizer steps.",
    )
    parser.add_argument(
        "--noise_scale_probe_batch_size",
        type=int,
        default=16,
        help="Per-probe batch size B (>= 2).",
    )
```

- [ ] **Step 5.2: Construct the probe before `train_epoch` is defined**

Find the `def train_epoch(epoch):` line (around line 469). Insert immediately before it:

```python
    # ----------------------------------------------------------- noise-scale probe
    noise_probe = None
    if args.track_noise_scale and accelerator.is_main_process:
        from utils.noise_scale import NoiseScaleProbe
        noise_probe = NoiseScaleProbe(
            model=model,
            accelerator=accelerator,
            probe_dataset=train_dataset,
            collate_fn=data_collator,
            probe_batch_size=args.noise_scale_probe_batch_size,
            every_n_steps=args.noise_scale_every_n_steps,
            seed=args.seed + 7919,
            output_path=os.path.join(args.output_dir, "noise_scale.jsonl"),
        )
```

(Note: BlockTT uses `data_collator` as a named local variable, unlike LoRA which inlines `DataCollatorForSupervisedDataset(...)`. We reuse the existing variable here.)

- [ ] **Step 5.3: Call `maybe_run` inside the training loop**

Find the block (around line 480–494):

```python
            if accelerator.sync_gradients:
                _t0 = time.time()
                optimizer.step()
                if args.blocktt_normalize_after_update:
                    unwrapped = accelerator.unwrap_model(model)
                    normalize_trainable_blocktt_cores_(unwrapped)
                lr_scheduler.step()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                sysmon.record_step(time.time() - _t0)
                progress_bar.update(1)
                args.completed_steps += 1
                if args.max_steps > 0 and args.completed_steps >= args.max_steps:
                    return
```

Insert the probe call between `args.completed_steps += 1` and the `if args.max_steps > 0 ...` line:

```python
                args.completed_steps += 1
                if noise_probe is not None:
                    noise_probe.maybe_run(args.completed_steps, max_train_steps)
                if args.max_steps > 0 and args.completed_steps >= args.max_steps:
                    return
```

- [ ] **Step 5.4: Close the probe after training ends**

Find the spot after the epoch loop in `main()` (near where `save_hf_format` or analogous final save runs) and add:

```python
    if noise_probe is not None:
        noise_probe.close()
```

If the BlockTT script has multiple `return` paths or early-exits, place `noise_probe.close()` just before the first one that follows the training loop. The probe's `close()` is idempotent, so duplicating it across exit paths is safe.

- [ ] **Step 5.5: Syntax-check**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m py_compile ref/LIFT/src/finetune_blocktt.py
```

Expected: silent success.

- [ ] **Step 5.6: Commit**

```bash
git add ref/LIFT/src/finetune_blocktt.py
git commit -m "finetune_blocktt: wire noise-scale probe behind --track_noise_scale flag"
```

---

## Task 6: argparse smoke test for both finetune scripts

**Files:**
- Create: `ref/LIFT/src/tests/test_noise_scale_cli.py`

A quick parser-level test that confirms the three new flags are accepted by both finetune scripts. We avoid invoking the full `main()` (which would try to load Llama-3-8B); we only exercise the argparse layer.

- [ ] **Step 6.1: Find the parser-construction function in each finetune script**

Read `ref/LIFT/src/finetune_lora.py` and `ref/LIFT/src/finetune_blocktt.py`. Identify how the argparse parser is built. If the scripts construct the parser inside `main()`, the test will need to either (a) call a small helper that returns the parser, or (b) patch `sys.argv` and call `main()` with an early-exit. Approach (a) is cleaner if the scripts already factor a `parse_args()` function.

If neither script has a separate parser-builder function, **create one** in each as part of this task. The simplest refactor: move the argparse construction into a top-level function `_build_parser() -> argparse.ArgumentParser` and have `main()` call `_build_parser().parse_args()`.

If `_build_parser` already exists (or an equivalent), use it as-is. Otherwise, do the small refactor as Step 6.1.5 below.

- [ ] **Step 6.1.5: (only if needed) Refactor parser construction into `_build_parser()`**

In each finetune script, replace the current `parser = argparse.ArgumentParser(...)` ... `args = parser.parse_args()` block at the top of `main()` with:

```python
def _build_parser():
    parser = argparse.ArgumentParser(...)
    # ... all parser.add_argument calls ...
    return parser

def main():
    args = _build_parser().parse_args()
    ...
```

Run `python -m py_compile` on both scripts to confirm the refactor is syntactically clean.

- [ ] **Step 6.2: Write the test**

Create `ref/LIFT/src/tests/test_noise_scale_cli.py`:

```python
"""argparse smoke test for the new noise-scale flags.

Verifies that finetune_lora.py and finetune_blocktt.py accept:
    --track_noise_scale
    --noise_scale_every_n_steps <int>
    --noise_scale_probe_batch_size <int>
without raising and that the resulting Namespace exposes them as attributes.
"""
import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_LIFT_SRC = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _LIFT_SRC not in sys.path:
    sys.path.insert(0, _LIFT_SRC)

# Minimum required argv for each script. We pass placeholders for paths;
# argparse won't validate filesystem existence at parse time.
_MIN_ARGV_LORA = [
    "--model_name_or_path", "dummy",
    "--data_path", "dummy.json",
    "--output_dir", "/tmp/dummy",
]
_MIN_ARGV_BTT = [
    "--model_name_or_path", "dummy",
    "--data_path", "dummy.json",
    "--output_dir", "/tmp/dummy",
]


def _parse(module_name, argv):
    mod = __import__(module_name)
    parser = mod._build_parser()
    return parser.parse_args(argv)


class TestNoiseScaleCli(unittest.TestCase):
    def test_lora_accepts_flags(self):
        argv = _MIN_ARGV_LORA + [
            "--track_noise_scale",
            "--noise_scale_every_n_steps", "200",
            "--noise_scale_probe_batch_size", "16",
        ]
        ns = _parse("finetune_lora", argv)
        self.assertTrue(ns.track_noise_scale)
        self.assertEqual(ns.noise_scale_every_n_steps, 200)
        self.assertEqual(ns.noise_scale_probe_batch_size, 16)

    def test_lora_defaults_when_flag_absent(self):
        ns = _parse("finetune_lora", _MIN_ARGV_LORA)
        self.assertFalse(ns.track_noise_scale)
        self.assertEqual(ns.noise_scale_every_n_steps, 200)
        self.assertEqual(ns.noise_scale_probe_batch_size, 16)

    def test_blocktt_accepts_flags(self):
        argv = _MIN_ARGV_BTT + [
            "--track_noise_scale",
            "--noise_scale_every_n_steps", "100",
            "--noise_scale_probe_batch_size", "8",
        ]
        ns = _parse("finetune_blocktt", argv)
        self.assertTrue(ns.track_noise_scale)
        self.assertEqual(ns.noise_scale_every_n_steps, 100)
        self.assertEqual(ns.noise_scale_probe_batch_size, 8)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6.3: Run the test**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m unittest ref.LIFT.src.tests.test_noise_scale_cli -v
```

Expected: 3 tests pass.

If the test fails because either script doesn't have `_build_parser()`, perform the refactor described in Step 6.1.5 and re-run.

- [ ] **Step 6.4: Commit**

```bash
git add ref/LIFT/src/tests/test_noise_scale_cli.py ref/LIFT/src/finetune_lora.py ref/LIFT/src/finetune_blocktt.py
git commit -m "noise-scale: argparse smoke test for both finetune scripts"
```

---

## Task 7: Launcher shell scripts

**Files:**
- Create: `ref/LIFT/bash_scripts/finetune_commonsense_lora_cbs.sh`
- Create: `ref/LIFT/bash_scripts/finetune_commonsense_blocktt_cbs.sh`

These are the user-facing entry points. They mirror the existing scripts but force 1 epoch, enable probing, and stamp run dirs distinctively.

- [ ] **Step 7.1: Read the source scripts to copy from**

```bash
cat /home/yequan/Project/lora/lora-without-regret/ref/LIFT/bash_scripts/finetune_commonsense_lora.sh
cat /home/yequan/Project/lora/lora-without-regret/ref/LIFT/bash_scripts/finetune_commonsense_blocktt.sh
```

- [ ] **Step 7.2: Create the LoRA CBS launcher**

Write to `ref/LIFT/bash_scripts/finetune_commonsense_lora_cbs.sh`:

```bash
#!/bin/bash
# Launcher: LoRA + noise-scale CBS probe, 1 epoch on commonsense_170k.
# Companion to finetune_commonsense_lora.sh; do not modify the original.

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"

SRC_DIR=/home/yequan/Project/lora/lora-without-regret/ref/LIFT
DATA_DIR=LLM-Adapters
OUTPUT_SRC_DIR=/data/yequan/fura/lift

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
adapter_name="${adapter_name:-lora}"
lr="${lr:-2e-4}"
lora_r="${lora_r:-64}"
lora_alpha="${lora_alpha:-128}"
seed="${seed:-43}"
MAX_STEPS="${MAX_STEPS:-0}"
model_tag="${MODEL##*/}"

target_modules="${target_modules:-q_proj k_proj v_proj up_proj down_proj}"
_tm_count=$(echo $target_modules | wc -w)
if [ "$_tm_count" = "5" ] && [ "$target_modules" = "q_proj k_proj v_proj up_proj down_proj" ]; then
    _tm_tag=""
else
    _tm_tag="-tgt_${_tm_count}mod"
fi

NOISE_EVERY_N="${NOISE_EVERY_N:-200}"
NOISE_BATCH="${NOISE_BATCH:-16}"

wandb_project="${wandb_project:-commonsense-${model_tag}-cbs}"
run_name="${run_name:-${adapter_name}-cbs-lr_${lr}-rank_${lora_r}${_tm_tag}-seed_${seed}}"
wandb_run_id="${wandb_run_id:-$(python -c 'import wandb; print(wandb.util.generate_id())')}"

export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/commonsense/${MODEL}/${adapter_name}-cbs-lr_${lr}-rank_${lora_r}${_tm_tag}-seed_${seed}}"
mkdir -p $OUTPUT

cd $SRC_DIR

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_lora.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --load_last_model \
    --adapter_name ${adapter_name} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --target_modules ${target_modules} \
    --data_path ${DATA_DIR}/ft-training_set/commonsense_170k.json \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    --max_steps ${MAX_STEPS} \
    --save_interval 100000 \
    --track_noise_scale \
    --noise_scale_every_n_steps ${NOISE_EVERY_N} \
    --noise_scale_probe_batch_size ${NOISE_BATCH} \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

# CBS runs are for measurement, not benchmark scoring. Skip auto-eval.
echo "CBS run finished. JSONL at: $OUTPUT/noise_scale.jsonl"
```

- [ ] **Step 7.3: Create the BlockTT CBS launcher**

Write to `ref/LIFT/bash_scripts/finetune_commonsense_blocktt_cbs.sh`:

```bash
#!/bin/bash
# Launcher: BlockTT (fura defaults) + noise-scale CBS probe, 1 epoch on commonsense_170k.
# Companion to finetune_commonsense_blocktt.sh; do not modify the original.
# fura defaults (per CLAUDE.md): output_one_block / small / keep_trainable / full rank.

pwd
hostname
date
echo starting job...
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export HF_HOME="${HF_HOME:-/data/yequan/huggingface}"

SRC_DIR=/home/yequan/Project/lora/lora-without-regret/ref/LIFT
DATA_DIR=LLM-Adapters
OUTPUT_SRC_DIR=/data/yequan/fura/lift

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
decomp_mode="${decomp_mode:-output_one_block}"
train_position="${train_position:-small}"
s_merged_to="${s_merged_to:-keep_trainable}"
blocktt_rank="${blocktt_rank:-full}"
trainable_type="${trainable_type:-all}"
lr="${lr:-2e-4}"
seed="${seed:-43}"
MAX_STEPS="${MAX_STEPS:-0}"
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-8}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-2}"
model_tag="${MODEL##*/}"

calib_mode="${calib_mode:-none}"
calib_source="${calib_source:-training_data}"
calib_num_seqs="${calib_num_seqs:-128}"
calib_batch_size="${calib_batch_size:-4}"

NOISE_EVERY_N="${NOISE_EVERY_N:-200}"
NOISE_BATCH="${NOISE_BATCH:-16}"

wandb_project="${wandb_project:-commonsense-${model_tag}-cbs}"
wandb_run_id="${wandb_run_id:-$(python -c 'import wandb; print(wandb.util.generate_id())')}"
export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

OUTPUT="${OUTPUT:-${OUTPUT_SRC_DIR}/commonsense/${MODEL}/blocktt-cbs-calib_${calib_mode}-lr_${lr}-decomp_${decomp_mode}_pos_${train_position}_smerge_${s_merged_to}-seed_${seed}}"
run_name="${run_name:-$(basename "$OUTPUT")}"

mkdir -p $OUTPUT

cd ${SRC_DIR}

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision="bf16" \
    src/finetune_blocktt.py \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BS} \
    --per_device_eval_batch_size 16 \
    --logging_steps 10 \
    --max_seq_len 2048 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --mixed_precision bf16 \
    --gradient_accumulation_steps ${GRAD_ACC_STEPS} \
    --lr_scheduler_type linear \
    --num_warmup_steps 0.03 \
    --seed ${seed} \
    --gradient_checkpointing \
    --instruction_type single \
    --decomp_mode ${decomp_mode} \
    --train_position ${train_position} \
    --blocktt_rank ${blocktt_rank} \
    --s_merged_to ${s_merged_to} \
    --trainable_type ${trainable_type} \
    --calib_mode ${calib_mode} \
    --calib_source ${calib_source} \
    --calib_num_seqs ${calib_num_seqs} \
    --calib_batch_size ${calib_batch_size} \
    --save_interval 100000 \
    --load_last_model \
    --data_path ${DATA_DIR}/ft-training_set/commonsense_170k.json \
    --wandb_project "${wandb_project}" \
    --wandb_run_name "${run_name}" \
    --max_steps ${MAX_STEPS} \
    --track_noise_scale \
    --noise_scale_every_n_steps ${NOISE_EVERY_N} \
    --noise_scale_probe_batch_size ${NOISE_BATCH} \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log

echo "CBS run finished. JSONL at: $OUTPUT/noise_scale.jsonl"
```

- [ ] **Step 7.4: Make both scripts executable and bash-syntax-check them**

```bash
cd /home/yequan/Project/lora/lora-without-regret
chmod +x ref/LIFT/bash_scripts/finetune_commonsense_lora_cbs.sh
chmod +x ref/LIFT/bash_scripts/finetune_commonsense_blocktt_cbs.sh
bash -n ref/LIFT/bash_scripts/finetune_commonsense_lora_cbs.sh
bash -n ref/LIFT/bash_scripts/finetune_commonsense_blocktt_cbs.sh
```

Expected: silent success on both `bash -n` invocations.

- [ ] **Step 7.5: Commit**

```bash
git add ref/LIFT/bash_scripts/finetune_commonsense_lora_cbs.sh ref/LIFT/bash_scripts/finetune_commonsense_blocktt_cbs.sh
git commit -m "noise-scale: launcher scripts for LoRA and BlockTT CBS runs"
```

---

## Task 8: Standalone plot script

**Files:**
- Create: `ref/LIFT/scripts/plot_noise_scale.py`

A self-contained, dependency-light plot script. Imports only `pandas`, `matplotlib`, and stdlib.

- [ ] **Step 8.1: Verify dependencies are available**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -c "import pandas, matplotlib, matplotlib.pyplot; print('ok')"
```

Expected: `ok`. If imports fail, install via `uv add pandas matplotlib` or the project's standard process.

- [ ] **Step 8.2: Create the plot script**

Write to `ref/LIFT/scripts/plot_noise_scale.py`:

```python
#!/usr/bin/env python
"""Standalone plot for McCandlish noise-scale runs.

Usage:
    python plot_noise_scale.py \\
        --runs <run_dir1> <run_dir2> ... \\
        [--labels "LoRA r=64" "BlockTT" ...] \\
        --out figures/noise_scale_comparison.png \\
        [--smooth_window N] \\
        [--mask_invalid / --no-mask_invalid]

Each run dir must contain noise_scale.jsonl. Produces a 5-subplot comparison
figure (b_simple, tr_sigma, g_norm_sq, g_norm_sq_naive, loss_at_probe) saved
as both PNG and PDF.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


_METRICS: List[Tuple[str, str, bool]] = [
    # (column, subplot_title, log_y)
    ("b_simple", r"$B_{\mathrm{simple}}$", True),
    ("tr_sigma", r"$\mathrm{tr}(\Sigma)$", True),
    ("g_norm_sq", r"$\|G\|^2$ (bias-corrected)", False),
    ("g_norm_sq_naive", r"$\|\bar g\|^2$ (naive)", True),
    ("loss_at_probe", r"loss at probe", False),
]


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=False).mean()


def _summarize(label: str, df: pd.DataFrame, masked: int) -> str:
    bs = df["b_simple"].dropna()
    if len(bs) == 0:
        return f"{label}: 0 probes (empty)"
    return (
        f"{label}: {len(df)} probes, "
        f"B_simple range [{bs.min():.1f}, {bs.max():.1f}], "
        f"median {bs.median():.1f}, "
        f"masked {masked}/{len(df)}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="One or more run directories containing noise_scale.jsonl.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Legend labels (one per run). Defaults to basenames of the run dirs.")
    parser.add_argument("--out", required=True,
                        help="Output PNG path. A matching .pdf is also saved.")
    parser.add_argument("--smooth_window", type=int, default=5,
                        help="Moving-average window over consecutive probes. 1 disables smoothing.")
    mask = parser.add_mutually_exclusive_group()
    mask.add_argument("--mask_invalid", dest="mask_invalid", action="store_true",
                      help="Drop rows where b_simple_valid == False from the b_simple "
                           "and g_norm_sq subplots (default).")
    mask.add_argument("--no-mask_invalid", dest="mask_invalid", action="store_false",
                      help="Do not mask invalid rows.")
    parser.set_defaults(mask_invalid=True)
    args = parser.parse_args(argv)

    runs = [Path(p) for p in args.runs]
    labels = args.labels if args.labels is not None else [p.name for p in runs]
    if len(labels) != len(runs):
        print(f"--labels has {len(labels)} entries but --runs has {len(runs)}.", file=sys.stderr)
        return 2

    # Load all runs.
    loaded: List[Tuple[str, pd.DataFrame]] = []
    for label, run in zip(labels, runs):
        jsonl = run / "noise_scale.jsonl"
        if not jsonl.exists():
            print(f"ERROR: {jsonl} does not exist.", file=sys.stderr)
            return 2
        df = _load_jsonl(jsonl)
        if df.empty:
            print(f"WARNING: {jsonl} is empty; skipping.", file=sys.stderr)
            continue
        loaded.append((label, df))
    if not loaded:
        print("No runs with data; nothing to plot.", file=sys.stderr)
        return 2

    # Figure: 2 rows x 3 cols, last cell hidden.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    flat_axes = axes.flatten()

    summaries: List[str] = []
    for label, df in loaded:
        if "b_simple_valid" not in df.columns:
            df = df.assign(b_simple_valid=True)
        masked = int((~df["b_simple_valid"].astype(bool)).sum())
        summaries.append(_summarize(label, df, masked))

    for idx, (col, title, log_y) in enumerate(_METRICS):
        ax = flat_axes[idx]
        for label, df in loaded:
            if col not in df.columns:
                continue
            view = df
            if args.mask_invalid and col in ("b_simple", "g_norm_sq"):
                view = df[df["b_simple_valid"].astype(bool)]
            if view.empty:
                continue
            x = view["step"].to_numpy()
            y_raw = view[col].astype(float)
            ax.plot(x, y_raw, alpha=0.3, label=None)
            y_smooth = _smooth(y_raw, args.smooth_window)
            ax.plot(x, y_smooth, alpha=1.0, label=label)
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # Hide the unused 6th subplot.
    flat_axes[-1].axis("off")
    fig.suptitle("Noise-scale (McCandlish B_simple) — across runs", fontsize=12)

    out_png = Path(args.out)
    out_pdf = out_png.with_suffix(".pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)

    for line in summaries:
        print(line)
    print(f"Saved {out_png} and {out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8.3: Smoke-test the plot script on synthetic JSONL**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python - <<'PY'
import json, os, random
random.seed(0)
for run in ("/tmp/_ns_run_a", "/tmp/_ns_run_b"):
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, "noise_scale.jsonl"), "w") as f:
        for step in range(200, 2001, 200):
            row = {
                "step": step,
                "epoch_frac": step / 2000.0,
                "probe_batch_size": 16,
                "tr_sigma": 10 + random.gauss(0, 2),
                "g_norm_sq": max(0.001, 0.5 - step / 5000 + random.gauss(0, 0.05)),
                "g_norm_sq_naive": 1.0 + random.gauss(0, 0.1),
                "b_simple": 20 + step / 100 + random.gauss(0, 5),
                "b_simple_valid": True,
                "loss_at_probe": 1.5 - step / 4000 + random.gauss(0, 0.05),
                "num_trainable_params": 1234567,
                "wall_time_s": 6.0 + random.gauss(0, 0.5),
                "error": None,
            }
            f.write(json.dumps(row) + "\n")
PY
mkdir -p /tmp/_ns_fig
python ref/LIFT/scripts/plot_noise_scale.py \
    --runs /tmp/_ns_run_a /tmp/_ns_run_b \
    --labels "Run A" "Run B" \
    --out /tmp/_ns_fig/test.png
ls /tmp/_ns_fig/
```

Expected: stdout shows two summary lines and a `Saved ...` line. `/tmp/_ns_fig/test.png` and `/tmp/_ns_fig/test.pdf` both exist.

- [ ] **Step 8.4: Smoke-test missing-file error path**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python ref/LIFT/scripts/plot_noise_scale.py \
    --runs /tmp/__no_such_dir__ \
    --out /tmp/_ns_fig/should_not_exist.png ; echo "exit=$?"
```

Expected: a stderr `ERROR: ... does not exist.` and `exit=2`.

- [ ] **Step 8.5: Commit**

```bash
git add ref/LIFT/scripts/plot_noise_scale.py
git commit -m "noise-scale: standalone multi-run plot script"
```

---

## Task 9: Final integration check

**Files:** none (verification only).

- [ ] **Step 9.1: Run all tests touched by this plan**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m unittest ref.LIFT.src.tests.test_noise_scale_estimator -v
python -m unittest ref.LIFT.src.tests.test_noise_scale_probe_compat -v
python -m unittest ref.LIFT.src.tests.test_noise_scale_cli -v
```

Expected: all pass. Tally: 4 + 4 + 3 = 11 tests, all green.

- [ ] **Step 9.2: Confirm the existing SVD test still passes**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m unittest ref.LIFT.src.tests.test_finetune_svd -v 2>&1 | tail -5
```

Expected: pre-existing pass count unchanged (no regressions).

- [ ] **Step 9.3: Confirm both finetune scripts compile**

```bash
cd /home/yequan/Project/lora/lora-without-regret
python -m py_compile ref/LIFT/src/finetune_lora.py ref/LIFT/src/finetune_blocktt.py ref/LIFT/src/utils/noise_scale.py ref/LIFT/scripts/plot_noise_scale.py
```

Expected: silent success.

- [ ] **Step 9.4: Confirm both shell scripts parse**

```bash
cd /home/yequan/Project/lora/lora-without-regret
bash -n ref/LIFT/bash_scripts/finetune_commonsense_lora_cbs.sh
bash -n ref/LIFT/bash_scripts/finetune_commonsense_blocktt_cbs.sh
```

Expected: silent success.

- [ ] **Step 9.5: Final commit if anything changed during integration check**

```bash
git status
# If clean, no commit needed. Otherwise commit any small fixes:
# git add -p ; git commit -m "noise-scale: integration fixes"
```

---

## Self-Review

**Spec coverage:**

- ✅ §1 Goal/scope → Tasks 1–8 collectively
- ✅ §2 Estimator math + numerical stability → Task 1 (`compute_noise_scale_stats`) + Task 3 (fp32 accumulation, NaN/Inf check, OOM handling, `b_simple_valid` flag)
- ✅ §3 Probe execution semantics → Task 3 (per-example backward, save/restore grads, separate dataloader cursor, `train()` mode preserved, error handling)
- ✅ §4 Module API and integration → Task 2 (class skeleton), Task 3 (maybe_run), Task 4/5 (call-site wiring)
- ✅ §5 Launchers and plot → Task 7 (shell scripts), Task 8 (plot script)
- ✅ §6 Testing → Task 1 estimator tests, Task 2/3 compat tests, Task 6 CLI smoke tests
- ✅ §7 File manifest → matches Tasks 1–8 outputs
- ✅ Bit-identity guarantee → Task 3 Step 3.1 explicitly tests it

**Placeholder scan:**

- No "TBD"/"TODO" in steps. All code blocks are complete.
- One conditional refactor (Task 6.1.5: add `_build_parser()` if not already present) — has full code shown.
- One probe-edit location depends on existing line numbers (~612, ~614, ~626, ~480) — these are anchors derived from reading the files; the surrounding context strings are quoted so a search will find the right spot even if line numbers drift.

**Type/name consistency:**

- `compute_noise_scale_stats` returns dict with keys `tr_sigma, g_norm_sq, g_norm_sq_naive, b_simple, b_simple_valid` — used identically in Task 3 (`row.update(stats)`) and Task 8 (column names in plot script).
- `NoiseScaleProbe.maybe_run(step, total_steps)` signature matches every call site (Task 4.3, Task 5.3, test in Task 3.1).
- JSONL row keys (`step`, `epoch_frac`, `probe_batch_size`, `tr_sigma`, `g_norm_sq`, `g_norm_sq_naive`, `b_simple`, `b_simple_valid`, `loss_at_probe`, `num_trainable_params`, `wall_time_s`, `error`) match between Task 3 (writer), Task 3 (test asserts), and Task 8 (`_METRICS`).

No issues found. Plan is ready for execution.
