"""Equivalence tests between run_rl.py (legacy plain BTT/SVD path via
btt_layer.py / svd_layer.py) and run_rl_new.py (unified plain BTT/SVD path
via the src/compress package).

The two paths must agree on:
  1. Materialized dense weight immediately after conversion.
  2. Forward output on a fixed input.
  3. Materialized dense weight after one AdamW step on identical gradients.

If any of these differ, training behavior between the two scripts diverges.

Tests require CUDA because the legacy converters perform SVD/QR on GPU.
"""
import copy
import os
import sys
import unittest

# Make the src/compress package and the repo root importable from the test file.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

# Legacy reference path (still imported here only to assert equivalence; the
# production run_rl_new.py never imports these).
from btt_layer import (
    BTTLayer,
    configure_blocktt_trainability,
    convert_linear_to_btt,
)
from svd_layer import (
    SVDLayer,
    configure_svd_trainability,
    convert_linear_to_svd,
)
# Unified src/compress path under test.
from compress.btt.btt_linear import BTTLinear
from compress.svd.svd_linear import SVDCompressedLinear
from compress.integration import (
    convert_linear_to_btt_compress,
    configure_compress_btt_trainability,
    convert_linear_to_svd_compress,
    configure_compress_svd_trainability,
)


class _ToyAttnConfig:
    """Mimics the bits of a HuggingFace config that convert_linear_to_btt reads."""
    num_attention_heads = 4
    num_key_value_heads = 4
    head_dim = 8
    hidden_size = 32


class _ToyAttnMLPModel(nn.Module):
    """Smallest model that exercises attention-style and MLP-style Linear
    layers with the names the converters scan for."""

    def __init__(self):
        super().__init__()
        H = _ToyAttnConfig.hidden_size  # 32
        FF = 64
        self.q_proj = nn.Linear(H, H, bias=True)
        self.k_proj = nn.Linear(H, H, bias=True)
        self.v_proj = nn.Linear(H, H, bias=True)
        self.o_proj = nn.Linear(H, H, bias=True)
        self.gate_proj = nn.Linear(H, FF, bias=True)
        self.up_proj = nn.Linear(H, FF, bias=True)
        self.down_proj = nn.Linear(FF, H, bias=True)
        self.config = _ToyAttnConfig()

    def forward(self, x):
        # Not a real attention block; just exercises every Linear so we can
        # check that converted outputs match end-to-end.
        attn = self.o_proj(
            self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
        )
        ff = self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))
        return attn + ff


def _make_models(device, dtype):
    """Return two deep-copied models built from the same RNG state."""
    torch.manual_seed(0)
    base = _ToyAttnMLPModel().to(device=device, dtype=dtype)
    return copy.deepcopy(base), copy.deepcopy(base)


def _materialized_weights(model):
    """Return {module_name: dense_weight} for every factored module."""
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, (BTTLayer, SVDLayer, BTTLinear, SVDCompressedLinear)):
            out[name] = module.materialize_dense_weight().detach().clone()
    return out


def _assert_dict_close(test, a, b, *, atol, rtol):
    test.assertEqual(set(a.keys()), set(b.keys()))
    for name in sorted(a.keys()):
        ta, tb = a[name], b[name]
        test.assertEqual(ta.shape, tb.shape, msg=f"shape mismatch on {name}")
        test.assertTrue(
            torch.allclose(ta, tb, atol=atol, rtol=rtol),
            msg=(
                f"{name} mismatch: max_abs_diff="
                f"{(ta.float() - tb.float()).abs().max().item():.6g}"
            ),
        )


@unittest.skipUnless(torch.cuda.is_available(), "BTT/SVD conversion requires CUDA")
class TestRunRLNewBTTEquivalence(unittest.TestCase):
    """Plain BTT path: btt_layer.convert_linear_to_btt vs
    compress_integration.convert_linear_to_btt_compress."""

    def setUp(self):
        self.device = "cuda"
        self.dtype = torch.float32  # keep precision tight
        self.atol = 1e-6
        self.rtol = 1e-6

    def _run_btt_pair(self, *, decomp_mode, train_position, s_merged_to,
                      factorize_by_head, convert_mode="svd"):
        legacy_model, compress_model = _make_models(self.device, self.dtype)

        include_names = ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj")

        # Legacy path
        convert_linear_to_btt(
            legacy_model,
            btt_rank="full",
            decomp_mode=decomp_mode,
            init_mode="default",
            include_names=include_names,
            skip_names=("lm_head",),
            lr_act=False,
            s_merged_to=s_merged_to,
            train_position=train_position,
            factorize_by_head=factorize_by_head,
            model_config=legacy_model.config,
            convert_mode=convert_mode,
        )
        configure_blocktt_trainability(
            legacy_model,
            train_bias=True,
            train_position=train_position,
            train_singular_values=(s_merged_to == "keep_trainable"),
        )

        # New unified path
        convert_linear_to_btt_compress(
            compress_model,
            btt_rank="full",
            decomp_mode=decomp_mode,
            init_mode="default",
            include_names=include_names,
            skip_names=("lm_head",),
            lr_act=False,
            s_merged_to=s_merged_to,
            train_position=train_position,
            factorize_by_head=factorize_by_head,
            model_config=compress_model.config,
            convert_mode=convert_mode,
        )
        configure_compress_btt_trainability(
            compress_model,
            train_bias=True,
            train_position=train_position,
            train_singular_values=(s_merged_to == "keep_trainable"),
        )
        return legacy_model, compress_model

    def _assert_pair_identical(self, legacy_model, compress_model):
        # Module-type sanity
        legacy_types = {type(m).__name__ for n, m in legacy_model.named_modules()
                        if isinstance(m, (BTTLayer, SVDLayer))}
        compress_types = {type(m).__name__ for n, m in compress_model.named_modules()
                          if isinstance(m, (BTTLinear, SVDCompressedLinear))}
        self.assertIn("BTTLayer", legacy_types)
        self.assertIn("BTTLinear", compress_types)

        # 1) Initial materialized weights must match.
        _assert_dict_close(
            self,
            _materialized_weights(legacy_model),
            _materialized_weights(compress_model),
            atol=self.atol, rtol=self.rtol,
        )

        # 2) Forward output on random input must match.
        torch.manual_seed(123)
        x = torch.randn(2, 5, _ToyAttnConfig.hidden_size, device=self.device, dtype=self.dtype)
        y_legacy = legacy_model(x).detach()
        y_compress = compress_model(x).detach()
        self.assertTrue(
            torch.allclose(y_legacy, y_compress, atol=self.atol, rtol=self.rtol),
            msg=(
                f"forward output mismatch: max_abs_diff="
                f"{(y_legacy - y_compress).abs().max().item():.6g}"
            ),
        )

        # 3) One AdamW step on identical (synthetic) targets — gradients
        # flow through each parametrization but the materialized weights
        # should match again.
        opt_legacy = torch.optim.AdamW(
            [p for p in legacy_model.parameters() if p.requires_grad],
            lr=1e-3,
        )
        opt_compress = torch.optim.AdamW(
            [p for p in compress_model.parameters() if p.requires_grad],
            lr=1e-3,
        )
        torch.manual_seed(7)
        target = torch.randn(2, 5, _ToyAttnConfig.hidden_size,
                             device=self.device, dtype=self.dtype)
        loss_legacy = (legacy_model(x) - target).pow(2).mean()
        loss_compress = (compress_model(x) - target).pow(2).mean()
        self.assertTrue(
            torch.allclose(loss_legacy, loss_compress, atol=self.atol, rtol=self.rtol),
            msg=f"loss mismatch: {loss_legacy.item()} vs {loss_compress.item()}",
        )
        opt_legacy.zero_grad()
        opt_compress.zero_grad()
        loss_legacy.backward()
        loss_compress.backward()
        opt_legacy.step()
        opt_compress.step()

        _assert_dict_close(
            self,
            _materialized_weights(legacy_model),
            _materialized_weights(compress_model),
            atol=1e-5, rtol=1e-5,
        )

    def test_btt_input_one_block_keep_trainable(self):
        """Project default config for fura/qfura."""
        legacy, compress = self._run_btt_pair(
            decomp_mode="output_one_block",
            train_position="small",
            s_merged_to="keep_trainable",
            factorize_by_head=True,
        )
        self._assert_pair_identical(legacy, compress)

    def test_btt_input_one_block_split(self):
        legacy, compress = self._run_btt_pair(
            decomp_mode="input_one_block",
            train_position="both",
            s_merged_to="split",
            factorize_by_head=False,
        )
        self._assert_pair_identical(legacy, compress)

    def test_btt_square_frozen_no_head_factorize(self):
        legacy, compress = self._run_btt_pair(
            decomp_mode="square",
            train_position="small",
            s_merged_to="frozen",
            factorize_by_head=False,
        )
        self._assert_pair_identical(legacy, compress)

    def test_btt_square_qr_convert_mode(self):
        legacy, compress = self._run_btt_pair(
            decomp_mode="square",
            train_position="small",
            s_merged_to=None,
            factorize_by_head=False,
            convert_mode="qr",
        )
        self._assert_pair_identical(legacy, compress)


@unittest.skipUnless(torch.cuda.is_available(), "SVD conversion requires CUDA")
class TestRunRLNewSVDEquivalence(unittest.TestCase):
    """Plain SVD path: svd_layer.convert_linear_to_svd vs
    compress_integration.convert_linear_to_svd_compress."""

    def setUp(self):
        self.device = "cuda"
        self.dtype = torch.float32
        self.atol = 1e-6
        self.rtol = 1e-6

    def _run_svd_pair(self, *, train_position, s_merged_to):
        legacy_model, compress_model = _make_models(self.device, self.dtype)
        include_names = ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj")

        convert_linear_to_svd(
            legacy_model,
            skip_names=("lm_head",),
            include_names=include_names,
            s_merged_to=s_merged_to,
            train_position=train_position,
        )
        configure_svd_trainability(
            legacy_model,
            train_position=train_position,
            train_bias=True,
            train_embed_lm_head=False,
            train_singular_values=(s_merged_to == "keep_trainable"),
        )

        convert_linear_to_svd_compress(
            compress_model,
            skip_names=("lm_head",),
            include_names=include_names,
            s_merged_to=s_merged_to,
            train_position=train_position,
        )
        configure_compress_svd_trainability(
            compress_model,
            train_position=train_position,
            train_bias=True,
            train_embed_lm_head=False,
            train_singular_values=(s_merged_to == "keep_trainable"),
        )
        return legacy_model, compress_model

    def _assert_pair_identical(self, legacy_model, compress_model):
        # 1) Initial materialized weights.
        _assert_dict_close(
            self,
            _materialized_weights(legacy_model),
            _materialized_weights(compress_model),
            atol=self.atol, rtol=self.rtol,
        )

        # 2) Forward output on random input.
        torch.manual_seed(123)
        x = torch.randn(2, 5, _ToyAttnConfig.hidden_size,
                        device=self.device, dtype=self.dtype)
        y_legacy = legacy_model(x).detach()
        y_compress = compress_model(x).detach()
        self.assertTrue(
            torch.allclose(y_legacy, y_compress, atol=self.atol, rtol=self.rtol),
            msg=(
                f"forward output mismatch: max_abs_diff="
                f"{(y_legacy - y_compress).abs().max().item():.6g}"
            ),
        )

        # 3) One AdamW step. AdamW is elementwise so the transposed
        # parametrization (V_r = svd_b.T, U_r = svd_a.T) yields identical
        # materialized weights after the update.
        opt_legacy = torch.optim.AdamW(
            [p for p in legacy_model.parameters() if p.requires_grad], lr=1e-3,
        )
        opt_compress = torch.optim.AdamW(
            [p for p in compress_model.parameters() if p.requires_grad], lr=1e-3,
        )
        torch.manual_seed(7)
        target = torch.randn(2, 5, _ToyAttnConfig.hidden_size,
                             device=self.device, dtype=self.dtype)
        loss_legacy = (legacy_model(x) - target).pow(2).mean()
        loss_compress = (compress_model(x) - target).pow(2).mean()
        self.assertTrue(
            torch.allclose(loss_legacy, loss_compress, atol=self.atol, rtol=self.rtol),
            msg=f"loss mismatch: {loss_legacy.item()} vs {loss_compress.item()}",
        )
        opt_legacy.zero_grad()
        opt_compress.zero_grad()
        loss_legacy.backward()
        loss_compress.backward()
        opt_legacy.step()
        opt_compress.step()

        _assert_dict_close(
            self,
            _materialized_weights(legacy_model),
            _materialized_weights(compress_model),
            atol=1e-5, rtol=1e-5,
        )

    def test_svd_output_keep_trainable(self):
        legacy, compress = self._run_svd_pair(
            train_position="output", s_merged_to="keep_trainable",
        )
        self._assert_pair_identical(legacy, compress)

    def test_svd_output_split(self):
        legacy, compress = self._run_svd_pair(
            train_position="output", s_merged_to="split",
        )
        self._assert_pair_identical(legacy, compress)

    def test_svd_input_frozen(self):
        legacy, compress = self._run_svd_pair(
            train_position="input", s_merged_to="frozen",
        )
        self._assert_pair_identical(legacy, compress)


if __name__ == "__main__":
    unittest.main()
