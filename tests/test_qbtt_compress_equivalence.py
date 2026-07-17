"""Equivalence tests for the qfura (NF4 frozen BTT core) port.

Compares the legacy ``btt_layer.QBTTLayer`` path against the new
``compress.btt.qbtt_linear.QBTTLinear`` path on:

  1. Materialized dense weight after NF4 quantization.
  2. Forward output on a random input.
  3. Post-AdamW-step materialized weight (the frozen NF4 side is byte-identical
     across steps; the trainable bf16 side must move identically).

Tests require CUDA + bitsandbytes (NF4 quantization is a CUDA-only path).
"""

import copy
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

try:
    import bitsandbytes  # noqa: F401
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False

# Legacy
from btt_layer import (
    BTTLayer,
    QBTTLayer,
    configure_blocktt_trainability,
    convert_linear_to_btt,
    quantize_frozen_core_ as legacy_quantize_frozen_core_,
    convert_btt_to_qbtt_ as legacy_convert_btt_to_qbtt_,
)
# Compress
from compress.btt.btt_linear import BTTLinear
from compress.btt.qbtt_linear import (
    QBTTLinear,
    quantize_frozen_core_ as compress_quantize_frozen_core_,
    convert_btt_to_qbtt_ as compress_convert_btt_to_qbtt_,
)
from compress.integration import (
    convert_linear_to_btt_compress,
    configure_compress_btt_trainability,
    convert_and_quantize_linear_to_qbtt_streaming,
)


class _ToyAttnConfig:
    num_attention_heads = 4
    num_key_value_heads = 4
    head_dim = 8
    hidden_size = 32


class _ToyAttnMLPModel(nn.Module):
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
        attn = self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))
        ff = self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))
        return attn + ff


def _make_models(device, dtype):
    torch.manual_seed(0)
    base = _ToyAttnMLPModel().to(device=device, dtype=dtype)
    return copy.deepcopy(base), copy.deepcopy(base)


def _materialized_weights(model):
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, (BTTLayer, BTTLinear)):  # BTTLinear covers QBTTLinear
            out[name] = module.materialize_dense_weight().detach().clone()
    return out


def _assert_dict_close(test, a, b, *, atol, rtol, label):
    test.assertEqual(set(a.keys()), set(b.keys()))
    for name in sorted(a.keys()):
        ta, tb = a[name], b[name]
        test.assertEqual(ta.shape, tb.shape, msg=f"[{label}] shape mismatch on {name}")
        test.assertTrue(
            torch.allclose(ta, tb, atol=atol, rtol=rtol),
            msg=(
                f"[{label}] {name} mismatch: max_abs_diff="
                f"{(ta.float() - tb.float()).abs().max().item():.6g}"
            ),
        )


@unittest.skipUnless(torch.cuda.is_available() and _HAS_BNB,
                     "qfura requires CUDA + bitsandbytes")
class TestQBTTCompressEquivalence(unittest.TestCase):
    """Confirms compress.QBTTLinear behaves identically to the legacy
    btt_layer.QBTTLayer for every documented qfura config."""

    def setUp(self):
        self.device = "cuda"
        self.dtype = torch.bfloat16  # qfura's production dtype
        # NF4 quantization introduces ~5-15% relative error per block — both
        # paths inherit it, so they should agree to bf16 precision.
        self.atol = 1e-3
        self.rtol = 1e-3

    def _build_quantized_pair(self, *, decomp_mode, train_position, s_merged_to,
                              factorize_by_head, quant_layout, convert_mode="svd"):
        legacy_model, compress_model = _make_models(self.device, self.dtype)
        include_names = ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj")

        # Legacy path: BTTLayer -> QBTTLayer
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
        legacy_stats = legacy_convert_btt_to_qbtt_(legacy_model, layout=quant_layout)
        self.assertGreater(legacy_stats["num_converted"], 0)

        # Compress path: BTTLinear -> QBTTLinear
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
        compress_stats = compress_convert_btt_to_qbtt_(compress_model, layout=quant_layout)
        self.assertEqual(compress_stats["num_converted"], legacy_stats["num_converted"])

        # Sanity: every targeted module ended up quantized.
        n_qlegacy = sum(1 for _, m in legacy_model.named_modules() if isinstance(m, QBTTLayer))
        n_qcompress = sum(1 for _, m in compress_model.named_modules() if isinstance(m, QBTTLinear))
        self.assertEqual(n_qlegacy, n_qcompress)
        self.assertEqual(n_qlegacy, len(include_names))

        return legacy_model, compress_model

    def _assert_pair_identical(self, legacy_model, compress_model):
        # 1) Materialized weights after quantization.
        _assert_dict_close(
            self,
            _materialized_weights(legacy_model),
            _materialized_weights(compress_model),
            atol=self.atol, rtol=self.rtol,
            label="post-quant materialize",
        )

        # 2) Forward output.
        torch.manual_seed(123)
        x = torch.randn(
            2, 5, _ToyAttnConfig.hidden_size,
            device=self.device, dtype=self.dtype,
        )
        y_legacy = legacy_model(x).detach()
        y_compress = compress_model(x).detach()
        max_abs = (y_legacy - y_compress).abs().max().item()
        self.assertTrue(
            torch.allclose(y_legacy, y_compress, atol=self.atol, rtol=self.rtol),
            msg=f"forward output mismatch: max_abs_diff={max_abs:.6g}",
        )

        # 3) One AdamW step. The frozen NF4 side is byte-identical across
        # both paths (same input bytes -> same bnb quantization), so the
        # post-step trainable side must also stay identical under an
        # elementwise optimizer.
        opt_legacy = torch.optim.AdamW(
            [p for p in legacy_model.parameters() if p.requires_grad], lr=1e-3,
        )
        opt_compress = torch.optim.AdamW(
            [p for p in compress_model.parameters() if p.requires_grad], lr=1e-3,
        )
        torch.manual_seed(7)
        target = torch.randn(
            2, 5, _ToyAttnConfig.hidden_size,
            device=self.device, dtype=self.dtype,
        )
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
            atol=self.atol, rtol=self.rtol,
            label="post-AdamW materialize",
        )

    # ------------------------------------------------------------------
    # Quant-conversion cases mirror project default qfura recipes.
    # ------------------------------------------------------------------

    def test_qfura_default_recipe_flat(self):
        """output_one_block + small + keep_trainable + flat layout
        (the project default for qfura)."""
        legacy, compress = self._build_quantized_pair(
            decomp_mode="output_one_block",
            train_position="small",
            s_merged_to="keep_trainable",
            factorize_by_head=True,
            quant_layout="flat",
        )
        self._assert_pair_identical(legacy, compress)

    def test_qfura_default_recipe_per_core_block(self):
        """Same as above but with per_core_block quant layout."""
        legacy, compress = self._build_quantized_pair(
            decomp_mode="output_one_block",
            train_position="small",
            s_merged_to="keep_trainable",
            factorize_by_head=True,
            quant_layout="per_core_block",
        )
        self._assert_pair_identical(legacy, compress)

    def test_qfura_input_one_block_frozen_flat(self):
        """input_one_block + small + frozen (no btt_s) + flat layout."""
        legacy, compress = self._build_quantized_pair(
            decomp_mode="input_one_block",
            train_position="small",
            s_merged_to="frozen",
            factorize_by_head=False,
            quant_layout="flat",
        )
        self._assert_pair_identical(legacy, compress)

    def test_qfura_square_frozen_per_core_block(self):
        """square + small + frozen + per_core_block."""
        legacy, compress = self._build_quantized_pair(
            decomp_mode="square",
            train_position="small",
            s_merged_to="frozen",
            factorize_by_head=False,
            quant_layout="per_core_block",
        )
        self._assert_pair_identical(legacy, compress)


@unittest.skipUnless(torch.cuda.is_available() and _HAS_BNB,
                     "qfura requires CUDA + bitsandbytes")
class TestQBTTCompressStreamingEquivalence(unittest.TestCase):
    """Confirms convert_and_quantize_linear_to_qbtt_streaming produces a
    model that matches the non-streaming compress path module-for-module."""

    def setUp(self):
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.atol = 1e-3
        self.rtol = 1e-3

    def test_streaming_matches_non_streaming(self):
        non_streaming, streaming = _make_models(self.device, self.dtype)
        include_names = ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj")

        # Non-streaming path: convert_linear_to_btt_compress + configure +
        # convert_btt_to_qbtt_.
        convert_linear_to_btt_compress(
            non_streaming,
            btt_rank="full",
            decomp_mode="output_one_block",
            init_mode="default",
            include_names=include_names,
            skip_names=("lm_head",),
            lr_act=False,
            s_merged_to="keep_trainable",
            train_position="small",
            factorize_by_head=True,
            model_config=non_streaming.config,
            convert_mode="svd",
        )
        configure_compress_btt_trainability(
            non_streaming,
            train_bias=True,
            train_position="small",
            train_singular_values=True,
        )
        compress_convert_btt_to_qbtt_(non_streaming, layout="flat")

        # Streaming path: single-shot convert + quantize per layer.
        stats = convert_and_quantize_linear_to_qbtt_streaming(
            streaming,
            btt_rank="full",
            decomp_mode="output_one_block",
            train_position="small",
            s_merged_to="keep_trainable",
            quant_layout="flat",
            target_modules=include_names,
            cuda_device=torch.device(self.device),
            skip_names=("lm_head",),
            factorize_by_head=True,
            convert_mode="svd",
            init_mode="default",
            compute_dtype=self.dtype,
        )
        self.assertEqual(stats["num_converted"], len(include_names))

        _assert_dict_close(
            self,
            _materialized_weights(non_streaming),
            _materialized_weights(streaming),
            atol=self.atol, rtol=self.rtol,
            label="streaming vs non-streaming",
        )

        # Forward parity.
        torch.manual_seed(123)
        x = torch.randn(
            2, 5, _ToyAttnConfig.hidden_size,
            device=self.device, dtype=self.dtype,
        )
        y_a = non_streaming(x).detach()
        y_b = streaming(x).detach()
        self.assertTrue(
            torch.allclose(y_a, y_b, atol=self.atol, rtol=self.rtol),
            msg=(
                f"forward mismatch: max_abs_diff="
                f"{(y_a - y_b).abs().max().item():.6g}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
