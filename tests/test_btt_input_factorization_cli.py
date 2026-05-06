"""Tests for --blocktt_input_factorization CLI plumbing.

Verifies that the three Llama-3-8B shape-ablation cells used by
launchers/shape_ablation_llama3_8b.sh produce the expected (n, b) on every
target module under decomp_mode=output_one_block.
"""
import types
import unittest

import torch
import torch.nn as nn

from btt_layer import (
    BTTLayer,
    _parse_input_factorization_arg,
    _parse_input_factorization_pair,
    _resolve_input_factorization_for_module,
    convert_linear_to_btt,
)


# Llama-3-8B-shaped Linear stack in a fake decoder layer.
LLAMA3_8B_SHAPES = {
    "q_proj": (4096, 4096),
    "k_proj": (1024, 4096),     # GQA: 8 kv-heads * 128
    "v_proj": (1024, 4096),
    "o_proj": (4096, 4096),
    "gate_proj": (14336, 4096),
    "up_proj": (14336, 4096),
    "down_proj": (4096, 14336),
}

LLAMA3_8B_FAKE_CONFIG = types.SimpleNamespace(
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    hidden_size=4096,
)


def _make_fake_decoder():
    """Build a tiny stand-in for a Llama decoder layer with the right Linear
    submodule names and shapes."""
    block = nn.Module()
    attn = nn.Module()
    mlp = nn.Module()
    for leaf in ("q_proj", "k_proj", "v_proj", "o_proj"):
        out, in_ = LLAMA3_8B_SHAPES[leaf]
        setattr(attn, leaf, nn.Linear(in_, out, bias=False))
    for leaf in ("gate_proj", "up_proj", "down_proj"):
        out, in_ = LLAMA3_8B_SHAPES[leaf]
        setattr(mlp, leaf, nn.Linear(in_, out, bias=False))
    block.self_attn = attn
    block.mlp = mlp
    return block


def _collect_btt_shapes(model):
    """Return {leaf_name: (n, b)} for every BTTLayer in `model`."""
    shapes = {}
    for name, mod in model.named_modules():
        if isinstance(mod, BTTLayer):
            leaf = name.split(".")[-1]
            shapes[leaf] = (mod.n, mod.b)
    return shapes


class TestParseInputFactorizationPair(unittest.TestCase):
    def test_sentinels(self):
        self.assertEqual(_parse_input_factorization_pair("head"), "head")
        self.assertEqual(_parse_input_factorization_pair("closest"), "closest")
        self.assertEqual(_parse_input_factorization_pair("CLOSEST"), "closest")

    def test_tuple_str(self):
        self.assertEqual(_parse_input_factorization_pair("32,128"), (32, 128))
        self.assertEqual(_parse_input_factorization_pair("(32, 128)"), (32, 128))
        self.assertEqual(_parse_input_factorization_pair("4096 1"), (4096, 1))

    def test_tuple_obj(self):
        self.assertEqual(_parse_input_factorization_pair((32, 128)), (32, 128))
        self.assertEqual(_parse_input_factorization_pair([32, 128]), (32, 128))

    def test_none_passthrough(self):
        self.assertIsNone(_parse_input_factorization_pair(None))

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            _parse_input_factorization_pair("nope")
        with self.assertRaises(ValueError):
            _parse_input_factorization_pair("1,2,3")


class TestParseInputFactorizationArg(unittest.TestCase):
    def test_none(self):
        self.assertIsNone(_parse_input_factorization_arg(None))

    def test_scalar(self):
        self.assertEqual(_parse_input_factorization_arg("head"), "head")
        self.assertEqual(_parse_input_factorization_arg("closest"), "closest")
        self.assertEqual(_parse_input_factorization_arg("32,128"), (32, 128))

    def test_dict_groups(self):
        spec = '{"qkv":"head","o":"head","mlp_upgate":"closest","mlp_down":"closest"}'
        out = _parse_input_factorization_arg(spec)
        self.assertEqual(out["q_proj"], "head")
        self.assertEqual(out["k_proj"], "head")
        self.assertEqual(out["v_proj"], "head")
        self.assertEqual(out["o_proj"], "head")
        self.assertEqual(out["gate_proj"], "closest")
        self.assertEqual(out["up_proj"], "closest")
        self.assertEqual(out["down_proj"], "closest")

    def test_dict_explicit_pairs(self):
        spec = '{"qkv":"512,8","o":"512,8","mlp_upgate":"512,8","mlp_down":"1792,8"}'
        out = _parse_input_factorization_arg(spec)
        self.assertEqual(out["q_proj"], (512, 8))
        self.assertEqual(out["down_proj"], (1792, 8))

    def test_unknown_group_treated_as_leaf(self):
        spec = '{"down_proj":"1792,8"}'
        out = _parse_input_factorization_arg(spec)
        self.assertEqual(out["down_proj"], (1792, 8))


class TestResolveInputFactorizationForModule(unittest.TestCase):
    def test_validates_product(self):
        with self.assertRaises(ValueError):
            _resolve_input_factorization_for_module(
                {"q_proj": (32, 100)}, "q_proj",
                in_features=4096, head_factorization=(32, 128),
            )

    def test_head_falls_back_when_no_head_info(self):
        # mlp module: no head_factorization is defined, "head" should defer to
        # the BTTLayer fallback (None -> closest_pair).
        out = _resolve_input_factorization_for_module(
            {"down_proj": "head"}, "down_proj",
            in_features=14336, head_factorization=None,
        )
        self.assertEqual(out, "head")

    def test_module_not_in_dict_returns_none(self):
        # If the dict doesn't mention this module, we fall back to
        # factorize_by_head behavior.
        out = _resolve_input_factorization_for_module(
            {"q_proj": "head"}, "down_proj",
            in_features=14336, head_factorization=None,
        )
        self.assertIsNone(out)


class TestLlama3_8BCellShapes(unittest.TestCase):
    """End-to-end: convert a fake Llama-3-8B decoder block via
    convert_linear_to_btt and verify (n, b) per module for each cell."""

    def setUp(self):
        # CPU init via allow_non_cuda; rank=1 to keep params tiny.
        self.kwargs = dict(
            btt_rank=1,
            decomp_mode="output_one_block",
            init_mode="default",
            include_names=(
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ),
            skip_names=("lm_head",),
            lr_act=False,
            s_merged_to="keep_trainable",
            train_position="small",
            model_config=LLAMA3_8B_FAKE_CONFIG,
            convert_mode="svd",
            allow_non_cuda=True,
        )

    def test_cell1_head_attn_closest_mlp(self):
        model = _make_fake_decoder()
        spec = '{"qkv":"head","o":"head","mlp_upgate":"closest","mlp_down":"closest"}'
        convert_linear_to_btt(
            model,
            factorize_by_head=False,
            input_factorization=spec,
            **self.kwargs,
        )
        shapes = _collect_btt_shapes(model)
        # q/k/v/o all have din=4096 except k/v whose din is also 4096 (we only
        # care about the input side for output_one_block). Head-aligned:
        # (32, 128) for all attn modules.
        self.assertEqual(shapes["q_proj"], (32, 128))
        self.assertEqual(shapes["k_proj"], (32, 128))
        self.assertEqual(shapes["v_proj"], (32, 128))
        self.assertEqual(shapes["o_proj"], (32, 128))
        # mlp closest: din=4096 -> (64, 64); din=14336 -> (112, 128).
        self.assertEqual(shapes["gate_proj"], (64, 64))
        self.assertEqual(shapes["up_proj"], (64, 64))
        self.assertEqual(shapes["down_proj"], (112, 128))

    def test_cell2_unbalanced_8col(self):
        model = _make_fake_decoder()
        spec = (
            '{"qkv":"512,8","o":"512,8","mlp_upgate":"512,8","mlp_down":"1792,8"}'
        )
        convert_linear_to_btt(
            model,
            factorize_by_head=False,
            input_factorization=spec,
            **self.kwargs,
        )
        shapes = _collect_btt_shapes(model)
        for leaf in ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj"):
            self.assertEqual(shapes[leaf], (512, 8), msg=f"{leaf} mismatch")
        self.assertEqual(shapes["down_proj"], (1792, 8))

    def test_cell3_extreme_b1(self):
        model = _make_fake_decoder()
        spec = (
            '{"qkv":"4096,1","o":"4096,1","mlp_upgate":"4096,1",'
            '"mlp_down":"14336,1"}'
        )
        convert_linear_to_btt(
            model,
            factorize_by_head=False,
            input_factorization=spec,
            **self.kwargs,
        )
        shapes = _collect_btt_shapes(model)
        for leaf in ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj"):
            self.assertEqual(shapes[leaf], (4096, 1), msg=f"{leaf} mismatch")
        self.assertEqual(shapes["down_proj"], (14336, 1))

    def test_legacy_factorize_by_head_unchanged(self):
        """Without input_factorization, factorize_by_head should behave as before:
        only o_proj's input side becomes (32, 128); everything else falls back
        to closest_pair."""
        model = _make_fake_decoder()
        convert_linear_to_btt(
            model,
            factorize_by_head=True,
            input_factorization=None,
            **self.kwargs,
        )
        shapes = _collect_btt_shapes(model)
        self.assertEqual(shapes["o_proj"], (32, 128))
        # q/k/v fall back to closest pair (din=4096) -> (64, 64).
        self.assertEqual(shapes["q_proj"], (64, 64))
        self.assertEqual(shapes["k_proj"], (64, 64))
        self.assertEqual(shapes["v_proj"], (64, 64))
        self.assertEqual(shapes["gate_proj"], (64, 64))
        self.assertEqual(shapes["up_proj"], (64, 64))
        self.assertEqual(shapes["down_proj"], (112, 128))


if __name__ == "__main__":
    unittest.main()
