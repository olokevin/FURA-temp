import unittest

import torch
import torch.nn as nn

from svft_layer import (
    SVFTLayer,
    _build_omega_indices,
    configure_svft_trainability,
    convert_linear_to_svft,
    get_svft_target_module_names,
)

_HAS_CUDA = torch.cuda.is_available()
_SKIP_CUDA = "SVFT init requires CUDA (torch.linalg.svd on GPU)."


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(16, 24, bias=False)
        self.down_proj = nn.Linear(24, 16, bias=False)
        self.o_proj = nn.Linear(16, 16, bias=True)
        self.other = nn.Linear(16, 16, bias=True)


class TestSVFTPatternIndices(unittest.TestCase):
    """Sparsity-pattern index construction is pure-CPU (no SVD needed)."""

    def test_plain_is_diagonal(self):
        idx = _build_omega_indices(10, "plain", 0, 0, 0, torch.device("cpu"))
        self.assertEqual(idx.shape[1], 10)
        self.assertTrue(torch.equal(idx[0], idx[1]))  # row == col => diagonal

    def test_banded_count_matches_formula(self):
        k, d = 10, 3
        idx = _build_omega_indices(k, "banded", d, 0, 0, torch.device("cpu"))
        expected = k + sum(2 * (k - off) for off in range(1, d + 1))
        self.assertEqual(idx.shape[1], expected)
        # All entries within the band |row - col| <= d.
        self.assertTrue((idx[0] - idx[1]).abs().max().item() <= d)

    def test_random_count_and_uniqueness(self):
        k = 12
        idx = _build_omega_indices(k, "random", 0, 40, 123, torch.device("cpu"))
        self.assertEqual(idx.shape[1], 40)
        flat = idx[0] * k + idx[1]
        self.assertEqual(flat.unique().numel(), 40)  # unique (row, col) pairs

    def test_random_is_deterministic_in_seed(self):
        a = _build_omega_indices(12, "random", 0, 30, 7, torch.device("cpu"))
        b = _build_omega_indices(12, "random", 0, 30, 7, torch.device("cpu"))
        self.assertTrue(torch.equal(a, b))


class TestSVFTTargetModules(unittest.TestCase):
    def test_udog_default(self):
        self.assertEqual(
            set(get_svft_target_module_names("udog")),
            {"up_proj", "down_proj", "o_proj", "gate_proj"},
        )

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            get_svft_target_module_names("bogus")


@unittest.skipUnless(_HAS_CUDA, _SKIP_CUDA)
class TestSVFTPipelineCompat(unittest.TestCase):
    def _mk_layer(self, di, do, **kw):
        torch.manual_seed(0)
        lin = nn.Linear(di, do, bias=True).cuda().to(torch.float32)
        layer = SVFTLayer(di, do, bias=True, **kw).cuda().to(torch.float32)
        layer.init_from_linear_weight(lin.weight.data, lin.bias.data)
        return lin, layer

    def test_zero_init_reconstructs_base_weight(self):
        for di, do in [(16, 16), (24, 16), (16, 24)]:
            lin, layer = self._mk_layer(di, do, pattern="banded", off_diag=4)
            rec = layer.materialize_dense_weight()
            self.assertLess((rec - lin.weight.data).abs().max().item(), 1e-4)

    def test_forward_matches_materialized(self):
        lin, layer = self._mk_layer(16, 24, pattern="banded", off_diag=4)
        with torch.no_grad():
            layer.m_coeffs.add_(0.05 * torch.randn_like(layer.m_coeffs))
        x = torch.randn(4, 16, device="cuda")
        w = layer.materialize_dense_weight()
        y_ref = torch.nn.functional.linear(x, w, layer.bias)
        self.assertTrue(torch.allclose(layer(x), y_ref, atol=1e-4, rtol=1e-4))

    def test_convert_and_trainability(self):
        model = _ToyModel().cuda().to(torch.float32)
        converted = convert_linear_to_svft(
            model,
            include_names=("up_proj", "down_proj", "o_proj"),
            pattern="banded",
            off_diag=2,
        )
        self.assertEqual(set(converted), {"up_proj", "down_proj", "o_proj"})
        self.assertIsInstance(model.up_proj, SVFTLayer)
        self.assertIsInstance(model.other, nn.Linear)  # not converted

        stats = configure_svft_trainability(model, train_bias=True)
        self.assertEqual(stats["num_svft_layers"], 3)
        # Only m_coeffs (+ biases) are trainable; U/V/s are frozen buffers.
        self.assertTrue(model.up_proj.m_coeffs.requires_grad)
        self.assertTrue(model.o_proj.bias.requires_grad)

    def test_plain_param_count_equals_k_per_layer(self):
        model = _ToyModel().cuda().to(torch.float32)
        convert_linear_to_svft(model, include_names=("o_proj",), pattern="plain")
        configure_svft_trainability(model, train_bias=False)
        # plain => k = min(16, 16) = 16 coefficients.
        self.assertEqual(model.o_proj.m_coeffs.numel(), 16)


if __name__ == "__main__":
    unittest.main()
