import unittest

import torch
import torch.nn as nn

from spectral_layer import (
    SpectralLayer,
    cayley,
    configure_spectral_trainability,
    convert_linear_to_spectral,
    get_spectral_target_module_names,
)

_HAS_CUDA = torch.cuda.is_available()
_SKIP = "Spectral init requires CUDA (torch.linalg.svd on GPU)."


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 32, bias=False)
        self.v_proj = nn.Linear(32, 16, bias=False)  # rectangular
        self.other = nn.Linear(32, 32, bias=False)


class TestCayleyCPU(unittest.TestCase):
    def test_cayley_orthogonal(self):
        torch.manual_seed(0)
        A = torch.randn(10, 10)
        Q = cayley(A)
        self.assertTrue(torch.allclose(Q @ Q.T, torch.eye(10), atol=1e-5))

    def test_cayley_zero_is_identity(self):
        Q = cayley(torch.zeros(6, 6))
        self.assertTrue(torch.allclose(Q, torch.eye(6), atol=1e-6))


class TestSpectralTargets(unittest.TestCase):
    def test_pissa5_default(self):
        self.assertEqual(
            set(get_spectral_target_module_names("pissa5")),
            {"q_proj", "k_proj", "v_proj", "up_proj", "down_proj"},
        )

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            get_spectral_target_module_names("bogus")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            SpectralLayer(16, 16, rank=4, mode="bogus")


@unittest.skipUnless(_HAS_CUDA, _SKIP)
class TestSpectralPipelineCompat(unittest.TestCase):
    def _mk(self, di, do, mode, rank=8):
        torch.manual_seed(0)
        lin = nn.Linear(di, do, bias=True).cuda().to(torch.float32)
        layer = SpectralLayer(di, do, rank=rank, mode=mode, bias=True).cuda().to(torch.float32)
        layer.init_from_linear_weight(lin.weight.data, lin.bias.data)
        return lin, layer

    def test_zero_init_reconstructs_weight_both_modes(self):
        for mode in ("additive", "rotation"):
            for di, do in [(32, 32), (32, 16), (16, 32)]:
                lin, layer = self._mk(di, do, mode)
                rec = layer.materialize_dense_weight()
                self.assertLess(
                    (rec - lin.weight.data).abs().max().item(), 1e-4,
                    f"{mode} {di}x{do}",
                )

    def test_forward_matches_materialized_after_perturb(self):
        for mode in ("additive", "rotation"):
            lin, layer = self._mk(32, 16, mode)
            with torch.no_grad():
                for p in layer.parameters():
                    if p.requires_grad:
                        p.add_(0.03 * torch.randn_like(p))
            x = torch.randn(4, 32, device="cuda")
            w = layer.materialize_dense_weight()
            y_ref = torch.nn.functional.linear(x, w, layer.bias)
            self.assertTrue(torch.allclose(layer(x), y_ref, atol=1e-4, rtol=1e-4), mode)

    def test_convert_and_trainability_additive(self):
        model = _ToyModel().cuda().to(torch.float32)
        conv = convert_linear_to_spectral(
            model, rank=8, mode="additive", include_names=("q_proj", "v_proj")
        )
        self.assertEqual(set(conv), {"q_proj", "v_proj"})
        self.assertIsInstance(model.q_proj, SpectralLayer)
        self.assertIsInstance(model.other, nn.Linear)
        stats = configure_spectral_trainability(model, train_bias=False)
        self.assertEqual(stats["num_spectral_layers"], 2)
        self.assertTrue(model.q_proj.delta_u.requires_grad)
        self.assertTrue(model.q_proj.delta_v.requires_grad)
        # frozen SVD lives in buffers
        self.assertFalse(model.q_proj.spec_u.requires_grad)

    def test_convert_and_trainability_rotation(self):
        model = _ToyModel().cuda().to(torch.float32)
        convert_linear_to_spectral(model, rank=8, mode="rotation", include_names=("q_proj",))
        stats = configure_spectral_trainability(model, train_bias=False)
        self.assertEqual(stats["num_spectral_layers"], 1)
        self.assertTrue(model.q_proj.rot_a.requires_grad)
        self.assertTrue(model.q_proj.rot_b.requires_grad)

    def test_rank_capped_by_min_dim(self):
        # v_proj is 32x16 -> rank capped at 16 even if 64 requested.
        _, layer = self._mk(32, 16, "additive", rank=64)
        self.assertEqual(layer.rank, 16)
        self.assertEqual(layer.delta_u.shape, (16, 16))
        self.assertEqual(layer.delta_v.shape, (32, 16))


if __name__ == "__main__":
    unittest.main()
