import unittest

import torch
import torch.nn as nn

from quanta_layer import (
    QuanTALayer,
    _gen_quanta_apply_expr,
    configure_quanta_trainability,
    convert_linear_to_quanta,
    get_quanta_target_module_names,
    parse_dims,
)

_HAS_CUDA = torch.cuda.is_available()


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)  # rectangular (GQA-like)
        self.other = nn.Linear(8, 8, bias=False)


class TestQuanTAConfig(unittest.TestCase):
    def test_parse_dims(self):
        self.assertEqual(parse_dims("16-8-8-4"), [16, 8, 8, 4])

    def test_num_tensors_per_pair(self):
        # N axes -> exactly N*(N-1)/2 tensors (one per unordered pair).
        for N in (3, 4, 5):
            _, pairs = _gen_quanta_apply_expr(N)
            self.assertEqual(len(pairs), N * (N - 1) // 2)

    def test_target_modules(self):
        self.assertEqual(set(get_quanta_target_module_names("qv")), {"q_proj", "v_proj"})
        with self.assertRaises(ValueError):
            get_quanta_target_module_names("bogus")


class TestQuanTALayerCPU(unittest.TestCase):
    """QuanTA math is device-agnostic; test the layer directly on CPU."""

    def _mk(self, di, do, dims):
        torch.manual_seed(0)
        lin = nn.Linear(di, do, bias=True)
        layer = QuanTALayer(di, do, dims=dims, bias=True)
        layer.init_from_linear_weight(lin.weight.data, lin.bias.data)
        return lin, layer

    def test_zero_update_at_init(self):
        # T == S at init => (T - S)x == 0 => layer reconstructs W0 exactly.
        for di, do in [(8, 8), (4, 8), (8, 4), (6, 8)]:
            lin, layer = self._mk(di, do, [2, 2, 2])
            rec = layer.materialize_dense_weight()
            self.assertLess((rec - lin.weight.data).abs().max().item(), 1e-5)
            x = torch.randn(3, di)
            self.assertTrue(torch.allclose(layer(x), lin(x), atol=1e-5))

    def test_nonuniform_dims_forward(self):
        # Regression: with non-uniform axis sizes the per-tensor axis order matters.
        # dims=[4,2,3] -> total 24; test square (24) and rectangular (12) outputs.
        for di, do in [(24, 24), (24, 12), (12, 24)]:
            lin, layer = self._mk(di, do, [4, 2, 3])
            rec = layer.materialize_dense_weight()
            self.assertLess((rec - lin.weight.data).abs().max().item(), 1e-5)
            with torch.no_grad():
                for t in layer.t_tensors:
                    t.add_(0.05 * torch.randn_like(t))
            x = torch.randn(4, di)
            w = layer.materialize_dense_weight()
            y_ref = torch.nn.functional.linear(x, w, layer.bias)
            self.assertTrue(torch.allclose(layer(x), y_ref, atol=1e-4, rtol=1e-4))

    def test_forward_matches_materialized_after_perturbation(self):
        lin, layer = self._mk(8, 8, [2, 2, 2])
        with torch.no_grad():
            for t in layer.t_tensors:
                t.add_(0.1 * torch.randn_like(t))
        x = torch.randn(5, 8)
        w = layer.materialize_dense_weight()
        y_ref = torch.nn.functional.linear(x, w, layer.bias)
        self.assertTrue(torch.allclose(layer(x), y_ref, atol=1e-4, rtol=1e-4))

    def test_gradients_flow_to_tensors_only(self):
        lin, layer = self._mk(8, 8, [2, 2, 2])
        configure_quanta_trainability_single(layer)
        x = torch.randn(4, 8)
        layer(x).pow(2).sum().backward()
        for t in layer.t_tensors:
            self.assertIsNotNone(t.grad)
        # Frozen base weight must not accumulate grad.
        self.assertFalse(layer.weight.requires_grad)


def configure_quanta_trainability_single(layer):
    for p in layer.parameters():
        p.requires_grad = False
    for t in layer.t_tensors:
        t.requires_grad = True


@unittest.skipUnless(_HAS_CUDA, "convert path requires CUDA weights")
class TestQuanTAConvertCUDA(unittest.TestCase):
    def test_convert_and_trainability(self):
        model = _ToyModel().cuda().to(torch.float32)
        converted = convert_linear_to_quanta(
            model, dims=[2, 2, 2], include_names=("q_proj", "v_proj")
        )
        self.assertEqual(set(converted), {"q_proj", "v_proj"})
        self.assertIsInstance(model.q_proj, QuanTALayer)
        self.assertIsInstance(model.other, nn.Linear)

        stats = configure_quanta_trainability(model, train_bias=False)
        self.assertEqual(stats["num_quanta_layers"], 2)
        self.assertTrue(all(t.requires_grad for t in model.q_proj.t_tensors))
        # frozen S copies live in buffers, not parameters
        self.assertFalse(model.q_proj.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
