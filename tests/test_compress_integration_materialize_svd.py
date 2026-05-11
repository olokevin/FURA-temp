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
