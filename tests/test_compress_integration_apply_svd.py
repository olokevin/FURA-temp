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

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Preserve (B, T) so calibration code can shift logits over the seq dim.
        B, T = input_ids.shape
        x = torch.randn(B, T, 16, device=input_ids.device,
                        dtype=next(self.parameters()).dtype)
        b = self.layers[0]
        # Run all projections so calibration covariances are collected for each
        h = b.gate_proj(x) * b.up_proj(x)
        h = b.down_proj(h)
        q = b.q_proj(x)
        k = b.k_proj(x)
        v = b.v_proj(x)
        h = h + b.o_proj(q + k + v)
        logits = self.lm_head(h)  # (B, T, 32)
        loss = logits.float().mean()  # real non-zero loss keeps backward gradients stable
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

    def test_apply_svd_v2_combined_replaces_nn_linear(self):
        # svd_v2_combined collects BOTH forward (activation) and backward
        # (gradient) covariances. The TinyModel.forward already constructs a
        # differentiable graph from input -> logits; the zero-loss trick is
        # documented in the loader fixture and keeps backward stable.
        model = TinyModel()
        args = _parse(["--calib-mode", "svd_v2_combined", "--calib-source", "training_data",
                       "--compression-ratio", "0.5"])
        loader = _ToyCalibLoader()
        out = ci.apply_calibrated_svd(model, args, calib_loader=loader, device="cpu")

        block = out.layers[0]
        for name in ("gate_proj", "up_proj", "down_proj", "q_proj",
                     "k_proj", "v_proj", "o_proj"):
            mod = getattr(block, name)
            self.assertIsInstance(mod, SVDCompressedLinear,
                                  f"{name} should be SVDCompressedLinear after svd_v2_combined")
            self.assertTrue(mod.U_r.requires_grad)
            self.assertTrue(mod.V_r.requires_grad)


if __name__ == "__main__":
    unittest.main()
