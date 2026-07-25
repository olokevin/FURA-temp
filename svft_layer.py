"""SVFT: Singular-Vectors guided Fine-Tuning (Lingam et al., NeurIPS 2024).

Reparametrizes an ``nn.Linear`` weight ``W0`` via its own SVD ``W0 = U S V^T`` and
adds a *structured, sparse* trainable update expressed in the frozen singular basis::

    h = W0 x + Delta_W x = U (S + M) V^T x

where ``U`` (left singular vectors) and ``V`` (right singular vectors) are frozen and
``M`` is a sparse ``k x k`` (``k = min(d_out, d_in)``) trainable coefficient matrix.
Only the coefficients on a fixed sparsity pattern ``Omega`` are learnable; the rest are
held at zero.  This mirrors the design of :mod:`svd_layer` (SVDLayer) so the trainer,
materialization, and eval pipeline can reuse the same plumbing.

Sparsity patterns for ``M`` (paper Sec. 3.1):
    * ``plain``  : M is diagonal (adapts singular values only) -> k trainable params/layer.
    * ``banded`` : M has the main diagonal plus ``off_diag`` bands on each side.
    * ``random`` : a fixed random set of off-diagonal coefficients is learnable.
    * ``topk``   : the top-``k`` |u_i . v_j| alignments are learnable (square W only).

At initialization ``M = 0`` so ``U (S + 0) V^T = W0`` reconstructs the pretrained
weight exactly (the adapter starts as identity), analogous to LoRA/SVD-layer init.
"""

import torch
import torch.nn as nn


VALID_SVFT_PATTERNS = {"plain", "banded", "random", "topk"}


def _raise_if_non_cuda_weight(full_name, weight):
    if weight.is_cuda:
        return
    raise RuntimeError(
        "Linear->SVFT conversion requires CUDA weights so SVD runs on GPU. "
        f"Module '{full_name}' is on device={weight.device}."
    )


def _build_omega_indices(k, pattern, off_diag, num_coeffs, seed, device):
    """Return a LongTensor of shape (2, nnz) with (row, col) indices of learnable M entries.

    ``k``         : side length of the square coefficient matrix M (= min(d_out, d_in)).
    ``pattern``   : one of VALID_SVFT_PATTERNS.
    ``off_diag``  : bandwidth ``d`` on each side of the diagonal (banded only).
    ``num_coeffs``: number of learnable coefficients (random / topk only).
    """
    if pattern == "plain":
        idx = torch.arange(k, device=device)
        return torch.stack([idx, idx], dim=0)

    if pattern == "banded":
        rows = []
        cols = []
        for offset in range(-off_diag, off_diag + 1):
            r0 = max(0, -offset)
            r1 = min(k, k - offset)
            r = torch.arange(r0, r1, device=device)
            c = r + offset
            rows.append(r)
            cols.append(c)
        rows = torch.cat(rows)
        cols = torch.cat(cols)
        return torch.stack([rows, cols], dim=0)

    if pattern == "random":
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        # Sample `num_coeffs` unique (row, col) pairs from the k x k grid.
        n_total = k * k
        num = min(num_coeffs, n_total)
        flat = torch.randperm(n_total, generator=g)[:num]
        rows = (flat // k).to(device)
        cols = (flat % k).to(device)
        return torch.stack([rows, cols], dim=0)

    raise ValueError(f"topk pattern is handled at init time; unexpected pattern={pattern}")


class SVFTLayer(nn.Module):
    """Drop-in replacement for ``nn.Linear`` implementing the SVFT update.

    The frozen ``U``/``V`` singular bases and the frozen singular values ``s_pre`` are
    stored as buffers; only the sparse coefficients ``m_coeffs`` (and optionally the
    bias) are trainable.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        pattern="banded",
        off_diag=8,
        num_coeffs=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(in_features, out_features)  # k
        self.pattern = pattern
        self.off_diag = off_diag
        self.num_coeffs = num_coeffs

        # Frozen singular bases / values (filled by init_from_linear_weight).
        self.register_buffer("svft_u", torch.empty(out_features, self.rank), persistent=True)
        self.register_buffer("svft_v", torch.empty(in_features, self.rank), persistent=True)
        self.register_buffer("svft_s", torch.empty(self.rank), persistent=True)

        # Learnable sparse coefficients + their (row, col) positions in M (k x k).
        self.register_buffer("m_index", torch.zeros(2, 0, dtype=torch.long), persistent=True)
        self.m_coeffs = nn.Parameter(torch.zeros(0))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def init_from_linear_weight(
        self,
        weight,
        bias=None,
        seed=0,
    ):
        if not weight.is_cuda:
            raise RuntimeError(
                "SVFT initialization requires CUDA weights so SVD runs on GPU. "
                f"Got device={weight.device}."
            )
        compute_dtype = (
            torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
        )
        u, s, vh = torch.linalg.svd(weight.to(dtype=compute_dtype), full_matrices=False)
        s = torch.clamp(s, min=0)
        v = vh.transpose(-2, -1).contiguous()  # (in_features, k)

        self.svft_u.copy_(u.to(device=weight.device, dtype=weight.dtype))
        self.svft_v.copy_(v.to(device=weight.device, dtype=weight.dtype))
        self.svft_s.copy_(s.to(device=weight.device, dtype=weight.dtype))

        k = self.rank
        if self.pattern == "topk":
            if self.out_features != self.in_features:
                raise ValueError(
                    "SVFT topk pattern requires a square weight (d_out == d_in); "
                    f"got ({self.out_features}, {self.in_features})."
                )
            # Alignment u_i . v_j over all (i, j); keep the top-`num_coeffs`.
            align = (u.to(compute_dtype).T @ v.to(compute_dtype)).abs()  # (k, k)
            num = min(self.num_coeffs, k * k) if self.num_coeffs > 0 else k
            flat = torch.topk(align.reshape(-1), num).indices
            rows = (flat // k).to(weight.device)
            cols = (flat % k).to(weight.device)
            index = torch.stack([rows, cols], dim=0)
        else:
            index = _build_omega_indices(
                k,
                self.pattern,
                self.off_diag,
                self.num_coeffs,
                seed,
                weight.device,
            )

        self.m_index = index
        # Reallocate coefficients to match |Omega|; zero init => Delta_W = 0 at start.
        self.m_coeffs = nn.Parameter(
            torch.zeros(index.shape[1], device=weight.device, dtype=weight.dtype)
        )

        if self.bias is not None and bias is not None:
            self.bias.copy_(bias.to(device=weight.device, dtype=weight.dtype))

    def _delta_diag_plus_m(self):
        """Return the dense ``(S + M)`` matrix of shape (k, k)."""
        k = self.rank
        sm = torch.diag(self.svft_s).to(self.m_coeffs.dtype)
        if self.m_index.shape[1] > 0:
            sm = sm.index_put(
                (self.m_index[0], self.m_index[1]),
                self.m_coeffs,
                accumulate=True,
            )
        return sm

    def materialize_dense_weight(self):
        # W = U (S + M) V^T ; shapes: (d_out,k)(k,k)(k,d_in) -> (d_out, d_in)
        sm = self._delta_diag_plus_m()
        return (self.svft_u @ sm) @ self.svft_v.transpose(-2, -1)

    def forward(self, x):
        return nn.functional.linear(x, self.materialize_dense_weight(), self.bias)


def get_svft_target_module_names(svft_type):
    """Modules to adapt.  Default (``udog``) follows the paper's LLaMA-3-8B recipe:
    up/down/o/gate projections (Table 16 / App. C.7)."""
    if svft_type in ("udog", "all_llama"):
        return ("up_proj", "down_proj", "o_proj", "gate_proj")
    if svft_type == "all":
        return (
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        )
    if svft_type == "mlp":
        return ("gate_proj", "up_proj", "down_proj")
    if svft_type == "attn":
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    raise ValueError("svft_type must be one of: udog, all, mlp, attn")


@torch.no_grad()
def convert_linear_to_svft(
    model,
    skip_names=("lm_head",),
    include_names=None,
    pattern="banded",
    off_diag=8,
    num_coeffs=0,
    seed=0,
):
    include_name_set = set(include_names) if include_names is not None else None
    modules_to_replace = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = name.split(".")[-1]
        if leaf_name in skip_names:
            continue
        if include_name_set is not None and leaf_name not in include_name_set:
            continue
        modules_to_replace.append((name, module))

    non_cuda_modules = [
        (name, module.weight.device)
        for name, module in modules_to_replace
        if not module.weight.is_cuda
    ]
    if non_cuda_modules:
        preview = ", ".join(f"{name}({device})" for name, device in non_cuda_modules[:3])
        if len(non_cuda_modules) > 3:
            preview += f", ... (+{len(non_cuda_modules) - 3} more)"
        raise RuntimeError(
            "Linear->SVFT conversion requires all target Linear weights on CUDA. "
            f"Found non-CUDA modules: {preview}"
        )

    print(f"Converting {len(modules_to_replace)} Linear layers to SVFT ({pattern})")

    for full_name, linear in modules_to_replace:
        _raise_if_non_cuda_weight(full_name, linear.weight)
        path = full_name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        child_name = path[-1]

        svft_layer = SVFTLayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            pattern=pattern,
            off_diag=off_diag,
            num_coeffs=num_coeffs,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        svft_layer.init_from_linear_weight(
            linear.weight.data,
            linear.bias.data if linear.bias is not None else None,
            seed=seed,
        )
        setattr(parent, child_name, svft_layer)

    print("Finished Linear->SVFT conversion")
    return [name for name, _ in modules_to_replace]


def configure_svft_trainability(model, train_bias=True):
    for p in model.parameters():
        p.requires_grad = False

    num_svft_layers = 0
    tuned_coeff_tensors = 0
    tuned_biases = 0

    for _, module in model.named_modules():
        if not isinstance(module, SVFTLayer):
            continue
        num_svft_layers += 1
        module.m_coeffs.requires_grad = True
        tuned_coeff_tensors += 1
        if module.bias is not None:
            module.bias.requires_grad = train_bias
            if train_bias:
                tuned_biases += 1

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    return {
        "num_svft_layers": num_svft_layers,
        "tuned_coeff_tensors": tuned_coeff_tensors,
        "tuned_biases": tuned_biases,
        "trainable_param_count": trainable_param_count,
        "total_param_count": total_param_count,
        "trainable_params": trainable_params,
    }
