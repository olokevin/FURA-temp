"""QuanTA: Quantum-informed Tensor Adaptation (Chen et al., NeurIPS 2024).

Reparametrizes an ``nn.Linear`` weight update as a sequence of small tensors that act
on individual axes of a *reshaped* hidden vector, analogous to two-qubit gates in a
quantum circuit.  This yields an efficient **high-rank** adapter with no inference
overhead (the tensor product can be merged back into a dense weight).

Given hidden dim ``d = d_1 * d_2 * ... * d_N`` (config e.g. ``16-8-8-4`` for LLaMA-3-8B,
4096 = 16*8*8*4), we reshape ``x`` to ``(..., d_1, ..., d_N)`` and, for every unordered
pair of axes ``(m, n)``, apply a trainable tensor ``T`` of shape
``(d_m, d_n, d_m, d_n)`` contracting the input axes.  There is exactly one tensor per
axis pair (``N*(N-1)/2`` tensors), matching the paper's ``N in {3,4,5}`` construction.

Zero-init trick (paper Eq. 8): a frozen copy ``S`` of the tensors is kept, and the layer
computes ``y = W0 x + T x - S x``.  Since ``T`` and ``S`` start identical, ``Tx - Sx = 0``
at initialization and the adapter is exactly the base model.  ``W0`` stays frozen.

Rectangular weights (``in_features != out_features``, e.g. GQA v_proj 4096->1024) are
handled by padding the reshaped input up to ``prod(d_n)`` and truncating the output to
``out_features`` (paper Appendix B).
"""

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_symbol(i):
    """opt_einsum-style symbol generator (letters then unicode), no external dep."""
    if i < 26:
        return chr(ord("a") + i)
    if i < 52:
        return chr(ord("A") + i - 26)
    return chr(0x100 + i - 52)


def _gen_quanta_apply_expr(N):
    """einsum string to apply {T} to a batched input x of shape (..., d_1, ..., d_N).

    Reproduces Appendix G ``quanta_apply_einsum_expr``: one tensor per unordered axis
    pair, contracted sequentially.  Returns (expr, pair_order) where pair_order[k] is
    the ordered axis pair ``(a, b)`` for tensor k, meaning tensor k must have shape
    ``(d_a, d_b, d_a, d_b)`` to line up with the einsum string.

    The generated per-tensor subscript is ``sym(s4)sym(s3)sym(s2)sym(s1)`` — i.e. its
    four axes are ``(out_b, out_a, in_b, in_a)`` where ``a = dim1`` and ``b = dim2`` are
    the two axes being contracted.  So the tensor's index order is (b, a): we record the
    pair as ``(dim2, dim1)`` so the caller allocates tensors with matching axis sizes.
    """
    current = list(range(N))
    expr = "..."
    for i in current:
        expr += _get_symbol(i)
    pair_order = []
    for (dim1, dim2) in itertools.combinations(range(-1, -N - 1, -1), 2):
        s1 = current[dim1]
        s2 = current[dim2]
        s3 = s1 + N
        s4 = s2 + N
        expr += "," + _get_symbol(s4) + _get_symbol(s3) + _get_symbol(s2) + _get_symbol(s1)
        current[dim1] = s3
        current[dim2] = s4
        # Tensor subscript trailing axes are (s2, s1) = (dim2, dim1); allocate to match.
        pair_order.append((dim2 % N, dim1 % N))
    expr += "->..."
    for i in current:
        expr += _get_symbol(i)
    return expr, pair_order


class QuanTALayer(nn.Module):
    """Drop-in replacement for ``nn.Linear`` implementing the QuanTA update.

    ``dims`` is the per-axis decomposition (e.g. ``[16, 8, 8, 4]``); ``prod(dims)`` must
    be >= max(in_features, out_features) (input is padded / output truncated otherwise).
    """

    def __init__(self, in_features, out_features, dims, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dims = list(dims)
        self.N = len(dims)
        self.total = int(math.prod(dims))
        if self.total < max(in_features, out_features):
            raise ValueError(
                f"prod(dims)={self.total} < max(in={in_features}, out={out_features}); "
                "choose a larger QuanTA decomposition."
            )

        self.apply_expr, self.pair_order = _gen_quanta_apply_expr(self.N)

        # Frozen base weight W0 (kept in the layer so we fully replace nn.Linear).
        self.register_buffer("weight", torch.empty(out_features, in_features), persistent=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # One tensor per unordered axis pair: shape (d_m, d_n, d_m, d_n).
        # T = trainable, S = frozen copy (for the zero-init cancellation).
        self.t_tensors = nn.ParameterList()
        self._s_names = []
        for p_idx, (m, n) in enumerate(self.pair_order):
            dm, dn = self.dims[m], self.dims[n]
            t = nn.Parameter(torch.empty(dm, dn, dm, dn))
            self.t_tensors.append(t)
            s_name = f"quanta_s_{p_idx}"
            self.register_buffer(s_name, torch.empty(dm, dn, dm, dn), persistent=True)
            self._s_names.append(s_name)

    @torch.no_grad()
    def reset_tensors(self):
        """Kaiming-init each trainable tensor (viewed as a (dm*dn, dm*dn) matrix) then
        copy it into the frozen S buffer so ``Tx - Sx == 0`` at init."""
        for t, s_name in zip(self.t_tensors, self._s_names):
            dm, dn = t.shape[0], t.shape[1]
            mat = t.view(dm * dn, dm * dn)
            nn.init.kaiming_uniform_(mat, a=math.sqrt(5))
            getattr(self, s_name).copy_(t.data)

    @torch.no_grad()
    def init_from_linear_weight(self, weight, bias=None):
        self.weight.copy_(weight.to(device=self.weight.device, dtype=self.weight.dtype))
        if self.bias is not None and bias is not None:
            self.bias.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))
        self.reset_tensors()

    def _apply_tensor_product(self, x, tensors):
        """Apply the sequence {tensors} to x (shape (..., in_features)); returns
        (..., total) reshaped-back vector.  Pads input up to `total` first."""
        lead = x.shape[:-1]
        if self.in_features < self.total:
            x = F.pad(x, (0, self.total - self.in_features))
        x = x.reshape(*lead, *self.dims)
        # Tensors are contracted in compute dtype for stability under bf16.
        out = torch.einsum(self.apply_expr, x, *tensors)
        out = out.reshape(*lead, self.total)
        return out

    def _quanta_delta(self, x):
        """(T - S) x, truncated/padded to out_features."""
        t_out = self._apply_tensor_product(x, list(self.t_tensors))
        s_tensors = [getattr(self, n) for n in self._s_names]
        s_out = self._apply_tensor_product(x, s_tensors)
        delta = t_out - s_out  # (..., total)
        if self.out_features <= self.total:
            return delta[..., : self.out_features]
        return F.pad(delta, (0, self.out_features - self.total))

    def materialize_dense_weight(self):
        """Construct the effective dense weight W0 + (T - S) merged, of shape
        (out_features, in_features).  Uses identity probing so it stays einsum-driven
        and dtype-faithful with the forward path."""
        eye = torch.eye(self.in_features, device=self.weight.device, dtype=self.t_tensors[0].dtype)
        delta_cols = self._quanta_delta(eye)  # (in_features, out_features)
        delta = delta_cols.transpose(0, 1).to(self.weight.dtype)  # (out_features, in_features)
        return self.weight + delta

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        compute_dtype = self.t_tensors[0].dtype
        delta = self._quanta_delta(x.to(compute_dtype)).to(base.dtype)
        return base + delta


def get_quanta_target_module_names(quanta_type):
    """Modules to adapt.  Default (``qv``) follows the paper's commonsense recipe
    (Table E.3): q_proj, v_proj only."""
    if quanta_type == "qv":
        return ("q_proj", "v_proj")
    if quanta_type == "qkvud":
        return ("q_proj", "k_proj", "v_proj", "up_proj", "down_proj")
    if quanta_type == "all":
        return (
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        )
    if quanta_type == "attn":
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    raise ValueError("quanta_type must be one of: qv, qkvud, all, attn")


def parse_dims(dims_str):
    """Parse a config like '16-8-8-4' into [16, 8, 8, 4]."""
    return [int(x) for x in str(dims_str).split("-") if x != ""]


@torch.no_grad()
def convert_linear_to_quanta(
    model,
    dims,
    skip_names=("lm_head",),
    include_names=None,
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
            "Linear->QuanTA conversion requires all target Linear weights on CUDA. "
            f"Found non-CUDA modules: {preview}"
        )

    print(f"Converting {len(modules_to_replace)} Linear layers to QuanTA (dims={dims})")

    for full_name, linear in modules_to_replace:
        path = full_name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        child_name = path[-1]

        quanta_layer = QuanTALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            dims=dims,
            bias=(linear.bias is not None),
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        quanta_layer.init_from_linear_weight(
            linear.weight.data,
            linear.bias.data if linear.bias is not None else None,
        )
        setattr(parent, child_name, quanta_layer)

    print("Finished Linear->QuanTA conversion")
    return [name for name, _ in modules_to_replace]


def configure_quanta_trainability(model, train_bias=False):
    for p in model.parameters():
        p.requires_grad = False

    num_quanta_layers = 0
    tuned_tensors = 0
    tuned_biases = 0

    for _, module in model.named_modules():
        if not isinstance(module, QuanTALayer):
            continue
        num_quanta_layers += 1
        for t in module.t_tensors:
            t.requires_grad = True
            tuned_tensors += 1
        if module.bias is not None and train_bias:
            module.bias.requires_grad = True
            tuned_biases += 1

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    return {
        "num_quanta_layers": num_quanta_layers,
        "tuned_tensors": tuned_tensors,
        "tuned_biases": tuned_biases,
        "trainable_param_count": trainable_param_count,
        "total_param_count": total_param_count,
        "trainable_params": trainable_params,
    }
