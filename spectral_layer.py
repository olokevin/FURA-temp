"""Spectral Adapter: Fine-Tuning in Spectral Space (Zhang & Pilanci, 2024).

SVD the pretrained weight ``W = U S Vᵀ``, keep the **top-r** singular triplet trainable
in spectral space and freeze the rest (the low-rank tail ``low``).  Two mechanisms
(paper Sec. 3, faithful to ref/spectral_adapter):

* ``additive`` — additively perturb the top-r singular *vectors* (S frozen)::

      W = (U_r + ΔU) diag(S_r) (V_r + ΔV)ᵀ + low          (ΔU, ΔV zero-init)

  This is the llama3_tune `spectral_top` LoRA variant: ΔU = lora_B (d_out×r),
  ΔV = lora_A ᵀ (d_in×r).

* ``rotation`` — orthogonally rotate the top-r singular vectors via Cayley
  parameterization (S frozen)::

      W = (U_r Q_A) diag(S_r) (V_r Q_B)ᵀ + low
      Q = (I − skew)(I + skew)⁻¹ ,  skew = ½(A − Aᵀ)       (A, B ∈ ℝ^{r×r} zero-init)

  At init both ΔU=ΔV=0 / A=B=0 ⇒ Q=I ⇒ ``W`` is reconstructed exactly (adapter = identity).

Related to PiSSA (both start from the top-r principal SVD subspace and are additive /
mergeable), but PiSSA trains a *free* low-rank update B·A seeded by the principal
components, whereas Spectral-additive perturbs the singular directions with ``diag(S_r)``
held fixed as a spectral weighting; Spectral-rotation constrains the update to be an
orthogonal rotation of those directions.

Mirrors :mod:`svft_layer` / :mod:`svd_layer` so the trainer, materialization, and eval
pipeline reuse the same plumbing (materializes to a dense ``nn.Linear`` at save time).
"""

import torch
import torch.nn as nn


VALID_SPECTRAL_MODES = {"additive", "rotation"}


def _raise_if_non_cuda_weight(full_name, weight):
    if weight.is_cuda:
        return
    raise RuntimeError(
        "Linear->Spectral conversion requires CUDA weights so SVD runs on GPU. "
        f"Module '{full_name}' is on device={weight.device}."
    )


def cayley(data):
    """Cayley map of a square matrix -> orthogonal matrix.

    ``skew = ½(A − Aᵀ)`` is skew-symmetric; ``Q = (I − skew)(I + skew)⁻¹`` is orthogonal.
    A = 0 -> Q = I.  (Faithful to ref/spectral_adapter edlora.cayley.)
    """
    r = data.shape[0]
    skew = 0.5 * (data - data.transpose(-2, -1))
    ident = torch.eye(r, device=data.device, dtype=data.dtype)
    return torch.linalg.solve(ident + skew, ident - skew)  # (I+skew)^{-1}(I-skew) = (I-skew)(I+skew)^{-1}


class SpectralLayer(nn.Module):
    """Drop-in replacement for ``nn.Linear`` implementing the Spectral Adapter.

    Frozen buffers: top-r singular vectors ``U_r`` (d_out×r), ``V_r`` (d_in×r),
    singular values ``S_r`` (r), and the frozen tail ``low`` (d_out×d_in).
    Trainable (mode-dependent): additive perturbations ``delta_u``/``delta_v`` or the
    Cayley generators ``rot_a``/``rot_b``.
    """

    def __init__(self, in_features, out_features, rank, mode="additive", bias=True):
        super().__init__()
        if mode not in VALID_SPECTRAL_MODES:
            raise ValueError(f"mode must be one of {VALID_SPECTRAL_MODES}; got {mode!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.rank = min(rank, in_features, out_features)

        r = self.rank
        self.register_buffer("spec_u", torch.empty(out_features, r), persistent=True)
        self.register_buffer("spec_v", torch.empty(in_features, r), persistent=True)
        self.register_buffer("spec_s", torch.empty(r), persistent=True)
        self.register_buffer("spec_low", torch.empty(out_features, in_features), persistent=True)

        if mode == "additive":
            self.delta_u = nn.Parameter(torch.zeros(out_features, r))
            self.delta_v = nn.Parameter(torch.zeros(in_features, r))
        else:  # rotation
            self.rot_a = nn.Parameter(torch.zeros(r, r))
            self.rot_b = nn.Parameter(torch.zeros(r, r))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def init_from_linear_weight(self, weight, bias=None):
        if not weight.is_cuda:
            raise RuntimeError(
                "Spectral initialization requires CUDA weights so SVD runs on GPU. "
                f"Got device={weight.device}."
            )
        compute_dtype = (
            torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
        )
        # torch.linalg.svd returns Vh (r×n); convert to V (n×r) to match reference (torch.svd).
        u, s, vh = torch.linalg.svd(weight.to(dtype=compute_dtype), full_matrices=False)
        v = vh.transpose(-2, -1).contiguous()
        r = self.rank
        u_r = u[:, :r]
        s_r = torch.clamp(s[:r], min=0)
        v_r = v[:, :r]
        # Frozen tail = the part of W outside the top-r spectral subspace.
        low = u[:, r:] @ torch.diag(s[r:]) @ v[:, r:].transpose(-2, -1)

        self.spec_u.copy_(u_r.to(device=weight.device, dtype=weight.dtype))
        self.spec_v.copy_(v_r.to(device=weight.device, dtype=weight.dtype))
        self.spec_s.copy_(s_r.to(device=weight.device, dtype=weight.dtype))
        self.spec_low.copy_(low.to(device=weight.device, dtype=weight.dtype))

        if self.bias is not None and bias is not None:
            self.bias.copy_(bias.to(device=weight.device, dtype=weight.dtype))

    def _u_v_effective(self):
        if self.mode == "additive":
            u_eff = self.spec_u + self.delta_u
            v_eff = self.spec_v + self.delta_v
        else:  # rotation
            q_a = cayley(self.rot_a.to(torch.float32)).to(self.spec_u.dtype)
            q_b = cayley(self.rot_b.to(torch.float32)).to(self.spec_v.dtype)
            u_eff = self.spec_u @ q_a
            v_eff = self.spec_v @ q_b
        return u_eff, v_eff

    def materialize_dense_weight(self):
        u_eff, v_eff = self._u_v_effective()
        # W = U_eff diag(S) V_effᵀ + low
        return (u_eff * self.spec_s.unsqueeze(0)) @ v_eff.transpose(-2, -1) + self.spec_low

    def forward(self, x):
        return nn.functional.linear(x, self.materialize_dense_weight(), self.bias)


def get_spectral_target_module_names(spectral_type):
    """Default (``pissa5``) matches PiSSA's 5-module recipe for a fair comparison."""
    if spectral_type in ("pissa5", "lora5"):
        return ("q_proj", "k_proj", "v_proj", "up_proj", "down_proj")
    if spectral_type in ("spectral6", "paper6"):  # spectral paper's llama3 set (+gate/o)
        return ("q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj")
    if spectral_type == "all":
        return ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    if spectral_type == "mlp":
        return ("gate_proj", "up_proj", "down_proj")
    if spectral_type == "attn":
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    raise ValueError("spectral_type must be one of: pissa5, spectral6, all, mlp, attn")


@torch.no_grad()
def convert_linear_to_spectral(
    model,
    rank,
    mode="additive",
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

    non_cuda = [
        (name, module.weight.device)
        for name, module in modules_to_replace
        if not module.weight.is_cuda
    ]
    if non_cuda:
        preview = ", ".join(f"{n}({d})" for n, d in non_cuda[:3])
        if len(non_cuda) > 3:
            preview += f", ... (+{len(non_cuda) - 3} more)"
        raise RuntimeError(
            "Linear->Spectral conversion requires all target Linear weights on CUDA. "
            f"Found non-CUDA modules: {preview}"
        )

    print(f"Converting {len(modules_to_replace)} Linear layers to Spectral ({mode}, rank={rank})")

    for full_name, linear in modules_to_replace:
        _raise_if_non_cuda_weight(full_name, linear.weight)
        path = full_name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        child_name = path[-1]

        spec = SpectralLayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            mode=mode,
            bias=(linear.bias is not None),
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        spec.init_from_linear_weight(
            linear.weight.data,
            linear.bias.data if linear.bias is not None else None,
        )
        setattr(parent, child_name, spec)

    print("Finished Linear->Spectral conversion")
    return [name for name, _ in modules_to_replace]


def configure_spectral_trainability(model, train_bias=False):
    for p in model.parameters():
        p.requires_grad = False

    num_layers = 0
    tuned = 0
    tuned_biases = 0
    for _, module in model.named_modules():
        if not isinstance(module, SpectralLayer):
            continue
        num_layers += 1
        if module.mode == "additive":
            module.delta_u.requires_grad = True
            module.delta_v.requires_grad = True
            tuned += 2
        else:
            module.rot_a.requires_grad = True
            module.rot_b.requires_grad = True
            tuned += 2
        if module.bias is not None and train_bias:
            module.bias.requires_grad = True
            tuned_biases += 1

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return {
        "num_spectral_layers": num_layers,
        "tuned_tensors": tuned,
        "tuned_biases": tuned_biases,
        "trainable_param_count": sum(p.numel() for p in trainable_params),
        "total_param_count": sum(p.numel() for p in model.parameters()),
        "trainable_params": trainable_params,
    }
