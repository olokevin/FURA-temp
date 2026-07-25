"""Optimizer construction shared by the LIFT finetuning entrypoints.

`--optimizer adamw` (default) keeps each script's original AdamW behaviour.
`--optimizer muon` swaps in the repo-root `optim.muon.Muon`, which routes

  * decoder weight matrices (2-D, not embedding / not `lm_head`) -> Muon
  * BlockTT cores (`btt_l` / `btt_r` / `btt_g`)                  -> BTT-Muon
  * everything else (embeddings, `lm_head`, norms, biases,
    BTT singular values `btt_s`, any non-matrix parameter)       -> AdamW

The routing lives inside `Muon` itself; this module just builds it from argparse
flags, restricts it to trainable parameters, and prints a compact summary so a
training log makes the split auditable.
"""

import os
import sys

import torch

# ref/LIFT/src/utils/optim_utils.py -> repo root is four levels up.
_REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir
    )
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from optim.muon import Muon  # noqa: E402

OPTIMIZER_CHOICES = ["adamw", "muon"]

_POLAR_METHODS = [
    "Keller",
    "Jiacheng",
    "polarexpress",
    "fast_polarexpress",
    "svd-exact",
]
_NORM_METHODS = ["row", "col", "row-col", "col-row", "shape"]
_STRUCTURED_ORTHO_METHODS = ["mup", "svd", "rms", "naive"]


def _parse_betas(value):
    parts = [v for v in str(value).replace("(", "").replace(")", "").split(",") if v.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated betas, got {value!r}")
    return (float(parts[0]), float(parts[1]))


def add_optimizer_args(parser):
    """Add the `--optimizer` switch plus Muon hyperparameters."""
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=OPTIMIZER_CHOICES,
        help=(
            "Optimizer to use. 'muon' applies orthogonalized updates to decoder "
            "weight matrices (and trainable BlockTT cores) and AdamW to "
            "embeddings / lm_head / norms / biases / btt_s."
        ),
    )
    parser.add_argument(
        "--muon_lr_adam",
        type=float,
        default=None,
        help="Muon-only: LR for the AdamW-routed parameters (default: --learning_rate).",
    )
    parser.add_argument(
        "--muon_lr_embedding",
        type=float,
        default=None,
        help="Muon-only: LR for embedding parameters (default: the AdamW LR).",
    )
    parser.add_argument(
        "--muon_momentum",
        type=float,
        default=0.95,
        help="Muon-only: momentum for the internal SGD (default: 0.95).",
    )
    parser.add_argument(
        "--muon_ns_steps",
        type=int,
        default=5,
        help="Muon-only: Newton-Schulz / PolarExpress iteration count (default: 5).",
    )
    parser.add_argument(
        "--muon_polar_method",
        type=str,
        default="polarexpress",
        choices=_POLAR_METHODS,
        help="Muon-only: polar factorization kernel (default: polarexpress).",
    )
    parser.add_argument(
        "--muon_structured_ortho_method",
        type=str,
        default="mup",
        choices=_STRUCTURED_ORTHO_METHODS,
        help=(
            "Muon-only: how BlockTT cores are orthogonalized. 'mup' treats each "
            "core slice as one matrix; 'svd'/'rms' orthogonalize per (rank x b) "
            "block; 'naive' flattens the core (default: mup)."
        ),
    )
    parser.add_argument(
        "--muon_norm_method",
        type=str,
        default=None,
        choices=_NORM_METHODS,
        help="Muon-only: optional row/col renormalization of the update.",
    )
    parser.add_argument(
        "--muon_adamw_betas",
        type=_parse_betas,
        default=(0.9, 0.95),
        help="Muon-only: betas for the internal AdamW fallback (default: 0.9,0.95).",
    )
    parser.add_argument(
        "--muon_no_rms_scaling",
        action="store_false",
        dest="muon_rms_scaling",
        help="Muon-only: disable the sqrt(fan_out/fan_in) update scaling.",
    )
    parser.set_defaults(muon_rms_scaling=True)
    return parser


def _bucket_names(optimizer, buckets):
    names = []
    for group_params in buckets:
        for p in group_params:
            names.extend(optimizer._param_names_by_id.get(id(p), []))
    return names


def describe_muon_routing(optimizer):
    """Return `{bucket: (num_params, num_elements)}` for a built `Muon`."""
    buckets = {
        "muon_matrix": optimizer._bucket_muon_matrix_by_group,
        "muon_btt": optimizer._bucket_btt_by_group,
        "muon_cola": (
            optimizer._bucket_muon_cola_a_by_group
            + optimizer._bucket_muon_cola_b_by_group
            + optimizer._bucket_muon_cola_g_by_group
        ),
        "adamw": optimizer._bucket_adamw_by_group,
        "adamw_embedding": optimizer._bucket_embedding_adamw_by_group,
    }
    summary = {}
    for label, by_group in buckets.items():
        params = [p for group_params in by_group for p in group_params]
        summary[label] = (len(params), sum(p.numel() for p in params))
    return summary


def _print_muon_routing(optimizer):
    summary = describe_muon_routing(optimizer)
    print("[muon] routing summary (trainable params only):")
    for label in ("muon_matrix", "muon_btt", "muon_cola", "adamw", "adamw_embedding"):
        count, numel = summary[label]
        if count == 0:
            continue
        print(f"  {label:>16}: {count:5d} tensors, {numel:,} params")

    # Guard the contract we advertise: nothing embedding-like should reach the
    # orthogonalized path.
    matrix_names = _bucket_names(optimizer, optimizer._bucket_muon_matrix_by_group)
    leaked = [
        n
        for n in matrix_names
        if any(tok in n for tok in ("embed_tokens", "embeddings", "wte", "wpe", "lm_head"))
    ]
    if leaked:
        raise RuntimeError(
            f"[muon] embedding/lm_head parameters were routed to matrix-Muon: {leaked[:5]}"
        )


def build_optimizer(args, model, adamw_factory):
    """Build the optimizer for `model`.

    `adamw_factory` is a zero-arg callable producing the script's original AdamW
    instance; it is used verbatim when `--optimizer adamw`.
    """
    optimizer_name = getattr(args, "optimizer", "adamw")
    if optimizer_name == "adamw":
        return adamw_factory()
    if optimizer_name != "muon":
        raise ValueError(f"Unsupported --optimizer: {optimizer_name}")

    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named_params:
        raise ValueError("--optimizer muon: no trainable parameters found.")

    lr_adam = args.muon_lr_adam if args.muon_lr_adam is not None else args.learning_rate
    optimizer = Muon(
        named_params,
        lr=args.learning_rate,
        lr_adam=lr_adam,
        lr_embedding=args.muon_lr_embedding,
        weight_decay=args.weight_decay,
        momentum=args.muon_momentum,
        nesterov=True,
        ns_steps=args.muon_ns_steps,
        rms_scaling=args.muon_rms_scaling,
        polar_method=args.muon_polar_method,
        adamw_betas=tuple(args.muon_adamw_betas),
        norm_method=args.muon_norm_method,
        structured_ortho_method=args.muon_structured_ortho_method,
    )
    print(
        f"[muon] lr={args.learning_rate} lr_adam={lr_adam} "
        f"lr_embedding={args.muon_lr_embedding} momentum={args.muon_momentum} "
        f"ns_steps={args.muon_ns_steps} polar={args.muon_polar_method} "
        f"structured_ortho={args.muon_structured_ortho_method} "
        f"rms_scaling={args.muon_rms_scaling} norm_method={args.muon_norm_method} "
        f"weight_decay={args.weight_decay}"
    )
    _print_muon_routing(optimizer)
    return optimizer


__all__ = [
    "OPTIMIZER_CHOICES",
    "add_optimizer_args",
    "build_optimizer",
    "describe_muon_routing",
]
