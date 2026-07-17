"""Shared helpers for integrating src/compress calibrated BTT with the
run_sft / run_rl / run_rl_dapo / LIFT training entrypoints.

Keeps legacy btt_layer.py / svd_layer.py paths untouched; activated only
when --calib-mode (or --calib_mode) is non-'none'.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure the sibling `compress` package under <repo_root>/src is importable.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from compress.decomposition import (  # noqa: E402
    DecompositionConfig,
    decompose_with_loader,
)
from compress.btt.btt_linear import BTTLinear  # noqa: E402
from compress.loaders import (  # noqa: E402
    build_c4_calib_loader,
    build_traces_jsonl_calib_loader,
)
from compress.topology import export_btt_topology, rebuild_btt_from_topology  # noqa: E402


# Public mapping from CLI --calib-mode value to compress train_mode.
CALIB_MODE_TO_TRAIN_MODE = {
    "v2": "btt_llm_v2",
    "v2_bp": "btt_llm_v2_bp",
    "v2_combined": "btt_llm_v2_combined",
    "twosteps": "btt_twosteps",
    "svd_v2": "svd_llm_v2",
    "svd_v2_combined": "svd_llm_v2_combined",
}
VALID_CALIB_MODES = ("none",) + tuple(CALIB_MODE_TO_TRAIN_MODE.keys())
VALID_CALIB_SOURCES = ("c4", "traces", "training_data")


def _arg_name(name: str, *, hyphen_style: bool) -> str:
    return "--" + (name if hyphen_style else name.replace("-", "_"))


def _attr_name(name: str) -> str:
    return name.replace("-", "_")


def add_calibrated_btt_args(parser, *, hyphen_style: bool = True) -> None:
    """Register --calib-mode / --calib-source / ... flags on the given parser."""
    parser.add_argument(
        _arg_name("calib-mode", hyphen_style=hyphen_style),
        type=str, default="none", choices=list(VALID_CALIB_MODES),
        help="Calibrated BTT mode. 'none' keeps the legacy blocktt path.",
    )
    parser.add_argument(
        _arg_name("calib-source", hyphen_style=hyphen_style),
        type=str, default="c4", choices=list(VALID_CALIB_SOURCES),
        help="Calibration data source. Only used when --calib-mode != none.",
    )
    parser.add_argument(
        _arg_name("calib-traces-path", hyphen_style=hyphen_style),
        type=str, default=None,
        help="Path to trace JSONL; required when --calib-source=traces.",
    )
    parser.add_argument(
        _arg_name("calib-num-seqs", hyphen_style=hyphen_style),
        type=int, default=128,
        help="Number of calibration sequences to sample.",
    )
    parser.add_argument(
        _arg_name("calib-max-length", hyphen_style=hyphen_style),
        type=int, default=2048,
        help="Max token length per calibration sample.",
    )
    parser.add_argument(
        _arg_name("calib-seed", hyphen_style=hyphen_style),
        type=int, default=3,
        help="RNG seed for calibration sampling.",
    )
    parser.add_argument(
        _arg_name("calib-batch-size", hyphen_style=hyphen_style),
        type=int, default=8,
        help="Batch size used by the calibration DataLoader.",
    )
    parser.add_argument(
        _arg_name("compression-ratio", hyphen_style=hyphen_style),
        type=float, default=1.0,
        help=(
            "Compression ratio for SVD calib modes (svd_v2 / svd_v2_combined). "
            "Fraction of compressible-layer params to retain. Must be in (0, 1]. "
            "Ignored for BTT calib modes (use --blocktt-rank instead)."
        ),
    )


# ---- helpers below; most are stubs filled in by later tasks ----

def validate_calibrated_btt_args(args, *, argv: Sequence[str], hyphen_style: bool = True) -> None:
    """Raise ValueError if the calib-* args are inconsistent with train-mode / blocktt-rank.

    Note: ``argv`` is unused by the current rules but kept in the signature for
    stability — Task 11 wires ``argv=argv`` from run_sft.py / run_rl.py, and
    future rules may need it.
    """
    del argv  # currently unused; see docstring
    calib_mode = getattr(args, "calib_mode", "none")
    calib_source = getattr(args, "calib_source", "c4")

    # 1. BTT calib modes require --train-mode blocktt; SVD calib modes are
    #    applied inside full-FT scripts so they have no train_mode constraint.
    if calib_mode != "none" and not calib_mode.startswith("svd_") \
            and hasattr(args, "train_mode"):
        if args.train_mode != "blocktt":
            raise ValueError(
                "--calib-mode only valid with --train-mode blocktt "
                f"(got --train-mode={args.train_mode!r})"
            )

    # 2. --calib-source=traces requires --calib-traces-path
    if calib_mode != "none" and calib_source == "traces":
        path = getattr(args, "calib_traces_path", None)
        if not path:
            flag = "--calib-traces-path" if hyphen_style else "--calib_traces_path"
            raise ValueError(f"{flag} must be set when --calib-source=traces")

    # 3. Integer --blocktt-rank rejected on the BTT calibrated path; float must be in (0, 1]
    if calib_mode != "none" and not calib_mode.startswith("svd_"):
        rank_raw = getattr(args, "blocktt_rank", "full")
        if isinstance(rank_raw, str):
            if rank_raw != "full":
                try:
                    int(rank_raw)
                    is_int = "." not in rank_raw
                except ValueError:
                    is_int = False
                if is_int:
                    raise ValueError(
                        "integer --blocktt-rank is only valid when --calib-mode=none; "
                        "for calibrated BTT pass 'full' or a float in (0, 1]"
                    )
                # Float rank must be in (0, 1]
                if "." in rank_raw or "e" in rank_raw.lower():
                    try:
                        val = float(rank_raw)
                    except ValueError:
                        val = None
                    if val is not None and not (0.0 < val <= 1.0):
                        raise ValueError(
                            "float --blocktt-rank must be in (0, 1] for calibrated BTT "
                            f"(got {rank_raw!r})"
                        )


_BLOCKTT_TARGET_NAMES = {
    "all": ("gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
}


def _resolve_ratio_from_rank(rank_raw) -> float:
    """--blocktt-rank on the calibrated path: 'full' -> 1.0; float in (0, 1] -> itself.

    Integer strings like '4' are rejected: on the calibrated path, rank is a
    compression ratio, not a fixed rank. `validate_calibrated_btt_args` also
    catches this, but we re-enforce here so direct callers (tests, LIFT) get
    a clear error.
    """
    if isinstance(rank_raw, bool):
        raise ValueError(f"--blocktt-rank must be 'full' or a float; got bool {rank_raw!r}")
    if isinstance(rank_raw, str):
        if rank_raw == "full":
            return 1.0
        # Reject integer-string form ('4', '+4', '-1') before the float parse.
        if "." not in rank_raw and "e" not in rank_raw.lower():
            stripped = rank_raw.lstrip("+-")
            if stripped.isdigit():
                raise ValueError(
                    "integer --blocktt-rank is only valid when --calib-mode=none; "
                    f"for calibrated BTT pass 'full' or a float in (0, 1] (got {rank_raw!r})"
                )
        try:
            val = float(rank_raw)
        except ValueError as exc:
            raise ValueError(f"--blocktt-rank must be 'full' or a float; got {rank_raw!r}") from exc
    elif isinstance(rank_raw, (int, float)):
        val = float(rank_raw)
    else:
        raise ValueError(f"--blocktt-rank has unsupported type {type(rank_raw).__name__}")
    if not math.isfinite(val):
        raise ValueError(f"--blocktt-rank must be finite; got {val}")
    if not (0.0 < val <= 1.0):
        raise ValueError(f"--blocktt-rank float must be in (0, 1]; got {val}")
    return val


def _build_skip_layers(model: nn.Module, trainable_type: str) -> str:
    """Return comma-separated leaf names to skip, i.e. every nn.Linear leaf whose
    name is NOT in the trainable_type target set. 'lm_head' is always included."""
    targets = set(_BLOCKTT_TARGET_NAMES.get(trainable_type, _BLOCKTT_TARGET_NAMES["all"]))
    skip = {"lm_head"}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets:
            skip.add(leaf)
    return ",".join(sorted(skip))


def build_decomposition_config(args, *, hyphen_style: bool = True, model=None) -> DecompositionConfig:
    """Translate CLI args to a DecompositionConfig. `model` is required so that
    --trainable-type can be inverted into a skip_layers list."""
    del hyphen_style  # kept for signature stability; routed via _arg_name upstream
    if model is None:
        raise ValueError("build_decomposition_config requires model= for skip_layers inversion")

    calib_mode = getattr(args, "calib_mode", None)
    if calib_mode is None:
        raise ValueError(
            "build_decomposition_config requires args.calib_mode; did you call "
            "add_calibrated_btt_args() on the parser?"
        )
    if calib_mode not in CALIB_MODE_TO_TRAIN_MODE:
        raise ValueError(
            f"calib_mode must be one of {sorted(CALIB_MODE_TO_TRAIN_MODE.keys())}; "
            f"got {calib_mode!r}"
        )
    train_mode = CALIB_MODE_TO_TRAIN_MODE[calib_mode]

    if calib_mode.startswith("svd_"):
        ratio_raw = getattr(args, "compression_ratio", 1.0)
        try:
            ratio = float(ratio_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"--compression-ratio must be a float; got {ratio_raw!r}"
            ) from exc
        if not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"--compression-ratio must be in (0, 1] for SVD calib modes; got {ratio}"
            )
    else:
        ratio = _resolve_ratio_from_rank(getattr(args, "blocktt_rank", "full"))

    trainable_type = getattr(args, "trainable_type", "all")
    skip_layers = _build_skip_layers(model, trainable_type)

    return DecompositionConfig(
        train_mode=train_mode,
        compression_ratio=ratio,
        calib_source=getattr(args, "calib_source", "c4"),
        calib_traces_path=getattr(args, "calib_traces_path", None),
        calib_num_seqs=getattr(args, "calib_num_seqs", 128),
        calib_max_length=getattr(args, "calib_max_length", 2048),
        calib_seed=getattr(args, "calib_seed", 3),
        skip_layers=skip_layers,
        decomp_mode=getattr(args, "decomp_mode", "square"),
        train_position=getattr(args, "train_position", "both"),
        s_merged_to=getattr(args, "s_merged_to", None),
        factorize_by_head=bool(getattr(args, "blocktt_factorize_by_head", True)),
    )


def build_training_data_calib_loader(
    dataset, collate_fn, *, num_seqs: int, batch_size: int, seed: int,
) -> DataLoader:
    """Take a deterministic subset of the training dataset and wrap it in a
    DataLoader using the training collate function. Yields the same batch shape
    the training loop sees."""
    import random
    n_available = len(dataset)
    n_take = min(int(num_seqs), n_available)
    rng = random.Random(int(seed))
    indices = list(range(n_available))
    rng.shuffle(indices)
    subset = Subset(dataset, indices[:n_take])
    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )


def build_rl_rollout_calib_loader(
    *, rl_rollout_fn: Callable[[int], List[Tuple[str, str]]],
    tokenizer, num_seqs: int, batch_size: int, max_length: int, seed: int,
) -> DataLoader:
    """Run a caller-supplied rollout function to produce (prompt, completion)
    text pairs; tokenize each pair; build a DataLoader of
    {input_ids, attention_mask, labels} where prompt tokens are masked to -100
    in labels (mirroring the RL loss mask)."""
    pairs = rl_rollout_fn(int(num_seqs))
    if len(pairs) == 0:
        raise ValueError("rl_rollout_fn returned no (prompt, completion) pairs")

    examples = []
    for prompt, completion in pairs[:int(num_seqs)]:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        full = (prompt_ids + completion_ids)[: int(max_length)]
        prompt_len = min(len(prompt_ids), len(full))
        labels = [-100] * prompt_len + full[prompt_len:]
        # If prompt was truncated away entirely, fall back to masking half.
        if all(l == -100 for l in labels):
            labels = full.copy()
        examples.append({
            "input_ids": torch.tensor(full, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        })

    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def _collate(batch):
        max_len = max(e["input_ids"].shape[0] for e in batch)
        input_ids = torch.full((len(batch), max_len), fill_value=pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
        for i, e in enumerate(batch):
            L = e["input_ids"].shape[0]
            input_ids[i, :L] = e["input_ids"]
            attn[i, :L] = 1
            labels[i, :L] = e["labels"]
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    return DataLoader(
        examples, batch_size=int(batch_size), shuffle=False, collate_fn=_collate,
    )


def build_calib_loader(
    args, *, tokenizer, training_dataset=None, training_collate_fn=None,
    rl_rollout_fn=None, hyphen_style: bool = True,
) -> Optional[DataLoader]:
    calib_mode = getattr(args, "calib_mode", "none")
    if calib_mode == "none":
        return None

    source = getattr(args, "calib_source", "c4")
    num_seqs = int(getattr(args, "calib_num_seqs", 128))
    max_length = int(getattr(args, "calib_max_length", 2048))
    seed = int(getattr(args, "calib_seed", 3))
    batch_size = int(getattr(args, "calib_batch_size", 8))

    if source == "c4":
        return build_c4_calib_loader(
            tokenizer, num_seqs=num_seqs, max_length=max_length,
            batch_size=batch_size, seed=seed,
        )
    if source == "traces":
        return build_traces_jsonl_calib_loader(
            tokenizer,
            jsonl_path=getattr(args, "calib_traces_path"),
            num_seqs=num_seqs, max_length=max_length, batch_size=batch_size,
        )
    if source == "training_data":
        if training_dataset is not None and training_collate_fn is not None:
            return build_training_data_calib_loader(
                training_dataset, training_collate_fn,
                num_seqs=num_seqs, batch_size=batch_size, seed=seed,
            )
        if rl_rollout_fn is not None:
            return build_rl_rollout_calib_loader(
                rl_rollout_fn=rl_rollout_fn, tokenizer=tokenizer,
                num_seqs=num_seqs, batch_size=batch_size,
                max_length=max_length, seed=seed,
            )
        raise ValueError(
            "--calib-source=training_data requires either "
            "(training_dataset, training_collate_fn) or rl_rollout_fn"
        )
    raise ValueError(f"Unknown --calib-source {source!r}")


def apply_calibrated_btt(
    model, args, *, calib_loader, device: Optional[str] = None,
    hyphen_style: bool = True,
) -> Tuple[nn.Module, dict]:
    cfg = build_decomposition_config(args, hyphen_style=hyphen_style, model=model)
    model, stats = decompose_with_loader(
        model, cfg, calib_loader=calib_loader, device=device,
        return_trainability_stats=True,
    )
    if stats is None or stats.get("num_btt_layers", 0) == 0:
        raise ValueError("No BTT layers were installed; check --trainable-type selection.")
    return model, stats


def apply_calibrated_svd(
    model, args, *, calib_loader, device: Optional[str] = None,
    hyphen_style: bool = True,
) -> nn.Module:
    """Apply an SVD-LLM-v2 compression in-place. Used by full-FT scripts that
    want to initialize weights on the low-rank manifold before running
    standard supervised fine-tuning. After compression, every
    SVDCompressedLinear's U_r/V_r factors are flipped to requires_grad=True
    so the optimizer trains the factored weights end-to-end."""
    cfg = build_decomposition_config(args, hyphen_style=hyphen_style, model=model)
    if not cfg.train_mode.startswith("svd_"):
        raise ValueError(
            f"apply_calibrated_svd called with non-SVD train_mode={cfg.train_mode!r}; "
            "use apply_calibrated_btt for BTT modes."
        )
    model = decompose_with_loader(
        model, cfg, calib_loader=calib_loader, device=device,
        return_trainability_stats=False,
    )
    # SVDCompressedLinear initialises U_r/V_r with requires_grad=False (it is
    # used by frozen-compress callers too). For full-FT after compression we
    # must flip them on explicitly — the model-level requires_grad_(True) call
    # inside decompose_with_loader does cascade through Parameters, but we
    # re-assert it here for clarity and as a guard against future changes.
    from compress.svd.svd_linear import SVDCompressedLinear
    for module in model.modules():
        if isinstance(module, SVDCompressedLinear):
            module.U_r.requires_grad = True
            module.V_r.requires_grad = True
            if module.bias is not None:
                module.bias.requires_grad = True
    return model


@torch.no_grad()
def materialize_calibrated_btt_weights(model) -> List[Tuple[str, torch.Tensor]]:
    """Return [(param_name, dense_tensor)] for every BTTLinear in the model.

    For each BTTLinear `M` at path `p`, yields (`p.weight`, dense) and
    optionally (`p.bias`, bias). This list is appended to any other
    (name, tensor) pairs the caller assembles for vLLM weight sync.
    """
    out: List[Tuple[str, torch.Tensor]] = []
    for name, module in model.named_modules():
        if not isinstance(module, BTTLinear):
            continue
        out.append((f"{name}.weight", module.materialize_dense_weight()))
        if module.bias is not None:
            out.append((f"{name}.bias", module.bias.detach()))
    return out


def restore_calibrated_btt_weights(model, saved_state) -> None:
    """No-op: BTTLinear weights are never overwritten by vLLM weight export,
    only materialized-and-copied. The real factored cores remain in place
    on the training model, so no restore is needed. Provided for API
    symmetry with the legacy SVDLayer/BTTLayer restore flow."""
    return None


@torch.no_grad()
def materialize_calibrated_btt_to_linear(model: nn.Module) -> nn.Module:
    """In-place: replace every BTTLinear in `model` with an nn.Linear whose
    weight is the materialized dense equivalent. Mirrors the legacy
    `materialize_btt_to_linear` in ref/LIFT/src/finetune_blocktt.py but for
    the `compress` package's BTTLinear."""
    replacements = [(n, m) for n, m in model.named_modules() if isinstance(m, BTTLinear)]
    for name, btt in replacements:
        dense = btt.materialize_dense_weight()
        linear = nn.Linear(
            btt.in_features,
            btt.out_features,
            bias=btt.bias is not None,
            device=dense.device,
            dtype=dense.dtype,
        )
        linear.weight.data.copy_(dense)
        if btt.bias is not None:
            linear.bias.data.copy_(btt.bias.data)
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], linear)
    return model


@torch.no_grad()
def materialize_svd_to_linear(model: nn.Module) -> nn.Module:
    """In-place: replace every SVDCompressedLinear in `model` with an
    nn.Linear whose weight is the dense product `(V_r @ U_r).T`. Mirrors
    `materialize_calibrated_btt_to_linear` but for the SVD calib path.

    The factored forward in SVDCompressedLinear is `(x @ V_r) @ U_r + b`,
    which equals `x @ (V_r @ U_r) + b`. The standard nn.Linear computes
    `x @ W.T + b`, so the materialized weight is `(V_r @ U_r).T`.

    Trade-off: the materialized checkpoint is dense (same on-disk size as
    the uncompressed base model). This is intentional: it lets eval_math.sh
    load the model as a vanilla HF checkpoint with no custom modules. The
    *training* still happened on the low-rank manifold, which is the
    experimental quantity of interest.
    """
    from compress.svd.svd_linear import SVDCompressedLinear

    replacements = [(n, m) for n, m in model.named_modules()
                    if isinstance(m, SVDCompressedLinear)]
    for name, svd in replacements:
        # V_r: (d_in, rank), U_r: (rank, d_out)
        # Dense weight for nn.Linear is (d_out, d_in) = (V_r @ diag(svd_s) @ U_r).T
        # when svd_s is present; otherwise (V_r @ U_r).T.
        dense_weight = svd.materialize_dense_weight()
        d_out, d_in = dense_weight.shape
        linear = nn.Linear(
            d_in,
            d_out,
            bias=svd.bias is not None,
            device=dense_weight.device,
            dtype=dense_weight.dtype,
        )
        linear.weight.data.copy_(dense_weight)
        if svd.bias is not None:
            linear.bias.data.copy_(svd.bias.data)

        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], linear)
    return model


def save_calibrated_btt_checkpoint(model, out_dir: str, tokenizer=None) -> None:
    """LIFT legacy HF format: materialize BTTLinear -> nn.Linear, then write
    exactly `pytorch_model.bin` + `config.json`. Byte-for-byte identical
    to `ref/LIFT/src/utils/model_utils.py:save_hf_format`, so a converted
    calib checkpoint is indistinguishable from a non-calib blocktt one.

    `tokenizer` is accepted for signature symmetry with non-calib callers
    but intentionally not written — `save_hf_format` does not save the
    tokenizer either, and LIFT eval loads it from `base_model`.
    """
    del tokenizer  # unused; signature kept for symmetry
    os.makedirs(out_dir, exist_ok=True)
    materialize_calibrated_btt_to_linear(model)
    torch.save(model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))
    model.config.to_json_file(os.path.join(out_dir, "config.json"))


def save_calibrated_btt_hf_pretrained(model, out_dir: str) -> None:
    """run_sft / run_rl / run_rl_dapo format: materialize BTTLinear ->
    nn.Linear, then `model.save_pretrained(out_dir)`. Byte-for-byte
    identical to the non-calib branch of those scripts, which also just
    call `model.save_pretrained(ckpt_dir)`.

    Does NOT save the tokenizer; callers in run_* already invoke
    `tokenizer.save_pretrained(ckpt_dir)` once, outside the calib/non-calib
    branch, so both branches produce the same on-disk file set.
    """
    os.makedirs(out_dir, exist_ok=True)
    materialize_calibrated_btt_to_linear(model)
    model.save_pretrained(out_dir)


def load_calibrated_btt_for_eval(model, checkpoint_dir: str) -> nn.Module:
    """Read btt_topology.json, rebuild BTTLinear modules in `model`, load
    model.safetensors. Returns the mutated model. No calibration pass."""
    from safetensors.torch import load_file

    topology_path = os.path.join(checkpoint_dir, "btt_topology.json")
    with open(topology_path) as f:
        topology = json.load(f)

    rebuild_btt_from_topology(model, topology)

    state = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys in checkpoint: {unexpected[:5]}")
    # 'missing' is OK: topology only rebuilt BTT paths; other params may have
    # been loaded from state directly.
    return model


# ---------------------------------------------------------------------------
# Plain (calibration-free) BTT / SVD conversion via the compress package.
#
# These helpers replace the legacy `btt_layer.convert_linear_to_btt` /
# `svd_layer.convert_linear_to_svd` entry points used by run_rl_new.py /
# run_sft_new.py / etc. The factorization math is intentionally delegated
# to the legacy `btt_layer.BTTLayer` / `svd_layer.SVDLayer` classes so the
# numerical result is bit-for-bit identical with the legacy path; only the
# resulting **module type** is swapped to `compress.btt.btt_linear.BTTLinear`
# / `compress.svd.svd_linear.SVDCompressedLinear` so the rest of the
# training pipeline (rollouts, checkpointing) sees a single canonical
# module type for both plain and calibrated runs.
# ---------------------------------------------------------------------------


@torch.no_grad()
def convert_linear_to_btt_compress(
    model: nn.Module,
    *,
    btt_rank,
    decomp_mode="square",
    init_mode: str = "default",
    skip_names: Tuple[str, ...] = ("lm_head",),
    include_names: Optional[Sequence[str]] = None,
    lr_act: bool = False,
    s_merged_to: Optional[str] = None,
    train_position: str = "small",
    factorize_by_head: bool = False,
    model_config=None,
    convert_mode: str = "svd",
    input_factorization=None,
) -> List[str]:
    """Replace every targeted nn.Linear in ``model`` with a
    ``compress.btt.btt_linear.BTTLinear`` initialized from the linear's dense
    weight.

    The numerical factorization (SVD / QR / s_merged_to logic /
    factorize_by_head shape resolution) is performed by the legacy
    ``btt_layer.BTTLayer`` class. After ``init_from_linear_weight`` runs we
    copy the resulting packed cores (``btt_l``, ``btt_r``, optional
    ``btt_s``) into a freshly constructed ``BTTLinear`` and install it in
    the parent module.

    All keyword names match ``btt_layer.convert_linear_to_btt`` so callers
    can swap implementations by changing only the import.
    """
    from btt_layer import (  # local import to avoid mandatory dep at module load
        BTTLayer,
        convert_linear_to_btt as _legacy_convert_linear_to_btt,
    )

    # Run the legacy converter first; this installs BTTLayer modules and
    # performs the per-layer SVD/QR decomposition on CUDA.
    converted_names = _legacy_convert_linear_to_btt(
        model,
        btt_rank=btt_rank,
        decomp_mode=decomp_mode,
        init_mode=init_mode,
        skip_names=skip_names,
        include_names=include_names,
        lr_act=lr_act,
        s_merged_to=s_merged_to,
        train_position=train_position,
        factorize_by_head=factorize_by_head,
        model_config=model_config,
        convert_mode=convert_mode,
        input_factorization=input_factorization,
    )

    # Walk the model again and replace each BTTLayer with the equivalent
    # compress.BTTLinear, transferring the freshly-initialized cores.
    converted_set = set(converted_names)
    leaf_to_full = {}
    for name, module in model.named_modules():
        if isinstance(module, BTTLayer):
            leaf_to_full[name] = module
    targets = [(name, leaf_to_full[name]) for name in leaf_to_full
               if name in converted_set]

    for full_name, btt_layer in targets:
        btt_l = btt_layer.btt_l.detach().clone()
        btt_r = btt_layer.btt_r.detach().clone()
        bias = btt_layer.bias.detach().clone() if btt_layer.bias is not None else None
        if btt_layer.btt_s is not None:
            btt_s_param = btt_layer.btt_s
            btt_s = btt_s_param.detach().clone()
            btt_s_requires_grad = bool(btt_s_param.requires_grad)
        else:
            btt_s = None
            btt_s_requires_grad = False
        new_layer = BTTLinear(
            btt_l,
            btt_r,
            bias=bias,
            m=btt_layer.m,
            a=btt_layer.a,
            n=btt_layer.n,
            b=btt_layer.b,
            rank=btt_layer.rank,
            btt_s=btt_s,
            btt_s_requires_grad=btt_s_requires_grad,
        ).to(device=btt_layer.btt_l.device, dtype=btt_layer.btt_l.dtype)
        # Mirror the legacy trainability convention: requires_grad is set
        # only by the downstream `configure_*_trainability` pass, but copy
        # the just-initialized state across so the modules are in lock-step
        # before any configure step runs.
        new_layer.btt_l.requires_grad_(btt_layer.btt_l.requires_grad)
        new_layer.btt_r.requires_grad_(btt_layer.btt_r.requires_grad)
        if new_layer.btt_s is not None:
            new_layer.btt_s.requires_grad_(btt_s_requires_grad)
        if new_layer.bias is not None:
            new_layer.bias.requires_grad_(btt_layer.bias.requires_grad)

        path = full_name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        setattr(parent, path[-1], new_layer)

    return converted_names


def configure_compress_btt_trainability(
    model: nn.Module,
    train_bias: bool = True,
    train_position: str = "small",
    train_singular_values: bool = False,
) -> dict:
    """Trainability configuration mirroring
    ``btt_layer.configure_blocktt_trainability`` but for
    ``compress.btt.btt_linear.BTTLinear`` modules."""
    if train_position not in {"small", "large", "both"}:
        raise ValueError("BlockTT train_position must be one of: small, large, both")

    for p in model.parameters():
        p.requires_grad = False

    num_btt_layers = 0
    tuned_left_cores = 0
    tuned_right_cores = 0
    tuned_biases = 0

    for _, module in model.named_modules():
        if not isinstance(module, BTTLinear):
            continue
        num_btt_layers += 1
        left_size = module.btt_l.numel()
        right_size = module.btt_r.numel()

        if train_position == "both":
            train_left, train_right = True, True
        elif train_position == "small":
            train_left = left_size <= right_size
            train_right = not train_left
        else:  # large
            train_left = left_size >= right_size
            train_right = not train_left

        module.btt_l.requires_grad = train_left
        module.btt_r.requires_grad = train_right
        if module.btt_s is not None:
            module.btt_s.requires_grad = bool(train_singular_values)
        tuned_left_cores += int(train_left)
        tuned_right_cores += int(train_right)

        if module.bias is not None:
            module.bias.requires_grad = train_bias
            if train_bias:
                tuned_biases += 1

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    return {
        "num_btt_layers": num_btt_layers,
        "tuned_left_cores": tuned_left_cores,
        "tuned_right_cores": tuned_right_cores,
        "tuned_biases": tuned_biases,
        "trainable_param_count": trainable_param_count,
        "total_param_count": total_param_count,
        "trainable_params": trainable_params,
    }


@torch.no_grad()
def convert_linear_to_svd_compress(
    model: nn.Module,
    *,
    skip_names: Tuple[str, ...] = ("lm_head",),
    include_names: Optional[Sequence[str]] = None,
    s_merged_to: Optional[str] = None,
    train_position: str = "output",
) -> List[str]:
    """Replace every targeted nn.Linear in ``model`` with a
    ``compress.svd.svd_linear.SVDCompressedLinear`` initialized from the
    linear's dense weight.

    The numerical factorization (SVD truncation / s_merged_to splitting) is
    performed by the legacy ``svd_layer.SVDLayer`` class. After
    ``init_from_linear_weight`` runs we copy the resulting factors into a
    freshly constructed ``SVDCompressedLinear`` and install it in the
    parent module.
    """
    from svd_layer import (  # local import to avoid mandatory dep at module load
        SVDLayer,
        convert_linear_to_svd as _legacy_convert_linear_to_svd,
    )
    from compress.svd.svd_linear import SVDCompressedLinear

    converted_names = _legacy_convert_linear_to_svd(
        model,
        skip_names=skip_names,
        include_names=include_names,
        s_merged_to=s_merged_to,
        train_position=train_position,
    )

    converted_set = set(converted_names)
    leaf_to_full = {
        name: m for name, m in model.named_modules() if isinstance(m, SVDLayer)
    }
    targets = [(name, leaf_to_full[name]) for name in leaf_to_full
               if name in converted_set]

    for full_name, svd_layer in targets:
        # SVDLayer parametrizes forward as `F.linear(x, svd_a @ svd_b)` =
        # `x @ (svd_a @ svd_b).T`. SVDCompressedLinear parametrizes forward
        # as `x @ V_r @ U_r`. To get identical forward output we set
        #   V_r = svd_b.T,  U_r = svd_a.T
        # so that V_r @ U_r = svd_b.T @ svd_a.T = (svd_a @ svd_b).T = W.T,
        # and `x @ V_r @ U_r = x @ W.T` matches `F.linear(x, W)`.
        svd_b = svd_layer.svd_b.detach().clone()
        svd_a = svd_layer.svd_a.detach().clone()
        V_r = svd_b.t().contiguous()
        U_r = svd_a.t().contiguous()
        bias = svd_layer.bias.detach().clone() if svd_layer.bias is not None else None
        if svd_layer.svd_s is not None:
            svd_s_param = svd_layer.svd_s
            svd_s = svd_s_param.detach().clone()
            svd_s_requires_grad = bool(svd_s_param.requires_grad)
        else:
            svd_s = None
            svd_s_requires_grad = False
        new_layer = SVDCompressedLinear(
            U_r,
            V_r,
            bias=bias,
            svd_s=svd_s,
            svd_s_requires_grad=svd_s_requires_grad,
        ).to(device=svd_a.device, dtype=svd_a.dtype)

        new_layer.V_r.requires_grad_(svd_layer.svd_b.requires_grad)
        new_layer.U_r.requires_grad_(svd_layer.svd_a.requires_grad)
        if new_layer.svd_s is not None:
            new_layer.svd_s.requires_grad_(svd_s_requires_grad)
        if new_layer.bias is not None:
            new_layer.bias.requires_grad_(svd_layer.bias.requires_grad)

        path = full_name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        setattr(parent, path[-1], new_layer)

    return converted_names


def configure_compress_svd_trainability(
    model: nn.Module,
    train_position: str = "output",
    train_bias: bool = True,
    train_embed_lm_head: bool = False,
    train_singular_values: bool = False,
) -> dict:
    """Trainability configuration mirroring
    ``svd_layer.configure_svd_trainability`` but for
    ``compress.svd.svd_linear.SVDCompressedLinear`` modules.

    Mapping legacy <-> compress parameter naming:
      svd_a  <->  U_r   (output-side factor, shape (rank, d_out))
      svd_b  <->  V_r   (input-side factor,  shape (d_in, rank))
      svd_s  <->  svd_s (per-rank scale, shape (rank,))
    """
    from compress.svd.svd_linear import SVDCompressedLinear

    if train_position not in {"output", "input", "both"}:
        raise ValueError("SVD train_position must be one of: output, input, both")

    for p in model.parameters():
        p.requires_grad = False

    num_svd_layers = 0
    tuned_output_cores = 0
    tuned_input_cores = 0
    tuned_biases = 0

    for _, module in model.named_modules():
        if not isinstance(module, SVDCompressedLinear):
            continue
        num_svd_layers += 1
        if train_position in {"output", "both"}:
            module.U_r.requires_grad = True
            tuned_output_cores += 1
        if train_position in {"input", "both"}:
            module.V_r.requires_grad = True
            tuned_input_cores += 1
        if module.svd_s is not None:
            module.svd_s.requires_grad = bool(train_singular_values)
        if module.bias is not None:
            module.bias.requires_grad = train_bias
            if train_bias:
                tuned_biases += 1

    if train_embed_lm_head:
        for name, module in model.named_modules():
            leaf_name = name.split(".")[-1]
            if leaf_name in ("embed_tokens", "lm_head"):
                for p in module.parameters():
                    p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    return {
        "num_svd_layers": num_svd_layers,
        "tuned_output_cores": tuned_output_cores,
        "tuned_input_cores": tuned_input_cores,
        "tuned_biases": tuned_biases,
        "trainable_param_count": trainable_param_count,
        "total_param_count": total_param_count,
        "trainable_params": trainable_params,
    }
