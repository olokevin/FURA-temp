"""Collect the first-step gradient G of a full fine-tuning run.

Replicates one optimizer-step worth of micro-batches (per_device_batch_size *
gradient_accumulation_steps) on a single GPU using the same seed, dataset,
collator, dtype and gradient-checkpointing setting as
``ref/LIFT/src/finetune_sft.py``. Saves G as a {key -> fp32 cpu tensor} dict
keyed by the HF state-dict name (``model.layers.{L}.self_attn.{mod}.weight``
or ``model.layers.{L}.mlp.{mod}.weight``) so it can be loaded alongside the
base / trained safetensors with the same key strings.

Honesty caveats:
- We do NOT call ``accelerator.prepare(...)``, so DataLoader RNG ordering is
  not byte-identical to the actual training run. This is an honest first-step
  gradient on the same data distribution + seed, not a byte-for-byte replay.
- bf16 + grad-checkpointing + flash-attn together introduce ~1e-3 relative
  drift between repeated runs. Acceptable for spectral / projection metrics.

Example::

    CUDA_VISIBLE_DEVICES=0 uv run python analysis/compute_first_step_grad.py \\
        --base-model meta-llama/Meta-Llama-3-8B \\
        --data-path ref/LIFT/LLM-Adapters/ft-training_set/commonsense_170k.json \\
        --seed 43 \\
        --output analysis_results/grad_step1__llama3-8b__seed43.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
LIFT_SRC = REPO_ROOT / "ref" / "LIFT" / "src"
if str(LIFT_SRC) not in sys.path:
    sys.path.insert(0, str(LIFT_SRC))

from accelerate.utils import set_seed  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from utils.data_utils import SupervisedDataset, DataCollatorForSupervisedDataset  # noqa: E402
from utils.model_utils import (  # noqa: E402
    load_hf_tokenizer,
    make_model_gradient_checkpointing_compatible,
)


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")
ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def _module_prefix(layer_idx: int, module_name: str) -> str:
    if module_name in ATTN_MODULES:
        return f"model.layers.{layer_idx}.self_attn.{module_name}"
    return f"model.layers.{layer_idx}.mlp.{module_name}"


def _name_is_target(name: str) -> bool:
    if not name.endswith(".weight"):
        return False
    parts = name.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        return False
    if not parts[2].isdigit():
        return False
    sub, mod = parts[3], parts[4]
    if sub == "self_attn" and mod in ATTN_MODULES:
        return True
    if sub == "mlp" and mod in ("gate_proj", "up_proj", "down_proj"):
        return True
    return False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True,
                   help="HF model id or local path (e.g. meta-llama/Meta-Llama-3-8B)")
    p.add_argument("--data-path", required=True,
                   help="JSON dataset path, e.g. ref/LIFT/LLM-Adapters/ft-training_set/commonsense_170k.json")
    p.add_argument("--instruction-type", default="single",
                   choices=["single", "multi"])
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--per-device-batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--use-flash-attn", default="True", choices=["True", "False"])
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="Disable gradient checkpointing (default: ENABLED to match training)")
    p.add_argument("--output", required=True,
                   help="Output .pt path. A sibling .json sidecar with config is also written.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def build_model(args, tokenizer):
    """Mirror finetune_sft.py:332-390."""
    config = AutoConfig.from_pretrained(args.base_model)
    base_kwargs = dict(config=config, dtype=torch.bfloat16)

    if args.use_flash_attn == "True":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model, attn_implementation="flash_attention_2", **base_kwargs)
        except (TypeError, ValueError, ImportError):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.base_model, use_flash_attention_2=True, **base_kwargs)
            except (TypeError, ValueError, ImportError):
                print("Warning: flash-attn unsupported here; loading without it.", flush=True)
                model = AutoModelForCausalLM.from_pretrained(args.base_model, **base_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **base_kwargs)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

    if not args.no_gradient_checkpointing:
        model = make_model_gradient_checkpointing_compatible(model)
        model.gradient_checkpointing_enable()

    return model


def main():
    args = parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"[skip] {output_path} already exists. Pass --overwrite to recompute.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[seed]   {args.seed}")
    print(f"[batch]  per_device={args.per_device_batch_size}, "
          f"grad_accum={args.gradient_accumulation_steps}")
    print(f"[model]  {args.base_model}  (bf16, "
          f"grad-ckpt={'OFF' if args.no_gradient_checkpointing else 'ON'}, "
          f"flash-attn={args.use_flash_attn})")

    set_seed(args.seed)

    tokenizer = load_hf_tokenizer(args.base_model, fast_tokenizer=True)
    tokenizer.model_max_length = args.max_seq_len

    print("[load]   building model ...")
    t0 = time.time()
    model = build_model(args, tokenizer)
    device = torch.device(args.device)
    model.to(device)
    model.train()
    print(f"[load]   model on {device} in {time.time() - t0:.1f}s")

    print(f"[data]   tokenizing {args.data_path} ...")
    fake_args = SimpleNamespace(
        model_name_or_path=args.base_model,
        global_rank=0,
    )
    dataset = SupervisedDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        instruction_type=args.instruction_type,
        args=fake_args,
    )
    print(f"[data]   {len(dataset)} examples")

    collator = DataCollatorForSupervisedDataset(tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    model.zero_grad(set_to_none=True)
    loader_iter = iter(loader)
    losses = []
    t0 = time.time()
    for step in range(args.gradient_accumulation_steps):
        batch = next(loader_iter)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / args.gradient_accumulation_steps
        loss.backward()
        losses.append(float(outputs.loss.detach()))
        print(f"[step {step + 1}/{args.gradient_accumulation_steps}]  "
              f"loss = {losses[-1]:.4f}  ({time.time() - t0:.1f}s elapsed)")

    print(f"[grad]   forward+backward done in {time.time() - t0:.1f}s")

    grads: dict[str, torch.Tensor] = {}
    n_missing = 0
    for name, param in model.named_parameters():
        if not _name_is_target(name):
            continue
        if param.grad is None:
            n_missing += 1
            continue
        grads[name] = param.grad.detach().to(torch.float32).cpu().contiguous()

    if not grads:
        raise RuntimeError("No target gradients collected — check naming / requires_grad.")
    if n_missing:
        print(f"[warn]   {n_missing} target params had no .grad")

    print(f"[save]   {len(grads)} tensors -> {output_path}")
    torch.save(grads, str(output_path))

    sidecar = output_path.with_suffix(".json")
    with open(sidecar, "w") as f:
        json.dump({
            "base_model": args.base_model,
            "data_path": args.data_path,
            "instruction_type": args.instruction_type,
            "seed": args.seed,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_seq_len": args.max_seq_len,
            "use_flash_attn": args.use_flash_attn,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "dtype": "bfloat16",
            "loss_per_microbatch": losses,
            "n_target_params": len(grads),
        }, f, indent=2)
    print(f"[save]   sidecar -> {sidecar}")


if __name__ == "__main__":
    main()
