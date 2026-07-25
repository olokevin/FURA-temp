"""Benchmark FuRA init overhead: time + memory to convert nn.Linear to BTTLayer.

Loads a base Llama-3-8B (bf16, GPU), measures parameter count and memory,
then runs convert_linear_to_btt() and re-measures. Reports:
  - Pre-convert: param count, peak GPU memory
  - Convert wall time (CPU + GPU sync)
  - Post-convert: param count, peak GPU memory
  - Stored extra (added params), trainable params after train_position=small
  - For LoRA/DoRA comparison: also time get_peft_model attach
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import argparse
import gc
import json
import time
from pathlib import Path


def _measure(fn, label):
    """Run fn(), return (result, wall_seconds, peak_gb_delta)."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
    t0 = time.time()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.time() - t0
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        peak_gb = peak / (1024**3)
        mem_after = torch.cuda.memory_allocated()
        delta_gb = (mem_after - mem_before) / (1024**3)
    else:
        peak_gb = 0.0
        delta_gb = 0.0
    print(f"  [{label}] wall={wall:.2f}s, peak_gb={peak_gb:.2f}, delta_gb={delta_gb:+.2f}")
    return result, wall, peak_gb, delta_gb


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def bench_fura(
    base_model: str,
    target_modules,
    decomp_mode: str = "output_one_block",
    s_merged_to: str = "keep_trainable",
    train_position: str = "small",
):
    import torch
    from transformers import AutoModelForCausalLM
    from btt_layer import (
        convert_linear_to_btt,
        configure_blocktt_trainability,
    )

    print(f"\n=== FuRA init overhead on {base_model} ===")
    torch.cuda.empty_cache()
    gc.collect()

    def _load():
        return AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="cuda"
        )

    model, load_wall, load_peak, load_delta = _measure(_load, "load base")
    base_total, base_trainable = _count_params(model)
    print(f"  base: total={base_total:,} trainable={base_trainable:,}")

    def _convert():
        return convert_linear_to_btt(
            model,
            btt_rank="full",
            decomp_mode=decomp_mode,
            init_mode="default",
            include_names=list(target_modules),
            skip_names=("lm_head",),
            lr_act=False,
            s_merged_to=s_merged_to,
            train_position=train_position,
        )

    converted, conv_wall, conv_peak, conv_delta = _measure(_convert, "convert linear -> BTT")
    n_converted = len(converted) if converted is not None else 0

    def _configure():
        configure_blocktt_trainability(model, train_position=train_position)

    _, cfg_wall, cfg_peak, cfg_delta = _measure(_configure, "configure trainability")

    after_total, after_trainable = _count_params(model)
    print(f"  after: total={after_total:,} trainable={after_trainable:,} converted_modules={n_converted}")

    return {
        "method": "fura",
        "base_model": base_model,
        "decomp_mode": decomp_mode,
        "s_merged_to": s_merged_to,
        "train_position": train_position,
        "n_converted_modules": n_converted,
        "load_wall_s": load_wall,
        "convert_wall_s": conv_wall,
        "configure_wall_s": cfg_wall,
        "total_init_wall_s": load_wall + conv_wall + cfg_wall,
        "load_peak_gb": load_peak,
        "convert_peak_gb": conv_peak,
        "configure_peak_gb": cfg_peak,
        "convert_mem_delta_gb": conv_delta,
        "base_total_params": base_total,
        "after_total_params": after_total,
        "after_trainable_params": after_trainable,
        "stored_extra_params": after_total - base_total,
        "stored_extra_pct": 100.0 * (after_total - base_total) / base_total,
        "trainable_pct": 100.0 * after_trainable / base_total,
    }


def bench_lora(base_model: str, rank: int, target_modules, use_dora: bool = False):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    label = "dora" if use_dora else "lora"
    print(f"\n=== {label.upper()}-r{rank} init overhead on {base_model} ===")
    torch.cuda.empty_cache()
    gc.collect()

    def _load():
        return AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="cuda"
        )

    model, load_wall, load_peak, load_delta = _measure(_load, "load base")
    base_total, _ = _count_params(model)

    def _attach():
        cfg = LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            target_modules=list(target_modules),
            use_dora=use_dora,
        )
        return get_peft_model(model, cfg)

    peft_model, attach_wall, attach_peak, attach_delta = _measure(_attach, f"attach {label}")
    after_total, after_trainable = _count_params(peft_model)

    return {
        "method": label,
        "rank": rank,
        "base_model": base_model,
        "load_wall_s": load_wall,
        "attach_wall_s": attach_wall,
        "total_init_wall_s": load_wall + attach_wall,
        "load_peak_gb": load_peak,
        "attach_peak_gb": attach_peak,
        "attach_mem_delta_gb": attach_delta,
        "base_total_params": base_total,
        "after_total_params": after_total,
        "after_trainable_params": after_trainable,
        "stored_extra_params": after_total - base_total,
        "stored_extra_pct": 100.0 * (after_total - base_total) / base_total,
        "trainable_pct": 100.0 * after_trainable / base_total,
    }


def bench_qfura_stream(
    base_model: str,
    target_modules,
    decomp_mode: str = "output_one_block",
    s_merged_to: str = "keep_trainable",
    train_position: str = "small",
    quant_layout: str = "flat",
):
    """Init cost of the layer-streaming BTT+NF4 path used for >=30B models.

    Mirrors finetune_qfura.py's `load_strategy=layer_stream` branch: the base
    model is loaded on CPU, each targeted Linear is factorised and NF4-quantised
    through a transient CUDA round-trip, and the residual (embed/norms/lm_head)
    is moved to GPU afterwards.
    """
    import torch
    from transformers import AutoModelForCausalLM
    from btt_layer import (
        QBTTLayer,
        convert_and_quantize_linear_to_qbtt_streaming,
    )

    print(f"\n=== qfura (streaming) init overhead on {base_model} ===")
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda")

    def _load():
        return AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )

    model, load_wall, load_peak, _ = _measure(_load, "load base (CPU)")
    base_total, _ = _count_params(model)
    print(f"  base: total={base_total:,}")

    def _convert():
        return convert_and_quantize_linear_to_qbtt_streaming(
            model,
            btt_rank="full",
            decomp_mode=decomp_mode,
            train_position=train_position,
            s_merged_to=s_merged_to,
            quant_layout=quant_layout,
            target_modules=tuple(target_modules),
            cuda_device=device,
            skip_names=("lm_head",),
            convert_mode="svd",
            init_mode="default",
        )

    stats, conv_wall, conv_peak, _ = _measure(_convert, "stream convert+quantise -> QBTT")

    def _residual():
        qbtt_ids = set()
        for _, mod in model.named_modules():
            if isinstance(mod, QBTTLayer):
                for p in mod.parameters(recurse=False):
                    qbtt_ids.add(id(p))
        for _, p in model.named_parameters():
            if id(p) not in qbtt_ids:
                p.requires_grad = False
        return model.to(device)

    _, res_wall, res_peak, _ = _measure(_residual, "move residual -> GPU")

    after_total, after_trainable = _count_params(model)
    return {
        "method": "qfura_stream",
        "base_model": base_model,
        "decomp_mode": decomp_mode,
        "s_merged_to": s_merged_to,
        "train_position": train_position,
        "n_converted_modules": stats["num_converted"],
        "bytes_saved": stats["bytes_saved"],
        "load_wall_s": load_wall,
        "convert_wall_s": conv_wall,
        "residual_move_wall_s": res_wall,
        "total_init_wall_s": load_wall + conv_wall + res_wall,
        "convert_peak_gb": conv_peak,
        "residual_peak_gb": res_peak,
        "base_total_params": base_total,
        "after_total_params": after_total,
        "after_trainable_params": after_trainable,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
    )
    ap.add_argument("--methods", nargs="+", default=["fura", "lora", "dora"])
    ap.add_argument("--decomp_mode", default="output_one_block")
    ap.add_argument("--s_merged_to", default="keep_trainable")
    ap.add_argument("--train_position", default="small")
    args = ap.parse_args()

    results = []
    if "fura" in args.methods:
        results.append(
            bench_fura(
                args.base_model,
                args.target_modules,
                decomp_mode=args.decomp_mode,
                s_merged_to=args.s_merged_to,
                train_position=args.train_position,
            )
        )
    if "qfura_stream" in args.methods:
        results.append(
            bench_qfura_stream(
                args.base_model,
                args.target_modules,
                decomp_mode=args.decomp_mode,
                s_merged_to=args.s_merged_to,
                train_position=args.train_position,
            )
        )
    if "lora" in args.methods:
        results.append(bench_lora(args.base_model, 64, args.target_modules, use_dora=False))
    if "dora" in args.methods:
        results.append(bench_lora(args.base_model, 64, args.target_modules, use_dora=True))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} results to {args.out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
