import sys
import os

# Add LIFT parent to path (same pattern as other LIFT scripts)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
# Add repo root to path for svft_layer.py
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
)

import copy
import time
import torch
import math
import argparse
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)

from utils.utils import (
    print_rank_0,
    get_all_reduce_mean,
    int_or_float,
)

from accelerate import Accelerator
from accelerate.utils import set_seed

from utils.model_utils import (
    load_hf_tokenizer,
    save_hf_format,
    make_model_gradient_checkpointing_compatible,
)

from utils.data_utils import SupervisedDataset, DataCollatorForSupervisedDataset

from svft_layer import (
    SVFTLayer,
    convert_linear_to_svft,
    configure_svft_trainability,
    get_svft_target_module_names,
)

from tools.system_metrics import SysMon


def materialize_svft_to_linear(model):
    """Replace all SVFTLayer modules with nn.Linear containing materialized dense
    weights so the saved HF checkpoint contains standard Linear weights instead of the
    frozen U/V bases + sparse coefficients.  Mirrors materialize_svd_to_linear."""
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, SVFTLayer):
            replacements.append((name, module))

    for name, svft_module in replacements:
        dense_weight = svft_module.materialize_dense_weight()
        linear = nn.Linear(
            svft_module.in_features,
            svft_module.out_features,
            bias=svft_module.bias is not None,
            device=dense_weight.device,
            dtype=dense_weight.dtype,
        )
        linear.weight.data.copy_(dense_weight)
        if svft_module.bias is not None:
            linear.bias.data.copy_(svft_module.bias.data)

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], linear)

    print(f"Materialized {len(replacements)} SVFTLayer modules to nn.Linear")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="SVFT Fine-Tuning (LIFT benchmark)")
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["./LLM-Adapters/ft-training_set/commonsense_170k.json"],
        help="Path to the training dataset (json).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=16,
        help="Batch size (per device) for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=16,
        help="Batch size (per device) for evaluation.",
    )
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--val_set_size", type=int, default=100,
        help="Size of the validation set. If 0, no validation set is used.")
    parser.add_argument("--load_last_model", action="store_true",
        help="Skip best-model tracking, save only the last model.")
    parser.add_argument("--eval_step", type=int, default=80)
    parser.add_argument("--eval_delay", type=int_or_float, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="If > 0, cap total optimizer steps at this value (for short-horizon system-eval runs).",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                 "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=float, default=0.03)
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--instruction_type", type=str, choices=["single", "multi"], default="single",
    )
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument(
        "--use_flash_attn", type=str, default="False",
    )

    # SVFT-specific arguments
    parser.add_argument(
        "--trainable_type", type=str, default="udog",
        choices=["udog", "all", "mlp", "attn"],
        help="Which modules to convert to SVFT. Default 'udog' = up/down/o/gate "
             "(paper's LLaMA-3-8B recipe).",
    )
    parser.add_argument(
        "--svft_pattern", type=str, default="banded",
        choices=["plain", "banded", "random", "topk"],
        help="Sparsity pattern for the trainable coefficient matrix M.",
    )
    parser.add_argument(
        "--svft_off_diag", type=int, default=8,
        help="Bandwidth d (off-diagonals per side) for the banded pattern.",
    )
    parser.add_argument(
        "--svft_num_coeffs", type=int, default=0,
        help="Number of learnable coefficients for random/topk patterns (0 -> k).",
    )
    parser.add_argument("--no_train_bias", action="store_true",
        help="Freeze biases on SVFT layers")
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="Weights & Biases run name.",
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable Weights & Biases logging.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    use_wandb = not args.no_wandb
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if use_wandb else None,
    )
    if not torch.cuda.is_available() or accelerator.device.type != "cuda":
        raise RuntimeError(
            "finetune_svft.py requires CUDA so SVD decomposition runs on GPU. "
            f"Current accelerator device: {accelerator.device}."
        )

    set_seed(args.seed)
    args.global_rank = 1

    if use_wandb:
        if args.wandb_project is None:
            args.wandb_project = "lift"
        tracker_config = vars(args).copy()
        wandb_init_kwargs = {}
        if args.wandb_run_name:
            wandb_init_kwargs["name"] = args.wandb_run_name
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=tracker_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    # Load tokenizer
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tokenizer.model_max_length = args.max_seq_len

    # Load model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.use_flash_attn == "True":
        model_kwargs["use_flash_attention_2"] = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        **model_kwargs,
    )
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))
    model = model.to(accelerator.device)

    # --- Dataset ---
    if len(args.data_path) == 1 and ".json" in args.data_path[0]:
        train_dataset = SupervisedDataset(
            data_path=args.data_path[0],
            tokenizer=tokenizer,
            instruction_type=args.instruction_type,
            args=args,
        )
        if args.val_set_size > 0:
            train_dataset, eval_dataset = torch.utils.data.random_split(
                train_dataset,
                [len(train_dataset) - args.val_set_size, args.val_set_size],
            )
    else:
        raise ValueError("Only json format is supported for now.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    if args.val_set_size > 0:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

    # Pretrained parameter footprint (counted BEFORE conversion; SVFT replaces the
    # target Linear weights with frozen U/V/s *buffers*, so the post-conversion
    # parameter total no longer reflects the base model).
    pretrained_param_count = sum(p.numel() for p in model.parameters())

    # --- SVFT conversion (reparametrize pretrained weights in the singular basis) ---
    target_modules = get_svft_target_module_names(args.trainable_type)
    train_bias = not args.no_train_bias

    converted_modules = convert_linear_to_svft(
        model,
        skip_names=("lm_head",),
        include_names=target_modules,
        pattern=args.svft_pattern,
        off_diag=args.svft_off_diag,
        num_coeffs=args.svft_num_coeffs,
        seed=args.seed,
    )
    stats = configure_svft_trainability(model, train_bias=train_bias)
    if stats["num_svft_layers"] == 0:
        raise ValueError("No layers were converted to SVFT; check --trainable_type.")

    print(f"Converted modules: {len(converted_modules)}")
    print(
        f"Trainable params: {stats['trainable_param_count']:,} / "
        f"{stats['total_param_count']:,} "
        f"({100 * stats['trainable_param_count'] / stats['total_param_count']:.4f}%)"
    )
    print(
        f"SVFT layers: {stats['num_svft_layers']}, pattern={args.svft_pattern}, "
        f"off_diag={args.svft_off_diag}, biases={stats['tuned_biases']}"
    )

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"param {name} is trainable")

    if args.gradient_checkpointing:
        model = make_model_gradient_checkpointing_compatible(model)
        model.gradient_checkpointing_enable()

    # --- Optimizer: standard AdamW on trainable SVFT coefficients ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    if args.max_steps > 0:
        max_train_steps = min(max_train_steps, args.max_steps)

    if args.num_warmup_steps < 1:
        args.num_warmup_steps = int(args.num_warmup_steps * max_train_steps)
    else:
        args.num_warmup_steps = int(args.num_warmup_steps)

    print(f"max trainable steps: {max_train_steps}, warmup steps: {args.num_warmup_steps}")
    total_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )

    print("***** Running SVFT training *****")
    print(f"  Num examples = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    print(f"  SVFT pattern = {args.svft_pattern} (off_diag={args.svft_off_diag})")

    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    args.completed_steps = 0

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if args.val_set_size > 0:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    best_model = None

    # base_params = pretrained footprint (captured before conversion). SVFT stores the
    # frozen singular bases as buffers, so trainable_pct / stored_extra_pct in SysMon
    # are measured against the true base model, not the reparametrized parameter total.
    sysmon = SysMon(
        out_dir=args.output_dir or ".",
        method="svft",
        rank=None,
        base_params=pretrained_param_count,
    )

    def train_epoch(epoch):
        nonlocal best_model, best_eval_loss
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                total_loss += loss.detach().float()

            if accelerator.sync_gradients:
                _t0 = time.time()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                sysmon.record_step(time.time() - _t0)
                progress_bar.update(1)
                args.completed_steps += 1
                if args.max_steps > 0 and args.completed_steps >= args.max_steps:
                    return

                if (
                    args.logging_steps
                    and args.completed_steps % args.logging_steps == 0
                ):
                    divisor = args.gradient_accumulation_steps * args.logging_steps
                    avg_loss = (
                        accelerator.gather(total_loss).mean().item() / divisor
                    )
                    print(
                        f"  Step: {args.completed_steps}, "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.8f}, "
                        f"Loss: {avg_loss:.6f}"
                    )
                    accelerator.log(
                        {
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "train_loss": avg_loss,
                        },
                        step=args.completed_steps,
                    )
                    total_loss = 0

                if (
                    args.completed_steps % args.eval_step == 0
                    and args.val_set_size > 0
                    and not args.load_last_model
                ):
                    perplexity, eval_loss = evaluate(model)
                    accelerator.print(
                        f"Epoch {epoch+1} Step {args.completed_steps}: "
                        f"Eval perplexity = {perplexity:.4f}, Eval loss = {eval_loss:.4f}"
                    )
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if accelerator.is_main_process and args.output_dir:
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            best_model = copy.deepcopy(unwrapped_model).to("cpu")
                            print("New best model")

        return total_loss / len(train_dataloader)

    def evaluate(model):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            losses = get_all_reduce_mean(losses)
        except Exception:
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        model.train()
        return perplexity, losses.item()

    # --- Training loop ---
    best_eval_loss = float("inf")
    for epoch in range(args.num_train_epochs):
        train_loss = train_epoch(epoch)
        if train_loss is not None:
            accelerator.print(f"Epoch {epoch+1}: Average loss = {train_loss:.4f}")
        if args.max_steps > 0 and args.completed_steps >= args.max_steps:
            break

    effective_tokens = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * args.max_seq_len
    )
    # "Extra parameters stored on top of the pretrained model": the SVFT adapter keeps
    # the frozen U/V/s buffers, whose factored footprint (d1*k + d2*k + k per layer) is
    # larger than the original dense weight (d1*d2) it replaced, plus the trainable
    # m_coeffs.  We report the net storage increase vs the base dense weights.
    unwrapped = accelerator.unwrap_model(model)
    factored_store = 0
    replaced_dense = 0
    trainable_adapter = 0
    for _, mod in unwrapped.named_modules():
        if isinstance(mod, SVFTLayer):
            factored_store += mod.svft_u.numel() + mod.svft_v.numel() + mod.svft_s.numel()
            replaced_dense += mod.out_features * mod.in_features
            trainable_adapter += mod.m_coeffs.numel()
    extra_params = (factored_store - replaced_dense) + trainable_adapter
    sysmon.dump(
        model,
        extra={
            "effective_tokens_per_step": effective_tokens,
            "learning_rate": args.learning_rate,
            "svft_pattern": args.svft_pattern,
            "svft_off_diag": args.svft_off_diag,
            "trainable_type": args.trainable_type,
            "extra_params": extra_params,
            "extra_params_pct": 100.0 * extra_params / pretrained_param_count,
            "trainable_params_true": trainable_adapter,
            "trainable_pct_true": 100.0 * trainable_adapter / pretrained_param_count,
        },
    )

    # --- Save policy: write last/ always, best/ if best-tracking ran.
    def _save_one(src_model, sub_folder):
        materialize_svft_to_linear(src_model)
        save_hf_format(src_model, tokenizer, args, sub_folder=sub_folder)

    if args.output_dir is not None and accelerator.is_main_process:
        accelerator.wait_for_everyone()

        if args.val_set_size > 0 and not args.load_last_model:
            ppl, val_loss = evaluate(model)
            print_rank_0(
                f"Validation perplexity: {ppl}, Validation loss: {val_loss}",
                args.global_rank,
            )
            if val_loss < best_eval_loss:
                best_eval_loss = val_loss
                if args.global_rank == 0:
                    best_model = copy.deepcopy(model.module).to("cpu")

        last_model = accelerator.unwrap_model(model)
        _save_one(last_model, "last")
        print_rank_0(f"Saved last-step checkpoint to {os.path.join(args.output_dir, 'last')}", args.global_rank)

        if best_model is not None:
            _save_one(best_model, "best")
            print_rank_0(
                f"Saved best-eval checkpoint to {os.path.join(args.output_dir, 'best')} "
                f"(val_loss={best_eval_loss:.4f})",
                args.global_rank,
            )

    if use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
