import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import logging
import os
import sys

# Make the lora-without-regret project root importable so we can pull in
# `btt_layer` (FURA's BlockTT layer + conversion helpers). This file lives at
# <repo>/ref/PiSSA/train.py, so the repo root is two levels up.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.distributed
import torch.nn as nn
import transformers
from transformers import Trainer, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
import datasets
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel, LoraRuntimeConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from btt_layer import (
    BTTLayer,
    convert_linear_to_btt,
    configure_blocktt_trainability,
    get_blocktt_target_module_names,
    resolve_blocktt_decomp_modes,
)

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Base model or residual model setting
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    attn_implementation : Optional[str] = field(default="flash_attention_2")
    # Lora or PiSSA setting
    full_finetune : Optional[bool] = field(default=True)
    adapter_name_or_path: Optional[str] = field(default=None,metadata={"help": ("Pre-initialized PiSSA adapter path; when this is not None, the following arguments are ignored."),},)
    init_weights: bool | str = field(default=True,metadata={"help": ("True -> LoRA; `pissa` -> PiSSA; `pissa_niter_16` -> Fast SVD PiSSA"),},)
    use_dora : Optional[bool] = field(default=False)
    target_modules : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_alpha : Optional[float] = field(default=32.)
    lora_dropout : Optional[float] = field(default=0.,metadata={"help": ("Must be set to 0 when using PiSSA."),},)
    # Quantization setting
    bits: int = field(default=16,metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True,metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    # DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    sub_task: List[str] = field(default=None)
    dataset_split: str = field(default="train", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    shuffle_dataset : Optional[bool] = field(default=False)
    # TrainingArguments
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512,metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    merge : Optional[bool] = field(default=False,metadata={"help": "Merge the PiSSA adapter to the residual model or LoRA to the base model"},)

    # FURA (BlockTT) settings — mirrors the project defaults documented in
    # CLAUDE.md and ref/LIFT/src/finetune_qfura.py. When `fura=True` the LoRA /
    # PiSSA branch in build_model is skipped entirely; instead every targeted
    # nn.Linear is replaced in place by a BTTLayer.
    fura : Optional[bool] = field(default=False, metadata={"help": "Use FURA (BlockTT) instead of LoRA/PiSSA/full-FT."})
    fura_trainable_type : Optional[str] = field(default="all", metadata={"help": "all|mlp|attn — which linears to convert to BTT."})
    fura_decomp_mode : Optional[str] = field(default="output_one_block", metadata={"help": "input_one_block|output_one_block, or a per-group dict literal."})
    fura_blocktt_rank : Optional[str] = field(default="full", metadata={"help": "'full' for lossless or a positive integer."})
    fura_train_position : Optional[str] = field(default="small", metadata={"help": "small|large|both."})
    fura_s_merged_to : Optional[str] = field(default="keep_trainable", metadata={"help": "frozen|trainable|output|input|split|keep_frozen|keep_trainable."})
    fura_factorize_by_head : Optional[bool] = field(default=True, metadata={"help": "Align attention BTT blocks with head structure."})
    fura_train_bias : Optional[bool] = field(default=True, metadata={"help": "Train bias on BTT layers (no-op when source linears are bias-less)."})

class SavePeftModelCallback(transformers.TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        tokenizer = kwargs.get("tokenizer", self.tokenizer)
        tokenizer.save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None # first training

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def resolve_blocktt_rank(rank_arg):
    """Parse --fura_blocktt_rank: 'full' or a positive integer."""
    if rank_arg == "full":
        return "full"
    try:
        rank = int(rank_arg)
    except ValueError as exc:
        raise ValueError("--fura_blocktt_rank must be 'full' or a positive integer") from exc
    if rank <= 0:
        raise ValueError("--fura_blocktt_rank must be > 0")
    return rank


def materialize_btt_to_linear(model):
    """Replace every BTTLayer with an nn.Linear holding its dense materialization.

    Mirrors `materialize_btt_to_linear` in ref/LIFT/src/finetune_blocktt.py.
    Used at save time so the checkpoint round-trips through standard HF I/O.
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, BTTLayer):
            replacements.append((name, module))
    for name, btt_module in replacements:
        dense_weight = btt_module.materialize_dense_weight()
        linear = nn.Linear(
            btt_module.in_features,
            btt_module.out_features,
            bias=btt_module.bias is not None,
            device=dense_weight.device,
            dtype=dense_weight.dtype,
        )
        linear.weight.data.copy_(dense_weight)
        if btt_module.bias is not None:
            linear.bias.data.copy_(btt_module.bias.data)
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], linear)
    return model


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length,truncation=True,)for text in strings]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def build_model(script_args, checkpoint_dir):
    if script_args.fura and script_args.full_finetune:
        raise ValueError("--fura and --full_finetune are mutually exclusive.")
    if script_args.fura and script_args.bits not in (16, 32):
        raise ValueError(
            "--fura currently requires --bits 16 (bf16/fp16). The base model needs "
            "to be on CUDA in a real dtype so SVD-based BTT init runs on GPU. "
            "Use qfura (NF4 quantization of the frozen core) for 4-bit setups; "
            "see ref/LIFT/src/finetune_qfura.py."
        )
    if script_args.full_finetune:
        assert script_args.bits in [16, 32]
    compute_dtype = (torch.bfloat16 if script_args.bf16 else torch.float32)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.double_quant,
            bnb_4bit_quant_type=script_args.quant_type,
        ) if script_args.bits in [4, 8] else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    # Tokenizer

    if script_args.fura:
        # FURA path: replace targeted Linear layers with BTTLayer, then freeze
        # everything except the chosen TT cores. Trainer treats the result like
        # a plain bf16 model with most params frozen — no PEFT adapter is built.
        if checkpoint_dir is not None or script_args.adapter_name_or_path is not None:
            raise NotImplementedError(
                "FURA does not support --adapter_name_or_path / checkpoint resume "
                "yet (the BTT cores are not stored as a PEFT adapter)."
            )
        # Make sure the base model lives on CUDA so SVD inside BTT init runs on GPU.
        if not next(model.parameters()).is_cuda:
            model = model.to(torch.cuda.current_device())

        blocktt_rank = resolve_blocktt_rank(script_args.fura_blocktt_rank)
        target_modules = get_blocktt_target_module_names(script_args.fura_trainable_type)
        decomp_mode_resolved, module_decomp_modes = resolve_blocktt_decomp_modes(
            script_args.fura_decomp_mode,
            include_names=target_modules,
            default_mode="output_one_block",
        )
        convert_linear_to_btt(
            model,
            btt_rank=blocktt_rank,
            decomp_mode=module_decomp_modes if module_decomp_modes is not None else decomp_mode_resolved,
            init_mode="default",
            include_names=target_modules,
            skip_names=("lm_head",),
            lr_act=False,
            s_merged_to=script_args.fura_s_merged_to,
            train_position=script_args.fura_train_position,
            factorize_by_head=script_args.fura_factorize_by_head,
            model_config=model.config,
        )
        stats = configure_blocktt_trainability(
            model,
            train_bias=script_args.fura_train_bias,
            train_position=script_args.fura_train_position,
            train_singular_values=(script_args.fura_s_merged_to == "keep_trainable"),
        )
        if stats["num_btt_layers"] == 0:
            raise ValueError(
                "FURA: no Linear layers were converted to BTT. "
                "Check --fura_trainable_type / target architecture."
            )
        if script_args.local_rank in (-1, 0):
            print(
                f"[fura] converted {stats['num_btt_layers']} BTT layers; "
                f"trainable params = {stats['trainable_param_count']:,} / "
                f"{stats['total_param_count']:,} "
                f"({100 * stats['trainable_param_count'] / stats['total_param_count']:.4f}%)"
            )
    elif not script_args.full_finetune:
        if script_args.bits < 16:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

        if checkpoint_dir is not None:
            logger.info(f"Loading adapters from {checkpoint_dir}.")
            # os.path.join(checkpoint_dir, 'adapter_model')
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
        elif script_args.adapter_name_or_path is not None:
            logger.info(f"Initilize LoRA/PiSSA/CLOVER adapters from {script_args.model_name_or_path}/{script_args.adapter_name_or_path}.")
            model = PeftModel.from_pretrained(model, script_args.model_name_or_path, subfolder = script_args.adapter_name_or_path, is_trainable=True)
        else:
            logger.info(f'Init LoRA/PiSSA modules...')
            peft_config = LoraConfig(
                use_dora=script_args.use_dora,
                runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=script_args.use_dora),
                task_type=TaskType.CAUSAL_LM,
                target_modules=script_args.target_modules.split(','),
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                init_lora_weights=script_args.init_weights,
            )
            model = get_peft_model(model, peft_config)

    for name, module in model.named_modules():
        if script_args.fura:
            # Skip BTT cores: forcing them to fp32 here would break the bf16
            # deepspeed flow (Trainer's autocast assumes BTT params live in the
            # configured compute dtype). Only cast LayerNorm-style modules.
            if isinstance(module, BTTLayer):
                continue
            if 'norm' in name:
                module = module.to(torch.float32)
        else:
            if 'norm' in name or 'gate' in name:
                module = module.to(torch.float32)
    return model

def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    log_level = script_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
        
    if script_args.local_rank == 0:
        logger.info('='*100)
        logger.info(script_args)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(script_args.model_name_or_path))
    
    resume_from_checkpoint_dir = get_last_checkpoint(script_args.output_dir)
    model = build_model(script_args, resume_from_checkpoint_dir)

    all_training_dataset = []
    for task in script_args.sub_task:
        if ":" in task: # e.g. math:500, gsm8k:100
            cur_task, num_split = task.split(":")
            cur_split = f"{script_args.dataset_split}[:{num_split}]"
        else:
            cur_task, cur_split = task, script_args.dataset_split

        ds = load_dataset(script_args.data_path, data_dir=cur_task, split=cur_split)
        if script_args.local_rank == 0:
            print(f"{script_args.data_path}/{cur_task}/{cur_split}/{ds.num_rows}")
            for k,v in ds[0].items():
                print("-"*100)
                print(k,end=':\t')
                print(v)
            print("+"*100)
        all_training_dataset.append(ds)
        
    raw_train_datasets = concatenate_datasets(all_training_dataset)
    if script_args.shuffle_dataset:
        if script_args.local_rank == 0:
            print(f"Shuffle dataset with seed={script_args.seed}")
        raw_train_datasets = raw_train_datasets.shuffle(seed=script_args.seed)

    if script_args.local_rank > 0 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={"tokenizer": tokenizer, "query": script_args.dataset_field[0], "response": script_args.dataset_field[1]}
    )

        
    if script_args.local_rank == 0:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        print(model)
        logger.info("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            logger.info(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
    if (not script_args.full_finetune) and (not script_args.fura):
        trainer.add_callback(SavePeftModelCallback(tokenizer))
    trainer.train(resume_from_checkpoint = resume_from_checkpoint_dir)
    trainer.save_state()
    if script_args.fura:
        # Materialize BTT cores back into dense nn.Linear so the saved checkpoint
        # is a standard HF model loadable by vLLM for HumanEval/MBPP evaluation.
        materialize_btt_to_linear(trainer.model)
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=script_args.output_dir)
        tokenizer.save_pretrained(script_args.output_dir)
    elif (not script_args.full_finetune) and script_args.merge:
        model = model.merge_and_unload()
        model.save_pretrained(script_args.output_dir)
        tokenizer.save_pretrained(script_args.output_dir)
    if script_args.full_finetune:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=script_args.output_dir)
        

if __name__ == "__main__":
    train()
