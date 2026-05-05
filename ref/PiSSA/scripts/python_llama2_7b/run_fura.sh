BASE_MODEL="meta-llama/Llama-2-7b-hf"
OUTPUT_PATH="output/python-FURA-Llama-2-7b"
DATA_PATH="pissa-dataset"
# export HF_ENDPOINT=https://hf-mirror.com

# Single H100 recipe.
# Effective batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
# (matches the PiSSA-paper Code-Feedback recipe; reduce per_device_train_batch_size and bump
# gradient_accumulation_steps if you OOM on 80GB cards.)
deepspeed --master_port=16971 --include=localhost:0 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --full_finetune False \
    --fura True \
    --fura_trainable_type all \
    --fura_decomp_mode output_one_block \
    --fura_blocktt_rank full \
    --fura_train_position small \
    --fura_s_merged_to keep_trainable \
    --bf16 \
    --data_path $DATA_PATH \
    --sub_task python \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \

python utils/gen_vllm.py --model $OUTPUT_PATH --sub_task python --output_file $OUTPUT_PATH/python_response.jsonl
python utils/code_process.py --path $OUTPUT_PATH/python_response.jsonl
evalplus.evaluate --dataset humaneval --samples $OUTPUT_PATH/humaneval.jsonl
evalplus.evaluate --dataset mbpp --samples $OUTPUT_PATH/mbpp.jsonl
