# Finetune VL-BART (BUTD features) on image-text multitask
# (VQAv2 / GQA / NLVR2 / COCO-Cap) with DoRA.
#
# Mirrors scripts/image/dora.sh budget but uses --feature butd (j-min layout)
# instead of --feature RN101 (broken gdrive). q_proj+v_proj target via
# --lora_settings + --use_dora is unchanged from upstream.
#
# Usage:
#   GPU=4 LR=1e-3 EPOCHS=1 RUN_NAME=dora_lr1e-3_short \
#     bash scripts/image/dora_butd.sh 1 [extra args...]

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${GPU}

task=multitask
model="bart"
folder_prefix="VLBart"
backbone="facebook/bart-base"
batch_size=${BATCH_SIZE:-300}

feature=${FEATURE:-butd}

lr=${LR:-1e-3}
epochs=${EPOCHS:-1}
lora_dim=${LORA_DIM:-128}
save_steps=${SAVE_STEPS:-500}
project_name=${PROJECT:-fura-vl-bart-image}
run_name=${RUN_NAME:-dora_${feature}_qv_lr${lr}_e${epochs}_d${lora_dim}}
output=snap/${folder_prefix}_${task}/${run_name}

resume_args=""
if [ -n "${RESUME:-}" ]; then
    resume_args="--resume ${RESUME}"
fi

TOKENIZERS_PARALLELISM=True PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port=${MASTER_PORT:-26464} \
    src/${task}.py \
    --distributed --multiGPU \
    --optim adamw \
    --warmup_ratio 0.1 \
    --clip_grad_norm 5 \
    --lr ${lr} \
    --epochs ${epochs} \
    --num_workers 4 \
    --backbone ${backbone} \
    --output ${output} ${@:2} \
    --num_beams 5 \
    --use_tasks_prompts \
    --batch_size ${batch_size} \
    --valid_batch_size ${batch_size} \
    --use_dora \
    --unfreeze_bias \
    --unfreeze_layer_norms \
    --lora_settings \
    --lora_dim ${lora_dim} \
    --tasks "vqa,gqa,nlvr,caption" \
    --feature ${feature} --n_boxes 36 --downsample \
    --image_size "(224,224)" \
    --save_steps ${save_steps} \
    ${resume_args} \
    --project_name ${project_name} \
    --run_name ${run_name}
