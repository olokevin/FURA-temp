# Finetune VL-BART (BUTD features) on image-text multitask
# (VQAv2 / GQA / NLVR2 / COCO-Cap) with fura (BlockTT).
#
# Usage:
#   GPU=4 LR=1e-3 EPOCHS=1 RUN_NAME=fura_lr1e-3_short \
#     bash scripts/image/fura.sh 1 [extra args...]
#
# fura defaults match project-wide CLAUDE.md (rank=full,
# decomp_mode=output_one_block, train_position=small,
# s_merged_to=keep_trainable). Targets q_proj+v_proj only,
# matching DoRA paper Table 3 setup.

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${GPU}

task=multitask
model="bart"
folder_prefix="VLBart"
backbone="facebook/bart-base"
batch_size=${BATCH_SIZE:-300}

# BUTD obj36 features (j-min/VL-T5 layout). DoRA's --feature RN101 gdrive
# is dead; we pivot to butd which the dataloaders also natively support.
feature=${FEATURE:-butd}

lr=${LR:-1e-3}
epochs=${EPOCHS:-1}
save_steps=${SAVE_STEPS:-500}
project_name=${PROJECT:-fura-vl-bart-image}
run_name=${RUN_NAME:-fura_${feature}_qv_lr${lr}_e${epochs}}
output=snap/${folder_prefix}_${task}/${run_name}

resume_args=""
if [ -n "${RESUME:-}" ]; then
    resume_args="--resume ${RESUME}"
fi

TOKENIZERS_PARALLELISM=True PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port=${MASTER_PORT:-26466} \
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
    --use_fura \
    --blocktt_rank full \
    --decomp_mode output_one_block \
    --train_position small \
    --s_merged_to keep_trainable \
    --convert_mode svd \
    --unfreeze_bias \
    --unfreeze_layer_norms \
    --tasks "vqa,gqa,nlvr,caption" \
    --feature ${feature} --n_boxes 36 --downsample \
    --image_size "(224,224)" \
    --save_steps ${save_steps} \
    ${resume_args} \
    --project_name ${project_name} \
    --run_name ${run_name}
