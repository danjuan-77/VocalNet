#!/bin/bash
export WANDB_MODE="offline"
export TOKENIZERS_PARALLELISM=false


CHECKPOINT_NAME="vocalnet-qwen25-7b-s2t"
CHECKPOINT_DIR="./checkpoints/${CHECKPOINT_NAME}"
BASE_MODEL=""
DATA_PATH=""
SPEECH_FOLDER=""
SPEECH_ENCODER="./models/speech_encoder/whisper-large-v3"

mkdir -p $CHECKPOINT_DIR

# 获取当前日期和时间
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${CHECKPOINT_DIR}/${CHECKPOINT_NAME}-${timestamp}.log"
echo "Starting training at $(date)"
echo "Logging to $log_file"


deepspeed ./omni_speech/train/train_mem.py \
    --lora_enable True \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $BASE_MODEL \
    --version qwen_2_5 \
    --input_type mel \
    --mel_size 128 \
    --speech_encoder_type whisper \
    --speech_projector_type linear \
    --freeze_backbone True \
    --mm_tunable_parts="speech_projector,backbone" \
    --is_multimodal \
    --data_path $DATA_PATH \
    --speech_folder $SPEECH_FOLDER \
    --speech_encoder $SPEECH_ENCODER \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$CHECKPOINT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb &> "$log_file"

CUDA_VISIBLE_DEVICES=0 python ./omni_speech/merge_lora_model.py --model-path $CHECKPOINT_DIR/lora_checkpoints --model-base $BASE_MODEL --save-model-path $CHECKPOINT_DIR


