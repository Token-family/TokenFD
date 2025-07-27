export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

BATCH_SIZE=${BATCH_SIZE:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


torchrun --nproc_per_node 8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=${MASTER_PORT} \
  internvl/train/train_tokenfd.py \
  --train_stage 1 \
  --vision_path "/path/to/InternViT" \
  --llm_path "/path/to/InternLM" \
  --conv_style "internlm2-chat" \
  --output_dir /path/to/save \
  --meta_path /path/to/data \
  --overwrite_output_dir True \
  --rewrite_temp_json False \
  --use_background_learning False \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --unfreeze_vit_layers 23 \
  --use_custom_trainer True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 2 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 2000 \
  --save_total_limit 2 \
  --learning_rate 5e-4 \
  --weight_decay 0.05 \
  --warmup_steps 10000 \
  --lr_scheduler_type "cosine" \
  --logging_steps 100 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "ds_configs/zero_stage1_config_custom_opt.json" \
  --report_to "tensorboard" \