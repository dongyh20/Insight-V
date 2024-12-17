EXP_NAME="llama3-llava-next-reason-dpo"
VISION_TOWER='/home/models/clip-vit-large-patch14-336'

echo $MASTER_ADDR; echo $nnode; echo $nrank

# ----------- finetune mlp+llm -------------

export DPO_FORWARD=1

torchrun  --nproc_per_node 8 --nnodes=$nnode --node_rank=$nrank --master_addr=$MASTER_ADDR --master_port=23333 \
    llava/train/train_dpo.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /path/to/reason_models \
    --version llava_llama_3 \
    --data_path /path/to/dpo_data \
    --image_folder ./playground/data \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --vision_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_use_think_token True \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints  "(1x1),...,(3x3)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --output_dir ./checkpoints_new/$EXP_NAME \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 12288 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
