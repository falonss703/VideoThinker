export PRIVATE_DATA_ROOT=$PRIVATE_DATA_ROOT
export WANDB_PROJECT=$WANDB_PROJECT
export MODEL_NAME=Qwen2.5-VL-3B-Instruct_clevrer_counterfactual_bias_model
export WANDB_NAME=$MODEL_NAME
export DEBUG_MODE=true
export LOG_PATH=$PRIVATE_DATA_ROOT/$WANDB_PROJECT/$WANDB_NAME/debug.log
export WANDB_MODE=offline
export SAMPLE_MODE=true
export LOG_PATH=$PRIVATE_DATA_ROOT/Training/$WANDB_NAME/debug.log

mkdir -p $PRIVATE_DATA_ROOT/Training/$MODEL_NAME

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12531" \
    src/open_r1/grpo.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $PRIVATE_DATA_ROOT/Training/$MODEL_NAME \
    --model_name_or_path $PRIVATE_DATA_ROOT/Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name xxx \
    --jsonl_path data/CLEVRER/clevrer_counterfactual_train_observational_bias_training.json \
    --max_prompt_length 4096 \
    --max_completion_length 768 \
    --reward_funcs accuracy format \
    --learning_rate 1e-6 \
    --beta 0.000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --question_type mixed \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --freeze_vision_modules false \
    --losstype grpo \
    --num_train_epochs 0.5 \
    --run_name $WANDB_NAME \
    --save_steps 500 \
    --max_grad_norm 20 \
    --save_only_model true \
    --num_generations 8 > $LOG_PATH 2>&1

model_paths=(
    "MODEL_PATH"
)

current_time=$(date +"%Y%m%d_%H%M")
file_names=(
    $current_time
)

export VIDEO_MAX_PIXELS=200704
export FPS_MAX_FRAMES=32
export SAMPLE_MODE=true
export DECORD_EOF_RETRY_MAX=40960 

for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    file_name="${file_names[0]}"
    echo "START $((i+1))/${#model_paths[@]}: $model"
    env CUDA_VISIBLE_DEVICES=0,1 python src/eval/eval_bench.py --model_path "$model" --file_name "$file_name" > ${model}/eval_${file_name}.log 2>&1
    echo "END $((i+1))/${#model_paths[@]}: $model"
done