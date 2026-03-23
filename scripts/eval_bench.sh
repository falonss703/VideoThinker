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