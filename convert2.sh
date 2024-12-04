#!/bin/bash

# 定义路径
LLM_PATH='/data/kesun/kesun/work_dirs/lllava_v15_7b_best/iter_14000_merged'
#LLM_PATH="/data/kesun/vicuna-7b-v1.5"
#path="/youtu-pangu_media_public_1511122_cq10/ziyinzhou/code/xtuner/work_dirs/llava_v15_7b_finetune_AIGC_progan_ours_10epoch3"
config_path="./xtuner/configs/llava/official/llava_v15_7b/llava_v16_finetune.py"
#config_path="./xtuner/configs/llava/official/llava_v15_7b/llava_v15_7b_finetune_AIGC_lora.py"
visual_encoder_name_or_path='/data/kesun/zzy_weights/aigcllmdetectvisual/image_visual'
#visual_encoder_name_or_path="/youtu-pangu_media_public_1511122_cq10/ziyinzhou/code/openai_clip"

max_number=0
max_path="/data/kesun/work_dirs/llava_v16_13b_finetune_AIGC_lora_best1202_1/iter_500.pth"
# 检查 max_save_path 是否存在
max_xtuner_path=$(echo "$max_path" | sed 's/\.pth$/_xtuner/')
max_merge_path=$(echo "$max_path" | sed 's/\.pth$/_merged/')
max_save_path=$(echo "$max_path" | sed 's/\.pth$/_hf/')

if [ -d "$max_save_path" ]; then
    echo "max_save_path 已存在，直接启动 vllm API 服务器"
    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model $max_save_path --tensor-parallel-size 2 --served-model-name xtuner_try --dtype=half --trust-remote-code --gpu-memory-utilization 0.9 --enforce-eager --max-model-len 4096 --port 8023 --chat_template /home/kesun/zzy/vllm/examples/template_llava.jinja
    exit 0
fi

# 使用 find 命令查找所有符合条件的文件夹
#folders=$(find "$path" -type d -name "iter_*.pth")

# 初始化最大数字和对应的路径


# 遍历所有找到的文件夹
#for folder in $folders; do
#    # 提取数字部分
#    number=$(echo "$folder" | grep -oP 'iter_\K[0-9]+')
#
#    # 比较数字大小
#    if [[ "$number" -gt "$max_number" ]]; then
#        max_number=$number
#        max_path="$folder"
#    fi
#done

# 输出最大数字对应的路径
echo "最大数字对应的路径是: $max_path"

# 将路径保存为一个变量
max_iter_path=$max_path

max_xtuner_path=$(echo "$max_path" | sed 's/\.pth$/_xtuner/')
max_merge_path=$(echo "$max_path" | sed 's/\.pth$/_merged/')
max_save_path=$(echo "$max_path" | sed 's/\.pth$/_hf/')

echo "最大数字对应的路径是: $max_xtuner_path"

LLM_ADAPTER="$max_xtuner_path/llm_adapter"
projector_weight="$max_xtuner_path/projector/pytorch_model.bin"

CUDA_VISIBLE_DEVICES=2 xtuner convert pth_to_hf $config_path $max_iter_path $max_xtuner_path

CUDA_VISIBLE_DEVICES=2 xtuner convert merge $LLM_PATH $LLM_ADAPTER $max_merge_path

CUDA_VISIBLE_DEVICES=2 python ../xtuner/xtuner/configs/llava/llama3_8b_instruct_clip_vit_large_p14_336/convert_xtuner_weights_to_hf.py --text_model_id $max_merge_path --vision_model_id $visual_encoder_name_or_path --projector_weight $projector_weight --save_path $max_save_path

#CONDA_BASE=$(conda info --base)
#source "$CONDA_BASE/etc/profile.d/conda.sh"

#conda activate myenv
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model $max_save_path --tensor-parallel-size 2 --served-model-name xtuner_try --dtype=half --trust-remote-code --gpu-memory-utilization 0.9 --enforce-eager --max-model-len 4096 --port 8023 --chat_template /home/kesun/zzy/vllm/examples/template_llava.jinja