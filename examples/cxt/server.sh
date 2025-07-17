#!/bin/bash

# python3 \
#   -m sglang.launch_server \
#   --model-path /cfs/xtchen/models/Llama-3-8B-Instruct \
#   --host 0.0.0.0 --port 30000

model_dir="/cfs/xtchen/model"
# deepseek_r1_w4a16_dir="${model_dir}/cfs2_models/deepseek-ai/DeepSeek-R1-awq"
# deepseek_r1_w4a16_dir="${model_dir}/vda_models/DeepSeek-R1-awq"
# deepseek_r1_w4a16_dir="${model_dir}/tmpfs_models/DeepSeek-R1-awq"
deepseek_r1_w4a16_dir="${model_dir}/nvme_models/DeepSeek-R1-awq"

# python3 \
#   -m sglang.launch_server \
#   --model-path ${deepseek_r1_w4a16_dir} \
#   --served-model-name deepseek-r1-w4a16 \
#   --tp 8 --trust-remote-code --quantization moe_wna16 \
#   --host 0.0.0.0 --port 30000

python3 \
  -m sglang.launch_server \
  --model-path ${deepseek_r1_w4a16_dir} \
  --served-model-name deepseek-r1-w4a16 \
  --tp 8 --trust-remote-code --quantization moe_wna16 \
  --page-size 1 \
  --mem-fraction-static 0.7 --disable-overlap-schedule \
  --host 0.0.0.0 --port 30000