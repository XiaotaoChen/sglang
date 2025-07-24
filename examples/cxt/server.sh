#!/bin/bash

# python3 \
#   -m sglang.launch_server \
#   --model-path /cfs/xtchen/models/Llama-3-8B-Instruct \
#   --host 0.0.0.0 --port 30000

deepseek_r1_w4fp8_dir="/mnt/xtchen/model/DeepSeek-R1-W4AFP8"


# SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m sglang.launch_server \
#     --model-path ${deepseek_r1_w4fp8_dir} \
#     --context-length 8192 \
#     --tp 8 \
#     --trust-remote-code \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --mem-fraction-static 0.8 \
#     --enable-ep-moe \
#     --cuda-graph-max-bs 256 \
#     --cuda-graph-bs 1 2 4 8 16 32 64 128 256 \
#     --max-running-requests 256 \
#     --disable-radix-cache


SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m sglang.launch_server \
    --model-path ${deepseek_r1_w4fp8_dir} \
    --context-length 25000 \
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --mem-fraction-static 0.8 \
    --enable-ep-moe \
    --cuda-graph-max-bs 32 \
    --cuda-graph-bs 1 2 4 8 16 32 \
    --max-running-requests 256 \
    --disable-radix-cache