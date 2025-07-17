#!/bin/bash

python3 -m sglang.bench_serving \
    --backend sglang \
    --base-url http://172.17.97.5:8000 \
    --tokenizer /cfs/models/deepseek-ai/DeepSeek-R1 \
    --model DeepSeek-R1 \
    --dataset-name sharegpt \
    --dataset-path /cfs/xtchen/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --random-input-len 1 \
    --random-output 1024 \
    --num-prompts 8 \
    --request-rate 8 \
    --profile \
    --output-file online.jsonl 2>&1