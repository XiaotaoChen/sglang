#!/bin/bash

# curl -X POST http://localhost:30000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "default",
#     "messages": [
#       {"role": "user", "content": "What is the capital of France?"}
#     ],
#     "temperature": 0.7,
#     "max_tokens": 512
#   }'

# model_name="/cfs/xtchen/models/Llama-3-8B-Instruct"
# model_name="deepseek-v3"

curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-r1-w4a16", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'