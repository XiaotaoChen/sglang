#!/bin/bash

# curl -X POST \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer ''" \
#   -d '{
#     "model": "/mnt/xtchen/model/Kimi-K2-Instruct",
#     "messages": [
#       {"role": "user", "content": "你好！请帮我制定一个去大理游玩的计划！"}
#     ],
#     "temperature": 0.7,
#     "max_tokens": 600,
#     "stream": false
# }' \
#   "http://127.0.0.1:8001/v1/chat/completions"

curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ''" \
  -d '{
    "model": "/mnt/xtchen/model/Kimi-K2-Instruct",
    "messages": [
      {"role": "user", "content": "法国的首都在哪里？"}
    ],
    "temperature": 0.7,
    "stream": false
}' \
  "http://127.0.0.1:8001/v1/chat/completions"
