#!/bin/bash
# master 节点，需要将--dist-init-addr对应的ip修改为自己服务器的master ip
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_DEBUG=INFO
export SGL_ENABLE_JIT_DEEPGEMM=1

# nohup \
python -m sglang.launch_server \
    --model-path /mnt/xtchen/model/Kimi-K2-Instruct \
    --tp 16 \
    --tool-call-parser kimi_k2  \
    --dist-init-addr 10.9.1.12:5000 \
    --nnodes 2 \
    --node-rank 0 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8001 \
    --disable-radix-cache
# 2>&1 > output/kimi_infer_server0_$(date +'%Y%m%d_%H%M%S').log & 
