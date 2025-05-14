#!/bin/bash
  
########################################################
# usage:
#
# bash prefill.sh "" "" "/mnt/attachment_ep_statistics/prefill_in4096.json" "./" 2>&1 | tee prefill.log &
#
########################################################

########################################################
# Fill <ip_list> with your machine ip <ifconfig eth0>
########################################################
# load iplist.txt
if [ -f "iplist.txt" ]; then
    readarray -t ip_list < iplist.txt
    if [ ${#ip_list[@]} -eq 0 ]; then
        echo "Error: iplist.txt file is empty"
        exit 1
    fi
else
    declare -a ip_list=(
        localhost
    )
fi

# Parameters to be filled
model_path="/mnt/DeepSeek-R1" # Your model path
device_name="mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7" # ib device
num_prefill=${1:-4} # Number of prefill nodes
num_decode=${2:-8} # Number of decode nodes
expert_location=${3:-"/mnt/attachment_ep_statistics/prefill_in4096.json"}
base_dir=${4:-"/cfs"}
timestamp="decode"_$(date +'%Y%m%d_%H%M%S')
log_dir="$base_dir/$timestamp"

remote_app_command="MC_TE_METRIC=true SGLANG_HACK_DEEPEP_NEW_MODE=0 SGL_ENABLE_JIT_DEEPGEMM=1 \
    NCCL_DEBUG=INFO NCCL_IB_TC=160 NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
    NCCL_NET_GDR_LEVEL=2 NCCL_IB_GID_INDEX=3 NCCL_WORK_FIFO_DEPTH=4194304 \
    NCCL_IB_QPS_PER_CONNECTION=4 NCCL_IB_TIMEOUT=22 NCCL_IB_DISABLE=0 \
    nohup python3 -m sglang.launch_server \
        --model-path ${model_path} \
        --disaggregation-mode prefill \
        --disaggregation-ib-device $device_name \
        --trust-remote-code \
        --host 0.0.0.0 \
        --nnodes ${num_prefill} \
        --tp-size $((${num_prefill}*8)) \
        --dp-size $((${num_prefill}*8)) \
        --enable-dp-attention \
        --enable-deepep-moe \
        --deepep-mode normal \
        --mem-fraction-static 0.85 \
        --chunked-prefill-size $((${num_prefill}*131072)) \
        --max-running-requests $((${num_prefill}*2048)) \
        --max-total-tokens 131072 \
        --context-length 8192 \
        --init-expert-location \"${expert_location}\" \
        --ep-num-redundant-experts 32 \
        --enable-two-batch-overlap \
        --moe-dense-tp-size 1 \
        --disable-radix-cache \
        --ep-dispatch-algorithm random"

cmd="mkdir -p $log_dir && ${remote_app_command}"


# Check if we have enough nodes for the specified prefill and decode counts
total_nodes=${#ip_list[@]}
if [[ $num_prefill -gt $total_nodes || $num_decode -gt $total_nodes ]]; then
    echo "Error: The total number of prefill and decode nodes cannot exceed the total available nodes ($total_nodes)."
    exit 1
fi

# Execute on prefill nodes
prefill_ip_list=("${ip_list[@]:0:$num_prefill}")
decode_ip_list=("${ip_list[@]:$num_prefill:$num_decode}")

# Master IP for prefill nodes: first prefill node IP
master_ip_prefill=${prefill_ip_list[0]}
# Master IP for decode nodes: first decode node IP
master_ip_decode=${decode_ip_list[0]}

# Prefill nodes: run the specified command with custom parameters inside Docker container
for i in ${!prefill_ip_list[@]}; do
    ip=${prefill_ip_list[$i]}
    node_rank=$i  # Prefill node rank is based on the index in prefill_ip_list
    echo; echo
    echo "Running prefill [ $cmd ] on server [ $ip ] with device [ $device_name ] and node_rank [ $node_rank ] inside container"
    ssh root@$ip "$cmd --dist-init-addr ${master_ip_prefill}:5757 --node-rank ${node_rank} 2>&1 | tee $log_dir/prefill_$ip.log &" &
done
