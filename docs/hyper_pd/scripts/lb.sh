#!/bin/bash
  
########################################################
# usage:
#
# bash lb.sh 2>&1 | tee lb.log &
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
device_names=("mlx5_bond_0" "mlx5_bond_1" "mlx5_bond_2" "mlx5_bond_3" "mlx5_bond_4" "mlx5_bond_5" "mlx5_bond_6" "mlx5_bond_7")
num_prefill=${1:-4} # Number of prefill nodes
num_decode=${2:-8} # Number of decode nodes
expert_location=${3:-"/mnt/attachment_ep_statistics/prefill_in4096.json"}
base_dir=${4:-"/cfs"}
timestamp="lb"_$(date +'%Y%m%d_%H%M%S')
log_dir="$base_dir/$timestamp"

cmd="python3 -m sglang.launch_server --model-path ${model_path}"

bg=0  # Change to 1 for background execution, 0 for serial

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


# Load balancer: connect to the first prefill and decode nodes inside Docker container
echo; echo "Running load balancer..."
cmd="nohup python3 -m sglang.srt.disaggregation.mini_lb --prefill "http://${prefill_ip_list[0]}:30000" --decode "http://${decode_ip_list[0]}:30000" --host 0.0.0.0 2>&1 | tee -a $log_path/lb$(date +'%Y%m%d_%H%M%S').log &"

ssh root@$master_ip_prefill "mkdir -p $log_dir && $cmd 2>&1 | tee $log_dir/lb.log &" &
