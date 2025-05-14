# 基础镜像：
```
# sglang合适的分支、deepep pr分支、tccl、mooncake、vllm
docker run \
    -itd \
    --gpus all \
    --privileged --cap-add=IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /mnt:/mnt \
    -v /cfs:/cfs \
    --net=host \
    --ipc=host \
    --name=sgl_tccl aicompute.tencentcloudcr.com/aibench/sglang:latest_deepep_tccl
```

# 启动脚本
需要提前准备attachment_ep_statistics.zip
```
cd /mnt/ && wget https://xmdong-1251001002.cos.ap-beijing.myqcloud.com/attachment_ep_statistics.zip && unzip attachment_ep_statistics.zip
```

1. 四个脚本参数1 为Prefill节点数默认为4、参数2为decode节点数默认为8、参数3为ep_statistics文件路径、参数4为log地址「log较多，建议写到cfs」

2. iplist.txt 为ip列表, 四个脚本均优先读txt文件

3. lb默认会在第一个ip机器上启动，如果需要修改，可以在脚本中修改

```
# 启动prefill
bash prefill.sh "" "" "/mnt/attachment_ep_statistics/prefill_in1000out1000.json" "/cfs/tmp/log"

# 启动decode
bash decode.sh "" "" "/mnt/attachment_ep_statistics/decode_in1000out1000.json" "/cfs/tmp/log"

# 启动lb，建议等待decode与prefill节点load成功之后启动
bash lb.sh "" "" "" "/cfs/tmp/log"

# 启动benchmark
bash benchmark.sh "" "" "" "/cfs/tmp/log"
```

# post测试

```python
import requests
from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process
url = f"http://172.21.16.8:8000/generate"
data = {"text": "What is the capital of France?"}

response = requests.post(url, json=data)
print_highlight(response.json())

```
