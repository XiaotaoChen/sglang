#!/bin/bash

# docker_id="aicompute.tencentcloudcr.com/aibench/sglang:latest_tencent"
# docker_id="aicompute.tencentcloudcr.com/aibench/sgl_dev:0.4.6.post5_sgl_kernel_cxt"
docker_id="aicompute.tencentcloudcr.com/aibench/sgl_dev:0.4.8.post1_sgl_kernel-0.2.1-w4ap8"
# docker_id="aicompute.tencentcloudcr.com/aibench/sgl_dev:0.4.6.post5_w4afp8"
# docker_id="aicompute.tencentcloudcr.com/aibench/sglang:v0.4.9.post2-cu126_tencent"
# docker_id="lmsysorg/sglang:dev"

docker run \
    -it \
    --rm \
    --gpus all \
    --privileged --cap-add=IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /data0:/data \
    -v /cfs:/cfs \
    -v /cfs2:/cfs2 \
    -v /mnt:/mnt \
    --net=host \
    --ipc=host \
    --name=sglang \
    --entrypoint /bin/bash \
    ${docker_id}

# docker run \
#     -it \
#     --rm \
#     --gpus all \
#     --privileged --cap-add=IPC_LOCK \
#     --ulimit memlock=-1 --ulimit stack=67108864 \
#     -v /data0:/data \
#     -v /cfs:/cfs \
#     -v /cfs2:/cfs2 \
#     -v /mnt:/mnt \
#     --net=host \
#     --ipc=host \
#     --name=sglang ${docker_id} \
#     /bin/bash