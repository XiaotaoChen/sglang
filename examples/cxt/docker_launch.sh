#!/bin/bash

docker_id="aicompute.tencentcloudcr.com/aibench/sgl_dev:official_dev_sgl_kernel-0.2.5-ptx-12.8-0723"

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