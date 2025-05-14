```
docker build  -t aicompute.tencentcloudcr.com/aibench/sglang:latest_deepep -f sglang_deepep.dockerfile .

docker push aicompute.tencentcloudcr.com/aibench/sglang:latest_deepep

docker build  -t aicompute.tencentcloudcr.com/aibench/sglang:latest_deepep_tccl -f sglang_deepep_tccl.dockerfile .
docker push aicompute.tencentcloudcr.com/aibench/sglang:latest_deepep_tccl

```
