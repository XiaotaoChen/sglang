#!/bin/bash
# ！！注意！！
# 提前登录ccr！

# 定义默认版本
DEFAULT_VERSIONS=("latest" "dev")

# 如果传入参数，则将这些参数添加到默认版本中
VERSIONS=("${DEFAULT_VERSIONS[@]}")
if [ $# -ne 0 ]; then
  VERSIONS+=("$@")  # 将传入的参数添加到默认版本数组中
fi

# 定义变量
SOURCE_REPO="lmsysorg/sglang"
TARGET_REPO="aicompute.tencentcloudcr.com/aibench/sglang"


# 1. 拉取官方镜像
echo "拉取官方镜像..."
for version in "${VERSIONS[@]}"; do
  echo "拉取 ${SOURCE_REPO}:${version} 镜像..."
  docker pull ${SOURCE_REPO}:${version}
done

# 2. 给官方镜像打上对应标签
echo "给官方镜像打标签为最新和指定版本..."
for version in "${VERSIONS[@]}"; do
  echo "给 ${SOURCE_REPO}:${version} 镜像打标签 ${TARGET_REPO}:${version}"
  docker tag ${SOURCE_REPO}:${version} ${TARGET_REPO}:${version}
done

# 3. 推送镜像到目标仓库
echo "推送官方镜像到目标仓库..."
for version in "${VERSIONS[@]}"; do
  echo "推送 ${TARGET_REPO}:${version} 镜像到目标仓库..."
  docker push ${TARGET_REPO}:${version}
done


# 4. 根据 Dockerfile 构建 _tencent 后缀的标签新镜像
echo "根据 Dockerfile 构建新镜像..."
for version in "${VERSIONS[@]}"; do
  echo "构建 ${TARGET_REPO}:${version}_tencent 镜像..."
  docker build  --build-arg VERSION=${version} -t ${TARGET_REPO}:${version}_tencent -f sglang.dockerfile .
done


# 5. 推送本地构建的镜像到目标仓库
echo "推送本地构建的镜像到目标仓库..."
for version in "${VERSIONS[@]}"; do
  echo "推送 ${TARGET_REPO}:${version}_tencent 镜像到目标仓库..."
  docker push ${TARGET_REPO}:${version}_tencent
done

echo "脚本执行完成！"
