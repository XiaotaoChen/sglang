# Docker 镜像构建与推送指南

## 简介
默认情况下，脚本会处理 `latest` 和 `dev` 两个版本。用户可以通过传入额外的版本参数来扩展需要处理的版本列表。

## 文件结构
- `sglang.dockerfile`: 用于构建镜像的 Dockerfile。
- `build_and_push.sh`: 用于拉取、构建和推送镜像的脚本。

## 使用方法

### 1. 准备工作
确保你已经安装了 Docker，并且登录到目标容器仓库。
```bash
docker login aicompute.tencentcloudcr.com
```

### 2. 运行脚本
将 `sglang.dockerfile` 和 `build_and_push.sh` 放在同一目录下，然后运行脚本。

#### 默认版本
运行以下命令，脚本将处理默认的 `latest` 和 `dev` 版本：
```bash
./build_and_push.sh
```

#### 添加额外版本
如果你需要处理额外的版本，可以通过命令行参数传入版本号。例如：
```bash
./build_and_push.sh v0.4.4.post1-cu124
```
这将处理 `latest`、`dev`、`v0.4.4.post1-cu124` 三个版本。

### 3. 脚本功能
- **拉取官方镜像**：从源仓库拉取 `latest` 镜像。
- **打标签**：为拉取的镜像打上目标仓库的标签。
- **推送镜像**：将打标签后的镜像推送到目标仓库。
- **构建新镜像**：根据 `sglang.dockerfile` 构建 `_tencent` 后缀的镜像。
- **推送新镜像**：将构建的新镜像推送到目标仓库。


### 4. 注意事项
- 确保 `sglang.dockerfile` 文件与脚本在同一目录下。
- 如果目标仓库需要认证，确保你已经通过 `docker login` 登录到目标仓库。
- 如果某些版本可能不存在于源仓库中，需要确保这些版本的镜像在源仓库中存在，否则 `docker tag` 和 `docker push` 会失败。

- 从官方仓库中查询tag，非GitHubtag
https://hub.docker.com/r/lmsysorg/sglang/tags
