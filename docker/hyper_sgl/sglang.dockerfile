# 定义版本参数
ARG VERSION=latest

# 根据版本调整构建逻辑
FROM aicompute.tencentcloudcr.com/aibench/sglang:${VERSION}

# configure timezone
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt update && apt install -y tzdata && \
        dpkg-reconfigure --frontend noninteractive tzdata

# ubuntu22.04 apt source
#RUN wget -O /etc/apt/sources.list https://mirrors.cloud.tencent.com/repo/ubuntu22_sources.list

# ubuntu24.04 apt source
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
    cat <<EOF > /etc/apt/sources.list
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-backports main restricted universe multiverse

# 以下安全更新软件源包含了官方源与镜像站配置，如有需要可自行修改注释切换
deb http://security.ubuntu.com/ubuntu/ noble-security main restricted universe multiverse
# deb-src http://security.ubuntu.com/ubuntu/ noble-security main restricted universe multiverse
EOF

RUN apt update && apt install -y \
        openssh-client openssh-server \
        cmake git net-tools pdsh tmux vim iputils-ping libnuma-dev libcap2 lrzsz curl

# support Chinese language
RUN apt install -y language-pack-zh-hans
ENV LANG="zh_CN.UTF-8"
ENV LANGUAGE="zh_CN:zh:en_US:en"

# allow OpenSSH to talk to containers without asking for confirmation
RUN mkdir -p /var/run/sshd
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
RUN sed -i 's/#Port 22/Port 3333/' /etc/ssh/sshd_config

# Install unzip and download/extract the .zip file
# RUN mkdir /root/.ssh
RUN apt update && apt install -y unzip && \
    wget https://simoon-test-1251783334.cos.ap-shanghai.myqcloud.com/dockerfile_ssh.zip && \
    unzip -o dockerfile_ssh.zip -d /tmp/ && \
    rm dockerfile_ssh.zip

RUN rm -rf /tmp/ssh/__MACOSX
RUN mkdir -p /root/.ssh
RUN cp /tmp/ssh/* /root/.ssh
RUN rm -rf /tmp/ssh

# Add SSH configuration

# Set correct permissions for SSH files
RUN chmod 700 /root/.ssh && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 644 /root/.ssh/id_rsa.pub

RUN mkdir -p /workspace

# install OFED UMD
# Ubuntu22.04
# RUN apt update && wget -q --show-progress https://taco-1251783334.cos.ap-shanghai.myqcloud.com/ofed/MLNX_OFED_LINUX-5.8-2.0.3.0-ubuntu22.04-x86_64.tgz && \
#     tar xf MLNX_OFED_LINUX-5.8-2.0.3.0-ubuntu22.04-x86_64.tgz && \
#     cd MLNX_OFED_LINUX-5.8-2.0.3.0-ubuntu22.04-x86_64 && \
#     ./mlnxofedinstall --user-space-only --without-fw-update --without-ucx-cuda --force && cd ../ && rm MLNX_OFED_LINUX* -rf

# Ubuntu24.04
RUN wget -q --show-progress https://taco-1251783334.cos.ap-shanghai.myqcloud.com/ofed/MLNX_OFED_LINUX-5.8-2.0.3.0-ubuntu22.04-x86_64.tgz && \
    tar xf MLNX_OFED_LINUX-5.8-2.0.3.0-ubuntu22.04-x86_64.tgz && \
    cd MLNX_OFED_LINUX-5.8-2.0.3.0-ubuntu22.04-x86_64 && \
    sed -i 's/dpatch//g' mlnxofedinstall && sed -i 's/ubuntu22/ubuntu2[24]/g' mlnxofedinstall && \
    ./mlnxofedinstall --user-space-only --without-fw-update --without-ucx-cuda --force --skip-distro-check && cd ../ && rm MLNX_OFED_LINUX* -rf

# export openmpi
ENV PATH=/usr/mpi/gcc/openmpi-4.1.5a1/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/mpi/gcc/openmpi-4.1.5a1/lib:$LD_LIBRARY_PATH



# install nccl-rdma-plugins (tencent specific)
RUN sed -i '/nccl_rdma_sharp_plugin/d' /etc/ld.so.conf.d/hpcx.conf
RUN wget -q --show-progress "https://taco-1251783334.cos.ap-shanghai.myqcloud.com/nccl/plugin/ubuntu22.04/nccl-rdma-sharp-plugins_1.4_amd64.deb" && \
    dpkg -i nccl-rdma-sharp-plugins_1.4_amd64.deb && rm -f nccl-rdma-sharp-plugins_1.4_amd64.deb

# avoid pdsh permission issue
RUN chown root:root /usr/lib /usr/bin /usr/include


# install nccl-tests
RUN mkdir -p /workspace && \
    cd /workspace && \
    git clone https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && \
    make MPI=1 MPI_HOME=/usr/mpi/gcc/openmpi-4.1.5a1 && \
    cp build/*perf /usr/local/bin/ && \
    cd /workspace && \
    rm -rf /workspace/nccl-tests


# nccl settings
ENV NCCL_SOCKET_IFNAME=eth0
ENV NCCL_IB_GID_INDEX=3
ENV NCCL_IB_DISABLE=0
ENV NCCL_NET_GDR_LEVEL=2
ENV NCCL_IB_QPS_PER_CONNECTION=4
ENV NCCL_IB_TC=160
ENV NCCL_IB_TIMEOUT=22
ENV GLOO_SOCKET_IFNAME=eth0


# use tencent pip source
RUN pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple
RUN pip config set global.trusted-host mirrors.cloud.tencent.com
RUN pip install pandas openpyxl --break-system-packages
RUN pip install concurrent-log-handler --break-system-packages
RUN if [ -f /usr/bin/python ]; then rm /usr/bin/python; fi && \
    ln -s /usr/bin/python3 /usr/bin/python


WORKDIR "/workspace/"

ENTRYPOINT ["/usr/sbin/sshd", "-D"]
