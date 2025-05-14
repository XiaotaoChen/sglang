FROM lmsysorg/sglang:latest
# configure timezone
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt update && apt install -y tzdata && \
        dpkg-reconfigure --frontend noninteractive tzdata

RUN wget -O /etc/apt/sources.list https://mirrors.cloud.tencent.com/repo/ubuntu22_sources.list

RUN apt update && apt install -y \
        openssh-client openssh-server \
        cmake git net-tools pdsh tmux vim iputils-ping libnuma-dev libcap2 lrzsz curl

# support Chinese language
RUN apt install -y language-pack-zh-hans
ENV LANG="zh_CN.UTF-8"
ENV LANGUAGE="zh_CN:zh:en_US:en"

RUN mkdir -p /workspace

# install OFED UMD
RUN apt update && wget -q --show-progress https://taco-1251783334.cos.ap-shanghai.myqcloud.com/ofed/MLNX_OFED_LINUX-5.8-2.0.3.0-ubuntu22.04-x86_64.tgz && \
    tar xf MLNX_OFED_LINUX-5.8-2.0.3.0-ubuntu22.04-x86_64.tgz && \
    cd MLNX_OFED_LINUX-5.8-2.0.3.0-ubuntu22.04-x86_64 && \
    ./mlnxofedinstall --user-space-only --without-fw-update --without-ucx-cuda --force && cd ../ && rm MLNX_OFED_LINUX* -rf

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
RUN pip install pandas openpyxl
RUN if [ -f /usr/bin/python ]; then rm /usr/bin/python; fi && \
    ln -s /usr/bin/python3 /usr/bin/python


WORKDIR "/workspace/"

# CMake
RUN apt-get update \
&& apt-get install -y --no-install-recommends \
build-essential \
wget \
libssl-dev \
&& wget https://github.com/Kitware/CMake/releases/download/v3.27.4/cmake-3.27.4-linux-x86_64.sh \
&& chmod +x cmake-3.27.4-linux-x86_64.sh \
&& ./cmake-3.27.4-linux-x86_64.sh --skip-license --prefix=/usr/local \
&& rm cmake-3.27.4-linux-x86_64.sh

# GDRCopy
WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/gdrcopy.git
WORKDIR /tmp/gdrcopy
RUN git checkout v2.4.4

RUN apt update
RUN apt install -y nvidia-dkms-535
RUN apt install -y build-essential devscripts debhelper fakeroot pkg-config dkms
RUN apt install -y check libsubunit0 libsubunit-dev

WORKDIR /tmp/gdrcopy/packages
RUN CUDA=/usr/local/cuda ./build-deb-packages.sh
RUN dpkg -i gdrdrv-dkms_*.deb
RUN dpkg -i libgdrapi_*.deb
RUN dpkg -i gdrcopy-tests_*.deb
RUN dpkg -i gdrcopy_*.deb

ENV GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/

# IBGDA dependency
RUN apt-get install -y libfabric-dev

# DeepEP branch
WORKDIR /sgl-workspace
RUN git clone https://github.com/deepseek-ai/DeepEP.git

# NVSHMEM
WORKDIR /sgl-workspace
RUN wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz
RUN tar -xf nvshmem_src_3.2.5-1.txz \
    && mv nvshmem_src nvshmem
RUN rm /sgl-workspace/nvshmem_src_3.2.5-1.txz

WORKDIR /sgl-workspace/nvshmem
RUN git apply /sgl-workspace/DeepEP/third-party/nvshmem.patch

WORKDIR /sgl-workspace/nvshmem
ENV CUDA_HOME=/usr/local/cuda
RUN NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=1 \
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/sgl-workspace/nvshmem/install -DCMAKE_CUDA_ARCHITECTURES=90 \
    && cd build \
    && make install -j

WORKDIR /sgl-workspace/DeepEP
RUN sed -i 's/constexpr int kNumWarpsPerGroup = 10;/constexpr int kNumWarpsPerGroup = 8;/g; s/constexpr int kNumWarpGroups = 3;/constexpr int kNumWarpGroups = 4;/g' /sgl-workspace/DeepEP/csrc/kernels/internode_ll.cu
ENV NVSHMEM_DIR=/sgl-workspace/nvshmem/install
RUN NVSHMEM_DIR=/sgl-workspace/nvshmem/install pip install .

# MOONCAKE
RUN apt install -y golang-go
RUN cd /sgl-workspace/ && git clone https://github.com/kvcache-ai/Mooncake.git
RUN cd /sgl-workspace/Mooncake && bash dependencies.sh
RUN export PATH="/usr/local/go/bin:$PATH" && cd /sgl-workspace/Mooncake && mkdir build && cd build && cmake .. && \
    make -j && sudo make install

# !! New branch Sglang
RUN mv /sgl-workspace/sglang /sgl-workspace/sglang_back
RUN cd /sgl-workspace && git clone --branch deepseek_ep https://gh-proxy.com/github.com/sgl-project/sglang.git
RUN cd /sgl-workspace/sglang/python && pip install -e .


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

RUN pip install vllm
# Set workspace
WORKDIR /workspace

ENTRYPOINT ["/usr/sbin/sshd", "-D"]
