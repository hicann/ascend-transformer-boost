ARG UBUNTU_VERSION=18.04

FROM ubuntu:${UBUNTU_VERSION}
LABEL maintainer = "jitao jitao12@huawei.com"
LABEL version = "1.0.0"
LABEL description = "Image for chatglm6b 910b based on Ubuntu18.04"

ENV GIT_SSL_NO_VERIFY=1
ENV ACLTRANSFORMER_TESTDATA=/data/acltransformer_testdata
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/driver/
ENV SET_NPU_DEVICE=1

WORKDIR /home

# Install Required Libraries
RUN apt-get update
RUN apt-get install -y \
    gcc \
    g++ \
    make \
    zlib1g \
    zlib1g-dev \
    openssl \
    git \
    wget \
    libsqlite3-dev \
    libssl-dev \
    libffi-dev \
    unzip \
    pciutils \
    net-tools \
    libblas-dev \ 
    gfortran \
    libblas3 \
    libopenblas-dev \
    libxml2-dev \
    libxml2 \
    vim

# Install CMake   
RUN wget https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/docker_file/cmake-3.27.1-linux-aarch64.sh --no-check-certificate
RUN yes| bash cmake-3.27.1-linux-aarch64.sh --prefix=/usr/bin \
    && rm -f cmake-3.27.1-linux-aarch64.sh
ENV PATH="/usr/bin/cmake-3.27.1-linux-aarch64/bin:${PATH}"

# Install Anaconda & Python3
RUN wget https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/docker_file/Anaconda3-2023.07-1-Linux-aarch64.sh --no-check-certificate

RUN bash Anaconda3-2023.07-1-Linux-aarch64.sh -b -p /opt/conda \
    && rm -f Anaconda3-2023.07-1-Linux-aarch64.sh
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
ENV PATH /opt/conda/bin:/opt/conda/condabin:$PATH

RUN conda create --name py3.7 python=3.7.10 
RUN echo "conda activate py3.7" >> ~/.bashrc
ENV PATH /opt/conda/envs/py3.7/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/py3.7/lib
RUN pip3 install \
    attrs \
    numpy \
    decorator \
    sympy \
    cffi \
    pyyaml \
    pathlib2 \
    psutil \
    scipy \
    requests \
    absl-py \
    protobuf==3.20.0 \
    transformers==4.27.1 \
    icetk \
    cpm_kernels \
    gradio \
    mdtex2html 

# Install Ascend Cann Library
RUN wget https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/docker_file/Ascend-cann-toolkit_CANN-6.4_linux-aarch64.run --no-check-certificate \
    && wget https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/docker_file/Ascend-cann-kernels-910b_CANN-6.4_linux.run --no-check-certificate
RUN chmod +x *.run
RUN ./Ascend-cann-toolkit_CANN-6.4_linux-aarch64.run --install
RUN ./Ascend-cann-kernels-910b_CANN-6.4_linux.run --install \
    && rm -f Ascend-cann-toolkit_CANN-6.4_linux-aarch64.run Ascend-cann-kernels-910b_CANN-6.4_linux.run

# Install Torch, Torch_npu, Apex
RUN wget https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/docker_file/torch-1.11.0-cp37-cp37m-linux_aarch64.whl --no-check-certificate \
    && pip3 install torch-1.11.0-cp37-cp37m-linux_aarch64.whl \
    && rm -f torch-1.11.0-cp37-cp37m-linux_aarch64.whl

RUN mkdir code \
    && cd code \
    && git clone https://gitee.com/ascend/pytorch.git -b v1.11.0 --depth=1 v1.11.0
RUN cd /home/code/v1.11.0 \
    && bash ci/build.sh --python=3.7 \
    && pip3 install dist/torch_npu-1.11.0.post1-cp37-cp37m-linux_aarch64.whl

RUN cd /home/code \
    && git clone -b master https://gitee.com/ascend/apex.git
RUN cd /home/code/apex \
    && git clone https://github.com/NVIDIA/apex.git \
    && cd apex \
    && git checkout 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a \
    && cd ../scripts \
    && bash gen.sh \
    && cd ../apex \
    && python3 setup.py --cpp_ext --npu_float_status bdist_wheel \
    && pip3 install dist/apex-0.1_ascend-cp37-cp37m-linux_aarch64.whl
RUN rm -rf /home/code

RUN pip3 install pydantic==1.10.9 --force-reinstall

# Load Model Weights
WORKDIR /data/acltransformer_testdata/weights/chatglm6b
RUN wget https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00001-of-00008.bin --no-check-certificate \
    && wget https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00002-of-00008.bin --no-check-certificate \
    && wget https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00003-of-00008.bin --no-check-certificate \
    && wget https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00004-of-00008.bin --no-check-certificate \
    && wget https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00005-of-00008.bin --no-check-certificate \
    && wget https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00006-of-00008.bin --no-check-certificate \
    && wget https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00007-of-00008.bin --no-check-certificate \
    && wget https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00008-of-00008.bin --no-check-certificate

# Create environment file
RUN touch set_env.sh \
    && echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> set_env.sh
RUN touch proxy.sh  \
    && echo "export http_proxy=http://90.253.11.99:6688" >> proxy.sh \
    && echo "export https_proxy=$http_proxy" >> proxy.sh \
    && echo "export no_proxy=127.0.0.1,*.huawei.com,localhost,local,.local" >> proxy.sh