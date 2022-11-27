ARG IMAGE_NAME
FROM ${IMAGE_NAME}:10.2-base-ubuntu18.04 as base

FROM base as base-x86_64

ENV NV_CUDA_LIB_VERSION 10.2.89-1
ENV NV_NVTX_VERSION 10.2.89-1
ENV NV_LIBNPP_VERSION 10.2.89-1
ENV NV_LIBCUSPARSE_VERSION 10.2.89-1

ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas10

ENV NV_LIBCUBLAS_VERSION 10.2.2.89-1
ENV NV_LIBCUBLAS_PACKAGE ${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}

ENV NV_LIBNCCL_PACKAGE_NAME "libnccl2"
ENV NV_LIBNCCL_PACKAGE_VERSION 2.15.5-1
ENV NCCL_VERSION 2.15.5
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda10.2
# FROM base as base-ppc64le

# ENV NV_CUDA_LIB_VERSION 10.2.89-1
# ENV NV_NVTX_VERSION 10.2.89-1
# ENV NV_LIBNPP_VERSION 10.2.89-1
# ENV NV_LIBCUSPARSE_VERSION 10.2.89-1

# ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas10

# ENV NV_LIBCUBLAS_VERSION 10.2.2.89-1
# ENV NV_LIBCUBLAS_PACKAGE ${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}

# ENV NV_LIBNCCL_PACKAGE_NAME "libnccl2"
# ENV NV_LIBNCCL_PACKAGE_VERSION 2.11.4-1
# ENV NCCL_VERSION 2.11.4
# ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda10.2

ARG TARGETARCH
FROM base-x86_64

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-10-2=${NV_CUDA_LIB_VERSION} \
    cuda-npp-10-2=${NV_LIBNPP_VERSION} \
    cuda-nvtx-10-2=${NV_NVTX_VERSION} \
    cuda-cusparse-10-2=${NV_LIBCUSPARSE_VERSION} \
    ${NV_LIBCUBLAS_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME}

ENV LIBRARY_PATH /usr/local/cuda-10.2/lib64
ENV CUDA_HOME /usr/local/cuda-10.2

# FROM ubuntu:18.04 as base

# FROM base as base-amd64

# ENV NVARCH x86_64
# ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2 brand=tesla,driver>=418,driver<419"
# ENV NV_CUDA_CUDART_VERSION 10.2.89-1

# FROM base as base-ppc64le

# ENV NVARCH ppc64el
# ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2"
# ENV NV_CUDA_CUDART_VERSION 10.2.89-1

# FROM base-amd64

# ARG TARGETARCH

# LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gnupg2 curl ca-certificates && \
#     curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/${NVARCH}/3bf863cc.pub | apt-key add - && \
#     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
#     apt-get purge --autoremove -y curl \
#     && rm -rf /var/lib/apt/lists/*

# ENV CUDA_VERSION 10.2.89

# # For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     cuda-cudart-10-2=${NV_CUDA_CUDART_VERSION} \
#     cuda-compat-10-2 \
#     && ln -s cuda-10.2 /usr/local/cuda && \
#     rm -rf /var/lib/apt/lists/*

# # Required for nvidia-docker v1
# RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
#     echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
# ENV CUDA_HOME /usr/local/cuda-10.2
# RUN export FORCE_CUDA="1"
# # COPY NGC-DL-CONTAINER-LICENSE /

# # nvidia-container-runtime
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
#--upgrade
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install" && \
    GIT_CLONE="git clone --depth 10" && \
    apt-get update && apt-get install -y --no-install-recommends &&\
     DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
	    nano \
        libx11-dev \
        fish \
        libsparsehash-dev \
        sqlite3 \
        libsqlite3-dev \
        curl \
        libcurl4-openssl-dev \
        python3-opengl \
        pkg-config \
        && \
        DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
    && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        # python3-pip \
        # python-wheel \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.8 ~/get-pip.py && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python && \
        $PIP_INSTALL \
        setuptools \ 
        numpy \
        torch==1.6.0 torchvision==0.7.0 \
        # torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html \
        # torchvision \
        && \
    # ==59.8.0
# ==================================================================
# python
# ------------------------------------------------------------------
    $PIP_INSTALL \
        scipy \
        matplotlib \
        Cython \
        tqdm \
        provider \
        imageio \
        # tfrecord \
        natsort \
        joblib \
        tensorboard \
        # coolname \
        # tabulate \
        # runx \
        # ninja \
        # nose \
        # memcnn \
        dominate \
        # cffi \
        # piexif \
        scikit-image \
        # jupyter \
        sklearn \
        # numba \
        einops \
        opencv-python \
        open3d \
        torchsummary \
        # pytictoc \
        # gdown \
        # timm \
        h5py \
        bz2file \
        hdf5storage \
        pandas \
        PyYAML \
        Pillow \
        plyfile \
        pyntcloud \
        # pycocotools \
        pickleshare \
        trimesh \
        pyrender \
        # p2j \
        mesh-to-sdf \
        && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN pip install torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html 
RUN pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html 
# RUN pip install torch-geometric 
RUN pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html 
# RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
#     GIT_CLONE="git clone --recurse-submodules" && \
#     $GIT_CLONE \
#         https://github.com/rubenwiersma/deltaconv.git && \
#     cd deltaconv && \
#     $PIP_INSTALL \
#         ./ \
#         # pyOpenGL_accelerate
#         && \
#     curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz && \
#     tar xzf 1.10.0.tar.gz

# workdir is where u work in dockers
# copy . /app copies content of ur supposed working dir to the docker wk dir
WORKDIR /app
COPY . /app

# ENV CUB_HOME=$PWD/cub-1.10.0
# RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
#     $PIP_INSTALL \
#         "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# ENV PATH="$PATH:/usr/lib/llvm-6.0/bin"
# ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/llvm-6.0/lib"

RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# RUN echo "CUB_HOME env path: ${CUB_HOME}"
RUN python -c "import torch; print(torch.__version__)" 
RUN python -c "import torch; print(torch.version.cuda)" 
RUN echo "CUB_HOME env path: ${PATH}" 
RUN echo "CUB_HOME env path: ${CPATH}" 

CMD ["python", "dataset.py "]
