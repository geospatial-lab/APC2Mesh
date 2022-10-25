ARG IMAGE_NAME
FROM ${IMAGE_NAME}:11.0.3-runtime-ubuntu20.04 as base

ENV NV_CUDA_LIB_VERSION 11.0.3-1

FROM base as base-amd64

# ENV NV_CUDA_CUDART_DEV_VERSION 11.0.221-1
# ENV NV_NVML_DEV_VERSION 11.0.167-1
ENV NV_LIBCUSPARSE_DEV_VERSION 11.1.1.245-1
ENV NV_LIBNPP_DEV_VERSION 11.1.0.245-1
ENV NV_LIBNPP_DEV_PACKAGE libnpp-dev-11-0=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_VERSION 11.2.0.252-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME libcublas-dev-11-0
ENV NV_LIBCUBLAS_DEV_PACKAGE ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

# ENV NV_NVPROF_VERSION 11.0.221-1
# ENV NV_NVPROF_DEV_PACKAGE cuda-nvprof-11-0=${NV_NVPROF_VERSION}

ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION 2.13.4-1
ENV NCCL_VERSION 2.13.4-1
ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda11.0

ARG TARGETARCH
FROM base-amd64
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
RUN apt-get update && apt-get install -y --no-install-recommends \
    # libtinfo5 libncursesw5 \
    # cuda-cudart-dev-11-6=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-11-0=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-11-0=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-11-0=${NV_CUDA_LIB_VERSION} \
    # cuda-nvml-dev-11-6=${NV_NVML_DEV_VERSION} \
    # cuda-nvprof-11-6=${NV_NVPROF_VERSION} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    # cuda-nvtx-11-6=${NV_NVTX_VERSION} \
    libcusparse-dev-11-0=${NV_LIBCUSPARSE_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME}

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
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
        # libosmesa6-dev \ 
        # llvm-6.0 \
        # llvm-6.0-tools \
        # freeglut3 \
        # freeglut3-dev \
        # libglfw3-dev \
        # libgles2-mesa-dev \
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
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.8 ~/get-pip.py && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python && \
        $PIP_INSTALL \
        setuptools \
        && \
# ==================================================================
# python
# ------------------------------------------------------------------
    $PIP_INSTALL \
        numpy \
        torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html \
        scipy \
        matplotlib \
        Cython \
        tqdm \
        provider \
        imageio \
        tfrecord \
        natsort \
        joblib \
        tensorboard \
        coolname \
        tabulate \
        runx \
        ninja \
        nose \
        memcnn \
        dominate \
        cffi \
        piexif \
        scikit-image \
        jupyter \
        sklearn \
        numba \
        einops \
        opencv-python \
        open3d \
        torchsummary \
        pytictoc \
        gdown \
        timm \
        h5py \
        bz2file \
        hdf5storage \
        pandas \
        PyYAML \
        Pillow \
        plyfile \
        pyntcloud \
        pycocotools \
        pickleshare \
        trimesh \
        pyrender \
        p2j \
        && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    $PIP_INSTALL \
        mesh-to-sdf \
        torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html \
        torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html \
        torch-geometric \
        torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html 
    # && \
# # Install osmesa from source
#     wget https://archive.mesa3d.org/mesa-18.3.3.tar.gz && \
#     tar -zxvf mesa-18.3.3.tar.gz && cd mesa-18.3.3 && \
#     ./configure --prefix=/usr/local \
#             --enable-opengl --disable-gles1 --disable-gles2   \
#             --disable-va --disable-xvmc --disable-vdpau       \
#             --enable-shared-glapi                             \
#             --disable-texture-float                           \
#             --enable-gallium-llvm --enable-llvm-shared-libs   \
#             --with-gallium-drivers=swrast,swr                 \
#             --disable-dri --with-dri-drivers=                 \
#             --disable-egl --with-egl-platforms= --disable-gbm \
#             --disable-glx                                     \
#             --disable-osmesa --enable-gallium-osmesa          \
#             ac_cv_path_LLVM_CONFIG=llvm-config-6.0 && \
#      make -j4 && make install && cd
# RUN wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb && \
#     dpkg -i ./mesa_18.3.3-0.deb || true && \
#     apt-get install -f

# # RUN MESA_HOME="/mesa-18.3.3"
# ENV LIBRARY_PATH="$LIBRARY_PATH:/mesa-18.3.3/lib"
# ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$/mesa-18.3.3/lib"
# ENV C_INCLUDE_PATH="$C_INCLUDE_PATH:$/mesa-18.3.3/include/"
# ENV CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/mesa-18.3.3/include/"

# Install pyopengl from github
# RUN pip uninstall pyopengl
RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --recurse-submodules" && \
    $GIT_CLONE \
        https://github.com/rubenwiersma/deltaconv.git && \
    cd deltaconv && \
    $PIP_INSTALL \
        ./ 
        # pyOpenGL_accelerate

# ENV PATH="$PATH:/usr/lib/llvm-6.0/bin"
# ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/llvm-6.0/lib"
# export PATH=my/path/to/llvm-install/bin:$PATH
# export LD_LIBRARY_PATH=my/path/to/llvm-install/lib:$LD_LIBRARY_PATH

# workdir is where u work in dockers
# copy . /app copies content of ur supposed working dir to the docker wk dir
WORKDIR /app
COPY . /app

RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

CMD ["python", "network.py "]
