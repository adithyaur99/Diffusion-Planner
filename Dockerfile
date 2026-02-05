# =============================================================================
# Diffusion-Planner Dockerfile for AWS Databricks
# =============================================================================

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# =============================================================================
# System Dependencies + Databricks Requirements
# =============================================================================
RUN apt-get update && apt-get install -y \
    # Required for Databricks Container Services
    openjdk-8-jdk \
    bash \
    iproute2 \
    coreutils \
    procps \
    sudo \
    # Build tools
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    # Python
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    # GIS/Geospatial (for nuplan-devkit)
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    # Graphics/Visualization
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Networking
    net-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# =============================================================================
# Databricks Requirements
# =============================================================================
RUN mkdir -p /databricks/python3 \
    && ln -s /usr/bin/python3.9 /databricks/python3/python \
    && ln -s /usr/bin/pip3 /databricks/python3/pip

# Java on PATH (required by Databricks)
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# =============================================================================
# nuplan-devkit Installation
# =============================================================================
# Install geospatial stack first
RUN pip install --no-cache-dir \
    numpy==1.23.4 \
    pandas \
    scipy \
    "geopandas>=0.12.1" \
    "Shapely>=2.0.0" \
    pyogrio \
    rtree \
    Fiona \
    rasterio \
    "pyquaternion>=0.9.5" \
    casadi \
    "control==0.9.1" \
    sympy \
    "hydra-core==1.1.0rc1" \
    "bokeh==2.4.3" \
    "SQLAlchemy==1.4.27" \
    joblib \
    psutil \
    tqdm \
    opencv-python \
    Pillow \
    matplotlib

# Clone and install nuplan-devkit
RUN git clone https://github.com/motional/nuplan-devkit.git /opt/nuplan-devkit \
    && cd /opt/nuplan-devkit \
    && pip install --no-cache-dir -e .

# =============================================================================
# PyTorch Installation (CUDA 11.8)
# =============================================================================
RUN pip install --no-cache-dir \
    torch==2.0.0+cu118 \
    torchvision==0.15.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Additional PyTorch ecosystem
RUN pip install --no-cache-dir \
    pytorch_lightning==2.0.1 \
    tensorboard==2.11.2 \
    timm==1.0.10 \
    mmengine \
    einops

# =============================================================================
# MLflow for Databricks
# =============================================================================
RUN pip install --no-cache-dir \
    "mlflow>=2.5.0" \
    databricks-sdk

# =============================================================================
# Diffusion-Planner Installation
# =============================================================================
COPY . /opt/diffusion-planner
WORKDIR /opt/diffusion-planner

# Install the package
RUN pip install --no-cache-dir -e .

# =============================================================================
# Environment Configuration
# =============================================================================
ENV PYTHONPATH=/opt/diffusion-planner:/opt/nuplan-devkit:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL optimizations for AWS
ENV NCCL_DEBUG=INFO
ENV NCCL_SOCKET_IFNAME=eth0

# =============================================================================
# Entrypoint (Databricks ignores CMD/ENTRYPOINT but useful for local testing)
# =============================================================================
CMD ["/bin/bash"]
