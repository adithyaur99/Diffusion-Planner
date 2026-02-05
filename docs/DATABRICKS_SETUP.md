# Diffusion-Planner: Databricks Training Guide

This guide covers how to train Diffusion-Planner on AWS Databricks using Docker containers and distributed GPU training.

## Overview

| Component | Technology |
|-----------|------------|
| Container | Docker with CUDA 11.8 + cuDNN 8 |
| Base OS | Ubuntu 20.04 |
| Python | 3.9 |
| ML Framework | PyTorch 2.0.0 |
| Distributed Training | TorchDistributor (DDP) |
| Experiment Tracking | MLflow |
| Cloud | AWS Databricks |

---

## Prerequisites

- AWS CLI configured with ECR permissions
- Docker installed locally
- AWS Databricks workspace
- nuPlan dataset (mounted or in DBFS)

---

## Quick Start

### 1. Build Docker Image

```bash
cd /path/to/Diffusion-Planner

# Build and test locally
./scripts/build_and_push.sh latest --test

# Build only (no push)
./scripts/build_and_push.sh latest --build-only
```

### 2. Push to AWS ECR

```bash
# Set your AWS region
export AWS_REGION=us-east-1

# Build and push
./scripts/build_and_push.sh latest
```

The script will:
- Auto-detect your AWS Account ID
- Create the ECR repository if needed
- Push the image with your specified tag

### 3. Configure Databricks Cluster

Create a cluster with the following configuration:

```json
{
  "cluster_name": "diffusion-planner-training",
  "spark_version": "13.3.x-gpu-ml-scala2.12",
  "node_type_id": "g5.12xlarge",
  "num_workers": 0,
  "driver_node_type_id": "g5.12xlarge",
  "docker_image": {
    "url": "<AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/diffusion-planner:latest"
  },
  "spark_conf": {
    "spark.task.resource.gpu.amount": "1"
  },
  "aws_attributes": {
    "first_on_demand": 1,
    "availability": "SPOT_WITH_FALLBACK"
  }
}
```

### 4. Run Training

In a Databricks notebook:

```python
from databricks.train_databricks import train_diffusion_planner

train_diffusion_planner(
    train_set_path="/dbfs/mnt/nuplan/processed",
    train_set_list_path="/dbfs/mnt/nuplan/train_list.json",
    output_path="/dbfs/mnt/diffusion-planner/checkpoints",
    num_gpus=4,
    epochs=500,
    batch_size=2048,
    learning_rate=5e-4
)
```

---

## GPU Instance Recommendations

| Use Case | Instance Type | GPUs | Memory | Notes |
|----------|--------------|------|--------|-------|
| Development | g5.xlarge | 1x A10G | 24 GB | Use spot instances |
| Training | g5.12xlarge | 4x A10G | 96 GB | Good price/performance |
| Production | p4d.24xlarge | 8x A100 | 320 GB | Fastest training |
| Large Scale | 4x p4d.24xlarge | 32x A100 | 1.2 TB | Multi-node |

---

## MLflow Integration

The project includes an MLflow logger as a drop-in replacement for WandB/TensorBoard.

### Enable MLflow Logging

In `train_predictor.py`, change the import:

```python
# Before (WandB/TensorBoard)
from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger

# After (MLflow)
from diffusion_planner.utils.mlflow_log import MLflowLogger as Logger
```

### View Experiments

1. Go to your Databricks workspace
2. Click **Experiments** in the sidebar
3. Find your experiment under `/Diffusion-Planner/`

MLflow automatically tracks:
- Training parameters
- Loss metrics per epoch
- Learning rate schedules
- Model checkpoints as artifacts

---

## Project Structure

```
Diffusion-Planner/
├── Dockerfile                          # Container definition
├── .dockerignore                       # Build exclusions
├── databricks/
│   ├── __init__.py
│   ├── train_databricks.py             # TorchDistributor wrapper
│   └── training_notebook.py            # Sample notebook
├── diffusion_planner/
│   └── utils/
│       ├── tb_log.py                   # Original WandB logger
│       └── mlflow_log.py               # MLflow logger (new)
├── scripts/
│   └── build_and_push.sh               # ECR deployment script
└── docs/
    └── DATABRICKS_SETUP.md             # This file
```

---

## Training API Reference

### `train_diffusion_planner()`

Launch distributed training on Databricks.

```python
train_diffusion_planner(
    train_set_path: str,           # Path to preprocessed data
    train_set_list_path: str,      # Path to scenario list JSON
    output_path: str,              # Checkpoint output directory
    num_gpus: int = 4,             # Number of GPUs
    epochs: int = 500,             # Training epochs
    batch_size: int = 2048,        # Global batch size
    learning_rate: float = 5e-4,   # Learning rate
    resume_path: str = None,       # Resume from checkpoint
    **extra_args                   # Additional train_predictor.py args
)
```

### `train_single_gpu()`

Single GPU training for development/debugging.

```python
train_single_gpu(
    train_set_path: str,
    train_set_list_path: str,
    output_path: str,
    epochs: int = 500,
    batch_size: int = 512,         # Smaller for single GPU
    learning_rate: float = 5e-4,
    **extra_args
)
```

### Helper Functions

```python
from databricks.train_databricks import (
    check_gpu_availability,    # Verify GPU setup
    verify_installation,       # Check all packages
    setup_mlflow_experiment    # Configure MLflow
)
```

---

## Troubleshooting

### NCCL Errors on AWS

If you see NCCL communication errors, the scripts already set:

```bash
NCCL_SOCKET_IFNAME=eth0
NCCL_P2P_DISABLE=1  # Required for G5 instances
```

If issues persist, try adding to your init script:

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
```

### Docker Build Fails

The image is ~15GB due to geospatial dependencies. Ensure sufficient disk space.

If nuplan-devkit fails to install:

```bash
# Try installing GDAL first
apt-get install -y libgdal-dev gdal-bin
pip install GDAL==$(gdal-config --version)
```

### Out of Memory

Reduce batch size or use gradient accumulation:

```python
train_diffusion_planner(
    ...,
    batch_size=1024,  # Halve the batch size
)
```

### MLflow Not Logging

Ensure you're on rank 0 (only the main process logs):

```python
# Check in your training code
if global_rank == 0:
    logger.log_metrics(...)
```

---

## Data Preparation

Before training, preprocess the nuPlan data:

```bash
python data_process.py \
    --data_path /path/to/nuplan/trainval \
    --map_path /path/to/nuplan/maps \
    --save_path /path/to/output
```

Or use the provided shell script:

```bash
./data_process.sh
```

---

## Cost Optimization Tips

1. **Use Spot Instances**: Set `"availability": "SPOT_WITH_FALLBACK"` in cluster config
2. **Auto-termination**: Enable auto-termination after idle period
3. **Start Small**: Test with g5.xlarge before scaling to larger instances
4. **Checkpoint Frequently**: Set `--save_utd 10` to save every 10 epochs

---

## References

- [Diffusion-Planner Paper](https://arxiv.org/abs/2404.XXXXX) (ICLR 2025)
- [nuPlan Dataset](https://www.nuscenes.org/nuplan)
- [Databricks TorchDistributor](https://docs.databricks.com/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
