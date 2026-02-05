"""
Databricks training launcher for Diffusion-Planner using TorchDistributor.

This module provides utilities to run distributed training on Databricks
GPU clusters using PyTorch's DistributedDataParallel (DDP).

Usage in Databricks notebook:
    from databricks.train_databricks import train_diffusion_planner

    train_diffusion_planner(
        train_set_path="/dbfs/mnt/nuplan/processed",
        train_set_list_path="/dbfs/mnt/nuplan/train_list.json",
        output_path="/dbfs/mnt/diffusion-planner/checkpoints",
        num_gpus=4
    )
"""
import os
import sys
from typing import Optional


def train_diffusion_planner(
    train_set_path: str,
    train_set_list_path: str,
    output_path: str,
    num_gpus: int = 4,
    epochs: int = 500,
    batch_size: int = 2048,
    learning_rate: float = 5e-4,
    resume_path: Optional[str] = None,
    use_mlflow: bool = True,
    **extra_args
):
    """
    Launch distributed Diffusion-Planner training on Databricks.

    Args:
        train_set_path: Path to preprocessed training data (DBFS or mounted storage)
        train_set_list_path: Path to training scenario list JSON
        output_path: Path for checkpoints and logs
        num_gpus: Number of GPUs to use for training
        epochs: Number of training epochs
        batch_size: Global batch size (will be divided across GPUs)
        learning_rate: Learning rate for optimizer
        resume_path: Path to resume training from checkpoint
        use_mlflow: Whether to use MLflow for logging (recommended for Databricks)
        **extra_args: Additional arguments passed to train_predictor.py
    """
    from pyspark.ml.torch.distributor import TorchDistributor

    # Prepare training arguments
    train_args = {
        'train_set': train_set_path,
        'train_set_list': train_set_list_path,
        'save_dir': output_path,
        'train_epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'ddp': True,
        **extra_args
    }

    if resume_path:
        train_args['resume_model_path'] = resume_path

    def training_function():
        """Inner function executed on each worker."""
        import os
        import sys
        import subprocess

        # NCCL configuration for AWS networking
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
        os.environ["NCCL_P2P_DISABLE"] = "1"  # Required for G5 instances
        os.environ["NCCL_DEBUG"] = "INFO"

        # Add project to Python path
        sys.path.insert(0, '/opt/diffusion-planner')

        # Get distributed training info from environment
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))

        print(f"Worker started: local_rank={local_rank}, world_size={world_size}")

        # Build command line arguments
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={local_world_size}",
            "/opt/diffusion-planner/train_predictor.py"
        ]

        # Add training arguments
        for key, value in train_args.items():
            if isinstance(value, bool):
                cmd.append(f"--{key}={str(value)}")
            else:
                cmd.append(f"--{key}={value}")

        print(f"Running command: {' '.join(cmd)}")

        # Execute training
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode

    # Launch distributed training
    print(f"Starting distributed training with {num_gpus} GPUs")
    distributor = TorchDistributor(
        num_processes=num_gpus,
        local_mode=False,
        use_gpu=True
    )

    return distributor.run(training_function)


def train_single_gpu(
    train_set_path: str,
    train_set_list_path: str,
    output_path: str,
    epochs: int = 500,
    batch_size: int = 512,
    learning_rate: float = 5e-4,
    **extra_args
):
    """
    Single GPU training for development/debugging.

    Args:
        train_set_path: Path to preprocessed training data
        train_set_list_path: Path to training scenario list JSON
        output_path: Path for checkpoints and logs
        epochs: Number of training epochs
        batch_size: Batch size (smaller for single GPU)
        learning_rate: Learning rate
        **extra_args: Additional arguments
    """
    import subprocess
    import sys

    cmd = [
        sys.executable, "/opt/diffusion-planner/train_predictor.py",
        f"--train_set={train_set_path}",
        f"--train_set_list={train_set_list_path}",
        f"--save_dir={output_path}",
        f"--train_epochs={epochs}",
        f"--batch_size={batch_size}",
        f"--learning_rate={learning_rate}",
        "--ddp=False"
    ]

    for key, value in extra_args.items():
        cmd.append(f"--{key}={value}")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


# Databricks notebook helper functions

def setup_mlflow_experiment(experiment_name: str = "Diffusion-Planner"):
    """
    Set up MLflow experiment for Databricks tracking.

    Args:
        experiment_name: Name of the MLflow experiment
    """
    import mlflow
    mlflow.set_experiment(f"/{experiment_name}")
    print(f"MLflow experiment set to: /{experiment_name}")
    return mlflow.get_experiment_by_name(f"/{experiment_name}")


def check_gpu_availability():
    """Check GPU availability on the cluster."""
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

    return torch.cuda.is_available()


def verify_installation():
    """Verify that all required packages are installed."""
    packages = [
        ('torch', 'PyTorch'),
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('timm', 'timm'),
        ('mlflow', 'MLflow'),
        ('diffusion_planner', 'Diffusion-Planner'),
    ]

    results = {}
    for package, name in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"{name}: {version}")
            results[package] = True
        except ImportError:
            print(f"{name}: NOT INSTALLED")
            results[package] = False

    # Check nuplan-devkit separately
    try:
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
        print("nuplan-devkit: installed")
        results['nuplan'] = True
    except ImportError:
        print("nuplan-devkit: NOT INSTALLED")
        results['nuplan'] = False

    return all(results.values())
