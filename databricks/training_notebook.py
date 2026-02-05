# Databricks notebook source
# MAGIC %md
# MAGIC # Diffusion-Planner Training on Databricks
# MAGIC
# MAGIC This notebook provides an end-to-end workflow for training Diffusion-Planner on AWS Databricks.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Cluster with custom Docker image (`diffusion-planner:latest` from ECR)
# MAGIC - nuPlan data mounted or accessible via DBFS
# MAGIC - GPU-enabled cluster (recommended: g5.12xlarge for 4x A10G GPUs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verify Installation

# COMMAND ----------

# Verify GPU availability and package installation
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

# COMMAND ----------

# Verify Diffusion-Planner installation
try:
    import diffusion_planner
    print("diffusion_planner: OK")

    from diffusion_planner.utils.mlflow_log import MLflowLogger
    print("MLflowLogger: OK")

    from diffusion_planner.model.diffusion_planner import Diffusion_Planner
    print("Diffusion_Planner model: OK")
except ImportError as e:
    print(f"Import error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configure Training Parameters

# COMMAND ----------

# Training configuration
config = {
    # Data paths - UPDATE THESE
    "train_set_path": "/dbfs/mnt/nuplan/processed",           # Path to preprocessed data
    "train_set_list_path": "/dbfs/mnt/nuplan/train_list.json", # Training scenario list

    # Output path
    "output_path": "/dbfs/mnt/diffusion-planner/checkpoints",

    # Training hyperparameters
    "num_gpus": 4,          # Number of GPUs (match your cluster)
    "epochs": 500,          # Training epochs
    "batch_size": 2048,     # Global batch size (divided across GPUs)
    "learning_rate": 5e-4,  # Learning rate

    # Optional: resume from checkpoint
    "resume_path": None,    # Set to checkpoint path to resume
}

print("Training configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Set Up MLflow Experiment

# COMMAND ----------

import mlflow

# Set experiment name
experiment_name = "/Diffusion-Planner/training-v1"
mlflow.set_experiment(experiment_name)
print(f"MLflow experiment: {experiment_name}")

# Get experiment info
experiment = mlflow.get_experiment_by_name(experiment_name)
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Artifact location: {experiment.artifact_location}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Launch Distributed Training

# COMMAND ----------

from databricks.train_databricks import train_diffusion_planner

# Start training
print("Starting distributed training...")
print(f"Using {config['num_gpus']} GPUs")

result = train_diffusion_planner(
    train_set_path=config["train_set_path"],
    train_set_list_path=config["train_set_list_path"],
    output_path=config["output_path"],
    num_gpus=config["num_gpus"],
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    learning_rate=config["learning_rate"],
    resume_path=config["resume_path"],
)

print(f"Training completed with return code: {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. View Training Results
# MAGIC
# MAGIC Navigate to **Experiments** in the Databricks sidebar to view:
# MAGIC - Training metrics (loss curves)
# MAGIC - Model checkpoints
# MAGIC - Training parameters

# COMMAND ----------

# List saved checkpoints
import os

checkpoint_dir = config["output_path"]
if os.path.exists(checkpoint_dir):
    print(f"Checkpoints in {checkpoint_dir}:")
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path):
            print(f"  {item}/")
            for subitem in os.listdir(item_path)[:5]:
                print(f"    {subitem}")
else:
    print(f"Checkpoint directory not found: {checkpoint_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative: Single GPU Training (for debugging)

# COMMAND ----------

# Uncomment to run single-GPU training for debugging
# from databricks.train_databricks import train_single_gpu
#
# train_single_gpu(
#     train_set_path=config["train_set_path"],
#     train_set_list_path=config["train_set_list_path"],
#     output_path=config["output_path"],
#     epochs=10,  # Fewer epochs for testing
#     batch_size=256,  # Smaller batch for single GPU
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster Configuration
# MAGIC
# MAGIC Recommended cluster settings for this notebook:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "spark_version": "13.3.x-gpu-ml-scala2.12",
# MAGIC   "node_type_id": "g5.12xlarge",
# MAGIC   "num_workers": 0,
# MAGIC   "driver_node_type_id": "g5.12xlarge",
# MAGIC   "docker_image": {
# MAGIC     "url": "<YOUR_ECR_REGISTRY>/diffusion-planner:latest"
# MAGIC   },
# MAGIC   "spark_conf": {
# MAGIC     "spark.task.resource.gpu.amount": "1"
# MAGIC   }
# MAGIC }
# MAGIC ```
