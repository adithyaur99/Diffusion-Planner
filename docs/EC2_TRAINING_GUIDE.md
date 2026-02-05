# Diffusion-Planner: EC2 Training Guide

Step-by-step guide to train Diffusion-Planner on an AWS EC2 GPU instance.

---

## Step 1: Launch EC2 Instance

### Option A: AWS Console

1. Go to **EC2 → Launch Instance**
2. Select **Deep Learning AMI GPU PyTorch 2.0 (Ubuntu 20.04)**
   - This comes with NVIDIA drivers, Docker, and CUDA pre-installed
3. Choose instance type:

   | Instance | GPUs | GPU Memory | Cost/hr (on-demand) | Recommended For |
   |----------|------|------------|---------------------|-----------------|
   | g5.xlarge | 1x A10G | 24 GB | ~$1.00 | Testing |
   | g5.2xlarge | 1x A10G | 24 GB | ~$1.21 | Development |
   | g5.12xlarge | 4x A10G | 96 GB | ~$5.67 | Training |
   | p4d.24xlarge | 8x A100 | 320 GB | ~$32.77 | Production |

4. Configure storage: **200+ GB** (for Docker image + data)
5. Security group: Allow SSH (port 22)
6. Launch with your key pair

### Option B: AWS CLI

```bash
# Launch g5.12xlarge with Deep Learning AMI
aws ec2 run-instances \
  --image-id ami-0123456789abcdef0 \  # Find latest Deep Learning AMI ID
  --instance-type g5.12xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=diffusion-planner-training}]'
```

---

## Step 2: Connect to Instance

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

---

## Step 3: Verify GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Expected output: Shows your GPU(s), driver version, CUDA version
```

If `nvidia-smi` fails, install drivers:
```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

---

## Step 4: Install Docker (if not pre-installed)

```bash
# Check if Docker exists
docker --version

# If not installed:
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify Docker GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

---

## Step 5: Clone Repository

```bash
cd ~
git clone https://github.com/adithyaur99/Diffusion-Planner.git
cd Diffusion-Planner
```

---

## Step 6: Build Docker Image

```bash
# Build the image (takes 15-20 minutes)
docker build -t diffusion-planner:latest .

# Verify build
docker images | grep diffusion-planner
```

**Alternative: Pull pre-built image (if pushed to ECR)**
```bash
# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Pull image
docker pull <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/diffusion-planner:latest
docker tag <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/diffusion-planner:latest diffusion-planner:latest
```

---

## Step 7: Prepare Data

### Option A: Download Mini Dataset Locally, Upload to EC2 (Recommended for Testing)

**On your local machine (Mac/Linux):**

```bash
# 1. Install nuplan-devkit
pip install nuplan-devkit

# 2. Download mini dataset (~5GB)
python -c "
from nuplan.database.nuplan_db_orm.nuplandb import download_nuplan_mini
download_nuplan_mini('$HOME/data/nuplan')
"

# 3. Compress for faster upload
tar -czvf nuplan-mini.tar.gz -C ~/data nuplan

# 4. Upload to EC2
scp -i your-key.pem nuplan-mini.tar.gz ubuntu@<EC2_IP>:~/
```

**On EC2:**

```bash
# 5. Extract data
cd ~
tar -xzvf nuplan-mini.tar.gz
mkdir -p ~/data
mv nuplan ~/data/

# Verify structure
ls ~/data/nuplan/
# Should show: nuplan-v1.1/  maps/
```

### Option B: Download Full Dataset to EC2

```bash
# Create data directory
mkdir -p ~/data/nuplan

# Register at https://www.nuscenes.org/nuplan (free)
# Download using nuplan-devkit
pip install nuplan-devkit
python -c "
from nuplan.database.nuplan_db_orm.nuplandb import download_nuplan
download_nuplan('/home/ubuntu/data/nuplan', version='v1.1')
"
```

### Option C: Sync from S3

```bash
# If you have data in S3
aws s3 sync s3://your-bucket/nuplan ~/data/nuplan
```

### Option D: Mount S3 Bucket (for large datasets)

```bash
# Install s3fs
sudo apt-get install -y s3fs

# Configure credentials
echo "ACCESS_KEY:SECRET_KEY" > ~/.passwd-s3fs
chmod 600 ~/.passwd-s3fs

# Mount bucket
mkdir -p ~/data/nuplan
s3fs your-nuplan-bucket ~/data/nuplan -o passwd_file=~/.passwd-s3fs
```

### Preprocess Data

**For mini dataset:**
```bash
docker run --rm --gpus all \
  -v ~/data/nuplan:/data/nuplan \
  -v ~/data/processed:/data/processed \
  diffusion-planner:latest \
  python /opt/diffusion-planner/data_process.py \
    --data_path /data/nuplan/nuplan-v1.1/mini \
    --map_path /data/nuplan/maps \
    --save_path /data/processed \
    --total_scenarios 500
```

**For full dataset:**
```bash
docker run --rm --gpus all \
  -v ~/data/nuplan:/data/nuplan \
  -v ~/data/processed:/data/processed \
  diffusion-planner:latest \
  python /opt/diffusion-planner/data_process.py \
    --data_path /data/nuplan/nuplan-v1.1/trainval \
    --map_path /data/nuplan/maps \
    --save_path /data/processed
```

---

## Step 8: Run Training

### Single GPU Training

```bash
docker run --rm --gpus all \
  -v ~/data/processed:/data/processed \
  -v ~/checkpoints:/checkpoints \
  diffusion-planner:latest \
  python /opt/diffusion-planner/train_predictor.py \
    --train_set /data/processed \
    --train_set_list /data/processed/diffusion_planner_training.json \
    --save_dir /checkpoints \
    --batch_size 512 \
    --train_epochs 500 \
    --ddp False
```

### Multi-GPU Training (Recommended)

```bash
# For 4 GPUs (g5.12xlarge)
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ~/data/processed:/data/processed \
  -v ~/checkpoints:/checkpoints \
  diffusion-planner:latest \
  torchrun --standalone --nproc_per_node=4 \
    /opt/diffusion-planner/train_predictor.py \
    --train_set /data/processed \
    --train_set_list /data/processed/diffusion_planner_training.json \
    --save_dir /checkpoints \
    --batch_size 2048 \
    --train_epochs 500 \
    --ddp True
```

### Run in Background with Logging

```bash
# Create log directory
mkdir -p ~/logs

# Run training in background
nohup docker run --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ~/data/processed:/data/processed \
  -v ~/checkpoints:/checkpoints \
  diffusion-planner:latest \
  torchrun --standalone --nproc_per_node=4 \
    /opt/diffusion-planner/train_predictor.py \
    --train_set /data/processed \
    --train_set_list /data/processed/diffusion_planner_training.json \
    --save_dir /checkpoints \
    --batch_size 2048 \
    --train_epochs 500 \
    --ddp True \
  > ~/logs/training.log 2>&1 &

# Check progress
tail -f ~/logs/training.log
```

---

## Step 9: Monitor Training

### Check GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use gpustat (prettier)
pip install gpustat
gpustat -i 1
```

### View Training Logs

```bash
# If running in background
tail -f ~/logs/training.log

# If running in Docker foreground, logs show directly
```

### TensorBoard (Optional)

```bash
# Start TensorBoard
docker run --rm -p 6006:6006 \
  -v ~/checkpoints:/checkpoints \
  diffusion-planner:latest \
  tensorboard --logdir /checkpoints --bind_all

# Access at http://<EC2_PUBLIC_IP>:6006
# (Add port 6006 to security group)
```

---

## Step 10: Save Checkpoints to S3

```bash
# Sync checkpoints to S3 periodically
aws s3 sync ~/checkpoints s3://your-bucket/diffusion-planner/checkpoints/

# Or set up automatic sync with cron
crontab -e
# Add: */30 * * * * aws s3 sync ~/checkpoints s3://your-bucket/checkpoints/
```

---

## Step 11: Resume Training

If training is interrupted:

```bash
docker run --rm --gpus all \
  --ipc=host \
  -v ~/data/processed:/data/processed \
  -v ~/checkpoints:/checkpoints \
  diffusion-planner:latest \
  torchrun --standalone --nproc_per_node=4 \
    /opt/diffusion-planner/train_predictor.py \
    --train_set /data/processed \
    --train_set_list /data/processed/diffusion_planner_training.json \
    --save_dir /checkpoints \
    --resume_model_path /checkpoints/training_log/diffusion-planner-training/<TIMESTAMP>/ \
    --batch_size 2048 \
    --ddp True
```

---

## Quick Reference: Docker Commands

```bash
# Build image
docker build -t diffusion-planner:latest .

# List images
docker images

# Run interactive shell
docker run --rm -it --gpus all diffusion-planner:latest bash

# Check running containers
docker ps

# Stop container
docker stop <container_id>

# Remove all stopped containers
docker container prune

# Check disk usage
docker system df
```

---

## Cost Optimization Tips

1. **Use Spot Instances**: Up to 90% cheaper
   ```bash
   aws ec2 request-spot-instances \
     --instance-count 1 \
     --type "one-time" \
     --launch-specification file://spot-spec.json
   ```

2. **Stop instance when not training**: You only pay for storage when stopped

3. **Use smaller instance for debugging**: Start with g5.xlarge, scale up for full training

4. **Checkpoint frequently**: Set `--save_utd 10` to save every 10 epochs

5. **Use gp3 volumes**: Better price/performance than gp2

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch_size 1024  # or 512 for single GPU

# Or enable gradient checkpointing (if supported)
```

### Docker Permission Denied

```bash
sudo usermod -aG docker $USER
newgrp docker
# Or logout and login again
```

### NCCL Errors (Multi-GPU)

```bash
# Add these environment variables
docker run --rm --gpus all \
  -e NCCL_DEBUG=INFO \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_IB_DISABLE=1 \
  ...
```

### Disk Full

```bash
# Check disk usage
df -h

# Clean Docker cache
docker system prune -a

# Remove old checkpoints
rm -rf ~/checkpoints/training_log/*/old_runs/
```

---

## Full Training Script

Save as `~/train.sh`:

```bash
#!/bin/bash
set -e

# Configuration
DATA_PATH=~/data/processed
CHECKPOINT_PATH=~/checkpoints
NUM_GPUS=4
BATCH_SIZE=2048
EPOCHS=500

echo "Starting Diffusion-Planner training..."
echo "GPUs: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"

docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $DATA_PATH:/data/processed \
  -v $CHECKPOINT_PATH:/checkpoints \
  diffusion-planner:latest \
  torchrun --standalone --nproc_per_node=$NUM_GPUS \
    /opt/diffusion-planner/train_predictor.py \
    --train_set /data/processed \
    --train_set_list /data/processed/diffusion_planner_training.json \
    --save_dir /checkpoints \
    --batch_size $BATCH_SIZE \
    --train_epochs $EPOCHS \
    --ddp True

echo "Training complete!"
```

Run with:
```bash
chmod +x ~/train.sh
./train.sh
```

---

## Summary

| Step | Command |
|------|---------|
| 1. Launch EC2 | AWS Console or CLI |
| 2. Connect | `ssh -i key.pem ubuntu@<IP>` |
| 3. Verify GPU | `nvidia-smi` |
| 4. Clone repo | `git clone https://github.com/adithyaur99/Diffusion-Planner.git` |
| 5. Build image | `docker build -t diffusion-planner:latest .` |
| 6. Prep data | Mount or download nuPlan dataset |
| 7. Train | `docker run --gpus all ... torchrun ...` |
| 8. Monitor | `tail -f ~/logs/training.log` |
| 9. Save | `aws s3 sync ~/checkpoints s3://bucket/` |
