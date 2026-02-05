#!/bin/bash
# =============================================================================
# Build and Push Diffusion-Planner Docker Image to AWS ECR
# =============================================================================
#
# Usage:
#   ./scripts/build_and_push.sh                    # Build and push with tag 'latest'
#   ./scripts/build_and_push.sh v1.0               # Build and push with tag 'v1.0'
#   ./scripts/build_and_push.sh latest --test      # Build, test locally, then push
#   ./scripts/build_and_push.sh latest --build-only  # Build only, don't push
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - Docker installed and running
#   - Environment variables (optional):
#     - AWS_REGION: AWS region (default: us-east-1)
#     - AWS_ACCOUNT_ID: Your AWS account ID (auto-detected if not set)
#
# =============================================================================

set -e

# Configuration
IMAGE_NAME="diffusion-planner"
VERSION="${1:-latest}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Auto-detect AWS Account ID if not set
if [ -z "$AWS_ACCOUNT_ID" ]; then
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
    if [ -z "$AWS_ACCOUNT_ID" ]; then
        echo "Error: Could not detect AWS Account ID. Please set AWS_ACCOUNT_ID or configure AWS CLI."
        exit 1
    fi
fi

REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${VERSION}"

echo "============================================="
echo "Diffusion-Planner Docker Build & Push"
echo "============================================="
echo "Image: ${IMAGE_NAME}"
echo "Version: ${VERSION}"
echo "Registry: ${REGISTRY}"
echo "Full name: ${FULL_IMAGE_NAME}"
echo "============================================="

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Building from: $(pwd)"

# Build the image
echo ""
echo "Step 1: Building Docker image..."
docker build \
    --platform linux/amd64 \
    -t ${IMAGE_NAME}:${VERSION} \
    -t ${FULL_IMAGE_NAME} \
    -f Dockerfile \
    .

echo "Build complete!"

# Optional: Test the image locally
if [ "$2" == "--test" ]; then
    echo ""
    echo "Step 2: Testing image locally..."

    # Test basic Python and CUDA
    echo "Testing Python environment..."
    docker run --rm ${IMAGE_NAME}:${VERSION} \
        python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

import diffusion_planner
print('diffusion_planner: imported successfully')

from diffusion_planner.utils.mlflow_log import MLflowLogger
print('MLflowLogger: imported successfully')
"

    # Test with GPU if available
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "Testing with GPU..."
        docker run --rm --gpus all ${IMAGE_NAME}:${VERSION} \
            python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
    fi

    echo "Tests passed!"
fi

# Skip push if --build-only flag
if [ "$2" == "--build-only" ]; then
    echo ""
    echo "Build-only mode. Skipping push to ECR."
    echo "Local image: ${IMAGE_NAME}:${VERSION}"
    exit 0
fi

# Push to ECR
echo ""
echo "Step 3: Authenticating with ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${REGISTRY}

echo ""
echo "Step 4: Creating ECR repository (if needed)..."
aws ecr describe-repositories --repository-names ${IMAGE_NAME} --region ${AWS_REGION} 2>/dev/null || \
    aws ecr create-repository --repository-name ${IMAGE_NAME} --region ${AWS_REGION}

echo ""
echo "Step 5: Pushing to ECR..."
docker push ${FULL_IMAGE_NAME}

# Also push as 'latest' if version is not 'latest'
if [ "$VERSION" != "latest" ]; then
    echo "Also tagging as 'latest'..."
    docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:latest
    docker push ${REGISTRY}/${IMAGE_NAME}:latest
fi

echo ""
echo "============================================="
echo "SUCCESS!"
echo "============================================="
echo "Image URI: ${FULL_IMAGE_NAME}"
echo ""
echo "Use in Databricks cluster config:"
echo '{'
echo '  "docker_image": {'
echo "    \"url\": \"${FULL_IMAGE_NAME}\""
echo '  }'
echo '}'
echo "============================================="
