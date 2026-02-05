#!/bin/bash
# =============================================================================
# Download nuPlan Mini Dataset using uv
# =============================================================================
#
# Usage:
#   ./scripts/setup_mini_data.sh
#   ./scripts/setup_mini_data.sh --data-dir ~/my-data
#
# Prerequisites:
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - nuPlan account (register at https://www.nuscenes.org/nuplan)
#
# =============================================================================

set -e

# Default data directory
DATA_DIR="${1:-$HOME/data/nuplan}"

echo "============================================="
echo "nuPlan Mini Dataset Setup (using uv)"
echo "============================================="
echo "Data directory: $DATA_DIR"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv version: $(uv --version)"
echo ""

# Create virtual environment and install nuplan-devkit
echo "Step 1: Creating virtual environment..."
cd "$(dirname "$0")/.."
uv venv .venv-data --python 3.9

echo ""
echo "Step 2: Installing nuplan-devkit..."
uv pip install --python .venv-data/bin/python nuplan-devkit

echo ""
echo "Step 3: Downloading nuPlan mini dataset..."
echo "This will download ~5GB of data to: $DATA_DIR"
echo ""

.venv-data/bin/python -c "
from nuplan.database.nuplan_db_orm.nuplandb import download_nuplan_mini
download_nuplan_mini('$DATA_DIR')
print('Download complete!')
"

echo ""
echo "Step 4: Verifying download..."
echo ""
echo "Directory structure:"
find "$DATA_DIR" -maxdepth 3 -type d | head -20

echo ""
echo "============================================="
echo "SUCCESS! Mini dataset downloaded to:"
echo "  $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Compress: tar -czvf nuplan-mini.tar.gz -C ~/data nuplan"
echo "  2. Upload:   scp -i key.pem nuplan-mini.tar.gz ubuntu@<EC2_IP>:~/"
echo "============================================="
