#!/bin/bash
# =============================================================================
# nuPlan Mini Dataset Setup Guide
# =============================================================================
#
# The nuPlan mini dataset must be downloaded MANUALLY from the website.
# This script helps with extraction and upload to EC2.
#
# =============================================================================

set -e

DATA_DIR="${1:-$HOME/data/nuplan}"

echo "============================================="
echo "nuPlan Mini Dataset Setup"
echo "============================================="
echo ""
echo "STEP 1: Download from website (manual)"
echo "---------------------------------------"
echo "1. Go to: https://www.nuscenes.org/nuplan#download"
echo "2. Sign in or register (free)"
echo "3. Download:"
echo "   - nuplan-v1.1_mini.zip (scenarios)"
echo "   - nuplan-maps-v1.1.zip (maps)"
echo ""
echo "STEP 2: Extract files"
echo "---------------------------------------"
echo "Run these commands after downloading:"
echo ""
echo "  mkdir -p $DATA_DIR"
echo "  unzip ~/Downloads/nuplan-v1.1_mini.zip -d $DATA_DIR/"
echo "  unzip ~/Downloads/nuplan-maps-v1.1.zip -d $DATA_DIR/"
echo ""
echo "STEP 3: Verify structure"
echo "---------------------------------------"
echo "Your data directory should look like:"
echo ""
echo "  $DATA_DIR/"
echo "  ├── nuplan-v1.1/"
echo "  │   └── mini/"
echo "  │       └── *.db"
echo "  └── maps/"
echo "      └── *.json"
echo ""
echo "STEP 4: Compress for EC2 upload"
echo "---------------------------------------"
echo ""
echo "  tar -czvf nuplan-mini.tar.gz -C ~/data nuplan"
echo ""
echo "STEP 5: Upload to EC2"
echo "---------------------------------------"
echo ""
echo "  scp -i your-key.pem nuplan-mini.tar.gz ubuntu@<EC2_IP>:~/"
echo ""
echo "============================================="
echo ""

# If data already exists, show what's there
if [ -d "$DATA_DIR" ]; then
    echo "Current contents of $DATA_DIR:"
    ls -la "$DATA_DIR" 2>/dev/null || echo "(empty)"
    echo ""
fi

# Interactive mode - ask if user wants to proceed with extraction
read -p "Have you downloaded the files? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Looking for downloaded files..."

    DOWNLOADS=~/Downloads

    if [ -f "$DOWNLOADS/nuplan-v1.1_mini.zip" ]; then
        echo "Found: nuplan-v1.1_mini.zip"
        mkdir -p "$DATA_DIR"
        unzip -o "$DOWNLOADS/nuplan-v1.1_mini.zip" -d "$DATA_DIR/"
    else
        echo "Not found: $DOWNLOADS/nuplan-v1.1_mini.zip"
    fi

    if [ -f "$DOWNLOADS/nuplan-maps-v1.1.zip" ]; then
        echo "Found: nuplan-maps-v1.1.zip"
        mkdir -p "$DATA_DIR"
        unzip -o "$DOWNLOADS/nuplan-maps-v1.1.zip" -d "$DATA_DIR/"
    else
        echo "Not found: $DOWNLOADS/nuplan-maps-v1.1.zip"
    fi

    echo ""
    echo "Done! Contents of $DATA_DIR:"
    ls -la "$DATA_DIR" 2>/dev/null
fi
