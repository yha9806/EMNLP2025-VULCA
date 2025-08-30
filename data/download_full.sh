#!/bin/bash
# Download full VULCA dataset

echo "========================================="
echo "VULCA Full Dataset Download Script"
echo "========================================="

# Check if we're in the right directory
if [ ! -f "../README.md" ]; then
    echo "Error: Please run this script from the data/ directory"
    exit 1
fi

echo ""
echo "This script will download the complete VULCA dataset including:"
echo "- High-resolution Qing Dynasty 'Twelve Months' paintings"
echo "- Complete MLLM evaluation outputs (1.2MB)"
echo "- Human expert annotations and benchmarks"
echo "- Pre-processed image slices"
echo ""
echo "Total download size: ~200MB"
echo "Required disk space: ~500MB"
echo ""

read -p "Do you want to continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

echo ""
echo "Downloading from GitHub Release..."

# Option 1: Download from GitHub Release
RELEASE_URL="https://github.com/yha9806/EMNLP2025-VULCA/releases/download/v1.0/vulca_full_dataset.zip"

if command -v wget &> /dev/null; then
    wget -O vulca_full_dataset.zip "$RELEASE_URL"
elif command -v curl &> /dev/null; then
    curl -L -o vulca_full_dataset.zip "$RELEASE_URL"
else
    echo "Error: Please install wget or curl to download the dataset"
    exit 1
fi

# Check if download was successful
if [ ! -f "vulca_full_dataset.zip" ]; then
    echo ""
    echo "Primary download failed. Trying alternative source..."
    
    # Option 2: Alternative download from Zenodo (with DOI)
    ZENODO_URL="https://zenodo.org/record/XXXXXX/files/vulca_dataset.zip"
    echo "Note: Zenodo DOI will be available after paper acceptance"
    echo "For now, please contact authors for full dataset access"
    exit 1
fi

echo ""
echo "Extracting dataset..."

# Extract the zip file
if command -v unzip &> /dev/null; then
    unzip -q vulca_full_dataset.zip
else
    echo "Error: Please install unzip to extract the dataset"
    exit 1
fi

# Organize the extracted files
echo "Organizing files..."

# Create necessary directories
mkdir -p paintings
mkdir -p mllm_outputs
mkdir -p human_annotations
mkdir -p processed_slices

# Move files to appropriate locations
if [ -d "vulca_dataset/paintings" ]; then
    mv vulca_dataset/paintings/* paintings/
fi

if [ -d "vulca_dataset/mllm_outputs" ]; then
    mv vulca_dataset/mllm_outputs/* mllm_outputs/
fi

if [ -d "vulca_dataset/human_annotations" ]; then
    mv vulca_dataset/human_annotations/* human_annotations/
fi

if [ -d "vulca_dataset/processed_slices" ]; then
    mv vulca_dataset/processed_slices/* processed_slices/
fi

# Clean up
rm -rf vulca_dataset
rm vulca_full_dataset.zip

echo ""
echo "========================================="
echo "Dataset download complete!"
echo "========================================="
echo ""
echo "Dataset structure:"
echo "  data/"
echo "  ├── paintings/          # Original paintings"
echo "  ├── mllm_outputs/       # MLLM evaluation results"
echo "  ├── human_annotations/  # Expert benchmarks"
echo "  └── processed_slices/   # Pre-processed patches"
echo ""
echo "To use the full dataset, update your configuration:"
echo "  - Set data_dir in configs/experiment.yaml"
echo "  - Or use --data-dir flag when running experiments"
echo ""
echo "For more information, see the README in the research branch:"
echo "  git checkout research"
echo "  cat README_research.md"
echo ""