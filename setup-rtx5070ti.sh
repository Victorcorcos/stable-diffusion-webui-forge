#!/bin/bash
################################################################################
# Stable Diffusion WebUI Forge - RTX 5070 Ti Setup Script
#
# This script configures Forge to work with NVIDIA RTX 5070 Ti (Blackwell GPU)
# by installing PyTorch with CUDA 12.8+ support
#
# Requirements:
# - NVIDIA driver R56x or newer (for Blackwell support)
# - Linux Mint 22.2 or compatible Ubuntu-based system
#
# Usage:
#   bash setup-rtx5070ti.sh
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "RTX 5070 Ti Setup for Forge"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [[ ! -f "webui.sh" ]]; then
    echo "ERROR: webui.sh not found. Are you in the Forge directory?"
    exit 1
fi

# Check NVIDIA driver
echo "[1/5] Checking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "✓ NVIDIA driver found: $DRIVER_VERSION"

    # Check if driver version is sufficient (550+)
    DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
    if [[ $DRIVER_MAJOR -lt 550 ]]; then
        echo "⚠ WARNING: Driver version $DRIVER_VERSION may be too old for RTX 5070 Ti"
        echo "  Blackwell GPUs require driver R56x (560+) or newer"
        echo "  Update your driver: https://www.nvidia.com/download/index.aspx"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "✗ ERROR: nvidia-smi not found. Is the NVIDIA driver installed?"
    echo "  Install driver: sudo ubuntu-drivers autoinstall"
    exit 1
fi

# Check GPU
echo ""
echo "[2/5] Checking GPU..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "✓ GPU detected: $GPU_NAME"

if [[ ! "$GPU_NAME" =~ "RTX 50" ]]; then
    echo "⚠ WARNING: This script is optimized for RTX 5070 Ti (Blackwell)"
    echo "  Your GPU: $GPU_NAME"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check CUDA
echo ""
echo "[3/5] Checking CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "✓ CUDA toolkit found: $CUDA_VERSION"
else
    echo "ℹ CUDA toolkit not found in PATH (this is OK)"
    echo "  PyTorch will use its bundled CUDA runtime"
fi

# Remove existing venv if it exists
echo ""
echo "[4/5] Setting up Python environment..."
if [[ -d "venv" ]]; then
    echo "⚠ Existing venv found. Removing to install correct PyTorch version..."
    rm -rf venv
    echo "✓ Old venv removed"
fi

# Set environment variable for CUDA 12.8+ PyTorch
export TORCH_COMMAND="pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio"
echo "✓ TORCH_COMMAND configured for CUDA 12.8+"
echo "  This ensures RTX 5070 Ti (sm_120) compatibility"

# Create launch script with proper environment
echo ""
echo "[5/5] Creating launch script..."
cat > launch-forge.sh << 'EOF'
#!/bin/bash
# Launch script for Stable Diffusion WebUI Forge with RTX 5070 Ti support

# Set PyTorch installation command for CUDA 12.8+
export TORCH_COMMAND="pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio"

# Optional: Uncomment to enable specific optimizations
# export COMMANDLINE_ARGS="--xformers --medvram --opt-sdp-attention"

# Launch Forge
bash webui.sh "$@"
EOF

chmod +x launch-forge.sh
echo "✓ launch-forge.sh created"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Launch Forge using the custom script:"
echo "   cd $SCRIPT_DIR"
echo "   ./launch-forge.sh"
echo ""
echo "2. First launch will:"
echo "   - Create a new venv"
echo "   - Install PyTorch 2.6+ with CUDA 12.8"
echo "   - Download and install all dependencies"
echo "   - This may take 10-20 minutes"
echo ""
echo "3. Download a model checkpoint:"
echo "   - Place .safetensors or .ckpt files in: models/Stable-diffusion/"
echo "   - Recommended: SDXL 1.0 or SD 1.5"
echo "   - Download from: https://civitai.com or https://huggingface.co"
echo ""
echo "4. Access the WebUI:"
echo "   - Open browser to: http://127.0.0.1:7860"
echo ""
echo "Important notes:"
echo "- Your RTX 5070 Ti has 16GB VRAM - perfect for SDXL!"
echo "- All files will sync to Google Drive hourly"
echo "- venv folder is excluded from sync (filters configured)"
echo ""
echo "Troubleshooting:"
echo "- If you get 'sm_120 not compatible' error, venv wasn't rebuilt"
echo "- Run: rm -rf venv && ./launch-forge.sh"
echo "- Check logs in: outputs/logs/"
echo ""
echo "Happy generating! 🎨"
echo ""
