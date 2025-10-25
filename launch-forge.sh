#!/bin/bash
# Launch script for Stable Diffusion WebUI Forge with RTX 5070 Ti support

# Set PyTorch installation command for CUDA 12.8+
export TORCH_COMMAND="pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio"

# Optional: Uncomment to enable specific optimizations
# export COMMANDLINE_ARGS="--xformers --medvram --opt-sdp-attention"

# Launch Forge
bash webui.sh "$@"
