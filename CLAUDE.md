# Stable Diffusion WebUI Forge - Setup Guide

## Overview

This is a **Stable Diffusion WebUI Forge** installation configured for **NVIDIA RTX 5070 Ti** (Blackwell architecture) with:

- **PyTorch 2.9.0 with CUDA 12.8** support
- **Python 3.10.16** (managed via asdf)
- **16GB VRAM** optimized configuration
- **Automatic Google Drive sync** (hourly via rclone bisync)

## Quick Start

```bash
./launch-forge.sh
```

Access at: **http://127.0.0.1:7860**

## RTX 5070 Ti Specific Configuration

The RTX 5070 Ti uses **Blackwell architecture** (compute capability sm_120) which requires:

- CUDA 12.8+ support
- PyTorch 2.6+ built with CUDA 12.8
- NVIDIA driver R56x (560+) or newer

**Critical Environment Variable:**
```bash
export TORCH_COMMAND="pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio"
```

This is set in `launch-forge.sh` to ensure correct PyTorch installation.

## System Requirements

### Python Environment (asdf)

```bash
python3 --version
cat .tool-versions
```

### System Libraries

Required for full functionality:

```bash
# For svglib/pycairo (SVG support)
sudo apt install -y libcairo2-dev pkg-config python3-dev

# Should already be installed:
# - python3-venv (for virtual environments)
# - NVIDIA driver 560+ (for RTX 5070 Ti)
```

### GPU Configuration

```
GPU: NVIDIA GeForce RTX 5070 Ti
VRAM: 15842 MB (~16GB)
CUDA: 12.8
Driver: R56x or newer
Compute Capability: sm_120 (Blackwell)
```

## Directory Structure

```
StableDiffusion/
├── .tool-versions              # asdf Python version (3.10.16)
├── CLAUDE.md                   # This file (AI Guidance)
├── AGENTS.md                   # Symlink to CLAUDE.md
├── README-RTX5070Ti.md         # Detailed RTX 5070 Ti guide
├── setup-rtx5070ti.sh          # Initial setup script
├── launch-forge.sh             # Launch script (USE THIS TO LAUNCH)
├── webui.sh                    # Original launcher (used by launch-forge.sh)
├── venv/                       # Python virtual environment (NOT synced)
├── models/
│   ├── Stable-diffusion/       # Model checkpoints (.safetensors)
│   ├── Lora/                   # LoRA models
│   ├── VAE/                    # VAE models
│   └── ControlNetPreprocessor/ # ControlNet preprocessors
├── extensions/                 # Extensions and plugins
├── extensions-builtin/         # Built-in extensions
│   ├── soft-inpainting/        # Soft inpainting (requires joblib)
│   └── sd-forge-controlnet/    # ControlNet
├── outputs/                    # Generated images (SYNCED to Google Drive)
└── repositories/               # Git submodules
```

## Development Standards

### Clean Code Techniques

Always apply these core principles when working with this codebase:

1. **DRY (Don't Repeat Yourself)**: If code is identical or very similar, extract it into a generalized function. Parameters are your friends.

2. **KISS (Keep It Simple Stupid)**: Make code so "stupid" that a 5-year-old could understand it.

3. **SRP (Single Responsibility Principle)**: Separate code into simple, well-defined, well-intentioned tasks with clear names. Prevents "spaghetti code".

4. **Avoid Hadouken IFs**: Avoid nested IFs → Solution: Early Returns and/or Switch-Cases.

5. **Avoid Negative Conditionals**: Positive conditionals reduce mental strain and make code easier to reason about.

6. **Avoid Comments**: Code should be self-documenting with intention-revealing names. If comments are necessary, explain the "why" not the "what". Use SRP and intention-revealing names as your primary tools.

7. **Intention Revealing Nomenclatures**: Use descriptive variable names that reveal intent. Use pronounceable and searchable names. Follow language, business, and team conventions.

8. **Use Vertical Formatting**: Code should read top to bottom without "jumping". Similar and dependent functions should be vertically close.

9. **Boy Scout Rule**: Always leave the codebase cleaner than you found it. Improve Clean Code whenever you touch existing code.

## Google Drive Sync

**Remote:** `gdrive:Sync/StableDiffusion`
**Local:** `/home/victorcorcos/Documents/Sync/StableDiffusion`
**Frequency:** Every hour
**Method:** rclone bisync (bidirectional sync)

**What's synced:**
- ✅ Model checkpoints (models/)
- ✅ LoRAs, VAEs, embeddings
- ✅ Generated images (outputs/)
- ✅ Extensions
- ✅ Configuration files

**What's NOT synced (filtered):**
- ❌ venv/ (Python virtual environment)
- ❌ Cache files
- ❌ Temporary files
- ❌ __pycache__/
- ❌ .git/

**Sync Commands:**
```bash
# Check sync status
systemctl --user status rclone-bisync.timer

# Manual sync
systemctl --user start rclone-bisync.service

# View logs
journalctl --user -u rclone-bisync.service --since today
```

## Getting Models

### Download Locations

1. **Civitai:** https://civitai.com (community models, huge variety)
2. **Hugging Face:** https://huggingface.co/models?pipeline_tag=text-to-image
3. **Stability AI:** https://huggingface.co/stabilityai

### Recommended Models for RTX 5070 Ti (16GB VRAM)

**SDXL (1024x1024, best quality):**
- SDXL 1.0 Base
- SDXL Turbo
- Juggernaut XL
- DreamShaper XL

**SD 1.5 (512x512, faster):**
- Realistic Vision v5.1
- DreamShaper 8
- Deliberate v2

### Installation

```bash
# Place models in:
cd ~/Documents/Sync/StableDiffusion/models/Stable-diffusion/

# They will automatically sync to Google Drive!
```

## Performance Optimization

### Recommended Settings for SDXL

```
Resolution: 1024x1024
Batch size: 1-2
Steps: 20-30
Sampler: DPM++ 2M Karras or Euler a
```

### If VRAM Issues Occur

Edit `launch-forge.sh` and uncomment:

```bash
export COMMANDLINE_ARGS="--medvram --opt-sdp-attention"
```

### For Maximum Speed

```bash
export COMMANDLINE_ARGS="--xformers"
```

Or use the cuda-malloc hint:
```bash
export COMMANDLINE_ARGS="--cuda-malloc"
```

## Troubleshooting

### "sm_120 not compatible" Error

Your venv has old PyTorch without Blackwell support.

**Fix:**
```bash
cd ~/Documents/Sync/StableDiffusion
rm -rf venv
./launch-forge.sh
```

This will reinstall PyTorch with CUDA 12.8 support.

### "CUDA out of memory"

**Solutions:**
1. Reduce image resolution
2. Reduce batch size
3. Add `--medvram` flag to COMMANDLINE_ARGS
4. Close other GPU-intensive applications

### Models Not Appearing

1. Check files are in `models/Stable-diffusion/`
2. Refresh the checkpoint dropdown in WebUI
3. Restart Forge: Ctrl+C then `./launch-forge.sh`

### Python venv Module Not Found (Again)

Make sure you're in the StableDiffusion directory where `.tool-versions` exists:

```bash
cd ~/Documents/Sync/StableDiffusion
cat .tool-versions  # Should show: python 3.10.16
python3 --version   # Should show: Python 3.10.16
```

## Useful Commands

```bash
# Check GPU status
nvidia-smi
watch -n 1 nvidia-smi  # Real-time monitoring

# Check PyTorch CUDA version
source venv/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# View Forge logs
tail -f outputs/logs/webui.log

# Check installed packages
./venv/bin/pip list

# Check dependency issues
./venv/bin/pip check

# Sync commands
systemctl --user status rclone-bisync.timer
journalctl --user -u rclone-bisync.service --since today
```

## File References

- **RTX 5070 Ti Guide:** `README-RTX5070Ti.md`
- **Setup Script:** `setup-rtx5070ti.sh`
- **Launch Script:** `launch-forge.sh` (use this to start!)
- **Python Version:** `.tool-versions`
- **Parent Sync Guide:** `../CLAUDE.md`

## Links

- **Forge Repository:** https://github.com/lllyasviel/stable-diffusion-webui-forge
- **NVIDIA Blackwell Info:** https://developer.nvidia.com/cuda-gpus
- **PyTorch CUDA 12.8:** https://download.pytorch.org/whl/cu128
- **Civitai Models:** https://civitai.com
- **Stable Diffusion Subreddit:** https://reddit.com/r/StableDiffusion

## Support

If issues occur:

1. ✅ Check this CLAUDE.md file
2. ✅ Check `README-RTX5070Ti.md`
3. ✅ Check parent `../CLAUDE.md` for sync info
4. ✅ View Forge logs in `outputs/logs/`
5. ✅ Check Forge issues: https://github.com/lllyasviel/stable-diffusion-webui-forge/issues
