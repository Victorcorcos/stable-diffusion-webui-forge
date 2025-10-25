# Stable Diffusion WebUI Forge - RTX 5070 Ti Configuration

## Quick Start

This installation is configured for **NVIDIA RTX 5070 Ti** (Blackwell GPU) with proper CUDA 12.8+ support.

### First Time Setup

```bash
cd ~/Documents/Sync/StableDiffusion
bash setup-rtx5070ti.sh
```

### Launch Forge

```bash
cd ~/Documents/Sync/StableDiffusion
./launch-forge.sh
```

Then open browser to: http://127.0.0.1:7860

## Why Special Configuration is Needed

The RTX 5070 Ti uses NVIDIA's **Blackwell architecture** (compute capability sm_120), which requires:

- **CUDA 12.8+** for native support
- **PyTorch 2.6+** built with CUDA 12.8
- **NVIDIA driver R56x (560+)** or newer

Without these, you'll get: `sm_120 not compatible` errors.

## What the Setup Does

The `setup-rtx5070ti.sh` script:

1. ✓ Checks your NVIDIA driver version
2. ✓ Detects your GPU
3. ✓ Removes old venv (if exists)
4. ✓ Sets `TORCH_COMMAND` for CUDA 12.8+ PyTorch
5. ✓ Creates `launch-forge.sh` with proper environment

## Directory Structure

```
StableDiffusion/
├── setup-rtx5070ti.sh      # Setup script (run once)
├── launch-forge.sh          # Launch script (use this to start)
├── webui.sh                 # Original launch script (don't use directly)
├── models/
│   ├── Stable-diffusion/   # Put your .safetensors checkpoints here
│   ├── Lora/                # LoRA models
│   ├── VAE/                 # VAE models
│   └── embeddings/          # Textual inversions
├── extensions/              # Custom extensions
├── outputs/                 # Generated images (synced to Google Drive!)
└── venv/                    # Python environment (NOT synced)
```

## Getting Model Checkpoints

### Recommended Models for RTX 5070 Ti (16GB VRAM)

**SDXL (best quality, needs more VRAM):**
- SDXL 1.0 Base
- SDXL Turbo (faster)
- Juggernaut XL

**SD 1.5 (faster, less VRAM):**
- Realistic Vision
- DreamShaper
- Deliberate

### Where to Download

1. **Civitai** (community models): https://civitai.com
2. **Hugging Face**: https://huggingface.co/models
3. **Stability AI**: https://huggingface.co/stabilityai

### How to Install

```bash
cd ~/Documents/Sync/StableDiffusion/models/Stable-diffusion/

# Download example (using wget or browser)
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```

Models will automatically sync to Google Drive!

## Sync Information

Your entire Stable Diffusion setup syncs to **Google Drive** every hour.

**What's synced:**
- ✓ Model checkpoints
- ✓ LoRAs, VAEs, embeddings
- ✓ Generated images (outputs/)
- ✓ Extensions
- ✓ Configs

**What's NOT synced (filtered out):**
- ✗ venv/ (Python environment)
- ✗ Cache files
- ✗ Temporary files
- ✗ Log files
- ✗ __pycache__/

## Performance Tips for RTX 5070 Ti

Your GPU has **16GB VRAM** - excellent for AI art!

### Optimal settings for SDXL:
```
Resolution: 1024x1024
Batch size: 1-2
Steps: 20-30
```

### If you run out of VRAM:
Add these to launch command in `launch-forge.sh`:
```bash
export COMMANDLINE_ARGS="--medvram --opt-sdp-attention"
```

### For maximum speed (if available):
```bash
export COMMANDLINE_ARGS="--xformers"
```

## Troubleshooting

### "sm_120 not compatible" error

Your venv has old PyTorch. Fix:
```bash
cd ~/Documents/Sync/StableDiffusion
rm -rf venv
./launch-forge.sh
```

### "CUDA out of memory"

Reduce image size or batch size, or add to `launch-forge.sh`:
```bash
export COMMANDLINE_ARGS="--medvram"
```

### Sync taking too long

- Large models (5GB+) take time to upload
- Check sync status: `systemctl --user status rclone-bisync.service`
- View logs: `journalctl --user -u rclone-bisync.service -f`

### Models not appearing in UI

- Refresh the checkpoint dropdown
- Check files are in: `models/Stable-diffusion/`
- Restart Forge

## Useful Commands

```bash
# Check GPU usage in real-time
nvidia-smi -l 1

# Check VRAM usage
watch -n 1 nvidia-smi

# Check PyTorch CUDA version (from venv)
source venv/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# View Forge logs
tail -f outputs/logs/webui.log

# Check sync status
systemctl --user status rclone-bisync.timer
journalctl --user -u rclone-bisync.service --since today
```

## References

- Forge GitHub: https://github.com/lllyasviel/stable-diffusion-webui-forge
- NVIDIA Blackwell: https://developer.nvidia.com/cuda-gpus
- PyTorch CUDA 12.8: https://download.pytorch.org/whl/cu128
- Civitai Models: https://civitai.com
- Stable Diffusion Reddit: https://reddit.com/r/StableDiffusion

## Support

If you encounter issues:

1. Check this README
2. Check `CLAUDE.md` in parent directory
3. View Forge logs: `outputs/logs/`
4. Check sync logs: `journalctl --user -u rclone-bisync.service`
5. Forge Issues: https://github.com/lllyasviel/stable-diffusion-webui-forge/issues
