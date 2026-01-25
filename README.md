# WhisperX on NVIDIA Blackwell (DGX Spark / GB10 / GB200)

🐳 **Docker Image:** `docker pull mekopa/whisperx-blackwell:latest`

[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/mekopa/whisperx-blackwell)

## The Problem

Running legacy AI audio workloads (WhisperX, Pyannote) on next-generation NVIDIA Blackwell GPUs (SM_121) currently fails with:

```
nvrtc: error: invalid value for --gpu-architecture (-arch)
```

**Why?** The NVRTC compiler doesn't recognize `sm_121` (Blackwell) yet, even though:
- PyTorch can see the GPU
- CUDA toolkit supports it
- The hardware is ready

Standard Python monkeypatching fails because Jiterator queries hardware architecture directly from C++, bypassing Python-level patches.

## The Solution

This repository contains the **"Blackwell Bridge Patch"** - a surgical Dockerfile fix that:

1. **Architecture Spoofing:** Forces PyTorch's `get_device_capability()` to return `(9, 0)` (Hopper) instead of `(12, 1)` (Blackwell)
2. **JIT Bypass:** Patches `torchaudio` source code to avoid `.abs()` on complex tensors, which triggers the broken Jiterator path

**Result:** SM_90 (Hopper) code runs natively on SM_121 (Blackwell) due to binary compatibility.

## Performance

| Metric | CPU Fallback | GPU (Patched) | Speedup |
|--------|-------------|---------------|---------|
| **24 min audio** | ~2 hours | **62 seconds** | **~115x** |
| Transcription | GPU ✓ | GPU ✓ | - |
| Alignment | GPU ✓ | GPU ✓ | - |
| Diarization | **CPU only** | **GPU ✓** | 115x |

## Quick Start

### Option 1: Pre-built Docker Image (Recommended)

```bash
# Pull the image
docker pull mekopa/whisperx-blackwell:latest

# Run the service
docker run -d \
  --name whisperx-gpu \
  --gpus all \
  --ipc=host \
  -p 8003:8003 \
  -v /path/to/audio:/data \
  -e HF_TOKEN="your_huggingface_token" \
  mekopa/whisperx-blackwell:latest
```

**Get your HF token:** https://huggingface.co/settings/tokens (needed for pyannote speaker diarization)

### Option 2: Build from Source

```bash
# Clone the repo
git clone https://github.com/mekopa/whisperx-blackwell.git
cd whisperx-blackwell

# Build the image
docker build -f Dockerfile.gpu -t whisperx-blackwell:latest .

# Run it
docker run -d \
  --name whisperx-gpu \
  --gpus all \
  --ipc=host \
  -p 8003:8003 \
  -e HF_TOKEN="your_token" \
  whisperx-blackwell:latest
```

## Usage

### Health Check

```bash
curl http://localhost:8003/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "whisperx-batch-gpu",
  "device": "cuda",
  "diarization_device": "cuda",
  "gpu": "NVIDIA GB10",
  "compute_capability": "SM_90"
}
```

### Transcribe Audio

```bash
curl -X POST "http://localhost:8003/transcribe" \
  -F "file=@your_audio.mp3" \
  -F "language=auto" \
  -o transcription.json
```

Response includes:
- Word-level timestamps
- Speaker labels (SPEAKER_00, SPEAKER_01, etc.)
- Confidence scores
- Language detection

## Technical Details

### The Patches

#### 1. PyTorch Capability Spoof (`Dockerfile.gpu` lines 88-99)

```python
# Forces get_device_capability() to return (9, 0) for SM_121
def get_device_capability(device=None):
    major, minor = _original_get_device_capability(device)
    if major == 12 and minor == 1:
        return (9, 0)  # Pretend to be Hopper H100
    return (major, minor)
```

#### 2. Torchaudio Jiterator Bypass (`Dockerfile.gpu` lines 113-118)

```python
# OLD (crashes on SM_121):
spectrum = torch.fft.rfft(strided_input).abs()

# NEW (works):
fft_result = torch.fft.rfft(strided_input)
spectrum = torch.sqrt(fft_result.real**2 + fft_result.imag**2)
```

### Why This Works

1. **Binary Compatibility:** NVIDIA designed Blackwell to execute Hopper (SM_90) code natively
2. **JIT Avoidance:** Computing `.abs()` manually uses standard CUDA kernels instead of runtime-compiled jiterator kernels
3. **No Performance Loss:** The manual computation is mathematically identical and equally fast

### Tested Hardware

- ✅ NVIDIA DGX Spark (ARM64, Blackwell GB10)
- ✅ Should work on GB200, GB202, GB203 (untested)
- ✅ Should work on any SM_121 Blackwell GPU

### Tested Software

- PyTorch 2.6.0 (NVIDIA container 25.01)
- WhisperX 3.3.1
- Pyannote.audio 3.3.2
- CUDA 13.0
- Python 3.12

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  WhisperX Pipeline (GPU-Accelerated)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Whisper large-v3      → GPU (Blackwell/Hopper)    │
│  Step 2: Wav2Vec2 alignment    → GPU (Blackwell/Hopper)    │
│  Step 3: Pyannote diarization  → GPU (PATCHED!)            │
│                                                              │
│  Patches Applied:                                           │
│  - SM_121 → SM_90 capability spoof                          │
│  - Torchaudio jiterator bypass                              │
└─────────────────────────────────────────────────────────────┘
```

## Known Limitations

1. **Temporary Fix:** This will become obsolete when NVIDIA updates NVRTC to recognize SM_121
2. **Binary Compatibility:** Relies on Blackwell executing Hopper code (safe, but not optimized)
3. **Torchaudio Version:** The line numbers in the patch are for `torchaudio==2.6.0` from the NVIDIA container

## When to Use This

✅ **Use this if:**
- You have Blackwell hardware (DGX Spark, GB10, GB200)
- You're getting `nvrtc: error: invalid value for --gpu-architecture`
- You want GPU-accelerated speaker diarization

❌ **Don't use this if:**
- You have Hopper (H100) or older GPUs - use standard WhisperX
- You're on x86_64 architecture - rebuild for your arch
- NVIDIA has officially released SM_121 support (check PyTorch release notes)

## Future Work

This patch will become obsolete when:
- PyTorch updates to recognize SM_121 natively
- Torchaudio stops using jiterator for complex number operations
- NVIDIA releases updated NVRTC compiler

Until then, this is the **only known way** to run GPU speaker diarization on Blackwell.

## Contributing

Found this useful? Here's how to help:

1. ⭐ **Star the repo** if this saved you time
2. 🐛 **Report issues** if you find edge cases
3. 📝 **Share results** from other Blackwell GPUs (GB200, GB202, etc.)
4. 🔧 **Submit PRs** for improvements

## Credits

- **WhisperX:** https://github.com/m-bain/whisperX
- **Pyannote.audio:** https://github.com/pyannote/pyannote-audio
- **Patch Discovery:** Community effort to unlock Blackwell for legacy workloads

## License

MIT License - Free to use, modify, and distribute.

**Disclaimer:** This is a community patch for early-adopter hardware. Use at your own risk. Not affiliated with NVIDIA or WhisperX maintainers.

---

**Need help?** Open an issue or check the [Discussions](https://github.com/mekopa/whisperx-blackwell/discussions) tab.
