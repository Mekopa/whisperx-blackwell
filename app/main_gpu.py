"""
WhisperX Batch Processing Service for NVIDIA DGX Spark (Blackwell GPU)
GPU DIARIZATION VERSION - Uses SM_90 spoof for full GPU acceleration

This service provides:
- Perfect transcription (Whisper large-v3)
- Word-level timestamps (Wav2Vec2 alignment)
- Speaker diarization (pyannote.audio) - GPU ACCELERATED!

Built from source to support ARM64 + Blackwell (SM_121) architecture.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager
import torch
import re

# Patch 1: PyTorch 2.6+ changed torch.load default to weights_only=True
# Pyannote models were saved with older PyTorch and need weights_only=False
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Patch 2: NVIDIA's torch version (2.6.0a0+ecf3bae40a) isn't valid semver
# Pyannote.audio tries to parse it and fails. Patch semver to handle this.
import semver.version
_original_semver_parse = semver.version.VersionInfo.parse
@classmethod
def _patched_semver_parse(cls, version):
    # Convert NVIDIA version format to valid semver
    # e.g., "2.6.0a0+ecf3bae40a" -> "2.6.0-alpha.0+ecf3bae40a"
    if isinstance(version, str) and 'a0+' in version:
        version = re.sub(r'(\d+\.\d+\.\d+)a0\+', r'\1-alpha.0+', version)
    elif isinstance(version, str) and 'a0' in version:
        version = re.sub(r'(\d+\.\d+\.\d+)a0', r'\1-alpha.0', version)
    return _original_semver_parse.__func__(cls, version)
semver.version.VersionInfo.parse = _patched_semver_parse

# Patch 3: torchaudio nightly removed APIs that pyannote.audio 3.3.2 still uses
# Create dummy implementations for compatibility
import torchaudio
from typing import NamedTuple

class AudioMetaData(NamedTuple):
    sample_rate: int
    num_frames: int
    num_channels: int

def list_audio_backends():
    return ["ffmpeg", "sox", "sox_io"]

torchaudio.AudioMetaData = AudioMetaData
torchaudio.list_audio_backends = list_audio_backends

import whisperx
import tempfile
import os
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global models
whisperx_model = None
align_model = None
diarize_pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    global whisperx_model

    logger.info("🚀 Loading WhisperX models (GPU DIARIZATION ENABLED)...")
    logger.info(f"   Device: {device}")
    logger.info(f"   Compute type: {compute_type}")
    logger.info(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        # Log the spoofed capability
        cap = torch.cuda.get_device_capability(0)
        logger.info(f"   Compute Capability (reported): SM_{cap[0]}{cap[1]}")

    # Load Whisper model
    logger.info("📦 Loading Whisper large-v3...")
    whisperx_model = whisperx.load_model(
        "large-v3",
        device=device,
        compute_type=compute_type
    )

    logger.info("✅ WhisperX ready for GPU-accelerated batch processing!")

    yield

    logger.info("👋 Shutting down WhisperX service...")


app = FastAPI(
    title="WhisperX Batch Processing Service (GPU)",
    description="Perfect transcription with GPU-accelerated speaker diarization. Built for NVIDIA DGX Spark (Blackwell).",
    version="1.1.0-gpu",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Health check"""
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    return {
        "status": "healthy" if whisperx_model else "loading",
        "service": "whisperx-batch-gpu",
        "device": device,
        "diarization_device": device,  # GPU diarization enabled!
        "model": "whisper-large-v3",
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": f"SM_{cap[0]}{cap[1]}"
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    num_speakers: int = Form(None),
    min_speakers: int = Form(None),
    max_speakers: int = Form(None)
):
    """
    Transcribe audio file with GPU-accelerated speaker diarization.

    Args:
        file: Audio file (WAV, MP3, M4A, etc.)
        language: Language code or "auto" for detection
        num_speakers: Exact number of speakers (if known)
        min_speakers: Minimum speakers (if uncertain)
        max_speakers: Maximum speakers (if uncertain)

    Returns:
        Transcript with speaker labels and word-level timestamps
    """
    logger.info(f"📥 Received file: {file.filename}")

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Step 1: Transcribe with WhisperX
        logger.info("🎤 Step 1: Transcribing audio...")
        result = whisperx_model.transcribe(
            tmp_path,
            batch_size=16,
            language=None if language == "auto" else language
        )

        language_detected = result["language"]
        logger.info(f"   Detected language: {language_detected}")

        # Step 2: Align (word-level timestamps)
        logger.info("⏱️  Step 2: Aligning word timestamps...")
        global align_model
        if align_model is None or align_model[1] != language_detected:
            align_model = whisperx.load_align_model(
                language_code=language_detected,
                device=device
            )

        result = whisperx.align(
            result["segments"],
            align_model[0],
            align_model[1],
            tmp_path,
            device,
            return_char_alignments=False
        )

        # Step 3: Diarization (speaker identification) - GPU ACCELERATED!
        diarize_device = device  # Use GPU!
        logger.info(f"👥 Step 3: Identifying speakers (device: {diarize_device}) - GPU ACCELERATED!")
        global diarize_pipeline
        if diarize_pipeline is None:
            diarize_pipeline = whisperx.diarize.DiarizationPipeline(
                token=os.getenv("HF_TOKEN"),
                device=diarize_device
            )

        # Run diarization
        diarize_segments = diarize_pipeline(
            tmp_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        # Step 4: Assign speakers to words
        logger.info("🔗 Step 4: Assigning speakers to words...")
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Format response
        logger.info("✅ Processing complete!")

        # Extract speaker stats
        speakers = {}
        for segment in result["segments"]:
            speaker = segment.get("speaker", "UNKNOWN")
            if speaker not in speakers:
                speakers[speaker] = {"duration": 0, "segments": 0}
            speakers[speaker]["duration"] += segment["end"] - segment["start"]
            speakers[speaker]["segments"] += 1

        return {
            "status": "success",
            "language": language_detected,
            "segments": result["segments"],
            "word_segments": result.get("word_segments", []),
            "speakers": speakers,
            "num_speakers": len(speakers),
            "diarization_device": diarize_device  # Confirm GPU was used
        }

    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        # Free GPU memory
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


@app.get("/")
async def root():
    """API info"""
    return {
        "service": "WhisperX Batch Processing (GPU)",
        "version": "1.1.0-gpu",
        "platform": "NVIDIA DGX Spark (Blackwell)",
        "endpoint": "POST /transcribe",
        "features": [
            "Perfect transcription (Whisper large-v3)",
            "Word-level timestamps (Wav2Vec2 alignment)",
            "Speaker diarization (pyannote.audio) - GPU ACCELERATED",
            "Full GPU acceleration (Blackwell SM_121 → SM_90 spoof)"
        ]
    }
