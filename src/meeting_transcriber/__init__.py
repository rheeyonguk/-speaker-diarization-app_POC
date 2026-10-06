"""Local Korean multi-speaker meeting transcription (WhisperX + pyannote Community-1 + WeSpeaker)."""

import os

# Privacy defaults, applied before any third-party ML library is imported.
# - pyannote.audio 4.x reads PYANNOTE_METRICS_ENABLED at import time (usage telemetry).
# - huggingface_hub / gradio send anonymous usage statistics unless disabled.
# setdefault: an explicit user choice in the environment always wins.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

__version__ = "0.1.0"
