"""Device selection, GPU memory hygiene and environment diagnostics."""

from __future__ import annotations

import gc
import platform
import shutil
import sys
from contextlib import contextmanager
from typing import Iterator

from .config import AppConfig, hf_token
from .errors import ConfigError, GpuOutOfMemoryError, is_cuda_oom
from .logging_utils import get_logger

logger = get_logger(__name__)


def resolve_device(requested: str) -> str:
    import torch

    requested = (requested or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigError(
            "CUDA 디바이스가 요청되었지만 사용할 수 없습니다.",
            hint="NVIDIA 드라이버/CUDA용 PyTorch 설치를 확인하거나 MT_RUNTIME__DEVICE=cpu 로 실행하세요.",
        )
    return requested


def torch_device_string(device: str, index: int) -> str:
    return f"cuda:{index}" if device == "cuda" else device


def resolve_compute_type(compute_type: str, device: str) -> str:
    """'auto' -> float16 on CUDA, int8 on CPU (CTranslate2 float16 is not supported on CPU)."""
    ct = (compute_type or "auto").lower()
    if ct in ("auto", "default"):
        return "float16" if device == "cuda" else "int8"
    if device == "cpu" and ct in ("float16", "int8_float16"):
        logger.warning("compute_type=%s 는 CPU 에서 지원되지 않아 int8 로 대체합니다.", ct)
        return "int8"
    return ct


def release_gpu_memory() -> None:
    """Called once at the end of a stage after the stage's model references were dropped."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


@contextmanager
def oom_guard(stage: str, hint: str) -> Iterator[None]:
    """Convert CUDA OOM (torch or CTranslate2) into a user-facing error and free the cache."""
    try:
        yield
    except GpuOutOfMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 - classify then re-raise
        if is_cuda_oom(exc):
            release_gpu_memory()
            raise GpuOutOfMemoryError("GPU 메모리가 부족합니다 (CUDA out of memory).", hint=hint, stage=stage) from exc
        raise


def gpu_memory_snapshot() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        return {
            "gpu_memory_total_gb": round(total / 1024**3, 2),
            "gpu_memory_free_gb": round(free / 1024**3, 2),
            "torch_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "torch_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        }
    except Exception:
        return {}


def collect_diagnostics(cfg: AppConfig) -> dict:
    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ffmpeg": shutil.which("ffmpeg") or "NOT FOUND",
        "ffprobe": shutil.which("ffprobe") or "NOT FOUND",
        "hf_token": "set" if hf_token() else "NOT SET",  # never the value
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda or "cpu-only build"
        info["cuda_available"] = torch.cuda.is_available()
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            idx = cfg.runtime.device_index
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(idx)
            cap = torch.cuda.get_device_capability(idx)
            info["gpu_compute_capability"] = f"{cap[0]}.{cap[1]}"
            info.update(gpu_memory_snapshot())
        else:
            info["gpu_name"] = None
        try:
            info["device"] = resolve_device(cfg.runtime.device)
        except ConfigError as exc:
            info["device"] = f"ERROR: {exc.message}"
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = repr(exc)

    for mod_name, key in (
        ("torchaudio", "torchaudio_version"),
        ("ctranslate2", "ctranslate2_version"),
        ("faster_whisper", "faster_whisper_version"),
        ("whisperx", "whisperx_version"),
        ("pyannote.audio", "pyannote_audio_version"),
        ("gradio", "gradio_version"),
    ):
        info[key] = _module_version(mod_name)
    info["wespeaker_version"] = _dist_version("wespeaker")
    try:
        import ctranslate2

        info["ctranslate2_cuda_devices"] = ctranslate2.get_cuda_device_count()
    except Exception:
        info["ctranslate2_cuda_devices"] = None

    device = info.get("device") if isinstance(info.get("device"), str) else "cpu"
    info["whisper_model"] = cfg.asr.model
    info["compute_type"] = resolve_compute_type(cfg.asr.compute_type, device if device in ("cuda", "cpu") else "cpu")
    info["pyannote_model"] = cfg.diarization.model
    info["wespeaker_model"] = cfg.speaker_id.wespeaker_model
    info["alignment_model_ko"] = default_alignment_model("ko")
    if device == "cpu":
        info["cpu_note"] = (
            "CPU 모드: large-v3 전사는 실시간의 수 배 이상 소요될 수 있습니다. "
            "실사용은 NVIDIA GPU 를 권장하며, CPU 에서는 large-v3-turbo + int8 을 권장합니다."
        )
    return info


def default_alignment_model(language: str) -> str | None:
    """The alignment model WhisperX itself would choose (read from the installed whisperx)."""
    try:
        from whisperx import alignment

        return alignment.DEFAULT_ALIGN_MODELS_TORCH.get(language) or alignment.DEFAULT_ALIGN_MODELS_HF.get(language)
    except Exception:
        return None


def _module_version(name: str) -> str | None:
    try:
        import importlib

        mod = importlib.import_module(name)
        return getattr(mod, "__version__", None) or _dist_version(name.replace(".", "-"))
    except Exception:
        return None


def _dist_version(dist: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:
        return None
