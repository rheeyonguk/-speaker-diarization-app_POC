"""Layer 3/4: WhisperX transcription (faster-whisper backend) and forced alignment.

API verified against whisperx 3.8.6:
  whisperx.load_model(whisper_arch, device, device_index, compute_type, asr_options, language,
                      vad_method, vad_options, download_root, threads, ...)
  FasterWhisperPipeline.transcribe(audio, batch_size, language, chunk_size, progress_callback)
      -> {"segments": [{"text", "start", "end", "avg_logprob"}], "language"}
  whisperx.load_align_model(language_code, device, model_name=None, model_dir=None)
  whisperx.align(transcript, model, align_model_metadata, audio, device, interpolate_method,
                 return_char_alignments, progress_callback)
      -> {"segments": [{"start","end","text","words":[{"word","start","end","score"}],"avg_logprob"}],
          "word_segments": [...]}
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..config import AppConfig
from ..device import default_alignment_model, release_gpu_memory, resolve_compute_type, torch_device_string
from ..errors import ModelLoadError, UserFacingError, is_auth_error, is_cuda_oom, is_network_error
from ..logging_utils import get_logger

logger = get_logger(__name__)

Progress = Optional[Callable[[float], None]]

ASR_OOM_HINT = (
    "asr.batch_size 를 낮추거나(16→8→4), asr.compute_type=int8_float16, "
    "또는 asr.model=large-v3-turbo 로 변경하세요."
)


def _model_load_error(what: str, name: str, exc: Exception) -> UserFacingError:
    if is_cuda_oom(exc):
        from ..errors import GpuOutOfMemoryError

        return GpuOutOfMemoryError(f"{what} 로딩 중 GPU 메모리 부족: {name}", hint=ASR_OOM_HINT)
    text = str(exc)
    if "libcudnn" in text or "libcublas" in text or "cudnn" in text.lower():
        return ModelLoadError(
            f"{what} 로딩 실패 - CUDA/cuDNN 라이브러리를 찾지 못했습니다: {name}",
            hint="CTranslate2 4.x 는 CUDA 12 + cuDNN 9 가 필요합니다. README 'GPU 트러블슈팅' 참고.",
        )
    if is_network_error(exc):
        return ModelLoadError(
            f"{what} 다운로드 실패(네트워크/프록시 차단): {name}",
            hint="최초 1회는 huggingface.co 접속이 필요합니다. 인터넷 가능한 PC에서 scripts/prefetch_models.py 로 "
                 "받은 캐시를 복사하거나 paths.model_cache_dir 지정 후 HF_HUB_OFFLINE=1 로 실행하세요.",
        )
    if is_auth_error(exc):
        return ModelLoadError(f"{what} 접근 실패(존재하지 않는 모델명 또는 권한 없음): {name}",
                              hint="모델 이름(asr.model / alignment.model_name)과 HF_TOKEN 권한을 확인하세요.")
    return ModelLoadError(f"{what} 로딩 실패: {name} ({type(exc).__name__}: {text[:300]})")


class WhisperXTranscriber:
    def __init__(self, cfg: AppConfig, device: str):
        self.cfg = cfg
        self.device = device
        self.compute_type = resolve_compute_type(cfg.asr.compute_type, device)
        self.language = None if cfg.asr.language.lower() == "auto" else cfg.asr.language
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        import whisperx

        asr_options = {"beam_size": self.cfg.asr.beam_size}
        if self.cfg.asr.initial_prompt:
            asr_options["initial_prompt"] = self.cfg.asr.initial_prompt
        logger.info("Loading Whisper model %s (device=%s, compute_type=%s)", self.cfg.asr.model, self.device, self.compute_type)
        try:
            self._model = whisperx.load_model(
                self.cfg.asr.model,
                device=self.device,
                device_index=self.cfg.runtime.device_index,
                compute_type=self.compute_type,
                asr_options=asr_options,
                language=self.language,
                vad_method=self.cfg.asr.vad_method,
                vad_options={"chunk_size": self.cfg.asr.chunk_size},
                download_root=self.cfg.model_cache_dir,
                threads=self.cfg.runtime.cpu_threads,
            )
        except UserFacingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _model_load_error("Whisper 모델", self.cfg.asr.model, exc) from exc

    def transcribe(self, audio: np.ndarray, progress: Progress = None) -> dict:
        self.load()
        cb = (lambda pct: progress(pct / 100.0)) if progress else None
        result = self._model.transcribe(
            audio,
            batch_size=self.cfg.asr.batch_size,
            language=self.language,
            chunk_size=self.cfg.asr.chunk_size,
            progress_callback=cb,
        )
        for seg in result.get("segments", []):
            seg["text"] = seg.get("text", "").strip()
        return result

    def unload(self) -> None:
        self._model = None
        release_gpu_memory()


def ensure_punkt_tab(language: str) -> None:
    """WhisperX alignment splits sentences with NLTK punkt_tab and downloads it on first use.
    nltk >= 3.10 refuses that download behind an HTTP(S) proxy unless explicitly allowed, which
    would otherwise surface as an obscure LookupError in the middle of alignment."""
    import nltk
    from whisperx.utils import PUNKT_LANGUAGES

    resource = f"tokenizers/punkt_tab/{PUNKT_LANGUAGES.get(language, 'english')}/"
    try:
        nltk.data.find(resource)
        return
    except LookupError:
        pass
    try:
        nltk.download("punkt_tab", quiet=True)
        nltk.data.find(resource)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            "WhisperX 단어 정렬에 필요한 NLTK 'punkt_tab' 데이터가 없고 자동 다운로드에 실패했습니다.",
            hint="인터넷 가능한 환경에서 `python scripts/prefetch_models.py` 실행 "
                 "(사내 프록시를 신뢰하는 경우 `--allow-proxied-nltk`), 또는 nltk_data 폴더를 복사하세요.",
        ) from exc


class WhisperXAligner:
    def __init__(self, cfg: AppConfig, device: str):
        self.cfg = cfg
        self.device = torch_device_string(device, cfg.runtime.device_index)
        self._model = None
        self._metadata = None
        self._language = None
        self.model_name: Optional[str] = None

    def load(self, language: str) -> bool:
        """Returns False when WhisperX has no alignment model for ``language``."""
        if self._model is not None and self._language == language:
            return True
        import whisperx

        self.model_name = self.cfg.alignment.model_name or default_alignment_model(language)
        if self.model_name is None:
            logger.warning("No WhisperX alignment model for language '%s'; word timestamps unavailable.", language)
            return False
        ensure_punkt_tab(language)
        logger.info("Loading alignment model %s for '%s'", self.model_name, language)
        try:
            self._model, self._metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
                model_name=self.cfg.alignment.model_name,  # None -> whisperx picks its own default
                model_dir=self.cfg.model_cache_dir,
            )
        except Exception as exc:  # noqa: BLE001
            raise _model_load_error("Alignment 모델", self.model_name, exc) from exc
        self._language = language
        return True

    def align(self, transcript: dict, audio: np.ndarray, progress: Progress = None) -> dict:
        language = transcript.get("language") or "ko"
        if not transcript.get("segments"):
            return {"segments": [], "word_segments": [], "language": language, "aligned": False}
        if not self.load(language):
            out = {"segments": [dict(s, words=[]) for s in transcript["segments"]], "word_segments": []}
            out.update(language=language, aligned=False)
            return out
        import whisperx

        cb = (lambda pct: progress(pct / 100.0)) if progress else None
        aligned = whisperx.align(
            transcript["segments"],
            self._model,
            self._metadata,
            audio,
            self.device,
            interpolate_method=self.cfg.alignment.interpolate_method,
            return_char_alignments=False,
            progress_callback=cb,
        )
        aligned["language"] = language
        aligned["aligned"] = True
        return aligned

    def unload(self) -> None:
        self._model = None
        self._metadata = None
        self._language = None
        release_gpu_memory()


__all__ = ["WhisperXTranscriber", "WhisperXAligner", "ASR_OOM_HINT"]
