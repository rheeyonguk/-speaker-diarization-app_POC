"""Layer 5/6: pyannote speaker-diarization-community-1.

Verified against pyannote.audio 4.0.7:
  Pipeline.from_pretrained(checkpoint, revision=None, hparams_file=None, subfolder=None,
                           token=None, cache_dir=None) -> Pipeline | None
  SpeakerDiarization.apply(file, num_speakers=None, min_speakers=None, max_speakers=None, hook=None)
      -> DiarizeOutput(speaker_diarization: Annotation,
                       exclusive_speaker_diarization: Annotation,
                       speaker_embeddings: np.ndarray | None)
     (or a bare Annotation when the pipeline was built with legacy=True)
Both annotations are kept: the overlap-aware one for overlap flags / cluster statistics, the
exclusive one for single-speaker attribution of transcript words.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from ..config import AppConfig, hf_token
from ..device import release_gpu_memory, torch_device_string
from ..errors import AuthError, ModelLoadError, is_auth_error, is_network_error
from ..logging_utils import get_logger
from .types import DiarizationResult, Turn

logger = get_logger(__name__)

Progress = Optional[Callable[[float], None]]

DIARIZATION_OOM_HINT = (
    "diarization.embedding_batch_size / segmentation_batch_size 를 낮추거나(예: 8), "
    "runtime.release_models_between_stages=true 인지 확인하세요."
)

_GATED_HINT = (
    "1) https://hf.co/pyannote/speaker-diarization-community-1 에서 사용 약관(user conditions)에 동의\n"
    "   2) https://hf.co/settings/tokens 에서 read 토큰 발급\n"
    "   3) 프로젝트 루트 .env 에 HF_TOKEN=hf_... 저장 후 재실행"
)


def annotation_to_turns(annotation: Any) -> list[Turn]:
    return [
        Turn(float(segment.start), float(segment.end), str(label))
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]


def from_pyannote_output(output: Any, params: dict | None = None, model: str = "") -> DiarizationResult:
    """Convert pyannote 4.x ``DiarizeOutput`` (or a legacy ``Annotation``) into ``DiarizationResult``."""
    if hasattr(output, "speaker_diarization"):
        full = annotation_to_turns(output.speaker_diarization)
        exclusive_ann = getattr(output, "exclusive_speaker_diarization", None)
        if exclusive_ann is not None:
            return DiarizationResult(full, annotation_to_turns(exclusive_ann), True, params or {}, model)
        return DiarizationResult(full, _exclusive_from_full(full), False, params or {}, model)
    if hasattr(output, "itertracks"):  # legacy Annotation
        full = annotation_to_turns(output)
        return DiarizationResult(full, _exclusive_from_full(full), False, params or {}, model)
    raise TypeError(f"Unsupported diarization output type: {type(output)!r}")


def _exclusive_from_full(full: list[Turn]) -> list[Turn]:
    """Fallback only (legacy pipelines): in overlaps keep the speaker with the longer turn, i.e. a
    short interjection is dropped. Community-1 provides the real exclusive diarization."""
    if not full:
        return []
    bounds = sorted({t.start for t in full} | {t.end for t in full})
    out: list[Turn] = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s <= 1e-6:
            continue
        mid = (s + e) / 2
        active = [t for t in full if t.start <= mid < t.end]
        if not active:
            continue
        best = max(active, key=lambda t: t.duration)
        if out and out[-1].speaker == best.speaker and abs(out[-1].end - s) < 1e-6:
            out[-1] = Turn(out[-1].start, e, best.speaker)
        else:
            out.append(Turn(s, e, best.speaker))
    return out


class PyannoteDiarizer:
    def __init__(self, cfg: AppConfig, device: str):
        self.cfg = cfg
        self.device = torch_device_string(device, cfg.runtime.device_index)
        self._pipeline = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        import torch
        from pyannote.audio import Pipeline

        model = self.cfg.diarization.model
        token = hf_token()
        logger.info("Loading diarization pipeline %s (HF token %s)", model, "set" if token else "not set")
        try:
            pipeline = Pipeline.from_pretrained(model, token=token, cache_dir=self.cfg.model_cache_dir)
        except Exception as exc:  # noqa: BLE001
            if is_network_error(exc):
                raise ModelLoadError(
                    f"pyannote 모델 다운로드 실패(네트워크/프록시 차단): {model}",
                    hint="최초 1회 huggingface.co 접속이 필요합니다. 캐시 후에는 HF_HUB_OFFLINE=1 로 오프라인 사용 가능.",
                ) from exc
            if is_auth_error(exc):
                raise AuthError(
                    f"pyannote 모델 접근 권한이 없습니다: {model}",
                    hint=_GATED_HINT if token else "HF_TOKEN 이 설정되지 않았습니다.\n   " + _GATED_HINT,
                ) from exc
            raise ModelLoadError(f"pyannote 파이프라인 로딩 실패: {type(exc).__name__}: {str(exc)[:300]}") from exc
        if pipeline is None:  # pyannote returns None when the config could not be downloaded
            raise AuthError(
                f"pyannote 파이프라인을 내려받지 못했습니다: {model}",
                hint=_GATED_HINT,
            )
        pipeline.to(torch.device(self.device))
        if self.cfg.diarization.embedding_batch_size:
            pipeline.embedding_batch_size = int(self.cfg.diarization.embedding_batch_size)
        if self.cfg.diarization.segmentation_batch_size:
            pipeline.segmentation_batch_size = int(self.cfg.diarization.segmentation_batch_size)
        self._pipeline = pipeline

    def diarize(self, audio: np.ndarray, sample_rate: int, speaker_kwargs: dict, progress: Progress = None) -> DiarizationResult:
        self.load()
        import torch

        waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None, :]
        file = {"waveform": waveform, "sample_rate": sample_rate, "uri": "meeting"}

        hook = None
        if progress is not None:
            ranges = {"segmentation": (0.0, 0.5), "embeddings": (0.5, 0.99)}
            last = [0.0]

            def hook(step_name, step_artifact, file=None, total=None, completed=None):  # noqa: ARG001
                if total and completed is not None:
                    lo, hi = ranges.get(step_name, (0.0, 0.99))
                    frac = lo + min(completed / total, 1.0) * (hi - lo)
                    if frac > last[0]:
                        last[0] = frac
                        progress(frac)

        logger.info("Running diarization with %s", speaker_kwargs or "AUTO speaker count")
        output = self._pipeline(file, hook=hook, **speaker_kwargs) if hook else self._pipeline(file, **speaker_kwargs)
        if progress:
            progress(1.0)
        params = {"mode_kwargs": dict(speaker_kwargs)}
        return from_pyannote_output(output, params=params, model=self.cfg.diarization.model)

    def unload(self) -> None:
        self._pipeline = None
        release_gpu_memory()


__all__ = [
    "DIARIZATION_OOM_HINT",
    "PyannoteDiarizer",
    "annotation_to_turns",
    "from_pyannote_output",
]
