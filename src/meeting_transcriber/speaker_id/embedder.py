"""WeSpeaker speaker-embedding adapter.

Verified against wespeaker @ 9fecd6c (wespeaker/cli/speaker.py):
  wespeaker.load_model(name_or_dir) -> Speaker
      name: hub key in wespeaker.cli.hub.Hub.Assets (downloaded to $WESPEAKER_HOME/<name>)
      dir : local directory with avg_model.pt + config.yaml
  Speaker.set_device(str), set_vad(bool), set_resample_rate(int),
  Speaker.set_wavform_norm(bool), set_window_type(str)
  Speaker.extract_embedding_from_pcm(pcm: Tensor[1, T], sample_rate) -> Tensor[D] | None
  Speaker.cosine_similarity(e1, e2) -> (cos + 1) / 2   (CLI convenience scale; we use raw cosine)

Upstream reads files with ``torchaudio.load(normalize=wavform_norm)``. With the default
``wavform_norm=False`` a 16-bit WAV arrives as int16-scale floats, so we feed the same scale
(x * 32768) to stay bit-compatible with the official CLI/recipes.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np

from ..errors import ConfigError, ModelLoadError, is_network_error
from ..logging_utils import get_logger

logger = get_logger(__name__)

# Frontend settings that wespeaker.cli.speaker.main() applies per hub model.
_HUB_FRONTEND_SETTINGS = {
    "campplus": {"wavform_norm": True, "window_type": "povey"},
    "eres2net": {"wavform_norm": True, "window_type": "povey"},
}


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def hub_model_names() -> list[str]:
    from wespeaker.cli.hub import Hub

    return list(Hub.Assets.keys())


class WeSpeakerEmbedder:
    def __init__(self, model: str, device: str = "cpu", cache_dir: Optional[str] = None):
        self.model_ref = model
        self.device = device
        self.cache_dir = cache_dir
        self._speaker = None
        self.embedding_dim: Optional[int] = None

    # ------------------------------------------------------------------ identity
    @property
    def model_id(self) -> str:
        """Stable identifier stored in each voice profile; embeddings from different models are
        not comparable, so profiles are only matched against the model that produced them."""
        local = Path(self.model_ref).expanduser()
        if local.is_dir():
            ckpt = local / "avg_model.pt"
            h = hashlib.sha256()
            h.update((local / "config.yaml").read_bytes() if (local / "config.yaml").exists() else b"")
            h.update(str(ckpt.stat().st_size if ckpt.exists() else 0).encode())
            return f"wespeaker:local:{local.name}:{h.hexdigest()[:12]}"
        return f"wespeaker:hub:{self.model_ref}"

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        if self._speaker is not None:
            return
        local = Path(self.model_ref).expanduser()
        is_dir = local.is_dir()
        if not is_dir:
            names = hub_model_names()
            if self.model_ref not in names:
                # wespeaker's Hub calls sys.exit() on unknown names - validate first.
                raise ConfigError(
                    f"알 수 없는 WeSpeaker 모델: {self.model_ref}",
                    hint=f"허브 이름 {names} 또는 avg_model.pt + config.yaml 이 있는 로컬 디렉터리 경로를 지정하세요.",
                )
            if self.cache_dir and "WESPEAKER_HOME" not in os.environ:
                os.environ["WESPEAKER_HOME"] = str(Path(self.cache_dir) / "wespeaker")
        else:
            missing = [f for f in ("avg_model.pt", "config.yaml") if not (local / f).exists()]
            if missing:
                raise ConfigError(f"WeSpeaker 모델 디렉터리에 {missing} 가 없습니다: {local}",
                                  hint="WeSpeaker 문서대로 체크포인트를 avg_model.pt / config.yaml 로 이름을 바꾸세요.")

        import wespeaker

        logger.info("Loading WeSpeaker model %s on %s", self.model_ref, self.device)
        try:
            speaker = wespeaker.load_model(str(local) if is_dir else self.model_ref)
        except SystemExit as exc:  # defensive: upstream Hub uses sys.exit on errors
            raise ModelLoadError(f"WeSpeaker 모델 로딩 실패: {self.model_ref}") from exc
        except Exception as exc:  # noqa: BLE001
            if is_network_error(exc) or isinstance(exc, StopIteration):
                raise ModelLoadError(
                    f"WeSpeaker 사전학습 모델 다운로드 실패: {self.model_ref}",
                    hint="WeSpeaker 허브는 modelscope.cn 에서 내려받습니다. 사내망에서 차단된 경우 "
                         "README '오프라인/수동 모델 설치' 절차로 avg_model.pt + config.yaml 을 받아 "
                         "speaker_id.wespeaker_model 에 디렉터리 경로를 지정하세요.",
                ) from exc
            raise ModelLoadError(f"WeSpeaker 모델 로딩 실패: {type(exc).__name__}: {str(exc)[:300]}") from exc

        settings = _HUB_FRONTEND_SETTINGS.get(self.model_ref, {})
        if "wavform_norm" in settings:
            speaker.set_wavform_norm(settings["wavform_norm"])
        if "window_type" in settings:
            speaker.set_window_type(settings["window_type"])
        speaker.set_resample_rate(16000)
        speaker.set_vad(False)  # speech regions are selected by us (diarization / silero for enrollment)
        speaker.set_device(self.device)
        self._speaker = speaker

    def unload(self) -> None:
        self._speaker = None

    # ------------------------------------------------------------------ inference
    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """L2-normalized embedding of a float32 mono waveform in [-1, 1]."""
        self.load()
        import torch

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size < int(0.1 * sample_rate):
            return None
        scale = 1.0 if getattr(self._speaker, "wavform_norm", False) else 32768.0
        pcm = torch.from_numpy(np.ascontiguousarray(audio * scale))[None, :]
        emb = self._speaker.extract_embedding_from_pcm(pcm, sample_rate)
        if emb is None:
            return None
        vec = emb.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if not np.all(np.isfinite(vec)):
            return None
        self.embedding_dim = int(vec.shape[0])
        return l2_normalize(vec)


# ---------------------------------------------------------------------- VAD for enrollment

_silero = None


def speech_only(audio: np.ndarray, sample_rate: int = 16000, min_silence_ms: int = 300) -> np.ndarray:
    """Concatenate speech regions found by Silero VAD (bundled with the ``silero-vad`` package,
    no download). Applied to float audio in [-1, 1] - the scale Silero expects."""
    global _silero
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    if _silero is None:
        _silero = load_silero_vad()
    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    stamps = get_speech_timestamps(wav, _silero, sampling_rate=sample_rate,
                                   min_silence_duration_ms=min_silence_ms, return_seconds=False)
    if not stamps:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([audio[s["start"]:s["end"]] for s in stamps]).astype(np.float32)
