"""Audio preprocessing: extraction/resampling (via io), normalization, optional noise-reduction hook."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..config import AudioConfig
from ..errors import AudioError
from ..logging_utils import get_logger
from .io import MediaInfo, load_audio_file, probe

logger = get_logger(__name__)

# Signature of a noise-reduction plug-in: (audio float32 mono, sample_rate) -> audio
NoiseReducer = Callable[[np.ndarray, int], np.ndarray]
_noise_reducer: Optional[NoiseReducer] = None


def register_noise_reducer(fn: Optional[NoiseReducer]) -> None:
    """Install a noise-reduction function used when ``audio.noise_reduction`` is true.

    Deliberately no built-in implementation: aggressive denoising tends to remove the spectral
    detail that speaker embeddings rely on and can introduce ASR hallucinations. Plug in a
    tested, mild method here if a recording environment really needs it.
    """
    global _noise_reducer
    _noise_reducer = fn


@dataclass
class PreparedAudio:
    audio: np.ndarray          # float32 mono, cfg.sample_rate
    sample_rate: int
    duration_sec: float
    wav_path: Path             # working copy (never the original)
    source_info: MediaInfo
    peak_dbfs_before: float
    rms_dbfs: float
    gain_db: float
    notes: list = field(default_factory=list)


def dbfs(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-10)))


def rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -200.0
    return dbfs(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))))


def peak_normalize(audio: np.ndarray, target_peak_dbfs: float = -1.0, max_gain_db: float = 30.0) -> tuple[np.ndarray, float]:
    """Linear gain so the absolute peak hits ``target_peak_dbfs``. No compression / limiting."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-6:
        return audio, 0.0
    gain_db = min(target_peak_dbfs - dbfs(peak), max_gain_db)
    gain = 10 ** (gain_db / 20.0)
    return (audio * gain).astype(np.float32), gain_db


def prepare_audio(path: str | Path, workdir: str | Path, cfg: AudioConfig) -> PreparedAudio:
    info = probe(path)
    if info.duration_sec and info.duration_sec > cfg.max_duration_hours * 3600:
        raise AudioError(
            f"오디오 길이 {info.duration_sec / 3600:.1f}시간이 허용 최대치 {cfg.max_duration_hours}시간을 초과합니다.",
            hint="파일을 분할하거나 audio.max_duration_hours 설정을 늘리세요.",
        )
    audio, wav_path = load_audio_file(path, workdir, cfg.sample_rate, cfg.channel_strategy)
    duration = len(audio) / cfg.sample_rate
    if duration < 1.0:
        raise AudioError(f"오디오가 너무 짧습니다 ({duration:.2f}s).")

    notes: list[str] = []
    peak_before = dbfs(float(np.max(np.abs(audio)))) if audio.size else -200.0
    if peak_before < -60:
        raise AudioError("오디오가 거의 무음입니다 (peak < -60 dBFS).", hint="녹음 파일/채널을 확인하세요.")
    if info.channels and info.channels > 1:
        notes.append(f"원본 {info.channels}채널 → mono ({cfg.channel_strategy})")
    if info.sample_rate and info.sample_rate < 16000:
        notes.append(f"원본 샘플레이트 {info.sample_rate} Hz (<16 kHz): 정확도 저하 가능")

    gain_db = 0.0
    if cfg.normalize:
        audio, gain_db = peak_normalize(audio, cfg.target_peak_dbfs)

    if cfg.noise_reduction:
        if _noise_reducer is None:
            notes.append("noise_reduction=true 이지만 등록된 noise reducer 가 없어 건너뜀")
        else:
            audio = np.asarray(_noise_reducer(audio, cfg.sample_rate), dtype=np.float32)
            notes.append("noise reduction 적용")

    return PreparedAudio(
        audio=audio,
        sample_rate=cfg.sample_rate,
        duration_sec=duration,
        wav_path=wav_path,
        source_info=info,
        peak_dbfs_before=peak_before,
        rms_dbfs=rms_dbfs(audio),
        gain_db=gain_db,
        notes=notes,
    )
