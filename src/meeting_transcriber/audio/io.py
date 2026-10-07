"""ffmpeg-based input handling.

The original file is only ever *read*. All processing happens on a 16 kHz PCM working copy.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import AudioError

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".m4a", ".mp4", ".mov")
# Extra containers ffmpeg handles fine; accepted but not advertised in the UI.
ALSO_ACCEPTED = (".flac", ".ogg", ".webm", ".aac", ".wma", ".mkv")


@dataclass
class MediaInfo:
    path: str
    duration_sec: float
    has_audio: bool
    audio_streams: int
    channels: int | None = None
    sample_rate: int | None = None
    codec: str | None = None
    bit_rate: int | None = None
    format_name: str | None = None
    extra: dict = field(default_factory=dict)


def require_ffmpeg() -> None:
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise AudioError(
            f"{', '.join(missing)} 실행 파일을 찾을 수 없습니다.",
            hint="ffmpeg 를 설치하고 PATH 에 추가하세요 (Windows: winget install Gyan.FFmpeg / Ubuntu: apt install ffmpeg).",
        )


def check_extension(path: str | Path) -> None:
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS + ALSO_ACCEPTED:
        raise AudioError(
            f"지원하지 않는 파일 형식입니다: {ext or '(확장자 없음)'}",
            hint=f"지원 형식: {', '.join(SUPPORTED_EXTENSIONS)}",
        )


def probe(path: str | Path) -> MediaInfo:
    require_ffmpeg()
    path = Path(path)
    if not path.exists():
        raise AudioError(f"파일이 존재하지 않습니다: {path}")
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise AudioError(f"미디어 파일을 읽을 수 없습니다 (손상되었거나 지원하지 않는 코덱): {path.name}",
                         hint=proc.stderr.strip()[-300:] or None)
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    fmt = data.get("format", {})
    duration = _to_float(fmt.get("duration"))
    if duration is None and audio:
        duration = _to_float(audio[0].get("duration"))
    first = audio[0] if audio else {}
    return MediaInfo(
        path=str(path),
        duration_sec=duration or 0.0,
        has_audio=bool(audio),
        audio_streams=len(audio),
        channels=_to_int(first.get("channels")),
        sample_rate=_to_int(first.get("sample_rate")),
        codec=first.get("codec_name"),
        bit_rate=_to_int(first.get("bit_rate")) or _to_int(fmt.get("bit_rate")),
        format_name=fmt.get("format_name"),
    )


def convert_to_pcm_wav(
    src: str | Path,
    dst: str | Path,
    sample_rate: int = 16000,
    channel_strategy: str = "downmix",
) -> Path:
    """Decode any supported input to 16-bit PCM WAV (mono) at ``sample_rate``.

    ``downmix`` averages all channels (ffmpeg -ac 1); ``first`` keeps only channel 0, which is
    preferable when e.g. channel 1 is a noisy room mic and channel 0 the table mic.
    """
    require_ffmpeg()
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise AudioError("변환 결과가 원본 파일을 덮어쓰게 됩니다. 출력 경로를 확인하세요.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-map", "0:a:0"]
    if channel_strategy == "first":
        cmd += ["-af", "pan=mono|c0=c0"]
    else:
        cmd += ["-ac", "1"]
    cmd += ["-ar", str(sample_rate), "-acodec", "pcm_s16le", "-f", "wav", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not dst.exists():
        raise AudioError(f"오디오 변환(ffmpeg) 실패: {src.name}", hint=proc.stderr.strip()[-300:] or None)
    return dst


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV into float32 mono in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise AudioError(f"16-bit PCM WAV 만 직접 읽을 수 있습니다 (sample width={width}).")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return path


def load_audio_file(path: str | Path, workdir: str | Path, sample_rate: int = 16000,
                    channel_strategy: str = "downmix") -> tuple[np.ndarray, Path]:
    """Convert ``path`` into ``workdir`` and return (float32 mono audio, wav path)."""
    check_extension(path)
    info = probe(path)
    if not info.has_audio:
        raise AudioError(f"오디오 트랙이 없는 파일입니다: {Path(path).name}")
    wav_path = convert_to_pcm_wav(path, Path(workdir) / "audio_16k_mono.wav", sample_rate, channel_strategy)
    audio, sr = read_wav(wav_path)
    if sr != sample_rate:  # pragma: no cover - ffmpeg guarantees -ar
        raise AudioError(f"예상하지 못한 샘플레이트: {sr}")
    return audio, wav_path


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
