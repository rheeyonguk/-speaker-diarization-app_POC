import hashlib

import numpy as np
import pytest

from conftest import SR, tone
from meeting_transcriber.audio import prepare_audio, probe, read_wav
from meeting_transcriber.audio.preprocess import peak_normalize, rms_dbfs
from meeting_transcriber.config import AudioConfig
from meeting_transcriber.errors import AudioError


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def stereo_44k(tmp_path, write_wav, ffmpeg):
    mono = tmp_path / "mono.wav"
    write_wav(mono, tone([180, 360, 540], 3.0), SR)
    src = tmp_path / "stereo44k.wav"
    ffmpeg("-i", mono, "-ac", "2", "-ar", "44100", src)
    return src


@pytest.mark.parametrize("ext,extra", [
    (".wav", []),
    (".mp3", ["-c:a", "libmp3lame"]),
    (".m4a", ["-c:a", "aac"]),
])
def test_audio_formats_converted_to_16k_mono(tmp_path, stereo_44k, ffmpeg, ext, extra):
    src = tmp_path / f"in{ext}"
    if ext == ".wav":
        src = stereo_44k
    else:
        ffmpeg("-i", stereo_44k, *extra, src)
    before = _sha(src)
    work = tmp_path / "job"
    prep = prepare_audio(src, work, AudioConfig())
    assert prep.sample_rate == SR
    assert prep.audio.dtype == np.float32 and prep.audio.ndim == 1
    assert prep.duration_sec == pytest.approx(3.0, abs=0.1)
    assert prep.source_info.channels == 2
    _, sr = read_wav(prep.wav_path)
    assert sr == SR
    assert _sha(src) == before, "original file must never be modified"


@pytest.mark.parametrize("ext", [".mp4", ".mov"])
def test_video_containers(tmp_path, stereo_44k, ffmpeg, ext):
    src = tmp_path / f"video{ext}"
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=64x64:d=3", "-i", stereo_44k, "-shortest",
           "-c:v", "libx264" if ext == ".mp4" else "mpeg4", "-c:a", "aac", src)
    info = probe(src)
    assert info.has_audio
    prep = prepare_audio(src, tmp_path / "job", AudioConfig())
    assert prep.duration_sec == pytest.approx(3.0, abs=0.15)


def test_unsupported_extension(tmp_path):
    f = tmp_path / "notes.docx"
    f.write_bytes(b"x")
    with pytest.raises(AudioError):
        prepare_audio(f, tmp_path / "job", AudioConfig())


def test_corrupt_file_gives_user_error(tmp_path):
    f = tmp_path / "broken.mp3"
    f.write_bytes(b"\x00\x01garbage" * 100)
    with pytest.raises(AudioError):
        prepare_audio(f, tmp_path / "job", AudioConfig())


def test_video_without_audio(tmp_path, ffmpeg):
    f = tmp_path / "silent_video.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=64x64:d=2", "-c:v", "libx264", f)
    with pytest.raises(AudioError, match="오디오 트랙"):
        prepare_audio(f, tmp_path / "job", AudioConfig())


def test_peak_normalization_is_pure_gain():
    x = tone([200], 1.0, amp=0.05)
    y, gain_db = peak_normalize(x, -1.0)
    assert np.max(np.abs(y)) == pytest.approx(10 ** (-1 / 20), rel=1e-3)
    assert gain_db > 0
    # pure gain: correlation 1
    assert np.corrcoef(x, y)[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_noise_reduction_off_by_default():
    assert AudioConfig().noise_reduction is False


def test_rms_dbfs():
    assert rms_dbfs(np.zeros(100, dtype=np.float32)) < -150
    assert rms_dbfs(np.ones(100, dtype=np.float32)) == pytest.approx(0.0, abs=1e-6)
