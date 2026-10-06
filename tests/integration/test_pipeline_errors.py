"""Pipeline orchestration and graceful error handling with real components (no mocks).

These run without downloaded models: failures must surface as UserFacingError with the stage name,
never as an unhandled traceback, and the original file must stay untouched.
"""

import hashlib

import pytest

from conftest import SR, tone
from meeting_transcriber.errors import AudioError, UserFacingError
from meeting_transcriber.pipeline import MeetingTranscriber, RunRequest


def test_bad_model_name_fails_gracefully_at_transcription(tmp_cfg, tmp_path, write_wav):
    src = tmp_path / "meeting.wav"
    write_wav(src, tone([150, 300], 3.0), SR)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    tmp_cfg.paths.model_cache_dir = str(tmp_path / "models")
    with pytest.raises(UserFacingError) as ei:
        MeetingTranscriber(tmp_cfg).run(RunRequest(str(src), whisper_model="no-such-whisper-model-xyz"))
    assert ei.value.stage == "2. Transcription"
    assert "Whisper" in ei.value.message
    assert hashlib.sha256(src.read_bytes()).hexdigest() == digest
    # intermediate 16 kHz working copy is cleaned up even on failure
    assert not list((tmp_path / "outputs").rglob("audio_16k_mono.wav"))


def test_corrupt_input_fails_at_preprocessing(tmp_cfg, tmp_path):
    bad = tmp_path / "broken.m4a"
    bad.write_bytes(b"not audio at all" * 50)
    with pytest.raises(AudioError) as ei:
        MeetingTranscriber(tmp_cfg).run(RunRequest(str(bad)))
    assert ei.value.stage == "1. Audio preprocessing"


def test_invalid_speaker_settings_rejected_before_processing(tmp_cfg, tmp_path, write_wav):
    src = tmp_path / "m.wav"
    write_wav(src, tone([150], 2.0), SR)
    with pytest.raises(UserFacingError):
        MeetingTranscriber(tmp_cfg).run(RunRequest(str(src), speaker_mode="FIXED", num_speakers=None))
    with pytest.raises(UserFacingError):
        MeetingTranscriber(tmp_cfg).run(RunRequest(str(src), speaker_mode="RANGE", min_speakers=6, max_speakers=3))
    assert not (tmp_path / "outputs").exists() or not any((tmp_path / "outputs").iterdir())


def test_cuda_request_without_gpu_is_clear(tmp_cfg, tmp_path, write_wav):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("GPU present")
    src = tmp_path / "m.wav"
    write_wav(src, tone([150], 2.0), SR)
    tmp_cfg.runtime.device = "cuda"
    with pytest.raises(UserFacingError, match="CUDA"):
        MeetingTranscriber(tmp_cfg).run(RunRequest(str(src)))
