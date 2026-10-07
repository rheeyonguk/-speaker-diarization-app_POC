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
    tmp_cfg.asr.backend = "whisperx"
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


def test_azure_backend_without_credentials_fails_clearly(tmp_cfg, tmp_path, write_wav, monkeypatch):
    monkeypatch.delenv("AZURE_SPEECH_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    src = tmp_path / "m.wav"
    write_wav(src, tone([150], 3.0), SR)
    tmp_cfg.asr.backend = "azure_mai"
    with pytest.raises(UserFacingError, match="AZURE_SPEECH_ENDPOINT") as ei:
        MeetingTranscriber(tmp_cfg).run(RunRequest(str(src)))
    assert ei.value.stage == "2. Transcription"


def test_torch_threads_survive_silero_import(tmp_cfg):
    torch = pytest.importorskip("torch")
    from meeting_transcriber.device import apply_torch_threads, effective_cpu_threads
    from meeting_transcriber.speaker_id.embedder import hub_model_names, speech_only

    apply_torch_threads(tmp_cfg)
    hub_model_names()  # imports wespeaker -> silero_vad (which calls torch.set_num_threads(1) on import)
    import numpy as np

    speech_only(np.zeros(16000, dtype=np.float32))
    assert torch.get_num_threads() == effective_cpu_threads(tmp_cfg)
