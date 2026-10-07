"""Azure MAI-Transcribe client against a local HTTP double of the documented REST contract
(tests/azure_stub.py), plus the full pipeline with backend=azure_mai.

The real Azure call is NOT exercised here (no credentials / network in CI): this verifies request
format (path, api-version, key header, multipart audio+definition), parsing, retries, error
mapping, chunking and pipeline integration. Use `meeting-transcriber azure-check` on a machine with
the company resource configured for the live check.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from azure_stub import KEY, AzureSpeechStub
from conftest import SR, tone

pytestmark = pytest.mark.integration


@pytest.fixture
def azure_env(monkeypatch):
    def _set(url, key=KEY):
        monkeypatch.setenv("AZURE_SPEECH_ENDPOINT", url)
        monkeypatch.setenv("AZURE_SPEECH_KEY", key)
    return _set


def _cfg(tmp_path):
    from meeting_transcriber.config import load_config

    cfg = load_config(env={})
    cfg.runtime.device = "cpu"
    cfg.paths.output_dir = str(tmp_path / "outputs")
    cfg.paths.enrollment_dir = str(tmp_path / "speakers")
    cfg.asr.backend = "azure_mai"
    cfg.asr.azure.max_retries = 2
    return cfg


def test_request_contract_and_parsing(tmp_path, azure_env):
    from meeting_transcriber.asr import AzureMaiTranscriber

    with AzureSpeechStub() as stub:
        azure_env(stub.url)
        cfg = _cfg(tmp_path)
        cfg.asr.azure.phrases = ["GMP"]
        out = AzureMaiTranscriber(cfg).transcribe(tone([180, 360], 13.0), SR, tmp_path, extra_phrases=["APQR"])
    req = stub.requests[0]
    assert req["path"] == "/speechtotext/transcriptions:transcribe"
    assert req["query"]["api-version"] == ["2025-10-15"]
    assert req["key"] == KEY
    assert req["audio_type"] == "audio/flac" and req["audio_bytes"] > 0
    assert req["definition"]["enhancedMode"]["model"] == "MAI-Transcribe-2"
    assert req["definition"]["enhancedMode"]["modelOptions"]["timestamps"] == "word"
    assert req["definition"]["phraseList"]["phrases"] == ["GMP", "APQR"]
    assert out["language"] == "ko" and out["aligned"] and out["requests"] == 1
    texts = [s["text"] for s in out["segments"]]
    assert "APQR 초안은 다음 주까지 공유드리겠습니다." in texts
    assert not list(tmp_path.glob("_mai_upload_*")), "upload temp files must be deleted"


def test_long_audio_is_chunked_and_offsets_are_global(tmp_path, azure_env):
    from meeting_transcriber.asr import AzureMaiTranscriber

    with AzureSpeechStub() as stub:
        azure_env(stub.url)
        cfg = _cfg(tmp_path)
        cfg.asr.azure.max_chunk_minutes = 1.0
        audio = tone([200, 400], 150.0)
        out = AzureMaiTranscriber(cfg).transcribe(audio, SR, tmp_path)
    assert out["requests"] == len(stub.requests) == 3
    starts = [s["start"] for s in out["segments"]]
    assert starts == sorted(starts) and max(s["end"] for s in out["segments"]) <= 150.0 + 1e-6
    assert any(s["start"] > 120 for s in out["segments"])  # third chunk mapped back to global time


def test_retry_on_429_then_success(tmp_path, azure_env):
    from meeting_transcriber.asr import AzureMaiTranscriber

    with AzureSpeechStub("429_then_ok") as stub:
        azure_env(stub.url)
        out = AzureMaiTranscriber(_cfg(tmp_path)).transcribe(tone([200], 8.0), SR, tmp_path)
    assert len(stub.requests) == 2 and out["segments"]


@pytest.mark.parametrize("mode,key,match", [
    ("ok", "wrong-key-000000000", "인증 실패"),
    ("400_region", KEY, "MAI-Transcribe 요청 거부"),
    ("500_always", KEY, "서버 오류"),
])
def test_error_mapping(tmp_path, azure_env, mode, key, match):
    from meeting_transcriber.asr import AzureMaiTranscriber
    from meeting_transcriber.errors import UserFacingError

    with AzureSpeechStub(mode) as stub:
        azure_env(stub.url, key)
        with pytest.raises(UserFacingError, match=match) as ei:
            AzureMaiTranscriber(_cfg(tmp_path)).transcribe(tone([200], 6.0), SR, tmp_path)
    assert key not in str(ei.value)
    if mode == "400_region":
        assert "southeastasia" in str(ei.value)


def test_unreachable_endpoint(tmp_path, azure_env):
    from meeting_transcriber.asr import AzureMaiTranscriber
    from meeting_transcriber.errors import UserFacingError

    azure_env("http://127.0.0.1:9")  # nothing listens on the discard port
    cfg = _cfg(tmp_path)
    cfg.asr.azure.max_retries = 0
    with pytest.raises(UserFacingError, match="연결 실패"):
        AzureMaiTranscriber(cfg).transcribe(tone([200], 3.0), SR, tmp_path)


def test_full_pipeline_with_azure_backend(tmp_path, azure_env):
    """M4A -> preprocess -> Azure MAI (double) -> alignment skipped -> pyannote (local config) ->
    WeSpeaker -> exports."""
    pytest.importorskip("whisper")
    from local_models import build_local_pyannote_pipeline, build_random_wespeaker

    from meeting_transcriber.audio.io import convert_to_pcm_wav, read_wav, write_wav
    from meeting_transcriber.pipeline import MeetingTranscriber, RunRequest

    from pyannote.audio.sample import SAMPLE_FILE

    audio, _ = read_wav(convert_to_pcm_wav(SAMPLE_FILE["audio"], tmp_path / "s16.wav"))
    meeting = write_wav(tmp_path / "meeting.wav", np.concatenate([audio, audio]), SR)
    cfg = _cfg(tmp_path)
    cfg.diarization.model = str(build_local_pyannote_pipeline(tmp_path / "pyannote"))
    cfg.speaker_id.wespeaker_model = str(build_random_wespeaker(tmp_path / "wespeaker"))
    with AzureSpeechStub() as stub:
        azure_env(stub.url)
        res = MeetingTranscriber(cfg).run(RunRequest(str(meeting), speaker_mode="FIXED", num_speakers=2,
                                                     phrases=["APQR"]))
    doc = res.document
    assert len(stub.requests) == 1
    assert doc["settings"]["asr_backend"] == "azure_mai"
    assert doc["models"]["asr"] == "MAI-Transcribe-2"
    assert doc["models"]["alignment"] == "azure_mai word timestamps"
    assert doc["language"] == "ko" and doc["speaker_count"] == 2
    assert any("APQR 초안은" in s["text"] for s in doc["segments"])
    assert all(s["identified_speaker"] for s in doc["segments"])
    assert any(s["words"] for s in doc["segments"])
    assert stub.requests[0]["definition"]["phraseList"]["phrases"] == ["APQR"]
    json.dumps(doc, allow_nan=False)
    assert not list(Path(res.job_dir).glob("*.wav")) and not list(Path(res.job_dir).glob("_mai_upload_*"))
