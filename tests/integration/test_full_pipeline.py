"""Real end-to-end run: WhisperX large model + alignment + pyannote Community-1 + WeSpeaker.

Opt-in (needs model downloads / HF_TOKEN with accepted Community-1 conditions / real speech):

  RUN_INTEGRATION=1 MT_TEST_AUDIO=path/to/korean_meeting.wav \
  [MT_TEST_NUM_SPEAKERS=3] [MT_TEST_ENROLL_DIR=data/enroll_raw] \
  pytest -m integration tests/integration/test_full_pipeline.py -s
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

AUDIO = os.environ.get("MT_TEST_AUDIO")
RUN = os.environ.get("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not (RUN and AUDIO), reason="set RUN_INTEGRATION=1 and MT_TEST_AUDIO")
def test_full_pipeline_real_models(tmp_path):
    from meeting_transcriber.config import hf_token, load_config
    from meeting_transcriber.pipeline import MeetingTranscriber, RunRequest
    from meeting_transcriber.speaker_id import SpeakerProfileStore, WeSpeakerEmbedder

    if not hf_token():
        pytest.skip("HF_TOKEN not set")
    cfg = load_config()
    cfg.paths.output_dir = str(tmp_path / "outputs")
    enroll_dir = os.environ.get("MT_TEST_ENROLL_DIR")
    if enroll_dir:
        cfg.paths.enrollment_dir = str(tmp_path / "speakers")
        store = SpeakerProfileStore(cfg.enrollment_dir)
        emb = WeSpeakerEmbedder(cfg.speaker_id.wespeaker_model, "cpu", cfg.model_cache_dir)
        for d in sorted(p for p in Path(enroll_dir).iterdir() if p.is_dir()):
            store.enroll(d.name, sorted(d.glob("*.wav")), emb, cfg.speaker_id.enrollment)

    n = os.environ.get("MT_TEST_NUM_SPEAKERS")
    req = RunRequest(AUDIO, speaker_mode="FIXED" if n else "AUTO", num_speakers=int(n) if n else None)
    res = MeetingTranscriber(cfg).run(req, lambda i, label, f: None)
    doc = res.document
    assert doc["language"] == "ko"
    assert doc["segments"], "no transcript produced"
    assert any(s["words"] for s in doc["segments"]), "no word-level alignment"
    assert doc["diarization"]["exclusive_available"]
    assert doc["speaker_count"] >= 1
    if n:
        assert doc["speaker_count"] == int(n)
    for fmt in ("txt", "json", "csv", "srt"):
        assert res.files[fmt].exists()
    print(res.files["txt"].read_text(encoding="utf-8")[:2000])
