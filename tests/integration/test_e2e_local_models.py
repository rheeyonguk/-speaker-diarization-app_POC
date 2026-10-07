"""Full pipeline, end to end, fully offline, through MeetingTranscriber.run():

  M4A upload -> ffmpeg -> WhisperX ASR (bundled pyannote VAD + local CTranslate2 Whisper)
  -> WhisperX alignment (local wav2vec2-CTC) -> pyannote SpeakerDiarization + VBx (local config.yaml)
  -> WeSpeaker identification (local checkpoint) -> TXT/JSON/CSV/SRT

Models are built by tests/local_models.py with the real architectures/formats and random weights
(segmentation is the real checkpoint bundled with whisperx). Speech is the real two-speaker sample
bundled with pyannote.audio, remixed into a meeting. Verifies integration only - not accuracy.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

SR = 16000


def _punkt_available():
    try:
        import nltk

        nltk.data.find("tokenizers/punkt_tab/english/")
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def local_models(tmp_path_factory):
    pytest.importorskip("whisper")
    from local_models import (
        build_local_pyannote_pipeline,
        build_random_wespeaker,
        build_tiny_wav2vec2_ctc,
        build_tiny_whisper_ct2,
    )

    d = tmp_path_factory.mktemp("local_models")
    return {
        "whisper": build_tiny_whisper_ct2(d / "whisper"),
        "align": build_tiny_wav2vec2_ctc(d / "w2v2", "가나다라마바사아자차카타파하회의를시작합니다abcdefghijklmnopqrstuvwxyz0123456789"),
        "pyannote": build_local_pyannote_pipeline(d / "pyannote"),
        "wespeaker": build_random_wespeaker(d / "wespeaker"),
    }


@pytest.fixture(scope="module")
def meeting(tmp_path_factory, ffmpeg_module):
    from pyannote.audio.sample import SAMPLE_FILE

    from meeting_transcriber.audio.io import convert_to_pcm_wav, read_wav, write_wav
    from meeting_transcriber.evaluation.synthetic import SynthConfig, load_clip_bank, synthesize

    work = tmp_path_factory.mktemp("meeting")
    audio, _ = read_wav(convert_to_pcm_wav(SAMPLE_FILE["audio"], work / "s16.wav"))
    turns = [(s.start, s.end, lab) for s, _, lab in SAMPLE_FILE["annotation"].itertracks(yield_label=True)]
    enroll = {}
    for spk in sorted({lab for *_, lab in turns}):
        own = [(s, e) for s, e, lab in turns if lab == spk and e - s > 1.0]
        (work / "clips" / spk).mkdir(parents=True)
        for k, (s, e) in enumerate(own[1:]):  # first turn is held out for enrollment
            write_wav(work / "clips" / spk / f"{k}.wav", audio[int(s * SR):int(e * SR)], SR)
        s, e = own[0]
        enroll[spk] = write_wav(work / f"enroll_{spk}.wav", audio[int(s * SR):int(e * SR)], SR)
    res = synthesize(load_clip_bank(work / "clips", work / "bank"),
                     SynthConfig(target_duration_sec=60, overlap_prob=0.15, interjection_prob=0.2, seed=1))
    wav = write_wav(work / "meeting.wav", res.audio, SR)
    m4a = work / "회의_녹음.m4a"
    ffmpeg_module("-i", wav, "-c:a", "aac", m4a)
    return {"m4a": m4a, "enroll": enroll, "duration": len(res.audio) / SR}


@pytest.fixture(scope="module")
def ffmpeg_module():
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")

    def run(*args):
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)

    return run


@pytest.mark.skipif(not _punkt_available(), reason="NLTK punkt_tab not available offline")
def test_full_pipeline_offline_with_local_models(tmp_path, local_models, meeting):
    from meeting_transcriber.config import load_config
    from meeting_transcriber.pipeline import STAGES, MeetingTranscriber, RunRequest
    from meeting_transcriber.speaker_id import SpeakerProfileStore, WeSpeakerEmbedder

    cfg = load_config(env={})
    cfg.runtime.device = "cpu"
    cfg.paths.output_dir = str(tmp_path / "outputs")
    cfg.paths.enrollment_dir = str(tmp_path / "speakers")
    cfg.asr.backend = "whisperx"
    cfg.asr.model = str(local_models["whisper"])
    cfg.asr.batch_size = 4
    cfg.alignment.model_name = str(local_models["align"])
    cfg.diarization.model = str(local_models["pyannote"])
    cfg.speaker_id.wespeaker_model = str(local_models["wespeaker"])
    cfg.speaker_id.enrollment.min_speech_sec = 1.0

    store = SpeakerProfileStore(cfg.enrollment_dir)
    emb = WeSpeakerEmbedder(cfg.speaker_id.wespeaker_model, "cpu")
    for name, path in meeting["enroll"].items():
        store.enroll(name, [path], emb, cfg.speaker_id.enrollment)  # Silero VAD on real speech

    seen = []
    result = MeetingTranscriber(cfg).run(
        RunRequest(str(meeting["m4a"]), speaker_mode="FIXED", num_speakers=2, language="ko"),
        lambda i, label, frac: seen.append(label) if frac == 0.0 else None,
    )
    doc = result.document

    assert seen == [label for _, label in STAGES]
    assert set(doc["processing"]["timings_sec"]) >= {"preprocess", "transcribe", "align", "diarize", "identify",
                                                     "export", "total"}
    assert doc["duration"] == pytest.approx(meeting["duration"], abs=0.2)
    assert doc["file"] == "회의_녹음.m4a"
    # diarization: both pyannote views kept, FIXED speaker count honoured
    assert doc["diarization"]["exclusive_available"] is True
    assert doc["speaker_count"] == 2
    assert doc["diarization"]["speaker_diarization"] and doc["diarization"]["exclusive_speaker_diarization"]
    # identification ran against both enrolled profiles
    si = doc["speaker_identification"]
    assert np.asarray(si["similarity_matrix"]).shape == (2, 2)
    assert sorted(si["profiles"]) == sorted(meeting["enroll"])
    labels = [s["label"] for s in doc["speakers"]]
    assert len(labels) == len(set(labels))
    assert all(lab in meeting["enroll"] or lab.startswith("UNKNOWN_") for lab in labels)
    # outputs
    for fmt in ("txt", "json", "csv", "srt"):
        assert result.files[fmt].exists() and result.files[fmt].stat().st_size > 0
    json.dumps(doc, allow_nan=False)
    assert not list(Path(result.job_dir).glob("*.wav")), "intermediate 16 kHz copy must be removed"
