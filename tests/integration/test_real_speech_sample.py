"""Uses the real 30 s two-speaker speech sample bundled with pyannote.audio (sample.wav + RTTM)."""

import numpy as np
import pytest

from conftest import SR


@pytest.fixture(scope="module")
def sample():
    pytest.importorskip("pyannote.audio")
    from pyannote.audio.sample import SAMPLE_FILE

    return SAMPLE_FILE


@pytest.fixture(scope="module")
def sample_audio(sample, tmp_path_factory):
    from meeting_transcriber.audio.io import convert_to_pcm_wav, read_wav

    wav = convert_to_pcm_wav(sample["audio"], tmp_path_factory.mktemp("s") / "sample16k.wav")
    audio, sr = read_wav(wav)
    assert sr == SR
    return audio


def _turns(sample):
    return [{"start": s.start, "end": s.end, "speaker": lab}
            for s, _, lab in sample["annotation"].itertracks(yield_label=True)]


def _speaker_audio(audio, turns, speaker):
    return np.concatenate([audio[int(t["start"] * SR):int(t["end"] * SR)] for t in turns if t["speaker"] == speaker])


def test_vad_enrollment_on_real_speech(tmp_path, sample, sample_audio, wespeaker_random_model_dir, write_wav):
    """Silero VAD (enrollment default) accepts real speech; storage / centroid pipeline runs end to end."""
    from meeting_transcriber.config import EnrollmentConfig
    from meeting_transcriber.speaker_id import SpeakerProfileStore, WeSpeakerEmbedder

    turns = _turns(sample)
    spk = max({t["speaker"] for t in turns}, key=lambda s: sum(t["end"] - t["start"] for t in turns if t["speaker"] == s))
    speech = _speaker_audio(sample_audio, turns, spk)
    f = tmp_path / "enroll.wav"
    write_wav(f, np.concatenate([np.zeros(SR, np.float32), speech, np.zeros(SR, np.float32)]), SR)
    store = SpeakerProfileStore(tmp_path / "speakers")
    emb = WeSpeakerEmbedder(str(wespeaker_random_model_dir), "cpu")
    rep = store.enroll("Speaker A", [f], emb, EnrollmentConfig(vad=True, min_speech_sec=3.0, chunk_sec=5.0))
    assert rep.accepted_samples == 1
    assert rep.total_speech_sec >= 3.0
    assert rep.total_speech_sec <= len(speech) / SR + 2.0  # leading/trailing silence was removed
    assert rep.chunks_total >= 2


def test_der_against_bundled_reference(sample):
    from meeting_transcriber.evaluation import diarization_error

    ref = _turns(sample)
    hyp = [{"start": s.start, "end": s.end, "speaker": lab}
           for s, _, lab in sample["diarization"].itertracks(yield_label=True)]
    d = diarization_error(ref, hyp)
    assert 0.0 <= d["der"] < 0.5
    assert d["der"] == pytest.approx(d["miss_rate"] + d["false_alarm_rate"] + d["confusion_rate"], abs=1e-9)


def test_synthetic_meeting_from_real_clips(tmp_path, sample, sample_audio, write_wav):
    from meeting_transcriber.evaluation.synthetic import SynthConfig, load_clip_bank, synthesize

    turns = _turns(sample)
    for spk in {t["speaker"] for t in turns}:
        d = tmp_path / "clips" / spk
        d.mkdir(parents=True)
        for k, t in enumerate(t for t in turns if t["speaker"] == spk and t["end"] - t["start"] > 1.0):
            write_wav(d / f"{k}.wav", sample_audio[int(t["start"] * SR):int(t["end"] * SR)], SR)
    bank = load_clip_bank(tmp_path / "clips", tmp_path / "work")
    res = synthesize(bank, SynthConfig(target_duration_sec=40, overlap_prob=0.5, interjection_prob=0.5, seed=3))
    assert set(res.speakers) == set(bank)
    assert len(res.audio) / SR >= 40
    by_spk = {}
    for t in res.turns:
        by_spk.setdefault(t["speaker"], []).append((t["start"], t["end"]))
    for ivs in by_spk.values():  # nobody overlaps themself
        ivs.sort()
        for (_, e1), (s2, _) in zip(ivs, ivs[1:]):
            assert s2 >= e1 - 1e-3
    assert np.max(np.abs(res.audio)) <= 1.0
