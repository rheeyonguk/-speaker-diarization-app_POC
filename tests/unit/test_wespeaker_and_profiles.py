"""Exercises the real WeSpeaker package (model build, checkpoint loading, kaldi fbank, inference)
through our adapter, with a randomly initialised ResNet34 (pretrained weights cannot be
downloaded in CI). Verifies API compatibility and data flow - NOT identification accuracy."""

import json

import numpy as np
import pytest

from conftest import SR, tone
from meeting_transcriber.config import EnrollmentConfig, SpeakerIdConfig
from meeting_transcriber.diarization import DiarizationResult, Turn
from meeting_transcriber.errors import ConfigError, EnrollmentError
from meeting_transcriber.speaker_id import SpeakerProfileStore, WeSpeakerEmbedder, identify_speakers, speech_only


@pytest.fixture(scope="module")
def embedder(wespeaker_random_model_dir):
    e = WeSpeakerEmbedder(str(wespeaker_random_model_dir), "cpu")
    e.load()
    return e


def test_embedding_shape_and_norm(embedder):
    v = embedder.embed(tone([150, 300, 450], 2.0), SR)
    assert v.shape == (256,)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)
    assert embedder.embedding_dim == 256
    assert embedder.model_id.startswith("wespeaker:local:")
    assert embedder.embed(np.zeros(100, dtype=np.float32), SR) is None  # too short


def test_embedding_is_deterministic(embedder):
    x = tone([170, 340], 2.0, seed=3)
    assert np.allclose(embedder.embed(x, SR), embedder.embed(x, SR), atol=1e-6)


def test_unknown_hub_name_is_config_error():
    with pytest.raises(ConfigError):
        WeSpeakerEmbedder("definitely-not-a-model", "cpu").load()


def test_hub_names_from_upstream():
    from meeting_transcriber.speaker_id.embedder import hub_model_names

    names = hub_model_names()
    assert "vblinkf" in names and "english" in names


def test_speech_only_vad_trims_silence():
    pytest.importorskip("silero_vad")
    x = np.concatenate([np.zeros(SR * 2, np.float32), tone([150, 300], 1.0), np.zeros(SR * 2, np.float32)])
    y = speech_only(x, SR)
    assert len(y) < len(x)


@pytest.fixture
def enrollment_files(tmp_path, write_wav):
    files = {}
    for name, freqs, seed in (("이용욱", [140, 280, 420], 1), ("홍길동", [220, 440, 660], 2)):
        files[name] = []
        for k in range(3):
            p = tmp_path / f"{seed}_{k}.wav"
            write_wav(p, np.concatenate([tone(freqs, 6.0, seed=seed * 10 + k), np.zeros(SR // 2, np.float32)]), SR)
            files[name].append(p)
    return files


def test_enroll_list_reenroll_delete(tmp_path, embedder, enrollment_files):
    store = SpeakerProfileStore(tmp_path / "speakers")
    cfg = EnrollmentConfig(vad=False, min_speech_sec=1.0)
    rep = store.enroll("이용욱", enrollment_files["이용욱"], embedder, cfg)
    assert rep.accepted_samples == 3 and rep.chunks_total == 3
    p = store.find_by_name("이용욱")
    sdir = store.root / p.speaker_id
    assert (sdir / "centroid.npy").exists() and (sdir / "embeddings.npy").exists()
    assert len(list((sdir / "samples").glob("*.wav"))) == 3
    assert p.model_id == embedder.model_id

    # appending a sample to the same name
    rep2 = store.enroll("이용욱", enrollment_files["홍길동"][:1], embedder, cfg)
    assert store.find_by_name("이용욱").num_samples == 4 and rep2.chunks_total == 4

    # re-enroll (replace) resets the profile
    store.enroll("이용욱", enrollment_files["이용욱"][:2], embedder, cfg, replace=True)
    assert store.find_by_name("이용욱").num_samples == 2

    # recompute from stored audio (model change workflow)
    rep3 = store.reenroll_from_stored(p.speaker_id, embedder, cfg)
    assert rep3.accepted_samples == 2

    # raw audio deletion keeps the profile usable
    assert store.delete_raw_audio(p.speaker_id) == 2
    prof = store.get(p.speaker_id)
    assert not prof.raw_audio_kept and (sdir / "centroid.npy").exists()
    with pytest.raises(EnrollmentError):
        store.reenroll_from_stored(p.speaker_id, embedder, cfg)

    # profile.json never contains embeddings
    raw = json.loads((sdir / "profile.json").read_text(encoding="utf-8"))
    assert "centroid" not in raw and "embeddings" not in raw

    assert store.delete(p.speaker_id)
    assert not sdir.exists() and store.list_profiles() == []


def test_enroll_rejects_silence_and_short(tmp_path, embedder, write_wav):
    """Default config (Silero VAD on): a silent sample has no speech -> rejected, nothing stored."""
    store = SpeakerProfileStore(tmp_path / "speakers")
    silent = tmp_path / "silent.wav"
    write_wav(silent, np.zeros(SR * 3, np.float32), SR)
    with pytest.raises(EnrollmentError):
        store.enroll("무음", [silent], embedder, EnrollmentConfig())
    with pytest.raises(EnrollmentError):
        store.enroll("", [silent], embedder, EnrollmentConfig())
    assert store.list_profiles() == []


def test_enroll_without_keeping_raw_audio(tmp_path, embedder, enrollment_files):
    store = SpeakerProfileStore(tmp_path / "speakers")
    store.enroll("홍길동", enrollment_files["홍길동"], embedder, EnrollmentConfig(vad=False, min_speech_sec=1.0), keep_raw_audio=False)
    p = store.find_by_name("홍길동")
    assert not (store.root / p.speaker_id / "samples").exists()
    assert not p.raw_audio_kept


def test_identify_end_to_end_data_flow(tmp_path, embedder, enrollment_files):
    """Full identification stage on real WeSpeaker inference. With random weights the decisions are
    not meaningful, so only structure / invariants are asserted."""
    store = SpeakerProfileStore(tmp_path / "speakers")
    cfg = EnrollmentConfig(vad=False, min_speech_sec=1.0)
    for name, files in enrollment_files.items():
        store.enroll(name, files, embedder, cfg)
    audio = np.concatenate([tone([140, 280, 420], 8.0, seed=7), tone([220, 440, 660], 8.0, seed=8),
                            tone([300, 600], 0.6, seed=9)])
    turns = [Turn(0.0, 8.0, "SPEAKER_00"), Turn(8.0, 16.0, "SPEAKER_01"), Turn(16.0, 16.6, "SPEAKER_02")]
    diar = DiarizationResult(turns, turns)
    res = identify_speakers(diar, audio, SR, store, embedder, SpeakerIdConfig())
    assert res.similarity.shape == (3, 2)
    assert np.all(res.similarity <= 1.0 + 1e-6) and np.all(res.similarity >= -1.0 - 1e-6)
    assert res.matches["SPEAKER_02"].status == "insufficient_audio"
    labels = [res.matches[c].label for c in res.clusters]
    unknowns = [x for x in labels if x.startswith("UNKNOWN_")]
    assert len(unknowns) == len(set(unknowns))  # unknown speakers stay distinct
    d = res.to_dict()
    assert "similarity_matrix" in d and "centroids" not in json.dumps(d)
