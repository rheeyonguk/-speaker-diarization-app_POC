import numpy as np
import pytest

from conftest import SR, tone
from meeting_transcriber.config import ClusterEmbeddingConfig
from meeting_transcriber.diarization import DiarizationResult, Turn, from_pyannote_output, subtract_intervals
from meeting_transcriber.speaker_id.cluster_embedding import select_windows


def _annotation(items):
    from pyannote.core import Annotation, Segment

    ann = Annotation(uri="t")
    for k, (s, e, spk) in enumerate(items):
        ann[Segment(s, e), k] = spk
    return ann


def test_real_pyannote_diarize_output_is_preserved():
    """Uses the real pyannote.audio 4.x DiarizeOutput dataclass (not a mock)."""
    from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput

    full = _annotation([(0.0, 5.0, "SPEAKER_00"), (4.0, 8.0, "SPEAKER_01"), (9.0, 10.0, "SPEAKER_00")])
    excl = _annotation([(0.0, 4.5, "SPEAKER_00"), (4.5, 8.0, "SPEAKER_01"), (9.0, 10.0, "SPEAKER_00")])
    out = DiarizeOutput(speaker_diarization=full, exclusive_speaker_diarization=excl,
                        speaker_embeddings=np.zeros((2, 256)))
    res = from_pyannote_output(out, params={"x": 1}, model="community-1")
    assert res.exclusive_available
    assert res.labels == ["SPEAKER_00", "SPEAKER_01"]
    assert len(res.speaker_diarization) == 3 and len(res.exclusive_speaker_diarization) == 3
    assert res.overlap_regions == [(4.0, 5.0)]          # overlap kept from the full diarization
    assert res.cooccurrence("SPEAKER_00", "SPEAKER_01") == pytest.approx(1.0)
    d = res.to_dict()
    assert {"speaker_diarization", "exclusive_speaker_diarization", "overlap_regions"} <= set(d)
    # DiarizeOutput.serialize (upstream) agrees with what we keep
    ser = out.serialize()
    assert len(ser["diarization"]) == len(d["speaker_diarization"])
    assert len(ser["exclusive_diarization"]) == len(d["exclusive_speaker_diarization"])


def test_legacy_annotation_fallback():
    full = _annotation([(0.0, 5.0, "A"), (4.0, 6.0, "B")])
    res = from_pyannote_output(full)
    assert not res.exclusive_available
    ex = res.exclusive_speaker_diarization
    # exclusive fallback has no overlapping turns
    for a, b in zip(ex, ex[1:]):
        assert a.end <= b.start + 1e-9


def test_overlap_regions_three_speakers():
    turns = [Turn(0, 10, "A"), Turn(2, 4, "B"), Turn(3, 6, "C"), Turn(8, 12, "B")]
    res = DiarizationResult(turns, turns)
    assert res.overlap_regions == [(2, 6), (8, 10)]


def test_subtract_intervals():
    assert subtract_intervals((0, 10), [(2, 3), (5, 6)]) == [(0, 2), (3, 5), (6, 10)]
    assert subtract_intervals((0, 1), [(0, 1)]) == []


def test_cluster_windows_skip_overlap_and_short_turns():
    audio = tone([200, 400], 30.0)
    turns = [
        Turn(0.0, 6.0, "S0"), Turn(5.0, 7.0, "S1"),   # 5-6 overlap
        Turn(8.0, 8.8, "S0"),                          # too short for primary selection
        Turn(10.0, 25.0, "S0"),                        # long -> split into <=10 s windows
        Turn(26.0, 26.7, "S2"),                        # 0.7 s total < min_total_sec -> not identifiable
        Turn(27.0, 27.8, "S3"), Turn(28.5, 29.3, "S3"),  # only short pieces -> low evidence
    ]
    diar = DiarizationResult(turns, turns)
    cfg = ClusterEmbeddingConfig()
    w0, low0, speech0, clean0 = select_windows(diar, "S0", audio, SR, cfg)
    assert not low0
    for s, e in w0:
        assert not (s < 6.0 and e > 5.0), "windows must exclude overlap regions"
        assert e - s <= cfg.max_segment_sec + 1e-6
        assert e - s >= cfg.min_segment_sec - 1e-6
    assert speech0 == pytest.approx(21.8)
    assert clean0 == pytest.approx(20.8)
    w2, _, _, _ = select_windows(diar, "S2", audio, SR, cfg)
    assert w2 == []
    w3, low3, _, _ = select_windows(diar, "S3", audio, SR, cfg)
    assert low3 and len(w3) == 2


def test_cluster_windows_drop_silence():
    audio = np.zeros(SR * 10, dtype=np.float32)
    diar = DiarizationResult([Turn(0, 5, "S0")], [Turn(0, 5, "S0")])
    w, _, _, _ = select_windows(diar, "S0", audio, SR, ClusterEmbeddingConfig())
    assert w == []
