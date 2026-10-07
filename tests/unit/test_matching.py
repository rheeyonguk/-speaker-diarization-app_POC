import numpy as np
import pytest

from meeting_transcriber.speaker_id.matching import match_clusters, passthrough_matches

NAMES = ["이용욱", "홍길동", "김철수"]


def run(S, clusters=None, mode="flexible", thr=0.5, margin=0.05, **kw):
    clusters = clusters or [f"SPEAKER_{i:02d}" for i in range(len(S))]
    ms = match_clusters(np.asarray(S), clusters, NAMES[: np.asarray(S).shape[1]], threshold=thr, margin=margin,
                        mode=mode, **kw)
    return {m.cluster: m for m in ms}


def test_example_matrix_from_spec():
    S = [[0.91, 0.34, 0.41],
         [0.29, 0.93, 0.51],
         [0.35, 0.47, 0.88]]
    for mode in ("flexible", "strict_one_to_one", "argmax"):
        out = run(S, mode=mode)
        assert [out[f"SPEAKER_0{i}"].label for i in range(3)] == NAMES
        assert out["SPEAKER_00"].similarity == pytest.approx(0.91)
        assert out["SPEAKER_00"].status == "identified"


def test_low_similarity_stays_unknown_and_unknowns_are_distinct():
    S = [[0.91, 0.10, 0.10],
         [0.20, 0.30, 0.25],   # guest 1
         [0.15, 0.22, 0.31]]   # guest 2
    out = run(S, first_appearance=[0.0, 5.0, 2.0])
    assert out["SPEAKER_00"].label == "이용욱"
    # numbered by first appearance: SPEAKER_02 appears at 2.0s, SPEAKER_01 at 5.0s
    assert out["SPEAKER_02"].label == "UNKNOWN_01"
    assert out["SPEAKER_01"].label == "UNKNOWN_02"
    assert out["SPEAKER_01"].status == "unknown" and "threshold" in out["SPEAKER_01"].reason


def test_argmax_double_assigns_but_global_does_not():
    # Two clusters both closest to 이용욱; the second is really 홍길동 (weaker match)
    S = [[0.85, 0.30, 0.10],
         [0.70, 0.62, 0.10]]
    argmax = run(S, mode="argmax")
    assert argmax["SPEAKER_00"].label == argmax["SPEAKER_01"].label == "이용욱"
    strict = run(S, mode="strict_one_to_one")
    assert strict["SPEAKER_00"].label == "이용욱"
    # global assignment hands the second cluster to 홍길동 (이용욱 is claimed by the stronger cluster)
    assert strict["SPEAKER_01"].label == "홍길동"
    assert strict["SPEAKER_01"].via == "hungarian"


def test_ambiguous_margin_goes_unknown():
    S = [[0.62, 0.60, 0.10]]
    out = run(S, mode="strict_one_to_one", margin=0.05)
    assert out["SPEAKER_00"].label == "UNKNOWN_01"
    assert "margin" in out["SPEAKER_00"].reason


def test_fragmented_speaker_flexible_vs_strict():
    # One person split into two clusters that never talk at the same time
    S = [[0.82, 0.20, 0.15],
         [0.74, 0.18, 0.20],
         [0.10, 0.85, 0.20]]
    C = np.zeros((3, 3))
    strict = run(S, mode="strict_one_to_one", cooccurrence=C)
    assert strict["SPEAKER_01"].status == "unknown"
    flex = run(S, mode="flexible", cooccurrence=C)
    assert flex["SPEAKER_00"].label == flex["SPEAKER_01"].label == "이용욱"
    assert flex["SPEAKER_01"].via == "fragment"
    assert flex["SPEAKER_02"].label == "홍길동"


def test_flexible_refuses_merge_of_simultaneous_speakers():
    S = [[0.82, 0.20, 0.15],
         [0.74, 0.18, 0.20]]
    C = np.array([[0.0, 12.0], [12.0, 0.0]])  # they talk over each other for 12s -> not the same person
    flex = run(S, mode="flexible", cooccurrence=C, fragment_max_cooccurrence_sec=2.0)
    assert flex["SPEAKER_01"].label == "UNKNOWN_01"
    assert "동시에" in flex["SPEAKER_01"].reason


def test_fragment_needs_extra_threshold():
    S = [[0.82, 0.20, 0.15],
         [0.52, 0.18, 0.20]]  # above base threshold 0.5 but below 0.5 + 0.05
    flex = run(S, mode="flexible", cooccurrence=np.zeros((2, 2)), fragment_extra_threshold=0.05)
    assert flex["SPEAKER_01"].status == "unknown"


def test_eight_enrolled_two_guests_ten_clusters():
    rng = np.random.default_rng(0)
    names = [f"P{i}" for i in range(8)]
    S = rng.uniform(0.0, 0.3, size=(10, 8))
    for i in range(8):
        S[i, i] = rng.uniform(0.65, 0.9)
    ms = match_clusters(S, [f"SPEAKER_{i:02d}" for i in range(10)], names, threshold=0.5, margin=0.05,
                        mode="flexible", cooccurrence=np.zeros((10, 10)), first_appearance=list(range(10)))
    labels = [m.label for m in ms]
    assert labels[:8] == names
    assert labels[8:] == ["UNKNOWN_01", "UNKNOWN_02"]
    assert len(set(labels)) == 10


def test_insufficient_audio_cluster_is_unknown():
    S = [[0.91, 0.1, 0.1], [-1.0, -1.0, -1.0]]
    out = run(S, valid=[True, False])
    assert out["SPEAKER_01"].status == "insufficient_audio"
    assert out["SPEAKER_01"].label.startswith("UNKNOWN_")


def test_more_profiles_than_clusters_and_single_profile():
    S = [[0.2, 0.7, 0.1]]
    assert run(S)["SPEAKER_00"].label == "홍길동"
    one = match_clusters(np.array([[0.7], [0.3]]), ["SPEAKER_00", "SPEAKER_01"], ["이용욱"],
                         threshold=0.5, margin=0.05)
    assert [m.label for m in one] == ["이용욱", "UNKNOWN_01"]


def test_passthrough_keeps_cluster_labels():
    ms = passthrough_matches(["SPEAKER_00", "SPEAKER_01"], "off")
    assert [m.label for m in ms] == ["SPEAKER_00", "SPEAKER_01"]
    assert all(m.status == "not_identified" for m in ms)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        match_clusters(np.zeros((2, 2)), ["a"], ["x", "y"], threshold=0.5, margin=0.0)
