import numpy as np
import pytest

from meeting_transcriber.evaluation.calibration import calibrate, eer, enrollment_trials, threshold_at_far
from meeting_transcriber.speaker_id.scoring import cosine_matrix, robust_centroid, weighted_centroid


def _cluster(center, n, noise, rng):
    return center + noise * rng.standard_normal((n, center.size))


def test_robust_centroid_rejects_outlier():
    rng = np.random.default_rng(1)
    c = rng.standard_normal(64)
    good = _cluster(c, 6, 0.3, rng)
    bad = rng.standard_normal((1, 64)) * 3  # another person / music / noise
    rc = robust_centroid(np.vstack([good, bad]))
    assert rc.keep.tolist() == [True] * 6 + [False]
    assert float(rc.centroid @ (c / np.linalg.norm(c))) > 0.9
    assert np.linalg.norm(rc.centroid) == pytest.approx(1.0, abs=1e-5)
    assert any("outlier" in w for w in rc.warnings)


def test_robust_centroid_small_sets():
    rng = np.random.default_rng(2)
    one = robust_centroid(rng.standard_normal((1, 16)))
    assert one.keep.tolist() == [True] and one.warnings
    two = robust_centroid(rng.standard_normal((2, 16)))
    assert two.keep.tolist() == [True, True]


def test_robust_centroid_keeps_majority():
    rng = np.random.default_rng(3)
    E = rng.standard_normal((6, 32))  # all inconsistent
    rc = robust_centroid(E, min_similarity=0.99)
    assert rc.keep.sum() >= 3


def test_cosine_matrix_and_mean_subtraction():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    b = np.array([[2.0, 0.0]])
    S = cosine_matrix(a, b)
    assert S.shape == (2, 1)
    assert S[0, 0] == pytest.approx(1.0)
    assert S[1, 0] == pytest.approx(0.0)
    S2 = cosine_matrix(np.array([[1.0, 1.0]]), np.array([[1.0, 0.9]]), mean_vec=np.array([1.0, 0.95]))
    assert S2[0, 0] < 0  # centring changes geometry (WeSpeaker-style mean normalisation)


def test_weighted_centroid_prefers_heavier():
    E = np.array([[1.0, 0.0], [0.0, 1.0]])
    c = weighted_centroid(E, np.array([9.0, 1.0]))
    assert c[0] > c[1]


def test_eer_and_far_threshold():
    rng = np.random.default_rng(4)
    tgt = rng.normal(0.7, 0.08, 2000).tolist()
    non = rng.normal(0.1, 0.1, 2000).tolist()
    e, thr = eer(tgt, non)
    assert e < 0.01 and 0.3 < thr < 0.5
    thr1, frr = threshold_at_far(tgt, non, 0.01)
    far = np.mean(np.asarray(non) >= thr1)
    assert far <= 0.01
    res = calibrate(tgt, non)
    assert res["recommended_threshold"] >= res["eer_threshold"]


def test_enrollment_trials():
    rng = np.random.default_rng(5)
    a = _cluster(rng.standard_normal(32), 4, 0.2, rng)
    b = _cluster(rng.standard_normal(32), 3, 0.2, rng)
    tgt, non = enrollment_trials([("a", a, np.ones(4, bool)), ("b", b, np.ones(3, bool))])
    assert len(tgt) == 7 and len(non) == 7
    assert min(tgt) > max(non)
