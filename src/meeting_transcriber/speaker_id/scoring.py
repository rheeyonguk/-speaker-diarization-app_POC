"""Profile construction and cosine scoring (pure numpy, model independent).

Scoring follows the WeSpeaker recipes (wespeaker/bin/score.py): cosine similarity between
embeddings, optionally after subtracting a cohort mean vector. Scores are raw cosine in [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .embedder import l2_normalize


@dataclass
class RobustCentroid:
    centroid: np.ndarray        # (D,) L2-normalized
    keep: np.ndarray            # (n,) bool - embeddings used for the centroid
    loo_similarity: np.ndarray  # (n,) cosine of each embedding to the centroid of the others
    warnings: list


def robust_centroid(
    embeddings: np.ndarray,
    mad_k: float = 3.0,
    min_similarity: float = 0.2,
    min_keep_ratio: float = 0.5,
) -> RobustCentroid:
    """Mean of L2-normalized embeddings after leave-one-out outlier rejection.

    Why: averaging length-normalized embeddings of several enrollment utterances is the standard
    multi-session enrollment in speaker verification; it reduces the session/channel variability of
    any single sample. A sample that is not like the others (wrong person, music, heavy noise,
    clipping) would drag the centroid, so it is rejected when its leave-one-out cosine is
      * a robust outlier: (s - median) / (1.4826 * MAD) < -mad_k, or
      * below an absolute floor ``min_similarity``.
    At least ``min_keep_ratio`` of the embeddings (the most consistent ones) are always kept.
    """
    E = l2_normalize(np.atleast_2d(np.asarray(embeddings, dtype=np.float32)))
    n = E.shape[0]
    warnings: list[str] = []
    if n == 0:
        raise ValueError("no embeddings")
    if n == 1:
        return RobustCentroid(E[0].copy(), np.array([True]), np.array([1.0], dtype=np.float32),
                              ["샘플 1개 기반 프로필: 3개 이상 등록을 권장합니다."])
    total = E.sum(axis=0)
    loo = np.array([float(E[i] @ l2_normalize(total - E[i])) for i in range(n)], dtype=np.float32)
    if n == 2:
        keep = np.array([True, True])
        if loo.min() < min_similarity:
            warnings.append(f"두 샘플 간 유사도가 낮습니다({loo.min():.2f}). 동일 인물/녹음 상태를 확인하세요.")
    else:
        med = float(np.median(loo))
        mad = float(np.median(np.abs(loo - med))) * 1.4826
        z = (loo - med) / max(mad, 1e-3)
        keep = ~((z < -mad_k) | (loo < min_similarity))
        min_keep = int(np.ceil(min_keep_ratio * n))
        if keep.sum() < min_keep:
            keep = np.zeros(n, dtype=bool)
            keep[np.argsort(-loo)[:min_keep]] = True
            warnings.append("일관성이 낮은 샘플이 많아 가장 일관된 샘플만 사용했습니다. 재녹음을 권장합니다.")
        dropped = int((~keep).sum())
        if dropped:
            warnings.append(f"{dropped}개 구간이 outlier 로 제외되었습니다.")
    centroid = l2_normalize(E[keep].mean(axis=0))
    return RobustCentroid(centroid, keep, loo, warnings)


def weighted_centroid(embeddings: np.ndarray, weights: np.ndarray) -> np.ndarray:
    E = l2_normalize(np.atleast_2d(embeddings))
    w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    return l2_normalize((E * w).sum(axis=0) / max(float(w.sum()), 1e-9))


def cosine_matrix(a: np.ndarray, b: np.ndarray, mean_vec: Optional[np.ndarray] = None) -> np.ndarray:
    """(M, D) x (N, D) -> (M, N) raw cosine; optional WeSpeaker-style mean subtraction first."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b = np.atleast_2d(np.asarray(b, dtype=np.float32))
    if mean_vec is not None:
        a = a - mean_vec
        b = b - mean_vec
    return l2_normalize(a) @ l2_normalize(b).T
