"""Layer 8: diarization cluster -> enrolled person assignment.

Modes
-----
argmax             Baseline for comparison only: every cluster independently takes its best
                   profile (threshold + margin). Two clusters can both become the same person.
strict_one_to_one  Global optimum (Hungarian / linear_sum_assignment) - each person at most once.
                   A fragmented speaker (one person split into 2 clusters) leaves the weaker
                   fragment as UNKNOWN.
flexible (default) Hungarian first, then a leftover cluster may join an already-assigned person
                   only if (a) its similarity clears threshold + fragment_extra_threshold,
                   (b) its margin is clear, and (c) it hardly ever talks *at the same time* as the
                   cluster(s) already mapped to that person (one person cannot overlap themself).

Acceptance (all modes): similarity >= threshold and similarity - best_alternative >= margin,
where alternatives exclude persons confidently claimed by another cluster in the global step.
Everything else stays UNKNOWN_xx (numbered by first appearance, kept distinct per cluster).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

MATCHING_MODES = ("flexible", "strict_one_to_one", "argmax")


@dataclass
class ClusterMatch:
    cluster: str
    label: str = ""                       # final display label (name / UNKNOWN_01 / SPEAKER_00)
    status: str = "unknown"               # identified | unknown | insufficient_audio | not_identified
    name: Optional[str] = None
    speaker_id: Optional[str] = None
    similarity: Optional[float] = None    # cosine to the assigned profile
    best_name: Optional[str] = None
    best_similarity: Optional[float] = None
    margin: Optional[float] = None
    via: Optional[str] = None             # hungarian | fragment | argmax
    reason: str = ""
    candidates: list = field(default_factory=list)  # top-3 [{"name", "similarity"}]

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("similarity", "best_similarity", "margin"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 4)
        return d


def _row_info(S: np.ndarray, i: int, names: Sequence[str]) -> tuple[int, float, list]:
    order = np.argsort(-S[i])
    cands = [{"name": names[j], "similarity": round(float(S[i, j]), 4)} for j in order[:3]]
    return int(order[0]), float(S[i, order[0]]), cands


def _margin(S: np.ndarray, i: int, j: int, excluded: set[int]) -> float:
    others = [S[i, k] for k in range(S.shape[1]) if k != j and k not in excluded]
    return float(S[i, j] - max(others)) if others else float("inf")


def match_clusters(
    S: np.ndarray,
    cluster_labels: Sequence[str],
    profile_names: Sequence[str],
    *,
    threshold: float,
    margin: float,
    mode: str = "flexible",
    profile_ids: Optional[Sequence[str]] = None,
    valid: Optional[Sequence[bool]] = None,
    cooccurrence: Optional[np.ndarray] = None,
    fragment_extra_threshold: float = 0.05,
    fragment_max_cooccurrence_sec: float = 2.0,
    first_appearance: Optional[Sequence[float]] = None,
) -> list[ClusterMatch]:
    """Assign clusters (rows of ``S``) to profiles (columns). Returns one ClusterMatch per cluster."""
    if mode not in MATCHING_MODES:
        raise ValueError(f"unknown matching mode {mode!r}")
    S = np.asarray(S, dtype=np.float64)
    M = len(cluster_labels)
    N = len(profile_names)
    profile_ids = list(profile_ids) if profile_ids is not None else list(profile_names)
    valid = list(valid) if valid is not None else [True] * M
    if S.shape != (M, N):
        raise ValueError(f"similarity matrix shape {S.shape} != ({M}, {N})")

    matches = [ClusterMatch(cluster=c) for c in cluster_labels]
    assigned: dict[int, int] = {}  # cluster row -> profile col

    for i in range(M):
        if not valid[i]:
            matches[i].status = "insufficient_audio"
            matches[i].reason = "식별에 충분한 비겹침 발화가 없음"
        elif N:
            b, s, cands = _row_info(S, i, profile_names)
            matches[i].best_name, matches[i].best_similarity, matches[i].candidates = profile_names[b], s, cands

    rows = [i for i in range(M) if valid[i]]
    if N and rows:
        if mode == "argmax":
            for i in rows:
                j = int(np.argmax(S[i]))
                m = _margin(S, i, j, set())
                matches[i].margin = m
                if S[i, j] >= threshold and m >= margin:
                    assigned[i] = j
                    matches[i].via = "argmax"
        else:
            from scipy.optimize import linear_sum_assignment

            sub = S[rows]
            r_idx, c_idx = linear_sum_assignment(-sub)
            pairs = {rows[r]: int(c) for r, c in zip(r_idx, c_idx)}
            # persons confidently explained by some cluster (above threshold in the global solution)
            claimed = {i: j for i, j in pairs.items() if S[i, j] >= threshold}
            for i, j in pairs.items():
                taken_elsewhere = {jj for ii, jj in claimed.items() if ii != i}
                m = _margin(S, i, j, taken_elsewhere)
                if S[i, j] >= threshold and m >= margin:
                    assigned[i] = j
                    matches[i].via, matches[i].margin = "hungarian", m
                else:
                    matches[i].margin = m

            if mode == "flexible":
                frag_thr = threshold + fragment_extra_threshold
                # strongest leftovers first so they get first claim
                leftovers = sorted((i for i in rows if i not in assigned), key=lambda i: -float(S[i].max()))
                for i in leftovers:
                    j = int(np.argmax(S[i]))
                    m = _margin(S, i, j, set())
                    if S[i, j] < frag_thr or m < margin:
                        continue
                    partners = [p for p, q in assigned.items() if q == j]
                    if cooccurrence is not None and any(
                        cooccurrence[i, p] > fragment_max_cooccurrence_sec for p in partners
                    ):
                        matches[i].reason = "동일 인물 후보와 동시에 발화(겹침)하여 병합 불가"
                        continue
                    assigned[i] = j
                    matches[i].via, matches[i].margin = "fragment", m

    for i in range(M):
        mt = matches[i]
        if i in assigned:
            j = assigned[i]
            mt.status = "identified"
            mt.name = profile_names[j]
            mt.speaker_id = profile_ids[j]
            mt.similarity = float(S[i, j])
            mt.label = profile_names[j]
            mt.reason = mt.reason or ("" if mt.via != "fragment" else "분할된 동일 화자 클러스터로 판단")
        elif valid[i] and N:
            mt.status = "unknown"
            if not mt.reason:
                if mt.best_similarity is not None and mt.best_similarity < threshold:
                    mt.reason = f"최고 유사도 {mt.best_similarity:.3f} < threshold {threshold:.3f}"
                elif mt.margin is not None and mt.margin < margin:
                    mt.reason = f"후보 간 차이(margin {mt.margin:.3f}) < {margin:.3f} - 모호"
                else:
                    mt.reason = "다른 클러스터가 해당 인물에 더 적합하게 배정됨"

    # UNKNOWN_xx numbering by first appearance (each cluster keeps its own number)
    unknown_rows = [i for i in range(M) if i not in assigned]
    if first_appearance is not None:
        unknown_rows.sort(key=lambda i: first_appearance[i])
    for k, i in enumerate(unknown_rows, start=1):
        matches[i].label = f"UNKNOWN_{k:02d}"
    return matches


def passthrough_matches(cluster_labels: Sequence[str], reason: str) -> list[ClusterMatch]:
    """Identification disabled / no profiles: keep anonymous cluster labels."""
    return [ClusterMatch(cluster=c, label=c, status="not_identified", reason=reason) for c in cluster_labels]
