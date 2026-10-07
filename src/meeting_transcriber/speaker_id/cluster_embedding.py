"""Cluster-level representative embeddings for diarization speakers (SPEAKER_xx).

A single short turn is a poor speaker sample. For every cluster we:
  1. take its turns from the overlap-aware diarization and cut out every overlap region
     (no other speaker present),
  2. keep pieces >= min_segment_sec (longer first); if none exist, fall back to
     >= fallback_min_segment_sec and flag the cluster as low-evidence,
  3. split long pieces into <= max_segment_sec windows, drop near-silent windows (RMS),
  4. embed up to max_segments windows / max_total_sec seconds with WeSpeaker,
  5. average the L2-normalized window embeddings weighted by duration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..audio.preprocess import rms_dbfs
from ..config import ClusterEmbeddingConfig
from ..diarization.types import DiarizationResult, IntervalSet, subtract_intervals
from .scoring import weighted_centroid


@dataclass
class ClusterEvidence:
    cluster: str
    windows: list = field(default_factory=list)     # [(start, end)] actually embedded
    embeddings: Optional[np.ndarray] = None         # (k, D) per-window (kept in memory only)
    centroid: Optional[np.ndarray] = None           # (D,)
    evidence_sec: float = 0.0
    speech_sec: float = 0.0                         # total diarized speech of the cluster
    clean_speech_sec: float = 0.0                   # non-overlapped speech
    low_evidence: bool = False

    @property
    def valid(self) -> bool:
        return self.centroid is not None

    def summary(self) -> dict:
        return {
            "cluster": self.cluster,
            "num_windows": len(self.windows),
            "evidence_sec": round(self.evidence_sec, 2),
            "speech_sec": round(self.speech_sec, 2),
            "clean_speech_sec": round(self.clean_speech_sec, 2),
            "low_evidence": self.low_evidence,
        }


def _split(piece: tuple[float, float], max_len: float) -> list[tuple[float, float]]:
    s, e = piece
    n = max(1, math.ceil((e - s) / max_len))
    step = (e - s) / n
    return [(s + k * step, s + (k + 1) * step) for k in range(n)]


def select_windows(
    diar: DiarizationResult,
    cluster: str,
    audio: np.ndarray,
    sample_rate: int,
    cfg: ClusterEmbeddingConfig,
) -> tuple[list[tuple[float, float]], bool, float, float]:
    """Returns (windows, low_evidence, speech_sec, clean_speech_sec)."""
    overlap = IntervalSet(diar.overlap_regions)
    turns = diar.speaker_timeline(cluster)
    speech_sec = sum(e - s for s, e in turns)
    pieces: list[tuple[float, float]] = []
    for turn in turns:
        pieces.extend(subtract_intervals(turn, overlap.window(*turn)))
    clean_sec = sum(e - s for s, e in pieces)

    low = False
    cands = [p for p in pieces if p[1] - p[0] >= cfg.min_segment_sec]
    if not cands:
        cands = [p for p in pieces if p[1] - p[0] >= cfg.fallback_min_segment_sec]
        low = bool(cands)

    windows: list[tuple[float, float]] = []
    for p in cands:
        windows.extend(_split(p, cfg.max_segment_sec))

    def window_ok(w: tuple[float, float]) -> bool:
        a, b = int(w[0] * sample_rate), int(w[1] * sample_rate)
        return b > a and rms_dbfs(audio[a:b]) >= cfg.min_rms_dbfs

    windows = [w for w in windows if window_ok(w)]
    windows.sort(key=lambda w: -(w[1] - w[0]))
    chosen, total = [], 0.0
    for w in windows:
        if len(chosen) >= cfg.max_segments or total >= cfg.max_total_sec:
            break
        chosen.append(w)
        total += w[1] - w[0]
    chosen.sort()
    if total < cfg.min_total_sec:
        return [], low, speech_sec, clean_sec
    if total < 2 * cfg.min_segment_sec:
        low = True
    return chosen, low, speech_sec, clean_sec


def build_cluster_evidence(
    diar: DiarizationResult,
    audio: np.ndarray,
    sample_rate: int,
    embedder,
    cfg: ClusterEmbeddingConfig,
) -> dict[str, ClusterEvidence]:
    out: dict[str, ClusterEvidence] = {}
    for cluster in diar.labels:
        windows, low, speech_sec, clean_sec = select_windows(diar, cluster, audio, sample_rate, cfg)
        ev = ClusterEvidence(cluster=cluster, speech_sec=speech_sec, clean_speech_sec=clean_sec, low_evidence=low)
        embs, weights, used = [], [], []
        for s, e in windows:
            vec = embedder.embed(audio[int(s * sample_rate):int(e * sample_rate)], sample_rate)
            if vec is not None:
                embs.append(vec)
                weights.append(e - s)
                used.append((s, e))
        if embs:
            ev.embeddings = np.stack(embs)
            ev.centroid = weighted_centroid(ev.embeddings, np.asarray(weights))
            ev.windows = used
            ev.evidence_sec = float(sum(weights))
        out[cluster] = ev
    return out
