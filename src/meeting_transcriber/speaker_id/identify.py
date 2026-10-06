"""Speaker identification stage: cluster evidence -> similarity matrix -> matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import SpeakerIdConfig
from ..diarization.types import DiarizationResult
from ..errors import ConfigError
from ..logging_utils import get_logger
from .cluster_embedding import build_cluster_evidence
from .matching import match_clusters, passthrough_matches
from .profiles import SpeakerProfileStore
from .scoring import cosine_matrix

logger = get_logger(__name__)


@dataclass
class IdentificationResult:
    matches: dict                       # cluster -> ClusterMatch
    profile_names: list = field(default_factory=list)
    similarity: Optional[np.ndarray] = None   # (M, N) clusters x profiles, raw cosine
    cooccurrence: Optional[np.ndarray] = None # (M, M) seconds of simultaneous speech
    clusters: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)  # cluster -> ClusterEvidence
    skipped_profiles: list = field(default_factory=list)
    model_id: Optional[str] = None
    enabled: bool = True
    notes: list = field(default_factory=list)
    centroids: Optional[np.ndarray] = field(default=None, repr=False)  # in-memory only, never exported

    def label_of(self, cluster: Optional[str]) -> str:
        if cluster is None:
            return "UNASSIGNED"
        m = self.matches.get(cluster)
        return m.label if m else cluster

    def window_agreement(self, cluster: str) -> Optional[float]:
        """Fraction of a cluster's windows whose own best profile equals the cluster decision."""
        ev = self.evidence.get(cluster)
        m = self.matches.get(cluster)
        if ev is None or ev.embeddings is None or m is None or m.status != "identified" or self.centroids is None:
            return None
        best = np.argmax(cosine_matrix(ev.embeddings, self.centroids), axis=1)
        j = self.profile_names.index(m.name)
        return float(np.mean(best == j))

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "embedding_model": self.model_id,
            "profiles": self.profile_names,
            "clusters": self.clusters,
            "similarity_matrix": None if self.similarity is None else [
                [round(float(x), 4) for x in row] for row in self.similarity
            ],
            "cooccurrence_sec": None if self.cooccurrence is None else [
                [round(float(x), 2) for x in row] for row in self.cooccurrence
            ],
            "evidence": {c: ev.summary() for c, ev in self.evidence.items()},
            "skipped_profiles": [{"name": p.name, "model_id": p.model_id} for p in self.skipped_profiles],
            "notes": self.notes,
        }


def cooccurrence_matrix(diar: DiarizationResult, clusters: list[str]) -> np.ndarray:
    M = len(clusters)
    C = np.zeros((M, M), dtype=np.float64)
    for a in range(M):
        for b in range(a + 1, M):
            C[a, b] = C[b, a] = diar.cooccurrence(clusters[a], clusters[b])
    return C


def load_cohort_mean(path: Optional[str], resolve) -> Optional[np.ndarray]:
    if not path:
        return None
    p = resolve(path)
    if not Path(p).exists():
        raise ConfigError(f"cohort_mean_path 파일이 없습니다: {p}")
    return np.load(p).astype(np.float32).reshape(-1)


def identify_speakers(
    diar: DiarizationResult,
    audio: np.ndarray,
    sample_rate: int,
    store: SpeakerProfileStore,
    embedder,
    cfg: SpeakerIdConfig,
    cohort_mean: Optional[np.ndarray] = None,
) -> IdentificationResult:
    clusters = diar.labels
    embedder.load()
    model_id = embedder.model_id
    profiles, centroids, skipped = store.matching_set(model_id)
    notes = []
    if skipped:
        notes.append(
            f"현재 임베딩 모델({model_id})과 다른 모델로 만든 프로필 {len(skipped)}개 제외: "
            + ", ".join(p.name for p in skipped) + " → 재등록 필요"
        )
    if not profiles:
        matches = {m.cluster: m for m in passthrough_matches(clusters, "등록된(호환) 화자 프로필 없음")}
        notes.append("등록된 화자 프로필이 없어 익명 화자(SPEAKER_xx)로 출력합니다.")
        return IdentificationResult(matches, clusters=clusters, skipped_profiles=skipped, model_id=model_id, notes=notes)

    evidence = build_cluster_evidence(diar, audio, sample_rate, embedder, cfg.cluster_embedding)
    names = [p.name for p in profiles]
    ids = [p.speaker_id for p in profiles]
    D = centroids.shape[1]
    cluster_vecs = np.stack([
        evidence[c].centroid if evidence[c].valid else np.zeros(D, dtype=np.float32) for c in clusters
    ]) if clusters else np.zeros((0, D), dtype=np.float32)
    S = cosine_matrix(cluster_vecs, centroids, cohort_mean) if clusters else np.zeros((0, len(names)))
    valid = [evidence[c].valid for c in clusters]
    S[~np.asarray(valid, dtype=bool)] = -1.0  # rows without evidence never match
    C = cooccurrence_matrix(diar, clusters)
    matches = match_clusters(
        S, clusters, names,
        threshold=cfg.match_threshold,
        margin=cfg.match_margin,
        mode=cfg.matching_mode,
        profile_ids=ids,
        valid=valid,
        cooccurrence=C,
        fragment_extra_threshold=cfg.fragment_extra_threshold,
        fragment_max_cooccurrence_sec=cfg.fragment_max_cooccurrence_sec,
        first_appearance=[diar.first_appearance(c) for c in clusters],
    )
    for c in clusters:
        if evidence[c].low_evidence:
            logger.info("Cluster %s identified from short/limited speech only", c)
    result = IdentificationResult(
        matches={m.cluster: m for m in matches},
        profile_names=names,
        similarity=S,
        cooccurrence=C,
        clusters=clusters,
        evidence=evidence,
        skipped_profiles=skipped,
        model_id=model_id,
        notes=notes,
        centroids=centroids,
    )
    return result

