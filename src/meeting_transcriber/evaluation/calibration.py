"""Threshold calibration from target / non-target cosine scores.

Two score sources are supported:
  * enrollment leave-one-out (``enrollment_trials``): every stored chunk embedding is scored
    against its own speaker's centroid built without that chunk (target) and against every other
    speaker's centroid (non-target). Cheap and needs no labelled meetings, but enrollment audio is
    cleaner than meeting audio, so the resulting threshold is optimistic.
  * labelled meetings (``scripts/evaluate.py`` collects cluster-vs-profile scores using the
    reference RTTM). Representative of the real operating condition - preferred.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..speaker_id.embedder import l2_normalize


def eer(target: list[float], nontarget: list[float]) -> tuple[float, float]:
    """Equal error rate and the threshold where FAR ~= FRR."""
    t = np.sort(np.asarray(target, dtype=float))
    n = np.sort(np.asarray(nontarget, dtype=float))
    if t.size == 0 or n.size == 0:
        raise ValueError("need both target and non-target scores")
    thresholds = np.unique(np.concatenate([t, n]))
    frr = np.searchsorted(t, thresholds, side="left") / t.size           # target < thr -> rejected
    far = 1.0 - np.searchsorted(n, thresholds, side="left") / n.size     # nontarget >= thr -> accepted
    k = int(np.argmin(np.abs(frr - far)))
    return float((frr[k] + far[k]) / 2), float(thresholds[k])


def threshold_at_far(target: list[float], nontarget: list[float], far: float) -> tuple[float, float]:
    """Smallest threshold whose false-accept rate <= ``far``; returns (threshold, frr at it)."""
    t = np.sort(np.asarray(target, dtype=float))
    n = np.sort(np.asarray(nontarget, dtype=float))
    if n.size == 0:
        raise ValueError("need non-target scores")
    k = int(np.ceil((1.0 - far) * n.size)) - 1
    k = min(max(k, 0), n.size - 1)
    thr = float(np.nextafter(n[k], np.inf))
    frr = float(np.searchsorted(t, thr, side="left") / t.size) if t.size else float("nan")
    return thr, frr


def calibrate(target: list[float], nontarget: list[float], fars=(0.01, 0.05)) -> dict:
    e, thr = eer(target, nontarget)
    out = {
        "n_target": len(target),
        "n_nontarget": len(nontarget),
        "target_mean": float(np.mean(target)),
        "nontarget_mean": float(np.mean(nontarget)),
        "eer": e,
        "eer_threshold": thr,
    }
    for far in fars:
        t, frr = threshold_at_far(target, nontarget, far)
        out[f"threshold_far_{far:g}"] = t
        out[f"frr_at_far_{far:g}"] = frr
    # conservative recommendation for meeting minutes: prefer UNKNOWN over a wrong name
    out["recommended_threshold"] = max(thr, out[f"threshold_far_{min(fars):g}"])
    return out


def enrollment_trials(profiles: list[tuple[str, np.ndarray, np.ndarray]]) -> tuple[list[float], list[float]]:
    """profiles: [(name, chunk_embeddings (n, D), keep_mask (n,))]"""
    cents = {}
    for name, E, keep in profiles:
        E = l2_normalize(E[keep] if keep is not None and keep.any() else E)
        cents[name] = (E, l2_normalize(E.mean(axis=0)))
    target, nontarget = [], []
    for name, (E, _) in cents.items():
        if E.shape[0] >= 2:
            total = E.sum(axis=0)
            for i in range(E.shape[0]):
                target.append(float(E[i] @ l2_normalize(total - E[i])))
        for other, (_, c) in cents.items():
            if other != name:
                nontarget.extend((E @ c).tolist())
    return target, nontarget


def summarize(target: list[float], nontarget: list[float]) -> Optional[dict]:
    if not target or not nontarget:
        return None
    return calibrate(target, nontarget)
