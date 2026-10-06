"""Evaluate one pipeline output document against references (used by scripts/evaluate.py)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..speaker_id.matching import MATCHING_MODES, match_clusters
from .metrics import (
    attribution_error_rate,
    cer,
    cluster_identity,
    diarization_error,
    per_speaker_time_accuracy,
    relabel_turns,
    speaker_id_metrics,
    wer,
)


def rematch_from_document(doc: dict, mode: str, threshold: Optional[float] = None,
                          margin: Optional[float] = None, fragment_extra: float = 0.05,
                          fragment_max_cooc: float = 2.0) -> Optional[dict]:
    """Re-run cluster->person matching on the similarity matrix stored in an output JSON."""
    si = doc.get("speaker_identification") or {}
    sim = si.get("similarity_matrix")
    if not sim:
        return None
    clusters = si["clusters"]
    profiles = si["profiles"]
    evidence = si.get("evidence", {})
    valid = [evidence.get(c, {}).get("num_windows", 1) > 0 for c in clusters]
    first = {}
    for t in doc["diarization"]["speaker_diarization"]:
        first[t["speaker"]] = min(first.get(t["speaker"], float("inf")), t["start"])
    matches = match_clusters(
        np.asarray(sim, dtype=float), clusters, profiles,
        threshold=si.get("threshold") if threshold is None else threshold,
        margin=si.get("margin") if margin is None else margin,
        mode=mode, valid=valid,
        cooccurrence=np.asarray(si["cooccurrence_sec"]) if si.get("cooccurrence_sec") else None,
        fragment_extra_threshold=fragment_extra,
        fragment_max_cooccurrence_sec=fragment_max_cooc,
        first_appearance=[first.get(c, float("inf")) for c in clusters],
    )
    return {m.cluster: m.label for m in matches}


def evaluate_document(doc: dict, reference_turns: list[dict], reference_text: Optional[str],
                      enrolled: Optional[list[str]] = None, collars=(0.0, 0.25)) -> dict:
    out: dict = {"file": doc.get("file"), "duration": doc.get("duration"),
                 "hyp_speaker_count": doc.get("speaker_count"),
                 "ref_speaker_count": len({t["speaker"] for t in reference_turns})}

    hyp_text = " ".join(s["text"] for s in sorted(doc["segments"], key=lambda s: s["start"]))
    if reference_text:
        out["stt"] = {"cer": cer(reference_text, hyp_text), "cer_with_spaces": cer(reference_text, hyp_text, False),
                      "wer": wer(reference_text, hyp_text)}

    hyp_turns = doc["diarization"]["speaker_diarization"]
    out["diarization"] = {f"collar_{c}": diarization_error(reference_turns, hyp_turns, collar=c) for c in collars}

    si = doc.get("speaker_identification") or {}
    enrolled = list(enrolled) if enrolled else list(si.get("profiles") or [])
    out["enrolled"] = enrolled
    as_run = {s["cluster"]: s["label"] for s in doc.get("speakers", [])}
    modes = {"as_run": as_run}
    for mode in MATCHING_MODES:
        m = rematch_from_document(doc, mode)
        if m is not None:
            modes[mode] = m
    out["speaker_id"] = {}
    for name, mapping in modes.items():
        named = relabel_turns(hyp_turns, mapping)
        out["speaker_id"][name] = {
            "cluster_level": speaker_id_metrics(reference_turns, hyp_turns, mapping, enrolled),
            "time_level": attribution_error_rate(reference_turns, named, enrolled),
            "per_speaker": per_speaker_time_accuracy(reference_turns, named, enrolled),
        }

    # cluster-vs-profile scores labelled by ground truth -> threshold calibration input
    target, nontarget = [], []
    sim = si.get("similarity_matrix")
    if sim:
        truth = {c: r for c, (r, _, _) in cluster_identity(reference_turns, hyp_turns).items()}
        for i, c in enumerate(si["clusters"]):
            for j, p in enumerate(si["profiles"]):
                (target if truth.get(c) == p else nontarget).append(float(sim[i][j]))
    out["similarity_scores"] = {"target": target, "nontarget": nontarget}
    return out


def markdown_summary(results: list[dict]) -> str:
    lines = ["| 시나리오 | 화자(ref/hyp) | CER | DER(c=0.25) | Miss | FA | Conf | ID mode | ID acc | FA rate | FR rate | Unknown rej. | Attr. err |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        d = r["diarization"].get("collar_0.25") or next(iter(r["diarization"].values()))
        cer_v = r.get("stt", {}).get("cer", {}).get("cer")
        for mode, m in r["speaker_id"].items():
            cl, tl = m["cluster_level"], m["time_level"]

            def f(v):
                return "-" if v is None else f"{v:.3f}"

            lines.append(
                f"| {r['id']} | {r['ref_speaker_count']}/{r['hyp_speaker_count']} | {f(cer_v)} | {f(d['der'])} | "
                f"{f(d['miss_rate'])} | {f(d['false_alarm_rate'])} | {f(d['confusion_rate'])} | {mode} | "
                f"{f(cl['accuracy'])} | {f(cl['false_accept_rate'])} | {f(cl['false_reject_rate'])} | "
                f"{f(cl['unknown_rejection_rate'])} | {f(tl['attribution_error_rate'])} |"
            )
    return "\n".join(lines) + "\n"
