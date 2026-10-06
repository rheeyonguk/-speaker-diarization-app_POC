"""Evaluation metrics.

STT          : CER (Korean primary metric; spaces removed by default because Korean spacing is
               inconsistent) and WER, via jiwer.
Diarization  : DER with Miss / False Alarm / Confusion components, via pyannote.metrics.
Speaker ID   : cluster-level decisions vs. reference identity -> accuracy, false accept
               (impostor accepted / wrong enrolled name), false reject, unknown rejection,
               confusion matrix; plus a time-based attribution error rate.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from ..diarization.types import intersect_length, merge_intervals

UNKNOWN = "UNKNOWN"
UNENROLLED = "<UNENROLLED>"

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- STT


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def cer(reference: str, hypothesis: str, ignore_spaces: bool = True) -> dict:
    import jiwer

    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if ignore_spaces:
        ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")
    if not ref:
        raise ValueError("empty reference text")
    out = jiwer.process_characters(ref, hyp)
    return {
        "cer": float(out.cer),
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "reference_chars": len(ref),
        "ignore_spaces": ignore_spaces,
    }


def wer(reference: str, hypothesis: str) -> dict:
    import jiwer

    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        raise ValueError("empty reference text")
    out = jiwer.process_words(ref, hyp)
    return {
        "wer": float(out.wer),
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "reference_words": len(ref.split()),
    }


# --------------------------------------------------------------------------- RTTM / annotations


def read_rttm(path: str | Path) -> list[dict]:
    """RTTM: SPEAKER <uri> <chan> <start> <dur> <NA> <NA> <label> <NA> <NA>"""
    turns = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, dur = float(parts[3]), float(parts[4])
        turns.append({"start": start, "end": start + dur, "speaker": parts[7]})
    return turns


def write_rttm(turns: Iterable[dict], path: str | Path, uri: str = "meeting") -> None:
    lines = [
        f"SPEAKER {uri} 1 {t['start']:.3f} {t['end'] - t['start']:.3f} <NA> <NA> {t['speaker']} <NA> <NA>"
        for t in sorted(turns, key=lambda t: t["start"])
        if t["end"] > t["start"]
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_annotation(turns: Iterable[dict], uri: str = "meeting"):
    from pyannote.core import Annotation, Segment

    ann = Annotation(uri=uri)
    for k, t in enumerate(turns):
        if t["end"] > t["start"]:
            ann[Segment(float(t["start"]), float(t["end"])), f"t{k}"] = t["speaker"]
    return ann


# --------------------------------------------------------------------------- DER


def diarization_error(reference: list[dict], hypothesis: list[dict], collar: float = 0.0,
                      skip_overlap: bool = False) -> dict:
    """DER = (miss + false alarm + confusion) / total reference speech (pyannote.metrics)."""
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    d = metric(to_annotation(reference), to_annotation(hypothesis), detailed=True)
    total = d["total"] or 1e-9
    return {
        "der": float(d["diarization error rate"]),
        "miss_rate": d["missed detection"] / total,
        "false_alarm_rate": d["false alarm"] / total,
        "confusion_rate": d["confusion"] / total,
        "miss_sec": float(d["missed detection"]),
        "false_alarm_sec": float(d["false alarm"]),
        "confusion_sec": float(d["confusion"]),
        "total_ref_speech_sec": float(d["total"]),
        "collar": collar,
        "skip_overlap": skip_overlap,
        "ref_speakers": len({t["speaker"] for t in reference}),
        "hyp_speakers": len({t["speaker"] for t in hypothesis}),
    }


# --------------------------------------------------------------------------- speaker identification


def _timelines(turns: Iterable[dict]) -> dict[str, list[tuple[float, float]]]:
    acc: dict[str, list] = defaultdict(list)
    for t in turns:
        acc[t["speaker"]].append((t["start"], t["end"]))
    return {k: merge_intervals(v) for k, v in acc.items()}


def cluster_identity(reference: list[dict], hypothesis: list[dict]) -> dict[str, tuple[Optional[str], float, float]]:
    """cluster -> (majority reference speaker, overlap seconds, cluster seconds)."""
    ref_tl, hyp_tl = _timelines(reference), _timelines(hypothesis)
    out = {}
    for c, tl in hyp_tl.items():
        best, best_d = None, 0.0
        for r, rtl in ref_tl.items():
            d = intersect_length(tl, rtl)
            if d > best_d:
                best, best_d = r, d
        out[c] = (best, best_d, sum(e - s for s, e in tl))
    return out


def speaker_id_metrics(
    reference: list[dict],
    hypothesis: list[dict],
    predicted: dict[str, str],
    enrolled: Iterable[str],
) -> dict:
    """Cluster-level identification metrics.

    reference : turns labelled with real names (guests may have any non-enrolled label)
    hypothesis: diarization turns labelled with cluster ids (SPEAKER_xx)
    predicted : cluster -> output label (enrolled name, UNKNOWN_xx, or SPEAKER_xx when ID is off)
    enrolled  : names that were enrolled for this meeting
    """
    enrolled = set(enrolled)
    ident = cluster_identity(reference, hypothesis)
    counts = defaultdict(float)
    dur = defaultdict(float)
    confusion: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    rows = []
    for c, (true, _, cdur) in sorted(ident.items()):
        if true is None:
            continue  # cluster on non-speech only (false alarm) - covered by DER
        pred = predicted.get(c, c)
        pred_n = pred if pred in enrolled else UNKNOWN
        true_n = true if true in enrolled else UNENROLLED
        if true_n != UNENROLLED:
            if pred_n == true_n:
                kind = "correct_accept"
            elif pred_n == UNKNOWN:
                kind = "false_reject"
            else:
                kind = "misidentification"
        else:
            kind = "correct_reject" if pred_n == UNKNOWN else "false_accept_impostor"
        counts[kind] += 1
        dur[kind] += cdur
        confusion[true_n][pred_n] += 1
        rows.append({"cluster": c, "true": true, "predicted": pred, "decision": kind, "duration_sec": round(cdur, 2)})

    def rate(num_keys, den_keys, src):
        den = sum(src[k] for k in den_keys)
        return None if den == 0 else sum(src[k] for k in num_keys) / den

    all_keys = ["correct_accept", "false_reject", "misidentification", "correct_reject", "false_accept_impostor"]
    target = ["correct_accept", "false_reject", "misidentification"]
    impostor = ["correct_reject", "false_accept_impostor"]
    labels_true = sorted(k for k in confusion if k != UNENROLLED) + ([UNENROLLED] if UNENROLLED in confusion else [])
    labels_pred = sorted({p for r in confusion.values() for p in r if p != UNKNOWN}) + [UNKNOWN]
    return {
        "clusters_evaluated": int(sum(counts[k] for k in all_keys)),
        "counts": {k: int(counts[k]) for k in all_keys},
        "accuracy": rate(["correct_accept", "correct_reject"], all_keys, counts),
        "false_reject_rate": rate(["false_reject"], target, counts),
        "misidentification_rate": rate(["misidentification"], target, counts),
        "false_accept_rate": rate(["false_accept_impostor"], impostor, counts),
        "unknown_rejection_rate": rate(["correct_reject"], impostor, counts),
        "accuracy_time_weighted": rate(["correct_accept", "correct_reject"], all_keys, dur),
        "confusion_matrix": {
            "rows_true": labels_true,
            "cols_pred": labels_pred,
            "matrix": [[int(confusion[r].get(p, 0)) for p in labels_pred] for r in labels_true],
        },
        "per_cluster": rows,
    }


def attribution_error_rate(reference: list[dict], hypothesis_named: list[dict], enrolled: Iterable[str]) -> dict:
    """Time-based identification error (pyannote IdentificationErrorRate) where every non-enrolled
    reference speaker and every UNKNOWN_xx hypothesis label are pooled as 'UNKNOWN'."""
    from pyannote.metrics.identification import IdentificationErrorRate

    enrolled = set(enrolled)

    def pool(turns):
        return [dict(t, speaker=t["speaker"] if t["speaker"] in enrolled else UNKNOWN) for t in turns]

    metric = IdentificationErrorRate()
    d = metric(to_annotation(pool(reference)), to_annotation(pool(hypothesis_named)), detailed=True)
    total = d["total"] or 1e-9
    return {
        "attribution_error_rate": float(d["identification error rate"]),
        "miss_rate": d["missed detection"] / total,
        "false_alarm_rate": d["false alarm"] / total,
        "wrong_name_rate": d["confusion"] / total,
    }


def per_speaker_time_accuracy(reference: list[dict], hypothesis_named: list[dict], enrolled: Iterable[str]) -> dict:
    """For each reference speaker: share of their speech time labelled with the right name
    (or with any UNKNOWN_xx for non-enrolled speakers)."""
    enrolled = set(enrolled)
    ref_tl = _timelines(reference)
    hyp_tl = _timelines(hypothesis_named)
    out = {}
    for r, rtl in ref_tl.items():
        total = sum(e - s for s, e in rtl)
        if r in enrolled:
            ok = intersect_length(rtl, hyp_tl.get(r, []))
        else:
            unk = merge_intervals(iv for lab, tl in hyp_tl.items() if lab.startswith("UNKNOWN") for iv in tl)
            ok = intersect_length(rtl, unk)
        out[r] = {"speech_sec": round(total, 2), "correct_sec": round(ok, 2),
                  "accuracy": round(ok / total, 4) if total else None, "enrolled": r in enrolled}
    return out


def relabel_turns(turns: list[dict], mapping: dict[str, str]) -> list[dict]:
    return [dict(t, speaker=mapping.get(t["speaker"], t["speaker"])) for t in turns]

