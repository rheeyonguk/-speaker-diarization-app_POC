"""Layer 9: combine WhisperX word timestamps with diarization.

Per word:
  * primary speaker  = max time-intersection with the *exclusive* diarization (one speaker at a time);
                       words outside every turn take the nearest turn within ``nearest_max_gap_sec``
  * overlap          = >= overlap_min_ratio of the word lies in an overlap region of the
                       *overlap-aware* diarization
  * secondary_speakers = other speakers active (in the overlap-aware diarization) for at least
                       overlap_min_ratio of the word
Utterances are split wherever the primary speaker changes, so one WhisperX segment holding a quick
exchange ("네." / "맞습니다.") becomes several speaker-attributed utterances.
"""

from __future__ import annotations

import bisect
import math
from typing import Optional

import numpy as np

from ..config import AssignmentConfig
from ..diarization.types import DiarizationResult, IntervalSet, Turn

NO_SPACE_LANGUAGES = {"ja", "zh", "th", "lo", "my"}


class _TurnIndex:
    def __init__(self, turns: list[Turn]):
        self.turns = sorted(turns, key=lambda t: t.start)
        self.starts = np.array([t.start for t in self.turns], dtype=np.float64)
        self.max_dur = max((t.duration for t in self.turns), default=0.0)

    def overlapping(self, s: float, e: float) -> dict[str, float]:
        if not self.turns:
            return {}
        lo = bisect.bisect_left(self.starts, s - self.max_dur)
        hi = bisect.bisect_right(self.starts, e)
        out: dict[str, float] = {}
        for t in self.turns[lo:hi]:
            inter = min(t.end, e) - max(t.start, s)
            if inter > 0:
                out[t.speaker] = out.get(t.speaker, 0.0) + inter
        return out

    def nearest(self, s: float, e: float, max_gap: float) -> Optional[str]:
        best, best_gap = None, max_gap
        lo = bisect.bisect_left(self.starts, s - self.max_dur - max_gap)
        hi = bisect.bisect_right(self.starts, e + max_gap)
        for t in self.turns[lo:hi]:
            gap = max(t.start - e, s - t.end, 0.0)
            if gap <= best_gap:
                best, best_gap = t.speaker, gap
        return best


def _num(v) -> Optional[float]:
    """float or None (WhisperX can emit numpy floats and NaN for unalignable tokens)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _word_interval(w: dict) -> Optional[tuple[float, float]]:
    s = _num(w.get("start"))
    if s is None:
        return None
    e = _num(w.get("end"))
    if e is None or e < s:
        e = s
    if e - s < 0.02:
        e = s + 0.02
    return float(s), float(e)


def assign_words(
    aligned: dict,
    diar: DiarizationResult,
    cfg: AssignmentConfig,
) -> list[dict]:
    """Returns flat list of word dicts with speaker fields, grouped info kept via 'segment_index'."""
    excl = _TurnIndex(diar.exclusive_speaker_diarization)
    full = _TurnIndex(diar.speaker_diarization)
    overlap = IntervalSet(diar.overlap_regions)
    words_out: list[dict] = []
    for si, seg in enumerate(aligned.get("segments", [])):
        words = seg.get("words") or []
        if not words:
            # alignment unavailable for this segment -> treat the whole segment as one "word"
            words = [{"word": seg.get("text", "").strip(), "start": seg.get("start"), "end": seg.get("end"),
                      "_segment_level": True}]
        for w in words:
            iv = _word_interval(w)
            rec = {
                "word": str(w.get("word", "")).strip(),
                "sep": w.get("sep"),  # original separator before this word (Azure MAI display text)
                "start": iv[0] if iv else None,
                "end": _num(w.get("end")) if iv else None,
                "score": None if _num(w.get("score")) is None else round(_num(w.get("score")), 4),
                "segment_index": si,
                "speaker_cluster": None,
                "speaker_source": None,
                "overlap": False,
                "secondary_clusters": [],
            }
            if w.get("_segment_level"):
                rec["segment_level"] = True
            if iv:
                s, e = iv
                dur = e - s
                inter = excl.overlapping(s, e)
                if inter:
                    rec["speaker_cluster"] = max(inter.items(), key=lambda kv: kv[1])[0]
                    rec["speaker_source"] = "exclusive"
                else:
                    near = excl.nearest(s, e, cfg.nearest_max_gap_sec)
                    if near is not None:
                        rec["speaker_cluster"] = near
                        rec["speaker_source"] = "nearest"
                ov_ratio = overlap.overlap_length(s, e) / dur
                rec["overlap"] = ov_ratio >= cfg.overlap_min_ratio
                active = full.overlapping(s, e)
                rec["secondary_clusters"] = sorted(
                    spk for spk, d in active.items()
                    if spk != rec["speaker_cluster"] and d >= cfg.overlap_min_ratio * dur
                )
            words_out.append(rec)

    # words without timestamps inherit their neighbour's speaker (previous, else next)
    last = None
    for rec in words_out:
        if rec["speaker_cluster"] is None and rec["start"] is None and last is not None:
            rec["speaker_cluster"], rec["speaker_source"] = last, "inherited"
        if rec["speaker_cluster"] is not None:
            last = rec["speaker_cluster"]
    nxt = None
    for rec in reversed(words_out):
        if rec["speaker_cluster"] is None and rec["start"] is None and nxt is not None:
            rec["speaker_cluster"], rec["speaker_source"] = nxt, "inherited"
        if rec["speaker_cluster"] is not None:
            nxt = rec["speaker_cluster"]
    return words_out


def _join_words(words: list[dict], joiner: str) -> str:
    """Join words using their original separators when the ASR provides them (keeps the service's
    spacing and punctuation), otherwise the language default (space, or nothing for CJK)."""
    parts = []
    for i, w in enumerate(words):
        if i:
            sep = w.get("sep")
            parts.append(joiner if sep is None else (sep if sep.strip() or sep == "" else " "))
        parts.append(w["word"])
    return "".join(parts).strip()


def build_utterances(
    aligned: dict,
    diar: DiarizationResult,
    labels: dict,
    similarities: dict,
    statuses: dict,
    cfg: AssignmentConfig,
    language: str = "ko",
) -> list[dict]:
    """Group speaker-attributed words into utterances.

    ``labels``: cluster -> display label, ``similarities``: cluster -> cosine or None,
    ``statuses``: cluster -> identification status.
    """
    words = assign_words(aligned, diar, cfg)
    segments = aligned.get("segments", [])
    joiner = "" if language in NO_SPACE_LANGUAGES else " "

    def display(cluster: Optional[str]) -> str:
        return "UNASSIGNED" if cluster is None else labels.get(cluster, cluster)

    utterances: list[dict] = []
    cur: Optional[dict] = None
    for rec in words:
        key = (rec["segment_index"], rec["speaker_cluster"])
        if cur is None or cur["_key"] != key:
            if cur is not None:
                utterances.append(cur)
            seg = segments[rec["segment_index"]]
            cur = {"_key": key, "_words": [], "_seg": seg}
        cur["_words"].append(rec)
    if cur is not None:
        utterances.append(cur)

    out = []
    for u in utterances:
        ws = u["_words"]
        seg = u["_seg"]
        cluster = u["_key"][1]
        starts = [w["start"] for w in ws if w["start"] is not None]
        ends = [w["end"] for w in ws if w.get("end") is not None]
        start = min(starts) if starts else _num(seg.get("start"))
        end = max(ends) if ends else _num(seg.get("end"))
        if start is None or end is None:
            continue
        scores = [w["score"] for w in ws if w.get("score") is not None]
        secondary = sorted({c for w in ws for c in w["secondary_clusters"]})
        text = _join_words([w for w in ws if w["word"]], joiner)
        if not text:
            continue
        out.append({
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "speaker_cluster": cluster,
            "identified_speaker": display(cluster),
            "speaker_status": statuses.get(cluster, "unassigned" if cluster is None else "not_identified"),
            "speaker_similarity": (round(float(similarities[cluster]), 4)
                                   if cluster is not None and similarities.get(cluster) is not None else None),
            "text": text,
            "overlap": any(w["overlap"] for w in ws),
            "secondary_speakers": [display(c) for c in secondary],
            "speaker_assignment": "nearest" if all(w["speaker_source"] == "nearest" for w in ws) else "diarization",
            "asr_avg_logprob": (round(_num(seg["avg_logprob"]), 4) if _num(seg.get("avg_logprob")) is not None else None),
            "word_confidence": round(float(np.mean(scores)), 4) if scores else None,
            "words": [
                {
                    "word": w["word"],
                    "start": None if w["start"] is None else round(float(w["start"]), 3),
                    "end": None if w.get("end") is None else round(float(w["end"]), 3),
                    "score": w.get("score"),
                    "speaker_cluster": w["speaker_cluster"],
                    "speaker": display(w["speaker_cluster"]),
                    "overlap": w["overlap"],
                    "secondary_speakers": [display(c) for c in w["secondary_clusters"]],
                }
                for w in ws if not w.get("segment_level")
            ],
        })
    out.sort(key=lambda u: (u["start"], u["end"]))
    return out
