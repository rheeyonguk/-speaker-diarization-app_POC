"""Library-agnostic diarization result (both overlap-aware and exclusive views)."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def intersect_length(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Total intersection between two *merged, sorted* interval lists."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


class IntervalSet:
    """Merged, sorted, non-overlapping intervals with O(log n) range queries (long meetings have
    thousands of overlap regions and tens of thousands of words)."""

    def __init__(self, intervals: Iterable[tuple[float, float]]):
        self.intervals = merge_intervals(intervals)
        self.ends = [e for _, e in self.intervals]

    def window(self, s: float, e: float) -> list[tuple[float, float]]:
        """Intervals intersecting [s, e)."""
        lo = bisect.bisect_right(self.ends, s)
        out = []
        for iv in self.intervals[lo:]:
            if iv[0] >= e:
                break
            out.append(iv)
        return out

    def overlap_length(self, s: float, e: float) -> float:
        return sum(min(b, e) - max(a, s) for a, b in self.window(s, e))


def subtract_intervals(base: tuple[float, float], holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """``base`` minus a merged, sorted list of ``holes``."""
    s0, e0 = base
    pieces = []
    cur = s0
    for hs, he in holes:
        if he <= cur:
            continue
        if hs >= e0:
            break
        if hs > cur:
            pieces.append((cur, min(hs, e0)))
        cur = max(cur, he)
        if cur >= e0:
            break
    if cur < e0:
        pieces.append((cur, e0))
    return [(a, b) for a, b in pieces if b - a > 1e-6]


@dataclass
class DiarizationResult:
    # Overlap-preserving diarization (pyannote ``speaker_diarization``). Never discarded.
    speaker_diarization: list[Turn]
    # One speaker at a time (pyannote ``exclusive_speaker_diarization``), used for transcript attribution.
    exclusive_speaker_diarization: list[Turn]
    exclusive_available: bool = True
    params: dict = field(default_factory=dict)
    model: str = ""

    def __post_init__(self) -> None:
        self.speaker_diarization = sorted(self.speaker_diarization, key=lambda t: (t.start, t.end))
        self.exclusive_speaker_diarization = sorted(self.exclusive_speaker_diarization, key=lambda t: (t.start, t.end))
        self._overlap: list[tuple[float, float]] | None = None

    @property
    def labels(self) -> list[str]:
        return sorted({t.speaker for t in self.speaker_diarization} | {t.speaker for t in self.exclusive_speaker_diarization})

    @property
    def num_speakers(self) -> int:
        return len(self.labels)

    def speaker_timeline(self, speaker: str) -> list[tuple[float, float]]:
        return merge_intervals((t.start, t.end) for t in self.speaker_diarization if t.speaker == speaker)

    @property
    def overlap_regions(self) -> list[tuple[float, float]]:
        """Regions where >= 2 speakers are simultaneously active (from the overlap-aware diarization)."""
        if self._overlap is None:
            events = []
            for spk in self.labels:
                for s, e in self.speaker_timeline(spk):
                    events.append((s, 1))
                    events.append((e, -1))
            events.sort(key=lambda x: (x[0], x[1]))
            regions, active, start = [], 0, None
            for t, delta in events:
                prev = active
                active += delta
                if prev < 2 <= active:
                    start = t
                elif prev >= 2 > active and start is not None:
                    if t > start:
                        regions.append((start, t))
                    start = None
            self._overlap = merge_intervals(regions)
        return self._overlap

    def speech_duration(self, speaker: str) -> float:
        return sum(e - s for s, e in self.speaker_timeline(speaker))

    def cooccurrence(self, a: str, b: str) -> float:
        """Seconds during which speakers ``a`` and ``b`` talk simultaneously."""
        return intersect_length(self.speaker_timeline(a), self.speaker_timeline(b))

    def first_appearance(self, speaker: str) -> float:
        times = [t.start for t in self.speaker_diarization if t.speaker == speaker]
        return min(times) if times else float("inf")

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "params": self.params,
            "labels": self.labels,
            "exclusive_available": self.exclusive_available,
            "speaker_diarization": [_turn_dict(t) for t in self.speaker_diarization],
            "exclusive_speaker_diarization": [_turn_dict(t) for t in self.exclusive_speaker_diarization],
            "overlap_regions": [{"start": round(s, 3), "end": round(e, 3)} for s, e in self.overlap_regions],
        }


def _turn_dict(t: Turn) -> dict:
    return {"start": round(t.start, 3), "end": round(t.end, 3), "speaker": t.speaker}
