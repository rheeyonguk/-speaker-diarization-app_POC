"""TXT / JSON / CSV / SRT writers. Inputs are the plain-dict transcript document (no embeddings)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

CSV_COLUMNS = ["start_time", "end_time", "speaker", "speaker_cluster", "confidence", "text"]


def fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_srt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_clock(seconds: float) -> str:
    """HH:MM:SS.mmm for CSV (sortable, Excel friendly)."""
    seconds = max(0.0, float(seconds))
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def merge_for_reading(segments: list[dict], max_gap: float) -> list[dict]:
    """Merge consecutive utterances of the same speaker separated by <= max_gap seconds (TXT only)."""
    merged: list[dict] = []
    for seg in segments:
        if (
            merged
            and merged[-1]["identified_speaker"] == seg["identified_speaker"]
            and seg["start"] - merged[-1]["end"] <= max_gap
        ):
            m = merged[-1]
            m["end"] = max(m["end"], seg["end"])
            m["text"] = f"{m['text']} {seg['text']}".strip()
            m["overlap"] = m["overlap"] or seg.get("overlap", False)
            m["secondary_speakers"] = sorted(set(m["secondary_speakers"]) | set(seg.get("secondary_speakers", [])))
        else:
            merged.append({
                "start": seg["start"],
                "end": seg["end"],
                "identified_speaker": seg["identified_speaker"],
                "text": seg["text"],
                "overlap": seg.get("overlap", False),
                "secondary_speakers": list(seg.get("secondary_speakers", [])),
            })
    return merged


def render_txt(doc: dict, merge_gap: float = 1.0) -> str:
    lines = [
        "회의록 (자동 생성 - 검토 필요)",
        f"파일: {doc.get('file')}",
        f"길이: {fmt_hms(doc.get('duration', 0))}   언어: {doc.get('language')}   화자 수: {doc.get('speaker_count')}",
    ]
    speakers = doc.get("speakers", [])
    if speakers:
        lines.append("화자: " + ", ".join(s["label"] for s in speakers))
    lines.append("=" * 60)
    lines.append("")
    for seg in merge_for_reading(doc.get("segments", []), merge_gap):
        lines.append(f"{fmt_hms(seg['start'])} - {fmt_hms(seg['end'])}")
        speaker = seg["identified_speaker"]
        if seg.get("overlap"):
            others = ", ".join(seg.get("secondary_speakers") or [])
            speaker += f"  (동시발화{': ' + others if others else ''})"
        lines.append(speaker)
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_srt(doc: dict) -> str:
    blocks = []
    for i, seg in enumerate(doc.get("segments", []), start=1):
        end = seg["end"] if seg["end"] > seg["start"] else seg["start"] + 0.5
        blocks.append(f"{i}\n{fmt_srt(seg['start'])} --> {fmt_srt(end)}\n[{seg['identified_speaker']}] {seg['text']}\n")
    return "\n".join(blocks)


def write_csv(doc: dict, path: Path) -> None:
    # utf-8-sig: Excel on Windows opens Korean text correctly only with a BOM
    with open(path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for seg in doc.get("segments", []):
            writer.writerow({
                "start_time": fmt_clock(seg["start"]),
                "end_time": fmt_clock(seg["end"]),
                "speaker": seg["identified_speaker"],
                "speaker_cluster": seg.get("speaker_cluster") or "",
                # speaker attribution confidence = cosine similarity to the matched voice profile
                "confidence": "" if seg.get("speaker_similarity") is None else f"{seg['speaker_similarity']:.4f}",
                "text": seg["text"],
            })


def strip_words(doc: dict) -> dict:
    out = dict(doc)
    out["segments"] = [{k: v for k, v in s.items() if k != "words"} for s in doc.get("segments", [])]
    return out


def write_outputs(doc: dict, out_dir: Path, stem: str, formats: list[str], merge_gap: float = 1.0,
                  include_words: bool = True) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if "txt" in formats:
        p = out_dir / f"{stem}.txt"
        p.write_text(render_txt(doc, merge_gap), encoding="utf-8")
        paths["txt"] = p
    if "json" in formats:
        p = out_dir / f"{stem}.json"
        payload = doc if include_words else strip_words(doc)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["json"] = p
    if "csv" in formats:
        p = out_dir / f"{stem}.csv"
        write_csv(doc, p)
        paths["csv"] = p
    if "srt" in formats:
        p = out_dir / f"{stem}.srt"
        p.write_text(render_srt(doc), encoding="utf-8")
        paths["srt"] = p
    return paths
