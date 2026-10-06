"""Build synthetic multi-speaker meetings with exact ground truth from single-speaker clips.

Real labelled Korean meeting recordings are the gold standard. Until those exist, mixing
per-person recordings (NOT the ones used for enrollment) gives controllable test scenarios:
speaker count, overlap rate, short interjections, similar voices, unenrolled guests.

Clip bank layout:  <clips_dir>/<speaker name>/*.wav   (+ optional same-stem .txt transcript)
                   <clips_dir>/<speaker name>/short/*.wav   (optional interjections: "네.", "맞습니다.")
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..audio.io import convert_to_pcm_wav, read_wav
from ..audio.preprocess import peak_normalize

AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".mp4", ".mov")

# Scenario presets (A-G) from the test plan
PRESETS = {
    "A": dict(num_speakers=2, overlap_prob=0.05, interjection_prob=0.05),
    "B": dict(num_speakers=5, overlap_prob=0.05, interjection_prob=0.1),
    "C": dict(num_speakers=10, overlap_prob=0.05, interjection_prob=0.1),
    "D": dict(num_speakers=10, overlap_prob=0.05, interjection_prob=0.1),  # + 2 unenrolled guests
    "E": dict(num_speakers=4, overlap_prob=0.35, interjection_prob=0.1),
    "F": dict(num_speakers=4, overlap_prob=0.05, interjection_prob=0.6),
    "G": dict(num_speakers=5, overlap_prob=0.05, interjection_prob=0.1),   # choose similar voices
}


@dataclass
class Clip:
    speaker: str
    path: str
    audio: np.ndarray
    text: Optional[str] = None
    short: bool = False

    @property
    def duration(self) -> float:
        return len(self.audio) / 16000


@dataclass
class SynthConfig:
    speakers: Optional[list] = None
    num_speakers: Optional[int] = None
    target_duration_sec: float = 300.0
    max_turn_sec: float = 15.0
    gap_range: tuple = (0.2, 1.0)
    overlap_prob: float = 0.0
    overlap_range: tuple = (0.3, 1.5)
    interjection_prob: float = 0.0
    interjection_sec: tuple = (0.4, 1.0)
    gain_db_jitter: float = 3.0
    noise_snr_db: Optional[float] = None
    seed: int = 0


@dataclass
class SynthResult:
    audio: np.ndarray
    turns: list = field(default_factory=list)      # reference [{"start","end","speaker"}]
    text: Optional[str] = None
    speakers: list = field(default_factory=list)


def load_clip_bank(clips_dir: str | Path, workdir: str | Path) -> dict[str, list[Clip]]:
    clips_dir, workdir = Path(clips_dir), Path(workdir)
    bank: dict[str, list[Clip]] = {}
    for spk_dir in sorted(p for p in clips_dir.iterdir() if p.is_dir()):
        clips = []
        for sub, is_short in ((spk_dir, False), (spk_dir / "short", True)):
            if not sub.is_dir():
                continue
            for f in sorted(sub.iterdir()):
                if f.suffix.lower() not in AUDIO_EXT:
                    continue
                wav = convert_to_pcm_wav(f, workdir / spk_dir.name / ("short_" if is_short else "") / f"{f.stem}.wav")
                audio, _ = read_wav(wav)
                txt = f.with_suffix(".txt")
                clips.append(Clip(spk_dir.name, str(f), audio,
                                  txt.read_text(encoding="utf-8").strip() if txt.exists() else None, is_short))
        if clips:
            bank[spk_dir.name] = clips
    return bank


def _excerpt(rng: random.Random, audio: np.ndarray, sec: float) -> np.ndarray:
    n = int(sec * 16000)
    if len(audio) <= n:
        return audio
    start = rng.randint(0, len(audio) - n)
    return audio[start:start + n]


def synthesize(bank: dict[str, list[Clip]], cfg: SynthConfig) -> SynthResult:
    rng = random.Random(cfg.seed)
    speakers = list(cfg.speakers or sorted(bank))
    missing = [s for s in speakers if s not in bank]
    if missing:
        raise ValueError(f"no clips for speakers: {missing}")
    if cfg.num_speakers:
        if cfg.num_speakers > len(speakers):
            raise ValueError(f"need {cfg.num_speakers} speakers, clip bank has {len(speakers)}")
        speakers = sorted(rng.sample(speakers, cfg.num_speakers))

    placed: list[tuple[float, np.ndarray, str, Optional[str]]] = []  # (start, audio, speaker, text)
    text_complete = True
    t, last = 0.5, None
    used_order = 0
    while t < cfg.target_duration_sec:
        # make sure everybody speaks at least once early on
        if used_order < len(speakers):
            spk = speakers[used_order]
            used_order += 1
        else:
            spk = rng.choice([s for s in speakers if s != last] or speakers)
        long_clips = [c for c in bank[spk] if not c.short] or bank[spk]
        clip = rng.choice(long_clips)
        if clip.duration > cfg.max_turn_sec:
            audio, text = _excerpt(rng, clip.audio, cfg.max_turn_sec), None
            text_complete = False
        else:
            audio, text = clip.audio, clip.text
            text_complete &= clip.text is not None
        # never overlap a speaker with their own previous clip
        if placed and placed[-1][2] != spk and rng.random() < cfg.overlap_prob:
            prev_start, prev_audio = placed[-1][0], placed[-1][1]
            prev_end = prev_start + len(prev_audio) / 16000
            start = max(prev_start + 0.5, prev_end - rng.uniform(*cfg.overlap_range))
        else:
            start = t + rng.uniform(*cfg.gap_range)
        gain = 10 ** (rng.uniform(-cfg.gain_db_jitter, cfg.gain_db_jitter) / 20)
        placed.append((start, audio * gain, spk, text))
        end = start + len(audio) / 16000

        if len(speakers) > 1 and rng.random() < cfg.interjection_prob:
            other = rng.choice([s for s in speakers if s != spk])
            shorts = [c for c in bank[other] if c.short]
            if shorts:
                ic = rng.choice(shorts)
                i_audio, i_text = ic.audio, ic.text
                text_complete &= ic.text is not None
            else:
                i_audio = _excerpt(rng, rng.choice(bank[other]).audio, rng.uniform(*cfg.interjection_sec))
                i_text = None
                text_complete = False
            if rng.random() < 0.5:   # during the turn (overlapping back-channel)
                i_start = rng.uniform(start + 0.3, max(start + 0.3, end - len(i_audio) / 16000))
            else:                    # right after the turn
                i_start = end + rng.uniform(0.1, 0.4)
            placed.append((i_start, i_audio * gain, other, i_text))
            end = max(end, i_start + len(i_audio) / 16000)
        t = max(t, end)
        last = spk

    total = max(s + len(a) / 16000 for s, a, _, _ in placed) + 0.5
    mix = np.zeros(int(total * 16000) + 1, dtype=np.float32)
    turns = []
    for s, a, spk, _ in placed:
        i = int(s * 16000)
        mix[i:i + len(a)] += a
        turns.append({"start": round(s, 3), "end": round(s + len(a) / 16000, 3), "speaker": spk})
    if cfg.noise_snr_db is not None:
        nrng = np.random.default_rng(cfg.seed)
        sig_pow = float(np.mean(mix[np.abs(mix) > 1e-4] ** 2)) if np.any(np.abs(mix) > 1e-4) else 1e-6
        noise_pow = sig_pow / (10 ** (cfg.noise_snr_db / 10))
        mix = mix + nrng.normal(0, np.sqrt(noise_pow), size=mix.shape).astype(np.float32)
    mix, _ = peak_normalize(mix, -1.0)
    text = None
    if text_complete:
        text = " ".join(tx for _, _, _, tx in sorted(placed, key=lambda p: p[0]) if tx)
    return SynthResult(mix, sorted(turns, key=lambda x: x["start"]), text, speakers)
