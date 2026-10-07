"""Create an evaluation scenario (A-G) by mixing single-speaker clips with exact ground truth.

  python scripts/make_synthetic_meeting.py --clips data/eval_clips --preset C --out data/eval/C
  python scripts/make_synthetic_meeting.py --clips data/eval_clips --preset D --out data/eval/D \
         --unenrolled 게스트1 게스트2
  python scripts/make_synthetic_meeting.py --clips data/eval_clips --preset G --out data/eval/G \
         --speakers 김철수 김민수 김영수 박철수 이철수     # similar voices: pick them yourself

Writes <out>/mixture.wav, reference.rttm, reference.txt (when every clip has a transcript),
scenario.yaml (manifest entry for scripts/evaluate.py).
IMPORTANT: use clips that were NOT used for enrollment, otherwise identification is overestimated.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meeting_transcriber.audio.io import write_wav  # noqa: E402
from meeting_transcriber.evaluation.metrics import write_rttm  # noqa: E402
from meeting_transcriber.evaluation.synthetic import PRESETS, SynthConfig, load_clip_bank, synthesize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="<dir>/<speaker>/*.wav clip bank")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--speakers", nargs="*")
    ap.add_argument("--num-speakers", type=int)
    ap.add_argument("--unenrolled", nargs="*", default=[], help="speakers that must NOT be enrolled (guests)")
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--overlap-prob", type=float)
    ap.add_argument("--interjection-prob", type=float)
    ap.add_argument("--snr", type=float, default=None, help="add white noise at this SNR (dB)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    params = dict(PRESETS.get(args.preset, {}))
    if args.num_speakers:
        params["num_speakers"] = args.num_speakers
    if args.overlap_prob is not None:
        params["overlap_prob"] = args.overlap_prob
    if args.interjection_prob is not None:
        params["interjection_prob"] = args.interjection_prob
    speakers = args.speakers
    if speakers:
        params["num_speakers"] = None
    with tempfile.TemporaryDirectory() as tmp:
        bank = load_clip_bank(args.clips, tmp)
        if args.unenrolled:
            missing = [u for u in args.unenrolled if u not in bank]
            if missing:
                print(f"clip bank 에 없는 게스트: {missing}")
                return 1
            if speakers is None and params.get("num_speakers"):
                enrolled_pool = sorted(s for s in bank if s not in args.unenrolled)
                need = params["num_speakers"] - len(args.unenrolled)
                import random

                speakers = sorted(random.Random(args.seed).sample(enrolled_pool, need)) + list(args.unenrolled)
                params["num_speakers"] = None
        cfg = SynthConfig(speakers=speakers, target_duration_sec=args.duration, noise_snr_db=args.snr,
                          seed=args.seed, **{k: v for k, v in params.items() if k in SynthConfig.__dataclass_fields__})
        res = synthesize(bank, cfg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_wav(out / "mixture.wav", res.audio, 16000)
    write_rttm(res.turns, out / "reference.rttm")
    if res.text:
        (out / "reference.txt").write_text(res.text + "\n", encoding="utf-8")
    enrolled = [s for s in res.speakers if s not in args.unenrolled]
    entry = {
        "id": args.preset or out.name,
        "audio": str(out / "mixture.wav"),
        "reference_rttm": str(out / "reference.rttm"),
        "reference_text": str(out / "reference.txt") if res.text else None,
        "speaker_mode": "FIXED",
        "num_speakers": len(res.speakers),
        "enrolled": enrolled,
        "unenrolled": list(args.unenrolled),
    }
    (out / "scenario.yaml").write_text(yaml.safe_dump(entry, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"{out}: {len(res.audio) / 16000:.1f}s, speakers={res.speakers}, turns={len(res.turns)}, "
          f"transcript={'yes' if res.text else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
