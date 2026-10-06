"""Evaluate STT (CER/WER), diarization (DER: miss / false alarm / confusion) and speaker
identification (accuracy, false accept, false reject, unknown rejection, confusion matrix) over a
manifest of scenarios, comparing matching modes (as_run / flexible / strict_one_to_one / argmax).

  python scripts/evaluate.py --manifest tests/scenarios/manifest.example.yaml
  python scripts/evaluate.py --manifest m.yaml --reuse        # use hypothesis_json if present

Manifest entry keys: id, audio, reference_rttm, reference_text (optional), speaker_mode,
num_speakers / min_speakers / max_speakers, enrolled (list, optional), hypothesis_json (optional).
Results: outputs/eval_<timestamp>/{report.json, report.md, similarity_scores.json}
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meeting_transcriber.config import load_config  # noqa: E402
from meeting_transcriber.errors import UserFacingError  # noqa: E402
from meeting_transcriber.evaluation.calibration import summarize  # noqa: E402
from meeting_transcriber.evaluation.metrics import read_rttm  # noqa: E402
from meeting_transcriber.evaluation.report import evaluate_document, markdown_summary  # noqa: E402
from meeting_transcriber.logging_utils import setup_logging  # noqa: E402


def _p(base: Path, v):
    if not v:
        return None
    p = Path(v)
    return p if p.is_absolute() else (base / p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--reuse", action="store_true", help="use hypothesis_json instead of re-running the pipeline")
    ap.add_argument("--only", nargs="*", help="scenario ids to run")
    args = ap.parse_args()
    setup_logging("WARNING")
    cfg = load_config()
    manifest_path = Path(args.manifest)
    base = ROOT
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    scenarios = data.get("scenarios", data if isinstance(data, list) else [])

    from meeting_transcriber.pipeline import MeetingTranscriber, RunRequest

    transcriber = MeetingTranscriber(cfg)
    out_dir = cfg.output_dir / f"eval_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results, all_t, all_n = [], [], []
    for sc in scenarios:
        if args.only and sc["id"] not in args.only:
            continue
        print(f"== {sc['id']}")
        try:
            hyp = _p(base, sc.get("hypothesis_json"))
            if args.reuse and hyp and hyp.exists():
                doc = json.loads(hyp.read_text(encoding="utf-8"))
            else:
                audio = _p(base, sc["audio"])
                if not audio or not audio.exists():
                    print(f"   skip: audio not found ({sc.get('audio')})")
                    continue
                res = transcriber.run(RunRequest(
                    input_path=str(audio),
                    speaker_mode=sc.get("speaker_mode"),
                    num_speakers=sc.get("num_speakers"),
                    min_speakers=sc.get("min_speakers"),
                    max_speakers=sc.get("max_speakers"),
                    language=sc.get("language"),
                ))
                doc = res.document
                print(f"   output: {res.job_dir}")
            ref_turns = read_rttm(_p(base, sc["reference_rttm"]))
            ref_text_path = _p(base, sc.get("reference_text"))
            ref_text = ref_text_path.read_text(encoding="utf-8") if ref_text_path and ref_text_path.exists() else None
            r = evaluate_document(doc, ref_turns, ref_text, sc.get("enrolled"))
            r["id"] = sc["id"]
            all_t += r["similarity_scores"]["target"]
            all_n += r["similarity_scores"]["nontarget"]
            results.append(r)
        except UserFacingError as exc:
            print(f"   failed: {exc}")
    if not results:
        print("평가된 시나리오가 없습니다.")
        return 1
    (out_dir / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md = markdown_summary(results)
    cal = summarize(all_t, all_n)
    if cal:
        md += (f"\n화자 식별 점수 calibration (클러스터 기준): EER={cal['eer']:.3f} @thr={cal['eer_threshold']:.3f}, "
               f"FAR1% thr={cal['threshold_far_0.01']:.3f}, 권장 threshold={cal['recommended_threshold']:.3f}\n")
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "similarity_scores.json").write_text(json.dumps({"target": all_t, "nontarget": all_n}), encoding="utf-8")
    print(md)
    print(f"report: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
