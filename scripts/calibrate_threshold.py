"""Estimate speaker_match_threshold from the locally enrolled voice profiles.

  python scripts/calibrate_threshold.py
  python scripts/calibrate_threshold.py --scores outputs/eval_*/similarity_scores.json   # from evaluate.py

Enrollment-only calibration (leave-one-out over enrollment chunks) needs >= 2 speakers with
>= 2 chunks each. It is optimistic (enrollment audio is clean); prefer scores collected on
labelled meetings with scripts/evaluate.py. Nothing is written unless --write is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meeting_transcriber.config import load_config  # noqa: E402
from meeting_transcriber.evaluation.calibration import calibrate, enrollment_trials  # noqa: E402
from meeting_transcriber.speaker_id import SpeakerProfileStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="*", help="similarity_scores.json files from scripts/evaluate.py")
    ap.add_argument("--write", help="write recommended threshold into this YAML override file")
    args = ap.parse_args()
    cfg = load_config()

    if args.scores:
        target, nontarget = [], []
        for f in args.scores:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            target += d["target"]
            nontarget += d["nontarget"]
        source = "labelled meetings"
    else:
        store = SpeakerProfileStore(cfg.enrollment_dir)
        profiles = []
        for p in store.list_profiles():
            emb = np.load(store.root / p.speaker_id / "embeddings.npy")
            profiles.append((p.name, emb, np.asarray(p.chunk_keep, dtype=bool)))
        models = {p.model_id for p in store.list_profiles()}
        if len(models) > 1:
            print(f"경고: 서로 다른 임베딩 모델의 프로필이 섞여 있습니다: {models}")
        target, nontarget = enrollment_trials(profiles)
        source = "enrollment leave-one-out (optimistic)"

    if len(target) < 2 or len(nontarget) < 2:
        print("점수가 부족합니다. 2명 이상, 각 2개 이상 구간(긴 샘플 또는 여러 샘플)을 등록하거나 --scores 를 사용하세요.")
        return 1
    res = calibrate(target, nontarget)
    print(f"source: {source}")
    print(json.dumps(res, indent=2))
    print(f"\n현재 설정 threshold = {cfg.speaker_id.match_threshold}")
    print(f"권장(보수적, max(EER thr, FAR1% thr)) = {res['recommended_threshold']:.3f}")
    if args.write:
        out = Path(args.write)
        out.write_text(f"speaker_id:\n  match_threshold: {res['recommended_threshold']:.3f}\n", encoding="utf-8")
        print(f"→ {out} 저장. MT_CONFIG={out} 로 적용하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
