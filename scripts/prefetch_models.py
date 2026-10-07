"""Download every model once (online machine), so later runs can be fully offline (HF_HUB_OFFLINE=1).

  python scripts/prefetch_models.py                  # models from config
  python scripts/prefetch_models.py --whisper large-v3 large-v3-turbo

Downloads: faster-whisper model(s), WhisperX alignment model for the language, pyannote
Community-1 (needs HF_TOKEN + accepted conditions), WeSpeaker model, NLTK punkt_tab (used by
WhisperX alignment for sentence splitting).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meeting_transcriber.config import hf_token, load_config  # noqa: E402
from meeting_transcriber.device import default_alignment_model  # noqa: E402


def step(name, fn) -> bool:
    print(f"[..] {name}", flush=True)
    try:
        fn()
        print(f"[OK] {name}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(exc).__name__}: {str(exc)[:300]}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper", nargs="*", help="faster-whisper model names (default: config asr.model)")
    ap.add_argument("--language", default=None)
    ap.add_argument("--skip", nargs="*", default=[], choices=["whisper", "align", "pyannote", "wespeaker", "nltk"])
    ap.add_argument("--allow-proxied-nltk", action="store_true",
                    help="nltk>=3.10 refuses downloads through an HTTP(S) proxy (SSRF guard); set this only if "
                         "your corporate proxy is trusted")
    args = ap.parse_args()
    cfg = load_config()
    cache = cfg.model_cache_dir
    lang = args.language or (cfg.asr.language if cfg.asr.language != "auto" else "ko")
    ok = True

    if "whisper" not in args.skip:
        from faster_whisper.utils import download_model

        for name in args.whisper or [cfg.asr.model]:
            ok &= step(f"faster-whisper {name}", lambda n=name: download_model(n, cache_dir=cache))

    if "align" not in args.skip:
        align_name = cfg.alignment.model_name or default_alignment_model(lang)

        def align():
            from huggingface_hub import snapshot_download

            if align_name is None:
                raise RuntimeError(f"no alignment model for {lang}")
            snapshot_download(align_name, cache_dir=cache)

        ok &= step(f"alignment {align_name}", align)

    if "pyannote" not in args.skip:
        def pyannote():
            from pyannote.audio import Pipeline

            if not hf_token():
                raise RuntimeError("HF_TOKEN not set (.env)")
            p = Pipeline.from_pretrained(cfg.diarization.model, token=hf_token(), cache_dir=cache)
            if p is None:
                raise RuntimeError("download refused - accept the model conditions on hf.co first")

        ok &= step(f"pyannote {cfg.diarization.model}", pyannote)

    if "wespeaker" not in args.skip:
        def wes():
            from meeting_transcriber.speaker_id import WeSpeakerEmbedder

            WeSpeakerEmbedder(cfg.speaker_id.wespeaker_model, "cpu", cache).load()

        ok &= step(f"wespeaker {cfg.speaker_id.wespeaker_model}", wes)

    if "nltk" not in args.skip:
        def punkt():
            import os

            if args.allow_proxied_nltk:
                os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"
            import nltk

            if not nltk.download("punkt_tab", quiet=True):
                raise RuntimeError("nltk download failed")

        ok &= step("nltk punkt_tab", punkt)

    print("완료" if ok else "일부 실패 - 위 로그 확인")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
