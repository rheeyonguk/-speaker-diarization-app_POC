"""Gradio entry point:  python app/main.py  [--host 127.0.0.1] [--port 7860] [--config config/local.yaml]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from meeting_transcriber.config import load_config  # noqa: E402
from meeting_transcriber.errors import UserFacingError  # noqa: E402
from meeting_transcriber.logging_utils import setup_logging  # noqa: E402
from meeting_transcriber.ui import launch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="한국어 회의 다화자 전사 PoC UI")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--config")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)
    try:
        cfg = load_config(args.config)
    except UserFacingError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2
    launch(cfg, server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
