"""Logging with secret redaction.

Secrets must never reach a log line. ``SecretRedactingFilter`` masks the current HF_TOKEN and
AZURE_SPEECH_KEY values and anything that looks like a Hugging Face token (``hf_...``) in every
record, including records emitted by third-party libraries attached to the root logger.
"""

from __future__ import annotations

import logging
import os
import re

_HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{8,}")
_MASK = "hf_***REDACTED***"


def redact(text: str) -> str:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "AZURE_SPEECH_KEY"):
        secret = os.environ.get(var)
        if secret and len(secret) >= 8:
            text = text.replace(secret, "***REDACTED***")
    return _HF_TOKEN_PATTERN.sub(_MASK, text)


class SecretRedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


_configured = False


def setup_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(SecretRedactingFilter())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("httpx", "urllib3", "matplotlib", "numba", "speechbrain", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
