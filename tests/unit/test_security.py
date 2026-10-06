import logging
import os
import subprocess
from pathlib import Path

from meeting_transcriber.logging_utils import SecretRedactingFilter, redact

ROOT = Path(__file__).resolve().parents[2]
FAKE = "hf_" + "Ab1" * 11  # built at runtime so the secret scanner below does not flag this file


def test_redact_env_token_and_pattern(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "my-custom-secret-value-123")
    assert "my-custom-secret-value-123" not in redact("token=my-custom-secret-value-123")
    assert FAKE not in redact(f"Authorization: Bearer {FAKE}")


def test_log_filter_masks_records(caplog):
    logger = logging.getLogger("test.secret")
    handler = logging.Handler()
    seen = []
    handler.emit = lambda rec: seen.append(rec.getMessage())
    handler.addFilter(SecretRedactingFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("loading with token %s", FAKE)
    logger.removeHandler(handler)
    assert seen and FAKE not in seen[0]


def test_privacy_env_defaults():
    import meeting_transcriber  # noqa: F401

    assert os.environ.get("PYANNOTE_METRICS_ENABLED") in ("0", "false", "False")
    assert os.environ.get("HF_HUB_DISABLE_TELEMETRY")
    assert os.environ.get("GRADIO_ANALYTICS_ENABLED")


def test_gitignore_protects_secrets_and_voice_data():
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".env", "data/speakers/*", "outputs/*", "*.wav", "*.npy"):
        assert pattern in gi
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "HF_TOKEN=\n" in env_example  # placeholder only


def test_no_token_literal_in_source():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    files += [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")]
    import re

    pat = re.compile(r"hf_[A-Za-z0-9]{30,}")
    for f in set(files):
        p = ROOT / f
        if p.suffix in {".py", ".md", ".yaml", ".toml", ".example", ".txt"} and p.exists():
            assert not pat.search(p.read_text(encoding="utf-8", errors="ignore")), f


def test_error_classification_proxy_403_is_network_not_auth():
    import requests

    from meeting_transcriber.errors import is_auth_error, is_network_error

    try:
        try:
            raise OSError("Tunnel connection failed: 403 Forbidden")
        except OSError as inner:
            raise requests.exceptions.ProxyError("Unable to connect to proxy") from inner
    except requests.exceptions.ProxyError as exc:
        assert is_network_error(exc)
        assert not is_auth_error(exc)


def test_error_classification_hub_403_is_auth():
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    from meeting_transcriber.errors import is_auth_error, is_network_error

    resp = requests.Response()
    resp.status_code = 403
    exc = HfHubHTTPError("403 Client Error: Forbidden for url", response=resp)
    assert is_auth_error(exc) and not is_network_error(exc)
