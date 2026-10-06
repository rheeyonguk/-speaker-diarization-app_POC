"""Launch the real Gradio app on a free port and drive it over HTTP with gradio_client."""

import socket
import urllib.request

import pytest

from conftest import SR, tone


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def app_url(tmp_path_factory):
    from meeting_transcriber.config import load_config
    from meeting_transcriber.ui import build_app, launch_kwargs

    tmp = tmp_path_factory.mktemp("app")
    cfg = load_config(env={})
    cfg.paths.enrollment_dir = str(tmp / "speakers")
    cfg.paths.output_dir = str(tmp / "outputs")
    cfg.runtime.device = "cpu"
    cfg.asr.model = "no-such-whisper-model-xyz"  # forces a model-load failure without network access
    demo = build_app(cfg)
    demo.queue(default_concurrency_limit=1)
    port = _free_port()
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True,
                allowed_paths=[str(tmp / "outputs")], **launch_kwargs(cfg))
    yield f"http://127.0.0.1:{port}/"
    demo.close()


def test_app_serves_three_tabs(app_url):
    import json

    cfg = json.loads(urllib.request.urlopen(app_url + "config", timeout=30).read())
    labels = [c.get("props", {}).get("label") for c in cfg["components"] if c.get("type") == "tabitem"]
    assert any("Transcription" in (x or "") for x in labels)
    assert any("Enrollment" in (x or "") for x in labels)
    assert any("Diagnostics" in (x or "") for x in labels)
    assert cfg.get("analytics_enabled") is False
    assert cfg.get("title") == "한미약품 AI 회의록"
    htmls = " ".join(str(c.get("props", {}).get("value", "")) for c in cfg["components"] if c.get("type") == "html")
    assert "mt-header" in htmls and "AI 회의록" in htmls and "외부 전송 없음" in htmls


def test_upload_and_run_returns_readable_error(app_url, tmp_path, write_wav):
    from gradio_client import Client, handle_file

    src = tmp_path / "meeting.wav"
    write_wav(src, tone([150, 300], 3.0), SR)
    client = Client(app_url, verbose=False)
    status, transcript, *_ = client.predict(
        handle_file(str(src)), "ko", "FIXED", 3, 2, 10, "no-such-whisper-model-xyz", True, "flexible", 0.5,
        api_name="/run_transcription",
    )
    assert status.startswith("❌")
    assert "Transcription" in status and "Whisper" in status
    assert "Traceback" not in status


def test_enrollment_without_files_is_rejected(app_url):
    from gradio_client import Client

    client = Client(app_url, verbose=False)
    msg, rows, _ = client.predict("이용욱", None, False, True, api_name="/enroll")
    assert msg.startswith("❌")


def test_diagnostics_endpoint(app_url):
    from gradio_client import Client

    info = Client(app_url, verbose=False).predict(api_name="/diagnostics")
    assert "cuda_available" in info and info["hf_token"] in ("set", "NOT SET")
    assert info["pyannote_model"] == "pyannote/speaker-diarization-community-1"
