from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import meeting_transcriber  # noqa: E402,F401  (sets privacy env defaults)

SR = 16000


def tone(freqs, seconds: float, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    """Harmonic 'voice-like' test signal with slight AM so it is not perfectly stationary."""
    t = np.arange(int(seconds * SR)) / SR
    sig = sum(np.sin(2 * np.pi * f * t) / (k + 1) for k, f in enumerate(freqs))
    am = 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t + seed)
    rng = np.random.default_rng(seed)
    out = amp * sig * am / max(1e-9, np.max(np.abs(sig))) + 0.003 * rng.standard_normal(t.size)
    return out.astype(np.float32)


@pytest.fixture
def write_wav():
    from meeting_transcriber.audio.io import write_wav as _w

    return _w


@pytest.fixture
def ffmpeg():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")

    def run(*args):
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)

    return run


@pytest.fixture
def tmp_cfg(tmp_path, monkeypatch):
    """Config pointing all writable paths into tmp_path; ignores the developer's .env overrides."""
    from meeting_transcriber.config import load_config

    for k in list(os.environ):
        if k.startswith("MT_"):
            monkeypatch.delenv(k, raising=False)
    cfg = load_config(env={})
    cfg.paths.enrollment_dir = str(tmp_path / "speakers")
    cfg.paths.output_dir = str(tmp_path / "outputs")
    cfg.runtime.device = "cpu"
    return cfg


@pytest.fixture(scope="session")
def wespeaker_random_model_dir(tmp_path_factory):
    """A real WeSpeaker ResNet34 checkpoint directory with *random* weights.

    Used only to exercise our adapter against the real wespeaker loading / feature / inference
    code path (API compatibility). It says nothing about identification accuracy.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("wespeaker")
    import yaml
    from wespeaker.models.speaker_model import get_speaker_model

    d = tmp_path_factory.mktemp("wespeaker_random")
    config = {"model": "ResNet34", "model_args": {"feat_dim": 80, "embed_dim": 256, "pooling_func": "TSTP",
                                                  "two_emb_layer": False}}
    torch.manual_seed(0)
    model = get_speaker_model(config["model"])(**config["model_args"])
    torch.save(model.state_dict(), d / "avg_model.pt")
    (d / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return d
