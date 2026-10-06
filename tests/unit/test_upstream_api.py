"""Guards against stale/deprecated upstream APIs: fail loudly if an upgrade changes what we call."""

import inspect

import pytest


def _params(fn):
    return set(inspect.signature(fn).parameters)


def test_whisperx_load_model_signature():
    from whisperx import asr

    assert {"whisper_arch", "device", "device_index", "compute_type", "asr_options", "language", "vad_method",
            "vad_options", "download_root", "threads"} <= _params(asr.load_model)
    assert {"audio", "batch_size", "language", "chunk_size", "progress_callback"} <= _params(
        asr.FasterWhisperPipeline.transcribe)


def test_whisperx_alignment_api_and_korean_model():
    from whisperx import alignment

    assert {"language_code", "device", "model_name", "model_dir"} <= _params(alignment.load_align_model)
    assert {"transcript", "model", "align_model_metadata", "audio", "device", "interpolate_method",
            "return_char_alignments", "progress_callback"} <= _params(alignment.align)
    # Korean is supported by WhisperX's default alignment table (value read, not hard-coded by us)
    assert alignment.DEFAULT_ALIGN_MODELS_HF.get("ko")


def test_whisperx_default_diarization_model_is_community1():
    from whisperx import diarize

    src = inspect.getsource(diarize.DiarizationPipeline.__init__)
    assert "pyannote/speaker-diarization-community-1" in src
    assert {"num_speakers", "min_speakers", "max_speakers", "return_embeddings"} <= _params(
        diarize.DiarizationPipeline.__call__)
    assert {"diarize_df", "transcript_result", "speaker_embeddings", "fill_nearest"} <= _params(
        diarize.assign_word_speakers)


def test_pyannote_community1_api():
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput, SpeakerDiarization

    assert {"checkpoint", "token", "cache_dir"} <= _params(Pipeline.from_pretrained)
    assert {"file", "num_speakers", "min_speakers", "max_speakers", "hook"} <= _params(SpeakerDiarization.apply)
    assert {"speaker_diarization", "exclusive_speaker_diarization", "speaker_embeddings"} == set(
        DiarizeOutput.__dataclass_fields__)


def test_wespeaker_python_api():
    pytest.importorskip("wespeaker")
    import wespeaker
    from wespeaker.cli.hub import Hub
    from wespeaker.cli.speaker import Speaker

    assert callable(wespeaker.load_model)
    for m in ("extract_embedding_from_pcm", "set_device", "set_vad", "set_resample_rate", "set_wavform_norm",
              "set_window_type", "cosine_similarity", "register", "recognize"):
        assert hasattr(Speaker, m), m
    assert {"pcm", "sample_rate"} <= _params(Speaker.extract_embedding_from_pcm)
    assert "vblinkf" in Hub.Assets


def test_faster_whisper_models_known():
    from faster_whisper.utils import available_models

    models = available_models()
    assert "large-v3" in models and "large-v3-turbo" in models


def test_versions_match_pins():
    from importlib.metadata import version

    assert version("whisperx") == "3.8.6"
    assert version("pyannote.audio").startswith("4.")
    assert version("faster-whisper").split(".")[0] == "1"
    assert version("ctranslate2").split(".")[0] == "4"
