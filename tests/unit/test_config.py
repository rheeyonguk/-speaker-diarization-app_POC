import pytest

from meeting_transcriber.config import AppConfig, load_config, speaker_count_kwargs, validate_config
from meeting_transcriber.errors import ConfigError


def test_defaults_load_from_yaml():
    cfg = load_config(env={})
    assert cfg.asr.language == "ko"
    assert cfg.diarization.model == "pyannote/speaker-diarization-community-1"
    assert cfg.diarization.max_speakers == 10
    assert cfg.diarization.speaker_limit == 10
    assert cfg.speaker_id.matching_mode == "flexible"
    assert set(cfg.export.formats) == {"txt", "json", "csv", "srt"}


def test_env_overrides_are_typed():
    cfg = load_config(env={
        "MT_ASR__MODEL": "large-v3-turbo",
        "MT_ASR__BATCH_SIZE": "4",
        "MT_SPEAKER_ID__MATCH_THRESHOLD": "0.42",
        "MT_RUNTIME__RELEASE_MODELS_BETWEEN_STAGES": "false",
        "MT_SPEAKER_ID__CLUSTER_EMBEDDING__MIN_SEGMENT_SEC": "2.5",
        "MT_DIARIZATION__NUM_SPEAKERS": "null",
        "MT_EXPORT__FORMATS": "txt,json",
    })
    assert cfg.asr.model == "large-v3-turbo"
    assert cfg.asr.batch_size == 4
    assert cfg.speaker_id.match_threshold == pytest.approx(0.42)
    assert cfg.runtime.release_models_between_stages is False
    assert cfg.speaker_id.cluster_embedding.min_segment_sec == 2.5
    assert cfg.diarization.num_speakers is None
    assert cfg.export.formats == ["txt", "json"]


def test_unknown_env_key_rejected():
    with pytest.raises(ConfigError):
        load_config(env={"MT_ASR__NOPE": "1"})


def test_yaml_override_file(tmp_path):
    f = tmp_path / "local.yaml"
    f.write_text("diarization:\n  speaker_limit: 20\n  max_speakers: 15\n", encoding="utf-8")
    cfg = load_config(f, env={})
    assert cfg.diarization.speaker_limit == 20
    assert cfg.diarization.max_speakers == 15


def test_speaker_limit_is_configurable_not_hardcoded():
    cfg = AppConfig()
    cfg.diarization.speaker_limit = 16
    cfg.diarization.max_speakers = 16
    validate_config(cfg)
    assert speaker_count_kwargs("FIXED", 16, None, None, 16) == {"num_speakers": 16}


@pytest.mark.parametrize("mode,num,mn,mx,expected", [
    ("AUTO", None, None, None, {}),
    ("auto", 3, 2, 5, {}),
    ("RANGE", None, 2, 10, {"min_speakers": 2, "max_speakers": 10}),
    ("FIXED", 10, 2, 5, {"num_speakers": 10}),
])
def test_speaker_count_modes(mode, num, mn, mx, expected):
    assert speaker_count_kwargs(mode, num, mn, mx, 10) == expected


@pytest.mark.parametrize("mode,num,mn,mx", [
    ("FIXED", None, None, None),     # FIXED needs an exact count
    ("FIXED", 11, None, None),       # above speaker_limit
    ("RANGE", None, 5, 3),           # min > max
    ("RANGE", None, None, None),
    ("SOMETIMES", None, None, None),
])
def test_speaker_count_invalid(mode, num, mn, mx):
    with pytest.raises(ConfigError):
        speaker_count_kwargs(mode, num, mn, mx, 10)


def test_threshold_range_validated():
    cfg = AppConfig()
    cfg.speaker_id.match_threshold = 1.5
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_hf_token_not_part_of_config(monkeypatch):
    token = "hf_" + "x1" * 16
    monkeypatch.setenv("HF_TOKEN", token)
    cfg = load_config(env={"HF_TOKEN": token})
    assert token not in repr(cfg.to_dict())
