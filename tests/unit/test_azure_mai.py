import json

import numpy as np
import pytest

from meeting_transcriber.asr.azure_mai import (
    build_definition,
    normalize_endpoint,
    parse_phrases_text,
    parse_transcription,
    plan_chunks,
    transcribe_url,
)
from meeting_transcriber.config import load_config
from meeting_transcriber.errors import ConfigError

# Shape taken from the documented sample response (lexical lower-case words, display text with
# punctuation, character-level CJK words, confidence always 0).
DOC_SAMPLE = {
    "durationMilliseconds": 20000,
    "phrases": [
        {"offsetMilliseconds": 80, "durationMilliseconds": 2000, "locale": "en-us", "confidence": 0,
         "text": "With custom speech,you can evaluate.",
         "words": [{"text": "with", "offsetMilliseconds": 80, "durationMilliseconds": 160},
                   {"text": "custom", "offsetMilliseconds": 240, "durationMilliseconds": 480},
                   {"text": "speech", "offsetMilliseconds": 720, "durationMilliseconds": 360},
                   {"text": "you", "offsetMilliseconds": 1100, "durationMilliseconds": 100},
                   {"text": "can", "offsetMilliseconds": 1200, "durationMilliseconds": 200},
                   {"text": "evaluate", "offsetMilliseconds": 1400, "durationMilliseconds": 600}]},
        {"offsetMilliseconds": 8000, "durationMilliseconds": 200, "locale": "zh-cn", "confidence": 0,
         "text": "现成的。",
         "words": [{"text": "现", "offsetMilliseconds": 8000, "durationMilliseconds": 40},
                   {"text": "成", "offsetMilliseconds": 8040, "durationMilliseconds": 40},
                   {"text": "的", "offsetMilliseconds": 8080, "durationMilliseconds": 40}]},
        {"offsetMilliseconds": 10000, "durationMilliseconds": 3000, "locale": "ko-KR", "confidence": 0,
         "text": "네, APQR 초안을 공유드리겠습니다.",
         "words": [{"text": "네", "offsetMilliseconds": 10000, "durationMilliseconds": 300},
                   {"text": "apqr", "offsetMilliseconds": 10400, "durationMilliseconds": 600},
                   {"text": "초안을", "offsetMilliseconds": 11000, "durationMilliseconds": 500},
                   {"text": "공유드리겠습니다", "offsetMilliseconds": 11600, "durationMilliseconds": 1300}]},
    ],
}


def _join(words):
    from meeting_transcriber.pipeline.assignment import _join_words

    return _join_words(words, " ")


def test_parse_keeps_display_text_spacing_and_punctuation():
    segs, langs = parse_transcription(DOC_SAMPLE, offset_sec=100.0)
    assert [s["text"] for s in segs] == ["With custom speech,you can evaluate.", "现成的。",
                                          "네, APQR 초안을 공유드리겠습니다."]
    en, zh, ko = segs
    assert en["start"] == pytest.approx(100.08) and en["end"] == pytest.approx(102.08)
    assert en["words"][0] == {"word": "With", "sep": "", "start": 100.08, "end": 100.24}
    assert _join(en["words"]) == "With custom speech,you can evaluate."
    assert _join(zh["words"]) == "现成的。"                       # CJK characters: no spaces inserted
    assert _join(ko["words"]) == "네, APQR 초안을 공유드리겠습니다."   # display casing + punctuation restored
    assert [(w["word"], w["sep"]) for w in ko["words"]] == [("네", ""), ("APQR", ", "), ("초안을", " "),
                                                             ("공유드리겠습니다.", " ")]
    assert langs["ko"] == pytest.approx(3.0) and langs["en"] == pytest.approx(2.0)


def test_unmatched_word_falls_back_to_token():
    payload = {"phrases": [{"offsetMilliseconds": 0, "durationMilliseconds": 1000, "text": "이천이십사 년",
                            "words": [{"text": "2024", "offsetMilliseconds": 0, "durationMilliseconds": 500},
                                      {"text": "년", "offsetMilliseconds": 500, "durationMilliseconds": 500}]}]}
    segs, _ = parse_transcription(payload)
    assert segs[0]["words"][0]["word"] == "2024"
    assert segs[0]["words"][1]["word"] == "년"


def test_phrase_without_words_is_kept_segment_level():
    segs, _ = parse_transcription({"phrases": [{"offsetMilliseconds": 0, "durationMilliseconds": 900, "text": "네."}]})
    assert segs[0]["words"] == [] and segs[0]["text"] == "네."


def test_definition_defaults_follow_ms_guidance():
    cfg = load_config(env={})
    d = build_definition(cfg, ["APQR", "CAPA", "APQR"])
    assert d["enhancedMode"] == {"enabled": True, "model": "MAI-Transcribe-2",
                                 "modelOptions": {"timestamps": "word", "transcribeStyle": "verbatim"}}
    assert d["diarization"] == {"enabled": False}
    assert d["phraseList"] == {"phrases": ["APQR", "CAPA"]}
    assert "locales" not in d  # auto-detection unless force_locale
    cfg.asr.azure.force_locale = True
    assert build_definition(cfg)["locales"] == ["ko"]
    json.dumps(d)


def test_parse_phrases_text():
    assert parse_phrases_text("APQR, CAPA\n일탈;변경관리 ,") == ["APQR", "CAPA", "일탈", "변경관리"]
    assert parse_phrases_text(None) == []


@pytest.mark.parametrize("raw,expected", [
    ("https://hanmi-speech.cognitiveservices.azure.com/", "https://hanmi-speech.cognitiveservices.azure.com"),
    ("hanmi-speech.cognitiveservices.azure.com", "https://hanmi-speech.cognitiveservices.azure.com"),
    ("https://hanmi-ai.services.ai.azure.com/api/projects/p1", "https://hanmi-ai.cognitiveservices.azure.com"),
    ("https://eastus.api.cognitive.microsoft.com/", "https://eastus.api.cognitive.microsoft.com"),
    ("http://example.cognitiveservices.azure.com", "https://example.cognitiveservices.azure.com"),
    ("http://127.0.0.1:8080", "http://127.0.0.1:8080"),
])
def test_normalize_endpoint(raw, expected):
    base, _ = normalize_endpoint(raw)
    assert base == expected
    assert transcribe_url(base, "2025-10-15").endswith("/speechtotext/transcriptions:transcribe?api-version=2025-10-15")


def test_plan_chunks_cuts_at_quiet_points_and_respects_limit():
    sr = 1000
    rng = np.random.default_rng(0)
    audio = rng.uniform(0.2, 0.5, size=sr * 250).astype(np.float32)
    audio[sr * 95: sr * 96] = 0.0   # silence near the 100 s boundary
    audio[sr * 188: sr * 189] = 0.0  # and near the next one
    chunks = plan_chunks(len(audio), sr, audio, max_chunk_sec=100, search_sec=10)
    assert chunks[0][0] == 0 and chunks[-1][1] == len(audio)
    assert all(b - a <= 100 * sr for a, b in chunks)
    assert all(chunks[i][1] == chunks[i + 1][0] for i in range(len(chunks) - 1))
    assert 95 * sr <= chunks[0][1] <= 96 * sr
    assert plan_chunks(50 * sr, sr, audio[: 50 * sr], 100) == [(0, 50 * sr)]


def test_missing_endpoint_or_key_is_clear(monkeypatch):
    from meeting_transcriber.asr import AzureMaiTranscriber

    monkeypatch.delenv("AZURE_SPEECH_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    cfg = load_config(env={})
    with pytest.raises(ConfigError, match="AZURE_SPEECH_ENDPOINT"):
        AzureMaiTranscriber(cfg)
    monkeypatch.setenv("AZURE_SPEECH_ENDPOINT", "https://x.cognitiveservices.azure.com")
    with pytest.raises(ConfigError, match="AZURE_SPEECH_KEY"):
        AzureMaiTranscriber(cfg)


def test_azure_key_is_redacted(monkeypatch):
    from meeting_transcriber.logging_utils import redact

    monkeypatch.setenv("AZURE_SPEECH_KEY", "abcd1234efgh5678")
    assert "abcd1234efgh5678" not in redact("header Ocp-Apim-Subscription-Key: abcd1234efgh5678")
