from meeting_transcriber.config import AssignmentConfig
from meeting_transcriber.diarization import DiarizationResult, Turn
from meeting_transcriber.pipeline.assignment import build_utterances

LABELS = {"SPEAKER_00": "이용욱", "SPEAKER_01": "홍길동", "SPEAKER_02": "UNKNOWN_01"}
SIMS = {"SPEAKER_00": 0.81, "SPEAKER_01": 0.77, "SPEAKER_02": None}
STATUS = {"SPEAKER_00": "identified", "SPEAKER_01": "identified", "SPEAKER_02": "unknown"}


def w(word, s, e, score=0.9):
    return {"word": word, "start": s, "end": e, "score": score}


def diar():
    full = [Turn(0.0, 5.0, "SPEAKER_00"), Turn(4.6, 6.0, "SPEAKER_01"), Turn(6.2, 9.0, "SPEAKER_02")]
    excl = [Turn(0.0, 4.6, "SPEAKER_00"), Turn(4.6, 6.0, "SPEAKER_01"), Turn(6.2, 9.0, "SPEAKER_02")]
    return DiarizationResult(full, excl)


def test_segment_split_on_speaker_change_and_overlap_flag():
    aligned = {"segments": [{
        "start": 0.0, "end": 6.0, "text": "오늘 회의를 시작하겠습니다 잠시만요", "avg_logprob": -0.2,
        "words": [w("오늘", 0.1, 0.5), w("회의를", 0.6, 1.2), w("시작하겠습니다", 1.3, 2.4), w("잠시만요", 4.7, 5.6)],
    }]}
    utts = build_utterances(aligned, diar(), LABELS, SIMS, STATUS, AssignmentConfig())
    assert [u["identified_speaker"] for u in utts] == ["이용욱", "홍길동"]
    assert utts[0]["text"] == "오늘 회의를 시작하겠습니다"
    assert utts[0]["speaker_similarity"] == 0.81
    assert utts[0]["overlap"] is False
    # "잠시만요" 4.7-5.6 overlaps SPEAKER_00 (until 5.0) -> overlap flag + secondary speaker kept
    assert utts[1]["text"] == "잠시만요"
    assert utts[1]["overlap"] is True
    assert utts[1]["secondary_speakers"] == ["이용욱"]
    assert utts[1]["words"][0]["speaker_cluster"] == "SPEAKER_01"
    assert utts[1]["asr_avg_logprob"] == -0.2


def test_unknown_label_and_nearest_fallback():
    aligned = {"segments": [{
        "start": 6.0, "end": 9.5, "text": "네 알겠습니다",
        "words": [w("네", 6.3, 6.5), w("알겠습니다", 9.2, 9.6)],   # second word after the turn ends (gap 0.2s)
    }]}
    utts = build_utterances(aligned, diar(), LABELS, SIMS, STATUS, AssignmentConfig())
    assert len(utts) == 1
    assert utts[0]["identified_speaker"] == "UNKNOWN_01"
    assert utts[0]["speaker_similarity"] is None
    assert utts[0]["text"] == "네 알겠습니다"


def test_words_far_from_any_turn_are_unassigned():
    aligned = {"segments": [{"start": 20.0, "end": 21.0, "text": "환각", "words": [w("환각", 20.0, 21.0)]}]}
    utts = build_utterances(aligned, diar(), LABELS, SIMS, STATUS, AssignmentConfig())
    assert utts[0]["identified_speaker"] == "UNASSIGNED"
    assert utts[0]["speaker_cluster"] is None


def test_segment_without_alignment_uses_segment_interval():
    aligned = {"segments": [{"start": 0.2, "end": 3.0, "text": "정렬 실패 구간", "words": []}]}
    utts = build_utterances(aligned, diar(), LABELS, SIMS, STATUS, AssignmentConfig())
    assert utts[0]["identified_speaker"] == "이용욱"
    assert utts[0]["text"] == "정렬 실패 구간"
    assert utts[0]["words"] == []


def test_words_without_timestamps_inherit_speaker():
    aligned = {"segments": [{"start": 0.0, "end": 2.0, "text": "2024년 실적",
                             "words": [{"word": "2024년"}, w("실적", 1.0, 1.5)]}]}
    utts = build_utterances(aligned, diar(), LABELS, SIMS, STATUS, AssignmentConfig())
    assert len(utts) == 1 and utts[0]["text"] == "2024년 실적"
    assert utts[0]["identified_speaker"] == "이용욱"


def test_numpy_and_nan_values_are_json_safe():
    import json
    import math

    import numpy as np

    aligned = {"segments": [{"start": np.float64(0.0), "end": np.float64(3.0), "text": "숫자 2024 포함",
                             "avg_logprob": np.float32(-0.25),
                             "words": [w("숫자", np.float64(0.1), np.float64(0.5), np.float64(0.9)),
                                       {"word": "2024", "start": float("nan"), "end": float("nan"), "score": float("nan")},
                                       w("포함", 1.0, 1.4, np.float32(0.7))]}]}
    utts = build_utterances(aligned, diar(), LABELS, SIMS, STATUS, AssignmentConfig())
    text = json.dumps(utts, ensure_ascii=False, allow_nan=False)  # raises on NaN / numpy types
    assert "2024" in text
    for u in utts:
        for x in u["words"]:
            assert x["start"] is None or not math.isnan(x["start"])
