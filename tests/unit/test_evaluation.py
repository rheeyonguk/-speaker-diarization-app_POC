import pytest

from meeting_transcriber.evaluation import (
    attribution_error_rate,
    cer,
    diarization_error,
    per_speaker_time_accuracy,
    read_rttm,
    speaker_id_metrics,
    wer,
    write_rttm,
)
from meeting_transcriber.evaluation.report import evaluate_document, markdown_summary


def test_cer_wer_korean():
    ref = "오늘 회의를 시작하겠습니다."
    assert cer(ref, "오늘 회의를 시작하겠습니다")["cer"] == 0.0           # punctuation ignored
    assert cer(ref, "오늘회의를 시작하겠습니다")["cer"] == 0.0             # spacing ignored by default
    assert cer(ref, "오늘회의를 시작하겠습니다", ignore_spaces=False)["cer"] > 0
    r = cer("가나다라", "가나다마")
    assert r["cer"] == pytest.approx(0.25) and r["substitutions"] == 1
    assert wer("오늘 회의를 시작합니다", "오늘 회의 시작합니다")["wer"] == pytest.approx(1 / 3)


REF = [{"start": 0, "end": 10, "speaker": "이용욱"}, {"start": 10, "end": 20, "speaker": "홍길동"},
       {"start": 20, "end": 30, "speaker": "GUEST_A"}]


def test_der_components():
    hyp = [{"start": 0, "end": 10, "speaker": "S0"}, {"start": 10, "end": 18, "speaker": "S1"},
           {"start": 18, "end": 20, "speaker": "S0"}, {"start": 30, "end": 32, "speaker": "S2"}]
    d = diarization_error(REF, hyp)
    assert d["miss_sec"] == pytest.approx(10.0)       # guest 20-30 missed
    assert d["false_alarm_sec"] == pytest.approx(2.0)
    assert d["confusion_sec"] == pytest.approx(2.0)
    assert d["der"] == pytest.approx(14 / 30)


def test_speaker_id_metrics_confusion():
    hyp = [{"start": 0, "end": 10, "speaker": "S0"}, {"start": 10, "end": 20, "speaker": "S1"},
           {"start": 20, "end": 30, "speaker": "S2"}]
    enrolled = ["이용욱", "홍길동"]
    perfect = speaker_id_metrics(REF, hyp, {"S0": "이용욱", "S1": "홍길동", "S2": "UNKNOWN_01"}, enrolled)
    assert perfect["accuracy"] == 1.0
    assert perfect["unknown_rejection_rate"] == 1.0
    assert perfect["false_accept_rate"] == 0.0

    bad = speaker_id_metrics(REF, hyp, {"S0": "UNKNOWN_01", "S1": "이용욱", "S2": "홍길동"}, enrolled)
    assert bad["counts"] == {"correct_accept": 0, "false_reject": 1, "misidentification": 1,
                             "correct_reject": 0, "false_accept_impostor": 1}
    assert bad["false_accept_rate"] == 1.0 and bad["false_reject_rate"] == 0.5
    cm = bad["confusion_matrix"]
    assert cm["rows_true"] == ["이용욱", "홍길동", "<UNENROLLED>"]
    assert cm["cols_pred"][-1] == "UNKNOWN"


def test_time_level_attribution():
    named = [{"start": 0, "end": 10, "speaker": "이용욱"}, {"start": 10, "end": 20, "speaker": "UNKNOWN_01"},
             {"start": 20, "end": 30, "speaker": "UNKNOWN_02"}]
    a = attribution_error_rate(REF, named, ["이용욱", "홍길동"])
    assert a["attribution_error_rate"] == pytest.approx(10 / 30)
    per = per_speaker_time_accuracy(REF, named, ["이용욱", "홍길동"])
    assert per["이용욱"]["accuracy"] == 1.0 and per["홍길동"]["accuracy"] == 0.0 and per["GUEST_A"]["accuracy"] == 1.0


def test_rttm_roundtrip(tmp_path):
    write_rttm(REF, tmp_path / "r.rttm")
    assert read_rttm(tmp_path / "r.rttm") == [dict(t, start=float(t["start"]), end=float(t["end"])) for t in REF]


def test_evaluate_document_rematch_modes():
    doc = {
        "file": "x", "duration": 30, "speaker_count": 3,
        "segments": [{"start": 0, "end": 10, "text": "안녕하세요"}],
        "speakers": [{"cluster": "S0", "label": "이용욱"}, {"cluster": "S1", "label": "홍길동"},
                     {"cluster": "S2", "label": "UNKNOWN_01"}],
        "diarization": {"speaker_diarization": [{"start": 0, "end": 10, "speaker": "S0"},
                                                {"start": 10, "end": 20, "speaker": "S1"},
                                                {"start": 20, "end": 30, "speaker": "S2"}]},
        "speaker_identification": {
            "clusters": ["S0", "S1", "S2"], "profiles": ["이용욱", "홍길동"],
            "similarity_matrix": [[0.8, 0.1], [0.2, 0.75], [0.3, 0.2]],
            "cooccurrence_sec": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "evidence": {"S0": {"num_windows": 2}, "S1": {"num_windows": 2}, "S2": {"num_windows": 1}},
            "threshold": 0.5, "margin": 0.05,
        },
    }
    r = evaluate_document(doc, REF, "안녕하세요")
    assert r["stt"]["cer"]["cer"] == 0.0
    assert set(r["speaker_id"]) == {"as_run", "flexible", "strict_one_to_one", "argmax"}
    for mode in r["speaker_id"].values():
        assert mode["cluster_level"]["accuracy"] == 1.0
    assert r["similarity_scores"]["target"] == [0.8, 0.75]
    r["id"] = "T"
    assert "| T |" in markdown_summary([r])
