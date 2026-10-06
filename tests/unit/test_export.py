import csv
import json
import re

from meeting_transcriber.export import CSV_COLUMNS, fmt_hms, fmt_srt, write_outputs

DOC = {
    "file": "회의.m4a",
    "language": "ko",
    "duration": 11.0,
    "speaker_count": 3,
    "speakers": [{"cluster": "SPEAKER_00", "label": "이용욱"}, {"cluster": "SPEAKER_01", "label": "김철수"},
                 {"cluster": "SPEAKER_02", "label": "홍길동"}],
    "segments": [
        {"start": 1.0, "end": 5.0, "speaker_cluster": "SPEAKER_00", "identified_speaker": "이용욱",
         "speaker_similarity": 0.8123, "text": "오늘 회의를 시작하겠습니다.", "overlap": False, "secondary_speakers": [],
         "words": [{"word": "오늘", "start": 1.0, "end": 1.4}]},
        {"start": 5.0, "end": 9.0, "speaker_cluster": "SPEAKER_01", "identified_speaker": "김철수",
         "speaker_similarity": 0.7, "text": "네, 먼저 지난주 진행 상황을 말씀드리겠습니다.", "overlap": False,
         "secondary_speakers": [], "words": []},
        {"start": 9.0, "end": 11.0, "speaker_cluster": "SPEAKER_02", "identified_speaker": "홍길동",
         "speaker_similarity": None, "text": "잠시만요.", "overlap": True, "secondary_speakers": ["김철수"], "words": []},
    ],
}


def test_time_formats():
    assert fmt_hms(3725.4) == "01:02:05"
    assert fmt_srt(3725.4567) == "01:02:05,457"
    assert fmt_srt(0) == "00:00:00,000"


def test_all_formats(tmp_path):
    paths = write_outputs(DOC, tmp_path, "meeting", ["txt", "json", "csv", "srt"])
    assert set(paths) == {"txt", "json", "csv", "srt"}

    txt = paths["txt"].read_text(encoding="utf-8")
    assert "00:00:01 - 00:00:05\n이용욱\n오늘 회의를 시작하겠습니다." in txt
    assert "00:00:05 - 00:00:09\n김철수\n" in txt
    assert "동시발화: 김철수" in txt

    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["segments"][0]["identified_speaker"] == "이용욱"
    assert data["segments"][0]["words"][0]["word"] == "오늘"

    with open(paths["csv"], encoding="utf-8-sig", newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert list(rows[0].keys()) == CSV_COLUMNS
    assert rows[0]["speaker"] == "이용욱" and rows[0]["confidence"] == "0.8123"
    assert rows[2]["confidence"] == ""
    assert rows[0]["start_time"] == "00:00:01.000"
    assert paths["csv"].read_bytes().startswith(b"\xef\xbb\xbf")  # BOM for Excel

    srt = paths["srt"].read_text(encoding="utf-8")
    blocks = [b for b in srt.strip().split("\n\n") if b]
    assert len(blocks) == 3
    assert re.match(r"1\n00:00:01,000 --> 00:00:05,000\n\[이용욱\] 오늘", blocks[0])


def test_txt_merges_same_speaker(tmp_path):
    doc = dict(DOC)
    doc["segments"] = [
        dict(DOC["segments"][0], start=1.0, end=2.0, text="첫 문장."),
        dict(DOC["segments"][0], start=2.3, end=3.0, text="둘째 문장."),
    ]
    txt = write_outputs(doc, tmp_path, "m", ["txt"], merge_gap=1.0)["txt"].read_text(encoding="utf-8")
    assert "첫 문장. 둘째 문장." in txt
    assert txt.count("이용욱\n") == 1


def test_json_without_words(tmp_path):
    p = write_outputs(DOC, tmp_path, "m", ["json"], include_words=False)["json"]
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "words" not in data["segments"][0]
