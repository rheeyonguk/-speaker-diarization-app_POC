"""Runs the real whisperx.load_align_model + whisperx.align code through our WhisperXAligner, then
speaker assignment and all exporters, with a tiny *randomly initialised* local Wav2Vec2-CTC model
whose vocabulary contains the Korean characters of the test sentence.

The pretrained Korean model (kresnik/wav2vec2-large-xlsr-korean) cannot be downloaded in CI, so this
verifies API compatibility and data flow only - timestamps are not meaningful.
"""

import pytest

from conftest import SR, tone

TEXT_A = "오늘 회의를 시작하겠습니다"
TEXT_B = "네 맞습니다"


def _punkt_available():
    try:
        import nltk

        nltk.data.find("tokenizers/punkt_tab/english/")
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def tiny_ctc_dir(tmp_path_factory):
    torch = pytest.importorskip("torch")
    import json

    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
    )

    d = tmp_path_factory.mktemp("tiny_w2v2_ko")
    chars = sorted({c for c in TEXT_A + TEXT_B if c != " "})
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}
    vocab.update({c: i + 5 for i, c in enumerate(chars)})
    (d / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    tok = Wav2Vec2CTCTokenizer(str(d / "vocab.json"), unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR, padding_value=0.0, do_normalize=True,
                                  return_attention_mask=False)
    Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok).save_pretrained(d)
    torch.manual_seed(0)
    cfg = Wav2Vec2Config(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, conv_dim=(32,) * 7, num_conv_pos_embeddings=16,
                         num_conv_pos_embedding_groups=2, pad_token_id=0)
    Wav2Vec2ForCTC(cfg).save_pretrained(d)
    return d


@pytest.mark.skipif(not _punkt_available(), reason="NLTK punkt_tab not available offline")
def test_whisperx_align_then_assign_and_export(tmp_cfg, tmp_path, tiny_ctc_dir):
    import numpy as np

    from meeting_transcriber.asr import WhisperXAligner
    from meeting_transcriber.diarization import DiarizationResult, Turn
    from meeting_transcriber.export import write_outputs
    from meeting_transcriber.pipeline.assignment import build_utterances

    tmp_cfg.alignment.model_name = str(tiny_ctc_dir)
    audio = np.concatenate([tone([150, 300], 3.0, seed=1), tone([250, 500], 2.0, seed=2)])
    transcript = {"language": "ko", "segments": [
        {"start": 0.0, "end": 3.0, "text": TEXT_A, "avg_logprob": -0.3},
        {"start": 3.0, "end": 5.0, "text": TEXT_B, "avg_logprob": -0.4},
    ]}
    aligner = WhisperXAligner(tmp_cfg, "cpu")
    aligned = aligner.align(transcript, audio)
    aligner.unload()
    assert aligned["aligned"] is True
    words = [w for s in aligned["segments"] for w in s["words"]]
    assert [w["word"] for w in words] == TEXT_A.split() + TEXT_B.split()
    for w in words:
        assert 0.0 <= w["start"] <= w["end"] <= 5.0 + 1e-6
    assert all("avg_logprob" in s for s in aligned["segments"])  # ASR confidence preserved

    turns = [Turn(0.0, 3.0, "SPEAKER_00"), Turn(3.0, 5.0, "SPEAKER_01")]
    diar = DiarizationResult(turns, turns)
    utts = build_utterances(aligned, diar, {"SPEAKER_00": "이용욱", "SPEAKER_01": "UNKNOWN_01"},
                            {"SPEAKER_00": 0.8, "SPEAKER_01": None},
                            {"SPEAKER_00": "identified", "SPEAKER_01": "unknown"}, tmp_cfg.assignment)
    assert {u["identified_speaker"] for u in utts} <= {"이용욱", "UNKNOWN_01"}
    doc = {"file": "t.wav", "language": "ko", "duration": 5.0, "speaker_count": 2, "speakers": [], "segments": utts}
    files = write_outputs(doc, tmp_path / "out", "t", ["txt", "json", "csv", "srt"])
    assert all(p.exists() and p.stat().st_size > 0 for p in files.values())
