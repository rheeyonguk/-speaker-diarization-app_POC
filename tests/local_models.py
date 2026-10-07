"""Build fully local, offline model directories with the *real* upstream architectures/formats but
random weights, so the complete pipeline (WhisperX ASR -> alignment -> pyannote SpeakerDiarization
-> WeSpeaker) can be executed end to end without Hugging Face / ModelScope access.

Outputs are meaningless (random weights) - these fixtures verify integration, formats, offline
local-path configuration, orchestration, memory and timing. Never use them for accuracy claims.

Exception: the pyannote *segmentation* model is the real pretrained checkpoint bundled inside the
whisperx package (whisperx/assets/pytorch_model.bin, used by WhisperX as its VAD).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- Whisper (CTranslate2)

def whisper_special_tokens(num_languages: int = 99) -> list[str]:
    from whisper.tokenizer import LANGUAGES

    return [
        "<|endoftext|>", "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        "<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]


def build_whisper_tokenizer(out_dir: Path, num_languages: int = 99) -> list[str]:
    """tokenizer.json + tokenizer_config.json equivalent to openai-whisper's multilingual tiktoken
    encoding (identical token ids), built from the vocabulary file shipped in the openai-whisper wheel."""
    import whisper
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from transformers.convert_slow_tokenizer import TikTokenConverter

    tiktoken_file = Path(whisper.__file__).parent / "assets" / "multilingual.tiktoken"
    vocab, merges = TikTokenConverter(vocab_file=str(tiktoken_file)).extract_vocab_merges_from_model(
        str(tiktoken_file))
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = whisper_special_tokens(num_languages)
    tok.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in specials])
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))
    (out_dir / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "WhisperTokenizer", "bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>", "pad_token": "<|endoftext|>", "model_max_length": 1024,
        "additional_special_tokens": specials[1:],
    }, ensure_ascii=False), encoding="utf-8")
    return specials


def build_tiny_whisper_ct2(out_dir: Path, seed: int = 0) -> Path:
    import ctranslate2
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    out_dir = Path(out_dir)
    hf_dir = out_dir.parent / (out_dir.name + "_hf")
    specials = build_whisper_tokenizer(hf_dir)
    torch.manual_seed(seed)
    cfg = WhisperConfig(
        vocab_size=51865, num_mel_bins=80, encoder_layers=2, decoder_layers=2, d_model=64,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=128, decoder_ffn_dim=128,
        max_source_positions=1500, max_target_positions=448,
        decoder_start_token_id=50258, eos_token_id=50257, pad_token_id=50257, bos_token_id=50257,
    )
    model = WhisperForConditionalGeneration(cfg)
    base = 50257
    model.generation_config.lang_to_id = {t: base + i for i, t in enumerate(specials) if 2 <= i < 2 + 99}
    model.save_pretrained(hf_dir)
    (hf_dir / "preprocessor_config.json").write_text(json.dumps({"feature_size": 80}), encoding="utf-8")
    ctranslate2.converters.TransformersConverter(
        str(hf_dir), copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(str(out_dir), force=True)
    return out_dir


# --------------------------------------------------------------------------- wav2vec2 CTC (alignment)

def build_tiny_wav2vec2_ctc(out_dir: Path, chars: str, seed: int = 0) -> Path:
    import torch
    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}
    vocab.update({c: i + 5 for i, c in enumerate(sorted({c for c in chars if not c.isspace()}))})
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    tok = Wav2Vec2CTCTokenizer(str(out_dir / "vocab.json"), unk_token="<unk>", pad_token="<pad>",
                               word_delimiter_token="|")
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                  return_attention_mask=False)
    Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok).save_pretrained(out_dir)
    torch.manual_seed(seed)
    cfg = Wav2Vec2Config(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, conv_dim=(32,) * 7, num_conv_pos_embeddings=16,
                         num_conv_pos_embedding_groups=2, pad_token_id=0)
    Wav2Vec2ForCTC(cfg).save_pretrained(out_dir)
    return out_dir


# --------------------------------------------------------------------------- pyannote pipeline

def _save_pyannote_checkpoint(model, path: Path) -> None:
    import lightning
    import pyannote.audio
    import torch

    ckpt = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": lightning.__version__,
        "pyannote.audio": {
            "versions": {"torch": torch.__version__, "pyannote.audio": pyannote.audio.__version__},
            "architecture": {"module": type(model).__module__, "class": type(model).__name__},
            "specifications": model.specifications,
        },
    }
    torch.save(ckpt, path)


def build_local_pyannote_pipeline(out_dir: Path, seed: int = 0) -> Path:
    """Directory with config.yaml for pyannote.audio.pipelines.SpeakerDiarization + VBx clustering
    (the pipeline class Community-1 uses): real WhisperX-bundled segmentation checkpoint,
    random WeSpeakerResNet34 embedding, random-but-valid PLDA."""
    import torch
    import whisperx
    import yaml
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from pyannote.audio.models.embedding import WeSpeakerResNet34

    out_dir = Path(out_dir)
    (out_dir / "embedding").mkdir(parents=True, exist_ok=True)
    (out_dir / "plda").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    emb = WeSpeakerResNet34()
    if getattr(emb, "_specifications", None) is None:
        emb.specifications = Specifications(problem=Problem.REPRESENTATION, resolution=Resolution.CHUNK,
                                            duration=10.0)
    _save_pyannote_checkpoint(emb, out_dir / "embedding" / "pytorch_model.bin")

    rng = np.random.default_rng(seed)
    dim, lda_dim = 256, 128
    lda = np.linalg.qr(rng.standard_normal((dim, dim)))[0][:, :lda_dim]
    np.savez(out_dir / "plda" / "xvec_transform.npz", mean1=np.zeros(dim), mean2=np.zeros(lda_dim), lda=lda)
    tr = np.linalg.qr(rng.standard_normal((lda_dim, lda_dim)))[0]
    np.savez(out_dir / "plda" / "plda.npz", mu=np.zeros(lda_dim), tr=tr, psi=np.linspace(5.0, 0.5, lda_dim))

    seg = Path(whisperx.__file__).parent / "assets" / "pytorch_model.bin"
    config = {
        "pipeline": {
            "name": "pyannote.audio.pipelines.SpeakerDiarization",
            "params": {
                "segmentation": str(seg),
                "embedding": str(out_dir / "embedding" / "pytorch_model.bin"),
                "plda": str(out_dir / "plda"),
                "clustering": "VBxClustering",
                "segmentation_step": 0.1,
                "embedding_batch_size": 8,
                "segmentation_batch_size": 8,
            },
        },
        "params": {
            "segmentation": {"threshold": 0.5, "min_duration_off": 0.0},
            "clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8},
        },
    }
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return out_dir / "config.yaml"


# --------------------------------------------------------------------------- WeSpeaker

def build_random_wespeaker(out_dir: Path, model: str = "ResNet34", embed_dim: int = 256, seed: int = 0) -> Path:
    import torch
    import yaml
    from wespeaker.models.speaker_model import get_speaker_model

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"model": model, "model_args": {"feat_dim": 80, "embed_dim": embed_dim, "pooling_func": "TSTP",
                                             "two_emb_layer": False}}
    torch.manual_seed(seed)
    net = get_speaker_model(model)(**config["model_args"])
    torch.save(net.state_dict(), out_dir / "avg_model.pt")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return out_dir


def offline_env() -> dict:
    """Environment that makes any accidental Hub download fail fast."""
    return {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


__all__ = [
    "build_local_pyannote_pipeline",
    "build_random_wespeaker",
    "build_tiny_wav2vec2_ctc",
    "build_tiny_whisper_ct2",
    "build_whisper_tokenizer",
    "whisper_special_tokens",
    "offline_env",
    "os",
]
