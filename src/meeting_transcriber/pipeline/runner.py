"""End-to-end orchestration:
preprocess -> WhisperX ASR -> WhisperX alignment -> pyannote Community-1 -> WeSpeaker ID -> export.

Each stage is timed, wrapped in a CUDA-OOM guard and (by default) releases its model before the
next stage starts so that peak VRAM is bounded by the largest single model, not their sum.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ..asr import ASR_OOM_HINT, AzureMaiTranscriber, WhisperXAligner, WhisperXTranscriber
from ..audio.preprocess import PreparedAudio, prepare_audio
from ..config import AppConfig, speaker_count_kwargs, validate_config
from ..device import (
    apply_torch_threads,
    gpu_memory_snapshot,
    oom_guard,
    release_gpu_memory,
    resolve_device,
    torch_device_string,
)
from ..diarization import DIARIZATION_OOM_HINT, DiarizationResult, PyannoteDiarizer
from ..errors import ConfigError, UserFacingError
from ..export import write_outputs
from ..logging_utils import get_logger, redact
from ..speaker_id import (
    IdentificationResult,
    SpeakerProfileStore,
    WeSpeakerEmbedder,
    identify_speakers,
    load_cohort_mean,
    passthrough_matches,
)
from .assignment import build_utterances

logger = get_logger(__name__)

STAGES = [
    ("preprocess", "1. Audio preprocessing"),
    ("transcribe", "2. Transcription"),
    ("align", "3. Alignment"),
    ("diarize", "4. Diarization"),
    ("identify", "5. Speaker identification"),
    ("export", "6. Export"),
]

ProgressFn = Callable[[int, str, float], None]

SCHEMA_VERSION = "1.0"


@dataclass
class RunRequest:
    input_path: str
    language: Optional[str] = None          # "ko", "auto", ... (None = config)
    speaker_mode: Optional[str] = None      # AUTO | RANGE | FIXED
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    whisper_model: Optional[str] = None
    asr_backend: Optional[str] = None       # azure_mai | whisperx (None = config)
    phrases: Optional[list] = None          # extra keyword-biasing terms (Azure MAI)
    identify: Optional[bool] = None
    matching_mode: Optional[str] = None
    match_threshold: Optional[float] = None
    original_name: Optional[str] = None     # display name when input_path is a temp upload


@dataclass
class RunResult:
    document: dict
    files: dict
    job_dir: Path
    timings: dict
    identification: Optional[IdentificationResult] = None
    warnings: list = field(default_factory=list)


def effective_config(base: AppConfig, req: RunRequest) -> AppConfig:
    cfg = base.copy()
    if req.language:
        cfg.asr.language = req.language
    if req.whisper_model:
        cfg.asr.model = req.whisper_model
    if req.asr_backend:
        cfg.asr.backend = req.asr_backend
    if req.speaker_mode:
        cfg.diarization.mode = req.speaker_mode.upper()
        if cfg.diarization.mode == "FIXED":
            cfg.diarization.num_speakers = req.num_speakers
        if cfg.diarization.mode == "RANGE":
            cfg.diarization.min_speakers = req.min_speakers
            cfg.diarization.max_speakers = req.max_speakers
    if req.identify is not None:
        cfg.speaker_id.enabled = bool(req.identify)
    if req.matching_mode:
        cfg.speaker_id.matching_mode = req.matching_mode
    if req.match_threshold is not None:
        cfg.speaker_id.match_threshold = float(req.match_threshold)
    validate_config(cfg)
    return cfg


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")
    return (stem or "meeting")[:40]


class MeetingTranscriber:
    """Reusable pipeline object (the UI keeps one instance)."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._cache: dict = {}

    # ------------------------------------------------------------------ model handles
    def _get(self, key: tuple, factory):
        if self.cfg.runtime.release_models_between_stages:
            return factory()
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def _done(self, model) -> None:
        if self.cfg.runtime.release_models_between_stages and model is not None:
            model.unload()

    def make_embedder(self, cfg: AppConfig, device: str) -> WeSpeakerEmbedder:
        dev = torch_device_string(device, cfg.runtime.device_index)
        return self._get(
            ("wespeaker", cfg.speaker_id.wespeaker_model, dev),
            lambda: WeSpeakerEmbedder(cfg.speaker_id.wespeaker_model, dev, cfg.model_cache_dir),
        )

    def release_embedder(self, embedder) -> None:
        self._done(embedder)
        if self.cfg.runtime.release_models_between_stages:
            release_gpu_memory()

    def clear_cache(self) -> None:
        for model in self._cache.values():
            try:
                model.unload()
            except Exception:  # pragma: no cover
                pass
        self._cache.clear()
        release_gpu_memory()

    # ------------------------------------------------------------------ run
    def run(self, req: RunRequest, progress: Optional[ProgressFn] = None) -> RunResult:
        cfg = effective_config(self.cfg, req)
        device = resolve_device(cfg.runtime.device)
        apply_torch_threads(cfg)
        speaker_kwargs = speaker_count_kwargs(
            cfg.diarization.mode, cfg.diarization.num_speakers, cfg.diarization.min_speakers,
            cfg.diarization.max_speakers, cfg.diarization.speaker_limit,
        )
        src = Path(req.input_path)
        display_name = req.original_name or src.name
        job_dir = cfg.output_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{_safe_stem(display_name)}"
        job_dir.mkdir(parents=True, exist_ok=True)

        timings: dict[str, float] = {}
        warnings: list[str] = []
        report = progress or (lambda i, label, frac: None)
        t_total = time.perf_counter()
        if device == "cuda":
            try:
                import torch

                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        def stage(idx: int, fn, oom_hint: str = ""):
            key, label = STAGES[idx]
            report(idx, label, 0.0)
            t0 = time.perf_counter()
            try:
                with oom_guard(label, oom_hint):
                    out = fn(lambda frac: report(idx, label, max(0.0, min(1.0, frac))))
            except UserFacingError as exc:
                exc.stage = exc.stage or label
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Stage %s failed", key)
                raise UserFacingError(
                    f"처리 중 오류가 발생했습니다: {type(exc).__name__}: {redact(str(exc))[:400]}", stage=label
                ) from exc
            timings[key] = round(time.perf_counter() - t0, 2)
            report(idx, label, 1.0)
            return out

        try:
            # 1. preprocess
            prepared: PreparedAudio = stage(0, lambda p: prepare_audio(src, job_dir, cfg.audio))
            warnings.extend(prepared.notes)
            audio, sr = prepared.audio, prepared.sample_rate

            # 2. ASR
            def run_asr(p):
                if cfg.asr.backend == "azure_mai":
                    # cloud ASR: the 16 kHz working copy is uploaded to the configured Azure Speech resource
                    mai = AzureMaiTranscriber(cfg)
                    out = mai.transcribe(audio, sr, job_dir, progress=p, extra_phrases=req.phrases)
                    out["asr_info"] = mai.describe()
                    warnings.extend(mai.notes)
                    return out
                asr = self._get(("asr", cfg.asr.model, device, cfg.asr.compute_type, cfg.asr.language, cfg.asr.vad_method),
                                lambda: WhisperXTranscriber(cfg, device))
                try:
                    return asr.transcribe(audio, progress=p)
                finally:
                    self._done(asr)

            transcript = stage(1, run_asr, ASR_OOM_HINT)
            language = transcript.get("language") or cfg.asr.language
            if not transcript.get("segments"):
                warnings.append("음성 구간에서 전사된 텍스트가 없습니다.")

            # 3. alignment
            def run_align(p):
                if transcript.get("word_timestamps_source"):
                    # Azure MAI already returns word timestamps: forced alignment is not needed
                    return dict(transcript, align_model=f"{transcript['word_timestamps_source']} word timestamps")
                if not cfg.alignment.enabled:
                    return {"segments": [dict(s, words=[]) for s in transcript["segments"]], "aligned": False,
                            "language": language}
                aligner = self._get(("align", language, cfg.alignment.model_name, device), lambda: WhisperXAligner(cfg, device))
                try:
                    out = aligner.align(transcript, audio, progress=p)
                    out["align_model"] = aligner.model_name
                    return out
                finally:
                    self._done(aligner)

            aligned = stage(2, run_align, "alignment.enabled=false 로 단어 정렬을 끄거나 GPU 메모리를 확보하세요.")
            if not aligned.get("aligned") and transcript.get("segments"):
                warnings.append(f"언어 '{language}' 단어 단위 정렬을 수행하지 못했습니다 (세그먼트 단위로 화자 할당).")

            # 4. diarization
            def run_diar(p):
                diarizer = self._get(("diar", cfg.diarization.model, device), lambda: PyannoteDiarizer(cfg, device))
                try:
                    return diarizer.diarize(audio, sr, speaker_kwargs, progress=p)
                finally:
                    self._done(diarizer)

            apply_torch_threads(cfg)  # guard against import-time thread changes in earlier stages
            diar: DiarizationResult = stage(3, run_diar, DIARIZATION_OOM_HINT)
            if diar.num_speakers == 0:
                warnings.append("화자 분리 결과 발화 구간이 없습니다.")
            if not diar.exclusive_available:
                warnings.append("exclusive diarization 미제공 파이프라인 - 대체 규칙으로 단일 화자 할당")

            # 5. speaker identification
            def run_identify(p):
                if not cfg.speaker_id.enabled:
                    matches = passthrough_matches(diar.labels, "화자 식별 OFF")
                    return IdentificationResult({m.cluster: m for m in matches}, clusters=diar.labels, enabled=False)
                store = SpeakerProfileStore(cfg.enrollment_dir)
                if not store.list_profiles():
                    matches = passthrough_matches(diar.labels, "등록된 화자 없음")
                    return IdentificationResult({m.cluster: m for m in matches}, clusters=diar.labels,
                                                notes=["등록된 화자 프로필이 없어 익명 화자(SPEAKER_xx)로 출력합니다."])
                embedder = self.make_embedder(cfg, device)
                try:
                    cohort = load_cohort_mean(cfg.speaker_id.cohort_mean_path, cfg.resolve_path)
                    return identify_speakers(diar, audio, sr, store, embedder, cfg.speaker_id, cohort)
                finally:
                    self.release_embedder(embedder)

            ident: IdentificationResult = stage(4, run_identify, "speaker_id.cluster_embedding.max_segments 를 줄이세요.")
            warnings.extend(ident.notes)

            # 6. export
            def run_export(p):
                doc = self._document(cfg, device, display_name, prepared, transcript, aligned, diar, ident,
                                     speaker_kwargs, language, timings, warnings)
                files = write_outputs(doc, job_dir, _safe_stem(display_name), cfg.export.formats,
                                      cfg.export.txt_merge_gap_sec, cfg.export.include_words_in_json)
                return doc, files

            doc, files = stage(5, run_export)
        finally:
            if not cfg.audio.keep_intermediate_wav:
                for f in job_dir.glob("audio_16k_mono.wav"):
                    f.unlink(missing_ok=True)

        total = round(time.perf_counter() - t_total, 2)
        timings["total"] = total
        doc["processing"]["timings_sec"] = timings
        doc["processing"]["total_sec"] = total
        doc["processing"]["real_time_factor"] = round(total / prepared.duration_sec, 4) if prepared.duration_sec else None
        doc["processing"].update(gpu_memory_snapshot())
        # rewrite JSON with final timings (export stage itself is included)
        if "json" in files:
            payload = doc if cfg.export.include_words_in_json else {
                **doc, "segments": [{k: v for k, v in s.items() if k != "words"} for s in doc["segments"]]
            }
            files["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return RunResult(doc, files, job_dir, timings, ident, warnings)

    # ------------------------------------------------------------------ document
    def _document(self, cfg, device, display_name, prepared, transcript, aligned, diar, ident,
                  speaker_kwargs, language, timings, warnings) -> dict:
        labels = {c: ident.label_of(c) for c in diar.labels}
        sims = {c: (m.similarity if m.status == "identified" else None) for c, m in ident.matches.items()}
        statuses = {c: m.status for c, m in ident.matches.items()}
        segments = build_utterances(aligned, diar, labels, sims, statuses, cfg.assignment, language)

        speakers = []
        for c in sorted(diar.labels, key=diar.first_appearance):
            m = ident.matches.get(c)
            ev = ident.evidence.get(c)
            speakers.append({
                "cluster": c,
                "label": labels[c],
                "status": m.status if m else "not_identified",
                "name": m.name if m else None,
                "similarity": None if not m or m.similarity is None else round(m.similarity, 4),
                "best_candidate": m.best_name if m else None,
                "best_similarity": None if not m or m.best_similarity is None else round(m.best_similarity, 4),
                "margin": None if not m or m.margin is None or m.margin == float("inf") else round(m.margin, 4),
                "matched_via": m.via if m else None,
                "reason": m.reason if m else "",
                "candidates": m.candidates if m else [],
                "speech_duration": round(diar.speech_duration(c), 2),
                "num_segments": sum(1 for s in segments if s["speaker_cluster"] == c),
                "first_appearance": round(diar.first_appearance(c), 3),
                "evidence_sec": round(ev.evidence_sec, 2) if ev else None,
                "low_evidence": ev.low_evidence if ev else None,
                "window_agreement": (None if ident.window_agreement(c) is None
                                     else round(ident.window_agreement(c), 3)),
            })

        ident_dict = ident.to_dict()
        ident_dict.update({
            "threshold": cfg.speaker_id.match_threshold,
            "margin": cfg.speaker_id.match_margin,
            "matching_mode": cfg.speaker_id.matching_mode,
            "score": "raw cosine" + (" (cohort mean-subtracted)" if cfg.speaker_id.cohort_mean_path else ""),
        })
        return {
            "schema_version": SCHEMA_VERSION,
            "file": display_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "language": language,
            "duration": round(prepared.duration_sec, 3),
            "speaker_count": diar.num_speakers,
            "speakers": speakers,
            "segments": segments,
            "diarization": diar.to_dict(),
            "speaker_identification": ident_dict,
            "models": {
                "asr": cfg.asr.azure.model if cfg.asr.backend == "azure_mai" else cfg.asr.model,
                "asr_backend": ("Azure Speech (Microsoft Foundry) - " + (transcript.get("asr_info") or {}).get("endpoint_host", "")
                                if cfg.asr.backend == "azure_mai" else "faster-whisper (via WhisperX)"),
                "alignment": aligned.get("align_model"),
                "diarization": cfg.diarization.model,
                "speaker_embedding": ident.model_id,
            },
            "settings": {
                "device": device,
                "asr_backend": cfg.asr.backend,
                "asr_requests": transcript.get("requests"),
                "asr_languages_detected": transcript.get("languages_detected"),
                "asr_phrase_list": transcript.get("phrase_list"),
                "asr_locale_forced": transcript.get("locale_forced"),
                "compute_type": cfg.asr.compute_type,
                "language_requested": cfg.asr.language,
                "speaker_mode": cfg.diarization.mode,
                "speaker_kwargs": speaker_kwargs,
                "speaker_identification": cfg.speaker_id.enabled,
                "matching_mode": cfg.speaker_id.matching_mode,
                "match_threshold": cfg.speaker_id.match_threshold,
            },
            "audio": {
                "source_duration": round(prepared.source_info.duration_sec or 0.0, 3),
                "source_channels": prepared.source_info.channels,
                "source_sample_rate": prepared.source_info.sample_rate,
                "source_codec": prepared.source_info.codec,
                "gain_db": round(prepared.gain_db, 2),
                "rms_dbfs": round(prepared.rms_dbfs, 2),
            },
            "processing": {"timings_sec": dict(timings)},
            "warnings": list(warnings),
        }


def run_pipeline(cfg: AppConfig, input_path: str, **kwargs) -> RunResult:
    """Convenience wrapper for scripts."""
    if not Path(input_path).exists():
        raise ConfigError(f"입력 파일이 없습니다: {input_path}")
    return MeetingTranscriber(cfg).run(RunRequest(input_path=input_path, **kwargs))


__all__ = ["STAGES", "MeetingTranscriber", "RunRequest", "RunResult", "effective_config", "run_pipeline"]
