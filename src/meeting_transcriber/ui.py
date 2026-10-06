"""Gradio UI: Transcription / Speaker Enrollment / Diagnostics tabs."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Optional

from .config import AppConfig
from .errors import UserFacingError
from .logging_utils import get_logger, redact

logger = get_logger(__name__)

LANGUAGES = [("한국어 (ko)", "ko"), ("자동 감지", "auto"), ("English (en)", "en"), ("日本語 (ja)", "ja"), ("中文 (zh)", "zh")]
WHISPER_MODELS = ["large-v3", "large-v3-turbo", "medium", "small"]
SPEAKER_HEADERS = ["클러스터", "최종 화자", "상태", "유사도(cos)", "최고 후보", "후보 유사도", "margin", "발화(초)", "근거(초)", "사유"]
PROFILE_HEADERS = ["이름", "ID", "샘플 수", "사용 구간", "유효 음성(초)", "원본 음성 보관", "임베딩 모델", "갱신"]

# One pipeline run at a time on a single GPU; a second click waits instead of OOM-ing.
_RUN_LOCK = threading.Lock()


def _fmt(v, nd=3):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else v)


def build_app(cfg: AppConfig):
    import gradio as gr

    from .device import collect_diagnostics
    from .export.writers import render_txt
    from .pipeline import STAGES, MeetingTranscriber, RunRequest
    from .speaker_id import SpeakerProfileStore

    transcriber = MeetingTranscriber(cfg)
    store = SpeakerProfileStore(cfg.enrollment_dir)
    limit = cfg.diarization.speaker_limit
    last_run: dict = {}

    # ------------------------------------------------------------------ helpers
    def profile_rows():
        return [[p.name, p.speaker_id, p.num_samples, f"{p.num_chunks_used}/{len(p.chunk_keep)}",
                 round(p.total_speech_sec, 1), "예" if p.raw_audio_kept else "아니오", p.model_id, p.updated_at]
                for p in store.list_profiles()]

    def profile_choices():
        return [(f"{p.name} ({p.speaker_id})", p.speaker_id) for p in store.list_profiles()]

    def error_text(exc: Exception) -> str:
        if isinstance(exc, UserFacingError):
            return f"❌ {exc}"
        logger.error("Unexpected error: %s", redact(traceback.format_exc()))
        return f"❌ 예상하지 못한 오류: {type(exc).__name__}: {redact(str(exc))[:400]}"

    # ------------------------------------------------------------------ transcription
    def on_mode(mode):
        return (gr.update(visible=mode == "FIXED"), gr.update(visible=mode == "RANGE"), gr.update(visible=mode == "RANGE"))

    def run_transcription(file, language, mode, num, min_s, max_s, model, identify, matching_mode, threshold,
                          progress=gr.Progress()):  # noqa: B008 - Gradio injects progress via this default
        empty = ("", [], [], None)
        if not file:
            return ("❌ 파일을 업로드하세요.", *empty)
        path = file if isinstance(file, str) else getattr(file, "name", None)

        def cb(i, label, frac):
            progress((i + frac) / len(STAGES), desc=f"{label} ({frac * 100:.0f}%)")

        req = RunRequest(
            input_path=path,
            language=language,
            speaker_mode=mode,
            num_speakers=int(num) if num else None,
            min_speakers=int(min_s) if min_s else None,
            max_speakers=int(max_s) if max_s else None,
            whisper_model=model,
            identify=bool(identify),
            matching_mode=matching_mode,
            match_threshold=float(threshold),
            original_name=Path(path).name,
        )
        try:
            with _RUN_LOCK:
                result = transcriber.run(req, cb)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            return (error_text(exc), *empty)

        doc = result.document
        last_run.clear()
        last_run.update({
            "file": doc["file"],
            "detected_speakers": doc["speaker_count"],
            "audio_duration_sec": doc["duration"],
            "processing_sec": doc["processing"]["total_sec"],
            "real_time_factor": doc["processing"]["real_time_factor"],
            "stage_timings_sec": doc["processing"]["timings_sec"],
            "device": doc["settings"]["device"],
            "whisper_model": doc["models"]["asr"],
            "alignment_model": doc["models"]["alignment"],
            "pyannote_model": doc["models"]["diarization"],
            "speaker_embedding_model": doc["models"]["speaker_embedding"],
            "gpu_peak_allocated_gb": doc["processing"].get("torch_peak_allocated_gb"),
            "output_dir": str(result.job_dir),
        })
        speakers = [[s["cluster"], s["label"], s["status"], _fmt(s["similarity"]), s["best_candidate"] or "",
                     _fmt(s["best_similarity"]), _fmt(s["margin"]), s["speech_duration"], s["evidence_sec"] or "",
                     s["reason"]] for s in doc["speakers"]]
        sim = doc["speaker_identification"].get("similarity_matrix")
        sim_rows = []
        if sim:
            profiles = doc["speaker_identification"]["profiles"]
            sim_rows = [[c] + [round(x, 3) for x in row] for c, row in zip(doc["speaker_identification"]["clusters"], sim)]
            sim_header = ["클러스터"] + profiles
        else:
            sim_header = ["클러스터"]
        status = (f"✅ 완료: 화자 {doc['speaker_count']}명, 길이 {doc['duration']:.1f}s, 처리 {doc['processing']['total_sec']}s "
                  f"(RTF {doc['processing']['real_time_factor']}) → {result.job_dir}")
        if result.warnings:
            status += "\n" + "\n".join(f"⚠ {w}" for w in result.warnings)
        files = [str(p) for p in result.files.values()]
        return (status, render_txt(doc, cfg.export.txt_merge_gap_sec), speakers,
                gr.update(value=sim_rows, headers=sim_header), files)

    # ------------------------------------------------------------------ enrollment
    def enroll(name, files, replace, keep_audio):
        if not files:
            return "❌ 등록할 음성 파일을 업로드하세요.", profile_rows(), gr.update(choices=profile_choices())
        paths = [f if isinstance(f, str) else getattr(f, "name", None) for f in files]
        try:
            with _RUN_LOCK:
                emb = transcriber.make_embedder(cfg, _device())
                try:
                    rep = store.enroll(name, paths, emb, cfg.speaker_id.enrollment, replace=bool(replace),
                                       keep_raw_audio=bool(keep_audio))
                finally:
                    transcriber.release_embedder(emb)
            msg = "✅ " + rep.message()
        except Exception as exc:  # noqa: BLE001
            msg = error_text(exc)
        return msg, profile_rows(), gr.update(choices=profile_choices())

    def _device():
        from .device import resolve_device

        return resolve_device(cfg.runtime.device)

    def delete_profile(speaker_id):
        if not speaker_id:
            return "❌ 삭제할 화자를 선택하세요.", profile_rows(), gr.update(choices=profile_choices(), value=None)
        p = store.get(speaker_id)
        store.delete(speaker_id)
        return (f"🗑 삭제 완료: {p.name if p else speaker_id} (등록 음성 + 임베딩)", profile_rows(),
                gr.update(choices=profile_choices(), value=None))

    def delete_raw(speaker_id):
        if not speaker_id:
            return "❌ 화자를 선택하세요.", profile_rows()
        n = store.delete_raw_audio(speaker_id)
        return f"🗑 원본 등록 음성 {n}개 삭제 (임베딩 프로필은 유지)", profile_rows()

    def reenroll_stored(speaker_id):
        if not speaker_id:
            return "❌ 화자를 선택하세요.", profile_rows()
        try:
            with _RUN_LOCK:
                emb = transcriber.make_embedder(cfg, _device())
                try:
                    rep = store.reenroll_from_stored(speaker_id, emb, cfg.speaker_id.enrollment)
                finally:
                    transcriber.release_embedder(emb)
            return "✅ 재계산 " + rep.message(), profile_rows()
        except Exception as exc:  # noqa: BLE001
            return error_text(exc), profile_rows()

    # ------------------------------------------------------------------ diagnostics
    def diagnostics():
        info = collect_diagnostics(cfg)
        info["last_run"] = dict(last_run) if last_run else "아직 실행 기록 없음"
        return info

    # ------------------------------------------------------------------ layout
    with gr.Blocks(title="한국어 회의 다화자 전사 PoC", analytics_enabled=False) as demo:
        gr.Markdown("## 한국어 회의 다화자 전사 (WhisperX → pyannote Community-1 → WeSpeaker 화자 식별)\n"
                    "모든 추론은 로컬에서 수행됩니다. 결과는 자동 생성물이므로 검토 후 사용하세요.")
        with gr.Tab("1. 전사 (Transcription)"):
            with gr.Row():
                with gr.Column(scale=1):
                    f_in = gr.File(label="회의 음성/영상 (WAV, MP3, M4A, MP4, MOV)",
                                   file_types=[".wav", ".mp3", ".m4a", ".mp4", ".mov"], type="filepath")
                    lang = gr.Dropdown(LANGUAGES, value=cfg.asr.language, label="Language")
                    mode = gr.Radio(["AUTO", "RANGE", "FIXED"], value=cfg.diarization.mode, label="Speaker Count Mode",
                                    info="참석 인원을 정확히 알면 FIXED 가 가장 정확합니다.")
                    num = gr.Slider(1, limit, value=cfg.diarization.num_speakers or min(2, limit), step=1,
                                    label="Exact Speaker Count", visible=cfg.diarization.mode == "FIXED")
                    min_s = gr.Slider(1, limit, value=cfg.diarization.min_speakers or 1, step=1, label="Min Speakers",
                                      visible=cfg.diarization.mode == "RANGE")
                    max_s = gr.Slider(1, limit, value=cfg.diarization.max_speakers or limit, step=1, label="Max Speakers",
                                      visible=cfg.diarization.mode == "RANGE")
                    model = gr.Dropdown(WHISPER_MODELS, value=cfg.asr.model, label="Whisper Model", allow_custom_value=True)
                    identify = gr.Checkbox(value=cfg.speaker_id.enabled, label="Registered Speaker Identification (ON/OFF)")
                    with gr.Accordion("화자 식별 고급 설정", open=False):
                        matching = gr.Radio(["flexible", "strict_one_to_one", "argmax"], value=cfg.speaker_id.matching_mode,
                                            label="Matching mode", info="argmax 는 비교용 baseline")
                        thr = gr.Slider(-1.0, 1.0, value=cfg.speaker_id.match_threshold, step=0.01,
                                        label="speaker_match_threshold (raw cosine, 환경별 calibration 필요)")
                    run_btn = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    status = gr.Textbox(label="상태", lines=3)
                    transcript = gr.Textbox(label="Transcript", lines=22, buttons=["copy"])
            spk_table = gr.Dataframe(headers=SPEAKER_HEADERS, label="Speaker list / 식별 결과", interactive=False, wrap=True)
            sim_table = gr.Dataframe(label="Speaker similarity (클러스터 × 등록 화자, raw cosine)", interactive=False)
            downloads = gr.File(label="Download (TXT / JSON / CSV / SRT)", file_count="multiple", interactive=False)
            mode.change(on_mode, mode, [num, min_s, max_s])
            run_btn.click(run_transcription,
                          [f_in, lang, mode, num, min_s, max_s, model, identify, matching, thr],
                          [status, transcript, spk_table, sim_table, downloads])

        with gr.Tab("2. 화자 등록 (Speaker Enrollment)"):
            gr.Markdown("한 사람당 **3개 이상, 각 10~30초**의 조용한 환경 발화를 권장합니다. "
                        "같은 이름으로 다시 등록하면 샘플이 **추가**되고, '재등록(덮어쓰기)'를 체크하면 기존 프로필을 교체합니다. "
                        "음성/임베딩은 이 PC의 `" + str(cfg.enrollment_dir) + "` 에만 저장됩니다.")
            with gr.Row():
                with gr.Column():
                    e_name = gr.Textbox(label="Speaker Name", placeholder="예: 이용욱")
                    e_files = gr.File(label="Enrollment Audio Upload", file_count="multiple",
                                      file_types=[".wav", ".mp3", ".m4a", ".mp4", ".mov", ".flac"], type="filepath")
                    e_replace = gr.Checkbox(label="재등록(덮어쓰기, Re-enroll)", value=False)
                    e_keep = gr.Checkbox(label="원본 등록 음성 보관 (모델 변경 시 재계산용)", value=cfg.speaker_id.enrollment.keep_raw_audio)
                    e_btn = gr.Button("등록", variant="primary")
                with gr.Column():
                    e_status = gr.Textbox(label="결과", lines=6)
                    sel = gr.Dropdown(choices=profile_choices(), label="화자 선택", value=None)
                    with gr.Row():
                        del_btn = gr.Button("Delete (프로필 삭제)", variant="stop")
                        raw_btn = gr.Button("원본 음성만 삭제")
                        re_btn = gr.Button("저장 음성으로 재계산")
                    refresh = gr.Button("목록 새로고침")
            profiles = gr.Dataframe(value=profile_rows(), headers=PROFILE_HEADERS, label="Registered Speaker List",
                                    interactive=False)
            e_btn.click(enroll, [e_name, e_files, e_replace, e_keep], [e_status, profiles, sel])
            del_btn.click(delete_profile, sel, [e_status, profiles, sel])
            raw_btn.click(delete_raw, sel, [e_status, profiles])
            re_btn.click(reenroll_stored, sel, [e_status, profiles])
            refresh.click(lambda: (profile_rows(), gr.update(choices=profile_choices())), None, [profiles, sel])

        with gr.Tab("3. 진단 (Diagnostics)"):
            d_btn = gr.Button("새로고침")
            d_json = gr.JSON(label="Device / CUDA / Models / 최근 실행")
            d_btn.click(diagnostics, None, d_json)
            demo.load(diagnostics, None, d_json)
    return demo


def launch(cfg: AppConfig, server_name: Optional[str] = None, server_port: Optional[int] = None,
           prevent_thread_lock: bool = False):
    demo = build_app(cfg)
    demo.queue(default_concurrency_limit=1)
    host = server_name or cfg.ui.server_name
    if host not in ("127.0.0.1", "localhost"):
        logger.warning("UI is bound to %s - other machines on the network can reach it.", host)
    return demo.launch(
        server_name=host,
        server_port=server_port or cfg.ui.server_port,
        share=False,  # never tunnel voice data through gradio.live
        max_file_size=cfg.ui.max_file_size,
        allowed_paths=[str(cfg.output_dir)],
        show_error=True,
        prevent_thread_lock=prevent_thread_lock,
    )
