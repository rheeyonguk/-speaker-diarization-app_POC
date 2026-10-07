"""Gradio UI: Transcription / Speaker Enrollment / Diagnostics tabs."""

from __future__ import annotations

import html
import threading
import traceback
from pathlib import Path
from typing import Optional

from .config import AppConfig, azure_speech_endpoint, azure_speech_key
from .errors import UserFacingError
from .logging_utils import get_logger, redact

logger = get_logger(__name__)

LANGUAGES = [("한국어 (ko)", "ko"), ("자동 감지", "auto"), ("English (en)", "en"), ("日本語 (ja)", "ja"), ("中文 (zh)", "zh")]
WHISPER_MODELS = ["large-v3", "large-v3-turbo", "medium", "small"]
SPEAKER_MODES = [("자동 추정 (AUTO)", "AUTO"), ("범위 지정 (RANGE)", "RANGE"), ("인원 고정 (FIXED)", "FIXED")]
ASR_BACKENDS = [("Azure MAI-Transcribe-2 (사내 Azure)", "azure_mai"), ("WhisperX large-v3 (로컬)", "whisperx")]
SPEAKER_HEADERS = ["클러스터", "최종 화자", "상태", "유사도(cos)", "최고 후보", "후보 유사도", "margin", "발화(초)", "근거(초)", "사유"]
PROFILE_HEADERS = ["이름", "ID", "샘플 수", "사용 구간", "유효 음성(초)", "원본 음성 보관", "임베딩 모델", "갱신"]

# One pipeline run at a time on a single GPU; a second click waits instead of OOM-ing.
_RUN_LOCK = threading.Lock()


def _fmt(v, nd=3):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else v)


def build_app(cfg: AppConfig):
    import gradio as gr

    from .asr.azure_mai import parse_phrases_text
    from .branding import footer_html, header_html, page_title, steps_html
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

    def on_backend(backend):
        local = backend == "whisperx"
        return gr.update(visible=local), gr.update(visible=not local)

    def run_transcription(file, language, mode, num, min_s, max_s, model, identify, matching_mode, threshold,
                          backend=None, phrases=None,
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
            asr_backend=backend or None,
            phrases=parse_phrases_text(phrases),
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
            "asr_backend": doc["settings"].get("asr_backend"),
            "asr_model": doc["models"]["asr"],
            "asr_service": doc["models"]["asr_backend"],
            "word_timestamps": doc["models"]["alignment"],
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

    def azure_check():
        import tempfile

        from .asr import AzureMaiTranscriber

        try:
            with tempfile.TemporaryDirectory() as tmp:
                res = AzureMaiTranscriber(cfg).check(Path(tmp))
            msg = (f"✅ Azure MAI 연결 성공: {res['endpoint_host']} / {res['model']} / 응답 {res['latency_sec']}초")
            if res["notes"]:
                msg += "\n" + "\n".join(f"ℹ {n}" for n in res["notes"])
            return msg
        except Exception as exc:  # noqa: BLE001
            return error_text(exc)

    def azure_save(endpoint_text, key_text):
        """Store endpoint (URL, portal resource URL or resource name) + key in the local .env, then test."""
        from .asr.azure_mai import normalize_endpoint
        from .config import save_env_values

        keep = (gr.update(), gr.update())
        try:
            raw = (endpoint_text or "").strip() or azure_speech_endpoint(cfg) or ""
            if not raw:
                return ("엔드포인트 또는 Azure 포털 리소스 URL을 입력하세요.", *keep)
            base, notes = normalize_endpoint(raw)
            values = {"AZURE_SPEECH_ENDPOINT": base}
            key = (key_text or "").strip()
            if key:
                values["AZURE_SPEECH_KEY"] = key
            elif cfg.asr.azure.auth == "key" and not azure_speech_key():
                return ("키(KEY 1 또는 KEY 2)를 입력하세요. 포털 > 리소스 > 키 및 엔드포인트", *keep)
            path = save_env_values(values)
        except Exception as exc:  # noqa: BLE001
            return (error_text(exc), *keep)
        msg = "\n".join([f"💾 저장 완료: {base} → {path.name} (이 PC에만 저장, Git 제외)",
                          *(f"ℹ {n}" for n in notes), azure_check()])
        # clear the key box so the secret does not stay in the page
        return msg, gr.update(value=base), gr.update(value="", placeholder="설정됨 — 변경할 때만 입력")

    # ------------------------------------------------------------------ layout
    with gr.Blocks(title=page_title(cfg), analytics_enabled=False) as demo:
        gr.HTML(header_html(cfg))
        with gr.Tab("회의록 생성 (Transcription)"):
            gr.HTML(steps_html())
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=340):
                    gr.HTML('<div class="mt-section-title">회의 파일</div>')
                    f_in = gr.File(label="회의 음성/영상 업로드 (WAV · MP3 · M4A · MP4 · MOV)",
                                   file_types=[".wav", ".mp3", ".m4a", ".mp4", ".mov"], type="filepath", height=170)
                    gr.HTML('<div class="mt-section-title">인식 설정</div>')
                    with gr.Group():
                        backend = gr.Radio(ASR_BACKENDS, value=cfg.asr.backend, label="음성인식 엔진 (ASR)",
                                           info="Azure: 회의 음성이 사내 Azure Speech 리소스로 전송됩니다 / 로컬: 외부 전송 없음")
                        lang = gr.Dropdown(LANGUAGES, value=cfg.asr.language, label="언어 (Language)",
                                           info="Azure MAI 는 자동 감지가 기본입니다 (asr.azure.force_locale)")
                        phrases = gr.Textbox(label="전문용어 우선 인식 (Azure MAI keyword biasing)", lines=2,
                                             placeholder="쉼표/줄바꿈 구분 예: APQR, CAPA, 일탈, 변경관리, 제품명",
                                             visible=cfg.asr.backend == "azure_mai")
                        model = gr.Dropdown(WHISPER_MODELS, value=cfg.asr.model, label="로컬 음성인식 모델 (Whisper Model)",
                                            info="GPU 가 없으면 large-v3-turbo 권장", allow_custom_value=True,
                                            visible=cfg.asr.backend == "whisperx")
                    gr.HTML('<div class="mt-section-title">참석자 수</div>')
                    with gr.Group():
                        mode = gr.Radio(SPEAKER_MODES, value=cfg.diarization.mode, label="화자 수 설정 (Speaker Count Mode)",
                                        info="참석 인원을 정확히 알면 '인원 고정'이 가장 정확합니다.")
                        num = gr.Slider(1, limit, value=cfg.diarization.num_speakers or min(2, limit), step=1,
                                        label="참석 인원 (Exact Speaker Count)", visible=cfg.diarization.mode == "FIXED")
                        min_s = gr.Slider(1, limit, value=cfg.diarization.min_speakers or 1, step=1,
                                          label="최소 인원 (Min Speakers)", visible=cfg.diarization.mode == "RANGE")
                        max_s = gr.Slider(1, limit, value=cfg.diarization.max_speakers or limit, step=1,
                                          label="최대 인원 (Max Speakers)", visible=cfg.diarization.mode == "RANGE")
                    gr.HTML('<div class="mt-section-title">화자 식별</div>')
                    with gr.Group():
                        identify = gr.Checkbox(value=cfg.speaker_id.enabled,
                                               label="등록 화자 자동 식별 (Registered Speaker Identification ON/OFF)")
                        with gr.Accordion("고급 설정 (매칭 방식 · 기준값)", open=False):
                            matching = gr.Radio(["flexible", "strict_one_to_one", "argmax"],
                                                value=cfg.speaker_id.matching_mode, label="매칭 방식 (Matching mode)",
                                                info="argmax 는 비교용 baseline")
                            thr = gr.Slider(-1.0, 1.0, value=cfg.speaker_id.match_threshold, step=0.01,
                                            label="화자 일치 기준값 (speaker_match_threshold, 환경별 calibration 필요)")
                    run_btn = gr.Button("회의록 생성 (Run)", variant="primary", size="lg", elem_classes=["mt-run"])
                with gr.Column(scale=3, min_width=420):
                    gr.HTML('<div class="mt-section-title">진행 상태</div>')
                    status = gr.Textbox(show_label=False, lines=3, placeholder="파일을 올리고 '회의록 생성'을 누르세요.")
                    gr.HTML('<div class="mt-section-title">회의록 (Transcript)</div>')
                    transcript = gr.Textbox(show_label=False, lines=24, buttons=["copy"],
                                            placeholder="[시간] / 화자명 / 발화 내용 형식으로 표시됩니다.")
            gr.HTML('<div class="mt-section-title">화자 식별 결과 (Speaker list)</div>')
            spk_table = gr.Dataframe(headers=SPEAKER_HEADERS, show_label=False, interactive=False, wrap=True)
            with gr.Accordion("화자 유사도 행렬 (Speaker similarity: 클러스터 × 등록 화자, raw cosine)", open=False):
                sim_table = gr.Dataframe(headers=["클러스터"], show_label=False, interactive=False)
            gr.HTML('<div class="mt-section-title">결과 파일 다운로드</div>')
            downloads = gr.File(label="TXT · JSON · CSV · SRT", file_count="multiple", interactive=False, height=120)
            mode.change(on_mode, mode, [num, min_s, max_s])
            backend.change(on_backend, backend, [model, phrases])
            run_btn.click(run_transcription,
                          [f_in, lang, mode, num, min_s, max_s, model, identify, matching, thr, backend, phrases],
                          [status, transcript, spk_table, sim_table, downloads])

        with gr.Tab("화자 등록 (Speaker Enrollment)"):
            gr.HTML('<div class="mt-note">한 사람당 <b>3개 이상, 각 10~30초</b>의 조용한 환경 발화를 권장합니다. '
                    "같은 이름으로 다시 등록하면 샘플이 <b>추가</b>되고, '재등록(덮어쓰기)'을 체크하면 기존 프로필을 교체합니다. "
                    "등록 음성과 음성 특징값은 이 PC(<code>" + html.escape(str(cfg.enrollment_dir)) + "</code>)에만 저장됩니다.</div>")
            with gr.Row(equal_height=False):
                with gr.Column():
                    gr.HTML('<div class="mt-section-title">신규 등록 / 샘플 추가</div>')
                    with gr.Group():
                        e_name = gr.Textbox(label="이름 (Speaker Name)", placeholder="예: 홍길동")
                        e_files = gr.File(label="등록 음성 업로드 (Enrollment Audio, 여러 개 선택 가능)", file_count="multiple",
                                          file_types=[".wav", ".mp3", ".m4a", ".mp4", ".mov", ".flac"], type="filepath",
                                          height=150)
                        e_replace = gr.Checkbox(label="재등록 - 기존 프로필 덮어쓰기 (Re-enroll)", value=False)
                        e_keep = gr.Checkbox(label="원본 등록 음성 보관 (모델 변경 시 재계산용)",
                                             value=cfg.speaker_id.enrollment.keep_raw_audio)
                    e_btn = gr.Button("화자 등록", variant="primary", size="lg", elem_classes=["mt-run"])
                with gr.Column():
                    gr.HTML('<div class="mt-section-title">등록 화자 관리</div>')
                    with gr.Group():
                        sel = gr.Dropdown(choices=profile_choices(), label="화자 선택", value=None)
                        with gr.Row():
                            re_btn = gr.Button("저장 음성으로 재계산", variant="secondary")
                            raw_btn = gr.Button("원본 음성만 삭제", variant="secondary")
                            del_btn = gr.Button("프로필 삭제 (Delete)", variant="stop")
                    e_status = gr.Textbox(label="처리 결과", lines=6)
            gr.HTML('<div class="mt-section-title">등록 화자 목록 (Registered Speaker List)</div>')
            with gr.Row():
                refresh = gr.Button("목록 새로고침", variant="secondary", size="sm", scale=0, min_width=140)
            profiles = gr.Dataframe(value=profile_rows(), headers=PROFILE_HEADERS, show_label=False, interactive=False)
            e_btn.click(enroll, [e_name, e_files, e_replace, e_keep], [e_status, profiles, sel])
            del_btn.click(delete_profile, sel, [e_status, profiles, sel])
            raw_btn.click(delete_raw, sel, [e_status, profiles])
            re_btn.click(reenroll_stored, sel, [e_status, profiles])
            refresh.click(lambda: (profile_rows(), gr.update(choices=profile_choices())), None, [profiles, sel])

        with gr.Tab("시스템 진단 (Diagnostics)") as diag_tab:
            gr.HTML('<div class="mt-note">GPU/CUDA 인식 여부, 음성인식 엔진(Azure MAI 엔드포인트), 라이브러리·모델 버전, '
                    '최근 실행의 화자 수·처리 시간·RTF(처리시간 ÷ 오디오 길이)를 확인합니다. '
                    'HF 토큰·Azure 키는 설정 여부만 표시합니다.</div>')
            gr.HTML('<div class="mt-section-title">Azure MAI 연결 설정</div>')
            with gr.Row():
                az_ep = gr.Textbox(
                    label="엔드포인트 또는 Azure 포털 리소스 URL", value=azure_speech_endpoint(cfg) or "", scale=3,
                    placeholder="https://<리소스명>.cognitiveservices.azure.com 또는 포털 주소창 URL 그대로")
                az_key = gr.Textbox(
                    label="키 (KEY 1)", type="password", scale=2,
                    placeholder="설정됨 — 변경할 때만 입력" if azure_speech_key() else "포털 > 리소스 > 키 및 엔드포인트")
            with gr.Row():
                az_save = gr.Button("저장 후 연결 테스트", variant="primary")
                az_btn = gr.Button("연결 테스트만 (2초 음성 전송)", variant="secondary")
            az_out = gr.Textbox(show_label=False, lines=3,
                                placeholder="키는 이 PC의 .env 에만 저장되고 화면·로그에 표시되지 않습니다.")
            az_save.click(azure_save, [az_ep, az_key], [az_out, az_ep, az_key])
            az_btn.click(azure_check, None, az_out)
            d_btn = gr.Button("새로고침", variant="secondary")
            d_json = gr.JSON(label="Device / CUDA / Models / 최근 실행")
            d_btn.click(diagnostics, None, d_json)
            # Gradio renders hidden tabs lazily: refresh when the tab is opened, not on page load
            diag_tab.select(diagnostics, None, d_json)
        gr.HTML(footer_html(cfg))
    return demo


def launch_kwargs(cfg: AppConfig) -> dict:
    """Look & feel passed to ``Blocks.launch`` (Gradio 6 takes theme/css at launch time)."""
    from .branding import favicon_path, make_css, make_theme

    return {
        "theme": make_theme(cfg),
        "css": make_css(cfg),
        "footer_links": [],  # hide the Gradio/API footer links; our own footer is rendered in the page
        "favicon_path": favicon_path(cfg),
    }


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
        **launch_kwargs(cfg),
    )
