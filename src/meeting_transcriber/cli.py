"""Command line interface.

  meeting-transcriber transcribe meeting.m4a --mode FIXED --num-speakers 5
  meeting-transcriber enroll --name 이용욱 s1.wav s2.wav s3.wav
  meeting-transcriber enroll-dir data/enroll_raw          (sub-folder name = speaker name)
  meeting-transcriber speakers | delete-speaker 이용욱 | delete-raw-audio 이용욱 | reenroll 이용욱
  meeting-transcriber diagnostics
  meeting-transcriber azure-check                         (Azure MAI-Transcribe connectivity test)
  meeting-transcriber ui
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .errors import UserFacingError
from .logging_utils import setup_logging

AUDIO_GLOBS = ("*.wav", "*.mp3", "*.m4a", "*.mp4", "*.mov", "*.flac")


def _resolve_speaker(store, key: str):
    p = store.find_by_name(key)
    if p is None:
        try:
            p = store.get(key)
        except UserFacingError:
            p = None
    if p is None:
        raise UserFacingError(f"등록된 화자를 찾을 수 없습니다: {key}")
    return p


def _embedder(cfg):
    from .device import resolve_device, torch_device_string
    from .speaker_id import WeSpeakerEmbedder

    device = resolve_device(cfg.runtime.device)
    return WeSpeakerEmbedder(cfg.speaker_id.wespeaker_model, torch_device_string(device, cfg.runtime.device_index),
                             cfg.model_cache_dir)


def cmd_transcribe(cfg, args) -> int:
    from .asr.azure_mai import parse_phrases_text
    from .pipeline import MeetingTranscriber, RunRequest

    def progress(i, label, frac):
        print(f"\r{label}: {frac * 100:5.1f}%", end="" if frac < 1 else "\n", file=sys.stderr, flush=True)

    req = RunRequest(
        input_path=args.input,
        language=args.language,
        speaker_mode=args.mode,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        whisper_model=args.model,
        asr_backend=args.backend,
        phrases=parse_phrases_text(args.phrases),
        identify=False if args.no_identify else None,
        matching_mode=args.matching_mode,
        match_threshold=args.threshold,
    )
    result = MeetingTranscriber(cfg).run(req, progress)
    print(f"\n출력 폴더: {result.job_dir}")
    for fmt, p in result.files.items():
        print(f"  {fmt.upper():4s} {p}")
    doc = result.document
    print(f"화자 {doc['speaker_count']}명 | 길이 {doc['duration']:.1f}s | 처리 {doc['processing']['total_sec']}s | "
          f"RTF {doc['processing']['real_time_factor']}")
    for s in doc["speakers"]:
        sim = "" if s["similarity"] is None else f" (cos={s['similarity']:.3f})"
        print(f"  {s['cluster']} → {s['label']}{sim}  [{s['status']}] {s['reason']}")
    for w in result.warnings:
        print(f"  ! {w}")
    return 0


def cmd_enroll(cfg, args) -> int:
    from .speaker_id import SpeakerProfileStore

    store = SpeakerProfileStore(cfg.enrollment_dir)
    rep = store.enroll(args.name, args.files, _embedder(cfg), cfg.speaker_id.enrollment, replace=args.replace,
                       keep_raw_audio=False if args.no_keep_audio else None)
    print(rep.message())
    return 0


def cmd_enroll_dir(cfg, args) -> int:
    from .speaker_id import SpeakerProfileStore

    root = Path(args.directory)
    if not root.is_dir():
        raise UserFacingError(f"디렉터리가 없습니다: {root}")
    store = SpeakerProfileStore(cfg.enrollment_dir)
    emb = _embedder(cfg)
    n = 0
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for g in AUDIO_GLOBS for f in d.glob(g))
        if not files:
            continue
        try:
            print(store.enroll(d.name, files, emb, cfg.speaker_id.enrollment, replace=args.replace).message())
            n += 1
        except UserFacingError as exc:
            print(f"[{d.name}] 실패: {exc}")
    print(f"{n}명 등록/갱신")
    return 0


def cmd_speakers(cfg, args) -> int:
    from .speaker_id import SpeakerProfileStore

    store = SpeakerProfileStore(cfg.enrollment_dir)
    profiles = store.list_profiles()
    if not profiles:
        print("등록된 화자가 없습니다.")
    for p in profiles:
        print(f"{p.name}\t{p.speaker_id}\tsamples={p.num_samples}\tchunks_used={p.num_chunks_used}\t"
              f"speech={p.total_speech_sec:.0f}s\traw_audio={'yes' if p.raw_audio_kept else 'no'}\tmodel={p.model_id}")
    return 0


def cmd_delete(cfg, args) -> int:
    from .speaker_id import SpeakerProfileStore

    store = SpeakerProfileStore(cfg.enrollment_dir)
    p = _resolve_speaker(store, args.speaker)
    store.delete(p.speaker_id)
    print(f"삭제 완료: {p.name} (음성 샘플 + 임베딩)")
    return 0


def cmd_delete_raw(cfg, args) -> int:
    from .speaker_id import SpeakerProfileStore

    store = SpeakerProfileStore(cfg.enrollment_dir)
    p = _resolve_speaker(store, args.speaker)
    n = store.delete_raw_audio(p.speaker_id)
    print(f"{p.name}: 원본 등록 음성 {n}개 삭제 (임베딩 프로필은 유지)")
    return 0


def cmd_reenroll(cfg, args) -> int:
    from .speaker_id import SpeakerProfileStore

    store = SpeakerProfileStore(cfg.enrollment_dir)
    p = _resolve_speaker(store, args.speaker)
    print(store.reenroll_from_stored(p.speaker_id, _embedder(cfg), cfg.speaker_id.enrollment).message())
    return 0


def cmd_azure_check(cfg, args) -> int:
    import tempfile

    from .asr import AzureMaiTranscriber

    with tempfile.TemporaryDirectory() as tmp:
        res = AzureMaiTranscriber(cfg).check(Path(tmp))
    print(f"Azure MAI 연결 성공: {res['endpoint_host']} / {res['model']} / 응답 {res['latency_sec']}초")
    for n in res["notes"]:
        print(f"  ℹ {n}")
    return 0


def cmd_diagnostics(cfg, args) -> int:
    from .device import collect_diagnostics

    print(json.dumps(collect_diagnostics(cfg), ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_ui(cfg, args) -> int:
    from .ui import launch

    launch(cfg, server_name=args.host, server_port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="meeting-transcriber", description="로컬 한국어 다화자 회의 전사")
    ap.add_argument("--config", help="추가 YAML 설정 파일")
    ap.add_argument("--log-level", default="INFO")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="회의 파일 전사")
    t.add_argument("input")
    t.add_argument("--language", help="ko | auto | en ...")
    t.add_argument("--mode", choices=["AUTO", "RANGE", "FIXED"], type=str.upper)
    t.add_argument("--num-speakers", type=int)
    t.add_argument("--min-speakers", type=int)
    t.add_argument("--max-speakers", type=int)
    t.add_argument("--backend", choices=["azure_mai", "whisperx"], help="음성인식 엔진 (기본: 설정 asr.backend)")
    t.add_argument("--phrases", help="Azure MAI 전문용어 우선 인식 목록 (쉼표 구분)")
    t.add_argument("--model", help="WhisperX 사용 시 Whisper 모델 (large-v3, large-v3-turbo, ...)")
    t.add_argument("--no-identify", action="store_true", help="등록 화자 식별 끄기")
    t.add_argument("--matching-mode", choices=["flexible", "strict_one_to_one", "argmax"])
    t.add_argument("--threshold", type=float, help="speaker_match_threshold (raw cosine)")
    t.set_defaults(func=cmd_transcribe)

    e = sub.add_parser("enroll", help="화자 등록 (같은 이름이면 샘플 추가)")
    e.add_argument("--name", required=True)
    e.add_argument("files", nargs="+")
    e.add_argument("--replace", action="store_true", help="기존 프로필 덮어쓰기(재등록)")
    e.add_argument("--no-keep-audio", action="store_true", help="임베딩 추출 후 원본 음성 저장 안 함")
    e.set_defaults(func=cmd_enroll)

    ed = sub.add_parser("enroll-dir", help="<dir>/<이름>/*.wav 일괄 등록")
    ed.add_argument("directory")
    ed.add_argument("--replace", action="store_true")
    ed.set_defaults(func=cmd_enroll_dir)

    sub.add_parser("speakers", help="등록 화자 목록").set_defaults(func=cmd_speakers)
    d = sub.add_parser("delete-speaker", help="화자 프로필 삭제(음성+임베딩)")
    d.add_argument("speaker")
    d.set_defaults(func=cmd_delete)
    dr = sub.add_parser("delete-raw-audio", help="원본 등록 음성만 삭제")
    dr.add_argument("speaker")
    dr.set_defaults(func=cmd_delete_raw)
    r = sub.add_parser("reenroll", help="저장된 등록 음성으로 프로필 재계산(모델 변경 시)")
    r.add_argument("speaker")
    r.set_defaults(func=cmd_reenroll)
    sub.add_parser("diagnostics", help="CUDA/GPU/버전 진단").set_defaults(func=cmd_diagnostics)
    sub.add_parser("azure-check", help="Azure MAI-Transcribe 연결 테스트 (2초 음성 전송)").set_defaults(
        func=cmd_azure_check)
    u = sub.add_parser("ui", help="Gradio UI 실행")
    u.add_argument("--host")
    u.add_argument("--port", type=int)
    u.set_defaults(func=cmd_ui)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        cfg = load_config(args.config)
        return args.func(cfg, args)
    except UserFacingError as exc:
        print(f"\n오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
