"""Layer 7: voice-profile store (local disk only).

Layout (git-ignored, see .gitignore):
  <enrollment_dir>/<speaker_id>/
      profile.json      name, model id, per-sample stats, which chunk embeddings are used
      embeddings.npy    (n_chunks, D) L2-normalized WeSpeaker embeddings
      centroid.npy      (D,) robust profile vector used for matching
      samples/          16 kHz mono WAV copies of the enrollment audio (optional, deletable)

Voice embeddings are treated as biometric-like personal data: they never leave this directory,
are never logged, and are never written into transcripts/exports. Deleting a speaker removes the
whole directory.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..audio.io import check_extension, convert_to_pcm_wav, read_wav
from ..audio.preprocess import peak_normalize
from ..config import EnrollmentConfig
from ..errors import EnrollmentError, UserFacingError
from ..logging_utils import get_logger
from .scoring import robust_centroid

logger = get_logger(__name__)

PROFILE_VERSION = 1
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


@dataclass
class SampleRecord:
    sample_index: int
    original_name: str
    stored_file: Optional[str]      # relative to the speaker dir, None once raw audio is deleted
    duration_sec: float
    speech_sec: float
    chunk_indices: list             # rows in embeddings.npy produced by this sample
    accepted: bool = True
    note: str = ""


@dataclass
class SpeakerProfile:
    speaker_id: str
    name: str
    model_id: str
    embedding_dim: int
    created_at: str
    updated_at: str
    samples: list = field(default_factory=list)       # list[SampleRecord as dict]
    chunk_keep: list = field(default_factory=list)    # bool per embeddings.npy row
    chunk_loo_similarity: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    version: int = PROFILE_VERSION

    @property
    def num_samples(self) -> int:
        return sum(1 for s in self.samples if s.get("accepted", True))

    @property
    def num_chunks_used(self) -> int:
        return int(sum(bool(k) for k in self.chunk_keep))

    @property
    def total_speech_sec(self) -> float:
        return float(sum(s.get("speech_sec", 0.0) for s in self.samples if s.get("accepted", True)))

    @property
    def raw_audio_kept(self) -> bool:
        return any(s.get("stored_file") for s in self.samples)


@dataclass
class EnrollmentReport:
    speaker_id: str
    name: str
    accepted_samples: int
    rejected_samples: list            # [(file name, reason)]
    chunks_total: int
    chunks_used: int
    total_speech_sec: float
    warnings: list

    def message(self) -> str:
        lines = [
            f"'{self.name}' 등록 완료: 샘플 {self.accepted_samples}개, 임베딩 구간 {self.chunks_used}/{self.chunks_total}개 사용, "
            f"유효 음성 {self.total_speech_sec:.1f}초"
        ]
        for fname, reason in self.rejected_samples:
            lines.append(f"  - 제외: {fname} ({reason})")
        for w in self.warnings:
            lines.append(f"  - 주의: {w}")
        return "\n".join(lines)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _chunks(audio: np.ndarray, sr: int, chunk_sec: float) -> list[np.ndarray]:
    n = int(chunk_sec * sr)
    if len(audio) <= int(1.5 * n):
        return [audio]
    parts = [audio[i:i + n] for i in range(0, len(audio), n)]
    if len(parts) > 1 and len(parts[-1]) < n // 2:  # fold a short tail into the previous chunk
        parts[-2] = np.concatenate([parts[-2], parts[-1]])
        parts.pop()
    return parts


class SpeakerProfileStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ queries
    def _dir(self, speaker_id: str) -> Path:
        if not _ID_RE.match(speaker_id):
            raise EnrollmentError(f"잘못된 speaker_id: {speaker_id!r}")
        return self.root / speaker_id

    def list_profiles(self) -> list[SpeakerProfile]:
        out = []
        for d in sorted(self.root.iterdir()) if self.root.exists() else []:
            pj = d / "profile.json"
            if d.is_dir() and pj.exists():
                try:
                    out.append(self._read_profile(pj))
                except Exception as exc:  # noqa: BLE001 - one bad profile must not break the app
                    logger.warning("Skipping unreadable profile %s: %s", d.name, type(exc).__name__)
        return sorted(out, key=lambda p: p.name)

    def get(self, speaker_id: str) -> Optional[SpeakerProfile]:
        pj = self._dir(speaker_id) / "profile.json"
        return self._read_profile(pj) if pj.exists() else None

    def find_by_name(self, name: str) -> Optional[SpeakerProfile]:
        name = name.strip()
        return next((p for p in self.list_profiles() if p.name == name), None)

    def centroid(self, speaker_id: str) -> np.ndarray:
        return np.load(self._dir(speaker_id) / "centroid.npy")

    def matching_set(self, model_id: str) -> tuple[list[SpeakerProfile], np.ndarray, list[SpeakerProfile]]:
        """Profiles usable with ``model_id`` + their centroids, and profiles skipped as incompatible."""
        usable, skipped, cents = [], [], []
        for p in self.list_profiles():
            if p.model_id != model_id:
                skipped.append(p)
                continue
            try:
                cents.append(self.centroid(p.speaker_id))
                usable.append(p)
            except FileNotFoundError:
                skipped.append(p)
        mat = np.stack(cents).astype(np.float32) if cents else np.zeros((0, 0), dtype=np.float32)
        return usable, mat, skipped

    # ------------------------------------------------------------------ mutations
    def enroll(
        self,
        name: str,
        audio_paths: Iterable[str | Path],
        embedder,
        cfg: EnrollmentConfig,
        replace: bool = False,
        keep_raw_audio: Optional[bool] = None,
    ) -> EnrollmentReport:
        """Create a profile, add samples to an existing one (same name), or re-enroll (replace=True)."""
        name = (name or "").strip()
        if not name:
            raise EnrollmentError("화자 이름을 입력하세요.")
        if len(name) > 50 or any(c in name for c in "\n\r\t"):
            raise EnrollmentError("화자 이름이 올바르지 않습니다 (50자 이하, 줄바꿈 불가).")
        paths = [Path(p) for p in audio_paths if p]
        if not paths:
            raise EnrollmentError("등록할 음성 파일을 1개 이상 업로드하세요.")
        keep_raw = cfg.keep_raw_audio if keep_raw_audio is None else keep_raw_audio

        embedder.load()
        model_id = embedder.model_id
        existing = self.find_by_name(name)
        if existing and existing.model_id != model_id and not replace:
            raise EnrollmentError(
                f"'{name}' 프로필은 다른 임베딩 모델({existing.model_id})로 생성되었습니다.",
                hint="'재등록(덮어쓰기)'으로 현재 모델 기준 프로필을 다시 만드세요.",
            )
        speaker_id = existing.speaker_id if existing else f"spk_{uuid.uuid4().hex[:10]}"
        sdir = self._dir(speaker_id)

        prev_samples: list = []
        prev_embs = None
        if existing and not replace:
            prev_samples = existing.samples
            prev_embs = np.load(sdir / "embeddings.npy")

        staging = Path(tempfile.mkdtemp(prefix="enroll_", dir=self.root))
        try:
            new_records, new_embs, rejected = [], [], []
            base_index = max((s["sample_index"] for s in prev_samples), default=0)
            row = 0 if prev_embs is None else prev_embs.shape[0]
            for k, src in enumerate(paths, start=1):
                idx = base_index + k
                try:
                    check_extension(src)
                    wav = convert_to_pcm_wav(src, staging / f"sample_{idx:02d}.wav", 16000, "downmix")
                except UserFacingError as exc:
                    rejected.append((src.name, exc.message))
                    continue
                audio, sr = read_wav(wav)
                audio, _ = peak_normalize(audio, -1.0)
                if cfg.vad:
                    from .embedder import speech_only

                    speech = speech_only(audio, sr)
                else:
                    speech = audio
                speech_sec = len(speech) / sr
                if speech_sec < cfg.min_speech_sec:
                    rejected.append((src.name, f"유효 음성 {speech_sec:.1f}s < 최소 {cfg.min_speech_sec:.0f}s"))
                    continue
                chunk_rows = []
                for chunk in _chunks(speech, sr, cfg.chunk_sec):
                    vec = embedder.embed(chunk, sr)
                    if vec is not None:
                        new_embs.append(vec)
                        chunk_rows.append(row)
                        row += 1
                if not chunk_rows:
                    rejected.append((src.name, "임베딩 추출 실패"))
                    continue
                new_records.append(asdict(SampleRecord(
                    sample_index=idx,
                    original_name=src.name,
                    stored_file=f"samples/{wav.name}" if keep_raw else None,
                    duration_sec=round(len(audio) / sr, 2),
                    speech_sec=round(speech_sec, 2),
                    chunk_indices=chunk_rows,
                )))
            if not new_records:
                detail = "; ".join(f"{n}: {r}" for n, r in rejected)
                raise EnrollmentError(f"유효한 등록 샘플이 없습니다. {detail}",
                                      hint=f"잡음이 적은 환경에서 {cfg.min_speech_sec:.0f}초 이상 말한 음성을 사용하세요.")

            all_embs = np.stack(new_embs) if prev_embs is None else np.concatenate([prev_embs, np.stack(new_embs)])
            rc = robust_centroid(all_embs, cfg.outlier_mad_k, cfg.outlier_min_similarity)
            samples = prev_samples + new_records
            total_speech = sum(s["speech_sec"] for s in samples)
            warnings = list(rc.warnings)
            if total_speech < cfg.recommended_total_speech_sec:
                warnings.append(f"총 유효 음성 {total_speech:.0f}s - 권장 {cfg.recommended_total_speech_sec:.0f}s 이상")

            now = _now()
            profile = SpeakerProfile(
                speaker_id=speaker_id,
                name=name,
                model_id=model_id,
                embedding_dim=int(all_embs.shape[1]),
                created_at=existing.created_at if (existing and not replace) else now,
                updated_at=now,
                samples=samples,
                chunk_keep=[bool(x) for x in rc.keep],
                chunk_loo_similarity=[round(float(x), 4) for x in rc.loo_similarity],
                warnings=warnings,
            )

            # commit: write into the speaker dir only after everything succeeded
            if replace and sdir.exists():
                shutil.rmtree(sdir)
            sdir.mkdir(parents=True, exist_ok=True)
            if keep_raw:
                (sdir / "samples").mkdir(exist_ok=True)
                for rec in new_records:
                    shutil.move(str(staging / Path(rec["stored_file"]).name), str(sdir / rec["stored_file"]))
            np.save(sdir / "embeddings.npy", all_embs.astype(np.float32))
            np.save(sdir / "centroid.npy", rc.centroid.astype(np.float32))
            self._write_profile(sdir / "profile.json", profile)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        return EnrollmentReport(
            speaker_id=speaker_id,
            name=name,
            accepted_samples=len(new_records),
            rejected_samples=rejected,
            chunks_total=int(len(rc.keep)),
            chunks_used=int(rc.keep.sum()),
            total_speech_sec=float(total_speech),
            warnings=warnings,
        )

    def delete(self, speaker_id: str) -> bool:
        d = self._dir(speaker_id)
        if not d.exists():
            return False
        shutil.rmtree(d)
        return True

    def delete_raw_audio(self, speaker_id: str) -> int:
        """Remove stored enrollment WAVs but keep the embeddings (profile stays usable).
        Re-enrolling with a different embedding model will then require new recordings."""
        d = self._dir(speaker_id)
        profile = self.get(speaker_id)
        if profile is None:
            return 0
        removed = 0
        for s in profile.samples:
            if s.get("stored_file"):
                f = d / s["stored_file"]
                if f.exists():
                    f.unlink()
                    removed += 1
                s["stored_file"] = None
        shutil.rmtree(d / "samples", ignore_errors=True)
        profile.updated_at = _now()
        self._write_profile(d / "profile.json", profile)
        return removed

    def reenroll_from_stored(self, speaker_id: str, embedder, cfg: EnrollmentConfig) -> EnrollmentReport:
        """Rebuild a profile from its stored WAVs (e.g. after switching the WeSpeaker model)."""
        profile = self.get(speaker_id)
        if profile is None:
            raise EnrollmentError(f"등록된 화자를 찾을 수 없습니다: {speaker_id}")
        paths = [p for p in self.stored_sample_paths(speaker_id) if p.exists()]
        if not paths:
            raise EnrollmentError(
                f"'{profile.name}' 의 원본 등록 음성이 삭제되어 재계산할 수 없습니다.",
                hint="새 음성 파일을 업로드하여 '재등록(덮어쓰기)' 하세요.",
            )
        return self.enroll(profile.name, paths, embedder, cfg, replace=True, keep_raw_audio=True)

    def stored_sample_paths(self, speaker_id: str) -> list[Path]:
        profile = self.get(speaker_id)
        d = self._dir(speaker_id)
        return [d / s["stored_file"] for s in (profile.samples if profile else []) if s.get("stored_file")]

    # ------------------------------------------------------------------ io
    @staticmethod
    def _read_profile(path: Path) -> SpeakerProfile:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in SpeakerProfile.__dataclass_fields__}
        return SpeakerProfile(**{k: v for k, v in data.items() if k in known})

    @staticmethod
    def _write_profile(path: Path, profile: SpeakerProfile) -> None:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(profile), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
