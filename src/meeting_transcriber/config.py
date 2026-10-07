"""Typed configuration.

Precedence (low -> high):
  dataclass defaults  <  config/default.yaml  <  YAML in $MT_CONFIG  <  env MT_<SECTION>__<KEY>
  (nested: MT_SPEAKER_ID__CLUSTER_EMBEDDING__MIN_SEGMENT_SEC=2.0)

Secrets are never part of this object: the HF token is read from the environment only when a
gated model is loaded (see ``hf_token()``), and is never stored, printed or exported.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, get_args, get_origin, get_type_hints

import yaml

from .errors import ConfigError

ENV_PREFIX = "MT_"


@dataclass
class RuntimeConfig:
    device: str = "auto"
    device_index: int = 0
    cpu_threads: int = 0   # 0 = all logical CPUs
    release_models_between_stages: bool = True


@dataclass
class PathsConfig:
    enrollment_dir: str = "data/speakers"
    output_dir: str = "outputs"
    model_cache_dir: Optional[str] = None


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channel_strategy: str = "downmix"
    normalize: bool = True
    target_peak_dbfs: float = -1.0
    noise_reduction: bool = False
    max_duration_hours: float = 4.0
    keep_intermediate_wav: bool = False


@dataclass
class AzureMaiConfig:
    """Azure Speech (Microsoft Foundry) MAI-Transcribe via the LLM Speech / fast transcription REST API.

    Endpoint and key are read from the environment (AZURE_SPEECH_ENDPOINT / AZURE_SPEECH_KEY);
    ``endpoint`` here is only a non-secret fallback. The key is never stored in config."""

    endpoint: Optional[str] = None
    model: str = "MAI-Transcribe-2"
    api_version: str = "2025-10-15"
    auth: str = "key"                 # key (Ocp-Apim-Subscription-Key) | entra (Microsoft Entra ID token)
    force_locale: bool = False        # MS guidance: force a locale only if auto-detection fails
    transcribe_style: str = "verbatim"  # verbatim | clean
    phrases: list = field(default_factory=list)  # keyword biasing (phraseList.phrases)
    upload_format: str = "flac"       # flac (smaller upload) | wav
    max_chunk_minutes: float = 120.0  # split longer recordings (API limit: < 5 h and < 500 MB per file)
    timeout_sec: float = 1800.0
    max_retries: int = 3


@dataclass
class AsrConfig:
    backend: str = "azure_mai"        # azure_mai | whisperx
    model: str = "large-v3"
    compute_type: str = "auto"
    batch_size: int = 16
    language: str = "ko"
    vad_method: str = "pyannote"
    chunk_size: int = 30
    beam_size: int = 5
    initial_prompt: Optional[str] = None
    azure: AzureMaiConfig = field(default_factory=AzureMaiConfig)


@dataclass
class AlignmentConfig:
    enabled: bool = True
    model_name: Optional[str] = None
    interpolate_method: str = "nearest"


@dataclass
class DiarizationConfig:
    model: str = "pyannote/speaker-diarization-community-1"
    mode: str = "AUTO"
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = 2
    max_speakers: Optional[int] = 10
    speaker_limit: int = 10
    embedding_batch_size: Optional[int] = None
    segmentation_batch_size: Optional[int] = None


@dataclass
class ClusterEmbeddingConfig:
    min_segment_sec: float = 1.5
    fallback_min_segment_sec: float = 0.5
    max_segment_sec: float = 10.0
    max_segments: int = 20
    max_total_sec: float = 120.0
    min_total_sec: float = 1.0
    min_rms_dbfs: float = -45.0


@dataclass
class EnrollmentConfig:
    vad: bool = True
    min_speech_sec: float = 3.0
    recommended_total_speech_sec: float = 30.0
    chunk_sec: float = 10.0
    outlier_mad_k: float = 3.0
    outlier_min_similarity: float = 0.2
    keep_raw_audio: bool = True


@dataclass
class SpeakerIdConfig:
    enabled: bool = True
    wespeaker_model: str = "english"
    match_threshold: float = 0.5
    match_margin: float = 0.05
    matching_mode: str = "flexible"
    fragment_extra_threshold: float = 0.05
    fragment_max_cooccurrence_sec: float = 2.0
    cohort_mean_path: Optional[str] = None
    cluster_embedding: ClusterEmbeddingConfig = field(default_factory=ClusterEmbeddingConfig)
    enrollment: EnrollmentConfig = field(default_factory=EnrollmentConfig)


@dataclass
class AssignmentConfig:
    overlap_min_ratio: float = 0.3
    nearest_max_gap_sec: float = 1.0


@dataclass
class ExportConfig:
    formats: list = field(default_factory=lambda: ["txt", "json", "csv", "srt"])
    txt_merge_gap_sec: float = 1.0
    include_words_in_json: bool = True


@dataclass
class BrandConfig:
    company: str = "한미약품"
    org_label: str = "AX PoC · 사내 전용"
    app_title: str = "AI 회의록"
    subtitle: str = "한국어 회의 음성 → 화자별 회의록 자동 생성 · 등록 화자 자동 식별"
    # shown when asr.backend=whisperx (everything local)
    security_badge: str = "로컬 처리 · 외부 전송 없음"
    footer: str = "모든 음성·화자 정보는 이 PC 안에서만 처리·저장됩니다. 자동 생성 결과는 검토 후 사용하세요."
    # shown when asr.backend=azure_mai (audio goes to the company Azure Speech resource)
    security_badge_cloud: str = "음성인식: 사내 Azure · 화자 정보는 이 PC에만 저장"
    footer_cloud: str = ("회의 음성은 음성인식을 위해 사내 Azure Speech 리소스로 전송되며, 화자 음성 등록 정보는 이 PC에만 "
                         "저장됩니다. 자동 생성 결과는 검토 후 사용하세요.")
    badges: list = field(default_factory=lambda: ["pyannote 화자 분리", "WeSpeaker 화자 식별"])
    primary_color: str = "#e12319"   # Hanmi CI single red
    accent_color: str = "#333333"    # neutral charcoal for text / secondary accents
    logo_path: Optional[str] = "assets/brand/hanmi_emblem.svg"
    favicon_path: Optional[str] = "assets/brand/hanmi_emblem.svg"


@dataclass
class UiConfig:
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    max_file_size: str = "4gb"
    brand: BrandConfig = field(default_factory=BrandConfig)


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    speaker_id: SpeakerIdConfig = field(default_factory=SpeakerIdConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    ui: UiConfig = field(default_factory=UiConfig)

    # ----------------------------------------------------------------- helpers
    def copy(self) -> "AppConfig":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def resolve_path(self, value: str | Path) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (project_root() / p)

    @property
    def enrollment_dir(self) -> Path:
        return self.resolve_path(self.paths.enrollment_dir)

    @property
    def output_dir(self) -> Path:
        return self.resolve_path(self.paths.output_dir)

    @property
    def model_cache_dir(self) -> Optional[str]:
        return str(self.resolve_path(self.paths.model_cache_dir)) if self.paths.model_cache_dir else None

    def validate(self) -> None:
        validate_config(self)


# --------------------------------------------------------------------------- loading


def project_root() -> Path:
    env = os.environ.get("MT_HOME")
    if env:
        return Path(env).expanduser().resolve()
    src_root = Path(__file__).resolve().parents[2]
    if (src_root / "config" / "default.yaml").exists():
        return src_root
    return Path.cwd()


def load_dotenv_if_present() -> None:
    """Load .env from the project root (does not override variables already set)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a hard dependency
        return
    env_file = project_root() / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)


def azure_speech_key() -> Optional[str]:
    key = (os.environ.get("AZURE_SPEECH_KEY") or "").strip()
    return key or None


def azure_speech_endpoint(cfg: "AppConfig") -> Optional[str]:
    value = (os.environ.get("AZURE_SPEECH_ENDPOINT") or cfg.asr.azure.endpoint or "").strip()
    return value.rstrip("/") or None


def hf_token() -> Optional[str]:
    """HF access token from the environment (HF_TOKEN; HUGGING_FACE_HUB_TOKEN as fallback)."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token = token.strip() if token else None
    return token or None


def load_config(config_path: str | Path | None = None, env: Optional[dict] = None) -> AppConfig:
    load_dotenv_if_present()
    env = dict(os.environ) if env is None else env
    cfg = AppConfig()

    default_file = project_root() / "config" / "default.yaml"
    if default_file.exists():
        _merge_dict(cfg, _read_yaml(default_file), "config/default.yaml")

    override = config_path or env.get("MT_CONFIG")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = project_root() / path
        if not path.exists():
            raise ConfigError(f"설정 파일을 찾을 수 없습니다: {path}")
        _merge_dict(cfg, _read_yaml(path), str(path))

    _apply_env(cfg, env)
    validate_config(cfg)
    return cfg


def _read_yaml(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 파싱 오류: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"설정 파일 최상위는 mapping 이어야 합니다: {path}")
    return data


def _merge_dict(obj: Any, data: dict, source: str, prefix: str = "") -> None:
    hints = get_type_hints(type(obj))
    for key, value in data.items():
        if key not in hints:
            raise ConfigError(f"알 수 없는 설정 키 '{prefix}{key}' ({source})")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ConfigError(f"'{prefix}{key}' 는 mapping 이어야 합니다 ({source})")
            _merge_dict(current, value, source, prefix=f"{prefix}{key}.")
        else:
            setattr(obj, key, _coerce(value, hints[key], f"{prefix}{key}"))


def _apply_env(cfg: AppConfig, env: dict) -> None:
    for raw_key, raw_value in env.items():
        if not raw_key.upper().startswith(ENV_PREFIX) or "__" not in raw_key:
            continue
        path = [p.lower() for p in raw_key[len(ENV_PREFIX):].split("__")]
        target: Any = cfg
        for part in path[:-1]:
            if not dataclasses.is_dataclass(target) or not hasattr(target, part):
                raise ConfigError(f"알 수 없는 환경변수 설정: {raw_key}")
            target = getattr(target, part)
        leaf = path[-1]
        hints = get_type_hints(type(target))
        if leaf not in hints or dataclasses.is_dataclass(getattr(target, leaf)):
            raise ConfigError(f"알 수 없는 환경변수 설정: {raw_key}")
        setattr(target, leaf, _coerce_str(raw_value, hints[leaf], raw_key))


def _coerce(value: Any, hint: Any, name: str) -> Any:
    if isinstance(value, str):
        return _coerce_str(value, hint, name)
    base, optional = _unwrap_optional(hint)
    if value is None:
        if optional:
            return None
        raise ConfigError(f"'{name}' 에 null 을 사용할 수 없습니다")
    try:
        if base is bool:
            if not isinstance(value, bool):
                raise ValueError
            return value
        if base is int:
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise ValueError
            return int(value)
        if base is float:
            return float(value)
        if base is list:
            if not isinstance(value, list):
                raise ValueError
            return list(value)
    except (TypeError, ValueError):
        raise ConfigError(f"'{name}' 값 형식 오류: {value!r}") from None
    return value


def _coerce_str(value: str, hint: Any, name: str) -> Any:
    base, optional = _unwrap_optional(hint)
    v = value.strip()
    if optional and v.lower() in ("", "null", "none"):
        return None
    try:
        if base is bool:
            if v.lower() in ("1", "true", "yes", "on"):
                return True
            if v.lower() in ("0", "false", "no", "off"):
                return False
            raise ValueError
        if base is int:
            return int(v)
        if base is float:
            return float(v)
        if base is list:
            return [item.strip() for item in v.split(",") if item.strip()]
    except ValueError:
        raise ConfigError(f"'{name}' 값 형식 오류: {value!r}") from None
    return v


def _unwrap_optional(hint: Any) -> tuple[Any, bool]:
    origin = get_origin(hint)
    if origin is list:
        return list, False
    args = get_args(hint)
    if args and type(None) in args:
        rest = [a for a in args if a is not type(None)]
        base = rest[0] if len(rest) == 1 else Any
        return (list if get_origin(base) is list else base), True
    return hint, False


# --------------------------------------------------------------------------- validation

VALID_DEVICES = {"auto", "cuda", "cpu"}
VALID_ASR_BACKENDS = {"azure_mai", "whisperx"}
VALID_MODES = {"AUTO", "RANGE", "FIXED"}
VALID_MATCHING = {"flexible", "strict_one_to_one", "argmax"}  # argmax = comparison baseline
VALID_COMPUTE = {"auto", "default", "float16", "int8_float16", "int8", "float32", "bfloat16", "int8_bfloat16", "int8_float32"}
VALID_FORMATS = {"txt", "json", "csv", "srt"}
_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def validate_config(cfg: AppConfig) -> None:
    if cfg.runtime.device not in VALID_DEVICES:
        raise ConfigError(f"runtime.device 는 {sorted(VALID_DEVICES)} 중 하나여야 합니다: {cfg.runtime.device}")
    if cfg.asr.backend not in VALID_ASR_BACKENDS:
        raise ConfigError(f"asr.backend 는 {sorted(VALID_ASR_BACKENDS)} 중 하나여야 합니다: {cfg.asr.backend}")
    if cfg.asr.azure.auth not in ("key", "entra"):
        raise ConfigError("asr.azure.auth 는 key | entra")
    if cfg.asr.azure.transcribe_style not in ("verbatim", "clean"):
        raise ConfigError("asr.azure.transcribe_style 는 verbatim | clean")
    if cfg.asr.azure.upload_format not in ("flac", "wav"):
        raise ConfigError("asr.azure.upload_format 는 flac | wav")
    if not 1 <= cfg.asr.azure.max_chunk_minutes <= 290:
        raise ConfigError("asr.azure.max_chunk_minutes 는 1~290 (API 한도: 파일당 5시간 미만)")
    if cfg.asr.compute_type not in VALID_COMPUTE:
        raise ConfigError(f"asr.compute_type 지원값 아님: {cfg.asr.compute_type}")
    if cfg.asr.batch_size < 1:
        raise ConfigError("asr.batch_size 는 1 이상이어야 합니다")
    if cfg.audio.sample_rate != 16000:
        raise ConfigError("audio.sample_rate 는 16000 이어야 합니다 (WhisperX/pyannote/WeSpeaker 입력 규격)")
    if cfg.audio.channel_strategy not in ("downmix", "first"):
        raise ConfigError("audio.channel_strategy 는 downmix | first")
    if cfg.speaker_id.matching_mode not in VALID_MATCHING:
        raise ConfigError(f"speaker_id.matching_mode 는 {sorted(VALID_MATCHING)} 중 하나여야 합니다")
    if not -1.0 <= cfg.speaker_id.match_threshold <= 1.0:
        raise ConfigError("speaker_id.match_threshold 는 cosine 범위 [-1, 1] 이어야 합니다")
    if cfg.speaker_id.match_margin < 0:
        raise ConfigError("speaker_id.match_margin 은 0 이상이어야 합니다")
    for key in ("primary_color", "accent_color"):
        value = getattr(cfg.ui.brand, key)
        if not _HEX_RE.match(value or ""):
            raise ConfigError(f"ui.brand.{key} 는 #RRGGBB 형식이어야 합니다: {value!r}")
    bad = set(cfg.export.formats) - VALID_FORMATS
    if bad:
        raise ConfigError(f"export.formats 지원하지 않는 형식: {sorted(bad)}")
    validate_speaker_count(
        cfg.diarization.mode,
        cfg.diarization.num_speakers,
        cfg.diarization.min_speakers,
        cfg.diarization.max_speakers,
        cfg.diarization.speaker_limit,
    )


def validate_speaker_count(
    mode: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    speaker_limit: int,
) -> None:
    """Validates config-level values. FIXED without num_speakers is allowed here (UI supplies it)."""
    mode_u = (mode or "").upper()
    if mode_u not in VALID_MODES:
        raise ConfigError(f"diarization.mode 는 AUTO | RANGE | FIXED 중 하나여야 합니다: {mode}")
    if speaker_limit < 1:
        raise ConfigError("diarization.speaker_limit 는 1 이상이어야 합니다")
    for label, v in (("num_speakers", num_speakers), ("min_speakers", min_speakers), ("max_speakers", max_speakers)):
        if v is not None and not (1 <= int(v) <= speaker_limit):
            raise ConfigError(f"{label}={v} 는 1 ~ speaker_limit({speaker_limit}) 범위여야 합니다")
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise ConfigError(f"min_speakers({min_speakers}) 가 max_speakers({max_speakers}) 보다 큽니다")


def speaker_count_kwargs(
    mode: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    speaker_limit: int,
) -> dict:
    """Translate a speaker-count mode into pyannote ``num_speakers/min_speakers/max_speakers`` kwargs."""
    validate_speaker_count(mode, num_speakers, min_speakers, max_speakers, speaker_limit)
    mode_u = mode.upper()
    if mode_u == "FIXED":
        if num_speakers is None:
            raise ConfigError("FIXED 모드에서는 정확한 화자 수(num_speakers)가 필요합니다")
        return {"num_speakers": int(num_speakers)}
    if mode_u == "RANGE":
        if min_speakers is None and max_speakers is None:
            raise ConfigError("RANGE 모드에서는 min_speakers 또는 max_speakers 가 필요합니다")
        kwargs = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            kwargs["max_speakers"] = int(max_speakers)
        return kwargs
    return {}
