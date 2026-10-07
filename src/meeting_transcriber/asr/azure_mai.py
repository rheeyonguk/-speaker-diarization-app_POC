"""Speech-to-text with Microsoft MAI-Transcribe-2 on Azure Speech (Microsoft Foundry).

API (MicrosoftDocs/azure-ai-docs, speech-service/mai-transcribe.md + llm-speech REST quickstart):
  POST {endpoint}/speechtotext/transcriptions:transcribe?api-version=2025-10-15
  multipart/form-data:  audio=<file>   definition=<JSON>
  definition = {"enhancedMode": {"enabled": true, "model": "MAI-Transcribe-2",
                                  "modelOptions": {"timestamps": "word", "transcribeStyle": "verbatim"}},
                "phraseList": {"phrases": [...]}, "locales": ["ko"], "diarization": {"enabled": false}}
  auth: Ocp-Apim-Subscription-Key: <key>   or   Authorization: Bearer <Entra ID token>
  response: {"durationMilliseconds", "combinedPhrases": [{"text"}],
             "phrases": [{"offsetMilliseconds", "durationMilliseconds", "text", "locale", "confidence",
                          "words": [{"text", "offsetMilliseconds", "durationMilliseconds"}]}]}
  limits: < 500 MB and < 5 hours per file; `confidence` is always 0 (not used).

Speaker diarization stays local (pyannote Community-1) because the pipeline needs the overlap-aware
and exclusive diarization views and the per-speaker audio for voice-profile matching; MAI's own
diarization is therefore not requested.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import numpy as np

from ..audio.io import write_wav
from ..config import AppConfig, azure_speech_endpoint, azure_speech_key
from ..errors import AuthError, ConfigError, UserFacingError
from ..logging_utils import get_logger, redact

logger = get_logger(__name__)

Progress = Optional[Callable[[float], None]]

MAI_REGIONS = ("centralindia", "eastus", "northeurope", "southeastasia", "westus", "westus2")
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}
_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


class AzureSpeechError(UserFacingError):
    pass


# --------------------------------------------------------------------------- request building


_PORTAL_ACCOUNT = re.compile(r"/providers/Microsoft\.CognitiveServices/accounts/([A-Za-z0-9][A-Za-z0-9-]{0,62})",
                             re.IGNORECASE)
_RESOURCE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{1,62}$")


def normalize_endpoint(raw: str) -> tuple[str, list[str]]:
    """Accept what people copy from the portal and return the Speech REST base URL (+ notes).

    Besides endpoint URLs this takes the Azure portal URL / resource ID of the account or its bare
    name; those map to ``https://<name>.cognitiveservices.azure.com`` (the custom subdomain the portal
    assigns by default)."""
    notes: list[str] = []
    value = raw.strip()
    account = _PORTAL_ACCOUNT.search(value)
    if account or (_RESOURCE_NAME.match(value) and value.lower() != "localhost"):
        name = (account.group(1) if account else value).lower()
        notes.append(f"리소스 이름 '{name}' 으로 엔드포인트를 구성했습니다. 연결이 안 되면 포털 '키 및 엔드포인트'의 "
                     "엔드포인트 값을 그대로 입력하세요 (사용자 지정 도메인이 다른 경우).")
        return f"https://{name}.cognitiveservices.azure.com", notes
    if not re.match(r"^https?://", value):
        value = "https://" + value
    p = urlparse(value)
    host = p.netloc.lower()
    scheme = "https"
    if p.scheme == "http":
        if host.split(":")[0] in ("127.0.0.1", "localhost"):
            scheme = "http"  # local gateway / test double only
        else:
            notes.append("http 엔드포인트는 https 로 변경했습니다 (음성 데이터 암호화 전송).")
    if not host:
        raise ConfigError(f"Azure Speech 엔드포인트 형식이 올바르지 않습니다: {raw}")
    if host.endswith(".services.ai.azure.com"):
        host = host.replace(".services.ai.azure.com", ".cognitiveservices.azure.com")
        notes.append("Foundry 프로젝트 엔드포인트를 같은 리소스의 Speech 엔드포인트(*.cognitiveservices.azure.com)로 변환했습니다.")
    elif host.endswith(".openai.azure.com"):
        host = host.replace(".openai.azure.com", ".cognitiveservices.azure.com")
        notes.append("OpenAI 엔드포인트를 같은 리소스의 Speech 엔드포인트(*.cognitiveservices.azure.com)로 변환했습니다.")
    if p.path not in ("", "/"):
        notes.append(f"엔드포인트 경로 '{p.path}' 는 무시하고 호스트만 사용합니다.")
    return f"{scheme}://{host}", notes


def transcribe_url(base: str, api_version: str) -> str:
    return f"{base}/speechtotext/transcriptions:transcribe?api-version={api_version}"


def build_definition(cfg: AppConfig, extra_phrases: Optional[list[str]] = None) -> dict:
    az = cfg.asr.azure
    definition: dict = {
        "enhancedMode": {
            "enabled": True,
            "model": az.model,
            "modelOptions": {"timestamps": "word", "transcribeStyle": az.transcribe_style},
        },
        "diarization": {"enabled": False},  # local pyannote does diarization
    }
    phrases = [p.strip() for p in list(az.phrases) + list(extra_phrases or []) if p and p.strip()]
    if phrases:
        definition["phraseList"] = {"phrases": list(dict.fromkeys(phrases))}
    lang = (cfg.asr.language or "").lower()
    if az.force_locale and lang and lang != "auto":
        definition["locales"] = [lang]
    return definition


def parse_phrases_text(text: Optional[str]) -> list[str]:
    """UI/CLI input: comma, semicolon or newline separated terms."""
    if not text:
        return []
    return [t.strip() for t in re.split(r"[,;\n]", text) if t.strip()]


# --------------------------------------------------------------------------- response parsing


def _display_words(phrase_text: str, words: list[dict]) -> list[tuple[str, str]]:
    """Map service word tokens onto the phrase's display text.

    Word tokens can differ from the display text (lower-cased/lexical form, punctuation stripped,
    character-level tokens for CJK). Each word is located in the display text (case-insensitive)
    and returned as (display_text, separator_before) so utterance text keeps the original spacing
    and punctuation for any language. Unmatched words fall back to their own text and a space.
    """
    out: list[tuple[str, str]] = []
    low = phrase_text.lower()
    cursor = 0
    for i, w in enumerate(words):
        token = str(w.get("text", "")).strip()
        if not token:
            out.append(("", ""))
            continue
        pos = low.find(token.lower(), cursor)
        if pos < 0 or (i > 0 and pos - cursor > max(12, 3 * len(token))):
            out.append((token, " " if i else ""))
            continue
        sep = phrase_text[cursor:pos] if i else ""
        out.append((phrase_text[pos:pos + len(token)], sep))
        cursor = pos + len(token)
    if out and cursor < len(phrase_text):
        tail = phrase_text[cursor:].rstrip()
        if tail and not tail.isspace():
            disp, sep = out[-1]
            out[-1] = (disp + tail, sep)
    return out


def _base_lang(locale: Optional[str]) -> Optional[str]:
    return locale.split("-")[0].lower() if locale else None


def parse_transcription(payload: dict, offset_sec: float = 0.0) -> tuple[list[dict], Counter]:
    """Service JSON -> pipeline segments (WhisperX-aligned format) + per-language speech seconds."""
    segments: list[dict] = []
    lang_sec: Counter = Counter()
    for ph in payload.get("phrases") or []:
        text = str(ph.get("text", "")).strip()
        if not text:
            continue
        start = offset_sec + float(ph.get("offsetMilliseconds", 0)) / 1000.0
        end = start + float(ph.get("durationMilliseconds", 0)) / 1000.0
        raw_words = ph.get("words") or []
        words = []
        for w, (disp, sep) in zip(raw_words, _display_words(text, raw_words)):
            if not disp:
                continue
            item = {"word": disp, "sep": sep}
            if w.get("offsetMilliseconds") is not None and w.get("durationMilliseconds") is not None:
                ws = offset_sec + float(w["offsetMilliseconds"]) / 1000.0
                item["start"] = round(ws, 3)
                item["end"] = round(ws + float(w["durationMilliseconds"]) / 1000.0, 3)
            words.append(item)
        lang = _base_lang(ph.get("locale"))
        if lang:
            lang_sec[lang] += max(0.0, end - start)
        segments.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "words": words,
            "locale": ph.get("locale"),
            "speaker_service": ph.get("speaker"),
        })
    return segments, lang_sec


# --------------------------------------------------------------------------- long audio


def plan_chunks(n_samples: int, sr: int, audio: np.ndarray, max_chunk_sec: float,
                search_sec: float = 10.0, win_sec: float = 0.2) -> list[tuple[int, int]]:
    """Split points near every max_chunk_sec boundary, placed at the quietest 200 ms window
    within +/- search_sec so a word is not cut in half."""
    max_len = int(max_chunk_sec * sr)
    if n_samples <= max_len:
        return [(0, n_samples)]
    cuts = [0]
    win = max(1, int(win_sec * sr))
    while n_samples - cuts[-1] > max_len:
        target = cuts[-1] + max_len
        lo = max(cuts[-1] + max_len // 2, target - int(search_sec * sr))
        hi = min(n_samples - win, target + int(search_sec * sr))
        hi = min(hi, cuts[-1] + max_len)  # never exceed the limit
        best = target if target < n_samples else n_samples
        if hi > lo:
            seg = audio[lo:hi + win].astype(np.float64) ** 2
            csum = np.concatenate([[0.0], np.cumsum(seg)])
            energies = csum[win:] - csum[:-win]
            step = max(1, win // 4)
            idx = np.arange(0, len(energies), step)
            k = int(idx[np.argmin(energies[idx])])
            best = lo + k + win // 2
        cuts.append(int(min(best, cuts[-1] + max_len)))
    cuts.append(n_samples)
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def encode_upload(audio: np.ndarray, sr: int, dst: Path, fmt: str) -> Path:
    wav = write_wav(dst.with_suffix(".wav"), audio, sr)
    if fmt == "wav":
        return wav
    out = dst.with_suffix(".flac")
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
                           "-c:a", "flac", str(out)], capture_output=True, text=True)
    wav.unlink(missing_ok=True)
    if proc.returncode != 0 or not out.exists():
        raise AzureSpeechError("업로드용 FLAC 인코딩(ffmpeg) 실패", hint=proc.stderr[-300:] or None)
    return out


# --------------------------------------------------------------------------- HTTP


@dataclass
class _Auth:
    mode: str
    key: Optional[str]
    _credential: object = None
    _token: Optional[str] = None
    _expires: float = 0.0

    def headers(self) -> dict:
        if self.mode == "key":
            return {"Ocp-Apim-Subscription-Key": self.key or ""}
        if self._token is None or time.time() > self._expires - 120:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError as exc:
                raise ConfigError("asr.azure.auth=entra 를 쓰려면 azure-identity 패키지가 필요합니다.",
                                  hint='pip install -e ".[azure]"  (또는 pip install azure-identity)') from exc
            if self._credential is None:
                self._credential = DefaultAzureCredential()
            tok = self._credential.get_token(_TOKEN_SCOPE)
            self._token, self._expires = tok.token, float(tok.expires_on)
        return {"Authorization": f"Bearer {self._token}"}


def _service_message(resp) -> str:
    try:
        data = resp.json()
        err = data.get("error", data) if isinstance(data, dict) else {}
        code = err.get("code") or ""
        msg = err.get("message") or ""
        inner = (err.get("innererror") or {}).get("code") or ""
        text = " / ".join(x for x in (code, inner, msg) if x)
        return redact(text)[:400] or f"HTTP {resp.status_code}"
    except Exception:
        return redact((resp.text or "")[:400]) or f"HTTP {resp.status_code}"


def _raise_for(resp, host: str) -> None:
    msg = _service_message(resp)
    status = resp.status_code
    if status == 401:
        raise AuthError(f"Azure Speech 인증 실패 (401): {msg}",
                        hint=".env 의 AZURE_SPEECH_KEY 가 이 리소스의 키(Key 1/2)인지, 엔드포인트와 같은 리소스인지 확인하세요.")
    if status == 403:
        raise AuthError(f"Azure Speech 접근 거부 (403): {msg}",
                        hint="리소스 네트워크 설정(방화벽/프라이빗 엔드포인트)·Entra ID 역할(Cognitive Services User)·"
                             "키 인증 비활성화 여부를 확인하세요.")
    if status == 404:
        raise AzureSpeechError(f"Azure Speech 엔드포인트를 찾을 수 없습니다 (404): {msg}",
                               hint=f"AZURE_SPEECH_ENDPOINT({host})가 Speech/Foundry 리소스의 엔드포인트인지 확인하세요.")
    if status == 413:
        raise AzureSpeechError("업로드 파일이 너무 큽니다 (413).", hint="asr.azure.max_chunk_minutes 를 줄이세요.")
    if status == 400:
        low = msg.lower()
        if any(k in low for k in ("region", "not supported", "not available", "unsupported", "model")):
            raise AzureSpeechError(f"MAI-Transcribe 요청 거부 (400): {msg}",
                                   hint="리소스 리전이 MAI-Transcribe 지원 리전(" + ", ".join(MAI_REGIONS) +
                                        ")인지, asr.azure.model / api_version 이 올바른지 확인하세요.")
        raise AzureSpeechError(f"Azure Speech 요청 오류 (400): {msg}")
    if status == 429:
        raise AzureSpeechError(f"Azure Speech 요청 한도 초과 (429): {msg}",
                               hint="잠시 후 다시 시도하거나 리소스의 요청 한도(분당 요청 수)를 확인하세요.")
    raise AzureSpeechError(f"Azure Speech 서버 오류 (HTTP {status}): {msg}", hint="잠시 후 다시 시도하세요.")


def _post(session, url: str, auth: _Auth, audio_path: Path, definition: dict, timeout: float,
          max_retries: int, host: str) -> dict:
    import requests

    mime = "audio/flac" if audio_path.suffix == ".flac" else "audio/wav"
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            with open(audio_path, "rb") as fh:
                resp = session.post(
                    url,
                    headers=auth.headers(),
                    files={"audio": (audio_path.name, fh, mime),
                           "definition": (None, json.dumps(definition, ensure_ascii=False), "application/json")},
                    timeout=(30, timeout),
                )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(min(30.0, 2 ** (attempt + 1)) + random.random())
                continue
            raise AzureSpeechError(
                f"Azure Speech 연결 실패: {host} ({type(exc).__name__})",
                hint="사내 프록시/방화벽에서 *.cognitiveservices.azure.com 허용 여부와 HTTPS_PROXY 설정을 확인하세요.",
            ) from exc
        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError as exc:
                raise AzureSpeechError("Azure Speech 응답을 해석할 수 없습니다 (JSON 아님).") from exc
        if resp.status_code in _RETRY_STATUS and attempt < max_retries:
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else 2 ** (attempt + 1)
            except ValueError:
                wait = 2 ** (attempt + 1)
            logger.warning("Azure Speech HTTP %s, retry %d/%d in %.1fs", resp.status_code, attempt + 1,
                           max_retries, wait)
            time.sleep(min(wait, 60.0))
            continue
        _raise_for(resp, host)
    raise AzureSpeechError("Azure Speech 요청 실패") from last_exc


# --------------------------------------------------------------------------- transcriber


class AzureMaiTranscriber:
    """Same role as WhisperXTranscriber + WhisperXAligner: returns word-timestamped segments."""

    provider = "azure_mai"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.notes: list[str] = []
        raw = azure_speech_endpoint(cfg)
        if not raw:
            raise ConfigError(
                "Azure Speech 엔드포인트가 설정되지 않았습니다 (AZURE_SPEECH_ENDPOINT).",
                hint="UI '시스템 진단' 탭 > Azure MAI 연결 설정에 포털 리소스 URL(또는 엔드포인트)과 키를 넣고 저장하거나, "
                     ".env 에 AZURE_SPEECH_ENDPOINT=https://<리소스명>.cognitiveservices.azure.com 을 입력하세요 "
                     "(CLI: meeting-transcriber azure-setup). 로컬 음성인식을 쓰려면 asr.backend=whisperx.",
            )
        self.base, notes = normalize_endpoint(raw)
        self.notes.extend(notes)
        self.host = urlparse(self.base).netloc
        az = cfg.asr.azure
        key = azure_speech_key()
        if az.auth == "key" and not key:
            raise ConfigError("Azure Speech 키가 설정되지 않았습니다 (AZURE_SPEECH_KEY).",
                              hint="UI '시스템 진단' 탭 > Azure MAI 연결 설정에 키를 넣고 저장하거나 .env 에 "
                                   "AZURE_SPEECH_KEY=<리소스 키> 를 입력하세요 (키 없는 인증은 asr.azure.auth=entra).")
        self.auth = _Auth(az.auth, key)
        self.url = transcribe_url(self.base, az.api_version)

    # pipeline interface (no local model to load)
    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def describe(self) -> dict:
        return {"provider": "Azure Speech (Microsoft Foundry)", "model": self.cfg.asr.azure.model,
                "endpoint_host": self.host, "auth": self.cfg.asr.azure.auth, "api_version": self.cfg.asr.azure.api_version}

    def transcribe(self, audio: np.ndarray, sample_rate: int, workdir: Path, progress: Progress = None,
                   extra_phrases: Optional[list[str]] = None) -> dict:
        import requests

        az = self.cfg.asr.azure
        definition = build_definition(self.cfg, extra_phrases)
        chunks = plan_chunks(len(audio), sample_rate, audio, az.max_chunk_minutes * 60)
        workdir = Path(workdir)
        segments: list[dict] = []
        lang_sec: Counter = Counter()
        with requests.Session() as session:
            for i, (a, b) in enumerate(chunks):
                if progress:
                    progress(i / len(chunks))
                upload = encode_upload(audio[a:b], sample_rate, workdir / f"_mai_upload_{i:03d}", az.upload_format)
                try:
                    size_mb = upload.stat().st_size / 1e6
                    if size_mb >= 500:
                        raise AzureSpeechError(f"업로드 파일이 API 한도(500 MB)를 넘습니다: {size_mb:.0f} MB",
                                               hint="asr.azure.max_chunk_minutes 를 줄이세요.")
                    logger.info("MAI-Transcribe request %d/%d (%.1f min, %.1f MB) -> %s", i + 1, len(chunks),
                                (b - a) / sample_rate / 60, size_mb, self.host)
                    payload = _post(session, self.url, self.auth, upload, definition, az.timeout_sec,
                                    az.max_retries, self.host)
                finally:
                    upload.unlink(missing_ok=True)
                segs, langs = parse_transcription(payload, offset_sec=a / sample_rate)
                segments.extend(segs)
                lang_sec.update(langs)
        if progress:
            progress(1.0)
        segments.sort(key=lambda s: (s["start"], s["end"]))
        language = lang_sec.most_common(1)[0][0] if lang_sec else (
            self.cfg.asr.language if self.cfg.asr.language != "auto" else "unknown")
        has_words = any(s["words"] for s in segments)
        return {
            "segments": segments,
            "language": language,
            "aligned": has_words,
            "word_timestamps_source": "azure_mai" if has_words else None,
            "provider": self.provider,
            "requests": len(chunks),
            "languages_detected": {k: round(v, 1) for k, v in lang_sec.items()},
            "phrase_list": definition.get("phraseList", {}).get("phrases", []),
            "locale_forced": definition.get("locales"),
        }

    def check(self, workdir: Path) -> dict:
        """Connectivity / permission test with a 2-second tone (billed as ~2 s of audio)."""
        sr = 16000
        t = np.arange(2 * sr) / sr
        tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        t0 = time.perf_counter()
        result = self.transcribe(tone, sr, workdir)
        return {
            "ok": True,
            "endpoint_host": self.host,
            "model": self.cfg.asr.azure.model,
            "latency_sec": round(time.perf_counter() - t0, 2),
            "segments": len(result["segments"]),
            "notes": self.notes,
        }


__all__ = [
    "MAI_REGIONS",
    "AzureMaiTranscriber",
    "AzureSpeechError",
    "build_definition",
    "normalize_endpoint",
    "parse_phrases_text",
    "parse_transcription",
    "plan_chunks",
    "transcribe_url",
]
