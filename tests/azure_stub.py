"""Local HTTP double of the Azure Speech fast-transcription endpoint used by MAI-Transcribe-2.

It follows the documented contract only (MicrosoftDocs/azure-ai-docs: mai-transcribe.md and the
LLM speech REST quickstart): POST /speechtotext/transcriptions:transcribe?api-version=...,
multipart fields `audio` + `definition`, key header, JSON `phrases[].words[]` response with
millisecond offsets, lower-cased lexical word tokens and display text with punctuation.
It does not recognise speech - it returns scripted Korean phrases spread over the uploaded audio.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

KEY = "stub-key-0123456789abcdef"
SCRIPT = [
    ("오늘 회의를 시작하겠습니다.", ["오늘", "회의를", "시작하겠습니다"]),
    ("네, 맞습니다.", ["네", "맞습니다"]),
    ("APQR 초안은 다음 주까지 공유드리겠습니다.", ["apqr", "초안은", "다음", "주까지", "공유드리겠습니다"]),
]


def _duration_ms(audio: bytes, suffix: str) -> int:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio)
        path = f.name
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True).stdout.strip()
        return int(float(out) * 1000)
    finally:
        Path(path).unlink(missing_ok=True)


def scripted_response(duration_ms: int, phrase_ms: int = 4000) -> dict:
    phrases, t, i = [], 300, 0
    while t + 1500 < duration_ms:
        text, words = SCRIPT[i % len(SCRIPT)]
        dur = min(phrase_ms - 500, duration_ms - t)
        step = dur // len(words)
        phrases.append({
            "offsetMilliseconds": t, "durationMilliseconds": dur, "text": text, "locale": "ko-KR", "confidence": 0,
            "words": [{"text": w, "offsetMilliseconds": t + k * step, "durationMilliseconds": max(40, step - 40)}
                      for k, w in enumerate(words)],
        })
        t += phrase_ms
        i += 1
    return {"durationMilliseconds": duration_ms, "combinedPhrases": [{"text": " ".join(p["text"] for p in phrases)}],
            "phrases": phrases}


class AzureSpeechStub:
    """mode: ok | 429_then_ok | 401 | 400_region | 500_always"""

    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _send(self, status, payload, headers=None):
                body = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                url = urlparse(self.path)
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                msg = BytesParser(policy=policy.default).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
                parts = {p.get_param("name", header="content-disposition"): p for p in msg.iter_parts()}
                audio = parts.get("audio")
                definition = json.loads(parts["definition"].get_content()) if "definition" in parts else None
                rec = {
                    "path": url.path, "query": parse_qs(url.query), "key": self.headers.get("Ocp-Apim-Subscription-Key"),
                    "definition": definition,
                    "audio_filename": audio.get_filename() if audio else None,
                    "audio_type": audio.get_content_type() if audio else None,
                    "audio_bytes": len(audio.get_payload(decode=True)) if audio else 0,
                }
                stub.requests.append(rec)
                if url.path != "/speechtotext/transcriptions:transcribe":
                    return self._send(404, {"error": {"code": "NotFound", "message": "Resource not found"}})
                if rec["key"] != KEY or stub.mode == "401":
                    return self._send(401, {"error": {"code": "401", "message": "Access denied due to invalid subscription key."}})
                if stub.mode == "400_region":
                    return self._send(400, {"error": {"code": "InvalidArgument",
                                                      "message": "Model MAI-Transcribe-2 is not supported in this region."}})
                if stub.mode == "500_always":
                    return self._send(503, {"error": {"code": "ServiceUnavailable", "message": "busy"}})
                if stub.mode == "429_then_ok" and len(stub.requests) == 1:
                    return self._send(429, {"error": {"code": "429", "message": "Too many requests"}}, {"Retry-After": "0"})
                suffix = ".flac" if rec["audio_type"] == "audio/flac" else ".wav"
                return self._send(200, scripted_response(_duration_ms(audio.get_payload(decode=True), suffix)))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
