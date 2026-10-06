"""User-facing error types.

Every failure that a user can act on is raised as ``UserFacingError`` with a Korean message and a
hint. The UI / CLI display ``str(err)``; stack traces only go to the debug log.
"""

from __future__ import annotations


class UserFacingError(Exception):
    def __init__(self, message: str, hint: str | None = None, stage: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.stage = stage

    def __str__(self) -> str:
        parts = []
        if self.stage:
            parts.append(f"[{self.stage}] ")
        parts.append(self.message)
        if self.hint:
            parts.append(f"\n→ 조치: {self.hint}")
        return "".join(parts)


class ConfigError(UserFacingError):
    pass


class AudioError(UserFacingError):
    pass


class ModelLoadError(UserFacingError):
    pass


class AuthError(ModelLoadError):
    pass


class GpuOutOfMemoryError(UserFacingError):
    pass


class EnrollmentError(UserFacingError):
    pass


def is_cuda_oom(exc: BaseException) -> bool:
    """True for torch and CTranslate2 CUDA out-of-memory errors."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # torch missing or very old
        pass
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cublas" in text or "cudnn" in text)


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def is_network_error(exc: BaseException) -> bool:
    """Connection / proxy / timeout / offline-cache-miss anywhere in the exception chain."""
    try:
        import requests

        net_types: tuple = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
    except Exception:  # pragma: no cover
        net_types = ()
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

        net_types += (LocalEntryNotFoundError, OfflineModeIsEnabled)
    except Exception:  # pragma: no cover
        pass
    net_types += (ConnectionError, TimeoutError)
    for e in _chain(exc):
        if net_types and isinstance(e, net_types):
            return True
        text = str(e).lower()
        if any(k in text for k in ("tunnel connection failed", "max retries exceeded", "unable to connect to proxy",
                                   "name or service not known", "temporary failure in name resolution",
                                   "cannot find the requested files in the disk cache", "timed out")):
            return True
    return False


def is_auth_error(exc: BaseException) -> bool:
    """401/403 from the Hub (token missing/invalid, gated model conditions not accepted) or unknown repo."""
    if is_network_error(exc):  # e.g. a proxy answering CONNECT with 403 is a network problem, not auth
        return False
    try:
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except Exception:  # pragma: no cover
        GatedRepoError = HfHubHTTPError = RepositoryNotFoundError = ()  # type: ignore
    for e in _chain(exc):
        if GatedRepoError and isinstance(e, (GatedRepoError, RepositoryNotFoundError)):
            return True
        if HfHubHTTPError and isinstance(e, HfHubHTTPError):
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                return True
        text = str(e)
        if any(k in text for k in ("401 Client Error", "403 Client Error", "gated", "Gated", "Unauthorized")):
            return True
    return False
