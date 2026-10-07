"""Bounded YouTube download used only by automatic marker transcription.

The downloader runs in a disposable subprocess so a stream or stalled extractor
cannot keep the processor's only worker slot forever. No shell or credentials in
command arguments are used; yt-dlp reads its existing proxy/cookies environment.
"""
from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 180

# Source failures are emitted before transcription. Never return provider text:
# it may contain proxy credentials, signed media URLs or cookie diagnostics.
SOURCE_ERROR_MESSAGES = {
    "YOUTUBE_ACCESS_RESTRICTED": "YouTube restringió el acceso al video desde el servidor. Revisa el enlace o usa una grabación accesible.",
    "NO_AUDIO_FORMAT": "YouTube no ofreció un formato de audio descargable para esta grabación.",
    "JS_RUNTIME_UNAVAILABLE": "Falta un motor JavaScript o un componente EJS compatible para descargar esta grabación.",
    "VIDEO_UNAVAILABLE": "No se pudo descargar el audio del video.",
}
CHILD_ERROR_DETAILS = {
    **{code: (502, message) for code, message in SOURCE_ERROR_MESSAGES.items()},
    "INVALID_AUDIO": (400, "No se pudo obtener un audio válido y acotado de esta grabación."),
    "DURATION_LIMIT": (400, "La grabación supera la duración máxima permitida."),
}


class MarkerDownloadError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _source_error_code(message: str) -> str:
    """Classify locally; the raw text is never stored in a result or log."""
    text = str(message).casefold().replace("’", "'")
    if any(pattern in text for pattern in (
        "sign in to confirm", "confirm you're not a bot", "sign in to view",
        "login required", "authentication required", "log in to", "login to",
        "requires login", "requires authentication", "age-restricted", "age restricted",
        "private video", "video is private", "members-only", "members only",
        "not available in your country", "not available in your region",
        "not available from your location", "geo-restricted", "geographic restriction",
    )):
        return "YOUTUBE_ACCESS_RESTRICTED"
    if any(pattern in text for pattern in (
        "no supported javascript runtime", "no js challenge providers",
        "javascript runtime is not available", "javascript runtime not found",
        "cannot find a javascript runtime",
    )) or (("ejs" in text or "challenge solver" in text) and any(pattern in text for pattern in (
        "skipped", "not installed", "unavailable", "not found", "missing",
    ))):
        return "JS_RUNTIME_UNAVAILABLE"
    if any(pattern in text for pattern in (
        "requested format is not available", "no video formats found", "no formats found",
        "no audio formats", "only images are available",
    )):
        return "NO_AUDIO_FORMAT"
    return "VIDEO_UNAVAILABLE"


def _classify_source_error(error: Exception, observed_codes=()) -> MarkerDownloadError:
    codes = {_source_error_code(str(error)), *observed_codes}
    # A missing runtime can cause the final error to mention only missing audio
    # formats. Explicit access restrictions still take priority over warnings.
    for code in ("YOUTUBE_ACCESS_RESTRICTED", "JS_RUNTIME_UNAVAILABLE", "NO_AUDIO_FORMAT"):
        if code in codes:
            return MarkerDownloadError(502, code, SOURCE_ERROR_MESSAGES[code])
    return MarkerDownloadError(502, "VIDEO_UNAVAILABLE", SOURCE_ERROR_MESSAGES["VIDEO_UNAVAILABLE"])


def _validate_metadata(info: dict, maximum_duration: float) -> None:
    if not isinstance(info, dict) or info.get("is_live") or info.get("is_upcoming") or (
        str(info.get("live_status") or "").lower() in {"is_live", "is_upcoming", "post_live"}
    ):
        raise MarkerDownloadError(400, "INVALID_AUDIO", "Usa una grabacion terminada, no una transmision en vivo")
    duration = info.get("duration")
    try:
        duration = float(duration) if not isinstance(duration, bool) else 0
    except (TypeError, ValueError, OverflowError):
        duration = 0
    if not math.isfinite(duration) or duration <= 0:
        raise MarkerDownloadError(400, "INVALID_AUDIO", "La grabacion no tiene una duracion verificable")
    if duration > maximum_duration:
        raise MarkerDownloadError(400, "DURATION_LIMIT", "La grabacion supera la duracion maxima permitida")


def _download_in_child(video_id: str, workdir: str, maximum_duration: float) -> str:
    import yt_dlp

    exceeded_size = False
    observed_codes = set()

    class QuietLogger:
        # Keep stdout exclusive to the structured parent/child response.
        def debug(self, message):
            pass
        def warning(self, message):
            observed_codes.add(_source_error_code(message))
        def error(self, message):
            observed_codes.add(_source_error_code(message))

    def progress(event):
        nonlocal exceeded_size
        if (event.get("downloaded_bytes") or 0) > MAX_DOWNLOAD_BYTES:
            exceeded_size = True
            raise MarkerDownloadError(400, "INVALID_AUDIO", "El audio del video excede 25 MB")

    def match_filter(info, *, incomplete=False):
        if not incomplete:
            _validate_metadata(info, maximum_duration)
        return None

    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio", "outtmpl": os.path.join(workdir, "audio.%(ext)s"),
        "noplaylist": True, "quiet": True, "no_warnings": False,
        "socket_timeout": 15, "retries": 0, "fragment_retries": 0,
        "max_filesize": MAX_DOWNLOAD_BYTES, "progress_hooks": [progress],
        "match_filter": match_filter, "logger": QuietLogger(),
    }
    if os.getenv("YTDLP_PROXY"):
        opts["proxy"] = os.environ["YTDLP_PROXY"]
    if os.getenv("YTDLP_COOKIES_B64"):
        cookie_path = Path(workdir) / "cookies.txt"
        cookie_path.write_bytes(base64.b64decode(os.environ["YTDLP_COOKIES_B64"], validate=True))
        opts["cookiefile"] = str(cookie_path)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            url = f"https://www.youtube.com/watch?v={video_id}"
            info = ydl.extract_info(url, download=False)
            _validate_metadata(info, maximum_duration)
            info = ydl.extract_info(url, download=True)
            _validate_metadata(info, maximum_duration)
            path = Path(ydl.prepare_filename(info)).resolve()
    except MarkerDownloadError:
        raise
    except Exception as exc:
        if exceeded_size:
            raise MarkerDownloadError(400, "INVALID_AUDIO", "El audio del video excede 25 MB") from exc
        raise _classify_source_error(exc, observed_codes) from exc
    directory = Path(workdir).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise MarkerDownloadError(502, "VIDEO_UNAVAILABLE", "La descarga no produjo audio")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_DOWNLOAD_BYTES:
        raise MarkerDownloadError(400, "INVALID_AUDIO", "El archivo de audio tiene un tamano no valido")
    return str(path)


def download_marker_audio(video_id: str, workdir: str, maximum_duration: float) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{11}", video_id):
        raise MarkerDownloadError(400, "INVALID_SOURCE", "URL de YouTube no valida")
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), video_id, workdir, str(maximum_duration)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=DOWNLOAD_TIMEOUT_SECONDS, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise MarkerDownloadError(502, "VIDEO_UNAVAILABLE", "La descarga excedio el tiempo permitido") from exc
    except OSError as exc:
        raise MarkerDownloadError(502, "VIDEO_UNAVAILABLE", "No se pudo iniciar la descarga del audio") from exc
    try:
        result = json.loads(completed.stdout)
    except (ValueError, TypeError):
        result = None
    if completed.returncode or not isinstance(result, dict) or not result.get("path"):
        if isinstance(result, dict) and isinstance(result.get("code"), str) and result["code"] in CHILD_ERROR_DETAILS:
            code = result["code"]
            status, message = CHILD_ERROR_DETAILS[code]
            raise MarkerDownloadError(status, code, message)
        raise MarkerDownloadError(502, "VIDEO_UNAVAILABLE", "No se pudo descargar el audio del video")
    path = Path(result["path"]).resolve()
    if not path.is_relative_to(Path(workdir).resolve()) or not path.is_file():
        raise MarkerDownloadError(502, "VIDEO_UNAVAILABLE", "La descarga no produjo audio")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_DOWNLOAD_BYTES:
        raise MarkerDownloadError(400, "INVALID_AUDIO", "El archivo de audio tiene un tamano no valido")
    return str(path)


if __name__ == "__main__":
    try:
        downloaded = _download_in_child(sys.argv[1], sys.argv[2], float(sys.argv[3]))
        print(json.dumps({"path": downloaded}))
    except MarkerDownloadError as error:
        print(json.dumps({"code": error.code, "message": error.message}))
        sys.exit(1)
    except Exception:
        print(json.dumps({"code": "VIDEO_UNAVAILABLE", "message": "No se pudo descargar el audio del video"}))
        sys.exit(1)
