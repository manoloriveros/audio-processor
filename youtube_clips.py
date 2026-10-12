"""Recortes de YouTube para un video de setlist, sin transcripción ni persistencia."""
import math
import os
import re
import shutil
import subprocess
import tempfile

from fastapi import HTTPException

MAX_CLIP_SECONDS = 5
MAX_CLIP_BYTES = 48000 * 2 * 2 * MAX_CLIP_SECONDS + 4096


def render_youtube_clip(video_id, start, duration, download_audio, max_duration=720):
    if not isinstance(video_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{11}", video_id):
        raise HTTPException(400, "Video de YouTube inválido")
    if (isinstance(start, bool) or isinstance(duration, bool)
            or not isinstance(start, (int, float)) or not isinstance(duration, (int, float))
            or not math.isfinite(start) or not math.isfinite(duration)
            or start < 0 or not 0 < duration <= MAX_CLIP_SECONDS
            or start + duration > max_duration):
        raise HTTPException(400, "El fragmento debe durar hasta cinco segundos y tener un inicio válido")

    workdir = tempfile.mkdtemp(prefix="setlist_clip_")
    try:
        audio_path = download_audio(video_id, workdir)
        # El archivo viene del descargador interno, nunca de una ruta enviada por el cliente.
        if os.path.commonpath([os.path.realpath(audio_path), os.path.realpath(workdir)]) != os.path.realpath(workdir):
            raise HTTPException(502, "No se pudo preparar el fragmento")
        result = subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", str(start), "-i", audio_path, "-t", str(duration),
            "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "48000",
            "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
        ], capture_output=True, check=True, timeout=45)
        if len(result.stdout) <= 128 or len(result.stdout) > MAX_CLIP_BYTES:
            raise HTTPException(422, "No hay audio en el marcador elegido")
        return result.stdout
    except FileNotFoundError as exc:
        raise HTTPException(503, "El servicio de recorte necesita FFmpeg") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(502, "No se pudo recortar el audio de esta grabación") from exc
    finally:
        # Incluye audio descargado y cualquier archivo auxiliar del descargador.
        shutil.rmtree(workdir, ignore_errors=True)

