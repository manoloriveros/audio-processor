"""
Separacion vocal/instrumental self-hosted con audio-separator (MDX-Net ONNX, CPU).

La separacion es el mayor salto de calidad del pipeline local:
  - La VOZ aislada va a Whisper: elimina las alucinaciones causadas por
    instrumentos y bateria, y mejora los timestamps por palabra.
  - El INSTRUMENTAL va al detector de acordes: el cromagrama deja de estar
    contaminado por la melodia vocal.

Se ejecuta como subproceso (CLI de audio-separator) con timeout duro: si el
paquete no esta instalado, el audio es demasiado largo, falla o tarda de mas,
se devuelve (None, None, None) y el pipeline continua con el mix original.

Variables de entorno:
  AUDIO_SEPARATION          "0" para desactivar (default: activada)
  SEPARATION_MODEL          modelo MDX (default: Kim_Vocal_2.onnx)
  SEPARATION_TIMEOUT        segundos maximos del subproceso (default: 210)
  SEPARATION_MAX_DURATION   segundos maximos de audio para intentarla (default: 480)
  MODEL_FILE_DIR            carpeta de modelos descargados (default: /models)
"""

import logging
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger("audio-processor.separation")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def is_available() -> bool:
    if _env("AUDIO_SEPARATION", "1") == "0":
        return False
    return shutil.which("audio-separator") is not None


def _audio_duration(audio_path: str) -> float | None:
    """Duracion en segundos via ffprobe (ffmpeg ya esta en la imagen)."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0", audio_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def separate(audio_path: str):
    """Devuelve (vocals_path, instrumental_path, workdir) o (None, None, None).

    El llamador debe borrar workdir cuando termine de usar los stems.
    """
    if not is_available():
        return None, None, None

    max_duration = float(_env("SEPARATION_MAX_DURATION", "480"))
    duration = _audio_duration(audio_path)
    if duration is not None and duration > max_duration:
        logger.info(
            "Separacion omitida: audio de %.0fs supera el maximo de %.0fs",
            duration, max_duration,
        )
        return None, None, None

    model = _env("SEPARATION_MODEL", "Kim_Vocal_2.onnx")
    timeout_s = float(_env("SEPARATION_TIMEOUT", "210"))
    workdir = tempfile.mkdtemp(prefix="stems_")

    cmd = [
        "audio-separator", audio_path,
        "--model_filename", model,
        "--model_file_dir", _env("MODEL_FILE_DIR", "/models"),
        "--output_dir", workdir,
        "--output_format", "WAV",
    ]
    # Nota: sin --custom_output_names por compatibilidad con audio-separator
    # 0.30.x (la version que resuelve pip con numpy<2); los stems se localizan
    # por nombre de archivo ("(Vocals)" / "(Instrumental)").
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
        if proc.returncode != 0:
            logger.warning(
                "audio-separator fallo (rc=%d): %s",
                proc.returncode, (proc.stderr or proc.stdout or "")[-500:],
            )
            shutil.rmtree(workdir, ignore_errors=True)
            return None, None, None

        vocals = instrumental = None
        for name in os.listdir(workdir):
            lowered = name.lower()
            full = os.path.join(workdir, name)
            if "vocal" in lowered and "instrumental" not in lowered:
                vocals = full
            elif "instrumental" in lowered:
                instrumental = full

        if not vocals or not instrumental:
            logger.warning(
                "Separacion sin stems reconocibles; archivos: %s", os.listdir(workdir),
            )
            shutil.rmtree(workdir, ignore_errors=True)
            return None, None, None

        logger.info(
            "Separacion completada (%s): vocals=%s, instrumental=%s",
            model, os.path.basename(vocals), os.path.basename(instrumental),
        )
        return vocals, instrumental, workdir
    except subprocess.TimeoutExpired:
        logger.warning("Separacion cancelada: supero el timeout de %.0fs", timeout_s)
        shutil.rmtree(workdir, ignore_errors=True)
        return None, None, None
    except Exception as exc:
        logger.warning("Separacion fallida: %s", exc)
        shutil.rmtree(workdir, ignore_errors=True)
        return None, None, None
