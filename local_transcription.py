"""Optional local singing transcription. No remote transcription or paid fallback.

Prepare the weights explicitly with ``python local_transcription.py --download``.
Requests use only cached weights, so a cold model cannot stall an HTTP request
while downloading gigabytes. FFmpeg/FFprobe are the same as the API pipeline.
"""
import argparse
from functools import lru_cache
import logging
import os

from transcription_chunks import iter_audio_chunks, merge_chunk_transcripts

logger = logging.getLogger("audio-processor.local-transcription")


def _settings():
    device = os.getenv("LOCAL_WHISPER_DEVICE", "cpu")
    return {
        "model_size_or_path": os.getenv("LOCAL_WHISPER_MODEL", "large-v3-turbo"),
        "device": device,
        "compute_type": os.getenv("LOCAL_WHISPER_COMPUTE_TYPE", "int8" if device == "cpu" else "float16"),
        "cpu_threads": max(1, int(os.getenv("LOCAL_WHISPER_THREADS", "4"))),
        "download_root": os.getenv("LOCAL_WHISPER_CACHE") or None,
    }


@lru_cache(maxsize=1)
def _load_model(model_size_or_path, device, compute_type, cpu_threads, download_root):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Falta faster-whisper. Instala requirements-local.txt antes de activar el motor local.") from exc
    try:
        return WhisperModel(model_size_or_path, device=device, compute_type=compute_type,
                            cpu_threads=cpu_threads, download_root=download_root, local_files_only=True)
    except Exception as exc:
        raise RuntimeError("No se pudo cargar el modelo local. Prepara sus pesos con "
                           "python local_transcription.py --download y comprueba dispositivo y memoria.") from exc


def transcribe_local_audio(audio_path, *, chunk_seconds=None, overlap_seconds=2.0):
    """Consume each temporary chunk before it is released, then restore its times."""
    if chunk_seconds is None:
        chunk_seconds = float(os.getenv("LOCAL_WHISPER_CHUNK_SECONDS", "120"))
    settings = _settings()
    model = _load_model(**settings)
    model_name = f"faster-whisper/{settings['model_size_or_path']}/{settings['compute_type']}"
    results = []
    with iter_audio_chunks(audio_path, chunk_seconds=chunk_seconds,
                           overlap_seconds=overlap_seconds) as chunks:
        for chunk in chunks:
            generated, _ = model.transcribe(
                chunk.path, language="es", beam_size=5, word_timestamps=True,
                condition_on_previous_text=False, vad_filter=False, temperature=0,
                hallucination_silence_threshold=2.0,
            )
            segments, words = [], []
            for segment in generated:
                segments.append({"text": segment.text.strip(), "start": segment.start, "end": segment.end})
                words.extend({"word": word.word.strip(), "start": word.start, "end": word.end,
                              "probability": word.probability} for word in segment.words or [])
            result = {"text": "\n".join(s["text"] for s in segments), "segments": segments,
                      "words": words, "model": model_name, "warnings": []}
            results.append((chunk, result))
            logger.info("Transcripcion local: fragmento %d, %.1f-%.1fs", chunk.index + 1,
                        chunk.core_start, chunk.core_end)
    merged = merge_chunk_transcripts(results)
    merged["chunkCount"] = len(results)
    merged["chunkSeconds"] = chunk_seconds
    merged["overlapSeconds"] = overlap_seconds
    merged["duration"] = results[-1][0].core_end if results else 0.0
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparar el modelo gratuito de transcripcion local")
    parser.add_argument("--download", action="store_true", required=True)
    parser.parse_args()
    from faster_whisper import WhisperModel
    WhisperModel(**_settings(), local_files_only=False)
    print("Modelo local preparado. Las peticiones usaran esta misma configuracion y cache.")
