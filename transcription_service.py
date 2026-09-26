"""Bounded transcription requests with a recording-wide, measured timeline."""

import logging
import math

from transcription_alignment import align_corrected_words
from transcription_chunks import iter_audio_chunks, merge_chunk_transcripts

logger = logging.getLogger("audio-processor.transcription")


def _field(item, name, default=None):
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def _timed_items(response, field, text_field):
    items = []
    for item in _field(response, field, []) or []:
        text = str(_field(item, text_field, "") or "").strip()
        try:
            start, end = float(_field(item, "start")), float(_field(item, "end"))
        except (TypeError, ValueError):
            continue
        if text and math.isfinite(start) and math.isfinite(end) and 0 <= start < end:
            items.append({text_field: text, "start": start, "end": end})
    return sorted(items, key=lambda item: item["start"])


def _segments_with_words(words, segments):
    if not words:
        return segments
    groups = [[] for _ in segments]
    unmatched = []
    for word in words:
        midpoint = (word["start"] + word["end"]) / 2
        target = next((i for i, seg in enumerate(segments)
                       if seg["start"] <= midpoint < seg["end"]), None)
        if target is None:
            unmatched.append(word)
        else:
            groups[target].append(word)
    # A partial word response must not erase whole timed lyric segments.
    retained_segments = [dict(segment) for segment, group in zip(segments, groups) if not group]
    groups.extend([[word] for word in unmatched])
    retained_segments.extend(
        {"text": " ".join(w["word"] for w in group),
         "start": group[0]["start"], "end": max(w["end"] for w in group)}
        for group in groups if group
    )
    return sorted(retained_segments, key=lambda seg: seg["start"])


def transcribe_chunk(client, audio_path, *, timestamp_model, text_models, prompt):
    """Transcribe one window. Never fabricate word times from character counts."""
    segments, words, warnings = [], [], []
    timed_text, corrected_text, text_model = "", "", None
    timestamp_succeeded = False
    try:
        with open(audio_path, "rb") as audio:
            response = client.audio.transcriptions.create(
                model=timestamp_model, file=audio, response_format="verbose_json",
                timestamp_granularities=["word", "segment"], language="es",
                **({"prompt": prompt} if prompt else {}), temperature=0,
            )
        timestamp_succeeded = True
        timed_text = str(_field(response, "text", "") or "").strip()
        words = _timed_items(response, "words", "word")
        segments = _timed_items(response, "segments", "text")
    except Exception as exc:
        logger.warning("Transcripcion temporal fallida: %s", exc)
        warnings.append("timestamp_model_failed")

    # A successful empty response is a valid instrumental/silent window.
    if timed_text or words or segments or not timestamp_succeeded:
        for candidate in dict.fromkeys(text_models):
            if not candidate or candidate == timestamp_model:
                continue
            try:
                with open(audio_path, "rb") as audio:
                    response = client.audio.transcriptions.create(
                        model=candidate, file=audio, response_format="json",
                        language="es", **({"prompt": prompt} if prompt else {}), temperature=0,
                    )
                corrected_text = str(_field(response, "text", "") or "").strip()
                text_model = candidate
                break
            except Exception as exc:
                logger.warning("Transcripcion textual fallida (%s): %s", candidate, exc)

    if not timestamp_succeeded and text_model is None:
        raise RuntimeError("No se pudo transcribir un fragmento del audio; no se devuelve una cancion incompleta")

    model = timestamp_model if timestamp_succeeded else text_model
    if words:
        if corrected_text:
            words, accepted = align_corrected_words(corrected_text, words)
            if accepted:
                model = f"{text_model}+{timestamp_model}"
            else:
                warnings.append("text_correction_rejected_to_preserve_timing")
        segments = _segments_with_words(words, segments)
        text = "\n".join(segment["text"] for segment in segments)
    elif segments:
        text = " ".join(seg["text"] for seg in segments)
        warnings.append("word_timestamps_unavailable")
    else:
        text = timed_text or corrected_text
        if text:
            warnings.append("word_timestamps_unavailable")
    return {"text": text, "words": words, "segments": segments,
            "model": model, "warnings": warnings}


def transcribe_audio(audio_path, *, api_key, timestamp_model, text_models, prompt,
                     chunk_seconds=120.0, overlap_seconds=2.0):
    import openai

    results = []
    # Client resources and each temporary WAV are released even on cancellation/error.
    with openai.OpenAI(api_key=api_key, timeout=90, max_retries=1) as client:
        with iter_audio_chunks(audio_path, chunk_seconds=chunk_seconds,
                               overlap_seconds=overlap_seconds) as chunks:
            for chunk in chunks:
                result = transcribe_chunk(
                    client, chunk.path, timestamp_model=timestamp_model,
                    text_models=text_models, prompt=prompt,
                )
                results.append((chunk, result))
    merged = merge_chunk_transcripts(results)
    merged["chunkCount"] = len(results)
    merged["duration"] = results[-1][0].core_end if results else 0.0
    return merged
