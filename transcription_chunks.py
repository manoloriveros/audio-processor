"""Bounded audio preparation and timestamp-aware transcript assembly.

No transcription service is called here. Each yielded WAV is valid only until
the iterator advances (or its context exits), so callers must consume it eagerly.
Only metadata and transcript dictionaries need to be retained for merging.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from typing import Iterable, Iterator
import wave


MAX_CHUNK_BYTES = 24_000_000


@dataclass(frozen=True)
class AudioChunk:
    path: str
    index: int
    start: float
    core_start: float
    core_end: float
    end: float
    is_last: bool

    @property
    def duration(self) -> float:
        return self.end - self.start

    def owns(self, absolute_midpoint: float) -> bool:
        return self.core_start <= absolute_midpoint and (
            absolute_midpoint < self.core_end
            or (self.is_last and absolute_midpoint == self.core_end)
        )


def _positive(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _run(args: list[str], timeout_seconds: float) -> str:
    try:
        completed = subprocess.run(
            args, check=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout_seconds,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{Path(args[0]).name} exceeded its audio-processing timeout") from exc
    except subprocess.CalledProcessError as exc:
        detail = str(exc.stderr or "").strip()[-1500:]
        raise RuntimeError(f"{Path(args[0]).name} could not process audio: {detail}") from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot start {Path(args[0]).name}: {exc}") from exc
    return completed.stdout


def probe_audio_duration(
    audio_path: str, *, ffprobe: str = "ffprobe", timeout_seconds: float = 60.0,
) -> float:
    """Read duration and verify an audio stream without decoding PCM into memory."""
    source = Path(audio_path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("Audio input must be a file")
    timeout_seconds = _positive(timeout_seconds, "timeout_seconds")
    raw = _run([
        ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type",
        "-of", "json", str(source),
    ], timeout_seconds)
    try:
        data = json.loads(raw)
        if not any(stream.get("codec_type") == "audio" for stream in data.get("streams", [])):
            raise ValueError("Input contains no audio stream")
        return _positive(data.get("format", {}).get("duration"), "audio duration")
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        raise ValueError("ffprobe returned invalid audio metadata") from exc


def _verify_wav(path: Path, expected_duration: float, sample_rate: int) -> None:
    if not path.is_file() or path.stat().st_size <= 44:
        raise RuntimeError("ffmpeg did not produce a nonempty audio chunk")
    if path.stat().st_size >= MAX_CHUNK_BYTES:
        raise RuntimeError("Normalized audio chunk exceeds the upload size budget")
    try:
        with wave.open(str(path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, sample_rate):
                raise RuntimeError("Audio chunk must be mono PCM16 at the requested sample rate")
            actual_duration = audio.getnframes() / sample_rate
            if actual_duration <= 0 or abs(actual_duration - expected_duration) > max(0.5, expected_duration * 0.005):
                raise RuntimeError("Normalized audio chunk is truncated or has an unexpected duration")
    except (wave.Error, EOFError) as exc:
        raise RuntimeError("ffmpeg produced an invalid WAV chunk") from exc


@contextmanager
def iter_audio_chunks(
    audio_path: str, *, chunk_seconds: float = 120.0, overlap_seconds: float = 2.0,
    ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", timeout_seconds: float = 180.0,
    temp_dir: str | None = None, sample_rate: int = 16000,
) -> Iterator[Iterator[AudioChunk]]:
    """Yield a lazy iterator of normalized WAVs, keeping just one chunk on disk.

    ``chunk_seconds`` is the owned core duration; ``overlap_seconds`` adds context
    on each side. Even short inputs are normalized. The optional sample rate lets
    other audio analyses reuse bounded decoding; transcription should use 16000.
    All paths are passed as subprocess arguments, never interpreted by a shell.
    """
    chunk_seconds = _positive(chunk_seconds, "chunk_seconds")
    timeout_seconds = _positive(timeout_seconds, "timeout_seconds")
    try:
        overlap_seconds = float(overlap_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("overlap_seconds must be finite and nonnegative") from exc
    if not math.isfinite(overlap_seconds) or not 0 <= overlap_seconds < chunk_seconds:
        raise ValueError("overlap_seconds must be finite, nonnegative, and shorter than a core")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
        raise ValueError("sample_rate must be an integer between 8000 and 192000")
    if (chunk_seconds + 2 * overlap_seconds) * sample_rate * 2 + 4096 >= MAX_CHUNK_BYTES:
        raise ValueError("Chunk duration and sample rate exceed the PCM upload size budget")
    source = Path(audio_path).resolve(strict=True)
    duration = probe_audio_duration(str(source), ffprobe=ffprobe, timeout_seconds=timeout_seconds)
    with TemporaryDirectory(prefix="song-transcription-", dir=temp_dir) as directory:
        def generate() -> Iterator[AudioChunk]:
            for index in range(math.ceil(duration / chunk_seconds)):
                core_start = index * chunk_seconds
                core_end = min(duration, (index + 1) * chunk_seconds)
                start = max(0.0, core_start - overlap_seconds)
                end = min(duration, core_end + overlap_seconds)
                path = Path(directory) / f"chunk-{index:05d}.wav"
                chunk = AudioChunk(str(path), index, start, core_start, core_end, end, core_end == duration)
                try:
                    _run([
                        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{start:.9f}", "-i", str(source), "-t", f"{chunk.duration:.9f}",
                        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate),
                        "-c:a", "pcm_s16le", "-f", "wav", str(path),
                    ], timeout_seconds)
                    _verify_wav(path, chunk.duration, sample_rate)
                    yield chunk
                finally:
                    path.unlink(missing_ok=True)
        iterator = generate()
        try:
            yield iterator
        finally:
            iterator.close()


def _timestamped(item: dict, chunk: AudioChunk, field: str) -> dict | None:
    """Convert genuine chunk-relative timestamps; 0/0 placeholders are invalid."""
    text = str(item.get(field) or "").strip()
    if not text:
        return None
    if item.get("timing_estimated"):
        return None
    try:
        start, end = float(item["start"]), float(item["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if (not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end
            or start >= chunk.duration or end > chunk.duration + 0.5):
        return None
    end = min(end, chunk.duration)
    return {**item, field: text, "start": start + chunk.start, "end": end + chunk.start}


def _midpoint(item: dict) -> float:
    return (item["start"] + item["end"]) / 2


def _same_overlap(item: dict, other: dict, field: str) -> bool:
    """Catch timestamp jitter across a chunk boundary, never distant repetitions."""
    normalize = lambda text: re.sub(r"\W+", "", text.casefold())
    if not normalize(item[field]) or normalize(item[field]) != normalize(other[field]):
        return False
    intersection = min(item["end"], other["end"]) - max(item["start"], other["start"])
    shortest = min(item["end"] - item["start"], other["end"] - other["start"])
    return intersection + 1e-6 >= shortest * 0.5 and abs(_midpoint(item) - _midpoint(other)) <= 0.25 + 1e-6


def _resolve_word_ownership(entries: list[tuple[AudioChunk, dict]]) -> tuple[list[list[dict]], list[list[dict]]]:
    """Choose owned words first, then recover corroborated boundary omissions.

    Opposite timestamp jitter can put each observation in its neighbour's core.
    Recover only when adjacent chunks corroborate the same temporal word, both
    rejected it, and neither retained an equivalent event. Keep the earlier
    chunk's observed times rather than inventing an averaged timestamp.
    """
    all_words: list[list[dict]] = []
    retained: list[list[dict]] = [[] for _ in entries]
    selected: list[tuple[int, dict]] = []
    for position, (chunk, response) in enumerate(entries):
        words = sorted(
            [event for item in response.get("words") or [] if isinstance(item, dict)
             if (event := _timestamped(item, chunk, "word")) is not None],
            key=lambda word: (word["start"], word["end"]),
        )
        all_words.append(words)
        for word in words:
            if not chunk.owns(_midpoint(word)):
                continue
            if any(source != position and _same_overlap(word, previous, "word")
                   for source, previous in selected[-12:]):
                continue
            retained[position].append(word)
            selected.append((position, word))

    for position in range(len(entries) - 1):
        left, right = entries[position][0], entries[position + 1][0]
        if abs(left.core_end - right.core_start) > 1e-6:
            continue  # Missing chunks are not evidence for filling a gap.
        left_context = [word for word in all_words[position]
                        if not left.owns(_midpoint(word)) and right.owns(_midpoint(word))]
        right_context = [word for word in all_words[position + 1]
                         if not right.owns(_midpoint(word)) and left.owns(_midpoint(word))]
        for word in left_context:
            if not any(_same_overlap(word, other, "word") for other in right_context):
                continue
            nearby_retained = retained[position] + retained[position + 1]
            if any(_same_overlap(word, other, "word") for other in nearby_retained):
                continue
            retained[position].append({**word, "boundary_recovered": True})

    for words in retained:
        words.sort(key=lambda word: (word["start"], word["end"]))
    return all_words, retained


def _word_segment(words: list[dict]) -> dict:
    return {
        "text": " ".join(word["word"] for word in words),
        "start": min(word["start"] for word in words),
        "end": max(word["end"] for word in words),
    }


def merge_chunk_transcripts(results: Iterable[tuple[AudioChunk, dict]]) -> dict:
    """Merge local words/segments onto the source timeline by midpoint ownership.

    Repeated choruses are preserved. Text similarity alone never removes an event.
    If words cover a segment, its text/times use owned words and corroborated
    boundary recoveries. Recoveries preserve observed times and are marked on
    the word with ``boundary_recovered=True``.
    Untimed fallback text remains explicitly estimated and gets an honest warning;
    timestamps alone cannot resolve overlap in text-only recognition responses.
    """
    entries = sorted(results, key=lambda pair: pair[0].core_start)
    merged_words: list[dict] = []
    merged_segments: list[dict] = []
    warnings: list[str] = []
    models: list[str] = []
    segment_sources: list[tuple[int, dict]] = []
    words_by_entry, retained_by_entry = _resolve_word_ownership(entries)

    for position, (chunk, response) in enumerate(entries):
        for warning in response.get("warnings") or []:
            if str(warning) not in warnings:
                warnings.append(str(warning))
        model = response.get("model")
        if model and str(model) not in models:
            models.append(str(model))
        raw_words = response.get("words") or []
        all_words = words_by_entry[position]
        owned_words = retained_by_entry[position]
        merged_words.extend(owned_words)

        chunk_segments: list[dict] = []
        assigned: set[int] = set()
        invalid_text: list[str] = []
        valid_segment_count = 0
        for item in response.get("segments") or []:
            if not isinstance(item, dict):
                continue
            segment = _timestamped(item, chunk, "text")
            if segment is None:
                if str(item.get("text") or "").strip():
                    invalid_text.append(str(item["text"]).strip())
                continue
            valid_segment_count += 1
            indices = [i for i, word in enumerate(owned_words)
                       if i not in assigned and segment["start"] <= _midpoint(word) < segment["end"]]
            if indices:
                chunk_segments.append(_word_segment([owned_words[i] for i in indices]))
                assigned.update(indices)
            elif not any(segment["start"] <= _midpoint(word) < segment["end"] for word in all_words):
                if chunk.owns(_midpoint(segment)):
                    chunk_segments.append(segment)

        # Missing/incomplete segment data must not discard reliable owned words.
        remaining: list[dict] = []
        for i, word in enumerate(owned_words):
            if i in assigned:
                continue
            if remaining and word["start"] - remaining[-1]["end"] > 1.0:
                chunk_segments.append(_word_segment(remaining))
                remaining = []
            remaining.append(word)
        if remaining:
            chunk_segments.append(_word_segment(remaining))

        if not all_words and not valid_segment_count:
            fallback = " ".join(invalid_text) or str(response.get("text") or "").strip()
            if fallback:
                chunk_segments.append({
                    "text": fallback, "start": chunk.core_start, "end": chunk.core_end,
                    "timing_estimated": True, "timestamp_source": "chunk_interval",
                })
                warnings.append(
                    f"Bloque {chunk.index + 1}: letra sin tiempos fiables; se conserva el intervalo "
                    "del bloque como estimación. El solape textual puede requerir revisión."
                )
        elif invalid_text or len(all_words) < len(raw_words):
            warnings.append(f"Bloque {chunk.index + 1}: se descartaron tiempos inválidos; se priorizaron eventos con tiempos fiables.")

        for segment in chunk_segments:
            if not segment.get("timing_estimated") and any(
                source != chunk.index and not previous.get("timing_estimated")
                and _same_overlap(segment, previous, "text")
                for source, previous in segment_sources[-12:]
            ):
                continue
            merged_segments.append(segment)
            segment_sources.append((chunk.index, segment))

    merged_words.sort(key=lambda word: (word["start"], word["end"]))
    merged_segments.sort(key=lambda segment: (segment["start"], segment["end"]))
    return {
        "text": "\n".join(segment["text"] for segment in merged_segments),
        "segments": merged_segments, "words": merged_words,
        "model": "|".join(models) or None, "warnings": list(dict.fromkeys(warnings)),
    }
