"""
Motor Music.ai (plataforma API de Moises) para extraer letra, acordes,
secciones y beats de una cancion con calidad profesional.

Flujo:
  1. Subir el audio a Music.ai (URL firmada temporal).
  2. Crear un job con el workflow configurado (MUSIC_AI_WORKFLOW).
  3. Esperar el resultado y descargar el JSON de cada modulo.
  4. Combinar letra + acordes + secciones + beats al formato del SongEditor.
  5. Pasada opcional de estructuracion con LLM (ortografia litugica y
     nombres canonicos de seccion), sin alterar el numero de lineas.

Variables de entorno:
  MUSIC_AI_API_KEY        API key (obligatoria para activar este motor)
  MUSIC_AI_WORKFLOW       slug del workflow (default: songlory-transcription)
  MUSIC_AI_JOB_TIMEOUT    segundos maximos de espera del job (default: 150)
  MUSIC_AI_POLL_INTERVAL  segundos entre sondeos de estado (default: 3)
  OPENAI_STRUCTURE_MODEL  modelo de la pasada de estructura (default: gpt-4o-mini);
                          usa OPENAI_API_KEY, si falta se omite la pasada.

El modulo es tolerante a fallos: cualquier excepcion la captura main.py,
que recurre al pipeline local (Whisper + Chordino) como fallback.
"""

import json
import logging
import os
import time

import requests

import structuring
from structuring import SECTION_LABEL_MAP

logger = logging.getLogger("audio-processor.musicai")

API_BASE = "https://api.music.ai/v1"

# Campos donde los modulos de Music.ai pueden reportar la etiqueta del acorde
_CHORD_FIELDS = (
    "chord_majmin", "chord_complex_pop", "chord_basic_pop",
    "chord_complex_jazz", "chord_basic_jazz", "chord", "name", "label", "value",
)


def _main():
    """Import perezoso de main para reutilizar helpers sin import circular."""
    import main
    return main


def _env(name: str, default=None):
    value = os.getenv(name)
    return value if value not in (None, "") else default


def is_configured() -> bool:
    return bool(_env("MUSIC_AI_API_KEY"))


def _headers() -> dict:
    return {"Authorization": _env("MUSIC_AI_API_KEY", "")}


# ---------------------------------------------------------------------------
# Cliente HTTP: upload + job + resultados
# ---------------------------------------------------------------------------
def _upload(audio_path: str) -> str:
    resp = requests.get(f"{API_BASE}/upload", headers=_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    with open(audio_path, "rb") as fh:
        put = requests.put(
            data["uploadUrl"],
            data=fh,
            headers={"Content-Type": "application/octet-stream"},
            timeout=180,
        )
    put.raise_for_status()
    return data["downloadUrl"]


def _create_job(input_url: str) -> str:
    workflow = _env("MUSIC_AI_WORKFLOW", "songlory-transcription")
    resp = requests.post(
        f"{API_BASE}/job",
        headers={**_headers(), "Content-Type": "application/json"},
        json={
            "name": "songlory-audio-import",
            "workflow": workflow,
            "params": {"inputUrl": input_url},
        },
        timeout=30,
    )
    if resp.status_code == 404:
        raise RuntimeError(
            f"El workflow '{workflow}' no existe en Music.ai; revisa MUSIC_AI_WORKFLOW"
        )
    resp.raise_for_status()
    return resp.json()["id"]


def _wait_job(job_id: str) -> dict:
    timeout_s = float(_env("MUSIC_AI_JOB_TIMEOUT", "150"))
    poll = max(1.0, float(_env("MUSIC_AI_POLL_INTERVAL", "3")))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = requests.get(f"{API_BASE}/job/{job_id}", headers=_headers(), timeout=30)
        resp.raise_for_status()
        job = resp.json()
        status = job.get("status")
        if status == "SUCCEEDED":
            return job.get("result") or {}
        if status == "FAILED":
            err = job.get("error") or {}
            raise RuntimeError(
                f"Job de Music.ai fallido: {err.get('code')}: {err.get('message')}"
            )
        time.sleep(poll)
    raise TimeoutError(f"El job de Music.ai no termino en {timeout_s:.0f}s")


def _fetch_payload(value):
    """Descarga el resultado de un modulo (URL a JSON) o lo devuelve inline."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value.startswith("http"):
        resp = requests.get(value, timeout=60)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return resp.text
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _pick(result: dict, *keywords):
    """Encuentra la salida del workflow cuyo nombre contiene alguna palabra clave."""
    for key, value in result.items():
        lowered = key.lower()
        if any(word in lowered for word in keywords):
            return value
    return None


# ---------------------------------------------------------------------------
# Parsers defensivos de los modulos
# ---------------------------------------------------------------------------
def _as_items(payload, *container_keys):
    if isinstance(payload, dict):
        for key in container_keys:
            if isinstance(payload.get(key), list):
                return payload[key]
    return payload if isinstance(payload, list) else []


def _parse_chords(payload) -> list[dict]:
    main = _main()
    items = _as_items(payload, "chords", "data", "annotations", "events", "progression")
    events: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = None
        for field in _CHORD_FIELDS:
            if item.get(field):
                label = str(item[field])
                break
        start = item.get("start", item.get("time", item.get("timestamp")))
        if label is None or start is None:
            continue
        chord = main._normalize_chord_label(label)
        if chord is None:  # "N" / silencio armonico
            continue
        events.append({"chord": chord, "time": round(float(start), 2)})

    events.sort(key=lambda e: e["time"])
    merged: list[dict] = []
    for ev in events:
        if merged and merged[-1]["chord"] == ev["chord"]:
            continue
        merged.append(ev)
    return merged


def _words_to_segments(words: list[dict], max_gap: float = 1.0) -> list[dict]:
    segments: list[dict] = []
    current: list[dict] = []
    for w in words:
        if current and (w["start"] - current[-1]["end"]) > max_gap:
            segments.append({
                "text": " ".join(x["word"] for x in current),
                "start": current[0]["start"],
                "end": current[-1]["end"],
            })
            current = []
        current.append(w)
    if current:
        segments.append({
            "text": " ".join(x["word"] for x in current),
            "start": current[0]["start"],
            "end": current[-1]["end"],
        })
    return segments


def _parse_lyrics(payload) -> dict:
    items = _as_items(payload, "lines", "segments", "lyrics", "data", "transcription", "words")
    segments: list[dict] = []
    words: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        end = item.get("end")
        word = item.get("word")
        text = (item.get("text") or item.get("line") or item.get("value") or "").strip()
        if word and not text:
            if start is None or end is None:
                continue
            words.append({"word": str(word).strip(), "start": float(start), "end": float(end)})
            continue
        if not text or start is None or end is None:
            continue
        segments.append({"text": text, "start": float(start), "end": float(end)})
        for w in item.get("words") or []:
            if isinstance(w, dict) and w.get("word") is not None:
                try:
                    words.append({
                        "word": str(w["word"]).strip(),
                        "start": float(w.get("start", start)),
                        "end": float(w.get("end", end)),
                    })
                except (TypeError, ValueError):
                    continue

    words.sort(key=lambda w: w["start"])
    if not segments and words:
        segments = _words_to_segments(words)
    segments.sort(key=lambda s: s["start"])
    return {"segments": segments, "words": words}


def _parse_sections(payload) -> list[dict]:
    items = _as_items(payload, "sections", "segments", "data", "annotations")
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(
            item.get("label") or item.get("section") or item.get("name") or item.get("value") or ""
        ).strip().lower()
        start = item.get("start")
        end = item.get("end")
        if start is None:
            continue
        out.append({
            "label": label,
            "start": float(start),
            "end": float(end) if end is not None else float(start),
        })
    out.sort(key=lambda s: s["start"])
    return out


def _parse_beats(payload) -> list[float]:
    items = _as_items(payload, "beats", "data", "annotations", "events")
    times: list[float] = []
    for item in items:
        if isinstance(item, dict):
            t = item.get("time", item.get("start", item.get("timestamp")))
            if t is None:
                continue
            times.append(float(t))
        else:
            try:
                times.append(float(item))
            except (TypeError, ValueError):
                continue
    times.sort()
    return times


# ---------------------------------------------------------------------------
# Post-proceso musical
# ---------------------------------------------------------------------------
def _snap_to_beats(events: list[dict], beats: list[float]) -> list[dict]:
    """Ajusta cada cambio de acorde al beat mas cercano (si esta cerca)."""
    if len(beats) < 4 or not events:
        return events
    gaps = [b - a for a, b in zip(beats, beats[1:]) if b > a]
    if not gaps:
        return events
    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    tol = 0.35 * median_gap

    snapped: list[dict] = []
    for ev in events:
        t = ev["time"]
        # busqueda binaria simple del beat mas cercano
        lo, hi = 0, len(beats) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if beats[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        best = beats[lo]
        if lo > 0 and abs(beats[lo - 1] - t) < abs(best - t):
            best = beats[lo - 1]
        if abs(best - t) <= tol:
            t = best
        snapped.append({"chord": ev["chord"], "time": round(float(t), 2)})

    snapped.sort(key=lambda e: e["time"])
    cleaned: list[dict] = []
    for ev in snapped:
        if cleaned and cleaned[-1]["chord"] == ev["chord"]:
            continue
        if cleaned and abs(cleaned[-1]["time"] - ev["time"]) < 1e-6:
            cleaned[-1] = ev  # dos acordes en el mismo beat: gana el ultimo
            continue
        cleaned.append(ev)
    return cleaned


def _group_segments(segments: list[dict], sections: list[dict]):
    """Agrupa lineas de letra dentro de las secciones temporales de Music.ai."""
    if not sections:
        return None
    groups = [
        {"label": sec["label"], "start": sec["start"], "end": sec["end"], "segments": []}
        for sec in sections
    ]
    for seg in segments:
        mid = (seg["start"] + seg["end"]) / 2
        target = None
        for g in groups:
            if g["start"] - 0.25 <= mid < g["end"] + 0.25:
                target = g
                break
        if target is None:
            target = min(groups, key=lambda g: min(abs(mid - g["start"]), abs(mid - g["end"])))
        target["segments"].append(seg)
    groups = [g for g in groups if g["segments"]]
    for g in groups:
        g["segments"].sort(key=lambda s: s["start"])
    return groups or None


def _group_by_pauses(segments: list[dict], pause: float = 2.5):
    """Fallback sin modulo de secciones: corta por pausas largas entre lineas."""
    groups = []
    current: list[dict] = []
    for i, seg in enumerate(segments):
        current.append(seg)
        is_last = i == len(segments) - 1
        if not is_last and (segments[i + 1]["start"] - seg["end"]) > pause:
            groups.append({"label": "", "segments": current})
            current = []
    if current:
        groups.append({"label": "", "segments": current})
    return groups


def _build_sections(groups, chords: list[dict], words: list[dict]) -> list[dict]:
    """Asigna acordes a cada linea y construye las secciones del SongEditor."""
    main = _main()
    sections_out: list[dict] = []
    last_line_chord: str | None = None

    all_segments = [seg for g in groups for seg in g["segments"]]
    seg_next_start: dict[int, float] = {}
    for idx, seg in enumerate(all_segments):
        seg_next_start[id(seg)] = (
            all_segments[idx + 1]["start"] if idx + 1 < len(all_segments) else seg["end"] + 999.0
        )

    for g in groups:
        lines: list[dict] = []
        for j, seg in enumerate(g["segments"]):
            seg_chords: list[dict] = []

            # Acorde activo al inicio de la linea
            active = None
            for ev in chords:
                if ev["time"] <= seg["start"] + 0.05:
                    active = ev
                else:
                    break
            if active and (j == 0 or active["chord"] != last_line_chord):
                seg_chords.append({"chord": active["chord"], "charIndex": 0, "_time": active["time"]})

            # Acordes dentro de la linea
            for ev in chords:
                if seg["start"] + 0.05 < ev["time"] < seg["end"]:
                    ci = main._time_to_char_index(
                        ev["time"], seg["text"], seg["start"], seg["end"], words,
                    )
                    seg_chords.append({"chord": ev["chord"], "charIndex": ci, "_time": ev["time"]})

            # Acordes entre el final de la linea y la siguiente, acotados al
            # final de la seccion: los acordes de un interludio pertenecen a la
            # seccion siguiente (apareceran alli como acorde activo inicial).
            next_start = seg_next_start[id(seg)]
            boundary = min(next_start, g.get("end", float("inf")) + 0.25)
            for ev in chords:
                if seg["end"] <= ev["time"] < boundary:
                    seg_chords.append({
                        "chord": ev["chord"],
                        "charIndex": len(seg["text"]),
                        "_time": ev["time"],
                    })

            # Deduplicar y espaciar
            seg_chords.sort(key=lambda c: (c.get("_time", 0), c["charIndex"]))
            unique: list[dict] = []
            for c in seg_chords:
                ci = max(0, int(c["charIndex"]))
                if unique and c["chord"] == unique[-1]["chord"] and abs(ci - unique[-1]["charIndex"]) <= 1:
                    continue
                unique.append({"chord": c["chord"], "charIndex": ci})
            unique = structuring.respace(unique)

            if unique:
                last_line_chord = unique[-1]["chord"]
            lines.append({
                "lyrics": seg["text"],
                "chords": unique,
                "timestamps": [],
                "_startTime": round(float(seg["start"]), 3),
            })

        if lines:
            sections_out.append({"label": g.get("label", ""), "lines": lines})

    # Nombres de seccion a partir de las etiquetas de Music.ai
    verse_n = 0
    chorus_seen = 0
    labeled = 0
    for section in sections_out:
        base = SECTION_LABEL_MAP.get(section.pop("label", ""), None)
        if base == "Verso":
            verse_n += 1
            section["name"] = f"Verso {verse_n}"
            labeled += 1
        elif base == "Coro":
            chorus_seen += 1
            section["name"] = "Coro"
            labeled += 1
        elif base:
            section["name"] = base
            labeled += 1
        else:
            verse_n += 1
            section["name"] = f"Verso {verse_n}"

    # Sin etiquetas utiles: aplicar deteccion de coros por similitud (legacy)
    if labeled == 0 and len(sections_out) > 1:
        main._detect_choruses(sections_out)

    return sections_out


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------
def process(audio_path: str) -> dict:
    main = _main()
    t0 = time.monotonic()

    download_url = _upload(audio_path)
    job_id = _create_job(download_url)
    logger.info("Job Music.ai creado: %s", job_id)
    result = _wait_job(job_id)
    logger.info(
        "Job Music.ai completado en %.1fs; salidas: %s",
        time.monotonic() - t0, sorted(result.keys()),
    )

    chords_raw = _pick(result, "chord", "acorde")
    lyrics_raw = _pick(result, "lyric", "letra", "transcript", "text", "word")
    sections_raw = _pick(result, "section", "seccion", "structure")
    beats_raw = _pick(result, "beat", "bpm")

    if lyrics_raw is None and chords_raw is None:
        raise RuntimeError(
            f"El workflow no devolvio salidas reconocibles; claves: {sorted(result.keys())}"
        )

    chords = _parse_chords(_fetch_payload(chords_raw)) if chords_raw is not None else []
    lyrics = _parse_lyrics(_fetch_payload(lyrics_raw)) if lyrics_raw is not None else {"segments": [], "words": []}
    sections_t = _parse_sections(_fetch_payload(sections_raw)) if sections_raw is not None else []
    beats = _parse_beats(_fetch_payload(beats_raw)) if beats_raw is not None else []

    logger.info(
        "Music.ai: %d lineas, %d palabras, %d acordes, %d secciones, %d beats",
        len(lyrics["segments"]), len(lyrics["words"]), len(chords), len(sections_t), len(beats),
    )

    if not lyrics["segments"]:
        raise RuntimeError("Music.ai no devolvio letra utilizable")

    chords = _snap_to_beats(chords, beats)

    segments = main._split_long_segments(lyrics["segments"])
    groups = _group_segments(segments, sections_t) or _group_by_pauses(segments)
    sections_out = _build_sections(groups, chords, lyrics["words"])

    all_chords = [c["chord"] for c in chords]
    detected_key, key_type = main._detect_key(all_chords) if all_chords else ("C", "major")

    sections_out = structuring.apply_structure(sections_out, detected_key, key_type)

    # Enarmonia coherente con la tonalidad (A# -> Bb, etc.)
    if main._use_flats(detected_key, key_type):
        detected_key = main._to_flat(detected_key)
        for section in sections_out:
            for line in section["lines"]:
                for chord in line["chords"]:
                    chord["chord"] = main._to_flat(chord["chord"])

    return {
        "sections": sections_out,
        "detectedKey": detected_key,
        "keyType": key_type,
        "engine": "music.ai",
        "transcriptionModel": "music.ai",
    }
