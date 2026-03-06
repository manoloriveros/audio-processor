"""
Audio Processor - Servicio de extraccion de letras y acordes.
Recibe un archivo de audio, transcribe la letra con Whisper (OpenAI)
y detecta los acordes con Librosa (HPSS + beat-sync + plantillas extendidas).
"""

import os
import tempfile
import logging
from difflib import SequenceMatcher

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
API_SECRET = os.getenv("API_SECRET", "change-me-in-production")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio-processor")

app = FastAPI(title="Audio Processor - Song Editor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Plantillas de acordes (mayor, menor, 7)
# 12 notas x 3 tipos = 36 acordes
# ---------------------------------------------------------------------------
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

ENHARMONIC_TO_FLAT = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}


def _build_template(root_idx: int, intervals: list[tuple[int, float]]) -> np.ndarray:
    t = np.zeros(12)
    for semitone, weight in intervals:
        t[(root_idx + semitone) % 12] = weight
    return t


def _to_flat(chord_name: str) -> str:
    """Convierte nombre con sostenido a bemol (A#m -> Bbm)."""
    for sharp, flat in ENHARMONIC_TO_FLAT.items():
        if chord_name.startswith(sharp):
            return flat + chord_name[len(sharp):]
    return chord_name


def _use_flats(key: str, key_type: str) -> bool:
    """Determina si la tonalidad usa bemoles."""
    flat_minor_roots = {2, 7, 0, 5}  # D, G, C, F
    flat_major_roots = {5, 10, 3, 8}  # F, Bb, Eb, Ab
    key_idx = NOTES.index(key) if key in NOTES else -1
    if key_type == "minor":
        return key_idx in flat_minor_roots
    return key_idx in flat_major_roots


def _build_diatonic_set(key: str, key_type: str) -> set[str]:
    """Construye el conjunto de acordes diatonicos para una tonalidad."""
    key_idx = NOTES.index(key) if key in NOTES else 0
    if key_type == "minor":
        intervals = [0, 2, 3, 5, 7, 8, 10]
        qualities = ["m", "", "", "m", "m", "", ""]
    else:
        intervals = [0, 2, 4, 5, 7, 9, 11]
        qualities = ["", "m", "m", "", "", "m", ""]

    diatonic = set()
    for iv, q in zip(intervals, qualities):
        note = NOTES[(key_idx + iv) % 12]
        diatonic.add(note + q)
        if iv == 7:  # V7 es comun en ambos modos
            diatonic.add(note + "7")
            if key_type == "minor":
                diatonic.add(note)  # V mayor en menor armonico
    return diatonic


CHORD_TEMPLATES: dict[str, np.ndarray] = {}
for _i, _note in enumerate(NOTES):
    # Mayor: 1 - 3M - 5J
    CHORD_TEMPLATES[_note] = _build_template(_i, [(0, 1.0), (4, 0.8), (7, 0.8)])
    # Menor: 1 - 3m - 5J
    CHORD_TEMPLATES[f"{_note}m"] = _build_template(_i, [(0, 1.0), (3, 0.8), (7, 0.8)])
    # Septima dominante: 1 - 3M - 5J - 7m
    CHORD_TEMPLATES[f"{_note}7"] = _build_template(_i, [(0, 1.0), (4, 0.7), (7, 0.7), (10, 0.5)])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    x_api_secret: str = Header(None),
):
    if x_api_secret != API_SECRET:
        raise HTTPException(status_code=401, detail="No autorizado")

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")

    allowed = {
        "audio/mpeg", "audio/wav", "audio/mp4", "audio/x-m4a",
        "audio/ogg", "audio/webm", "audio/mp3", "audio/wave",
        "audio/x-wav", "audio/aac", "audio/flac",
    }
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: {file.content_type}",
        )

    suffix = os.path.splitext(file.filename or "audio.mp3")[1] or ".mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="El archivo excede el limite de 25 MB")
        tmp.write(content)
        tmp.close()

        logger.info("Procesando archivo: %s (%.1f MB)", file.filename, len(content) / 1e6)

        lyrics_data = transcribe_with_whisper(tmp.name)
        chords_data = detect_chords(tmp.name)
        result = synchronize(lyrics_data, chords_data)

        logger.info(
            "Procesamiento completado: %d secciones, clave detectada: %s, %d acordes detectados",
            len(result["sections"]),
            result["detectedKey"],
            sum(len(line["chords"]) for s in result["sections"] for line in s["lines"]),
        )
        return result

    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Paso 1: Transcripcion con Whisper
# ---------------------------------------------------------------------------
WHISPER_PROMPT = (
    "Cancion cristiana catolica en espanol. "
    "Letra de alabanza, adoracion y musica liturgica. "
    "La cancion tiene versos y coros que se repiten varias veces. "
    "Transcribir todas las repeticiones completas. "
    "Señor, Dios, Jesús, Cristo, Espíritu Santo, María, aleluya, amén, "
    "cordero, gloria, bendito, misericordia, alabanza, adoración."
)


def transcribe_with_whisper(audio_path: str) -> dict:
    """Transcribe audio usando Whisper con prompt contextual y timestamps."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language="es",
            prompt=WHISPER_PROMPT,
        )

    # Extraer texto completo
    full_text = ""
    if hasattr(response, "text"):
        full_text = response.text
    elif isinstance(response, dict):
        full_text = response.get("text", "")

    # Extraer segmentos con timestamps
    raw_segments = None
    if hasattr(response, "segments"):
        raw_segments = response.segments
    elif isinstance(response, dict) and "segments" in response:
        raw_segments = response["segments"]

    segments = []
    if raw_segments:
        for seg in raw_segments:
            text = seg["text"] if isinstance(seg, dict) else getattr(seg, "text", "")
            start = seg["start"] if isinstance(seg, dict) else getattr(seg, "start", 0)
            end = seg["end"] if isinstance(seg, dict) else getattr(seg, "end", 0)
            text = text.strip()
            if text:
                segments.append({"text": text, "start": start, "end": end})

    if not segments and full_text:
        logger.warning("No se recibieron segmentos con timestamps, usando division por frases")
        sentences = [s.strip() for s in full_text.replace(".", ".\n").split("\n") if s.strip()]
        for i, sentence in enumerate(sentences):
            segments.append({"text": sentence, "start": 0, "end": 0})

    return {"text": full_text, "segments": segments}


# ---------------------------------------------------------------------------
# Paso 2: Deteccion de acordes con Librosa (mejorada)
# ---------------------------------------------------------------------------
def detect_chords(audio_path: str) -> list[dict]:
    """
    Detecta acordes usando:
    1. HPSS (separacion armonica/percusiva) para aislar contenido tonal
    2. Cromagrama CENS sobre la componente armonica (robusto para acordes)
    3. Analisis sincronizado por beats (un acorde por beat)
    4. Filtro de mediana para suavizar transiciones espurias
    5. Dos pasadas: primera detecta tonalidad, segunda aplica sesgo diatonico
    """
    import librosa
    from scipy.ndimage import median_filter

    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Separar componente armonica (quita percusion)
    y_harmonic, _ = librosa.effects.hpss(y)

    # Detectar beats para sincronizar analisis
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    hop_length = 512

    if len(beat_frames) < 2:
        hop_length = 2048
        chroma = librosa.feature.chroma_cens(y=y_harmonic, sr=sr, hop_length=hop_length)
        times = librosa.frames_to_time(range(chroma.shape[1]), sr=sr, hop_length=hop_length)
    else:
        chroma_full = librosa.feature.chroma_cens(y=y_harmonic, sr=sr, hop_length=hop_length)
        chroma = librosa.util.sync(chroma_full, beat_frames, aggregate=np.median)
        times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    # Filtro de mediana temporal
    if chroma.shape[1] > 5:
        chroma = median_filter(chroma, size=(1, 5))

    # Normalizar cada frame
    norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-10
    chroma_norm = chroma / norms

    # Pre-normalizar plantillas
    template_names = list(CHORD_TEMPLATES.keys())
    template_matrix = np.array([
        CHORD_TEMPLATES[name] / (np.linalg.norm(CHORD_TEMPLATES[name]) + 1e-10)
        for name in template_names
    ])

    BASIC_CHORDS = {n for n in template_names if not n.endswith("7")}
    EXTENDED_BONUS = 0.15
    scores = template_matrix @ chroma_norm

    def _classify_frame(frame_idx: int, diatonic_bonus: dict[str, float] | None = None):
        frame_scores = scores[:, frame_idx].copy()
        if diatonic_bonus:
            for j, name in enumerate(template_names):
                frame_scores[j] += diatonic_bonus.get(name, 0.0)
        if np.max(frame_scores) < 0.55:
            return None
        best_basic_score, best_basic = -1.0, None
        best_ext_score, best_ext = -1.0, None
        for j, name in enumerate(template_names):
            s = frame_scores[j]
            if name in BASIC_CHORDS:
                if s > best_basic_score:
                    best_basic_score, best_basic = s, name
            else:
                if s > best_ext_score:
                    best_ext_score, best_ext = s, name
        if best_ext and best_ext_score > best_basic_score + EXTENDED_BONUS:
            return best_ext
        return best_basic

    # --- Pasada 1: clasificacion sin sesgo -> detectar tonalidad ---
    raw_pass1 = [_classify_frame(i) for i in range(chroma.shape[1])]
    names_pass1 = [c for c in raw_pass1 if c is not None]
    det_key, det_type = _detect_key(names_pass1)
    diatonic = _build_diatonic_set(det_key, det_type)
    logger.info("Tonalidad detectada (pasada 1): %s %s, diatonicos: %s", det_key, det_type, diatonic)

    # --- Pasada 2: clasificacion con sesgo diatonico ---
    bonus = {name: 0.08 for name in diatonic}
    raw_chords: list[str | None] = [_classify_frame(i, bonus) for i in range(chroma.shape[1])]

    # Filtro de mediana sobre la secuencia
    chord_to_idx = {name: idx for idx, name in enumerate(template_names)}
    chord_to_idx[None] = -1
    idx_sequence = np.array([chord_to_idx.get(c, -1) for c in raw_chords])

    if len(idx_sequence) > 5:
        filtered = median_filter(idx_sequence.astype(float), size=5).astype(int)
        idx_to_chord = {idx: name for name, idx in chord_to_idx.items()}
        raw_chords = [idx_to_chord.get(int(idx)) for idx in filtered]

    # Generar eventos: solo cuando el acorde CAMBIA, duracion minima 2.0s
    chord_events: list[dict] = []
    current_chord: str | None = None
    current_start = 0.0
    min_duration = 2.0

    for i, chord in enumerate(raw_chords):
        if chord != current_chord:
            if current_chord is not None and i < len(times):
                duration = times[i] - current_start
                if duration >= min_duration:
                    chord_events.append({"chord": current_chord, "time": round(current_start, 2)})
            current_chord = chord
            current_start = times[i] if i < len(times) else current_start

    if current_chord is not None:
        chord_events.append({"chord": current_chord, "time": round(current_start, 2)})

    logger.info("Acordes detectados: %d eventos", len(chord_events))
    return chord_events


# ---------------------------------------------------------------------------
# Paso 3: Sincronizacion letras + acordes
# ---------------------------------------------------------------------------
def _split_long_segments(segments: list[dict], max_len: int = 40, min_len: int = 15) -> list[dict]:
    """Divide segmentos largos en lineas mas cortas, respetando frases naturales."""
    result = []
    for seg in segments:
        text = seg["text"].strip()
        if len(text) <= max_len:
            result.append(seg)
            continue

        duration = seg["end"] - seg["start"]

        # Buscar puntos de corte naturales: comas, puntos, punto y coma
        split_chars = {",", ".", ";", "?", "!"}
        candidates: list[int] = []
        for idx, ch in enumerate(text):
            if ch in split_chars and idx > 0:
                candidates.append(idx + 1)  # incluir el signo de puntuacion

        # Generar partes cortando en los puntos naturales
        parts: list[str] = []
        start_idx = 0
        for cut in candidates:
            part = text[start_idx:cut].strip()
            rest = text[cut:].strip()
            # Solo cortar si ambos lados quedan con longitud razonable
            if len(part) >= min_len and len(rest) >= min_len:
                parts.append(part)
                start_idx = cut

        # Agregar lo que quede
        remaining = text[start_idx:].strip()
        if remaining:
            # Si lo que queda es muy largo, cortar por palabras
            if len(remaining) > max_len:
                words = remaining.split()
                current = ""
                for w in words:
                    test = (current + " " + w).strip()
                    if len(test) > max_len and len(current) >= min_len:
                        parts.append(current)
                        current = w
                    else:
                        current = test
                if current:
                    # No dejar fragmentos muy cortos solos
                    if len(current) < min_len and parts:
                        parts[-1] = parts[-1] + " " + current
                    else:
                        parts.append(current)
            else:
                # Si es corto pero hay partes previas y es muy pequeno, pegarlo a la anterior
                if len(remaining) < min_len and parts:
                    parts[-1] = parts[-1] + " " + remaining
                else:
                    parts.append(remaining)

        if len(parts) <= 1:
            result.append(seg)
            continue

        # Distribuir tiempos proporcionalmente
        total_chars = sum(len(p) for p in parts)
        time_cursor = seg["start"]
        for part in parts:
            part_dur = duration * (len(part) / max(total_chars, 1))
            result.append({
                "text": part.strip(),
                "start": round(time_cursor, 3),
                "end": round(time_cursor + part_dur, 3),
            })
            time_cursor += part_dur

    return result


def synchronize(lyrics_data: dict, chords_data: list[dict]) -> dict:
    """Cruza letras (con tiempos) y acordes (con tiempos) en secciones estructuradas."""
    segments = lyrics_data["segments"]
    if not segments:
        return {"sections": [], "detectedKey": "C", "keyType": "major"}

    # Paso A: Dividir segmentos largos respetando frases naturales
    segments = _split_long_segments(segments)

    # Paso B: Asignar acordes a cada linea
    sections: list[dict] = []
    current_lines: list[dict] = []
    section_counter = 1

    for i, seg in enumerate(segments):
        seg_chords: list[dict] = []

        # Acorde activo al inicio: el ultimo acorde que suena ANTES o AL INICIO de esta linea
        last_before = None
        for chord_ev in chords_data:
            if chord_ev["time"] <= seg["start"]:
                last_before = chord_ev
            else:
                break
        if last_before:
            seg_chords.append({"chord": last_before["chord"], "charIndex": 0})

        # Acordes que caen DENTRO del rango temporal de esta linea
        for chord_ev in chords_data:
            if seg["start"] < chord_ev["time"] < seg["end"]:
                seg_duration = max(seg["end"] - seg["start"], 0.01)
                relative_pos = (chord_ev["time"] - seg["start"]) / seg_duration
                char_index = int(relative_pos * len(seg["text"]))
                char_index = max(1, min(char_index, len(seg["text"]) - 1))
                seg_chords.append({"chord": chord_ev["chord"], "charIndex": char_index})

        # Acordes entre el fin de esta linea y el inicio de la siguiente (intermedios)
        next_start = segments[i + 1]["start"] if i < len(segments) - 1 else seg["end"] + 999
        for chord_ev in chords_data:
            if seg["end"] <= chord_ev["time"] < next_start:
                seg_chords.append({"chord": chord_ev["chord"], "charIndex": len(seg["text"])})

        # Deduplicar: quitar acordes repetidos consecutivos y misma posicion
        seen_positions: set[int] = set()
        unique_chords: list[dict] = []
        prev_chord_name: str | None = None
        for c in seg_chords:
            if c["chord"] == prev_chord_name:
                continue
            if c["charIndex"] not in seen_positions:
                seen_positions.add(c["charIndex"])
                unique_chords.append(c)
                prev_chord_name = c["chord"]

        # Limitar a maximo 3 acordes por linea
        if len(unique_chords) > 3:
            first = unique_chords[0]
            last = unique_chords[-1]
            mid = unique_chords[len(unique_chords) // 2]
            unique_chords = [first, mid, last]

        current_lines.append({
            "lyrics": seg["text"],
            "chords": unique_chords,
            "timestamps": [],
        })

        # Detectar cambio de seccion por pausa larga (> 2.5 segundos)
        if i < len(segments) - 1:
            pause = segments[i + 1]["start"] - seg["end"]
            if pause > 2.5:
                sections.append({
                    "name": f"Verso {section_counter}",
                    "lines": current_lines,
                })
                current_lines = []
                section_counter += 1

    if current_lines:
        sections.append({"name": f"Verso {section_counter}", "lines": current_lines})

    _detect_choruses(sections)

    all_chords = [c["chord"] for c in chords_data if c.get("chord")]
    detected_key, key_type = _detect_key(all_chords)

    # Renombrar enarmonicos: A# -> Bb, D# -> Eb, etc. segun tonalidad
    if _use_flats(detected_key, key_type):
        detected_key = _to_flat(detected_key)
        for section in sections:
            for line in section["lines"]:
                for chord in line["chords"]:
                    chord["chord"] = _to_flat(chord["chord"])

    return {"sections": sections, "detectedKey": detected_key, "keyType": key_type}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _detect_choruses(sections: list[dict]) -> None:
    """Renombra secciones repetidas como 'Coro'."""
    if len(sections) < 2:
        return

    texts = [
        " ".join(line["lyrics"].lower().strip() for line in s["lines"])
        for s in sections
    ]

    chorus_indices: set[int] = set()
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if SequenceMatcher(None, texts[i], texts[j]).ratio() > 0.65:
                chorus_indices.add(i)
                chorus_indices.add(j)

    verse_n = 1
    chorus_n = 1
    assigned_chorus_text: dict[str, int] = {}

    for i, section in enumerate(sections):
        if i in chorus_indices:
            matched = None
            for ref_text, ref_num in assigned_chorus_text.items():
                if SequenceMatcher(None, texts[i], ref_text).ratio() > 0.65:
                    matched = ref_num
                    break
            if matched is not None:
                section["name"] = "Coro" if matched == 1 else f"Coro {matched}"
            else:
                assigned_chorus_text[texts[i]] = chorus_n
                section["name"] = "Coro" if chorus_n == 1 else f"Coro {chorus_n}"
                chorus_n += 1
        else:
            section["name"] = f"Verso {verse_n}"
            verse_n += 1


def _detect_key(chord_names: list[str]) -> tuple[str, str]:
    """Detecta la tonalidad mas probable a partir de la frecuencia de acordes."""
    if not chord_names:
        return "C", "major"

    # Normalizar: "Am7" -> "Am", "G7" -> "G", "Cmaj7" -> "C", etc.
    freq: dict[str, int] = {}
    for name in chord_names:
        # Extraer solo root + m/dim para el analisis de tonalidad
        base = name
        for suffix in ("maj7", "m7", "7", "sus4", "sus2", "dim"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                if suffix in ("m7",):
                    base += "m"
                elif suffix == "dim":
                    base += "dim"
                break
        if not base:
            base = name
        freq[base] = freq.get(base, 0) + 1

    major_intervals = [0, 2, 4, 5, 7, 9, 11]
    minor_intervals = [0, 2, 3, 5, 7, 8, 10]
    major_qualities = ["", "m", "m", "", "", "m", "dim"]
    minor_qualities = ["m", "dim", "", "m", "m", "", ""]

    best_key = "C"
    best_type = "major"
    best_score = -1

    for i, note in enumerate(NOTES):
        diatonic = [NOTES[(i + iv) % 12] + major_qualities[j] for j, iv in enumerate(major_intervals)]
        score = sum(freq.get(c, 0) for c in diatonic)
        if score > best_score:
            best_score = score
            best_key = note
            best_type = "major"

        diatonic = [NOTES[(i + iv) % 12] + minor_qualities[j] for j, iv in enumerate(minor_intervals)]
        score = sum(freq.get(c, 0) for c in diatonic)
        if score > best_score:
            best_score = score
            best_key = note
            best_type = "minor"

    return best_key, best_type


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
