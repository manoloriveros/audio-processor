"""
Audio Processor - Servicio de extraccion de letras y acordes.
Recibe un archivo de audio, transcribe la letra con Whisper (OpenAI)
y detecta los acordes con Librosa.
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
# Plantillas de acordes para deteccion (12 notas x mayor/menor = 24 acordes)
# ---------------------------------------------------------------------------
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORD_TEMPLATES: dict[str, np.ndarray] = {}
for _i, _note in enumerate(NOTES):
    # Acorde mayor: root + 3ra mayor (4 semitonos) + 5ta justa (7 semitonos)
    _major = np.zeros(12)
    _major[_i % 12] = 1.0
    _major[(_i + 4) % 12] = 0.8
    _major[(_i + 7) % 12] = 0.8
    CHORD_TEMPLATES[_note] = _major

    # Acorde menor: root + 3ra menor (3 semitonos) + 5ta justa (7 semitonos)
    _minor = np.zeros(12)
    _minor[_i % 12] = 1.0
    _minor[(_i + 3) % 12] = 0.8
    _minor[(_i + 7) % 12] = 0.8
    CHORD_TEMPLATES[f"{_note}m"] = _minor

    # Acorde de septima: root + 3ra mayor + 5ta justa + 7ma menor (10 semitonos)
    _dom7 = np.zeros(12)
    _dom7[_i % 12] = 1.0
    _dom7[(_i + 4) % 12] = 0.7
    _dom7[(_i + 7) % 12] = 0.7
    _dom7[(_i + 10) % 12] = 0.5
    CHORD_TEMPLATES[f"{_note}7"] = _dom7


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
    # Validar secreto interno
    if x_api_secret != API_SECRET:
        raise HTTPException(status_code=401, detail="No autorizado")

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")

    # Validar tipo de archivo
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

    # Guardar archivo temporal
    suffix = os.path.splitext(file.filename or "audio.mp3")[1] or ".mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="El archivo excede el limite de 25 MB")
        tmp.write(content)
        tmp.close()

        logger.info("Procesando archivo: %s (%.1f MB)", file.filename, len(content) / 1e6)

        # Paso 1: Transcribir con Whisper API
        lyrics_data = transcribe_with_whisper(tmp.name)

        # Paso 2: Detectar acordes con Librosa
        chords_data = detect_chords(tmp.name)

        # Paso 3: Sincronizar letras + acordes y estructurar en secciones
        result = synchronize(lyrics_data, chords_data)

        logger.info(
            "Procesamiento completado: %d secciones, clave detectada: %s",
            len(result["sections"]),
            result["detectedKey"],
        )
        return result

    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Paso 1: Transcripcion con OpenAI Audio API
# ---------------------------------------------------------------------------
def transcribe_with_whisper(audio_path: str) -> dict:
    """Transcribe audio usando la API de OpenAI (gpt-4o-mini-transcribe) con marcas de tiempo."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
            response_format="json",
            timestamp_granularities=["segment"],
            language="es",
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

    # Fallback: si el modelo no devolvio segmentos, dividir el texto en frases
    if not segments and full_text:
        logger.warning("No se recibieron segmentos con timestamps, usando division por frases")
        sentences = [s.strip() for s in full_text.replace(".", ".\n").split("\n") if s.strip()]
        for i, sentence in enumerate(sentences):
            segments.append({"text": sentence, "start": 0, "end": 0})

    return {"text": full_text, "segments": segments}


# ---------------------------------------------------------------------------
# Paso 2: Deteccion de acordes con Librosa
# ---------------------------------------------------------------------------
def detect_chords(audio_path: str) -> list[dict]:
    """Detecta acordes usando analisis de cromagrama (Librosa)."""
    import librosa

    # Cargar audio (mono, 22050 Hz)
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Cromagrama CQT — buena resolucion tonal
    hop_length = 2048
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(range(chroma.shape[1]), sr=sr, hop_length=hop_length)

    # Clasificar cada frame al acorde mas probable
    raw_chords: list[str | None] = []
    for i in range(chroma.shape[1]):
        frame = chroma[:, i]
        if np.max(frame) < 0.15:
            raw_chords.append(None)
            continue

        best_chord = None
        best_score = -1.0
        frame_norm = frame / (np.linalg.norm(frame) + 1e-10)
        for chord_name, template in CHORD_TEMPLATES.items():
            score = float(np.dot(frame_norm, template / (np.linalg.norm(template) + 1e-10)))
            if score > best_score:
                best_score = score
                best_chord = chord_name
        raw_chords.append(best_chord)

    # Suavizar: unir acordes consecutivos iguales, filtrar cambios muy cortos
    chord_events: list[dict] = []
    current_chord: str | None = None
    current_start = 0.0
    min_duration = 0.8  # duracion minima de un acorde en segundos

    for i, chord in enumerate(raw_chords):
        if chord != current_chord:
            if current_chord is not None and i < len(times):
                duration = times[i] - current_start
                if duration >= min_duration:
                    chord_events.append({"chord": current_chord, "time": round(current_start, 2)})
            current_chord = chord
            current_start = times[i] if i < len(times) else current_start

    # Agregar ultimo acorde
    if current_chord is not None:
        chord_events.append({"chord": current_chord, "time": round(current_start, 2)})

    return chord_events


# ---------------------------------------------------------------------------
# Paso 3: Sincronizacion letras + acordes
# ---------------------------------------------------------------------------
def synchronize(lyrics_data: dict, chords_data: list[dict]) -> dict:
    """Cruza letras (con tiempos) y acordes (con tiempos) en secciones estructuradas."""
    segments = lyrics_data["segments"]
    if not segments:
        return {"sections": [], "detectedKey": "C", "keyType": "major"}

    # Construir secciones basandose en pausas entre segmentos
    sections: list[dict] = []
    current_lines: list[dict] = []
    section_counter = 1

    for i, seg in enumerate(segments):
        # Encontrar acordes dentro del rango temporal de este segmento
        seg_chords: list[dict] = []
        for chord_ev in chords_data:
            if seg["start"] <= chord_ev["time"] < seg["end"]:
                seg_duration = max(seg["end"] - seg["start"], 0.01)
                relative_pos = (chord_ev["time"] - seg["start"]) / seg_duration
                char_index = int(relative_pos * len(seg["text"]))
                char_index = max(0, min(char_index, len(seg["text"])))
                seg_chords.append({"chord": chord_ev["chord"], "charIndex": char_index})

        # Tambien incluir el acorde activo al inicio del segmento (si no hay acorde interno)
        if not seg_chords:
            # Buscar el ultimo acorde antes del inicio de este segmento
            active_chord = None
            for chord_ev in chords_data:
                if chord_ev["time"] <= seg["start"]:
                    active_chord = chord_ev["chord"]
                else:
                    break
            if active_chord:
                seg_chords.append({"chord": active_chord, "charIndex": 0})

        # Eliminar acordes duplicados en la misma posicion
        seen: set[int] = set()
        unique_chords: list[dict] = []
        for c in seg_chords:
            if c["charIndex"] not in seen:
                seen.add(c["charIndex"])
                unique_chords.append(c)

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

    # Agregar lineas restantes
    if current_lines:
        sections.append({"name": f"Verso {section_counter}", "lines": current_lines})

    # Intentar detectar coros (secciones con texto repetido)
    _detect_choruses(sections)

    # Detectar tonalidad
    all_chords = [c["chord"] for c in chords_data if c.get("chord")]
    detected_key, key_type = _detect_key(all_chords)

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
            # Buscar si ya asignamos un numero a un coro con texto similar
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

    freq: dict[str, int] = {}
    for name in chord_names:
        freq[name] = freq.get(name, 0) + 1

    major_intervals = [0, 2, 4, 5, 7, 9, 11]
    minor_intervals = [0, 2, 3, 5, 7, 8, 10]
    major_qualities = ["", "m", "m", "", "", "m", "dim"]
    minor_qualities = ["m", "dim", "", "m", "m", "", ""]

    best_key = "C"
    best_type = "major"
    best_score = -1

    for i, note in enumerate(NOTES):
        # Probar tonalidad mayor
        diatonic = [NOTES[(i + iv) % 12] + major_qualities[j] for j, iv in enumerate(major_intervals)]
        score = sum(freq.get(c, 0) for c in diatonic)
        if score > best_score:
            best_score = score
            best_key = note
            best_type = "major"

        # Probar tonalidad menor
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
