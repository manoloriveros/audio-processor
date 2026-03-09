"""
Audio Processor - Servicio de extraccion de letras y acordes.
Recibe un archivo de audio, transcribe la letra con Whisper (OpenAI)
y detecta los acordes con Essentia (HPCP + ChordsDetection), con fallback a Librosa.
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
# Verificar disponibilidad de Essentia
# ---------------------------------------------------------------------------
_ESSENTIA_AVAILABLE = False
try:
    import essentia
    import essentia.standard as _es_std
    _ESSENTIA_AVAILABLE = True
    logger.info("Motor de acordes: Essentia (HPCP + ChordsDetection)")
except ImportError:
    logger.warning("Essentia no disponible — se usara Librosa como fallback")

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
    # Septima dominante: 1 - 3M - 5J - 7m (pesos bajos para evitar falsos positivos)
    CHORD_TEMPLATES[f"{_note}7"] = _build_template(_i, [(0, 1.0), (4, 0.5), (7, 0.5), (10, 0.3)])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "engine": "essentia" if _ESSENTIA_AVAILABLE else "librosa"}


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
            timestamp_granularities=["word", "segment"],
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

    # Extraer palabras con timestamps individuales
    words: list[dict] = []
    raw_words = None
    if hasattr(response, "words"):
        raw_words = response.words
    elif isinstance(response, dict) and "words" in response:
        raw_words = response["words"]

    if raw_words:
        for w in raw_words:
            w_text = w["word"] if isinstance(w, dict) else getattr(w, "word", "")
            w_start = w["start"] if isinstance(w, dict) else getattr(w, "start", 0)
            w_end = w["end"] if isinstance(w, dict) else getattr(w, "end", 0)
            w_text = w_text.strip()
            if w_text:
                words.append({"word": w_text, "start": w_start, "end": w_end})

    logger.info("Whisper: %d segmentos, %d palabras con timestamps", len(segments), len(words))
    return {"text": full_text, "segments": segments, "words": words}


# ---------------------------------------------------------------------------
# Paso 2A: Deteccion de acordes con Essentia (primario)
# ---------------------------------------------------------------------------
def _detect_chords_essentia(audio_path: str) -> list[dict]:
    """
    Detecta acordes usando Essentia:
    1. HPCP (36-bin Harmonic Pitch Class Profile) — mejor resolucion que chroma
    2. ChordsDetection con plantillas Gaussianas (mayor, menor, dim, aug)
    3. Post-proceso para detectar septimas desde HPCP
    4. min_duration adaptativo segun tempo
    """
    import essentia
    import essentia.standard as es

    sr = 44100
    audio = es.MonoLoader(filename=audio_path, sampleRate=sr)()

    if len(audio) < sr:  # Menos de 1 segundo
        return []

    # --- Detectar tempo para min_duration adaptativo ---
    try:
        bpm = es.RhythmExtractor2013(method="multifeature")(audio)[0]
    except Exception:
        bpm = 120.0
    beat_dur = 60.0 / max(bpm, 60)
    logger.info("Tempo (Essentia): %.1f BPM (beat=%.2fs)", bpm, beat_dur)

    # --- Extraccion HPCP frame a frame ---
    frame_size = 8192   # ~186ms a 44100Hz — buena resolucion para acordes
    hop_size = 2048     # ~46ms

    win = es.Windowing(type="blackmanharris62")
    spec_algo = es.Spectrum()
    peaks_algo = es.SpectralPeaks(
        orderBy="magnitude",
        magnitudeThreshold=1e-5,
        minFrequency=40,
        maxFrequency=5000,
        maxPeaks=100,
        sampleRate=sr,
    )
    hpcp_algo = es.HPCP(
        size=36,
        referenceFrequency=440,
        harmonics=8,
        bandPreset=True,
        minFrequency=40,
        maxFrequency=5000,
        weightType="cosine",
        nonLinear=False,
        windowSize=1.0,
        sampleRate=sr,
    )

    hpcp_frames = []
    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
        s = spec_algo(win(frame))
        f, m = peaks_algo(s)
        hpcp_frames.append(hpcp_algo(f, m))

    if not hpcp_frames:
        return []

    hpcp_array = np.array(hpcp_frames)  # shape: (n_frames, 36)
    n_frames = hpcp_array.shape[0]
    times = np.arange(n_frames) * hop_size / sr

    # --- ChordsDetection: plantillas Gaussianas sobre HPCP 36-bin ---
    chords_det = es.ChordsDetection(hopSize=hop_size, sampleRate=sr, windowSize=2)
    chords, strengths = chords_det(hpcp_array)

    # --- Post-proceso: detectar septimas desde HPCP ---
    # Reducir HPCP 36-bin a 12-bin para analisis de intervalos
    hpcp_12 = np.zeros((n_frames, 12))
    for i in range(n_frames):
        for j in range(12):
            hpcp_12[i, j] = np.mean(hpcp_array[i, j * 3 : (j + 1) * 3])

    NOTE_TO_IDX = {n: idx for idx, n in enumerate(NOTES)}

    enhanced_chords: list[str] = []
    for i in range(len(chords)):
        chord = chords[i]
        if chord == "N" or i >= n_frames:
            enhanced_chords.append(chord)
            continue

        # Parsear raiz y calidad
        is_minor = chord.endswith("m") and not chord.endswith("dim")
        root = chord[:-1] if is_minor else chord
        root = root.replace("dim", "").replace("aug", "")

        if root not in NOTE_TO_IDX:
            enhanced_chords.append(chord)
            continue

        root_idx = NOTE_TO_IDX[root]
        m7_idx = (root_idx + 10) % 12  # 10 semitonos = 7ma menor
        root_energy = hpcp_12[i, root_idx]
        m7_energy = hpcp_12[i, m7_idx]

        # Si la 7ma menor tiene energia significativa relativa a la raiz
        if root_energy > 0.01 and m7_energy / (root_energy + 1e-10) > 0.60:
            if is_minor:
                enhanced_chords.append(root + "m7")
            elif not any(chord.endswith(s) for s in ("dim", "aug")):
                enhanced_chords.append(chord + "7")
            else:
                enhanced_chords.append(chord)
        else:
            enhanced_chords.append(chord)

    # --- Generar eventos con min_duration adaptativo ---
    chord_events: list[dict] = []
    current_chord: str | None = None
    current_start = 0.0
    min_duration = max(0.5, beat_dur * 0.75)  # ~75% de un beat, minimo 0.5s

    for i, chord in enumerate(enhanced_chords):
        t = times[i] if i < len(times) else times[-1]

        if chord == "N":
            if current_chord is not None:
                dur = t - current_start
                if dur >= min_duration:
                    chord_events.append({"chord": current_chord, "time": round(current_start, 2)})
                current_chord = None
            continue

        if chord != current_chord:
            if current_chord is not None:
                dur = t - current_start
                if dur >= min_duration:
                    chord_events.append({"chord": current_chord, "time": round(current_start, 2)})
            current_chord = chord
            current_start = t

    if current_chord is not None:
        chord_events.append({"chord": current_chord, "time": round(current_start, 2)})

    logger.info("Acordes (Essentia): %d eventos", len(chord_events))
    return chord_events


# ---------------------------------------------------------------------------
# Deteccion de tonalidad desde cromagrama (Krumhansl-Kessler)
# ---------------------------------------------------------------------------
def _detect_key_from_chroma(chroma: np.ndarray) -> tuple[str, str]:
    """Detecta tonalidad usando perfiles Krumhansl-Kessler sobre el cromagrama global."""
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    chroma_sum = np.sum(chroma, axis=1)
    chroma_sum = chroma_sum / (np.linalg.norm(chroma_sum) + 1e-10)

    best_key = 0
    best_type = "major"
    best_corr = -1.0

    for shift in range(12):
        shifted = np.roll(chroma_sum, -shift)
        corr_maj = np.corrcoef(shifted, major_profile)[0, 1]
        if corr_maj > best_corr:
            best_corr, best_key, best_type = corr_maj, shift, "major"
        corr_min = np.corrcoef(shifted, minor_profile)[0, 1]
        if corr_min > best_corr:
            best_corr, best_key, best_type = corr_min, shift, "minor"

    return NOTES[best_key], best_type


# ---------------------------------------------------------------------------
# Paso 2B: Deteccion de acordes con Librosa (fallback)
# ---------------------------------------------------------------------------
def _detect_chords_librosa(audio_path: str) -> list[dict]:
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
    EXTENDED_BONUS = 0.35
    scores = template_matrix @ chroma_norm

    def _classify_frame(frame_idx: int, diatonic_bonus: dict[str, float] | None = None):
        frame_scores = scores[:, frame_idx].copy()
        if diatonic_bonus:
            for j, name in enumerate(template_names):
                frame_scores[j] += diatonic_bonus.get(name, 0.0)
        if np.max(frame_scores) < 0.60:
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

    # --- Detectar tonalidad desde cromagrama (Krumhansl-Kessler) ---
    det_key, det_type = _detect_key_from_chroma(chroma)
    diatonic = _build_diatonic_set(det_key, det_type)
    logger.info("Tonalidad detectada (K-K): %s %s, diatonicos: %s", det_key, det_type, diatonic)

    # --- Clasificacion con sesgo diatonico fuerte ---
    bonus: dict[str, float] = {}
    for name in template_names:
        if name in diatonic:
            bonus[name] = 0.20
        else:
            bonus[name] = -0.12
    raw_chords: list[str | None] = [_classify_frame(i, bonus) for i in range(chroma.shape[1])]

    # Filtro de moda sobre la secuencia (mediana no aplica a datos categoricos)
    if len(raw_chords) > 5:
        from collections import Counter
        smoothed = list(raw_chords)
        half = 2
        for i in range(len(raw_chords)):
            window = raw_chords[max(0, i - half):min(len(raw_chords), i + half + 1)]
            counter = Counter(window)
            smoothed[i] = counter.most_common(1)[0][0]
        raw_chords = smoothed

    # Generar eventos: solo cuando el acorde CAMBIA, duracion minima 1.0s
    chord_events: list[dict] = []
    current_chord: str | None = None
    current_start = 0.0
    min_duration = 1.0

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

    logger.info("Acordes detectados (Librosa fallback): %d eventos", len(chord_events))
    return chord_events


# ---------------------------------------------------------------------------
# Paso 2: Wrapper — Essentia primario, Librosa fallback
# ---------------------------------------------------------------------------
def detect_chords(audio_path: str) -> list[dict]:
    """Detecta acordes: intenta Essentia primero, Librosa como fallback."""
    if _ESSENTIA_AVAILABLE:
        try:
            return _detect_chords_essentia(audio_path)
        except Exception as e:
            logger.error("Error en Essentia chord detection: %s — fallback a Librosa", e, exc_info=True)
    return _detect_chords_librosa(audio_path)


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


def _time_to_char_index(
    chord_time: float,
    line_text: str,
    line_start: float,
    line_end: float,
    words: list[dict],
) -> int:
    """Mapea timestamp de acorde a indice de caracter usando posiciones de palabras.

    Si hay words con timestamps disponibles, busca la palabra mas cercana al
    momento del acorde y calcula el charIndex real.  Fallback: interpolacion lineal.
    """
    # Filtrar palabras que pertenecen a esta linea (con tolerancia de 0.1s)
    line_words = [
        w for w in words
        if w["start"] >= line_start - 0.1 and w["end"] <= line_end + 0.1
    ]

    if not line_words:
        # Fallback: interpolacion lineal
        seg_dur = max(line_end - line_start, 0.01)
        rel = (chord_time - line_start) / seg_dur
        return max(1, min(int(rel * len(line_text)), len(line_text) - 1))

    # Mapear cada palabra a su posicion de caracter en line_text
    search_from = 0
    word_positions: list[dict] = []
    text_lower = line_text.lower()
    for w in line_words:
        clean = w["word"].strip()
        pos = text_lower.find(clean.lower(), search_from)
        if pos == -1:
            pos = search_from
        word_positions.append({
            "char_start": pos,
            "char_end": pos + len(clean),
            "time_start": w["start"],
            "time_end": w["end"],
        })
        search_from = pos + len(clean)

    # Buscar la palabra donde cae el acorde
    for wp in word_positions:
        if chord_time <= wp["time_start"]:
            return max(1, wp["char_start"])
        if wp["time_start"] <= chord_time <= wp["time_end"]:
            # Interpolar dentro de la palabra
            word_dur = max(wp["time_end"] - wp["time_start"], 0.01)
            progress = (chord_time - wp["time_start"]) / word_dur
            char_within = int(progress * (wp["char_end"] - wp["char_start"]))
            return max(1, min(wp["char_start"] + char_within, len(line_text) - 1))

    # El acorde cae despues de todas las palabras
    return max(1, min(word_positions[-1]["char_end"], len(line_text) - 1))


def synchronize(lyrics_data: dict, chords_data: list[dict]) -> dict:
    """Cruza letras (con tiempos) y acordes (con tiempos) en secciones estructuradas."""
    segments = lyrics_data["segments"]
    words = lyrics_data.get("words", [])
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
                char_index = _time_to_char_index(
                    chord_ev["time"], seg["text"], seg["start"], seg["end"], words,
                )
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

        # Enforcar espaciado minimo para evitar superposicion visual
        if len(unique_chords) > 1:
            spaced = [unique_chords[0]]
            for c in unique_chords[1:]:
                prev = spaced[-1]
                min_pos = prev["charIndex"] + len(prev["chord"]) + 2
                if c["charIndex"] < min_pos:
                    if min_pos < len(seg["text"]):
                        spaced.append({**c, "charIndex": min_pos})
                    # sin espacio: descartar este acorde
                else:
                    spaced.append(c)
            unique_chords = spaced

        # Limitar a maximo 6 acordes por linea
        if len(unique_chords) > 6:
            step = len(unique_chords) / 6
            unique_chords = [unique_chords[int(i * step)] for i in range(6)]

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
