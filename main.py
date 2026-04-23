"""
Audio Processor - Servicio de extraccion de letras y acordes.
Recibe un archivo de audio, transcribe la letra con Whisper (OpenAI)
y detecta los acordes con Chordino/Librosa, con Essentia como motor legacy opcional.
"""

import os
import tempfile
import logging
import re
from difflib import SequenceMatcher

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
API_SECRET = os.getenv("API_SECRET", "change-me-in-production")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-transcribe")
CHORD_ENGINE = os.getenv("CHORD_ENGINE", "chordino").strip().lower()
CHORDINO_ROLL_ON = float(os.getenv("CHORDINO_ROLL_ON", "1"))
CHORDINO_BOOST_N = float(os.getenv("CHORDINO_BOOST_N", "0.05"))

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
# Verificar disponibilidad de motores de acordes
# ---------------------------------------------------------------------------
_CHORDINO_AVAILABLE = False
try:
    from chord_extractor.extractors import Chordino as _Chordino
    _CHORDINO_AVAILABLE = True
    logger.info("Motor de acordes disponible: Chordino (NNLS Chroma + HMM)")
except Exception as exc:
    logger.warning("Chordino no disponible — se usara Librosa si es necesario: %s", exc)

_ESSENTIA_AVAILABLE = False
try:
    import essentia
    import essentia.standard as _es_std
    _ESSENTIA_AVAILABLE = True
    logger.info("Motor de acordes disponible: Essentia (HPCP + ChordsDetection)")
except Exception as exc:
    logger.info("Essentia no disponible — motor legacy desactivado: %s", exc)

# ---------------------------------------------------------------------------
# Plantillas de acordes (mayor, menor, 7)
# 12 notas x 3 tipos = 36 acordes
# ---------------------------------------------------------------------------
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

ENHARMONIC_TO_FLAT = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}
FLAT_TO_ENHARMONIC = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
CHORD_RE = re.compile(r"^([A-G](?:#|b)?)(.*)$")


def _build_template(root_idx: int, intervals: list[tuple[int, float]]) -> np.ndarray:
    t = np.zeros(12)
    for semitone, weight in intervals:
        t[(root_idx + semitone) % 12] = weight
    return t


def _normalize_note_name(note: str) -> str:
    """Normaliza notas a sostenidos para el procesamiento interno."""
    return FLAT_TO_ENHARMONIC.get(note, note)


def _to_flat_note(note: str) -> str:
    return ENHARMONIC_TO_FLAT.get(note, note)


def _split_chord_root(chord_name: str) -> tuple[str | None, str, str | None]:
    """Separa raiz, calidad e inversion opcional en acordes como C#m7/G#."""
    chord_name = chord_name.strip()
    main, slash, bass = chord_name.partition("/")
    match = CHORD_RE.match(main)
    if not match:
        return None, chord_name, None
    root = _normalize_note_name(match.group(1))
    quality = match.group(2)
    bass_note = None
    if slash and bass:
        bass_match = CHORD_RE.match(bass)
        bass_note = _normalize_note_name(bass_match.group(1)) if bass_match else bass
    return root, quality, bass_note


def _to_flat(chord_name: str) -> str:
    """Convierte nombre con sostenido a bemol, incluyendo inversiones (C#/G# -> Db/Ab)."""
    root, quality, bass = _split_chord_root(chord_name)
    if not root:
        return chord_name
    flat_chord = _to_flat_note(root) + quality
    if bass:
        flat_chord += "/" + _to_flat_note(bass)
    return flat_chord


def _normalize_chord_label(chord_name: str | None) -> str | None:
    """Normaliza etiquetas de motores externos al formato usado por SongEditor."""
    if not chord_name:
        return None

    cleaned = str(chord_name).strip().replace("♭", "b").replace(" ", "")
    if not cleaned or cleaned.upper() in {"N", "NOCHORD", "NONE"}:
        return None

    main, slash, bass = cleaned.partition("/")
    match = CHORD_RE.match(main)
    if not match:
        return None

    root = _normalize_note_name(match.group(1))
    quality = match.group(2)
    if quality.startswith(":"):
        quality = quality[1:]

    quality_map = {
        "": "",
        "maj": "",
        "major": "",
        "min": "m",
        "minor": "m",
        "m": "m",
        "7": "7",
        "maj7": "maj7",
        "major7": "maj7",
        "min7": "m7",
        "minor7": "m7",
        "m7": "m7",
        "sus2": "sus2",
        "sus4": "sus4",
        "dim": "dim",
        "aug": "aug",
        "m7b5": "m7b5",
        "min7b5": "m7b5",
    }
    normalized_quality = quality_map.get(quality.lower(), quality)

    normalized = root + normalized_quality
    if slash and bass:
        bass_match = CHORD_RE.match(bass)
        if bass_match:
            normalized += "/" + _normalize_note_name(bass_match.group(1))
    return normalized


def _use_flats(key: str, key_type: str) -> bool:
    """Determina si la tonalidad usa bemoles."""
    flat_minor_roots = {2, 7, 0, 5}  # D, G, C, F
    flat_major_roots = {5, 10, 3, 8}  # F, Bb, Eb, Ab
    key_idx = NOTES.index(key) if key in NOTES else -1
    if key_type == "minor":
        return key_idx in flat_minor_roots
    return key_idx in flat_major_roots


def _build_diatonic_set(key: str, key_type: str) -> set[str]:
    """Construye el conjunto de acordes diatonicos para una tonalidad.

    Excluye ii° (menor) y vii° (mayor) porque los acordes disminuidos
    practicamente no aparecen en musica popular/liturgica, y su inclusion
    como acordes mayores generaba falsos positivos (ej. E mayor en Dm).
    """
    key_idx = NOTES.index(key) if key in NOTES else 0
    if key_type == "minor":
        # i, III, iv, v, VI, VII  (sin ii°)
        intervals_q = [(0, "m"), (3, ""), (5, "m"), (7, "m"), (8, ""), (10, "")]
    else:
        # I, ii, iii, IV, V, vi  (sin vii°)
        intervals_q = [(0, ""), (2, "m"), (4, "m"), (5, ""), (7, ""), (9, "m")]

    diatonic = set()
    for iv, q in intervals_q:
        note = NOTES[(key_idx + iv) % 12]
        diatonic.add(note + q)

    # V7 es comun en ambos modos; V mayor en menor armonico
    v_note = NOTES[(key_idx + 7) % 12]
    diatonic.add(v_note + "7")
    if key_type == "minor":
        diatonic.add(v_note)  # V mayor (menor armonico)
    return diatonic


CHORD_TEMPLATES: dict[str, np.ndarray] = {}
for _i, _note in enumerate(NOTES):
    # Mayor: 1 - 3M - 5J
    CHORD_TEMPLATES[_note] = _build_template(_i, [(0, 1.0), (4, 0.8), (7, 0.8)])
    # Menor: 1 - 3m - 5J
    CHORD_TEMPLATES[f"{_note}m"] = _build_template(_i, [(0, 1.0), (3, 0.8), (7, 0.8)])
    # Septima dominante: 1 - 3M - 5J - 7m (pesos bajos para evitar falsos positivos)
    CHORD_TEMPLATES[f"{_note}7"] = _build_template(_i, [(0, 1.0), (4, 0.5), (7, 0.5), (10, 0.3)])
    # Septima mayor
    CHORD_TEMPLATES[f"{_note}maj7"] = _build_template(_i, [(0, 1.0), (4, 0.74), (7, 0.68), (11, 0.36)])
    # Menor septima
    CHORD_TEMPLATES[f"{_note}m7"] = _build_template(_i, [(0, 1.0), (3, 0.74), (7, 0.68), (10, 0.36)])
    # Suspendidos comunes
    CHORD_TEMPLATES[f"{_note}sus2"] = _build_template(_i, [(0, 1.0), (2, 0.72), (7, 0.7)])
    CHORD_TEMPLATES[f"{_note}sus4"] = _build_template(_i, [(0, 1.0), (5, 0.76), (7, 0.7)])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    available = ["librosa"]
    if _CHORDINO_AVAILABLE:
        available.insert(0, "chordino")
    if _ESSENTIA_AVAILABLE:
        available.append("essentia")
    return {"status": "ok", "configuredEngine": CHORD_ENGINE, "availableEngines": available}


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
        result["transcriptionModel"] = lyrics_data.get("model")

        logger.info(
            "Procesamiento completado: %d secciones, clave detectada: %s, %d acordes detectados, modelo STT: %s",
            len(result["sections"]),
            result["detectedKey"],
            sum(len(line["chords"]) for s in result["sections"] for line in s["lines"]),
            result.get("transcriptionModel"),
        )
        return result

    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Paso 1: Transcripcion con Whisper
# ---------------------------------------------------------------------------
WHISPER_PROMPT = (
    "Cancion cristiana catolica cantada en espanol. "
    "Es una interpretacion vocal musical, no habla conversacional. "
    "Puede haber melismas, vocales sostenidas y silabas alargadas por el canto. "
    "Transcribe la palabra real y canonica, sin repetir letras por el sostenido musical. "
    "No inventes palabras por adornos melodicos ni por respiraciones. "
    "La cancion tiene versos y coros que se repiten varias veces; conserva las repeticiones reales. "
    "Vocabulario frecuente: Señor, Dios, Jesús, Cristo, Espíritu Santo, María, aleluya, amén, "
    "cordero, gloria, bendito, misericordia, alabanza, adoración."
)

TRANSCRIPTION_MODEL_FALLBACKS = [
    OPENAI_TRANSCRIPTION_MODEL,
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "whisper-1",
]


def transcribe_with_whisper(audio_path: str) -> dict:
    """Transcribe audio con el mejor modelo disponible de OpenAI para speech-to-text."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    response = None
    selected_model = None
    last_error: Exception | None = None

    with open(audio_path, "rb") as f:
        for candidate_model in dict.fromkeys(TRANSCRIPTION_MODEL_FALLBACKS):
            try:
                f.seek(0)
                response = client.audio.transcriptions.create(
                    model=candidate_model,
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                    language="es",
                    prompt=WHISPER_PROMPT,
                )
                selected_model = candidate_model
                logger.info("Modelo de transcripcion usado: %s", selected_model)
                break
            except Exception as exc:
                last_error = exc
                logger.warning("Fallo transcripcion con %s: %s", candidate_model, exc)

    if response is None:
        raise RuntimeError(f"No se pudo transcribir el audio con ningun modelo disponible: {last_error}")

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

    logger.info("Transcripcion (%s): %d segmentos, %d palabras con timestamps", selected_model, len(segments), len(words))
    return {"text": full_text, "segments": segments, "words": words, "model": selected_model}


# ---------------------------------------------------------------------------
# Paso 2A: Deteccion de acordes con Chordino (NNLS Chroma + HMM)
# ---------------------------------------------------------------------------
def _detect_chords_chordino(audio_path: str) -> list[dict]:
    """
    Detecta acordes con Chordino.

    Chordino usa NNLS Chroma para estimar contenido armonico y un HMM interno
    para estabilizar la secuencia de acordes. Es mas apropiado que el matching
    directo por plantillas cuando hay voz, guitarras con armonicos fuertes o
    cambios con ruido melodico.
    """
    from chord_extractor.extractors import Chordino

    chordino = Chordino(
        roll_on=CHORDINO_ROLL_ON,
        boost_n_likelihood=CHORDINO_BOOST_N,
        spectral_whitening=1,
        spectral_shape=0.7,
    )
    changes = chordino.extract(audio_path)

    chord_events: list[dict] = []
    last_chord: str | None = None
    for change in changes:
        chord = _normalize_chord_label(getattr(change, "chord", None))
        timestamp = float(getattr(change, "timestamp", 0.0))
        if chord is None:
            last_chord = None
            continue
        if chord == last_chord:
            continue
        chord_events.append({"chord": chord, "time": round(timestamp, 2)})
        last_chord = chord

    logger.info("Acordes detectados (Chordino): %d eventos", len(chord_events))
    return chord_events


# ---------------------------------------------------------------------------
# Paso 2B: Deteccion de acordes con Essentia (legacy opcional)
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
        if root_energy > 0.01 and m7_energy / (root_energy + 1e-10) > 0.70:
            if is_minor:
                enhanced_chords.append(root + "m7")
            elif not any(chord.endswith(s) for s in ("dim", "aug")):
                enhanced_chords.append(chord + "7")
            else:
                enhanced_chords.append(chord)
        else:
            enhanced_chords.append(chord)

    # --- Suavizado: filtro de moda (ventana ~400ms a 46ms/frame) ---
    if len(enhanced_chords) > 9:
        from collections import Counter
        smoothed_e = list(enhanced_chords)
        half_e = 4  # ventana de 9 frames ≈ 414ms
        for idx_e in range(len(enhanced_chords)):
            win = enhanced_chords[max(0, idx_e - half_e):min(len(enhanced_chords), idx_e + half_e + 1)]
            non_null = [c for c in win if c != "N"]
            if non_null and enhanced_chords[idx_e] != "N":
                smoothed_e[idx_e] = Counter(non_null).most_common(1)[0][0]
        enhanced_chords = smoothed_e

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
# Paso 2C: Deteccion de acordes con Librosa (fallback optimizado)
# ---------------------------------------------------------------------------
def _detect_chords_librosa(audio_path: str) -> list[dict]:
    """
    Detecta acordes usando Librosa como motor local sin dependencias Vamp.

    Cambios frente a la version anterior:
    1. No colapsa a beat-sync; conserva resolucion temporal de ~46 ms.
    2. Combina Chroma CQT + CENS para equilibrar detalle y estabilidad.
    3. Usa Viterbi con transiciones armonicas en vez de argmax frame-a-frame.
    4. Aplica un sesgo diatonico suave, no una correccion agresiva.
    5. Ajusta cambios a onsets armonicos cercanos sin moverlos demasiado.
    """
    import librosa
    from scipy.ndimage import median_filter

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    if len(y) < sr:
        return []

    hop_length = 1024

    # Separar componente armonica con margen alto para reducir bateria y ruido.
    y_harmonic, _ = librosa.effects.hpss(y, margin=(1.0, 5.0))

    try:
        tuning = float(librosa.estimate_tuning(y=y_harmonic, sr=sr))
    except Exception:
        tuning = 0.0

    chroma_cqt = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=hop_length,
        n_chroma=12,
        bins_per_octave=36,
        tuning=tuning,
    )
    chroma_cens = librosa.feature.chroma_cens(y=y_harmonic, sr=sr, hop_length=hop_length)
    chroma = (0.75 * chroma_cqt) + (0.25 * chroma_cens)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

    # Detectar onsets armonicos para snap posterior
    onset_env = librosa.onset.onset_strength(y=y_harmonic, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        y=y_harmonic, sr=sr, hop_length=hop_length, onset_envelope=onset_env,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Filtro temporal leve: estabiliza vibrato/melodia sin borrar cambios cortos.
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

    scores = template_matrix @ chroma_norm

    # --- Detectar tonalidad desde cromagrama (Krumhansl-Kessler) ---
    det_key, det_type = _detect_key_from_chroma(chroma)
    diatonic = _build_diatonic_set(det_key, det_type)
    logger.info("Tonalidad detectada (K-K): %s %s, diatonicos: %s", det_key, det_type, diatonic)

    # --- Emisiones: sesgo diatonico suave + penalizacion para extensiones dudosas ---
    emission = scores.astype(np.float32).T * 2.2
    for j, name in enumerate(template_names):
        if name in diatonic:
            emission[:, j] += 0.08
        else:
            emission[:, j] -= 0.02
        if name.endswith(("maj7", "m7")):
            emission[:, j] -= 0.16
        elif name.endswith("7"):
            emission[:, j] -= 0.18
        elif name.endswith(("sus2", "sus4")):
            emission[:, j] -= 0.08

    # --- Transiciones armonicas para Viterbi ---
    n_states = len(template_names)
    transition = np.full((n_states, n_states), -0.18, dtype=np.float32)
    np.fill_diagonal(transition, 0.32)

    state_meta: list[tuple[int, str]] = []
    for name in template_names:
        root, quality, _ = _split_chord_root(name)
        state_meta.append((NOTES.index(root) if root in NOTES else 0, quality))

    for prev_idx, (prev_root, prev_quality) in enumerate(state_meta):
        for next_idx, (next_root, next_quality) in enumerate(state_meta):
            if prev_idx == next_idx:
                continue
            root_motion = (next_root - prev_root) % 12
            if root_motion in {5, 7}:  # IV/V, cadencias y circulo de quintas
                transition[prev_idx, next_idx] += 0.07
            elif root_motion in {2, 10}:  # movimiento por tono, comun en pop/liturgico
                transition[prev_idx, next_idx] += 0.03
            if prev_root == next_root and prev_quality != next_quality:
                transition[prev_idx, next_idx] -= 0.07
            if template_names[prev_idx] in diatonic and template_names[next_idx] in diatonic:
                transition[prev_idx, next_idx] += 0.03

    def _viterbi_decode(emission_scores: np.ndarray) -> list[int]:
        if emission_scores.shape[0] == 0:
            return []
        n_frames, n_labels = emission_scores.shape
        backptr = np.zeros((n_frames, n_labels), dtype=np.int16)
        dp = emission_scores[0].copy()
        for frame_idx in range(1, n_frames):
            candidates = dp[:, None] + transition
            backptr[frame_idx] = np.argmax(candidates, axis=0)
            dp = emission_scores[frame_idx] + np.max(candidates, axis=0)
        path = np.zeros(n_frames, dtype=np.int16)
        path[-1] = int(np.argmax(dp))
        for frame_idx in range(n_frames - 2, -1, -1):
            path[frame_idx] = backptr[frame_idx + 1, path[frame_idx + 1]]
        return path.tolist()

    path = _viterbi_decode(emission)
    best_raw_scores = np.max(scores, axis=0)
    raw_chords: list[str | None] = []
    for i, state_idx in enumerate(path):
        if best_raw_scores[i] < 0.48:
            raw_chords.append(None)
        else:
            raw_chords.append(template_names[state_idx])

    # Generar eventos: solo cuando el acorde CAMBIA, duracion minima adaptativa
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    _tempo_val = float(np.atleast_1d(tempo)[0]) if hasattr(tempo, '__len__') else float(tempo)
    beat_dur = 60.0 / max(_tempo_val, 60)
    min_duration = max(0.4, beat_dur * 0.50)
    logger.info("Tempo (Librosa): %.1f BPM, min_duration: %.2fs", _tempo_val, min_duration)

    runs: list[dict] = []
    current_chord: str | None = raw_chords[0] if raw_chords else None
    current_start = float(times[0]) if len(times) else 0.0
    for i, chord in enumerate(raw_chords):
        if chord != current_chord:
            end_time = float(times[i]) if i < len(times) else current_start
            runs.append({"chord": current_chord, "start": current_start, "end": end_time})
            current_chord = chord
            current_start = end_time

    if raw_chords:
        end_time = float(times[-1] + hop_length / sr)
        runs.append({"chord": current_chord, "start": current_start, "end": end_time})

    chord_events: list[dict] = []
    for run in runs:
        chord = run["chord"]
        if chord is None:
            continue
        duration = run["end"] - run["start"]
        if duration < min_duration:
            continue
        if chord_events and chord_events[-1]["chord"] == chord:
            continue
        chord_events.append({"chord": chord, "time": round(float(run["start"]), 2)})

    # Snap cada cambio de acorde al onset armonico mas cercano (mejora timing)
    if len(onset_times) > 0:
        for event in chord_events:
            closest_idx = np.argmin(np.abs(onset_times - event["time"]))
            # Solo snap si el onset esta cerca; evita mover cambios de compas completos.
            if abs(onset_times[closest_idx] - event["time"]) < 0.22:
                event["time"] = round(float(onset_times[closest_idx]), 2)

    logger.info("Acordes detectados (Librosa): %d eventos", len(chord_events))
    return chord_events


# ---------------------------------------------------------------------------
# Paso 2: Wrapper — motor configurable + fallback + post-proceso
# ---------------------------------------------------------------------------
def _configured_engine_sequence() -> list[str]:
    """Resuelve el orden de motores segun CHORD_ENGINE."""
    if CHORD_ENGINE in {"auto", "default"}:
        return ["chordino", "librosa"]
    if CHORD_ENGINE in {"chordino", "nnls"}:
        return ["chordino", "librosa"]
    if CHORD_ENGINE == "librosa":
        return ["librosa"]
    if CHORD_ENGINE == "essentia":
        return ["essentia", "librosa"]

    logger.warning("CHORD_ENGINE=%s no reconocido; usando chordino con fallback a librosa", CHORD_ENGINE)
    return ["chordino", "librosa"]


def detect_chords(audio_path: str) -> list[dict]:
    """Detecta acordes con motor configurable y aplica limpieza final."""
    last_error: Exception | None = None
    for engine in _configured_engine_sequence():
        if engine == "chordino" and not _CHORDINO_AVAILABLE:
            logger.warning("Chordino configurado pero no disponible; probando siguiente motor")
            continue
        if engine == "essentia" and not _ESSENTIA_AVAILABLE:
            logger.warning("Essentia configurado pero no disponible; probando siguiente motor")
            continue

        try:
            if engine == "chordino":
                events = _detect_chords_chordino(audio_path)
            elif engine == "essentia":
                events = _detect_chords_essentia(audio_path)
            else:
                events = _detect_chords_librosa(audio_path)

            events = _postprocess_chord_events(events)
            if events:
                logger.info("Motor de acordes usado: %s", engine)
                return events
            logger.warning("Motor %s no produjo acordes; probando siguiente motor", engine)
        except Exception as e:
            last_error = e
            logger.error("Error en motor de acordes %s: %s", engine, e, exc_info=True)

    if last_error:
        logger.error("Todos los motores de acordes fallaron; ultimo error: %s", last_error)
    return []


def _find_nearest_diatonic(chord: str, diatonic: set[str]) -> str | None:
    """Encuentra el acorde diatonico mas cercano a uno no diatonico.

    Busca por proximidad de raiz (±1-2 semitonos) manteniendo la calidad,
    y como ultimo recurso prueba la calidad opuesta en la misma raiz.
    """
    is_m7 = chord.endswith("m7")
    is_7 = chord.endswith("7") and not is_m7
    is_minor = chord.endswith("m") and not is_m7 and not chord.endswith("dim")

    if is_m7:
        root = chord[:-2]
        quality = "m"
    elif is_7:
        root = chord[:-1]
        quality = ""
    elif is_minor:
        root = chord[:-1]
        quality = "m"
    else:
        root = chord
        quality = ""

    if root not in NOTES:
        return None

    root_idx = NOTES.index(root)

    # Buscar raices vecinas (±1-2 semitonos) con la misma calidad
    for offset in [1, -1, 2, -2]:
        candidate = NOTES[(root_idx + offset) % 12] + quality
        if candidate in diatonic:
            return candidate

    # Intentar calidad opuesta en la misma raiz
    alt_quality = "" if quality == "m" else "m"
    alt_chord = root + alt_quality
    if alt_chord in diatonic:
        return alt_chord

    return None


def _postprocess_chord_events(chord_events: list[dict]) -> list[dict]:
    """Post-procesa eventos de acordes:
    1. Ordena y fusiona duplicados inmediatos
    2. Elimina parpadeos A→B→A donde B dura muy poco
    3. Conserva acordes cromaticos reales: no fuerza sustituciones diatonicas
    """
    if len(chord_events) < 2:
        return chord_events

    ordered_events = sorted(
        (
            {"chord": event["chord"], "time": round(float(event["time"]), 2)}
            for event in chord_events
            if event.get("chord") is not None and event.get("time") is not None
        ),
        key=lambda event: (event["time"], event["chord"]),
    )

    if len(ordered_events) < 2:
        return ordered_events

    deduped = [ordered_events[0]]
    for event in ordered_events[1:]:
        prev = deduped[-1]
        if event["chord"] == prev["chord"] and abs(event["time"] - prev["time"]) < 0.18:
            continue
        deduped.append(event)

    # Calcular duracion de cada evento
    for i in range(len(deduped)):
        if i < len(deduped) - 1:
            deduped[i]["_dur"] = deduped[i + 1]["time"] - deduped[i]["time"]
        else:
            deduped[i]["_dur"] = 999.0

    # Eliminar parpadeos breves sin reescribir la armonia real
    corrected = deduped
    if len(corrected) >= 3:
        filtered = [corrected[0]]
        for i in range(1, len(corrected) - 1):
            prev = filtered[-1]
            curr = corrected[i]
            nxt = corrected[i + 1]
            if (prev["chord"] == nxt["chord"]
                    and curr["chord"] != prev["chord"]
                    and curr.get("_dur", 999) < 0.85):
                continue
            filtered.append(curr)
        filtered.append(corrected[-1])
        corrected = filtered

    # Fusionar consecutivos iguales tras limpiar parpadeos
    merged = [corrected[0]]
    for event in corrected[1:]:
        if event["chord"] == merged[-1]["chord"]:
            continue
        merged.append(event)

    # Limpiar campo temporal
    for event in merged:
        event.pop("_dur", None)

    logger.info("Post-proceso: %d → %d eventos", len(chord_events), len(merged))
    return merged


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
            seg_chords.append({
                "chord": last_before["chord"],
                "charIndex": 0,
                "_time": last_before["time"],
            })

        # Acordes que caen DENTRO del rango temporal de esta linea
        for chord_ev in chords_data:
            if seg["start"] < chord_ev["time"] < seg["end"]:
                char_index = _time_to_char_index(
                    chord_ev["time"], seg["text"], seg["start"], seg["end"], words,
                )
                seg_chords.append({
                    "chord": chord_ev["chord"],
                    "charIndex": char_index,
                    "_time": chord_ev["time"],
                })

        # Acordes entre el fin de esta linea y el inicio de la siguiente (intermedios)
        next_start = segments[i + 1]["start"] if i < len(segments) - 1 else seg["end"] + 999
        for chord_ev in chords_data:
            if seg["end"] <= chord_ev["time"] < next_start:
                seg_chords.append({
                    "chord": chord_ev["chord"],
                    "charIndex": len(seg["text"]),
                    "_time": chord_ev["time"],
                })

        # Deduplicar solo duplicados reales, sin descartar acordes distintos
        seg_chords.sort(key=lambda chord: (chord.get("_time", 0), chord["charIndex"]))
        unique_chords: list[dict] = []
        for c in seg_chords:
            normalized_char_index = max(0, int(c["charIndex"]))
            if unique_chords:
                prev = unique_chords[-1]
                same_chord = c["chord"] == prev["chord"]
                same_spot = abs(normalized_char_index - prev["charIndex"]) <= 1
                if same_chord and same_spot:
                    continue
            unique_chords.append({
                "chord": c["chord"],
                "charIndex": normalized_char_index,
                "_time": c.get("_time", 0),
            })

        # Enforcar espaciado minimo para evitar superposicion visual
        if len(unique_chords) > 1:
            spaced = [unique_chords[0]]
            for c in unique_chords[1:]:
                prev = spaced[-1]
                min_pos = prev["charIndex"] + len(prev["chord"]) + 2
                spaced.append({
                    **c,
                    "charIndex": max(c["charIndex"], min_pos),
                })
            unique_chords = spaced

        for chord in unique_chords:
            chord.pop("_time", None)

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
        normalized = _normalize_chord_label(name) or name
        main = normalized.split("/", 1)[0]
        root, quality, _ = _split_chord_root(main)
        if not root:
            continue
        quality_l = quality.lower()
        if quality_l in {"m", "m7", "m6", "m9", "min", "min7"}:
            base = root + "m"
        elif quality_l in {"dim", "m7b5"}:
            base = root + "dim"
        else:
            base = root
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
