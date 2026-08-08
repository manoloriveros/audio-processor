# Audio Processor — extracción de letra y acordes

Servicio FastAPI (Railway) que recibe un audio y devuelve la canción estructurada
(secciones con líneas de letra + acordes con `charIndex`) para el SongEditor.

## Pipeline (v2, jul-2026)

```
audio (mix)
   │
   ├─ Paso 0 · separación de stems (self-hosted, GRATIS) ── audio-separator (MDX-Net ONNX, CPU)
   │      ├─ vocals.wav       → Paso 1
   │      └─ instrumental.wav → Paso 2
   │      (si falla/tarda/desactivada → se usa el mix en ambos pasos, como antes)
   │
   ├─ Paso 1 · letra ── whisper-1 (timestamps por palabra) + gpt-4o-transcribe
   │                    (texto de calidad) alineados palabra a palabra
   ├─ Paso 2 · acordes ── Chordino (NNLS Chroma + HMM) sobre el instrumental;
   │                      fallback Librosa. Beat-snap con beats del MIX original
   ├─ Paso 3 · sincronización ── acordes → charIndex por línea vía timestamps
   └─ Paso 4 · estructura (LLM) ── gpt-4o-mini corrige ortografía litúrgica y
                                   nombra secciones (Verso 1, Coro, Puente…)
```

Coste por canción de 4 min: **~$0.06** (2 transcripciones OpenAI + pasada mini).
La separación y los acordes son locales (solo CPU de Railway).

## Endpoints

- `POST /process` — multipart con archivo de audio (≤25 MB). Devuelve
  `{sections, detectedKey, keyType, engine, transcriptionModel}`.
- `POST /resolve-spotify` — JSON `{"url": "https://open.spotify.com/track/..."}`.
  Usa el enlace únicamente para identificar la pista y devuelve hasta cinco
  versiones candidatas de YouTube. No descarga ni analiza audio de Spotify:
  el usuario confirma una versión y después esa URL pasa por `/process-url`.
  El campo opcional `searchQuery` permite añadir artista o versión cuando el
  título por sí solo es ambiguo.
- `POST /process-url` — JSON `{"url": "https://youtube.com/watch?v=..."}`.
  Descarga el audio con **yt-dlp**, ejecuta el mismo pipeline y además:
  - adjunta `timestamps: [{time, order}]` por línea, sincronizados con el
    video (los marcadores del editor quedan listos);
  - devuelve `videoId` y `youtubeLink` para vincular el player.
  Límite de duración: `YT_MAX_DURATION` (default 12 min).

  ⚠️ YouTube bloquea con frecuencia descargas desde IPs de datacenter
  ("Sign in to confirm you're not a bot"). En ese caso el servicio responde
  422 con un mensaje que invita a subir el MP3. Mitigación: exportar cookies
  de un navegador logueado y ponerlas en `YTDLP_COOKIES_B64` (base64 de
  cookies.txt). Nota: descargar contenido puede infringir los ToS de YouTube;
  usar solo con canciones cuyo uso esté permitido.

### Por qué la separación importa

- Whisper alucina mucho menos transcribiendo la **voz aislada** (sin batería/instrumentos).
- El cromagrama del **instrumental** no está contaminado por la melodía vocal:
  Chordino acierta muchos más acordes.

## Motor premium opcional (dormido por defecto)

`musicai_engine.py` integra la API de [Music.ai](https://music.ai) (plataforma de
Moises): letra + acordes + secciones + beats profesionales (~$0.25–0.35/min).
**Solo se activa si `MUSIC_AI_API_KEY` está configurada**; sin ella no cuesta nada
y no se usa. Si el job falla, cae automáticamente al pipeline local.

Para activarlo algún día: crear cuenta en music.ai → Workflows → nuevo workflow
con módulos *Chords*, *Lyrics Transcription* (idioma: es), *Sections* y *Beats*,
salidas en JSON con esos nombres → copiar el slug a `MUSIC_AI_WORKFLOW`.

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `API_SECRET` | — | obligatoria; el backend la envía en `x-api-secret` |
| `OPENAI_API_KEY` | — | obligatoria para letra y pasada de estructura |
| `OPENAI_TRANSCRIPTION_MODEL` | `gpt-4o-transcribe` | modelo de texto (pasada 2) |
| `OPENAI_STRUCTURE_MODEL` | `gpt-4o-mini` | modelo de la pasada de estructura |
| `LLM_STRUCTURE` | `1` | `0` desactiva la pasada de estructura |
| `AUDIO_SEPARATION` | `1` | `0` desactiva la separación de stems |
| `SEPARATION_MODEL` | `Kim_Vocal_2.onnx` | modelo MDX (ver `audio-separator --list_models`) |
| `SEPARATION_TIMEOUT` | `210` | segundos máx. del subproceso de separación |
| `SEPARATION_MAX_DURATION` | `480` | no separar audios más largos (segundos) |
| `MODEL_FILE_DIR` | `/models` | carpeta del modelo ONNX (horneado en la imagen) |
| `CHORD_ENGINE` | `chordino` | `chordino` \| `librosa` \| `essentia` |
| `MUSIC_AI_API_KEY` | — | activa el motor premium Music.ai |
| `MUSIC_AI_WORKFLOW` | `songlory-transcription` | slug del workflow |
| `MUSIC_AI_JOB_TIMEOUT` | `150` | segundos máx. de espera del job |
| `YT_MAX_DURATION` | `720` | duración máxima (s) de videos de YouTube |
| `YTDLP_PROXY` | — | proxy residencial (http://user:pass@host:puerto) — la vía recomendada contra el anti-bot de YouTube, transparente para todos los usuarios |
| `YTDLP_COOKIES_B64` | — | cookies.txt en base64 (capa extra opcional; caducan y son por-cuenta) |

## Notas de despliegue (Railway)

- La imagen crece ~1.5–2 GB (torch CPU + onnxruntime); el modelo Kim_Vocal_2
  (~66 MB) se pre-descarga en el build.
- RAM recomendada: **4 GB** (la separación MDX usa 2–3 GB pico). Si hay OOM,
  bajar `SEPARATION_MODEL` a un modelo más pequeño o `AUDIO_SEPARATION=0`.
- La separación añade ~1–3 min por canción en CPU; los timeouts de frontend
  (`AudioImporter.jsx`) y backend (`server.js`) están en 8 min.
- `GET /health` muestra qué piezas están activas:
  `{"stemSeparation": true, "llmStructure": true, "musicai": false, ...}`.

## Tests

```
python -m pytest test_pipeline.py -q      # o: python test_pipeline.py
```

Cubren: normalización de etiquetas de acordes (incl. formato `C:maj` de
Music.ai), parsers del motor Music.ai, agrupación por secciones, asignación de
acordes a líneas, remapeo de `charIndex` tras la corrección LLM y la
sincronización legacy completa con datos simulados.
