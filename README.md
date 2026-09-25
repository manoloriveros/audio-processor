# Audio Processor — extracción de letra y acordes

Servicio FastAPI (Railway) que recibe un audio y devuelve la canción estructurada
(secciones con líneas de letra + acordes con `charIndex`) para el SongEditor.

## Pipeline (revisión de precisión, septiembre de 2026)

1. Separación opcional de voz/instrumental con audio-separator. Si no hay stems,
   el procesamiento continúa sobre la mezcla original.
2. Transcripción por ventanas de 120 segundos, con 2 segundos de contexto a cada
   lado. FFmpeg crea un solo WAV mono PCM16 de 16 kHz a la vez; incluso los audios
   cortos se normalizan. Los tiempos vuelven al eje temporal de la grabación.
3. La segunda transcripción solo corrige palabras cuando conserva su cantidad y
   correspondencia. No se reparten tiempos inventados entre palabras nuevas ni
   se elimina un coro porque el otro modelo lo haya resumido.
4. Chordino (o el fallback configurado) conserva inversiones, cambios breves,
   intervalos y eventos `N` sin armonía. Los beats de toda la grabación se analizan
   por ventanas y se adjuntan como referencia, sin mover los tiempos detectados.
5. Las líneas se cortan en tiempos reales de palabras cuando existen. Intro,
   interludios y finales instrumentales conservan sus acordes. `charIndex` expresa
   el anclaje musical; el espaciado para que las etiquetas no se tapen pertenece
   al editor.
6. La agrupación gratuita reconoce bloques de varias líneas repetidas y propone
   coros aunque no haya pausas. La pasada LLM opcional puede reagrupar IDs de línea
   consecutivos; se rechaza completa si pierde, duplica o reordena contenido. Solo
   puede cambiar puntuación, mayúsculas y tildes, conservando los tiempos.

Los timestamps por palabra del proveedor siguen usando `whisper-1` con
`verbose_json`; la pasada textual usa JSON, según la
[documentación oficial de transcripción](https://developers.openai.com/api/docs/guides/speech-to-text).
No se ha incorporado un proveedor de pago nuevo. La transcripción existente y la
pasada LLM opcional siguen usando la API de OpenAI; esta revisión no las convierte
súbitamente en procesamiento gratuito en el navegador. `LLM_STRUCTURE=0` permite
usar únicamente la agrupación local de secciones y ahora es el valor por defecto.
La pasada adicional requiere `LLM_STRUCTURE=1` explícito; una configuración
existente con ese valor sigue teniendo prioridad.

La respuesta conserva `chordTimeline` con `{chord, time, end?}` y metadatos del
motor cuando existen. El pipeline local añade `analysisWarnings`,
`analysisDuration` y `transcriptionChunks`. Los tiempos ausentes se marcan como
estimados internamente; no producen marcadores de video de falsa precisión.
Los acordes `N` quedan en la cronología y no se dibujan como acordes en la letra.

### Límites y validación

- Detectar una repetición textual no demuestra que sea un coro: un verso repetido
  puede ser ambiguo. Variaciones o saltos de línea diferentes pueden impedir la
  agrupación. Las secciones acústicas de Music.ai se conservan cuando existen.
- Las ventanas acotan la memoria de transcripción y beats. Los detectores
  Librosa/Essentia todavía cargan el audio completo. La separación conserva su
  límite de 480 segundos por defecto, y YouTube su límite de 720 segundos.
- La corrección textual conservadora prioriza no borrar contenido medido; puede
  dejar sin aplicar una corrección útil que cambie la cantidad de palabras.
- Los tiempos por palabra no son una alineación forzada específica para canto.
  La posición dentro de una palabra sigue siendo aproximada. La cronología de
  acordes se devuelve para diagnóstico, pero el editor guarda sus anclas de letra,
  no esta cronología completa.
- Las pruebas son de regresión, con respuestas simuladas y una prueba sintética
  de conversión FFmpeg. No establecen un porcentaje de precisión musical ni
  sustituyen una comparación con grabaciones largas anotadas.

Para probar, instalar las dependencias del servicio y `pytest`, después usar
`python -m pytest -q`. La conversión sintética requiere FFmpeg; usa metadatos de
WAV conocidos para poder probar sin FFprobe. El servicio real requiere ambos.

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
| `OPENAI_API_KEY` | — | necesaria para transcripción y pasada LLM opcional |
| `OPENAI_TRANSCRIPTION_MODEL` | `gpt-4o-transcribe` | modelo de texto (pasada 2) |
| `OPENAI_STRUCTURE_MODEL` | `gpt-4o-mini` | modelo de la pasada de estructura |
| `LLM_STRUCTURE` | `0` | `0` desactiva solo la pasada LLM; la agrupación local sigue activa |
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
