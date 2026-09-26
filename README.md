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
4. ChordMini con contraste de séptimas en Chordino (o el fallback) conserva cambios breves,
   intervalos y eventos `N` sin armonía. Los beats de toda la grabación se analizan
   por ventanas y se adjuntan como referencia, sin mover los tiempos detectados.
5. Las líneas se cortan en tiempos reales de palabras cuando existen. Las pausas
   vocales menores de ocho segundos no generan nuevas secciones instrumentales:
   los cambios de acorde permanecen al final de la frase. Un acorde sostenido
   ya mostrado no se repite en cada línea. Intro, interludios y finales
   instrumentales conservan sus acordes. `charIndex` expresa
   el anclaje musical; el espaciado para que las etiquetas no se tapen pertenece
   al editor.
6. La agrupación gratuita reconoce bloques de varias líneas repetidas y propone
   coros aunque no haya pausas. Dos repeticiones contiguas, por sí solas, conservan
   el nombre de verso: también pueden ser estrofas iniciales repetidas. La pasada
   LLM opcional puede reagrupar IDs de línea
   consecutivos; se rechaza completa si pierde, duplica o reordena contenido. Solo
   puede cambiar puntuación, mayúsculas y tildes, conservando los tiempos.

Los timestamps por palabra del proveedor siguen usando `whisper-1` con
`verbose_json`; la pasada textual usa JSON, según la
[documentación oficial de transcripción](https://developers.openai.com/api/docs/guides/speech-to-text).
No se ha incorporado un proveedor de pago nuevo. La transcripción existente y la
pasada LLM opcional siguen usando la API de OpenAI por defecto. El motor local
opcional descrito abajo no requiere esa API y corre donde se ejecute este servicio;
todavía no es inferencia en el navegador. `LLM_STRUCTURE=0` permite
usar únicamente la agrupación local de secciones y ahora es el valor por defecto.
La pasada adicional requiere `LLM_STRUCTURE=1` explícito; una configuración
existente con ese valor sigue teniendo prioridad.

Las peticiones de transcripción ya no llevan una lista de vocabulario religioso.
Ese texto podía introducir palabras ausentes durante pasajes instrumentales. La
coincidencia observada en la captura motivó retirarlo; no es una prueba causal A/B
contra la API desplegada.

### Transcripción gratuita opcional en el equipo local

Se añadió `TRANSCRIPTION_ENGINE=faster-whisper`, con `large-v3-turbo` multilingüe,
CPU int8 y cuatro hilos por defecto. Se conserva el ensamblado de ventanas de
120 segundos con contexto, incluyendo las repeticiones lejanas. Este modo omite
Music.ai y la pasada editorial de OpenAI incluso si hay claves configuradas; si
falla no cambia automáticamente a un servicio de pago.

En un entorno con las dependencias del servicio y FFmpeg/FFprobe instalados:

```powershell
python -m pip install -r requirements-local.txt
$env:LOCAL_WHISPER_CACHE = 'C:\models\songlory-whisper'
python local_transcription.py --download
$env:TRANSCRIPTION_ENGINE = 'faster-whisper'
python main.py
```

Mantener la misma caché y configuración al iniciar el servicio. Preparar el
modelo descarga aproximadamente 1,6 GB; las peticiones HTTP solo admiten pesos
ya descargados. `API_SECRET` sigue siendo obligatorio para los endpoints.
`LOCAL_WHISPER_MODEL`, `LOCAL_WHISPER_DEVICE`, `LOCAL_WHISPER_COMPUTE_TYPE` y
`LOCAL_WHISPER_THREADS` permiten ajustar el equipo. GPU requiere las bibliotecas
CUDA/cuDNN compatibles con faster-whisper; la prueba realizada fue en CPU.

La imagen Docker predeterminada conserva OpenAI: no instala ni descarga este
modelo automáticamente. El modo local requiere preparar su entorno y dirigir el
backend de Songlory a ese procesador. No cambia por sí solo una instancia ya
desplegada. El reconocimiento todavía puede inventar texto o equivocarse al
cantar; no se activa el VAD de habla, que eliminó casi toda la canción en la
prueba con la grabación de Athenas.

La respuesta conserva `chordTimeline` con `{chord, time, end?}` y metadatos del
motor cuando existen. El pipeline local añade `analysisWarnings`,
`analysisDuration` y `transcriptionChunks`. Los tiempos ausentes se marcan como
estimados internamente; no producen marcadores de video de falsa precisión.
Los acordes `N` quedan en la cronología y no se dibujan como acordes en la letra.

### Límites y validación

- Detectar una repetición textual no demuestra que sea un coro: un verso repetido
  puede ser ambiguo. Variaciones o saltos de línea diferentes pueden impedir la
  agrupación. Las secciones acústicas de Music.ai se conservan cuando existen.
- Ocho segundos es un umbral conservador de presentación, no un detector de
  instrumentales. Todavía faltan límites y etiquetas basados en la estructura
  musical para distinguir de forma fiable versos, coros y puentes.
- Las ventanas acotan la memoria de transcripción y beats. Los detectores
  Librosa/Essentia todavía cargan el audio completo. La separación conserva su
  límite de 480 segundos por defecto, y YouTube su límite de 720 segundos.
- La corrección textual conservadora prioriza no borrar contenido medido; puede
  dejar sin aplicar una corrección útil que cambie la cantidad de palabras.
- Los tiempos por palabra no son una alineación forzada específica para canto.
  La posición dentro de una palabra sigue siendo aproximada. La cronología de
  acordes se devuelve para diagnóstico, pero el editor guarda sus anclas de letra,
  no esta cronología completa.
- La suite automática es de regresión, con respuestas simuladas y audio
  sintético. Además se hizo un piloto manual con el MP3 de Athenas de 450,49 s.
  Ninguno establece un porcentaje de precisión musical: falta una anotación
  completa de esa grabación y una evaluación con más canciones.

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

La voz aislada puede reducir la interferencia de los instrumentos; el
acompañamiento puede reducir la de la melodía vocal sobre los acordes. La
separación también introduce artefactos. Hay que comparar ambas entradas con
canto real antes de atribuirle una mejora de precisión.

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
| `TRANSCRIPTION_ENGINE` | `openai` | `openai` o `faster-whisper`; el segundo bloquea las llamadas de pago del pipeline |
| `OPENAI_API_KEY` | — | necesaria para la transcripción OpenAI y su pasada LLM opcional; no para faster-whisper |
| `OPENAI_TRANSCRIPTION_MODEL` | `gpt-4o-transcribe` | modelo de texto (pasada 2) |
| `OPENAI_STRUCTURE_MODEL` | `gpt-4o-mini` | modelo de la pasada de estructura |
| `LLM_STRUCTURE` | `0` | `0` desactiva solo la pasada LLM; la agrupación local sigue activa |
| `AUDIO_SEPARATION` | `1` | `0` desactiva la separación de stems |
| `SEPARATION_MODEL` | `Kim_Vocal_2.onnx` | modelo MDX (ver `audio-separator --list_models`) |
| `SEPARATION_TIMEOUT` | `210` | segundos máx. del subproceso de separación |
| `SEPARATION_MAX_DURATION` | `480` | no separar audios más largos (segundos) |
| `MODEL_FILE_DIR` | `/models` | carpeta del modelo ONNX (horneado en la imagen) |
| `CHORD_ENGINE` | `auto` | ChordMini + contraste de séptimas; admite `chordmini` solo, `chordino`, `librosa`, `essentia` |
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


## Acordes y colocación: actualización del 25 de septiembre

El valor predeterminado `CHORD_ENGINE=auto` usa [ChordMini](https://github.com/ptnghia-j/ChordMini)
con el [export ONNX](https://huggingface.co/musetric/chordmini-onnx) fijado en la
revisión `086162411b8c4772774392be195e5c6f065d67ad`. Los dos archivos se verifican
con SHA-256. La CQT sigue el contrato Python publicado; la normalización ya está
dentro del modelo. No se normaliza la amplitud de la grabación. CPU, dos hilos;
no hay llamadas de inferencia externas ni cobros por este detector.

Preparación local con las dependencias del servicio instaladas:

```powershell
python -m pip install -r requirements.txt
$env:CHORDMINI_MODEL_DIR = 'C:\models\songlory-chordmini'
python chordmini.py --download
$env:CHORD_ENGINE = 'auto'
python main.py
```

Mantener `CHORDMINI_MODEL_DIR` al iniciar el servicio. Docker prepara estos pesos
al construir la imagen y falla si no puede verificarlos. Una variable existente
`CHORD_ENGINE=chordino` tiene prioridad: cambiarla a `auto` para activar el modo
nuevo. No se ha desplegado ni reiniciado una instancia remota desde esta revisión.
`CHORDMINI_THREADS` permite de uno a ocho hilos. Las peticiones nunca descargan
pesos; si faltan, continúa Chordino y luego Librosa, con aviso en la respuesta.
`/health` muestra ChordMini solo si el runtime y los pesos pueden cargarse.

En modo `auto`, Chordino contrasta las tríadas de ChordMini. Conserva una séptima
con la misma raíz y tercera si ambos intervalos coinciden durante al menos 0,8 s.
Se mantienen los límites medidos del intervalo que aporta la extensión. Esta regla
no modifica suspendidos, séptimas, inversiones o cambios breves ya emitidos por
ChordMini. No añade sextas ni sustituye notas para encajarlas en una tonalidad.
Las extensiones añadidas llevan `extensionNeedsReview`, `extensionEngine` y
`primaryChord`: son discrepancias entre detectores, no acordes certificados.
El modo explícito `chordmini` omite ese contraste. Si Chordino no está instalado,
`auto` sigue usando ChordMini sin contraste. Si el primario falla, siguen los
fallbacks anteriores. Todos estos detectores pueden equivocarse.

La salida retiene séptimas y suspendidos, sin reducirlos a acordes básicos.
El vocabulario neural tiene 170 clases, pero NO reconoce inversiones con bajo
independiente ni todas las extensiones posibles. Las inversiones emitidas por
los otros motores se conservan; no se inventan inversiones para el modelo neural.
La tonalidad se calcula por duración de acordes, de modo que cien subdivisiones
de un acorde corto no pesen más que un acorde largo. Si falta el final de la
cronología legacy, se conserva el cálculo por frecuencias, sin inventar duración.

Cada acorde colocado incluye `audioTime`, `audioEnd` cuando existe y
`alignmentSource`. Esto permite distinguir palabras con tiempos, estimaciones
por segmento, acordes sostenidos y pausas vocales. Los beats no desplazan el
acorde. Dentro de una palabra todavía se interpola: no es alineación por sílaba.
SongEditor conserva esta evidencia durante la importación; el formato de guardado
actual sigue persistiendo las anclas de letra, no la cronología acústica completa.

La prueba real cubrió el MP3 completo de Athenas (450,49 s), mezcla y acompañamiento
separado. Se usó el Demucs de seis pistas del navegador mediante un adaptador CPU
local: tardó 626,8 s y por eso NO sustituye a Kim_Vocal_2 en el servicio. Este ensayo
no comprueba el separador Kim del Docker ni la aplicación desplegada. La prueba de
Chordino en Windows usa el mismo algoritmo/parámetros con el build comunitario
nnls-chroma Win64 y Sonic Annotator 1.7; no el binario Linux del contenedor.

En un único audio sintético de 84 acordes, la coincidencia exacta de etiqueta en
los dos segundos centrales de cada acorde fue 69,05% con Chordino, 54,05% con
ChordMini solo y 84,00% con el contraste. El último método se ajustó tras observar
ese mismo ensayo: no es un conjunto independiente ni un porcentaje de precisión
sobre canciones. Siguen fallando suspendidos y algunas séptimas. La grabación de
Athenas no tiene anotación completa para medir precisión o exactitud temporal;
las capturas son una referencia parcial. La prueba de asignación comprueba que
ningún cambio detectado se pierde ni mueve al ensamblar, no que cada predicción
sea musicalmente correcta. También quedan errores de versos/coros/puente.
