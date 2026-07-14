"""
Pasada de estructuracion con LLM para el resultado de la transcripcion.

Toma las secciones ya sincronizadas (letra + acordes con charIndex) y usa un
modelo economico (default gpt-4o-mini) para:
  1. Corregir ortografia, tildes, mayusculas y puntuacion (espanol liturgico).
  2. Asignar nombres canonicos de seccion: Intro, Verso N, Pre-coro, Coro,
     Puente, Final (detecta coros por repeticion).

Garantias:
  - Nunca cambia el numero de secciones ni de lineas (si el LLM lo altera, se
    descarta su salida y se conserva el original).
  - Los charIndex de los acordes se reubican proporcionalmente cuando cambia
    la longitud de una linea.
  - Fail-open: ante cualquier error se devuelven las secciones sin tocar.

Variables de entorno:
  OPENAI_API_KEY          si falta, la pasada se omite.
  OPENAI_STRUCTURE_MODEL  modelo a usar (default: gpt-4o-mini).
  LLM_STRUCTURE           "0" para desactivar la pasada (default: activada).
"""

import json
import logging
import os

logger = logging.getLogger("audio-processor.structuring")

SECTION_LABEL_MAP = {
    "verse": "Verso",
    "verso": "Verso",
    "chorus": "Coro",
    "coro": "Coro",
    "refrain": "Coro",
    "estribillo": "Coro",
    "pre-chorus": "Pre-coro",
    "prechorus": "Pre-coro",
    "pre chorus": "Pre-coro",
    "bridge": "Puente",
    "puente": "Puente",
    "intro": "Intro",
    "outro": "Final",
    "ending": "Final",
    "coda": "Final",
    "instrumental": "Instrumental",
    "interlude": "Instrumental",
    "solo": "Instrumental",
}

_SYSTEM_PROMPT = (
    "Eres editor de cancioneros de musica catolica en espanol. Recibes un JSON con "
    "las secciones de una cancion transcrita automaticamente desde audio. Tu tarea:\n"
    "1. Corregir SOLO ortografia, tildes, mayusculas y puntuacion de cada linea "
    "(vocabulario liturgico: Señor, Dios, Jesús, Cristo, Espíritu Santo, María, "
    "aleluya, amén, misericordia...). Elimina muletillas imposibles de una letra "
    "cantada solo si son claramente ruido de transcripcion (p. ej. 'Subtitulos por...').\n"
    "2. Asignar a cada seccion su nombre canonico: Intro, Verso 1..N, Pre-coro, "
    "Coro, Puente o Final. Las secciones cuya letra se repite casi identica son el Coro.\n"
    "PROHIBIDO: añadir, eliminar, traducir o reordenar palabras, lineas o secciones. "
    "El resultado debe tener exactamente el mismo numero de secciones y de lineas "
    "por seccion que el original.\n"
    'Responde UNICAMENTE con JSON: {"secciones": [{"name": "...", "lines": ["..."]}]}'
)


def respace(chords: list[dict]) -> list[dict]:
    """Espaciado minimo entre acordes de una linea para evitar superposicion."""
    if len(chords) <= 1:
        return chords
    spaced = [chords[0]]
    for c in chords[1:]:
        prev = spaced[-1]
        min_pos = prev["charIndex"] + len(prev["chord"]) + 2
        spaced.append({**c, "charIndex": max(c["charIndex"], min_pos)})
    return spaced


def remap_chords(chords: list[dict], old_text: str, new_text: str) -> list[dict]:
    """Reubica los charIndex proporcionalmente cuando cambia el texto de la linea."""
    if old_text == new_text or not chords:
        return chords
    scale = len(new_text) / max(len(old_text), 1)
    remapped = []
    for c in chords:
        idx = int(round(c["charIndex"] * scale))
        remapped.append({**c, "charIndex": max(0, min(idx, max(len(new_text) - 1, 0)))})
    return respace(remapped)


def apply_structure(sections: list[dict], detected_key: str, key_type: str) -> list[dict]:
    """Aplica la pasada LLM sobre las secciones. Fail-open ante cualquier error."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not sections or os.getenv("LLM_STRUCTURE", "1") == "0":
        return sections
    model = os.getenv("OPENAI_STRUCTURE_MODEL", "gpt-4o-mini")
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        compact = [
            {"name": s["name"], "lines": [line["lyrics"] for line in s["lines"]]}
            for s in sections
        ]
        payload = json.dumps(
            {"tonalidad": f"{detected_key} {key_type}", "secciones": compact},
            ensure_ascii=False,
        )
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=0,
            timeout=45,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        new_sections = data.get("secciones")
        if not isinstance(new_sections, list) or len(new_sections) != len(sections):
            logger.warning("Estructura LLM descartada: numero de secciones no coincide")
            return sections

        for original, updated in zip(sections, new_sections):
            if not isinstance(updated, dict):
                continue
            if isinstance(updated.get("name"), str) and updated["name"].strip():
                original["name"] = updated["name"].strip()[:40]
            new_lines = updated.get("lines")
            if not isinstance(new_lines, list) or len(new_lines) != len(original["lines"]):
                continue  # conservar las lineas originales de esta seccion
            for line, new_text in zip(original["lines"], new_lines):
                if not isinstance(new_text, str) or not new_text.strip():
                    continue
                cleaned = new_text.strip()
                line["chords"] = remap_chords(line["chords"], line["lyrics"], cleaned)
                line["lyrics"] = cleaned
        logger.info("Pasada de estructura LLM aplicada (%s)", model)
        return sections
    except Exception as exc:
        logger.warning("Pasada de estructura LLM omitida: %s", exc)
        return sections
