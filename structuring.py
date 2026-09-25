"""Conservative lyric grouping, with an optional validated editorial pass.

Repeated multi-line blocks can suggest a chorus without making an API call.
This is a textual inference, not acoustic verse/chorus recognition. Different
line breaks, variations in the words, and repeated verses remain ambiguous.
Existing semantic sections and instrumental lines are preserved.

The optional LLM may regroup stable line IDs and edit only punctuation, case
and diacritics. Its complete result is validated before a copy is changed.
Chord anchors follow corresponding characters, never a length ratio; visual
spacing belongs to the renderer. No words, repetitions or timing are dropped.

LLM_STRUCTURE=0 disables only the optional API pass. OPENAI_API_KEY is required
for that pass; offline grouping always works without it. Call with
preserve_boundaries=True when sections already come from an acoustic model.
"""

from bisect import bisect_left
from collections import defaultdict
from copy import deepcopy
import json
import logging
import os
import re
import unicodedata

logger = logging.getLogger("audio-processor.structuring")

SECTION_LABEL_MAP = {
    "verse": "Verso", "verso": "Verso",
    "chorus": "Coro", "coro": "Coro", "refrain": "Coro", "estribillo": "Coro",
    "pre-chorus": "Pre-coro", "prechorus": "Pre-coro", "pre chorus": "Pre-coro",
    "pre-coro": "Pre-coro", "precoro": "Pre-coro",
    "bridge": "Puente", "puente": "Puente", "intro": "Intro",
    "outro": "Final", "ending": "Final", "coda": "Final", "final": "Final",
    "instrumental": "Instrumental", "interlude": "Instrumental", "solo": "Instrumental",
}

_SYSTEM_PROMPT = (
    "Organiza la letra de una cancion en espanol. Cada linea tiene un ID estable. "
    "Puedes reagrupar lineas consecutivas en secciones y cambiar el numero de "
    "secciones. Conserva TODOS los IDs exactamente una vez y en el mismo orden, "
    "incluidas todas las repeticiones del coro y las lineas instrumentales vacias. "
    "Nombres permitidos: Intro, Verso 1..N, Pre-coro, Coro, Puente, Instrumental, Final. "
    "No deduzcas un coro solo de una palabra repetida. Conserva las secciones "
    "instrumentales y los nombres semanticos ya asignados. Si preserve_boundaries "
    "es true conserva ademas todas las agrupaciones y nombres existentes. "
    "En lyrics solo puedes cambiar puntuacion, mayusculas y tildes. PROHIBIDO "
    "anadir, borrar, sustituir, traducir, unir o reordenar palabras. No borres "
    "supuesto ruido ni anadas etiquetas dentro de lyrics. No dividas ni unas lineas. "
    "Si no estas seguro, conserva el texto. Responde solo JSON con este formato: "
    '{"secciones":[{"name":"Verso 1","lines":[{"id":"L000001",'
    '"lyrics":"Texto original"}]}]}'
)


def respace(chords: list[dict]) -> list[dict]:
    """Keep musical anchors intact; chord-label width is a rendering concern."""
    return chords


def _unaccent(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text.casefold())
        if not unicodedata.combining(char)
    )


def _words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+", _unaccent(text), flags=re.UNICODE))


def _editorial_change_only(old_text: str, new_text: str) -> bool:
    if not isinstance(new_text, str) or "\n" in new_text or "\r" in new_text:
        return False
    if not old_text.strip():
        return not new_text.strip()
    if _words(old_text) != _words(new_text):
        return False
    return all(
        char.isalnum() or char.isspace()
        or unicodedata.category(char).startswith(("P", "M"))
        or char in old_text
        for char in new_text
    )


def _character_anchors(text: str) -> list[tuple[str, int]]:
    """Map normalized lexical characters back to their original offsets."""
    return [
        (base, index)
        for index, char in enumerate(text)
        for base in _unaccent(char)
        if base.isalnum()
    ]


def remap_chords(chords: list[dict], old_text: str, new_text: str) -> list[dict]:
    """Follow the same lexical characters after a validated editorial edit.

    An anchor on whitespace/punctuation follows the next lexical character;
    end-of-line anchors remain at the end. Instrumental and unchanged lines
    retain their exact positions, including intentionally shared anchors.
    """
    if old_text == new_text or not chords:
        return chords
    if not _editorial_change_only(old_text, new_text):
        raise ValueError("La correccion cambia las palabras de la letra")
    old_anchors = _character_anchors(old_text)
    new_anchors = _character_anchors(new_text)
    if [char for char, _ in old_anchors] != [char for char, _ in new_anchors]:
        raise ValueError("No hay un mapeo de caracteres exacto")
    if not old_anchors:
        return chords
    offsets = [offset for _, offset in old_anchors]
    result = []
    for chord in chords:
        index = max(0, int(chord["charIndex"]))
        anchor = bisect_left(offsets, index)
        new_index = new_anchors[anchor][1] if anchor < len(new_anchors) else len(new_text)
        result.append({**chord, "charIndex": new_index})
    return result


def _section_kind(name: str) -> str | None:
    clean = re.sub(r"\s+\d+$", "", str(name).strip().lower())
    return SECTION_LABEL_MAP.get(clean)


def _protected_section(section: dict) -> bool:
    """Keep supplied semantic labels; only generic verses need inference."""
    return _section_kind(section.get("name", "")) not in (None, "Verso")


def _flatten(sections: list[dict]) -> tuple[list[dict], list[int]]:
    lines, sources = [], []
    for index, section in enumerate(sections):
        for line in section["lines"]:
            lines.append(line)
            sources.append(index)
    return lines, sources


def _nonoverlapping(starts: list[int], length: int) -> list[int]:
    result = []
    for start in starts:
        if not result or start >= result[-1] + length:
            result.append(start)
    return result


def _meaningful_refrain(block: tuple) -> bool:
    if len(set(block)) < 2:
        return False
    tokens = [word for line in block for word in line]
    if len(tokens) < 6 or len(set(tokens)) < 4:
        return False
    # Prefer one complete repetition over an ABAB... concatenation of it.
    for period in range(1, len(block) // 2 + 1):
        if len(block) % period == 0 and block == block[:period] * (len(block) // period):
            return False
    return True


def infer_repeated_sections(sections: list[dict]) -> list[dict]:
    """Split generic verses around repeated blocks of complete lines.

    Exact normalized word matches are deliberate. Single repeated phrases or
    words, fuzzy matches, and a fixed every-four-lines rule are too speculative.
    Existing verse boundaries survive outside an inferred refrain. Returns the
    input unchanged if there is no defensible new grouping.
    """
    if not sections or any(not section.get("lines") for section in sections):
        return sections
    lines, sources = _flatten(sections)
    keys = [
        _words(line.get("lyrics", ""))
        if not _protected_section(sections[source]) else ()
        for line, source in zip(lines, sources)
    ]
    candidates = []
    # Short indexed seeds avoid comparing every line pair. Extend matching
    # seeds below so a long refrain is not cut at an arbitrary line count.
    for length in range(2, min(16, len(lines) // 2) + 1):
        occurrences = defaultdict(list)
        for start in range(len(lines) - length + 1):
            block = tuple(keys[start:start + length])
            if all(block):
                occurrences[block].append(start)
        for block, starts in occurrences.items():
            starts = _nonoverlapping(starts, length)
            if len(starts) < 2:
                continue
            block_length = length
            limit = min(len(lines) - starts[-1], min(b - a for a, b in zip(starts, starts[1:])))
            while block_length < limit:
                next_key = keys[starts[0] + block_length]
                if not next_key or any(keys[start + block_length] != next_key for start in starts[1:]):
                    break
                block_length += 1
            extended = tuple(keys[starts[0]:starts[0] + block_length])
            if _meaningful_refrain(extended):
                candidates.append((block_length * len(starts), block_length, starts))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2][0]))
    occupied: set[int] = set()
    refrains: dict[int, int] = {}
    for _, length, starts in candidates:
        available = [
            start for start in starts
            if all(index not in occupied for index in range(start, start + length))
        ]
        if len(available) < 2:
            continue
        for start in available:
            refrains[start] = start + length
            occupied.update(range(start, start + length))
    if not refrains:
        return sections

    result, index, verse_number = [], 0, 0
    while index < len(lines):
        source = sources[index]
        original = sections[source]
        is_refrain = index in refrains
        end = refrains.get(index, index + 1)
        if not is_refrain:
            while end < len(lines) and sources[end] == source and end not in refrains:
                end += 1
        section = {key: deepcopy(value) for key, value in original.items() if key != "lines"}
        section["lines"] = deepcopy(lines[index:end])
        if is_refrain:
            section["name"] = "Coro"
        elif _section_kind(original.get("name", "")) == "Verso":
            verse_number += 1
            section["name"] = f"Verso {verse_number}"
        result.append(section)
        index = end
    return result


def _canonical_name(name: object, verse_number: int) -> str:
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 40:
        raise ValueError("Nombre de seccion invalido")
    kind = _section_kind(name)
    if not kind:
        raise ValueError("Nombre de seccion no reconocido")
    return f"Verso {verse_number}" if kind == "Verso" else kind


def _validated_regroup(sections: list[dict], proposal: object, preserve_boundaries: bool) -> list[dict]:
    """Validate the entire proposal before producing any modified line copies."""
    if not isinstance(proposal, list) or not proposal:
        raise ValueError("Faltan las secciones propuestas")
    originals, sources = _flatten(sections)
    expected_ids = [f"L{index + 1:06d}" for index in range(len(originals))]
    by_id = dict(zip(expected_ids, originals))
    groups, seen_ids = [], []
    verse_number = 0
    for proposed in proposal:
        if not isinstance(proposed, dict) or not isinstance(proposed.get("lines"), list) or not proposed["lines"]:
            raise ValueError("Seccion propuesta vacia o invalida")
        if _section_kind(proposed.get("name", "")) == "Verso":
            verse_number += 1
        name = _canonical_name(proposed.get("name"), verse_number)
        group = []
        for item in proposed["lines"]:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] not in by_id:
                raise ValueError("ID de linea desconocido")
            line_id = item["id"]
            original = by_id[line_id]
            text = item.get("lyrics", original["lyrics"])
            if not _editorial_change_only(original["lyrics"], text):
                raise ValueError("La correccion cambia las palabras de la letra")
            group.append((line_id, text.strip()))
            seen_ids.append(line_id)
        groups.append((name, group))
    if seen_ids != expected_ids:
        raise ValueError("Los IDs no cubren exactamente las lineas en su orden original")

    original_groups, offset = [], 0
    for section in sections:
        group_ids = expected_ids[offset:offset + len(section["lines"])]
        original_groups.append(group_ids)
        offset += len(group_ids)
    proposed_groups = [[line_id for line_id, _ in group] for _, group in groups]
    if preserve_boundaries and proposed_groups != original_groups:
        raise ValueError("La propuesta cambia limites acusticos protegidos")
    protected = {}
    for section, group_ids in zip(sections, original_groups):
        if preserve_boundaries or _protected_section(section) or any(not line["lyrics"].strip() for line in section["lines"]):
            protected[tuple(group_ids)] = section["name"]
            if group_ids not in proposed_groups:
                raise ValueError("La propuesta cambia una seccion semantica o instrumental")

    result, offset = [], 0
    for name, group in groups:
        source = sources[offset]
        updated = {key: deepcopy(value) for key, value in sections[source].items() if key != "lines"}
        group_ids = tuple(line_id for line_id, _ in group)
        updated["name"] = protected.get(group_ids, name)
        updated["lines"] = []
        for line_id, text in group:
            line = deepcopy(by_id[line_id])
            line["chords"] = remap_chords(line.get("chords", []), line["lyrics"], text)
            line["lyrics"] = text
            updated["lines"].append(line)
        result.append(updated)
        offset += len(group)
    return result


def apply_structure(
    sections: list[dict], detected_key: str, key_type: str, *, preserve_boundaries: bool = False,
) -> list[dict]:
    """Free grouping first; optional LLM failure leaves that result untouched."""
    if not sections or any(not section.get("lines") for section in sections):
        return sections
    base = sections if preserve_boundaries else infer_repeated_sections(sections)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or os.getenv("LLM_STRUCTURE", "0") == "0":
        return base
    model = os.getenv("OPENAI_STRUCTURE_MODEL", "gpt-4o-mini")
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        compact, ordinal = [], 0
        for section in base:
            compact_lines = []
            for line in section["lines"]:
                ordinal += 1
                compact_lines.append({"id": f"L{ordinal:06d}", "lyrics": line["lyrics"]})
            compact.append({"name": section["name"], "lines": compact_lines})
        payload = json.dumps({
            "tonalidad": f"{detected_key} {key_type}", "secciones": compact,
            "preserve_boundaries": preserve_boundaries,
        }, ensure_ascii=False)
        response = client.chat.completions.create(
            model=model, response_format={"type": "json_object"}, temperature=0, timeout=45,
            messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": payload}],
        )
        data = json.loads(response.choices[0].message.content)
        result = _validated_regroup(base, data.get("secciones"), preserve_boundaries)
        logger.info("Pasada de estructura validada aplicada (%s)", model)
        return result
    except Exception as exc:
        logger.warning("Pasada de estructura LLM descartada; se conserva agrupacion local: %s", exc)
        return base
