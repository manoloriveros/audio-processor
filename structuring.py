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
from difflib import SequenceMatcher
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
    "outro": "Outro", "ending": "Final", "coda": "Final", "final": "Final",
    "instrumental": "Instrumental", "interlude": "Instrumental", "solo": "Instrumental",
}

_SYSTEM_PROMPT = (
    "Organiza la letra de una cancion en espanol. Cada linea tiene un ID estable. "
    "Puedes reagrupar lineas consecutivas en secciones y cambiar el numero de "
    "secciones. Conserva TODOS los IDs exactamente una vez y en el mismo orden, "
    "incluidas todas las repeticiones del coro y las lineas instrumentales vacias. "
    "Nombres permitidos: Intro, Verso 1..N, Pre-coro, Coro, Puente, Instrumental, Outro, Final. "
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
    for period in range(1, (len(block) + 1) // 2 + 1):
        if len(block) - period >= 2 and all(block[i] == block[i % period] for i in range(len(block))):
            return False
    return True


def _similar_phrase(left, right, threshold=.82):
    if not left or not right:
        return False
    if ({w for w in left if w.isdigit()} != {w for w in right if w.isdigit()}
            or ("no" in left) != ("no" in right)):
        return False
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    shared = sum(block.size for block in matcher.get_matching_blocks())
    fillers = {"y", "porque", "oh", "ah", "el", "la"}
    connective_only = all(set(left[a:b]) | set(right[c:d]) <= fillers
                          for tag, a, b, c, d in matcher.get_opcodes() if tag != "equal")
    return shared >= 2 and (matcher.ratio() >= threshold or connective_only)


def _pattern_ranges(keys, pattern):
    """Recover complete learned refrains split differently by ASR.

    Match all slots before accepting a range. Only a duplicated phrase may
    be shortened, with at least two other distinct full phrases corroborating
    the same block. Classification does not rewrite any lyric or line.
    """
    result, start = [], 0
    duplicated = {phrase for phrase in pattern if pattern.count(phrase) > 1}
    while start < len(keys):
        def match(slot, cursor, strong):
            if slot == len(pattern):
                return cursor if len(strong) >= min(2, len(set(pattern))) else None
            phrase = pattern[slot]
            for count in range(1, 4):
                parts = keys[cursor:cursor + count]
                if len(parts) != count or not all(parts):
                    break
                if count > 1 and any(len(part) == 1 and part[0] not in {"porque", "y", "oh", "ah"} for part in parts):
                    continue
                combined = tuple(word for part in parts for word in part)
                full = _similar_phrase(phrase, combined)
                relaxed = (count == 1 and phrase in duplicated and len(combined) <= len(phrase)
                           and _similar_phrase(phrase, combined, .65))
                first_variant = (slot == 0 and len(pattern) >= 3 and len(set(phrase) & set(combined)) >= 3
                                 and _similar_phrase(phrase, combined, .75))
                if full or relaxed or first_variant:
                    end = match(slot + 1, cursor + count, strong | ({phrase} if full else set()))
                    if end is not None:
                        return end
            return None
        end = match(0, start, set())
        if end is None:
            start += 1
        else:
            result.append((start, end))
            start = end
    return result


def _phrase_keys(lines, sources, sections):
    """Compare complete phrases with conservative ASR variations, never edit them."""
    keys, representatives = [], []
    fillers = {"y", "porque", "oh", "ah", "el", "la"}
    for line, source in zip(lines, sources):
        key = _words(line.get("lyrics", "")) if not _protected_section(sections[source]) else ()
        representative = key
        if len(key) >= 2:
            for previous in representatives:
                if len(previous) < 2 or abs(len(key) - len(previous)) > 2:
                    continue
                if ({w for w in key if w.isdigit()} != {w for w in previous if w.isdigit()}
                        or ("no" in key) != ("no" in previous)):
                    continue
                matcher = SequenceMatcher(None, previous, key, autojunk=False)
                shared = sum(block.size for block in matcher.get_matching_blocks())
                connective_only = all(
                    set(previous[a:b]) | set(key[c:d]) <= fillers
                    for tag, a, b, c, d in matcher.get_opcodes() if tag != "equal")
                if shared >= 2 and (matcher.ratio() >= .82 or connective_only):
                    representative = previous
                    break
        if key and representative == key and key not in representatives:
            representatives.append(key)
        keys.append(representative)
    return keys


def infer_repeated_sections(sections: list[dict]) -> list[dict]:
    """Split generic verses around repeated blocks of complete lines.

    Complete repeated blocks establish the reference; conservative phrase
    variations may recover ASR boundaries. A single repeated phrase cannot
    establish a chorus by itself. No fixed line count or lyric repair is used.
    Existing verse boundaries survive outside an inferred refrain. Returns the
    input unchanged if there is no defensible new grouping.
    """
    if not sections or any(not section.get("lines") for section in sections):
        return sections
    lines, sources = _flatten(sections)
    keys = _phrase_keys(lines, sources, sections)
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
            # Two adjacent repeats alone cannot distinguish an opening verse
            # repeated twice from a chorus. Retain the existing labels in that
            # ambiguous case; acoustic labels are handled separately.
            adjacent_pair = (len(starts) == 2 and not any(
                line.get("lyrics", "").strip()
                for line in lines[starts[0] + block_length:starts[1]]
            ))
            if _meaningful_refrain(extended):
                candidates.append((block_length * len(starts), block_length, starts, not adjacent_pair))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2][0]))
    occupied: set[int] = set()
    refrains: dict[int, tuple[int, bool]] = {}
    families = []
    for _, length, starts, chorus_candidate in candidates:
        available = [
            start for start in starts
            if all(index not in occupied for index in range(start, start + length))
        ]
        if len(available) < 2:
            continue
        for start in available:
            refrains[start] = (start + length, chorus_candidate)
            occupied.update(range(start, start + length))
        families.append((available, length, chorus_candidate))
    if not refrains:
        return sections

    # The widest recurring complete family is the chorus reference. Secondary
    # short verse fragments must not create additional chorus boundaries.
    eligible = [family for family in families if family[2]]
    dominant = max(eligible, key=lambda f: (f[0][-1] - f[0][0]) * f[1], default=None)
    bridges, outros = {}, {}
    if dominant:
        starts, length, _ = dominant
        pattern = tuple(keys[starts[0]:starts[0] + length])
        core_ranges = _pattern_ranges(keys, pattern)
        # Apply the dominant-family filter only with at least three complete
        # occurrences. Two refrains alone cannot establish competing roles.
        established = len(core_ranges) >= 3
        if established:
            refrains = {start: (end, True) for start, end in core_ranges}
            for other_starts, other_length, candidate in families:
                if not candidate:
                    for start in other_starts:
                        refrains.setdefault(start, (start + other_length, False))
            # Recover a repeated opening verse with complete phrase evidence.
            # Its final repetition may include additional lines before the chorus.
            opening = next((i for i, key in enumerate(keys) if key), 0)
            first_chorus = core_ranges[0][0]
            for size in range(3, min(8, first_chorus - opening) + 1):
                verse_pattern = tuple(keys[opening:opening + size])
                if not all(verse_pattern) or len(set(verse_pattern)) < 3:
                    continue
                verse_ranges = _pattern_ranges(keys[:first_chorus], verse_pattern)
                if (len(verse_ranges) == 2 and verse_ranges[0][0] == opening
                        and verse_ranges[0][1] == verse_ranges[1][0]):
                    first, second = verse_ranges
                    refrains[first[0]] = (first[1], False)
                    refrains[second[0]] = (first_chorus, False)
                    break
            # Attach a recurrent short closing phrase learned immediately after
            # complete choruses; do not absorb arbitrary adjacent verse lyrics.
            endings = defaultdict(list)
            for start, end in core_ranges:
                for count in (1, 2):
                    parts = keys[end:end + count]
                    phrase = tuple(word for part in parts for word in part)
                    if (len(parts) != count or not all(parts) or not 2 <= len(phrase) <= 4
                            or any(i in refrains for i in range(end, end + count))
                            or any(_similar_phrase(phrase, core) for core in pattern)):
                        continue
                    gap = lines[end].get("_startTime", 0) - lines[end - 1].get("_endTime", 0)
                    if gap <= 6:
                        endings[phrase].append((start, end, count))
            for phrase, occurrences in endings.items():
                support = {start for other, spans in endings.items()
                           if _similar_phrase(phrase, other, .8)
                           for start, _, _ in spans}
                if len(support) >= 2:
                    for start, end, count in occurrences:
                        refrains[start] = (max(refrains[start][0], end + count), True)
            for other_starts, other_length, candidate in families:
                if not candidate or len(other_starts) < 3 or other_starts == starts:
                    continue
                end = other_starts[-1] + other_length
                compact = all(b - a <= other_length + 2 for a, b in zip(other_starts, other_starts[1:]))
                surrounded = any(s < other_starts[0] for s, _ in core_ranges) and any(s >= end for s, _ in core_ranges)
                if not compact or not surrounded or not all(keys[i] for i in range(other_starts[0], end)):
                    continue
                bridge_pattern = keys[other_starts[0]:other_starts[0] + other_length]
                cursor = 0
                while end < len(keys) and keys[end] and end not in refrains:
                    expected = bridge_pattern[cursor % other_length]
                    actual = keys[end]
                    partial = len(actual) >= 1 and len(actual) < len(expected) and tuple(expected[:len(actual)]) == actual
                    if actual != expected and not partial:
                        break
                    end += 1
                    cursor += 1
                bridges[other_starts[0]] = end
            # A repeated chorus phrase after the last complete chorus may be an
            # outro. Require three full repeats; retain intervening fragments.
            last_start, _ = core_ranges[-1]
            tail = refrains[last_start][0]
            for phrase in set(pattern):
                end, repeats = tail, 0
                while end < len(keys) and keys[end]:
                    actual = keys[end]
                    if _similar_phrase(phrase, actual):
                        repeats += 1
                    elif not (len(actual) <= 3 and set(actual) <= set(phrase)):
                        break
                    end += 1
                if repeats >= 3:
                    outros[tail] = end
                    break

    result, index, verse_number = [], 0, 0
    while index < len(lines):
        source = sources[index]
        original = sections[source]
        is_refrain = index in refrains
        end, chorus_candidate = refrains.get(index, (index + 1, False))
        is_bridge = index in bridges
        is_outro = index in outros
        if is_outro:
            end = outros[index]
        elif is_bridge:
            end = bridges[index]
        elif not is_refrain:
            while end < len(lines) and sources[end] == source and end not in refrains and end not in bridges and end not in outros:
                end += 1
        section = {key: deepcopy(value) for key, value in original.items() if key != "lines"}
        section["lines"] = deepcopy(lines[index:end])
        if is_outro:
            section["name"] = "Outro"
        elif is_bridge:
            section["name"] = "Puente"
        elif is_refrain and chorus_candidate:
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
