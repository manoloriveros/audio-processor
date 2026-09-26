"""Lossless timeline transforms between detector events and editor lines.

Detector times remain authoritative. Display spacing must never change a lyric
anchor; only empty instrumental lines use spaced character positions.
"""

import math
import re
import unicodedata
from bisect import bisect_right
from difflib import SequenceMatcher

# Short vocal rests occur inside musical phrases. This is a conservative layout
# threshold, not an acoustic verse/chorus classifier.
SECTION_REST_SECONDS = 8.0


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def normalize_events(events, duration=None):
    """Keep short changes and explicit silence, filling only known interval ends."""
    ordered = []
    for event in events:
        start = _number(event.get("time", event.get("start")))
        if start is None or start < 0 or "chord" not in event:
            continue
        chord = event["chord"]
        if chord is None or str(chord).upper() in {"N", "NONE", "NOCHORD"}:
            chord = "N"
        entry = {**event, "chord": chord, "time": round(start, 6)}
        end = _number(event.get("end"))
        entry.pop("end", None)
        if end is not None and end >= start:
            entry["end"] = round(end, 6)
        ordered.append(entry)
    ordered.sort(key=lambda item: item["time"])
    unique = []
    for item in ordered:
        if unique and all(unique[-1].get(key) == item.get(key) for key in ("time", "end", "chord")):
            continue
        unique.append(item)
    result = []
    for index, item in enumerate(unique):
        next_start = unique[index + 1]["time"] if index + 1 < len(unique) else _number(duration)
        end = item.get("end")
        if next_start is not None:
            end = min(end, next_start) if end is not None else next_start
        if end is not None:
            item["end"] = max(item["time"], end)
        result.append(item)
        if next_start is not None and end is not None and end < next_start and item["chord"] != "N":
            result.append({"chord": "N", "time": end, "end": next_start})
    return result


def _word_key(text):
    text = unicodedata.normalize("NFD", str(text).casefold())
    return "".join(char for char in text if char.isalnum() and not unicodedata.combining(char))


def word_positions(text, start, end, words):
    tokens = list(re.finditer(r"\S+", text))
    candidates = []
    for word in words:
        word_start, word_end = _number(word.get("start")), _number(word.get("end"))
        # A word ending exactly at a section boundary belongs to the preceding
        # phrase. Including it can attach an identical repeated word twice.
        if word_start is not None and word_end is not None and word_end > start and word_start < end:
            candidates.append({**word, "start": word_start, "end": word_end})
    candidates.sort(key=lambda word: (word["start"], word["end"]))
    matcher = SequenceMatcher(None, [_word_key(token.group()) for token in tokens],
                              [_word_key(word.get("word", "")) for word in candidates], autojunk=False)
    mapped = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            token_index = block.a + offset
            token = tokens[token_index]
            word = candidates[block.b + offset]
            mapped[token_index] = {**word, "char_start": token.start(), "char_end": token.end()}
    return tokens, mapped


def split_segments(segments, words=(), max_len=40, min_len=15):
    """Split at actual word boundaries; retain unsplittable untimed phrases."""
    result = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        start, end = _number(segment.get("start")), _number(segment.get("end"))
        if not text or start is None or end is None:
            continue
        tokens, mapped = word_positions(text, start, end, words)
        cuts = [0]
        for index in range(1, len(tokens)) if not segment.get("_repeated_phrase") else ():
            if index not in mapped or index - 1 not in mapped:
                continue
            first = tokens[cuts[-1]].start()
            previous_length = tokens[index - 1].end() - first
            candidate_length = tokens[index].end() - first
            gap = mapped[index]["start"] - mapped[index - 1]["end"]
            punctuation = tokens[index - 1].group().endswith((".", ",", ";", "!", "?"))
            if ((candidate_length > max_len and previous_length >= min_len)
                    or (gap >= 0.8 and previous_length >= 4)
                    or (punctuation and previous_length >= min_len and len(text) - tokens[index].start() >= min_len)):
                cuts.append(index)
        cuts.append(len(tokens))
        for left, right in zip(cuts, cuts[1:]):
            if left == right:
                continue
            char_start, char_end = tokens[left].start(), tokens[right - 1].end()
            line_start = float(mapped[left]["start"]) if left in mapped else start
            line_end = float(mapped[right - 1]["end"]) if right - 1 in mapped else end
            line_words = [{**mapped[index], "char_start": mapped[index]["char_start"] - char_start,
                           "char_end": mapped[index]["char_end"] - char_start}
                          for index in range(left, right) if index in mapped]
            result.append({**segment, "text": text[char_start:char_end], "start": line_start,
                           "end": max(line_start, line_end), "_words": line_words})
    return result


def time_to_char_index(time, text, start, end, words):
    if not text:
        return 0
    if words and all("char_start" in word for word in words):
        positions = words
    else:
        _, mapped = word_positions(text, start, end, words)
        positions = list(mapped.values())
    if not positions:
        fraction = (time - start) / max(end - start, 0.001)
        return max(0, min(int(fraction * len(text)), len(text)))
    for word in positions:
        if time <= word["start"]:
            return max(0, min(word["char_start"], len(text)))
        if time < word["end"]:
            fraction = (time - word["start"]) / max(word["end"] - word["start"], 0.001)
            return min(len(text), word["char_start"] + int(fraction * (word["char_end"] - word["char_start"])))
    return min(len(text), positions[-1]["char_end"])


def build_sections(segments, events):
    """Preserve instrumental events and silence without moving lyric anchors."""
    segments = sorted(segments, key=lambda segment: segment["start"])
    events = normalize_events(events)
    starts = [event["time"] for event in events]

    def placed_chord(event, anchor, source):
        # Keep the measured onset even when presentation repeats a held chord or
        # places a rest at the end of a line. charIndex is not an audio timestamp.
        return {"chord": event["chord"], "charIndex": anchor,
                "audioTime": event["time"], "alignmentSource": source,
                **({"audioEnd": event["end"]} if "end" in event else {}),
                **({"extensionNeedsReview": True} if event.get("extensionNeedsReview") else {})}

    def in_span(start, end):
        index = bisect_right(starts, start) - 1
        chosen = []
        if index >= 0:
            active = events[index]
            if active["chord"] != "N" and active.get("end", float("inf")) > start:
                chosen.append(active)
        for event in events[index + 1:]:
            if event["time"] >= end:
                break
            if event["chord"] != "N":
                chosen.append(event)
        return chosen

    def instrumental(start, end, name="Instrumental"):
        chosen = in_span(start, end)
        if not chosen:
            return None
        lines = []
        for offset in range(0, len(chosen), 8):
            group = chosen[offset:offset + 8]
            cursor, chords = 0, []
            for event in group:
                chords.append(placed_chord(event, cursor, "instrumental"))
                cursor += len(event["chord"]) + 2
            line_end = chosen[offset + 8]["time"] if offset + 8 < len(chosen) else end
            lines.append({"lyrics": "", "chords": chords, "timestamps": [],
                          "_startTime": max(start, group[0]["time"]), "_endTime": line_end})
        return {"name": name, "lines": lines}

    known_end = max((event.get("end", event["time"] + 0.001) for event in events), default=0)
    if not segments:
        section = instrumental(0, known_end, "Instrumental")
        return [section] if section else []

    sections, current = [], []
    verse = 0
    last_rendered_event = None

    def flush():
        nonlocal current, verse
        if current:
            verse += 1
            sections.append({"name": f"Verso {verse}", "lines": current})
            current = []

    first_start = segments[0]["start"]
    intro_events = in_span(0, first_start)
    if intro_events and (first_start - intro_events[0]["time"] >= 2 or len(intro_events) > 1):
        intro = instrumental(0, first_start, "Intro")
        if intro:
            sections.append(intro)

    for index, segment in enumerate(segments):
        start, end = segment["start"], segment["end"]
        next_start = segments[index + 1]["start"] if index + 1 < len(segments) else max(end, known_end)
        is_last = index + 1 == len(segments)
        gap_duration = next_start - end
        separate_gap = bool(in_span(end, next_start)) and gap_duration >= SECTION_REST_SECONDS
        chord_end = end if separate_gap else max(end, next_start)
        chords = []
        for event in in_span(start, chord_end):
            # An ongoing chord was already shown on the preceding lyric line.
            # Do not turn one held chord into many apparent harmonic changes.
            if (current and event["time"] < start
                    and last_rendered_event == (event["time"], event["chord"])):
                continue
            last_rendered_event = (event["time"], event["chord"])
            if event["time"] <= start:
                anchor = 0
                source = "held" if event["time"] < start else "line-start"
            elif event["time"] >= end:
                anchor = len(segment["text"])
                source = "vocal-rest"
            else:
                anchor = time_to_char_index(event["time"], segment["text"], start, end, segment.get("_words", []))
                source = "word-timestamps" if segment.get("_words") else "segment-estimate"
            chords.append(placed_chord(event, anchor, source))
        current.append({"lyrics": segment["text"], "chords": chords, "timestamps": [],
                        "_startTime": start, "_endTime": end,
                        **{key: segment[key] for key in ("timing_estimated", "timestamp_source")
                           if key in segment}})
        if separate_gap:
            flush()
            gap = instrumental(end, next_start, "Final" if is_last else "Instrumental")
            if gap:
                sections.append(gap)
        elif not is_last and gap_duration >= SECTION_REST_SECONDS:
            flush()
    flush()
    return sections
