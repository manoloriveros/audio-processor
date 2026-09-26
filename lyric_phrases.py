"""Recording-wide phrase boundaries learned from measured, repeated lyrics.

No song-specific words, spelling repairs or estimated word timestamps. Existing
segments without complete word coverage remain intact. Reflow keeps every word.
"""
from collections import defaultdict
from copy import deepcopy
import re

from timeline import word_positions, _word_key


def repeated_phrase_segments(segments, words):
    if not words or len(segments) < 3:
        return segments
    runs, current, seeds = [], [], set()
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        tokens, mapped = word_positions(text, segment["start"], segment["end"], words)
        valid = (tokens and len(mapped) == len(tokens) and not segment.get("timing_estimated"))
        if not valid:
            if current:
                runs.append((True, current))
                current = []
            runs.append((False, [segment]))
            continue
        entries = [{**mapped[i], "word": token.group(), "_segment": segment}
                   for i, token in enumerate(tokens)]
        if current and entries[0]["start"] - current[-1]["end"] >= 8:
            runs.append((True, current))
            current = []
        current.extend(entries)
        key = tuple(_word_key(token.group()) for token in tokens)
        if 2 <= len(key) <= 12 and all(key) and segment["end"] - segment["start"] <= 14:
            seeds.add(key)
    if current:
        runs.append((True, current))
    matches = defaultdict(list)
    for run_id, (valid, entries) in enumerate(runs):
        if not valid:
            continue
        keys = [_word_key(entry["word"]) for entry in entries]
        for seed in seeds:
            for start in range(len(keys) - len(seed) + 1):
                end = start + len(seed)
                if tuple(keys[start:end]) == seed and entries[end - 1]["end"] - entries[start]["start"] <= 14:
                    matches[seed].append((run_id, start, end))
    repeated = {seed: spans for seed, spans in matches.items() if len(spans) >= 2}
    # One repeated short phrase is not sufficient evidence to reflow a song.
    if len(repeated) < 2:
        return segments
    # Prefer the shorter complete source phrases when they tile a compound
    # phrase. This keeps a repeated two-line ASR segment as two sung phrases.
    primitive = {}
    for seed, spans in repeated.items():
        reachable = {0}
        for index in range(len(seed)):
            if index not in reachable:
                continue
            for other in repeated:
                if len(other) < len(seed) and seed[index:index + len(other)] == other:
                    reachable.add(index + len(other))
        compound = len(seed) in reachable
        if not compound:
            for other in repeated:
                if not 3 <= len(other) < len(seed):
                    continue
                for prefix in (True, False):
                    if (seed[:len(other)] if prefix else seed[-len(other):]) != other:
                        continue
                    remainder = seed[len(other):] if prefix else seed[:-len(other)]
                    if any(len(remainder) < len(part) < len(seed)
                           and (part[:len(remainder)] if prefix else part[-len(remainder):]) == remainder
                           for part in repeated):
                        compound = True
                        break
                if compound:
                    break
        if not compound:
            primitive[seed] = spans
    selected = defaultdict(list)
    occupied = defaultdict(set)
    for seed, spans in sorted(primitive.items(), key=lambda item: (-len(item[0]) * len(item[1]), -len(item[0]), item[0])):
        for run_id, start, end in spans:
            if (start and _word_key(runs[run_id][1][start - 1]["word"]) in {"porque", "y", "oh", "ah"}
                    and runs[run_id][1][start - 1]["_segment"] is runs[run_id][1][start]["_segment"]
                    and start - 1 not in occupied[run_id]):
                start -= 1
            if not any(i in occupied[run_id] for i in range(start, end)):
                selected[run_id].append((start, end))
                occupied[run_id].update(range(start, end))
    output = []
    for run_id, (valid, entries) in enumerate(runs):
        if not valid:
            output.extend(deepcopy(entries))
            continue
        intervals = sorted(selected[run_id])
        protected_cuts = {i for start, end in intervals for i in range(start + 1, end)}
        cuts = {0, len(entries), *(i for span in intervals for i in span)}
        for i in range(1, len(entries)):
            if i in protected_cuts:
                continue
            if (entries[i]["_segment"] is not entries[i - 1]["_segment"]
                    or entries[i]["start"] - entries[i - 1]["end"] >= 1.3
                    or entries[i - 1]["word"].endswith((".", ",", ";", "!", "?"))):
                cuts.add(i)
        cuts = sorted(cuts)
        for start, end in zip(cuts, cuts[1:]):
            group = entries[start:end]
            original = group[0]["_segment"]
            output.append({**original, "text": " ".join(entry["word"] for entry in group),
                           "start": group[0]["start"], "end": max(entry["end"] for entry in group),
                           "_repeated_phrase": (start, end) in intervals})
    before = [_word_key(token) for segment in segments for token in str(segment.get("text", "")).split()]
    after = [_word_key(token) for segment in output for token in segment["text"].split()]
    if before != after:
        raise ValueError("Phrase reflow would change the lyric word sequence")
    return output


_CREDIT = re.compile(r"^(?:subtitulos\s+(?:realizados|creados|hechos)\s+por\s+.+|"
                     r"subtitulos\s+(?:realizados|creados|hechos)\s+(?:amara\s+)?org|"
                     r"subtitulos\s+de\s+la\s+comunidad\s+de\s+amara\s+org|"
                     r"(?:muchas\s+)?gracias\s+(?:por\s+)?ver\s+(?:el|este)\s+video)$")


def quarantine_credit_segments(transcript):
    """Quarantine explicit ASR credit lines near the end; keep review evidence.

    A word like 'subtítulos' alone is never filtered. This is a narrow textual
    heuristic, not proof that a phrase is absent from the audio.
    """
    duration = transcript.get("duration")
    if not isinstance(duration, (float, int)) or duration <= 0:
        return transcript
    removed, retained = [], []
    for segment in transcript.get("segments", []):
        key = " ".join(re.findall(r"[^\W_]+", _word_key_text(segment.get("text", ""))))
        if segment["start"] >= max(duration * .9, duration - 45) and _CREDIT.fullmatch(key):
            removed.append({**segment, "reason": "possible_asr_credits"})
        else:
            retained.append(segment)
    if not removed:
        return transcript
    words = [word for word in transcript.get("words", []) if not any(
        segment["start"] <= (word["start"] + word["end"]) / 2 < segment["end"] for segment in removed)]
    return {**transcript, "segments": retained, "words": words,
            "text": "\n".join(segment["text"] for segment in retained),
            "reviewSegments": [*transcript.get("reviewSegments", []), *removed],
            "warnings": [*transcript.get("warnings", []), "possible_asr_credits_quarantined"]}


def _word_key_text(text):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", str(text).casefold())
                   if not unicodedata.combining(c))
