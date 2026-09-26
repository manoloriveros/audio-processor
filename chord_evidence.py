"""Retain sustained seventh evidence when the neural model emits a triad.

This is detector evidence, not a claim of a calibrated confidence or a verified
transcription. No key-based substitution, global simplification or beat snapping.
"""
import re

_BASIC = re.compile(r"^([A-G][#b]?)(m?)$")
_SEVENTH = re.compile(r"^([A-G][#b]?)(maj7|mMaj7|m7|7)$")


def retain_supported_sevenths(primary, secondary):
    """Add only same-root, same-third sevenths supported for >=0.8 seconds.

    Neural sevenths, suspensions, inversions and short changes stay untouched.
    Intersections retain each detector's measured boundary; an extension never
    moves to the beginning of an entire phrase. Sixths are not inferred here.
    """
    result = []
    for event in primary:
        basic = _BASIC.fullmatch(event["chord"])
        if not basic or "end" not in event:
            result.append(dict(event))
            continue
        cursor = event["time"]
        for support in secondary:
            if "end" not in support or support["end"] <= cursor:
                continue
            if support["time"] >= event["end"]:
                break
            rich = _SEVENTH.fullmatch(support["chord"])
            if not rich or basic[1] != rich[1]:
                continue
            minor = rich[2] in {"m7", "mMaj7"}
            if bool(basic[2]) != minor:
                continue
            start = max(cursor, support["time"])
            end = min(event["end"], support["end"])
            if end - start < .8:
                continue
            if start > cursor:
                result.append({**event, "time": cursor, "end": start})
            result.append({**event, "chord": support["chord"], "time": start, "end": end,
                           "extensionEngine": "chordino", "extensionNeedsReview": True,
                           "primaryChord": event["chord"]})
            cursor = end
        if cursor < event["end"]:
            result.append({**event, "time": cursor})
    return result
