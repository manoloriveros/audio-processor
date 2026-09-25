"""Conservative text correction on an existing, measured word timeline."""

import re
import unicodedata
from difflib import SequenceMatcher


def normalize_word(value):
    folded = unicodedata.normalize("NFD", str(value).casefold())
    return re.sub(r"[^a-z0-9]", "", "".join(
        char for char in folded if unicodedata.category(char) != "Mn"
    ))


def align_corrected_words(corrected_text, timed_words):
    """Apply only corrections that retain a one-to-one, ordered word mapping.

    Text comparison cannot locate an inserted word or identify which repeated
    chorus was removed. In those cases keep the measured transcript instead of
    inventing timestamps or deleting entire occurrences. This is token mapping,
    not acoustic forced alignment.
    """
    tokens = str(corrected_text or "").split()
    if not timed_words or not tokens:
        return timed_words, False
    original = [normalize_word(item["word"]) for item in timed_words]
    corrected = [normalize_word(token) for token in tokens]
    if len(original) != len(corrected):
        return timed_words, False
    matcher = SequenceMatcher(None, original, corrected, autojunk=False)
    edits = matcher.get_opcodes()
    if matcher.ratio() < 0.65 or any(
        tag in {"insert", "delete"} or i2 - i1 != j2 - j1
        for tag, i1, i2, j1, j2 in edits
    ):
        return timed_words, False
    return [
        {**word, "word": token}
        for word, token in zip(timed_words, tokens)
    ], True
