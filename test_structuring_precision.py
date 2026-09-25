"""Regression checks for lossless offline and optional editorial structuring."""

from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

import structuring


def line(text, ordinal=0):
    return {
        "lyrics": text,
        "chords": [{"chord": "Cmaj7", "charIndex": 0, "_time": ordinal * 2.0}],
        "timestamps": [{"time": ordinal * 2.0, "order": ordinal + 1}],
        "_startTime": ordinal * 2.0,
        "_endTime": ordinal * 2.0 + 1.9,
    }


def song(texts):
    return [{"name": "Verso 1", "lines": [line(text, i) for i, text in enumerate(texts)]}]


def flatten(sections):
    return [item for section in sections for item in section["lines"]]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_STRUCTURE", "0")


def mock_llm(monkeypatch, proposal):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"secciones": proposal}, ensure_ascii=False),
        ))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **_: client))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used-on-network")
    monkeypatch.setenv("LLM_STRUCTURE", "1")
    return requests


def proposed(name, *ids, texts=None):
    result = []
    for index, ordinal in enumerate(ids):
        item = {"id": f"L{ordinal:06d}"}
        if texts is not None:
            item["lyrics"] = texts[index]
        result.append(item)
    return {"name": name, "lines": result}


REFRAIN = ["Tu amor sostiene mi camino", "Tu luz ilumina nuestra vida"]


def test_repeated_block_becomes_chorus_without_pauses_or_api():
    sections = song(["Camino por la tierra", *REFRAIN, "Hoy mi voz te canta", *REFRAIN])
    before = deepcopy(sections)
    result = structuring.apply_structure(sections, "C", "major")
    assert [section["name"] for section in result] == ["Verso 1", "Coro", "Verso 2", "Coro"]
    assert [len(section["lines"]) for section in result] == [1, 2, 1, 2]
    assert flatten(result) == flatten(before)
    assert sections == before
    result[1]["lines"][0]["chords"][0]["chord"] = "G"
    assert sections == before


def test_all_adjacent_refrain_repetitions_remain_separate_and_in_order():
    sections = song(["Comienza esta cancion", *REFRAIN, *REFRAIN, *REFRAIN, *REFRAIN])
    result = structuring.apply_structure(sections, "C", "major")
    assert [len(section["lines"]) for section in result] == [1, 2, 2, 2, 2]
    assert [section["name"] for section in result].count("Coro") == 4
    assert flatten(result) == flatten(sections)


def test_long_repeated_block_is_not_cut_at_a_fixed_line_count():
    refrain = [f"Cantamos la frase numero {i} juntos" for i in range(20)]
    sections = song(["La estrofa comienza aqui", *refrain, "La segunda estrofa es distinta", *refrain])
    result = structuring.apply_structure(sections, "C", "major")
    assert [len(section["lines"]) for section in result] == [1, 20, 1, 20]
    assert flatten(result) == flatten(sections)


@pytest.mark.parametrize("texts", [
    ["Aleluya"] * 12,
    ["Santo santo santo"] * 10,
    ["Ven", "Senor", "Ven", "Senor"],
    ["Tu amor sostiene mi camino", "Una estrofa diferente", "Tu amor sostiene mi camino"],
    ["Tu amor sostiene mi camino", "Tu luz ilumina nuestra vida", "Solo esta vez"],
])
def test_single_repeated_word_or_line_is_not_assumed_chorus(texts):
    sections = song(texts)
    assert structuring.apply_structure(sections, "C", "major") is sections


def test_punctuation_case_and_accents_do_not_hide_a_repetition():
    sections = song([
        "Dame tu mano senor", "Canta conmigo esta cancion",
        "Hoy mi voz te busca",
        "¡Dame tu mano, Señor!", "Canta conmigo esta canción",
    ])
    result = structuring.apply_structure(sections, "C", "major")
    assert [section["name"] for section in result] == ["Coro", "Verso 1", "Coro"]
    assert flatten(result) == flatten(sections)


def test_existing_verse_boundaries_survive_and_refrain_may_cross_pause_boundaries():
    sections = [
        {"name": "Verso 1", "lines": [line("Primer verso diferente"), line(REFRAIN[0], 1)]},
        {"name": "Verso 2", "lines": [line(REFRAIN[1], 2)]},
        {"name": "Verso 3", "lines": [line("Otro verso distinto", 3)]},
        {"name": "Verso 4", "lines": [line("Continua esta estrofa", 4), line(REFRAIN[0], 5), line(REFRAIN[1], 6)]},
    ]
    result = structuring.apply_structure(sections, "C", "major")
    assert [section["name"] for section in result] == ["Verso 1", "Coro", "Verso 2", "Verso 3", "Coro"]
    assert flatten(result) == flatten(sections)


def test_instrumental_sections_and_all_chords_survive_grouping():
    intro = {"name": "Intro", "lines": [line("", 0)]}
    bridge = {"name": "Instrumental", "lines": [line("", 5)]}
    sections = [intro, {"name": "Verso 1", "lines": [line(text, i + 1) for i, text in enumerate(REFRAIN)]},
                bridge, {"name": "Verso 2", "lines": [line(text, i + 6) for i, text in enumerate(REFRAIN)]}]
    result = structuring.apply_structure(sections, "C", "major")
    assert [section["name"] for section in result] == ["Intro", "Coro", "Instrumental", "Coro"]
    assert result[0] == intro and result[2] == bridge
    assert flatten(result) == flatten(sections)


def test_no_refrain_can_cross_a_silent_or_instrumental_line():
    sections = song([REFRAIN[0], "", REFRAIN[1], "Otro verso", REFRAIN[0], "", REFRAIN[1]])
    assert structuring.apply_structure(sections, "C", "major") is sections


def test_acoustic_boundaries_and_labels_are_not_reinferred():
    sections = song(["Entrada diferente", *REFRAIN, "Salida diferente", *REFRAIN])
    assert structuring.apply_structure(sections, "C", "major", preserve_boundaries=True) is sections


def test_existing_semantic_labels_are_kept_without_acoustic_flag():
    sections = [{"name": "Puente", "lines": [line(text, i) for i, text in enumerate(REFRAIN * 2)]}]
    assert structuring.apply_structure(sections, "C", "major") is sections


def test_spacing_does_not_change_musical_character_anchors():
    chords = [{"chord": "Cmaj7", "charIndex": 5}, {"chord": "G", "charIndex": 6}, {"chord": "D", "charIndex": 6}]
    assert structuring.respace(chords) is chords
    assert [item["charIndex"] for item in chords] == [5, 6, 6]


def test_editorial_remapping_follows_characters_instead_of_line_length():
    old = "senor ten piedad de mi"
    new = "¡Señor! Ten piedad de mí."
    chords = [{"chord": "C", "charIndex": 0},
              {"chord": "G", "charIndex": old.index("piedad"), "_time": 4.2},
              {"chord": "Am", "charIndex": len(old)}]
    result = structuring.remap_chords(chords, old, new)
    assert [item["charIndex"] for item in result] == [new.index("S"), new.index("piedad"), len(new)]
    assert result[1]["_time"] == 4.2
    assert chords[1]["charIndex"] == old.index("piedad")


def test_decomposed_accent_is_mapped_without_drifting_the_next_word():
    old = "mi cancio\u0301n vive"
    new = "Mi canción vive"
    result = structuring.remap_chords([{"chord": "C", "charIndex": old.index("vive")}], old, new)
    assert result[0]["charIndex"] == new.index("vive")


def test_punctuation_after_anchor_does_not_shift_that_anchor():
    old = "la voz del senor"
    new = "La voz del Señor..."
    result = structuring.remap_chords([{"chord": "G", "charIndex": 3}], old, new)
    assert result[0]["charIndex"] == 3


@pytest.mark.parametrize("corrected", [
    "Señor ten piedad de todos", "Ten Señor piedad de mi", "Señor ten piedad",
    "Señor ten gran piedad de mi", "Señorten piedad de mi", "Señor\nten piedad de mi",
])
def test_lexical_rewrites_are_rejected_by_remapper(corrected):
    with pytest.raises(ValueError):
        structuring.remap_chords([{"chord": "G", "charIndex": 0}], "senor ten piedad de mi", corrected)


def test_llm_can_split_and_merge_sections_by_ids_without_touching_original(monkeypatch):
    sections = song(["senor ten piedad de mi", "tu luz me acompana", "hoy camino contigo", "mi esperanza sigue viva"])
    before = deepcopy(sections)
    requests = mock_llm(monkeypatch, [
        proposed("Verso 1", 1, texts=["¡Señor, ten piedad de mí!"]),
        proposed("Coro", 2, 3), proposed("Final", 4),
    ])
    result = structuring.apply_structure(sections, "C", "major")
    assert [len(section["lines"]) for section in result] == [1, 2, 1]
    assert sections == before
    assert flatten(result)[0]["lyrics"] == "¡Señor, ten piedad de mí!"
    for original, updated in zip(flatten(before), flatten(result)):
        assert updated["timestamps"] == original["timestamps"]
        assert updated["_startTime"] == original["_startTime"]
        assert updated["_endTime"] == original["_endTime"]
        assert updated["chords"][0]["_time"] == original["chords"][0]["_time"]
    payload = json.loads(requests[0]["messages"][1]["content"])
    assert [item["id"] for item in payload["secciones"][0]["lines"]] == ["L000001", "L000002", "L000003", "L000004"]

    sections2 = [{"name": "Verso 1", "lines": [line("primera frase cantada")]},
                 {"name": "Verso 2", "lines": [line("segunda frase distinta", 1)]}]
    mock_llm(monkeypatch, [proposed("Verso 1", 1, 2)])
    merged = structuring.apply_structure(sections2, "C", "major")
    assert len(merged) == 1 and flatten(merged) == flatten(sections2)


@pytest.mark.parametrize("proposal", [
    [proposed("Verso 1", 1)],
    [proposed("Verso 1", 1, 1)],
    [proposed("Verso 1", 2, 1)],
    [proposed("Verso 1", 1, 99)],
    [{"name": "Coro", "lines": ["primera linea", "segunda linea"]}],
    [proposed("Nombre inventado", 1, 2)],
    [{"name": "Coro", "lines": []}],
    [proposed("Verso 1", 1, texts=["¡Primera línea!"]), proposed("Coro", 2, texts=["letra inventada"])],
])
def test_bad_llm_response_is_rejected_atomically(monkeypatch, proposal):
    sections = song(["primera linea", "segunda linea"])
    before = deepcopy(sections)
    mock_llm(monkeypatch, proposal)
    assert structuring.apply_structure(sections, "C", "major") is sections
    assert sections == before


def test_bad_llm_response_keeps_successful_offline_grouping(monkeypatch):
    sections = song(["Una estrofa distinta", *REFRAIN, "Otra estrofa diferente", *REFRAIN])
    before = deepcopy(sections)
    mock_llm(monkeypatch, [proposed("Verso 1", 1)])
    result = structuring.apply_structure(sections, "C", "major")
    assert [section["name"] for section in result] == ["Verso 1", "Coro", "Verso 2", "Coro"]
    assert flatten(result) == flatten(sections) and sections == before


def test_acoustic_flag_allows_safe_text_edits_but_preserves_names_and_groups(monkeypatch):
    sections = [{"name": "Verso 7", "lines": [line("senor ten piedad")]},
                {"name": "Coro 2", "lines": [line("ilumina nuestro camino", 1)]}]
    before = deepcopy(sections)
    mock_llm(monkeypatch, [proposed("Puente", 1, texts=["Señor, ten piedad"]), proposed("Final", 2)])
    result = structuring.apply_structure(sections, "C", "major", preserve_boundaries=True)
    assert [section["name"] for section in result] == ["Verso 7", "Coro 2"]
    assert result[0]["lines"][0]["lyrics"] == "Señor, ten piedad"
    assert sections == before
    mock_llm(monkeypatch, [proposed("Verso 1", 1, 2)])
    assert structuring.apply_structure(sections, "C", "major", preserve_boundaries=True) is sections


def test_llm_cannot_absorb_an_instrumental_section(monkeypatch):
    sections = [{"name": "Intro", "lines": [line("")]},
                {"name": "Verso 1", "lines": [line("cantamos esta cancion", 1)]}]
    mock_llm(monkeypatch, [proposed("Verso 1", 1, 2)])
    assert structuring.apply_structure(sections, "C", "major") is sections
    mock_llm(monkeypatch, [proposed("Intro", 1, texts=["musica"]), proposed("Verso 1", 2)])
    assert structuring.apply_structure(sections, "C", "major") is sections


def test_empty_sections_are_preserved_defensively():
    sections = [{"name": "Instrumental", "lines": []}, *song(REFRAIN * 2)]
    assert structuring.apply_structure(sections, "C", "major") is sections


def test_llm_disabled_never_constructs_client(monkeypatch):
    def fail(**_):
        raise AssertionError("No debe llamar la API")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fail))
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    sections = song(REFRAIN * 2)
    result = structuring.apply_structure(sections, "C", "major")
    assert [section["name"] for section in result] == ["Coro", "Coro"]


def test_paid_editorial_pass_is_opt_in_even_with_transcription_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    monkeypatch.delenv("LLM_STRUCTURE", raising=False)
    # Importing a provider would fail: a present transcription key is not opt-in.
    import sys
    monkeypatch.setitem(sys.modules, "openai", None)
    original = [{"name": "Verso 1", "lines": [{"lyrics": "Texto original", "chords": [], "timestamps": []}]}]
    assert structuring.apply_structure(original, "C", "major") is original
