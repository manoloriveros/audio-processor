"""Tests del pipeline de audio (sin red, sin OpenAI, sin audio real).

Ejecutar:  python test_pipeline.py   (o pytest test_pipeline.py -q)
"""

import os
import sys

os.environ.setdefault("LLM_STRUCTURE", "0")  # sin llamadas LLM en tests
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("MUSIC_AI_API_KEY", None)

import main  # noqa: E402
import musicai_engine as me  # noqa: E402
import structuring  # noqa: E402


def test_normalize_chord_labels():
    cases = {
        "C:maj": "C", "D:min": "Dm", "D:min7": "Dm7", "G:7": "G7",
        "Bb:maj": "A#", "F#:min": "F#m", "N": None, "noChord": None,
        "C/E": "C/E", "Cmaj7": "Cmaj7", "A:sus4": "Asus4",
        "E:min/5": "Em",  # bajo por grado se descarta con seguridad
    }
    for raw, expected in cases.items():
        got = main._normalize_chord_label(raw)
        assert got == expected, f"{raw}: esperaba {expected}, obtuve {got}"


def test_parse_chords_shapes():
    payload_a = [
        {"start": 0.5, "end": 2.0, "chord_majmin": "D:min"},
        {"start": 2.0, "end": 4.0, "chord_majmin": "D:min"},   # duplicado consecutivo
        {"start": 4.0, "end": 6.0, "chord_majmin": "Bb:maj"},
        {"start": 6.0, "end": 7.0, "chord_majmin": "N"},        # se descarta
        {"start": 7.0, "end": 9.0, "chord_majmin": "A:7"},
    ]
    events = me._parse_chords(payload_a)
    assert [e["chord"] for e in events] == ["Dm", "A#", "A7"], events
    payload_b = {"data": [{"time": 1.0, "chord": "Em"}, {"time": 3.0, "name": "C"}]}
    events_b = me._parse_chords(payload_b)
    assert [e["chord"] for e in events_b] == ["Em", "C"]


def test_parse_lyrics_lines_and_words():
    payload = {
        "lines": [
            {"start": 10.0, "end": 13.0, "text": "Vine a alabar a Dios",
             "words": [
                 {"word": "Vine", "start": 10.0, "end": 10.4},
                 {"word": "a", "start": 10.4, "end": 10.5},
                 {"word": "alabar", "start": 10.5, "end": 11.4},
                 {"word": "a", "start": 11.4, "end": 11.5},
                 {"word": "Dios", "start": 11.5, "end": 12.6},
             ]},
            {"start": 14.0, "end": 17.0, "text": "vine a alabarle"},
        ]
    }
    lyrics = me._parse_lyrics(payload)
    assert len(lyrics["segments"]) == 2
    assert len(lyrics["words"]) == 5
    only_words = me._parse_lyrics([
        {"word": "hola", "start": 0.0, "end": 0.4},
        {"word": "mundo", "start": 0.5, "end": 0.9},
        {"word": "adios", "start": 3.0, "end": 3.4},  # pausa > 1s → nueva linea
    ])
    assert len(only_words["segments"]) == 2, only_words["segments"]


def test_group_segments_and_sections():
    segments = [
        {"text": "linea uno", "start": 1.0, "end": 3.0},
        {"text": "linea dos", "start": 4.0, "end": 6.0},
        {"text": "coro uno", "start": 20.0, "end": 23.0},
        {"text": "coro dos", "start": 24.0, "end": 27.0},
    ]
    sections = [
        {"label": "verse", "start": 0.0, "end": 10.0},
        {"label": "instrumental", "start": 10.0, "end": 19.0},  # sin lineas → fuera
        {"label": "chorus", "start": 19.0, "end": 30.0},
    ]
    groups = me._group_segments(segments, sections)
    assert len(groups) == 2
    assert [len(g["segments"]) for g in groups] == [2, 2]

    chords = [
        {"chord": "Dm", "time": 0.5},
        {"chord": "A#", "time": 4.5},
        {"chord": "C", "time": 19.5},
        {"chord": "Dm", "time": 25.0},
    ]
    built = me._build_sections(groups, chords, words=[])
    assert [s["name"] for s in built] == ["Verso 1", "Coro"], built
    first_line = built[0]["lines"][0]
    assert first_line["chords"][0] == {"chord": "Dm", "charIndex": 0}
    # linea 2: activo Dm suprimido (igual que el ultimo de la linea 1), A# dentro
    second_line = built[0]["lines"][1]
    assert [c["chord"] for c in second_line["chords"]] == ["A#"], second_line
    # coro: C activo al inicio de la seccion aunque venga de antes
    coro_line = built[1]["lines"][0]
    assert coro_line["chords"][0]["chord"] == "C"


def test_snap_to_beats():
    beats = [float(i) * 0.5 for i in range(200)]  # 120 BPM
    events = [
        {"chord": "C", "time": 1.06},   # → 1.0
        {"chord": "G", "time": 4.74},   # → 4.5 (dentro de tolerancia 0.175)? no: dif 0.24 → queda
        {"chord": "Am", "time": 8.51},  # → 8.5
    ]
    snapped = me._snap_to_beats(events, beats)
    assert snapped[0]["time"] == 1.0
    assert snapped[1]["time"] == 4.74
    assert snapped[2]["time"] == 8.5


def test_remap_chords_proportional():
    chords = [{"chord": "C", "charIndex": 0}, {"chord": "G", "charIndex": 10}]
    remapped = structuring.remap_chords(chords, "senor ten piedad", "Señor, ten piedad")
    assert remapped[0]["charIndex"] == 0
    assert 9 <= remapped[1]["charIndex"] <= 12, remapped
    # respace evita superposicion
    tight = [{"chord": "Cmaj7", "charIndex": 5}, {"chord": "G", "charIndex": 6}]
    spaced = structuring.respace(tight)
    assert spaced[1]["charIndex"] >= 5 + len("Cmaj7") + 2


def test_legacy_synchronize_end_to_end():
    lyrics_data = {
        "text": "vine a alabar a dios vine a alabarle",
        "segments": [
            {"text": "vine a alabar a dios", "start": 1.0, "end": 4.0},
            {"text": "vine a alabarle", "start": 8.0, "end": 11.0},
        ],
        "words": [
            {"word": "vine", "start": 1.0, "end": 1.4},
            {"word": "a", "start": 1.4, "end": 1.5},
            {"word": "alabar", "start": 1.5, "end": 2.4},
            {"word": "a", "start": 2.4, "end": 2.5},
            {"word": "dios", "start": 2.5, "end": 3.6},
            {"word": "vine", "start": 8.0, "end": 8.4},
            {"word": "a", "start": 8.4, "end": 8.5},
            {"word": "alabarle", "start": 8.5, "end": 10.2},
        ],
    }
    chords_data = [
        {"chord": "G", "time": 0.5},
        {"chord": "C", "time": 2.4},
        {"chord": "D", "time": 9.0},
    ]
    result = main.synchronize(lyrics_data, chords_data)
    assert result["detectedKey"] == "G"
    assert len(result["sections"]) == 2  # pausa de 4s → dos secciones
    line1 = result["sections"][0]["lines"][0]
    assert line1["chords"][0] == {"chord": "G", "charIndex": 0}
    c_chord = [c for c in line1["chords"] if c["chord"] == "C"]
    assert c_chord and 7 <= c_chord[0]["charIndex"] <= 16, line1["chords"]


def test_apply_structure_disabled_is_noop():
    sections = [{"name": "Verso 1", "lines": [{"lyrics": "hola", "chords": [], "timestamps": []}]}]
    out = structuring.apply_structure(sections, "C", "major")
    assert out is sections


def test_separation_unavailable_returns_none():
    import separation
    os.environ["AUDIO_SEPARATION"] = "0"
    try:
        assert separation.separate("/tmp/nope.mp3") == (None, None, None)
        assert not separation.is_available()
    finally:
        os.environ.pop("AUDIO_SEPARATION")


def test_musicai_pick_and_payload():
    result = {"Chords JSON": "u1", "lyrics_output": "u2", "Sections": "u3", "beat map": "u4"}
    assert me._pick(result, "chord") == "u1"
    assert me._pick(result, "lyric", "transcript") == "u2"
    assert me._pick(result, "section") == "u3"
    assert me._pick(result, "beat") == "u4"
    assert me._fetch_payload('{"a": 1}') == {"a": 1}
    assert me._fetch_payload([1, 2]) == [1, 2]


def test_extract_youtube_id():
    cases = {
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=10": "dQw4w9WgXcQ",
        "https://m.youtube.com/watch?app=desktop&v=dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://vimeo.com/12345678901": None,
        "no-es-url": None,
        "": None,
    }
    for url, expected in cases.items():
        got = main._extract_youtube_id(url)
        assert got == expected, f"{url!r}: esperaba {expected}, obtuve {got}"


def test_finalize_timestamps_attach_and_strip():
    def build():
        return {"sections": [
            {"name": "Verso 1", "lines": [
                {"lyrics": "a", "chords": [], "timestamps": [], "_startTime": 12.5},
                {"lyrics": "b", "chords": [], "timestamps": [], "_startTime": 15.0},
            ]},
            {"name": "Coro", "lines": [
                {"lyrics": "c", "chords": [], "timestamps": [], "_startTime": 30.2},
            ]},
        ]}

    out = main._finalize_timestamps(build(), attach=True)
    lines = [l for s in out["sections"] for l in s["lines"]]
    assert all("_startTime" not in l for l in lines)
    assert lines[0]["timestamps"] == [{"time": 12.5, "order": 1}]
    assert lines[1]["timestamps"] == [{"time": 15.0, "order": 2}]
    assert lines[2]["timestamps"] == [{"time": 30.2, "order": 3}]

    out2 = main._finalize_timestamps(build(), attach=False)
    lines2 = [l for s in out2["sections"] for l in s["lines"]]
    assert all(l["timestamps"] == [] and "_startTime" not in l for l in lines2)


def test_synchronize_emits_start_times():
    lyrics_data = {
        "text": "x",
        "segments": [
            {"text": "linea uno", "start": 2.0, "end": 4.0},
            {"text": "linea dos", "start": 9.0, "end": 11.0},
        ],
        "words": [],
    }
    result = main.synchronize(lyrics_data, [{"chord": "C", "time": 1.0}])
    raw_lines = [l for s in result["sections"] for l in s["lines"]]
    assert [l["_startTime"] for l in raw_lines] == [2.0, 9.0]
    final = main._finalize_timestamps(result, attach=True)
    lines = [l for s in final["sections"] for l in s["lines"]]
    assert lines[0]["timestamps"] == [{"time": 2.0, "order": 1}]
    assert lines[1]["timestamps"] == [{"time": 9.0, "order": 2}]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  OK  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print("—" * 40)
    sys.exit(1 if failures else print("Todos los tests pasaron") or 0)
