"""Regression tests for preserving musical evidence, independent of API calls."""

from contextlib import contextmanager
from types import SimpleNamespace
import sys

import numpy as np
import pytest

import main
from timeline import normalize_events
import transcription_chunks


@pytest.mark.parametrize("source,expected", [
    ("C:maj/3", "C/E"), ("C:min/b3", "Cm/D#"), ("E:min/5", "Em/B"),
    ("Bb:7/3", "A#7/D"), ("C:maj7/G", "Cmaj7/G"),
    ("F#:min/b3", "F#m/A"), ("G:maj/5", "G/D"),
])
def test_degree_inversions_are_converted_to_bass_notes(source, expected):
    assert main._normalize_chord_label(source) == expected


def test_short_harmonic_change_is_not_removed_or_moved():
    events = [{"chord": "C", "time": 0}, {"chord": "G", "time": 1},
              {"chord": "C", "time": 1.125, "end": 2, "confidence": 0.8}]
    result = main._postprocess_chord_events(events)
    assert [(event["chord"], event["time"]) for event in result] == [("C", 0), ("G", 1), ("C", 1.125)]
    assert result[1]["end"] == 1.125
    assert result[2]["confidence"] == 0.8
    assert "end" not in events[0]  # caller data is not mutated


def test_known_interval_end_inserts_silence_and_preserves_repeat():
    result = normalize_events([{"chord": "C", "time": 0, "end": 2},
                               {"chord": "C", "time": 5, "end": 7}])
    assert result == [{"chord": "C", "time": 0, "end": 2},
                      {"chord": "N", "time": 2, "end": 5},
                      {"chord": "C", "time": 5, "end": 7}]


def test_chordino_keeps_explicit_no_chord_events(monkeypatch):
    changes = [SimpleNamespace(chord=label, timestamp=time) for label, time in [("C", 0), ("N", 2), ("C", 5)]]
    extractor = SimpleNamespace(Chordino=lambda **kwargs: SimpleNamespace(extract=lambda path: changes))
    monkeypatch.setitem(sys.modules, "chord_extractor.extractors", extractor)
    result = main._detect_chords_chordino("unused.wav")
    assert [(event["chord"], event["time"]) for event in result] == [("C", 0), ("N", 2), ("C", 5)]


def test_silence_does_not_extend_the_previous_chord_into_lyrics():
    result = main.synchronize({"segments": [{"text": "voz sola", "start": 3, "end": 4}], "words": []},
                              [{"chord": "C", "time": 0, "end": 2}, {"chord": "G", "time": 5, "end": 8}])
    vocal = next(line for section in result["sections"] for line in section["lines"] if line["lyrics"])
    assert all(chord["charIndex"] >= len(vocal["lyrics"]) for chord in vocal["chords"])  # solo tras el final de la voz
    assert result["chordTimeline"][1] == {"chord": "N", "time": 2, "end": 5}


def test_intro_and_interlude_progressions_survive_without_lyrics():
    events = [{"chord": chord, "time": time} for chord, time in
              [("C", 0), ("F", 2), ("G", 4), ("Am", 8), ("F", 13), ("G", 15), ("C", 18)]]
    result = main.synchronize({"segments": [{"text": "primera frase", "start": 10, "end": 12},
                                            {"text": "segunda frase", "start": 20, "end": 22}], "words": []}, events)
    intro = result["sections"][0]
    assert intro["name"] == "Intro"
    assert [chord["chord"] for line in intro["lines"] for chord in line["chords"]] == ["C", "F", "G", "Am"]
    interlude = next(section for section in result["sections"] if section["name"] == "Instrumental")
    assert all(line["lyrics"] == "" for line in interlude["lines"])
    assert [chord["chord"] for line in interlude["lines"] for chord in line["chords"]][-3:] == ["F", "G", "C"]


def test_instrumental_only_audio_keeps_chords_and_times():
    result = main.synchronize({"segments": [], "words": []},
                              [{"chord": "Dm", "time": 0, "end": 2}, {"chord": "A", "time": 2, "end": 4}])
    assert result["sections"][0]["name"] == "Instrumental"
    line = result["sections"][0]["lines"][0]
    assert line["lyrics"] == ""
    assert [chord["chord"] for chord in line["chords"]] == ["Dm", "A"]
    assert line["_startTime"] == 0 and line["_endTime"] == 4


def test_instrumental_only_unknown_last_duration_is_not_lost():
    result = main.synchronize({"segments": []}, [{"chord": "G", "time": 7}])
    assert result["sections"][0]["lines"][0]["chords"][0]["chord"] == "G"


def test_line_splits_follow_real_words_with_a_sustained_final_note():
    text = "El Señor es mi pastor nada me falta y por siempre cantaré su misericordia"
    words = [{"word": token, "start": index * 0.25, "end": index * 0.25 + 0.2}
             for index, token in enumerate(text.split())]
    words[-1]["end"] = 20
    result = main._split_long_segments([{"text": text, "start": 0, "end": 20}], words=words)
    assert len(result) == 2
    assert result[1]["start"] == 2.25
    assert result[0]["end"] == 2.2
    assert result[1]["end"] == 20
    assert " ".join(segment["text"] for segment in result) == text
    position = main._time_to_char_index(2.25, result[1]["text"], result[1]["start"],
                                        result[1]["end"], result[1]["_words"])
    assert position == 0


def test_untimed_long_phrase_is_not_given_fabricated_line_timestamps():
    text = "Una frase larga sin palabras con tiempos no permite dividir la sincronización con precisión"
    result = main._split_long_segments([{"text": text, "start": 0, "end": 20}])
    assert len(result) == 1
    assert result[0]["text"] == text
    assert result[0]["start"] == 0 and result[0]["end"] == 20


def test_repeated_word_on_adjacent_segment_boundary_is_owned_once():
    words = [{"word": "aleluya", "start": 0, "end": 1},
             {"word": "aleluya", "start": 1, "end": 2}]
    result = main._split_long_segments([
        {"text": "aleluya", "start": 0, "end": 1},
        {"text": "aleluya", "start": 1, "end": 2},
    ], words=words)
    assert [segment["start"] for segment in result] == [0, 1]
    assert [len(segment["_words"]) for segment in result] == [1, 1]


def test_visual_chord_label_width_does_not_change_lyric_anchor():
    result = main.synchronize({"segments": [{"text": "aleluya", "start": 1, "end": 8}], "words": []},
                              [{"chord": "Cmaj7", "time": 0}, {"chord": "G", "time": 1.01}])
    vocal = next(line for section in result["sections"] for line in section["lines"] if line["lyrics"])
    expected = main._time_to_char_index(1.01, "aleluya", 1, 8, [])
    assert vocal["chords"][1]["chord"] == "G"
    assert vocal["chords"][1]["charIndex"] == expected
    assert vocal["chords"][1]["audioTime"] == 1.01
    assert expected < 2


def test_estimated_timing_provenance_reaches_lines_and_prevents_video_markers():
    result = main.synchronize({"segments": [{"text": "sin tiempos fiables", "start": 0, "end": 20,
                                             "timing_estimated": True, "timestamp_source": "estimated"}]}, [])
    line = result["sections"][0]["lines"][0]
    assert line["timing_estimated"] is True
    assert line["timestamp_source"] == "estimated"
    finalized = main._finalize_timestamps(result, attach=True)
    assert finalized["sections"][0]["lines"][0]["timestamps"] == []


def test_beat_evidence_covers_after_ten_minutes_without_mutating_events(monkeypatch):
    seen = []

    @contextmanager
    def chunks(path, **kwargs):
        assert kwargs == {"chunk_seconds": 60.0, "overlap_seconds": 4.0}
        yield iter([transcription_chunks.AudioChunk("first.wav", 0, 0, 0, 60, 60, False),
                    transcription_chunks.AudioChunk("last.wav", 11, 660, 660, 720, 720, True)])

    def load(path, **kwargs):
        assert kwargs == {"sr": None, "mono": True}
        seen.append(path)
        return np.zeros(32), 16000

    monkeypatch.setattr(transcription_chunks, "iter_audio_chunks", chunks)
    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(
        load=load, beat=SimpleNamespace(beat_track=lambda **kwargs: (120, [0])),
        frames_to_time=lambda frames, **kwargs: np.array([0.0])))
    events = [{"chord": "C", "time": 0.08, "end": 0.1},
              {"chord": "G", "time": 0.1, "end": 1},
              {"chord": "N", "time": 660.01, "end": 660.08},
              {"chord": "Am", "time": 660.08, "end": 661}]
    result = main._snap_chords_to_beats("long.wav", events)
    assert seen == ["first.wav", "last.wav"]
    assert result[-1]["beatTime"] == 660
    assert all({key: event[key] for key in ("chord", "time", "end")} == original
               for event, original in zip(result, events))
    assert "beatTime" not in result[2]
    assert all("beatTime" not in event for event in events)
