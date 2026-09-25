"""Music.ai adapter regressions: preserve provider evidence and boundaries."""

import musicai_engine as me


def test_complex_labels_win_and_explicit_silence_ends_survive():
    result = me._parse_chords([
        {"start": 0, "end": 2, "chord_majmin": "C:maj", "chord_complex_pop": "C:maj7/G", "confidence": 0.91},
        {"start": 2, "end": 5, "chord_complex_pop": "N"},
        {"start": 5, "end": 7, "chord_complex_pop": "C:maj7/G"},
    ])
    assert [event["chord"] for event in result] == ["Cmaj7/G", "N", "Cmaj7/G"]
    assert [event["end"] for event in result] == [2, 5, 7]
    assert result[0]["confidence"] == 0.91


def test_unrecognized_labels_are_not_converted_into_false_silence():
    assert me._parse_chords([{"start": 0, "end": 2, "chord": "unknown-format"}]) == []


def test_provider_instrumental_sections_are_kept_without_lyric_segments():
    groups = me._group_segments([
        {"text": "verso", "start": 2, "end": 4}, {"text": "coro", "start": 12, "end": 14},
    ], [{"label": "verse", "start": 0, "end": 5},
        {"label": "instrumental", "start": 5, "end": 10},
        {"label": "chorus", "start": 10, "end": 16}])
    assert len(groups) == 3
    assert groups[1]["segments"] == []
    result = me._build_sections(groups, [
        {"chord": "C", "time": 0, "end": 5},
        {"chord": "Am", "time": 5, "end": 7},
        {"chord": "F", "time": 7, "end": 9},
        {"chord": "G", "time": 9, "end": 12},
    ], [])
    assert [section["name"] for section in result] == ["Verso 1", "Instrumental", "Coro"]
    assert [chord["chord"] for line in result[1]["lines"] for chord in line["chords"]] == ["Am", "F", "G"]
    assert all(line["lyrics"] == "" for line in result[1]["lines"])


def test_beat_reference_keeps_offbeat_changes_and_same_beat_candidates():
    events = [{"chord": "C", "time": 1.04, "end": 1.08},
              {"chord": "G", "time": 1.08, "end": 1.12},
              {"chord": "N", "time": 1.12, "end": 2}]
    result = me._snap_to_beats(events, [0, 0.5, 1, 1.5, 2])
    assert len(result) == 3
    assert [event["time"] for event in result] == [1.04, 1.08, 1.12]
    assert [event["end"] for event in result] == [1.08, 1.12, 2]
    assert result[0]["beatTime"] == result[1]["beatTime"] == 1
    assert "beatTime" not in result[2]
    assert all("beatTime" not in event for event in events)


def test_instrumental_workflow_result_is_usable_without_lyrics(monkeypatch):
    monkeypatch.setattr(me, "_upload", lambda path: "uploaded")
    monkeypatch.setattr(me, "_create_job", lambda url: "job")
    monkeypatch.setattr(me, "_wait_job", lambda job: {
        "Chords": [{"start": 0, "end": 2, "chord_complex_pop": "C:maj/3"},
                   {"start": 2, "end": 4, "chord_complex_pop": "G:maj"}],
        "Sections": [{"start": 0, "end": 4, "label": "instrumental"}],
    })
    calls = []

    def structure(sections, key, key_type, **kwargs):
        calls.append(kwargs)
        return sections

    monkeypatch.setattr(me.structuring, "apply_structure", structure)
    result = me.process("unused.wav")
    assert result["engine"] == "music.ai"
    assert result["sections"][0]["name"] == "Instrumental"
    assert [event["chord"] for event in result["chordTimeline"]] == ["C/E", "G"]
    assert calls == [{"preserve_boundaries": True}]


def test_workflow_without_acoustic_sections_allows_offline_grouping(monkeypatch):
    monkeypatch.setattr(me, "_upload", lambda path: "uploaded")
    monkeypatch.setattr(me, "_create_job", lambda url: "job")
    monkeypatch.setattr(me, "_wait_job", lambda job: {
        "Lyrics": {"lines": [{"text": "El Señor", "start": 0, "end": 2}]},
        "Chords": [{"start": 0, "end": 2, "chord": "C"}],
    })
    calls = []
    monkeypatch.setattr(me.structuring, "apply_structure", lambda sections, key, key_type, **kwargs:
                        calls.append(kwargs) or sections)
    assert me.process("unused.wav")["sections"]
    assert calls == [{"preserve_boundaries": False}]
