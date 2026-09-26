"""Regressions motivated by the two supplied screenshots.

Words/line starts are transcribed from the user's reference. Segment end times
and intervening chord changes are constructed cases, not a measured ASR replay.
These tests prove layout rules only, not recognition quality on the recording.
"""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import main
import structuring
from timeline import build_sections
from transcription_service import transcribe_chunk


def test_ordinary_vocal_rests_do_not_create_instrumentals_or_one_line_verses():
    segments = [
        {"text": "Todo lo haces nuevo, Jesús", "start": 98, "end": 100},
        {"text": "Todo lo haces nuevo, Jesús", "start": 101, "end": 104},
        {"text": "Mi vida es tuya Señor", "start": 109, "end": 111},
        {"text": "Renuévame con tu amor", "start": 113, "end": 117},
    ]
    events = [{"chord": "E", "time": 98}, {"chord": "A", "time": 100.5},
              {"chord": "B", "time": 106}, {"chord": "A", "time": 112}]
    result = build_sections(segments, events)
    assert len(result) == 1
    assert [line["lyrics"] for line in result[0]["lines"]] == [s["text"] for s in segments]
    # Every actual change remains; one held chord is not reprinted on each line.
    assert [chord["chord"] for line in result[0]["lines"] for chord in line["chords"]] == ['E', 'A', 'B', 'A']


def test_long_instrumental_and_its_progression_are_still_kept():
    result = build_sections([
        {"text": "Frase anterior", "start": 0, "end": 4},
        {"text": "Frase siguiente", "start": 18, "end": 22},
    ], [{"chord": "E", "time": 0}, {"chord": "A", "time": 6},
        {"chord": "B", "time": 10}, {"chord": "E", "time": 17}])
    assert [s['name'] for s in result] == ['Verso 1', 'Instrumental', 'Verso 2']
    assert [c['chord'] for line in result[1]['lines'] for c in line['chords']][-3:] == ['A', 'B', 'E']


def test_a_brief_trailing_rest_does_not_invent_a_final_section():
    result = build_sections([{'text': 'Final cantado', 'start': 0, 'end': 5}],
                            [{'chord': 'E', 'time': 0}, {'chord': 'A', 'time': 5.5, 'end': 7}])
    assert len(result) == 1
    assert result[0]['lines'][0]['chords'][-1]['chord'] == 'A'


def test_two_adjacent_repeated_opening_verses_are_not_called_choruses(monkeypatch):
    monkeypatch.setenv('LLM_STRUCTURE', '0')
    verse = ['Tú obras maravillas con tu gracia', 'Nada es imposible en tus manos',
             'Ven hasta lo más hondo de mi ser']
    sections = [{'name': 'Verso 1', 'lines': [
        {'lyrics': text, 'chords': [], 'timestamps': []} for text in verse * 2
    ]}]
    before = deepcopy(sections)
    result = structuring.apply_structure(sections, 'E', 'major')
    assert [section['name'] for section in result] == ['Verso 1', 'Verso 2']
    assert [len(section['lines']) for section in result] == [3, 3]
    assert [line for section in result for line in section['lines']] == sections[0]['lines']
    assert sections == before


def test_default_transcription_does_not_seed_unheard_religious_words(tmp_path):
    path = tmp_path / 'mock.wav'
    path.write_bytes(b'test')
    create = Mock(return_value={'text': '', 'words': [], 'segments': []})
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    transcribe_chunk(client, str(path), timestamp_model='whisper-1', text_models=[], prompt=main.WHISPER_PROMPT)
    assert 'prompt' not in create.call_args.kwargs
