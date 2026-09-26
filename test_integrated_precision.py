"""Recording-wide regressions; external recognition is mocked, never billed."""
from copy import deepcopy
from types import SimpleNamespace

import main
import structuring


def test_long_recording_preserves_all_refrains_and_timeline(monkeypatch):
    phrases = [(1, 'Venimos a cantar'), (120, 'Gloria al Señor'),
               (122, 'Su amor es eterno'), (300, 'Caminamos en tu luz'),
               (660, 'Gloria al Señor'), (662, 'Su amor es eterno')]
    transcript = {
        'segments': [{'start': time, 'end': time + 1.8, 'text': text} for time, text in phrases],
        'words': [{'word': word, 'start': time + i * .3, 'end': time + i * .3 + .25}
                  for time, text in phrases for i, word in enumerate(text.split())],
        'duration': 720, 'chunkCount': 6, 'model': 'mock',
        'warnings': ['text_correction_rejected_to_preserve_timing'],
    }
    chords = [{'chord': 'C', 'time': 0}, {'chord': 'G', 'time': 120},
              {'chord': 'C', 'time': 120.2}, {'chord': 'N', 'time': 130},
              {'chord': 'F', 'time': 300}, {'chord': 'G7/B', 'time': 660}]
    monkeypatch.setattr(main, 'musicai_engine', None)
    monkeypatch.setattr(main, 'separation', SimpleNamespace(separate=lambda path: (None, None, None)))
    monkeypatch.setattr(main, 'transcribe_with_whisper', lambda path: deepcopy(transcript))
    monkeypatch.setattr(main, 'detect_chords', lambda *args, **kwargs: [
        {**event, 'engine': 'chordmini'} for event in deepcopy(chords)])
    monkeypatch.setenv('LLM_STRUCTURE', '0')
    result = main.run_pipeline('long-recording.wav')
    result = main._finalize_timestamps(result, attach=True)
    vocal_lines = [line for section in result['sections'] for line in section['lines'] if line['lyrics']]
    assert [line['lyrics'] for line in vocal_lines] == [text for _, text in phrases]
    assert [line['timestamps'][0]['time'] for line in vocal_lines] == [time for time, _ in phrases]
    assert sum(section['name'] == 'Coro' for section in result['sections']) == 2
    assert result['transcriptionChunks'] == 6
    assert result['analysisDuration'] == 720
    assert result['analysisWarnings'] == transcript['warnings']
    assert [event['chord'] for event in result['chordTimeline']] == [event['chord'] for event in chords]
    assert result['chordTimeline'][1]['end'] == 120.2
    assert result['chordTimeline'][-1]['end'] == 720
    assert not any(key.startswith('_') for line in vocal_lines for key in line)


def test_estimated_fallback_never_becomes_precise_video_marker():
    result = main.synchronize({
        'segments': [{'text': 'Una frase sin tiempos de voz', 'start': 600, 'end': 720,
                      'timing_estimated': True, 'timestamp_source': 'chunk_interval'}],
        'words': [], 'duration': 720,
    }, [{'chord': 'Am', 'time': 602, 'end': 610}])
    final = main._finalize_timestamps(result, attach=True)
    line = next(line for section in final['sections'] for line in section['lines'] if line['lyrics'])
    assert line['timestamps'] == []
    assert '_startTime' not in line and '_endTime' not in line


def test_structure_keeps_time_and_instrumental_lines_after_regrouping(monkeypatch):
    monkeypatch.setenv('LLM_STRUCTURE', '0')
    sections = [{'name': 'Verso 1', 'lines': [
        {'lyrics': text, 'chords': [{'chord': 'Cmaj7', 'charIndex': 2}],
         '_startTime': index * 120, '_endTime': index * 120 + 5, 'timestamps': []}
        for index, text in enumerate(['Gloria al Señor', 'Su amor es eterno',
                                      'Caminamos a tu lado', 'Gloria al Señor', 'Su amor es eterno'])
    ]}, {'name': 'Final', 'lines': [{'lyrics': '', 'chords': [{'chord': 'C', 'charIndex': 0}],
                                   '_startTime': 610, '_endTime': 620, 'timestamps': []}]}]
    original = deepcopy(sections)
    output = structuring.apply_structure(sections, 'C', 'major')
    assert [line for sec in output for line in sec['lines']] == [line for sec in original for line in sec['lines']]
    assert output[-1] == original[-1]
    assert sections == original
