"""Instrumental marker and overlapping chorus-boundary regressions."""
from copy import deepcopy
import pytest
import main
from lyric_phrases import is_music_annotation
from structuring import infer_repeated_sections
from test_structuring_precision import song, flatten


@pytest.mark.parametrize('marker', ['♪', '♫ ♫', '(♪)', '♩. ♪', '[Música]', '(instrumental)'])
def test_explicit_music_marker_is_not_lyric(marker):
    assert is_music_annotation(marker)


@pytest.mark.parametrize('lyric', ['Música', 'Oh', 'J', '♪ Mi vida es tuya ♪', 'A', 'Gracias.', '...'])
def test_ordinary_words_or_ambiguous_noise_are_not_filtered(lyric):
    assert not is_music_annotation(lyric)


def test_opening_music_markers_become_one_intro_and_first_sung_section_is_verse_one():
    segments = [{'text': '♪', 'start': t, 'end': t + 3} for t in (0, 4, 8)]
    segments.append({'text': 'Mi vida es tuya', 'start': 24, 'end': 28})
    chords = [{'chord': c, 'time': t, 'end': end} for c, t, end in
              [('E', 0, 4), ('Amaj7', 4, 8), ('E', 8, 16), ('Bsus4', 16, 24), ('C#m7', 24, 28)]]
    original = deepcopy(segments)
    result = main.synchronize({'segments': segments, 'duration': 28}, chords)
    assert [s['name'] for s in result['sections']] == ['Intro', 'Verso 1']
    intro = result['sections'][0]
    assert len(intro['lines']) == 1 and intro['lines'][0]['lyrics'] == ''
    assert [(c['chord'], c['audioTime']) for c in intro['lines'][0]['chords']] == [
        ('E', 0), ('Amaj7', 4), ('E', 8), ('Bsus4', 16)]
    assert segments == original


def test_interior_music_marker_preserves_instrumental_timeline_without_fake_verses():
    segments = [{'text': 'Una frase cantada', 'start': 0, 'end': 3},
                {'text': '[Música]', 'start': 4, 'end': 15},
                {'text': 'Otra frase cantada', 'start': 16, 'end': 20}]
    events = [{'chord': 'Cmaj7', 'time': 0, 'end': 4},
              {'chord': 'Gsus4', 'time': 4, 'end': 12}, {'chord': 'G', 'time': 12, 'end': 16},
              {'chord': 'C', 'time': 16, 'end': 20}]
    sections = main.synchronize({'segments': segments}, events)['sections']
    assert [s['name'] for s in sections] == ['Verso 1', 'Instrumental', 'Verso 2']
    assert all(line['lyrics'] == '' for line in sections[1]['lines'])
    actual = {(c['chord'], c['audioTime'], c['audioEnd']) for s in sections for l in s['lines'] for c in l['chords']}
    assert actual == {(c['chord'], c['time'], c['end']) for c in events}


def test_music_only_recording_remains_instrumental_with_all_changes():
    sections = main.synchronize({'segments': [{'text': '♫', 'start': 0, 'end': 12}]},
                                [{'chord': 'E', 'time': 0, 'end': 8}, {'chord': 'A', 'time': 8, 'end': 12}])['sections']
    assert sections[0]['name'] == 'Instrumental'
    assert sections[0]['lines'][0]['lyrics'] == ''
    assert [c['chord'] for c in sections[0]['lines'][0]['chords']] == ['E', 'A']


def test_frequent_chorus_core_is_not_absorbed_by_long_repeated_verse_tails():
    verse_tail = ['Un camino diferente empieza', 'Una nueva esperanza crece']
    prefix = ['Cantamos siempre esta cancion', 'Cantamos esta cancion']
    core = ['Nuestra esperanza sigue viva', 'Tu amor sostiene cada paso']
    sections = song(['La primera estrofa comienza', *verse_tail, *prefix, *core,
                     'La segunda estrofa comienza', *verse_tail, *prefix, *core,
                     *prefix, *core, 'Un puente distinto comienza', *prefix, *core,
                     *prefix, *core])
    result = infer_repeated_sections(sections)
    assert [s['name'] for s in result].count('Coro') == 5
    for section in result:
        if section['name'] == 'Coro':
            assert [l['lyrics'] for l in section['lines']] == [*prefix, *core]
    assert flatten(result) == flatten(sections)


def test_shortened_chorus_openings_still_do_not_borrow_repeated_verse_tails():
    verse_tail = ['Un camino diferente empieza', 'Una nueva esperanza crece']
    first = 'Porque cantamos juntos con mucha esperanza'
    short = 'Cantamos con esperanza'
    core = ['Nuestra esperanza sigue viva', 'Tu amor sostiene cada paso']
    sections = song(['La primera estrofa comienza', *verse_tail, first, short, *core,
                     'La segunda estrofa comienza', *verse_tail, first, short, *core,
                     first, 'Juntos con esperanza', *core, 'Un puente distinto comienza',
                     first, 'Cantamos mucha esperanza', *core, first, short, *core])
    result = infer_repeated_sections(sections)
    assert [s['name'] for s in result].count('Coro') == 5
    assert all(len(s['lines']) == 4 for s in result if s['name'] == 'Coro')
    assert all(l['lyrics'] not in verse_tail for s in result if s['name'] == 'Coro' for l in s['lines'])
    assert flatten(result) == flatten(sections)
