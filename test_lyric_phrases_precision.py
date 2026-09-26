"""Regression checks for complete phrases and lossless structural grouping."""
from copy import deepcopy

import pytest
import main
from lyric_phrases import repeated_phrase_segments, quarantine_credit_segments
from structuring import infer_repeated_sections
from test_structuring_precision import song, flatten, REFRAIN


def timed(texts):
    segments, words = [], []
    clock = 1.0
    for text in texts:
        start = clock
        for token in text.split():
            words.append({"word": token, "start": clock, "end": clock + .2})
            clock += .3
        segments.append({"text": text, "start": start, "end": clock - .1})
        clock += .1
    return segments, words


def test_complete_phrases_cross_asr_segment_boundaries_without_changing_words():
    first = "Esta es una frase cantada suficientemente larga para probar"
    second = "Otra frase distinta completa acompana esta cancion"
    texts = [first, second, "Esta es una frase cantada", "suficientemente larga para probar " + second]
    segments, words = timed(texts)
    result = repeated_phrase_segments(segments, words)
    assert [s['text'] for s in result] == [first, second, first, second]
    assert [s['start'] for s in result] == [words[i]['start'] for i in (0, 9, 16, 25)]
    split = main._split_long_segments(result, words=words)
    assert [s['text'] for s in split] == [first, second, first, second]


@pytest.mark.parametrize('mode', ['missing', 'estimated'])
def test_reflow_does_not_guess_missing_word_times(mode):
    segments, words = timed([*REFRAIN, *REFRAIN])
    if mode == 'missing':
        words = []
    else:
        for segment in segments:
            segment['timing_estimated'] = True
    assert repeated_phrase_segments(segments, words) == segments


def test_chords_follow_actual_word_times_after_phrase_reflow():
    texts = [*REFRAIN, REFRAIN[0] + ' ' + REFRAIN[1]]
    segments, words = timed(texts)
    target = next(word for word in words[10:] if word['word'] == 'ilumina')
    events = [{"chord": "Cmaj7", "time": words[0]['start'], "end": target['start']},
              {"chord": "Gsus4", "time": target['start'], "end": words[-1]['end']}]
    result = main.synchronize({'segments': segments, 'words': words}, events)
    placed = [(line, chord) for section in result['sections'] for line in section['lines']
              for chord in line['chords'] if chord['audioTime'] == pytest.approx(target['start'])]
    actual = next((line, chord) for line, chord in placed if 'ilumina' in line['lyrics']
                  and line['_startTime'] <= target['start'] < line['_endTime'])
    assert actual[1]['chord'] == 'Gsus4'
    assert actual[1]['charIndex'] == actual[0]['lyrics'].index('ilumina')
    assert actual[1]['alignmentSource'] == 'word-timestamps'


@pytest.mark.parametrize('text', ['Subtítulos realizados org', 'Subtítulos realizados por Amara.org',
                                 'Gracias por ver el video.'])
def test_explicit_final_credit_is_kept_for_review_instead_of_song(text):
    segments, words = timed(['La cancion sigue viva', text])
    segments[1]['start'], segments[1]['end'] = 98, 99
    for word in words[4:]:
        word['start'], word['end'] = 98, 99
    original = {'segments': segments, 'words': words, 'duration': 100}
    before = deepcopy(original)
    result = quarantine_credit_segments(original)
    assert result['segments'] == segments[:1]
    assert result['reviewSegments'][0]['text'] == text
    assert len(result['words']) == 4
    assert original == before


@pytest.mark.parametrize('text', ['Gracias.', 'Subtítulos', 'Gracias por tu amor', 'Toda tu gracia me renueva'])
def test_ordinary_lyrics_are_not_filtered(text):
    data = {'segments': [{'text': text, 'start': 98, 'end': 99}], 'duration': 100}
    assert quarantine_credit_segments(data) is data


def test_credit_inside_song_is_not_filtered():
    data = {'segments': [{'text': 'Gracias por ver el video', 'start': 30, 'end': 31}], 'duration': 100}
    assert quarantine_credit_segments(data) is data


def test_clustered_repeated_bridge_and_tail_are_not_extra_choruses():
    chorus = ['Cantamos siempre esta cancion', 'Cantamos siempre esta cancion',
              'Nuestra esperanza sigue viva', 'Tu amor sostiene cada paso']
    bridge = ['Dejamos nuestro miedo atras', 'Volvemos juntos a empezar', 'Ven conmigo caminemos']
    verses = ['La primera estrofa abre', 'Hoy volvemos a cantarla']
    sections = song([*verses, *chorus, 'Otra estrofa diferente', *chorus, *chorus,
                     *(bridge * 4), *chorus, *chorus, *([chorus[0]] * 4)])
    result = infer_repeated_sections(sections)
    assert [s['name'] for s in result].count('Coro') == 5
    assert [s['name'] for s in result].count('Puente') == 1
    assert [l['lyrics'] for l in next(s for s in result if s['name'] == 'Puente')['lines']] == bridge * 4
    assert result[-1]['name'] == 'Outro'
    assert flatten(result) == flatten(sections)


def test_fragmented_chorus_and_closing_phrase_keep_exact_lines_and_chords():
    chorus = ['Cantamos siempre esta cancion', 'Cantamos siempre esta cancion',
              'Nuestra esperanza sigue viva', 'Tu amor sostiene cada paso']
    sections = song(['Una estrofa comienza', *chorus, 'Senor contigo', 'Otra estrofa comienza',
                     *chorus, *chorus, 'Senor y contigo', 'Un puente diferente',
                     'Cantamos siempre', 'esta cancion', *chorus[1:]])
    result = infer_repeated_sections(sections)
    assert [s['name'] for s in result].count('Coro') == 4
    assert len(result[-1]['lines']) == 5
    assert flatten(result) == flatten(sections)


def test_different_numbers_and_negation_do_not_merge_phrase_identities():
    sections = song(['Cantamos esta frase numero 1', 'Cantamos esta frase numero 2',
                     'No quiero volver a empezar', 'Quiero volver a empezar'] * 2)
    result = infer_repeated_sections(sections)
    assert flatten(result) == flatten(sections)
    from structuring import _similar_phrase
    assert not _similar_phrase(('no', 'quiero', 'volver', 'a', 'empezar'), ('quiero', 'volver', 'a', 'empezar'))
    assert not _similar_phrase(('frase', 'numero', '1'), ('frase', 'numero', '2'))
