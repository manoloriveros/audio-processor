from transcription_alignment import align_corrected_words


def test_missing_repeated_chorus_preserves_original_timeline():
    words = [
        {"word": word, "start": start + i, "end": start + i + 0.8}
        for start in (0, 300, 600)
        for i, word in enumerate("aleluya gloria al señor".split())
    ]
    output, accepted = align_corrected_words(
        "aleluya gloria al Señor aleluya gloria al Señor", words,
    )
    assert not accepted
    assert output == words
    assert [output[i]["start"] for i in (0, 4, 8)] == [0, 300, 600]


def test_inserted_word_never_receives_fabricated_overlapping_time():
    words = [{"word": "a", "start": 0, "end": 1}, {"word": "b", "start": 1, "end": 2}]
    assert align_corrected_words("a c b", words) == (words, False)


def test_one_to_one_correction_keeps_each_measured_time_and_metadata():
    words = [
        {"word": "senor", "start": 1, "end": 7, "confidence": 0.8},
        {"word": "ten", "start": 7.1, "end": 7.4},
        {"word": "piedra", "start": 7.4, "end": 9},
    ]
    output, accepted = align_corrected_words("Señor, ten piedad", words)
    assert accepted
    assert [w["word"] for w in output] == ["Señor,", "ten", "piedad"]
    assert [(w["start"], w["end"]) for w in output] == [(1, 7), (7.1, 7.4), (7.4, 9)]
    assert output[0]["confidence"] == 0.8
    assert words[0]["word"] == "senor"


def test_unrelated_or_reordered_text_keeps_timed_transcript():
    words = [
        {"word": word, "start": i, "end": i + 0.8}
        for i, word in enumerate("gloria al padre y al hijo".split())
    ]
    assert align_corrected_words("hoy vamos todos juntos para casa", words) == (words, False)
    assert align_corrected_words("y al hijo gloria al padre", words) == (words, False)
