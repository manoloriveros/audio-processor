"""Inference/placement regressions. Acoustic evaluation is documented separately."""
from types import SimpleNamespace

import numpy as np
import pytest

import chordmini
import main
from chord_evidence import retain_supported_sevenths


@pytest.mark.parametrize("raw,canonical", [
    ("A:maj7", "Amaj7"), ("B:sus4", "Bsus4"), ("C#:min7", "C#m7"),
    ("D:minmaj7", "DmMaj7"), ("F:hdim7", "Fm7b5"), ("G:dim7", "Gdim7"),
    ("A:maj6", "A6"), ("F#:min6", "F#m6"), ("E:sus2", "Esus2"),
])
def test_rich_vocabulary_is_retained(raw, canonical):
    assert main._normalize_chord_label(raw) == canonical


def test_long_inference_keeps_global_frame_order_and_partial_batch():
    # Four model batches: more than ten minutes, plus an incomplete last window.
    frames = 16 * 108 * 4 + 37
    features = np.zeros((frames, 144), np.float32)
    features[:, 0] = np.arange(frames)
    seen = []

    def run(outputs, inputs):
        batch = inputs["features"]
        assert batch.shape == (16, 108, 144)
        seen.append(batch.copy())
        logits = np.zeros((16, 108, 170), np.float32)
        logits[:, :, 0] = batch[:, :, 0]
        return [logits]

    scores = chordmini.predict_scores(features, SimpleNamespace(run=run))
    np.testing.assert_array_equal(scores[:, 0], np.arange(frames))
    assert len(seen) == 5
    assert np.count_nonzero(seen[-1].reshape(-1, 144)[37:]) == 0


def test_decoder_retains_sevenths_sus_and_unknown_without_extending_duration():
    vocab = ["A:maj7", "B:sus4", "C#:min7", "X", "N"]
    scores = np.zeros((150, 5), np.float32)
    for block in range(5):
        scores[block * 30:(block + 1) * 30, block] = 10
    duration = (150 - .3) * chordmini.HOP / chordmini.SAMPLE_RATE
    result = chordmini.decode_scores(scores, vocab, duration)
    assert [event["rawLabel"] for event in result] == vocab
    assert result[3]["chord"] == "N" and result[3]["uncertain"]
    assert result[-1]["end"] == duration
    assert all(a["end"] == b["time"] for a, b in zip(result, result[1:]))


def test_digital_silence_overrides_a_hallucinated_chord():
    scores = np.tile([9., 0.], (30, 1))
    events = chordmini.decode_scores(scores, ["E", "N"], 2.5, np.ones(30, dtype=bool))
    assert len(events) == 1 and events[0]["chord"] == "N"
    assert events[0]["end"] == 2.5


def test_invalid_weights_fail_before_creating_a_runtime(tmp_path):
    (tmp_path / "chordnet.onnx").write_bytes(b"not the reviewed model")
    with pytest.raises(ValueError, match="checksum"):
        chordmini._runtime(str(tmp_path))


def test_model_failure_uses_existing_detector_and_records_engine(monkeypatch):
    monkeypatch.setattr(main, "CHORD_ENGINE", "auto")
    monkeypatch.setattr(main, "_CHORDINO_AVAILABLE", True)
    monkeypatch.setattr(chordmini, "detect", lambda path: (_ for _ in ()).throw(FileNotFoundError("weights")))
    monkeypatch.setattr(main, "_detect_chords_chordino", lambda path: [{"chord": "D7/F#", "time": 0, "end": 2}])
    monkeypatch.setattr(main, "_snap_chords_to_beats", lambda path, events: events)
    import transcription_chunks
    monkeypatch.setattr(transcription_chunks, "probe_audio_duration", lambda path: 2.)
    result = main.detect_chords("fixture.wav")
    assert result == [{"chord": "D7/F#", "time": 0., "end": 2., "engine": "chordino"}]


def test_key_is_invariant_to_fragmenting_a_short_chord():
    names = ["E", "Amaj7", "Bsus4", "C#m7"]
    durations = [30., 10., 8., 4.]
    expected = main._detect_key(names, durations)
    split_names = ["E"] + ["Amaj7"] * 100 + ["Bsus4", "C#m7"]
    split_durations = [30.] + [.1] * 100 + [8., 4.]
    assert expected == ("E", "major")
    assert main._detect_key(split_names, split_durations) == expected


def test_chords_stay_at_timestamped_words_in_a_late_repeated_phrase():
    text = "Ven hasta lo más hondo de mi ser"
    starts = [420., 420.3, 420.5, 420.7, 421., 421.5, 422., 423.]
    words = [{"word": word, "start": start, "end": start + .25}
             for word, start in zip(text.split(), starts)]
    data = {"duration": 425., "segments": [{"text": text, "start": 420., "end": 424.}], "words": words}
    changes = [{"chord": "Bsus4", "time": 420., "end": 421.},
               {"chord": "C#m7", "time": 421., "end": 423.},
               {"chord": "Amaj7", "time": 423., "end": 425.}]
    result = main.synchronize(data, changes)
    line = next(line for section in result["sections"] for line in section["lines"] if line["lyrics"])
    assert [(c["chord"], c["charIndex"], c["audioTime"]) for c in line["chords"]] == [
        ("Bsus4", 0, 420.), ("C#m7", text.index("hondo"), 421.), ("Amaj7", text.index("ser"), 423.)]
    assert line["chords"][1]["alignmentSource"] == "word-timestamps"
    assert result["chordTimeline"] == changes


def test_rest_and_instrumental_chords_keep_measured_times():
    result = main.synchronize({"segments": [{"text": "frase", "start": 10., "end": 12.},
                                            {"text": "siguiente", "start": 14., "end": 16.}]},
                              [{"chord": "E", "time": 0.}, {"chord": "Amaj7", "time": 13.},
                               {"chord": "Bsus4", "time": 14., "end": 17.}])
    chords = [c for s in result["sections"] for line in s["lines"] for c in line["chords"]]
    rest = next(c for c in chords if c["chord"] == "Amaj7")
    assert rest["audioTime"] == 13. and rest["alignmentSource"] == "vocal-rest"
    assert chords[0]["audioTime"] == 0. and chords[0]["alignmentSource"] == "instrumental"


def test_supported_seventh_keeps_its_own_onset_instead_of_moving_to_the_phrase_start():
    primary = [{"chord": "C", "time": 10., "end": 20.}]
    secondary = [{"chord": "Cmaj7", "time": 12., "end": 18.}]
    result = retain_supported_sevenths(primary, secondary)
    assert [(e["chord"], e["time"], e["end"]) for e in result] == [
        ("C", 10., 12.), ("Cmaj7", 12., 18.), ("C", 18., 20.)]
    assert result[1]["extensionNeedsReview"] is True
    assert primary == [{"chord": "C", "time": 10., "end": 20.}]


@pytest.mark.parametrize("chord", ["Csus4", "Csus2", "Cm7", "Cmaj7", "C/E", "N"])
def test_seventh_contrast_does_not_replace_suspensions_inversions_or_silence(chord):
    primary = [{"chord": chord, "time": 0., "end": 10.}]
    assert retain_supported_sevenths(primary, [{"chord": "C7", "time": 0., "end": 10.}]) == primary


@pytest.mark.parametrize("chord,end", [("Dm7", 10.), ("Cm7", 10.), ("C6", 10.), ("Cmaj7", .3)])
def test_seventh_contrast_rejects_changed_root_third_sixths_and_brief_evidence(chord, end):
    primary = [{"chord": "C", "time": 0., "end": 10.}]
    assert retain_supported_sevenths(primary, [{"chord": chord, "time": 0., "end": end}]) == primary
