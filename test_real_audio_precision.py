"""Small real DSP regressions, with generated audio and no external services."""

import numpy as np
import pytest
import soundfile as sf

import main


SAMPLE_RATE = 22050
HOP_SECONDS = 1024 / SAMPLE_RATE


def tone(notes, seconds=2.0, amplitude=0.1):
    samples = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return sum(amplitude * np.sin(2 * np.pi * (440 * 2 ** ((note - 69) / 12)) * samples)
               for note in notes).astype(np.float32)


def detect(tmp_path, samples):
    path = tmp_path / "generated.wav"
    # FLOAT preserves genuinely quiet test music instead of quantizing it to 0.
    sf.write(path, samples, SAMPLE_RATE, subtype="FLOAT")
    return main._detect_chords_librosa(str(path))


def active(events, second):
    return next((event for event in events if event["time"] <= second < event["end"]), None)


def assert_valid_intervals(events, duration):
    assert events
    assert events[0]["time"] == 0
    assert events[-1]["end"] == pytest.approx(duration)
    assert all(0 <= event["time"] < event["end"] <= duration for event in events)
    assert all(left["end"] == right["time"] for left, right in zip(events, events[1:]))


def test_digital_silence_between_real_chords_is_not_normalized_into_harmony(tmp_path):
    audio = np.concatenate([tone([60, 64, 67]), np.zeros(2 * SAMPLE_RATE, dtype=np.float32),
                            tone([67, 71, 74]), tone([60, 64, 67])])
    events = detect(tmp_path, audio)
    assert_valid_intervals(events, 8)
    assert active(events, 1)["chord"] != "N"
    assert active(events, 3)["chord"] == "N"
    assert active(events, 5)["chord"] != "N"
    silent = active(events, 3)
    # Centered RMS frames touch the neighboring music for at most one hop;
    # the gate never pretends to resolve a boundary more finely than its grid.
    assert silent["time"] <= 2 + 2 * HOP_SECONDS
    assert silent["end"] >= 4 - 2 * HOP_SECONDS


def test_silence_at_edges_and_partial_final_frame_has_valid_coverage(tmp_path):
    audio = np.concatenate([np.zeros(SAMPLE_RATE, dtype=np.float32), tone([60, 64, 67]),
                            np.zeros(SAMPLE_RATE + 137, dtype=np.float32)])
    events = detect(tmp_path, audio)
    assert_valid_intervals(events, len(audio) / SAMPLE_RATE)
    assert active(events, 0.5)["chord"] == "N"
    assert active(events, 2)["chord"] != "N"
    assert active(events, 3.5)["chord"] == "N"


def test_very_quiet_music_above_digital_floor_is_not_gated(tmp_path):
    # Roughly -100 dBFS, far quieter than ordinary recordings but not silence.
    audio = tone([60, 64, 67], seconds=3.0, amplitude=1e-5)
    events = detect(tmp_path, audio)
    assert_valid_intervals(events, 3)
    assert active(events, 1.5)["chord"] != "N"
