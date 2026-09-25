from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from transcription_service import transcribe_chunk


def call_chunk(tmp_path, responses):
    path = tmp_path / "window.wav"
    path.write_bytes(b"test input consumed by mocked client")
    create = Mock(side_effect=responses)
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    result = transcribe_chunk(client, str(path), timestamp_model="whisper-1",
                              text_models=["text-model"], prompt="canto")
    return result, create


def test_repetition_omission_cannot_shorten_measured_transcript(tmp_path):
    words = [{"word": word, "start": i, "end": i + 0.8}
             for i, word in enumerate("gloria al señor gloria al señor".split())]
    result, _ = call_chunk(tmp_path, [
        {"text": "gloria al señor gloria al señor", "words": words,
         "segments": [{"text": "gloria al señor gloria al señor", "start": 0, "end": 6}]},
        {"text": "Gloria al Señor"},
    ])
    assert result["text"] == "gloria al señor gloria al señor"
    assert result["words"] == words
    assert "text_correction_rejected_to_preserve_timing" in result["warnings"]


def test_silent_window_is_valid_and_does_not_trigger_text_request(tmp_path):
    result, create = call_chunk(tmp_path, [{"text": "", "segments": [], "words": []}])
    assert result["text"] == ""
    assert create.call_count == 1


def test_timestamp_failure_returns_text_without_false_zero_timestamps(tmp_path):
    result, _ = call_chunk(tmp_path, [RuntimeError("offline"), {"text": "Gloria al Señor"}])
    assert result["text"] == "Gloria al Señor"
    assert result["words"] == result["segments"] == []
    assert "timestamp_model_failed" in result["warnings"]


def test_failed_chunk_never_silently_skips_part_of_recording(tmp_path):
    with pytest.raises(RuntimeError, match="fragmento"):
        call_chunk(tmp_path, [RuntimeError("offline"), RuntimeError("offline")])


def test_word_corrections_keep_actual_durations(tmp_path):
    result, _ = call_chunk(tmp_path, [
        {"text": "senor ten piedad", "words": [
            {"word": "senor", "start": 1, "end": 8},
            {"word": "ten", "start": 8, "end": 8.2},
            {"word": "piedad", "start": 8.2, "end": 9},
        ], "segments": [{"text": "senor ten piedad", "start": 1, "end": 9}]},
        {"text": "Señor, ten piedad."},
    ])
    assert result["segments"][0]["text"] == "Señor, ten piedad."
    assert result["words"][0] == {"word": "Señor,", "start": 1, "end": 8}


def test_partial_word_timestamps_do_not_erase_a_complete_timed_phrase(tmp_path):
    result, _ = call_chunk(tmp_path, [
        {"text": "Gloria al Señor Su amor es eterno", "words": [
            {"word": "Gloria", "start": 1, "end": 2},
            {"word": "al", "start": 2, "end": 2.5},
            {"word": "Señor", "start": 2.5, "end": 3},
        ], "segments": [
            {"text": "Gloria al Señor", "start": 1, "end": 3},
            {"text": "Su amor es eterno", "start": 5, "end": 9},
        ]},
        {"text": "Gloria al Señor Su amor es eterno"},
    ])
    assert [segment["text"] for segment in result["segments"]] == ["Gloria al Señor", "Su amor es eterno"]
    assert result["segments"][1]["start"] == 5
    assert "Su amor es eterno" in result["text"]
