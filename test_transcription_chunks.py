"""Offline coverage for bounded decoding and absolute transcript assembly."""

import json
from contextlib import nullcontext
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

from transcription_chunks import AudioChunk, iter_audio_chunks, merge_chunk_transcripts, probe_audio_duration, _run


def available_ffmpeg():
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, AttributeError, RuntimeError):
        return None


FFMPEG = available_ffmpeg()


def chunk(index=0, duration=240.0):
    start = index * 120.0
    end = min(duration, start + 120.0)
    return AudioChunk("unused.wav", index, max(0.0, start - 2.0), start, end,
                      min(duration, end + 2.0), end == duration)


def transcript(words, segments=None):
    return {"words": words, "segments": segments or [], "model": "test-model"}


class ChunkPreparationTests(unittest.TestCase):
    def test_lazy_decode_keeps_one_file_and_removes_it_on_advance(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "input with spaces & symbols.wav"
            source.write_bytes(b"source")
            calls = []

            def decode(args, timeout):
                calls.append(args)
                Path(args[-1]).write_bytes(b"temporary")
                return ""

            with patch("transcription_chunks.probe_audio_duration", return_value=241.0), \
                 patch("transcription_chunks._run", side_effect=decode), \
                 patch("transcription_chunks._verify_wav"):
                with iter_audio_chunks(str(source), temp_dir=directory) as chunks:
                    self.assertEqual(calls, [])
                    first = next(chunks)
                    self.assertEqual((first.start, first.core_end, first.end), (0, 120, 122))
                    self.assertTrue(Path(first.path).exists())
                    second = next(chunks)
                    self.assertFalse(Path(first.path).exists())
                    self.assertEqual((second.start, second.core_start, second.core_end, second.end), (118, 120, 240, 241))
                    self.assertEqual(len(list(Path(second.path).parent.iterdir())), 1)
                    third = next(chunks)
                    self.assertEqual((third.start, third.core_start, third.core_end, third.end), (238, 240, 241, 241))
                    self.assertTrue(third.is_last)
                    self.assertEqual(calls[0][calls[0].index("-i") + 1], str(source.resolve()))
                    self.assertEqual(calls[0][calls[0].index("-ar") + 1], "16000")
                self.assertFalse(Path(third.path).parent.exists())

    def test_exception_in_consumer_also_cleans_chunk(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            source.write_bytes(b"source")
            def decode(args, timeout):
                Path(args[-1]).write_bytes(b"temporary")
                return ""
            with patch("transcription_chunks.probe_audio_duration", return_value=20), \
                 patch("transcription_chunks._run", side_effect=decode), \
                 patch("transcription_chunks._verify_wav"):
                with self.assertRaisesRegex(RuntimeError, "consumer"):
                    with iter_audio_chunks(str(source), temp_dir=directory) as chunks:
                        current = next(chunks)
                        raise RuntimeError("consumer failed")
                self.assertFalse(Path(current.path).exists())

    def test_decode_failure_cleans_partial_file(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            source.write_bytes(b"source")
            paths = []
            def decode(args, timeout):
                paths.append(Path(args[-1]))
                paths[-1].write_bytes(b"partial")
                raise RuntimeError("ffmpeg failed")
            with patch("transcription_chunks.probe_audio_duration", return_value=20), \
                 patch("transcription_chunks._run", side_effect=decode):
                with self.assertRaisesRegex(RuntimeError, "ffmpeg"):
                    with iter_audio_chunks(str(source), temp_dir=directory) as chunks:
                        next(chunks)
                self.assertFalse(paths[0].parent.exists())

    def test_unsafe_chunk_parameters_fail_before_subprocess(self):
        invalid = [dict(chunk_seconds=0), dict(chunk_seconds=float("inf")),
                   dict(overlap_seconds=-1), dict(overlap_seconds=120),
                   dict(chunk_seconds=1000), dict(sample_rate=0), dict(timeout_seconds=0)]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), patch("transcription_chunks._run") as run:
                with self.assertRaises(ValueError):
                    with iter_audio_chunks("not-needed.wav", **kwargs):
                        pass
                run.assert_not_called()

    def test_probe_validates_duration_and_stream(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "test.wav"
            source.write_bytes(b"test")
            for value in ["NaN", "inf", "0", "-2", None]:
                data = {"format": {"duration": value}, "streams": [{"codec_type": "audio"}]}
                with self.subTest(value=value), patch("transcription_chunks._run", return_value=json.dumps(data)):
                    with self.assertRaises(ValueError):
                        probe_audio_duration(str(source))
            with patch("transcription_chunks._run", return_value='{"format":{"duration":"5"},"streams":[]}'):
                with self.assertRaisesRegex(ValueError, "no audio"):
                    probe_audio_duration(str(source))
            with patch("transcription_chunks._run", return_value='{"format":{"duration":"5.5"},"streams":[{"codec_type":"audio"}]}'):
                self.assertEqual(probe_audio_duration(str(source)), 5.5)

    def test_subprocess_timeout_is_bounded_and_no_shell_is_used(self):
        with patch("transcription_chunks.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 3)) as run:
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                _run(["ffmpeg", "path & name"], 3)
            self.assertEqual(run.call_args.kwargs["timeout"], 3)
            self.assertNotIn("shell", run.call_args.kwargs)

    @unittest.skipUnless(FFMPEG, "ffmpeg optional integration check")
    def test_real_ffmpeg_normalizes_short_audio_and_chunks_timeline(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "stereo.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setnchannels(2)
                audio.setsampwidth(2)
                audio.setframerate(44100)
                block = b"\0" * (4410 * 4)
                for _ in range(13):
                    audio.writeframes(block)
            chunks_seen = []
            # Minimal developer runtimes may bundle ffmpeg without ffprobe. In
            # that case only metadata is stubbed, using the known fixture length;
            # normalization/seek/decode and WAV validation still run for real.
            probe = nullcontext() if shutil.which("ffprobe") else patch("transcription_chunks.probe_audio_duration", return_value=1.3)
            with probe:
                with iter_audio_chunks(str(source), chunk_seconds=0.5, overlap_seconds=0.1, ffmpeg=FFMPEG) as chunks:
                    for item in chunks:
                        with wave.open(item.path, "rb") as audio:
                            self.assertEqual((audio.getnchannels(), audio.getframerate(), audio.getsampwidth()), (1, 16000, 2))
                            self.assertAlmostEqual(audio.getnframes() / 16000, item.duration, places=3)
                        chunks_seen.append(item)
            self.assertEqual(len(chunks_seen), 3)
            self.assertAlmostEqual(chunks_seen[-1].core_end, 1.3, places=3)
            self.assertFalse(Path(chunks_seen[-1].path).exists())


class TranscriptMergeTests(unittest.TestCase):
    def test_overlap_words_are_owned_once_and_segment_text_is_cropped(self):
        first, second = chunk(0), chunk(1)
        words_first = [{"word": "Antes", "start": 119.0, "end": 119.4},
                       {"word": "después", "start": 120.1, "end": 120.5}]
        words_second = [{"word": "Antes", "start": 1.0, "end": 1.4},
                        {"word": "después", "start": 2.1, "end": 2.5}]
        result = merge_chunk_transcripts([
            (first, transcript(words_first, [{"text": "Antes después", "start": 119, "end": 121}])),
            (second, transcript(words_second, [{"text": "Antes después", "start": 1, "end": 3}])),
        ])
        self.assertEqual([word["word"] for word in result["words"]], ["Antes", "después"])
        self.assertEqual([seg["text"] for seg in result["segments"]], ["Antes", "después"])
        self.assertAlmostEqual(result["segments"][1]["start"], 120.1)

    def test_half_open_ownership_assigns_exact_boundary_to_next_core(self):
        first, second = chunk(0), chunk(1)
        event1 = {"word": "luz", "start": 119.8, "end": 120.2}
        event2 = {"word": "luz", "start": 1.8, "end": 2.2}
        result = merge_chunk_transcripts([(first, transcript([event1])), (second, transcript([event2]))])
        self.assertEqual(len(result["words"]), 1)
        self.assertFalse(first.owns(120))
        self.assertTrue(second.owns(120))

    def test_boundary_timestamp_jitter_does_not_duplicate_the_same_word(self):
        result = merge_chunk_transcripts([
            (chunk(0), transcript([{"word": "Santo", "start": 119.6, "end": 120.2}])),
            (chunk(1), transcript([{"word": "Santo", "start": 1.8, "end": 2.4}])),
        ])
        self.assertEqual([word["word"] for word in result["words"]], ["Santo"])
        self.assertEqual(result["text"], "Santo")

    def test_opposite_jitter_recovers_word_rejected_by_both_owners(self):
        result = merge_chunk_transcripts([
            (chunk(0), transcript(
                [{"word": "Santo", "start": 119.9, "end": 120.3}],
                [{"text": "Santo", "start": 119.5, "end": 120.5}],
            )),
            (chunk(1), transcript(
                [{"word": "Santo", "start": 1.7, "end": 2.1}],
                [{"text": "Santo", "start": 1.5, "end": 2.5}],
            )),
        ])
        self.assertEqual(result["text"], "Santo")
        self.assertEqual(len(result["words"]), 1)
        self.assertEqual((result["words"][0]["start"], result["words"][0]["end"]), (119.9, 120.3))
        self.assertTrue(result["words"][0]["boundary_recovered"])

    def test_context_recovery_at_last_short_chunk_preserves_final_word(self):
        result = merge_chunk_transcripts([
            (chunk(1, 241), transcript([{"word": "Santo", "start": 121.9, "end": 122.3}])),
            (chunk(2, 241), transcript([
                {"word": "Santo", "start": 1.7, "end": 2.1},
                {"word": "amén", "start": 2.8, "end": 3.0},
            ])),
        ])
        self.assertEqual([word["word"] for word in result["words"]], ["Santo", "amén"])
        self.assertEqual(result["words"][-1]["end"], 241)
        self.assertEqual(sum(bool(word.get("boundary_recovered")) for word in result["words"]), 1)

    def test_context_recovery_does_not_duplicate_existing_owned_word_or_later_chorus(self):
        result = merge_chunk_transcripts([
            (chunk(0), transcript([{"word": "Santo", "start": 119.9, "end": 120.3}])),
            (chunk(1), transcript([
                {"word": "Santo", "start": 1.7, "end": 2.1},
                {"word": "Santo", "start": 1.9, "end": 2.3},
                {"word": "Santo", "start": 12.0, "end": 12.4},
            ])),
        ])
        self.assertEqual([word["word"] for word in result["words"]], ["Santo", "Santo"])
        self.assertEqual(result["words"][-1]["start"], 130)
        self.assertFalse(any(word.get("boundary_recovered") for word in result["words"]))

    def test_context_recovery_requires_valid_temporal_agreement(self):
        invalid = [
            {"word": "Santo", "start": float("nan"), "end": 2.1},
            {"word": "Santo", "start": 1.7, "end": 9999},
            {"word": "Santo", "start": 0, "end": 0},
            {"word": "Santo", "start": 1.7, "end": 2.1, "timing_estimated": True},
            {"word": "diferente", "start": 1.7, "end": 2.1},
            {"word": "Santo", "start": 0.1, "end": 0.5},
        ]
        for other in invalid:
            with self.subTest(other=other):
                result = merge_chunk_transcripts([
                    (chunk(0), transcript([{"word": "Santo", "start": 119.9, "end": 120.3}])),
                    (chunk(1), transcript([other])),
                ])
                self.assertEqual(result["words"], [])

    def test_context_is_not_recovered_when_adjacent_chunk_is_missing(self):
        result = merge_chunk_transcripts([
            (chunk(0), transcript([{"word": "Santo", "start": 119.9, "end": 120.3}])),
        ])
        self.assertEqual(result["words"], [])

    def test_repeated_choruses_at_different_times_are_preserved(self):
        result = merge_chunk_transcripts([
            (chunk(0), transcript([{"word": "Santo", "start": 10, "end": 11}])),
            (chunk(1), transcript([{"word": "Santo", "start": 12, "end": 13}])),
        ])
        self.assertEqual([word["start"] for word in result["words"]], [10, 130])
        self.assertEqual(result["text"], "Santo\nSanto")

    def test_segments_without_words_use_absolute_time(self):
        result = merge_chunk_transcripts([(chunk(1), transcript([], [{"text": "Coro", "start": 5, "end": 8}]))])
        self.assertEqual(result["segments"][0], {"text": "Coro", "start": 123, "end": 126})

    def test_context_only_segment_does_not_reappear_as_untimed_fallback(self):
        response = {"text": "Antes", "segments": [{"text": "Antes", "start": 0, "end": 1}], "words": []}
        result = merge_chunk_transcripts([(chunk(1), response)])
        self.assertEqual(result["text"], "")
        self.assertEqual(result["segments"], [])

    def test_zero_or_missing_timestamps_get_honest_nonzero_interval(self):
        for segment in [{"text": "Coro", "start": 0, "end": 0}, {"text": "Coro"},
                        {"text": "Coro", "start": float("nan"), "end": 4},
                        {"text": "Coro", "start": 2, "end": 9999}]:
            with self.subTest(segment=segment):
                result = merge_chunk_transcripts([(chunk(1), transcript([], [segment]))])
                output = result["segments"][0]
                self.assertEqual((output["start"], output["end"]), (120, 240))
                self.assertTrue(output["timing_estimated"])
                self.assertEqual(output["timestamp_source"], "chunk_interval")
                self.assertTrue(result["warnings"])

    def test_text_only_and_empty_results_are_safe(self):
        result = merge_chunk_transcripts([(chunk(0, 50), {"text": "Sin tiempos", "model": "text-only"})])
        self.assertEqual(result["segments"][0]["end"], 50)
        self.assertEqual(result["model"], "text-only")
        self.assertEqual(merge_chunk_transcripts([])["text"], "")

    def test_unassigned_words_and_input_are_preserved(self):
        words = [{"word": "primero", "start": 3, "end": 4}, {"word": "segundo", "start": 10, "end": 11}]
        response = transcript(words, [{"text": "primero", "start": 2, "end": 5}])
        result = merge_chunk_transcripts([(chunk(1), response)])
        self.assertEqual(result["text"], "primero\nsegundo")
        self.assertEqual(words[0]["start"], 3)


if __name__ == "__main__":
    unittest.main()
