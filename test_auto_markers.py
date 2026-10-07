"""Offline contract tests for timing-only transcription, without optional ML dependencies.

The selected production functions are compiled from their source AST. This permits
running these tests with the Python standard library; every remote/ML dependency
is mocked and no real audio or OpenAI request is used.
"""
import ast
import asyncio
from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class HttpError(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail


class App:
    def __init__(self):
        self.routes = {}

    def post(self, route):
        def register(function):
            self.routes[route] = function
            return function
        return register

    get = post


def _load_functions(filename, names, namespace):
    path = Path(__file__).with_name(filename)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _marker_namespace():
    namespace = {
        "asyncio": asyncio, "os": os, "tempfile": tempfile, "shutil": shutil,
        "math": math, "re": re, "_secrets": secrets, "HTTPException": HttpError,
        "Header": lambda default: default, "app": App(), "ProcessUrlBody": SimpleNamespace,
        "API_SECRET": "test-secret", "OPENAI_API_KEY": "test-only-key",
        "TRANSCRIPTION_ENGINE": "openai", "YT_MAX_DURATION": 720,
        "_YT_ID_RE": re.compile(r"^[a-zA-Z0-9_-]{11}$"),
        "logger": logging.getLogger("test-markers"), "_JOB_SEMAPHORE": asyncio.Semaphore(1),
    }
    return _load_functions("main.py", ["_extract_youtube_id", "_marker_error", "_download_marker_youtube_audio",
        "_transcribe_marker_recording", "_await_marker_thread", "transcribe_url"], namespace)


class MarkerEndpointTests(unittest.TestCase):
    def setUp(self):
        self.ns = _marker_namespace()
        self.real_download = self.ns["_download_marker_youtube_audio"]
        self.transcribe = Mock(return_value={"words": [
            {"word": "Alabar", "start": 4, "end": 4.8},
            {"word": "Dios", "start": 8, "end": 8.8}], "model": "whisper-1"})
        self.local = Mock(return_value={"words": [
            {"word": "Alabar", "start": 4, "end": 5}], "model": "faster-whisper/mock"})
        self.probe = Mock(return_value=12.5)
        self.modules = patch.dict(sys.modules, {
            "transcription_chunks": SimpleNamespace(probe_audio_duration=self.probe),
            "transcription_service": SimpleNamespace(transcribe_audio=self.transcribe),
            "local_transcription": SimpleNamespace(transcribe_local_audio=self.local),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.directories = []

        def download(video_id, workdir):
            self.directories.append(workdir)
            path = Path(workdir) / "audio.m4a"
            path.write_bytes(b"test-only mock audio")
            return str(path)
        self.download = Mock(side_effect=download)
        self.ns["_download_marker_youtube_audio"] = self.download

    def request(self, secret="test-secret", url="https://youtu.be/dQw4w9WgXcQ"):
        return asyncio.run(self.ns["transcribe_url"](SimpleNamespace(url=url), secret))

    def error(self, code, operation):
        with self.assertRaises(HttpError) as caught:
            operation()
        self.assertEqual(caught.exception.detail["code"], code)
        return caught.exception

    def test_success_only_single_timing_pass_and_cleanup(self):
        result = self.request()["recording"]
        self.assertEqual(result["videoId"], "dQw4w9WgXcQ")
        self.assertEqual(result["duration"], 12.5)
        self.assertEqual(result["analysisVersion"], "auto-markers-v1")
        self.assertEqual(result["words"][0], {"word": "Alabar", "start": 4., "end": 4.8})
        self.transcribe.assert_called_once()
        self.assertEqual(self.transcribe.call_args.kwargs,
            {"api_key": "test-only-key", "timestamp_model": "whisper-1",
             "text_models": [], "prompt": None, "max_retries": 0})
        self.local.assert_not_called()
        self.assertTrue(self.directories)
        self.assertTrue(all(not Path(path).exists() for path in self.directories))
        self.assertIs(self.ns["app"].routes["/transcribe-url"], self.ns["transcribe_url"])

    def test_missing_and_wrong_secret_fail_before_download(self):
        self.error("UNAUTHORIZED", lambda: self.request(None))
        self.error("UNAUTHORIZED", lambda: self.request("wrong-secret"))
        self.download.assert_not_called()
        self.transcribe.assert_not_called()

    def test_configuration_fails_closed_before_download(self):
        self.ns["API_SECRET"] = None
        self.assertEqual(self.error("CONFIGURATION", self.request).status_code, 503)
        self.ns["API_SECRET"] = "test-secret"
        self.ns["OPENAI_API_KEY"] = None
        self.error("CONFIGURATION", self.request)
        self.ns["TRANSCRIPTION_ENGINE"] = "typo"
        self.error("CONFIGURATION", self.request)
        self.download.assert_not_called()

    def test_spotify_and_invalid_source_are_not_transcribed(self):
        for url in ["https://open.spotify.com/track/123", "https://example.com/song.mp3", ""]:
            self.error("INVALID_SOURCE", lambda: self.request(url=url))
        self.download.assert_not_called()

    def test_local_engine_has_no_paid_fallback_or_key_requirement(self):
        self.ns["TRANSCRIPTION_ENGINE"] = "faster-whisper"
        self.ns["OPENAI_API_KEY"] = None
        self.assertEqual(self.request()["recording"]["model"], "faster-whisper/mock")
        self.local.assert_called_once()
        self.transcribe.assert_not_called()

    def test_duration_is_measured_before_any_paid_request(self):
        for duration in [0, -1, float("nan"), float("inf")]:
            self.probe.return_value = duration
            self.error("INVALID_AUDIO", self.request)
        self.probe.return_value = 720.1
        self.error("DURATION_LIMIT", self.request)
        self.ns["YT_MAX_DURATION"] = 1200
        self.error("DURATION_LIMIT", self.request)
        self.transcribe.assert_not_called()
        self.assertTrue(all(not Path(path).exists() for path in self.directories))

    def test_probe_failure_cleans_download_without_transcribing(self):
        self.probe.side_effect = RuntimeError("invalid media")
        self.error("INVALID_AUDIO", self.request)
        self.transcribe.assert_not_called()
        self.assertTrue(all(not Path(path).exists() for path in self.directories))

    def test_unsafe_or_estimated_word_times_never_leave_endpoint(self):
        invalid = [
            {"word": "empty", "start": 0, "end": 0},
            {"word": "negative", "start": -1, "end": 1},
            {"word": "missing", "start": 1},
            {"word": "outside", "start": 13, "end": 14},
            {"word": "estimated", "start": 1, "end": 2, "timing_estimated": True},
            {"word": "nan", "start": float("nan"), "end": 2},
            {"word": "inf", "start": 1, "end": float("inf")},
            {"word": "bad", "start": "bad", "end": 2},
            {"word": "boolean", "start": False, "end": True},
            {"word": "overflow", "start": 10 ** 500, "end": 2},
            {"word": "", "start": 1, "end": 2},
        ]
        self.transcribe.return_value = {"words": invalid}
        self.error("NO_MEASURED_WORDS", self.request)
        valid = {"word": "medido", "start": 0, "end": .4}
        self.transcribe.return_value = {"words": invalid + [valid], "model": "whisper-1"}
        self.assertEqual(self.request()["recording"]["words"], [valid])

    def test_failed_transcription_is_not_retried_and_temp_audio_is_removed(self):
        self.transcribe.side_effect = RuntimeError("failed after one paid chunk")
        self.error("TRANSCRIPTION_FAILED", self.request)
        self.transcribe.assert_called_once()
        self.assertTrue(all(not Path(path).exists() for path in self.directories))

    def test_download_failure_is_reported_and_cleaned(self):
        def fail(video_id, workdir):
            self.directories.append(workdir)
            raise HttpError(422, "YouTube bloqueo la descarga")
        self.download.side_effect = fail
        self.error("VIDEO_UNAVAILABLE", self.request)
        self.transcribe.assert_not_called()
        self.assertTrue(all(not Path(path).exists() for path in self.directories))

    def test_bounded_downloader_validation_error_is_preserved(self):
        self.download.side_effect = HttpError(400, {"code": "INVALID_AUDIO", "message": "Live stream rejected"})
        self.assertEqual(self.error("INVALID_AUDIO", self.request).status_code, 400)
        self.transcribe.assert_not_called()

    def test_source_errors_have_safe_logs_and_never_start_transcription(self):
        self.ns["_download_marker_youtube_audio"] = self.real_download
        for code in ["YOUTUBE_ACCESS_RESTRICTED", "NO_AUDIO_FORMAT", "JS_RUNTIME_UNAVAILABLE"]:
            with self.subTest(code=code), patch("marker_download.subprocess.run", return_value=SimpleNamespace(
                returncode=1, stdout=json.dumps({"code": code, "message": "secret-password https://user:pass@proxy.invalid"}))):
                with self.assertLogs("test-markers", level="WARNING") as observed:
                    failure = self.error(code, self.request)
                self.assertEqual(failure.status_code, 502)
                rendered = " ".join(observed.output) + str(failure.detail)
                self.assertIn("dQw4w9WgXcQ", rendered)
                self.assertIn(code, rendered)
                self.assertIn("MarkerDownloadError", rendered)
                self.assertNotIn("secret-password", rendered)
                self.assertNotIn("proxy.invalid", rendered)
        self.transcribe.assert_not_called()
        self.local.assert_not_called()
        self.probe.assert_not_called()

    def test_unexpected_download_exception_is_not_logged_or_returned_raw(self):
        self.download.side_effect = RuntimeError("secret-password https://user:pass@proxy.invalid")
        with self.assertLogs("test-markers", level="WARNING") as observed:
            failure = self.error("VIDEO_UNAVAILABLE", self.request)
        rendered = " ".join(observed.output) + str(failure.detail)
        self.assertIn("RuntimeError", rendered)
        self.assertNotIn("secret-password", rendered)
        self.assertNotIn("proxy.invalid", rendered)
        self.transcribe.assert_not_called()

    def test_semaphore_covers_download_and_transcription(self):
        observed = []
        previous_download = self.download.side_effect
        def download(*args):
            observed.append(self.ns["_JOB_SEMAPHORE"].locked())
            return previous_download(*args)
        self.download.side_effect = download
        self.transcribe.side_effect = lambda *args, **kwargs: (
            observed.append(self.ns["_JOB_SEMAPHORE"].locked()) or
            {"words": [{"word": "test", "start": 1, "end": 2}], "model": "whisper-1"})
        self.request()
        self.assertEqual(observed, [True, True])
        self.assertFalse(self.ns["_JOB_SEMAPHORE"].locked())

    def test_cancelled_caller_keeps_slot_and_audio_until_thread_finishes(self):
        started, release = threading.Event(), threading.Event()
        def infer(*args, **kwargs):
            started.set()
            if not release.wait(timeout=3):
                raise AssertionError("test thread was not released")
            self.assertTrue(Path(self.directories[0]).exists())
            self.assertTrue(self.ns["_JOB_SEMAPHORE"].locked())
            return {"words": [{"word": "test", "start": 1, "end": 2}], "model": "whisper-1"}
        self.transcribe.side_effect = infer
        async def cancel():
            task = asyncio.create_task(self.ns["transcribe_url"](
                SimpleNamespace(url="dQw4w9WgXcQ"), "test-secret"))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                task.cancel()
                await asyncio.sleep(.01)
                self.assertFalse(task.done())
                self.assertTrue(self.ns["_JOB_SEMAPHORE"].locked())
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(self.ns["_JOB_SEMAPHORE"].locked())
        asyncio.run(cancel())
        self.assertTrue(all(not Path(path).exists() for path in self.directories))


class HealthConfigurationTests(unittest.TestCase):
    def test_marker_configuration_does_not_claim_local_model_readiness(self):
        namespace = {"app": App(), "os": os, "_CHORDINO_AVAILABLE": False,
            "_ESSENTIA_AVAILABLE": False, "CHORD_ENGINE": "auto",
            "TRANSCRIPTION_TIMESTAMP_MODEL": "whisper-1",
            "OPENAI_TRANSCRIPTION_MODEL": "gpt-4o-transcribe", "separation": None,
            "musicai_engine": None, "YT_MAX_DURATION": 720}
        _load_functions("main.py", ["health"], namespace)
        load_model = Mock(side_effect=AssertionError("Health must not load model weights"))
        with patch.dict(sys.modules, {
            "chordmini": SimpleNamespace(is_available=lambda: False),
            "local_transcription": SimpleNamespace(_load_model=load_model),
        }):
            for secret, engine, key, expected in [
                (None, "openai", "mock", False),
                ("secret", "openai", None, False),
                ("secret", "unknown", "mock", False),
                ("secret", "openai", "mock", True),
                ("secret", "faster-whisper", None, True),
            ]:
                with self.subTest(secret=bool(secret), engine=engine, key=bool(key)):
                    namespace.update(API_SECRET=secret, TRANSCRIPTION_ENGINE=engine, OPENAI_API_KEY=key)
                    metadata = asyncio.run(namespace["health"]())["markerTranscription"]
                    self.assertEqual(metadata["configured"], expected)
                    self.assertNotIn("available", metadata)
                    self.assertEqual(metadata["maximumDuration"], 720)
                    self.assertEqual(metadata["analysisVersion"], "auto-markers-v1")
        load_model.assert_not_called()


class TimingServiceTests(unittest.TestCase):
    def setUp(self):
        self.ns = _load_functions("transcription_service.py", ["_field", "_timed_items",
            "_segments_with_words", "transcribe_chunk", "transcribe_audio"], {
            "math": math, "logger": logging.getLogger("test-service"),
            "align_corrected_words": Mock(side_effect=AssertionError("second text pass called")),
        })
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = str(Path(self.directory.name) / "mock.wav")
        Path(self.path).write_bytes(b"mock audio")

    def test_no_text_model_means_one_request_with_measured_word_timestamps(self):
        create = Mock(return_value={"text": "Alabar", "words": [
            {"word": "Alabar", "start": 1, "end": 1.5}], "segments": []})
        client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
        result = self.ns["transcribe_chunk"](client, self.path,
            timestamp_model="whisper-1", text_models=[], prompt=None)
        create.assert_called_once()
        self.assertEqual(create.call_args.kwargs["model"], "whisper-1")
        self.assertEqual(create.call_args.kwargs["timestamp_granularities"], ["word", "segment"])
        self.assertNotIn("prompt", create.call_args.kwargs)
        self.assertEqual(result["words"], [{"word": "Alabar", "start": 1., "end": 1.5}])

    def test_timestamp_failure_has_no_paid_text_fallback(self):
        create = Mock(side_effect=RuntimeError("mock API timeout"))
        client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
        with self.assertRaises(RuntimeError):
            self.ns["transcribe_chunk"](client, self.path,
                timestamp_model="whisper-1", text_models=[], prompt=None)
        create.assert_called_once()

    def test_sdk_retries_zero_and_client_chunks_cleaned_on_failure(self):
        completed = []
        @contextmanager
        def client(**kwargs):
            completed.append(kwargs)
            try:
                yield object()
            finally:
                completed.append("client_closed")
        @contextmanager
        def chunks(*args, **kwargs):
            try:
                yield iter([SimpleNamespace(path=self.path, core_end=12)])
            finally:
                completed.append("chunks_closed")
        self.ns["iter_audio_chunks"] = chunks
        self.ns["transcribe_chunk"] = Mock(side_effect=RuntimeError("mock API failure"))
        self.ns["merge_chunk_transcripts"] = Mock()
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=client)}):
            with self.assertRaises(RuntimeError):
                self.ns["transcribe_audio"](self.path, api_key="mock", timestamp_model="whisper-1",
                    text_models=[], prompt=None, max_retries=0)
        self.assertEqual(completed[0], {"api_key": "mock", "timeout": 90, "max_retries": 0})
        self.assertEqual(completed[1:], ["chunks_closed", "client_closed"])
        self.ns["merge_chunk_transcripts"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
