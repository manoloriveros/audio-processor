"""Offline private-file timing contract and pre-parser upload limits."""
import asyncio
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_auto_markers import _load_functions, _marker_namespace, HttpError


MAX_BYTES = 25 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


def file_namespace():
    namespace = _marker_namespace()
    namespace.update(UploadFile=SimpleNamespace, File=lambda default: default,
        Form=lambda default: default, json=json, _MARKER_FILE_MAX_BYTES=MAX_BYTES,
        _MARKER_BODY_MAX_BYTES=MAX_BYTES + 64 * 1024, _MARKER_UPLOAD_CHUNK_BYTES=CHUNK_BYTES)
    return _load_functions("main.py", ["transcribe_file", "_marker_upload_response",
        "MarkerUploadLimitMiddleware"], namespace)


class Upload:
    filename = "../../secret-password@proxy.invalid.m4a"
    content_type = "application/octet-stream"

    def __init__(self, chunks=(b"offline mock audio",)):
        self.chunks = iter(chunks)
        self.read_sizes = []
        self.closed = False

    async def read(self, size):
        self.read_sizes.append(size)
        return next(self.chunks, b"")

    async def close(self):
        self.closed = True


class MarkerFileEndpointTests(unittest.TestCase):
    def setUp(self):
        self.ns = file_namespace()
        self.paths = []

        def probe(path, **kwargs):
            self.paths.append(path)
            self.assertEqual(Path(path).name, "recording.audio")
            self.assertTrue(self.ns["_JOB_SEMAPHORE"].locked())
            return 30
        self.probe = Mock(side_effect=probe)
        self.transcribe = Mock(return_value={"model": "whisper-1", "words": [
            {"word": "Cantamos", "start": 1, "end": 2},
            {"word": "alabanza", "start": 3, "end": 4}]})
        self.local = Mock()
        self.modules = patch.dict(sys.modules, {
            "transcription_chunks": SimpleNamespace(probe_audio_duration=self.probe),
            "transcription_service": SimpleNamespace(transcribe_audio=self.transcribe),
            "local_transcription": SimpleNamespace(transcribe_local_audio=self.local),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.upload = Upload()

    def request(self, video_id="dQw4w9WgXcQ", secret="test-secret"):
        return asyncio.run(self.ns["transcribe_file"](self.upload, video_id, secret))

    def assert_error(self, code, operation):
        with self.assertRaises(HttpError) as caught:
            operation()
        self.assertEqual(caught.exception.detail["code"], code)
        return caught.exception

    def test_success_uses_controlled_path_one_timing_pass_and_closes_upload(self):
        result = self.request()["recording"]
        self.assertEqual(result["videoId"], "dQw4w9WgXcQ")
        self.assertEqual(result["analysisVersion"], "auto-markers-v1")
        self.assertEqual(result["duration"], 30)
        self.transcribe.assert_called_once()
        self.assertEqual(self.transcribe.call_args.kwargs, {"api_key": "test-only-key",
            "timestamp_model": "whisper-1", "text_models": [], "prompt": None, "max_retries": 0})
        self.assertEqual(set(self.upload.read_sizes), {CHUNK_BYTES})
        self.assertTrue(self.upload.closed)
        self.assertTrue(all(not Path(path).parent.exists() for path in self.paths))
        self.local.assert_not_called()
        self.assertIs(self.ns["app"].routes["/transcribe-file"], self.ns["transcribe_file"])

    def test_auth_configuration_and_invalid_context_fail_before_reading(self):
        self.assert_error("UNAUTHORIZED", lambda: self.request(secret="wrong"))
        self.assert_error("UNAUTHORIZED", lambda: self.request(secret=None))
        for invalid in ["", "../../audio", "https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQextra"]:
            self.assert_error("INVALID_SOURCE", lambda: self.request(video_id=invalid))
        self.ns["OPENAI_API_KEY"] = None
        self.assert_error("CONFIGURATION", self.request)
        self.ns["API_SECRET"] = None
        self.assert_error("CONFIGURATION", self.request)
        self.assertFalse(self.upload.read_sizes)
        self.assertTrue(self.upload.closed)
        self.probe.assert_not_called()
        self.transcribe.assert_not_called()

    def test_empty_and_oversized_files_are_closed_before_any_model_call(self):
        self.upload = Upload(())
        self.assert_error("INVALID_AUDIO", self.request)
        self.upload = Upload([b"x" * CHUNK_BYTES] * 26)
        self.assertEqual(self.assert_error("INVALID_AUDIO", self.request).status_code, 413)
        self.assertEqual(set(self.upload.read_sizes), {CHUNK_BYTES})
        self.assertTrue(self.upload.closed)
        self.probe.assert_not_called()
        self.transcribe.assert_not_called()

    def test_invalid_audio_and_overlong_duration_are_measured_before_whisper(self):
        self.probe.side_effect = ValueError("untrusted media diagnostic secret-password")
        failure = self.assert_error("INVALID_AUDIO", self.request)
        self.assertNotIn("secret-password", str(failure.detail))
        self.upload = Upload()
        self.probe.side_effect = None
        self.probe.return_value = 721
        self.assert_error("DURATION_LIMIT", self.request)
        self.transcribe.assert_not_called()
        self.assertTrue(self.upload.closed)

    def test_failed_inference_has_no_retry_and_removes_audio(self):
        self.transcribe.side_effect = RuntimeError("uncertain paid chunk")
        self.assert_error("TRANSCRIPTION_FAILED", self.request)
        self.transcribe.assert_called_once()
        self.assertTrue(all(not Path(path).parent.exists() for path in self.paths))

    def test_cancelled_request_holds_slot_and_file_until_inference_finishes(self):
        started, release = threading.Event(), threading.Event()

        def infer(*args, **kwargs):
            started.set()
            if not release.wait(3):
                raise AssertionError("test did not release inference")
            self.assertTrue(Path(self.paths[0]).exists())
            self.assertTrue(self.ns["_JOB_SEMAPHORE"].locked())
            return {"model": "whisper-1", "words": [{"word": "test", "start": 1, "end": 2}]}
        self.transcribe.side_effect = infer

        async def cancel():
            task = asyncio.create_task(self.ns["transcribe_file"](self.upload, "dQw4w9WgXcQ", "test-secret"))
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
        asyncio.run(cancel())
        self.assertFalse(self.ns["_JOB_SEMAPHORE"].locked())
        self.assertTrue(self.upload.closed)
        self.assertTrue(all(not Path(path).parent.exists() for path in self.paths))


class MarkerUploadBodyLimitTests(unittest.TestCase):
    def setUp(self):
        self.ns = file_namespace()
        self.app_calls = 0
        self.read_calls = 0
        self.messages = []
        self.replayed = []

        async def app(scope, receive, send):
            self.app_calls += 1
            while True:
                message = await receive()
                self.replayed.append(message["body"])
                if not message.get("more_body"):
                    break
            await send({"type": "http.response.start", "status": 200, "headers": []})
        self.middleware = self.ns["MarkerUploadLimitMiddleware"](app)

    def request(self, bodies=(b"small multipart fixture",), headers=None, path="/transcribe-file"):
        chunks = iter(bodies)

        async def receive():
            self.read_calls += 1
            data = next(chunks)
            return {"type": "http.request", "body": data, "more_body": self.read_calls < len(bodies)}

        async def send(message):
            self.messages.append(message)
        scope = {"type": "http", "method": "POST", "path": path,
                 "headers": headers if headers is not None else [(b"x-api-secret", b"test-secret")]}
        asyncio.run(self.middleware(scope, receive, send))
        return self.messages[0]["status"]

    def test_unauthorized_and_declared_oversize_are_rejected_before_body_read(self):
        self.assertEqual(self.request(headers=[]), 401)
        self.assertEqual(self.read_calls, 0)
        self.messages = []
        self.assertEqual(self.request(headers=[(b"x-api-secret", b"test-secret"),
            (b"content-length", str(MAX_BYTES + 64 * 1024 + 1).encode())]), 413)
        self.assertEqual(self.read_calls, 0)
        self.assertEqual(self.app_calls, 0)
        self.ns["API_SECRET"] = None
        self.messages = []
        self.assertEqual(self.request(), 503)

    def test_untrusted_content_length_and_streamed_oversize_do_not_reach_parser(self):
        self.assertEqual(self.request(headers=[(b"x-api-secret", b"test-secret"), (b"content-length", b"invalid")]), 400)
        self.messages = []
        self.assertEqual(self.request(bodies=[b"x" * CHUNK_BYTES] * 26), 413)
        self.assertEqual(self.app_calls, 0)

    def test_replay_is_bounded_and_preserves_bytes_for_multipart_parser(self):
        content = b"a" * (CHUNK_BYTES + 128)
        self.assertEqual(self.request(bodies=[content, b"end"]), 200)
        self.assertEqual(b"".join(self.replayed), content + b"end")
        self.assertTrue(all(len(chunk) <= CHUNK_BYTES for chunk in self.replayed))
        self.assertEqual(self.app_calls, 1)

    def test_other_endpoints_pass_through(self):
        self.assertEqual(self.request(headers=[], path="/transcribe-url"), 200)
        self.assertEqual(self.app_calls, 1)


if __name__ == "__main__":
    unittest.main()
