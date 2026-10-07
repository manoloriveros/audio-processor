"""Offline safety checks for the automatic-marker-only YouTube downloader."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from marker_download import (
    DOWNLOAD_TIMEOUT_SECONDS, MAX_DOWNLOAD_BYTES, MarkerDownloadError,
    SOURCE_ERROR_MESSAGES, _classify_source_error, _source_error_code,
    _download_in_child, _validate_metadata, download_marker_audio,
)


class MarkerDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workdir = self.directory.name
        self.path = Path(self.workdir) / "audio.m4a"
        self.path.write_bytes(b"offline mock audio")
        self.info = {"duration": 30, "is_live": False, "live_status": "not_live"}
        self.ydl = SimpleNamespace(
            extract_info=Mock(side_effect=lambda url, download: dict(self.info)),
            prepare_filename=Mock(return_value=str(self.path)),
        )
        self.options = None

        class Context:
            def __enter__(inner):
                return self.ydl
            def __exit__(inner, *args):
                return False

        def youtube(options):
            self.options = options
            return Context()
        self.module = patch.dict(sys.modules, {"yt_dlp": SimpleNamespace(YoutubeDL=youtube)})
        self.module.start()
        self.addCleanup(self.module.stop)
        self.environment = patch.dict("os.environ", {"YTDLP_PROXY": "", "YTDLP_COOKIES_B64": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def child(self):
        return _download_in_child("dQw4w9WgXcQ", self.workdir, 720)

    def assert_error(self, code, operation):
        with self.assertRaises(MarkerDownloadError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_completed_recording_uses_bounded_audio_download(self):
        self.assertEqual(self.child(), str(self.path.resolve()))
        self.assertEqual([call.kwargs["download"] for call in self.ydl.extract_info.call_args_list], [False, True])
        self.assertEqual(self.options["max_filesize"], MAX_DOWNLOAD_BYTES)
        self.assertEqual(self.options["retries"], 0)
        self.assertEqual(self.options["fragment_retries"], 0)
        self.assertNotIn("best", self.options["format"].split("/"))

    def test_live_upcoming_and_post_live_fail_before_download(self):
        for change in [{"is_live": True}, {"is_upcoming": True},
                       {"live_status": "is_live"}, {"live_status": "is_upcoming"},
                       {"live_status": "post_live"}]:
            with self.subTest(change=change):
                self.ydl.extract_info.reset_mock()
                self.info = {"duration": 30, **change}
                self.assert_error("INVALID_AUDIO", self.child)
                self.assertEqual(self.ydl.extract_info.call_count, 1)
                self.assertFalse(self.ydl.extract_info.call_args.kwargs["download"])

    def test_unknown_invalid_and_overlong_duration_fail_before_download(self):
        for duration in [None, "unknown", 0, -1, False, True, float("nan"), float("inf"), 10 ** 500]:
            with self.subTest(duration=str(duration)[:20]):
                self.ydl.extract_info.reset_mock()
                self.info = {"duration": duration}
                self.assert_error("INVALID_AUDIO", self.child)
                self.assertEqual(self.ydl.extract_info.call_count, 1)
        self.info = {"duration": 721}
        self.assert_error("DURATION_LIMIT", self.child)
        _validate_metadata({"duration": 720}, 720)
        _validate_metadata({"duration": 30, "was_live": True, "live_status": "was_live"}, 720)

    def test_download_filter_rechecks_metadata_after_preflight(self):
        self.child()
        match_filter = self.options["match_filter"]
        self.assertIsNone(match_filter({}, incomplete=True))
        self.assert_error("INVALID_AUDIO", lambda: match_filter({"duration": 30, "is_live": True}))
        self.assert_error("DURATION_LIMIT", lambda: match_filter({"duration": 800}))

    def test_progress_limit_stops_even_when_remote_filesize_is_unknown(self):
        def extract(url, download):
            if download:
                try:
                    self.options["progress_hooks"][0]({"downloaded_bytes": MAX_DOWNLOAD_BYTES + 1})
                except MarkerDownloadError as exc:
                    raise RuntimeError("yt-dlp wrapped the progress hook error") from exc
            return dict(self.info)
        self.ydl.extract_info.side_effect = extract
        self.assert_error("INVALID_AUDIO", self.child)

    def test_child_rejects_paths_outside_its_workdir(self):
        self.ydl.prepare_filename.return_value = __file__
        self.assert_error("VIDEO_UNAVAILABLE", self.child)

    def test_download_failure_never_exposes_remote_error_or_credentials(self):
        self.ydl.extract_info.side_effect = RuntimeError("proxy user:secret or cookies")
        failure = self.assert_error("VIDEO_UNAVAILABLE", self.child)
        self.assertNotIn("secret", str(failure))

    def test_source_error_classification_is_static_and_safe(self):
        cases = [
            ("Sign in to confirm you’re not a bot", "YOUTUBE_ACCESS_RESTRICTED"),
            ("Sign in to confirm your age", "YOUTUBE_ACCESS_RESTRICTED"),
            ("This video is private", "YOUTUBE_ACCESS_RESTRICTED"),
            ("Please log in to watch this video", "YOUTUBE_ACCESS_RESTRICTED"),
            ("This video is not available in your country", "YOUTUBE_ACCESS_RESTRICTED"),
            ("Join this channel for members-only content", "YOUTUBE_ACCESS_RESTRICTED"),
            ("Requested format is not available. Use --list-formats", "NO_AUDIO_FORMAT"),
            ("Only images are available for download", "NO_AUDIO_FORMAT"),
            ("No supported JavaScript runtime could be found", "JS_RUNTIME_UNAVAILABLE"),
            ("yt-dlp-ejs is not installed", "JS_RUNTIME_UNAVAILABLE"),
            ("Remote components challenge solver script (deno) were skipped", "JS_RUNTIME_UNAVAILABLE"),
            ("HTTP Error 503: Service Unavailable", "VIDEO_UNAVAILABLE"),
            ("connection timed out", "VIDEO_UNAVAILABLE"),
        ]
        for message, code in cases:
            with self.subTest(message=message):
                self.assertEqual(_source_error_code(message), code)
                raw = RuntimeError(message + " https://user:secret-password@proxy.invalid/video?token=private-token")
                failure = _classify_source_error(raw)
                self.assertEqual(failure.code, code)
                self.assertEqual(failure.status, 502)
                self.assertEqual(failure.message, SOURCE_ERROR_MESSAGES[code])
                self.assertNotIn("secret-password", str(failure))
                self.assertNotIn("private-token", str(failure))

    def test_child_preserves_source_code_without_raw_provider_text(self):
        self.ydl.extract_info.side_effect = RuntimeError(
            "Sign in to confirm you're not a bot https://user:secret-password@proxy.invalid")
        failure = self.assert_error("YOUTUBE_ACCESS_RESTRICTED", self.child)
        self.assertNotIn("secret-password", str(failure))
        self.assertEqual(self.ydl.extract_info.call_count, 1)

    def test_runtime_warning_explains_missing_formats_without_leaking_warning(self):
        def fail(url, download):
            self.options["logger"].warning("No supported JavaScript runtime could be found; proxy secret-password")
            raise RuntimeError("Requested format is not available")
        self.ydl.extract_info.side_effect = fail
        self.assert_error("JS_RUNTIME_UNAVAILABLE", self.child)
        self.assertFalse(self.options["no_warnings"])
        failure = _classify_source_error(RuntimeError("Sign in to confirm you're not a bot"), {"JS_RUNTIME_UNAVAILABLE"})
        self.assertEqual(failure.code, "YOUTUBE_ACCESS_RESTRICTED")

    def test_parent_source_payload_preserves_whitelisted_code_and_discards_message(self):
        for code, expected in SOURCE_ERROR_MESSAGES.items():
            with self.subTest(code=code), patch("marker_download.subprocess.run", return_value=SimpleNamespace(
                returncode=1, stdout=json.dumps({"code": code, "message": "secret-password https://user:pass@proxy.invalid"}))):
                failure = self.assert_error(code, lambda: download_marker_audio("dQw4w9WgXcQ", self.workdir, 720))
                self.assertEqual(failure.status, 502)
                self.assertEqual(failure.message, expected)
        for untrusted in ["HTTP403-secret-password", {"message": "secret-password"}]:
            with patch("marker_download.subprocess.run", return_value=SimpleNamespace(
                returncode=1, stdout=json.dumps({"code": untrusted, "message": "secret-password"}))):
                failure = self.assert_error("VIDEO_UNAVAILABLE", lambda: download_marker_audio("dQw4w9WgXcQ", self.workdir, 720))
                self.assertNotIn("secret-password", str(failure))

    def test_parent_enforces_hard_process_timeout_and_no_shell(self):
        with patch("marker_download.subprocess.run", side_effect=subprocess.TimeoutExpired(["test"], 180)) as run:
            failure = self.assert_error("VIDEO_UNAVAILABLE", lambda:
                download_marker_audio("dQw4w9WgXcQ", self.workdir, 720))
        self.assertEqual(failure.status, 502)
        self.assertEqual(run.call_args.kwargs["timeout"], DOWNLOAD_TIMEOUT_SECONDS)
        self.assertFalse(run.call_args.kwargs.get("shell", False))
        self.assertEqual(run.call_args.args[0][-3:], ["dQw4w9WgXcQ", self.workdir, "720"])

    def test_parent_validates_source_before_spawning(self):
        with patch("marker_download.subprocess.run") as run:
            self.assert_error("INVALID_SOURCE", lambda: download_marker_audio("$(bad)", self.workdir, 720))
        run.assert_not_called()

    def test_parent_accepts_verified_file_and_preserves_child_validation_error(self):
        with patch("marker_download.subprocess.run", return_value=SimpleNamespace(
            returncode=0, stdout=json.dumps({"path": str(self.path)}))):
            self.assertEqual(download_marker_audio("dQw4w9WgXcQ", self.workdir, 720), str(self.path.resolve()))
        with patch("marker_download.subprocess.run", return_value=SimpleNamespace(
            returncode=1, stdout=json.dumps({"code": "DURATION_LIMIT", "message": "exceeds limit"}))):
            self.assert_error("DURATION_LIMIT", lambda: download_marker_audio("dQw4w9WgXcQ", self.workdir, 720))

    def test_parent_rejects_missing_unsafe_or_oversized_output(self):
        for result in ["not json", json.dumps({"path": __file__}), json.dumps({"path": str(self.path) + ".missing"})]:
            with patch("marker_download.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=result)):
                self.assert_error("VIDEO_UNAVAILABLE", lambda: download_marker_audio("dQw4w9WgXcQ", self.workdir, 720))
        with self.path.open("wb") as file:
            file.truncate(MAX_DOWNLOAD_BYTES + 1)
        with patch("marker_download.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps({"path": str(self.path)}))):
            self.assert_error("INVALID_AUDIO", lambda: download_marker_audio("dQw4w9WgXcQ", self.workdir, 720))


if __name__ == "__main__":
    unittest.main()
