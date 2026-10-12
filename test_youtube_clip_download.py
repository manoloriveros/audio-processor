"""El recorte valida la duración antes de descargar y consulta metadatos una vez."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
import main


class ClipDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'audio.m4a'
        self.path.write_bytes(b'offline recording')
        self.info = {'duration': 120}
        self.ydl = SimpleNamespace(
            extract_info=Mock(return_value=self.info),
            process_ie_result=Mock(return_value=self.info),
            prepare_filename=Mock(return_value=str(self.path)),
        )
        self.options = None
        context = Mock()
        context.__enter__ = Mock(return_value=self.ydl)
        context.__exit__ = Mock(return_value=False)

        def youtube(options):
            self.options = options
            return context

        modules = patch.dict(sys.modules, {'yt_dlp': SimpleNamespace(YoutubeDL=youtube)})
        modules.start()
        self.addCleanup(modules.stop)
        environment = patch.dict(os.environ, {'YTDLP_PROXY': '', 'YTDLP_COOKIES_B64': ''})
        environment.start()
        self.addCleanup(environment.stop)

    def test_clip_download_uses_validated_metadata_once(self):
        result = main._download_setlist_clip_audio('abcdefghijk', self.directory.name)
        self.assertEqual(result, str(self.path))
        self.ydl.extract_info.assert_called_once_with('https://www.youtube.com/watch?v=abcdefghijk', download=False)
        self.ydl.process_ie_result.assert_called_once_with(self.info, download=True)
        self.assertEqual(self.options['max_filesize'], 25 * 1024 * 1024)

    def test_duration_limit_prevents_media_download(self):
        self.info['duration'] = main.YT_MAX_DURATION + 1
        with self.assertRaises(HTTPException) as error:
            main._download_setlist_clip_audio('abcdefghijk', self.directory.name)
        self.assertEqual(error.exception.status_code, 400)
        self.ydl.process_ie_result.assert_not_called()

    def test_source_restriction_is_not_retried(self):
        self.ydl.extract_info.side_effect = RuntimeError('Sign in to confirm you are not a bot')
        with self.assertRaises(HTTPException) as error:
            main._download_setlist_clip_audio('abcdefghijk', self.directory.name)
        self.assertEqual(error.exception.status_code, 422)
        self.assertEqual(self.ydl.extract_info.call_count, 1)
        self.ydl.process_ie_result.assert_not_called()


if __name__ == '__main__':
    unittest.main()

