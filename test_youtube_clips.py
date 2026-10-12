"""Verifica el recorte con FFmpeg real y la limpieza también cuando falla."""
import io
import math
import os
import shutil
import struct
import subprocess
import unittest
import wave
from unittest.mock import patch

from fastapi import HTTPException
from youtube_clips import render_youtube_clip


class YouTubeClipTests(unittest.TestCase):
    def recording(self, video_id, workdir):
        self.workdir = workdir
        path = os.path.join(workdir, "audio.wav")
        with wave.open(path, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(48000)
            # El primer segundo es silencio, los siguientes son una señal conocida.
            samples = (0 if i < 48000 else int(8000 * math.sin(2 * math.pi * 440 * i / 48000)) for i in range(48000 * 7))
            output.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
        return path

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg no está instalado")
    def test_exact_clip_and_cleanup(self):
        result = render_youtube_clip("abcdefghijk", 1, 5, self.recording)
        with wave.open(io.BytesIO(result), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 2)
            self.assertEqual(audio.getframerate(), 48000)
            # El WAV por pipe tiene tamaño desconocido en la cabecera; contar PCM real.
            pcm = audio.readframes(48000 * 6)
            self.assertEqual(len(pcm), 48000 * 5 * 4)
            self.assertGreater(max(abs(value[0]) for value in struct.iter_unpack("<h", pcm[:48000])), 1000)
        self.assertFalse(os.path.exists(self.workdir))

    def test_failure_also_removes_download(self):
        with patch("youtube_clips.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 45)):
            with self.assertRaises(HTTPException) as error:
                render_youtube_clip("abcdefghijk", 1, 5, self.recording)
        self.assertEqual(error.exception.status_code, 502)
        self.assertFalse(os.path.exists(self.workdir))

    def test_rejects_invalid_ranges_before_download(self):
        for start, duration in [(-1, 5), (1, 6), (float("nan"), 5), (1, float("inf")), (719, 5), (True, 5)]:
            with self.subTest(start=start, duration=duration), self.assertRaises(HTTPException):
                render_youtube_clip("abcdefghijk", start, duration, lambda *_: self.fail("No debe descargar"))


if __name__ == "__main__":
    unittest.main()

