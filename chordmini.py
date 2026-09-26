"""Local ChordMini inference; pinned weights, no downloads during requests.

CQT follows ChordMini's AudioChordDataset, including arithmetic stereo mixdown.
Normalization is already inside the ONNX graph. Model scores are not calibrated
probabilities. The vocabulary has extensions but does not distinguish inversions.
"""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import tempfile

import numpy as np

REVISION = "086162411b8c4772774392be195e5c6f065d67ad"
REPOSITORY = "musetric/chordmini-onnx"
HASHES = {
    "chordnet.onnx": "cfe7703434ebd1c28ba2ded6601581ab41d8f7d1b40f285110445460e1b11154",
    "config.json": "1f26c11ebea51ec08f12e813eb213a729fa0ecc407ac7632dfdc7bad67e65aa4",
}
SAMPLE_RATE = 22050
HOP = 2048
SEQUENCE = 108
BATCH = 16  # The pinned graph has a static batch, despite its config metadata.


def model_directory():
    return Path(os.getenv("CHORDMINI_MODEL_DIR", str(Path.home() / ".cache" / "songlory" / "chordmini" / REVISION)))


def _verify(path, expected):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"ChordMini artifact checksum mismatch: {Path(path).name}")


def download():
    """Explicit setup command; atomic, verified artifact installation."""
    import requests
    directory = model_directory()
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in HASHES.items():
        target = directory / name
        if target.exists():
            try:
                _verify(target, digest)
                continue
            except ValueError:
                pass
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                temporary = Path(output.name)
                url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
                with requests.get(url, stream=True, timeout=(15, 120)) as response:
                    response.raise_for_status()
                    for block in response.iter_content(1024 * 1024):
                        output.write(block)
            _verify(temporary, digest)
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return directory


@lru_cache(maxsize=1)
def _runtime(directory):
    import onnxruntime as ort
    directory = Path(directory)
    for name, digest in HASHES.items():
        _verify(directory / name, digest)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(8, int(os.getenv("CHORDMINI_THREADS", "2"))))
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(directory / "chordnet.onnx"), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [BATCH, SEQUENCE, 144]:
        raise ValueError("Unexpected ChordMini input dimensions")
    return session, config


def is_available():
    try:
        _runtime(str(model_directory()))
        return True
    except (ImportError, OSError, ValueError, RuntimeError):
        return False


def predict_scores(features, session):
    """Bound memory to 16 windows; keep absolute frame order past batch edges."""
    if features.ndim != 2 or features.shape[1] != 144:
        raise ValueError("Expected CQT frames with 144 bins")
    frames = len(features)
    scores = np.empty((frames, 170), dtype=np.float32)
    group = BATCH * SEQUENCE
    for start in range(0, frames, group):
        count = min(group, frames - start)
        batch = np.zeros((group, 144), dtype=np.float32)
        batch[:count] = features[start:start + count]
        output = session.run(["logits"], {"features": batch.reshape(BATCH, SEQUENCE, 144)})[0]
        if output.shape != (BATCH, SEQUENCE, 170) or not np.isfinite(output).all():
            raise ValueError("Invalid ChordMini output")
        scores[start:start + count] = output.reshape(group, 170)[:count]
    return scores


def decode_scores(scores, vocabulary, duration, silent_frames=None):
    """Decode intervals without inventing tails, removing short chords or keys."""
    if not len(scores) or duration <= 0:
        return []
    from scipy.ndimage import uniform_filter1d
    # Reference model's temporal decoder; apply once over the entire timeline.
    smoothed = uniform_filter1d(scores, size=9, axis=0, mode="constant")
    indices = smoothed.argmax(axis=1)
    if silent_frames is not None:
        indices[np.asarray(silent_frames, dtype=bool)] = vocabulary.index("N")
    events = []
    for frame, index in enumerate(indices):
        time = frame * HOP / SAMPLE_RATE
        if time >= duration:
            break
        raw = vocabulary[int(index)]
        if not events or events[-1]["rawLabel"] != raw:
            if events:
                events[-1]["end"] = time
            events.append({"chord": "N" if raw == "X" else raw, "rawLabel": raw,
                           "time": time, "engine": "chordmini", "modelRevision": REVISION,
                           **({"uncertain": True} if raw == "X" else {})})
    if events:
        events[-1]["end"] = duration
    return events


def detect(audio_path):
    import librosa
    session, config = _runtime(str(model_directory()))
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if not len(audio):
        return []
    duration = len(audio) / sr
    if not np.isfinite(audio).all():
        raise ValueError("Non-finite audio samples")
    if np.max(np.abs(audio)) <= 1e-7:
        return [{"chord": "N", "rawLabel": "N", "time": 0., "end": duration,
                 "engine": "chordmini", "modelRevision": REVISION}]
    features = np.log(np.abs(librosa.cqt(audio, sr=sr, hop_length=HOP,
        fmin=config["fmin"], n_bins=144, bins_per_octave=24, norm=1,
        sparsity=0.01, window="hann", scale=True, pad_mode="constant")) + 1e-6).T.astype(np.float32)
    scores = predict_scores(features, session)
    # Only digital silence, never a relative loudness gate that removes quiet music.
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=HOP)[0]
    return decode_scores(scores, config["chordVocab"], duration, rms[:len(scores)] <= 1e-7)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Prepare verified local model weights")
    args = parser.parse_args()
    if args.download:
        print(download())
    else:
        parser.error("Use --download to prepare the model before starting the service")
