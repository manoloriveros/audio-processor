from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
from fastapi.testclient import TestClient

import local_transcription as local
import main
from transcription_chunks import AudioChunk


@contextmanager
def _chunks(*args, **kwargs):
    yield iter([AudioChunk('first.wav', 0, 0, 0, 120, 122, False),
                AudioChunk('second.wav', 1, 118, 120, 240, 240, True)])


def _segment(text, start, end):
    word = SimpleNamespace(word=text, start=start, end=end, probability=.95)
    return SimpleNamespace(text=text, start=start, end=end, words=[word])


def test_local_chunks_keep_late_repetitions_at_absolute_times(monkeypatch):
    transcribe = Mock(side_effect=[(iter([_segment('Volver', 10, 11)]), None),
                                   (iter([_segment('Volver', 12, 13)]), None)])
    monkeypatch.setattr(local, '_load_model', lambda **kwargs: SimpleNamespace(transcribe=transcribe))
    monkeypatch.setattr(local, 'iter_audio_chunks', _chunks)
    result = local.transcribe_local_audio('song.mp3')
    assert [word['start'] for word in result['words']] == [10, 130]
    assert [word['word'] for word in result['words']] == ['Volver', 'Volver']
    assert result['duration'] == 240
    assert result['chunkCount'] == 2
    assert all('initial_prompt' not in call.kwargs for call in transcribe.call_args_list)
    assert all(call.kwargs['condition_on_previous_text'] is False for call in transcribe.call_args_list)


def test_local_failure_never_returns_a_partial_song(monkeypatch):
    transcribe = Mock(side_effect=[(iter([_segment('Volver', 10, 11)]), None), RuntimeError('out of memory')])
    monkeypatch.setattr(local, '_load_model', lambda **kwargs: SimpleNamespace(transcribe=transcribe))
    monkeypatch.setattr(local, 'iter_audio_chunks', _chunks)
    with pytest.raises(RuntimeError, match='out of memory'):
        local.transcribe_local_audio('song.mp3')


def test_http_inference_loads_only_prepared_weights(monkeypatch):
    factory = Mock(return_value=object())
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=factory))
    local._load_model.cache_clear()
    try:
        local._load_model('large-v3-turbo', 'cpu', 'int8', 4, None)
        assert factory.call_args.kwargs['local_files_only'] is True
    finally:
        local._load_model.cache_clear()


def test_local_mode_never_calls_configured_paid_engines(monkeypatch):
    monkeypatch.setattr(main, 'TRANSCRIPTION_ENGINE', 'faster-whisper')
    monkeypatch.setattr(main, 'OPENAI_API_KEY', None)
    monkeypatch.setenv('LLM_STRUCTURE', '1')
    premium = Mock()
    premium.is_configured.return_value = True
    structure = SimpleNamespace(infer_repeated_sections=lambda sections: sections,
                                apply_structure=Mock(side_effect=AssertionError('paid API called')))
    monkeypatch.setattr(main, 'musicai_engine', premium)
    monkeypatch.setattr(main, 'structuring', structure)
    monkeypatch.setattr(main, 'separation', None)
    monkeypatch.setattr(local, 'transcribe_local_audio', lambda path: {
        'text': 'Volver', 'segments': [{'text': 'Volver', 'start': 1, 'end': 2}],
        'words': [], 'model': 'faster-whisper/mock', 'duration': 3, 'chunkCount': 1})
    monkeypatch.setattr(main, 'detect_chords', lambda *args, **kwargs: [{'chord': 'E', 'time': 0}])
    result = main.run_pipeline('song.mp3')
    assert result['transcriptionModel'] == 'faster-whisper/mock'
    premium.process.assert_not_called()
    structure.apply_structure.assert_not_called()


def test_unknown_engine_fails_before_any_premium_request(monkeypatch):
    monkeypatch.setattr(main, 'TRANSCRIPTION_ENGINE', 'misspelled')
    premium = Mock()
    premium.is_configured.return_value = True
    monkeypatch.setattr(main, 'musicai_engine', premium)
    with pytest.raises(ValueError, match='Motor de transcripcion'):
        main.run_pipeline('song.mp3')
    premium.process.assert_not_called()


@pytest.mark.parametrize('route', ['/process', '/process-url'])
def test_local_endpoints_work_without_openai_key_and_still_require_secret(monkeypatch, tmp_path, route):
    monkeypatch.setattr(main, 'TRANSCRIPTION_ENGINE', 'faster-whisper')
    monkeypatch.setattr(main, 'OPENAI_API_KEY', None)
    monkeypatch.setattr(main, 'API_SECRET', 'local-test-secret')
    audio = tmp_path / 'song.wav'
    audio.write_bytes(b'mocked audio')
    monkeypatch.setattr(main, '_download_youtube_audio', lambda *args: str(audio))
    monkeypatch.setattr(main, 'run_pipeline', lambda path: {'sections': [], 'detectedKey': 'E'})
    payload = {'files': {'file': ('song.wav', b'mocked audio', 'audio/wav')}} if route == '/process' else {
        'json': {'url': 'https://youtu.be/dQw4w9WgXcQ'}}
    with TestClient(main.app) as client:
        assert client.post(route, **payload).status_code == 401
        response = client.post(route, headers={'x-api-secret': 'local-test-secret'}, **payload)
        assert response.status_code == 200, response.text

@pytest.mark.parametrize('engine,expected_model,expected_text', [
    ('faster-whisper', 'large-v3', None),
    ('openai', 'whisper-1', 'gpt-4o-transcribe'),
])
def test_health_exposes_transcription_model_without_loading_weights(monkeypatch, engine, expected_model, expected_text):
    monkeypatch.setattr(main, 'TRANSCRIPTION_ENGINE', engine)
    monkeypatch.setattr(main, 'OPENAI_TRANSCRIPTION_MODEL', 'gpt-4o-transcribe')
    monkeypatch.setenv('LOCAL_WHISPER_MODEL', 'large-v3')
    monkeypatch.setattr(local, '_load_model', Mock(side_effect=AssertionError('Health must not load gigabyte weights')))
    with TestClient(main.app) as client:
        data = client.get('/health').json()
    assert data['transcriptionEngine'] == engine
    assert data['transcriptionModel'] == expected_model
    assert data['transcriptionTextModel'] == expected_text

@pytest.mark.parametrize('explicit,expected', [(None, 60), (90, 90)])
def test_local_chunk_configuration_is_applied_and_explicit_argument_takes_priority(monkeypatch, explicit, expected):
    calls = []
    @contextmanager
    def configured_chunks(*args, **kwargs):
        calls.append(kwargs)
        yield iter([AudioChunk('song.wav', 0, 0, 0, 10, 10, True)])
    monkeypatch.setenv('LOCAL_WHISPER_CHUNK_SECONDS', '60')
    monkeypatch.setattr(local, 'iter_audio_chunks', configured_chunks)
    monkeypatch.setattr(local, '_load_model', lambda **_: SimpleNamespace(
        transcribe=lambda *args, **kwargs: (iter([_segment('Volver', 1, 2)]), None)))
    result = local.transcribe_local_audio('song.mp3', chunk_seconds=explicit)
    assert calls[0]['chunk_seconds'] == expected
    assert calls[0]['overlap_seconds'] == 2
    assert result['chunkSeconds'] == expected
    assert result['overlapSeconds'] == 2


def test_invalid_local_chunk_text_fails_without_loading_a_model(monkeypatch):
    monkeypatch.setenv('LOCAL_WHISPER_CHUNK_SECONDS', 'not-seconds')
    model = Mock(side_effect=AssertionError('Invalid configuration must not load weights'))
    monkeypatch.setattr(local, '_load_model', model)
    with pytest.raises(ValueError):
        local.transcribe_local_audio('song.mp3')
    model.assert_not_called()
