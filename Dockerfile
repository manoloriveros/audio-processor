FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV CHORD_ENGINE=chordino

# ffmpeg y librerias de sistema para Chordino + Librosa + audio-separator
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libyaml-0-2 \
    libchromaprint1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# 1. numpy<2 primero: vamp (chord-extractor) lo necesita durante su compilacion.
# 2. torch/torchvision desde el indice CPU de PyTorch: evita descargar ~2.5 GB
#    de librerias CUDA (cublas, cudnn, nccl...) inutiles en Railway (solo CPU).
# 3. resto de requirements: vamp y diffq compilan extensiones nativas, asi que
#    build-essential se instala solo para este paso y se purga en la misma capa.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir "numpy>=1.23.0,<2" \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY main.py separation.py structuring.py musicai_engine.py ./

# Pre-descargar el modelo de separacion vocal (MDX-Net ONNX) en la imagen para
# evitar la descarga en el primer request. Kim_Vocal_2: buen equilibrio
# calidad/velocidad en CPU para voz e instrumental.
ENV MODEL_FILE_DIR=/models
RUN audio-separator --download_model_only --model_filename "Kim_Vocal_2.onnx" --model_file_dir /models || \
    python -c "from audio_separator.separator import Separator; s = Separator(model_file_dir='/models'); s.download_model_files('Kim_Vocal_2.onnx')" || \
    echo "AVISO: no se pudo pre-descargar el modelo; se descargara en el primer uso"

EXPOSE 8000

# Railway inyecta la variable PORT automaticamente
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
