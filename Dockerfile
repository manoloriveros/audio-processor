FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV CHORD_ENGINE=chordino

# Instalar ffmpeg y dependencias de sistema para Chordino + Librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libyaml-0-2 \
    libchromaprint1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# vamp (dependencia de chord-extractor) necesita numpy disponible durante su instalacion.
RUN pip install --no-cache-dir "numpy>=1.23.0,<2"
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py separation.py structuring.py musicai_engine.py ./

# Pre-descargar el modelo de separacion vocal (MDX-Net ONNX) en la imagen
# para evitar la descarga en el primer request. Kim_Vocal_2: buen equilibrio
# calidad/velocidad en CPU para voz e instrumental.
ENV MODEL_FILE_DIR=/models
RUN audio-separator --download_model_only --model_filename "Kim_Vocal_2.onnx" --model_file_dir /models || \
    python -c "from audio_separator.separator import Separator; s = Separator(model_file_dir='/models'); s.download_model_files('Kim_Vocal_2.onnx')" || \
    echo "AVISO: no se pudo pre-descargar el modelo; se descargara en el primer uso"

EXPOSE 8000

# Railway inyecta la variable PORT automaticamente
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
