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

COPY main.py .

EXPOSE 8000

# Railway inyecta la variable PORT automaticamente
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
