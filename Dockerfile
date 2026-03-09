FROM python:3.11-slim

# Instalar ffmpeg y dependencias de sistema para Essentia + Librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libyaml-0-2 \
    libchromaprint1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Essentia: instalar best-effort (no hay release estable para Python 3.11)
# Si falla, el servicio usa Librosa como fallback automaticamente
RUN pip install --no-cache-dir essentia==2.1b6.dev1389 || true

COPY main.py .

EXPOSE 8000

# Railway inyecta la variable PORT automaticamente
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
