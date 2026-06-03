FROM python:3.10-slim

ARG BUILD_VERSION=0.8.12
ARG BUILD_ARCH=amd64
LABEL io.hass.version="${BUILD_VERSION}" \
      io.hass.type="app" \
      io.hass.arch="${BUILD_ARCH}" \
      io.hass.name="PlateVision" \
      io.hass.description="Lokale Kennzeichenerkennung mit RTSP, Test-Upload und OCR-Labor"

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System-Abhängigkeiten (OpenCV/EasyOCR + ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-deu \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Python deps zuerst (bessere Docker-Cache-Nutzung)
COPY src/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# App-Code
COPY src/ /app/

# Startscript
COPY run.sh /run.sh
RUN chmod +x /run.sh

EXPOSE 5000
CMD ["/run.sh"]
