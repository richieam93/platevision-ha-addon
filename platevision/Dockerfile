FROM python:3.10-slim

ARG BUILD_VERSION=0.10.0
ARG BUILD_ARCH=amd64

LABEL org.opencontainers.image.title="PlateVision" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.source="https://github.com/richieam93/platevision-ha-addon" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      io.hass.version="${BUILD_VERSION}" \
      io.hass.type="app" \
      io.hass.arch="${BUILD_ARCH}"

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System-Abhängigkeiten (OpenCV/EasyOCR + ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
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

# Lizenz- und Herkunftshinweise im ausgelieferten Image
COPY LICENSE /app/LICENSE
COPY NOTICE.md /app/NOTICE.md
COPY THIRD_PARTY_NOTICES.md /app/THIRD_PARTY_NOTICES.md
COPY MODEL_PROVENANCE.md /app/MODEL_PROVENANCE.md
COPY third_party_licenses/ /app/third_party_licenses/
COPY legacy_licenses/ /app/legacy_licenses/

# App-Code
COPY src/ /app/

# Startscript
COPY run.sh /run.sh
RUN chmod +x /run.sh

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import json,urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:5000/api/system/live', timeout=8)); raise SystemExit(0 if data.get('ok') else 1)"

CMD ["/run.sh"]
