FROM python:3.10-slim

LABEL org.opencontainers.image.title="PlateVision" \
      org.opencontainers.image.version="0.9.0" \
      org.opencontainers.image.source="https://github.com/richieam93/platevision-ha-addon" \
      org.opencontainers.image.licenses="AGPL-3.0-only"

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
CMD ["/run.sh"]
