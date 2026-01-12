#!/bin/sh
set -e

# Persistente Verzeichnisse (bleiben über Updates/Neustarts erhalten)
mkdir -p /data/uploads /data/models /data/data

# Initiale Daten einmalig rüberkopieren, falls /data leer ist
if [ -d /app/models ] && [ -z "$(ls -A /data/models 2>/dev/null)" ]; then
  cp -a /app/models/. /data/models/ || true
fi

if [ -d /app/data ] && [ ! -f /data/data/.initialized ]; then
  cp -a /app/data/. /data/data/ || true
  touch /data/data/.initialized
fi

# /app/* auf persistente Pfade umbiegen
rm -rf /app/uploads /app/models /app/data
ln -s /data/uploads /app/uploads
ln -s /data/models /app/models
ln -s /data/data /app/data

cd /app
exec python /app/app.py
