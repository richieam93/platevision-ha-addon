#!/bin/sh
set -e

# Persistente Verzeichnisse (bleiben über Updates/Neustarts erhalten)
mkdir -p /data/uploads /data/models /data/data

# Modelle synchronisieren: neue Modelle aus dem Add-on-Image nach /data/models übernehmen,
# ohne bestehende oder vom Benutzer angepasste Modelle zu überschreiben.
# Wichtig für Updates: best.pt/last.pt erscheinen auch dann, wenn /data/models schon alte Modelle enthält.
if [ -d /app/models ]; then
  cp -an /app/models/. /data/models/ || true
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
