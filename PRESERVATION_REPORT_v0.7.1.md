# PlateVision v0.7.1 – Preservation Report

Ziel: Alle Einstellungen und Funktionen aus dem Original-Add-on bleiben vorhanden. Die ProTraffic-Funktionen werden nur ergänzt.

## Original-Konfigurationsbereiche erhalten

- `rtsp`
- `detection`
- `ocr`
- `general`
- `history`
- `models`

## Original-Einstellungsseite wiederhergestellt und erweitert

Die Datei `platevision/src/templates/settings.html` basiert wieder auf der ursprünglichen Einstellungsseite mit den Original-Tabs:

- Erkennung
- OCR / Texterkennung
- Bildvorverarbeitung
- Historie
- Speicher
- Allgemein
- Modelle
- Über

Zusätzlich wurden neue Tabs ergänzt:

- Kennzeichen Pro
- Suche
- Statistik / Verkehr
- Dashboard & Layout
- Datenschutz & Alerts

## Original-API-Routen erhalten

Die bekannten Endpunkte für Stream, Config, History, Storage, Processing, Latest, Models und System bleiben vorhanden. Neue Endpunkte wurden additiv ergänzt.

## Add-on-Konfiguration

Der ursprüngliche `webui`-Portverweis wurde auf `http://[HOST]:[PORT:8087]` zurückgesetzt, damit die ursprüngliche Home-Assistant-Zuordnung erhalten bleibt.

## Hinweis

Diese Version löscht keine Original-Einstellung. Wo neue Defaults hinzugefügt wurden, werden sie über den bestehenden Merge-Mechanismus ergänzt, ohne bestehende `data/config.json` Werte zu überschreiben.
