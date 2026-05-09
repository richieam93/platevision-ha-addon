# PlateVision v0.6.0 Pro+

Diese Version wurde zusätzlich geprüft und erweitert.

## Korrigierte Fehler
- `requirements.txt` von UTF-16 auf UTF-8 umgestellt, damit Docker/Pip die Datei zuverlässig lesen kann.
- Home-Assistant `webui` Port auf den Container-Port `5000` korrigiert.
- Versionsstände auf `0.6.0` aktualisiert.
- Python-Syntax und Jinja-Templates validiert.

## Neue Funktionen
- Mehrsprachige Oberfläche: DE, EN, FR, IT als konfigurierbare Sprache.
- Neue Diagnose-Seite mit Health Checks und Audit-Empfehlungen.
- Neues Einstellungs-Dashboard für Sprache, Layout, Erkennung, Suche, Datenschutz und Backup.
- Erkennungsprofile: `fast`, `balanced`, `strict`, `night`.
- Batch-Nummern-Test für mehrere OCR-Zeilen.
- Erweiterte Kennzeichen-Kandidatenbildung mit Score, Format und Maskierung.
- Autocomplete für historische Kennzeichen.
- Timeline API für Dashboard-Charts.
- Duplikat-Gruppen-API für ähnliche Kennzeichen.
- Config- und Watchlist-Import/Export.
- Historien-Cleanup nach Aufbewahrungsdauer.

## Neue API-Endpunkte
- `GET /diagnostics`
- `GET /api/i18n`
- `GET /api/settings/schema`
- `POST /api/config/general`
- `POST /api/config/ui`
- `POST /api/config/privacy`
- `GET /api/config/export`
- `POST /api/config/import`
- `POST /api/profile/apply/<profile>`
- `POST /api/plate/analyze`
- `POST /api/plate/batch-analyze`
- `GET /api/history/autocomplete`
- `GET /api/history/timeline`
- `GET /api/history/duplicates`
- `POST /api/history/cleanup`
- `GET /api/watchlist/export`
- `POST /api/watchlist/import`
- `GET /api/system/health`
- `GET /api/system/audit`
