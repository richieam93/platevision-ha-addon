# PlateVision v0.5.0 - Pro-Erweiterung

Diese Version erweitert das Add-on um mehrere neue Funktionen:

## Neue Oberfläche
- Komplett überarbeitetes Dashboard Pro mit Live-Metriken, Charts, Top-Kennzeichen und Schnellaktionen.
- Neue Seite **Suche & Analyse** mit erweiterten Filtern und Watchlist-Verwaltung.
- Neue **Pro-Erkennung** in den Einstellungen.

## Bessere Nummern-/Kennzeichenerkennung
- Zentrale Kennzeichen-Normalisierung.
- Intelligente OCR-Korrektur für typische Verwechslungen wie `O/0`, `I/1`, `S/5`, `B/8`.
- Länder-/Format-Hinweis für Auto, CH, DE und AT.
- Validierung per Länge, automatischem Format oder optionaler Regex.
- Speicherung von normalisiertem Kennzeichen, Format und Validierungsstatus in der Historie.

## Mehr Suchfunktionen
- Fuzzy-Suche.
- Regex-Suche.
- Filter nach Datum, Konfidenz, Quelle, Fahrzeugtyp, Farbe, gültigem Format und Watchlist-Treffern.
- Sortierung nach Zeit, Kennzeichen oder Konfidenz.
- CSV- und JSON-Export der gefilterten Treffer.

## Watchlist
- Kennzeichen hinzufügen, benennen und kategorisieren.
- Automatische Markierung von Treffern in Historie, Suche und Dashboard.

## Neue API-Endpunkte
- `GET/POST /api/history/search`
- `GET /api/history/facets`
- `GET /api/history/export?format=csv|json`
- `GET /api/dashboard/overview`
- `POST /api/plate/normalize`
- `GET/POST /api/watchlist`
- `DELETE /api/watchlist/<item_id>`
- `POST /api/config/advanced`
- `POST /api/config/reset`
