# PlateVision ProTraffic People Pro v0.8.1 Audit

## Ziel
Alle bestehenden Funktionen und Einstellungen aus v0.8.0 bleiben erhalten. Die Personenanalyse wurde additiv erweitert.

## Wichtige Korrektur
- Config-Merge bewahrt jetzt auch unbekannte/alte Custom-Keys. Dadurch gehen bei Updates keine benutzerdefinierten Einstellungen mehr verloren.

## Neue Personenfunktionen
- Modellscan für `models/` und Custom-Modellordner.
- Personen-Modellauswahl direkt im Tab `Personenanalyse & Personenzählung`.
- Endpunkte: `/api/models/available`, `/api/models/people/options`, `/api/models/people/select`.
- Validierung: `/api/config/validate`.
- Anwesenheitsschätzung: `/api/people/presence`.
- Personen-Historien-Cleanup: `/api/people/cleanup`.
- Testdaten-Simulation: `/api/people/test/simulate`.
- Filter für YOLO-Bildgröße, IoU, Fläche, Seitenverhältnis und optionale Personen-Zone.
- Zähl-Debounce und Mindest-Track-Alter gegen Fehlzählungen.

## Prüfung
- Python-Syntax geprüft.
- Alle Jinja/HTML-Templates geparst.
- Neue Routen statisch geprüft.
- Keine Original-Settingsbereiche entfernt.
