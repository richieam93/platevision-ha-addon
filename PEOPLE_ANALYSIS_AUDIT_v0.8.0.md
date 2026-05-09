# PlateVision v0.8.0 ProTraffic People - Änderungs- und Prüfbericht

## Ziel

Diese Version erweitert PlateVision additiv um eine Personenanalyse, ohne bestehende Einstellungen oder Funktionen zu entfernen.

## Erhaltene Originalbereiche

Alle bisherigen Haupt-Einstellungsbereiche bleiben vorhanden:

- Erkennung
- OCR / Texterkennung
- Bildvorverarbeitung
- Historie
- Speicher
- Allgemein
- Modelle
- Über

Zusätzlich bleibt die vorhandene Kennzeichen-, Auto-, RTSP-, Suche-, Dashboard-, Statistik- und Test-Funktionalität erhalten.

## Neue Personenfunktionen

- Neue Seite `/people` für Personenstatistik und Personen-Historie.
- Separater Personen-History-Manager mit Datei `data/people/history.json`.
- Personenerkennung per Schalter aktivierbar/deaktivierbar.
- Auswahl zwischen Standard YOLOv8 COCO-Personenklasse und eigenem YOLOv8-Human-Modell.
- Pfad-Auswahl für Modelle wie `models/human_best.pt`.
- Konfigurierbare Class IDs und Class Names.
- Virtuelle Zähllinie mit X-/Y-Achse, Position und Richtung.
- Einfacher Tracker mit Track-ID, Distanz und Timeout.
- Zählstrategien: virtuelle Linie, erstes Auftauchen, jede Erkennung.
- Eigene Personen-Historie mit Export als CSV/JSON.
- Live-Snapshot-Test für Personenanalyse.
- Test-Seite zeigt zusätzlich erkannte Personen an.
- Dashboard zeigt Personen heute an.
- Live-Overlay kann Personenboxen und Zähllinie anzeigen.

## Neue API-Endpunkte

- `/people`
- `/api/config/people`
- `/api/people/statistics`
- `/api/people/history`
- `/api/people/history/clear`
- `/api/people/export`
- `/api/people/test/snapshot`

## Kompatibilitätsprüfung

Gegenüber v0.7.2 wurden keine Default-Config-Sektionen entfernt. Auch keine bisherigen Top-Level-Keys der vorhandenen Config-Sektionen wurden entfernt.

Zusätzlich wurde nur die neue Sektion `people` ergänzt.

## Technische Prüfungen

Durchgeführt:

- Python Syntax-Check für `app.py`, `rtsp_handler.py`, `detector.py`.
- Jinja/HTML-Parsing aller Templates.
- Statischer Abgleich der `url_for(...)`-Verweise.
- Statischer Abgleich vorhandener API-Routen gegenüber v0.7.2.
- Statischer Abgleich vorhandener Config-Keys gegenüber v0.7.2.

Nicht durchgeführt:

- Echter RTSP-Livetest.
- Echter YOLO-/EasyOCR-Laufzeittest mit Kamera.
- Download oder Bündelung externer HumanDetection-Modelle.

## Hinweis zur Genauigkeit

Personenzählung ist abhängig von Kameraposition, Bildrate, Zähllinie und Bewegungsrichtung. Für zuverlässige Werte sollte die virtuelle Linie so positioniert werden, dass jede Person sie genau einmal überquert.
