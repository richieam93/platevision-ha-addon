# Changelog

## 0.8.28

- Dashboard komplett überarbeitet: keine Modell-Details mehr, Fokus auf letztes Auto, letzte Person und heutige Kennzahlen.
- RTSP/Webstream-Autostart mit Startverzögerung ergänzt.
- Autostart-Einstellungen in RTSP Stream und in den allgemeinen Einstellungen sichtbar.
- `examples/` auf aktuelle API-Endpunkte, Personenanalyse und Lovelace-Beispiele aktualisiert; alte/nicht mehr passende Dateien entfernt.



## 0.8.27

- Frontend stabilisiert: optionale/verschobene UI-Elemente verursachen keine `Cannot read properties of null (reading style)` Fehler mehr.
- Gemeinsame DOM-Helfer ergänzt und harte `.style`-Zugriffe abgesichert.
- UI-Fehler-Toast zeigt jetzt zusätzlich Datei/Zeile/Spalte, damit künftige Probleme schneller gefunden werden.
- Funktionalität der 0.8.26 bleibt erhalten.

## 0.8.26

- Home-Assistant-Add-on-Metadaten korrigiert, damit die neue Version im Update-Dialog sauber erkannt wird.
- `CHANGELOG.md` für die Add-on-Update-Ansicht ergänzt.
- Version in `config.yaml`, Webinterface, API-Health und About-Ausgabe vereinheitlicht.
- Basis bleibt die funktionierende 0.8.25 mit Personen-Zähllinie, fast-plate-ocr, YOLO-Kennzeichen, YOLO-Fahrzeuganalyse und CPU-/RTSP-ROI-Fix.

## 0.8.25

- Personen-Zähllinie auf `/people` wieder einstellbar gemacht.
- Linie kann gespeichert und direkt auf RTSP angewendet werden.
- Test & Upload, Personenanalyse und RTSP verwenden dieselben Linienwerte.

## 0.8.24

- Webinterface stabilisiert.
- Personen-Crop-Speicherung korrigiert.
- Doppelte Kennzeichen und Unique-Filter verbessert.
- Historie, Suche und Test & Upload korrigiert.

## 0.8.23

- CPU-/RTSP-Fix für Straßenbereich/Polygon.
- YOLO läuft bei RTSP wieder im gepufferten Analysebereich statt permanent auf dem kompletten Frame.

## 0.8.22

- Diagnose & Audit in die Einstellungen verschoben.
- Personenanalyse Demo-Daten entfernt.

## 0.8.21

- fast-plate-ocr als Standard-OCR integriert.
- YOLO-Kennzeichendetektor und YOLO-Fahrzeuganalyse integriert.
- Fahrzeugtyp, Kennzeichen-Land und Fahrzeugfarbe ergänzt.
