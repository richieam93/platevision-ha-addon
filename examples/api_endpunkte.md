# PlateVision API-Endpunkte 0.8.28

Basis-URL im Beispiel: `http://192.168.1.240:8087`

## Dashboard und Tagesdaten

| Methode | Endpoint | Beschreibung |
|---|---|---|
| GET | `/api/dashboard/overview` | Kompakte Übersicht: letzter Wagen, letzte Person, heutige Zahlen, Streamstatus |
| GET | `/api/history/statistics` | Historie-Statistik |
| GET | `/api/history/timeline?bucket=hour` | Verlauf nach Stunde oder Tag |
| GET | `/api/statistics/traffic?days=1` | Durchfahrten, Wiederkehrer, Kommen/Gehen |

## RTSP Stream

| Methode | Endpoint | Beschreibung |
|---|---|---|
| POST | `/api/stream/start` | RTSP-Stream starten |
| POST | `/api/stream/stop` | RTSP-Stream stoppen |
| GET | `/api/stream/status` | Stream-Status abrufen |
| GET | `/api/stream/resolution` | Stream-Auflösung abrufen |
| GET | `/api/stream/feed` | MJPEG-Livebild |
| GET | `/api/stream/feed?raw=1` | MJPEG-Rohbild ohne Overlays, falls verfügbar |
| GET | `/api/stream/snapshot` | Rohes Einzelbild für Vorschau/Kalibrierung |
| GET | `/api/stream/geometry` | Geometrie für Straßenbereich/ROI |

## Letzte Erkennung

| Methode | Endpoint | Beschreibung |
|---|---|---|
| GET | `/api/latest` | Letzter kompletter Kennzeichen-Historieeintrag |
| GET | `/api/latest/plate` | Letztes Kennzeichen als JSON |
| GET | `/api/latest/plate/image` | Letzter Kennzeichen-Crop als JPEG |
| GET | `/api/latest/vehicle` | Letztes Fahrzeug als JSON |
| GET | `/api/latest/vehicle/image` | Letzter Fahrzeug-Crop als JPEG |
| GET | `/api/latest/person` | Letzte gespeicherte Person als JSON |
| GET | `/api/latest/person/image` | Letzter Personen-Crop als JPEG |
| GET | `/api/latest/full` | Letzte Fahrzeug-/Kennzeichenerkennung inkl. Base64-Bilder |
| GET | `/api/latest/image` | Bestes letztes Bild, Fahrzeug vor Kennzeichen |

## Test & Upload

| Methode | Endpoint | Beschreibung |
|---|---|---|
| POST | `/api/process/image` | Bild hochladen und mit den gespeicherten Test-/RTSP-Werten analysieren |
| POST | `/api/process/video` | Videoverarbeitung starten, falls aktiviert |
| GET | `/api/process/jobs` | Verarbeitungsjobs anzeigen |

## Historie und Suche

| Methode | Endpoint | Beschreibung |
|---|---|---|
| GET | `/api/history?unique=true` | Historie, eindeutige Kennzeichen |
| GET | `/api/history/search?q=ZH12345&unique=true` | Suche mit Unique-Filter |
| GET | `/api/history/export?format=json&unique=true` | Export |
| GET | `/api/history/<id>/image/vehicle` | Fahrzeugbild eines Historie-Eintrags |
| GET | `/api/history/<id>/image/plate` | Kennzeichenbild eines Historie-Eintrags |

## Personenanalyse

| Methode | Endpoint | Beschreibung |
|---|---|---|
| GET | `/api/people/statistics?days=1` | Personenstatistik |
| GET | `/api/people/history` | Personenereignisse |
| GET | `/api/people/images/history?limit=50&counted_only=false` | gespeicherte Personen-Crops |
| GET | `/api/people/presence` | aktuelle Personen-/Track-Übersicht |
| GET | `/api/people/test/snapshot` | Test-Snapshot |
| POST | `/api/config/people` | Personen-Einstellungen speichern |

## Konfiguration und Modelle

| Methode | Endpoint | Beschreibung |
|---|---|---|
| GET/POST | `/api/config` | Gesamtkonfiguration lesen/speichern |
| GET/POST | `/api/config/rtsp` | RTSP und Autostart |
| GET/POST | `/api/config/detection` | Kennzeichen-/YOLO-Erkennung |
| GET/POST | `/api/config/ocr` | fast-plate-ocr oder EasyOCR |
| GET/POST | `/api/config/history` | Historie/Speicher |
| GET | `/api/models/status` | Modellstatus |
| GET | `/api/models/available` | auswählbare Modelle |
| POST | `/api/models/upload` | eigenes Modell hochladen |
| POST | `/api/models/reload` | Modelle neu laden |
| GET | `/api/system/health` | Healthcheck |
| GET | `/api/system/about` | Version/Info |
| GET | `/api/system/audit` | Diagnose/Audit |
