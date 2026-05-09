# PlateVision v0.7.2 ProTraffic Plus - Erweiterte Original-Einstellungen

## Grundsatz
Diese Version ist additiv aufgebaut: Die ursprünglichen Einstellungsbereiche bleiben erhalten und wurden nur erweitert. Bestehende Felder, API-Routen und Grundfunktionen werden nicht entfernt.

## Erweiterte Bereiche

### Erkennung
- Fahrzeugklassenfilter: PKW, LKW, Bus, Motorrad
- Maximale Erkennungen pro Frame
- Kennzeichen-Seitenverhältnis min/max
- Kennzeichenmodell-Confidence-Faktor
- Mindest-Fahrzeuggröße
- Option für zusätzliche Vollbildprüfung
- Annotationen und Confidence-Anzeige ein-/ausschaltbar

### OCR / Texterkennung
- OCR Engine Feld
- Decoder-Auswahl
- Paragraph-Modus
- Rotation-Varianten
- Early-Stop-Konfidenz
- Limit für OCR-Varianten
- Mindest-Textlänge
- Fragment-Zusammenführung
- Großschreibung und EasyOCR-Allowlist

### Bildvorverarbeitung
- Gamma-Korrektur
- Gamma-Wert
- CLAHE Clip Limit und Rastergröße
- Denoise-Stärke
- Threshold Block Size und C-Wert
- Invertierte Variante
- Bilateralfilter
- Perspektiv-Option vorbereitet
- Rand-Padding um Kennzeichen

### Historie
- Fuzzy-Duplikat-Erkennung sichtbar konfigurierbar
- Ähnlichkeitsschwelle
- OCR-Rohtext speichern
- Kandidaten speichern
- Positionsdaten speichern
- Gruppierung nach Besuch
- Auto-Cleanup
- Standard-Exportformat
- Markierung für niedrige Konfidenz

### Speicher
- JPEG-Qualität für Kennzeichen, Fahrzeuge und Frames
- Tagesordner
- Metadaten-JSON
- Bild-Cleanup
- Speicherlimit in MB
- Export-Kompression
- Thumbnails
- Dateiname-Muster
- Unbekannte Kennzeichen separat ablegen

### Allgemein
- Sprache DE/EN/FR/IT
- Theme dark/light/auto
- Zeitzone und Datumsformat
- Startseite
- 24h-Zeit
- Auto-Refresh global
- Log-Level
- Barrierearm-Modus
- Kompakte Zahlen
- Standortname und Bediener

### Modelle
- Modellpfade editierbar
- Gerät: auto/cpu/cuda/mps
- Modellgröße-Hinweis
- Custom Modellordner
- Label-Beschreibungen
- Auto-Reload
- Warmup
- FP16
- CPU-Fallback
- Speichern & neu laden

### Über
- Version auf 0.7.2 aktualisiert
- Release-Kanal
- Support-URL
- Dokumentations-URL
- Lizenzhinweis
- System-About-API

## Neue APIs
- `POST /api/config/storage`
- `POST /api/config/models`
- `POST /api/config/about`
- `GET /api/system/about`

## Prüfungen
- Python-Syntax geprüft: `app.py`, `detector.py`, `rtsp_handler.py`
- Alle Jinja/HTML-Templates syntaktisch geparst
- API-Routen statisch geprüft
- CSS erweitert

## Einschränkung
Ein realer Kamera-/RTSP-/YOLO-/EasyOCR-Livetest kann nur in der Zielumgebung mit Kamera und Modellen erfolgen.
