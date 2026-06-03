# PlateVision Fix Report - Fahrzeugbilder und OCR nach Update

## Ursache

Das letzte Update hat die RTSP-Analysezone vor der Modell-Erkennung zugeschnitten bzw. maskiert. Dadurch sieht YOLO oft nur einen Teil des Fahrzeuges. Die Kennzeichen-Erkennung kann trotzdem noch funktionieren, aber das Fahrzeug wird nicht mehr sauber erkannt und deshalb wird kein `vehicle_image` gespeichert.

Zusätzlich wurden OCR-Fragmente bisher fast nur nach Konfidenz ausgewählt. EasyOCR liefert Kennzeichen häufig in Teilen, z. B. `ZH` und `12345`. Wenn `ZH` die höchste Einzel-Konfidenz hatte, wurde der vollständige Kandidat verworfen oder schlechter bewertet.

## Änderungen

- RTSP-ROI wird standardmäßig nach der Erkennung gefiltert, nicht mehr vor der Fahrzeugerkennung zugeschnitten.
- Optionaler alter Crop/Mask-Modus bleibt über `detection.crop_before_detection` bzw. `detection.mask_before_detection` verfügbar.
- Separater Fahrzeug-Schwellwert (`vehicle_confidence_threshold`, Standard 0.25) und Kennzeichen-Schwellwert (`plate_confidence_threshold`, Standard 0.25).
- OCR-Kandidaten werden jetzt nach Plausibilität, Format, Länge und Konfidenz bewertet, damit zusammengesetzte Kennzeichen-Fragmente bevorzugt werden.
- Falls kein Fahrzeug-Box-Match existiert, wird trotzdem ein größeres Kontextbild rund um das Kennzeichen als Fahrzeugbild gespeichert.
- Die UI-Schalter `Fahrzeug-Bild speichern` und `Kennzeichen-Bild speichern` werden mit den Detector-Schaltern synchronisiert.
- Full-frame-Base64 wird wieder an Historie/API weitergereicht, wenn aktiviert.
- Requirements haben Major-Version-Obergrenzen bekommen, damit ein Add-on-Rebuild nicht still eine inkompatible neue OCR/YOLO/OpenCV-Generation zieht.

## Geänderte Dateien

- `platevision/src/app.py`
- `platevision/src/rtsp_handler.py`
- `platevision/src/requirements.txt`

## Validierung

- Python Syntax/Compile Check erfolgreich:
  - `python3 -m py_compile platevision/src/app.py platevision/src/rtsp_handler.py platevision/src/detector.py`
  - `python3 -m compileall -q platevision/src`

## Empfehlung nach Installation

1. Add-on neu bauen/starten.
2. In den Einstellungen prüfen:
   - Fahrzeug-Erkennung aktiv
   - Fahrzeug-Bild speichern aktiv
   - Kennzeichen-Bild speichern aktiv
3. Falls die Kamera weit entfernt ist: `vehicle_confidence_threshold` bei 0.20-0.30 testen.
4. ROI/Analysebereich nicht zu eng um das Kennzeichen legen; die Fahrzeugbox darf außerhalb der ROI liegen, solange Fahrzeugmitte/Kennzeichen später im ROI gefiltert wird.
