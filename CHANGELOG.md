# Changelog

## 0.8.11 - 2026-06-03

### Added
- Test-&-Upload-Erkennungslabor mit direkt einstellbaren OCR-, YOLO- und Preprocessing-Parametern.
- OCR-Umschaltung zwischen EasyOCR, Tesseract und Auto-Modus.
- Zusätzliche OCR-Optimierungen: Resize, Denoise, Schärfen, CLAHE, Threshold, Morphology, Invertierung und Padding.
- Möglichkeit, Test-Einstellungen direkt zu speichern und danach für den RTSP-Stream zu verwenden.

### Changed
- Fahrzeugerkennung läuft wieder auf dem ganzen Frame; ROI wird nur noch zum Filtern verwendet.
- Fahrzeugbild wird zuverlässiger gespeichert, auch wenn keine vollständige Fahrzeugbox erkannt wird.
- OCR-Bewertung und Fragment-Merge für Kennzeichen verbessert.
- Separate Confidence-Werte für Fahrzeug-, Kennzeichen- und OCR-Erkennung.

### Fixed
- Problem behoben, bei dem nach dem Update nur noch das Kennzeichenbild, aber kein Fahrzeugbild angezeigt wurde.
- Problem behoben, bei dem Fahrzeuge durch zu frühes ROI-Cropping schlechter oder gar nicht erkannt wurden.
- Abhängigkeiten eingegrenzt, damit zukünftige Builds stabiler bleiben.

## 0.8.10

- Vorherige veröffentlichte Version.
