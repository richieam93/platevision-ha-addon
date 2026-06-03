# Changelog
## 0.8.16 - 2026-06-03

- Fahrzeug-Farberkennung robuster gemacht: Karosserie-Zonen statt nur Mittelpunkt.
- Fenster, Kennzeichen/Scheinwerfer, Reifen/Strasse und helle Reflexionen werden optional ignoriert.
- KMeans-Farbcluster und HSV/Lab-Bewertung kombiniert.
- Test & Upload um Fahrzeugfarbe-/Karosserie-Farbe-Labor erweitert.
- Farb-Konfidenz und Methode werden an Detektion/RTSP-Historie weitergegeben.


## 0.8.15 - 2026-06-03

### Added
- PaddleOCR als dritte OCR-Engine ergänzt.
- Auto-Modus erweitert auf PaddleOCR + EasyOCR + Tesseract mit Kandidaten-Bewertung.
- Test-&-Upload-Labor um PaddleOCR-Sprache, Mindest-Konfidenz, Variantenlimit und Ausrichtungsprüfung erweitert.
- Einstellungen-Seite um PaddleOCR-Optionen und Statusanzeige erweitert.

### Changed
- `auto_best` nutzt jetzt alle verfügbaren OCR-Engines und überspringt fehlende Engines automatisch.
- OCR-Reader werden nach Test-/Speicheränderungen sauber neu geladen.

## 0.8.14 - 2026-06-03

### Added
- Echter Auto-Länder-Modus: bei Land/Format „Auto“ werden DE, CH, FL, AT, CZ, EU und NL automatisch getestet und gegeneinander bewertet.
- Neuer Lesemodus im Test-&-Upload-Labor: „Rohtext: alle Buchstaben/Zahlen lesen“. Damit wird weniger landesspezifisch korrigiert und der erkannte Text des Kennzeichens direkter angezeigt.
- Rohtext-Fallback: wenn Länderlogik unsicher ist, wird ein lesbarer alphanumerischer OCR-Treffer trotzdem als Kandidat angezeigt statt komplett verworfen.
- Optionaler OCR-Fallback ohne Kennzeichenbox: wenn YOLO keine Kennzeichenbox findet, kann OCR auf Fahrzeug-/Bildbereichen trotzdem einen Textversuch machen.

### Changed
- Auto-Modus bevorzugt vollständige Kennzeichen stärker und verwirft kurze Fragmente wie OB36 aggressiver, wenn längere Kandidaten vorhanden sind.
- Standard für neue Installationen ist jetzt OCR Engine „auto_best“ und Land/Format „auto“.

## 0.8.12 - 2026-06-03

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
