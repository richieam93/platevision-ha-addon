# Herkunft der mitgelieferten Machine-Learning-Modelle

Diese Datei dokumentiert die im PlateVision-Repository mitgelieferten
Modellgewichte. Die SHA-256-Werte erlauben eine eindeutige Prüfung der Dateien.

| Datei | Verwendung | Quelle | Ausgewiesene Lizenz der Quelle | SHA-256 |
|---|---|---|---|---|
| `yolov8n.pt` | Fahrzeug- und COCO-Objekterkennung | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | AGPL-3.0 oder Enterprise | `31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95` |
| `license_plate_detector.pt` | Kennzeichen-Lokalisierung | [Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8) | MIT im Ursprungsrepository; zusätzlich Ultralytics beachten | `8ec3b254a6c87610f037a90957462cafa11a9c03224e33a28c6a1d1ac2ac51b0` |
| `best.pt` | Personenerkennung, bester Trainingscheckpoint | [J3lly-Been/YOLOv8-HumanDetection](https://github.com/J3lly-Been/YOLOv8-HumanDetection) | GPL-3.0 im Ursprungsrepository; zusätzlich Ultralytics beachten | `a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43` |
| `last.pt` | Personenerkennung, letzter Trainingscheckpoint | [J3lly-Been/YOLOv8-HumanDetection](https://github.com/J3lly-Been/YOLOv8-HumanDetection) | GPL-3.0 im Ursprungsrepository; zusätzlich Ultralytics beachten | `db0f2dfe996d997cd33388003069a4685215c84dbca0f7a6de4b66c9d915f85f` |

## Prüfen der Dateien

```bash
sha256sum platevision/src/models/*.pt
```

Die Weboberfläche zeigt auf der Seite **Lizenz & Modelle** zusätzlich den
aktuell gefundenen Hash und ob er mit dem dokumentierten Wert übereinstimmt.

## Wichtiger Hinweis

Die Lizenz eines Repositorys, die Lizenz des Trainingsframeworks, die Lizenz
eines Basismodells und die Bedingungen des Trainingsdatensatzes können
nebeneinander gelten. Diese Dokumentation ist eine nachvollziehbare
Bestandsaufnahme, aber keine verbindliche Rechtsberatung.

## Laufzeit-OCR-Modell von fast-plate-ocr

PlateVision verwendet standardmässig `cct-s-v2-global-model` über das Paket
`fast-plate-ocr`. Dieses OCR-Modell ist nicht als `.pt`-Datei im Repository
enthalten. Je nach Paketcache kann es beim ersten Einsatz heruntergeladen werden.
Das Paket und seine veröffentlichten Modellkonfigurationen stammen aus
https://github.com/ankandrew/fast-plate-ocr und stehen dort unter MIT.
Der zugehörige Lizenztext wird unter
`third_party_licenses/fast-plate-ocr-MIT.txt` mitgeliefert.

