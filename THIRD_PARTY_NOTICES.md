# Drittanbieter-Hinweise

PlateVision enthält beziehungsweise verwendet Software und Modellgewichte von
Drittanbietern. Die Projektlizenz ersetzt deren eigene Lizenzbedingungen nicht.

## Ultralytics

- Komponente: Python-Paket `ultralytics` und offizielles YOLOv8n-Modell
- Projekt: https://github.com/ultralytics/ultralytics
- Lizenz: GNU Affero General Public License v3.0 oder eine separat erworbene
  Ultralytics Enterprise License
- Mitgelieferter Lizenztext: `third_party_licenses/Ultralytics-AGPL-3.0.txt`

## Kennzeichenerkennungsmodell

- Datei: `license_plate_detector.pt`
- Quelle: https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8
- Urheberhinweis des Ursprungsrepositorys: Copyright (c) 2023 Muhammad Zeerak Khan
- Repository-Lizenz: MIT
- Mitgelieferter Lizenztext: `third_party_licenses/Muhammad-Zeerak-Khan-MIT.txt`
- Hinweis: Das Modell basiert auf Ultralytics YOLOv8; anwendbare
  Ultralytics-Lizenzpflichten bleiben zusätzlich zu beachten.

## Personenerkennungsmodelle

- Dateien: `best.pt`, `last.pt`
- Quelle: https://github.com/J3lly-Been/YOLOv8-HumanDetection
- Repository-Lizenz: GNU General Public License v3.0
- Mitgelieferter Lizenztext: `third_party_licenses/J3lly-Been-GPL-3.0.txt`
- Hinweis: Die Modelle basieren auf Ultralytics YOLOv8; anwendbare
  Ultralytics-Lizenzpflichten bleiben zusätzlich zu beachten.

## Weitere Python-Abhängigkeiten

Die übrigen Bibliotheken sind in `platevision/src/requirements.txt` aufgeführt
und unterliegen ihren jeweiligen Lizenzen. Beim Erstellen des Docker-Images
werden diese Abhängigkeiten aus ihren Paketquellen installiert.

PlateVision ist nicht mit Ultralytics oder den genannten Modellprojekten
verbunden und wird von diesen nicht offiziell unterstützt.
