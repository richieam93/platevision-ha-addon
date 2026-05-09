# PlateVision v0.6.0 Pro+ - Audit Report

## Durchgeführte Prüfungen
- Python-Syntax geprüft mit `python3 -m py_compile` für `app.py`, `detector.py`, `rtsp_handler.py`.
- Jinja-Templates statisch geparst: keine Template-Syntaxfehler gefunden.
- `url_for(...)`-Referenzen in Templates gegen Flask-Endpunkte geprüft: keine fehlenden Endpunkte gefunden.
- `requirements.txt` Encoding geprüft und von UTF-16 auf UTF-8 korrigiert.
- Home-Assistant `config.yaml` geprüft und Web-UI-Port auf `5000` korrigiert.

## Gefundene und behobene Hauptprobleme
- `requirements.txt` war UTF-16 Little Endian. Das kann `pip install -r requirements.txt` im Docker-Build brechen.
- Versionen und Metadaten waren uneinheitlich (`0.5.0` vs. interne 2.1-Texte). Aktualisiert auf `0.6.0`.
- Root-Seite `/` zeigt nun direkt das überarbeitete Dashboard.
- Basislayout um Diagnose, Sprachumschalter und UI-Konfigurationslogik erweitert.

## Hinweise
Die Prüfung war eine statische Code-/Template-Prüfung. Ein echter Laufzeittest mit YOLO/EasyOCR-Modellen und RTSP-Kamera wurde nicht ausgeführt, weil dafür die Zielumgebung, Modelle und Kamera/Stream benötigt werden.
