# PlateVision Add-on Dokumentation

PlateVision erkennt Kennzeichen, Fahrzeuge und Personen über RTSP oder Foto-Upload.

Wichtige Bereiche:

- **Dashboard**: zeigt letztes Auto, letzte Person und Tageszahlen ohne Modell-Details.
- **Test & Upload**: Modelle, OCR, YOLO-Parameter und Personen-Zähllinie testen und speichern.
- **RTSP Stream**: Kamera/Stream, Autostart und Straßenbereich/Polygon einstellen.
- **Einstellungen**: Historie, Speicher, Modell-Upload, Suche, Statistik, Datenschutz, Alerts und Diagnose.

## RTSP Autostart

Der Live-/RTSP-Stream kann nach Docker- oder Home-Assistant-Neustart automatisch gestartet werden.
Die Option findest du unter **RTSP Stream** und zusätzlich in den allgemeinen Einstellungen.

## Einstellungen testen und anwenden

Nach Änderungen in **Test & Upload** bitte **Speichern & RTSP anwenden** verwenden. Danach gelten die getesteten Werte auch für den RTSP-Stream.

## Beispiele

Aktuelle Home-Assistant-Beispiele findest du im Ordner `examples/`.
