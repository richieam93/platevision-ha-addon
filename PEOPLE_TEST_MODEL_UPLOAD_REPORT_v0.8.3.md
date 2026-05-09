# PlateVision v0.8.3 - Personen-Foto-Test & Modell-Upload

## Neu

- `/test` enthält einen separaten Personen-Foto-Test.
- Neue API `/api/people/test/image` für Bild-Upload, temporäres Aktivieren der Personenanalyse, Ergebnisbild und Personenanzahl.
- Neuer Einstellungs-Tab **Modelle hochladen**.
- Neue API `/api/models/upload` zum Hochladen eigener `.pt`, `.onnx` und `.engine` Modelle.
- Modelle werden bevorzugt nach `/data/models` gespeichert und können direkt als Personenmodell ausgewählt werden.
- Modellscan und Personenmodell-Dropdown werden nach Upload aktualisiert.

## Erhalten

- Bestehende Test-Funktion für Fahrzeuge/Kennzeichen bleibt erhalten.
- Bestehende Personenanalyse, Personen-Historie und Modellauswahl bleiben erhalten.
- Original-Einstellungsbereiche bleiben erhalten und werden nur erweitert.

## Hinweise

- Für zuverlässige Personenzählung bleiben Kamerawinkel, virtuelle Linie und Modellqualität entscheidend.
- Ein echter RTSP-/YOLO-Livetest muss in der Zielumgebung erfolgen.
