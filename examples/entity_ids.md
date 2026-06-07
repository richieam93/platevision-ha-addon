# Home Assistant Entity-IDs

Diese Entity-IDs entstehen, wenn du `examples/configuration.yaml` übernimmst. Home Assistant kann Namen je nach Sprache/Installation leicht anders normalisieren.

## Kennzeichen und Fahrzeug

| Entity | Typ | Inhalt |
|---|---|---|
| `sensor.platevision_kennzeichen` | Sensor | letztes Kennzeichen |
| `sensor.platevision_fahrzeugtyp` | Sensor | Auto / PKW, LKW, Bus, Motorrad, Fahrrad |
| `sensor.platevision_fahrzeugfarbe` | Sensor | erkannte Fahrzeugfarbe |
| `sensor.platevision_land` | Sensor | Kennzeichen-Land / Region |
| `sensor.platevision_konfidenz` | Sensor | OCR-/Kennzeichen-Konfidenz in Prozent |
| `sensor.platevision_zuletzt_gesehen` | Sensor | relative Zeit seit letzter Erkennung |
| `sensor.platevision_quelle` | Sensor | `rtsp`, `image_upload`, `video_upload` |

## Personen und Tageszahlen

| Entity | Typ | Inhalt |
|---|---|---|
| `sensor.platevision_personen_heute` | Sensor | gezählte Personen heute |
| `sensor.platevision_kennzeichen_heute` | Sensor | Kennzeichenereignisse heute |
| `sensor.platevision_autos_heute` | Sensor | eindeutige Fahrzeuge heute |
| `sensor.platevision_durchfahrten_heute` | Sensor | Durchfahrten/Besuche heute |
| `sensor.platevision_wiederkehrer_heute` | Sensor | Wiederkehrer heute |
| `sensor.platevision_letzte_person` | Sensor | letzte Person mit Richtung/Confidence/Bild-URL |

## Stream und Kameras

| Entity | Typ | Inhalt |
|---|---|---|
| `binary_sensor.platevision_stream` | Binary Sensor | Stream läuft/verbunden |
| `camera.platevision_live_stream` | Kamera | MJPEG Live Stream |
| `camera.platevision_fahrzeug_bild` | Kamera | letzter Fahrzeug-Crop |
| `camera.platevision_kennzeichen_bild` | Kamera | letzter Kennzeichen-Crop |
| `camera.platevision_person_bild` | Kamera | letzter Personen-Crop |
| `camera.platevision_snapshot` | Kamera | aktueller Stream-Snapshot |
