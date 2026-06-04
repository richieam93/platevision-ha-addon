# PlateVision 0.8.20

## Hauptfix

Die RTSP-Analysezone/ROI wird standardmäßig nicht mehr vor der Fahrzeug- und Kennzeichenerkennung zugeschnitten. Die Modelle bekommen wieder das komplette Kamerabild und die ROI wird erst nach der Erkennung als Filter angewendet.

## Behobene Probleme

- Fahrzeuge verschwanden nach dem letzten Update teilweise, weil nur der ausgeschnittene Analysebereich an YOLO übergeben wurde.
- Dadurch wurde oft nur noch das Kennzeichenbild gespeichert, aber kein Fahrzeugbild.
- Fahrzeuge oder Kennzeichen am Rand der Analysezone wurden zu streng verworfen, weil nur der Mittelpunkt geprüft wurde.
- OCR bekam teilweise zu eng geschnittene Kennzeichenbilder ohne Rand.

## Änderungen

- `rtsp.analysis_area.crop_before_detection` neu, Standard `false`.
- `rtsp.analysis_area.mask_before_detection` neu, Standard `false`.
- ROI-Filterung für Fahrzeuge/Kennzeichen toleranter: Mittelpunkt, Fußpunkt oder Mindestüberlappung reicht.
- Kennzeichen-Crop erhält kleinen Padding-Rand vor OCR.
- Fahrzeugbild-Fallback: Wenn kein Fahrzeugmodell-Treffer zum Kennzeichen vorhanden ist, wird ein größerer Kontextausschnitt aus dem Vollbild gespeichert.
- Version auf `0.8.20` erhöht.

## Hinweis

Für die normale Nutzung sollte `crop_before_detection` deaktiviert bleiben. Nur so sieht das Modell das komplette Fahrzeug.
