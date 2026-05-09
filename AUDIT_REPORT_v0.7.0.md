# PlateVision v0.7.0 ProTraffic - Änderungs- und Prüfbericht

## Geprüft und erweitert

- Backend um Verkehrsstatistik, Sessions und Kennzeichenprofile erweitert.
- Neue Statistik-API und Export-API eingebaut.
- RTSP-Historieneinträge speichern nun Positionsdaten, soweit verfügbar.
- Dashboard, Sidebar, Einstellungen, Historie, Startseite und Fehlerseiten erweitert.
- Alle Templates erhalten durch globale CSS-Erweiterungen ein konsistenteres Layout; wichtige Seiten wurden zusätzlich um Hinweise/Links ergänzt.
- Neue `statistics.html` Seite erstellt.

## Grenzen

- Ohne echte Richtungsdaten kann „gegangen“ nicht absolut sicher erkannt werden. Das Add-on kennzeichnet solche Fälle als Timeout-Heuristik.
- Live-RTSP, YOLO und EasyOCR wurden nicht mit echter Kamera getestet; geprüft wurden Syntax, Routing und Template-Parsing.
