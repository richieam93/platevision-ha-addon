# PlateVision v0.8.0 ProTraffic People

Diese Version erweitert das Add-on additiv um eine Personenanalyse. Bestehende Kennzeichen-, Fahrzeug-, Statistik-, RTSP-, Test- und Einstellungsmodule bleiben erhalten.

## Neue Funktionen

- Neue Seite `/people` für Personenzählung und Personen-Historie.
- Separate Personen-Historie unter `data/people/history.json`.
- Aktivierungsschalter für das gesamte Personenmodul.
- Modell-Auswahl: Standard YOLOv8 COCO Person-Klasse oder eigenes HumanDetection YOLOv8 `.pt` Modell.
- Virtuelle Zähllinie mit X/Y-Achse, Linienposition, Richtung und Zählstrategie.
- Einfacher Personen-Tracker mit Track-ID, Timeout und Maximaldistanz.
- API-Endpunkte für Statistik, Historie, Export, Konfiguration und Snapshot-Test.
- Live-Overlay mit Personenboxen und Zähllinie.

## Hinweis zur Genauigkeit

Personenzählung ist stark von Kameraposition, Bildrate und Zähllinie abhängig. Für zuverlässige Werte sollte die Zähllinie so platziert werden, dass jede Person sie genau einmal überquert. Ohne klare Linie ist die Zählung heuristisch.

## Modell-Hinweis

Das Add-on lädt keine externen Modelle automatisch herunter. Ein Modell wie `best.pt` aus einem HumanDetection-Repository kann im Add-on als `models/human_best.pt` abgelegt und in den Einstellungen ausgewählt werden. Bitte Lizenz und Herkunft des Modells selbst prüfen.
