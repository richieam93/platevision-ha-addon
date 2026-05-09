# PlateVision v0.7.1 ProTraffic

## Neue Hauptfunktionen

- Neue Seite `/statistics` für Verkehrsstatistik.
- Tageszählung: Durchfahrten/Besuche, eindeutige Fahrzeuge, Roh-Erkennungen und Wiederkehrer.
- Kennzeichen-Profil: wie oft dasselbe Auto erkannt wurde, erste/letzte Sichtung und Sessions.
- Kommen/Gehen-Heuristik:
  - nutzt vorhandene explizite Richtungsdaten, wenn vorhanden;
  - erkennt Bewegungsrichtung heuristisch über gespeicherte Positionen, wenn mehrere Punkte vorhanden sind;
  - markiert Fahrzeuge nach Timeout als vermutlich gegangen.
- Neue API-Endpunkte:
  - `GET /api/statistics/traffic`
  - `GET /api/statistics/plate/<plate>`
  - `GET /api/statistics/export?format=csv|json`
  - `POST /api/config/traffic`
- Neue Einstellungen im Bereich Statistik.
- Dashboard zeigt nun auch Durchfahrten und eindeutige Autos für heute.
- CSS und Templates wurden erweitert, neue Navigation und ProTraffic-Komponenten ergänzt.

## Hinweis

Eine absolut sichere Ein-/Ausfahrtserkennung benötigt explizite Richtungsdaten, zwei Kameras oder eine klare Bewegungsachse. Diese Version markiert unklare Abfahrten transparent als `gegangen*` beziehungsweise `departed_assumed`.
