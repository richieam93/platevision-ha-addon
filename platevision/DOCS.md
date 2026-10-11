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


## 0.8.30

Neue RTSP-Leistungsoptionen befinden sich unter Einstellungen -> Dashboard & Layout + Allgemein. Praktische YOLO/OCR/Personenwerte werden im Test & Upload Menü gespeichert und für RTSP verwendet.


## Lizenz und Modellherkunft

- Weboberfläche: `/legal`
- Maschinenlesbare Übersicht: `/api/system/licenses`
- Projektlizenz ab Version 0.9.0: `AGPL-3.0-only`
- Modellquellen und Prüfsummen: `MODEL_PROVENANCE.md`


## Benutzeroberfläche ab 0.12.9

Die Seitenleiste ist in fünf Bereiche gruppiert. Die Suche am Menükopf kann auch mit `Strg+K`/`Cmd+K` geöffnet werden. Bis zu sechs Register können per Stern im jeweiligen Menüeintrag oder am Seitenkopf favorisiert werden. Favoriten und eingeklappte Gruppen liegen nur im Browser (`localStorage`) und werden nicht als Add-on-Konfiguration gespeichert. Details in `NAVIGATION_0.12.9_DE.md` im Repository.


## Home Assistant Ingress ab 0.13.0

Die PlateVision-Oberfläche erscheint jetzt unter **Home Assistant → PlateVision**
in der HA-Seitenleiste. Die Anmeldung erfolgt durch Home Assistant.
`ingress: true` und `ingress_port: 5000` sind voreingestellt.
Das Netzwerkport-Mapping auf 8087 ist standardmäßig **deaktiviert**.
Für eigenständige REST-Sensoren kann ein direkter Port manuell zugewiesen werden;
dieser umgeht die Home-Assistant-Anmeldung und darf nicht ungeschützt im Internet
freigegeben werden. Weitere Hinweise stehen im Repository in
`HOME_ASSISTANT_INGRESS_0.13.0_DE.md`.


**Update 0.13.5:** Neuer Alarm-Posteingang mit sicherem Watchlist-Opt-out, Sperrzeit und Postfach. Details: `WATCHLIST_ALARMZENTRALE_0.13.5_DE.md`.

**Update 0.13.4:** Speicherschonendere Personen-Sitzungen und Bildkalender, Schutz referenzierter Personenbilder bei Retention sowie verlustarme Video-Job-Aufbewahrung. Details: `LANGZEITSTABILITAET_0.13.4_DE.md`.

**Update 0.13.6:** Optional Home-Assistant-Mitteilungen bei neuen Watchlist-Treffern: dauerhaftes SQLite-Zustellpostfach, sichere Geräteauswahl, Smartphone-Push über Companion-App, Datenschutzschalter und Verbindungsprüfung. Standardmäßig ausgeschaltet. Details: `HOME_ASSISTANT_MELDUNGEN_0.13.6_DE.md`.



**Update 0.13.7:** Sichere Home-Assistant-Zustellung mit Lease-Fencing, zusätzlicher Datenschutzkontrolle, geschützten Konfigurationsimporten und einem privaten Zustellprotokoll in der Alarmzentrale. Details: `HA_ZUSTELLUNG_0.13.7_DE.md`.


**Update 0.13.9:** Neue Opt-out-Absicherung bei parallelen Erkennungen sowie Importen, pausierter HA-Versand bei deaktivierter Watchlist, getrennte Anzeige für fällige und verzögerte Retry-Versuche. Details: `DATENSCHUTZ_UND_ZUSTELLUNG_0.13.9_DE.md`.

**Update 0.13.8:** Synchronisierter HA-Empfängerwechsel, Abschalten ausstehender Benachrichtigungen und neuer Zustellmonitor in Diagnose. Details: `HA_ZUSTELLMONITOR_0.13.8_DE.md`.

