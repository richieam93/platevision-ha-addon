# PlateVision 0.13.7 – Sicherere Home-Assistant-Zustellung

## Überblick

Diese Version verbessert die in 0.13.6 eingeführte Home-Assistant-Benachrichtigungsfunktion. Die 14 Register, die lokale Watchlist-Alarmzentrale, die Bild-/Personen-/Kennzeichenerkennung, die bisherigen API-Pfade und die Home-Assistant-Ingress-Einbettung bleiben erhalten.

## Korrekturen

- **Abgelaufene Zustellaufträge:** Jede Übernahme eines SQLite-Auftrags erhält jetzt eine eigene, monoton steigende Lease-Generation. Ein alter Hintergrundprozess kann den Status einer neueren Übernahme nicht mehr fälschlich als erfolgreich oder fehlgeschlagen markieren.
- **Bestehende Datenbanken:** Die Watchlist-Alarmdatenbank bekommt beim Öffnen automatisch die zusätzliche Spalte `lease_generation`. Das ist ein additives Upgrade, kein Neuaufbau und keine Migration der Kennzeichen- oder Personenhistorien.
- **Datenschutz bei wartenden Meldungen:** Wird die Übermittlung von Kennzeichen und Watchlist-Notizen ausgeschaltet, versenden bereits wartende Aufträge diese Daten nicht mehr. Eine spätere Zustimmung ergänzt ältere, ursprünglich anonyme Meldungen nicht rückwirkend um Kennzeichen.
- **Einstellungswechsel während der Zustellung:** Konfigurationsänderungen und die laufende HA-Sendeoperation werden innerhalb des Add-on-Prozesses serialisiert. Nach einer erfolgreich gespeicherten Änderung werden ausstehende Aufträge für das vorherige Ziel entfernt.
- **Andere Konfigurationswege:** Die allgemeinen `/api/config`- und `/api/config/advanced`-Schnittstellen akzeptieren keine Änderungen der `alerts.ha_*`-Schalter mehr. Diese werden nur im Bereich **Alarmzentrale → Home Assistant & Smartphone-Benachrichtigungen** geändert. Ein JSON-Konfigurationsimport übernimmt bestehende HA-Benachrichtigungseinstellungen der laufenden Installation statt externe Ziele und die Zustimmung aus der Datei zu aktivieren.
- **Standardwerte zurücksetzen:** Ein vollständiger Konfigurationsreset bereinigt ausstehende Zustellungen, wenn dadurch die HA-Benachrichtigungseinstellungen geändert werden.

## Neue Funktion: Zustellprotokoll

Unter **Alarmzentrale → Home Assistant & Smartphone-Benachrichtigungen → Zustellprotokoll** zeigt PlateVision die letzten 20 Zustellaufträge mit Kanal, Status, Anzahl der Versuche, Zeitstempel und begrenzter Fehlermeldung. Die API `/api/alerts/homeassistant/deliveries?limit=20` liefert maximal 50 Datensätze und **keine** Kennzeichen, Notizen, Telefonnummern, Smartphone-Dienstnamen oder Tokens. Die Historie ist von diesem Protokoll unabhängig.

Ein als „angenommen“ markierter Auftrag wurde von Home Assistant angenommen. Das bestätigt **nicht**, dass eine Push-Nachricht auf dem Smartphone angekommen oder dort angezeigt worden ist. Nach einem Netzwerkabbruch direkt nach dem Versand kann eine Zustellung erneut versucht werden. Eine garantierte Exactly-Once-Zustellung ist technisch nicht gegeben.

## Aktualisierung

1. Ein vollständiges Home-Assistant-Backup erstellen.
2. Den lokalen Add-on-Quellcode auf 0.13.7 aktualisieren und neu bauen; das bestehende `/data`-Verzeichnis nicht überschreiben.
3. Das Add-on in Home Assistant starten und die neue Version prüfen.
4. In der Alarmzentrale den Zustand der HA-Dienste prüfen und eine **neutrale** Testmeldung verschicken. Erst danach bei Bedarf echte Watchlist-Push-Meldungen aktivieren.
5. Bei Meldungsfehlern das neue, aufklappbare Zustellprotokoll ansehen. Alte in einer Konfigurationsdatei enthaltene HA-Ziele werden beim Import bewusst nicht aktiviert.

## Getestet und offen

Automatisch geprüft wurden die SQLite-Schema-Erweiterung, Stale-Lease-Fencing, parallele Zustellungen, die aktuellen Datenschutz-Schalter, generische Konfigurationsänderungen, der Konfigurationsimport und das private Zustellprotokoll. Chromium wurde für 320, 390 und 1366 Pixel mit simulierten Home-Assistant-Antworten geprüft. Die vorhandenen Backend-/Frontend-/Ingress-Regressionen wurden wiederholt.

**Nicht auf echter Hardware geprüft:** echter Supervisor-Zugriff mit `SUPERVISOR_TOKEN`, Companion-App-Push, Live-RTSP-Verbindung, YOLO-/OCR-Ausführung und Docker-Build auf der Zielplattform.
