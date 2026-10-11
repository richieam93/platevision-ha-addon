# PlateVision 0.13.6 – Home-Assistant-Benachrichtigungen

## Was neu ist

In **PlateVision → Alarmzentrale → Home Assistant & Smartphone-Benachrichtigungen** lassen sich zwei Weiterleitungen getrennt aktivieren:

- **Dauerhafte Home-Assistant-Mitteilung:** Meldung im Home-Assistant-Benachrichtigungsbereich via `persistent_notification.create`.
- **Smartphone-Push:** Standarddienst `notify.mobile_app_<geraet>` eines bereits eingerichteten Smartphones in der Home-Assistant-Companion-App.

Beides ist **standardmäßig aus**. Das bestehende lokale Alarm-Postfach bleibt unabhängig davon verfügbar und wird nicht gelöscht oder verändert.

## Schritt für Schritt

1. Vor dem Update ein **vollständiges Home-Assistant-Backup** erstellen. Das ZIP ist Quellcode, kein fertiges Docker-Image. Keine Daten unter `/data` ersetzen.
2. Add-on auf Version **0.13.6** aktualisieren und neu bauen/starten. Die neue Konfiguration nutzt `homeassistant_api: true` für den internen Core-Zugriff. Die App bleibt über die Home-Assistant-Seitenleiste erreichbar.
3. Für Smartphone-Push die Home-Assistant-Companion-App auf dem Gerät einrichten und unter **Home Assistant → Entwicklerwerkzeuge → Aktionen** den vorhandenen Dienst `notify.mobile_app_...` nachsehen.
4. In PlateVision unter **Alarmzentrale → Home Assistant & Smartphone-Benachrichtigungen** gewünschte Kanäle aktivieren. Für Push den **genauen** Dienstnamen eingeben (mit oder ohne das Präfix `notify.`).
5. Mit **HA-Einstellungen speichern** bestätigen und erst danach **Neutrale Testmeldung senden** drücken. Der Test enthält **keine Kennzeichen**. Home Assistant kann eine Meldung annehmen, auch wenn ein Smartphone vorübergehend offline ist.
6. Neue Watchlist-Treffer lösen nun – nach erfolgreicher Historien- und Alarm-Speicherung – optional externe Meldungen aus. Bestehende Alarme werden beim Aktivieren **nicht nachträglich** versendet.

## Datenschutz & Sicherheit

- Übertragungen erfolgen **nur** an `http://supervisor/core/api/services/…` über die interne Home-Assistant-Schnittstelle. Host, URLs und Dienste sind nicht frei konfigurierbar. Das `SUPERVISOR_TOKEN` bleibt im Add-on und wird nicht im Browser angezeigt, protokolliert oder gespeichert.
- **Kennzeichen und Watchlist-Notiz mitsenden** ist ein **separater, standardmäßig ausgeschalteter** Schalter. Ohne ihn ist die Nachricht allgemein gehalten.
- Bei Smartphone-Push können Betriebssystem-Pushdienste beteiligt sein. Bei sensiblen Kennzeichen nur einschalten, wenn diese Übertragung erwünscht ist.
- Der optionale Direktport ist weiterhin standardmäßig deaktiviert. Öffne ihn nicht ohne zusätzliche Absicherung.
- Das ZIP-Datenbackup enthält auch die neue Outbox und ihre Zustellstatus. Eine Rücksicherung kann ausstehende Meldungen wieder aktivieren.

## Verhalten bei Fehlern

- **Kein Token:** PlatteVision zeigt „HA-Token fehlt“. Nur lokale Alarme werden weiter gespeichert. Die API-Berechtigung im Add-on prüfen und den Docker-Container neu bauen/starten.
- **Nicht erreichbares Home Assistant, HTTP 429 oder 5xx:** Begrenzte Wiederholungen mit zunehmendem Abstand, ohne die Kameraerkennung zu blockieren.
- **Falscher Smartphone-Dienst / HTTP 400, 401, 403 oder 404:** Meldung wird als **fehlgeschlagen** markiert. Dienst prüfen und nach Korrektur **Fehlgeschlagene erneut versuchen** auswählen.
- **Nach Neustart:** Ausstehende Meldungen werden weiter bearbeitet. Im ungünstigen Fall eines Absturzes genau nach der Bestätigung durch Home Assistant kann eine Meldung erneut zugestellt werden. Feste Benachrichtigungs-IDs und mobile Tags reduzieren diesen Effekt, garantieren aber nicht für jedes Gerät exakt eine Nachricht.
- **Zielwechsel oder Deaktivierung:** Die App entfernt offene Zustellaufträge für den früheren Empfänger. Ein bereits laufender HTTP-Aufruf kann im Ausnahmefall noch abgeschlossen werden.

## Was nicht getestet werden konnte

Die Tests nutzen einen simulierten HA-HTTP-Dienst sowie SQLite-/API-/Browserprüfungen. Die **tatsächliche Annahme durch Home Assistant Core und Zustellung auf dein Smartphone** müssen auf deiner Anlage geprüft werden. Die KI-, RTSP- und sonstigen Schnittstellen bleiben in dieser Version unverändert.
