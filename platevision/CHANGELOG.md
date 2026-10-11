# Changelog

## 0.13.11 – Schwarzes Livebild in Home Assistant Ingress korrigiert (11.10.2026)

- **Ingress-Livestream:** Neue Einzelbildvorschau `/api/stream/preview`, die echte JPEGs über gewöhnliche HTTP-Antworten liefert. Der blockierende MJPEG-Feed bleibt für direkte Zugriffe bestehen; bei Fehlern wird dort automatisch auf JPEG umgestellt.
- **Verbindungsanzeige:** Live-Funktionen überschreiben keine gleichnamigen globalen Funktionen aus `base.html` mehr. Die vorher widersprüchlichen Statusanzeigen „Verbinde …“ und „Verbunden“ arbeiten unabhängig und korrekt.
- **FPS und Fehlerhinweise:** Anzeige real gemessener Kamera-FPS aus der API, lesbare Warte-/Fehlerhinweise statt rein schwarzer Fläche und Pausieren unsichtbarer Tab-Vorschauen.
- **Datenschutz:** Die Live-Seite zeigt nur noch die anonymisierte RTSP-Hostanzeige, keine Benutzerkennungen, Passwörter oder Stream-Pfade.
- **Prüfungen:** `tools/test_stream_preview_01311.py` und `tools/qa_browser_stream_preview_01311.py` ergänzen alle vorhandenen Regressionstests. Kein Datenbankumbau und keine Änderungen an den mitgelieferten KI-Modellen. Siehe `AENDERUNGSBERICHT_0.13.11.md`.

## 0.13.10 – Stabiler RTSP-Dauerbetrieb und weniger SQLite-Schreiblast (11.10.2026)

- **SQLite-Historie:** `sync()` vergleicht jetzt wirklich den gespeicherten JSON-Payload statt irrtümlich das letzte Suchindex-Feld. Unveränderte Datensätze werden bei Abgleichen nicht mehr alle überschrieben. Bei einer manuellen Neuordnung werden zuvor temporär verschobene SQL-Ränge dennoch vollständig wiederhergestellt.
- **RTSP-Erkennung:** Die lokale Duplikatsperre wird erst nach einer erfolgreich gespeicherten Erkennung aktualisiert. Vorübergehende Speicherfehler oder vom HistoryManager abgewiesene Einträge blockieren dadurch keine spätere gültige Erkennung. Die Sperrzeit ist gegen ungültige, negative, unendliche und übergroße Werte abgesichert und verwendet eine monotone Uhr.
- **RTSP-Fehlermeldungen:** Verbindungsausnahmen geben keine eingebetteten Benutzer, Passwörter, Stream-Pfade oder URL-Token mehr in den Capture-Log aus. Auch abweichend formatierte RTSP- und RTSPS-URLs in FFmpeg/OpenCV-Fehlern werden maskiert.
- **Kamera-Neuverbindung:** Der Bewegungsfilter verwirft seinen alten Vergleichsframe beim Trennen der Kamera. Ein neuer Stream startet somit mit einem frischen Bewegungs-Referenzbild.
- **Neue Diagnose:** Alter des zuletzt empfangenen RTSP-Bildes und Anzahl erfolgreicher Neuverbindungen in der Systemdiagnose. Ist die Kamera laut Verbindung aktiv, liefert aber seit mindestens 20 Sekunden kein Bild, erscheint eine eigene Warnung. Bewusst deaktivierte Kameras bleiben fehlerfrei.
- **Prüfungen:** Neues `tools/test_runtime_hardening_01310.py` mit echten SQLite-Triggern, Fehlersimulation und API-Gesundheitsprüfung; `tools/qa_browser_camera_01310.py` für 320/390/1360 Pixel, Ingress und JavaScript. Keine SQLite-Schemaänderung, keine Änderung an KI-Modellen, Datenpfaden oder Portfreigaben. Details: `AENDERUNGSBERICHT_0.13.10.md` und `LANGZEITBETRIEB_0.13.10_DE.md`.

## 0.13.9 – Datenschutz bei parallelen Erkennungen und präzisere Zustelldiagnose (11.10.2026)

- **Nebenläufige Erkennungen:** Das Einreihen eines Watchlist-Alarms hält nun dieselbe Sperre wie HA-Empfängerwechsel und globale Opt-outs. Nach dem Abschalten kann kein gerade verarbeiteter Frame unbemerkt eine neue Nachricht an den alten Empfänger einreihen.
- **Generische Konfiguration/Import:** `/api/config`, `/api/config/advanced` und `/api/config/import` prüfen Änderungen an globaler Alarmierung und Watchlist-Aktivierung unter der HA-Zustellsperre. Alte externe Aufträge werden vor der neuen Konfigurationsspeicherung entfernt; die ausdrücklichen HA-Empfängerfreigaben bleiben beim Import erhalten.
- **Fail-closed und Rollback:** Schlägt das Bereinigen der SQLite-Warteschlange fehl, wird die Einstellungsänderung verworfen (HTTP 503). Ungültige Abschnitte und Speicherfehler rollen den Arbeitsspeicher auf den letzten gültigen Stand zurück. Bei fehlgeschlagener JSON-Speicherung können zuvor verworfene externe Aufträge nicht wiederhergestellt werden; lokale Historien und Alarme bleiben bestehen.
- **Reset/Standardwerte:** Beide Wege bereinigen bei Änderungen der HA-Zustellberechtigung jetzt ebenfalls vor der Persistierung. SQLite-Fehler verursachen HTTP 503 und bewahren die vorherige Konfiguration.
- **Wiedereinschalten und Retry:** Globale Alarmierung sowie Watchlist-Abgleich werden beim HA-Senden erneut geprüft. Auch das Wiedereinschalten leert veraltete Aufträge; manuelle Wiederholungen sind bei deaktivierter Watchlist gesperrt.
- **Diagnose:** `due_now`, `oldest_due_seconds` und `next_retry_at` trennen tatsächlich überfällige Versuche von korrekt geplanten Wiederholungen. Diagnose und Alarmzentrale zeigen eine globale Versandpause an, ohne Kennzeichen oder Empfänger zu verraten.
- **Tests:** Neue SQLite-/Thread-/Import-/Fehlerfall-Regressionen in `tools/test_consent_hardening_0139.py`, bisherige Tests bleiben erhalten. Keine neue Datenbankmigration, keine Änderung der Erkennungsmodelle oder des Ingress-Ports. Siehe `DATENSCHUTZ_UND_ZUSTELLUNG_0.13.9_DE.md`.

## 0.13.8 – HA-Zustellmonitor und synchronisiertes Abschalten (10.10.2026)

- **Benachrichtigungskonfiguration:** HA-Empfängerwechsel, Konfigurationsreset und neutrale Testmeldung sind explizit mit dem laufenden Zustellprozess synchronisiert (zusätzlich zur bisherigen Anfrage-Sperre).
- **Watchlist-Opt-out:** Ein ausgeschalteter globaler Watchlist-Benachrichtigungsschalter entfernt auch noch wartende externe Zustellungen, ohne die lokalen Alarm- oder Erkennungsdaten zu löschen.
- **Neue Diagnose:** Aufklappbarer HA-Zustellmonitor mit Wartezeit, Fehlversuchen, abgelaufenen Leases und Token-Prüfung. Keine Kennzeichen, Smartphone-Ziele oder Passwörter in der Diagnose.
- **Konfigurationsspeicherung:** Zurücksetzen der Erkennungswerte prüft den Schreibvorgang und liefert bei vollem Datenträger HTTP 503 statt eines falschen Erfolgs.
- **API kompatibel:** `/api/system/health` und der interne Zustellstatus erhalten nur zusätzliche Felder. Keine Datenbankmigration und keine Änderung der KI-Modelle.
- **Regressionen:** Gezielte Parallelitäts- und Rücksetztests gegen echte SQLite-Outbox, mit blockiertem Netzwerksender sowie Fehler- und Datenschutzprüfung. Siehe `HA_ZUSTELLMONITOR_0.13.8_DE.md`.

## 0.13.7 – Datenschutz und Zustellabsicherung (10.10.2026)

- **Lease-Fencing:** SQLite-Zustellaufträge nutzen eine monotone Lease-Generation; ein abgelaufener Worker kann keinen neueren Auftrag mehr abschließen. Vorhandene Alarmdatenbanken werden ohne Datenverlust erweitert.
- **Datenschutz:** Bei wartenden Benachrichtigungen gilt auch die aktuelle Kennzeichenfreigabe. Änderungen des HA-Ziels und der Datenschutzschalter sind gegenüber laufender Zustellung serialisiert.
- **Einstellungen:** Generische Konfigurations-APIs dürfen keine HA-Ziele oder externe Benachrichtigungen aktivieren. Importe behalten die HA-Opt-in-Einstellungen der aktuellen Installation; Zurücksetzen räumt betroffene wartende Aufträge auf.
- **Neue Anzeige:** Aufklappbares Zustellprotokoll in der Alarmzentrale (max. 20 standardmäßig, API bis 50) ohne Kennzeichen, Watchlist-Notizen oder Gerätedienstnamen.
- **Tests:** Ergänzte SQLite-, Parallelitäts-, Datenschutz- und Konfigurations-Regressionen sowie Chromium-Prüfung für Ingress, XSS und Smartphone-Breiten.
- **Kompatibilität:** Keine Änderungen an KI-Modellen, Kennzeichen-/Personenhistorien oder bestehender Ingress-Konfiguration.

## 0.13.6 - Optionale Home-Assistant-Benachrichtigungen (10.10.2026)

- **Neu:** Dauerhafte Meldungen in Home Assistant über `persistent_notification.create` und optionaler Smartphone-Push über `notify.mobile_app_*`. Die Benachrichtigung wird über den internen Supervisor/Core-API-Proxy gesendet (`homeassistant_api: true`) und nutzt das Home-Assistant-Token ausschließlich innerhalb des Add-ons.
- **Datenschutz:** Beide externen Kanäle sind standardmäßig deaktiviert. Kennzeichen und Watchlist-Notiz werden nur nach separater ausdrücklicher Zustimmung übertragen. Zielgeräte werden gegen ein festes Muster geprüft; keine fremden URLs, Webhooks oder beliebigen Service-Aufrufe.
- **Zuverlässigkeit:** Zustellaufträge werden mit neuen lokalen Alarmen in einer SQLite-Transaktion gespeichert und unabhängig vom KI-Thread in einem Hintergrund-Worker verarbeitet. Bei vorübergehenden Verbindungsfehlern erfolgen begrenzte Wiederholungen mit Backoff. Fehlerhafte Dienste werden sichtbar als fehlgeschlagen markiert.
- **Sicherheit bei Neustart:** Ausstehende Meldungen und Fehlstatus überleben einen Neustart; parallele Prozesse beanspruchen dank SQLite-Leases nicht denselben Auftrag. Bei Zielwechsel oder Abschalten werden noch ausstehende Meldungen entfernt. Bestätigte Zustellungen besitzen feste IDs/Tags zur Reduzierung mehrfacher Nachrichten.
- **Bedienung:** Einklappbare Home-Assistant-Einstellungen direkt in der Alarmzentrale, expliziter neutraler Verbindungstest mit Rate-Limit, Statuszähler und Knopf für erneute Zustellversuche.
- **Kompatibilität:** Bestehende Historien, KI-Modelle, RTSP-Schnittstellen und bestehende lokale Alarmfunktionen unverändert. Nur eine additive Tabelle in der Watchlist-Alarmdatenbank; keine Migration der Kennzeichen- oder Personenhistorie.
- **Tests:** SQLite-Outbox-/Neustart-/Parallel-/HTTP-Fehler-/Datenschutztests, API-Rollbacktests, Chromium-Browserprüfung und bisherige Regressionstests. Echte HA-Supervisor-/Companion-App-Auslieferung bleibt ein Praxistest.

## 0.13.5 - Alarmzentrale und sichere Watchlist-Meldungen (10.10.2026)

- **Neu:** Eigenstaendige Alarmzentrale mit persistentem SQLite-Postfach, gelesen/ungelesen, Einzel- und Sammelbestaetigung sowie optionalem Entfernen einzelner Meldungen. Die eigentliche Kennzeichenhistorie wird dabei nie geloescht.
- **Vermeidet Alarmfluten:** Wiederholte Erkennungen desselben Watchlist-Kennzeichens derselben Quelle werden innerhalb einer einstellbaren Sperrzeit (Standard 180 Sekunden) zusammengefasst. Nach Neustarts und bei parallelen Threads bleiben Event-IDs eindeutig.
- **Sinnvolle Grenzen:** Standardmaessig mindestens 65 % OCR-Konfidenz, getrennte Video-Upload-Auftraege und optionale Sperrzeit von 0-3600 Sekunden. Pro Watchlist-Eintrag kann die Benachrichtigung deaktiviert werden. Die lokale Alarmdatenbank enthaelt keine duplizierten Bilder und ist auf 5.000 Meldungen begrenzt.
- **Video-Abbruch ohne Fehlalarm:** Watchlist-Meldungen aus Video-Uploads entstehen erst, nachdem die erfolgreiche Verarbeitung dauerhaft vermerkt wurde. Ein abgebrochenes Video hinterlaesst keine solchen Meldungen.
- **Fehlerkorrektur:** Importiertes `notify: "false"` gilt wirklich als deaktiviert; unbekannte Wahrheitswerte werden beim Speichern abgewiesen. Fehler beim Schreiben oder Loeschen der Watchlist werden in der API als HTTP 503 statt irrefuehrend als ungueltige Anfrage gemeldet.
- **Sicherungen:** Der PlateVision-ZIP-Export enthaelt die neue Alarmdatenbank als konsistenten SQLite-Snapshot, inklusive SHA-256-Manifest. Keine automatische Wiederherstellung oder externe Push-Nachricht aktiviert.
- **Sicherheit:** Texte aus Kennzeichen und Labels werden in der neuen Oberflaeche ausschliesslich als Text gerendert. API und Frontend nutzen weiterhin die Home-Assistant-Ingress-Pfade und die bisherigen Zugangskontrollen.
- **Tests:** Isolierte SQLite-/Thread-/Abbruch-/Fehlersimulationen, MP4-Probeverarbeitung sowie Release-, Frontend- und vorhandene Regressionspruefungen. Ein echter RTSP- und Supervisor-Praxistest ist weiterhin erforderlich.

## 0.13.4 — Speicheroptimierung und sichere Aufbewahrung (10.10.2026)

- **Personenbild-Aufbewahrung:** Noch referenzierte oder gemeinsam verwendete Bilder sind bei der automatischen Bereinigung geschützt, selbst wenn ihr Dateidatum alt ist. Die Löschung läuft erst nach erfolgreicher Speicherung der aktualisierten Historie.
- **Videoaufbewahrung:** Abgelaufene Jobmetadaten werden vor dem Entfernen von Videos gespeichert. Bei vollem Datenträger bleiben Jobliste und Downloads erhalten.
- **Personen-Sitzungen:** Ereignisse werden für die Sitzungsgruppierung aus SQLite gestreamt, statt alle JSON-Datensätze gemeinsam zu laden; beide Sortierrichtungen bleiben kompatibel.
- **Personenbild-Kalender:** Tages-/Bildkategorie-Zähler werden in SQLite aggregiert, ohne die komplette Galerie in Python einzulesen.
- **Fehlerfälle und Zeitzonen:** Robustere Positionsdaten, korrekt umgerechnete Personen-Zeitfilter, überprüfte Retention-Grenzen und HTTP 400 statt HTTP 500 für ungültige Personen-APIs.
- **Tests:** Regressionen für 740 Ereignisse, SQL-Parität, beschädigte Felder, Zeitfilter, retentionkritische Bilder und simulierte Video-Datenträgerfehler. Keine neue SQLite-Migration oder Modelländerungen; reale Kamera-/Supervisor-Tests bleiben offen.

## 0.13.3 — Langzeitstabilität, Datenintegrität und verständliche Diagnose (10.10.2026)

- **Indizierte Bildabrufe:** `/api/history/<id>/image/<kind>` liest gezielt den Eintrag per SQLite-Schlüssel statt die vollständige Kennzeichenliste zu durchsuchen. Dieselben Bild-URLs und Inhalte bleiben erhalten.
- **Atomare Galerie-Löschung:** `/api/people/images/delete` verwendet die bereits transaktionssichere Sammellöschung, validiert maximal 500 IDs und meldet ungültige Anfragen als HTTP 400. Ein Speicherausfall darf keinen teilweisen Erfolg vortäuschen. Die Lösch- und Cleanup-APIs verstehen `false`/`true` auch als Text korrekt; ungültige JSON-Typen werden abgewiesen.
- **Personengalerie und Sitzungen:** Die Galerie liest nur die angeforderte SQLite-Seite mit passender Trefferzahl und nicht mehr die gesamte Personenhistorie. Ungültige Limit-/Offset-Werte sowie beschädigte Bild-/Positionsfelder in Sitzungen führen nicht mehr zu einer Ausnahme.
- **Kleine JSON-Anfragen:** REST-JSON erhält ein per Anfrage geltendes 1-MiB-Limit; explizite Watchlist-/Konfigurationsimporte erhalten bis 5 MiB. Multipart-Video-Uploads und Backup-Prüfungen sind davon nicht betroffen. Flask mindestens 3.1 erforderlich.
- **Browser-Origin-Schutz:** `X-Forwarded-Host` zählt nur noch dann für die Origin-Prüfung, wenn die Anfrage dem verifizierten Home-Assistant-Ingress-Proxy zugeordnet wurde. Kein Vertrauen in das frei setzbare Headerfeld anderer Clients.
- **Systemmonitor:** `/api/system/health` klassifiziert bewusst deaktivierte RTSP- und Personenmodule als deaktiviert, nicht als defekt. Eine aktive, aber unterbrochene Kamera gilt jetzt als Handlungsbedarf. `summary` und `camera` ergänzen die bisherige API, ohne die bisherigen Felder zu entfernen; keine Zugangsdaten im neuen Monitor.
- **Diagnose-Oberfläche:** Übersichtliche Ampel, Kamera-Bildrate, verarbeitete Frames und Erkennungszahl, einklappbare technische Prüfungen, robuste Fehlermeldungen.
- **Tests:** Sieben neue isolierte Regressionstestbereiche sowie bestehende SQLite-, Kamera-, Video-, Ingress- und Frontendtests. Keine SQLite- oder Modelle-Migration. Echte RTSP-/Home-Assistant-/YOLO-Laufzeittests bleiben auf Zielhardware offen.


## 0.13.0 — Home-Assistant-Seitenleiste und Ingress (10.10.2026)

- **Echte Home-Assistant-Einbettung:** Ingress aktiviert (`ingress: true`, Port 5000, Streaming-Unterstützung) und Panel-Symbol/Titel übernommen. Der bereits aktivierte Seitenleisten-Schalter funktioniert damit nicht nur als Direktlink.
- **Sichere Standardkonfiguration:** Der bisher offen erreichbare Hostport 8087 wird standardmäßig nicht mehr veröffentlicht (`5000/tcp: null`). Für bestehende REST-Sensoren ist eine bewusste optionale Portfreigabe nötig; sie umgeht die Home-Assistant-Anmeldung.
- **Proxykompatible Pfade:** Interne Seitenlinks, statische Dateien, API-, Bild- und Uploadadressen sowie Socket.IO berücksichtigen jetzt den von Home Assistant übermittelten Ingress-Präfix. Direktbetrieb ohne Proxy bleibt bei optionaler Freigabe möglich.
- **Dashboard-Fehler behoben:** Verfrühte `onload`/`onerror`-Aufrufe für Fahrzeugbilder verursachen keine nicht definierte JavaScript-Funktion mehr. Fehlende Bilder zeigen einen Platzhalter.
- **Vorlagen-Regressionsschutz:** „Letzte Erkennung“ wird darauf geprüft, dass das Home-Assistant-YAML-Beispiel `{{ value_json.plate_text }}` nicht durch die App-Vorlagen-Engine ausgewertet wird (vorher HTTP 500 in einem Benutzerprotokoll).
- **Testgrenzen:** Offline gerenderte 26 Direkt-/Ingress-Ansichten, Pfadprüfungen und simulierte Chromium-Einbettung. Kein echter Home-Assistant-Supervisor-Proxylauf im Testcontainer; Live-RTSP und Modelle auf Zielhardware nicht getestet.
- **Kompatibilität:** Keine SQLite-Migration und keine Änderung von Erkennungsmodellen oder Datenformaten. Details in `HOME_ASSISTANT_INGRESS_0.13.0_DE.md`.


## 0.12.9 — Aufgeräumte Navigation und Schnellzugriff (10.10.2026)

- **Neu strukturiertes Hauptmenü:** Die zwölf bisherigen Register bleiben erhalten und sind in fünf verständliche Bereiche gegliedert: Übersicht, Kennzeichen, Personen, Werkzeuge und System. Die vorhandene Diagnose ist jetzt zusätzlich direkt als Menüpunkt erreichbar (insgesamt 13). Bei deaktivierter Personenanalyse verschwindet die Personengruppe automatisch.
- **Suche und Tastatur:** Die Menüsuchfunktion findet Seiten sofort nach Namen und sinnvollen Stichwörtern. `Strg+K` beziehungsweise `Cmd+K` fokussiert die Suche; `Enter` öffnet einen einzelnen Treffer. `Escape` leert zuerst die Suche und schließt danach auf Mobilgeräten das Menü. Ohne JavaScript bleiben alle Links sichtbar.
- **Aufklappbare Gruppen:** Menügruppen lassen sich getrennt öffnen/schließen, optional alle auf einmal. Die Auswahl wird lokal im Browser gespeichert; bei einem Seitenwechsel öffnet die Gruppe des aktuellen Registers automatisch.
- **Favoriten:** Bis zu sechs Register mit ★ anheften, per Seitenleiste oder Kontextleiste. Ein eigener Favoritenabschnitt dient dem schnellen Wechsel. Es werden ausschließlich bekannte Menü-IDs im Browser (`localStorage`) gespeichert – keine Kennzeichen, Bilder oder persönlichen Erkennungsdaten.
- **Kontext-Schnellwege:** Alle 13 Seiten erhalten einen dezenten Bereichspfad und passende Links zu verwandten Registern. Vorhandene APIs, Webadressen und Funktionen der Fachseiten bleiben erhalten.
- **Darstellung und Zugänglichkeit:** Desktop- und Smartphone-Menü verbessert (320 bis 1440 px), lokale Symbol-Fallbacks ohne Internet, sichtbare Tastatur-Fokusmarkierungen, `aria-current`, `aria-expanded`, `aria-pressed`, Suchergebnisstatus und geprüfte deutsche/englische/französische/italienische Menülokalisierung.
- **Tests:** Neues Offline-Regressionsskript prüft 13 Register in vier Sprachen, Routen und den deaktivierten Personenbereich. Chromium-Tests prüfen Auf-/Zuklappen, Favoriten, Tastatur, Suche, mobile Darstellung und untrusted Speicherwerte. Bestehende Backend-/SQLite-Tests bleiben unverändert.
- **Kompatibilität:** Reine UI-Erweiterung; **keine** Datenbankmigration, kein Konfigurationswechsel, keine Änderung an KI-Modellen oder Kamerapipeline. Keine tatsächliche Home-Assistant-/RTSP-/Modell-Laufzeitausführung in der Testumgebung.
- Einzelheiten: `NAVIGATION_0.12.9_DE.md`.

## 0.12.8 — Qualitätsprüfung und gespeicherte Suchfilter (09.10.2026)

- **Neue Prüf-Funktion für Kennzeichen:** Jede Erkennung kann in der Suche als `unreviewed`, `confirmed`, `needs_review` oder `ignored` markiert und mit einer Notiz von maximal 1.200 Zeichen versehen werden. Die originalen OCR-, Fahrzeug- und Bilddaten werden nicht verändert. Eine Statusübersicht ermöglicht den direkten Einstieg in gefilterte Einträge.
- **Transaktionale SQLite-Updates:** Neue gezielte Eintragsaktualisierung per Primärschlüssel statt Full-History-Sync; Änderung und SQL-Prüfstatusindex werden gemeinsam mit einer Revisionsnummer geschrieben. Simulierte Dateifehler rollen alles zurück. `expected_updated_at` verhindert unbemerktes Überschreiben einer anderen Prüfung (HTTP 409). Der bestehende JSON-Rückfallpfad rollt gescheiterte Änderungen ebenfalls zurück.
- **API/Exporte:** `GET /api/history/review-summary` und `PATCH|POST /api/history/<id>/review`; `review_status` als zusätzlicher SQL- und Legacy-Filter in Historien-/Such- und Export-APIs. CSV-Export enthält `review_status`, `review_note`, `review_updated_at`, inklusive bestehendem Schutz gegen CSV-Formelinjektion.
- **Neue Suchprofile:** Bis zu zwölf Filterkonfigurationen (Suchtext, Suchmodus, Datum, Konfidenz, Quelle, Kennzeichenfilter, Prüfstatus, Sortierung) können im Browser gespeichert, wieder angewendet und gelöscht werden; keine Speicherung auf dem PlateVision-Server. Speicherung nutzt ausschließlich `localStorage`, sicher gerenderte Optionen und explizite Bestätigung vor dem Überschreiben/Löschen.
- **Tests:** Neue SQLite-/JSON-/API-Regressionsfälle für Schreibfehler, 409-Konflikte, unzulässige Eingaben und Statusfilter; zusätzliche Chromium-Oberflächentests für Desktop, mobile Dialoge, XSS-Notizen und Suchprofile. Alle bisherigen Regressionstests laufen weiter.
- **Kompatibilität/Grenzen:** SQLite-Schema bleibt v2, vorhandene Kennzeichen und Datenbanken werden nicht automatisch verändert. Kein echter Kamera-, KI- oder Home-Assistant-Laufzeittest; ein vollständiges Backup vor dem Upgrade bleibt Pflicht. `ignored` ist eine Review-Markierung, **kein Löschen und kein Ausschluss aus bisherigen Verkehrszahlen**.

## 0.12.7 — Indexierte SQLite-Suche, Statistik und Abfragen (09.10.2026)

- **Datenbankschema v2:** Zusätzliche, abfragbare Metadatenfelder und Indizes für lokale Zeitstempel, normierte Kennzeichen, Quelle, Fahrzeugtyp/Farbe sowie Personenfilter. Bestehende v1-Datenbanken werden beim Start unter Prozesssperre in **einer SQLite-Transaktion** aktualisiert. JSON-Originale und gespeicherte Ereignis-Payloads bleiben unverändert; bei Fehler erfolgt Rollback und Schreibschutz.
- **Such-API:** Gefilterte Trefferzahl und Seitenresultate kommen direkt aus einer konsistenten SQLite-Lesetransaktion. Normale Fuzzy-Suche wird mittels begrenzter SQLite-Funktion gefiltert, ohne alle JSON-Ereignisse als Python-Liste zu laden. Neu: `match_mode=exact|starts_with|contains` (Standard `contains`) für schnelle indexgestützte Exakt-/Präfixsuche.
- **Personenhistorie:** Datums-/Quellen-/Konfidenz-/Review-/Track-/Bildfilter und Paginierung erfolgen in SQLite; auch wichtige Personenaggregate, Anwesenheits-Zeitraum und die neuesten Ereignisse lesen gezielt aus SQL.
- **Statistik/Export:** Kennzeichen-Dashboard-Aggregate, Facetten, Autovervollständigung, Tages-/Stundenverlauf und die Vorfilterung der Verkehrsstatistik laufen über SQLite. CSV/JSON-Export lädt Ergebnisse blockweise aus einer konsistenten SQLite-Lesetransaktion (sofern keine spezielle Python-Gruppierung nötig ist).
- **Fehlerbehebung:** Ungültige Konfidenzwerte in Personensitzungen führen nicht mehr zu Ausnahmen. Die Suchmaske bietet die zwei neuen Modi und die Diagnose zeigt die SQL-Indexversion.
- **Regressionen:** Neue v1->v2-Upgrade- und Rollbacktests, SQL/Legacy-Vergleichstests für Kennzeichen und Personen, Transaktions-, Sortier-, Sicherheits-, Export- und Statistikfälle. Keine Realhardware-/YOLO-Ausführung erfolgt.
- **Grenzen:** Komplexe Sitzungsheuristiken, Regex und Fuzzy-Gruppierung eindeutiger Kennzeichen brauchen weiterhin Python; der historische Schreibpfad hält vorerst eine In-Memory-Kopie. **Kein Downgrade ohne sorgfältige Rücksicherung/Rückexport.**

## 0.12.6 — SQLite-Migration für Kennzeichen und Personen (09.10.2026)

- **Neue Datenhaltung:** Kennzeichen- und Personenhistorien werden beim ersten Start automatisch und kontrolliert aus ihren bestehenden JSON-Dateien in SQLite-Datenbanken übernommen. Die JSON-Originale und ihre `.bak` bleiben unverändert; zusätzlich werden einmalige `*.pre-sqlite.json`-Sicherungen angelegt.
- **Datenintegrität:** Migration prüft jeden Datensatz, Anzahl, Reihenfolge und SQLite-`integrity_check`. Fehlende oder mehrfach verwendete Eintrags-IDs werden in der neuen DB eindeutig ergänzt; der Originalzustand bleibt im gesicherten JSON erhalten. Fehler aktivieren den Schreibschutz statt auf eine leere Historie zurückzufallen.
- **Mehr Stabilität:** SQLite-Transaktionen mit WAL, `synchronous=FULL`, Revisionsprüfung bei konkurrierenden Schreibern und Rollback bei Fehlern. Neue Erkennungen werden gezielt eingefügt, statt die ganze Historie als JSON zu speichern. Indexgestützte Listenansicht und Einzelabfrage; komplexe Such- und Statistikberechnungen verwenden zunächst weiterhin den kompatiblen In-Memory-Datenbestand.
- **Backups/Rückexport:** ZIP-Datensicherung enthält jetzt echte SQLite-Online-Snapshots und aktuelle JSON-Exporte (statt veralteter Altdateien), jeweils mit SHA-256-Manifest. Neues Offline-Werkzeug `tools/export_sqlite_to_json.py` erzeugt einen überprüfbaren, niemals automatisch eingespielten Rückexport.
- **Oberfläche:** Diagnose zeigt Datenbankstatus, Eintragszahlen und Vorhandensein der Migrationssicherung; kein automatisches Zurückspielen von Daten.
- **Kompatibilität:** API-Endpunkte und JSON-Rückgabe bleiben bestehen. Zehn neue Migrations-/Integritätstestbereiche. Eine Rückkehr zur alten Version **ohne vorherigen SQLite-Rückexport** ist nicht sicher; siehe `SQLITE_MIGRATION_0.12.6_DE.md`.
- **Grenzen:** Kein Test mit Home Assistant/RTSP/KI auf Zielhardware; dies ist eine Source-Release-Version, kein gebautes Docker-Image.

## 0.12.5 — Schreibschutz bei Datenkorruption, geprüfte Backups und korrekter Export (09.10.2026)

- **Schutz vor Datenverlust:** Ist eine Haupt-JSON-Datei und ihre `.bak` beschädigt oder strukturell unbrauchbar, wird sie nicht mehr mit leeren Standarddaten überschrieben. Die Anwendung aktiviert einen Schreibschutz für verändernde API-Anfragen, lässt aber Diagnose, Datensicherung und Stoppen des Streams zu.
- **Sicherungsprüfung (neu):** `POST /api/system/backup/verify` validiert das PlateVision-ZIP-Format, enthaltene Dateipfade, Dateianzahl und Größenlimits sowie SHA-256-Prüfsummen, ohne Archive zu entpacken oder produktive Dateien zu überschreiben. Upload und Prüfergebnis im Diagnose-Layout.
- **Historienexport:** Alter Exportknopf nutzte die paginierte API (max. 1.000 Datensätze trotz größerem Limit). Er verwendet jetzt den unpaginierten Serverexport. CSV und JSON werden zeilenweise ausgegeben statt vollständig im Antwortpuffer aufgebaut.
- **Personenbilder:** Speicherung über temporäre Dateien und atomaren Austausch; sicherere Dateinamen, Verzeichnis-/Symlinkschutz und sauberer Rückfall bei fehlgeschlagenem Austausch.
- **Tests:** Fünf neue Testbereiche mit echten ZIP-Archiven, beschädigten Sicherungen, Pfadmanipulation, Schreibfehlern und über 1.500 exportierten Einträgen; bestehende Tests werden mit neuen Fail-safe-Erwartungen fortgeführt.
- **Grenzen:** Weiterhin JSON als primäre Speicherung, kein automatischer ZIP-Import oder Restore. Kein Test mit echtem Home Assistant, realen KI-Gewichten oder RTSP-Hardware.
- Details: `TIEFENPRUEFUNG_0.12.5_DE.md`.

## 0.12.4 — Transaktionale Importe, stabilere Video-Jobs und Kamera-Ressourcen (09.10.2026)

- **Watchlist:** Mehrfacheinträge werden als Ganzes validiert und in einem einzigen atomaren Schreibvorgang übernommen. Bei Fehlern bleibt die vorherige Watchlist unverändert; Duplikate werden kontrolliert aktualisiert.
- **Modelle/KI:** Zusätzliche Einstellungswege (u.a. OCR, Personen, Testkonfiguration, Modell-Dateiupload) verwenden nun durchgängig die Inferenzsperre. Reloads werden zusammengefasst, statt sich als Threads unbegrenzt zu vermehren.
- **Kamera/RTSP:** Blockierendes `VideoCapture.read()` und natives `release()` laufen nicht länger gleichzeitig in zwei Threads. OpenCV-Lese-/Öffnungszeitlimits werden unterstützt, sofern die Laufzeit sie anbietet.
- **Video-Job-Verwaltung:** Neue Jobs werden vor dem Threadstart dauerhaft registriert. Bei Neustart werden vorläufige Erkennungen unterbrochener Jobs entfernt. Speicherfehler hinterlassen einen sichtbaren Retry-Status; Löschvorgänge setzen den Job bei fehlender Persistenz zurück.
- **Uploads/API-Grenzen:** JSON-Import maximal 4 MiB, Watchlist maximal 5.000 Einträge, Modell-Uploads in begrenzten Blöcken, Batch-Analyse maximal 200 Kennzeichen. Begrenzte Timeline-/Autocomplete-Ausgaben und keine unnötigen Base64-Bilder in der Duplikatvorschau.
- **Tests:** Neun zusätzliche automatisierte Testbereiche mit realen Funktionen und Fehlerdoubles sowie bestehende Personen-/Kern-/Resilience-/Integritäts-/MP4-Tests. Neue Tests sind Bestandteil der CI-Pipeline.
- **Kompatibilität:** Keine Änderung an vorhandenen JSON-Schemata oder REST-Pfaden, keine automatische Datenmigration. Vollständiger Home-Assistant-/Kamera-/KI-Laufzeittest bleibt auf Zielhardware erforderlich.
- Details: `TIEFENPRUEFUNG_0.12.4_DE.md`.

## 0.12.3 — Sichere Wiederherstellung, parallele Zugriffe und Datenexport (09.10.2026)

- **Konfiguration:** Speichern ist transaktionsähnlich gegen Schreibfehler abgesichert. Fehlgeschlagene Änderungen werden im Speicher auf den zuletzt erfolgreich gespeicherten Stand zurückgesetzt; API meldet HTTP 503 statt Erfolg.
- **Gleichzeitige Zugriffe:** HTTP-Konfigurationsänderungen werden über ihren gesamten Änderungsablauf serialisiert; fremde Browser-Schreibanfragen via Origin/Fetch-Metadaten werden abgewehrt. Bestehende Home-Assistant-Clients ohne Browser-Origin bleiben kompatibel.
- **Kamera:** Ein Stop/Start kann nicht über noch lebende RTSP-Threads hinweg erfolgen. Langsame Kamera-Verbindungen und die KI-Verarbeitung prüfen Abbruchsignale erneut.
- **KI/Modellverwaltung:** Exklusive Modelloperationen für Livestream, Video und Bildtests; Modell-Neustart wartet auf laufende Erkennungen und blockiert zusätzliche Neustart-Anfragen. Gleichzeitige interaktive Bildanalysen werden begrenzt.
- **Datensicherung (neu):** `/api/system/backup/preview` und `/api/system/backup` plus Download in der Diagnose. Streaming-ZIP mit SHA-256 pro enthaltener Datei, Größen-/Dateilimit und Ausschluss von Symlinks. Speichert ausgewählte JSON-Konfigurationen/-Historien, Watchlist, Jobmetadaten und Kennzeichen-/Fahrzeug-/Personenbilder. **Keine Videos oder KI-Modelle** und **keine automatische ZIP-Wiederherstellung**.
- **Speicherbereinigung:** Historie wird vor endgültiger Mediendatei-Löschung gespeichert; partielle Dateilöschfehler und fehlerhafte Video-Job-Metadatenspeicherung werden nicht länger als erfolgreich gemeldet.
- **Darstellung/Sicherheit:** Zusätzlicher XSS-Schutz in Live-Erkennungskarten, Personenarchiv-Datumsnavigation und Bildlinks. Diagnose zeigt Größe/Dateianzahl des ZIP-Datenexports an.
- **Statistik/API:** Zeitzonen-Offsets bei Dashboard-Datumsvergleichen korrekt lokalisiert; letzte Erkennung in Statistiken ohne unnötige Base64-Bilddaten.
- **Qualitätssicherung:** Neue Regressionstests für Konfigurations-Rollback, parallele Änderungen, Browser-Origin, ZIP-Prüfsummen, Modell-/RTSP-Neustart und Löschfehler; zusätzliche Chromium-XSS-Tests in Live und Personenarchiv; Testfälle in CI.
- **Grenzen:** Kein realer Home-Assistant-/RTSP-/KI-Modell-Laufzeittest in dieser Umgebung; ZIP-Export ist keine zeitpunktgenaue Datenbank-Snapshot-Transaktion. Produktiv erst nach Backup und Hardwaretest aktualisieren.
- Details: `TIEFENPRUEFUNG_0.12.3_DE.md`.

## 0.12.2 — Datensicherheit, Suche und Langzeitstabilität (09.10.2026)

- HistoryManager / WatchlistManager / PersonHistoryManager: atomare Schreibvorgänge, konsistentes Zurückrollen nach Datenträgerfehlern, verlässliche Fehlerantworten, Thread-Schutz für Kernoperationen.
- Zusätzlich atomische Rotation der letzten intakten `.bak`-Kopie; defekte Dateien werden in der Diagnose sichtbar.
- Kennzeichen-Suche ohne stille Begrenzung auf 1.000 Einträge; Regex-Muster werden vor Ausführung auf ein risikoarmes Subset beschränkt.
- Browser-Suche, Watchlist, Facetten, Batch-Analyse, Statistik und Modellverwaltung härten HTML/Text gegen XSS ab; lange Watchlist-Werte beschädigen das mobile Layout nicht mehr.
- Kompakte Suchantworten auf Wunsch ohne Bild-Base64; JSON-Historienexport ohne Bilder als datensparsamer Standard (ausdrückliche Freigabe für Bildexport nötig).
- Statistik robust gegen ungültige Konfidenzen, NaN/Inf, unterschiedliche Zeitzonen und zukünftige Ereignisse.
- Video-Jobs über Neustarts hinweg wiederfindbar, abgebrochene Jobs rollen zugehörige Kennzeichenereignisse zurück; Schreibfehler während eines Videos beenden den Job mit Fehler statt mit scheinbarem Erfolg.
- Videozeitbasierte Kennzeichen-Deduplizierung und sichtbare Zahl wirklich gespeicherter Erkennungen in der Videoverwaltung.
- Neue Speicherintegritätsprüfung (`/api/system/storage-audit`) mit Oberfläche; optionaler HTTP-Basic-Netzwerkzugriffsschutz (nur mit HTTPS/VPN).
- 7 neue Fehlersimulationsbereiche, 13 Deep-Regression-Testkategorien, Chromium XSS-/Mobiltests sowie Syntax- und Metadatenprüfungen. Realtest mit Home Assistant, Kamera, GPU/YOLO weiterhin offen.
- Siehe `TIEFENPRUEFUNG_0.12.2_DE.md`.


## 0.12.1 — Vertiefte Qualitäts- und Sicherheitsprüfung (09.10.2026)

- Video-Jobsteuerung mit begrenzter Parallelität, Fortschritt, Abbruch-API und Abbruchschaltfläche.
- Vollständige Bereinigung von VideoCapture/Writer nach Fehlern oder Abbruch; keine unvollständigen MP4-Dateien.
- Videouploads werden vor dem Multipart-Einlesen und beim Schreiben in Blöcken auf Größe geprüft (standardmäßig 400 MB).
- Speicherbereinigung und Upload-Start sind synchronisiert; aktive oder wartende Jobs können nicht versehentlich gelöscht werden.
- Automatische TTL-Bereinigung für fertige Video-Jobs und UUID-benannte, nach Neustart verwaiste Dateien.
- RTSP-Verbindungsfehler geben OpenCV-Captures zuverlässig frei; aktueller statt alter Frames wird bevorzugt und ungültige Intervallwerte werden abgefangen.
- Kennzeichenhistorie: Zeitstempel mit Zeitzone, gültige Konfidenzwerte und vollständiger CSV-/JSON-Export jenseits von 1000 Treffern.
- CSV-Exportwerte werden gegen Tabellen-Formelinjektion geschützt, Diagnoseansichten maskieren Zugangsdaten.
- Personen-Foto-Vorschau überschreibt keine globalen Live-Einstellungen mehr (thread-lokale Konfiguration).
- Uploadbilder werden vor dem Dekodieren auf Dateigröße und Pixelanzahl geprüft; Testseiten gegen zusätzliche HTML-Injektionen gehärtet.
- Neuer datensparsamer Support-Bericht unter `/api/system/support-report` und in der Diagnoseoberfläche.
- 12 zusätzliche Regressionstestbereiche, darunter eine echte MP4-Verarbeitung und deren Abbruch; Tests in CI eingebunden.
- Vollständige Dokumentation unter `TIEFENPRUEFUNG_0.12.1_DE.md`; Realtest mit Home Assistant/RTSP/YOLO weiterhin nötig.

## 0.12.0

### Nachprüfung und UI-Optimierung (09.10.2026, überarbeitete Quellfassung)
- Responsives, barriereärmeres Hell-/Dunkel-Layout mit offlinefähigen SVG-Menüsymbolen.
- Sicherere API-Pagination und JSON-Sicherung; zeitzonenfeste Filter; verbesserter RTSP-Stopp.
- RTSP-Zugangsdaten in Status und Logs maskiert, exponierter Debugger deaktiviert.
- Robustere Dashboard-Aktualisierung und Schutz vor HTML-Injektionen.
- Zusätzliche Backend-, Frontend- und Browser-Regressionstests.
- Details und Testgrenzen unter `OPTIMIERUNGEN_2026_DE.md` im Repository.


### Neu
- Eigenes Menü **Personenbilder** mit Kalender, Tagesnavigation und tageweiser Galerie.
- Verfügbare Bildtage werden über `/api/people/images/days` zusammengefasst.
- Bildaufbewahrung kann direkt in der Galerie gespeichert und sofort angewendet werden.

### Verbessert
- Standard-Aufbewahrung für Personenbilder bei Neuinstallationen: 10 Tage.
- Automatisches Bild-Cleanup ist bei Neuinstallationen standardmässig aktiv.
- Abgelaufene Bilder werden beim Start, beim Speichern neuer Ereignisse und beim Öffnen der Galerie bereinigt.
- Die statistische Personen-Historie kann länger erhalten bleiben als die Bilddateien.
- Eigene Obergrenze für Personenereignisse, standardmässig 20’000, damit das 10-Tage-Archiv nicht frühzeitig abgeschnitten wird.

## 0.11.0

- Personen-Historie mit serverseitiger Filterung, Pagination und Gesamtzahl erweitert.
- Neue Personen-Sitzungen fassen zusammengehörige Track- und Positionsereignisse zusammen.
- Detailansicht mit Crop, annotiertem Bild, Vollbild, Metadaten, Label, Notiz und Prüfstatus ergänzt.
- Mehrfachauswahl und Sammellöschen für Personenereignisse hinzugefügt.
- Neue Filter für Ereignistyp, Zählstatus, Quelle, Track/ID, Bilder und Freitext.
- Personenstatistik um aktive Personen, Sitzungen, Wiederholsperren, Bildanzahl, Quellen und Spitzenstunde erweitert.
- Bild-Historie erhält Seitennavigation und konsistente Filter.
- CSV-/JSON-Export enthält zusätzliche Prüf- und Kommentarfelder.
- API-Endpunkte für Sitzungen, Ereignisdetails, Aktualisierung und Sammellöschen ergänzt.
- Projektmetadaten, Versionsangaben und Release-Prüfung auf 0.11.0 aktualisiert.
- Leichter Regressionstest für Personen-Historie, Sitzungen, Reviews und Sammellöschen ergänzt.

## 0.10.0

### Stabilität und Sicherheit

- Konfiguration, Kennzeichenhistorie, Watchlist und Personenhistorie werden atomar gespeichert und erhalten eine `.bak`-Sicherung.
- Beschädigte JSON-Dateien können beim Start automatisch aus der letzten Sicherung wiederhergestellt werden.
- Frische Installationen starten nicht mehr automatisch mit der Platzhalter-RTSP-Adresse; Autostart erfordert ein aktiviertes RTSP-Modul und eine echte URL.
- Modell-Uploads werden zunächst temporär gespeichert, danach auf tatsächliche Grösse geprüft und atomar übernommen.
- Nach einem Modell-Upload wird die SHA-256-Prüfsumme angezeigt; die Oberfläche warnt vor nicht vertrauenswürdigen `.pt`-Dateien.
- Socket.IO akzeptiert standardmässig nur Same-Origin-Verbindungen; zusätzliche Ursprünge können über `PLATEVISION_CORS_ORIGINS` gesetzt werden.
- Sicherheitsheader, Request-ID und ein verständlicher 413-Fehler für zu grosse Uploads ergänzt.
- Modellpfade im Health-Check werden über die tatsächlichen Laufzeitpfade aufgelöst.
- Docker-Healthcheck und Home-Assistant-Watchdog ergänzt.
- Doppelte Modellpfad-Hilfsfunktionen und ungenutzte Python-Abhängigkeiten entfernt.
- `repository.yaml`, `SECURITY.md`, Upgrade-Anleitung und automatischer statischer Release-Validator ergänzt.

## 0.9.0

- PlateVision für neue Veröffentlichungen auf `AGPL-3.0-only` umgestellt.
- Drittanbieter-Lizenztexte und Modellherkunft mit SHA-256-Prüfsummen ergänzt.
- Neue Seite `/legal` und neue API `/api/system/licenses` hinzugefügt.
- Lizenzhinweis ist nicht mehr frei editierbar und alte MIT-Defaultwerte werden automatisch migriert.
- Lizenz- und Herkunftsdateien werden in das Docker-Image kopiert.
- Feste öffentliche Flask-Secret-Key-Konstante durch persistenten Zufallsschlüssel ersetzt.
- Versionsanzeige und Einstellungs-Export auf die zentrale App-Version umgestellt.
- Python-Abhängigkeiten gegen unerwartete neue Hauptversionen begrenzt.

## 0.8.30
- Einstellungen-Menü erweitert und verbessert: Module & Autostart, Stream-Retry, Reconnect, Buffer, Stream-Auflösung, sichere Defaults und Test/RTSP-Defaults.
- Test & Upload erweitert: Schnellprofile, Kennzeichen-Scanstrategie, Fahrzeugfilter, Fahrzeugfarbe, OCR-Detailwerte, Bildvorverarbeitung, Personen-Zählung, Verkehrslogik, RTSP-CPU-Sparmodus und Speicheroptionen.
- Test-&-Upload-Speichern übernimmt jetzt auch RTSP-Analysebereich/CPU-Gate-Werte, YOLO-Zoom, Deduplikation, Fahrzeugklassen, OCR-Preprocessing und Traffic-Labels.
- API `/api/test/settings` kann jetzt zusätzlich RTSP-Analysebereich-Werte sicher zusammenführen; gespeicherter Straßenbereich bleibt erhalten.
- Fehlende Schnellprofil-Funktion im Testmodus ergänzt und alle neuen Testfelder mit Laden, Default und Speichern verbunden.

## 0.8.28

- Dashboard komplett überarbeitet: keine Modell-Details mehr, Fokus auf letztes Auto, letzte Person und heutige Kennzahlen.
- RTSP/Webstream-Autostart mit Startverzögerung ergänzt.
- Autostart-Einstellungen in RTSP Stream und in den allgemeinen Einstellungen sichtbar.
- `examples/` auf aktuelle API-Endpunkte, Personenanalyse und Lovelace-Beispiele aktualisiert; alte/nicht mehr passende Dateien entfernt.



## 0.8.27

- Frontend stabilisiert: optionale/verschobene UI-Elemente verursachen keine `Cannot read properties of null (reading style)` Fehler mehr.
- Gemeinsame DOM-Helfer ergänzt und harte `.style`-Zugriffe abgesichert.
- UI-Fehler-Toast zeigt jetzt zusätzlich Datei/Zeile/Spalte, damit künftige Probleme schneller gefunden werden.
- Funktionalität der 0.8.26 bleibt erhalten.

## 0.8.26

- Home-Assistant-Add-on-Metadaten korrigiert, damit die neue Version im Update-Dialog sauber erkannt wird.
- `CHANGELOG.md` für die Add-on-Update-Ansicht ergänzt.
- Version in `config.yaml`, Webinterface, API-Health und About-Ausgabe vereinheitlicht.
- Basis bleibt die funktionierende 0.8.25 mit Personen-Zähllinie, fast-plate-ocr, YOLO-Kennzeichen, YOLO-Fahrzeuganalyse und CPU-/RTSP-ROI-Fix.

## 0.8.25

- Personen-Zähllinie auf `/people` wieder einstellbar gemacht.
- Linie kann gespeichert und direkt auf RTSP angewendet werden.
- Test & Upload, Personenanalyse und RTSP verwenden dieselben Linienwerte.

## 0.8.24

- Webinterface stabilisiert.
- Personen-Crop-Speicherung korrigiert.
- Doppelte Kennzeichen und Unique-Filter verbessert.
- Historie, Suche und Test & Upload korrigiert.

## 0.8.23

- CPU-/RTSP-Fix für Straßenbereich/Polygon.
- YOLO läuft bei RTSP wieder im gepufferten Analysebereich statt permanent auf dem kompletten Frame.

## 0.8.22

- Diagnose & Audit in die Einstellungen verschoben.
- Personenanalyse Demo-Daten entfernt.

## 0.8.21

- fast-plate-ocr als Standard-OCR integriert.
- YOLO-Kennzeichendetektor und YOLO-Fahrzeuganalyse integriert.
- Fahrzeugtyp, Kennzeichen-Land und Fahrzeugfarbe ergänzt.
