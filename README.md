# PlateVision – Home Assistant Add-on



![Version](https://img.shields.io/badge/version-0.13.9-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0--only-blue)
![Architecture](https://img.shields.io/badge/architecture-amd64-lightgrey)

**PlateVision** ist ein Home-Assistant-Add-on für lokale Kennzeichen-, Fahrzeug- und Personenerkennung mit RTSP-Kameras.

Die Erkennung und Datenspeicherung laufen auf dem eigenen System. Es gibt keinen verpflichtenden Cloud-Dienst und kein Abonnement.

**Quellfassung 0.13.3 (10.10.2026):** Zusätzliche Absicherung
für die Langzeitnutzung: historische Bilder werden direkt über den SQLite-ID-Index
geladen (ohne linearen Gesamtscan), die Personenbild-Galerie nutzt
SQLite-Paginierung und transaktionssichere Sammellöschung. Ungültige
Auswahlwerte oder fehlerhafte Seitenparameter erzeugen keine stillen Teil-Löschungen.
Der Bildlösch-Schalter akzeptiert `false` auch in älteren API-Aufrufen korrekt.
JSON-API-Anfragen haben jetzt ein eigenes Größenlimit (1 MiB; bis zu 5 MiB
für Konfigurations-/Watchlist-Importe), während große Video-Uploads getrennt
behandelt bleiben. Der Origin-Check vertraut einem Forwarded-Host nur bei
verifiziertem Home-Assistant-Ingress. **Die Flask-Mindestversion ist 3.1**,
da nur dort ein individuelles JSON-Request-Limit sicher einstellbar ist.

**Neue Systemdiagnose:** Kamera absichtlich ausgeschaltet ≠ Kamera defekt.
Die Diagnose zeigt eine verständliche Ampel mit Bildrate, Frame-Zahl und
Stream-Erkennungen; technische Detailprüfungen sind aufklappbar. Der Docker-
Watchdog nutzt unverändert `/api/system/live`. Keine neue SQLite-Migration,
keine Änderung an vorhandenen Historien oder KI-Modellen. Details in
[STABILITAET_UND_DIAGNOSE_0.13.3_DE.md](STABILITAET_UND_DIAGNOSE_0.13.3_DE.md).

**Bisherige Quellfassung 0.13.2 (10.10.2026):** Verbesserte Bildauswahl
für **Dashboard** und **Letzte Erkennung**: PlateVision kombiniert auf Wunsch
**OCR-Konfidenz** mit einer lokal berechneten **Schärfe-Heuristik** auf dem
Kennzeichenausschnitt. Der Mindestwert von 80–100 % OCR-Konfidenz bleibt
verbindlich. Zwei verschiedene Videoaufträge bzw. hochgeladene Dateien werden
auch bei engem zeitlichem Abstand nicht mehr als dieselbe Durchfahrt behandelt.
Die alte Auswahl nach reiner Konfidenz ist per Schalter weiterhin möglich.

Die Seite **Letzte Erkennung** verwendet jetzt auf Wunsch kleine JSON-Antworten
mit Bild-URLs statt Base64-Bildern alle fünf Sekunden erneut zu übertragen.
`GET /api/latest/display` ohne Parameter bleibt kompatibel;
`?images=0` liefert nur Metadaten und geschützte Bild-URLs. Bei unverändertem
Treffer bleiben die Bildknoten im Browser erhalten. Die Dashboard-Tageszähler
lesen leichte SQLite-Daten; unscharfe Kennzeichengruppierung bleibt berücksichtigt.
**Keine erneute SQLite-Schema-Migration und keine Änderung vorhandener Bilder.**
Vor dem Update wie üblich ein Home-Assistant-Backup erstellen. Details in
[BILDQUALITAET_UND_LEISTUNG_0.13.2_DE.md](BILDQUALITAET_UND_LEISTUNG_0.13.2_DE.md).

**Bisherige Quellfassung 0.13.1 (10.10.2026):** Vereinfachte
Arbeitsoberflächen, optisch zurückgenommene Personenbilder mit größerem
Bildausschnitt bei neuen Aufnahmen, verlegte Personen-Zähllinienkalibrierung
unter **Test & Upload** und selektierte Fahrzeugbilder für **Dashboard** und
**Letzte Erkennung**. Der optionale Qualitätsfilter ist bei neu angelegter
Konfiguration standardmäßig aktiv (min. 80 %), kann unter
**Einstellungen → Anzeige & Fahrzeugbilder** deaktiviert oder angepasst
werden; vorhandene explizite Einstellungen bleiben erhalten. **Alle bisher
aufgezeichneten Erkennungen bleiben in Historie, Suche, Statistiken und
Exporten**. Erkennungsmodelle und SQLite-Schema bleiben unverändert.
Personenbilder mit bereits abgeschnittenen Körperteilen können nicht
nachträglich wiederhergestellt werden. Details in
[LAYOUT_UND_BILDQUALITAET_0.13.1_DE.md](LAYOUT_UND_BILDQUALITAET_0.13.1_DE.md).

**Bisherige Quellfassung 0.13.0 (10.10.2026):** PlateVision kann nun direkt in
Home Assistant über die **eingebaute, angemeldete Ingress-Seitenleiste** bedient werden,
auch mit Suche, Liveansicht, Uploads und Personenbildern. Die Einbettung läuft über
Port 5000 **intern**; ein Port 8087 wird zum Schutz vor Umgehung der
Home-Assistant-Anmeldung **standardmäßig nicht mehr ins Heimnetz veröffentlicht**.
Ein Direktzugriff bleibt über das Netzwerk-Menü des Add-ons optional.
Dashboard-Bildfehler bei frühen oder gecachten Bildantworten wurden behoben;
die Vorlagen für „Letzte Erkennung“ werden erneut auf korrekt dargestellte
Home-Assistant-YAML-Beispiele geprüft. Keine SQLite-Migration nötig.
**Wichtige Aktualisierungsanweisungen:**
[HOME_ASSISTANT_INGRESS_0.13.0_DE.md](HOME_ASSISTANT_INGRESS_0.13.0_DE.md).

**Bisherige Quellfassung 0.12.9 (10.10.2026):** Die Hauptnavigation ist jetzt in
fünf Bereiche gegliedert und enthält ein Schnell-Suchfeld, lokale Favoriten,
einklappbare Gruppen sowie eine Kontextleiste mit passenden Links zwischen
verwandten Registern. Neben den zwölf bisherigen Seiten ist die **Diagnose** nun
direkt im Menü erreichbar. Tastatur: `Strg+K` / `Cmd+K` zum Suchen.
Die Änderungen betreffen ausschließlich das Frontend; alle bestehenden
Datenbanken, URLs und Erkennungsfunktionen bleiben unverändert.
Siehe [Navigation 0.12.9](NAVIGATION_0.12.9_DE.md).

**Bisherige Quellfassung 0.12.8 (09.10.2026):** Neue manuelle Qualitätsprüfung
für einzelne Kennzeichenerkennungen mit Status **Ungeprüft / Bestätigt /
Erneut prüfen / Ignoriert** und optionaler Notiz. Speichern erfolgt als
kleine atomare SQLite-Transaktion, auch mit Sperre gegen überschriebene
Änderungen aus einem zweiten Browserfenster. Filter, zusammengefasste
Prüfstatistik und CSV-/JSON-Export berücksichtigen die Informationen.
Die Suchseite bietet außerdem bis zu 12 lokal gespeicherte Filterprofile.
Die Filterprofile werden **nur im jeweiligen Browser gespeichert** und
synchronisieren sich nicht mit Home Assistant. Die manuelle Prüfung ändert
nicht die ursprünglichen OCR-Ergebnisse, Bilder, Watchlist oder Statistiken.
Details: [QUALITAETSPRUEFUNG_0.12.8_DE.md](QUALITAETSPRUEFUNG_0.12.8_DE.md).
**Keine automatische Datenmigration nötig; SQLite-Indexschema bleibt v2.**

**Bisherige Quellfassung 0.12.7 (09.10.2026):** Die normalen Suchabfragen,
Filter, Seitenlisten, Facetten, Tages-/Stundenstatistiken und Exporte nutzen jetzt
SQLite direkt. Die Suchoberfläche bietet zusätzlich exakte und präfixbasierte
Kennzeichensuche mit Datenbankindex; die Standard-Suche bleibt erhalten.
Installierte SQLite-Datenbanken aus 0.12.6 werden beim ersten Start
**transaktionssicher von Indexschema 1 auf 2 aktualisiert**. Das ist ein
Datenbank-Schema-Update; davor ein vollständiges Home-Assistant-Backup erstellen.
Details und verbleibende Grenzen: [SQLITE_ABFRAGEN_0.12.7_DE.md](SQLITE_ABFRAGEN_0.12.7_DE.md).

**Bisherige Quellfassung 0.12.6 (09.10.2026):** Kennzeichen- und
Personenhistorien nutzen nach einer gesicherten Einmalmigration SQLite statt
fortlaufend neu geschriebener JSON-Dateien. Originaldateien bleiben unverändert;
Transaktionen, Integritätsprüfungen und manueller JSON-Rückexport sind vorhanden.
**Vor dem Update ein vollständiges Home-Assistant-Backup erstellen.**
Ein Downgrade auf eine ältere Version ist ohne aktuellen SQLite-Rückexport
nicht sicher. Anleitung: [SQLITE_MIGRATION_0.12.6_DE.md](SQLITE_MIGRATION_0.12.6_DE.md).

**Diese ZIP-Datei enthält Quellcode – kein vorgefertigtes Add-on-Image.**

**Vertiefte überarbeitete Quellfassung 0.12.1 (09.10.2026):** Die neue
[Tiefenprüfung](TIEFENPRUEFUNG_0.12.1_DE.md) beschreibt gefundene Fehler, Änderungen,
Regressionstests und verbleibende Risiken. Frühere Layout- und Sicherheitsanpassungen
sind in [OPTIMIERUNGEN_2026_DE.md](OPTIMIERUNGEN_2026_DE.md) dokumentiert.
Dies ist eine lokal weiterentwickelte Quellfassung; vor dem Produktiveinsatz selbst bauen und mit echter Hardware testen.

> **Aktueller Stand:** Die Bedienoberfläche ist hauptsächlich deutsch. Das Standard-Add-on verwendet die CPU. GPU-Beschleunigung benötigt eine dafür angepasste Laufzeitumgebung.

## Unterstützung

PlateVision ist frei verfügbar. Wer die Entwicklung unterstützen möchte:

<a href="https://www.buymeacoffee.com/geartec" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>


## Inhalt

- [Funktionen](#funktionen)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Erste Einrichtung](#erste-einrichtung)
- [Personenbilder und Aufbewahrung](#personenbilder-und-aufbewahrung)
- [Home-Assistant-Integration](#home-assistant-integration)
- [Daten und Backups](#daten-und-backups)
- [Datenschutz und Sicherheit](#datenschutz-und-sicherheit)
- [API](#api)
- [Fehlerbehebung](#fehlerbehebung)
- [Lizenz und Modellherkunft](#lizenz-und-modellherkunft)
- [English summary](#english-summary)

---

## Datenbankmigration ab 0.12.6

Beim ersten Start werden die bisherigen Kennzeichen- und Personenhistorien automatisch nach `data/history/plates.sqlite3` bzw. `data/people/people.sqlite3` übernommen. Die alten `.json`-, `.bak`- und neuen `*.pre-sqlite.json`-Dateien bleiben als Migrationsnachweis erhalten, werden danach jedoch **nicht aktualisiert**. Die normalen API-Antworten bleiben JSON.

**Achtung beim Downgrade:** Ein älteres Add-on könnte nur veraltete JSON-Historien sehen. Deshalb vor jedem Rückwechsel den aktuellen SQLite-Inhalt in ein neues JSON-Sicherungspaket exportieren. Nutze `tools/export_sqlite_to_json.py` aus diesem Quellcode und die [vollständige Migrations-/Rückfallanleitung](SQLITE_MIGRATION_0.12.6_DE.md). Das normale Backup-ZIP enthält ab 0.12.6 sowohl SQLite-Schnappschüsse als auch aktuelle JSON-Exporte.

Seit 0.12.7 finden normale Kennzeichen- und Personensuche, Trefferzählung,
Paginierung, Facetten, Autovervollständigung, Zeitverlauf sowie die wesentlichen
Statistikaggregate direkt in SQLite statt. Das reduziert große JSON-Kopien
und unnötige Arbeit im Python-Prozess. Komplexe Besuchs-/Personensitzungen,
Regex sowie unscharfe Gruppierung einmaliger Kennzeichen benötigen weiterhin
Python-Nachverarbeitung. Die Historien bleiben für alte Schreibpfade vorerst
zusätzlich im Arbeitsspeicher; dies ist **noch keine vollständige
In-Memory-Abschaffung**. Auch Fuzzy-Suchen müssen über Kandidaten vergleichen.
Die exakte und präfixbasierte Kennzeichensuche profitiert besonders von Indizes.

## Funktionen

### Kennzeichen- und Fahrzeugerkennung

- RTSP-Livestream mit Start, Stopp, Status und Snapshot
- lokale YOLO-Erkennung für Fahrzeuge und Kennzeichen
- OCR mit FastPlateOCR sowie optionalem EasyOCR-Fallback
- Kennzeichen-Normalisierung und typische OCR-Korrekturen
- Länder- und Formathinweise für CH, DE und AT
- Fahrzeugtyp-Erkennung, beispielsweise Auto, Motorrad, Bus oder LKW
- optionale Fahrzeugfarben-Analyse
- einstellbare Konfidenzen, Bildgrössen, IoU-Werte und Scanstrategien
- Analysebereich als Rechteck oder Polygon
- CPU-Sparmodus durch Crop- und Bewegungsfilter
- Schutz vor doppelten Erkennungen im selben Bild und innerhalb eines Zeitfensters

### Historie, Suche und Verkehrsauswertung

- Kennzeichenhistorie mit Bildern, Zeitstempel und technischen Metadaten
- Suche nach Kennzeichen, Datum, Quelle, Fahrzeugtyp, Farbe und Konfidenz
- Fuzzy-Suche, Regex-Suche, exakte und präfixbasierte Datenbanksuche sowie Anzeige eindeutiger Kennzeichen
- Watchlist für bekannte oder wichtige Kennzeichen
- Besuchs- und Session-Gruppierung
- Kommen-/Gehen-Auswertung
- Tages-, Stunden- und Wiederkehrerstatistiken
- CSV- und JSON-Export
- Manuelle Qualitätsprüfung pro Kennzeichentreffer (Status und Notiz), SQL-Prüfstatusfilter und Prüfstatistik
- Gespeicherte Suchfilter (bis zu 12 Profile pro Browser, lokal)
- automatische und manuelle Bereinigung alter Daten

### Personenerkennung

- Personenerkennung über die COCO-Personenklasse oder ein eigenes Human-Modell
- virtuelle Zähllinie mit Richtungserkennung
- Tracking, Debounce und Schutz vor Mehrfachzählungen
- einstellbare Personen-Zone und Grössenfilter
- Anwesenheits- und Belegungsabschätzung
- Personenhistorie mit serverseitiger Filterung und Seitennavigation
- Sitzungsgruppierung nach Track, Position und Zeitabstand
- Detailansicht mit Crop, annotiertem Bild und Vollbild
- Labels, Notizen und Prüfstatus pro Ereignis
- Sammelauswahl und Sammellöschen
- CSV- und JSON-Export

> Die Sitzungsgruppierung ist keine Gesichtserkennung. Sie verwendet Track-ID, Position, Grösse, Quelle und Zeitabstand.

### Personenbilder

- eigenes Menü **Personenbilder** unter `/people/gallery`
- Tagesnavigation und Kalender
- Anzeige aller Tage mit gespeicherten Bildern
- separate Ansicht für Crop, annotiertes Bild und Vollbild
- frei einstellbare Aufbewahrungsfrist
- automatisches oder manuelles Löschen abgelaufener Bilder
- Bildaufbewahrung unabhängig von der statistischen Personenhistorie

Bei Neuinstallationen gelten standardmässig:

| Einstellung | Standard |
|---|---:|
| Aufbewahrung der Personenbilder | 10 Tage |
| Automatische Bildbereinigung | aktiviert |
| Personenhistorie | 90 Tage |
| Maximale Personenereignisse | 20.000 |

### Test und Diagnose

- Testanalyse für einzelne Bilder
- Videoverarbeitung mit Fortschritt, Abbruch, parallelem Job-Limit und automatischer Dateibereinigung
- separater Personentest
- Modell-Upload für `.pt`, `.onnx` und `.engine`
- Modellstatus und manuelles Neuladen
- Systemstatus, Datenschutz-Supportbericht, Healthcheck und Home-Assistant-Watchdog
- Lizenz- und Modellprüfung unter `/legal`
- Konfigurations-Export und -Import
- Diagnose- und Audit-Endpunkte

### Video-Limits und Diagnose (0.12.1)

- Unter **Test & Upload → Verarbeitungsjobs** können laufende Videos kontrolliert abgebrochen werden.
- Standard: höchstens **ein** aktiver Video-Job (`PLATEVISION_MAX_VIDEO_JOBS`, 1–4).
- Standard: **400 MB** maximale Video-Datei (`PLATEVISION_MAX_VIDEO_MB`, 16–480). Flask begrenzt den gesamten HTTP-Upload auf 500 MB.
- Standard: **25 MB** je Bild (`PLATEVISION_MAX_IMAGE_MB`, 1–128) und maximal 32 Megapixel.
- Fertige oder abgebrochene Video-Dateien werden nach **168 Stunden** beim Aufräumen entfernt (`PLATEVISION_VIDEO_RETENTION_HOURS`, 0–8760; 0 deaktiviert). Auch verwaiste UUID-Dateien nach Neustarts werden bereinigt.
- Unter **Diagnose** lässt sich ein Bericht über den Systemzustand **ohne Kennzeichen, Kamerapasswörter und IP-Adressen** herunterladen.

**Hinweis:** Der reguläre Home-Assistant-Seitenleisten-Zugriff ist jetzt durch die
Home-Assistant-Anmeldung geschützt. PlateVision besitzt weiterhin **keine
separate Benutzeranmeldung** für den direkten Port. Nur wenn du den externen
Netzwerkport bewusst aktivierst (z. B. 8087), ist die Web-API von diesem Netz
aus direkt erreichbar und kann die Home-Assistant-Anmeldung umgehen. Für
externen Zugriff VPN oder einen Authentifizierungsproxy nutzen. Video-Jobs
werden nach Neustarts aus gespeicherten Jobdaten wiederhergestellt.

### Bedienoberfläche

Das Seitenmenü enthält aktuell:

- Dashboard
- Live-Ansicht
- Historie
- Suche
- Statistik
- Personen
- Personenbilder
- Letzte Erkennung
- Test & Upload
- RTSP Stream
- Einstellungen
- Lizenz & Modelle

---

## Voraussetzungen

| Anforderung | Beschreibung |
|---|---|
| Home Assistant | Home Assistant OS oder Supervised mit Add-on-Unterstützung |
| Architektur | `amd64` / x86-64 |
| Kamera | RTSP-Stream, der über FFmpeg/OpenCV erreichbar ist |
| Netzwerk | Home Assistant muss die Kamera erreichen können |
| Hardware | Ausreichend CPU und RAM für Torch, YOLO und OCR |

Die erste Installation kann länger dauern, weil unter anderem Torch, Ultralytics, OpenCV und OCR-Bibliotheken installiert werden.

Das Standard-Image enthält keine speziell konfigurierte CUDA-Laufzeit. Eine in der Oberfläche auswählbare GPU funktioniert nur, wenn die verwendete Umgebung und die installierten Bibliotheken diese tatsächlich unterstützen.

---

## Installation

1. In Home Assistant **Einstellungen → Add-ons → Add-on Store** öffnen.
2. Oben rechts **⋮ → Repositories** auswählen.
3. Folgende Repository-Adresse hinzufügen:

   ```text
   https://github.com/richieam93/platevision-ha-addon
   ```

4. Den Add-on Store neu laden.
5. **PlateVision** auswählen, installieren und starten.
6. **In Seitenleiste anzeigen** aktivieren (bei deiner Installation war dies bereits
   eingeschaltet). Die neue **PlateVision**-Seitenleiste in Home Assistant öffnen,
   oder in der Add-on-Info auf **Benutzeroberfläche öffnen** klicken.

PlateVision wird jetzt direkt innerhalb von Home Assistant angezeigt
(`http://HOME-ASSISTANT-IP:8123`). **Port 8087 ist standardmäßig deaktiviert.**
Falls du bewusst einen separaten Direktzugriff für bestehende REST-Sensoren
benötigst, kannst du im Add-on unter **Netzwerk → 5000/tcp** einen freien Hostport,
z. B. `8087`, eintragen. **Achtung:** Der direkte Port läuft außerhalb der
Home-Assistant-Anmeldung, also nur geschützt/vertrauenswürdig verwenden.

| Container-Port | Externer Standardport | Einbettung |
|---:|---:|---|
| 5000 | nicht freigegeben | Home Assistant Ingress (intern) |

Mehr dazu: [Ingress und Aktualisierung 0.13.0](HOME_ASSISTANT_INGRESS_0.13.0_DE.md).

---

## Erste Einrichtung

### 1. RTSP-Kamera verbinden

1. **RTSP Stream** öffnen.
2. RTSP-Adresse eintragen, beispielsweise:

   ```text
   rtsp://benutzer:passwort@192.168.1.100:554/stream
   ```

3. Verbindung testen.
4. Auflösung und Analysebereich kontrollieren.
5. Stream aktivieren und bei Bedarf Autostart einschalten.

Eine neue Installation startet nicht automatisch mit der Beispieladresse. Der Autostart wird erst verwendet, wenn RTSP aktiviert und eine echte URL gespeichert wurde.

### 2. Erkennung einstellen

Unter **Einstellungen** können unter anderem angepasst werden:

- Erkennungsprofile: schnell, ausgewogen, streng und Nacht
- Fahrzeug- und Kennzeichenmodell
- OCR-Engine und Vorverarbeitung
- Konfidenz- und IoU-Grenzen
- Fahrzeugklassen und Fahrzeugfarbe
- Historie, Speicher und Datenschutz
- Personenmodell, Zähllinie und Bildaufbewahrung
- RTSP-Wiederverbindung, Puffer und CPU-Sparfunktionen

### 3. Mit einem Testbild prüfen

Unter **Test & Upload** zuerst ein typisches Bild aus der späteren Kameraperspektive hochladen. Dort lassen sich Kennzeichen-, Fahrzeug- und Personenanalyse kontrollieren, bevor die Live-Verarbeitung dauerhaft gestartet wird.

---

## Personenbilder und Aufbewahrung

Die Galerie ist über das eingebettete Home-Assistant-Register
**PlateVision → Personenbilder** erreichbar. Ein direkter Pfad
`http://HOME-ASSISTANT-IP:8087/people/gallery` funktioniert nur, wenn ein
externer Port manuell aktiviert wurde.

Dort kann die Aufbewahrungsfrist zwischen einem festen Zeitraum und unbegrenzter Speicherung gewählt werden. `0 Tage` bedeutet unbegrenzt.

Beim Löschen abgelaufener Bilder bleiben die statistischen Ereignisse erhalten. Entfernt werden nur die zugehörigen Bilddateien, sofern nicht ausdrücklich das vollständige Ereignis gelöscht wird.

Die Bereinigung wird ausgeführt:

- beim Start von PlateVision
- beim Speichern neuer Personenereignisse
- beim Öffnen der Galerie
- über **Jetzt bereinigen**

Vor einer langen Aufbewahrungszeit sollte der verfügbare Speicherplatz geprüft werden.

---

## Home-Assistant-Integration

PlateVision stellt JSON-Endpunkte bereit, die sich für REST-Sensoren, Automatisierungen und Lovelace-Karten verwenden lassen.

Aktuelle Beispiele befinden sich im Ordner [`examples/`](examples/):

- [`examples/README.md`](examples/README.md)
- [`examples/configuration.yaml`](examples/configuration.yaml)
- [`examples/automations.yaml`](examples/automations.yaml)
- [`examples/lovelace_dashboard.yaml`](examples/lovelace_dashboard.yaml)
- [`examples/scripts.yaml`](examples/scripts.yaml)
- [`examples/api_endpunkte.md`](examples/api_endpunkte.md)
- [`examples/entity_ids.md`](examples/entity_ids.md)

Typische Anwendungen:

- Garagentor bei einem bekannten Kennzeichen öffnen
- Benachrichtigung bei einem Watchlist-Treffer senden
- letztes Kennzeichen und letzte Person im Dashboard anzeigen
- Tagesstatistik als Sensor übernehmen
- Systemzustand und Streamstatus überwachen

Die Beispiele müssen an die eigene Home-Assistant-Adresse, Entity-IDs und
Sicherheitsanforderungen angepasst werden. **Bestehende REST-Sensoren auf Port 8087
funktionieren nach dem Update nicht mehr**, solange du den Direktport nicht
bewusst neu aktivierst oder auf eine geeignete abgesicherte Integration umstellst.
Die wechselnden Ingress-Session-URLs eignen sich **nicht** als feste REST-Adressen
für Hintergrundautomatisierungen.

---

## Daten und Backups

PlateVision verwendet unter Home Assistant den persistenten Ordner `/data`.

| Zweck | Persistenter Pfad |
|---|---|
| Uploads und erkannte Bilder | `/data/uploads` |
| mitgelieferte und hochgeladene Modelle | `/data/models` |
| Konfiguration, Historien und Metadaten | `/data/data` |

Konfiguration, Kennzeichenhistorie, Watchlist und Personenhistorie werden atomar gespeichert. Zusätzlich werden `.bak`-Sicherungen verwendet, aus denen beschädigte JSON-Dateien wiederhergestellt werden können.

Vor grösseren Updates empfiehlt sich trotzdem ein Home-Assistant-Backup.

---

## Datenschutz und Sicherheit

- Kamera möglichst nur auf den eigenen, erforderlichen Bereich ausrichten.
- Gesetzliche Vorgaben zur Videoüberwachung, Information betroffener Personen und Speicherdauer beachten.
- RTSP-Zugangsdaten nicht in Screenshots, Issues oder öffentlichen Logs veröffentlichen.
- Das Webinterface nur in einem vertrauenswürdigen Netzwerk oder hinter einer geeigneten Authentifizierung beziehungsweise einem Reverse Proxy betreiben.
- Nur Modelle aus vertrauenswürdigen Quellen hochladen. PyTorch-Modelle können beim Laden ausführbaren Python-/Pickle-Inhalt enthalten.
- Aufbewahrungsfristen so kurz wie für den Zweck notwendig einstellen.

Die Erkennung und die gespeicherten Daten bleiben lokal. Die aktuelle Weboberfläche lädt jedoch einige Frontend-Ressourcen wie Bootstrap, Font Awesome, Google Fonts und Socket.IO von öffentlichen CDNs. Ohne Internetzugang können deshalb Teile der Darstellung oder Bedienung eingeschränkt sein, obwohl die Erkennungslogik lokal läuft.

Sicherheitsprobleme bitte gemäss [`SECURITY.md`](SECURITY.md) melden.

---

## API

Einige wichtige Endpunkte:

| Endpunkt | Zweck |
|---|---|
| `/api/system/live` | leichter Healthcheck für Watchdog |
| `/api/system/health` | ausführlicher Systemstatus |
| `/api/system/version` | Version und Edition |
| `/api/system/licenses` | Lizenz- und Modellprüfung |
| `/api/stream/status` | Streamstatus |
| `/api/latest/full` | letzte vollständige Erkennung |
| `/api/history` | Kennzeichenhistorie |
| `/api/statistics/traffic` | Verkehrsauswertung |
| `/api/people/history` | Personenhistorie |
| `/api/people/sessions` | gruppierte Personensitzungen |
| `/api/people/images/days` | verfügbare Bildtage |

Die vollständige Übersicht befindet sich in [`examples/api_endpunkte.md`](examples/api_endpunkte.md).

---

## Screenshots

Die folgenden Bilder zeigen zentrale Bereiche. Einzelne Details können sich je nach Konfiguration unterscheiden.

| Dashboard | RTSP Stream | Letzte Erkennung |
|---|---|---|
| ![Dashboard](Bilder/dashboard.JPG) | ![RTSP](Bilder/rtsp.JPG) | ![Letzte Erkennung](Bilder/Letzte%20Erkennung.JPG) |

| Historie | Einstellungen | Test & Upload |
|---|---|---|
| ![Historie](Bilder/Historie.JPG) | ![Einstellungen](Bilder/einstellungen.JPG) | ![Test](Bilder/test.JPG) |

---

## Fehlerbehebung

### Der Stream startet nicht

- RTSP-Adresse mit VLC oder FFmpeg testen.
- Benutzername, Passwort, Port und Stream-Pfad prüfen.
- Erreichbarkeit aus dem Home-Assistant-Netz kontrollieren.
- Unter **RTSP Stream** Status und Snapshot testen.
- Logs des Add-ons prüfen.

### Home Assistant zeigt keine neue Version

- Add-on Store neu laden.
- Repository entfernen und erneut hinzufügen, falls der Cache hängen bleibt.
- Prüfen, ob `platevision/config.yaml` die erwartete Versionsnummer enthält.

### Erkennung ist langsam

- Analysebereich verkleinern.
- Bewegungsfilter aktivieren.
- Bildgrössen und Erkennungsintervall reduzieren.
- nicht benötigte Fahrzeug- oder Personenfunktionen deaktivieren.
- ein schnelleres Erkennungsprofil auswählen.

### Es werden zu viele Bilder gespeichert

- Aufbewahrungsfrist unter **Personenbilder** oder **Einstellungen** reduzieren.
- automatische Bereinigung aktivieren.
- nicht benötigte Vollbilder oder annotierte Bilder deaktivieren.
- Speicherübersicht kontrollieren.

---

## Entwicklung und Prüfung

Lokale statische Prüfung:

```bash
python3 tools/validate_release.py
python3 tools/test_people_history.py
```

Änderungen und bekannte Versionshinweise stehen in:

- [`CHANGELOG.md`](CHANGELOG.md)
- [`UPGRADE_0.12.0.md`](UPGRADE_0.12.0.md)
- [`RELEASE_CHECKLIST_0.12.0.md`](RELEASE_CHECKLIST_0.12.0.md)

Fehler und Funktionswünsche können über [GitHub Issues](https://github.com/richieam93/platevision-ha-addon/issues) gemeldet werden.

---

## Lizenz und Modellherkunft

PlateVision wird unter der **GNU Affero General Public License Version 3** (`AGPL-3.0-only`) veröffentlicht. Der vollständige Lizenztext befindet sich in [`LICENSE`](LICENSE).

Das Projekt verwendet Drittanbieterbibliotheken und mitgelieferte Modelle mit eigenen Lizenzbedingungen. Herkunft, Prüfsummen und Lizenztexte sind dokumentiert in:

- [`NOTICE.md`](NOTICE.md)
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
- [`MODEL_PROVENANCE.md`](MODEL_PROVENANCE.md)
- [`third_party_licenses/`](third_party_licenses/)

Die Lizenz- und Modellinformationen können in der laufenden Anwendung zusätzlich unter `/legal` kontrolliert werden.

Copyright © 2025–2026 `richieam93`

---

## Unterstützung

PlateVision ist frei verfügbar. Wer die Entwicklung unterstützen möchte:

<a href="https://www.buymeacoffee.com/geartec" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

---

## English summary

PlateVision is an `amd64` Home Assistant add-on for local RTSP-based license plate, vehicle and person detection.

Main features include:

- RTSP live stream and configurable analysis area
- local YOLO vehicle, plate and person detection
- FastPlateOCR with optional EasyOCR fallback
- plate history, search, watchlist and traffic statistics
- person counting, tracking, review workflow and sessions
- person image gallery with configurable retention, defaulting to 10 days on new installations
- JSON APIs for Home Assistant REST sensors and automations
- persistent storage under `/data`
- health, diagnostics, model provenance and license pages

The standard add-on is CPU-based and supports `amd64`. Detection and stored data remain local, while parts of the current web interface load frontend assets from public CDNs.

Installation repository:

```text
https://github.com/richieam93/platevision-ha-addon
```

### Sichere Suche und Speicherdiagnose (0.12.2)

- Die Suchseite lädt standardmäßig nur Metadaten (`include_images=false`); Bilder werden bei Bedarf
  über die geschützten Einzelbild-Endpunkte geladen. Ältere Clients können weiter `include_images=true` nutzen.
- Regex-Suchen erlauben nur einfache, begrenzte Muster ohne Gruppen, Alternativen oder Rückverweise,
  damit aufwendige reguläre Ausdrücke den Server nicht blockieren.
- CSV-/JSON-Exporte enthalten weiterhin alle Treffer; JSON exportiert Bilddaten standardmäßig nicht.
  Nur die Kombination aus `include_images=true` und erlaubender Datenschutzkonfiguration
  `privacy.export_include_images` schaltet sie frei.
- `/api/system/storage-audit` meldet freien Speicher, Haupt-/Backup-JSON-Konsistenz,
  Video-Jobs und ob der optionale Passwortschutz aktiviert ist, ohne Nutzdaten preiszugeben.
- Optionaler LAN-Zugriffsschutz: Umgebungsvariablen `PLATEVISION_AUTH_USERNAME` (Standard
  `platevision`) und `PLATEVISION_AUTH_PASSWORD` setzen. Nur hinter **HTTPS oder VPN** verwenden:
  HTTP Basic ist keine Verschlüsselung! Ohne Passwort bleibt die bisherige Zugriffskonfiguration
  unverändert. Home-Assistant-Watchdog (`/api/system/live`) ist weiterhin zugänglich.
- Die Video-Job-Liste wird datensparsam in `data/video_jobs.json` gespeichert. Nach einem Neustart
  sind fertige Downloads wieder verfügbar, solange die Dateien vorhanden sind. Unterbrochene
  Jobs werden mit Fehlerstatus angezeigt und nicht unbeaufsichtigt neu gestartet.
- Video-Kennzeichen-Speicherung wird anhand der **Videozeit** entdoppelt.
  `PLATEVISION_VIDEO_PLATE_COOLDOWN_SECONDS` regelt den Abstand (Standard 10 Sek.).

**Vor einem Update:** Home-Assistant-Backup erstellen, Quell-Add-on neu bauen/aktualisieren,
Einstellungen und Modelle prüfen und mit einer echten RTSP-Kamera testen. Bei einem
Upgrade aus 0.12.1 bleiben die bisherigen Konfigurationsfelder erhalten.

### Sicherung und Betriebszuverlässigkeit (0.12.3)

Die Seite **Diagnose** bietet unter **Datensicherung herunterladen** eine
ZIP-Datei mit gespeicherten JSON-Daten (Konfiguration, Kennzeichen- und
Personenhistorie, Watchlist, Video-Job-Metadaten) sowie lokalen
Kennzeichen-/Fahrzeug-/Personenbildern. Der Inhalt wird durch
`platevision-backup-manifest.json` mit SHA-256-Prüfsummen dokumentiert. Über
`GET /api/system/backup/preview` können erwartete Dateianzahl und Umfang
angefragt werden; `GET /api/system/backup` startet den Download.

**Achtung:** Die Sicherung kann Kennzeichen, Personenbilder und Zugangsdaten
für Kameras enthalten. Sie muss vertraulich und möglichst verschlüsselt
aufbewahrt werden. **Videos, KI-Modelle und geheime Dateien außerhalb des
Anwendungsdatenverzeichnisses sind nicht enthalten.** Zur vollständigen
Wiederherstellung vorerst Home Assistant selbst sichern; es gibt **keinen
automatischen Import** des ZIPs. Die Sicherung ist nicht atomar gegenüber
allen laufenden Schreibvorgängen: nach Möglichkeit bei ruhendem Stream
exportieren. Die exportierte Datenmenge wird durch die optionale
Umgebungsvariable `PLATEVISION_BACKUP_MAX_MB` begrenzt (Standard 512 MB,
zugelassen 1–2048 MB, zusätzlich maximal 20.000 Dateien).

Gleichzeitige Konfigurationsspeicherungen und Kamera-Neustarts sind in 0.12.3
stärker abgesichert. Nach einem Speicherfehler kann die Weboberfläche eine
Fehlermeldung anzeigen; bitte Datenträger und `/api/system/storage-audit`
prüfen und die Änderung erneut speichern. Ein manuelles Modell-Neuladen
während laufender Erkennung wartet, statt Modellreferenzen zu löschen.

**Für den Produktivbetrieb:** Nach vorherigem Home-Assistant-Backup neues
Quell-Add-on bauen, Einstellungen kontrollieren und RTSP, YOLO/OCR,
Personenerkennung, Exporte, Video-Jobs und Neustarts auf der Zielhardware
testen. SQLite statt großer JSON-Historien bleibt eine mögliche spätere,
explizit zu testende Migration und ist **nicht** Bestandteil von 0.12.3.

### Weitere Stabilitätskorrekturen (0.12.4)

- Watchlist-Import (`POST /api/watchlist/import`) verarbeitet alle Einträge
  transaktional und nimmt Änderungen bei ungültigen Daten oder Datenträgerfehlern
  zurück. Bis zu 5.000 Listeneinträge pro Import sind zulässig.
- Modell- und OCR-Einstellungen laden KI-Komponenten nicht mehr ungeschützt
  parallel zu einer laufenden Erkennung neu. Mehrere gleichzeitige Reload-Anfragen
  werden zusammengefasst. Das kann auf langsamer Hardware kurze Wartezeiten erzeugen.
- Video-Jobs werden vor ihrer Verarbeitung gespeichert. Nach einem Neustart werden
  vorläufige Erkennungen unterbrochener Videos rückgängig gemacht. Der Status
  `history_cleanup_pending` weist auf eine wegen Speicherfehlern noch ausstehende
  Bereinigung hin; bei einem späteren Neustart wird sie erneut versucht.
- Begrenzungen für JSON-Konfigurations- und Watchlist-Importe (4 MiB),
  Modell-Dateiuploads, Autocomplete, Zeitreihen und Sammelanalysen schützen
  Arbeitsspeicher und Antwortzeiten.
- Die Kamera gibt den OpenCV-Videograbber nach Möglichkeit im selben Thread
  frei, der ihn verwendet. Ein hängendes natives Kamera-Backend kann trotz
  Zeitlimits einen Neustart verzögern, wird aber nicht von einem anderen Thread
  unkontrolliert freigegeben.

Details, Prüfergebnisse und Grenzen unter
[TIEFENPRUEFUNG_0.12.4_DE.md](TIEFENPRUEFUNG_0.12.4_DE.md).


### Schutz vor Datenverlust und Sicherungen prüfen (0.12.5)

Unter **Diagnose** lässt sich eine lokal gespeicherte PlateVision-ZIP-Sicherung
ohne Wiederherstellung auf erlaubte Dateinamen, SHA-256-Prüfsummen und
Archivintegrität prüfen (`POST /api/system/backup/verify`).
Bei einer beschädigten Hauptdatei wird zunächst eine gültige `.bak` geladen.
Sind **beide** Datendateien unbrauchbar, aktiviert PlateVision einen globalen
Schreibschutz für verändernde API-Anfragen (außer `Stream stoppen`); dadurch
bleiben beschädigte Originaldateien für eine spätere Reparatur erhalten. Die
Oberfläche zeigt eine deutliche Warnung.

**Wiederherstellung:** Home Assistant stoppen, **beide** betroffenen Dateien
sichern, ein externes Home-Assistant-Backup wiederherstellen und danach
PlateVision neu starten. Die neue ZIP-Prüfung spielt keine Daten zurück.
Vor einer Reparatur keine beschädigten Originaldateien manuell überschreiben.

Der CSV-/JSON-Export über die Historienseite liefert jetzt alle gefilterten
Treffer per Streaming, nicht nur die erste API-Seite. JSON enthält standardmäßig
keine Bild-Base64-Inhalte. Bei Hunderttausenden Erkennungen bleibt die
JSON-Historie selbst weiterhin eine Leistungsgrenze; eine SQLite-Migration
wurde für dieses Wartungsrelease bewusst nicht aktiviert.
