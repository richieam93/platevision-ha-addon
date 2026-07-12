

# PlateVision – Home Assistant Add-on

🚗 **Automatic License Plate Recognition for Home Assistant**

🚗 **Automatische Kennzeichenerkennung für Home Assistant**

> ⚠️ **Current Status:** German UI only | CPU only (GPU coming soon)

> ⚠️ **Aktueller Status:** Nur deutsche Oberfläche | Nur CPU (GPU kommt bald)

[English](#-english) | [Deutsch](#-deutsch)

---

# 🇬🇧 English

PlateVision is a license plate recognition webapp (Flask + Socket.IO + OpenCV + Ultralytics/YOLO + EasyOCR), packaged as a **Home Assistant Add-on**.

**100% local** – no cloud, no subscription, your data stays with you.

---

## 🎯 What does PlateVision do?

PlateVision connects to your **RTSP camera** and automatically recognizes license plates in real-time.

| Use Cases | |
|-----------|---|
| 🚘 Monitor driveway | Who comes and goes? |
| 🚗 Manage parking | Automatic detection |
| 🔓 Open garage door | When known plate is detected |
| 📊 Statistics | Which vehicles, how often, when? |

---

## ✨ Features

### 📹 RTSP Camera Integration (Main Feature)
- **Live connection** to any RTSP-capable camera
- **Automatic analysis** – runs in background
- **Real-time recognition** of license plates
- **Camera crop** adjustable (saves computing power)

### 🤖 Recognition
- **YOLO-based** – runs 100% locally on CPU or GPU
- **EasyOCR** for text recognition
- **Vehicle type detection** (car, truck, motorcycle, etc.)
- **CPU/GPU selectable** in web interface

### 🖥️ Web UI
- **Live view** of camera
- **Detected plates** with timestamp
- **Statistics & history**
- **Settings** directly in browser

### 🔌 Home Assistant Integration
- **JSON API** for REST sensors
- **Automations** possible (e.g., open gate)
- **Persistent data** – survives updates


### 🧠 Pro-Erkennung & Analyse (v0.7.2)
- Verbesserte Kennzeichen-Normalisierung mit OCR-Korrekturen für typische Verwechslungen wie O/0, I/1, S/5
- Länder-/Format-Hinweis für CH/DE/AT sowie automatische Format-Erkennung
- Erweiterte Suche mit Fuzzy-Suche, Regex, Datum, Konfidenz, Quelle, Fahrzeugtyp, Farbe und eindeutigen Kennzeichen
- CSV/JSON-Export der Historie inklusive Filter
- Watchlist für bekannte oder wichtige Kennzeichen
- Neues Dashboard Pro mit Live-Kennzahlen, Charts, Top-Kennzeichen und Schnellaktionen
- Neue Pro-Einstellungen für OCR-Logik, Suche, Dashboard-Refresh und Benachrichtigungen

### 🧪 Test Mode
- **Upload function** for images/videos
- Perfect for **testing and adjusting** recognition
- **Optimize** crop and settings before going live

---

## 📋 Requirements

| Requirement | Details |
|-------------|---------|
| **Home Assistant** | OS / Supervised with Add-on Support |
| **Architecture** | amd64 (x86_64) |
| **Camera** | RTSP-capable (most IP cameras) |
| **Hardware** | Min. Intel i3 or comparable recommended |

> ⚠️ **Note:** First build may take longer depending on hardware/RAM (Torch/Ultralytics/EasyOCR).

---

## 📸 Screenshots

| Dashboard | RTSP Stream | Erkennung |
|-----------|-------------|-----------|
| ![Dashboard](Bilder/dashboard.JPG) | ![RTSP](Bilder/rtsp.JPG) | ![Erkennung](Bilder/Letzte%20Erkennung.JPG) |

| Historie | Einstellungen | Test-Modus |
|----------|---------------|------------|
| ![Historie](Bilder/Historie.JPG) | ![Einstellungen](Bilder/einstellungen.JPG) | ![Test](Bilder/test.JPG) |


## 🚀 Installation

1. In Home Assistant open:
   - **Settings → Add-ons → Add-on Store**
2. Top right **(⋮) → Repositories**
3. Add repo URL: https://github.com/richieam93/platevision-ha-addon
4. **Reload** Add-on Store
5. Install and start **PlateVision**

---

## ⚙️ Setup

### Connect RTSP Camera

1. Open Web UI: http://<HA-IP>:8087
2. Go to **Settings**
3. Enter your RTSP URL, e.g.: rtsp://user:password@192.168.1.100:554/stream
4. **Adjust crop** – only analyze relevant area
5. Select **CPU/GPU**
6. Save – done! 🎉

---

## 🌐 Ports / Web UI

| Internal (Container) | External (Host) |
|---------------------|-----------------|
| 5000 | 8087 |

**Web UI:** http://<HA-IP>:8087

---

## 🏠 Home Assistant Integration

### Ready-to-use Examples / Fertige Beispiele

In folder **[/examples](https://github.com/richieam93/platevision-ha-addon/tree/main/examples)** you'll find:

Im Ordner **[/examples](https://github.com/richieam93/platevision-ha-addon/tree/main/examples)** findest du:

#### ⚙️ Configuration / Konfiguration

| File / Datei | Description / Beschreibung |
|--------------|---------------------------|
| 📄 [configuration.yaml](examples/configuration.yaml) | Basic configuration / Grundkonfiguration |
| 📄 [Zusammenfassung der Entity-IDs.txt](examples/Zusammenfassung%20der%20Entity-IDs.txt) | Entity ID overview / Entity-ID Übersicht |

#### 🤖 Automations / Automatisierungen

| File / Datei | Description / Beschreibung |
|--------------|---------------------------|
| 📄 [automations_Bekanntes Kennzeichen.yaml](examples/automations_%20Bekanntes%20Kennzeichen.yaml) | Trigger on known plate / Bei bekanntem Kennzeichen |
| 📄 [automations_Neue Erkennung.yaml](examples/automations_Neue%20Erkennung.yaml) | Trigger on new detection / Bei neuer Erkennung |

#### 🖼️ Lovelace Cards / Dashboard-Karten

| File / Datei | Description / Beschreibung |
|--------------|---------------------------|
| 📄 [lovelace_card-Dashboard Karte.yaml](examples/lovelace_card-%20Dashboard%20Karte.yaml) | Standard dashboard card / Standard Dashboard-Karte |
| 📄 [lovelace_mini_card_Kompakte_Karte.yaml](examples/lovelace_mini_card_Kompakte_Karte.yaml) | Compact card / Kompakte Karte |
| 📄 [lovelace_picture_elements_Erweiterte Karte.yaml](examples/lovelace_picture_elements_Erweiterte%20Karte.yaml) | Advanced card / Erweiterte Karte |

#### 📜 Scripts

| File / Datei | Description / Beschreibung |
|--------------|---------------------------|
| 📄 [scripts_Nützliche Scripts.yaml](examples/scripts_N%C3%BCtzliche%20Scripts.yaml) | Useful scripts / Nützliche Scripts |

#### 📡 API

| File / Datei | Description / Beschreibung |
|--------------|---------------------------|
| 📄 [Alle API-Endpunkte - Vollständige Übersicht.txt](examples/Alle%20API-Endpunkte%20-%20Vollst%C3%A4ndige%20%C3%9Cbersicht.txt) | All API endpoints / Alle API-Endpunkte |
---

## 💾 Persistent Data

| App Path | Persistent |
|----------|------------|
| /app/uploads | /data/uploads |
| /app/models | /data/models |
| /app/data | /data/data |

---

# 🇩🇪 Deutsch

PlateVision ist eine Kennzeichen-Erkennungs-Webapp (Flask + Socket.IO + OpenCV + Ultralytics/YOLO + EasyOCR), verpackt als **Home Assistant Add-on**.

**100% lokal** – keine Cloud, kein Abo, deine Daten bleiben bei dir.

---

## 🎯 Was macht PlateVision?

PlateVision verbindet sich mit deiner **RTSP-Kamera** und erkennt automatisch Nummernschilder in Echtzeit.

| Anwendungsbeispiele | |
|---------------------|---|
| 🚘 Einfahrt überwachen | Wer kommt und geht? |
| 🚗 Parkplatz verwalten | Automatische Erfassung |
| 🔓 Garagentor öffnen | Wenn bekanntes Kennzeichen erkannt |
| 📊 Statistiken | Welche Fahrzeuge, wie oft, wann? |

---

## ✨ Features

### 📹 RTSP Kamera-Integration (Hauptfunktion)
- **Live-Verbindung** zu jeder RTSP-fähigen Kamera
- **Automatische Analyse** – läuft im Hintergrund
- **Echtzeit-Erkennung** von Nummernschildern
- **Kamera-Zuschnitt** einstellbar (spart Rechenleistung)

### 🤖 Erkennung
- **YOLO-basiert** – läuft 100% lokal auf CPU oder GPU
- **EasyOCR** für Texterkennung
- **Fahrzeugtyp-Erkennung** (Auto, LKW, Motorrad, etc.)
- **CPU/GPU wählbar** im Webinterface

### 🖥️ Web UI
- **Live-View** der Kamera
- **Erkannte Kennzeichen** mit Zeitstempel
- **Statistiken & History**
- **Einstellungen** direkt im Browser

### 🔌 Home Assistant Integration
- **JSON API** für REST-Sensoren
- **Automatisierungen** möglich (z.B. Tor öffnen)
- **Persistente Daten** – bleibt bei Updates erhalten

### 🧪 Test-Modus
- **Upload-Funktion** für Bilder/Videos
- Perfekt zum **Testen und Einstellen** der Erkennung
- **Optimiere** Zuschnitt und Einstellungen bevor du live gehst

---

## 📋 Voraussetzungen

| Anforderung | Details |
|-------------|---------|
| **Home Assistant** | OS / Supervised mit Add-on Support |
| **Architektur** | amd64 (x86_64) |
| **Kamera** | RTSP-fähig (die meisten IP-Kameras) |
| **Hardware** | Min. Intel i3 oder vergleichbar empfohlen |

> ⚠️ **Hinweis:** Der erste Build kann je nach Hardware/RAM länger dauern (Torch/Ultralytics/EasyOCR).

---

## 📸 Screenshots

| Dashboard | RTSP Stream | Erkennung |
|-----------|-------------|-----------|
| ![Dashboard](Bilder/dashboard.JPG) | ![RTSP](Bilder/rtsp.JPG) | ![Erkennung](Bilder/Letzte%20Erkennung.JPG) |

| Historie | Einstellungen | Test-Modus |
|----------|---------------|------------|
| ![Historie](Bilder/Historie.JPG) | ![Einstellungen](Bilder/einstellungen.JPG) | ![Test](Bilder/test.JPG) |

---


## 🚀 Installation

1. In Home Assistant öffnen:
   - **Einstellungen → Add-ons → Add-on Store**
2. Oben rechts **(⋮) → Repositories**
3. Repo-URL hinzufügen: https://github.com/richieam93/platevision-ha-addon
4. **Reload** im Add-on Store
5. Add-on **PlateVision** installieren und starten

---

## ⚙️ Einrichtung

### RTSP-Kamera verbinden

1. Öffne das Web UI: http://<HA-IP>:8087
2. Gehe zu **Einstellungen**
3. Trage deine RTSP-URL ein, z.B.: rtsp://benutzer:passwort@192.168.1.100:554/stream
4. **Zuschnitt anpassen** – nur den relevanten Bereich analysieren
5. **CPU/GPU** auswählen
6. Speichern – fertig! 🎉

---

## 🌐 Ports / Web UI

| Intern (Container) | Extern (Host) |
|--------------------|---------------|
| 5000 | 8087 |

**Web UI:** http://<HA-IP>:8087

---

## 💾 Persistente Daten

| App-Pfad | Persistent |
|----------|------------|
| /app/uploads | /data/uploads |
| /app/models | /data/models |
| /app/data | /data/data |

---

## ☕ Support this Project / Unterstütze dieses Projekt

This project is **100% free and open source**. Dieses Projekt ist **100% gratis und Open Source**.

Over ** several hours of development** went into PlateVision. Über **Mehre Stunden Entwicklung** stecken in PlateVision.

If it helps you, I'd appreciate a coffee. Wenn es dir hilft, freue ich mich über einen Kaffee:

<a href="https://www.buymeacoffee.com/geartec" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

---

## 📝 Feedback & Support

- 🐛 **Issues:** [GitHub Issues](https://github.com/richieam93/platevision-ha-addon/issues)
- 💬 **Questions / Fragen:** Just open an issue!

---

## 📜 License / Lizenz

PlateVision wird ab Version **0.9.0** unter der **GNU Affero General Public
License Version 3** (`AGPL-3.0-only`) veröffentlicht. Der vollständige Text
befindet sich in [`LICENSE`](LICENSE).

Frühere Versionen, die bereits unter MIT veröffentlicht wurden, bleiben für
die damals erhaltenen Kopien unter MIT nutzbar. Der Wechsel gilt nicht
rückwirkend.

PlateVision verwendet Ultralytics YOLO und mitgelieferte Modelle aus weiteren
Open-Source-Projekten. Die Quellen, SHA-256-Prüfsummen und Lizenzhinweise sind
hier dokumentiert:

- [`NOTICE.md`](NOTICE.md)
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
- [`MODEL_PROVENANCE.md`](MODEL_PROVENANCE.md)
- [`third_party_licenses/`](third_party_licenses/)
- [`legacy_licenses/PlateVision-MIT-through-0.8.30.txt`](legacy_licenses/PlateVision-MIT-through-0.8.30.txt)
- [`RELEASE_CHECKLIST_0.9.0.md`](RELEASE_CHECKLIST_0.9.0.md)

Die AGPL erlaubt Nutzung, Änderung, Weitergabe und auch entgeltliche
Weitergabe. Dabei müssen die anwendbaren Lizenzbedingungen, insbesondere die
Bereitstellung des entsprechenden Quellcodes, eingehalten werden.

---

Made with ❤️ in Switzerland 🇨🇭 | Entwickelt mit ❤️ in der Schweiz 🇨🇭



- Neue Statistik-Seite `/statistics` mit Autos pro Tag, eindeutigen Fahrzeugen und Roh-Erkennungen.
- Wiederkehrer-Analyse: zeigt, wie oft dasselbe Kennzeichen erkannt wurde.
- Besuchs-/Session-Logik: Mehrfach-Erkennungen eines Autos werden zu einem Besuch gruppiert.
- Kommen/Gehen-Auswertung: explizite Richtung wird genutzt; ohne Richtung wird „gegangen“ nach Timeout angenommen.
- CSV-/JSON-Export für Tagesstatistik und Top-Kennzeichen.
- Neue Traffic-Einstellungen für Besuchslücke, Timeout, Mindestkonfidenz und Bewegungsachse.


- Neue Personenanalyse mit eigenem Menüpunkt `/people`.
- Aktivierbare Personenerkennung mit YOLOv8 COCO-Personenklasse oder eigenem HumanDetection-Modell.
- Eigene Personen-Historie, Tagesstatistik, Stundenverteilung, Richtung und Export.
- Virtuelle Zähllinie, Tracking, Zählstrategie und Testumgebung über die Einstellungen.
- Bestehende Original-Einstellungen bleiben erhalten und werden nur erweitert.


- Personenmodell-Auswahl direkt in den Einstellungen mit Modellscan für `models/`.
- Neue Endpunkte für Modellliste, Modellauswahl, Config-Validierung, Personen-Anwesenheit, Cleanup und Simulation.
- Erweiterte Personenerkennung mit Bildgröße, IoU, Flächen-/Seitenverhältnisfilter, optionaler Personen-Zone, Track-Alter und Debounce.
- Config-Merge bewahrt unbekannte/alte Custom-Keys, damit Updates keine Einstellungen verstecken.


- Neue Test-Funktion unter `/test`: Foto hochladen und ausschließlich die Personenanalyse prüfen.
- Der Personen-Test zeigt erkannte Personen, gezählte Personen, Track-ID, Ereignistyp, Richtung und ein annotiertes Ergebnisbild.
- Die Personenerkennung kann für den Einzeltest temporär aktiviert werden, ohne die gespeicherte Live-Einstellung zu ändern.
- Neuer Einstellungs-Tab **Modelle hochladen** für `.pt`, `.onnx` und `.engine`.
- Hochgeladene Modelle werden bevorzugt unter `/data/models` gespeichert, damit sie Add-on-Updates überstehen.
- Personenmodelle können nach dem Upload automatisch ausgewählt und optional direkt neu geladen werden.

## Version 0.9.0

- Projektlizenz für neue Veröffentlichungen auf `AGPL-3.0-only` umgestellt.
- Herkunft, Lizenzen und SHA-256-Prüfsummen aller vier mitgelieferten Modelle dokumentiert.
- Originale MIT-, GPL-3.0- und AGPL-3.0-Lizenztexte werden mit Repository und Docker-Image ausgeliefert.
- Neue Webseite `/legal` und API `/api/system/licenses` prüfen die tatsächlich installierten Modell-Dateien.
- Versionsanzeigen im Webinterface auf gemeinsame App-Metadaten umgestellt.
- Öffentlich bekannten festen Flask-Schlüssel durch persistenten Zufallsschlüssel ersetzt.
- Abhängigkeiten mit sicheren Major-Version-Obergrenzen versehen, um unerwartete Breaking Changes zu vermeiden.

## Version 0.8.24

- Neue Paketversion auf Basis der funktionierenden 0.8.22 CPU-/RTSP-Korrektur.
- Straßenbereich/Polygon bleibt als RTSP-Analysebereich aktiv, mit CPU-Gate und gepuffertem ROI-Crop.
- Test & Upload, Modelle, Personen-Crops und neue OCR-/YOLO-Pipeline bleiben enthalten.

- Webinterface-Stabilisierung: Settings-Formulare werden nicht mehr durch fehlende/verschobene UI-Elemente blockiert.
- Sichtbare UI-Fehlermeldungen im Browser, damit stille JavaScript-Fehler sofort auffallen.
- Personen-Test und Personen-Fotoanalyse speichern erkannte Personen jetzt zuverlässig als Crop-Bilder in der Personenanalyse.
- Personen-Bild-History zeigt standardmäßig alle gespeicherten Personen-Crops, nicht nur gezählte Linienereignisse.
- Historie und Suche: „Nur einzigartige“ funktioniert mit normalisierten/fuzzy Kennzeichen; doppelte Kennzeichen werden in der Anzeige zusammengefasst.
- RTSP-Quelle und Fahrzeugtyp-Filter wurden an die tatsächlich gespeicherten Werte angepasst.
- Fahrzeug- und Kennzeichen-YOLO verwenden die im Testmodus gespeicherten Modell-/ImgSz-/Confidence-Werte konsistent.


## Version 0.8.26

- Personen-Zähllinie auf der Personenanalyse-Seite wieder direkt einstellbar.
- Linienposition, Achse, Richtung und ROI-Bezug können gespeichert und sofort für RTSP angewendet werden.
- Test & Upload synchronisiert die Personen-Linie wieder sauber mit dem RTSP-Profil.


## Version 0.8.30

- Dashboard komplett vereinfacht: letztes Auto, letzte Person und Tageszahlen ohne Modell-Details.
- RTSP/Webstream-Autostart nach Docker-/Home-Assistant-Neustart als Einstellung ergänzt.
- Beispiele im Ordner `examples/` aktualisiert und veraltete Beispiele entfernt.


### Version 0.8.30

Diese Version erweitert Einstellungen und Test & Upload. Zusätzliche RTSP-Leistungswerte, OCR-/YOLO-Filter, Fahrzeugfarbe, Personenfilter und Verkehrslogik können jetzt im Webinterface gesetzt und für den RTSP-Stream gespeichert werden.
