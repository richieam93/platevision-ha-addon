# PlateVision – Home Assistant Add-on Repository

PlateVision ist eine Kennzeichen-Erkennungs-Webapp (Flask + Socket.IO + OpenCV + Ultralytics/YOLO + EasyOCR), verpackt als **Home Assistant Add-on**.

Dieses Repository ist als **Add-on Repository** aufgebaut, damit du es direkt in Home Assistant unter  
**Einstellungen → Add-ons → Add-on Store → (⋮) Repositories** hinzufügen kannst.

---

## Inhalte / Features

- Web UI (Flask) zum Live-View und zur Auswertung
- JSON API (z.B. „latest plate“) für Home Assistant REST-Sensoren
- Persistente Daten über `/data` (bleibt bei Updates/Rebuild erhalten)
  - Uploads (Bilder/Videos)
  - History / Detected Plates
  - Modelle (optional)

---

## Voraussetzungen

- Home Assistant OS / Supervised mit Add-on Support
- Architektur: `amd64` (x86_64)
- Netzwerkzugriff von Home Assistant auf das Add-on (lokal ist Standard)
- Hinweis: Der erste Build kann je nach Hardware/RAM länger dauern (Torch/Ultralytics/EasyOCR).

---

## Installation (über Home Assistant UI)

1. In Home Assistant öffnen:
   - **Einstellungen → Add-ons → Add-on Store**
   - oben rechts **(⋮) → Repositories**
2. Repo-URL hinzufügen:
   - `https://github.com/richieam93/platevision-ha-addon
3. Danach **Reload** im Add-on Store.
4. Add-on **PlateVision** installieren und starten.

---

## Ports / Web UI

Standard:
- Intern im Container: `5000`
- Extern am Home Assistant Host: **`8087`**

Web UI:
- `http://<HA-IP>:8087`

---

## Persistente Daten (`/data`)

Home Assistant Add-ons speichern persistent nur unter `/data`.

Dieses Add-on sorgt beim Start automatisch dafür, dass folgende Pfade persistent sind:

- `/app/uploads` → `/data/uploads`
- `/app/models` → `/data/models`
- `/app/data` → `/data/data`

Damit bleiben Uploads/History/Modelle nach Neustarts und Updates erhalten.