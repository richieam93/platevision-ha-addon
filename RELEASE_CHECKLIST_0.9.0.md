# PlateVision 0.9.0 veröffentlichen

Diese Checkliste gilt für den Wechsel der öffentlichen Hauptversion von 0.8.30
(MIT) auf 0.9.0 (AGPL-3.0-only).

## 1. Bestehendes Repository sichern

```bash
git clone --mirror https://github.com/richieam93/platevision-ha-addon.git platevision-backup.git
```

## 2. Letzten alten Stand markieren

Im bestehenden lokalen Git-Repository vor dem Ersetzen der Dateien:

```bash
git tag -a v0.8.30-mit -m "Letzter veröffentlichter PlateVision-Stand unter MIT"
git push origin v0.8.30-mit
```

Falls dieser Tag bereits existiert, nicht nochmals erstellen.

## 3. Dateien aus dieser ZIP übernehmen

Den Inhalt dieser Version über das lokale Repository kopieren. Danach:

```bash
git status
git diff --check
git diff
```

## 4. Funktion prüfen

- Add-on bauen und starten
- Dashboard, RTSP-Stream und Test-Upload prüfen
- Kennzeichen- und Personenerkennung prüfen
- `/legal` öffnen
- `/api/system/licenses` öffnen
- Bei allen vier Modellen muss `hash_matches` den Wert `true` zeigen

## 5. Neue Version veröffentlichen

```bash
git add .
git commit -m "release: PlateVision 0.9.0 with AGPL licensing and model provenance"
git push origin main
git tag -a v0.9.0 -m "PlateVision 0.9.0"
git push origin v0.9.0
```

## 6. Forum-Beitrag

Der bestehende Link darf auf das Repository zeigen. Ergänze im Forum bei
Gelegenheit nur den Hinweis:

> PlateVision wird ab Version 0.9.0 unter AGPL-3.0-only veröffentlicht. Die
> verwendeten Modellquellen und Lizenztexte sind im Repository dokumentiert.

## Hinweis

Die Dateien dokumentieren den nachvollziehbaren aktuellen Stand. Sie ersetzen
keine individuelle Rechtsberatung, insbesondere nicht für eine spätere
proprietäre oder geschlossene kommerzielle Version.
