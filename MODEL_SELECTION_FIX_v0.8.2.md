# PlateVision v0.8.2 - Personen-Modell-Auswahl Fix

## Behoben

- `best.pt` und `last.pt` werden als Personenmodelle erkannt, wenn sie im Modellordner liegen.
- Modellscan sucht nun in mehreren sinnvollen Orten: `models/`, `/data/models`, `/app/models`, `platevision/src/models` sowie im Custom-Modellordner.
- Repo-Pfade wie `platevision/src/models/best.pt` werden zur Laufzeit auf `/app/models/best.pt` beziehungsweise `/data/models/best.pt` abgebildet.
- Beim Add-on-Start werden neue Modelle aus `/app/models` nach `/data/models` synchronisiert, ohne bestehende Modelle zu überschreiben. Dadurch erscheinen nach Updates neu hinzugefügte Modelle auch dann, wenn `/data/models` bereits existiert.
- Das Dropdown zeigt nur vorhandene Modelle auswählbar an und weist zusätzlich auf erwartete, aber fehlende Standardmodelle hin.

## Wichtig

Die Modell-Binärdateien selbst werden nicht künstlich erzeugt. Damit `best.pt` und `last.pt` angezeigt werden, müssen sie im Add-on unter `platevision/src/models/` oder persistent unter `/data/models` vorhanden sein.
