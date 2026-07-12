# Sicherheit

## Unterstützte Version

Sicherheitskorrekturen werden für die aktuelle Hauptversion von PlateVision gepflegt.

## Sicherheitsrelevante Hinweise

- PlateVision ist für ein vertrauenswürdiges lokales Netzwerk gedacht. Veröffentliche Port 8087 nicht ungefiltert im Internet.
- Verwende für den Zugriff von ausserhalb des Heimnetzes einen authentifizierten Reverse-Proxy oder VPN.
- Lade nur Machine-Learning-Modelle aus vertrauenswürdigen Quellen. PyTorch-`.pt`-Dateien können beim Laden ausführbaren Python-Code enthalten.
- Prüfe bei fremden Modellen Quelle, Lizenz und SHA-256-Prüfsumme.
- RTSP-Zugangsdaten werden lokal in `/data/data/config.json` gespeichert. Sichere diesen Ordner entsprechend.

## Schwachstellen melden

Bitte veröffentliche Sicherheitsprobleme nicht sofort als öffentliches Issue. Kontaktiere den Maintainer zunächst über das GitHub-Profil `@richieam93` und beschreibe Problem, betroffene Version und mögliche Auswirkungen.
