#!/usr/bin/env python3
"""Static release validation for PlateVision. Does not load model files."""
from __future__ import annotations
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = '0.10.0'
EXPECTED_MODELS = {
    'yolov8n.pt': '31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95',
    'license_plate_detector.pt': '8ec3b254a6c87610f037a90957462cafa11a9c03224e33a28c6a1d1ac2ac51b0',
    'best.pt': 'a6aead7bf0eccb35bd56731bfaa6ea19a4645a66150d2d0b19dd3fb1b116ef43',
    'last.pt': 'db0f2dfe996d997cd33388003069a4685215c84dbca0f7a6de4b66c9d915f85f',
}
errors=[]
warnings=[]

def require(condition, message):
    if not condition:
        errors.append(message)

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

# Python syntax and duplicate top-level functions.
for path in sorted((ROOT/'platevision/src').glob('*.py')):
    try:
        tree=ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except SyntaxError as exc:
        errors.append(f'Python-Syntax: {path}: {exc}')
        continue
    funcs={}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            funcs.setdefault(node.name,[]).append(node.lineno)
    for name,lines in funcs.items():
        if len(lines)>1:
            errors.append(f'Doppelte Top-Level-Funktion {name} in {path}: {lines}')

# Basic version consistency.
config=(ROOT/'platevision/config.yaml').read_text(encoding='utf-8')
docker=(ROOT/'platevision/Dockerfile').read_text(encoding='utf-8')
app=(ROOT/'platevision/src/app.py').read_text(encoding='utf-8')
require(f'version: "{EXPECTED_VERSION}"' in config, 'config.yaml hat nicht die erwartete Version')
require('ARG BUILD_VERSION=0.10.0' in docker, 'Docker BUILD_VERSION fehlt oder ist falsch')
require('io.hass.version="${BUILD_VERSION}"' in docker, 'Home-Assistant-Version-Label fehlt')
require('io.hass.type="app"' in docker, 'Home-Assistant-Typ-Label fehlt')
require('io.hass.arch="${BUILD_ARCH}"' in docker, 'Home-Assistant-Architektur-Label fehlt')
require('/api/system/live' in docker, 'Docker-Healthcheck verwendet nicht den Liveness-Endpunkt')
require('/api/system/live' in config, 'Home-Assistant-Watchdog verwendet nicht den Liveness-Endpunkt')
require(f"APP_VERSION = '{EXPECTED_VERSION}'" in app, 'app.py hat nicht die erwartete Version')
require("@app.route('/api/system/live')" in app, 'Liveness-Route fehlt')
require('GNU AFFERO GENERAL PUBLIC LICENSE' in (ROOT/'LICENSE').read_text(encoding='utf-8'), 'Root-LICENSE ist nicht AGPL-3.0')
require((ROOT/'third_party_licenses/fast-plate-ocr-MIT.txt').is_file(), 'fast-plate-ocr-Lizenz fehlt im Repository')
require((ROOT/'platevision/third_party_licenses/fast-plate-ocr-MIT.txt').is_file(), 'fast-plate-ocr-Lizenz fehlt im Docker-Build-Kontext')
require(not any('#U00' in path.name for path in ROOT.rglob('*')), 'Fehlerhaft kodierter Dateiname mit #U00 gefunden')

# Docker COPY sources.
for match in re.finditer(r'^COPY\s+([^\s]+)', docker, re.M):
    source=match.group(1).rstrip('/')
    if any(ch in source for ch in '*$'):
        continue
    require((ROOT/'platevision'/source).exists(), f'Docker COPY-Quelle fehlt: platevision/{source}')

# Model fingerprints without loading pickle/torch files.
for name,expected in EXPECTED_MODELS.items():
    path=ROOT/'platevision/src/models'/name
    require(path.is_file(), f'Modell fehlt: {name}')
    if path.is_file():
        require(sha256(path)==expected, f'Modell-Hash stimmt nicht: {name}')

# JSON and optional YAML/Jinja parsing.
json.loads((ROOT/'repository.json').read_text(encoding='utf-8'))
try:
    import yaml
    yaml.safe_load(config)
    yaml.safe_load((ROOT/'repository.yaml').read_text(encoding='utf-8'))
except ImportError:
    warnings.append('PyYAML fehlt; YAML wurde nur textuell geprüft.')
except Exception as exc:
    errors.append(f'YAML-Fehler: {exc}')

try:
    from jinja2 import Environment
    env=Environment()
    for path in sorted((ROOT/'platevision/src/templates').glob('*.html')):
        try:
            env.parse(path.read_text(encoding='utf-8'))
        except Exception as exc:
            errors.append(f'Jinja-Fehler {path.name}: {exc}')
except ImportError:
    warnings.append('Jinja2 fehlt; Templates wurden nicht geparst.')

print(f'PlateVision static validation {EXPECTED_VERSION}')
for item in warnings:
    print('WARN:',item)
for item in errors:
    print('ERROR:',item)
if errors:
    sys.exit(1)
print('OK: alle statischen Prüfungen bestanden')
