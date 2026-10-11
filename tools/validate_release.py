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
EXPECTED_VERSION = '0.13.9'
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
require('ingress: true' in config and 'ingress_port: 5000' in config, 'Home-Assistant-Ingress fehlt')
require('ingress_stream: true' in config, 'Stream-Unterstützung für Ingress fehlt')
require('5000/tcp: null' in config, 'Unverschlüsselter Direktport ist standardmäßig offen')
require((ROOT/'tools/test_ingress_0130.py').is_file(), 'Ingress-Regressionsprüfung fehlt')
require(f'ARG BUILD_VERSION={EXPECTED_VERSION}' in docker, 'Docker BUILD_VERSION fehlt oder ist falsch')
require('io.hass.version="${BUILD_VERSION}"' in docker, 'Home-Assistant-Version-Label fehlt')
require('io.hass.type="app"' in docker, 'Home-Assistant-Typ-Label fehlt')
require('io.hass.arch="${BUILD_ARCH}"' in docker, 'Home-Assistant-Architektur-Label fehlt')
require('/api/system/live' in docker, 'Docker-Healthcheck verwendet nicht den Liveness-Endpunkt')
require('/api/system/live' in config, 'Home-Assistant-Watchdog verwendet nicht den Liveness-Endpunkt')
require(f"APP_VERSION = '{EXPECTED_VERSION}'" in app, 'app.py hat nicht die erwartete Version')
require("@app.route('/api/system/live')" in app, 'Liveness-Route fehlt')
require("@app.route('/api/people/sessions')" in app, 'Personen-Sitzungsroute fehlt')
require("@app.route('/people/gallery')" in app, 'Personenbild-Galerieseite fehlt')
require("@app.route('/api/people/images/days')" in app, 'Personenbild-Tagesindex fehlt')
require("@app.route('/api/process/video/<job_id>/cancel', methods=['POST'])" in app, 'Video-Abbruch-Route fehlt')
require("@app.route('/api/system/support-report')" in app, 'Support-Bericht-Route fehlt')
require('def _cleanup_orphan_video_files():' in app, 'Video-Bereinigung nach Neustart fehlt')
require((ROOT/'tools/test_deep_regressions.py').is_file(), 'Vertiefte Regressionstests fehlen')
require((ROOT/'tools/test_resilience_regressions.py').is_file(), 'Persistenz- und Sicherheitstests fehlen')
require("@app.route('/api/system/storage-audit')" in app, 'Speicher-Audit-Route fehlt')
require('def _restore_video_jobs():' in app, 'Video-Jobs-Neustartfunktion fehlt')
require("@app.route('/api/system/backup')" in app, 'ZIP-Datensicherung fehlt')
require("@app.route('/api/system/backup/preview')" in app, 'Datensicherungs-Vorschau fehlt')
require('_model_reload_lock' in app, 'Gesperrter Modell-Neustart fehlt')
require((ROOT/'tools/test_integrity_0123.py').is_file(), 'Erweiterte Konfigurations-/Backup-/RTSP-Tests fehlen')
require((ROOT/'tools/test_hardening_0124.py').is_file(), 'Import-/Video-Recovery-/RTSP-Hardening-Tests fehlen')
require((ROOT/'tools/test_safemode_0125.py').is_file(), 'Neue Datenintegritäts- und Backup-Verifikationstests fehlen')
require((ROOT/'platevision/src/history_sqlite.py').is_file(), 'SQLite-Historienmodul fehlt')
require((ROOT/'tools/test_sqlite_migration_0126.py').is_file(), 'SQLite-Migrationstests fehlen')
require((ROOT/'tools/export_sqlite_to_json.py').is_file(), 'JSON-Rückexport fehlt')
require((ROOT/'tools/test_review_workflow_0128.py').is_file(), 'Tests für manuelle Qualitätsprüfung fehlen')
require((ROOT/'tools/test_navigation_0129.py').is_file(), 'Navigations-Regressionstests fehlen')
require((ROOT/'tools/test_image_quality_0132.py').is_file(), 'Qualitäts-Regressionsprüfung 0.13.2 fehlt')
require((ROOT/'tools/test_stability_0133.py').is_file(), 'Stabilitäts- und Diagnoseprüfungen 0.13.3 fehlen')
require((ROOT/'tools/test_large_history_0134.py').is_file(), 'Großhistorien-Regression 0.13.4 fehlt')
require((ROOT/'tools/test_video_retention_0134.py').is_file(), 'Video-Aufbewahrungstest 0.13.4 fehlt')
require((ROOT/'platevision/src/watchlist_alerts.py').is_file(), 'Watchlist-Alarmdatenbank fehlt')
require((ROOT/'platevision/src/templates/alerts.html').is_file(), 'Alarmzentrale fehlt')
require('homeassistant_api: true' in config, 'HA-Core-API-Zugriffsrecht fehlt')
require((ROOT/'platevision/src/ha_notifications.py').is_file(), 'HA-Notification-Modul fehlt')
require((ROOT/'tools/test_ha_notifications_0136.py').is_file(), 'HA-Outbox-Regressionstests fehlen')
require((ROOT/'tools/test_ha_api_0136.py').is_file(), 'HA-API-Regressionstests fehlen')
require((ROOT/'tools/test_ha_hardening_0137.py').is_file(), 'HA-Zustellhardening-Tests fehlen')
require((ROOT/'tools/test_ha_queue_monitor_0138.py').is_file(), 'Zustellmonitor-Regression 0.13.8 fehlt')
require((ROOT/'tools/test_config_defaults_0138.py').is_file(), 'Konfigurationsrollback-Test 0.13.8 fehlt')
require((ROOT/'tools/test_consent_hardening_0139.py').is_file(), 'Opt-out-/Import-Test 0.13.9 fehlt')
require('def _persist_config_update(' in app and 'def _ha_worker_settings(' in app, 'Sicherer Einstellungswechsel 0.13.9 fehlt')
require('oldest_due_seconds' in (ROOT/'platevision/src/watchlist_alerts.py').read_text(), 'Fällige HA-Queue-Metrik fehlt')
require('HA-Versand' in (ROOT/'platevision/src/templates/diagnostics.html').read_text(), 'HA-Zustellmonitor fehlt')
require('lease_generation' in (ROOT/'platevision/src/watchlist_alerts.py').read_text(),
        'HA-Lease-Generationsschutz fehlt')
require('def recent_deliveries(' in (ROOT/'platevision/src/watchlist_alerts.py').read_text(),
        'Privates HA-Zustellprotokoll fehlt')
require("@app.route('/api/alerts/homeassistant/deliveries')" in app,
        'HA-Zustellprotokoll-API fehlt')
require('id="pvHADeliveryLog"' in (ROOT/'platevision/src/templates/alerts.html').read_text(),
        'Zustellprotokoll in der Alarmzentrale fehlt')
require("@app.route('/api/alerts/homeassistant', methods=['GET', 'POST'])" in app,
        'Sichere HA-Notification-Konfiguration fehlt')
require("@app.route('/api/alerts/homeassistant/test', methods=['POST'])" in app,
        'HA-Verbindungstest fehlt')
require("@app.route('/api/alerts/homeassistant/retry', methods=['POST'])" in app,
        'HA-Zustellwiederholung fehlt')

require((ROOT/'tools/test_watchlist_alerts_0135.py').is_file(), 'Alarm-Regressionspruefung fehlt')
require("@app.route('/api/alerts')" in app, 'Alarm-API fehlt')
require("@app.route('/api/alerts/settings', methods=['GET', 'POST'])" in app,
        'Einstellbare Watchlist-Alarmierung fehlt')
require("_record_watchlist_notification(saved_entry, alert_center)" in app,
        'Benachrichtigung nach History-Commit fehlt')
require('pending_video_alerts' in app, 'Nachgelagerte Videoalarme fehlen')

require((ROOT/'platevision/src/static/js/navigation.js').is_file(), 'Navigations-JavaScript fehlt')
require((ROOT/'platevision/src/static/css/navigation.css').is_file(), 'Navigations-Stylesheet fehlt')
require((ROOT/'tools/qa_browser_review_0128.py').is_file(), 'Browsertests für Qualität und Suchfilter fehlen')
require("@app.route('/api/history/review-summary')" in app, 'API für Prüfstatistik fehlt')
require("@app.route('/api/history/<entry_id>/review', methods=['POST', 'PATCH'])" in app, 'Review-API fehlt')
require('def update_plate_review(' in (ROOT/'platevision/src/history_sqlite.py').read_text(), 'Atomare SQLite-Prüfung fehlt')
require('id="reviewDialog"' in (ROOT/'platevision/src/templates/search.html').read_text(), 'Prüf-Dialog fehlt')
require('id="presetSelect"' in (ROOT/'platevision/src/templates/search.html').read_text(), 'Gespeicherte Suchfilter fehlen')
require('def _load_history_sqlite(' in app, 'Sicherer SQLite-Initialisierungspfad fehlt')
require('sqlite-with-json-export' in app, 'SQLite-Backup-Snapshots fehlen')
require("@app.route('/api/system/backup/verify', methods=['POST'])" in app, 'ZIP-Backup-Prüfung fehlt')
require('_prevent_mutations_in_storage_safe_mode' in app, 'Sicherheits-Schreibschutz fehlt')
require('def _queue_model_reload():' in app, 'Sicherer Modell-Neustart-Scheduler fehlt')
require('def _read_limited_import_json(' in app, 'Begrenzter JSON-Import fehlt')
require('def _store_limited_model_upload(' in app, 'Begrenzter Modell-Upload fehlt')
require('history_cleanup_pending' in app, 'Video-Recovery ohne Retry-Status')

require((ROOT/'platevision/src/templates/people_gallery.html').is_file(), 'Template people_gallery.html fehlt')
require("'image_history_retention_days': 10" in app, 'Standard-Bildaufbewahrung ist nicht 10 Tage')
require("'max_history_entries': 20000" in app, 'Personen-Historienlimit fehlt')
require((ROOT/'CHANGELOG.md').read_text(encoding='utf-8').count(f'## {EXPECTED_VERSION}') == 1, 'CHANGELOG enthält die aktuelle Version mehrfach')
require("@app.route('/api/people/history/bulk-delete', methods=['POST'])" in app, 'Personen-Sammellöschung fehlt')
require("methods=['GET', 'PATCH']" in app and "/api/people/history/<event_id>" in app, 'Personen-Detail-/Review-Route fehlt')
require('richieam93' in (ROOT/'NOTICE.md').read_text(encoding='utf-8'), 'Projektpseudonym fehlt in NOTICE.md')
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
