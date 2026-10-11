#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Export current PlateVision SQLite histories to a verifiable offline rollback set.

This tool NEVER overwrites production databases or JSON files. To downgrade:
stop PlateVision, create a full Home Assistant backup, export the latest data,
then restore the exported JSONs and move the DB files offline *manually*.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platevision/src'))
from history_sqlite import SQLiteHistoryStore

SOURCES = (
    ('plates', 'history/detections.json', 'history/plates.sqlite3'),
    ('people', 'people/history.json', 'people/people.sqlite3'),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export(data_root, output):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Zielordner existiert bereits; vorhandene Sicherungen niemals überschreiben')
    if output == data_root or data_root in output.parents:
        raise ValueError('Rückexport außerhalb des produktiven Datenverzeichnisses ablegen')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.platevision-rollback-', dir=output.parent))
    manifest = {'format': 'platevision-sqlite-rollback', 'schema': 1,
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'warning': 'KEINE automatische Wiederherstellung. Anwendung vorher stoppen.',
                'files': [], 'counts': {}}
    try:
        for kind, json_rel, db_rel in SOURCES:
            source = data_root / db_rel
            if not source.is_file():
                raise ValueError(f'SQLite-Quelle fehlt: {source}')
            store = SQLiteHistoryStore(data_root / json_rel, kind=kind)
            db_copy = stage / db_rel
            db_copy.parent.mkdir(parents=True, exist_ok=True)
            store.backup_to(db_copy)
            rows = SQLiteHistoryStore.read_snapshot(db_copy)
            manifest['counts'][kind] = len(rows)
            json_out = stage / json_rel
            json_out.parent.mkdir(parents=True, exist_ok=True)
            with json_out.open('x', encoding='utf-8') as handle:
                json.dump(rows, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            # Re-read the exported JSON and compare it with the snapshotted DB.
            if json.loads(json_out.read_text(encoding='utf-8')) != rows:
                raise ValueError('Kontrollvergleich des JSON-Rückexports fehlgeschlagen')
            for rel, exported in ((json_rel, json_out), (db_rel, db_copy)):
                manifest['files'].append({'path': rel, 'bytes': exported.stat().st_size,
                                          'sha256': sha256(exported)})
        (stage / 'ROLLBACK_ANLEITUNG.txt').write_text(
            'PlateVision SQLite -> JSON (manueller Rueckfall)\n'
            '1. Zuerst vollstaendiges Home-Assistant-Backup erstellen und PlateVision stoppen.\n'
            '2. Alte JSON-Dateien UND beide SQLite-Dateien mit -wal/-shm separat sichern.\n'
            '3. Die beiden exportierten JSON-Dateien in ihre vorherigen Datenpfade kopieren.\n'
            '4. Beide SQLite-Datenbanken NUR BEI GESTOPPTEM DIENST aus dem aktiven Datenpfad entfernen,\n'
            '   einschliesslich eventueller -wal/-shm-Dateien; separat aufbewahren.\n'
            '5. PLATEVISION_HISTORY_BACKEND=json setzen fuer 0.12.6 oder vorherige Version starten.\n'
            '6. Erst nach Kontrolle der Datensaetze wieder automatische Erkennung aktivieren.\n'
            'Achtung: Dieser Export enthaelt personenbezogene Daten und ist nicht verschluesselt.\n',
            encoding='utf-8')
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data', help='Persistenter PlateVision data/-Ordner')
    parser.add_argument('--output-dir', required=True, help='Neuer Zielordner außerhalb des produktiven Datenpfads')
    args = parser.parse_args()
    try:
        result = export(args.data_root, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f'FEHLER: {exc}', file=sys.stderr)
        return 2
    print('SQLite-Rückexport erfolgreich; keine Live-Dateien verändert')
    print(json.dumps({'counts': result['counts'], 'files': len(result['files'])}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
