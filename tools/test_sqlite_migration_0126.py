#!/usr/bin/env python3
"""PlateVision 0.12.6 SQLite migration, recovery, export and API-compatibility tests."""
from __future__ import annotations
import ast
import copy
import hashlib
import io
import json
import logging
import os
import subprocess
import sqlite3
import sys
import tempfile
import threading
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'platevision/src/app.py'
sys.path.insert(0, str(ROOT / 'platevision/src'))
from history_sqlite import SQLiteHistoryStore, HistoryStorageError
sys.path.insert(0, str(ROOT / 'tools'))
from export_sqlite_to_json import export


def extract(symbols, env):
    tree = ast.parse(APP.read_text(encoding='utf-8'), filename=str(APP))
    nodes = []
    for name in symbols:
        node = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name), None)
        assert node is not None, name
        node = copy.deepcopy(node)
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(APP), 'exec'), env)


def expect_error(fn, error=Exception):
    try:
        fn()
    except error:
        return
    raise AssertionError('Expected exception')


def test_store():
    old = Path('data/history/detections.json')
    old.parent.mkdir(parents=True)
    original = [dict(id='A', plate_text='ZH 123', timestamp='2026-10-01T07:30:00+02:00', confidence=.6),
                dict(id='A', plate_text='BE 45', timestamp='2026-10-01T07:00:00Z'),
                dict(plate_text='AG 99', timestamp='2026-09-30T23:00:00')]
    old.write_text(json.dumps(original, ensure_ascii=False), encoding='utf-8')
    baseline = old.read_bytes()
    store = SQLiteHistoryStore(old, kind='plates')
    repaired = store.initialize(lambda:json.loads(old.read_text()))
    assert len(repaired) == len(original)
    assert len({r['id'] for r in repaired}) == 3
    assert [r['plate_text'] for r in repaired] == [r['plate_text'] for r in original]
    assert old.read_bytes() == baseline
    assert json.loads(store.snapshot.read_text()) == original
    assert store.quick_status()['repaired_ids'] == 2
    assert store.path.exists()
    with sqlite3.connect(store.path) as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        indexes = {row[1] for row in conn.execute('PRAGMA index_list(events)')}
        assert {'idx_events_order', 'idx_events_plate', 'idx_events_timestamp'} <= indexes
        assert conn.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
    assert store.get(repaired[0]['id'])['plate_text'] == 'ZH 123'
    assert store.page(1,1)[0]['plate_text'] == 'BE 45'
    print('  OK: SQLite migration preserves records and original, repairs duplicate/missing IDs, uses indexes/WAL')
    changed = [dict(id='NEW', plate_text='SG 123'), *repaired]
    changed[1]['note'] = 'bearbeitet'
    store.sync(changed)
    assert store.page(2,0)[0]['id'] == 'NEW'
    assert store.get('A')['note'] == 'bearbeitet'
    assert SQLiteHistoryStore(old, kind='plates').initialize(lambda: (_ for _ in ()).throw(AssertionError('should not load JSON'))) == changed
    changed.pop(2)
    store.sync(changed)
    assert store.get(repaired[1]['id']) is None
    store.sync([changed[-1], changed[0], changed[1]])  # rare explicit reorder
    assert [r['id'] for r in store.load()] == [changed[-1]['id'], 'NEW', 'A']
    assert store.quick_status()['count'] == 3
    print('  OK: Ordered writes, indexed pages, updates, deletions, reordering and restart without JSON reload')
    before = store.load()
    with patch.object(SQLiteHistoryStore, '_event_values', side_effect=OSError('disk full')):
        expect_error(lambda: store.sync([dict(id='failure'), *before]), OSError)
    assert store.load() == before
    assert store.quick_status()['count'] == len(before)
    shadow = SQLiteHistoryStore(old, kind='plates')
    assert shadow.initialize(lambda:[]) == before
    store.sync([dict(id='after'), *before])
    expect_error(lambda: shadow.sync([dict(id='stale'), *before]), HistoryStorageError)
    assert store.get('after') is not None and store.get('stale') is None
    print('  OK: Transaction rollback on write failure and rejection of stale parallel writers')
    snapshot = Path('checked.sqlite3')
    store.backup_to(snapshot)
    assert SQLiteHistoryStore.read_snapshot(snapshot) == store.load()
    output = Path('rollback-export')
    Path('data/people').mkdir(parents=True)
    ppath = Path('data/people/history.json')
    ppath.write_text('[{"id":"P1","counted":true}]')
    person = SQLiteHistoryStore(ppath, kind='people')
    person.initialize(lambda:json.loads(ppath.read_text()))
    person.sync([{'id':'P2', 'counted':False}, {'id':'P1','counted':True}])
    result = export(Path('data'), output)
    assert result['counts'] == {'plates': 4, 'people': 2}
    assert (output/'history/detections.json').is_file()
    assert (output/'people/history.json').is_file()
    assert json.loads((output/'history/detections.json').read_text()) == store.load()
    for item in result['files']:
        assert hashlib.sha256((output/item['path']).read_bytes()).hexdigest() == item['sha256']
    expect_error(lambda: export(Path('data'), output), ValueError)
    print('  OK: SQLite online snapshot, verifiable external JSON rollback export; live files unchanged')
    old.write_text('[{"id":"changed after migration"}]')
    expect_error(store.load, HistoryStorageError)
    assert store.path.is_file()  # never silently discard a newer sqlite DB
    old.write_bytes(baseline)
    assert store.quick_status()['ok']
    print('  OK: Detects old-version JSON drift; protects current SQLite data')


def test_parallel_migration():
    """Two independent processes must not overwrite each other's first DB."""
    base = Path('two-process-sim')
    base.mkdir(exist_ok=True)
    legacy = base/'data/history/detections.json'
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps([{'id': f'row-{i}', 'plate_text': f'ZH {i}'} for i in range(750)]), encoding='utf-8')
    code = '''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from history_sqlite import SQLiteHistoryStore
p=Path('data/history/detections.json')
s=SQLiteHistoryStore(p,kind='plates')
rows=s.initialize(lambda:json.loads(p.read_text()))
assert len(rows)==750
assert s.quick_status()['count']==750
'''
    pythonpath = str(ROOT/'platevision/src')
    procs = [subprocess.Popen([sys.executable, '-c', code, pythonpath], cwd=base,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    for proc in procs:
        out, err = proc.communicate(timeout=25)
        assert proc.returncode == 0, (proc.returncode, out, err)
    assert SQLiteHistoryStore(legacy, kind='plates').initialize(lambda:[]) and \
           SQLiteHistoryStore(legacy, kind='plates').quick_status()['count'] == 750
    assert not list(legacy.parent.glob('*.migrating'))
    print('  OK: Two simultaneous independent startups produce one intact SQLite database')


def test_invalid_legacy_guard():
    """Neither a broken source nor its broken .bak may become an empty DB."""
    name = Path('invalid-legacy/history/detections.json')
    name.parent.mkdir(parents=True, exist_ok=True)
    name.write_text('{broken')
    Path(str(name)+'.bak').write_text('[null]')
    guard={}
    env=dict(os=os, Path=Path, logger=logging.getLogger('sqlite-invalid-test'),
             _read_only_storage=guard, SQLiteHistoryStore=SQLiteHistoryStore,
             HistoryStorageError=HistoryStorageError, sqlite3=sqlite3, json=json)
    extract(('_json_clone', '_load_json_with_backup', '_load_history_sqlite'),env)
    store, rows=env['_load_history_sqlite'](str(name),'plates')
    assert store is None and rows==[] and guard
    assert not (name.parent/'plates.sqlite3').exists()
    assert name.read_text()=='{broken'
    print('  OK: Corrupt source + backup activate write protection, without initializing an empty SQLite DB')


def test_corruption():
    path = Path('damaged/history.json')
    path.parent.mkdir(parents=True)
    path.write_text('[{"id":"good"}]')
    store = SQLiteHistoryStore(path, kind='people')
    original = path.read_bytes()
    with patch('history_sqlite._write_once_json', side_effect=OSError('disk full before migration')):
        expect_error(lambda: store.initialize(lambda:json.loads(path.read_text())), OSError)
    assert not store.path.exists() and path.read_bytes() == original
    store.initialize(lambda:json.loads(path.read_text()))
    store.path.write_bytes(b'not an SQLite database')
    expect_error(lambda: SQLiteHistoryStore(path, kind='people').initialize(lambda: []), sqlite3.DatabaseError)
    assert path.read_bytes() == original
    print('  OK: Migration fails safe on disk full and on damaged databases without overwriting old JSON')


def test_managers():
    logger = logging.getLogger('sqlite-migration-test')
    config_data = {'history': {'filter_duplicates': False}, 'plate_recognition': {},
                   'general': {'max_history_entries': 20000},
                   'people': {'history_enabled': True, 'image_history_auto_cleanup_enabled': False,
                              'auto_cleanup_enabled': False, 'max_history_entries':20000}}
    class Config:
        def get(self, *keys):
            cur = config_data
            for key in keys:
                cur = cur.get(key) if isinstance(cur, dict) else None
            return cur
    env = dict(os=os, Path=Path, datetime=datetime, timedelta=timedelta, uuid=uuid,
               threading=threading, json=json, sqlite3=sqlite3, logger=logger, contextmanager=contextmanager,
               SQLiteHistoryStore=SQLiteHistoryStore, HistoryStorageError=HistoryStorageError,
               config_manager=Config(), re=__import__('re'), SequenceMatcher=__import__('difflib').SequenceMatcher,
               _read_only_storage={}, _storage_recovered_from_backup=set(),
               _page_integer=lambda v,d,minimum=0,maximum=200: max(minimum, min(int(v or d), maximum)))
    extract(('_json_clone','_atomic_write_json','_load_json_with_backup','_load_history_sqlite', 'PlateUtils',
             'HistoryManager','PersonHistoryManager'),env)
    cwd=Path.cwd()
    Path('manager-fixture').mkdir(exist_ok=True)
    os.chdir('manager-fixture')
    Path('data/history').mkdir(parents=True,exist_ok=True)
    Path('data/people').mkdir(parents=True,exist_ok=True)
    (Path('data/history')/'detections.json').write_text('[{"id":"A","plate_text":"ZH 1"}]')
    (Path('data/people')/'history.json').write_text('[{"id":"P","counted":true}]')
    plate = env['HistoryManager']()
    person = env['PersonHistoryManager']()
    assert plate._sqlite_store and person._sqlite_store
    assert plate.get_by_id('A')['plate_text'] == 'ZH 1'
    assert person.get_event('P')['counted']
    before = plate.get_all()[0]['id']
    plate.add_entry({'plate_text':'AG 123','confidence':.9},check_duplicate=False)
    assert len(plate._sqlite_store.load()) == 2
    assert plate.get_all()[0]['plate_text'] in ('AG 123','AG123')
    assert plate.delete_entry(before)
    assert len(plate._sqlite_store.load()) == 1
    person.history.insert(0, {'id':'P2','counted':False})
    assert person.save_history()
    assert person.get_all(1)[0]['id'] == 'P2'
    with patch.object(person._sqlite_store, 'sync',side_effect=OSError('simulated full disk')):
        assert not person.save_history()
    assert len(person._sqlite_store.load())==2
    os.chdir(cwd)
    print('  OK: Existing history manager APIs use SQLite for both people and plates and report failed writes')


def test_archive():
    from importlib.util import spec_from_file_location, module_from_spec
    # Isolated extraction of actual backup functions without Flask/ML dependencies.
    import tempfile as tempfile_module
    env = dict(Path=Path, os=os, json=json, zipfile=zipfile, hashlib=hashlib,
               re=__import__('re'), secrets=__import__('secrets'),
               datetime=datetime, SQLiteHistoryStore=SQLiteHistoryStore,
               tempfile=tempfile_module, _BACKUP_METADATA=('data/config.json', 'data/history/detections.json', 'data/people/history.json', 'data/watchlist.json','data/video_jobs.json'),
               _BACKUP_IMAGE_DIRS=('data/plates_detected','data/vehicles_detected','data/people/images'),
               _SQLITE_BACKUP_DATA=(('plates', 'data/history/detections.json','data/history/plates.sqlite3'),
                                    ('people', 'data/people/history.json','data/people/people.sqlite3')),
               APP_VERSION='0.12.6')
    extract(('_backup_inventory','_write_backup_archive','_verify_backup_archive'),env)
    # Existing sqlite DBs from test_store were written under data/. The JSON
    # file still has old entries; the ZIP must contain the *current* DB state.
    buffer = io.BytesIO()
    manifest = env['_write_backup_archive'](buffer)
    assert manifest['storage_backend']=='sqlite-with-json-export'
    buffer.seek(0)
    assert env['_verify_backup_archive'](buffer)['success']
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        expected=json.loads(archive.read('data/history/detections.json'))
        assert len(expected)==4
        assert expected[0]['id']=='after'
        assert 'data/history/plates.sqlite3' in archive.namelist()
        assert 'data/people/people.sqlite3' in archive.namelist()
        assert 'data/history/detections.json.bak' not in archive.namelist()
    print('  OK: Online ZIP contains current SQLite snapshots AND correct JSON exports, not stale pre-migration data')


def main():
    cwd=Path.cwd()
    with tempfile.TemporaryDirectory(prefix='platevision-sqlite-0126-') as folder:
        os.chdir(folder)
        try:
            test_store()
            # Restore original fingerprint before archive is created.
            test_managers()
            test_archive()
            test_corruption()
            test_parallel_migration()
            test_invalid_legacy_guard()
        finally:
            os.chdir(cwd)
    print('PlateVision 0.12.6 SQLite migration tests: 10 Bereiche bestanden')

if __name__=='__main__':
    main()
