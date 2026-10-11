#!/usr/bin/env python3
"""Fault-injection and restart tests using REAL PlateVision source definitions.

Fast to run, no Flask, camera, GPU or ML weights needed. Standard library only.
"""
from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / 'platevision/src/app.py'
LOG = logging.getLogger('pv-resilience-test')

class Config:
    def __init__(self):
        self.data = {
            'search': {'default_limit': 30, 'allow_regex_search': True, 'enable_fuzzy_search': False},
            'general': {'max_history_entries': 1000},
            'history': {'filter_duplicates': False, 'duplicate_timeout': 60},
            'plate_recognition': {'min_length': 3, 'max_length': 12},
            'people': {'history_enabled': True, 'person_recount_block_enabled': False,
                       'image_history_enabled': False, 'image_history_auto_cleanup_enabled': False,
                       'max_history_entries': 3000},
        }
    def get(self, *keys):
        item = self.data
        for key in keys:
            item = item.get(key) if isinstance(item, dict) else None
        return item


def load_symbols(names, namespace):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'), filename=str(SOURCE))
    nodes = []
    for name in names:
        found = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                      and node.name == name), None)
        assert found is not None, name
        found = copy.deepcopy(found)
        if isinstance(found, ast.FunctionDef):
            found.decorator_list = []  # Do not need a Flask object to test route logic.
        nodes.append(found)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), 'exec'), namespace)
    return namespace


def build_ns(config):
    ns = dict(Path=Path, json=json, logging=logging, logger=LOG, os=os,
              hashlib=hashlib, shutil=shutil, threading=threading, uuid=uuid, time=time,
              re=re, Counter=Counter, defaultdict=defaultdict, datetime=datetime,
              timedelta=timedelta, SequenceMatcher=SequenceMatcher,
              config_manager=config)
    load_symbols(('_page_integer', '_safe_numeric', '_json_clone', '_atomic_write_json',
                  '_load_json_with_backup', '_compile_plate_search_regex',
                  'PlateUtils', 'HistoryManager', 'WatchlistManager',
                  '_history_api_entries', '_invalid_search_regex'), ns)
    return ns


def check_history_and_failures(ns):
    history = ns['HistoryManager']()
    old_writer = ns['_atomic_write_json']
    inserted = history.add_entry({'plate_text': 'ZH 12345', 'confidence': .92})
    assert inserted['id'] and history.get_by_id(inserted['id'])
    recorded = Path(history.HISTORY_FILE).read_bytes()
    def broken_writer(*args, **kwargs):
        raise OSError('simulated disk full')
    ns['_atomic_write_json'] = broken_writer
    try:
        for method, args in [(history.add_entry, ({'plate_text': 'BE 87654'},)),
                             (history.delete_entry, (inserted['id'],)),
                             (history.clear_history, ())]:
            try:
                method(*args)
                raise AssertionError('Failed write incorrectly reported success')
            except OSError as exc:
                assert 'konnte nicht gespeichert' in str(exc)
            assert len(history.history) == 1
            assert history.get_by_id(inserted['id'])
            assert Path(history.HISTORY_FILE).read_bytes() == recorded
    finally:
        ns['_atomic_write_json'] = old_writer

    for index in range(1250):
        history.history.append({'id': str(index), 'plate_text': 'BE-9876',
                                'timestamp': datetime.now().isoformat(), 'confidence': 'n/a' if index % 3 == 0 else .7})
    assert history.search_advanced({'offset': 1000, 'limit': 300})['total'] == 1251
    assert len(history.search_advanced({'offset': 1000, 'limit': 300})['entries']) == 251
    history.history.insert(0, {'id': 'future', 'plate_text': 'GE 55',
                              'timestamp': (datetime.now() + timedelta(days=7)).isoformat(),
                              'confidence': float('nan')})
    history.history.insert(0, {'id': 'aware', 'plate_text': 'OW 1234',
                              'timestamp': datetime.now().astimezone().isoformat(),
                              'confidence': 'bad-confidence'})
    stats = history.get_statistics()
    assert stats['total_detections'] == 1253
    assert stats['last_hour_detections'] == 1252, stats['last_hour_detections']
    assert 0 <= stats['average_confidence'] <= 1
    assert stats['daily'] and stats['hourly']

    # Reject a catastrophic backtracking pattern BEFORE compiling it.
    compile_rx = ns['_compile_plate_search_regex']
    for malicious in ['(A+)+$', '((A|AA)+)+$', '(?=A).*', r'(A)\1', 'A{3000}', 'A{1,30}', 'A'*100]:
        assert compile_rx(malicious) is None, malicious
    assert compile_rx('^ZH[ -]?123[0-9]{1,3}$') is not None
    assert ns['_invalid_search_regex']({'regex': True, 'q': '(a+)+$'})
    assert history.search_all({'regex': True, 'q': '(a+)+$'}) == []
    assert len(history.search_all({'regex': True, 'q': '^GE 55$'})) == 1
    assert not history.search_all({'q': 'GE', 'fuzzy': False, 'source': 'no-such-source'})

    source = [{'id': 'one', 'plate_text': 'ZH 11', 'plate_image': 'AAAA', 'vehicle_image': 'BBBB',
               'full_frame': 'CCCC', 'watchlist_match': {'label': '<img onerror=1>'}}]
    compact = ns['_history_api_entries'](source, {'include_images': False})
    assert compact[0]['has_plate_image'] and compact[0]['has_vehicle_image']
    assert all(field not in compact[0] for field in ('plate_image', 'vehicle_image', 'full_frame'))
    assert source[0]['plate_image'] == 'AAAA'  # No accidental mutation.
    assert ns['_history_api_entries'](source, {'include_images': True}) is source
    # Multi-writer simulation using the actual RLock and on-disk atomic writer.
    history.history = []
    errors = []
    def write_batch(batch):
        try:
            for index in range(8):
                history.add_entry({'plate_text': f'BE{batch}{index:04d}', 'confidence': .8}, check_duplicate=False)
        except Exception as exc:
            errors.append(exc)
    writers = [threading.Thread(target=write_batch, args=(i,)) for i in range(4)]
    for thread in writers: thread.start()
    for thread in writers: thread.join(timeout=10)
    assert not errors, errors
    assert all(not thread.is_alive() for thread in writers)
    assert len(history.history) == 32
    assert len(json.loads(Path(history.HISTORY_FILE).read_text())) == 32
    print('  OK: Kennzeichen-Statistik, Regex-Kontrolle, Pagination, Rollback und parallele Schreiber')


def check_watchlist(ns):
    watch = ns['WatchlistManager']()
    item = watch.add('ZH 1234', label='Original')
    original = json.loads(Path(watch.WATCHLIST_FILE).read_text())
    writer = ns['_atomic_write_json']
    ns['_atomic_write_json'] = lambda *_: (_ for _ in ()).throw(OSError('disk failure'))
    try:
        for method, args in ((watch.add, ('ZH 1234', 'Neues Label')),
                             (watch.add, ('AG 333',)),
                             (watch.delete, (item['id'],))):
            try:
                method(*args)
                raise AssertionError('Watchlist saved despite simulated failure')
            except OSError:
                pass
            assert watch.list()[0]['label'] == 'Original'
            assert len(watch.list()) == 1
            assert json.loads(Path(watch.WATCHLIST_FILE).read_text()) == original
    finally:
        ns['_atomic_write_json'] = writer
    assert watch.delete(item['id'])
    print('  OK: Watchlist-Update/Erstellen/Loeschen bei vollem Datentraeger')


def check_people(ns):
    # Keep actual globals of extracted class; no monkeypatched internal fake manager.
    people_ns = dict(ns, contextmanager=__import__('contextlib').contextmanager)
    load_symbols(('PersonHistoryManager',), people_ns)
    people = people_ns['PersonHistoryManager']()
    existing = people.add_event({'id': 'person-a', 'source': 'camera', 'event_type': 'appearance', 'counted': True})
    assert existing and people.get_event('person-a')
    old = people_ns['_atomic_write_json']
    people_ns['_atomic_write_json'] = lambda *_: (_ for _ in ()).throw(OSError('disk full'))
    try:
        try:
            people.add_event({'id': 'person-b', 'source': 'camera', 'counted': True})
            raise AssertionError('Personen-History write was silently ignored')
        except OSError:
            pass
        assert people.get_event('person-b') is None and people.get_event('person-a')
        try:
            people.update_event('person-a', {'label': 'Should rollback'})
            raise AssertionError('Review update write was silently ignored')
        except OSError:
            pass
        assert people.get_event('person-a')['label'] == ''
    finally:
        people_ns['_atomic_write_json'] = old
    assert len(people.get_all(limit='oops', offset='-5')) == 1
    assert people.get_event('person-a')['id'] == 'person-a'
    print('  OK: Personen-Ereignisse und Reviews bei Schreibfehlern')


def check_registry(ns):
    video = dict(ns, VIDEO_JOB_STATE_FILE=Path('data/video_jobs.json'),
                 video_jobs_lock=threading.RLock(), video_processing_jobs={},
                 history_manager=SimpleNamespace(delete_video_job_entries=lambda *_: 0))
    load_symbols(('_video_status_payload', '_persist_video_jobs', '_restore_video_jobs'), video)
    job_id = str(uuid.uuid4())
    invalid_id = '../../path/escape'
    Path('uploads/processed').mkdir(parents=True)
    output = Path('uploads/processed') / (job_id + '_output.mp4')
    output.write_bytes(b'fake completed file')
    video['video_processing_jobs'][job_id] = {
        'id': job_id, 'filename': '../../malicious.mp4', 'status': 'completed',
        'created_at': datetime.now().isoformat(), 'output_path': str(output),
        'saved_history_count': 13, 'video_path': '../../wrong-file.mp4'
    }
    assert video['_persist_video_jobs']()
    dumped = json.loads(Path('data/video_jobs.json').read_text())
    assert 'output_path' not in dumped[0] and 'video_path' not in dumped[0]
    dumped += [{'id': invalid_id, 'filename': '../../sneaky.mp4', 'status': 'completed'},
               {'id': str(uuid.uuid4()), 'filename': 'unavailable.mov', 'status': 'processing'}]
    Path('data/video_jobs.json').write_text(json.dumps(dumped), encoding='utf-8')
    video['video_processing_jobs'].clear()
    assert video['_restore_video_jobs']() == 2
    assert video['video_processing_jobs'][job_id]['status'] == 'completed'
    assert video['video_processing_jobs'][job_id]['filename'] == 'malicious.mp4'
    assert video['video_processing_jobs'][job_id]['saved_history_count'] == 13
    assert all(v['status'] in ('completed', 'error', 'cancelled') for v in video['video_processing_jobs'].values())
    assert invalid_id not in video['video_processing_jobs']
    assert not any('wrong-file' in str(v) for v in video['video_processing_jobs'].values())
    print('  OK: Neustart-Recovery gespeicherter Video-Jobs ohne Dateipfad-Leckage')


def check_security(ns):
    auth = dict(ns, secrets=secrets, Response=lambda text, **kw: SimpleNamespace(text=text, headers={}, status_code=200),
                jsonify=lambda data: SimpleNamespace(data=data, headers={}, status_code=200),
                g=SimpleNamespace(), request=SimpleNamespace(path='/api/private', headers={}, authorization=None),
                APP_VERSION='test', VIDEO_JOB_STATE_FILE=Path('data/video_jobs.json'),
                video_jobs_lock=threading.RLock(), video_processing_jobs={})
    # This isolated test exercises direct/LAN Basic Auth. The new trusted
    # ingress-prefix helper is covered separately in test_ingress_0130.py.
    auth['_platevision_ingress_prefix'] = lambda: ''
    load_symbols(('_basic_auth_valid', '_begin_request_context', '_storage_audit'), auth)
    auth['request'].headers = {}
    assert auth['_basic_auth_valid']()  # explicitly opt-in to avoid breaking Home Assistant.
    os.environ['PLATEVISION_AUTH_PASSWORD'] = 'very-secret'
    try:
        assert not auth['_basic_auth_valid']()
        blocked = auth['_begin_request_context']()
        assert blocked.status_code == 401
        assert 'WWW-Authenticate' in blocked.headers
        auth['request'].authorization = SimpleNamespace(type='Basic', username='platevision', password='wrong')
        assert not auth['_basic_auth_valid']()
        auth['request'].authorization.password = 'very-secret'
        assert auth['_basic_auth_valid']()
        assert auth['_begin_request_context']() is None
        auth['request'].authorization = None
        auth['request'].path = '/api/system/live'
        assert auth['_begin_request_context']() is None
        auth['request'].path = '/api/whatever'
        os.environ['PLATEVISION_AUTH_USERNAME'] = 'operator'
        auth['request'].authorization = SimpleNamespace(type='Basic', username='platevision', password='very-secret')
        assert not auth['_basic_auth_valid']()
        auth['request'].authorization.username = 'operator'
        assert auth['_basic_auth_valid']()
    finally:
        os.environ.pop('PLATEVISION_AUTH_PASSWORD', None)
        os.environ.pop('PLATEVISION_AUTH_USERNAME', None)

    Path('data/config.json').parent.mkdir(parents=True, exist_ok=True)
    Path('data/config.json').write_text('{not-json', encoding='utf-8')
    Path('data/config.json.bak').write_text('{}', encoding='utf-8')
    result = auth['_storage_audit']()
    config_entry = next(item for item in result['files'] if item['category'] == 'configuration')
    assert config_entry['valid'] is False and config_entry['backup_valid'] is True
    assert any('configuration' in issue for issue in result['issues'])
    assert result['disk']['free_mb'] is not None
    assert not result['authentication_enabled']
    assert 'very-secret' not in json.dumps(result)
    print('  OK: HTTP-Basic-Zugriff, Health-Ausnahme, Speicher- und Backupprobleme')


def check_atomic_failure(ns):
    writer = ns['_atomic_write_json']
    target = Path('data/persist/atomic.json')
    writer(target, {'revision': 1})
    writer(target, {'revision': 2})
    assert json.loads(target.read_text())['revision'] == 2
    assert json.loads(Path(str(target) + '.bak').read_text())['revision'] == 1
    old_replace = os.replace
    def crash_before_replace(src, dst):
        if Path(dst) == target:
            raise OSError('power interruption while publishing primary')
        return old_replace(src, dst)
    os.replace = crash_before_replace
    try:
        try:
            writer(target, {'revision': 3})
            raise AssertionError('Atomic publishing should propagate OS failure')
        except OSError:
            pass
    finally:
        os.replace = old_replace
    assert json.loads(target.read_text())['revision'] == 2
    assert json.loads(Path(str(target) + '.bak').read_text())['revision'] == 2
    assert not list(target.parent.glob('*.tmp'))
    print('  OK: Atomares Schreiben und Sicherungsrotation bei unterbrochenem Replace')


def check_malformed_rows(ns):
    # 0.12.5: mixed-type persisted histories are suspect. Never silently drop
    # bad rows and overwrite the only copy on the next save.
    value = [{'id': 'valid', 'plate_text': 'ZH 4'}, None, 'broken']
    for path, cls_name, method in (
        ('data/history/detections.json', 'HistoryManager', 'history'),
        ('data/watchlist.json', 'WatchlistManager', 'items')):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.with_name(target.name + '.bak').unlink(missing_ok=True)
        target.write_text(json.dumps(value), encoding='utf-8')
        raw = target.read_bytes()
        manager = ns[cls_name]()
        assert getattr(manager, method) == []
        assert str(target.resolve()) in ns['_read_only_storage']
        assert not (manager.save_history() if cls_name == 'HistoryManager' else manager.save())
        assert target.read_bytes() == raw
    people_ns = dict(ns, contextmanager=__import__('contextlib').contextmanager)
    load_symbols(('PersonHistoryManager',), people_ns)
    target = Path('data/people/history.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.with_name(target.name + '.bak').unlink(missing_ok=True)
    target.write_text(json.dumps(value), encoding='utf-8')
    assert people_ns['PersonHistoryManager']().history == []
    assert str(target.resolve()) in people_ns['_read_only_storage']
    print('  OK: beschädigte Datensätze nicht still verwerfen/überschreiben; sicherer Schreibschutz')


def main():
    cwd = Path.cwd()
    ns = build_ns(Config())
    with tempfile.TemporaryDirectory(prefix='platevision-resilience-') as td:
        os.chdir(td)
        try:
            check_history_and_failures(ns)
            check_watchlist(ns)
            check_people(ns)
            check_registry(ns)
            check_security(ns)
            check_atomic_failure(ns)
            check_malformed_rows(ns)
        finally:
            os.chdir(cwd)
    print('PlateVision resilience regression tests: 7 Bereiche bestanden')

if __name__ == '__main__':
    main()
