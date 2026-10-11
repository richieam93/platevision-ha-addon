#!/usr/bin/env python3
"""PlateVision 0.12.8 review workflow regression tests, no ML/Flask required.

Covers indexed filtering, SQL atomic writes, lost-update protection, error
rollback, legacy JSON fallback and API request validation.
"""
import ast
import copy
import json
import io
import logging
import os
import re
import sqlite3
import sys
import tempfile
import threading
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'platevision/src'))
from history_sqlite import SQLiteHistoryStore, HistoryStorageError, ReviewConflictError, REVIEW_STATUSES


def extracted(names, env):
    tree = ast.parse((ROOT / 'platevision/src/app.py').read_text(encoding='utf-8'))
    collected = []
    for name in names:
        node = next(node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
        node = copy.deepcopy(node)
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        collected.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=collected, type_ignores=[])), '<review-tests>', 'exec'), env)

class Config:
    def get(self, *path):
        data = {'search': {'default_limit': 50, 'enable_fuzzy_search': False},
                'history': {'fuzzy_duplicate_detection': False},
                'general': {'max_history_entries': 10000}}
        for part in path:
            data = data.get(part) if isinstance(data, dict) else None
        return data


def make_manager(store, rows):
    env = dict(re=re, sqlite3=sqlite3, logging=logging, os=os, uuid=uuid,
               logger=logging.getLogger('review-tests'), datetime=datetime, timedelta=timedelta,
               threading=threading, Counter=Counter, defaultdict=defaultdict,
               SequenceMatcher=SequenceMatcher, config_manager=Config(),
               HistoryStorageError=HistoryStorageError, ReviewConflictError=ReviewConflictError,
               SQLiteHistoryStore=SQLiteHistoryStore,
               _read_only_storage={},
               _page_integer=lambda v,d,minimum=0,maximum=1000: max(minimum, min(maximum, int(d if v in ('',None) else v))),
               _safe_numeric=lambda v,d=0.0: float(v) if v is not None else d)
    extracted(('PlateUtils', 'HistoryManager'), env)
    man = env['HistoryManager'].__new__(env['HistoryManager'])
    man.history = rows
    man.lock = threading.RLock()
    man._sqlite_store = store
    return man, env


def sample_rows():
    return [
        {'id':'p1','timestamp':'2026-10-08T10:00:00','plate_text':'ZH 100','confidence':.91,'source':'rtsp','plate_image':'base64-sensitive'},
        {'id':'p2','timestamp':'2026-10-08T11:00:00','plate_text':'AG 200','confidence':.73,'review_status':'confirmed'},
        {'id':'p3','timestamp':'2026-10-08T12:00:00','plate_text':'BE 300','confidence':.52,'review_status':'needs_review'},
        {'id':'p4','timestamp':'2026-10-08T13:00:00','plate_text':'AG 200','confidence':.88,'review_status':'ignored'},
        {'id':'p5','timestamp':'2026-10-08T14:00:00','plate_text':'VD 500','confidence':.55,'review_status':'alien'},
    ]


def test_sqlite():
    with tempfile.TemporaryDirectory() as td:
        legacy = Path(td) / 'detections.json'
        rows = sample_rows()
        original_json = json.dumps(rows)
        legacy.write_text(original_json, encoding='utf-8')
        store = SQLiteHistoryStore(legacy, kind='plates')
        assert store.initialize(lambda: json.loads(legacy.read_text())) == rows
        manager, env = make_manager(store, rows)
        assert manager.review_counts() == {'unreviewed':2,'confirmed':1,'needs_review':1,'ignored':1,'total':5}
        for status, count in [('unreviewed',2),('confirmed',1),('needs_review',1),('ignored',1)]:
            filters = {'review_status':status,'fuzzy':False, 'unique':False, 'sort':'timestamp'}
            sql = manager.search_advanced(filters)
            assert sql['total'] == count, (status,sql)
            manager._sqlite_store = None
            fallback = manager.search_advanced(filters)
            manager._sqlite_store = store
            assert sql == fallback, (status,sql,fallback)
        revision_before = store.revision
        original = store.get('p1')
        note = '<img src=x onerror="window.__pwned=1"> Wichtig!'
        edited = manager.update_review('p1','confirmed',note,expected_updated_at='')
        assert edited['review_note'] == note and edited['review_status'] == 'confirmed'
        assert edited['review_updated_at'] and store.revision == revision_before + 1
        assert store.get('p1') == edited and manager.history[0] == edited
        assert store.get('p2') == rows[1]
        assert store.plate_review_counts()['confirmed'] == 2
        assert legacy.read_text(encoding='utf-8') == original_json
        assert store.snapshot.is_file()
        with __import__('contextlib').closing(store._connect(read_only=True)) as conn:
            assert conn.execute('SELECT review_status_lower FROM events WHERE event_key=?',('p1',)).fetchone()[0] == 'confirmed'
            assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        try:
            manager.update_review('p1','ignored','stale', expected_updated_at='')
            raise AssertionError('stale update accepted')
        except ReviewConflictError:
            pass
        assert store.get('p1')['review_note'] == note
        for s, n in [('invalid','x'),('confirmed','x'*1201),('confirmed','bad\x00byte')]:
            try:
                manager.update_review('p1',s,n)
                raise AssertionError('invalid review accepted')
            except ValueError:
                pass
        assert manager.update_review('missing','confirmed','') is None
        assert store.get('p1') == edited
        # SQL errors must roll back the payload, review index and revision.
        def storage_failure(*args, **kwargs):
            raise sqlite3.OperationalError('Disk full')
        with patch.object(store,'_put_meta', side_effect=storage_failure):
            try:
                store.update_plate_review('p3','ignored','try')
                raise AssertionError('disk full wrongly committed')
            except sqlite3.OperationalError:
                pass
        assert store.get('p3') == rows[2]
        assert store.revision == revision_before + 1
        assert store.plate_review_counts()['needs_review'] == 1
        # A second process must not silently overwrite the first process's changes.
        other = SQLiteHistoryStore(legacy, kind='plates')
        other.initialize(lambda: [])
        manager.update_review('p2','needs_review','recheck')
        try:
            other.update_plate_review('p2','ignored','stale process')
            raise AssertionError('outdated process revision accepted')
        except HistoryStorageError:
            pass
        assert other.get('p2')['review_status'] == 'needs_review'
        assert not env['_read_only_storage'], env['_read_only_storage']
        print('  OK SQLite: indexed review + notes + optimistic locking + durable rollback')


def test_json_fallback():
    rows = sample_rows()
    manager, env = make_manager(None, rows)
    stored=[]
    def saved():
        stored.append(copy.deepcopy(manager.history))
        return True
    manager.save_history = saved
    edited = manager.update_review('p1','confirmed','Ein Test', expected_updated_at='')
    assert edited['review_status']=='confirmed' and stored[0][0]==edited
    assert manager.review_counts()['confirmed']==2
    try:
        manager.update_review('p1','ignored','',expected_updated_at='')
        raise AssertionError('old edit accepted')
    except ReviewConflictError:
        pass
    old = manager.history
    manager.save_history = lambda: False
    try:
        manager.update_review('p3','confirmed','Test')
        raise AssertionError('write failure accepted')
    except OSError:
        pass
    assert manager.history is old and manager.history[2]['review_status']=='needs_review'
    print('  OK JSON: review saved, conflicts blocked, failed writes reverted')


def test_route_validation():
    class DummyRequest:
        content_length = 0
        value = {}
        is_json = True
        raw = None
        @property
        def stream(self):
            return io.BytesIO(self.raw if self.raw is not None else json.dumps(self.value).encode('utf-8'))
    request = DummyRequest()
    manager=type('Manager',(),{'update_review':lambda self, *args, **kwargs: {'id':args[0], 'plate_text':'ZH 100', 'review_status':args[1]}})()
    def jsonify(obj): return obj
    env={'request':request,'history_manager':manager,'jsonify':jsonify,'json':json,
         'ReviewConflictError':ReviewConflictError,
         '_history_api_entries':lambda items,opts:items}
    extracted(('api_history_review_entry',), env)
    call=env['api_history_review_entry']
    request.value = ['oops']
    assert call('p1')[1]==400
    request.value={'review_status':'confirmed','review_note':''}
    assert call('p1')['success']
    request.content_length=8193
    assert call('p1')[1]==413
    request.content_length=0
    request.raw = b'not-json'
    assert call('p1')[1] == 400
    request.raw = b'\xff\xfe'
    assert call('p1')[1] == 400
    request.raw = b' ' * 8193
    assert call('p1')[1] == 413
    request.raw = None
    request.is_json = False
    assert call('p1')[1] == 400
    request.is_json = True
    manager.update_review=lambda *args,**kwargs: (_ for _ in ()).throw(ReviewConflictError('conflict'))
    assert call('p1')[1]==409
    manager.update_review=lambda *args,**kwargs: (_ for _ in ()).throw(ValueError('bad'))
    assert call('p1')[1]==400
    manager.update_review=lambda *args,**kwargs: None
    assert call('p1')[1]==404
    print('  OK API: 400/404/409/413 and explicit success')


if __name__=='__main__':
    test_sqlite()
    test_json_fallback()
    test_route_validation()
    print('PlateVision 0.12.8 quality review regression: OK')
