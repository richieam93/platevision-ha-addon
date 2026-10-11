#!/usr/bin/env python3
"""0.13.5: durable watchlist alerts, opt-outs, visit dedup, rollback, backups.

Runs without Flask, Torch, network or camera hardware.
"""
import ast
import copy
import json
import logging
import os
import sqlite3
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'platevision/src'))
from watchlist_alerts import WatchlistAlertStore, validate_notify, notification_enabled


def sample(event_id, *, plate='ZH 101', match_id='w1', notify=True,
           source='rtsp', job=None, confidence=.91):
    row = {'id':event_id, 'plate_text':plate,
           'plate_text_normalized':''.join(c for c in plate.upper() if c.isalnum()),
           'confidence':confidence, 'source':source,
           'watchlist_match':{'id':match_id,'label':'Geprueft','category':'known','notify':notify}}
    if job:
        row['video_job_id' if source == 'video_upload' else 'upload_id'] = job
    return row


def test_db_workflow():
    with tempfile.TemporaryDirectory(prefix='pv-alerts-') as tmp:
        db = Path(tmp) / 'alerts.sqlite3'
        store = WatchlistAlertStore(db)
        assert store.status()['total'] == 0
        assert store.record(sample('low',confidence=.33), minimum_confidence=.65) is None
        assert store.record(sample('disabled',notify='false')) is None
        assert store.record(sample('missing',match_id='')) is None
        assert store.record(sample('a1'),now=10000)['created']
        assert not store.record(sample('a1'),now=10000)['created']
        assert not store.record(sample('a2'),now=10005)['created']
        assert store.status() == {'total':1, 'unread':1, 'backend':'sqlite'}
        page = store.page()
        assert page['items'][0]['occurrences'] == 2
        assert page['items'][0]['plate_text'] == 'ZH 101'
        assert page['items'][0]['latest_event_id'] == 'a2'
        assert store.acknowledge(page['items'][0]['id'])
        assert store.status()['unread'] == 0
        assert not store.record(sample('a3'), now=10010)['created']
        assert store.status()['unread'] == 0, 'The same visit must not reopen a read alert'
        assert store.record(sample('a4'),now=10200)['created']
        assert store.record(sample('b1',source='video_upload',job='job1'),now=10200)['created']
        assert store.record(sample('b2',source='video_upload',job='job2'),now=10200)['created']
        assert store.record(sample('c1',source='image_upload',job='upload1'),now=10200)['created']
        assert store.status() == {'total':5,'unread':4,'backend':'sqlite'}
        assert store.page(unread_only=True)['total'] == 4
        assert store.acknowledge_all() == 4
        assert store.page(unread_only=True)['total'] == 0
        store2 = WatchlistAlertStore(db)
        assert store2.status() == {'total':5,'unread':0,'backend':'sqlite'}
        assert store2.record(sample('a2'),now=10300)['duplicate_event']
        assert store2.delete(store2.page()['items'][0]['id'])
        with sqlite3.connect(db) as conn:
            assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
        backup = Path(tmp) / 'backup.sqlite3'
        store2.backup_to(backup)
        with sqlite3.connect(backup) as conn:
            assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            assert conn.execute('SELECT COUNT(*) FROM alerts').fetchone()[0] == 4
        print('  OK persistent inbox / cooldown / opt-out / read+delete / WAL backup')


def test_concurrent_and_bounded():
    with tempfile.TemporaryDirectory(prefix='pv-alerts-concurrent-') as tmp:
        store = WatchlistAlertStore(Path(tmp) / 'alert.sqlite3', max_alerts=100)
        def send(n):
            return store.record(sample(f'thread-{n}'),now=15000+n*.01)
        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(send, range(35)))
        assert sum(bool(r['created']) for r in results) == 1
        assert store.page()['items'][0]['occurrences'] == 35
        for i in range(130):
            r = store.record(sample('unique-'+str(i), plate=f'BE {i:05d}'), now=16000+i)
            assert r and r['created']
        assert store.status()['total'] == 100
        assert store.page(limit=500)['limit'] == 100
        assert store.page(offset=-30)['offset']==0
        with sqlite3.connect(store.path) as conn:
            assert conn.execute('SELECT COUNT(*) FROM seen_events').fetchone()[0] == 100
            assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        print('  OK concurrent inserts / cross-thread cooldown / bounded storage and cascade')


def test_errors_and_optouts():
    for val in (False, 'false', 'OFF','nein', '0', 0):
        assert validate_notify(val) is False
        assert not notification_enabled(val)
    for val in (True, 'true','yes','ja', '1', 1):
        assert validate_notify(val) is True
        assert notification_enabled(val)
    for val in ('maybe', [], {}, -1, 2, None):
        try:
            validate_notify(val)
            raise AssertionError(f'Invalid notify switch accepted: {val}')
        except ValueError:
            pass
    with tempfile.TemporaryDirectory(prefix='pv-alert-errors-') as tmp:
        store = WatchlistAlertStore(Path(tmp) / 'alert.sqlite3')
        store.record(sample('good'))
        with patch.object(store, '_connect', side_effect=sqlite3.OperationalError('disk full')):
            try:
                store.record(sample('failed'))
                raise AssertionError('Disk failure incorrectly succeeded')
            except sqlite3.OperationalError:
                pass
        assert store.status()['total'] == 1
        assert store.page()['items'][0]['first_event_id'] == 'good'
        print('  OK strict imported booleans / failed writes do not damage alerts')


def test_app_integration():
    app = ROOT/'platevision/src/app.py'
    tree = ast.parse(app.read_text(encoding='utf-8'))
    nodes = []
    for name in ('WatchlistManager', '_record_watchlist_notification'):
        node = copy.deepcopy(next(n for n in tree.body if isinstance(n, (ast.ClassDef,ast.FunctionDef)) and n.name == name))
        if isinstance(node,ast.FunctionDef): node.decorator_list=[]
        nodes.append(node)
    with tempfile.TemporaryDirectory(prefix='pv-alert-app-') as tmp:
        os.chdir(tmp)
        try:
            config = SimpleNamespace(get=lambda *args: {
                ('alerts','watchlist_notifications'):True,
                ('plate_recognition','watchlist_enabled'):True,
                ('alerts','min_alert_confidence'):.65,
                ('alerts','watchlist_alert_cooldown_seconds'):180}.get(args))
            env = {'config_manager':config,'logger':logging.getLogger('pv-alert-test'),
                   '_safe_numeric':lambda value, default: value if value is not None else default,
                   '_page_integer':lambda v,d,minimum=0,maximum=3600: min(maximum,max(minimum,int(v if v is not None else d))),
                   'notification_enabled':notification_enabled,
                   'threading':threading,'uuid':__import__('uuid'), 'datetime':datetime,
                   'PlateUtils':SimpleNamespace(normalize=lambda v,compact:''.join(c for c in str(v).upper() if c.isalnum()),
                                                pretty=lambda v:str(v)),
                   '_load_json_with_backup':lambda *args:[],
                   '_atomic_write_json':lambda *args:None}
            exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(app),'exec'),env)
            watchlist=env['WatchlistManager']()
            assert not watchlist.add('ZH 1234',notify='false')['notify']
            assert not watchlist.import_items([{'plate_text':'ZH1234','notify':'no'}]) is None
            assert watchlist.list()[0]['notify'] is False
            orig=watchlist.list()
            try:
                watchlist.import_items([{'plate_text':'BE 111','notify':'maybe'}])
                raise AssertionError('Bad imported boolean accepted')
            except ValueError:
                assert watchlist.list() == orig
            store=WatchlistAlertStore(Path(tmp)/'alerts.sqlite3')
            ingest=env['_record_watchlist_notification']
            assert ingest(sample('1'),store)['created']
            config.get=lambda *args: False if args == ('alerts','watchlist_notifications') else .65
            assert ingest(sample('2'),store) is None
            assert store.status()['total']==1
        finally:
            os.chdir(ROOT)
    print('  OK real watchlist import and committed-event notification gate')



def test_video_lifecycle_no_phantom_alerts():
    # Uses the REAL MP4 processing function, with fake OCR instead of any ML
    # model; cancellation and a storage failure must not leave ghost alerts.
    sys.path.insert(0, str(ROOT/'tools'))
    from test_deep_regressions import Config, make_mp4, make_video_ns
    with tempfile.TemporaryDirectory(prefix='pv-alert-video-') as tmp:
        current=Path.cwd()
        os.chdir(tmp)
        try:
            Path('data/history').mkdir(parents=True)
            Path('uploads/videos').mkdir(parents=True)
            Path('uploads/processed').mkdir(parents=True)
            config=Config()
            config.config['alerts']={'watchlist_notifications':True,
                                     'min_alert_confidence':.65,'watchlist_alert_cooldown_seconds':180}
            config.config['plate_recognition']['watchlist_enabled']=True
            video=make_video_ns(config)
            store=WatchlistAlertStore(Path('data/alerts/watchlist.sqlite3'))
            video['watchlist_alert_center']=store
            video['notification_enabled']=notification_enabled
            app=ROOT/'platevision/src/app.py'
            tree=ast.parse(app.read_text(encoding='utf-8'))
            node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                                    and n.name=='_record_watchlist_notification'))
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(app),'exec'),video)
            class HistoryWithAlerts:
                def __init__(self): self.rows=[]
                def add_entry(self, row, check_duplicate=True):
                    saved=dict(row, id=f'e-{len(self.rows)}',
                               plate_text_normalized='ZH1234',
                               watchlist_match={'id':'w1','notify':True,'label':'ALARM'})
                    self.rows.append(saved)
                    return saved
                def delete_video_job_entries(self, job_id):
                    before=len(self.rows)
                    self.rows=[r for r in self.rows if r.get('video_job_id')!=job_id]
                    return before-len(self.rows)
            video['history_manager']=HistoryWithAlerts()
            path=Path('uploads/videos/complete.mp4')
            make_mp4(path, 24)
            video['video_processing_jobs']['complete']={'id':'complete','status':'queued',
                'filename':'video.mp4','created_at':datetime.now().isoformat()}
            video['video_job_cancel_events']['complete']=threading.Event()
            video['process_video_job']('complete',str(path),'video.mp4')
            assert video['video_processing_jobs']['complete']['status']=='completed'
            assert store.status()['total']==1
            path2=Path('uploads/videos/cancel.mp4')
            make_mp4(path2,24)
            video['video_processing_jobs']['cancel']={'id':'cancel','status':'queued',
                'filename':'cancel.mp4','created_at':datetime.now().isoformat()}
            video['video_job_cancel_events']['cancel']=threading.Event()
            video['video_job_cancel_events']['cancel'].set()
            video['process_video_job']('cancel',str(path2),'cancel.mp4')
            assert video['video_processing_jobs']['cancel']['status']=='cancelled'
            assert store.status()['total']==1
            print('  OK real MP4 completion alerts; cancellation creates no phantom alerts')
        finally:
            os.chdir(current)


def test_ui_guards():
    template=(ROOT/'platevision/src/templates/alerts.html').read_text(encoding='utf-8')
    app=(ROOT/'platevision/src/app.py').read_text(encoding='utf-8')
    assert 'replaceChildren' in template and '.textContent =' in template
    assert 'newElement(' in template
    assert 'innerHTML' not in template
    for route in ("@app.route('/api/alerts')", "@app.route('/api/alerts/status')",
                  "@app.route('/api/alerts/read-all', methods=['POST'])"):
        assert route in app
    assert '_record_watchlist_notification(saved_entry, alert_center)' in app
    assert "_ALERT_BACKUP_DB = 'data/alerts/watchlist.sqlite3'" in app
    print('  OK routes + XSS-safe text rendering + backup inventory wiring')


if __name__ == '__main__':
    original=Path.cwd()
    try:
        test_db_workflow()
        test_concurrent_and_bounded()
        test_errors_and_optouts()
        test_app_integration()
        test_video_lifecycle_no_phantom_alerts()
        test_ui_guards()
    finally:
        os.chdir(original)
    print('PlateVision 0.13.5 watchlist alert tests: OK')
