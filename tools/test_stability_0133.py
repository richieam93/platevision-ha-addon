#!/usr/bin/env python3
"""0.13.3: isolated real-code regression tests (no running ML/Flask service needed)."""
from __future__ import annotations
import ast
import base64
import copy
import logging
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta
from types import MethodType
import uuid
import hashlib
import threading
from types import SimpleNamespace
import os
import tempfile
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'platevision/src'
sys.path.insert(0,str(SRC))
from history_sqlite import SQLiteHistoryStore
TREE=ast.parse((SRC/'app.py').read_text(encoding='utf-8'))


def function(name, env):
    source=copy.deepcopy(next(node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name==name))
    source.decorator_list=[]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[source], type_ignores=[])),
                 str(SRC/'app.py'),'exec'),env)
    return env[name]


def test_indexed_image_lookup():
    class IndexedHistory:
        def __init__(self): self.lookups=[]
        @property
        def history(self): raise AssertionError('full history scanned for one image')
        def get_by_id(self, entry_id):
            self.lookups.append(entry_id)
            if entry_id=='target':
                return {'id':entry_id,'vehicle_image':base64.b64encode(b'one-photo').decode('ascii')}
            return None
    manager=IndexedHistory()
    env={'history_manager':manager, 'jsonify':lambda d:d, 'base64':base64,
         'Response':lambda bytes_data, mimetype: (bytes_data,mimetype),
         'logger':logging.getLogger('test-index')}
    fn=function('api_history_entry_image',env)
    assert fn('target','vehicle')==(b'one-photo','image/jpeg')
    assert manager.lookups==['target']
    assert fn('missing','vehicle')[1]==404
    assert fn('target','../danger')[1]==404
    # Real SQLite backend can retrieve a single image by event_key.
    with tempfile.TemporaryDirectory(prefix='pv-0133-img-') as tmp:
        source=Path(tmp)/'history/detections.json';source.parent.mkdir(parents=True)
        rows=[{'id':str(i),'plate_text':f'X{i}','timestamp':'2026-10-10T11:10:00',
               'vehicle_image':base64.b64encode(str(i).encode()).decode()} for i in range(1500)]
        source.write_text(json.dumps(rows),encoding='utf-8')
        store=SQLiteHistoryStore(source,kind='plates')
        store.initialize(lambda:json.loads(source.read_text()))
        assert store.get('1499')['id']=='1499'
        assert store.get('absent') is None
    print('  OK: indexed image endpoint, no scan of entire history, SQLite indexed event lookup')


def test_atomic_gallery():
    class FakePeople:
        def __init__(self): self.calls=[]
        def bulk_delete(self, ids, delete_images):
            self.calls.append((ids,delete_images))
            return {'success':True,'deleted':2,'deleted_images':3,'remaining':5}
        def delete_event(self,*args,**kwargs): raise AssertionError('non-transactional individual delete')
    mgr=FakePeople()
    env={'person_history_manager':mgr,'request':SimpleNamespace(get_json=lambda silent: {'ids':['one','two']}),
         'jsonify':lambda v:v}
    function('_parse_api_bool',env)
    function('_people_delete_payload',env)
    fn=function('api_people_images_delete',env)
    result=fn()
    assert result['deleted_events']==2 and result['deleted_images']==3
    assert mgr.calls==[(['one','two'],True)]
    for ids in (42, ['x']*501, [None], [123], [{'bad':'input'}]):
        env['request']=SimpleNamespace(get_json=lambda silent, ids=ids: {'ids':ids})
        result=fn()
        assert result[1]==400, ids
    assert len(mgr.calls)==1
    assert env['_parse_api_bool']('false') is False
    assert env['_parse_api_bool']('true') is True
    assert env['_parse_api_bool'](0) is False
    try:
        env['_parse_api_bool']('garbage')
        raise AssertionError('Invalid boolean accepted')
    except ValueError:
        pass
    # Historic bulk-delete preserves images when a client sends "false".
    hist_delete=function('api_people_history_bulk_delete',env)
    env['request']=SimpleNamespace(get_json=lambda silent: {'ids':['one','two'], 'delete_images':'false'})
    hist_delete()
    assert mgr.calls[-1]==(['one','two'],False)
    # A non-object JSON body must not cause an unhandled 500.
    env['request']=SimpleNamespace(get_json=lambda silent: [1,2,3])
    assert fn()[1]==400
    print('  OK: transactional gallery and history delete, validated IDs and false string flags')


def test_json_limits():
    env={'jsonify':lambda payload:payload,'_API_JSON_MAX_BYTES':1024*1024,
         '_API_JSON_IMPORT_MAX_BYTES':5*1024*1024}
    fn=function('_limit_json_api_request',env)
    req=SimpleNamespace(path='/api/history/search',method='POST',is_json=True,
                        content_length=100,max_content_length=None)
    env['request']=req
    assert fn() is None and req.max_content_length==1024*1024
    req.content_length=1024*1024+1
    assert fn()[1]==413
    req.path='/api/config/import';req.content_length=4*1024*1024
    assert fn() is None and req.max_content_length==5*1024*1024
    req.path='/api/process/video';req.is_json=False;req.max_content_length=None
    assert fn() is None and req.max_content_length is None
    req.path='/dashboard';req.is_json=True
    assert fn() is None and req.max_content_length is None
    req.path='/api/history/search';req.is_json=True;req.content_length=None
    assert fn() is None and req.max_content_length==1024*1024
    print('  OK: JSON API limited without blocking video multipart and bounded imports')


def test_forwarded_host_not_trusted_from_lan():
    fake_request=SimpleNamespace(method='POST',path='/api/config/general',
                                 headers={'Origin':'https://attacker.example',
                                          'X-Forwarded-Host':'attacker.example'},
                                 host='ha.local',endpoint='api_save_general_config')
    fake_g=SimpleNamespace()
    lock=threading.RLock()
    env={'request':fake_request,'g':fake_g,'jsonify':lambda value:value,
         'config_manager':SimpleNamespace(lock=lock),
         '_platevision_ingress_prefix':lambda: ''}
    guard=function('_protect_mutating_requests',env)
    blocked=guard()
    assert blocked[1]==403,blocked
    assert not hasattr(fake_g,'platevision_config_write_lock')
    fake_request.headers={'Origin':'https://ha.local'}
    assert guard() is None
    assert fake_g.platevision_config_write_lock
    lock.release()
    print('  OK: malicious X-Forwarded-Host cannot authorize foreign Origin on direct port')


def test_health_disabled_not_failure():
    with tempfile.TemporaryDirectory(prefix='pv-0133-health-') as td:
        base=Path(td)
        (base/'vehicle.pt').write_bytes(b'weights')
        (base/'plate.pt').write_bytes(b'weights')
        config={
            'models': {'vehicle_detector':'vehicle.pt','license_plate_detector':'plate.pt'},
            'people':{'enabled':False,'model_mode':'model_file','custom_model_path':'missing.pt'},
            'rtsp': {'enabled':False, 'url':'rtsp://placeholder', 'autostart_enabled':False}
        }
        class FakeConfig:
            DEFAULT_CONFIG={'rtsp':{'url':'rtsp://placeholder'}}
            def get(self, group, key=None):
                return config[group] if key is None else config[group].get(key)
        class FakeStore:
            def quick_status(self): return {'ok':True,'count':18,'legacy_snapshot_exists':True}
        def manager(path): return SimpleNamespace(HISTORY_FILE=path,_sqlite_store=FakeStore())
        stream=SimpleNamespace(get_status=lambda: {'connected':False, 'fps':0,'frame_count':0,
                                                   'detection_count':0,'error':'No connection'})
        env={'config_manager':FakeConfig(), 'ConfigManager':FakeConfig,
             'stream_manager':stream, 'history_manager':manager('data/history.json'),
             'person_history_manager':manager('data/people.json'),
             'watchlist_manager':SimpleNamespace(WATCHLIST_FILE='data/watchlist.json'),
             '_resolve_model_path':lambda model: str(base/model) if model else '',
             'os':SimpleNamespace(path=SimpleNamespace(isdir=lambda v:True,dirname=os.path.dirname),
                                  access=lambda *a:True,W_OK=os.W_OK,environ={}),
             'Path':Path, '_read_only_storage':{}, 'APP_VERSION':'0.13.3',
             'APP_LICENSE':'AGPL-3.0-only','APP_SOURCE_URL':'http://example.test',
             '_safe_numeric':lambda v,d=0:float(v) if v is not None else d,
             'channels_for_settings':lambda settings: (),
             '_ha_settings_snapshot':lambda: {},
             '_alert_delivery_active':lambda: True,
             'watchlist_alert_center':SimpleNamespace(delivery_status=lambda: {
                 'pending':0, 'sending':0, 'failed':0, 'accepted':0,
                 'oldest_waiting_seconds':None, 'expired_leases':0,
                 'last_accepted_at':None}),
             'jsonify':lambda value:value}
        function('_health_summary',env)
        fn=function('api_system_health',env)
        report=fn()
        assert report['ok'] and report['summary']['problems']==0, report['summary']
        assert report['camera']['enabled'] is False
        assert any(c['name']=='stream_connected' and c['state']=='disabled' for c in report['checks'])
        assert any(c['name']=='person_model_path' and c['state']=='disabled' for c in report['checks'])
        assert any(c['name']=='ha_notifications' and c['state']=='disabled' for c in report['checks'])
        # An explicitly enabled HA destination without a Supervisor token is
        # a real configuration problem; a deliberately disabled one is not.
        env['channels_for_settings']=lambda settings: (('persistent',''),)
        report=fn()
        assert not report['ok'] and 'ha_notifications' in report['summary']['problem_checks']
        env['channels_for_settings']=lambda settings: ()
        config['rtsp']['enabled']=True
        report=fn()
        assert not report['ok'] and {'rtsp_configured','stream_connected'} <= set(report['summary']['problem_checks'])
        config['rtsp']['url']='rtsp://user:private@host.local:554/path'
        stream.get_status=lambda:{'connected':True,'fps':9.7,'frame_count':420,'detection_count':14, 'error':None}
        report=fn()
        assert report['ok'] and report['camera']['fps']==9.7, report['summary']
        assert 'private' not in json.dumps(report['checks']), 'private RTSP credential leaked'
        config['people']['enabled']=True
        report=fn()
        assert not report['ok'] and 'person_model_path' in report['summary']['problem_checks']
        env['_read_only_storage']={'bad':'disk'}
        report=fn()
        assert not report['ok'] and 'storage_integrity' in report['summary']['problem_checks']
    print('  OK: disabled components vs genuine failures, RTSP secrets withheld, diagnostics data')


def test_people_gallery_sql_paging_and_session_resilience():
    manager_class = next(node for node in TREE.body if isinstance(node,ast.ClassDef) and node.name=='PersonHistoryManager')
    def method(name, env):
        node = copy.deepcopy(next(fn for fn in manager_class.body
                                  if isinstance(fn,ast.FunctionDef) and fn.name == name))
        node.decorator_list=[]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                     str(SRC/'app.py'),'exec'),env)
        return env[name]
    page_fn=function('_page_integer',{})
    env={'_page_integer':page_fn,'datetime':datetime,'timedelta':timedelta,
         'Counter':Counter,'hashlib':hashlib,'uuid':uuid,
         '_safe_numeric':lambda value, default=0: default}
    gallery=method('image_history',env)
    sessions=method('get_sessions',env)
    with tempfile.TemporaryDirectory(prefix='pv-0133-gallery-') as tmp:
        json_file=Path(tmp)/'people/history.json';json_file.parent.mkdir(parents=True)
        rows=[]
        for i in range(250):
            rows.append({'id':str(i),'timestamp':f'2026-10-10T11:{i%60:02d}:00',
                         'source':'image_upload','counted':i%2==0,'confidence':.9,
                         'images':({'crop':f'crops/{i}.jpg'} if i%4==0 else {})})
        json_file.write_text(json.dumps(rows))
        store=SQLiteHistoryStore(json_file,kind='people')
        store.initialize(lambda:json.loads(json_file.read_text()))
        class FakePeople:
            _sqlite_store=store
            def __init__(self):
                import threading
                self.lock=threading.RLock()
            def _maybe_auto_cleanup_images(self,force=False): return None
            def _filters(self,filters):
                return dict(date_from='2026-10-10T00:00:00',date_to='2026-10-11T00:00:00',
                            counted=None,has_images=filters.get('has_images')=='true',
                            min_confidence=0,sort='desc')
            def _cfg(self): return {'image_history_retention_days':10}
            def _public_item(self,item): return item
            def search(self, filters): raise AssertionError('gallery must not deserialize the entire history')
        fake=FakePeople()
        first=gallery(fake,{'limit':'7','offset':'0','counted_only':'false'})
        second=gallery(fake,{'limit':'7','offset':'7','counted_only':'false'})
        assert first['total']==63 and second['total']==63 and first['has_more']
        assert len(first['entries'])==7 and len(second['entries'])==7
        assert set(r['id'] for r in first['entries']).isdisjoint(r['id'] for r in second['entries'])
        assert all(r['image_urls']['crop'].startswith('/api/people/images/') for r in first['entries'])
        assert gallery(fake,{'limit':'garbage','offset':'nan'})['limit']==60
    class FakeSessions:
        def search(self,filters):
            return [{'id':'event','timestamp':'2026-10-10T10:00:00','images':'bad-record',
                     'bbox':123,'confidence':'invalid','event_type':'appearance','source':'camera'}]
        def _cfg(self): return {'session_gap_minutes':5,'present_timeout_minutes':'corrupt'}
        def _event_dt(self,item): return datetime.fromisoformat(item['timestamp'])
        def _position_signature(self,item): return 'position'
    result=sessions(FakeSessions(),{'limit':'abc','offset':'bad'})
    assert result['total']==1 and result['limit']==30 and result['offset']==0
    assert result['sessions'][0]['image_count']==0
    print('  OK: SQL gallery indexes/pagination, person sessions tolerate malformed parameters and media')


def test_frontend_monitor():
    content=(SRC/'templates/diagnostics.html').read_text(encoding='utf-8')
    for expected in ('id="healthSummary"','id="cameraMonitor"',
                     'id="healthTechnicalDetails"','pvMakeMonitorTile','data.summary',
                     "check.state === 'disabled'", "{{ pv_prefix }}/api/system/health"):
        assert expected in content, expected
    print('  OK: new live camera/system health view and progressive disclosure')


if __name__=='__main__':
    test_indexed_image_lookup()
    test_atomic_gallery()
    test_json_limits()
    test_forwarded_host_not_trusted_from_lan()
    test_health_disabled_not_failure()
    test_people_gallery_sql_paging_and_session_resilience()
    test_frontend_monitor()
    print('PASS PlateVision 0.13.3 reliability and diagnostics checks')
