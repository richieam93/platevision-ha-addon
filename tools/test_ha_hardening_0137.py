#!/usr/bin/env python3
"""Offline regressions for lease fencing, HA privacy and safe config imports.

No camera, Home Assistant token, HTTP access or ML runtime is required.
"""
import ast
import copy
import json
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'platevision/src'))
from watchlist_alerts import WatchlistAlertStore
from ha_notifications import HADeliveryWorker, HomeAssistantSender, ha_settings_from_config
from watchlist_alerts import notification_enabled


def event(i, plate='PRIVATE-123'):
    return {'id': str(i), 'plate_text': plate,
            'plate_text_normalized': plate.upper(), 'confidence': .91,
            'source': 'rtsp', 'watchlist_match': {'id': 'watch1', 'notify': True,
                                                'label': 'PRIVATE-NOTE'}}


class MockResponse:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *args): return None


class MockOpener:
    def __init__(self): self.calls = []
    def open(self, req, timeout=None):
        self.calls.append((req.full_url, json.loads(req.data)))
        return MockResponse()


def test_lease_fencing_and_upgrade():
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / 'alerts.sqlite3'
        first = WatchlistAlertStore(db)
        first.record(event(1), now=1000, ha_channels=(('mobile','mobile_app_one'),))
        # Force exactly the legacy 0.13.6 schema to test an in-place upgrade.
        with sqlite3.connect(db) as conn:
            conn.execute('ALTER TABLE ha_delivery_outbox DROP COLUMN lease_generation')
        upgraded = WatchlistAlertStore(db)
        with sqlite3.connect(db) as conn:
            assert 'lease_generation' in [r[1] for r in conn.execute('PRAGMA table_info(ha_delivery_outbox)')]
        stale = first.claim_delivery(now=1001, lease_seconds=10)
        assert stale['lease_generation'] == 1
        latest = upgraded.claim_delivery(now=1012, lease_seconds=10)
        assert latest['lease_generation'] == 2
        assert stale['id'] == latest['id']
        assert not first.finish_delivery(stale['id'], lease_generation=stale['lease_generation'],
                                         success=True, now=1013)
        assert upgraded.delivery_status()['sending'] == 1
        assert upgraded.finish_delivery(latest['id'], lease_generation=latest['lease_generation'],
                                        success=True, now=1014)
        assert upgraded.delivery_status()['accepted'] == 1
        assert not upgraded.finish_delivery(latest['id'],lease_generation=latest['lease_generation'],
                                            success=False, retryable=True)
        # Explicit user retry resets the displayed attempts, never the
        # monotonic claim-generation used to fence old processes.
        upgraded.record(event(100),now=1200,cooldown_seconds=0,
                        ha_channels=(('persistent',''),))
        failing=upgraded.claim_delivery(now=1201)
        assert upgraded.finish_delivery(failing['id'],lease_generation=failing['lease_generation'],
                                        success=False,retryable=False,error='HTTP 404',now=1202)
        assert upgraded.retry_failed_deliveries(now=1203)==1
        restarted=upgraded.claim_delivery(now=1203)
        assert restarted['lease_generation'] > failing['lease_generation']
        assert restarted['attempts']==1
        assert not upgraded.finish_delivery(failing['id'],lease_generation=failing['lease_generation'],
                                            success=True,now=1204)
        print('  OK existing alarm database auto-upgrade; expired worker cannot overwrite new lease')


def test_privacy_downgrade_and_audit():
    with tempfile.TemporaryDirectory() as directory:
        store = WatchlistAlertStore(Path(directory)/'alerts.sqlite3')
        conf = {'ha_mobile_enabled': True, 'ha_persistent_enabled': False,
                'ha_mobile_service': 'mobile_app_one', 'ha_include_plate': False}
        store.record(event(2), now=time.time()-1, ha_channels=(('mobile','mobile_app_one'),),
                     ha_include_plate=True)
        opener = MockOpener()
        worker = HADeliveryWorker(store, lambda:conf,
                sender=HomeAssistantSender(token_getter=lambda:'testtoken', opener=opener))
        assert worker.deliver_once()
        assert len(opener.calls) == 1
        content = str(opener.calls[0][1])
        assert 'PRIVATE-123' not in content and 'PRIVATE-NOTE' not in content
        store.record(event(3), now=time.time()+1, ha_channels=(('mobile','mobile_app_one'),),
                     ha_include_plate=False, cooldown_seconds=0)
        conf['ha_include_plate'] = True
        # Consent granted later must never upgrade an old anonymized task.
        with sqlite3.connect(store.path) as c:
            c.execute('UPDATE ha_delivery_outbox SET next_attempt_at=0 WHERE state=\'pending\'')
        assert worker.deliver_once()
        assert 'PRIVATE-123' not in str(opener.calls[-1][1])
        records = store.recent_deliveries(limit=2000)
        assert len(records) == 2 and len(records) <= 50
        for rec in records:
            for secret in ('plate_text','normalized','label','target','include_plate','alert_id'):
                assert secret not in rec
            assert 'PRIVATE-123' not in str(rec) and 'mobile_app_one' not in str(rec)
        print('  OK queued data respects CURRENT privacy; HA audit contains no plate or phone name')


def test_serialized_delivery_change():
    class BlockingSender:
        def __init__(self): self.started=threading.Event(); self.release=threading.Event(); self.calls=0
        def available(self): return True
        def send_alert(self, row, *, include_plate):
            self.calls+=1; self.started.set()
            assert self.release.wait(5), 'test sender not released'
            return True
    with tempfile.TemporaryDirectory() as directory:
        store = WatchlistAlertStore(Path(directory)/'alerts.sqlite3')
        store.record(event(4), now=time.time()-1, ha_channels=(('persistent',''),))
        conf={'ha_persistent_enabled':True,'ha_mobile_enabled':False,'ha_include_plate':False}
        sender=BlockingSender()
        worker=HADeliveryWorker(store,lambda:conf,sender=sender)
        thread=threading.Thread(target=worker.deliver_once)
        thread.start(); assert sender.started.wait(5)
        changed=threading.Event()
        def change_settings():
            with worker.delivery_lock:
                conf['ha_persistent_enabled']=False
                store.cancel_pending_deliveries()
            changed.set()
        config_thread=threading.Thread(target=change_settings)
        config_thread.start()
        assert not changed.wait(.2), 'settings changed during an in-flight send'
        sender.release.set()
        thread.join(5); config_thread.join(5)
        assert changed.is_set() and sender.calls==1
        assert not worker.deliver_once()
        print('  OK notification sending and opt-out cannot overlap inside one process')


def test_indirect_config_block_and_safe_import():
    file = ROOT / 'platevision/src/app.py'
    tree = ast.parse(file.read_text(encoding='utf-8'))
    names={'_indirect_ha_settings_present','_ha_settings_snapshot', '_alert_delivery_active',
           '_cancel_delivery_after_settings_change','_persist_config_update',
           'api_save_config','api_save_advanced_config',
           'api_config_import','api_alerts_homeassistant_deliveries'}
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            node=copy.deepcopy(node); node.decorator_list=[]; nodes.append(node)
    assert len(nodes) == len(names)
    class Config:
        def __init__(self):
            self.lock=threading.RLock()
            self.DEFAULT_CONFIG={'alerts':{'ha_mobile_enabled':False,
                'ha_mobile_service':'','ha_include_plate':False,'ha_persistent_enabled':False},
                'ui':{'theme':'dark'}}
            self.config=copy.deepcopy(self.DEFAULT_CONFIG)
            self._committed_config=copy.deepcopy(self.config)
        def save_config(self):
            self._committed_config=copy.deepcopy(self.config); return True
        def get(self, *keys):
            value = self.config
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            return value
        def _normalize_analysis_area(self, value): return value
        def _merge_configs(self, defaults, payload):
            merged=copy.deepcopy(defaults)
            for section, values in payload.items():
                if isinstance(values,dict) and isinstance(merged.get(section),dict):
                    merged[section].update(values)
                else: merged[section]=values
            return merged
    class Req:
        def __init__(self):self.payload={};self.args={};self.json={}
        def get_json(self,silent=True):return self.payload
    cfg=Config(); req=Req()
    store=WatchlistAlertStore(Path(tempfile.mkdtemp())/'alerts.sqlite3')
    env={'config_manager':cfg, 'request':req, 'jsonify':lambda payload:payload,
         'ha_settings_from_config':ha_settings_from_config,
         'notification_enabled':notification_enabled,
         '_json_clone':copy.deepcopy, '_invalidate_model_cache':lambda:None,
         '_public_config':lambda:copy.deepcopy(cfg.config),
         '_read_limited_import_json':lambda:req.payload,
         'watchlist_alert_center':store,
         '_alert_store_or_error':lambda:(store,None),
         '_page_integer':lambda val, default, minimum=0, maximum=9999:
                 min(max(int(val) if val is not None else default,minimum),maximum),
         'sqlite3':sqlite3, 'OSError':OSError}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(file),'exec'),env)
    req.json={'alerts':{'ha_include_plate':True}}
    req.payload=req.json
    answer, code=env['api_save_config']()
    assert code == 400 and not cfg.config['alerts']['ha_include_plate']
    req.payload={'alerts':{'ha_mobile_enabled':True,'ha_mobile_service':'mobile_app_stranger'}}
    answer, code=env['api_save_advanced_config']()
    assert code == 400 and not cfg.config['alerts']['ha_mobile_enabled']
    cfg.config['alerts'].update(ha_mobile_enabled=True,ha_mobile_service='mobile_app_current')
    req.payload={'alerts':{'ha_mobile_enabled':False,'ha_mobile_service':'mobile_app_stranger',
                           'ha_include_plate':True},'ui':{'theme':'light'}}
    imported=env['api_config_import']()
    assert imported['success'] and imported['ha_notification_settings_preserved']
    assert cfg.config['alerts']['ha_mobile_enabled'] is True
    assert cfg.config['alerts']['ha_mobile_service']=='mobile_app_current'
    assert cfg.config['alerts']['ha_include_plate'] is False
    assert cfg.config['ui']['theme']=='light'
    req.args={'limit':'5'}
    log=env['api_alerts_homeassistant_deliveries']()
    assert log['success'] and log['items']==[]
    assert set(log) == {'success','items','status'}
    print('  OK generic config endpoints block HA opt-in; import keeps current HA choice; audit API safe')


def test_reset_cancels_queued_notifications():
    with tempfile.TemporaryDirectory() as directory:
        store=WatchlistAlertStore(Path(directory)/'alarms.sqlite3')
        store.record(event(220),now=time.time()-1,ha_channels=(('persistent',''),))
        assert store.delivery_status()['pending']==1
        config={'ha_persistent_enabled':True, 'ha_mobile_enabled':False,
                'ha_include_plate':False, 'ha_mobile_service':''}
        old=dict(config)
        config['ha_persistent_enabled']=False
        file=ROOT/'platevision/src/app.py'
        tree=ast.parse(file.read_text())
        node=next(copy.deepcopy(n) for n in tree.body
                  if isinstance(n,ast.FunctionDef) and
                  n.name=='_cancel_delivery_after_settings_change')
        env={'_ha_settings_snapshot':lambda:dict(config),
             'watchlist_alert_center':store}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                     str(file),'exec'),env)
        env['_cancel_delivery_after_settings_change'](old)
        assert store.delivery_status()['pending']==0
        assert store.status()['total']==1, 'local alert must survive reset'
        print('  OK config reset clears old outbound tasks without deleting local alerts')


if __name__ == '__main__':
    test_lease_fencing_and_upgrade()
    test_privacy_downgrade_and_audit()
    test_serialized_delivery_change()
    test_indirect_config_block_and_safe_import()
    test_reset_cancels_queued_notifications()
    print('PlateVision HA hardening 0.13.7: OK')
