#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""0.13.8 regressions: notification config/sender races and private queue health."""
import ast
import copy
import logging
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'platevision' / 'src'))
from watchlist_alerts import WatchlistAlertStore
from ha_notifications import (HADeliveryWorker, ha_settings_from_config,
                              channels_for_settings, valid_mobile_service)


def event(event_id):
    return {'id': event_id, 'plate_text': 'SECRET-123',
            'plate_text_normalized': 'SECRET123', 'confidence': .96,
            'source': 'rtsp', 'watchlist_match': {'id': 'watch1', 'notify': True}}


class Settings:
    def __init__(self):
        self.lock = threading.RLock()
        self.config = {'alerts': {'ha_persistent_enabled': False,
                                  'ha_mobile_enabled': True,
                                  'ha_mobile_service': 'mobile_app_old',
                                  'ha_include_plate': False,
                                  'watchlist_notifications': True,
                                  'min_alert_confidence': .65,
                                  'watchlist_alert_cooldown_seconds': 180}}
        self.saved = copy.deepcopy(self.config)

    def save_config(self):
        self.saved = copy.deepcopy(self.config)
        return True

    def _normalize_analysis_area(self, value):
        return value

    def get(self, *keys):
        current=self.config
        for key in keys:
            current=current.get(key) if isinstance(current,dict) else None
        return current


class BlockingSender:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def available(self):
        return True

    def send_alert(self, row, *, include_plate=False):
        self.started.set()
        assert self.release.wait(5), 'Blocked HA sender was not released'
        self.calls.append((row['target'], include_plate))
        return True


def handler_env(config, store, worker):
    source = ROOT / 'platevision/src/app.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    names = {'_ha_settings_snapshot', '_alert_delivery_active', '_cancel_delivery_after_settings_change', '_persist_config_update',
             'api_alerts_homeassistant', 'api_reset_config', 'api_alerts_settings'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node = copy.deepcopy(node)
            node.decorator_list = []
            nodes.append(node)
    assert len(nodes) == len(names)
    request = SimpleNamespace(method='POST', content_length=300)
    request.get_json = lambda silent=True: request.payload
    env = {'config_manager': config, 'request': request, 'ha_delivery_worker': worker,
           'watchlist_alert_center': store, 'jsonify': lambda value: value,
           'ha_settings_from_config': ha_settings_from_config,
           'notification_enabled': lambda value: value is not False, '_json_clone': copy.deepcopy,
           'valid_mobile_service': valid_mobile_service, 'sqlite3': sqlite3,
           '_safe_numeric': lambda value,default: value if value is not None else default,
           '_page_integer': lambda value,default,minimum=0,maximum=3600:
                min(max(default if value is None else int(value),minimum),maximum),
           'logger': logging.getLogger('platevision-0138'),
           '_config_defaults_copy': lambda: {'alerts': {
               'ha_persistent_enabled': False, 'ha_mobile_enabled': False,
               'ha_mobile_service': '', 'ha_include_plate': False}},
           '_invalidate_model_cache': lambda: None,
           '_public_config': lambda: copy.deepcopy(config.config)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                 str(source), 'exec'), env)
    return env, request


def test_setting_change_waits_for_delivery():
    with tempfile.TemporaryDirectory() as td:
        store = WatchlistAlertStore(Path(td)/'alerts.db')
        config = Settings()
        store.record(event('first'), now=time.time()-3,
                     ha_channels=(('mobile', 'mobile_app_old'),), cooldown_seconds=0)
        store.record(event('second'), now=time.time()-2,
                     ha_channels=(('mobile', 'mobile_app_old'),), cooldown_seconds=0)
        assert store.delivery_status()['pending'] == 2
        sender = BlockingSender()
        worker = HADeliveryWorker(store,
              lambda: ha_settings_from_config(config.config['alerts']), sender=sender)
        env, req = handler_env(config, store, worker)
        running = threading.Thread(target=worker.deliver_once)
        running.start()
        assert sender.started.wait(4)
        req.payload = {'ha_persistent_enabled': False,
                       'ha_mobile_enabled': True,
                       'ha_mobile_service': 'mobile_app_new',
                       'ha_include_plate': False}
        changed = threading.Event()
        response = []
        def update():
            response.append(env['api_alerts_homeassistant']())
            changed.set()
        update_thread = threading.Thread(target=update)
        update_thread.start()
        assert not changed.wait(.2), 'Configuration was changed during an in-flight send'
        assert config.config['alerts']['ha_mobile_service'] == 'mobile_app_old'
        sender.release.set()
        running.join(5); update_thread.join(5)
        assert not running.is_alive() and not update_thread.is_alive()
        assert changed.is_set() and response[0]['success']
        assert config.config['alerts']['ha_mobile_service'] == 'mobile_app_new'
        assert sender.calls == [('mobile_app_old', False)]
        stats = store.delivery_status()
        assert stats['pending'] == 0 and stats['accepted'] == 1, stats
        print('  OK smartphone target change waits for in-flight delivery and cancels stale queued tasks')


def test_reset_waits_for_delivery():
    with tempfile.TemporaryDirectory() as td:
        store = WatchlistAlertStore(Path(td)/'alerts.db')
        config = Settings()
        store.record(event('reset-1'), now=time.time()-3,
                     ha_channels=(('mobile', 'mobile_app_old'),))
        store.record(event('reset-2'), now=time.time()-2,
                     ha_channels=(('mobile', 'mobile_app_old'),), cooldown_seconds=0)
        sender=BlockingSender()
        worker=HADeliveryWorker(store,
              lambda: ha_settings_from_config(config.config['alerts']), sender=sender)
        env, req=handler_env(config, store, worker)
        running = threading.Thread(target=worker.deliver_once)
        running.start()
        assert sender.started.wait(4)
        changed = threading.Event()
        result = []
        def reset():
            result.append(env['api_reset_config']())
            changed.set()
        reset_thread=threading.Thread(target=reset)
        reset_thread.start()
        assert not changed.wait(.2), 'Config reset crossed a network delivery'
        sender.release.set()
        running.join(5); reset_thread.join(5)
        assert changed.is_set() and result[0]['success']
        assert not config.config['alerts']['ha_mobile_enabled']
        assert store.delivery_status()['pending'] == 0
        assert store.status()['total'] >= 1
        print('  OK full settings reset waits for delivery, preserves local alarms and cancels outbound tasks')


def test_local_alert_opt_out_cancels_ha_queue():
    with tempfile.TemporaryDirectory() as td:
        store=WatchlistAlertStore(Path(td)/'alerts.db')
        config=Settings()
        store.record(event('off-1'),now=time.time()-3,
                     ha_channels=(('mobile','mobile_app_old'),),cooldown_seconds=0)
        store.record(event('off-2'),now=time.time()-2,
                     ha_channels=(('mobile','mobile_app_old'),),cooldown_seconds=0)
        sender=BlockingSender()
        worker=HADeliveryWorker(store,
             lambda:ha_settings_from_config(config.config['alerts']),sender=sender)
        env,request=handler_env(config,store,worker)
        sender_thread=threading.Thread(target=worker.deliver_once)
        sender_thread.start(); assert sender.started.wait(4)
        request.payload={'enabled':False,'min_confidence':.65,'cooldown_seconds':180}
        done=threading.Event()
        outcome=[]
        def disable():
            outcome.append(env['api_alerts_settings']())
            done.set()
        change_thread=threading.Thread(target=disable)
        change_thread.start()
        assert not done.wait(.2), 'Local alarm opt-out raced an in-flight push'
        sender.release.set()
        sender_thread.join(5);change_thread.join(5)
        assert done.is_set() and outcome[0]['success']
        assert config.get('alerts','watchlist_notifications') is False
        assert store.delivery_status()['pending']==0
        assert store.status()['total']==2, 'Local inbox must be preserved'
        print('  OK disabling watchlist messages blocks an in-flight sender then cancels outstanding HA tasks')


def test_queue_health_private_and_recoverable():
    with tempfile.TemporaryDirectory() as td:
        store=WatchlistAlertStore(Path(td)/'alerts.db')
        empty=store.delivery_status()
        assert empty['oldest_waiting_seconds'] is None and empty['expired_leases']==0
        assert empty['last_accepted_at'] is None
        store.record(event('health'), now=time.time()-1800,
                     ha_channels=(('mobile','mobile_app_secret'),))
        data=store.delivery_status()
        assert data['pending']==1 and data['oldest_waiting_seconds']>=1700
        claimed=store.claim_delivery(now=time.time()-45,lease_seconds=5)
        assert claimed['channel']=='mobile'
        data=store.delivery_status()
        assert data['expired_leases']==1 and data['sending']==1
        # An expired lease is recoverable, and the older worker is fenced off.
        newer=store.claim_delivery(now=time.time(),lease_seconds=30)
        assert newer['lease_generation'] > claimed['lease_generation']
        assert not store.finish_delivery(claimed['id'],lease_generation=claimed['lease_generation'],
                                         success=True)
        assert store.finish_delivery(newer['id'],lease_generation=newer['lease_generation'],
                                     success=True)
        accepted=store.delivery_status()
        assert accepted['accepted']==1 and accepted['last_accepted_at'] is not None
        assert accepted['oldest_waiting_seconds'] is None
        for secret in ('SECRET-123','mobile_app_secret','watch1'):
            assert secret not in repr(accepted), f'Queue health leaked {secret}'
        print('  OK queue health detects stale leases/backlog, recovers safely and does not expose plate or recipient')


if __name__=='__main__':
    test_setting_change_waits_for_delivery()
    test_reset_waits_for_delivery()
    test_local_alert_opt_out_cancels_ha_queue()
    test_queue_health_private_and_recoverable()
    print('PlateVision HA queue and diagnostics 0.13.8: OK')
