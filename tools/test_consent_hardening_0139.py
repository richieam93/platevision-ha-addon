#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""0.13.9 offline: consent races, generic settings, fail-closed outbox and retries.

Isolated AST execution avoids importing Flask, YOLO or downloading models.
Actual SQLite transactions are exercised. No network calls leave this test.
"""
from __future__ import annotations

import ast
import copy
import json
import logging
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'platevision/src'))
from ha_notifications import (HADeliveryWorker, channels_for_settings,
                              ha_settings_from_config)
from watchlist_alerts import WatchlistAlertStore, notification_enabled


class Config:
    DEFAULT_CONFIG = {
        'alerts': {'watchlist_notifications': True, 'min_alert_confidence': .65,
                   'watchlist_alert_cooldown_seconds': 0,
                   'ha_persistent_enabled': False, 'ha_mobile_enabled': False,
                   'ha_mobile_service': '', 'ha_include_plate': False},
        'plate_recognition': {'watchlist_enabled': True},
        'ui': {'theme': 'dark'},
    }

    def __init__(self):
        self.lock = threading.RLock()
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.config['alerts'].update(ha_mobile_enabled=True,
                                     ha_mobile_service='mobile_app_old')
        self._committed_config = copy.deepcopy(self.config)
        self.fail_save = False

    def get(self, *keys):
        value = self.config
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        return value

    def save_config(self):
        if self.fail_save:
            self.config = copy.deepcopy(self._committed_config)
            return False
        self._committed_config = copy.deepcopy(self.config)
        return True

    def _merge_configs(self, defaults, payload):
        out = copy.deepcopy(defaults)
        for section, values in payload.items():
            if isinstance(out.get(section), dict) and isinstance(values, dict):
                out[section].update(copy.deepcopy(values))
            else:
                out[section] = copy.deepcopy(values)
        return out

    def _normalize_analysis_area(self, config):
        return config


class Request:
    method = 'POST'
    content_length = 100
    args = {}

    def __init__(self):
        self.payload = {}

    def get_json(self, silent=True):
        return self.payload


class RecordingSender:
    def __init__(self):
        self.sent = []

    def available(self):
        return True

    def send_alert(self, row, *, include_plate=False):
        self.sent.append((row['target'], include_plate))


def sample(eid):
    return {'id': eid, 'plate_text': 'PRIVATE-123',
            'plate_text_normalized': 'PRIVATE123', 'confidence': .94,
            'source': 'rtsp', 'watchlist_match': {'id': 'watch-1',
                                               'notify': True, 'label': 'PRIVATE NOTE'}}


def invoke(result):
    return result if isinstance(result, tuple) else (result, 200)


def environment(store, config=None):
    config = config or Config()
    request = Request()
    logger = logging.getLogger('platevision-consent-0139')
    symbols = {
        '_indirect_ha_settings_present', '_ha_settings_snapshot',
        '_ha_worker_settings', '_alert_delivery_active',
        '_cancel_delivery_after_settings_change', '_persist_config_update',
        '_config_defaults_copy', '_apply_default_scope',
        'api_reset_config', 'api_config_defaults',
        '_record_watchlist_notification', 'api_save_config',
        'api_save_advanced_config', 'api_config_import',
        'api_alerts_settings', 'api_alerts_homeassistant_retry',
    }
    path = ROOT / 'platevision/src/app.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in symbols:
            node = copy.deepcopy(node)
            node.decorator_list = []
            nodes.append(node)
    assert len(nodes) == len(symbols), symbols - {node.name for node in nodes}
    env = {
        'config_manager': config, 'watchlist_alert_center': store,
        'request': request, 'json': json, 'jsonify': lambda obj: obj, 'threading': threading,
        'logger': logger, 'sqlite3': sqlite3, 'notification_enabled': notification_enabled,
        'ha_settings_from_config': ha_settings_from_config,
        'channels_for_settings': channels_for_settings,
        '_json_clone': copy.deepcopy,
        '_safe_numeric': lambda value, default: default if value is None else float(value),
        '_page_integer': lambda value, default, minimum=0, maximum=3600:
            min(max(int(value if value is not None else default), minimum), maximum),
        '_public_config': lambda: copy.deepcopy(config.config),
        '_invalidate_model_cache': lambda: None,
        '_queue_model_reload': lambda: None, '_scan_model_files': lambda: [],
        '_read_limited_import_json': lambda: copy.deepcopy(request.payload),
        '_alert_store_or_error': lambda: (store, None),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), 'exec'), env)
    sender = RecordingSender()
    worker = HADeliveryWorker(store, env['_ha_worker_settings'], sender=sender)
    env['ha_delivery_worker'] = worker
    return env, request, config, sender, worker


def test_capture_vs_global_opt_out():
    """A paused INSERT must not sneak a notification in after opt-out."""
    with tempfile.TemporaryDirectory() as directory:
        store = WatchlistAlertStore(Path(directory) / 'alerts.db')
        env, req, cfg, sender, worker = environment(store)
        entered, release = threading.Event(), threading.Event()
        original = store.record
        def delayed_record(*args, **kwargs):
            entered.set()
            assert release.wait(5), 'record guard not released'
            return original(*args, **kwargs)
        store.record = delayed_record
        results = []
        capture = threading.Thread(target=lambda: results.append(
            env['_record_watchlist_notification'](sample('capture-before-optout'), store)))
        capture.start()
        assert entered.wait(4)
        req.payload = {'alerts': {'watchlist_notifications': False}}
        changed = threading.Event()
        outcome = []
        def disable():
            outcome.append(invoke(env['api_save_config']()))
            changed.set()
        setter = threading.Thread(target=disable)
        setter.start()
        assert not changed.wait(.15), 'setting write crossed unfinished outbox INSERT'
        assert cfg.get('alerts', 'watchlist_notifications') is True
        release.set()
        capture.join(6)
        setter.join(6)
        assert not capture.is_alive() and not setter.is_alive()
        assert outcome[0][1] == 200 and outcome[0][0]['success']
        assert len(results) == 1 and results[0]['created']
        assert store.status()['total'] == 1
        assert store.delivery_status()['pending'] == 0
        assert worker.deliver_once() is False and not sender.sent
        print('  OK: simultaneous detection and general opt-out never leave an old push')


def test_import_disable_and_reenable():
    with tempfile.TemporaryDirectory() as directory:
        store = WatchlistAlertStore(Path(directory) / 'alerts.db')
        env, req, cfg, sender, worker = environment(store)
        old_target = cfg.get('alerts', 'ha_mobile_service')
        assert env['_record_watchlist_notification'](sample('one'), store)['created']
        assert store.delivery_status()['pending'] == 1
        req.payload = {'alerts': {'watchlist_notifications': False,
                                  'ha_mobile_enabled': False,
                                  'ha_mobile_service': 'mobile_app_attacker',
                                  'ha_include_plate': True}, 'ui': {'theme': 'light'}}
        outcome, status = invoke(env['api_config_import']())
        assert status == 200 and outcome['ha_notification_settings_preserved']
        assert cfg.get('alerts', 'ha_mobile_service') == old_target
        assert cfg.get('alerts', 'ha_mobile_enabled') is True
        assert cfg.get('alerts', 'ha_include_plate') is False
        assert cfg.get('alerts', 'watchlist_notifications') is False
        assert cfg.get('ui', 'theme') == 'light'
        assert store.status()['total'] == 1
        assert store.delivery_status()['pending'] == 0
        # Even a pre-existing outbox from 0.13.8 cannot escape an opt-out.
        store.record(sample('stale'), now=time.time()-2, cooldown_seconds=0,
                     ha_channels=(('mobile', old_target),))
        assert store.delivery_status()['pending'] == 1
        assert worker.deliver_once() is False and not sender.sent
        # Enabling the global switch must also clean up stale legacy tasks.
        req.payload = {'alerts': {'watchlist_notifications': True}}
        result, code = invoke(env['api_save_advanced_config']())
        assert code == 200 and result['success']
        assert store.delivery_status()['pending'] == 0
        assert worker.deliver_once() is False
        print('  OK: import preserves phone consent; opt-out and later opt-in discard old jobs')


def test_watchlist_matching_disable_and_retry_gate():
    with tempfile.TemporaryDirectory() as directory:
        store = WatchlistAlertStore(Path(directory) / 'alerts.db')
        env, req, cfg, sender, worker = environment(store)
        old_target = cfg.get('alerts', 'ha_mobile_service')
        env['_record_watchlist_notification'](sample('before'), store)
        assert store.delivery_status()['pending'] == 1
        req.payload = {'plate_recognition': {'watchlist_enabled': False}}
        response, code = invoke(env['api_save_config']())
        assert code == 200 and response['success']
        assert store.delivery_status()['pending'] == 0
        assert env['_record_watchlist_notification'](sample('after'), store) is None
        # Existing failed job cannot be manually retried while disabled.
        store.record(sample('failed'), now=time.time()-2, cooldown_seconds=0,
                     ha_channels=(('mobile', old_target),))
        delivery = store.claim_delivery(now=time.time()+1)
        assert delivery is not None
        store.finish_delivery(delivery['id'], lease_generation=delivery['lease_generation'],
                              success=False, retryable=False, error='HTTP 404')
        req.payload = {}
        response, code = invoke(env['api_alerts_homeassistant_retry']())
        assert code == 400 and response['success'] is False
        assert store.delivery_status()['failed'] == 1
        assert worker.deliver_once() is False and not sender.sent
        print('  OK: disabling watchlist matching suppresses send and manual retries')


def test_failed_changes_rollback_and_do_not_opt_in():
    with tempfile.TemporaryDirectory() as directory:
        store = WatchlistAlertStore(Path(directory) / 'alerts.db')
        env, req, cfg, sender, worker = environment(store)
        old = copy.deepcopy(cfg.config)
        req.payload = {'alerts': {'watchlist_notifications': False},
                       'general': ['not-a-dict']}
        response, code = invoke(env['api_save_advanced_config']())
        assert code == 400 and not response['success']
        assert cfg.config == old
        req.payload = {'alerts': {'ha_mobile_enabled': True}}
        response, code = invoke(env['api_save_config']())
        assert code == 400 and not response['success']
        assert cfg.config == old
        env['_record_watchlist_notification'](sample('pending'), store)
        assert store.delivery_status()['pending'] == 1
        original = store.cancel_pending_deliveries
        def fail_purge():
            raise sqlite3.OperationalError('simulated storage failure')
        store.cancel_pending_deliveries = fail_purge
        req.payload = {'alerts': {'watchlist_notifications': False}}
        response, code = invoke(env['api_save_config']())
        assert code == 503 and not response['success']
        assert cfg.config == old and store.delivery_status()['pending'] == 1
        store.cancel_pending_deliveries = original
        cfg.fail_save = True
        response, code = invoke(env['api_save_config']())
        assert code == 503 and not response['success']
        assert cfg.config == old
        assert store.delivery_status()['pending'] == 0  # privacy-first before failing JSON commit
        print('  OK: invalid edits/disk errors roll back, failed queue purge blocks opt-out')



def test_reset_defaults_never_commit_when_queue_cleanup_fails():
    """A failed SQLite cleanup must not persist reset/default consent changes."""
    with tempfile.TemporaryDirectory() as directory:
        store = WatchlistAlertStore(Path(directory) / 'alerts.db')
        env, req, cfg, sender, worker = environment(store)
        env['_record_watchlist_notification'](sample('before-reset'), store)
        assert store.delivery_status()['pending'] == 1
        original = copy.deepcopy(cfg.config)
        original_purge = store.cancel_pending_deliveries
        def fail_cleanup():
            raise sqlite3.OperationalError('simulated outbox cleanup failure')
        store.cancel_pending_deliveries = fail_cleanup
        for route, payload in (
            ('api_reset_config', {}),
            ('api_config_defaults', {'scope': 'global', 'reload_models': False}),
        ):
            req.payload = payload
            response, status = invoke(env[route]())
            assert status == 503 and not response['success'], (route, response, status)
            assert cfg.config == original, route
            assert store.delivery_status()['pending'] == 1, route
        store.cancel_pending_deliveries = original_purge
        req.payload = {'scope': 'global', 'reload_models': False}
        response, status = invoke(env['api_config_defaults']())
        assert status == 200 and response['success']
        assert cfg.get('alerts', 'ha_mobile_enabled') is False
        assert store.delivery_status()['pending'] == 0
        assert store.status()['total'] == 1  # Local history survives
        print('  OK: reset/defaults fail closed on SQLite cleanup and preserve local alarms')


def test_due_vs_backoff_monitor():
    with tempfile.TemporaryDirectory() as directory:
        store = WatchlistAlertStore(Path(directory) / 'alerts.db')
        event = sample('retry')
        store.record(event, now=time.time()-1200,
                     ha_channels=(('persistent', ''),))
        claimed = store.claim_delivery(now=time.time()-1199)
        assert claimed is not None
        # Retry is deliberately set for a later time, not an overdue message.
        store.finish_delivery(claimed['id'], lease_generation=claimed['lease_generation'],
                              success=False, retryable=True, error='connection error',
                              now=time.time()+250)
        state = store.delivery_status()
        assert state['pending'] == 1 and state['due_now'] == 0, state
        assert state['oldest_due_seconds'] is None and state['oldest_waiting_seconds'] > 900
        assert state['next_retry_at'] is not None and state['next_retry_at'] > time.time()
        with sqlite3.connect(store.path) as conn:
            conn.execute('UPDATE ha_delivery_outbox SET next_attempt_at=?', (time.time()-5,))
        state = store.delivery_status()
        assert state['due_now'] == 1 and state['oldest_due_seconds'] > 900
        assert state['next_retry_at'] is None
        for hidden in ('PRIVATE-123', 'mobile_app_old', 'PRIVATE NOTE'):
            assert hidden not in str(state)
        print('  OK: scheduled retry is not mislabeled overdue; privacy-safe next retry')


if __name__ == '__main__':
    test_capture_vs_global_opt_out()
    test_import_disable_and_reenable()
    test_watchlist_matching_disable_and_retry_gate()
    test_failed_changes_rollback_and_do_not_opt_in()
    test_reset_defaults_never_commit_when_queue_cleanup_fails()
    test_due_vs_backoff_monitor()
    print('PlateVision 0.13.9 consent, recovery and due-monitor regressions: OK')
