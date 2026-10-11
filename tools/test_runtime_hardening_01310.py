#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""0.13.10: SQLite write amplification, RTSP persistence, reconnect and privacy regressions.

Uses the real persistence and RTSP classes; no camera, Flask server or ML weights.
"""
import ast
import copy
import os
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'platevision/src'))
from history_sqlite import SQLiteHistoryStore
import rtsp_handler


def test_sqlite_sync_writes_only_changed_rows():
    with tempfile.TemporaryDirectory(prefix='pv-01310-sqlite-') as tmp:
        path = Path(tmp) / 'detections.json'
        original = [
            {'id': key, 'timestamp': f'2026-10-11T10:00:0{i}', 'plate_text': f'ZH {i}',
             'confidence': .9, 'source': 'rtsp'}
            for i, key in enumerate(('first', 'middle', 'last'))
        ]
        path.write_text(json.dumps(original), encoding='utf-8')
        store = SQLiteHistoryStore(path, kind='plates')
        assert store.initialize(lambda: original) == original
        with sqlite3.connect(store.path) as conn:
            conn.executescript('''
                CREATE TABLE test_row_updates(id TEXT);
                CREATE TRIGGER audit_event_update AFTER UPDATE ON events
                    BEGIN INSERT INTO test_row_updates VALUES (NEW.event_key); END;
            ''')
        # Persisting the same state must not rewrite every large JSON/BLOB payload.
        store.sync([dict(row) for row in original])
        with sqlite3.connect(store.path) as conn:
            count = conn.execute('SELECT COUNT(*) FROM test_row_updates').fetchone()[0]
        assert count == 0, f'Unmodified SQLite records rewritten: {count}'
        edited = [dict(row) for row in original]
        edited[1]['review_note'] = 'Neu bewertet, aber gleiche Suchfelder'
        store.sync(edited)
        with sqlite3.connect(store.path) as conn:
            changed = conn.execute('SELECT id FROM test_row_updates').fetchall()
        assert changed == [('middle',)], f'Unexpected row updates: {changed}'
        assert store.get('middle')['review_note'] == edited[1]['review_note']
        # Reordering moves all old ranks into a temporary range; restore even
        # rows whose final rank equals their original one.
        reverse = list(reversed(edited))
        store.sync(reverse)
        assert [row['id'] for row in store.load()] == ['last', 'middle', 'first']
        with sqlite3.connect(store.path) as conn:
            ranks = conn.execute('SELECT event_key,rank FROM events ORDER BY rank DESC').fetchall()
        assert [key for key, rank in ranks] == ['last', 'middle', 'first']
        assert [rank for key, rank in ranks] == [3, 2, 1]
        print('  OK SQLite: no needless JSON rewrites, single-row edits, exact rank reorder')


class SimpleConfig:
    def __init__(self, url='rtsp://admin:very-secret@192.0.2.50:554/stream?token=private'):
        self.data = {'rtsp': {'url': url, 'reconnect_delay': 0.2},
                     'history': {'filter_duplicates': True, 'duplicate_timeout': 30},
                     'people': {'enabled': False}}

    def get(self, *path):
        value = self.data
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        return value


class FailingThenSavingHistory:
    def __init__(self):
        self.calls = 0
        self.rows = []

    def add_entry(self, entry, check_duplicate=True):
        self.calls += 1
        if self.calls == 1:
            raise OSError('disk full')
        self.rows.append(entry.copy())
        return entry.copy()


def test_rtsp_only_caches_committed_detections():
    history = FailingThenSavingHistory()
    stream = rtsp_handler.RTSPHandler(SimpleConfig(), history, None)
    plate = {'plate_text': 'ZH 123', 'confidence': .96}
    try:
        stream._handle_detection(plate, {})
    except OSError:
        pass
    else:
        raise AssertionError('A failed history write must reach the caller')
    assert stream.recent_plates == {}, 'Failed save blocked the next valid detection'
    stream._handle_detection(plate, {})
    assert history.calls == 2 and len(history.rows) == 1
    stream._handle_detection(plate, {})
    assert history.calls == 2, 'Committed detection not deduplicated'
    assert stream.detection_count == 1
    # Older imported settings may contain unusable timeout values.
    stream.config_manager.data['history']['duplicate_timeout'] = 'not-a-number'
    stream._handle_detection({'plate_text': 'BE 9', 'confidence': .8}, {})
    assert history.calls == 3
    print('  OK RTSP: failed SQLite write never poisons duplicate cache')


def test_rtsp_scrubs_capture_errors_and_resets_motion_baseline():
    config = SimpleConfig()
    stream = rtsp_handler.RTSPHandler(config, None, None)
    hidden_url = config.get('rtsp', 'url')
    disguised = 'rtsps://other:another-password@192.0.2.50:8554/private?token=other-secret'
    safe = stream._safe_error_text('Capture error: ' + disguised)
    assert disguised not in safe
    assert 'another-password' not in safe and 'other-secret' not in safe
    assert 'rtsps://192.0.2.50:8554' in safe

    class FailingCapture:
        def read(self):
            stream.running = False
            stream._stop_event.set()
            raise RuntimeError('Unexpected RTSP failure: ' + hidden_url)

        def release(self):
            return None

    stream.cap = FailingCapture()
    stream.connected = True
    stream.running = True
    stream._motion_prev_gray = object()
    stream._motion_prev_shape = (20, 20)
    stream._last_analysis_run_time = 99
    with patch.object(rtsp_handler.logger, 'error') as logger_error:
        stream._capture_loop()
        logs = ' '.join(map(str, logger_error.call_args.args))
    assert 'very-secret' not in logs and 'token=private' not in logs, logs
    assert stream._motion_prev_gray is None and stream._motion_prev_shape is None
    assert stream._last_analysis_run_time == 0
    print('  OK RTSP: camera credentials redacted, reconnection clears motion gate baseline')


def test_camera_heartbeat_and_reconnect_counters():
    import numpy as np
    stream = rtsp_handler.RTSPHandler(SimpleConfig(), None, None)

    class SyntheticCapture:
        def set(self, *_):
            return True

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((24, 32, 3), dtype=np.uint8)

        def release(self):
            return None

    with patch.object(rtsp_handler.cv2, 'VideoCapture', return_value=SyntheticCapture()):
        with patch.object(rtsp_handler.time, 'monotonic', return_value=100.0):
            assert stream.connect()
        with patch.object(rtsp_handler.time, 'monotonic', return_value=120.1):
            status = stream.get_status()
        assert status['reconnect_count'] == 0
        assert status['last_frame_age_seconds'] == 20.1
        with patch.object(rtsp_handler.time, 'monotonic', return_value=125.0):
            assert stream.connect()
    assert stream.get_status()['reconnect_count'] == 1
    stream.disconnect()
    assert stream.get_status()['reconnect_count'] == 1
    print('  OK Camera: last real frame timestamp and successful reconnections')


def test_stale_camera_health_warning():
    # Real Flask route body, with mocked dependencies but no Flask/ML import.
    app = ROOT / 'platevision/src/app.py'
    tree = ast.parse(app.read_text(encoding='utf-8'), filename=str(app))
    names = {'_health_summary', 'api_system_health'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node = copy.deepcopy(node)
            node.decorator_list = []
            nodes.append(node)
    assert len(nodes) == 2
    with tempfile.TemporaryDirectory(prefix='pv-01310-health-') as temp:
        base = Path(temp)
        (base/'vehicle.pt').write_bytes(b'fake')
        (base/'plate.pt').write_bytes(b'fake')
        cfg = {'models': {'vehicle_detector': 'vehicle.pt',
                          'license_plate_detector': 'plate.pt'},
               'people': {'enabled': False},
               'rtsp': {'enabled': True, 'url': 'rtsp://camera.example/stream'}}

        class FakeConfig:
            DEFAULT_CONFIG = {'rtsp': {'url': 'rtsp://placeholder'}}

            def get(self, section, key=None):
                return cfg[section] if key is None else cfg[section].get(key)

        fake_status = {'connected': True, 'fps': 0, 'frame_count': 88,
                       'detection_count': 5, 'last_frame_age_seconds': 25.1,
                       'reconnect_count': 4}
        stream = SimpleNamespace(get_status=lambda: fake_status)
        manager = lambda: SimpleNamespace(HISTORY_FILE=str(base/'history.json'),
                                          _sqlite_store=None)
        env = {
            'config_manager': FakeConfig(), 'ConfigManager': FakeConfig,
            'stream_manager': stream, 'history_manager': manager(),
            'person_history_manager': manager(),
            'watchlist_manager': SimpleNamespace(WATCHLIST_FILE=str(base/'watchlist.json')),
            '_resolve_model_path': lambda name: str(base/name),
            'os': SimpleNamespace(path=SimpleNamespace(isdir=lambda path: True,
                                                       dirname=os.path.dirname),
                                  access=lambda *args: True, W_OK=os.W_OK,
                                  environ={}),
            'Path': Path, '_read_only_storage': {},
            '_safe_numeric': lambda value, default=0: float(value) if value is not None else default,
            'channels_for_settings': lambda settings: (),
            '_ha_settings_snapshot': lambda: {},
            '_alert_delivery_active': lambda: True,
            'watchlist_alert_center': SimpleNamespace(delivery_status=lambda: {
                'pending': 0, 'sending': 0, 'accepted': 0, 'failed': 0,
                'oldest_waiting_seconds': None, 'oldest_due_seconds': None,
                'expired_leases': 0, 'due_now': 0}),
            'APP_VERSION': '0.13.10', 'APP_LICENSE': 'AGPL-3.0-only',
            'APP_SOURCE_URL': 'https://example.test',
            'jsonify': lambda value: value,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     str(app), 'exec'), env)
        report = env['api_system_health']()
        assert not report['ok']
        assert 'stream_frames_fresh' in report['summary']['problem_checks']
        assert report['camera']['reconnect_count'] == 4
        assert report['camera']['last_frame_age_seconds'] == 25.1
        fake_status['last_frame_age_seconds'] = 3.0
        report = env['api_system_health']()
        assert report['ok']
        cfg['rtsp']['enabled'] = False
        fake_status['last_frame_age_seconds'] = 999
        report = env['api_system_health']()
        assert report['ok'] and 'stream_frames_fresh' not in {
            row['name'] for row in report['checks']}
    print('  OK Health: frozen RTSP frames warn; fresh/disabled cameras remain healthy')


if __name__ == '__main__':
    test_sqlite_sync_writes_only_changed_rows()
    test_rtsp_only_caches_committed_detections()
    test_rtsp_scrubs_capture_errors_and_resets_motion_baseline()
    test_camera_heartbeat_and_reconnect_counters()
    test_stale_camera_health_warning()
    print('PlateVision runtime hardening 0.13.10: OK')
