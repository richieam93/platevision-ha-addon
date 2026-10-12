#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""0.13.13 regression: RTSP generation, capture-age and config write failures.

All tests use the real handler/API method with fake local camera data; no RTSP
credentials, model weights, or hardware are required.
"""
import ast
import copy
import logging
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'platevision/src'))
from rtsp_handler import RTSPHandler


class Config:
    def __init__(self):
        self.config = {
            'rtsp': {'url': 'rtsp://fake-camera.local/test', 'analysis_area': {},
                     'resolution': {'width': 64, 'height': 48}},
            'detection': {'process_interval': 0.01},
            'history': {'duplicate_timeout': 60, 'filter_duplicates': True},
            'people': {'enabled': False}, 'storage': {},
        }
        self.committed = copy.deepcopy(self.config)
        self.can_save = True

    def get(self, *path):
        result = self.config
        for key in path:
            result = result.get(key) if isinstance(result, dict) else None
        return result

    def _normalize_analysis_area(self, value):
        return value

    def save_config(self):
        if self.can_save:
            self.committed = copy.deepcopy(self.config)
            return True
        # The real ConfigManager restores the last-known-good state.
        self.config = copy.deepcopy(self.committed)
        return False


class History:
    def __init__(self):
        self.items = []
    def add_entry(self, row, **kwargs):
        self.items.append(row)
        return row


def run_processing(stream, payload):
    stream.running = True
    stream.frame_queue.put_nowait(payload)
    worker = threading.Thread(target=stream._process_loop, daemon=True)
    worker.start()
    return worker


def stop_processing(stream, worker):
    stream.running = False
    stream._stop_event.set()
    worker.join(timeout=5)
    assert not worker.is_alive(), 'Worker did not stop'


def test_reconnect_invalidates_inflight_analysis_and_detections():
    entered, unblock = threading.Event(), threading.Event()
    class SlowModel:
        models_loaded = True
        def process_frame(self, frame, **_):
            entered.set()
            assert unblock.wait(4), 'Synthetic inference got stuck'
            return {'vehicles': [], 'detections': [{'plate_text': 'TEST123'}], 'people': []}

    history = History()
    stream = RTSPHandler(Config(), history, SlowModel())
    stream._get_analysis_area = lambda *args: {'enabled': False}
    stream._apply_analysis_mask = lambda frame, _: (frame, 0, 0)
    stream._draw_analysis_area = lambda *args: None
    stream._draw_people_line = lambda *args: None
    with stream.frame_lock:
        stream._stream_generation = 19
        stream.current_frame = np.zeros((48, 64, 3), np.uint8)
        stream._last_frame_monotonic = time.monotonic()
    worker = run_processing(stream, (np.zeros((48, 64, 3), np.uint8), time.monotonic() - 1, 19))
    try:
        assert entered.wait(4), 'Fake detector never started'
        stream.disconnect()  # read error/reconnect, but the process worker is still running
        assert stream._stream_generation != 19
        with stream.frame_lock:
            stream.current_frame = np.full((48, 64, 3), 75, np.uint8)
            stream._last_frame_monotonic = time.monotonic()
        unblock.set()
        time.sleep(0.15)
        assert history.items == [], 'Old RTSP detection was saved after reconnection'
        assert stream.annotated_frame is None, 'Old camera image covered the new stream'
        assert stream.frame_count == 0
    finally:
        unblock.set()
        stop_processing(stream, worker)
    print('  OK RTSP: delayed inference from previous connection is discarded')


def test_true_capture_time_keeps_preview_fresh():
    stream = RTSPHandler(Config(), History(), None)
    raw = np.full((48, 64, 3), 17, np.uint8)
    previous_capture = time.monotonic() - 9
    with stream.frame_lock:
        stream.current_frame = raw.copy()
        stream._last_frame_monotonic = time.monotonic()
        generation = stream._stream_generation
    worker = run_processing(stream, (np.full((48, 64, 3), 230, np.uint8),
                                     previous_capture, generation))
    try:
        deadline = time.monotonic() + 4
        while stream.annotated_frame is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert stream.annotated_frame is not None, 'Frame not processed'
        state = stream.get_status()
        assert state['last_analysis_frame_age_seconds'] >= 8, state
        assert state['preview_source'] == 'raw', state
        assert int(stream.get_current_frame()[0, 0, 0]) == 17
    finally:
        stop_processing(stream, worker)
    print('  OK RTSP: queued frames retain their original camera timestamp')


def test_capture_stop_does_not_restore_a_deleted_frame():
    stream = RTSPHandler(Config(), History(), None)
    stream.running = True
    stream.connected = True
    class SlowCapture:
        def read(self):
            stream.running = False
            stream._stop_event.set()
            return True, np.full((48, 64, 3), 110, np.uint8)
        def release(self):
            pass
    stream.cap = SlowCapture()
    stream._capture_loop()
    assert stream.current_frame is None
    assert stream.frame_queue.qsize() == 0
    assert stream._last_frame_monotonic is None
    print('  OK RTSP: read finishing after stop cannot reinsert a camera frame')


def test_queue_pressure_tracks_discarded_images():
    stream = RTSPHandler(Config(), History(), None)
    stream.running = True
    stream.connected = True
    done, resume = threading.Event(), threading.Event()
    class FastCapture:
        count = 0
        def read(self):
            self.count += 1
            if self.count == 4:
                done.set()
                resume.wait(4)
                stream.running = False
            return True, np.full((48, 64, 3), self.count, np.uint8)
        def release(self):
            pass
    stream.cap = FastCapture()
    worker = threading.Thread(target=stream._capture_loop, daemon=True)
    worker.start()
    try:
        assert done.wait(4)
        state = stream.get_status()
        assert state['queued_frames'] == 2 and state['discarded_frames'] == 1, state
        queued = list(stream.frame_queue.queue)
        assert all(isinstance(item, tuple) and len(item) == 3 for item in queued)
        assert [int(item[0][0, 0, 0]) for item in queued] == [2, 3]
        assert all(item[1] > 0 and item[2] == stream._stream_generation for item in queued)
    finally:
        resume.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
    print('  OK RTSP: oldest packet dropped, queue/drop metrics are trustworthy')


def function(name, env):
    source = ROOT / 'platevision/src/app.py'
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), 'exec'), env)
    return env[name]


def test_api_config_cannot_claim_failed_write_is_saved():
    cfg = Config()
    request = SimpleNamespace(method='POST', json={}, get_json=lambda **kw: {}, args={})
    def geometry():
        return {'active': {'width': 64, 'height': 48}}
    scope = {'config_manager': cfg, 'request': request, 'jsonify': lambda data: data,
             '_current_stream_geometry': geometry, '_public_config': lambda: cfg.config}
    sections = (
        ('api_save_rtsp_config', 'rtsp', {'url': 'rtsp://fake-camera/new'}),
        ('api_save_detection_config', 'detection', {'process_interval': 0.6}),
        ('api_save_history_config', 'history', {'duplicate_timeout': 23}),
        ('api_save_storage_config', 'storage', {'image_history_limit': 8}),
    )
    for method, section, payload in sections:
        endpoint = function(method, scope)
        request.json = payload.copy()
        request.get_json = lambda **kw: payload.copy()
        cfg.can_save = False
        before = copy.deepcopy(cfg.config)
        result = endpoint()
        assert isinstance(result, tuple) and result[1] == 503, (method, result)
        assert result[0]['success'] is False
        assert cfg.config == before, (method, cfg.config)
        cfg.can_save = True
        result = endpoint()
        assert result['success'] is True, (method, result)
        assert cfg.config[section].get(next(iter(payload))) == next(iter(payload.values()))
    roi = function('api_rtsp_analysis_area', scope)
    request.json = {'polygon': [[2, 2], [40, 2], [40, 35]], 'coordinate_width': 64, 'coordinate_height': 48}
    request.get_json = lambda **kw: request.json.copy()
    cfg.can_save = False
    before = copy.deepcopy(cfg.config)
    result = roi()
    assert result[1] == 503 and result[0]['success'] is False
    assert cfg.config == before
    cfg.can_save = True
    result = roi()
    assert result['success'] is True
    assert len(cfg.config['rtsp']['analysis_area']['polygon']) == 3
    print('  OK API: five settings routes return HTTP 503 + rollback on storage failure')


if __name__ == '__main__':
    test_reconnect_invalidates_inflight_analysis_and_detections()
    test_true_capture_time_keeps_preview_fresh()
    test_capture_stop_does_not_restore_a_deleted_frame()
    test_queue_pressure_tracks_discarded_images()
    test_api_config_cannot_claim_failed_write_is_saved()
    print('PlateVision 0.13.13 RTSP sessions/config: OK')
