#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""0.13.12 RTSP regression: fresh preview, safe cancellation and bounded caches.

Exercises the production RTSPHandler with synthetic frames; no live camera or
model weights are needed.
"""
import logging
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'platevision/src'))
from rtsp_handler import RTSPHandler


class Config:
    def __init__(self):
        self.settings = {
            'rtsp': {'url': 'rtsp://camera.local/test'},
            'detection': {'process_interval': 0.5},
            'history': {'duplicate_timeout': 600, 'filter_duplicates': True},
            'people': {'enabled': False},
        }

    def get(self, *path):
        result = self.settings
        for key in path:
            result = result.get(key) if isinstance(result, dict) else None
        return result


class History:
    def __init__(self):
        self.items = []

    def add_entry(self, row, **_):
        self.items.append(row)
        return row


def test_stale_annotations_fall_back_to_latest_camera_image():
    stream = RTSPHandler(Config(), History(), None)
    raw = np.full((32, 48, 3), 22, dtype=np.uint8)
    annotated = np.full((32, 48, 3), 230, dtype=np.uint8)
    with stream.frame_lock:
        stream.current_frame = raw.copy()
        stream.annotated_frame = annotated.copy()
        stream._last_frame_monotonic = 100.0
        stream._last_annotated_monotonic = 99.5
    with patch('rtsp_handler.time.monotonic', return_value=100.2):
        assert int(stream.get_current_frame()[0, 0, 0]) == 230
        assert stream.get_status()['preview_source'] == 'annotated'
    with stream.frame_lock:
        stream._last_frame_monotonic = 103.0
    with patch('rtsp_handler.time.monotonic', return_value=103.1):
        displayed = stream.get_current_frame()
        status = stream.get_status()
    assert int(displayed[0, 0, 0]) == 22, 'Old YOLO frame should not freeze the live view'
    assert status['preview_source'] == 'raw'
    assert status['last_frame_age_seconds'] == 0.1
    assert status['last_analysis_frame_age_seconds'] == 3.6
    # Do not hand out internal NumPy references to HTTP threads.
    displayed[:, :] = 99
    assert int(stream.get_raw_frame()[0, 0, 0]) == 22
    with patch('rtsp_handler.time.monotonic', return_value=103.2):
        with stream.frame_lock:
            stream.annotated_frame = annotated.copy()
            stream._last_annotated_monotonic = 103.2
        assert int(stream.get_current_frame()[0, 0, 0]) == 230
    stream.disconnect()
    assert stream.get_current_frame() is None
    assert stream.get_status()['last_frame_age_seconds'] is None
    assert stream.get_status()['last_analysis_frame_age_seconds'] is None
    print('  OK RTSP: stale analysis displays newest raw image; frames and clocks reset')


def test_interval_updates_do_not_need_restart():
    config = Config()
    stream = RTSPHandler(config, History(), None)
    assert stream._configured_process_interval() == 0.5
    config.settings['detection']['process_interval'] = '3.4'
    assert stream._configured_process_interval() == 3.4
    config.settings['detection']['process_interval'] = 0
    assert stream._configured_process_interval() == 0.01
    for bad in ('nan', 'inf', [], 'invalid'):
        config.settings['detection']['process_interval'] = bad
        assert stream._configured_process_interval() == 0.5, bad
    print('  OK RTSP: live interval slider re-read and corrupt settings bounded')


def test_stop_during_inference_never_publishes_old_frames():
    entered = threading.Event()
    proceed = threading.Event()

    class SlowModel:
        models_loaded = True

        def process_frame(self, frame, **_):
            entered.set()
            assert proceed.wait(timeout=5), 'Fake inference did not finish'
            return {'vehicles': [], 'detections': [{'plate_text': 'ZH 123'}], 'people': []}

    history = History()
    stream = RTSPHandler(Config(), history, SlowModel())
    stream._get_analysis_area = lambda *_: {'enabled': False}
    stream._apply_analysis_mask = lambda frame, _: (frame, 0, 0)
    stream._draw_analysis_area = lambda *_: None
    stream._draw_people_line = lambda *_: None
    stream.running = True
    stream.frame_queue.put(np.zeros((80, 100, 3), dtype=np.uint8))
    worker = threading.Thread(target=stream._process_loop, daemon=True)
    worker.start()
    assert entered.wait(timeout=5), 'Model was never called'
    stream.running = False
    stream._stop_event.set()
    proceed.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert stream.annotated_frame is None, 'Late YOLO result overwrote stopped camera'
    assert history.items == [], 'Late YOLO result created history after stop'
    print('  OK RTSP: stop during long inference cannot revive old live frames')


def test_duplicate_cache_is_bounded_without_breaking_cooldown():
    history = History()
    config = Config()
    stream = RTSPHandler(config, history, None)
    with patch.object(logging.getLogger('rtsp_handler'), 'info'):
        for i in range(4300):
            stream._handle_detection({'plate_text': f'ZH {i:05d}', 'confidence': 0.94}, {})
    assert len(history.items) == 4300
    assert len(stream.recent_plates) <= 4096, len(stream.recent_plates)
    stream._handle_detection({'plate_text': 'ZH 04299', 'confidence': 0.94}, {})
    assert len(history.items) == 4300, 'Recent plates must still be deduplicated'
    assert 'ZH04299' in stream.recent_plates
    print('  OK RTSP: duplicate cache stays bounded after thousands of vehicles')


if __name__ == '__main__':
    test_stale_annotations_fall_back_to_latest_camera_image()
    test_interval_updates_do_not_need_restart()
    test_stop_during_inference_never_publishes_old_frames()
    test_duplicate_cache_is_bounded_without_breaking_cooldown()
    print('PlateVision 0.13.12 frame freshness: OK')
