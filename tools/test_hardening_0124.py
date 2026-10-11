#!/usr/bin/env python3
"""PlateVision 0.12.4: real regression cases for previously untested edge paths.

Uses AST-extracted original implementations and realistic fakes, so expensive
Torch/EasyOCR dependencies and HA credentials are not required to run these.
"""
import ast
import copy
import io
import json
import logging
import os
import queue
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'platevision/src/app.py'
RTSP = ROOT / 'platevision/src/rtsp_handler.py'
LOGGER = logging.getLogger('platevision-0124-tests')


def extract(path, names, ns):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    nodes = []
    for name in names:
        node = next((n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name), None)
        assert node is not None, f'{name} not found'
        node = copy.deepcopy(node)
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), ns)
    return ns


class Plate:
    @staticmethod
    def normalize(value, compact=False):
        return ''.join(c for c in str(value or '').upper() if c.isalnum())

    @staticmethod
    def pretty(value):
        return str(value).upper().strip()

    @staticmethod
    def similarity(x, y):
        return 1.0 if x == y else 0.0

    @staticmethod
    def best_candidate(v, **kw):
        return v

    @staticmethod
    def generate_candidates(v, **kw):
        return [v]


def checks():
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='platevision-0124-') as tmp:
        os.chdir(tmp)
        try:
            from collections import Counter
            from datetime import timedelta
            import hashlib
            import shutil
            import re
            import cv2
            import numpy as np
            import zipfile

            ns = {'Path': Path, 'json': json, 'os': os, 'shutil': shutil,
                  'logging': logging, 'logger': LOGGER, 'uuid': uuid, 'threading': threading,
                  'datetime': datetime, 'timedelta': timedelta, 'Counter': Counter,
                  'time': time, 'PlateUtils': Plate, 'jsonify': lambda x: x,
                  'config_manager': SimpleNamespace(get=lambda *args: .88),
                  'request': SimpleNamespace(files={}, stream=io.BytesIO(), content_length=None)}
            extract(APP, ('_json_clone', '_atomic_write_json', '_load_json_with_backup', '_read_limited_import_json',
                          'WatchlistManager'), ns)
            manager = ns['WatchlistManager']()
            entries = [{'plate_text': 'ZH 1234', 'label': 'Erster'},
                       {'plate_text': 'BE5678', 'label': 'Zweiter'}]
            assert manager.import_items(entries) == 2
            saved = Path('data/watchlist.json').read_bytes()
            assert len(manager.items) == 2
            assert manager.import_items([{'plate_text': 'ZH-1234', 'notes': 'Update'}]) == 1
            assert len(manager.items) == 2
            assert manager.items[0]['label'] == 'Erster'
            # First entry valid but the second invalid: none can be committed.
            before = json.loads(Path('data/watchlist.json').read_text())
            try:
                manager.import_items([{'plate_text': 'NEW123'}, {'plate_text': ['BAD']}])
                raise AssertionError('Malformed import accepted')
            except ValueError:
                pass
            assert manager.items == before
            # A disk failure after processing hundreds of rows rolls back all.
            writer = ns['_atomic_write_json']
            ns['_atomic_write_json'] = lambda *a: (_ for _ in ()).throw(OSError('disk full'))
            try:
                manager.import_items([{'plate_text': f'ZH-{i}'} for i in range(120)])
                raise AssertionError('Disk failure was silently ignored')
            except OSError:
                pass
            ns['_atomic_write_json'] = writer
            assert manager.items == before
            assert json.loads(Path('data/watchlist.json').read_text()) == before
            print('  OK: Atomic watchlist import, duplicate updates and full rollback')

            req = ns['request']
            req.stream = io.BytesIO(b'{"general":{"theme":"light"}}')
            req.content_length = 29
            assert ns['_read_limited_import_json']()['general']['theme'] == 'light'
            req.content_length = None
            req.stream = io.BytesIO(b'x' * 128)
            try:
                ns['_read_limited_import_json'](max_bytes=16)
                raise AssertionError('Oversized JSON import accepted')
            except ValueError:
                pass
            req.files = {'file': SimpleNamespace(stream=io.BytesIO(b'\xef\xbb\xbf{"items":[]}'))}
            req.stream = io.BytesIO(b'ignored')
            assert ns['_read_limited_import_json']()['items'] == []
            req.files = {}
            req.stream = io.BytesIO(b'not json')
            try:
                ns['_read_limited_import_json']()
                raise AssertionError('Invalid JSON accepted')
            except ValueError:
                pass
            print('  OK: Strict JSON import size and encoding checks')

            model_ns = {'Path': Path, 'os': os, 'threading': threading, 'logger': LOGGER}
            extract(APP, ('_store_limited_model_upload',), model_ns)
            Path('data/models').mkdir(parents=True)
            model = Path('data/models/check.pt')
            assert model_ns['_store_limited_model_upload'](SimpleNamespace(stream=io.BytesIO(b'abc')), model, 3) == 3
            assert model.read_bytes() == b'abc'
            try:
                model_ns['_store_limited_model_upload'](SimpleNamespace(stream=io.BytesIO(b'x' * 2048)), model, 64)
                raise AssertionError('Oversized model accepted')
            except ValueError:
                pass
            assert model.stat().st_size <= 64
            print('  OK: Model uploads stopped before configured limit is exceeded')

            class Model:
                def __init__(self):
                    self.inference_lock = threading.RLock()
                    self.models_loaded = True
                    self.coco_model = 'old-coco'
                    self.license_model = 'old-plate'
                    self.human_model = 'old-human'
                    self.ocr_reader = 'old-ocr'
                    self.fast_plate_recognizer = 'old-fast'
                    self.fast_plate_cache_key = 'old-cache'
                    self.load_count = 0
                def load_models(self):
                    self.load_count += 1
                    time.sleep(.025)
                    self.models_loaded = True
            d = Model()
            model_ns.update({'detector': d, '_model_reload_lock': threading.Lock(),
                             '_model_reload_followup_lock': threading.Lock(),
                             '_model_reload_pending_lock': threading.Lock(),
                             '_model_reload_queued': False})
            extract(APP, ('_reload_models_background', '_invalidate_model_cache',
                          '_pending_model_reload_worker', '_queue_model_reload'), model_ns)
            with d.inference_lock:
                assert model_ns['_queue_model_reload']() == 'started'
                assert model_ns['_queue_model_reload']() == 'queued'
                for _ in range(5):
                    assert model_ns['_queue_model_reload']() == 'queued'
                assert d.license_model == 'old-plate'  # no unsafe live reset
            for _ in range(250):
                if d.load_count >= 2 and not model_ns['_model_reload_lock'].locked() and not model_ns['_model_reload_followup_lock'].locked():
                    break
                time.sleep(.01)
            assert d.load_count == 2, f'coalescing spawned {d.load_count} threads'
            assert not model_ns['_model_reload_lock'].locked()
            assert not model_ns['_model_reload_followup_lock'].locked()
            # Cache invalidation must wait for an in-flight inference.
            gate = threading.Event()
            with d.inference_lock:
                t = threading.Thread(target=lambda: (model_ns['_invalidate_model_cache'](), gate.set()))
                t.start()
                time.sleep(.03)
                assert not gate.is_set() and d.models_loaded
            t.join(2)
            assert gate.is_set() and not d.models_loaded and d.coco_model is None
            print('  OK: Serialized model reload and coalesced concurrent settings changes')

            # A video must be durably registered BEFORE its processing thread
            # can create recognition events; a registry disk error aborts upload.
            Path('uploads/videos').mkdir(parents=True, exist_ok=True)
            class VideoRequest:
                files = {'file': SimpleNamespace(filename='start.mp4', read=lambda n: b'' )}
            no_threads = []
            fake_registry = {}
            def temp_video_store(upload, path):
                Path(path).write_bytes(b'fake video')
                return 10
            route_ns = {'threading': threading, 'Path': Path, 'uuid': uuid, 'datetime': datetime,
                        'logger': LOGGER, 'jsonify': lambda x: x,
                        'request': VideoRequest(), 'video_jobs_lock': threading.RLock(),
                        'video_processing_jobs': fake_registry, 'video_job_cancel_events': {},
                        '_cleanup_old_video_jobs': lambda: 0,
                        '_video_active_jobs': lambda: 0, '_video_max_active_jobs': lambda: 1,
                        '_store_limited_video': temp_video_store,
                        '_persist_video_jobs': lambda: False,
                        'process_video_job': lambda *a: no_threads.append(a)}
            extract(APP, ('api_process_video',), route_ns)
            payload, code = route_ns['api_process_video']()
            assert code == 503 and not fake_registry and not no_threads, payload
            assert not list(Path('uploads/videos').iterdir())
            # The successful path must observe an already-committed registry.
            saved = []
            class FakeThread:
                def __init__(self, **kw):
                    pass
                def start(self):
                    assert saved == [True], 'Video worker started before persistent registry commit'
                    no_threads.append('started')
            route_ns['threading'] = SimpleNamespace(Event=threading.Event, Thread=FakeThread)
            route_ns['_persist_video_jobs'] = lambda: saved.append(True) or True
            payload = route_ns['api_process_video']()
            assert payload['success'] and no_threads == ['started']
            print('  OK: Video jobs are persisted before a processing thread is allowed to start')

            # Finished jobs may only be removed if their updated registry was
            # successfully committed. A full disk must preserve the download.
            job_to_delete = str(uuid.uuid4())
            output_to_delete = Path('uploads/processed') / f'{job_to_delete}_output.mp4'
            output_to_delete.parent.mkdir(parents=True, exist_ok=True)
            output_to_delete.write_bytes(b'completed video')
            deleted_registry = {job_to_delete: {
                'id': job_to_delete, 'status': 'completed',
                'output_path': str(output_to_delete), 'video_path': None}}
            delete_ns = {'jsonify': lambda x: x, 'logger': LOGGER, 'Path': Path,
                         'video_jobs_lock': threading.RLock(),
                         'video_processing_jobs': deleted_registry,
                         'video_job_cancel_events': {},
                         '_persist_video_jobs': lambda: False}
            extract(APP, ('api_delete_job',), delete_ns)
            response, status = delete_ns['api_delete_job'](job_to_delete)
            assert status == 503 and not response['success']
            assert job_to_delete in deleted_registry and output_to_delete.is_file()
            delete_ns['_persist_video_jobs'] = lambda: True
            response = delete_ns['api_delete_job'](job_to_delete)
            assert response['success'] and job_to_delete not in deleted_registry
            assert not output_to_delete.exists()
            print('  OK: Failed video-job deletion rolls back registry before touching media')

            class HM:
                def __init__(self):
                    self.rows = {}
                    self.deleted = []
                def delete_video_job_entries(self, job_id):
                    self.deleted.append(job_id)
                    return self.rows.pop(job_id, 0)
            histories = HM()
            video_ns = {'Path': Path, 'uuid': uuid, 'json': json, 'logger': LOGGER,
                        'threading': threading, 'video_jobs_lock': threading.RLock(),
                        'video_processing_jobs': {}, 'history_manager': histories,
                        'VIDEO_JOB_STATE_FILE': Path('data/video_jobs.json'),
                        'datetime': datetime, '_atomic_write_json': ns['_atomic_write_json']}
            extract(APP, ('_video_status_payload', '_persist_video_jobs',
                          '_restore_video_jobs', '_load_json_with_backup'), video_ns)
            jobs = video_ns['video_processing_jobs']
            completed = str(uuid.uuid4())
            crash = str(uuid.uuid4())
            failed = str(uuid.uuid4())
            histories.rows.update({completed: 1, crash: 4, failed: 3})
            out = Path('uploads/processed') / f'{completed}_output.mp4'
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b'mp4')
            Path('data').mkdir(exist_ok=True)
            Path('data/video_jobs.json').write_text(json.dumps([
                {'id': completed, 'filename': 'good.mp4', 'status': 'completed', 'saved_history_count': 1},
                {'id': crash, 'filename': 'crash.mp4', 'status': 'processing', 'saved_history_count': 4},
                {'id': failed, 'filename': 'failed.mp4', 'status': 'error', 'saved_history_count': 3},
            ]))
            assert video_ns['_restore_video_jobs']() == 3
            assert histories.rows == {completed: 1}
            assert set(histories.deleted) == {crash, failed}
            assert jobs[crash]['status'] == 'error' and jobs[crash]['saved_history_count'] == 0
            assert jobs[completed]['status'] == 'completed' and jobs[completed]['saved_history_count'] == 1
            persisted = json.loads(Path('data/video_jobs.json').read_text())
            assert all('video_path' not in job and 'output_path' not in job for job in persisted)
            # A cleanup failure must stay visible and be retryable next boot.
            video_ns['history_manager'] = SimpleNamespace(delete_video_job_entries=lambda *_: (_ for _ in ()).throw(OSError('disk full')))
            jobs.clear()
            Path('data/video_jobs.json').write_text(json.dumps([{'id': crash, 'filename': 'retry.mp4', 'status': 'processing'}]))
            assert video_ns['_restore_video_jobs']() == 1
            assert jobs[crash]['history_cleanup_pending'] is True
            assert json.loads(Path('data/video_jobs.json').read_text())[0]['history_cleanup_pending'] is True
            print('  OK: Crash recovery cleans incomplete video history, retains retry flag on disk errors')

            class Data:
                def __init__(self, events):
                    self.history = events
                    self.lock = threading.RLock()
                @staticmethod
                def _parse_datetime(v):
                    return datetime.fromisoformat(v) if v else None
            history = Data([{'plate_text': f'ZH {i}', 'timestamp': '2026-10-09T09:00:00',
                             'id': str(i), 'plate_image': 'LARGE-BASE64'} for i in range(600)])
            class Req:
                args = {'limit': '0', 'bucket': 'day'}
                def get_json(self, **kwargs):
                    return {}
            req = Req()
            api_ns = {'_page_integer': lambda value, default, minimum=0, maximum=200: max(minimum, min(int(value) if str(value).lstrip('-').isdigit() else default, maximum)),
                      '_safe_numeric': lambda v, default=0: float(v) if v and v != 'nan' else default,
                      'request': req, 'history_manager': history,
                      'PlateUtils': Plate, 'Counter': Counter, 'datetime': datetime,
                      'jsonify': lambda x: x,
                      'config_manager': SimpleNamespace(get=lambda *x: .88)}
            extract(APP, ('_history_api_entries', 'api_history_autocomplete',
                          'api_history_timeline', 'api_history_duplicates',
                          'api_plate_batch_analyze'), api_ns)
            assert len(api_ns['api_history_autocomplete']()['items']) == 1
            assert len(api_ns['api_history_timeline']()['items']) == 1
            req.args = {'limit': '50000', 'bucket': 'hour'}
            assert len(api_ns['api_history_autocomplete']()['items']) <= 50
            assert len(api_ns['api_history_timeline']()['items']) <= 366
            req.args = {'similarity': '-999'}
            result = api_ns['api_history_duplicates']()
            assert result['similarity'] >= .5
            assert all('plate_image' not in row for g in result['groups'] for row in g['entries'])
            req.get_json = lambda **kwargs: {'plates': [f'ZH{i}' for i in range(201)]}
            assert api_ns['api_plate_batch_analyze']()[1] == 413
            req.get_json = lambda **kwargs: {'plates': ['BE123', 'ZH456']}
            assert api_ns['api_plate_batch_analyze']()['total'] == 2
            print('  OK: Bounded autocomplete, timelines, duplicate previews and batch analysis')

            # Native stream read() must never race with VideoCapture.release().
            rtsp_ns = {'threading': threading, 'queue': queue, 'time': time,
                       'logging': logging, 'logger': LOGGER, 'np': np, 'cv2': cv2,
                       'datetime': datetime}
            extract(RTSP, ('RTSPHandler',), rtsp_ns)
            class Capture:
                def __init__(self):
                    self.reading = threading.Event()
                    self.unblock = threading.Event()
                    self.released_during_read = False
                    self.read_active = False
                    self.closed = False
                def read(self):
                    self.read_active = True
                    self.reading.set()
                    self.unblock.wait(7)
                    self.read_active = False
                    return False, None
                def release(self):
                    self.released_during_read |= self.read_active
                    self.closed = True
            capture = Capture()
            class Cfg:
                def get(self, *keys):
                    return .2 if keys == ('rtsp', 'reconnect_delay') else None
            camera = rtsp_ns['RTSPHandler'](Cfg(), None, None)
            camera.cap = capture
            camera.connected = True
            camera._process_loop = lambda: camera._stop_event.wait(0.1)
            assert camera.start()
            assert capture.reading.wait(3)
            assert camera.stop() is False
            assert not capture.closed and not capture.released_during_read
            capture.unblock.set()
            camera.capture_thread.join(4)
            assert capture.closed and not capture.released_during_read
            assert camera.start()
            camera.stop()
            print('  OK: RTSP native read handle is released only by the capture thread')

        finally:
            os.chdir(original)
    print('PlateVision 0.12.4 additional hardening regression tests: OK')


if __name__ == '__main__':
    checks()
