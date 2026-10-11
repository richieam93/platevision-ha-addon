#!/usr/bin/env python3
"""Regression tests for 0.12.3: configuration, CSRF, backup, RTSP and inference.

Exercises definitions from real source by AST extraction without expensive ML imports.
"""
import ast
import copy
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'platevision/src/app.py'
RTSP = ROOT / 'platevision/src/rtsp_handler.py'
LOGGER = logging.getLogger('platevision-0123-tests')


def extract(source, names, namespace):
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    chosen = []
    for name in names:
        node = next((n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name), None)
        assert node, name
        node = copy.deepcopy(node)
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        chosen.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=chosen, type_ignores=[])), str(source), 'exec'), namespace)
    return namespace


def fake_jsonify(value):
    return value


class LocalRequest(threading.local):
    def __init__(self):
        self.method = 'POST'
        self.path = '/api/config/general'
        self.endpoint = 'api_save_general_config'
        self.headers = {}
        self.host = 'ha.example:8123'


def run():
    ns = {
        'Path': Path, 'json': json, 'logging': logging, 'logger': LOGGER,
        'os': os, 'shutil': shutil, 'tempfile': tempfile, 'threading': threading, 'uuid': uuid,
        'hashlib': hashlib, 'time': time, 'datetime': datetime, 'timedelta': timedelta,
        're': re, 'Counter': Counter, 'defaultdict': defaultdict,
        'SequenceMatcher': SequenceMatcher, 'zipfile': zipfile, 'wraps': wraps,
        'has_request_context': lambda: False, '_platevision_ingress_prefix': lambda: '',
        'g': threading.local(), 'request': LocalRequest(), 'jsonify': fake_jsonify,
        '_BACKUP_METADATA': ('data/config.json', 'data/history/detections.json', 'data/people/history.json', 'data/watchlist.json', 'data/video_jobs.json'),
        '_BACKUP_IMAGE_DIRS': ('data/plates_detected', 'data/vehicles_detected', 'data/people/images'),
        '_SQLITE_BACKUP_DATA': (('plates','data/history/detections.json','data/history/plates.sqlite3'), ('people','data/people/history.json','data/people/people.sqlite3')),
        'APP_VERSION': '0.12.3',
    }
    extract(APP, ('_json_clone', '_atomic_write_json', '_load_json_with_backup',
                  '_sha256_file', '_page_integer', '_safe_numeric', 'ConfigManager',
                  '_protect_mutating_requests', '_release_config_request_lock',
                  '_serialize_model_operation', '_backup_inventory', '_write_backup_archive'), ns)
    origdir = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix='platevision-0123-test-') as temporary:
            os.chdir(temporary)
            cm = ns['ConfigManager']()
            ns['config_manager'] = cm
            assert cm.save_config()
            first_contents = Path('data/config.json').read_bytes()
            previous_site = cm.config['general']['site_name']
            real_writer = ns['_atomic_write_json']
            ns['_atomic_write_json'] = lambda *_: (_ for _ in ()).throw(OSError('ENOSPC simulated'))
            cm.config['general']['site_name'] = 'should-not-be-used'
            assert cm.save_config() is False
            assert cm.config['general']['site_name'] == previous_site
            assert Path('data/config.json').read_bytes() == first_contents
            try:
                cm.set('still-not-persisted', 'general', 'site_name')
                raise AssertionError('Config.set accepted unsaved edit')
            except OSError:
                pass
            assert cm.config['general']['site_name'] == previous_site
            ns['_atomic_write_json'] = real_writer
            cm.config['rtsp'] = 'wrong type'
            assert not cm.save_config() and isinstance(cm.config['rtsp'], dict)
            assert Path('data/config.json').read_bytes() == first_contents
            assert cm.save_config()
            print('  OK: Config changes rollback on disk-full and malformed sections')

            req = ns['request']
            g = ns['g']
            req.headers = {'Sec-Fetch-Site': 'cross-site'}
            response = ns['_protect_mutating_requests']()
            assert response[1] == 403, response
            req.headers = {'Origin': 'https://evil.example'}
            assert ns['_protect_mutating_requests']()[1] == 403
            req.headers = {'Origin': 'http://ha.example:8123'}
            assert ns['_protect_mutating_requests']() is None
            assert g.platevision_config_write_lock
            ns['_release_config_request_lock']()
            req.headers = {'Origin': 'https://ha.example:8123', 'Sec-Fetch-Site': 'same-origin'}
            assert ns['_protect_mutating_requests']() is None
            ns['_release_config_request_lock']()
            print('  OK: Origin/Sec-Fetch cross-site requests blocked; same-origin allowed')

            req.headers = {}
            failures = []
            def change(key, value):
                try:
                    assert ns['_protect_mutating_requests']() is None
                    cm.config['general'][key] = value
                    time.sleep(.03)
                    assert cm.save_config()
                except Exception as exc:
                    failures.append(exc)
                finally:
                    ns['_release_config_request_lock']()
            workers = [threading.Thread(target=change, args=('operator_name', 'Alice')),
                       threading.Thread(target=change, args=('site_name', 'Garage'))]
            for worker in workers: worker.start()
            for worker in workers: worker.join(5)
            assert not failures and all(not worker.is_alive() for worker in workers), failures
            readback = json.loads(Path('data/config.json').read_text())
            assert readback['general']['operator_name'] == 'Alice'
            assert readback['general']['site_name'] == 'Garage'
            print('  OK: Concurrent config updates remain committed together')

            Path('data/history').mkdir(parents=True, exist_ok=True)
            Path('data/people/images/crops').mkdir(parents=True, exist_ok=True)
            Path('data/plates_detected').mkdir(parents=True, exist_ok=True)
            Path('data/history/detections.json').write_text('[{"id":"abc"}]')
            Path('data/people/images/crops/person.jpg').write_bytes(b'person crop')
            Path('data/plates_detected/plate.jpg').write_bytes(b'plate img')
            Path('data/people/images/crops/external-link.jpg').symlink_to('/etc/hosts')
            try:
                Path('data/people/images/other-link').symlink_to('/etc', target_is_directory=True)
            except OSError:
                pass
            paths, size = ns['_backup_inventory']()
            assert any(name == 'data/config.json' for name, _, _ in paths)
            assert any(name.endswith('/person.jpg') for name, _, _ in paths)
            assert not any('external-link' in name or 'other-link' in name for name, _, _ in paths)
            try:
                ns['_backup_inventory'](max_bytes=2)
                raise AssertionError('Oversized backup not blocked')
            except ValueError:
                pass
            with io.BytesIO() as buffer:
                manifest = ns['_write_backup_archive'](buffer)
                buffer.seek(0)
                with zipfile.ZipFile(buffer) as archive:
                    assert archive.testzip() is None
                    members = archive.namelist()
                    assert 'data/config.json' in members and 'platevision-backup-manifest.json' in members
                    assert not any('models' in member or 'external-link' in member for member in members)
                    stored = json.loads(archive.read('platevision-backup-manifest.json'))
                    assert stored == manifest
                    for row in stored['included']:
                        assert hashlib.sha256(archive.read(row['path'])).hexdigest() == row['sha256']
            print('  OK: Portable ZIP with checksums, symlink filtering and size limit')

            decorator = ns['_serialize_model_operation']
            class FakeDetector:
                inference_lock = threading.RLock()
                active = 0
                max_active = 0

                @decorator
                def work(self):
                    self.active += 1
                    self.max_active = max(self.active, self.max_active)
                    time.sleep(.015)
                    self.active -= 1
            detector = FakeDetector()
            threads = [threading.Thread(target=detector.work) for _ in range(6)]
            for thread in threads: thread.start()
            for thread in threads: thread.join(3)
            assert detector.max_active == 1
            tree = ast.parse(APP.read_text())
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LicensePlateDetector')
            for method_name in ('load_models', 'process_frame'):
                method = next(m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == method_name)
                assert any(isinstance(d, ast.Name) and d.id == '_serialize_model_operation' for d in method.decorator_list)
            print('  OK: Shared detector model operations are serialized')

            # A model reload may be queued while inference is holding the
            # detector lock. It must not clear models in the active session.
            ready = threading.Event()
            class ReloadDetector:
                inference_lock = threading.RLock()
                models_loaded = True
                coco_model = 'old'
                license_model = 'old'
                human_model = 'old'
                ocr_reader = 'old'
                fast_plate_recognizer = 'old'
                fast_plate_cache_key = 'old'
                def load_models(self):
                    self.models_loaded = True
                    ready.set()
            reload_detector = ReloadDetector()
            reload_lock = threading.Lock()
            reload_ns = {'threading': threading, 'detector': reload_detector,
                         '_model_reload_lock': reload_lock, 'logger': LOGGER,
                         'jsonify': fake_jsonify}
            extract(APP, ('_reload_models_background', 'api_reload_models'), reload_ns)
            with reload_detector.inference_lock:
                assert reload_ns['api_reload_models']()['success']
                assert reload_ns['api_reload_models']()[1] == 409
                time.sleep(.02)
                assert reload_detector.coco_model == 'old', 'model cleared during inference'
            assert ready.wait(3), 'model reload did not finish'
            for _ in range(100):
                if not reload_lock.locked(): break
                time.sleep(.01)
            assert not reload_lock.locked() and reload_detector.models_loaded
            assert reload_detector.coco_model is None
            print('  OK: Exclusive model reload and bounded concurrent reload requests')

            # The storage-clear API must stop immediately on a failed history
            # commit, and explicitly report partial deletion/persist errors.
            clear_ns = {'Path': Path, 'jsonify': fake_jsonify, 'logger': LOGGER,
                        'video_jobs_lock': threading.RLock(),
                        '_video_active_jobs': lambda: False,
                        'video_processing_jobs': {'job-1': {'status': 'completed'}},
                        'video_job_cancel_events': {'job-1': threading.Event()},
                        '_persist_video_jobs': lambda: False,
                        'history_manager': SimpleNamespace(clear_history=lambda: (_ for _ in ()).throw(OSError('disk full')))}
            extract(APP, ('_delete_top_level_files', '_deletion_response',
                          'api_clear_all_storage', 'api_clear_plates'), clear_ns)
            Path('data/plates_detected/keep.jpg').write_bytes(b'keep')
            failed = clear_ns['api_clear_all_storage']()
            assert failed[1] == 503 and Path('data/plates_detected/keep.jpg').exists()
            assert len(clear_ns['video_processing_jobs']) == 1
            clear_ns['history_manager'] = SimpleNamespace(clear_history=lambda: 1)
            failed = clear_ns['api_clear_all_storage']()
            assert failed[1] == 503 and failed[0]['cleared_history'] == 1
            assert failed[0]['deleted_count'] >= 1
            assert len(clear_ns['video_processing_jobs']) == 1
            assert clear_ns['api_clear_plates']()[1] == 200
            # A symlink inside an image folder must never be removed by the API.
            Path('data/plates_detected/linked.jpg').symlink_to('/etc/hosts')
            assert clear_ns['api_clear_plates']()[0]['deleted_count'] == 0
            assert Path('data/plates_detected/linked.jpg').is_symlink()
            print('  OK: Storage clear does not lie on disk failure or follow symlinks')

            rtsp = dict(logger=LOGGER, time=time, threading=threading, Path=Path,
                        queue=__import__('queue'), np=__import__('numpy'),
                        cv2=__import__('cv2'), datetime=datetime, urlsplit=urlsplit)
            extract(RTSP, ('RTSPHandler',), rtsp)
            class Config:
                def get(self, *args):
                    return None
            handler = rtsp['RTSPHandler'](Config(), None, None)
            long_running = threading.Event()
            handler._capture_loop = lambda: long_running.wait(6)
            handler._process_loop = lambda: handler._stop_event.wait(.2)
            assert handler.start()
            assert handler.stop() is False  # capture worker still blocked after timeout
            assert handler.start() is False  # cannot resurrect old worker
            assert handler.running is False
            long_running.set()
            handler.capture_thread.join(3)
            assert handler.start() is True
            handler.stop()
            print('  OK: Slow RTSP worker cannot be resurrected by a quick restart')

            ns['datetime'] = datetime
            extract(APP, ('_parse_iso_datetime',), ns)
            parsed = ns['_parse_iso_datetime']('2026-10-09T23:30:00-05:00')
            local = datetime.fromisoformat('2026-10-09T23:30:00-05:00').astimezone().replace(tzinfo=None)
            assert parsed == local
            assert ns['_parse_iso_datetime']('garbage') is None
            print('  OK: Dashboard timestamp offsets converted to local time')
    finally:
        os.chdir(origdir)
    print('PlateVision 0.12.3 additional integrity tests: OK')


if __name__ == '__main__':
    run()
