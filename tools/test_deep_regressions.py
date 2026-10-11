#!/usr/bin/env python3
"""Extra real-world regression tests without importing Flask or ML weights.

Exercises the actual methods extracted from app.py/rtsp_handler.py, including
an encoded MP4 pipeline, cancellation, limits, HTML-safe API responses, and
thread-local people settings.  Needs OpenCV, NumPy and Pillow.
"""
from __future__ import annotations

import ast
import copy
import csv
import io
import json
import logging
import os
import shutil
import queue
import tempfile
import threading
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'platevision/src/app.py'
RTSP = ROOT / 'platevision/src/rtsp_handler.py'
LOGGER = logging.getLogger('pv-deep-regression')


class Config:
    def __init__(self):
        self.config = {
            'general': {'max_history_entries': 5000},
            'search': {'default_limit': 50, 'allow_regex_search': True, 'enable_fuzzy_search': False},
            'history': {'filter_duplicates': True, 'duplicate_timeout': 60},
            'plate_recognition': {'min_length': 3, 'max_length': 12},
            'rtsp': {'url': 'rtsp://admin:supersecret@192.168.1.5:554/secret?token=42', 'reconnect_delay': 5},
            'people': {'enabled': False, 'image_history_enabled': False},
            'api_key': 'sensitive-123',
            'nested': {'password': 'hidden', 'items': [{'token': 'abc'}]},
        }

    def get(self, *keys):
        data = self.config
        for key in keys:
            if not isinstance(data, dict):
                return None
            data = data.get(key)
        return data


def extracted(path, names, ns):
    """Compile function/class definitions from real source, without route bindings."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    found = []
    for name in names:
        selected = next((x for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name == name), None)
        assert selected is not None, (path, name)
        selected = copy.deepcopy(selected)
        if isinstance(selected, ast.FunctionDef):
            selected.decorator_list = []
        found.append(selected)
    exec(compile(ast.fix_missing_locations(ast.Module(body=found, type_ignores=[])), str(path), 'exec'), ns)
    return ns


def fake_people_detector_method_class():
    """Only load detector's actual two concurrency methods, not ML packages."""
    tree = ast.parse(APP.read_text(encoding='utf-8'))
    detector = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'LicensePlateDetector')
    methods = [copy.deepcopy(x) for x in detector.body if isinstance(x, ast.FunctionDef) and x.name in {'_person_config', 'override_people_config'}]
    assert len(methods) == 2
    cls = ast.ClassDef(name='MinimalDetector', bases=[], keywords=[], body=methods, decorator_list=[])
    ns = {'contextmanager': contextmanager}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), 'minimal_detector', 'exec'), ns)
    return ns['MinimalDetector']


class VideoDetector:
    def __init__(self, delay=0, on_detect=None):
        self.delay = delay
        self.on_detect = on_detect
        self.calls = 0

    def process_frame(self, frame):
        self.calls += 1
        if self.on_detect:
            self.on_detect.set()
        if self.delay:
            time.sleep(self.delay)
        return {'annotated_frame': frame, 'detections': [
            {'plate_text': 'ZH-1234', 'confidence': .87, 'vehicle_type': 'Auto'}
        ]}


class History:
    def __init__(self):
        self.rows = []

    def add_entry(self, value, check_duplicate=True):
        self.rows.append(value)

    def delete_video_job_entries(self, job_id):
        before = len(self.rows)
        self.rows = [row for row in self.rows if row.get('video_job_id') != job_id]
        return before - len(self.rows)


class Socket:
    def __init__(self):
        self.emitted = []

    def emit(self, *args):
        self.emitted.append(args)


def make_mp4(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 12.0, (80, 64))
    assert writer.isOpened(), 'OpenCV FFmpeg/mp4v encoder unavailable'
    for frame_no in range(frames):
        frame = np.full((64, 80, 3), frame_no % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert Path(path).stat().st_size > 200


def make_video_ns(config):
    ns = {
        'os': os, 'json': json, 'shutil': shutil, 'threading': threading, 'time': time, 'datetime': datetime,
        'timedelta': timedelta, 'Path': Path, 'logger': LOGGER, 'uuid': uuid,
        'cv2': cv2, 'socketio': Socket(), 'detector': VideoDetector(),
        'history_manager': History(), 'jsonify': lambda x: x,
        'request': SimpleNamespace(files={}), 'video_processing_jobs': {},
        'video_jobs_lock': threading.RLock(), 'video_job_cancel_events': {},
        'config_manager': config, 'VIDEO_JOB_STATE_FILE': Path('data/video_jobs.json'),
    }
    names = (
        '_page_integer', '_safe_numeric', '_atomic_write_json', '_load_json_with_backup',
        '_video_history_cooldown', '_persist_video_jobs', '_restore_video_jobs', 'VideoJobCancelled', '_video_max_active_jobs',
        '_video_max_upload_mb', '_store_limited_video', '_video_retention_hours',
        '_video_active_jobs', '_video_status_payload', '_cleanup_old_video_jobs', '_cleanup_orphan_video_files',
        'process_video_job', '_reject_oversized_video_request', 'api_video_job_status', 'api_video_job_cancel',
        'api_get_jobs', 'api_delete_job', 'api_process_video', 'api_clear_uploads', 'api_clear_all_storage',
        '_json_clone', '_csv_safe_cell', '_safe_image_upload', '_diagnostics_config',
    )
    extracted(APP, names, ns)
    return ns


def run():
    previous = Path.cwd()
    config = Config()
    passed = []
    with tempfile.TemporaryDirectory(prefix='platevision-deep-test-') as temp:
        os.chdir(temp)
        try:
            Path('data/history').mkdir(parents=True)
            Path('uploads/videos').mkdir(parents=True)
            Path('uploads/processed').mkdir(parents=True)

            # 1. License plates: real aware/naive timestamp handling and large exports.
            history_ns = {
                'config_manager': config, 'logging': logging, 'logger': LOGGER,
                'datetime': datetime, 'timedelta': timedelta, 'Path': Path,
                'threading': threading, 'json': json, 'os': os, 'hashlib': __import__('hashlib'),
                'uuid': uuid, 'time': time, 'SequenceMatcher': SequenceMatcher,
                're': __import__('re'), 'Counter': Counter, 'defaultdict': defaultdict,
            }
            extracted(APP, ('_page_integer', '_safe_numeric', '_json_clone', '_atomic_write_json',
                            '_load_json_with_backup', 'PlateUtils', 'HistoryManager'), history_ns)
            hm = history_ns['HistoryManager']()
            now = datetime.now().astimezone()
            hm.history = [
                {'id': 'fresh', 'plate_text': 'ZH 1234', 'timestamp': now.isoformat(), 'confidence': 'NaN'},
                {'id': 'old', 'plate_text': 'BE 1', 'timestamp': (now - timedelta(days=2)).isoformat(), 'confidence': .85},
                {'id': 'invalid', 'plate_text': 'GE 1', 'timestamp': 'broken!', 'confidence': 'Infinity'},
            ]
            assert hm._is_duplicate_in_history('ZH-1234', 60), 'Timezone-aware duplicate bypass'
            assert not hm._is_duplicate_in_history('BE-1', 60)
            assert hm.search_advanced({'date_from': now.date().isoformat()})['total'] == 1
            assert hm.search_all({'sort': 'confidence'})[0]['id'] == 'old'
            assert hm._safe_float('NaN', .7) == .7
            assert hm._safe_float('inf', .6) == .6
            hm.history = [
                {'id': str(i), 'plate_text': f'BE {i}', 'timestamp': (now - timedelta(seconds=i)).isoformat(), 'confidence': .6}
                for i in range(1505)
            ]
            assert hm.search_advanced({'limit': 10000})['limit'] == 1000
            assert len(hm.search_all({})) == 1505, 'CSV export would silently drop rows'
            passed.append('Historie: Zeitzonen, Zeitfilter, Konfidenz und >1000 Export-Zeilen')

            # 2. CSV protection against formulas interpreted by spreadsheet software.
            video = make_video_ns(config)
            csv_safe = video['_csv_safe_cell']
            for attack in ('=IMPORTXML(1)', '+cmd', '-2+3', '@SUM(A1)', '\t=HYPERLINK(1)'):
                assert str(csv_safe(attack)).lstrip().startswith("'"), attack
            assert csv_safe('normal') == 'normal' and csv_safe(10) == 10
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow([csv_safe('=1+1'), csv_safe('ZH1234')])
            assert next(csv.reader(io.StringIO(buf.getvalue()))) == ["'=1+1", 'ZH1234']
            passed.append('CSV: Schutz vor Tabellen-Formelinjektion')

            # 3. Safe diagnostics and image upload constraints.
            scrubbed = video['_diagnostics_config']()
            serialized = json.dumps(scrubbed)
            assert 'supersecret' not in serialized and 'sensitive-123' not in serialized
            assert 'hidden' not in serialized and 'token=42' not in serialized
            assert scrubbed['nested']['items'][0]['token'] == '[geschützt]'
            video['secure_filename'] = lambda filename: Path(filename).name
            video['np'] = np
            video['Image'] = Image
            video['io'] = io
            def image_file(data, filename='test.png'):
                return SimpleNamespace(filename=filename, read=io.BytesIO(data).read)
            png = io.BytesIO()
            Image.new('RGB', (48, 48), (20, 30, 40)).save(png, format='PNG')
            data, error = video['_safe_image_upload'](image_file(png.getvalue()))
            assert error is None and data['image'].shape[:2] == (48, 48)
            data, error = video['_safe_image_upload'](image_file(b'bad!', 'a.png'))
            assert data is None and 'Ungültiges' in error
            data, error = video['_safe_image_upload'](image_file(png.getvalue(), 'attack.php'))
            assert data is None and 'Bilddateien' in error
            os.environ['PLATEVISION_MAX_IMAGE_MB'] = '1'
            try:
                data, error = video['_safe_image_upload'](image_file(b'0' * (1024*1024 + 1)))
                assert data is None and 'zu groß' in error
            finally:
                os.environ.pop('PLATEVISION_MAX_IMAGE_MB', None)
            # Large pixel metadata rejected before expensive OpenCV decoder.
            original_image_class = video['Image']
            class HugeImage:
                size = (10000, 5000)
                def __enter__(self): return self
                def __exit__(self, *args): pass
            video['Image'] = SimpleNamespace(open=lambda *_: HugeImage())
            data, error = video['_safe_image_upload'](image_file(png.getvalue()))
            assert data is None and 'Megapixel' in error
            video['Image'] = original_image_class
            passed.append('Diagnose: sensible Konfiguration anonymisiert; Bild-Upload-Limits')

            # 4. Real MP4 file through actual process_video_job; exposes only safe metadata.
            video_path = Path('uploads/videos/success.mp4')
            make_mp4(video_path, 24)
            video['video_processing_jobs']['success'] = {'id': 'success', 'status': 'queued', 'filename': 'dashcam.mp4',
                'video_path': str(video_path), 'progress': 0, 'created_at': datetime.now().isoformat()}
            video['video_job_cancel_events']['success'] = threading.Event()
            video['process_video_job']('success', str(video_path), 'dashcam.mp4')
            result = video['video_processing_jobs']['success']
            assert result['status'] == 'completed' and result['progress'] == 100
            assert result['current_frame'] == 24 and result['detections_count'] >= 1
            assert Path(result['output_path']).exists() and cv2.VideoCapture(result['output_path']).isOpened()
            assert len(video['history_manager'].rows) > 0
            status = video['api_video_job_status']('success')
            assert status['output_available'] is True
            assert 'output_path' not in status and 'video_path' not in status
            assert 'results' not in status and status['recent_plates'] == ['ZH-1234']
            assert not any('uploads/videos' in str(v) for v in video['api_get_jobs']().values())
            passed.append('Video: echte MP4-Verarbeitung, Kennzeichen-Historie und sichere API-Metadaten')

            # A durable history I/O failure must invalidate the job, not silently
            # count a detection while promising successful persistence.
            failed_source = Path('uploads/videos/io_fail.mp4')
            make_mp4(failed_source, 24)
            video['video_processing_jobs']['io_fail'] = {
                'id': 'io_fail', 'filename': 'io_fail.mp4', 'status': 'queued',
                'created_at': datetime.now().isoformat(), 'video_path': str(failed_source)}
            video['video_job_cancel_events']['io_fail'] = threading.Event()
            old_history = video['history_manager']
            class DiskFailureHistory(History):
                def add_entry(self, value, check_duplicate=True):
                    raise OSError('disk full while saving license plate')
            video['history_manager'] = DiskFailureHistory()
            video['process_video_job']('io_fail', str(failed_source), 'io_fail.mp4')
            assert video['video_processing_jobs']['io_fail']['status'] == 'error'
            assert video['video_processing_jobs']['io_fail'].get('saved_history_count', 0) == 0
            assert not failed_source.exists()
            assert not Path('uploads/processed/io_fail_output.mp4').exists()
            video['history_manager'] = old_history
            passed.append('Video: volle Festplatte bricht den Job fehlerfrei ab statt Erfolg vorzutäuschen')

            # 5. Abort a running video, wait for worker exit, delete both file remnants.
            cancel_source = Path('uploads/videos/abort.mp4')
            make_mp4(cancel_source, 160)
            video['video_processing_jobs']['abort'] = {'id': 'abort', 'filename': 'to_cancel.mp4',
                'video_path': str(cancel_source), 'status': 'queued', 'created_at': datetime.now().isoformat()}
            video['video_job_cancel_events']['abort'] = threading.Event()
            detection_started = threading.Event()
            video['detector'] = VideoDetector(delay=.025, on_detect=detection_started)
            worker = threading.Thread(target=video['process_video_job'], args=('abort', str(cancel_source), 'cancel.mp4'))
            worker.start()
            assert detection_started.wait(6), 'Video test worker never started analysis'
            payload, code = video['api_video_job_cancel']('abort')
            assert code == 202 and payload['status'] == 'cancelling'
            worker.join(timeout=6)
            assert not worker.is_alive(), 'Video cancellation deadlocked'
            assert video['video_processing_jobs']['abort']['status'] == 'cancelled'
            assert not cancel_source.exists() and not Path('uploads/processed/abort_output.mp4').exists()
            assert 'abort' not in video['video_job_cancel_events']
            passed.append('Video: laufenden Job abbrechen ohne beschädigte Restdateien')

            # 6. Guard against parallel overload and uploads >configured disk budget.
            video['video_processing_jobs']['busy'] = {'id': 'busy', 'status': 'processing'}
            video['request'] = SimpleNamespace(files={'file': SimpleNamespace(filename='clip.mp4')})
            body, code = video['api_process_video']()
            assert code == 429 and 'Video-Verarbeitung' in body['error']
            video['video_processing_jobs'].pop('busy')
            class Upload:
                def __init__(self, data):
                    self.stream = io.BytesIO(data)
                def read(self, count):
                    return self.stream.read(count)
            os.environ['PLATEVISION_MAX_VIDEO_MB'] = '16'
            try:
                assert video['_video_max_upload_mb']() == 16
                try:
                    video['_store_limited_video'](Upload(b'v' * (16*1024*1024 + 1)), 'uploads/videos/limit.mp4')
                    raise AssertionError('Oversized video was accepted')
                except ValueError as exc:
                    assert 'Upload-Limit' in str(exc)
                assert Path('uploads/videos/limit.mp4').stat().st_size <= 16*1024*1024
                assert video['_store_limited_video'](Upload(b'abcd'), 'uploads/videos/tiny.mp4') == 4
            finally:
                os.environ.pop('PLATEVISION_MAX_VIDEO_MB', None)
            passed.append('Video: parallele Jobs blockiert und begrenzter Streaming-Upload')

            # Multipart metadata is checked before Werkzeug spends disk/RAM
            # buffering a very large video body.
            video['request'] = SimpleNamespace(path='/api/process/video', method='POST',
                content_length=490 * 1024 * 1024)
            body, code = video['_reject_oversized_video_request']()
            assert code == 413 and 'Upload-Limit' in body['error']
            video['request'].content_length = 1000
            assert video['_reject_oversized_video_request']() is None
            passed.append('API: frühzeitige Größenprüfung großer Multipart-Uploads')

            # Clearing storage holds the same lock as incoming job creation;
            # an active/queued job blocks destructive cleanup immediately.
            preserved = Path('uploads/videos/in_progress.txt')
            preserved.write_text('keep')
            video['video_processing_jobs']['queue_guard'] = {
                'id': 'queue_guard', 'status': 'queued',
                'created_at': datetime.now().isoformat()}
            body, code = video['api_clear_uploads']()
            assert code == 409 and preserved.exists()
            body, code = video['api_clear_all_storage']()
            assert code == 409 and preserved.exists()
            video['video_processing_jobs'].pop('queue_guard')
            passed.append('Speicher: keine Löschung aktiver oder wartender Videoverarbeitungen')

            # 7. Automatic retention of *finished* jobs, not running ones.
            stale_path = Path('uploads/videos/stale.mp4')
            stale_path.write_bytes(b'stale')
            video['video_processing_jobs']['old'] = {
                'id': 'old', 'status': 'completed', 'video_path': str(stale_path),
                'created_at': (datetime.now() - timedelta(days=20)).isoformat(),
                'completed_at': (datetime.now() - timedelta(days=20)).isoformat()}
            video['video_processing_jobs']['live'] = {
                'id': 'live', 'status': 'processing', 'created_at': (datetime.now() - timedelta(days=20)).isoformat()}
            assert video['_cleanup_old_video_jobs']() == 1
            assert 'old' not in video['video_processing_jobs'] and not stale_path.exists()
            assert 'live' in video['video_processing_jobs']
            body, code = video['api_delete_job']('live')
            assert code == 409 and body['success'] is False
            video['video_processing_jobs'].pop('live')
            assert video['api_delete_job']('success')['success'] is True
            assert not Path(result['output_path']).exists()
            passed.append('Video: automatische Aufbewahrung und Löschschutz bei aktiven Jobs')

            # In-memory records vanish on HA restart; UUID-named persisted
            # video uploads still need TTL cleanup without touching user files.
            orphan_id = str(uuid.uuid4())
            orphan = Path('uploads/videos') / f'{orphan_id}.mp4'
            orphan.write_bytes(b'old video')
            old_time = time.time() - 20*24*3600
            os.utime(orphan, (old_time, old_time))
            untouched = Path('uploads/videos/holiday.mp4')
            untouched.write_bytes(b'user video')
            os.utime(untouched, (old_time, old_time))
            assert video['_cleanup_orphan_video_files']() == 1
            assert not orphan.exists() and untouched.exists()
            kept_id = str(uuid.uuid4())
            live_path = Path('uploads/videos') / f'{kept_id}.mp4'
            live_path.write_bytes(b'live video')
            os.utime(live_path, (old_time, old_time))
            video['video_processing_jobs']['kept'] = {'id':'kept','status':'processing',
                'video_path':str(live_path),'created_at':datetime.now().isoformat()}
            assert video['_cleanup_orphan_video_files']() == 0 and live_path.exists()
            video['video_processing_jobs'].pop('kept')
            # Timezone-aware completed_at values must be comparable with cutoff.
            video['video_processing_jobs']['aware'] = {'id': 'aware', 'status': 'completed',
                'completed_at': (datetime.now().astimezone() - timedelta(days=20)).isoformat()}
            assert video['_cleanup_old_video_jobs']() == 1
            assert 'aware' not in video['video_processing_jobs'] and not live_path.exists()
            passed.append('Aufbewahrung: verwaiste Dateien nach Neustart und ISO-Zeitzonen')

            # 8. Thread-local overrides, including cleanup during exceptions.
            person_cls = fake_people_detector_method_class()
            people = person_cls()
            people.config_manager = config
            people._people_config_local = threading.local()
            assert people._person_config()['enabled'] is False
            started = threading.Event()
            end = threading.Event()
            def test_request():
                with people.override_people_config({'enabled': True, 'model_mode': 'custom_human'}):
                    assert people._person_config()['enabled'] is True
                    started.set()
                    assert end.wait(3)
                assert people._person_config()['enabled'] is False
            thread = threading.Thread(target=test_request)
            thread.start()
            assert started.wait(3)
            assert people._person_config()['enabled'] is False, 'Live thread saw preview-only settings'
            end.set()
            thread.join(timeout=3)
            assert not thread.is_alive()
            try:
                with people.override_people_config({'enabled': True}):
                    raise RuntimeError('preview failed')
            except RuntimeError:
                pass
            assert people._person_config()['enabled'] is False
            assert config.get('people', 'enabled') is False
            passed.append('Personenerkennung: Thread-Trennung und korrekte Rücksetzung von Vorschau-Einstellungen')

            # 9. Real RTSPHandler connect/read/release via in-memory OpenCV fakes.
            rtsp_ns = {'threading': threading, 'queue': queue, 'time': time,
                       'logger': LOGGER, 'datetime': datetime, 'urlsplit': urlsplit}
            extracted(RTSP, ('RTSPHandler',), rtsp_ns)
            class Capture:
                instances = []
                def __init__(self, url, backend):
                    self.closed = False
                    self.opened = False
                    Capture.instances.append(self)
                def set(self, *_): pass
                def isOpened(self): return self.opened
                def read(self): return False, None
                def release(self): self.closed = True
            rtsp_ns['cv2'] = SimpleNamespace(VideoCapture=Capture, CAP_FFMPEG=10, CAP_PROP_BUFFERSIZE=38)
            handler = rtsp_ns['RTSPHandler'](config, None, None)
            assert not handler.connect()
            assert Capture.instances[0].closed and handler.cap is None
            assert handler.frame_queue.maxsize <= 2
            handler.frame_queue.put_nowait(np.zeros((2,2,3), dtype=np.uint8))
            handler.disconnect()
            assert handler.frame_queue.empty()
            class ExceptionCapture(Capture):
                def set(self, *_): raise RuntimeError(config.get('rtsp', 'url'))
            rtsp_ns['cv2'].VideoCapture = ExceptionCapture
            assert not handler.connect() and Capture.instances[-1].closed
            assert 'supersecret' not in handler.last_error
            passed.append('RTSP: Release nach fehlgeschlagener Verbindung, Frame-Puffer und Geheimnisschutz')

        finally:
            os.chdir(previous)
    for case in passed:
        print('  OK:', case)
    print(f'PlateVision deep regression tests: {len(passed)} Kategorien bestanden')


if __name__ == '__main__':
    run()
