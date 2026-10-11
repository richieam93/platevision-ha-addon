#!/usr/bin/env python3
"""0.13.4 regression checks: streamed person sessions, image calendar and safe cleanup.

These tests exercise the actual class and SQLite module without Flask/YOLO/RTSP.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'platevision/src'))
from history_sqlite import SQLiteHistoryStore
from test_people_history import _load_manager_class


def rows_for_test(now):
    result = []
    for index in range(740):
        day = (now - timedelta(days=index % 17)).date().isoformat()
        images = ({'crop': f'crops/test/img-{index}.jpg', 'annotated': f'annotated/test/{index}.jpg'}
                  if index % 3 == 0 else {})
        result.append({'id': f'evt-{index}', 'timestamp': f'{day}T12:{index%60:02d}:00',
                       'source': 'video_upload' if index % 5 else 'rtsp',
                       'track_id': index % 19 if index % 6 else None,
                       'bbox': [10, 20, 80, 180], 'center_x': 40, 'center_y': 90,
                       'frame_width': 1280, 'frame_height': 720,
                       'confidence': .8, 'direction': 'down', 'counted': index % 4 != 0,
                       'event_type': 'line_crossing', 'images': images})
    return result


def test_calendar_and_sessions():
    cls = _load_manager_class()
    with tempfile.TemporaryDirectory(prefix='pv-0134-large-') as folder:
        old_cwd = Path.cwd()
        os.chdir(folder)
        try:
            manager = cls()
            rows = rows_for_test(datetime.now())
            manager.history = rows
            manager._save_or_revert([])
            classic_days = manager.image_days({'days': '3650', 'counted_only': 'false'})
            classic_sessions = manager.get_sessions({'days': '3650', 'counted_only': 'false', 'limit': 30})
            classic_reverse = manager.get_sessions({'days': '3650', 'counted_only': 'false', 'sort': 'asc', 'limit': 30})
            store = SQLiteHistoryStore(manager.HISTORY_FILE, kind='people')
            store.initialize(lambda: json.loads(Path(manager.HISTORY_FILE).read_text()))
            manager._sqlite_store = store
            # An index must calculate the calendar without iterating all person JSON payloads.
            manager.search = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError('Full SQLite people history unnecessarily deserialized'))
            days = manager.image_days({'days': '3650', 'counted_only': 'false'})
            assert days['days'] == classic_days['days'], 'SQL image day aggregation differs from legacy'
            assert days['total_events_with_images'] == classic_days['total_events_with_images']
            # Full sessions should stream in chronological order instead of loading all JSON rows.
            sessions = manager.get_sessions({'days': '3650', 'counted_only': 'false', 'limit': 30})
            assert sessions['total'] == classic_sessions['total']
            assert sessions['sessions'] == classic_sessions['sessions'], 'Streaming session aggregation differs'
            backward = manager.get_sessions({'days': '3650', 'counted_only': 'false', 'sort': 'asc', 'limit': 30})
            assert backward['sessions'] == classic_reverse['sessions'], 'Reverse session ordering changed'
        finally:
            os.chdir(old_cwd)
    print('  OK: SQLite image-day aggregation, streaming sessions, legacy parity (740 records)')


def test_image_retention_protects_live_references():
    cls = _load_manager_class()
    with tempfile.TemporaryDirectory(prefix='pv-0134-retention-') as folder:
        old_cwd = Path.cwd()
        os.chdir(folder)
        try:
            manager = cls()
            current = datetime.now()
            root = manager.IMAGE_ROOT
            live = root / 'crops' / 'test' / 'still-in-use.jpg'
            shared = root / 'crops' / 'test' / 'shared.jpg'
            orphan = root / 'crops' / 'test' / 'orphan.jpg'
            for path in (live, shared, orphan):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'preserve if linked')
                old_time = (current-timedelta(days=50)).timestamp()
                os.utime(path, (old_time, old_time))
            manager.history = [
                {'id': 'live', 'timestamp': current.isoformat(), 'images': {'crop': live.relative_to(root).as_posix()}},
                {'id': 'shared-new', 'timestamp': current.isoformat(), 'images': {'crop': shared.relative_to(root).as_posix()}},
                {'id': 'shared-old', 'timestamp': (current - timedelta(days=30)).isoformat(),
                 'images': {'crop': shared.relative_to(root).as_posix()}},
            ]
            manager._save_or_revert([])
            result = manager.cleanup_images(10, delete_orphan_files=True)
            assert live.exists(), 'A retained event image was removed based on file mtime alone'
            assert shared.exists(), 'Shared image still referenced by current event removed'
            assert not orphan.exists(), 'Actual unreferenced expired image was not removed'
            assert result['cleared_image_events'] == 1
        finally:
            os.chdir(old_cwd)
    print('  OK: retention protects referenced images even when mtime is older')


def test_malformed_people_geometry_and_cleanup_api():
    cls = _load_manager_class()
    with tempfile.TemporaryDirectory(prefix='pv-0134-corrupt-') as folder:
        old_cwd = Path.cwd()
        os.chdir(folder)
        try:
            manager = cls()
            manager.history = [{'id': 'bad1', 'timestamp': datetime.now().isoformat(),
                                'center_x': 'bad-value', 'bbox': {'x': 1},
                                'frame_width': 'unusable', 'confidence': 'bad', 'counted': False},
                               {'id': 'bad2', 'timestamp': datetime.now().isoformat(),
                                'center_x': None, 'bbox': 42, 'counted': False}]
            assert manager.get_sessions({'days': 1})['total'] == 2
            # An unsupported retention setting should not crash a periodic cleanup.
            result = manager.cleanup('invalid')
            assert result['retention_days'] == 0 and result['removed'] == 0
        finally:
            os.chdir(old_cwd)
    print('  OK: corrupted person geometry and invalid retention do not cause HTTP 500')



def test_invalid_cleanup_requests_and_timezone():
    # Directly execute the real HTTP handler with simple fakes, without loading ML.
    tree = ast.parse((ROOT/'platevision/src/app.py').read_text())
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='api_people_cleanup'))
    node.decorator_list=[]
    recorded=[]
    class FakeRequest:
        value={}
        def get_json(self, silent=False): return self.value
    fake=FakeRequest()
    env={'request':fake,'jsonify':lambda value:value,
         'person_history_manager':SimpleNamespace(cleanup=lambda days: recorded.append(days) or {'removed':0})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])), '<people cleanup>', 'exec'),env)
    handle=env['api_people_cleanup']
    for payload in ([1,2], {'retention_days':'nope'}, {'retention_days':-1}, {'retention_days':4000}):
        fake.value=payload
        assert handle()[1]==400, payload
        assert recorded==[]
    fake.value={'retention_days':10}
    assert handle()['success'] and recorded==[10]

    # Offset-qualified API date filters must use the local-time indexed timestamp.
    if hasattr(time, 'tzset'):
        original_tz=os.environ.get('TZ')
        try:
            os.environ['TZ']='Europe/Berlin'
            time.tzset()
            cls=_load_manager_class()
            with tempfile.TemporaryDirectory(prefix='pv-0134-tz-') as td:
                previous=Path.cwd()
                os.chdir(td)
                try:
                    mgr=cls()
                    mgr.history=[{'id':'early', 'timestamp':'2026-10-10T00:30:00+02:00','images':{'crop':'crops/early.jpg'}},
                                 {'id':'later', 'timestamp':'2026-10-10T04:30:00+02:00','images':{'crop':'crops/later.jpg'}}]
                    mgr._save_or_revert([])
                    query={'date_from':'2026-10-10T00:00:00Z','date_to':'2026-10-10T23:59:00Z','counted_only':'false'}
                    fallback=mgr.search(query)
                    assert [row['id'] for row in fallback]==['later'],fallback
                    store=SQLiteHistoryStore(mgr.HISTORY_FILE,kind='people')
                    store.initialize(lambda:json.loads(Path(mgr.HISTORY_FILE).read_text()))
                    mgr._sqlite_store=store
                    assert [row['id'] for row in mgr.search(query)]==['later']
                finally:
                    os.chdir(previous)
        finally:
            if original_tz is None: os.environ.pop('TZ',None)
            else: os.environ['TZ']=original_tz
            time.tzset()
    print('  OK: 400 on bad cleanup requests and timezone-aware SQLite filters')

if __name__ == '__main__':
    test_calendar_and_sessions()
    test_image_retention_protects_live_references()
    test_malformed_people_geometry_and_cleanup_api()
    test_invalid_cleanup_requests_and_timezone()
    print('PASS PlateVision 0.13.4 large-history safety and compatibility')
