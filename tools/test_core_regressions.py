#!/usr/bin/env python3
"""Fast dependency-free regression tests. Never imports torch, YOLO, or model files."""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = ROOT / 'platevision/src/app.py'
RTSP_SOURCE = ROOT / 'platevision/src/rtsp_handler.py'


class Config:
    def __init__(self):
        self.data = {
            'search': {'default_limit': 20, 'allow_regex_search': True, 'enable_fuzzy_search': False},
            'history': {'filter_duplicates': True, 'duplicate_timeout': 60, 'fuzzy_duplicate_similarity': .88},
            'general': {'max_history_entries': 1000},
            'rtsp': {'url': 'rtsp://alice:p%40ss@192.168.0.22:554/private?token=sensitive', 'reconnect_delay': 30},
            'people': {'enabled': False},
        }

    def get(self, *keys):
        value = self.data
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value


def get_classes(path: Path, names: tuple[str, ...], namespace: dict):
    parsed = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    selected = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert len(selected) == len(names), f'Missing symbol in {path}: {set(names) - {n.name for n in selected}}'
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


def main():
    config = Config()
    logger = logging.getLogger('platevision-core-test')
    namespace = {
        'Path': Path, 'json': json, 'logging': logging, 'logger': logger,
        'os': os, 'shutil': shutil, 'threading': threading, 'uuid': uuid, 'time': time,
        'hashlib': hashlib, 're': re, 'Counter': Counter, 'defaultdict': defaultdict,
        'SequenceMatcher': SequenceMatcher, 'datetime': datetime, 'timedelta': timedelta,
        'config_manager': config,
    }
    get_classes(APP_SOURCE, ('_page_integer', '_safe_numeric', '_json_clone', '_atomic_write_json', '_load_json_with_backup', 'PlateUtils', 'HistoryManager', 'WatchlistManager'), namespace)
    page = namespace['_page_integer']
    assert page('no', 20, 1, 200) == 20
    assert page('-2', 20, 1, 200) == 1
    assert page('999999999999', 20, 1, 200) == 200
    assert page('', 20, 1, 200) == 20
    assert page(None, 20, 1, 200) == 20
    assert namespace['_safe_numeric']('NaN', .35) == .35
    assert namespace['PlateUtils'].normalize('  zh - 1234 ', compact=True) == 'ZH1234'

    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='platevision-regression-') as temp:
        os.chdir(temp)
        try:
            atomic = namespace['_atomic_write_json']
            load = namespace['_load_json_with_backup']
            target = Path('data/config.json')
            atomic(target, {'revision': 1})
            atomic(target, {'revision': 2})
            backup = target.with_name('config.json.bak')
            assert json.loads(backup.read_text()) == {'revision': 1}
            target.write_text('{broken json', encoding='utf-8')
            assert load(target, {}) == {'revision': 1}, 'corrupt primary must recover'
            atomic(target, {'revision': 3})
            assert json.loads(backup.read_text()) == {'revision': 1}, 'healthy backup may not be clobbered'
            assert load(target, {}) == {'revision': 3}
            histories = namespace['HistoryManager']()
            histories.history = [
                {'id': 'one', 'plate_text': 'ZH-1234', 'confidence': .9, 'timestamp': '2026-09-09T13:00:00'},
                {'id': 'two', 'plate_text': 'ZH 1234', 'confidence': .8, 'timestamp': '2026-09-08T13:00:00'},
                {'id': 'three', 'plate_text': 'BE 9876', 'confidence': .8, 'timestamp': '2026-09-07T13:00:00'},
            ]
            paged = histories.search_advanced({'limit': 'garbage', 'offset': '-30'})
            assert paged['limit'] == 20 and paged['offset'] == 0 and paged['total'] == 3
            assert histories.search_advanced({'unique': True, 'limit': 100})['total'] == 2
            assert histories.search_advanced({'limit': 999999})['limit'] == 1000
            assert histories.search_advanced({'limit': -200})['limit'] == 1
            assert len(histories.get_all(limit=-9, offset=-9)) == 1
            histories.history.append({'id':'tz', 'plate_text':'GE 11', 'timestamp':'2026-09-07T12:00:00+02:00'})
            assert histories.search_advanced({'date_from':'2026-09-07', 'date_to':'2026-09-09'})['total'] == 4
            assert histories.search_advanced({'min_confidence':'bad'})['total'] == 4

            watch = namespace['WatchlistManager']()
            entry = watch.add('ZH-1234', label='Besucher')
            assert watch.check('ZH 1234')['label'] == 'Besucher'
            assert watch.delete(entry['id'])
            assert watch.check('ZH1234') is None
        finally:
            os.chdir(original_cwd)

    rtsp_ns = {
        'threading': threading, 'queue': queue, 'time': time, 'logger': logger,
        'urlsplit': urlsplit, 'datetime': datetime,
    }
    get_classes(RTSP_SOURCE, ('RTSPHandler',), rtsp_ns)
    Handler = rtsp_ns['RTSPHandler']
    stream = Handler(config, None, detector=None, person_history_manager=None)
    visible = stream.get_status()
    assert visible['status'] == 'stopped'
    assert visible['url'] == 'rtsp://192.168.0.22:554'
    assert not any(secret in str(visible) for secret in ('alice', 'sensitive', 'p%40ss', '/private'))
    assert stream._safe_error_text('Fehler: ' + config.get('rtsp', 'url')) == 'Fehler: rtsp://192.168.0.22:554'
    stream.running = True
    assert stream.get_status()['status'] == 'connecting'
    stream.running = False
    stream.connected = True
    stream.running = True
    assert stream.get_status()['status'] == 'running'
    stream.running = False
    stream.connected = False
    # Failed connections should not block stop() during long reconnect delays.
    stream.connect = lambda: False
    assert stream.start()
    time.sleep(.08)
    start = time.monotonic()
    stream.stop()
    elapsed = time.monotonic() - start
    assert elapsed < 1.8, f'reconnect wait should stop quickly ({elapsed:.2f}s)'
    assert not stream.capture_thread.is_alive()

    # Werkzeug debugger must not be enabled on the network port.
    source = APP_SOURCE.read_text()
    assert 'debug=False, use_reloader=False' in source
    assert "if job.get('status') in ('queued', 'processing', 'cancelling'):" in source
    print('PlateVision core regression tests: OK')


if __name__ == '__main__':
    main()
