#!/usr/bin/env python3
"""0.12.5: damage-safe JSON writes, restore-free backup verification and image paths.

Tests execute the real source implementations (AST extracted) without loading
heavy ML model weights or starting the Flask application.
"""
import ast
import copy
import hashlib
import io
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

APP = Path(__file__).resolve().parents[1] / 'platevision/src/app.py'
LOG = logging.getLogger('platevision-0125-test')


def extract(names, env):
    tree = ast.parse(APP.read_text(encoding='utf-8'), filename=str(APP))
    nodes = []
    for name in names:
        node = next((n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name), None)
        assert node is not None, name
        node = copy.deepcopy(node)
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(APP), 'exec'), env)


def expect_error(fn, exc_type=Exception):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError('Expected exception was not raised')


def safe_mode_tests(env):
    writer = env['_atomic_write_json']
    reader = env['_load_json_with_backup']
    cfg = Path('data/config.json')
    cfg.parent.mkdir(parents=True)
    assert reader(cfg, {}) == {}
    assert not env.get('_read_only_storage')
    writer(cfg, {'revision': 1})
    writer(cfg, {'revision': 2})
    cfg.write_text('{broken', encoding='utf-8')
    assert reader(cfg, {}) == {'revision': 1}
    assert str(cfg.resolve()) in env['_storage_recovered_from_backup']
    writer(cfg, {'revision': 3})
    assert json.loads(cfg.read_text())['revision'] == 3
    # Invalid main and backup MUST be left untouched, even if JSON parses but
    # has an incorrect root type (or the manager blindly tries to save).
    original = b'{unrecoverable primary'
    cfg.write_bytes(original)
    backup = Path(str(cfg) + '.bak')
    backup.write_text('[]', encoding='utf-8')
    assert reader(cfg, {}) == {}
    assert env['_read_only_storage'][str(cfg.resolve())]
    expect_error(lambda: writer(cfg, {'revision': 4}), OSError)
    assert cfg.read_bytes() == original
    assert backup.read_text() == '[]'
    # Array with partial damage also blocks before any record can be discarded.
    plate = Path('data/history/detections.json')
    plate.parent.mkdir(parents=True)
    plate.write_text('[{"id":"ok"},123]', encoding='utf-8')
    assert reader(plate, []) == []
    expect_error(lambda: writer(plate, [{'id': 'replacement'}]), OSError)
    assert '123' in plate.read_text()
    # A missing brand-new file is safe to initialize.
    clean = Path('data/people/history.json')
    assert reader(clean, []) == []
    expect_error(lambda: writer(clean, [{'id': 'new'}]), OSError)  # global safe mode
    # Simulate an administrator fixing files and restarting the service.
    env['_read_only_storage'].clear()
    writer(clean, [{'id': 'new'}])
    assert json.loads(clean.read_text()) == [{'id': 'new'}]
    print('  OK: Corrupt primary+backup protected, valid fallback, shape checks and clean initialization')


def archive_bytes(rows, *, extras=None, amend_manifest=None):
    manifest = {'format': 'platevision-backup', 'schema': 1, 'app_version': '0.12.5',
                'created_at': datetime.now().isoformat(), 'included': [], 'uncompressed_bytes': 0}
    fileobj = io.BytesIO()
    with zipfile.ZipFile(fileobj, 'w', compression=zipfile.ZIP_DEFLATED) as arc:
        for name, content in rows:
            arc.writestr(name, content)
            manifest['included'].append({'path': name, 'bytes': len(content),
                                         'sha256': hashlib.sha256(content).hexdigest()})
            manifest['uncompressed_bytes'] += len(content)
        if extras:
            for name, content in extras:
                arc.writestr(name, content)
        if amend_manifest:
            amend_manifest(manifest)
        arc.writestr('platevision-backup-manifest.json', json.dumps(manifest))
    fileobj.seek(0)
    return fileobj


def archive_tests(env):
    verify = env['_verify_backup_archive']
    metadata = [('data/config.json', b'{"general":{"enabled":true}}'),
                ('data/history/detections.json', b'[\n{"id":"x"}\n]'),
                ('data/people/images/crops/2026-01-01/first.jpg', b'JPEG-DATA')]
    archive = archive_bytes(metadata)
    result = verify(archive)
    assert result['verified_files'] == len(metadata)
    assert result['verified_bytes'] == sum(map(lambda r: len(r[1]), metadata))
    assert result['source_version'] == '0.12.5'
    assert archive.tell() > 0
    wrong_hash = archive_bytes(metadata, amend_manifest=lambda m: m['included'][0].update(sha256='0' * 64))
    expect_error(lambda: verify(wrong_hash), ValueError)
    extra = archive_bytes(metadata, extras=[('data/watchlist.json', b'[]')])
    expect_error(lambda: verify(extra), ValueError)
    traversal = archive_bytes([('../secrets.txt', b'bad')])
    expect_error(lambda: verify(traversal), ValueError)
    bad_dir = archive_bytes([('data/people/images/../../escape.jpg', b'x')])
    expect_error(lambda: verify(bad_dir), ValueError)
    dup = archive_bytes([('data/config.json', b'one'), ('data/config.json', b'two')])
    expect_error(lambda: verify(dup), ValueError)
    oversize = archive_bytes(metadata)
    expect_error(lambda: verify(oversize, max_bytes=4), ValueError)
    unsupported_schema = archive_bytes(metadata, amend_manifest=lambda m: m.update(schema=8))
    expect_error(lambda: verify(unsupported_schema), ValueError)
    fake_manifest = io.BytesIO()
    with zipfile.ZipFile(fake_manifest, 'w') as arc:
        arc.writestr('data/config.json', '{}')
    fake_manifest.seek(0)
    expect_error(lambda: verify(fake_manifest), ValueError)
    # Verify an actual archive generated by the app's own writer.
    backup_root = Path('backuptest')
    (backup_root/'data/people/images/crops').mkdir(parents=True)
    (backup_root/'data/config.json').write_text('{"some":"secret"}', encoding='utf-8')
    (backup_root/'data/people/images/crops/a.jpg').write_bytes(b'preview')
    generated = io.BytesIO()
    write = env['_write_backup_archive']
    write(generated, root=backup_root)
    generated.seek(0)
    assert verify(generated)['verified_files'] == 2
    print('  OK: Real backup roundtrip, manifests, checksum mismatch, duplicates, ZIP-slip, size and schema limits')


def image_tests(env):
    import numpy as np
    import cv2
    class Config:
        def get(self, *keys):
            values = {'people': {'history_enabled': True, 'image_history_enabled': True,
                                 'save_person_crops': True, 'image_history_store_crop': True,
                                 'image_history_store_annotated': False,
                                 'image_history_auto_cleanup_enabled': False,
                                 'blur_strength': 35}}
            value = values
            for k in keys:
                value = value.get(k) if isinstance(value, dict) else None
            return value
    ns = dict(env, cv2=cv2, np=np, config_manager=Config(),
              contextmanager=__import__('contextlib').contextmanager)
    extract(('PersonHistoryManager',), ns)
    people = ns['PersonHistoryManager']()
    sample = np.zeros((16, 16, 3), dtype=np.uint8)
    assert people._save_jpeg(sample, Path('crops/a.jpg')) == 'crops/a.jpg'
    filename = people.IMAGE_ROOT / 'crops/a.jpg'
    old = filename.read_bytes()
    expect_error(lambda: people._save_jpeg(sample, Path('../bad.jpg')), ValueError)
    outside = Path('notperson.jpg')
    assert not outside.exists()
    other_root = Path('outside-folder')
    other_root.mkdir()
    link = people.IMAGE_ROOT/'crops/escape'
    link.symlink_to(other_root.resolve(), target_is_directory=True)
    expect_error(lambda: people._save_jpeg(sample, Path('crops/escape/x.jpg')), ValueError)
    assert not (other_root/'x.jpg').exists()
    original_replace = os.replace
    try:
        os.replace = lambda *a: (_ for _ in ()).throw(OSError('no space for replacement'))
        expect_error(lambda: people._save_jpeg(sample, Path('crops/a.jpg')), OSError)
    finally:
        os.replace = original_replace
    assert filename.read_bytes() == old
    assert not list(filename.parent.glob('.a.jpg.*.tmp'))
    event = {'id': '../../../unsafe', 'timestamp': '../../..', 'bbox': [0,0,16,16]}
    people.attach_images(event, frame=sample)
    assert event.get('images', {}).get('crop', '').startswith('crops/')
    assert '..' not in event['images']['crop']
    print('  OK: Atomic person image updates, safe date and ID paths, symlink and traversal protection')


def request_gate_tests(env):
    env['_read_only_storage']['simulated'] = 'file unreadable'
    fake = dict(env, request=SimpleNamespace(method='POST', endpoint='api_history_clear'),
                jsonify=lambda v: v)
    extract(('_prevent_mutations_in_storage_safe_mode',), fake)
    response = fake['_prevent_mutations_in_storage_safe_mode']()
    assert response[1] == 503 and 'Schreibschutz' in response[0]['error']
    fake['request'].endpoint = 'api_stream_stop'
    assert fake['_prevent_mutations_in_storage_safe_mode']() is None
    fake['request'].endpoint = 'api_system_backup_verify'
    assert fake['_prevent_mutations_in_storage_safe_mode']() is None
    fake['request'].method = 'GET'
    assert fake['_prevent_mutations_in_storage_safe_mode']() is None
    print('  OK: Mutations blocked in safe mode, stop-stream and read-only verification allowed')


def export_tests(env):
    import csv
    class Args(dict):
        def to_dict(self):
            return dict(self)
    class Response:
        def __init__(self, iterable, mimetype=None, **kwargs):
            self.iterable = iterable
            self.mimetype = mimetype
            self.headers = {}
        def read(self):
            return ''.join(self.iterable)
    captured = []
    class History:
        def search_all(self, filters):
            captured.append(dict(filters))
            return [dict(id=str(n), timestamp='2026-01-01', plate_text=("=2+3" if n == 0 else f'ZH{n}'),
                         plate_image='SECRETBASE64', confidence=.9) for n in range(1507)]
    request = SimpleNamespace(args=Args(format='json', limit='1000', include_images='false'))
    api_ns = dict(env, request=request, history_manager=History(),
                  Response=Response, io=io, csv=csv,
                  _csv_safe_cell=lambda x: "'" + x if isinstance(x, str) and x.startswith('=') else x,
                  _invalid_search_regex=lambda f: False,
                  _history_api_entries=lambda entries, f: [{k: v for k, v in row.items() if k != 'plate_image'} for row in entries],
                  config_manager=SimpleNamespace(get=lambda *args: False),
                  jsonify=lambda x: x)
    extract(('api_history_export',), api_ns)
    response = api_ns['api_history_export']()
    result = json.loads(response.read())
    assert len(result) == 1507 and all('plate_image' not in row for row in result)
    assert 'limit' not in captured[-1] and 'offset' not in captured[-1]
    request.args['format'] = 'csv'
    csvdata = api_ns['api_history_export']().read()
    assert len(csvdata.splitlines()) == 1508
    assert list(csv.reader(io.StringIO(csvdata)))[1][1] == "'=2+3"
    request.args['format'] = 'bad'
    assert api_ns['api_history_export']()[1] == 400
    print('  OK: Full history streaming exports >1000 rows, metadata only and safe CSV')


def main():
    cwd = Path.cwd()
    env = dict(Path=Path, os=os, json=json, logger=LOG, logging=logging,
               threading=threading, uuid=uuid, shutil=shutil, hashlib=hashlib,
               tempfile=tempfile, re=re, secrets=secrets, zipfile=zipfile,
               datetime=datetime, APP_VERSION='0.12.5',
               _BACKUP_METADATA=('data/config.json', 'data/history/detections.json', 'data/people/history.json', 'data/watchlist.json', 'data/video_jobs.json'),
               _BACKUP_IMAGE_DIRS=('data/plates_detected', 'data/vehicles_detected', 'data/people/images'),
               _SQLITE_BACKUP_DATA=(('plates', 'data/history/detections.json', 'data/history/plates.sqlite3'),
                                    ('people', 'data/people/history.json', 'data/people/people.sqlite3')))
    extract(('_json_clone', '_atomic_write_json', '_load_json_with_backup',
             '_backup_inventory', '_write_backup_archive', '_verify_backup_archive'), env)
    with tempfile.TemporaryDirectory(prefix='platevision-0125-') as folder:
        os.chdir(folder)
        try:
            safe_mode_tests(env)
            archive_tests(env)
            image_tests(env)
            request_gate_tests(env)
            export_tests(env)
        finally:
            os.chdir(cwd)
    print('PlateVision 0.12.5 safe-mode and backup tests: 5 Bereiche bestanden')


if __name__ == '__main__':
    main()
