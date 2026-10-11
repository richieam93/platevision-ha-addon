#!/usr/bin/env python3
"""PlateVision 0.13.2: blur-aware selection, upload isolation, compact API and daily SQL.

Self-contained deterministic tests. No full Flask app, RTSP camera or ML weights needed.
"""
import ast
import copy
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'platevision/src'
sys.path.insert(0, str(SRC))
from history_sqlite import SQLiteHistoryStore
from vehicle_display import best_image_candidate, plate_image_sharpness


def capture(id, text='JK A 0011', conf=.9, sharpness=None, source='rtsp', timestamp='2026-10-10T11:30:00', **kwargs):
    row=dict(id=id, plate_text=text, confidence=conf, timestamp=timestamp,
             source=source, plate_image='QUJD', plate_sharpness=sharpness)
    row.update(kwargs)
    return row


def test_sharpness_and_blur():
    clear=np.full((60,220,3),220,dtype=np.uint8)
    cv2.putText(clear,'ZH 0011',(12,42),cv2.FONT_HERSHEY_SIMPLEX,1.1,(20,20,20),3)
    blur=cv2.GaussianBlur(clear,(21,21),4)
    sharp=plate_image_sharpness(clear)
    soft=plate_image_sharpness(blur)
    assert sharp is not None and soft is not None and 0 <= soft < sharp <= 1, (sharp,soft)
    assert plate_image_sharpness(None) is None
    assert plate_image_sharpness(np.zeros((2,3,3),dtype=np.uint8)) is None
    rows=[capture('latest',conf=.83,sharpness=.95),
          capture('blurred',conf=.98,sharpness=.02,timestamp='2026-10-10T11:29:50')]
    assert best_image_candidate(rows,prefer_sharpness=True)['id']=='latest'
    assert best_image_candidate(rows,prefer_sharpness=False)['id']=='blurred'
    assert best_image_candidate(rows, minimum=.90,prefer_sharpness=True)['id']=='blurred'
    assert best_image_candidate(rows,minimum=.99) is None
    # Missing metadata on any qualifying row must not be treated as sharpness 0.
    rows[0].pop('plate_sharpness')
    assert best_image_candidate(rows)['id']=='blurred'
    assert best_image_candidate(rows,sharpness_weight=float('nan')) is None
    print(f'  OK: deterministic image blur metric ({soft:.2f} vs {sharp:.2f}), threshold and legacy fallback')


def test_video_job_and_upload_isolation():
    rows=[capture('a',conf=.82,source='video_upload',video_job_id='job-1',filename='a.mp4'),
          capture('b',conf=.99,source='video_upload',video_job_id='job-2',filename='a.mp4',
                  timestamp='2026-10-10T11:29:59'),
          capture('c',conf=.90,source='video_upload',video_job_id='job-1',filename='a.mp4',
                  timestamp='2026-10-10T11:29:58')]
    assert best_image_candidate(rows)['id']=='c'
    rows=[capture('a',conf=.83,source='video_upload',filename='a.mp4'),
          capture('b',conf=.96,source='video_upload',filename='b.mp4',timestamp='2026-10-10T11:29:58')]
    assert best_image_candidate(rows)['id']=='a'
    rows=[capture('a',conf=.83,source='image_upload',filename='a.jpg',upload_id='batch-1'),
          capture('b',conf=.96,source='image_upload',filename='a.jpg',upload_id='batch-2',timestamp='2026-10-10T11:29:58')]
    assert best_image_candidate(rows)['id']=='a'  # Same filename, distinct uploads.
    rows=[capture('a',conf=.83,source='image_upload',filename='a.jpg'),
          capture('b',conf=.96,source='image_upload',filename='b.jpg',timestamp='2026-10-10T11:29:58')]
    assert best_image_candidate(rows)['id']=='a'
    # Older records with no identifiers preserve the old source/time/plate behavior.
    rows=[capture('a',conf=.83,source='video_upload'),
          capture('b',conf=.96,source='video_upload',timestamp='2026-10-10T11:29:58')]
    assert best_image_candidate(rows)['id']=='b'
    assert best_image_candidate([capture('different-source', source='rtsp'),rows[1]])['id']=='different-source'
    print('  OK: no mixing of different video jobs, manual uploads or cameras')


def test_sqlite_and_daily():
    rows=[capture('a',conf=.83,sharpness=.95,source='video_upload',video_job_id='job-1',filename='a.mp4'),
          capture('wrong',conf=.99,sharpness=1.0,source='video_upload',video_job_id='job-2',filename='a.mp4',timestamp='2026-10-10T11:29:59'),
          capture('blur',conf=.98,sharpness=.02,source='video_upload',video_job_id='job-1',filename='a.mp4',timestamp='2026-10-10T11:29:55'),
          capture('nearly-same','JK A 0012',conf=.86,sharpness=.60,source='video_upload',video_job_id='job-1',filename='a.mp4',timestamp='2026-10-10T11:29:54'),
          capture('unrelated', 'AG 999',conf=.93,sharpness=.99, source='video_upload',video_job_id='job-1',filename='a.mp4',timestamp='2026-10-10T11:29:53')]
    with tempfile.TemporaryDirectory(prefix='pv-0132-') as td:
        path=Path(td)/'history/detections.json';path.parent.mkdir()
        path.write_text(json.dumps(rows),encoding='utf-8')
        before=path.read_bytes()
        store=SQLiteHistoryStore(path,kind='plates')
        assert store.initialize(lambda:json.loads(path.read_text()))==rows
        assert store.best_vehicle_for_display(prefer_sharpness=True)['id']=='a'
        assert store.best_vehicle_for_display(prefer_sharpness=False)['id']=='blur'
        assert store.best_vehicle_for_display(minimum=.99) is None
        assert store.search_plates({},limit=20)[1]==5
        assert len(store.load())==5 and path.read_bytes()==before
        # No timestamp leakage: date and exact/fuzzy plate grouping must match
        # legacy newest-first order, without loading image JSON on the SQL route.
        assert store.today_plate_counts(now=datetime(2026,10,10),fuzzy=False)==(5,3)
        assert store.today_plate_counts(now=datetime(2026,10,10),fuzzy=True,similarity=.80)==(5,2)
        assert store.today_plate_counts(now=datetime(2026,10,11))==(0,0)
    print('  OK: compact SQL reads, fuzzy/exact daily counters and preserved event data')


def test_compact_display_endpoint():
    source=ast.parse((SRC/'app.py').read_text(encoding='utf-8'))
    extracts=[]
    for name in ('_entry_image_flags','api_latest_display'):
        fn=copy.deepcopy(next(f for f in source.body if isinstance(f,ast.FunctionDef) and f.name==name))
        fn.decorator_list=[]
        extracts.append(fn)
    env={'request':SimpleNamespace(args={}), 'jsonify':lambda payload:payload,
         '_presentation_vehicle_settings':lambda:(True,.8,180,True,.25)}
    row=capture('x',conf=.97,sharpness=.84)
    row['vehicle_image']='dGhpcyBpcyBub3QgYSBqcGVn'
    env['_display_vehicle_entry']=lambda:row
    exec(compile(ast.fix_missing_locations(ast.Module(body=extracts,type_ignores=[])),
                 str(SRC/'app.py'),'exec'),env)
    endpoint=env['api_latest_display']
    env['request'].args={'images':'0'}
    compact=endpoint()
    assert 'vehicle_image' not in compact and 'plate_image' not in compact
    assert compact['vehicle_image_url']=='/api/history/x/image/vehicle'
    assert compact['plate_image_url']=='/api/history/x/image/plate'
    assert compact['plate_sharpness']==.84
    env['request'].args={}
    legacy=endpoint()
    assert legacy['vehicle_image']==row['vehicle_image']
    assert legacy['plate_image']==row['plate_image']
    tpl=(SRC/'templates/latest.html').read_text()
    assert '/api/latest/display?images=0' in tpl
    assert 'pvSafeImageUrl(data.vehicle_image_url)' in tpl
    assert 'pvSafeImageUrl(data.plate_image_url)' in tpl
    print('  OK: compact API avoids base64, legacy API retained; URLs sanitized by UI')


if __name__=='__main__':
    test_sharpness_and_blur()
    test_video_job_and_upload_isolation()
    test_sqlite_and_daily()
    test_compact_display_endpoint()
    print('PASS PlateVision 0.13.2 image quality, upload isolation, performance & compatibility')
