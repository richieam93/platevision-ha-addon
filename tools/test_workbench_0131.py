#!/usr/bin/env python3
"""PlateVision 0.13.1 presentation policy, SQLite parity and expanded crop tests.

No trained model, Flask server, or real camera required.
"""
from __future__ import annotations
import ast
import copy
import json
import logging
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'platevision/src'
sys.path.insert(0,str(SRC))
from vehicle_display import best_image_candidate
from history_sqlite import SQLiteHistoryStore


def item(i, plate, conf, time, source='video_upload', image=True):
    return dict(id=i,plate_text=plate,confidence=conf,timestamp=time,source=source,
                vehicle_image=('QUJD' if image else None))


def test_best_vehicle():
    newest = item('last','JK A 0011',.57,'2026-10-10T11:35:15')
    good = item('best','JK A 0011',.93,'2026-10-10T11:35:03')
    lower = item('earlier','JK A 0011',.85,'2026-10-10T11:34:56')
    wrong = item('othercar','XYZ 987',.999,'2026-10-10T11:35:04')
    rows = [newest,wrong,good,lower]
    assert best_image_candidate(rows, minimum=.8)['id']=='best'
    assert best_image_candidate(rows, minimum=.94) is None
    assert best_image_candidate(rows, minimum=.8,window_seconds=10) is None
    assert best_image_candidate([newest,lower], minimum=.8,window_seconds=10) is None
    assert best_image_candidate([item('only','ABC123',.5,'2026-10-10T09:00:00')]) is None
    assert best_image_candidate([item('only','ABC123',.95,'2026-10-10T09:00:00',image=False)]) is None
    assert best_image_candidate([item('only','ABC123','NaN','2026-10-10T09:00:00')]) is None
    assert best_image_candidate([]) is None
    assert best_image_candidate([item('only','ABC123',.95,'garbage')])['id']=='only'
    assert best_image_candidate([item('only','ABC123',.95,'2026-10-10T09:00:00')],minimum=2) is None
    print('  OK: high-confidence selection, variants, no image, nonnumeric scores and time boundaries')


def test_sqlite_display():
    with tempfile.TemporaryDirectory(prefix='pv-workbench-') as td:
        p=Path(td)/'history/detections.json'
        p.parent.mkdir(parents=True)
        original=[item('latest','JK A 0011',.571,'2026-10-10T11:35:15'),
                  item('other','XX 9988',.992,'2026-10-10T11:35:06'),
                  item('best','JK A 0011',.932,'2026-10-10T11:35:02'),
                  item('low','JK A 0011',.42,'2026-10-10T11:34:55'),
                  item('old','JK A 0011',.995,'2026-10-10T11:12:35')]
        p.write_text(json.dumps(original))
        before=p.read_bytes()
        store=SQLiteHistoryStore(p,kind='plates')
        assert store.initialize(lambda:json.loads(p.read_text()))==original
        assert store.best_vehicle_for_display(minimum=.8)['id']=='best'
        assert store.best_vehicle_for_display(minimum=.95) is None
        assert len(store.load())==5 and p.read_bytes()==before
        ids=[r['id'] for r in store.load()]
        assert ids==['latest','other','best','low','old']
        assert store.search_plates({},limit=20)[1]==5
        assert len(store.search_plates({},limit=20)[0])==5
        assert store.best_vehicle_for_display(minimum=.80,window_seconds=10) is None
        print('  OK: SQLite selects best metadata+image, does not mutate search/history, minimum and burst gate')


def test_crop():
    app_ast=ast.parse((SRC/'app.py').read_text())
    mgr=next(n for n in app_ast.body if isinstance(n,ast.ClassDef) and n.name=='PersonHistoryManager')
    method=copy.deepcopy(next(n for n in mgr.body if isinstance(n,ast.FunctionDef) and n.name=='attach_images'))
    method.decorator_list=[]
    namespace={'datetime':datetime,'Path':Path,'re':re,'uuid':uuid,'logger':logging.getLogger('test-crop')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(SRC/'app.py'),'exec'),namespace)
    class FakeMgr:
        attach_images=namespace['attach_images']
        def __init__(self,padding):
            self.padding=padding
            self.shapes=[]
        def _cfg(self):
            return dict(image_history_enabled=True,save_person_crops=True,image_history_store_annotated=False,
                        image_history_crop_padding_percent=self.padding,image_history_crop_padding_px=0)
        def _parse_datetime(self,s): return datetime.fromisoformat(s)
        def _save_jpeg(self,crop,path,quality):
            self.shapes.append(crop.shape)
            return str(path)
    frame=np.zeros((480,640,3),dtype=np.uint8)
    for padding in (0,25,100,150):
        fake=FakeMgr(padding)
        entry={'id':f'crop-{padding}','timestamp':'2026-10-10T10:00:00','bbox':[220,170,310,340]}
        images=fake.attach_images(entry,frame=frame)
        assert images.get('crop') and fake.shapes and fake.shapes[0][0]>=170
        if padding==100:
            assert fake.shapes[0][0]>=380, fake.shapes
            assert fake.shapes[0][1]>170, fake.shapes
    print('  OK: person crop gains surrounding pixels, frame bounds are respected')


def test_templates():
    test=(SRC/'templates/test.html').read_text()
    people=(SRC/'templates/people.html').read_text()
    dashboard=(SRC/'templates/dashboard.html').read_text()
    latest=(SRC/'templates/latest.html').read_text()
    settings=(SRC/'templates/settings.html').read_text()
    assert test.index('id="testUploadWorkspace"') < test.index('id="recognitionSettingsDetails"')
    assert test.index('id="peopleCalibrationWorkspace"') < test.index('id="recognitionSettingsDetails"')
    assert 'peopleLinePreviewShell' in test and 'peopleLinePreviewShell' not in people
    assert 'id="testPeopleCropPadding"' in test
    assert 'id="showOnlyBestVehicle"' in settings and 'id="bestVehicleMinConfidence"' in settings
    assert '/api/latest/display' in latest
    assert 'vehicleQualityHint' in dashboard
    assert 'object-fit:contain' in dashboard
    print('  OK: moved calibration, progressive settings, existing IDs, display endpoint')


if __name__=='__main__':
    test_best_vehicle()
    test_sqlite_display()
    test_crop()
    test_templates()
    print('PASS PlateVision 0.13.1 functionality, data preservation and layout invariants')
