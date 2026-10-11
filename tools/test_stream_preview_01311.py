#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Regression for HA-Ingress-safe JPEG preview (no live RTSP hardware)."""
import ast
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
class Response:
    def __init__(self, data=b'', *, status=200, mimetype=None):
        self.data = data
        self.status_code = status
        self.mimetype = mimetype
        self.headers = {}


def jsonify(obj):
    return obj

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'platevision/src'

class PreviewStream:
    def __init__(self):
        self.frame = None

    def get_current_frame(self):
        return None if self.frame is None else self.frame.copy()


def check_preview_endpoint():
    tree=ast.parse((SRC/'app.py').read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='api_stream_preview')
    isolated=ast.FunctionDef(name=function.name,args=function.args,body=function.body,decorator_list=[],returns=function.returns,type_comment=None)
    module=ast.fix_missing_locations(ast.Module(body=[isolated],type_ignores=[]))
    stream=PreviewStream()
    scope={'stream_manager':stream,'cv2':cv2,'Response':Response,'jsonify':jsonify,'logger':logging.getLogger('pv-preview-test')}
    exec(compile(module,str(SRC/'app.py'),'exec'),scope)
    if True:
        response=scope['api_stream_preview']()
        assert response.status_code==204 and not response.data, response
        # This must use the detection/annotation image, not the empty-camera
        # calibration placeholder, and must bound JPEG dimensions.
        sample=np.zeros((960,1280,3),dtype=np.uint8)
        sample[:,:]=(28,160,70)
        cv2.rectangle(sample,(200,150),(1000,600),(255,255,255),3)
        stream.frame=sample
        response=scope['api_stream_preview']()
        assert response.status_code==200
        assert response.mimetype=='image/jpeg'
        assert 'no-store' in response.headers.get('Cache-Control','')
        decoded=cv2.imdecode(np.frombuffer(response.data,dtype=np.uint8),cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.shape[:2]==(720,960),decoded.shape
        assert decoded[30,30,1]>decoded[30,30,0], decoded[30,30]
    source=(SRC/'templates/live.html').read_text()
    assert 'config.rtsp.url' not in source, 'RTSP credentials must not appear in Live markup'
    assert 'stream_status.url' in source
    assert 'const isIngressView' in source and '/api/stream/preview' in source
    assert 'fpsCounter.frames++' not in source, 'MJPEG onload is not a per-frame counter'
    assert 'function checkStreamStatus()' not in source, 'Live page must not override base status heartbeat'
    assert 'function startStream()' not in source and 'function stopStream()' not in source, 'Live page must not override base controls'
    assert 'function checkLiveStreamStatus()' in source
    print('  OK: 204 without frame, correct JPEG, resized preview, no-store, privacy and FPS source')

if __name__=='__main__':
    check_preview_endpoint()
