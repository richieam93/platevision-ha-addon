#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Direct MJPEG should be CPU bounded while the camera is offline/online."""
import ast
import copy
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'platevision/src/app.py'


class Response:
    def __init__(self, iterable, *, mimetype=None):
        self.response = iterable
        self.mimetype = mimetype
        self.headers = {}


class FakeStream:
    def __init__(self):
        self.frame = None

    def get_current_frame(self):
        return None if self.frame is None else self.frame.copy()

    get_raw_frame = get_current_frame


def test_feed():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    method = copy.deepcopy(next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                                and node.name == 'stream_feed'))
    method.decorator_list = []
    stream = FakeStream()
    sleeps = []
    cv_calls = []
    def encode(*args, **kwargs):
        cv_calls.append(1)
        return cv2.imencode(*args, **kwargs)

    ns = {'request': SimpleNamespace(args={}), 'stream_manager': stream,
          'Response': Response, 'np': np, 'cv2': SimpleNamespace(
              imencode=encode, zeros=np.zeros, putText=cv2.putText,
              IMWRITE_JPEG_QUALITY=cv2.IMWRITE_JPEG_QUALITY,
              FONT_HERSHEY_SIMPLEX=cv2.FONT_HERSHEY_SIMPLEX,
              error=cv2.error),
          'time': SimpleNamespace(sleep=sleeps.append),
          'logger': logging.getLogger('pv-mjpeg-test')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(SOURCE), 'exec'), ns)
    response = ns['stream_feed']()
    assert response.mimetype == 'multipart/x-mixed-replace; boundary=frame'
    assert 'no-store' in response.headers['Cache-Control']
    assert response.headers['X-Accel-Buffering'] == 'no'
    images = response.response
    waiting_1 = next(images)
    waiting_2 = next(images)
    assert waiting_1 == waiting_2
    assert len(cv_calls) == 1, 'Offline placeholder encoded more than once'
    assert sleeps == [1.0], sleeps
    stream.frame = np.full((30, 50, 3), 100, dtype=np.uint8)
    frame_1 = next(images)
    frame_2 = next(images)
    assert frame_1 != waiting_1
    assert frame_1.startswith(b'--frame\r\nContent-Type: image/jpeg\r\n')
    assert sleeps[-2:] == [1.0, 0.1], sleeps
    assert len(cv_calls) == 3
    assert b'\xff\xd8' in frame_1, 'Real MJPEG should contain JPEG bytes'
    images.close()
    print('  OK MJPEG: offline placeholder reused at 1 fps; direct stream at 10 fps; no-store')

if __name__ == '__main__':
    test_feed()
