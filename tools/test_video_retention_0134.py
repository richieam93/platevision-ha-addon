#!/usr/bin/env python3
"""Verify video retention cannot orphan a completed job after disk failure."""
import ast
import copy
import logging
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'platevision/src/app.py'

def check():
    tree=ast.parse(APP.read_text())
    fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_cleanup_old_video_jobs'))
    fn.decorator_list=[]
    with tempfile.TemporaryDirectory(prefix='pv-video-retention-') as td:
        base=Path(td)
        older=(datetime.now()-timedelta(days=20)).isoformat()
        fresh=(datetime.now()-timedelta(minutes=1)).isoformat()
        old_file=base/'completed.mp4';old_file.write_bytes(b'old-completed-output')
        new_file=base/'fresh.mp4';new_file.write_bytes(b'new-output')
        jobs={'expired':{'id':'expired','status':'completed','created_at':older,'output_path':str(old_file)},
              'new':{'id':'new','status':'completed','created_at':fresh,'output_path':str(new_file)}}
        cancel={'expired':object()}
        calls=[]
        should_persist=[False]
        def persist():
            calls.append(('persist',old_file.exists()))
            return should_persist[0]
        def orphan():
            calls.append(('orphan',old_file.exists()))
        env={'_video_retention_hours':lambda:168,'datetime':datetime,'timedelta':timedelta,
             'video_processing_jobs':jobs,'video_job_cancel_events':cancel,
             'video_jobs_lock':threading.RLock(), '_persist_video_jobs':persist,
             '_cleanup_orphan_video_files':orphan,'Path':Path,
             'logger':logging.getLogger('retention-test')}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(APP),'exec'),env)
        cleanup=env['_cleanup_old_video_jobs']
        assert cleanup()==0
        assert old_file.exists() and 'expired' in jobs and 'expired' in cancel
        assert all(operation!='orphan' for operation,_ in calls)
        should_persist[0]=True
        assert cleanup()==1
        assert 'expired' not in jobs and not old_file.exists()
        assert 'new' in jobs and new_file.exists()
        assert calls[-2:]==[('persist',True),('orphan',False)],calls
    print('  OK: video expiration rolls back a failed registry write and preserves downloads')

if __name__=='__main__':
    check()
    print('PASS PlateVision 0.13.4 video retention integrity')
