#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Regression: saving recognition defaults must fail closed on a full disk."""
import ast
import copy
import json
import logging
import sqlite3
import threading
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class Config:
    DEFAULT_CONFIG={
        'detection':{'threshold': .50}, 'ocr':{'min_confidence':.25},
        'people':{'enabled':False}, 'traffic':{'enabled':True},
        'models':{'license_plate_detector':'lp.pt', 'vehicle_detector':'vehicle.pt',
                  'person_detector':'people.pt'},
        'rtsp':{'url':'rtsp://camera', 'enabled':False, 'analysis_area':{'crop_before_detection':False}},
        'alerts':{'ha_mobile_enabled':False},
    }
    def __init__(self):
        self.lock=threading.RLock()
        self.config=copy.deepcopy(self.DEFAULT_CONFIG)
        self.config['detection']['threshold']=.99
        self.config['rtsp']['url']='rtsp://my-private-camera'
        self.saved=copy.deepcopy(self.config)
        self.fail=False
    def save_config(self):
        if self.fail:
            self.config=copy.deepcopy(self.saved)
            return False
        self.saved=copy.deepcopy(self.config)
        return True
    def _normalize_analysis_area(self, value):return value
    def get(self, *keys):
        value=self.config
        for k in keys:value=value.get(k, {}) if isinstance(value,dict) else None
        return value


def run():
    manager=Config()
    source=ROOT/'platevision/src/app.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    targets={'_config_defaults_copy','_apply_default_scope','_persist_config_update','api_config_defaults'}
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in targets:
            node=copy.deepcopy(node);node.decorator_list=[];nodes.append(node)
    assert len(nodes)==4
    class Request:
        data={}
        def get_json(self,silent=True):return self.data
    req=Request()
    env={'config_manager':manager,'request':req,'json':json,'sqlite3':sqlite3,
         'jsonify':lambda v:v,'logger':logging.getLogger('default-0138'),
         '_ha_settings_snapshot':lambda:{},
         '_alert_delivery_active':lambda: True,
         '_json_clone':copy.deepcopy,
         '_queue_model_reload':lambda:None,
         '_invalidate_model_cache':lambda:None,
         '_public_config':lambda:manager.config,
         '_scan_model_files':lambda:[]}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                 str(source),'exec'),env)
    before=copy.deepcopy(manager.config)
    manager.fail=True
    req.data={'scope':'recognition','reload_models':False}
    response,status=env['api_config_defaults']()
    assert status==503 and response['success'] is False,(response,status)
    assert manager.config==before, 'Uncommitted defaults overwrote the last good config'
    # Global defaults must report the same persistence failure.
    req.data={'scope':'global','reload_models':False}
    response,status=env['api_config_defaults']()
    assert status==503 and not response['success']
    assert manager.config==before
    manager.fail=False
    req.data={'scope':'recognition','reload_models':False}
    result=env['api_config_defaults']()
    assert result['success'] is True and manager.config['detection']['threshold']==.50
    assert manager.config['rtsp']['url']=='rtsp://my-private-camera'
    print('PlateVision 0.13.8 config defaults: full-disk rollback, HTTP 503, camera preserved: OK')

if __name__=='__main__':run()
