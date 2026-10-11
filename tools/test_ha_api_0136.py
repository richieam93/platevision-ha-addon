#!/usr/bin/env python3
"""Isolated actual Flask handler logic with stubbed request/config/HTTP."""
import ast
import copy
import logging
import os
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'platevision/src'))
from watchlist_alerts import WatchlistAlertStore, notification_enabled
from ha_notifications import (ha_settings_from_config,valid_mobile_service,
                              HomeAssistantSender,HADeliveryError,channels_for_settings)

class Config:
    def __init__(self):
        self.lock=threading.RLock()
        self.config={'alerts':{}}
        self.snapshot=copy.deepcopy(self.config)
        self.fail=False
    def get(self, *keys):
        value = self.config
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        return value
    def save_config(self):
        if self.fail:
            self.config=copy.deepcopy(self.snapshot)
            return False
        self.snapshot=copy.deepcopy(self.config)
        return True

class Request:
    def __init__(self):
        self.method='GET';self.payload=None;self.content_length=0
    def get_json(self, silent=True):return self.payload

class Sender:
    calls=[]
    connected=True
    def available(self):return self.connected
    def send_test(self,channel,target=''):
        self.calls.append((channel,target))
        return True


def invoke(response):
    if isinstance(response,tuple):return response[0],response[1]
    return response,200


def run():
    with tempfile.TemporaryDirectory() as tmp:
        config=Config();request=Request();store=WatchlistAlertStore(Path(tmp)/'a.sqlite3')
        class DummyApp:
            def route(self,*args,**kwargs):return lambda fn:fn
        env={'app':DummyApp(),'request':request,'jsonify':lambda data:data,
             'config_manager':config,'watchlist_alert_center':store,
             'ha_settings_from_config':ha_settings_from_config,
             'valid_mobile_service':valid_mobile_service,
             'channels_for_settings':channels_for_settings, 'notification_enabled':notification_enabled,
             'threading':threading,'sqlite3':sqlite3,'os':os,'time':__import__('time'),
             'HADeliveryError':HADeliveryError,'HomeAssistantSender':Sender,
             'logger':logging.getLogger('api_ha_0136'),
             '_alert_store_or_error':lambda: (store,None),
             'ha_delivery_worker':SimpleNamespace(wake=lambda:None)}
        file=ROOT/'platevision/src/app.py'
        tree=ast.parse(file.read_text())
        nodes=[]
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in (
                '_ha_settings_snapshot','_alert_delivery_active','api_alerts_homeassistant',
                'api_alerts_homeassistant_test','api_alerts_homeassistant_retry'):
                nodes.append(copy.deepcopy(node))
            if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name)
                and n.id=='_ha_test_lock' for n in node.targets):
                nodes.append(copy.deepcopy(node))
            if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id=='_ha_last_test_at':
                nodes.append(copy.deepcopy(node))
            if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name)
                and n.id=='_ha_last_test_at' for n in node.targets):
                nodes.append(copy.deepcopy(node))
        assert len([n for n in nodes if isinstance(n,ast.FunctionDef)])==5
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(file),'exec'),env)
        query=env['api_alerts_homeassistant'];test=env['api_alerts_homeassistant_test']
        retry=env['api_alerts_homeassistant_retry']
        def post(payload):
            request.method='POST';request.payload=payload
            request.content_length=len(str(payload).encode())
            return invoke(query())
        request.method='GET'
        got,code=invoke(query())
        assert code==200 and not got['settings']['ha_mobile_enabled']
        payload={'ha_persistent_enabled':False,'ha_mobile_enabled':True,
                 'ha_mobile_service':'notify.mobile_app_alice', 'ha_include_plate':False}
        assert post(payload)[0]['settings']['ha_mobile_service']=='mobile_app_alice'
        assert config.snapshot['alerts']['ha_mobile_enabled'] is True
        invalid=dict(payload,ha_mobile_service='notify.mobile_app_x/../../admin')
        assert post(invalid)[1]==400
        invalid=dict(payload,ha_persistent_enabled='true')
        assert post(invalid)[1]==400
        assert post({'ha_persistent_enabled':True})[1]==400
        request.content_length=8000;assert invoke(query())[1]==413
        config.fail=True
        invalid=dict(payload,ha_mobile_enabled=False)
        assert post(invalid)[1]==503
        assert config.config['alerts']['ha_mobile_enabled'] is True
        config.fail=False
        request.method='GET'
        os.environ['SUPERVISOR_TOKEN']='test-secret-not-a-real-token'
        try:
            got,code=invoke(query())
            assert got['token_available'] is True
            request.method='POST'
            assert invoke(test())[0]['success'] is True
            assert Sender.calls[-1]==('mobile','mobile_app_alice')
            assert invoke(test())[1]==429, 'rate limit missing'
            assert invoke(retry())[0]['queued']==0
        finally:
            del os.environ['SUPERVISOR_TOKEN']
        print('PlateVision HA API 0.13.6: GET, POST, service validation, rate limit, rollback, status, retry: OK')

if __name__=='__main__':run()
