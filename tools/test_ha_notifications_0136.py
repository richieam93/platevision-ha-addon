#!/usr/bin/env python3
"""0.13.6 HA integration: opt-in, durable outbox, retry, privacy and no-SSRF.

Runs offline. HTTP calls are recorded in an in-memory fake; no Supervisor
access, camera or ML runtime is needed.
"""
import ast
import json
import logging
import os
import sqlite3
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'platevision/src'))
from watchlist_alerts import WatchlistAlertStore
from ha_notifications import (HADeliveryError, HADeliveryWorker,
                              HomeAssistantSender, channels_for_settings,
                              ha_settings_from_config, valid_mobile_service)


def event(eid, plate='BE 123', notify=True):
    return {'id': eid, 'plate_text': plate,
            'plate_text_normalized': plate.replace(' ','').upper(),
            'source': 'rtsp', 'confidence': .93,
            'watchlist_match': {'id': 'watch1', 'notify': notify,
                                'category':'important','label':'Privater Hinweis'}}


def settings(*, mobile=False, persistent=False, target='', include=False):
    return {'ha_mobile_enabled': mobile, 'ha_persistent_enabled': persistent,
            'ha_mobile_service': target, 'ha_include_plate': include}


class FakeResponse:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *args): return None


class FakeOpener:
    def __init__(self, issue=None):
        self.requests=[]
        self.issue=issue
    def open(self, req, timeout=None):
        self.requests.append((req,timeout))
        if self.issue:
            raise self.issue
        return FakeResponse()


def test_queued_after_new_alert_only():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'alerts.sqlite3'
        store=WatchlistAlertStore(path)
        channels=(('persistent',''), ('mobile','mobile_app_phone'))
        assert store.record(event('1'),now=10000,ha_channels=channels)['created']
        assert not store.record(event('2'),now=10001,ha_channels=channels)['created']
        assert store.delivery_status()['pending']==2
        assert store.status()['total']==1
        assert store.record(event('off',notify=False),now=10002,ha_channels=channels) is None
        assert store.delivery_status()['pending']==2
        store2=WatchlistAlertStore(path)
        assert store2.delivery_status()['pending']==2
        with sqlite3.connect(path) as c:
            assert c.execute('SELECT COUNT(*) FROM ha_delivery_outbox').fetchone()[0]==2
            assert c.execute('SELECT COUNT(*) FROM seen_events').fetchone()[0]==2
        first=store2.claim_delivery(now=10010)
        assert first and first['attempts']==1 and first['include_plate']==0
        assert store2.claim_delivery(now=10010) is not None
        assert store2.claim_delivery(now=10010) is None
        assert store2.finish_delivery(first['id'],lease_generation=first['lease_generation'],
                                      success=True,now=10020)
        assert store2.delivery_status()['accepted']==1
        print('  OK notification outbox is atomic and survives restarts; cooldown enforced')


def test_retry_and_leasing():
    with tempfile.TemporaryDirectory() as tmp:
        store=WatchlistAlertStore(Path(tmp)/'a.sqlite3')
        store.record(event('1'),now=900,ha_channels=(('persistent',''),))
        first=store.claim_delivery(now=901, lease_seconds=30)
        assert first['attempts']==1
        assert store.claim_delivery(now=902) is None
        second=store.claim_delivery(now=932)
        assert second['attempts']==2, 'stale lease must recover'
        assert not store.finish_delivery(first['id'],lease_generation=first['lease_generation'],
                                         success=True,now=933), 'stale worker must not finish newer lease'
        assert store.finish_delivery(second['id'],lease_generation=second['lease_generation'],
                                     success=False,retryable=True,
                                     error='network error',now=933)
        assert store.delivery_status()['pending']==1
        assert store.claim_delivery(now=952) is None
        third=store.claim_delivery(now=973)
        assert third and third['attempts']==3
        assert store.finish_delivery(third['id'],lease_generation=third['lease_generation'],
                                     success=False,retryable=False,
                                     error='Home Assistant HTTP 404',now=974)
        assert store.delivery_status()['failed']==1
        assert store.retry_failed_deliveries(now=975)==1
        assert store.claim_delivery(now=975)['attempts']==1
        print('  OK stale worker leases, backoff, permanent errors, user-controlled retries')


def test_transport_private_and_fixed_destination():
    opener=FakeOpener()
    token='very-secret-supervisor-token'
    sender=HomeAssistantSender(token_getter=lambda:token,opener=opener)
    row={'alert_id': 42,'channel':'mobile','target':'mobile_app_family',
         'plate_text':'BE 123','label':'Private account'}
    sender.send_alert(row,include_plate=False)
    request,timeout=opener.requests[-1]
    assert request.full_url=='http://supervisor/core/api/services/notify/mobile_app_family'
    body=json.loads(request.data)
    assert 'BE 123' not in str(body) and 'Private account' not in str(body)
    assert body['data']['tag']=='platevision-watchlist-42'
    assert timeout<=5
    assert request.get_header('Authorization')=='Bearer '+token
    sender.send_alert({**row,'channel':'persistent'},include_plate=True)
    request,_=opener.requests[-1]
    assert request.full_url=='http://supervisor/core/api/services/persistent_notification/create'
    body=json.loads(request.data)
    assert body['notification_id']=='platevision_watchlist_42' and 'BE 123' in body['message']
    sender.send_test(channel='mobile',target='mobile_app_family')
    assert 'BE 123' not in str(json.loads(opener.requests[-1][0].data))
    assert not valid_mobile_service('mobile_app_x/../../config')
    assert not valid_mobile_service('notify.admin')
    assert not valid_mobile_service('http://evil')
    try:
        sender.send_alert({**row,'target':'mobile_app_a/../../'},include_plate=True)
        raise AssertionError('malicious service accepted')
    except HADeliveryError as exc:
        assert not exc.retryable
    assert len(opener.requests)==3
    offline=HomeAssistantSender(token_getter=lambda:'',opener=opener)
    try:
        offline.send_test(channel='persistent')
        raise AssertionError('missing token did not fail')
    except HADeliveryError as exc:
        assert exc.retryable
    assert len(opener.requests)==3
    print('  OK fixed Supervisor URL, validated service and private notification text')


def test_worker_recovery_and_change_of_target():
    with tempfile.TemporaryDirectory() as tmp:
        store=WatchlistAlertStore(Path(tmp)/'a.sqlite3')
        store.record(event('1'),now=10000,ha_channels=(('mobile','mobile_app_old'),))
        conf=settings(mobile=True,target='mobile_app_new',include=True)
        opener=FakeOpener()
        sender=HomeAssistantSender(token_getter=lambda:'token',opener=opener)
        worker=HADeliveryWorker(store,lambda:conf,sender=sender)
        assert worker.deliver_once()
        assert store.delivery_status()['failed']==1
        assert len(opener.requests)==0, 'old device must never get stale notification'
        assert store.cancel_pending_deliveries()==1
        store.record(event('2'),now=12000,ha_channels=(('mobile','mobile_app_new'),),ha_include_plate=True)
        assert worker.deliver_once()
        assert store.delivery_status()['accepted']==1
        assert 'BE 123' in str(json.loads(opener.requests[0][0].data))
        # Disable switches pauses sending without changing the local inbox.
        conf['ha_mobile_enabled']=False
        store.record(event('3'),now=13000,ha_channels=(('persistent',''),))
        assert not worker.deliver_once()
        assert store.delivery_status()['pending']==1
        # Persisted pending work survives reinstantiation.
        restored=WatchlistAlertStore(store.path)
        assert restored.delivery_status()['pending']==1
        print('  OK target changes, opt-out pauses, durable work queue, accepted status')


def test_retry_worker_transient_and_http():
    with tempfile.TemporaryDirectory() as tmp:
        store=WatchlistAlertStore(Path(tmp)/'a.sqlite3')
        store.record(event('1'),now=time.time()-2,ha_channels=(('persistent',''),))
        problem=FakeOpener(issue=URLError('network down'))
        w=HADeliveryWorker(store,lambda:settings(persistent=True),
                           sender=HomeAssistantSender(token_getter=lambda:'token',opener=problem))
        assert w.deliver_once() and store.delivery_status()['pending']==1
        # No runaway immediate retry
        assert not w.deliver_once()
        assert store.delivery_status()['failed']==0
        # invalid HA endpoint is permanent failure (no endlessly recurring retries)
        store.retry_failed_deliveries()
        with sqlite3.connect(store.path) as conn:
            conn.execute('UPDATE ha_delivery_outbox SET next_attempt_at=0')
        problem.issue=HTTPError('internal',404,'service missing',None,None)
        assert w.deliver_once() and store.delivery_status()['failed']==1
        assert '404' in store.delivery_status()['last_error']
        print('  OK network retry with backoff, 404 service failures stay visible')


def test_concurrent_claims_and_backups():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'a.sqlite3'
        first=WatchlistAlertStore(path)
        second=WatchlistAlertStore(path)
        channels=(('persistent',''),)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results=list(pool.map(lambda i: first.record(event(str(i)),now=10000+i,
                                                     ha_channels=channels),range(30)))
        assert sum(r['created'] for r in results)==1
        assert first.delivery_status()['pending']==1
        with ThreadPoolExecutor(max_workers=2) as pool:
            result=list(pool.map(lambda s: s.claim_delivery(now=11000),[first,second]))
        assert sum(bool(r) for r in result)==1
        dbcopy=Path(tmp)/'backup.sqlite3'
        first.backup_to(dbcopy)
        with sqlite3.connect(dbcopy) as conn:
            assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            assert conn.execute('SELECT COUNT(*) FROM ha_delivery_outbox').fetchone()[0]==1
        print('  OK parallel processes claim once; online backup includes pending notifications')


def test_release_wiring():
    app=(ROOT/'platevision/src/app.py').read_text()
    config=(ROOT/'platevision/config.yaml').read_text()
    tpl=(ROOT/'platevision/src/templates/alerts.html').read_text()
    assert 'homeassistant_api: true' in config
    for url in ('/api/alerts/homeassistant','/api/alerts/homeassistant/test',
                '/api/alerts/homeassistant/retry'):
        assert url in app
    assert 'ha_channels=channels' in app
    assert 'pvHAMobileService' in tpl and 'pvHAIncludePlate' in tpl
    assert 'SUPERVISOR_TOKEN' not in tpl
    assert 'innerHTML' not in tpl
    assert "'ha_mobile_enabled': False" in app
    print('  OK integration, privacy defaults, HA manifest and XSS-safe display')


if __name__=='__main__':
    for fn in (test_queued_after_new_alert_only, test_retry_and_leasing,
               test_transport_private_and_fixed_destination,
               test_worker_recovery_and_change_of_target,
               test_retry_worker_transient_and_http,
               test_concurrent_claims_and_backups, test_release_wiring):
        fn()
    print('PlateVision HA notifications 0.13.6: OK')
