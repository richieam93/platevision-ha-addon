# SPDX-License-Identifier: AGPL-3.0-only
"""Opt-in, bounded Home Assistant notifications for durable PlateVision alerts.

Uses only the internal Supervisor -> Core proxy, never a user-supplied URL.
The token stays in the add-on environment and is never returned to the UI.
Phone push delivery is accepted by HA; the device may still be offline.
"""
import json
import os
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


CORE_API = 'http://supervisor/core/api/services/'
TARGET_PATTERN = re.compile(r'^mobile_app_[a-z0-9_]{1,75}$')


class HADeliveryError(Exception):
    def __init__(self, message, *, retryable=False):
        super().__init__(message)
        self.retryable = retryable


def valid_mobile_service(value):
    return isinstance(value, str) and bool(TARGET_PATTERN.fullmatch(value))


def channels_for_settings(settings):
    """Return only supported, explicitly enabled Home Assistant services."""
    channels = []
    if settings.get('ha_persistent_enabled') is True:
        channels.append(('persistent', ''))
    if settings.get('ha_mobile_enabled') is True:
        mobile = settings.get('ha_mobile_service', '')
        if valid_mobile_service(mobile):
            channels.append(('mobile', mobile))
    return tuple(channels)


def ha_settings_from_config(config):
    """Strict booleans: malformed legacy values never enable external sends."""
    return {
        'ha_persistent_enabled': config.get('ha_persistent_enabled') is True,
        'ha_mobile_enabled': config.get('ha_mobile_enabled') is True,
        'ha_mobile_service': str(config.get('ha_mobile_service') or '')[:90],
        'ha_include_plate': config.get('ha_include_plate') is True,
    }


def _single_line(value, limit):
    raw = ''.join(c if c.isprintable() and c not in '\r\n\t' else ' '
                  for c in str(value or ''))
    return raw.strip()[:limit].replace('<', '').replace('>', '')


class _NoRedirect(HTTPRedirectHandler):
    """Do not forward the Supervisor token to a redirected host."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HomeAssistantSender:
    """A narrow HTTP client with no configurable hosts or arbitrary services."""

    def __init__(self, *, token_getter=None, opener=None):
        self.token_getter = token_getter or (lambda: os.environ.get('SUPERVISOR_TOKEN', ''))
        # A proxy configured in the environment must not receive the token.
        self.opener = opener or build_opener(ProxyHandler({}), _NoRedirect())

    def available(self):
        return bool(self.token_getter())

    def _post(self, service_path, payload):
        token = self.token_getter()
        if not token:
            raise HADeliveryError('Home-Assistant-Token nicht verfügbar', retryable=True)
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        req = Request(CORE_API + service_path, data=data, method='POST', headers={
            'Authorization': 'Bearer ' + token,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        })
        try:
            # Time-bound request: capture threads never wait for HA.
            with self.opener.open(req, timeout=5) as response:
                if not 200 <= response.status < 300:
                    raise HADeliveryError('Home Assistant HTTP ' + str(response.status),
                                          retryable=response.status >= 500)
                return True
        except HTTPError as exc:
            # Avoid logging HTML bodies, credentials, headers, URLs or tokens.
            raise HADeliveryError('Home Assistant HTTP ' + str(exc.code),
                                  retryable=exc.code in (408, 429) or exc.code >= 500) from None
        except (URLError, TimeoutError, OSError):
            raise HADeliveryError('Home Assistant derzeit nicht erreichbar', retryable=True) from None

    def send_alert(self, row, *, include_plate=False):
        channel = row['channel']
        ident = int(row['alert_id'])
        if include_plate:
            plate = _single_line(row.get('plate_text') or row.get('normalized'), 60)
            label = _single_line(row.get('label'), 100)
            message = f'Watchlist-Treffer: {plate or "unbekannt"}'
            if label:
                message += '\nNotiz: ' + label
        else:
            message = 'Ein Watchlist-Treffer wurde erkannt. Details stehen in PlateVision.'
        title = 'PlateVision - Watchlist'
        if channel == 'persistent':
            return self._post('persistent_notification/create', {
                'title': title, 'message': message,
                'notification_id': f'platevision_watchlist_{ident}',
            })
        if channel == 'mobile' and valid_mobile_service(row.get('target')):
            return self._post('notify/' + row['target'], {
                'title': title, 'message': message,
                'data': {'tag': f'platevision-watchlist-{ident}'},
            })
        raise HADeliveryError('Ungültiges Benachrichtigungsziel', retryable=False)

    def send_test(self, *, channel, target=''):
        payload = 'PlateVision-Test: Home-Assistant-Benachrichtigungen sind erreichbar.'
        if channel == 'persistent':
            return self._post('persistent_notification/create', {
                'title': 'PlateVision - Verbindungstest', 'message': payload,
                'notification_id': 'platevision_connection_test',
            })
        if channel == 'mobile' and valid_mobile_service(target):
            return self._post('notify/' + target, {
                'title': 'PlateVision - Verbindungstest', 'message': payload,
                'data': {'tag': 'platevision-connection-test'},
            })
        raise HADeliveryError('Ungültiges Benachrichtigungsziel')


class HADeliveryWorker:
    """Periodic retry worker. SQLite leases prevent parallel add-on workers."""

    def __init__(self, store, get_settings, *, sender=None, logger=None):
        self.store = store
        self.get_settings = get_settings
        self.sender = sender or HomeAssistantSender()
        self.logger = logger
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        # Acquired by the HA settings endpoint as well: a settings change
        # cannot race an already claimed notification network request.
        self.delivery_lock = threading.RLock()

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name='platevision-ha-notify')
            self._thread.start()
        return self

    def wake(self):
        self._wake.set()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def deliver_once(self):
        """Perform zero or one bounded network attempt, useful for tests."""
        with self.delivery_lock:
            settings = self.get_settings()
            # The app may have disabled the entire watchlist while keeping
            # the channel credentials configured for a later opt-in. Never
            # deliver a queued message in that state, even after a bad import.
            if settings.get('alerts_enabled') is False:
                return False
            enabled = channels_for_settings(settings)
            if not enabled or not self.sender.available():
                return False
            row = self.store.claim_delivery()
            if row is None:
                return False
            claim = row['lease_generation']
            # Reject stale queue entries from a previously configured phone.
            if (row['channel'], row['target']) not in enabled:
                self.store.finish_delivery(row['id'], lease_generation=claim,
                                           success=False, error='Altes Benachrichtigungsziel',
                                           retryable=False)
                return True
            try:
                # A privacy opt-out also applies to ALREADY QUEUED alerts.
                # A later opt-in never upgrades old, anonymous queue entries.
                include_plate = (row.get('include_plate') == 1 and
                                 settings.get('ha_include_plate') is True)
                self.sender.send_alert(row, include_plate=include_plate)
            except HADeliveryError as exc:
                self.store.finish_delivery(row['id'], lease_generation=claim,
                                           success=False, retryable=exc.retryable, error=str(exc))
            except Exception:
                self.store.finish_delivery(row['id'], lease_generation=claim,
                                           success=False, retryable=True,
                                           error='Interner Zustellfehler')
                if self.logger:
                    self.logger.warning('HA notification delivery failed (unexpected error)')
            else:
                self.store.finish_delivery(row['id'], lease_generation=claim, success=True)
            return True

    def _loop(self):
        while not self._stop.is_set():
            try:
                for _ in range(8):
                    if self._stop.is_set() or not self.deliver_once():
                        break
            except Exception:
                if self.logger:
                    self.logger.warning('HA notification worker deferred due to storage error')
            self._wake.wait(timeout=5)
            self._wake.clear()
