# SPDX-License-Identifier: AGPL-3.0-only
"""Durable, bounded and privacy-conscious local watchlist alert inbox.

The full vehicle/plate images remain in the existing history database. This
SQLite database stores only a reference and minimal matching metadata. Each
saved history event may be recorded at most once, even after a restart.
"""

from contextlib import closing
from pathlib import Path
import os
import sqlite3
import threading
import time


FALSE_VALUES = {'false', '0', 'no', 'off', 'nein', 'non', 'falso'}
TRUE_VALUES = {'true', '1', 'yes', 'on', 'ja', 'oui', 'vero'}


def notification_enabled(value):
    """Accept real JSON booleans and explicitly known boolean representations.

    Unknown values fail closed to avoid accidentally notifying for a malformed
    imported watchlist row. Import/update paths validate strictly separately.
    """
    if value is None:
        return True  # legacy entries without an explicit notification field
    if isinstance(value, bool):
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in TRUE_VALUES
    return False


def validate_notify(value):
    if isinstance(value, bool):
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in TRUE_VALUES:
            return True
        if token in FALSE_VALUES:
            return False
    raise ValueError('Benachrichtigung muss true/false oder ja/nein sein')


class WatchlistAlertStore:
    """Thread-safe SQLite inbox with cross-process deduplication.

    The alert is created only after the recognition has committed to history.
    A repeated event ID is idempotent; new events in the same recent visit are
    counted under the original alert rather than generating a toast storm.
    """

    def __init__(self, path='data/alerts/watchlist.sqlite3', *, max_alerts=5000):
        self.path = Path(path)
        self.max_alerts = max(100, min(int(max_alerts), 20000))
        self.lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, closing(self._connect()) as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    first_event_id TEXT NOT NULL,
                    latest_event_id TEXT NOT NULL,
                    watchlist_id TEXT NOT NULL,
                    plate_text TEXT NOT NULL,
                    normalized TEXT NOT NULL,
                    label TEXT NOT NULL,
                    category TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_group TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    match_type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    occurrences INTEGER NOT NULL DEFAULT 1,
                    is_read INTEGER NOT NULL DEFAULT 0,
                    read_at REAL
                );
                CREATE TABLE IF NOT EXISTS seen_events (
                    event_id TEXT PRIMARY KEY,
                    alert_id INTEGER NOT NULL REFERENCES alerts(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS ha_delivery_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id INTEGER NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
                    channel TEXT NOT NULL CHECK(channel IN ('persistent','mobile')),
                    target TEXT NOT NULL DEFAULT '',
                    include_plate INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL,
                    lease_until REAL,
                    lease_generation INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    accepted_at REAL,
                    created_at REAL NOT NULL,
                    UNIQUE(alert_id, channel, target)
                );
                CREATE INDEX IF NOT EXISTS ha_delivery_due
                    ON ha_delivery_outbox(state, next_attempt_at, id);
                CREATE INDEX IF NOT EXISTS alerts_recent ON alerts(last_seen_at DESC, id DESC);
                CREATE INDEX IF NOT EXISTS alerts_unread ON alerts(is_read, last_seen_at DESC);
                CREATE INDEX IF NOT EXISTS alerts_visit
                    ON alerts(watchlist_id, normalized, source_group, last_seen_at DESC);
            ''')
            # Existing 0.13.6 alarm databases need only this additive column.
            # It fences off a worker whose lease expired while it was paused.
            columns = {row['name'] for row in connection.execute(
                'PRAGMA table_info(ha_delivery_outbox)')}
            if 'lease_generation' not in columns:
                try:
                    connection.execute('''ALTER TABLE ha_delivery_outbox
                        ADD COLUMN lease_generation INTEGER NOT NULL DEFAULT 0''')
                except sqlite3.OperationalError as exc:
                    # Two add-on processes may perform the same upgrade.
                    if 'duplicate column name' not in str(exc).lower():
                        raise
            connection.commit()
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=5.0)
        try:
            connection.execute('PRAGMA foreign_keys=ON')
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA busy_timeout=5000')
            connection.execute('PRAGMA synchronous=FULL')
            connection.row_factory = sqlite3.Row
        except Exception:
            connection.close()
            raise
        return connection

    @staticmethod
    def _source_group(entry):
        source = str(entry.get('source') or 'unknown')[:40]
        if source == 'video_upload':
            job = str(entry.get('video_job_id') or entry.get('upload_id') or '')[:80]
            return 'video:' + (job or 'legacy')
        if source == 'image_upload':
            upload = str(entry.get('upload_id') or '')[:80]
            return 'image:' + (upload or 'legacy')
        return source

    def record(self, entry, *, minimum_confidence=0.65, cooldown_seconds=180, now=None,
               ha_channels=(), ha_include_plate=False):
        """Record a matched saved event or return None if notifications are off.

        Any SQLite exception is allowed to bubble up to the caller, who reports
        it without rolling back already committed history data.
        """
        if not isinstance(entry, dict):
            return None
        match = entry.get('watchlist_match')
        if not isinstance(match, dict) or not notification_enabled(match.get('notify', True)):
            return None
        try:
            confidence = float(entry.get('confidence') or 0)
            minimum = float(minimum_confidence)
        except (TypeError, ValueError, OverflowError):
            return None
        if (not 0 <= confidence <= 1 or not 0 <= minimum <= 1):
            return None
        if confidence < minimum:
            return None
        event_id = str(entry.get('id') or '')[:128]
        watchlist_id = str(match.get('id') or '')[:128]
        normalized = str(entry.get('plate_text_normalized') or '').upper().strip()[:60]
        if not event_id or not watchlist_id or not normalized:
            return None
        try:
            cooldown = max(0, min(int(cooldown_seconds), 3600))
        except (ValueError, TypeError, OverflowError):
            cooldown = 180
        now = time.time() if now is None else float(now)
        if not 0 <= now < 100000000000:
            raise ValueError('Invalid alert timestamp')
        group = self._source_group(entry)
        fuzzy = bool(match.get('fuzzy_match'))
        with self.lock, closing(self._connect()) as conn:
            # A transaction serializes overlapping writes across threads and
            # across independent PlateVision processes.
            conn.execute('BEGIN IMMEDIATE')
            seen = conn.execute('SELECT alert_id FROM seen_events WHERE event_id=?',
                                (event_id,)).fetchone()
            if seen:
                conn.commit()
                return {'id': seen['alert_id'], 'created': False, 'duplicate_event': True}
            last = conn.execute('''SELECT id, last_seen_at FROM alerts
                WHERE watchlist_id=? AND normalized=? AND source_group=?
                ORDER BY last_seen_at DESC, id DESC LIMIT 1''',
                (watchlist_id, normalized, group)).fetchone()
            if last and abs(now - last['last_seen_at']) < cooldown:
                alert_id = last['id']
                conn.execute('''UPDATE alerts SET last_seen_at=MAX(last_seen_at,?),latest_event_id=?,
                    occurrences=occurrences+1,confidence=MAX(confidence,?) WHERE id=?''',
                    (now, event_id, confidence, alert_id))
                created = False
            else:
                cur = conn.execute('''INSERT INTO alerts
                    (first_event_id,latest_event_id,watchlist_id,plate_text,normalized,
                    label,category,source,source_group,confidence,match_type,
                    created_at,last_seen_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (event_id, event_id, watchlist_id,
                     str(entry.get('plate_text') or '')[:60], normalized,
                     str(match.get('label') or '')[:160],
                     str(match.get('category') or 'known')[:40],
                     str(entry.get('source') or 'unknown')[:40], group,
                     confidence, 'similar' if fuzzy else 'exact', now, now))
                alert_id = cur.lastrowid
                created = True
            conn.execute('INSERT INTO seen_events(event_id,alert_id) VALUES (?,?)',
                         (event_id, alert_id))
            # Queue HA delivery in the SAME transaction as the new local alarm.
            # A repeated OCR frame must never trigger a second push.
            if created:
                for channel, target in ha_channels:
                    if channel not in ('persistent', 'mobile'):
                        raise ValueError('Unsupported notification channel')
                    target = str(target or '')
                    if len(target) > 90:
                        raise ValueError('Notification target too long')
                    conn.execute("""INSERT OR IGNORE INTO ha_delivery_outbox
                        (alert_id,channel,target,include_plate,next_attempt_at,created_at)
                        VALUES (?,?,?,?,?,?)""",
                        (alert_id, channel, target, int(ha_include_plate is True), now, now))
            # Bound both the notifications and the event-ID dedupe table.
            conn.execute('''DELETE FROM alerts WHERE id IN
                (SELECT id FROM alerts ORDER BY last_seen_at DESC,id DESC
                 LIMIT -1 OFFSET ?)''', (self.max_alerts,))
            conn.commit()
            return {'id': alert_id, 'created': created, 'duplicate_event': False}

    def status(self):
        with self.lock, closing(self._connect()) as conn:
            total, unread = conn.execute('SELECT COUNT(*),COALESCE(SUM(1-is_read),0) FROM alerts').fetchone()
            return {'total': total, 'unread': unread, 'backend': 'sqlite'}

    def page(self, *, limit=30, offset=0, unread_only=False):
        limit = max(1, min(int(limit), 100))
        offset = max(0, min(int(offset), 1000000))
        where = ' WHERE is_read=0' if unread_only else ''
        with self.lock, closing(self._connect()) as conn:
            total = conn.execute('SELECT COUNT(*) FROM alerts' + where).fetchone()[0]
            rows = conn.execute('''SELECT id,first_event_id,latest_event_id,
                plate_text,normalized,label,category,source,confidence,
                match_type,created_at,last_seen_at,occurrences,is_read,read_at
                FROM alerts''' + where + ' ORDER BY last_seen_at DESC,id DESC LIMIT ? OFFSET ?',
                (limit,offset)).fetchall()
        return {'items': [dict(row) for row in rows], 'total': total,
                'limit': limit, 'offset': offset, 'unread_only': bool(unread_only)}

    def acknowledge(self, alert_id):
        with self.lock, closing(self._connect()) as conn:
            conn.execute('BEGIN IMMEDIATE')
            updated = conn.execute('UPDATE alerts SET is_read=1,read_at=? WHERE id=?',
                                   (time.time(), int(alert_id))).rowcount
            conn.commit()
            return bool(updated)

    def acknowledge_all(self):
        with self.lock, closing(self._connect()) as conn:
            conn.execute('BEGIN IMMEDIATE')
            changed = conn.execute('UPDATE alerts SET is_read=1,read_at=? WHERE is_read=0',
                                   (time.time(),)).rowcount
            conn.commit()
            return changed

    def delete(self, alert_id):
        """Delete an inbox entry, never its source recognition or image."""
        with self.lock, closing(self._connect()) as conn:
            conn.execute('BEGIN IMMEDIATE')
            changed = conn.execute('DELETE FROM alerts WHERE id=?',(int(alert_id),)).rowcount
            conn.commit()
            return bool(changed)

    def delivery_status(self):
        """Compact queue-health information, never user or recipient data.

        Distinguish slow backlogs, failed tasks and expired leases.  In
        particular, an expired lease is not a successful delivery: another
        worker may need to reclaim it after a restart.
        """
        with self.lock, closing(self._connect()) as conn:
            rows = conn.execute('''SELECT state,COUNT(*) AS number
                FROM ha_delivery_outbox GROUP BY state''').fetchall()
            stats = {'pending': 0, 'sending': 0, 'accepted': 0, 'failed': 0}
            for row in rows:
                stats[row['state']] = row['number']
            error = conn.execute('''SELECT last_error FROM ha_delivery_outbox
                WHERE state='failed' ORDER BY id DESC LIMIT 1''').fetchone()
            stats['last_error'] = error['last_error'] if error else ''
            oldest = conn.execute('''SELECT MIN(created_at) FROM ha_delivery_outbox
                WHERE state IN ('pending','sending')''').fetchone()[0]
            accepted = conn.execute('''SELECT MAX(accepted_at) FROM ha_delivery_outbox
                WHERE state='accepted' ''').fetchone()[0]
            now = time.time()
            stats['oldest_waiting_seconds'] = (max(0, int(now - oldest))
                                               if oldest is not None else None)
            stats['last_accepted_at'] = accepted
            stats['expired_leases'] = conn.execute('''SELECT COUNT(*)
                FROM ha_delivery_outbox WHERE state='sending'
                AND lease_until <= ?''', (now,)).fetchone()[0]
            due_row = conn.execute('''SELECT COUNT(*) AS due, MIN(created_at) AS oldest
                FROM ha_delivery_outbox
                WHERE (state='pending' AND next_attempt_at<=?)
                   OR (state='sending' AND lease_until<=?)''', (now, now)).fetchone()
            stats['due_now'] = due_row['due']
            stats['oldest_due_seconds'] = (max(0, int(now - due_row['oldest']))
                                           if due_row['oldest'] is not None else None)
            stats['next_retry_at'] = conn.execute('''SELECT MIN(next_attempt_at)
                FROM ha_delivery_outbox
                WHERE state='pending' AND next_attempt_at>?''', (now,)).fetchone()[0]
            return stats

    def recent_deliveries(self, *, limit=20):
        """Operational audit only: no plate, note, token or phone identifier."""
        limit = max(1, min(int(limit), 50))
        with self.lock, closing(self._connect()) as conn:
            rows = conn.execute('''SELECT id, channel, state, attempts,
                next_attempt_at, accepted_at, created_at, last_error
                FROM ha_delivery_outbox ORDER BY id DESC LIMIT ?''', (limit,)).fetchall()
        return [dict(row) for row in rows]

    def claim_delivery(self, *, now=None, lease_seconds=30):
        """Atomically claim a due item; a stale lease can be recovered."""
        now = time.time() if now is None else float(now)
        with self.lock, closing(self._connect()) as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('''SELECT d.id, d.alert_id, d.channel,d.target,d.include_plate,
                d.attempts,d.lease_generation,a.plate_text,a.normalized,a.label,a.category,
                a.source,a.confidence,a.created_at,a.occurrences
                FROM ha_delivery_outbox d JOIN alerts a ON a.id=d.alert_id
                WHERE d.next_attempt_at<=? AND
                    (d.state='pending' OR (d.state='sending' AND d.lease_until<=?))
                ORDER BY d.next_attempt_at,d.id LIMIT 1''', (now, now)).fetchone()
            if not row:
                conn.commit()
                return None
            conn.execute('''UPDATE ha_delivery_outbox SET state='sending',
                attempts=attempts+1, lease_generation=lease_generation+1,
                lease_until=? WHERE id=?''',
                (now + max(5, min(int(lease_seconds), 300)), row['id']))
            conn.commit()
            result = dict(row)
            result['attempts'] += 1
            result['lease_generation'] += 1
            return result

    def finish_delivery(self, delivery_id, *, lease_generation, success, retryable=False,
                        error='', now=None, max_attempts=12):
        """Only the worker holding the CURRENT lease may finish the job."""
        now = time.time() if now is None else float(now)
        with self.lock, closing(self._connect()) as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('''SELECT attempts,state,lease_generation
                FROM ha_delivery_outbox WHERE id=?''',
                               (int(delivery_id),)).fetchone()
            if (not row or row['state'] != 'sending' or
                    row['lease_generation'] != int(lease_generation)):
                conn.commit()
                return False
            if success:
                conn.execute('''UPDATE ha_delivery_outbox SET state='accepted',
                    accepted_at=?, lease_until=NULL, last_error='' WHERE id=?''',
                    (now, delivery_id))
            else:
                attempt = row['attempts']
                retry = bool(retryable) and attempt < max_attempts
                wait = min(3600, 10 * (2 ** min(attempt, 9))) if retry else 0
                conn.execute('''UPDATE ha_delivery_outbox SET state=?,next_attempt_at=?,
                    lease_until=NULL,last_error=? WHERE id=?''',
                    ('pending' if retry else 'failed', now + wait,
                     str(error or 'Unbekannter Zustellfehler')[:120], delivery_id))
            conn.commit()
            return True

    def cancel_pending_deliveries(self):
        """Avoid sending old plate data to a formerly configured phone."""
        with self.lock, closing(self._connect()) as conn:
            conn.execute('BEGIN IMMEDIATE')
            count = conn.execute('''DELETE FROM ha_delivery_outbox
                WHERE state IN ('pending','sending','failed')''').rowcount
            conn.commit()
            return count

    def retry_failed_deliveries(self, *, now=None):
        now = time.time() if now is None else float(now)
        with self.lock, closing(self._connect()) as conn:
            conn.execute('BEGIN IMMEDIATE')
            count = conn.execute('''UPDATE ha_delivery_outbox
                SET state='pending',attempts=0,lease_until=NULL,next_attempt_at=?,
                    last_error='' WHERE state='failed' ''', (now,)).rowcount
            conn.commit()
            return count

    def backup_to(self, destination):
        """Online consistent copy; never copy the WAL-backed file directly."""
        with self.lock, closing(self._connect()) as src, closing(sqlite3.connect(str(destination))) as dest:
            src.backup(dest)
