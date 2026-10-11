# SPDX-License-Identifier: AGPL-3.0-only
"""Crash-resistant SQLite persistence for PlateVision detection histories.

The public history API remains a list of dictionaries in this initial release.
SQLite is the authoritative storage; legacy JSON files are *not* updated after
migration. A byte-stable pre-migration snapshot is retained. Do not enable the
old JSON backend while an SQLite database exists.

No Flask, ML, external package or user data is imported by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import re
from difflib import SequenceMatcher
import threading
import uuid
from datetime import datetime, timezone, timedelta
from contextlib import closing, contextmanager
import fcntl
from pathlib import Path

SCHEMA_VERSION = 2

# V2 adds indexed filter values without modifying the original JSON payload.
# Columns must be appended in this exact order for _event_values().
QUERY_COLUMNS = (
    ('timestamp_local', 'TEXT'), ('traffic_timestamp', 'TEXT'), ('source_lower', 'TEXT'),
    ('vehicle_type', 'TEXT'), ('vehicle_type_lower', 'TEXT'),
    ('vehicle_type_en_lower', 'TEXT'), ('vehicle_color', 'TEXT'),
    ('vehicle_color_lower', 'TEXT'), ('plate_format_lower', 'TEXT'),
    ('valid_plate', 'INTEGER'), ('watchlist_hit', 'INTEGER'),
    ('counted', 'INTEGER'), ('review_status_lower', 'TEXT'),
    ('direction', 'TEXT'), ('track_id_lower', 'TEXT'),
    ('has_images', 'INTEGER'), ('repeat_blocked', 'INTEGER'), ('people_query_text', 'TEXT'),
)
BASE_COLUMNS = ('event_key', 'rank', 'timestamp', 'plate_text', 'plate_normalized',
                'source', 'event_type', 'confidence', 'payload')
EVENT_COLUMNS = BASE_COLUMNS + tuple(column for column, _ in QUERY_COLUMNS)
INSERT_SQL = (f"INSERT INTO events ({','.join(EVENT_COLUMNS)}) "
              f"VALUES ({','.join('?' for _ in EVENT_COLUMNS)})")
UPDATE_SQL = ("UPDATE events SET " + ','.join(f'{name}=?' for name in EVENT_COLUMNS[1:])
              + ' WHERE event_key=?')
QUERY_INDEXES = (
    'CREATE INDEX IF NOT EXISTS idx_events_local_time ON events(timestamp_local)',
    'CREATE INDEX IF NOT EXISTS idx_events_traffic_time ON events(traffic_timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_events_normalized_rank ON events(plate_normalized, rank DESC)',
    'CREATE INDEX IF NOT EXISTS idx_events_source_lower ON events(source_lower)',
    'CREATE INDEX IF NOT EXISTS idx_events_vehicle_type ON events(vehicle_type_lower)',
    'CREATE INDEX IF NOT EXISTS idx_events_vehicle_color ON events(vehicle_color_lower)',
    'CREATE INDEX IF NOT EXISTS idx_events_valid ON events(valid_plate)',
    'CREATE INDEX IF NOT EXISTS idx_events_watchlist ON events(watchlist_hit)',
    'CREATE INDEX IF NOT EXISTS idx_events_people_time ON events(counted, timestamp_local)',
    'CREATE INDEX IF NOT EXISTS idx_events_people_type ON events(event_type, timestamp_local)',
    'CREATE INDEX IF NOT EXISTS idx_events_people_review ON events(review_status_lower)',
)

_OCR_CHARS = str.maketrans({
    'Ä':'A', 'Ö':'O', 'Ü':'U', 'À':'A', 'Á':'A', 'Â':'A',
    'È':'E', 'É':'E', 'Ê':'E', 'Ì':'I', 'Í':'I', 'Î':'I',
    'Ò':'O', 'Ó':'O', 'Ô':'O', 'Ù':'U', 'Ú':'U', 'Û':'U',
    '|':'1',
})


def _normalize_plate(value):
    """Same compact plate representation as PlateUtils, without Flask imports."""
    if value is None:
        return ''
    value = re.sub(r'[^A-Z0-9ÄÖÜÀÁÂÈÉÊÌÍÎÒÓÔÙÚÛ\- _./:|·•*,;]',
                   '', str(value).upper().strip().replace('\n', ' ').replace('\r', ' '))
    value = value.translate(_OCR_CHARS)
    return re.sub('[^A-Z0-9]', '', value)


def _local_timestamp(value, *, kind):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except (ValueError, TypeError, OverflowError):
        if kind != 'plates':
            return None
        try:
            dt = datetime.fromisoformat(str(value)[:10])
        except (ValueError, TypeError, OverflowError):
            return None
    return (dt.astimezone().replace(tzinfo=None) if dt.tzinfo else dt).isoformat()


def _safe_confidence(value):
    try:
        value = float(value)
        if -1e300 < value < 1e300:
            return value
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def _query_values(row, *, kind):
    """Filter columns reflect search semantics; original payload is unchanged."""
    lower = lambda value: str(value or '').lower()
    query_text = ' '.join(str(row.get(key) or '') for key in
                          ('id', 'track_id', 'source', 'event_type', 'direction',
                           'label', 'note', 'filename')).lower()
    original_timestamp = row.get('timestamp')
    try:
        raw_dt = datetime.fromisoformat(str(original_timestamp).replace('Z', '+00:00'))
        traffic_dt = raw_dt.replace(tzinfo=None).isoformat()
    except (ValueError, TypeError, OverflowError):
        try:
            traffic_dt = (datetime.fromisoformat(str(original_timestamp)[:10]).isoformat()
                          if kind == 'plates' else None)
        except (ValueError, TypeError, OverflowError):
            traffic_dt = None
    return (
        _local_timestamp(row.get('timestamp'), kind=kind), traffic_dt,
        lower(row.get('source')),
        str(row.get('vehicle_type') or ''),
        lower(row.get('vehicle_type')), lower(row.get('vehicle_type_en')),
        str(row.get('vehicle_color') or ''), lower(row.get('vehicle_color')),
        lower(row.get('plate_format')),
        int(bool(row.get('is_valid_plate', True))), int(bool(row.get('watchlist_match'))),
        int(bool(row.get('counted', True))), lower(row.get('review_status') or 'unreviewed'),
        str(row.get('direction') or ''), lower(row.get('track_id')),
        int(bool(row.get('images') or {})),
        int(bool(row.get('repeat_blocked') or row.get('event_type') == 'repeat_blocked')), query_text,
    )



class HistoryStorageError(OSError):
    """A migration, consistency check or transaction failed without data loss."""


class ReviewConflictError(ValueError):
    """The browser tried to overwrite a review edited by another client."""


REVIEW_STATUSES = ('unreviewed', 'confirmed', 'needs_review', 'ignored')


def _sha256(path: Path):
    if not path.is_file():
        return 'MISSING'
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _write_once_json(path, rows):
    """Create (never overwrite) a durable pre-migration JSON backup."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise HistoryStorageError('Migrationssicherung darf kein Symlink sein')
    if path.exists():
        with path.open('r', encoding='utf-8') as stream:
            if json.load(stream) != rows:
                raise HistoryStorageError(f'Abweichende bestehende Migrationssicherung: {path}')
        return
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    try:
        with tmp.open('x', encoding='utf-8') as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            with path.open('r', encoding='utf-8') as handle:
                if json.load(handle) != rows:
                    raise HistoryStorageError('Migrationssicherung wurde parallel verändert')
        else:
            os.replace(tmp, path)
            _fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _fsync_directory(folder):
    if hasattr(os, 'O_DIRECTORY'):
        try:
            fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass  # Not every host filesystem permits directory fsync.


class SQLiteHistoryStore:
    """Per-history SQLite store with optimistic revisions and safe migration.

    Every sync runs in BEGIN IMMEDIATE, with revision checking. Changed JSON
    payloads are updated, deleted rows are removed, and unchanged rows remain
    untouched on disk. Event order uses monotonic ranks, not a full renumber on
    each prepend. Calls are serialized by their manager's RLock.
    """

    def __init__(self, legacy_json, *, kind):
        if kind not in ('plates', 'people'):
            raise ValueError('Unbekannte Historienart')
        self.legacy_json = Path(legacy_json)
        self.path = self.legacy_json.parent / f'{kind}.sqlite3'
        self.snapshot = self.legacy_json.with_name(self.legacy_json.stem + '.pre-sqlite.json')
        self.kind = kind
        self.revision = None
        self.migrated_at = None
        self._mutex = threading.RLock()
        self._legacy_signature = None
        self._legacy_digest = None

    @contextmanager
    def _migration_file_lock(self):
        """Keep two overlapping add-on startups from racing to replace a DB."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        path = self.path.with_name(self.path.name + '.migration.lock')
        if path.is_symlink():
            raise HistoryStorageError('Migrationssperre ist unerwartet ein Symlink')
        with path.open('a+b') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _legacy_hash(self):
        """Detect legacy-file changes without hashing megabytes on every event."""
        try:
            stat = self.legacy_json.stat()
            signature = (stat.st_dev, stat.st_ino, stat.st_size,
                         stat.st_mtime_ns, stat.st_ctime_ns)
        except FileNotFoundError:
            signature = None
        if signature != self._legacy_signature or self._legacy_digest is None:
            self._legacy_digest = _sha256(self.legacy_json)
            self._legacy_signature = signature
        return self._legacy_digest

    def _connect(self, path=None, *, read_only=False):
        path = Path(path or self.path)
        if read_only:
            conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=10)
        else:
            conn = sqlite3.connect(str(path), timeout=10)
        conn.execute('PRAGMA busy_timeout=10000')
        conn.create_function('plate_similarity', 2,
            lambda a, b: (SequenceMatcher(None, str(a), str(b)).ratio()
                          if a and b else 0.0), deterministic=True)
        if not read_only:
            if path == self.path and path.is_file():
                conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=FULL')
        return conn

    @staticmethod
    def _schema(conn):
        query_columns = ',\n'.join(f'{name} {kind}' for name, kind in QUERY_COLUMNS)
        conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (
            event_key TEXT PRIMARY KEY,
            rank INTEGER NOT NULL UNIQUE,
            timestamp TEXT, plate_text TEXT, plate_normalized TEXT,
            source TEXT, event_type TEXT, confidence REAL, payload TEXT NOT NULL,
            {query_columns}
        );
        CREATE INDEX IF NOT EXISTS idx_events_order ON events(rank DESC);
        CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate_normalized);
        CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
        CREATE INDEX IF NOT EXISTS idx_events_confidence ON events(confidence);
        """)
        for statement in QUERY_INDEXES:
            conn.execute(statement)

    def _upgrade_query_schema(self, conn):
        """One atomic v1->v2 upgrade, preserving every payload and old JSON backup."""
        metadata = self._meta(conn)
        if metadata.get('kind') != self.kind or metadata.get('schema_version') not in ('1', '2'):
            raise HistoryStorageError('Unbekanntes SQLite-Historienschema')
        if metadata.get('source_sha256') != self._legacy_hash():
            raise HistoryStorageError('Legacy-JSON wurde nach der Migration verändert')
        if metadata['schema_version'] == '2':
            current = {row[1] for row in conn.execute('PRAGMA table_info(events)')}
            if not set(EVENT_COLUMNS) <= current:
                raise HistoryStorageError('SQLite-Abfrageindizes oder Spalten fehlen')
            return
        conn.execute('BEGIN IMMEDIATE')
        try:
            for name, kind in QUERY_COLUMNS:
                conn.execute(f'ALTER TABLE events ADD COLUMN {name} {kind}')
            updates = ('UPDATE events SET plate_normalized=?, '
                       + ','.join(f'{name}=?' for name, _ in QUERY_COLUMNS)
                       + ' WHERE event_key=?')
            cursor = conn.execute('SELECT event_key,payload FROM events')
            while True:
                batch = cursor.fetchmany(500)
                if not batch:
                    break
                values = []
                for event_key, payload in batch:
                    row = json.loads(payload)
                    if not isinstance(row, dict) or str(row.get('id')) != event_key:
                        raise HistoryStorageError('Beschädigtes SQLite-Ereignis beim Schema-Upgrade')
                    values.append((_normalize_plate(row.get('plate_text')), *_query_values(row, kind=self.kind), event_key))
                conn.executemany(updates, values)
            for statement in QUERY_INDEXES:
                conn.execute(statement)
            self._put_meta(conn, 'schema_version', SCHEMA_VERSION)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _put_meta(conn, key, value):
        conn.execute('INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)', (key, str(value)))

    @staticmethod
    def _meta(conn):
        return dict(conn.execute('SELECT key,value FROM metadata'))

    @staticmethod
    def _normalize_rows(rows, kind):
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise HistoryStorageError('Historie enthält keine gültige Objektliste')
        used = set()
        prepared = []
        repaired = 0
        for index, original in enumerate(rows):
            row = dict(original)
            raw = str(row.get('id') or '').strip()
            if not raw or raw in used:
                raw = str(uuid.uuid5(uuid.NAMESPACE_URL,
                                    f'platevision-{kind}-{index}-{_json_text(original)}'))
                while raw in used:
                    raw = str(uuid.uuid4())
                row['id'] = raw
                repaired += 1
            used.add(raw)
            prepared.append(row)
        return prepared, repaired

    def _event_values(self, row, rank):
        return (str(row['id']), rank, str(row.get('timestamp') or ''),
                str(row.get('plate_text') or ''), _normalize_plate(row.get('plate_text')),
                str(row.get('source') or ''), str(row.get('event_type') or ''),
                _safe_confidence(row.get('confidence')), _json_text(row),
                *_query_values(row, kind=self.kind))

    def _validate(self, conn):
        status = conn.execute('PRAGMA integrity_check').fetchone()
        if not status or status[0] != 'ok':
            raise HistoryStorageError('SQLite-integrity_check fehlgeschlagen')
        meta = self._meta(conn)
        if meta.get('kind') != self.kind or meta.get('schema_version') != str(SCHEMA_VERSION):
            raise HistoryStorageError('SQLite-Datei enthält nicht das erwartete Schema')
        if not meta.get('revision', '').isdigit():
            raise HistoryStorageError('SQLite-Revisionsmetadaten fehlen')
        if meta.get('source_sha256') != self._legacy_hash():
            raise HistoryStorageError('Legacy-JSON wurde seit der SQLite-Übernahme verändert: manuell abgleichen; nicht automatisch überschreiben')
        return meta

    def initialize(self, legacy_loader):
        """Verify an existing DB or migrate validated JSON without altering it."""
        with self._mutex, self._migration_file_lock():
            if self.path.is_symlink():
                raise HistoryStorageError('SQLite-Datei darf kein Symlink sein')
            if self.path.is_file():
                with closing(self._connect()) as conn:
                    self._upgrade_query_schema(conn)
                    meta = self._validate(conn)
                    self.revision = int(meta['revision'])
                    self.migrated_at = meta.get('migrated_at')
                    return self._load_from_conn(conn)
            # Check the validated primary / .bak with the application's own
            # safe-mode loader. On invalid input it will set a write-protection
            # flag; the caller must refuse migration in that case.
            legacy = legacy_loader()
            if not isinstance(legacy, list) or any(not isinstance(row, dict) for row in legacy):
                raise HistoryStorageError('Ungültige JSON-Quelldaten')
            prepared, repaired = self._normalize_rows(legacy, self.kind)
            _write_once_json(self.snapshot, legacy)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            staging = self.path.with_name(f'.{self.path.name}.{os.getpid()}.{threading.get_ident()}.migrating')
            try:
                with self._connect(staging) as conn:
                    conn.execute('PRAGMA journal_mode=DELETE')
                    self._schema(conn)
                    now = datetime.now(timezone.utc).isoformat()
                    self._put_meta(conn, 'kind', self.kind)
                    self._put_meta(conn, 'schema_version', SCHEMA_VERSION)
                    self._put_meta(conn, 'revision', 1)
                    self._put_meta(conn, 'migrated_at', now)
                    self._put_meta(conn, 'source_sha256', self._legacy_hash())
                    self._put_meta(conn, 'source_records', len(legacy))
                    self._put_meta(conn, 'repaired_legacy_ids', repaired)
                    conn.executemany(INSERT_SQL,
                        [self._event_values(row, len(prepared) - index) for index, row in enumerate(prepared)])
                    conn.commit()
                    if conn.execute('SELECT COUNT(*) FROM events').fetchone()[0] != len(legacy):
                        raise HistoryStorageError('Zeilenzahl bei SQLite-Migration weicht ab')
                    if self._load_from_conn(conn) != prepared:
                        raise HistoryStorageError('SQLite-Migration: Zeilen oder Reihenfolge stimmen nicht')
                    if conn.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                        raise HistoryStorageError('SQLite-Migration: Integritätsprüfung fehlgeschlagen')
                if self.path.exists():
                    raise HistoryStorageError('Andere Migration hat bereits SQLite erstellt')
                os.replace(staging, self.path)
                _fsync_directory(self.path.parent)
                # Enable WAL only on the final path; the staging database used
                # DELETE journaling to remain a single-file atomic migration.
                with closing(self._connect()) as finalized:
                    if finalized.execute('PRAGMA journal_mode').fetchone()[0] != 'wal':
                        raise HistoryStorageError('SQLite WAL-Modus konnte nicht aktiviert werden')
            finally:
                for suffix in ('', '-journal', '-wal', '-shm'):
                    staging.with_name(staging.name + suffix).unlink(missing_ok=True)
            self.revision = 1
            self.migrated_at = now
            return prepared

    @staticmethod
    def _load_from_conn(conn):
        rows = []
        for event_key, raw in conn.execute('SELECT event_key,payload FROM events ORDER BY rank DESC'):
            row = json.loads(raw)
            if not isinstance(row, dict) or str(row.get('id')) != event_key:
                raise HistoryStorageError('SQLite-Eintrag beschädigt oder ID widersprüchlich')
            rows.append(row)
        return rows

    def load(self):
        with self._mutex, self._connect() as conn:
            meta = self._validate(conn)
            self.revision = int(meta['revision'])
            self.migrated_at = meta.get('migrated_at')
            return self._load_from_conn(conn)

    def sync(self, rows):
        """Atomically reconcile a list snapshot without replacing the whole DB."""
        with self._mutex:
            if self.revision is None:
                raise HistoryStorageError('SQLite wurde noch nicht erfolgreich geöffnet')
            if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
                raise HistoryStorageError('Ungültiger Historieninhalt')
            ids = [str(row.get('id') or '') for row in rows]
            if any(not x for x in ids) or len(ids) != len(set(ids)):
                raise HistoryStorageError('Doppelte oder fehlende Eintrags-IDs')
            conn = self._connect()
            try:
                conn.execute('BEGIN IMMEDIATE')
                meta = self._meta(conn)
                if int(meta.get('revision', '0')) != self.revision:
                    raise HistoryStorageError('SQLite wurde parallel von einem zweiten Prozess geändert; Dienst neu starten')
                if meta.get('source_sha256') != self._legacy_hash():
                    raise HistoryStorageError('Legacy-JSON wurde zwischenzeitlich verändert')
                existing = {key: (rank, payload) for key, rank, payload in
                            conn.execute('SELECT event_key, rank, payload FROM events')}
                wanted = set(ids)
                previous_order = [key for key, _ in sorted(existing.items(),
                                                          key=lambda p: p[1][0], reverse=True) if key in wanted]
                current_old = [key for key in ids if key in existing]
                head_new = len(ids) - len(current_old)
                simple_prepend = (previous_order == current_old and
                                  all(key not in existing for key in ids[:head_new]))
                rank_max = max((r for r, _ in existing.values()), default=0)
                rank_min = min((r for r, _ in existing.values()), default=0)
                rank_map = {}
                if simple_prepend:
                    for index, key in enumerate(ids[:head_new]):
                        rank_map[key] = rank_max + head_new - index
                    for key in current_old:
                        rank_map[key] = existing[key][0]
                else:
                    # Rare case: arbitrary reorder, insert in the middle, or
                    # rebuilding a migrated list. Reassign order without
                    # UNIQUE collisions by moving old ranks into a temporary
                    # disjoint range within the same transaction.
                    low = min(rank_min, 0) - len(existing) - 1
                    for index, key in enumerate(existing):
                        conn.execute('UPDATE events SET rank=? WHERE event_key=?', (low - index, key))
                    for index, key in enumerate(ids):
                        rank_map[key] = len(ids) - index
                removed = list(set(existing) - wanted)
                if removed:
                    conn.executemany('DELETE FROM events WHERE event_key=?', [(key,) for key in removed])
                for row in rows:
                    key = str(row['id'])
                    rank = rank_map[key]
                    values = self._event_values(row, rank)
                    if key not in existing:
                        conn.execute(INSERT_SQL, values)
                    elif existing[key][1] != values[-1] or existing[key][0] != rank:
                        conn.execute(UPDATE_SQL, values[1:] + (key,))
                self._put_meta(conn, 'revision', self.revision + 1)
                conn.commit()
                self.revision += 1
            except (Exception,):
                conn.rollback()
                raise
            finally:
                conn.close()

    def prepend(self, row, *, removed_ids=()):
        """O(log n) primary write path for a new detection + retention pruning.

        This path never rewrites old records, unlike a whole-JSON save. The
        manager still performs duplicate checks, ordering and in-memory
        retention with its RLock before calling this atomic SQL transaction.
        """
        key = str(row.get('id') or '')
        if not key:
            raise HistoryStorageError('Neuem Erkennungseintrag fehlt die ID')
        with self._mutex:
            if self.revision is None:
                raise HistoryStorageError('SQLite wurde nicht initialisiert')
            conn = self._connect()
            try:
                conn.execute('BEGIN IMMEDIATE')
                meta = self._meta(conn)
                if int(meta.get('revision', '0')) != self.revision:
                    raise HistoryStorageError('SQLite wurde parallel geändert; Neustart erforderlich')
                if meta.get('source_sha256') != self._legacy_hash():
                    raise HistoryStorageError('Legacy-JSON seit Migration verändert')
                highest = conn.execute('SELECT COALESCE(MAX(rank),0) FROM events').fetchone()[0]
                conn.execute(INSERT_SQL, self._event_values(row, highest + 1))
                removed = [(str(id_value),) for id_value in removed_ids if id_value is not None]
                if removed:
                    conn.executemany('DELETE FROM events WHERE event_key=?', removed)
                self._put_meta(conn, 'revision', self.revision + 1)
                conn.commit()
                self.revision += 1
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def update_plate_review(self, entry_id, status, note, *, expected_updated_at=None):
        """Atomically update one review without reconciling the entire history.

        Every write checks the same optimistic DB revision used by prepend/sync;
        the legacy migration fingerprint and original plate/image fields remain
        untouched. A client may supply the last displayed timestamp to detect
        edits made in another browser tab.
        """
        if self.kind != 'plates':
            raise ValueError('Reviews sind nur für Kennzeichen vorgesehen')
        if status not in REVIEW_STATUSES or not isinstance(note, str) or len(note) > 1200:
            raise ValueError('Ungültiger Prüfstatus oder zu lange Notiz')
        with self._mutex:
            if self.revision is None:
                raise HistoryStorageError('SQLite wurde nicht initialisiert')
            conn = self._connect()
            try:
                conn.execute('BEGIN IMMEDIATE')
                meta = self._meta(conn)
                if int(meta.get('revision', '0')) != self.revision:
                    raise HistoryStorageError('SQLite wurde parallel geändert; Neustart erforderlich')
                if meta.get('source_sha256') != self._legacy_hash():
                    raise HistoryStorageError('Legacy-JSON seit Migration verändert')
                found = conn.execute('SELECT payload FROM events WHERE event_key=?',
                                     (str(entry_id),)).fetchone()
                if found is None:
                    conn.rollback()
                    return None
                current = json.loads(found[0])
                if not isinstance(current, dict) or str(current.get('id')) != str(entry_id):
                    raise HistoryStorageError('SQLite-Eintrag beschädigt oder ID widersprüchlich')
                if (expected_updated_at is not None and
                        str(current.get('review_updated_at') or '') != expected_updated_at):
                    raise ReviewConflictError('Der Prüfstatus wurde zwischenzeitlich geändert. Bitte neu laden.')
                updated = dict(current)
                updated['review_status'] = status
                updated['review_note'] = note
                updated['review_updated_at'] = datetime.now(timezone.utc).isoformat()
                conn.execute('UPDATE events SET payload=?, review_status_lower=? WHERE event_key=?',
                             (_json_text(updated), status, str(entry_id)))
                self._put_meta(conn, 'revision', self.revision + 1)
                conn.commit()
                self.revision += 1
                return updated
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def plate_review_counts(self):
        """Small SQL aggregation for the review dashboard, no history scan."""
        if self.kind != 'plates':
            raise ValueError('Nur Kennzeichen haben diese Prüfstatistik')
        with self._mutex, closing(self._connect(read_only=True)) as conn:
            groups = dict(conn.execute('SELECT review_status_lower,COUNT(*) FROM events GROUP BY review_status_lower'))
        counts = {key: groups.get(key, 0) for key in REVIEW_STATUSES}
        counts['unreviewed'] += sum(n for key, n in groups.items() if key not in REVIEW_STATUSES)
        counts['total'] = sum(groups.values())
        return counts

    def get(self, entry_id):
        """Indexed lookup, without scanning the in-memory list."""
        with self._mutex, closing(self._connect(read_only=True)) as conn:
            row = conn.execute('SELECT payload FROM events WHERE event_key=?', (str(entry_id),)).fetchone()
            return json.loads(row[0]) if row else None

    def page(self, limit=100, offset=0):
        """Index-ordered reading with SQL LIMIT/OFFSET."""
        with self._mutex, closing(self._connect(read_only=True)) as conn:
            rows = conn.execute('SELECT payload FROM events ORDER BY rank DESC LIMIT ? OFFSET ?',
                                (int(limit), int(offset)))
            return [json.loads(row[0]) for row in rows]

    @staticmethod
    def _read_rows(conn, sql, parameters=()):
        return [json.loads(item[0]) for item in conn.execute(sql, parameters)]

    @staticmethod
    def _safe_float(value, default=0):
        return _safe_confidence(value) if _safe_confidence(value) is not None else default

    def _plate_where(self, filters):
        """Build a parameterized WHERE using the same semantics as HistoryManager.

        Regex and fuzzy *unique grouping* are deliberately handled in Python:
        those cases require a compiled pattern / greedy edit-distance grouping.
        A fuzzy ordinary query uses a bounded SQLite scalar function and never
        reads entire JSON payloads to serve one page.
        """
        conditions, params = [], []
        query = str(filters.get('q') or filters.get('search') or '').strip()[:256]
        if query:
            normalized = _normalize_plate(query)
            mode = filters.get('match_mode') or 'contains'
            if mode == 'exact':
                conditions.append('plate_normalized=?')
                params.append(normalized)
            elif mode == 'starts_with':
                conditions.append('plate_normalized>=? AND plate_normalized<?')
                params.extend((normalized, normalized + '['))
            elif filters.get('_fuzzy_enabled'):
                conditions.append('(instr(plate_normalized, ?) > 0 OR plate_similarity(?, plate_normalized) >= ?)')
                params.extend((normalized, normalized, float(filters.get('_fuzzy_threshold', .72))))
            elif mode not in ('exact','starts_with'):
                conditions.append('instr(plate_normalized, ?) > 0')
                params.append(normalized)
        if filters.get('source') and filters['source'] != 'all':
            wanted = str(filters['source']).strip().lower()
            aliases = {'rtsp_stream': 'rtsp', 'rtsp': 'rtsp_stream'}
            actuals = [wanted] + ([aliases[wanted]] if wanted in aliases else [])
            conditions.append(f"source_lower IN ({','.join('?' for _ in actuals)})")
            params.extend(actuals)
        if filters.get('vehicle_type') and filters['vehicle_type'] != 'all':
            wanted = str(filters['vehicle_type']).strip().lower()
            conditions.append("((vehicle_type_lower != '' AND "
                              "(instr(vehicle_type_lower, ?) > 0 OR instr(?, vehicle_type_lower) > 0)) "
                              "OR (vehicle_type_en_lower != '' AND "
                              "(instr(vehicle_type_en_lower, ?) > 0 OR instr(?, vehicle_type_en_lower) > 0)))")
            params.extend((wanted, wanted, wanted, wanted))
        for key, column in (('vehicle_color', 'vehicle_color_lower'),
                            ('plate_format', 'plate_format_lower')):
            if filters.get(key) and filters[key] != 'all':
                conditions.append(f'{column}=?')
                params.append(str(filters[key]).strip().lower())
        if filters.get('valid_only'):
            conditions.append('valid_plate=1')
        if filters.get('watchlist_only'):
            conditions.append('watchlist_hit=1')
        review = str(filters.get('review_status') or '').strip().lower()
        if review and review != 'all':
            # New entries already have the indexed 'unreviewed' default.
            # Unknown values from old imports are treated as unreviewed.
            if review == 'unreviewed':
                conditions.append("(review_status_lower='unreviewed' OR review_status_lower NOT IN ('confirmed','needs_review','ignored','unreviewed'))")
            elif review in REVIEW_STATUSES:
                conditions.append('review_status_lower=?')
                params.append(review)
            else:
                conditions.append('0=1')
        if filters.get('min_confidence') not in (None, ''):
            conditions.append('COALESCE(confidence,0)>=?')
            params.append(self._safe_float(filters['min_confidence'], 0))
        if filters.get('max_confidence') not in (None, ''):
            conditions.append('COALESCE(confidence,0)<=?')
            params.append(self._safe_float(filters['max_confidence'], 1))
        if filters.get('date_from'):
            if filters.get('_date_from_local'):
                conditions.append('timestamp_local>=?')
                params.append(filters['_date_from_local'])
            # Legacy behavior: invalid date inputs are ignored, but a row
            # without a valid timestamp never matches a date filter.
            conditions.append('timestamp_local IS NOT NULL')
        if filters.get('date_to'):
            if filters.get('_date_to_local'):
                conditions.append('timestamp_local<=?')
                params.append(filters['_date_to_local'])
            conditions.append('timestamp_local IS NOT NULL')
        return (' WHERE ' + ' AND '.join(conditions) if conditions else ''), params

    @staticmethod
    def _plate_order(filters):
        direction = 'ASC' if str(filters.get('order') or 'desc').lower() == 'asc' else 'DESC'
        sort = filters.get('sort') or 'timestamp'
        column = ('COALESCE(confidence,0)' if sort == 'confidence' else
                  'plate_normalized' if sort == 'plate' else
                  "COALESCE(timestamp_local, '')")
        return f' ORDER BY {column} {direction}, rank DESC'

    def _plate_sql(self, filters, *, paginated=False):
        where, params = self._plate_where(filters)
        order = self._plate_order(filters)
        if filters.get('unique'):
            # Exact uniqueness follows the original latest-rank-wins behavior.
            sql = ('WITH matches AS (SELECT *, '
                   'ROW_NUMBER() OVER (PARTITION BY plate_normalized ORDER BY rank DESC) AS nth '
                   'FROM events' + where + ') '
                   "SELECT payload FROM matches WHERE nth=1 AND plate_normalized != ''" + order)
            counter = ('WITH matches AS (SELECT plate_normalized FROM events' + where + ') '
                       "SELECT COUNT(DISTINCT NULLIF(plate_normalized,'')) FROM matches")
        else:
            sql = 'SELECT payload FROM events' + where + order
            counter = 'SELECT COUNT(*) FROM events' + where
        if paginated:
            sql += ' LIMIT ? OFFSET ?'
        return sql, counter, params

    def best_vehicle_for_display(self, *, minimum=0.8, window_seconds=180,
                                 prefer_sharpness=True, sharpness_weight=0.25):
        """Return one best capture with just two small DB reads.

        Read only metadata for up to 150 recent events, not their base64
        payloads. A missing qualifying result must not fall back to a low
        confidence image. The API/history feed itself stays unfiltered.
        """
        if self.kind != 'plates':
            raise ValueError('Only for plate history')
        from vehicle_display import best_image_candidate
        with self._mutex, closing(self._connect(read_only=True)) as conn:
            conn.execute('BEGIN')
            rows = conn.execute('''SELECT event_key,plate_normalized,plate_text,source,confidence,
                       timestamp_local,rank,
                       json_extract(payload, '$.video_job_id'),
                       json_extract(payload, '$.filename'),
                       json_extract(payload, '$.upload_id'),
                       json_extract(payload, '$.plate_sharpness'),
                       CASE WHEN COALESCE(json_extract(payload, '$.vehicle_image'),'')!=''
                                  OR COALESCE(json_extract(payload, '$.plate_image'),'')!=''
                            THEN 1 ELSE 0 END
                       FROM events ORDER BY rank DESC LIMIT 150''').fetchall()
            candidates = [dict(id=r[0],plate_normalized=r[1],plate_text=r[2],
                               source=r[3],confidence=r[4],timestamp_local=r[5],
                               rank=r[6],video_job_id=r[7],filename=r[8],upload_id=r[9],
                               plate_sharpness=r[10],images_available=bool(r[11])) for r in rows]
            chosen = best_image_candidate(candidates, minimum=minimum,
                                          window_seconds=window_seconds,
                                          prefer_sharpness=prefer_sharpness,
                                          sharpness_weight=sharpness_weight)
            if not chosen:
                return None
            result = conn.execute('SELECT payload FROM events WHERE event_key=?',
                                  (chosen['id'],)).fetchone()
            return json.loads(result[0]) if result else None

    def search_plates(self, filters, *, limit=None, offset=0):
        """One consistent read snapshot for SQL filtered count and page."""
        if self.kind != 'plates':
            raise ValueError('Nur für Kennzeichenhistorien')
        sql, counter, params = self._plate_sql(filters, paginated=limit is not None)
        with self._mutex, closing(self._connect(read_only=True)) as conn:
            conn.execute('BEGIN')
            total = conn.execute(counter, params).fetchone()[0]
            suffix = [int(limit), int(offset)] if limit is not None else []
            rows = self._read_rows(conn, sql, [*params, *suffix])
            return rows, total

    def iter_plates(self, filters, *, batch_size=256):
        """Stream a consistent SQLite snapshot instead of copying all history."""
        if self.kind != 'plates':
            raise ValueError('Nur für Kennzeichenhistorien')
        sql, _, params = self._plate_sql(filters)
        with closing(self._connect(read_only=True)) as conn:
            cursor = conn.execute(sql, params)
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                for (payload,) in batch:
                    yield json.loads(payload)

    def plate_facets(self):
        if self.kind != 'plates':
            raise ValueError('Nur für Kennzeichenhistorien')
        with closing(self._connect(read_only=True)) as conn:
            result = {}
            for name, col, default in (
                ('sources', 'source', 'Unbekannt'),
                ('vehicle_types', 'vehicle_type', 'Unbekannt'),
                ('vehicle_colors', 'vehicle_color', 'Unbekannt'),
                ('plate_formats', "json_extract(payload, '$.plate_format')", 'Unbekannt'),
            ):
                # Group on stored display labels, matching Counter semantics.
                expr = f"COALESCE(NULLIF({col},''),?)"
                rows = conn.execute(f'SELECT {expr} AS label,COUNT(*) AS n FROM events '
                                    f'GROUP BY label ORDER BY n DESC, MAX(rank) DESC', (default,)).fetchall()
                result[name] = [{'value': str(label), 'count': n} for label, n in rows]
            return result

    def autocomplete(self, query='', limit=12):
        """Plate suggestions aggregate counts in SQL without reading image JSON."""
        if self.kind != 'plates':
            raise ValueError('Nur für Kennzeichenhistorien')
        normalized = _normalize_plate(query)
        clauses = ["plate_normalized!=''"]
        params = []
        if normalized:
            clauses.append('(instr(plate_normalized,?)>0 OR plate_similarity(?,plate_normalized)>=0.65)')
            params.extend((normalized,normalized))
        statement = ('SELECT plate_text,COUNT(*) AS n FROM events WHERE '
                     + ' AND '.join(clauses) +
                     ' GROUP BY plate_text ORDER BY n DESC, MAX(rank) DESC LIMIT ?')
        with closing(self._connect(read_only=True)) as conn:
            return [{'plate_text':plate, 'count':count} for plate,count in
                    conn.execute(statement, [*params,int(limit)])]

    def timeline(self, *, bucket='day', limit=60):
        """Local-time day/hour chart buckets aggregated directly in SQLite."""
        if self.kind != 'plates':
            raise ValueError('Nur für Kennzeichenhistorien')
        expr = ("replace(substr(timestamp_local,1,13),'T',' ')||':00'"
                if bucket=='hour' else 'substr(timestamp_local,1,10)')
        with closing(self._connect(read_only=True)) as conn:
            statement = (f'SELECT {expr} AS t,COUNT(*) FROM events '
                         'WHERE timestamp_local IS NOT NULL GROUP BY t ORDER BY t DESC LIMIT ?')
            data = conn.execute(statement,(int(limit),)).fetchall()
        return [{'time':stamp,'count':count} for stamp,count in reversed(data)]

    def today_plate_counts(self, *, now, fuzzy=False, similarity=0.88):
        """Today's counters without deserializing the complete image history.

        With fuzzy duplicate grouping enabled, preserve the previous newest-
        first grouping algorithm on just small normalized plate strings.
        """
        if self.kind != 'plates':
            raise ValueError('Nur für Kennzeichenhistorien')
        today = now.date().isoformat()
        tomorrow = (now.date() + timedelta(days=1)).isoformat()
        with closing(self._connect(read_only=True)) as conn:
            if not fuzzy:
                events, unique = conn.execute(
                    "SELECT COUNT(*),COUNT(DISTINCT NULLIF(plate_normalized,'')) "
                    "FROM events WHERE timestamp_local>=? AND timestamp_local<?",
                    (today,tomorrow)).fetchone()
                return int(events), int(unique)
            try:
                threshold = max(0.0, min(1.0, float(similarity)))
                if not __import__('math').isfinite(threshold):
                    threshold = .88
            except (ValueError, TypeError, OverflowError):
                threshold = .88
            strings = conn.execute(
                "SELECT plate_normalized FROM events "
                "WHERE timestamp_local>=? AND timestamp_local<? ORDER BY rank DESC", (today,tomorrow))
            used = []
            total = 0
            for (plate,) in strings:
                total += 1
                if plate and not any(SequenceMatcher(None, plate, other).ratio() >= threshold
                                     for other in used):
                    used.append(plate)
            return total, len(used)

    def plate_summary(self, *, now):
        """Basic dashboard statistics are aggregated in SQL, not Python lists."""
        if self.kind != 'plates':
            raise ValueError('Nur für Kennzeichenhistorien')
        now_iso = now.isoformat()
        hour_start = (now - __import__('datetime').timedelta(hours=1)).isoformat()
        today = now.date().isoformat()
        with closing(self._connect(read_only=True)) as conn:
            conn.execute('BEGIN')
            total, unique, avg_conf, valid, watch = conn.execute("""
                SELECT COUNT(*),COUNT(DISTINCT NULLIF(plate_normalized,'')),
                       AVG(MIN(1,MAX(0,COALESCE(confidence,0)))),
                       COALESCE(SUM(valid_plate),0),COALESCE(SUM(watchlist_hit),0)
                FROM events""").fetchone()
            today_count = conn.execute('SELECT COUNT(*) FROM events WHERE substr(timestamp_local,1,10)=?', (today,)).fetchone()[0]
            last_hour_count = conn.execute('SELECT COUNT(*) FROM events WHERE timestamp_local>=? AND timestamp_local<=?',
                                           (hour_start, now_iso)).fetchone()[0]
            buckets = conn.execute("""SELECT
                SUM(CASE WHEN MIN(1,MAX(0,COALESCE(confidence,0))) < .4 THEN 1 ELSE 0 END),
                SUM(CASE WHEN MIN(1,MAX(0,COALESCE(confidence,0))) >= .4 AND MIN(1,MAX(0,COALESCE(confidence,0))) < .6 THEN 1 ELSE 0 END),
                SUM(CASE WHEN MIN(1,MAX(0,COALESCE(confidence,0))) >= .6 AND MIN(1,MAX(0,COALESCE(confidence,0))) < .8 THEN 1 ELSE 0 END),
                SUM(CASE WHEN MIN(1,MAX(0,COALESCE(confidence,0))) >= .8 THEN 1 ELSE 0 END)
                FROM events""").fetchone()
            def frequency(column, fallback):
                return dict(conn.execute('SELECT COALESCE(NULLIF('+column+",''),?),COUNT(*) "
                                         'FROM events GROUP BY 1', (fallback,)).fetchall())
            tops = conn.execute("""SELECT plate_text,COUNT(*) n FROM events
                WHERE plate_normalized!='' GROUP BY plate_text
                ORDER BY n DESC,MAX(rank) DESC LIMIT 10""").fetchall()
            hours = conn.execute("""SELECT substr(timestamp_local,12,2)||':00',COUNT(*)
                FROM events WHERE timestamp_local IS NOT NULL
                GROUP BY 1 ORDER BY 1""").fetchall()
            days = conn.execute("""SELECT substr(timestamp_local,1,10),COUNT(*)
                FROM events WHERE timestamp_local IS NOT NULL
                GROUP BY 1 ORDER BY 1 DESC LIMIT 30""").fetchall()
            latest = conn.execute('SELECT payload FROM events ORDER BY rank DESC LIMIT 1').fetchone()
            last_detection = json.loads(latest[0]) if latest else None
            if last_detection:
                for key in ('plate_image','vehicle_image','full_frame','full_frame_image'):
                    last_detection.pop(key, None)
            return dict(total_detections=total, today_detections=today_count,
                        last_hour_detections=last_hour_count, unique_plates=unique,
                        average_confidence=avg_conf or 0,
                        vehicle_types=frequency('vehicle_type','Unbekannt'),
                        vehicle_colors=frequency('vehicle_color','Unbekannt'),
                        sources=frequency('source','unknown'),
                        top_plates=[(plate,n) for plate,n in tops],
                        hourly=dict(hours),daily=dict(reversed(days)),
                        confidence_buckets=dict(zip(('0-40','40-60','60-80','80-100'),
                                                    (b or 0 for b in buckets))),
                        last_detection=last_detection,valid_plates=valid,watchlist_hits=watch)

    def traffic_rows(self, *, date_from=None, date_to=None, min_confidence=0,
                     ignore_unknown=True, plate=None):
        """Only fetch candidate rows needed for the Python visit-session heuristic."""
        where = ['traffic_timestamp IS NOT NULL','COALESCE(confidence,0)>=?']
        params = [float(min_confidence)]
        if date_from:
            where.append('traffic_timestamp>=?')
            params.append(date_from)
        if date_to:
            where.append('traffic_timestamp<=?')
            params.append(date_to)
        if ignore_unknown:
            where.append("plate_normalized NOT IN ('','UNKNOWN','UNBEKANNT')")
        if plate is not None:
            where.append('plate_normalized=?')
            params.append(plate)
        sql = 'SELECT payload FROM events WHERE '+ ' AND '.join(where) + ' ORDER BY traffic_timestamp ASC, rank DESC'
        with closing(self._connect(read_only=True)) as conn:
            return self._read_rows(conn, sql, params)

    def plate_recent(self, *, beginning, end):
        """Dashboard day snapshot in insertion order for fuzzy uniqueness."""
        with closing(self._connect(read_only=True)) as conn:
            return self._read_rows(conn, 'SELECT payload FROM events WHERE timestamp_local>=? AND timestamp_local<? ORDER BY rank DESC',
                                   (beginning, end))

    def _people_where(self, filters):
        where = ['timestamp_local IS NOT NULL']
        params = []
        for key, sign in (('date_from', '>='), ('date_to', '<=')):
            if filters.get(key):
                where.append(f'timestamp_local{sign}?')
                params.append(filters[key].isoformat() if hasattr(filters[key], 'isoformat') else str(filters[key]))
        where.append('COALESCE(confidence,0)>=?')
        params.append(float(filters.get('min_confidence',0) or 0))
        if filters.get('counted') is not None:
            where.append('counted=?')
            params.append(int(bool(filters['counted'])))
        for key, column in (('event_type', 'event_type'), ('direction','direction'),
                            ('source','source'), ('review_status','review_status_lower')):
            value = filters.get(key)
            if value and value != 'all':
                where.append(column+'=?')
                params.append(value)
        if filters.get('track_id'):
            where.append('instr(track_id_lower,?)>0')
            params.append(filters['track_id'].lower())
        if filters.get('has_images') is not None:
            where.append('has_images=?')
            params.append(int(bool(filters['has_images'])))
        if filters.get('query'):
            where.append('instr(people_query_text,?)>0')
            params.append(filters['query'])
        return ' WHERE ' + ' AND '.join(where), params

    def search_people(self, filters, *, limit=None, offset=0):
        if self.kind != 'people':
            raise ValueError('Nur für Personenhistorien')
        where, params = self._people_where(filters)
        # People history historically sorts by the original ISO timestamp text.
        order = ('ASC' if filters.get('sort') == 'asc' else 'DESC')
        query = 'SELECT payload FROM events' + where + f' ORDER BY timestamp {order},rank DESC'
        with self._mutex, closing(self._connect(read_only=True)) as conn:
            conn.execute('BEGIN')
            total = conn.execute('SELECT COUNT(*) FROM events' + where, params).fetchone()[0]
            if limit is not None:
                query += ' LIMIT ? OFFSET ?'
                params = [*params, int(limit), int(offset)]
            return self._read_rows(conn, query, params), total

    def iter_people(self, filters, *, chronological=False, batch_size=256):
        """Stream JSON events in bounded chunks for session aggregation.

        Uses one read transaction so events arriving during the scan do not
        change the result. Chronological order matches reversed(search()).
        """
        if self.kind != 'people':
            raise ValueError('Nur für Personenhistorien')
        where, params = self._people_where(filters)
        direction = ('DESC' if filters.get('sort') == 'asc' else 'ASC') if chronological else (
            'ASC' if filters.get('sort') == 'asc' else 'DESC')
        with closing(self._connect(read_only=True)) as conn:
            conn.execute('BEGIN')
            cursor = conn.execute('SELECT payload FROM events' + where +
                                  f' ORDER BY timestamp {direction}, rank {direction}', params)
            while True:
                chunk = cursor.fetchmany(max(1, min(1024, int(batch_size))))
                if not chunk:
                    break
                for (payload,) in chunk:
                    yield json.loads(payload)

    def people_image_days(self, filters):
        """SQL-only gallery calendar; do not deserialize every image event."""
        if self.kind != 'people':
            raise ValueError('Nur für Personenhistorien')
        query_filters = dict(filters)
        query_filters['has_images'] = True
        where, params = self._people_where(query_filters)
        # json_extract is available on the SQLite builds already required by
        # people_analytics(). Null, missing and empty image paths are not counted.
        count_exprs = []
        for key in ('crop', 'annotated', 'full_frame'):
            path = '$.images.' + key
            count_exprs.append(
                "COALESCE(SUM(CASE WHEN json_extract(payload,'" + path +
                "') IS NOT NULL AND json_extract(payload,'" + path +
                "') NOT IN ('',0) THEN 1 ELSE 0 END),0)")
        sql = ("SELECT substr(timestamp_local,1,10),COUNT(*)," + ','.join(count_exprs) +
               " FROM events" + where +
               " GROUP BY substr(timestamp_local,1,10) ORDER BY 1 DESC")
        with closing(self._connect(read_only=True)) as conn:
            return [dict(date=day, count=count, crop_count=crop,
                         annotated_count=annotated, full_frame_count=frame)
                    for day, count, crop, annotated, frame in conn.execute(sql, params)]

    def people_analytics(self, filters):
        """Aggregate person events in SQLite; leave heuristic sessions separate."""
        if self.kind != 'people':
            raise ValueError('Nur für Personenhistorien')
        where, params = self._people_where(filters)
        with closing(self._connect(read_only=True)) as conn:
            conn.execute('BEGIN')
            basic = conn.execute("""SELECT COUNT(*),COALESCE(SUM(counted),0),
                    COALESCE(SUM(CASE WHEN counted=0 THEN 1 ELSE 0 END),0),
                    COALESCE(SUM(repeat_blocked),0),COALESCE(SUM(has_images),0),
                    COALESCE(AVG(COALESCE(confidence,0)),0)
                FROM events""" + where,params).fetchone()
            daily = conn.execute("""SELECT substr(timestamp_local,1,10) day,
                    COUNT(*) detections,COALESCE(SUM(counted),0) persons,
                    SUM(CASE WHEN event_type='line_crossing' THEN 1 ELSE 0 END) crossings,
                    SUM(CASE WHEN event_type='appearance' THEN 1 ELSE 0 END) appearances,
                    COALESCE(SUM(repeat_blocked),0) repeated,COALESCE(SUM(has_images),0) images
                FROM events"""+where+' GROUP BY day ORDER BY day',params).fetchall()
            directional = conn.execute("""SELECT substr(timestamp_local,1,10),COALESCE(NULLIF(direction,''),'unknown'),COUNT(*)
                FROM events"""+where+' GROUP BY 1,2',params).fetchall()
            hours = conn.execute("""SELECT replace(substr(timestamp_local,1,13),'T',' ')||':00',COUNT(*) FROM events
                """ + where + ' AND counted=1 GROUP BY 1 ORDER BY 1 DESC',params).fetchall()
            def counter(column, default='unknown'):
                expr = 'COALESCE(NULLIF('+column+",''),?)"
                return dict(conn.execute('SELECT '+expr+',COUNT(*) FROM events'+where+
                                         ' GROUP BY 1', [default,*params]).fetchall())
            directions=counter('direction')
            types=counter('event_type')
            sources=counter('source')
            tracks = conn.execute("""SELECT COUNT(DISTINCT CASE
                   WHEN json_type(payload,'$.track_id') IS NOT NULL AND
                        json_type(payload,'$.track_id')!='null'
                   THEN COALESCE(NULLIF(source,''),'unknown')||':'||
                        CAST(json_extract(payload,'$.track_id') AS TEXT)
                   ELSE NULL END) FROM events"""+where,params).fetchone()[0]
            latest = self._read_rows(conn,'SELECT payload FROM events'+where+
                                     ' ORDER BY timestamp DESC,rank DESC LIMIT 100',params)
        direction_days = {}
        for day, direction, count in directional:
            direction_days.setdefault(day,{})[direction]=count
        daily_rows = [dict(date=day,persons=persons,detections=detections,
                           line_crossings=crossings,appearances=appearances,
                           repeat_blocked=repeated,images=images,
                           directions=direction_days.get(day,{}))
                      for day,detections,persons,crossings,appearances,repeated,images in daily]
        return {'counts': basic, 'daily': daily_rows, 'hourly': hours,
                'directions': directions, 'event_types': types, 'sources': sources,
                'unique_tracks': tracks, 'latest': latest}

    def people_since(self, cutoff):
        if self.kind != 'people':
            raise ValueError('Nur für Personenhistorien')
        with closing(self._connect(read_only=True)) as conn:
            return self._read_rows(conn,
                'SELECT payload FROM events WHERE timestamp_local>=? ORDER BY rank DESC',
                (cutoff,))

    def quick_status(self):
        try:
            with closing(self._connect(read_only=True)) as conn:
                meta = self._validate(conn)
                total = conn.execute('SELECT COUNT(*) FROM events').fetchone()[0]
            return {'kind': self.kind, 'backend': 'sqlite', 'ok': True, 'count': total,
                    'size_mb': round(self.path.stat().st_size / 1048576, 2),
                    'migrated_at': meta.get('migrated_at'),
                    'repaired_ids': int(meta.get('repaired_legacy_ids', '0')),
                    'legacy_snapshot_exists': self.snapshot.is_file(), 'query_schema': SCHEMA_VERSION, 'error': None}
        except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
            return {'kind': self.kind, 'backend': 'sqlite', 'ok': False, 'count': None,
                    'size_mb': None, 'legacy_snapshot_exists': self.snapshot.is_file(),
                    'error': str(exc)[:180]}

    def backup_to(self, destination):
        """Produce a transactionally consistent standalone SQLite snapshot."""
        destination = Path(destination)
        if destination.exists():
            raise HistoryStorageError('Backup-Zieldatei existiert bereits')
        source = self._connect(read_only=True)
        target = sqlite3.connect(str(destination), timeout=10)
        try:
            self._validate(source)
            source.backup(target)
            if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise HistoryStorageError('Erzeugter SQLite-Snapshot ist beschädigt')
            target.commit()
        except Exception:
            target.close()
            destination.unlink(missing_ok=True)
            raise
        else:
            target.close()
        finally:
            source.close()
        return destination

    @staticmethod
    def read_snapshot(path):
        with sqlite3.connect(str(path)) as conn:
            if conn.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise HistoryStorageError('SQLite-Snapshot ist beschädigt')
            return SQLiteHistoryStore._load_from_conn(conn)
