#!/usr/bin/env python3
"""PlateVision 0.12.7: SQL search, pagination, filters, aggregates and safe v1 upgrade.

Tests use the real old SQLite 0.12.6 schema and AST-extracted managers; no Flask/ML
packages, real camera, or a live Home Assistant installation are needed.
"""
import ast
import copy
import json
import hashlib
import logging
import os
import re
import sqlite3
import sys
import tempfile
import threading
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'platevision/src'))
from history_sqlite import SQLiteHistoryStore, HistoryStorageError, SCHEMA_VERSION


def extract(names, env):
    tree = ast.parse((ROOT/'platevision/src/app.py').read_text(encoding='utf-8'))
    nodes=[]
    for name in names:
        node=next((n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name), None)
        assert node is not None,name
        node=copy.deepcopy(node)
        if isinstance(node, ast.FunctionDef):
            node.decorator_list=[]
        nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(ROOT/'platevision/src/app.py'),'exec'),env)


class Config:
    data = {
        'search': {'default_limit': 50, 'enable_fuzzy_search':True, 'fuzzy_similarity':.72},
        'history': {'fuzzy_duplicate_detection': False, 'fuzzy_duplicate_similarity': .88},
        'dashboard': {'default_range_days':7},
        'people': {'present_timeout_minutes': 10, 'session_gap_minutes': 5},
        'traffic': {'min_confidence': .0, 'ignore_unknown_plates': True,
                    'include_duplicate_events': False, 'visit_gap_minutes': 15}
    }
    def get(self, *keys):
        data = self.data
        for key in keys:
            data = data.get(key) if isinstance(data,dict) else None
        return data


def make_managers():
    env=dict(re=re,sqlite3=sqlite3,logger=logging.getLogger('v127-tests'),
             datetime=datetime,timedelta=timedelta,threading=threading,uuid=uuid,
             os=os,Path=Path,hashlib=hashlib,contextmanager=contextmanager,SequenceMatcher=SequenceMatcher,
             Counter=Counter,defaultdict=defaultdict,config_manager=Config(),
             _page_integer=lambda v,d,minimum=0,maximum=200: max(minimum,min(int(d if v in (None,'') else v),maximum)),
             _safe_numeric=lambda v,d=0.0: float(v) if v not in (None,'') and str(v).replace('.','',1).isdigit() else d,
             SQLiteHistoryStore=SQLiteHistoryStore)
    extract(('PlateUtils','HistoryManager','PersonHistoryManager'),env)
    return env,env['HistoryManager'],env['PersonHistoryManager']


def make_store(path,kind,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    store=SQLiteHistoryStore(path,kind=kind)
    assert store.initialize(lambda:json.loads(path.read_text())) == rows
    return store


def manager_for(cls,store,rows):
    manager=cls.__new__(cls)
    manager.lock=threading.RLock()
    manager._sqlite_store=store
    manager.history=rows
    if cls.__name__=='PersonHistoryManager':
        manager._people_config_local=threading.local()
    return manager


def plate_tests():
    env,HP,_=make_managers()
    now=datetime.now().replace(microsecond=0)
    day=now.date().isoformat()
    rows=[]
    for n in range(1200):
        ts=(now-timedelta(minutes=(n%280))).isoformat()
        if n%17==0: ts=(now-timedelta(minutes=n%280)).astimezone().isoformat()
        rows.append({'id':f'p-{n}', 'plate_text': ('ZH 123' if n%19==0 else
            'BE 42' if n%11==0 else f'AG {100+n%370}'),
            'timestamp':ts, 'confidence':('bad' if n%71==0 else round((n%101)/100,2)),
            'source': ('rtsp_stream' if n%3 else 'rtsp'),
            'vehicle_type': ('Auto' if n%2 else 'LKW'),
            'vehicle_type_en': ('car' if n%2 else 'truck'),
            'vehicle_color': ('blau' if n%2 else 'rot'),
            'plate_format': ('CH' if n%5 else 'Generic'),
            'is_valid_plate': (n%7!=0),
            'watchlist_match': ({'plate':'ZH 123'} if n%13==0 else None)})
    rows.append({'id':'no-date','plate_text':'ZÜ 25','timestamp':'n/a','confidence':'nan'})
    store=make_store(Path('data/history/detections.json'),'plates',rows)
    man=manager_for(HP,store,rows)
    cases=[{}, {'q':'ZH 12'},{'q':'ZH 123','match_mode':'exact'},
        {'q':'ZH','match_mode':'starts_with'},
        {'q':'AG 34','fuzzy':False},
        {'q':'AC 344','fuzzy':True,'fuzzy_similarity':.85},
        {'source':'rtsp'}, {'vehicle_type':'car'}, {'vehicle_type':'auto'},
        {'vehicle_color':'BLAU'}, {'plate_format':'CH'},
        {'valid_only':True,'watchlist_only':True},
        {'min_confidence':.65,'max_confidence':.88},
        {'date_from':day,'date_to':day},
        {'date_from':day,'date_to':day, 'sort':'confidence','order':'asc'},
        {'sort':'plate','order':'asc'}, {'unique':True,'fuzzy':False},
        {'q':"X' OR 1=1 --", 'fuzzy':False},
        {'source':'rtsp','q':'ZH 123', 'limit':7,'offset':12},
        {'order':'asc','limit':45,'offset':101}]
    for idx,case in enumerate(cases):
        query=dict(case)
        query.setdefault('limit',22)
        query.setdefault('offset',0)
        man._sqlite_store=None
        expected=man.search_advanced(query)
        man._sqlite_store=store
        actual=man.search_advanced(query)
        assert actual==expected, (idx,query,actual['total'],expected['total'],
                                  [x['id'] for x in actual['entries'][:10]],
                                  [x['id'] for x in expected['entries'][:10]])
    man._sqlite_store=None
    expected_facets=man.get_facets()
    man._sqlite_store=store
    assert man.get_facets()==expected_facets
    # Whole-dashboard aggregations must match the reference code.
    man.get_traffic_statistics=lambda *_: {'summary':{'total_visits':0}}
    man._sqlite_store=None
    expected_stats=man.get_statistics()
    man._sqlite_store=store
    sql_stats=man.get_statistics()
    for k in expected_stats:
        if k=='average_confidence':
            assert abs(sql_stats[k]-expected_stats[k])<.000001,(k,sql_stats[k],expected_stats[k])
        else:
            assert sql_stats[k] == expected_stats[k], (k,sql_stats[k],expected_stats[k])
    print('  OK: 20 plate query variants, fuzzy, sorting, source aliases, unique, facets, dashboard SQL aggregates')
    # Verify SQLite reads never call full-table HistoryStore.load().
    with patch.object(store,'load',side_effect=AssertionError('unexpected full SQLite load')):
        assert len(man.search_advanced({'limit':9})['entries'])==9
        assert len(list(man.iter_search({'source':'rtsp','fuzzy':False})))>0
    assert len(list(man.iter_search({'fuzzy':False})))==len(rows)
    assert len(list(man.iter_search({'q':'ZH 123', 'fuzzy':False})))==man.search_advanced({'q':'ZH 123','fuzzy':False})['total']
    assert store.search_plates({'q':'ZH123','match_mode':'exact'})[1] > 0
    with sqlite3.connect(store.path) as conn:
        plan = conn.execute("EXPLAIN QUERY PLAN SELECT payload FROM events WHERE plate_normalized=?",('ZH123',)).fetchall()
        assert any('idx_events_normalized_rank' in str(row) for row in plan),plan
    # SQL helper outputs must equal previous Counter-based API outputs.
    from history_sqlite import _normalize_plate
    q='ZH123'
    counts=Counter()
    for row in rows:
        norm=_normalize_plate(row.get('plate_text'))
        if norm and (q in norm or SequenceMatcher(None,q,norm).ratio()>=.65):
            counts[row['plate_text']]+=1
    assert store.autocomplete(q,12)==[{'plate_text':plate,'count':count} for plate,count in counts.most_common(12)]
    for bucket in ('hour','day'):
        counts=Counter()
        for row in rows:
            parsed=man._local_datetime(man._parse_datetime(row.get('timestamp')))
            if parsed:
                key=parsed.strftime('%Y-%m-%d %H:00') if bucket=='hour' else parsed.date().isoformat()
                counts[key]+=1
        assert store.timeline(bucket=bucket,limit=30)==[{'time':k,'count':v} for k,v in sorted(counts.items())[-30:]]
    # Index state must track both updates and appends as well as deletions.
    patch_row=dict(rows[0],source='manual_upload',plate_text='TI 9999',vehicle_color='gruen')
    amended=[patch_row,*rows[1:]]
    store.sync(amended)
    assert store.search_plates({'q':'TI9999','match_mode':'exact'})[1]==1
    assert store.search_plates({'source':'manual_upload'})[1]==1
    store.prepend({'id':'newest','plate_text':'TI 9999','timestamp':datetime.now().isoformat()},removed_ids=['no-date'])
    assert store.search_plates({'q':'TI9999','match_mode':'exact'})[1]==2
    assert store.get('no-date') is None
    # Keep the later legacy-comparison tests on the original fixture.
    store.sync(rows)
    print('  OK: SQL autocomplete/timeline, indexed mutations and streamed exports >1000 rows')
    # SQL traffic preselection must equal original Python filtering.
    traffic_cases=({'days':1},{'days':7,'min_confidence':.7},
                   {'days':7,'ignore_unknown_plates':'false'})
    for q in traffic_cases:
        man._sqlite_store=None
        exp=man._traffic_entries(q)[0]
        man._sqlite_store=store
        got=man._traffic_entries(q)[0]
        assert got==exp,(q,len(got),len(exp))
    man._sqlite_store=None
    exp=man.get_plate_profile('ZH 123',{'days':7})
    man._sqlite_store=store
    got=man.get_plate_profile('ZH 123',{'days':7})
    assert got==exp
    print('  OK: SQL traffic time-window selection and indexed per-plate profile unchanged')


def people_tests():
    env,_,PP=make_managers()
    now=datetime.now().replace(microsecond=0)
    rows=[]
    for i in range(1700):
        ts=(now-timedelta(hours=i%48)).isoformat()
        rows.append({'id':f'person-{i}', 'timestamp':ts,
            'confidence': ('bad' if i%13==0 else i%101/100),
            'counted':i%3!=0, 'source':('cam1' if i%2 else 'cam2'),
            'event_type':('line_crossing' if i%5 else 'appearance'),
            'direction':('enter' if i%2 else 'exit'),
            'track_id':i%19, 'review_status':('confirmed' if i%7==0 else 'unreviewed'),
            'label':'Visitor' if i%4==0 else '',
            'images':{'crop':f'person-{i}.jpg'} if i%6==0 else {}})
    rows.append({'id':'invalid-time','timestamp':'corrupt','counted':True})
    store=make_store(Path('data/people/history.json'),'people',rows)
    man=manager_for(PP,store,rows)
    cases=[{}, {'days':1},{'days':2,'counted':'all'},
           {'days':2,'counted':'false'}, {'days':2,'event_type':'line_crossing'},
           {'days':2,'direction':'exit'}, {'days':2,'source':'cam1'},
           {'days':2,'track_id':'1'},{'days':2,'has_images':'true'},
           {'days':2,'has_images':'false'}, {'days':2,'review_status':'confirmed'},
           {'days':2,'query':'visitor'}, {'days':2,'min_confidence':.7},
           {'date_from':(now-timedelta(days=1)).date().isoformat(),
            'date_to':now.date().isoformat(), 'sort':'asc'}]
    for idx, filters in enumerate(cases):
        man._sqlite_store=None
        expected=man.search(filters)
        man._sqlite_store=store
        actual=man.search(filters)
        assert actual==expected,(idx,filters,len(actual),len(expected),
                                  [r['id'] for r in actual[:4]],[r['id'] for r in expected[:4]])
        result=man.paginated_history({**filters,'limit':17,'offset':19})
        assert result['total']==len(expected)
        assert [row['id'] for row in result['entries']] == [row['id'] for row in expected[19:36]]
    # Aggregated people statistics must preserve the existing response structure.
    man._sqlite_store=None
    expected_stats=man.get_statistics({'days':2})
    man._sqlite_store=store
    sql_stats=man.get_statistics({'days':2})
    for key in expected_stats:
        assert expected_stats[key] == sql_stats[key], (key,expected_stats[key],sql_stats[key])
    print('  OK: People SQL day/hour aggregates, totals, event types and track counts match the legacy statistics')
    print('  OK: Person search, counted, track, images, date, review, filters and SQL paging')
    man._sqlite_store=None
    old=man.get_presence()
    man._sqlite_store=store
    new=man.get_presence()
    assert old==new,(old,new)
    print('  OK: Recent-person presence SQL selection preserves public result')


def old_schema_upgrade():
    """Upgrade a real v1 file in one transaction; assert no JSON bytes change."""
    path=Path('upgrade/detections.json')
    rows=[{'id':'A','plate_text':'ZH 124', 'confidence':.6,
           'timestamp':'2026-10-09T10:00:00+02:00'},
          {'id':'B','plate_text':'BE 52','timestamp':'bad'}]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(rows),encoding='utf-8')
    baseline=path.read_bytes()
    p=path.parent/'plates.sqlite3'
    from history_sqlite import _sha256
    with sqlite3.connect(p) as conn:
        conn.executescript('''CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE events(event_key TEXT PRIMARY KEY,rank INTEGER NOT NULL UNIQUE,
           timestamp TEXT,plate_text TEXT,plate_normalized TEXT,source TEXT,event_type TEXT,
           confidence REAL,payload TEXT NOT NULL);''')
        conn.executemany('INSERT INTO metadata VALUES(?,?)',[
            ('kind','plates'),('schema_version','1'),('revision','1'),
            ('source_sha256',_sha256(path)),('migrated_at','2026-10-09T00:00:00Z')])
        for rank,row in zip((2,1),rows):
            conn.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)',
                (row['id'],rank,row.get('timestamp'),row['plate_text'],'',None,None,row.get('confidence'),json.dumps(row)))
        conn.commit()
    store=SQLiteHistoryStore(path,kind='plates')
    assert store.initialize(lambda: (_ for _ in ()).throw(AssertionError('do not load legacy'))) == rows
    assert path.read_bytes()==baseline
    assert store.quick_status()['query_schema']==2
    assert store.search_plates({'q':'ZH124'})[1]==1
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]=='2'
        assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert conn.execute('SELECT plate_normalized FROM events WHERE event_key=?',('A',)).fetchone()[0]=='ZH124'
        assert 'idx_events_local_time' in {r[1] for r in conn.execute('PRAGMA index_list(events)')}
    print('  OK: Atomic SQLite v1->v2 upgrade, original JSON and event payloads preserved')
    # A damaged payload must fail before any schema version change.
    bad=Path('invalid/history.json')
    bad.parent.mkdir(parents=True)
    bad.write_text('[]')
    with sqlite3.connect(bad.parent/'people.sqlite3') as conn:
        conn.executescript('''CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE events(event_key TEXT PRIMARY KEY,rank INTEGER NOT NULL UNIQUE,
           timestamp TEXT,plate_text TEXT,plate_normalized TEXT,source TEXT,event_type TEXT,
           confidence REAL,payload TEXT NOT NULL);''')
        conn.executemany('INSERT INTO metadata VALUES(?,?)',[
           ('kind','people'),('schema_version','1'),('revision','1'),('source_sha256',_sha256(bad))])
        conn.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)',('x',1,None,None,None,None,None,None,'{invalid'))
        conn.commit()
    invalid=SQLiteHistoryStore(bad,kind='people')
    try:
        invalid.initialize(lambda:[])
        raise AssertionError('Damaged JSON payload must refuse schema upgrade')
    except (json.JSONDecodeError,HistoryStorageError):
        pass
    with sqlite3.connect(invalid.path) as conn:
        assert conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]=='1'
        assert 'timestamp_local' not in {r[1] for r in conn.execute('PRAGMA table_info(events)')}
    print('  OK: Damaged SQLite v1 payload rolls back the schema upgrade')


def main():
    with tempfile.TemporaryDirectory(prefix='pv-sql-127-') as name:
        cwd=Path.cwd()
        try:
            os.chdir(name)
            old_schema_upgrade()
            plate_tests()
            people_tests()
        finally:
            os.chdir(cwd)
    print('PlateVision 0.12.7 indexed SQL regression tests: all cases passed')

if __name__=='__main__':
    main()
