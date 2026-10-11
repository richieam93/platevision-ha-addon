#!/usr/bin/env python3
"""Optional real Chromium smoke/security tests for PlateVision UI.

Requires Playwright + chromium but neither is needed by the production Add-on.
Uses actual Jinja templates and mocked API responses, not a static mock UI.
"""
import ast
import copy
import json
import os
import re
import shutil
from pathlib import Path
from jinja2 import ChainableUndefined, Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / 'platevision/src/templates'
STATIC = ROOT / 'platevision/src/static'
source = ast.parse((ROOT / 'platevision/src/app.py').read_text(encoding='utf-8'))
cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'ConfigManager')
config = ast.literal_eval(next(n.value for n in cls.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_CONFIG' for t in n.targets)))
env = Environment(loader=FileSystemLoader(str(TEMPLATES)), undefined=ChainableUndefined, autoescape=True)
malicious = '<img src=x onerror="window.__pwned=1">'

def render(name, payloads):
    ctx = dict(config=config, app_config=config, lang='de', page=name,
               app_version='0.12.7', app_license='AGPL-3.0-only',
               people_menu_enabled=True, stream_status={'status':'stopped'},
               entries=[], stats={}, t=lambda key: key,
               pv_prefix='', pv_url_for=lambda endpoint,**kwargs:'/static/'+kwargs['filename'] if endpoint == 'static' else '/'+endpoint)
    html = env.get_template(name+'.html').render(ctx)
    html = re.sub(r'<link\s+[^>]*?href="/static/([^\"]+\.css)"[^>]*>',
                  lambda m: '<style>' + (STATIC / m.group(1)).read_text() + '</style>', html)
    html = re.sub(r'<script\s+src="/static/([^\"]+\.js)"[^>]*></script>',
                  lambda m: '<script>' + (STATIC / m.group(1)).read_text() + '</script>', html)
    html = re.sub(r'<link\s+[^>]*?href="https://[^>]+>', '', html)
    html = re.sub(r'<script\s+src="https://[^>]+></script>', '', html)
    mock = '''<script>
window.io = () => ({on(){}});
window.Chart = class {constructor(){} destroy(){} };
window.__pwned = 0;
const __payloads = __PAYLOADS__;
window.fetch = async (input, options) => ({
  ok: true,
  status: 200,
  json: async () => __payloads[String(input).split('?')[0]] || {success:true,status:{status:'stopped'},checks:[],issues:[],suggestions:[]}
});
</script>'''.replace('__PAYLOADS__',json.dumps(payloads,ensure_ascii=False).replace('</', '<\\/'))
    return html.replace('<head>', '<head>' + mock)

payloads = {
    '/api/stream/status': {'status':'stopped','connected': False},
    '/api/settings/schema': {'config': config, 'schema':{}},
    '/api/history/search': {'total':2, 'limit':50, 'offset':0, 'entries':[
        {'id':'malicious-legacy-id', 'plate_text': malicious, 'plate_format': 'CH',
         'vehicle_type': malicious, 'vehicle_color': malicious, 'source': malicious,
         'timestamp': '2026-10-08T16:00:00', 'confidence': 'NaN', 'is_valid_plate': False,
         'has_plate_image': False, 'watchlist_match': {'label':malicious}},
        {'id':'d3b073b1-d79b-418e-92f8-0872c856d193', 'plate_text':'ZH 1234',
         'timestamp':'2026-10-09T12:30:00', 'confidence': 0.9, 'is_valid_plate': True,
         'has_plate_image': True}]},
    '/api/history/facets': {'sources':[{'value':malicious,'count':3}],
                            'vehicle_types':[{'value': malicious,'count':1}],
                            'vehicle_colors':[{'value': malicious,'count':1}]},
    '/api/history/autocomplete': {'items':[{'plate_text':malicious,'count':1}]},
    '/api/watchlist': {'items':[{'id':'row-1','plate_text':malicious,'label':malicious}]},
    '/api/plate/batch-analyze': {'rows':[{'input':malicious,'best':{'pretty':malicious,'score':'NaN','format':malicious}}]},
    '/api/models/people/options': {'models':[{'name':malicious,'kind_guess':'people',
                                              'path':malicious, 'size_mb': 12,'exists':True}], 'status':{}},
    '/api/models/available': {'models':[]},
    '/api/system/health': {'checks':[{'name':malicious,'detail':malicious,'ok':False}]},
    '/api/system/audit': {'issues':[{'title':malicious,'detail':malicious,'level':'warning'}], 'suggestions':[malicious]},
    '/api/system/backup/preview': {'success':True,'file_count':32, 'size_mb':4.52,'max_mb':512},
    '/api/system/backup/verify': {'success':True,'verified_files':32,'verified_bytes':1048576,'source_version':'0.12.7'},
    '/api/people/images/days': {'days':[{'date': "2026-10-09');window.__pwned=1;//",'count':malicious,'crop_count':malicious},
                                           {'date':'2026-10-09','count':1,'crop_count':1}],
                                 'total_events_with_images':1,'total_days':1, 'retention_days':10,
                                 'auto_cleanup_enabled':True,'newest_date':'2026-10-09'},
    '/api/people/images/history': {'entries':[{'id':"x');window.__pwned=1;//",'timestamp':'2026-10-09T12:00:00',
               'track_id':malicious,'direction':malicious,'confidence':0.8,
               'image_urls':{'crop':'javascript:window.__pwned=1','annotated':'/api/people/images/crops/2026-10-09/test.jpg'}}],
              'total':1,'limit':48,'offset':0,'has_more':False},
    '/api/system/storage-audit': {'disk':{'free_mb':120.5},'files':[{'category':malicious,'valid':False,'backup_valid':True,'size_mb': 0.8}],
                                  'issues':[malicious], 'video_jobs':{'active':1,'completed':2}, 'authentication_enabled':False,
                                  'sqlite_histories': {'plates':{'ok':True,'count':30000,'legacy_snapshot_exists':True},
                                                       'people':{'ok':True,'count':2500,'legacy_snapshot_exists':True}}},
}

with sync_playwright() as pw:
    binary = os.environ.get('PV_CHROMIUM_PATH') or shutil.which('chromium') or '/usr/bin/chromium'
    browser = pw.chromium.launch(executable_path=binary, headless=True, args=['--no-sandbox','--disable-dev-shm-usage'])
    try:
        search = browser.new_page(viewport={'width':1440,'height':900})
        search_errors = []
        search.on('pageerror', lambda exc: search_errors.append(str(exc)))
        search.set_content(render('search', payloads), wait_until='domcontentloaded')
        search.wait_for_function("document.querySelector('#resultTotal').textContent === '2'")
        assert search.locator('#resultsBody tr').count() == 2
        assert malicious in search.locator('#resultsBody tr').first.inner_text()
        assert search.locator('#resultsBody tr:nth-child(2) img').count() <= 1
        assert search.locator('#resultsBody tr:first-child img').count() == 0
        assert search.locator('#watchlist img').count() == 0
        assert search.locator('#source img').count() == 0
        assert search.evaluate('window.__pwned') == 0
        # Exercise the new SQL-indexed exact/prefix search selectors and prove
        # their requested modes reach the same backend endpoint.
        search.evaluate('''() => {
            window.__pvModeRequests = [];
            const originalFetch = window.fetch;
            window.fetch = (input, options) => {
                if (String(input) === '/api/history/search' && options && options.body) {
                    window.__pvModeRequests.push(JSON.parse(options.body));
                }
                return originalFetch(input, options);
            };
        }''')
        search.locator('#matchMode').select_option('exact')
        search.wait_for_function('window.__pvModeRequests.some(request => request.match_mode === "exact")')
        search.locator('#matchMode').select_option('starts_with')
        search.wait_for_function('window.__pvModeRequests.some(request => request.match_mode === "starts_with")')
        search.locator('#matchMode').select_option('contains')
        search.wait_for_function('window.__pvModeRequests.some(request => request.match_mode === "contains")')
        assert search_errors == [], 'Suchmodus JavaScript errors: ' + str(search_errors[:5])
        search.locator('#batchInput').fill(malicious)
        search.locator('button[onclick="batchAnalyze()"] ').click()
        search.wait_for_function('document.querySelector("#batchResult").childElementCount === 1')
        assert search.locator('#batchResult img').count() == 0
        assert search.evaluate('window.__pwned') == 0
        search.screenshot(path=str(ROOT/'PlateVision_0.12.7_Suche_Desktop.png'), full_page=True)
        for width in (320, 390, 768, 1024):
            mobile = browser.new_page(viewport={'width':width,'height':844})
            mobile.set_content(render('search',payloads), wait_until='domcontentloaded')
            mobile.wait_for_function("document.querySelector('#resultTotal').textContent === '2'")
            if not mobile.evaluate('document.documentElement.scrollWidth <= innerWidth + 3'):
                offenders=mobile.evaluate("""() => ({scrollWidth: document.documentElement.scrollWidth,innerWidth: innerWidth, offenders: [...document.querySelectorAll('body *')].filter(el => {const r=el.getBoundingClientRect();return r.width>0 && r.right>innerWidth+3 && getComputedStyle(el).position !== 'fixed'}).slice(0,100).map(el=>({tag:el.tagName,id:el.id,cl:String(el.className).slice(0,60),right:Math.round(el.getBoundingClientRect().right),width:Math.round(el.getBoundingClientRect().width)}))})""")
                ancestor=mobile.evaluate("""() => {let e=document.getElementById('resultsBody'); const result=[];while(e && result.length<12){const r=e.getBoundingClientRect(),s=getComputedStyle(e);result.push({tag:e.tagName,cl:String(e.className).slice(0,55),width:r.width,right:r.right,scroll:e.scrollWidth,client:e.clientWidth,overflowX:s.overflowX,minWidth:s.minWidth,maxWidth:s.maxWidth});e=e.parentElement}return result}""")
                other = [x for x in offenders['offenders'] if x['tag'] not in ['TD','TR','TABLE','TH','THEAD','TBODY','SPAN','DIV'] or x['width'] < 280]
                raise AssertionError(f'Search overflow at {width}: {other}\n{ancestor}')
            if width == 390:
                mobile.screenshot(path=str(ROOT/'PlateVision_0.12.7_Suche_Mobil.png'), full_page=True)
            mobile.close()
        print('  OK: Suche / Watchlist / Facetten / Batch XSS und Mobile-Layout')
        assert not search_errors, 'Search JavaScript errors: ' + str(search_errors[:6])

        settings = browser.new_page(viewport={'width':1360, 'height':860})
        settings_errors=[]
        settings.on('pageerror', lambda exc: settings_errors.append(str(exc)))
        settings.set_content(render('settings',payloads), wait_until='domcontentloaded')
        settings.evaluate('''() => renderModelInventory([{name:'<img src=x onerror="window.__pwned=1">', path:"' onclick=window.__pwned=2", kind_guess:'people', size_mb:3, exists:true}])''')
        assert settings.locator('#modelUploadInventory tbody tr').count() == 1
        assert settings.locator('#modelUploadInventory img').count() == 0
        assert settings.locator('#modelUploadInventory button').count() == 1
        assert settings.evaluate('window.__pwned') == 0
        settings.evaluate('loadPeopleModelOptions()')
        settings.wait_for_function('document.querySelector("#peopleSelectedModelFile option")?.value.includes("<img")')
        assert settings.locator('#peopleSelectedModelFile img').count() == 0
        assert settings.evaluate('window.__pwned') == 0
        print('  OK: Modellverwaltung: manipulierte Modellnamen/Pfade sicher im DOM')

        diag = browser.new_page(viewport={'width':1320, 'height':900})
        diag_errors=[]
        diag.on('pageerror', lambda exc: diag_errors.append(str(exc)))
        diag.set_content(render('diagnostics',payloads), wait_until='domcontentloaded')
        diag.wait_for_function('document.querySelector("#storageFiles tr:first-child td")?.textContent.includes("<img")')
        assert diag.locator('#storageFiles img').count() == 0
        assert diag.locator('#storageAlerts img').count() == 0
        assert diag.evaluate('window.__pwned') == 0
        assert '120.5 MB' in diag.locator('#storageFree').inner_text()
        assert '30000' in diag.locator('#sqliteStatus').inner_text()
        assert '2500' in diag.locator('#sqliteStatus').inner_text()
        diag.wait_for_function('document.querySelector("#backupPreview").textContent.includes("32 Dateien")')
        assert diag.locator('a[href="/api/system/backup"][download]').count() == 1
        diag.locator('#backupVerifyFile').set_input_files({'name': 'sample.zip', 'mimeType': 'application/zip', 'buffer': b'fake-test-data'})
        diag.locator('#backupVerifyButton').click()
        diag.wait_for_function('document.querySelector("#backupVerifyResult").textContent.includes("Prüfung bestanden")')
        assert '32 Dateien' in diag.locator('#backupVerifyResult').text_content()
        assert not diag_errors, str(diag_errors)
        diag.screenshot(path=str(ROOT/'PlateVision_0.12.7_Diagnose.png'), full_page=True)
        mobile_diag = browser.new_page(viewport={'width': 390, 'height': 844})
        mobile_diag.set_content(render('diagnostics',payloads),wait_until='domcontentloaded')
        mobile_diag.wait_for_function('document.querySelector("#backupPreview").textContent.includes("32 Dateien")')
        assert mobile_diag.evaluate('document.documentElement.scrollWidth <= innerWidth + 3'), 'Diagnose hat Mobile Overflow'
        assert mobile_diag.locator('#backupVerifyButton').count() == 1
        mobile_diag.screenshot(path=str(ROOT/'PlateVision_0.12.7_Diagnose_Mobil.png'), full_page=True)
        mobile_diag.close()
        print('  OK: Speicher-Diagnose mit manipulierten Daten ohne XSS')
        gallery = browser.new_page(viewport={'width':1100, 'height':800})
        gallery_errors=[]
        gallery.on('pageerror', lambda exc: gallery_errors.append(str(exc)))
        gallery.set_content(render('people_gallery',payloads), wait_until='domcontentloaded')
        gallery.wait_for_function('document.querySelector("#archiveGrid [data-delete-event-id]") !== null')
        assert gallery.locator('#archiveDays img').count() == 0
        assert gallery.locator('#archiveGrid .archive-card img').count() == 1
        assert gallery.locator('a[href^="javascript:"]').count() == 0
        assert gallery.locator('#archiveDays button[onclick]').count() == 0
        assert gallery.evaluate('window.__pwned') == 0
        gallery.locator('#archiveDays button').first.click()
        assert gallery.evaluate('window.__pwned') == 0
        assert not gallery_errors, str(gallery_errors[:5])
        print('  OK: Personenarchiv mit boesartigen Datums-, ID- und Bildlink-Daten sicher')

        live = browser.new_page(viewport={'width':1280,'height':800})
        live_errors=[]
        live.on('pageerror', lambda exc: live_errors.append(str(exc)))
        live.set_content(render('live',payloads), wait_until='domcontentloaded')
        live.evaluate('''(bad)=>addToRecentList({plate_text:bad,plate_image:'\" onerror=window.__pwned=1',confidence:0.7})''', malicious)
        assert live.locator('#recentDetectionsList .detection-card').count() == 1
        assert live.locator('#recentDetectionsList img').count() == 0
        assert malicious in live.locator('#recentDetectionsList').inner_text()
        assert live.evaluate('window.__pwned') == 0
        assert not live_errors, str(live_errors[:5])
        print('  OK: Live-Stream Ereignisliste weist HTML-/Bildattribut-Injektion ab')

        # Save a clean, reproducible visual example with synthetic vehicle rows.
        demo = copy.deepcopy(payloads)
        demo['/api/history/search']['entries'] = [
            {'id':'5904cc8d-6caf-48de-af7d-114e6a47d715','plate_text':'ZH 45837','plate_format':'CH',
             'vehicle_type':'Personenwagen','vehicle_color':'Blau','source':'RTSP',
             'timestamp':'2026-10-09T12:05:00','confidence':.92,'is_valid_plate':True,
             'has_plate_image':False},
            {'id':'4094a5af-c277-4115-9f4c-b8f5f2e0e962','plate_text':'BE 1263','plate_format':'CH',
             'vehicle_type':'Transporter','vehicle_color':'Weiss','source':'Video',
             'timestamp':'2026-10-09T11:30:00','confidence':.87,'is_valid_plate':True,
             'has_plate_image':False}]
        demo['/api/history/facets'] = {'sources':[{'value':'RTSP','count':1},{'value':'Video','count':1}],
                                       'vehicle_types':[{'value':'Personenwagen','count':1},{'value':'Transporter','count':1}],
                                       'vehicle_colors':[{'value':'Blau','count':1},{'value':'Weiss','count':1}]}
        demo['/api/watchlist'] = {'items':[{'id':'watch-1','plate_text':'ZH 45837','label':'Einfahrt Nord'}]}
        for width, height, filename in [
            (1440,900,'PlateVision_0.12.7_Suche_Desktop.png'),
            (390,844,'PlateVision_0.12.7_Suche_Mobil.png')]:
            preview = browser.new_page(viewport={'width':width,'height':height})
            preview.set_content(render('search',demo),wait_until='domcontentloaded')
            preview.wait_for_function('document.querySelector("#resultTotal").textContent === "2"')
            preview.wait_for_timeout(180)
            preview.screenshot(path=str(ROOT/filename),full_page=True)
            preview.close()
        print('  OK: Layout-Vorschauen mit simulierten, harmlosen Daten erstellt')
        print('Browser resilience checks: OK')
    finally:
        browser.close()
