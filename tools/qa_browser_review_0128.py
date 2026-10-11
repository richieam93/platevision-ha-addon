#!/usr/bin/env python3
"""Optional Chromium UI checks for 0.12.8. Uses rendered Jinja + mock APIs."""
import ast
import json
import re
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
from jinja2 import ChainableUndefined, Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'platevision/src/templates'
STATIC = ROOT / 'platevision/src/static'
ast_root = ast.parse((ROOT/'platevision/src/app.py').read_text(encoding='utf-8'))
config_cls = next(n for n in ast_root.body if isinstance(n,ast.ClassDef) and n.name=='ConfigManager')
config = ast.literal_eval(next(n.value for n in config_cls.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_CONFIG' for t in n.targets)))
env=Environment(loader=FileSystemLoader(TEMPLATE),undefined=ChainableUndefined,autoescape=True)
html=env.get_template('search.html').render(
    config=config,app_config=config,lang='de',page='search',app_version='0.12.8',
    app_license='AGPL-3.0-only',people_menu_enabled=True,stream_status={'status':'stopped'},
    entries=[],stats={},t=lambda name:name,
    pv_prefix='', pv_url_for=lambda endpoint,**kw:'/static/'+kw['filename'] if endpoint=='static' else '/'+endpoint)
html=re.sub(r'<link\s+[^>]*?href="/static/([^"]+\.css)"[^>]*>',
    lambda m:'<style>'+(STATIC / m.group(1)).read_text()+'</style>',html)
html=re.sub(r'<script\s+src="/static/([^"]+\.js)"[^>]*></script>',
    lambda m:'<script>'+(STATIC / m.group(1)).read_text()+'</script>',html)
html=re.sub(r'<link\s+[^>]*?href="https://[^>]+>', '',html)
html=re.sub(r'<script\s+src="https://[^>]+></script>', '',html)
# Fake API maintains state in the browser; no real backend or user data is used.
mock=r'''<script>
window.io=()=>({on(){}});
const __pvLocal = {};
Object.defineProperty(window, 'localStorage', {configurable: true, value: {
 getItem:(k)=>Object.prototype.hasOwnProperty.call(__pvLocal,k) ? __pvLocal[k] : null,
 setItem:(k,v)=>{__pvLocal[k]=String(v)},
 removeItem:(k)=>{delete __pvLocal[k]}
}});
window.__pwned=0;
window.__reviewRequests=[];
window.__records=[
  {id:'p1',plate_text:'ZH 100',timestamp:'2026-10-08T10:00:00',confidence:.87,review_status:'unreviewed',has_plate_image:false},
  {id:'p2',plate_text:'BE 200',timestamp:'2026-10-08T11:00:00',confidence:.91,review_status:'confirmed',has_plate_image:false},
  {id:'p3',plate_text:'AG 300',timestamp:'2026-10-08T12:00:00',confidence:.44,review_status:'needs_review',review_note:'<img src=x onerror="window.__pwned=1">',has_plate_image:false}
];
window.fetch=async (input,options={})=>{
 const url=String(input),path=url.split('?')[0];
 let data={success:true};
 if(path==='/api/history/search') {
   const f=JSON.parse(options.body||'{}');
   const matches=window.__records.filter(r=>!f.review_status||f.review_status==='all'||(r.review_status||'unreviewed')===f.review_status);
   data={success:true,total:matches.length,limit:50,offset:0,entries:matches};
 } else if(path==='/api/history/review-summary') {
   data={success:true,unreviewed:0,confirmed:0,needs_review:0,ignored:0};
   for(const row of window.__records)data[row.review_status||'unreviewed']++;
 } else if(path.startsWith('/api/history/')&&path.endsWith('/review')) {
   const body=JSON.parse(options.body||'{}');
   window.__reviewRequests.push(body);
   const id=decodeURIComponent(path.split('/')[3]);
   const record=window.__records.find(r=>r.id===id);
   if(!record||body.expected_updated_at !== String(record.review_updated_at||'')){
      return {ok:false,status:409,json:async()=>({success:false,error:'zwischenzeitlich geändert'})};
   }
   record.review_status=body.review_status;
   record.review_note=body.review_note;
   record.review_updated_at='2026-10-09T22:00:00Z';
   data={success:true,entry:record};
 } else if(path==='/api/history/facets') data={sources:[],vehicle_types:[],vehicle_colors:[]};
 else if(path==='/api/watchlist') data={items:[]};
 else if(path==='/api/history/autocomplete') data={items:[]};
 else if(path==='/api/stream/status') data={status:'stopped',connected:false};
 return {ok:true,status:200,json:async()=>data};
};
</script>'''
html=html.replace('<head>', '<head>'+mock)

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True,executable_path=shutil.which('chromium') or '/usr/bin/chromium',
                                 args=['--no-sandbox','--disable-dev-shm-usage'])
    try:
        page=browser.new_page(viewport={'width':1440,'height':900})
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.set_content(html, wait_until='domcontentloaded')
        page.wait_for_function('document.querySelector("#reviewCountUnreviewed").textContent === "1"')
        assert page.locator('#resultsBody tr').count()==3
        assert page.locator('#resultsBody img').count()==0
        assert page.evaluate('window.__pwned')==0
        page.screenshot(path=str(ROOT.parent/'PlateVision_0.12.8_Suche_Desktop.png'),full_page=False)
        page.locator('#resultsBody tr').nth(0).get_by_role('button',name='Prüfen').click()
        page.locator('#reviewStatusInput').select_option('needs_review')
        note='<svg/onload=window.__pwned=1> Geprüft'
        page.locator('#reviewNoteInput').fill(note)
        page.locator('#saveReviewButton').click()
        page.wait_for_function('document.querySelector("#reviewCountNeedsReview").textContent === "2"')
        assert page.evaluate('window.__pwned')==0
        assert page.evaluate('window.__reviewRequests[0].review_note')==note
        assert note in page.locator('#resultsBody').inner_text()
        page.get_by_role('button',name='Bestätigt').click()
        page.wait_for_function('document.querySelector("#resultTotal").textContent === "1"')
        assert page.locator('#reviewStatusFilter').input_value()=='confirmed'
        assert not page.locator('#unique').is_checked()
        # Presets are browser-local and survive a full reload.
        page.locator('#q').fill('ZH 100')
        page.locator('#presetName').fill('Einfahrt morgens')
        page.locator('button[onclick="savePreset()"] ').click()
        page.wait_for_function('document.querySelector("#presetSelect").value === "Einfahrt morgens"')
        page.locator('#q').fill('ABC')
        page.locator('button[onclick="applyPreset()"] ').click()
        assert page.locator('#q').input_value()=='ZH 100'
        assert page.locator('#reviewStatusFilter').input_value()=='confirmed'
        # Simulated reload: rebuild selector from persisted localStorage data.
        page.evaluate('renderPresets()')
        page.locator('#presetSelect').select_option('Einfahrt morgens')
        page.locator('button[onclick="applyPreset()"] ').click()
        assert page.locator('#q').input_value()=='ZH 100'
        # Old annotation is displayed as text, not as DOM markup.
        assert page.locator('#resultsBody svg').count()==0
        assert page.evaluate('window.__pwned')==0
        assert not errors,errors
        print('  OK Desktop: Review, optimistic API request, XSS, filters, local presets')
        for width in (320,390,768):
            mobile=browser.new_page(viewport={'width':width,'height':850})
            mobile_errors=[]
            mobile.on('pageerror',lambda e:mobile_errors.append(str(e)))
            mobile.set_content(html, wait_until='domcontentloaded')
            mobile.wait_for_function('document.querySelector("#resultTotal").textContent === "3"')
            bounds=mobile.evaluate('({scroll:document.documentElement.scrollWidth,width:innerWidth})')
            assert bounds['scroll'] <= width + 3,(width,bounds)
            if width==390:
                mobile.screenshot(path=str(ROOT.parent/'PlateVision_0.12.8_Suche_Mobil.png'),full_page=True)
            mobile.locator('#resultsBody tr').nth(0).get_by_role('button',name='Prüfen').click()
            assert mobile.locator('#reviewDialog').evaluate('(el)=>el.open')
            mobile.locator('#reviewNoteInput').fill('Sicher')
            mobile.locator('#saveReviewButton').click()
            mobile.wait_for_function('!document.querySelector("#reviewDialog").open')
            assert not mobile_errors,mobile_errors
            mobile.close()
        print('  OK Mobil: 320/390/768 px, Dialog und Speichern')
        print('PlateVision 0.12.8 Chromium quality review: OK')
    finally:
        browser.close()
