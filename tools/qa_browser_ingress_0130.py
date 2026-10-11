#!/usr/bin/env python3
"""Real Chromium iframe/DOM regression without Home Assistant or network.

Network navigation to localhost is restricted in some test environments, so this
uses a srcdoc frame with the real rendered PlateVision template, inlined local
assets and a mocked API. This does NOT validate the live Supervisor proxy.
"""
import ast
import json
import re
import shutil
from html import escape
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, ChainableUndefined
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'platevision/src'
PREFIX='/api/hassio_ingress/sessionToken123456'
TREE=ast.parse((SRC/'app.py').read_text())
CLS=next(n for n in TREE.body if isinstance(n,ast.ClassDef) and n.name=='ConfigManager')
CONFIG=ast.literal_eval(next(n.value for n in CLS.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_CONFIG' for t in n.targets)))
LANG=ast.literal_eval(next(n.value for n in TREE.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRANSLATIONS' for t in n.targets)))
ENV=Environment(loader=FileSystemLoader(SRC/'templates'),undefined=ChainableUndefined,autoescape=True)

def render(template):
    c=dict(config=CONFIG,app_config=CONFIG,lang='de',page=template,
           app_version='0.13.0',app_license='AGPL-3.0-only',people_menu_enabled=True,
           stream_status={'status':'stopped'},entries=[],stats={},
           t=lambda name: LANG['de'].get(name,name), pv_prefix=PREFIX,
           pv_url_for=lambda endpoint,**kw:PREFIX+('/static/'+kw['filename'] if endpoint=='static' else '/'+{'search_page':'search','latest_detection_page':'latest','people_gallery_page':'people/gallery','statistics_page':'statistics','people_page':'people','live_view':'live','test_page':'test','watchlist_alerts_page':'alerts','diagnostics_page':'diagnostics','legal_page':'legal','rtsp_settings':'rtsp-settings'}.get(endpoint,endpoint)))
    markup=ENV.get_template(template+'.html').render(c)
    markup=re.sub(r'<link\s+[^>]*?href="'+re.escape(PREFIX)+r'/static/([^\"]+\.css)"[^>]*>',
        lambda m:'<style>'+(SRC/'static'/m.group(1)).read_text()+'</style>',markup)
    markup=re.sub(r'<script\s+src="'+re.escape(PREFIX)+r'/static/([^\"]+\.js)"[^>]*></script>',
        lambda m:'<script>'+(SRC/'static'/m.group(1)).read_text()+'</script>',markup)
    markup=re.sub(r'<link\s+[^>]*?href="https://[^>]+>','',markup)
    markup=re.sub(r'<script\s+src="https://[^>]+></script>','',markup)
    mock='''<script>
    window.__requested = [];
    window.io=function(options){window.__socketPath=options?.path;return {on(){},emit(){}}};
    window.Chart=class{constructor(){}destroy(){}};
    window.fetch = async (input, options={}) => {
      window.__requested.push({url:String(input),method:options.method||'GET'});
      const data={success:true,status:'stopped',stream:{status:'stopped'},config:__CONFIG__,
        entries:[],items:[],results:[],models:[],stats:{},schema:{},total:0,limit:50,offset:0,latest:[],
        latest_vehicle:{id:'v1',plate_text:'ZH 123',vehicle_type:'PKW',image_url:'/api/latest/vehicle/image'},
        today:{raw_detections:4,unique_vehicles:2}};
      return {ok:true,status:200,json:async()=>data};
    };
    </script>'''.replace('__CONFIG__',json.dumps(CONFIG,ensure_ascii=False))
    return markup.replace('<head>', '<head>'+mock,1)


def run():
    errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=shutil.which('chromium') or '/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1600,'height':1050})
        page.on('pageerror',lambda exc:errors.append(str(exc)))
        def load(template):
            src=render(template)
            fake_ha=f'<html><body><header style="padding:15px;font-family:Arial">Home Assistant · PlateVision</header><iframe id="platevision" srcdoc="{escape(src,quote=True)}" style="width:1500px;height:940px;border:0"></iframe></body></html>'
            page.set_content(fake_ha,wait_until='domcontentloaded')
            return page.frame_locator('#platevision')
        frame=load('dashboard')
        frame.locator('#latestPlate').wait_for(timeout=15000)
        page.wait_for_timeout(550)
        assert frame.locator('#latestPlate').inner_text()=='ZH 123'
        assert frame.locator('#latestVehiclePlaceholder').is_visible(), 'Failed image should show placeholder'
        assert frame.locator('#latestVehicleImage').evaluate('(el)=>getComputedStyle(el).display')=='none'
        assert frame.locator('a[data-nav-link="search"]').get_attribute('href')==PREFIX+'/search'
        assert frame.locator('html').evaluate('(el)=>window.__socketPath')==PREFIX+'/socket.io'
        assert frame.locator('html').evaluate('(el)=>window.pvUrl("/api/latest")')==PREFIX+'/api/latest'
        assert frame.locator('html').evaluate('(el)=>window.pvUrl("/api/hassio_ingress/sessionToken123456/api/latest")')==PREFIX+'/api/latest'
        assert frame.locator('html').evaluate('(el)=>window.pvSafeImageUrl("/api/people/images/test.jpg")')==PREFIX+'/api/people/images/test.jpg'
        assert frame.locator('html').evaluate('(el)=>window.__requested.some(x=>x.url==="'+PREFIX+'/api/dashboard/overview")'), 'Dashboard bypasses ingress'
        assert not errors,errors
        frame=load('search')
        frame.locator('#sidebarNavSearch').wait_for(timeout=10000)
        assert frame.locator('a[data-nav-link="dashboard"]').get_attribute('href')==PREFIX+'/dashboard'
        assert frame.locator('html').evaluate('(el)=>window.PV_INGRESS_PREFIX')==PREFIX
        frame=load('people_gallery')
        frame.locator('#sidebarNavSearch').wait_for(timeout=10000)
        assert frame.locator('html').evaluate('(el)=>safeArchiveImageUrl("/api/people/images/crops/test.jpg")')==PREFIX+'/api/people/images/crops/test.jpg'
        assert not errors,errors
        print('PlateVision Ingress Chromium: 3 embedded pages, dashboard image fallback, JS URLs, Socket.IO path, gallery URL and navigation links: OK')
        browser.close()

if __name__=='__main__': run()
