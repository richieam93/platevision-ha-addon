#!/usr/bin/env python3
"""Optional Playwright/Chromium navigation QA. No external camera/backend required."""
import ast
import json
import re
import shutil
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, ChainableUndefined
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
APP = ast.parse((ROOT/'platevision/src/app.py').read_text())
CLS = next(node for node in APP.body if isinstance(node,ast.ClassDef) and node.name=='ConfigManager')
CONFIG = ast.literal_eval(next(node.value for node in CLS.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_CONFIG' for t in node.targets)))
LANGS = ast.literal_eval(next(node.value for node in APP.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRANSLATIONS' for t in node.targets)))
ENV = Environment(loader=FileSystemLoader(ROOT/'platevision/src/templates'),undefined=ChainableUndefined,autoescape=True)
STATIC = ROOT/'platevision/src/static'
PAGES = {'dashboard':'dashboard','live_view':'live','latest_detection_page':'latest',
         'watchlist_alerts_page':'alerts','history':'history','search_page':'search',
         'statistics_page':'statistics','people_page':'people','people_gallery_page':'people_gallery',
         'test_page':'test','rtsp_settings':'rtsp','settings':'settings','diagnostics_page':'diagnostics','legal_page':'legal'}

def render(page='dashboard',people=True,stored=None,full=False):
    html=ENV.get_template('dashboard.html' if full and page=='dashboard' else 'base.html').render(page=page,app_config=CONFIG,config=CONFIG,lang='de',
       people_menu_enabled=people,app_version='0.13.5',app_license='AGPL-3.0-only',stream_status={'status':'stopped'},
       t=lambda name: LANGS['de'].get(name,name),
       pv_prefix='', pv_url_for=lambda endpoint,**kw: '/static/'+kw['filename'] if endpoint=='static' else '/'+endpoint)
    html=re.sub(r'<link\s+[^>]*?href="/static/([^"]+\.css)"[^>]*>',
       lambda m:'<style>'+(STATIC/m.group(1)).read_text()+'</style>',html)
    html=re.sub(r'<script\s+src="/static/([^"]+\.js)"[^>]*></script>',
       lambda m:'<script>'+(STATIC/m.group(1)).read_text()+'</script>',html)
    html=re.sub(r'<link\s+[^>]*?href="https://[^>]+>','',html)
    html=re.sub(r'<script\s+src="https://[^>]+></script>','',html)
    storage=json.dumps(stored or {},ensure_ascii=False).replace('<', r'\u003c')
    stub='''<head><script>
    window.__pvTestStored=__STORED__;
    Object.defineProperty(window,'localStorage',{configurable:true,value:{
      getItem:k=>Object.hasOwn(window.__pvTestStored,k) ? window.__pvTestStored[k] : null,
      setItem:(k,v)=>{window.__pvTestStored[k]=String(v)},
      removeItem:k=>{delete window.__pvTestStored[k]}
    }});
    window.io=()=>({on(){}});
    window.Chart=class {constructor(){}destroy(){}};
    window.fetch=async ()=>({ok:true,status:200,json:async()=>({success:true, status:{status:'stopped'},config:{general:{language:'de',theme:'dark'}},items:[]})});
    </script>'''.replace('__STORED__',storage)
    return html.replace('<head>',stub,1)

def run():
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=shutil.which('chromium') or '/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        try:
            errors=[]
            def open_page(page_id='dashboard', people=True, stored=None, width=1440, height=940, full=False):
                page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
                page.on('pageerror',lambda error:errors.append(str(error)))
                page.set_content(render(page_id,people,stored,full),wait_until='domcontentloaded')
                page.wait_for_function("document.querySelector('#mainNavigation')?.dataset.enhanced === 'true'")
                return page
            page=open_page()
            assert page.locator('.pv-nav-group').count()==5
            assert page.locator('.pv-nav-entry').count()==14
            assert page.locator('.pv-nav-current-group').get_attribute('data-nav-group')=='overview'
            assert page.locator('a[aria-current="page"]').count()==1
            assert page.locator('#sidebar').evaluate('e=>getComputedStyle(e).display')=='flex'
            assert page.locator('.pv-nav-entry[data-nav-page="people"]').count()==1
            # Persisted group collapse and temporary search expansion.
            page.locator('[data-nav-group-toggle="plates"]').click()
            assert page.locator('[data-nav-group-toggle="plates"]').get_attribute('aria-expanded')=='true'
            page.locator('[data-nav-group-toggle="plates"]').click()
            assert page.locator('[data-nav-group-toggle="plates"]').get_attribute('aria-expanded')=='false'
            page.locator('#sidebarNavSearch').fill('kennzeichen')
            assert page.locator('.pv-nav-entry[data-nav-page="search"]').is_visible()
            assert page.locator('[data-nav-group-toggle="plates"]').get_attribute('aria-expanded')=='true'
            page.locator('#sidebarNavSearch').fill('keines_dieser_module_xyz')
            assert page.locator('#pvNavNoResults').is_visible()
            page.locator('#pvNavSearchClear').click()
            assert page.locator('[data-nav-group-toggle="plates"]').get_attribute('aria-expanded')=='false'
            page.locator('[data-nav-group-toggle="plates"]').click()
            page.locator('[data-nav-pin="search"]').click()
            assert page.locator('#pvNavFavorites a').count()==1
            assert page.locator('#pvNavFavorites a').first.get_attribute('href')=='/search_page'
            stored=page.evaluate('window.__pvTestStored')
            page.close()
            page=open_page('search',stored=stored)
            assert page.locator('#pvNavFavorites a').count()==1
            assert page.locator('.pv-nav-current-group').get_attribute('data-nav-group')=='plates'
            assert page.locator('#pvPageFavorite').get_attribute('aria-pressed')=='true'
            assert page.locator('a[aria-current="page"]').count()==2
            page.locator('#pvPageFavorite').click()
            assert page.locator('#pvNavFavorites').is_hidden()
            # Favoriten sind begrenzt, ungültige/unbekannte IDs haben keine Wirkung.
            page.evaluate('''() => [...document.querySelectorAll('[data-nav-pin]')].slice(0, 7).forEach(el => el.click())''')
            assert page.locator('#pvNavFavorites a').count()==6
            assert len(json.loads(page.evaluate('localStorage.getItem("pv_nav_favorites_v1")')))==6
            page.evaluate('''() => [...document.querySelectorAll('[data-nav-pin]')].slice(0, 6).forEach(el => el.click())''')
            assert page.locator('#pvNavFavorites').is_hidden()
            page.keyboard.press('Control+k')
            assert page.locator('#sidebarNavSearch').evaluate('e=>e===document.activeElement')
            page.locator('#sidebarNavSearch').fill('diagnose')
            assert page.locator('.pv-nav-entry:not([hidden])').count()==1
            page.locator('#sidebarNavSearch').press('Enter') if False else None
            # Enter opens a sole matching page. Here the href target is verified
            # because the sandboxed Chromium denies all HTTP navigation.
            assert page.locator('.pv-nav-entry:not([hidden]) > a').first.get_attribute('href')=='/diagnostics_page'
            page.close()
            # Each of the fourteen existing routes retains its own active state.
            for endpoint, current in PAGES.items():
                probe=open_page(current)
                assert probe.locator('.pv-nav-entry a[aria-current="page"]').count()==1,endpoint
                assert probe.locator('.pv-nav-entry a[aria-current="page"]').first.get_attribute('data-nav-link')==current,endpoint
                assert probe.locator('.pv-context-bar').count()==1,endpoint
                assert probe.locator('.pv-context-link').count()>=2,endpoint
                assert probe.locator('.pv-nav-current-group .pv-nav-group-toggle').get_attribute('aria-expanded')=='true',endpoint
                probe.close()
            malicious={'pv_nav_favorites_v1':json.dumps(['<img src=x onerror=alert(1)>','history'])}
            page=open_page('history',stored=malicious)
            assert page.locator('#pvNavFavorites a').count()==1
            assert page.locator('#pvNavFavorites img').count()==0
            assert 'Historie' in page.locator('#pvNavFavorites').inner_text()
            page.close()
            page=open_page(people=False,stored={'pv_nav_favorites_v1':json.dumps(['people','search'])})
            assert page.locator('[data-nav-group="persons"]').count()==0
            assert page.locator('.pv-nav-entry').count()==12
            assert page.locator('#pvNavFavorites a').count()==1
            page.close()
            print('Desktop: collapsible sections, search, favorites, persisted state, all 14 pages, disabled-person module and XSS: OK')
            for width in [320,390,768,1024,1440]:
                view=open_page(width=width,height=844)
                if width<=768:
                    view.locator('#sidebarToggle').click()
                    assert view.locator('#sidebarToggle').get_attribute('aria-expanded')=='true'
                    assert view.locator('#sidebarNavSearch').is_visible()
                    view.locator('#sidebarNavSearch').fill('person')
                    assert view.locator('.pv-nav-entry[data-nav-page="people_gallery"]').is_visible()
                    view.locator('#sidebarNavSearch').press('Escape')
                    assert view.locator('#sidebarNavSearch').input_value()==''
                    assert view.locator('#sidebarToggle').get_attribute('aria-expanded')=='true'
                    if width==390:
                        view.wait_for_timeout(350)
                        box=view.locator('#sidebar').bounding_box()
                        assert box and box['x'] >= -1 and box['width'] > 245,box
                        view.screenshot(path='/mnt/data/PlateVision_0.12.9_Navigation_Mobil.png')
                    view.locator('#sidebarNavSearch').press('Escape')
                    assert view.locator('#sidebarToggle').get_attribute('aria-expanded')=='false'
                assert view.evaluate('document.documentElement.scrollWidth <= innerWidth + 2'), f'overflow {width}'
                view.close()
            assert not errors, errors
            print('Responsive: 320/390/768/1024/1440px, mobile toggle, Escape, no overflow: OK')
            overview=open_page('dashboard',width=1440,height=850,full=True)
            overview.wait_for_timeout(250)
            overview.screenshot(path='/mnt/data/PlateVision_0.12.9_Navigation_Desktop.png',full_page=True)
            overview.close()
        finally:
            browser.close()

if __name__=='__main__': run()
