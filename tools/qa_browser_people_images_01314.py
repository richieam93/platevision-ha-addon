#!/usr/bin/env python3
"""Regression: person detail images work in Home Assistant Ingress and direct mode.

Renders the real people.html template in Chromium. Uses synthetic API records and a
no camera, private photos, supervisor, or external network required.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

from jinja2 import ChainableUndefined, Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'platevision' / 'src'
AST = ast.parse((SRC / 'app.py').read_text(encoding='utf-8'))
CLS = next(node for node in AST.body if isinstance(node, ast.ClassDef) and node.name == 'ConfigManager')
CONFIG = ast.literal_eval(next(node.value for node in CLS.body if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Name) and target.id == 'DEFAULT_CONFIG'
                                       for target in node.targets)))
JINJA = Environment(loader=FileSystemLoader(SRC / 'templates'),
                    undefined=ChainableUndefined, autoescape=True)
SUFFIX = '/api/people/images/crops/2026-10-10/person.jpg'
MISSING = '/api/people/images/crops/2026-10-10/missing.jpg'
EVENTS = {
    'found': {
        'id': 'found', 'timestamp': '2026-10-10T11:32:09', 'source': 'image_upload',
        'track_id': 1, 'direction': 'unknown', 'counted': False, 'confidence': .738,
        'image_urls': {'crop': SUFFIX},
    },
    'missing': {
        'id': 'missing', 'timestamp': '2026-10-10T11:32:09', 'source': 'image_upload',
        'image_urls': {'crop': MISSING},
    },
    'unsafe': {
        'id': 'unsafe', 'image_urls': {'crop': 'javascript:alert(1)', 'annotated': '//evil.test/track.png'},
    },
    'empty': {'id': 'empty', 'image_urls': {}},
}


def render(prefix: str) -> str:
    ctx = dict(config=CONFIG, app_config=CONFIG, lang='de', page='people',
               app_version='0.13.14', app_license='AGPL-3.0-only', people_menu_enabled=True,
               stream_status={'status':'stopped'}, entries=[], stats={},
               t=lambda k:k, pv_prefix=prefix,
               pv_url_for=lambda endpoint, **kw:prefix+'/static/'+kw['filename']
                     if endpoint == 'static' else prefix+'/'+endpoint)
    html = JINJA.get_template('people.html').render(ctx)
    html = re.sub(r'<link\s+[^>]*?href="'+re.escape(prefix)+r'/static/([^"]+\.css)"[^>]*>',
                  lambda m:'<style>'+(SRC/'static'/m.group(1)).read_text(encoding='utf-8')+'</style>',html)
    html = re.sub(r'<script\s+src="'+re.escape(prefix)+r'/static/([^"]+\.js)"[^>]*></script>',
                  lambda m:'<script>'+(SRC/'static'/m.group(1)).read_text(encoding='utf-8')+'</script>',html)
    html = re.sub(r'<link\s+[^>]*?href="https://[^>]+>', '', html)
    html = re.sub(r'<script\s+src="https://[^>]+></script>', '', html)
    fake_api = '''<script>
        window.io=() => ({on(){},emit(){}});
        window.Chart=class {constructor(){}destroy(){}};
        window.__peopleEvents=__EVENTS__;
        window.fetch=async (input) => {
            const path = String(input).split('?')[0];
            let data = {success:true,entries:[],sessions:[],summary:{},sources:{},
                        daily:[],hourly:[],total:0,limit:40,offset:0,has_more:false};
            const key = path.match(/\\/api\\/people\\/history\\/([^/]+)$/);
            if (key) data = {success:true,event:window.__peopleEvents[key[1]]};
            return {ok:true,status:200,json:async()=>data};
        };
    </script>'''.replace('__EVENTS__', json.dumps(EVENTS,ensure_ascii=False))
    return html.replace('<head>', '<head>'+fake_api, 1)


def run():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=shutil.which('chromium') or '/usr/bin/chromium',
                                     headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        try:
            for prefix in ('/api/hassio_ingress/sessionToken123456', ''):
                html = render(prefix)
                page = browser.new_page(viewport={'width':1200,'height':900})
                errors=[]
                page.on('pageerror',lambda error:errors.append(str(error)))
                # srcdoc-style test avoids network navigation; verify the exact
                # DOM URLs while the JavaScript task still owns the images.
                page.set_content(html, wait_until='domcontentloaded')
                found = page.evaluate('''async () => {
                    await openPeopleEvent('found');
                    const image = document.querySelector('#peopleEventImages img');
                    const anchor = document.querySelector('#peopleEventImages a');
                    return {src: image?.getAttribute('src'), href: anchor?.getAttribute('href')};
                }''')
                assert found == {'src': prefix+SUFFIX, 'href': prefix+SUFFIX}, (prefix,found)
                assert page.evaluate('historyThumb({image_urls:{crop:"'+SUFFIX+'"}})').find(prefix+SUFFIX)>=0
                assert page.evaluate('personCropSrc({image_urls:{crop:"'+SUFFIX+'"}})')==prefix+SUFFIX
                assert page.evaluate('personCropSrc({images:{crop:"crops/2026-10-10/person.jpg"}})')==prefix+SUFFIX
                # This used to request the root /api/... endpoint outside the Ingress session.
                if prefix:
                    assert found['src'] != SUFFIX, 'Ingress bypass detected'
                page.evaluate('openPeopleEvent("missing")')
                page.locator('#peopleEventImages [role=status]').wait_for(timeout=10000)
                assert 'nicht verfügbar' in page.locator('#peopleEventImages').inner_text()
                assert page.locator('#peopleEventImages img').count()==0
                page.evaluate('openPeopleEvent("unsafe")')
                assert page.locator('#peopleEventImages img').count()==0
                assert 'Keine Bilder gespeichert' in page.locator('#peopleEventImages').inner_text()
                page.evaluate('openPeopleEvent("empty")')
                assert 'Keine Bilder gespeichert' in page.locator('#peopleEventImages').inner_text()
                assert not errors, errors
                print('  OK', 'Ingress' if prefix else 'Direkt', ': Bild-Pfad, Bild-Link, Thumbnail, fehlende Datei, URL-Sicherheit')
                page.close()
        finally:
            browser.close()
    print('PlateVision 0.13.14 Personenbilder: Chromium OK')

if __name__ == '__main__': run()
