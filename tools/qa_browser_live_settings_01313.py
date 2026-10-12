#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Chromium check for debounced, ordered RTSP live slider writes and errors."""
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
from qa_browser_ingress_0130 import render, PREFIX

ROOT=Path(__file__).resolve().parents[1]
JS = '''<script>
window.__saves=[];
window.__release=[];
window.__notices=[];
window.__holdSaves=true;
window.__errorNext=false;
const originalFetch=window.fetch;
window.fetch=(url,opts={})=>{
 if (String(url).endsWith('/api/config/detection') && opts.method==='POST') {
   window.__saves.push(JSON.parse(opts.body));
   if (window.__errorNext) { window.__errorNext=false; return Promise.resolve({ok:false,status:503}); }
   if (window.__holdSaves) return new Promise(resolve=>window.__release.push(()=>resolve({ok:true,json:async()=>({success:true})})));
   return Promise.resolve({ok:true,json:async()=>({success:true})});
 }
 return originalFetch(url,opts);
};
</script>'''

def run():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=shutil.which('chromium') or '/usr/bin/chromium',
            headless=True, args=['--no-sandbox','--disable-dev-shm-usage'])
        try:
            page = browser.new_page(viewport={'width': 390, 'height': 920})
            errors=[]
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.set_content(render('live').replace('</head>',JS+'</head>',1),wait_until='domcontentloaded')
            page.evaluate('''() => {
                window.showToast=(title, message, level)=>window.__notices.push({title,message,level});
                updateInterval('7');
                updateInterval('8');
                updateInterval('9');
                updateLiveConfidence('74');
            }''')
            page.wait_for_function('window.__saves.length===1', timeout=6000)
            first=page.evaluate('window.__saves[0]')
            assert first == {'process_interval':0.9,'confidence_threshold':0.74},first
            page.evaluate("updateInterval('14')")
            page.wait_for_timeout(600)
            assert page.evaluate('window.__saves.length')==1,'Out-of-order request started while prior write pending'
            page.evaluate('window.__release.shift()()')
            page.wait_for_function('window.__saves.length===2',timeout=6000)
            assert page.evaluate('window.__saves[1]') == {'process_interval':1.4}
            page.evaluate('''() => { window.__release.shift()(); window.__holdSaves=false; }''')
            page.evaluate("updateLiveConfidence('68')")
            page.wait_for_function('window.__saves.length===3',timeout=6000)
            assert page.evaluate('window.__saves[2]') == {'confidence_threshold':0.68}
            page.evaluate('''() => {window.__errorNext=true; updateInterval('12');}''')
            page.wait_for_function('window.__notices.length>0', timeout=6000)
            assert 'nicht gespeichert' in page.evaluate('window.__notices[0].message')
            assert page.locator('#intervalValue').inner_text() == '1.2s'
            assert not errors, errors
            print('  OK Chromium: burst coalesced, requests serialized, write failure shown')
        finally:
            browser.close()

if __name__ == '__main__': run()
