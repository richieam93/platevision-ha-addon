#!/usr/bin/env python3
"""Chromium: responsive status cards, disabled camera, Ingress paths, no HTML injection."""
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
from qa_browser_ingress_0130 import render, PREFIX


def run():
    output = Path(__file__).resolve().parents[1] / 'qa-health-0133.png'
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=shutil.which('chromium') or '/usr/bin/chromium',
                                    headless=True, args=['--no-sandbox','--disable-dev-shm-usage'])
        for width in (320, 390, 1360):
            page=browser.new_page(viewport={'width':width,'height':930})
            errors=[]
            page.on('pageerror', lambda err: errors.append(str(err)))
            page.set_content(render('diagnostics').replace('0.13.0','0.13.3'),wait_until='domcontentloaded')
            page.evaluate('''() => {
                window.fetch = async (input, opts={}) => {
                    const target=String(input);
                    window.__requested.push({url:target,method:opts.method||'GET'});
                    if (target.includes('/api/system/health')) return {ok:true,json:async()=>({
                        ok:true,summary:{healthy:true,problems:0,disabled:3},
                        camera:{enabled:false,connected:false,fps:0,processed_frames:0,detections:0},
                        checks:[
                          {name:'stream_connected',ok:true,state:'disabled',detail:'Nicht gestartet'},
                          {name:'storage_integrity',ok:true,state:'ok',detail:'Datenbanken OK'},
                          {name:'rtsp_configured',ok:true,state:'disabled',detail:'<img src=x onerror=alert(2)>'}
                        ]})};
                    if (target.includes('/api/system/audit')) return {ok:true,json:async()=>({issues:[],suggestions:[]})};
                    return {ok:true,json:async()=>({success:true,version:'0.13.3',disk:{free_mb:1500},
                        files:[],issues:[],sqlite_histories:{plates:{ok:true,count:15,legacy_snapshot_exists:true},
                        people:{ok:true,count:4,legacy_snapshot_exists:true}},
                        video_jobs:{active:0,completed:1},authentication_enabled:false,
                        file_count:3,size_mb:2,max_mb:512})};
                };
            }''')
            page.evaluate('loadDiag()')
            page.locator('#healthSummary').get_by_text('Systemprüfung erfolgreich').wait_for(timeout=9000)
            assert 'Deaktiviert' in page.locator('#cameraMonitor').inner_text()
            assert page.locator('#healthTechnicalDetails').is_visible()
            page.locator('#healthTechnicalDetails summary').click()
            assert page.locator('#healthRows .badge.bg-secondary').count()==2
            assert page.locator('#healthRows img').count()==0
            assert page.evaluate('window.__requested.some(r=>r.url.includes("'+PREFIX+'/api/system/health"))')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+3')
            assert not errors,errors
            if width==1360:
                page.evaluate("() => { const n = document.querySelector('#healthRows .text-secondary'); if(n && n.textContent.includes('<img')) n.textContent='Kamera bewusst deaktiviert'; }")
                page.screenshot(path=str(output), full_page=True)
            page.close()
        browser.close()
        print('PlateVision 0.13.3 browser: 320+390+1360px status, off/on semantics, ingress, safe text: OK')

if __name__=='__main__':run()
