#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Visual Chromium regression for 0.13.10 camera heartbeat and Ingress UI."""
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
from qa_browser_ingress_0130 import render, PREFIX

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / 'PlateVision_0.13.10_Kameramonitor.png'


def run():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=shutil.which('chromium') or '/usr/bin/chromium',
                                             headless=True, args=['--no-sandbox','--disable-dev-shm-usage'])
        try:
            for width in (320, 390, 1360):
                page = browser.new_page(viewport={'width': width, 'height': 920})
                errors = []
                page.on('pageerror', lambda err: errors.append(str(err)))
                page.set_content(render('diagnostics').replace('0.13.0', '0.13.10'), wait_until='domcontentloaded')
                page.evaluate('''() => {
                    window.__cameraAge = 28;
                    window.fetch = async (input, opts={}) => {
                        window.__requested.push({url:String(input),method:opts.method||'GET'});
                        if (String(input).includes('/api/system/health')) {
                            const age = window.__cameraAge;
                            const fresh = age < 20;
                            return {ok:true,json:async()=>({
                                ok:fresh,
                                summary:{healthy:fresh,problems:fresh?0:1,disabled:1},
                                camera:{enabled:true,connected:true,fps:15.3,
                                        processed_frames:12345,detections:214,
                                        last_frame_age_seconds:age,reconnect_count:2},
                                homeassistant_notifications:{enabled:false,
                                      queue:{pending:0,sending:0,failed:0,due_now:0}},
                                checks:[
                                  {name:'stream_connected',ok:true,detail:'Verbunden'},
                                  {name:'stream_frames_fresh',ok:fresh,
                                   detail:fresh?'Kamerabilder treffen ein':'Letztes Kamerabild vor '+age+' Sekunden'},
                                  {name:'rtsp_configured',ok:true,detail:'Gültige RTSP-Adresse hinterlegt'}
                                ]})};
                        }
                        return {ok:true,json:async()=>({issues:[],suggestions:[]})};
                    };
                }''')
                page.evaluate('loadDiag()')
                page.locator('#healthSummary').get_by_text('Prüfungen benötigen Aufmerksamkeit').wait_for(timeout=10000)
                assert '28 s' in page.locator('#cameraMonitor').inner_text()
                assert 'Neuverbindungen' in page.locator('#cameraMonitor').inner_text()
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+3')
                assert page.evaluate('window.__requested.some(r=>r.url.includes("'+PREFIX+'/api/system/health"))')
                page.evaluate('window.__cameraAge = 3; loadDiag();')
                page.locator('#healthSummary').get_by_text('Systemprüfung erfolgreich').wait_for(timeout=10000)
                assert '3 s' in page.locator('#cameraMonitor').inner_text()
                assert '2' in page.locator('#cameraMonitor').inner_text()
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+3')
                assert not errors, errors
                if width == 1360:
                    page.evaluate('''() => {
                        const card=document.querySelector('[role="region"][aria-label="Systemmonitor"]');
                        const tag=document.createElement('p');
                        tag.className='text-secondary small mb-1';
                        tag.textContent='Beispielansicht mit simulierten Kameradaten';
                        card.insertBefore(tag, card.querySelector('#healthSummary'));
                    }''')
                    page.locator('[role="region"][aria-label="Systemmonitor"]').screenshot(path=str(PREVIEW))
                page.close()
        finally:
            browser.close()
    print('PlateVision 0.13.10 Chromium: stale/fresh RTSP, 320/390/1360px, Ingress and JS: OK')


if __name__ == '__main__':
    run()
