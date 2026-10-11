#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Responsive Chromium verification of the new privacy-safe HA health monitor."""
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
from qa_browser_ingress_0130 import PREFIX, render


def run():
    out = Path('/mnt/data/PlateVision_0.13.8_Zustellmonitor.png')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which('chromium') or '/usr/bin/chromium',
            headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
        for width in (320, 390, 1366):
            page = browser.new_page(viewport={'width':width,'height':900})
            errors=[]
            page.on('pageerror',lambda exc: errors.append(str(exc)))
            page.set_content(render('diagnostics'),wait_until='domcontentloaded')
            page.evaluate('''() => {
                window.__calls=[];
                window.fetch = async (input, opts={}) => {
                    const url=String(input); window.__calls.push(url);
                    if (url.includes('/api/system/health')) return {ok:true,json:async()=>({
                        ok:false, summary:{healthy:false,problems:1,disabled:2},
                        camera:{enabled:false,connected:false,fps:0,processed_frames:0,detections:0},
                        homeassistant_notifications:{enabled:true,token_available:true,
                          queue:{pending:2,sending:1,accepted:4,failed:3,
                            oldest_waiting_seconds:1900,expired_leases:1,
                            last_accepted_at:1780000000}},
                        checks:[
                          {name:'ha_notifications',ok:false,state:'warning',
                           detail:'3 fehlgeschlagene Zustellungen'},
                          {name:'rtsp_configured',ok:true,state:'disabled',
                           detail:'<img src=x onerror="window.__XSS=true">'}]
                    })};
                    if (url.includes('/api/system/audit')) return {ok:true,json:async()=>({issues:[],suggestions:[]})};
                    return {ok:true,json:async()=>({success:true,disk:{free_mb:1200},
                       issues:[],files:[],video_jobs:{active:0,completed:0},authentication_enabled:true})};
                };
            }''')
            page.evaluate('loadDiag()')
            page.locator('#healthSummary').get_by_text('1 Prüfungen').wait_for(timeout=10000)
            page.locator('#haDeliveryHealthDetails > summary').click()
            block=page.locator('#haDeliveryMonitor')
            content=block.inner_text()
            assert 'Aktiv' in content and 'Wartend' in content and '3' in content,content
            assert '31 min' in content,content
            assert '3 fehlgeschlagene Zustellungen' not in content
            assert page.locator('#healthTechnicalDetails').count()==1
            assert block.locator('img').count()==0
            assert not page.evaluate('!!window.__XSS')
            assert page.evaluate(f'window.__calls.some(url=>url.includes("{PREFIX}/api/system/health"))')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+3'),width
            assert not errors,(width,errors)
            if width==1366:
                page.add_style_tag(content='.top-header {position:static !important;}')
                page.locator('#haDeliveryHealthDetails').screenshot(path=str(out))
            page.close()
        browser.close()
    print('PlateVision 0.13.8 HA health: Chromium 320/390/1366px, privacy, status, Ingress OK')


if __name__=='__main__':
    run()
