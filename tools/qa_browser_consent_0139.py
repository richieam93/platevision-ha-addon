#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Chromium: responsive, ingress-safe and private HA pause/due-status UI."""
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
from qa_browser_ingress_0130 import ROOT, PREFIX, render
from qa_browser_alerts_0135 import FAKE_SCRIPT
from qa_browser_ha_0136 import FAKE_HA

PAUSED_FETCH = r'''<script>
const __previousHAFetch = window.fetch;
window.fetch=async (url,options={})=>{
  const response=await __previousHAFetch(url,options);
  if(String(url).endsWith('/api/alerts/homeassistant') &&
     (options.method||'GET').toUpperCase()==='GET') {
    return {ok:true,status:200,json:async()=>({...await response.json(),paused:true})};
  }
  return response;
};
</script>'''


def run():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=shutil.which('chromium') or '/usr/bin/chromium',
            headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
        for width in (320,390,1366):
            page=browser.new_page(viewport={'width':width,'height':900})
            errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.set_content(render('diagnostics'),wait_until='domcontentloaded')
            page.evaluate('''() => {
              window.__pvCalls=[];
              window.fetch=async (url,opts={})=>{
                const path=String(url); window.__pvCalls.push(path);
                if(path.includes('/api/system/health')) return {ok:true,json:async()=>({
                  ok:true,summary:{healthy:true,problems:0,disabled:3},
                  camera:{enabled:false,connected:false,fps:0,processed_frames:4,detections:0},
                  homeassistant_notifications:{enabled:false,configured:true,paused:true,
                    token_available:true,queue:{pending:2,sending:0,accepted:10,failed:0,
                      due_now:2,oldest_due_seconds:35,oldest_waiting_seconds:3600,
                      next_retry_at: Date.now()/1000+240}},
                  checks:[{name:'ha_notifications',ok:true,state:'disabled',
                    detail:'<img src=x onerror="window.__XSS=true">'}]
                })};
                if(path.includes('/api/system/audit')) return {ok:true,json:async()=>({issues:[],suggestions:[]})};
                return {ok:true,json:async()=>({success:true,disk:{free_mb:4096},issues:[],
                  files:[],video_jobs:{active:0,completed:0},authentication_enabled:true})};
              };
            }''')
            page.evaluate('loadDiag()')
            page.locator('#haDeliveryHealthDetails > summary').click()
            block=page.locator('#haDeliveryMonitor')
            page.wait_for_function("document.querySelector('#haDeliveryMonitor').textContent.includes('Pausiert')")
            content=block.inner_text()
            for part in ('Pausiert','Jetzt fällig','Nächster Versuch','2','4 min'):
                assert part in content,(width,part,content)
            assert not page.evaluate('!!window.__XSS')
            assert block.locator('img').count()==0
            assert page.evaluate(f'window.__pvCalls.some(u=>u.includes("{PREFIX}/api/system/health"))')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 3'),width
            assert not errors,(width,errors)
            if width==1366:
                page.add_style_tag(content='.top-header {position:static !important;}')
                block.screenshot(path=str(ROOT/'PlateVision_0.13.9_Zustellmonitor.png'))
            page.close()

            alerts=browser.new_page(viewport={'width':width,'height':900})
            errors=[]
            alerts.on('pageerror',lambda error:errors.append(str(error)))
            html=render('alerts').replace('</head>',FAKE_SCRIPT+FAKE_HA+PAUSED_FETCH+'</head>',1)
            alerts.set_content(html,wait_until='domcontentloaded')
            alerts.locator('#pvHASettings > summary').click()
            alerts.wait_for_function("document.querySelector('#pvHAStatus').textContent.includes('Versand pausiert')")
            assert 'Versand pausiert' in alerts.locator('#pvHAStatus').inner_text()
            assert alerts.locator('a[data-nav-link="alerts"]').get_attribute('href')==PREFIX+'/alerts'
            assert alerts.evaluate('document.documentElement.scrollWidth <= innerWidth+3'),width
            assert not errors,(width,errors)
            alerts.close()
        browser.close()
    print('PlateVision 0.13.9 Chromium 320/390/1366px, HA pause/due, safe text, ingress: OK')


if __name__=='__main__':
    run()
