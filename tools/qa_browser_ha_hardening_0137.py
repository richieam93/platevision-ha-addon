#!/usr/bin/env python3
"""Offline Chromium check for the HA delivery audit (real HTML and JS)."""
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
from qa_browser_ingress_0130 import render, PREFIX
from qa_browser_alerts_0135 import FAKE_SCRIPT
from qa_browser_ha_0136 import FAKE_HA

ROOT=Path(__file__).resolve().parents[1]
FAKE_AUDIT=r'''<script>
const __preAuditFetch=window.fetch;
window.fetch=async (url,opts={})=>{
  if(String(url).includes('/api/alerts/homeassistant/deliveries')) {
    window.__auditRequests=(window.__auditRequests || 0)+1;
    return {ok:true,status:200,json:async()=>({success:true,items:[
      {id:1,channel:'mobile',state:'failed',attempts:3,created_at:1780000000,
        last_error:'<img src=x onerror="window.__XSS=true">',target:'DO-NOT-EXPOSE'},
      {id:2,channel:'persistent',state:'accepted',attempts:1,created_at:1780000010,
        accepted_at:1780000020,plate_text:'PRIVATE-PLATE'}
    ],status:{pending:0,failed:1,accepted:1}})};
  }
  return __preAuditFetch(url,opts);
};
</script>'''


def main():
    with sync_playwright() as pw:
        browser=pw.chromium.launch(
            executable_path=shutil.which('chromium') or '/usr/bin/chromium',
            headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        for width in (320,390,1366):
            page=browser.new_page(viewport={'width':width,'height':900})
            errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            html=render('alerts').replace('</head>',FAKE_SCRIPT+FAKE_HA+FAKE_AUDIT+'</head>',1)
            page.set_content(html,wait_until='domcontentloaded')
            page.locator('#pvHASettings > summary').click()
            page.locator('#pvHADeliveryDetails > summary').click()
            page.wait_for_function("window.__auditRequests >= 1")
            log=page.locator('#pvHADeliveryLog')
            assert '3 Versuch(e)' in log.inner_text()
            assert 'Home Assistant' in log.inner_text()
            assert log.locator('img').count()==0
            assert not page.evaluate('!!window.__XSS')
            assert 'PRIVATE-PLATE' not in log.inner_text()
            assert 'DO-NOT-EXPOSE' not in log.inner_text()
            page.locator('#pvHALogRefresh').click()
            page.wait_for_function('window.__auditRequests >= 2')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 3'),width
            assert not errors,errors
            assert page.locator('a[data-nav-link="alerts"]').get_attribute('href')==PREFIX+'/alerts'
            if width==1366:
                page.add_style_tag(content='.top-header {position:static !important;z-index:0 !important;}')
                page.locator('#pvHASettings').screenshot(path=str(ROOT/'PlateVision_0.13.7_Zustellprotokoll.png'))
            page.close()
        browser.close()
    print('PlateVision HA delivery audit 0.13.7: Chromium 320/390/1366 px, XSS, Ingress: OK')


if __name__=='__main__':main()
