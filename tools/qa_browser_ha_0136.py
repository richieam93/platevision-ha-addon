#!/usr/bin/env python3
"""Chromium UI check of opt-in HA controls, privacy defaults and Ingress."""
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
from qa_browser_ingress_0130 import render, PREFIX
from qa_browser_alerts_0135 import FAKE_SCRIPT

ROOT=Path(__file__).resolve().parents[1]
FAKE_HA=r'''<script>
window.__haSaved={ha_persistent_enabled:false,ha_mobile_enabled:false,
  ha_mobile_service:'',ha_include_plate:false};
window.__haPosted=[];
const __originalFetch=window.fetch;
window.fetch=async (url,opts={})=>{
  const path=String(url),method=(opts.method||'GET').toUpperCase();
  if(path.includes('/api/alerts/homeassistant')){
    window.__haPosted.push({url:path,method});
    if(path.endsWith('/test')) return {ok:true,status:200,json:async()=>({success:true,message:'Angenommen'})};
    if(path.endsWith('/retry')) return {ok:true,status:200,json:async()=>({success:true,queued:1})};
    if(method==='POST'){
      window.__haSaved=JSON.parse(opts.body);
      window.__haSaved.ha_mobile_service=window.__haSaved.ha_mobile_service.replace(/^notify\./,'');
    }
    return {ok:true,status:200,json:async()=>({success:true,settings:window.__haSaved,
      token_available:true,delivery:{pending:1,accepted:2,failed:0}})};
  }
  return __originalFetch(url,opts);
};
</script>'''


def run():
  with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path=shutil.which('chromium') or '/usr/bin/chromium',
      headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
    for width in [320,390,1366]:
      page=browser.new_page(viewport={'width':width,'height':850})
      errors=[]
      page.on('pageerror',lambda err:errors.append(str(err)))
      html=render('alerts').replace('PLATEVISION V0.13.0','PLATEVISION V0.13.6').replace('</head>',FAKE_SCRIPT+FAKE_HA+'</head>',1)
      page.set_content(html,wait_until='domcontentloaded')
      page.locator('.pv-alert-card').first.wait_for(timeout=15000)
      page.locator('#pvHASettings > summary').click()
      page.wait_for_function("document.querySelector('#pvHAStatus').textContent.includes('HA-Token')")
      assert not page.locator('#pvHAMobile').is_checked()
      assert not page.locator('#pvHAPersistent').is_checked()
      assert not page.locator('#pvHAIncludePlate').is_checked()
      assert page.locator('a[data-nav-link="alerts"]').get_attribute('href') == PREFIX+'/alerts'
      page.locator('#pvHAMobile').check()
      page.locator('#pvHAMobileService').fill('invalid/path')
      page.locator('#pvHASave').click()
      assert 'gültigen Dienst' in page.locator('#pvHAFeedback').inner_text()
      page.locator('#pvHAMobileService').fill('notify.mobile_app_testtelefon')
      page.locator('#pvHASave').click()
      page.wait_for_function("window.__haSaved.ha_mobile_service==='mobile_app_testtelefon'")
      assert page.evaluate('window.__haSaved.ha_mobile_enabled')
      assert not page.evaluate('window.__haSaved.ha_include_plate')
      page.locator('#pvHATest').click()
      page.wait_for_function("window.__haPosted.some(x=>x.url.endsWith('/test'))")
      page.locator('#pvHARetry').click()
      page.wait_for_function("window.__haPosted.some(x=>x.url.endsWith('/retry'))")
      assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+3'),width
      assert not errors,errors
      if width==1366:
        page.add_style_tag(content='.top-header {position: static !important; z-index: 0 !important;}')
        page.locator('#pvHASettings').screenshot(path=str(ROOT/'PlateVision_0.13.6_HA_Benachrichtigungen.png'))
      page.close()
    browser.close()
  print('PlateVision 0.13.6 HA browser: mobile/desktop, settings, validation, test, retry, Ingress: OK')

if __name__=='__main__':run()
