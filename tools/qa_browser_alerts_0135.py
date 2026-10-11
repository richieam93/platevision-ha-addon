#!/usr/bin/env python3
"""Real Chromium: alert inbox, mobile layout, API actions, safe text and Ingress."""
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
from qa_browser_ingress_0130 import render, PREFIX

ROOT=Path(__file__).resolve().parents[1]

FAKE_SCRIPT=r'''<script>
window.__alertRows = [
{id:1,plate_text:'<img src=x onerror=window.__xss=true>',normalized:'ZH101',label:'Test <svg onload=alert(1)>',
 category:'known',source:'rtsp',confidence:0.89,match_type:'exact',created_at:1760000000,last_seen_at:1760000001,
 occurrences:3,is_read:0},
{id:2,plate_text:'BE 222',normalized:'BE222',label:'Privat',
 category:'known',source:'video_upload',confidence:0.91,match_type:'similar',created_at:1760000002,last_seen_at:1760000003,
 occurrences:1,is_read:0},
{id:3,plate_text:'LU 333',normalized:'LU333',label:'Fertig',
 category:'known',source:'rtsp',confidence:0.81,match_type:'exact',created_at:1760000004,last_seen_at:1760000005,
 occurrences:1,is_read:1}
];
window.__fakeSettings={enabled:true,min_confidence:0.65,cooldown_seconds:180};
window.__requested=[];
window.fetch=async (input, opts={})=>{
  const url=String(input),method=(opts.method||'GET').toUpperCase();
  window.__requested.push({url,method});
  let data={success:true};let status=200;
  if(url.includes('/api/alerts/settings')){
    if(method==='POST') window.__fakeSettings=JSON.parse(opts.body);
    data={success:true,settings:window.__fakeSettings};
  } else if(url.includes('/api/alerts/read-all')){
    let changed=0; for(const row of window.__alertRows){if(!row.is_read){row.is_read=1;changed++;}};
    data={success:true,updated:changed};
  } else if(/\/api\/alerts\/\d+\/read/.test(url)){
    const id=Number(url.match(/\/api\/alerts\/(\d+)\/read/)[1]);
    window.__alertRows.find(row=>row.id===id).is_read=1;
  } else if(/\/api\/alerts\/\d+$/.test(url)&&method==='DELETE'){
    const id=Number(url.match(/\/api\/alerts\/(\d+)$/)[1]);
    window.__alertRows=window.__alertRows.filter(row=>row.id!==id);
  } else if(url.includes('/api/alerts/status')){
    data={success:true,total:window.__alertRows.length,unread:window.__alertRows.filter(x=>!x.is_read).length};
  } else if(url.includes('/api/alerts')){
    const query=new URL(url,'https://example.invalid').searchParams;
    const unread=query.get('unread')==='true';
    const rows=window.__alertRows.filter(x=>!unread || !x.is_read);
    const offset=Number(query.get('offset')||0),limit=Number(query.get('limit')||24);
    data={success:true,items:rows.slice(offset,offset+limit),total:rows.length,limit,offset,
      status:{unread:window.__alertRows.filter(x=>!x.is_read).length,total:window.__alertRows.length}};
  } else data={success:true,status:'stopped',stream:{status:'stopped'}};
  return {ok:status===200,status,json:async()=>data};
};
</script>'''


def run():
    screenshot=ROOT/'PlateVision_0.13.5_Alarmzentrale.png'
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(executable_path=shutil.which('chromium') or '/usr/bin/chromium',
                                          headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        for width in (320,390,1380):
            page=browser.new_page(viewport={'width':width,'height':940})
            errors=[]
            page.on('pageerror',lambda exc:errors.append(str(exc)))
            html=render('alerts').replace('</head>',FAKE_SCRIPT+'</head>',1).replace('0.13.0','0.13.5')
            page.set_content(html,wait_until='domcontentloaded')
            page.locator('.pv-alert-card').first.wait_for(timeout=15000)
            assert page.locator('.pv-alert-card').count()==3
            assert page.locator('#pvAlertUnread').inner_text()=='2'
            assert page.locator('.pv-alert-card img').count()==0
            assert page.evaluate('window.__xss===undefined')
            assert page.locator('a[data-nav-link="alerts"]').get_attribute('href')==PREFIX+'/alerts'
            assert page.locator('.pv-alert-card a[href^="'+PREFIX+'/history"]').count()==3
            page.locator('#pvAlertSettings summary').click()
            page.locator('#pvAlertsMinimum').fill('80')
            page.locator('#pvAlertsCooldown').fill('300')
            page.locator('#pvAlertsSaveSettings').click()
            page.wait_for_function('window.__fakeSettings.cooldown_seconds===300')
            assert page.evaluate('window.__fakeSettings.min_confidence')==.8
            page.locator('#pvAlertsUnreadOnly').click()
            page.wait_for_function("document.querySelectorAll('.pv-alert-card').length===2")
            page.locator('.pv-alert-card').first.get_by_role('button',name='Als gelesen').click()
            page.wait_for_function("document.querySelectorAll('.pv-alert-card').length===1")
            assert page.locator('#pvAlertUnread').inner_text()=='1'
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+3')
            assert page.evaluate('window.__requested.some(r=>r.url.includes("'+PREFIX+'/api/alerts"))')
            assert not errors, errors
            if width == 1380:
                page.locator('#pvAlertsAll').click()
                # Browser security assertions use intentionally hostile text;
                # publish a neutral visual preview without that test fixture.
                page.locator('.pv-alert-card').first.wait_for()
                page.locator('.pv-alert-plate').first.evaluate('(el) => { el.textContent = "ZH 101"; }')
                page.locator('.pv-alert-subtitle').first.evaluate('(el) => { el.textContent = "Watchlist-Treffer"; }')
                page.locator('#pvAlertSettings summary').click()
                page.screenshot(path=str(screenshot),full_page=True)
            page.close()
        browser.close()
        print('PlateVision 0.13.5 browser: 320/390/1380px, filtering, read state, settings, XSS and Ingress: OK')

if __name__=='__main__':
    run()
