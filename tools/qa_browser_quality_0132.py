#!/usr/bin/env python3
"""Offline Chromium checks for PlateVision 0.13.2 (real production templates).

The mocked API exercises UI behavior, not a real Supervisor/RTSP deployment.
"""
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
from qa_browser_ingress_0130 import render, PREFIX


def run():
    errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(
            executable_path=shutil.which('chromium') or '/usr/bin/chromium',
            headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1250,'height':850})
        page.on('pageerror',lambda exc:errors.append(str(exc)))
        page.set_content(render('settings'),wait_until='domcontentloaded')
        check=page.locator('#bestVehiclePreferSharpness')
        assert check.count()==1 and check.is_checked()
        assert page.locator('#bestVehicleMinConfidence').count()==1
        check.evaluate('(el)=>{el.checked=false;el.dispatchEvent(new Event("change",{bubbles:true}))}')
        assert not check.is_checked()
        assert not errors,errors

        errors.clear()
        page.goto('about:blank', wait_until='domcontentloaded')
        page.set_content(render('latest'),wait_until='domcontentloaded')
        page.evaluate('''() => {
            window.fetch = async (input) => {
                window.__requested.push({url:String(input), method:'GET'});
                return {ok:true,status:200,json:async()=>({
                    id:'qa-event', plate_text:'ZH 123', timestamp:'2026-10-10T11:00:00',
                    confidence:.94,vehicle_type:'Auto',source:'video_upload',
                    vehicle_image_url:'/api/history/qa-event/image/vehicle',
                    plate_image_url:'/api/history/qa-event/image/plate'
                })};
            };
            lastDetectionId=null;
        }''')
        page.evaluate('loadLatestDetection()')
        page.locator('#latestAge').wait_for(timeout=5000)
        assert page.locator('.latest-image-box img').count()==2
        assert page.locator('.latest-image-box img').first.get_attribute('src')==PREFIX+'/api/history/qa-event/image/vehicle'
        assert page.evaluate('window.__requested.some(x=>x.url.includes("/api/latest/display?images=0"))')
        # Preserve the DOM and image elements for the unchanged record.
        page.evaluate('document.getElementById("latestContent").dataset.keep="sentinel"')
        page.evaluate('loadLatestDetection()')
        page.wait_for_timeout(180)
        assert page.locator('#latestContent').get_attribute('data-keep')=='sentinel'
        assert not errors,errors
        print('PlateVision 0.13.2 Chromium: settings toggle, compact Ingress image URLs, stable refresh: OK')
        browser.close()

if __name__=='__main__': run()
