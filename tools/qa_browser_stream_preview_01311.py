#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Chromium test: black HA Ingress MJPEG -> working JPEG fallback, errors visible."""
import base64
import shutil
from pathlib import Path
import cv2
import numpy as np
from playwright.sync_api import sync_playwright
from qa_browser_ingress_0130 import render, PREFIX

ROOT=Path(__file__).resolve().parents[1]
IMAGE=ROOT/'PlateVision_0.13.11_Ingress_Vorschau.png'

image=np.zeros((240,426,3),dtype=np.uint8)
image[:]=(38,64,45)
cv2.putText(image,'TESTBILD - SIMULIERT',(20,110),cv2.FONT_HERSHEY_SIMPLEX,.7,(244,244,244),2)
cv2.rectangle(image,(18,135),(410,220),(92,190,233),3)
ok,data=cv2.imencode('.jpg',image)
assert ok
JPEG_B64=base64.b64encode(data.tobytes()).decode()

MOCK='''<script>
window.__previewRequests=[];
window.__previewMode='ok';
window.__mockJpeg='__JPEG__';
window.__origFetch=window.fetch;
window.fetch=async (input,opts={})=>{
 const url=String(input);window.__previewRequests.push(url);
 if(url.includes('/api/stream/status')) return {ok:true,status:200,json:async()=>({status:'running',connected:true,fps:12.3,reconnect_count:2,resolution:{width:1280,height:960},last_frame_age_seconds:0.4})};
 if(url.includes('/api/stream/preview')) {
    if(window.__previewMode==='missing') return {ok:true,status:204};
    if(window.__previewMode==='bad') return {ok:false,status:503};
    const bytes=Uint8Array.from(atob(window.__mockJpeg),c=>c.charCodeAt(0));
    return new Response(bytes,{status:200,headers:{'Content-Type':'image/jpeg'}});
 }
 return window.__origFetch(input,opts);
};
</script>'''.replace('__JPEG__',JPEG_B64)


def run():
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=shutil.which('chromium') or '/usr/bin/chromium',headless=True,
                                   args=['--no-sandbox','--disable-dev-shm-usage'])
        try:
            for width in (390,1560):
                page=browser.new_page(viewport={'width':width,'height':920})
                errors=[]
                page.on('pageerror',lambda err:errors.append(str(err)))
                html=render('live').replace('</head>',MOCK+'</head>',1)
                page.set_content(html,wait_until='domcontentloaded')
                page.wait_for_function('''() => document.querySelector('#streamVideo').naturalWidth > 0 &&
                     document.querySelector('#streamVideo').src.startsWith('blob:')''',timeout=15000)
                assert page.locator('#offlineOverlay').evaluate('(e)=>e.classList.contains("hidden")')
                assert 'Einzelbildvorschau' in page.locator('#previewTransport').inner_text()
                assert page.locator('#fpsValue').inner_text()=='12.3'
                page.wait_for_function('document.getElementById("headerConnectionText").textContent === "Verbunden"', timeout=5000)
                assert '1280' in page.locator('#resolutionValue').inner_text()
                req=page.evaluate('window.__previewRequests')
                assert any(PREFIX+'/api/stream/preview' in x for x in req)
                assert not any('/api/stream/feed' in x for x in req), 'Ingress must not request blocking MJPEG feed'
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+5')
                assert not errors,errors
                if width==1560:
                    page.locator('#streamMain').screenshot(path=str(IMAGE))
                page.close()
            # A real backend might not have a frame yet. Never show a black,
            # unlabelled rectangle when the single-frame endpoint returns 204.
            page=browser.new_page(viewport={'width':1280,'height':900})
            errors=[]
            page.on('pageerror',lambda err:errors.append(str(err)))
            html=render('live').replace('</head>',MOCK+'</head>',1).replace("window.__previewMode='ok'","window.__previewMode='missing'")
            page.set_content(html,wait_until='domcontentloaded')
            page.locator('#offlineTitle').get_by_text('Warte auf Kamerabilder').wait_for(timeout=10000)
            assert page.locator('#offlineOverlay').is_visible()
            assert not errors,errors
            page.close()
            # Direct access (not Ingress) tries MJPEG and switches to JPEG
            # when img.onerror fires (mocked URL is unreachable by design).
            page=browser.new_page(viewport={'width':1280,'height':900})
            errors=[]
            page.on('pageerror',lambda err:errors.append(str(err)))
            html=render('live').replace('const isIngressView = true;', 'const isIngressView = false;',1).replace('</head>',MOCK+'</head>',1)
            page.set_content(html,wait_until='domcontentloaded')
            page.wait_for_function('''() => document.querySelector('#streamVideo').naturalWidth > 0 &&
                        document.querySelector('#streamVideo').src.startsWith('blob:')''',timeout=16000)
            assert any('/api/stream/feed' in x for x in page.evaluate('performance.getEntriesByType("resource").map(x=>x.name)')) or 'MJPEG' in page.locator('#previewTransport').inner_text()
            assert not errors,errors
            page.close()
        finally:
            browser.close()
    print('  OK Chromium: Ingress JPEG at 390/1560px, 204 waiting overlay, direct MJPEG fallback')

if __name__=='__main__':run()
