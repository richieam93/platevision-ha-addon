#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Browser regression: cancelled previews and asynchronous image decoding.

No camera/network access required; synthetic JPEGs exercise the actual live UI.
"""
import base64
import shutil
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright
from qa_browser_ingress_0130 import render

ROOT = Path(__file__).resolve().parents[1]

def image_b64(color):
    img = np.full((75, 100, 3), color, dtype=np.uint8)
    ok, data = cv2.imencode('.jpg', img)
    assert ok
    return base64.b64encode(data.tobytes()).decode()

JS = '''<script>
window.__a='__A__';
window.__b='__B__';
window.__heldResolve=null;
window.__previewCount=0;
window.__realFetch=window.fetch;
window.__realImage=window.Image;
window.__jpg=(base64) => new Response(Uint8Array.from(atob(base64), c => c.charCodeAt(0)),
    {status:200,headers:{'Content-Type':'image/jpeg'}});
window.fetch=async (url, opts={}) => {
 const path=String(url);
 if(path.includes('/api/stream/status')) return {ok:true,status:200,json:async()=>({
     status:'running',connected:true,fps:9.1,preview_source:'raw',
     last_frame_age_seconds:0.1,last_analysis_frame_age_seconds:12.3,
     reconnect_count:2,resolution:{width:1280,height:960}})};
 if(path.includes('/api/stream/preview')) {
    window.__previewCount++;
    if(window.__previewCount===1) {
      return new Promise(resolve=>{window.__heldResolve=resolve}); // ignores abort on purpose
    }
    return window.__jpg(window.__b);
 }
 return window.__realFetch(url,opts);
};
</script>'''.replace('__A__',image_b64((5, 5, 225))).replace('__B__',image_b64((5, 225, 5)))


def run():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=shutil.which('chromium') or '/usr/bin/chromium',
                            headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
        try:
            page = browser.new_page(viewport={'width': 390, 'height': 900})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.set_content(render('live').replace('</head>', JS + '</head>', 1), wait_until='domcontentloaded')
            page.wait_for_function('window.__previewCount === 1 && window.__heldResolve && isStreaming', timeout=12000)
            # Stop this preview and create a new session before the first
            # HTTP request (whose mock ignores AbortSignal) returns.
            page.evaluate("stopStreamFeed(); startSingleFramePreview('Neue Sitzung');")
            page.wait_for_function("window.__previewCount >= 2 && document.querySelector('#streamVideo').naturalWidth > 0", timeout=10000)
            assert page.locator('#previewSourceValue').inner_text() == 'Aktuelles Kamerabild'
            assert '12' in page.locator('#analysisAgeValue').inner_text()
            old_source = page.locator('#streamVideo').get_attribute('src')
            page.evaluate('window.__heldResolve(window.__jpg(window.__a))')
            page.wait_for_timeout(170)
            assert page.locator('#streamVideo').get_attribute('src') == old_source, \
                'Old HTTP response replaced new camera preview after restart'
            assert not errors, errors
            assert page.evaluate("""() => {
                const firstTimer = uptimeInterval;
                startUptimeCounter();
                if (!firstTimer || firstTimer !== uptimeInterval) return false;
                stopUptimeCounter();
                return uptimeInterval === null && streamStartTime === null;
            }"""), 'Stop/restart must not duplicate or reuse old uptime timers'
            # Decode may never finish (corrupt browser codec, tab suspension).
            # Stopping the stream must cancel it, and the next preview must work.
            page.evaluate('''() => {
              stopStreamFeed();
              window.__decodeWaiting=false;
              window.Image = class {
                set src(value) { window.__decodeWaiting=true; this.value=value; }
                get src() { return this.value; }
              };
              startSingleFramePreview('Decoder wartet');
            }''')
            page.wait_for_function('window.__decodeWaiting === true', timeout=10000)
            page.evaluate('''() => {
                stopStreamFeed();
                window.Image=window.__realImage;
                startSingleFramePreview('Decoder neu gestartet');
            }''')
            page.wait_for_function("document.querySelector('#streamVideo').naturalWidth > 0 && previewLastOk > 0", timeout=12000)
            assert page.locator('#offlineOverlay').evaluate('(el) => el.classList.contains("hidden")')
            assert not errors, errors
            page.close()
            print('  OK Browser: old response ignored; hung JPEG decoder aborts; session resumes at mobile width')
        finally:
            browser.close()

if __name__ == '__main__':
    run()
