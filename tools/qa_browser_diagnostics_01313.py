#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Chromium test: the 0.13.13 camera queue and dropped-frame metrics render safely."""
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
from qa_browser_ingress_0130 import render, PREFIX

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'PlateVision_0.13.13_Pufferdiagnose.png'

MOCK = '''<script>
const fetchPrevious=window.fetch;
window.fetch=(input,opts={})=>{
  if(String(input).includes('/api/system/health')) return Promise.resolve({ok:true,json:async()=>({
    ok:true, summary:{healthy:true,problems:0,disabled:1},
    camera:{enabled:true,connected:true,fps:15.1,processed_frames:9812,
      detections:43,last_frame_age_seconds:0.3,
      last_analysis_frame_age_seconds:1.5,
      reconnect_count:2,preview_source:'raw',queued_frames:2,discarded_frames:127},
    homeassistant_notifications:{enabled:false,queue:{pending:0,sending:0,failed:0}},
    checks:[{name:'stream_connected',ok:true,detail:'Verbunden'},
      {name:'stream_frames_fresh',ok:true,detail:'Kamerabilder treffen ein'}]
  })});
  return fetchPrevious(input,opts);
};
</script>'''

def run():
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=shutil.which('chromium') or '/usr/bin/chromium',
            headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        try:
            for width in (320,390,1360):
                page=browser.new_page(viewport={'width':width,'height':920})
                errors=[]
                page.on('pageerror',lambda e: errors.append(str(e)))
                page.set_content(render('diagnostics').replace('</head>',MOCK+'</head>',1),wait_until='domcontentloaded')
                page.evaluate('loadDiag()')
                page.wait_for_function('''() => document.querySelector('#cameraMonitor').textContent.includes('127')''',timeout=10000)
                text=page.locator('#cameraMonitor').inner_text()
                assert 'Bilder im Analysepuffer' in text,text
                assert 'Übersprungene Pufferbilder' in text,text
                assert '127' in text and '2' in text,text
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+5')
                assert not errors,errors
                if width==1360:
                    page.evaluate('''() => {
                      const block=document.querySelector('#cameraMonitor');
                      const line=document.createElement('p');
                      line.textContent='SIMULIERTE KAMERADATEN – keine echte Aufnahme';
                      line.className='small text-secondary';
                      block.parentElement.insertBefore(line,block);
                    }''')
                    page.locator('#cameraMonitor').screenshot(path=str(OUTPUT))
                page.close()
        finally:
            browser.close()
    print('  OK Chromium: frame queue and dropped counts on mobile/desktop without layout overflow')

if __name__ == '__main__':run()
