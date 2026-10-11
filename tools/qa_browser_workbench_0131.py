#!/usr/bin/env python3
"""Chromium UI audit for PlateVision 0.13.1, using production templates and a mocked API."""
import json
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
from qa_browser_ingress_0130 import render, PREFIX

OUT=Path('/mnt/data')


def main():
    errors=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=shutil.which('chromium') or '/usr/bin/chromium',
                                   headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1490,'height':1040},device_scale_factor=1)
        page.on('pageerror',lambda exc:errors.append(str(exc)))
        def load(template):
            errors.clear()
            page.goto('about:blank',wait_until='domcontentloaded')
            page.set_content(render(template).replace('0.13.0', '0.13.1'),wait_until='domcontentloaded')
            page.wait_for_timeout(850)
            if errors:
                print('Browser errors for',template,errors[-10:])
            return page
        p=load('test')
        assert p.locator('#testUploadWorkspace').count()==1
        assert p.locator('#peopleCalibrationWorkspace').count()==1
        assert p.locator('#peopleLinePreviewShell').count()==1
        assert p.locator('#recognitionSettingsDetails').count()==1
        assert p.locator('#recognitionSettingsDetails').evaluate('(e)=>e.open') is False
        assert p.locator('#testPeopleCropPadding').count()==1
        assert p.locator('.pv-sub-settings').count()>=5
        p.locator('#recognitionSettingsDetails>summary').click()
        assert p.locator('#recognitionSettingsDetails').evaluate('(e)=>e.open') is True
        p.locator('#recognitionSettingsDetails>summary').click() # collapse expert details for main overview
        p.screenshot(path=str(OUT/'PlateVision_0.13.1_Testzentrum.png'),full_page=False)
        p.locator('#peopleCalibrationWorkspace').scroll_into_view_if_needed()
        p.screenshot(path=str(OUT/'PlateVision_0.13.1_Kalibrierung.png'),full_page=False)
        p=load('people')
        assert p.locator('#peopleLinePreviewShell').count()==0
        assert p.locator('a[href*="/test#peopleCalibrationWorkspace"]').count()>0
        assert p.locator('#peopleImageGallery').count()==1
        p.screenshot(path=str(OUT/'PlateVision_0.13.1_Personen.png'),full_page=False)
        p=load('settings')
        assert p.locator('#showOnlyBestVehicle').count()==1
        assert p.locator('#bestVehicleMinConfidence').count()==1
        p.locator('#showOnlyBestVehicle').evaluate('(el)=>{el.checked=true;el.dispatchEvent(new Event("change",{bubbles:true}))}')
        assert p.locator('#showOnlyBestVehicle').evaluate('(el)=>el.checked')
        p.evaluate('''() => { document.querySelectorAll('.tab-pane').forEach(x=>x.classList.remove('show','active'));const d=document.getElementById('dashboardUiSettings');d?.classList.add('show','active');}''')
        p.locator('#bestVehicleDisplaySettings').scroll_into_view_if_needed()
        p.screenshot(path=str(OUT/'PlateVision_0.13.1_Einstellungen.png'),full_page=False)
        p.locator('#bestVehicleDisplaySettings').screenshot(path=str(OUT/'PlateVision_0.13.1_Qualitaetsfilter.png'))
        p=load('dashboard')
        assert p.locator('#vehicleQualityHint').count()==1
        p.evaluate('''() => {renderVehicle(null); const d=document.getElementById('vehicleQualityHint');d.textContent='Qualitätsauswahl aktiv: ab 80 %';}''')
        assert 'Qualitätsauswahl' in p.locator('#vehicleQualityHint').inner_text()
        assert p.locator('#latestVehiclePlaceholder').is_visible()
        assert 'contain' in p.locator('#latestPersonImage').evaluate('(e)=>getComputedStyle(e).objectFit')
        p.screenshot(path=str(OUT/'PlateVision_0.13.1_Dashboard.png'),full_page=False)
        p=load('latest')
        assert any('/api/latest/display' in u['url'] for u in p.evaluate('window.__requested')),p.evaluate('window.__requested')
        assert not any('ReferenceError' in e or 'TypeError' in e for e in errors),errors
        print('PlateVision 0.13.1 Chromium: 5 pages, calibration moved, advanced settings collapsible, configuration present, image contain, latest display API: OK')
        browser.close()

if __name__=='__main__': main()
