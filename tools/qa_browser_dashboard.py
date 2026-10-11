"""Optional Chromium/Playwright smoke test; requires Jinja2, playwright, Chromium."""
import ast, re, json
from pathlib import Path
import os, shutil
from jinja2 import Environment, FileSystemLoader, ChainableUndefined
from playwright.sync_api import sync_playwright
r=Path(__file__).resolve().parents[1]
source=ast.parse((r/'platevision/src/app.py').read_text())
cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='ConfigManager')
config=ast.literal_eval(next(n.value for n in cls.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_CONFIG' for t in n.targets)))
env=Environment(loader=FileSystemLoader(str(r/'platevision/src/templates')),undefined=ChainableUndefined, autoescape=True)
ctx={'config':config,'app_config':config,'lang':'de','page':'dashboard','app_version':'0.12.5','app_license':'AGPL-3.0-only','people_menu_enabled':True,'stream_status':{'status':'stopped'},'t':lambda k:k,'pv_prefix':'','pv_url_for':lambda endpoint,**kw:'/static/'+kw['filename'] if endpoint=='static' else '/'+endpoint}
html=env.get_template('dashboard.html').render(ctx)
static=r/'platevision/src/static'
html=re.sub(r'<link\s+[^>]*?href="/static/([^"]+\.css)"[^>]*>', lambda m:'<style>'+ (static/m.group(1)).read_text()+'</style>',html)
html=re.sub(r'<script\s+src="/static/([^"]+\.js)"[^>]*></script>',lambda m:'<script>'+(static/m.group(1)).read_text()+'</script>',html)
html=re.sub(r'<link\s+[^>]*?href="https://[^>]+>', '', html)
html=re.sub(r'<script\s+src="https://[^>]+></script>', '', html)
payloads={
 '/api/stream/status':{'status':'stopped','connected':False},
 '/api/settings/schema':{'config':config,'schema':{}},
 '/api/dashboard/overview':{'today':{'raw_detections':14,'unique_vehicles':6,'traffic_visits':8,'repeat_vehicles':2,'people_counted':7,'last_hour_detections':3},'stream':{'status':'stopped'},'config':config,'latest_vehicle':{'plate_text':'ZH 45837','vehicle_type':'PKW','vehicle_color':'Blau','plate_country_display':'Schweiz','confidence':.9,'timestamp':'2026-10-09T16:00:00'},'latest_person':{},'latest':[{'plate_text':'ZH 45837','vehicle_type':'PKW','vehicle_color':'Blau','plate_country':'CH','timestamp':'2026-10-09T16:00:00'}]},
 '/api/history/timeline':{'items':[]},
}
stubs="""<script>
window.io=()=>({on(){}});
window.Chart=class {constructor(){}destroy(){}};
const payloads = __PAYLOADS__;
window.fetch=async (url, options={})=>({ok: true,status:200,json:async()=>payloads[String(url).split('?')[0]]||{success:true,status:{status:'stopped'}}});
</script>""".replace('__PAYLOADS__',json.dumps(payloads,ensure_ascii=False))
html=html.replace('<head>', '<head>'+stubs)
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path=os.environ.get('PV_CHROMIUM_PATH') or shutil.which('chromium') or '/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
 page=browser.new_page(viewport={'width':1440,'height':960},device_scale_factor=1)
 page_errors=[]
 page.on('pageerror',lambda err: page_errors.append(str(err)))
 page.set_content(html,wait_until='domcontentloaded')
 page.wait_for_timeout(350)
 assert page.locator('#todayPlateEvents').inner_text()=='14', 'api mock not loaded'
 assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+2'),'desktop width overflow'
 assert page.evaluate("getComputedStyle(document.querySelector('.sidebar')).display")=='flex'
 page.evaluate("showToast('Sicherheit', '<img src=x onerror=alert(1)>', 'info')")
 assert page.locator('.toast-body img').count()==0, 'toast XSS'
 assert '<img src=x' in page.locator('.toast-body').last.inner_text()
 assert page.evaluate("pvSafeImageUrl('javascript:alert(1)')") == ''
 page.locator('.toast').evaluate_all('(items) => items.forEach(item => item.remove())')
 page.screenshot(path='PlateVision_0.12.5_Dashboard_Desktop.png',full_page=True)
 page.evaluate("applyUiConfig({general:{theme:'light'}})")
 assert page.locator('html').get_attribute('data-pv-theme')=='light'
 page.screenshot(path='qa_light.png',full_page=True)
 mobile=browser.new_page(viewport={'width':390,'height':844},device_scale_factor=1)
 mobile.set_content(html,wait_until='domcontentloaded')
 mobile.wait_for_timeout(350)
 assert mobile.evaluate('document.documentElement.scrollWidth <= innerWidth+2'), 'mobile width overflow'
 mobile.locator('#sidebarToggle').click()
 assert mobile.locator('#sidebarToggle').get_attribute('aria-expanded')=='true'
 assert mobile.locator('#sidebar').evaluate('(e)=>e.classList.contains("show")')
 mobile.wait_for_timeout(350)
 rect = mobile.locator('#sidebar').bounding_box()
 assert rect and rect['x'] >= -1 and rect['width'] > 250, f'mobile sidebar not fully visible: {rect}'
 mobile.screenshot(path='qa_mobile_menu.png')
 mobile.keyboard.press('Escape')
 assert mobile.locator('#sidebarToggle').get_attribute('aria-expanded')=='false'
 mobile.wait_for_timeout(350)
 mobile.screenshot(path='PlateVision_0.12.5_Dashboard_Mobil.png',full_page=True)
 for width in (320, 375, 768, 1024):
  probe=browser.new_page(viewport={'width':width,'height':850},device_scale_factor=1)
  probe.set_content(html,wait_until='domcontentloaded')
  probe.wait_for_timeout(100)
  overflow=probe.evaluate('''() => ({width: document.documentElement.scrollWidth, offenders: [...document.querySelectorAll('body *')].filter(el=> { const r=el.getBoundingClientRect(); return r.right>innerWidth+3 && r.width>0 && getComputedStyle(el).position!=='fixed'; }).slice(0,14).map(el=>({tag:el.tagName, id:el.id, cls:String(el.className).slice(0,70),right: Math.round(el.getBoundingClientRect().right), width: Math.round(el.getBoundingClientRect().width)}))})''')
  print('width',width,overflow)
  assert probe.evaluate('document.documentElement.scrollWidth <= innerWidth + 2'), f'overflow at {width}px'
  probe.close()
 print('Browser QA: desktop/mobile layout, responsive menu, XSS escaping, light theme, dashboard API render: OK')
 print('JS errors:',page_errors[:8])
 browser.close()
