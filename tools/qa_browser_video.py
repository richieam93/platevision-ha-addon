"""Optional Chromium/Playwright smoke test; requires Jinja2, playwright, Chromium."""
from pathlib import Path
import os, shutil
import ast, re, json
from jinja2 import Environment, FileSystemLoader, ChainableUndefined
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1]
src=ast.parse((root/'platevision/src/app.py').read_text(encoding='utf-8'))
config=ast.literal_eval(next(n.value for c in src.body if isinstance(c, ast.ClassDef) and c.name=='ConfigManager' for n in c.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_CONFIG' for t in n.targets)))
env=Environment(loader=FileSystemLoader(str(root/'platevision/src/templates')), undefined=ChainableUndefined, autoescape=True)
ctx={'config':config,'app_config':config,'lang':'de','page':'test','app_version':'0.12.5','app_license':'AGPL-3.0-only','people_menu_enabled':True,'stream_status':{'status':'stopped'},'t':lambda k:k,'pv_prefix':'','pv_url_for':lambda endpoint,**kw:'/static/'+kw['filename'] if endpoint=='static' else '/'+endpoint}
html=env.get_template('test.html').render(ctx)
static=root/'platevision/src/static'
html=re.sub(r'<link\s+[^>]*?href="/static/([^"]+\.css)"[^>]*>',lambda m:'<style>'+ (static/m.group(1)).read_text()+'</style>',html)
html=re.sub(r'<script\s+src="/static/([^"]+\.js)"[^>]*></script>',lambda m:'<script>'+(static/m.group(1)).read_text()+'</script>',html)
html=re.sub(r'<link\s+[^>]*?href="https://[^>]+>', '', html)
html=re.sub(r'<script\s+src="https://[^>]+></script>', '', html)
mock='''<script>
window.io=()=>({on(){},emit(){}});
window.__pvCalls=[];
window.fetch=async (url, options={})=>{
  window.__pvCalls.push({url:String(url), method: options.method||'GET'});
  const key=String(url).split('?')[0];
  if (key.endsWith('/cancel')) return {ok:true,status:202,json:async()=>({success:true,status:'cancelling'})};
  if (key==='/api/process/jobs') return {ok:true,status:200,json:async()=>({})};
  if (key==='/api/config') return {ok:true,status:200,json:async()=>({success:true,config:__CONFIG__})};
  if (key==='/api/test/settings') return {ok:true,status:200,json:async()=>({success:true,config:__CONFIG__,settings:__CONFIG__})};
  if (key==='/api/models/available') return {ok:true,status:200,json:async()=>([])};
  return {ok:true,status:200,json:async()=>({success:true,config:__CONFIG__,models:[], people:[],status:'stopped'})};
};
</script>'''.replace('__CONFIG__',json.dumps(config,ensure_ascii=False))
html=html.replace('<head>','<head>'+mock)
with sync_playwright() as p:
  browser=p.chromium.launch(executable_path=os.environ.get('PV_CHROMIUM_PATH') or shutil.which('chromium') or '/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
  page=browser.new_page(viewport={'width':1400,'height':940})
  errors=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.set_content(html,wait_until='domcontentloaded')
  page.wait_for_timeout(500)
  job='bc000001-0000-4000-a000-000000000001'
  mal='<img src=x onerror=window.__pvXss=1>'
  page.evaluate('''([id, mal]) => displayJobs({[id]:{id, filename:mal, status:'processing',progress:46,total_frames:120,current_frame:55,detections_count:4,created_at:'2026-10-09T16:00:00'}})''',[job,mal])
  assert page.locator('#jobsList .job-card').count()==1
  assert page.locator('#jobsList img').count()==0
  assert page.locator('#jobsList').inner_text().count(mal)==1
  assert page.evaluate('window.__pvXss') is None
  page.evaluate('window.confirm=()=>true')
  page.locator('#jobsList button[title="Verarbeitung abbrechen"]').click()
  page.wait_for_timeout(200)
  assert page.evaluate('''id=>window.__pvCalls.some(x=>x.url==`/api/process/video/${id}/cancel` && x.method==='POST')''',job)
  page.evaluate('''id=>displayJobs({[id]:{id,filename:'clip.mp4', status:'completed', progress:100,detections_count:5}})''',job)
  assert page.locator(f'a[href="/api/process/video/{job}/output"]').count()==1
  page.evaluate("displayPeopleList([{track_id:'x', event_type:'<img src=x onerror=window.__pvXss=1>',direction:'<svg onload=window.__pvXss=1>',confidence:'0.7'}])")
  assert page.evaluate('window.__pvXss') is None
  assert page.locator('#peopleList svg[onload]').count()==0
  page.evaluate("displayJobs({})")
  page.screenshot(path='PlateVision_0.12.5_Video_Jobs.png',full_page=True)
  mobile=browser.new_page(viewport={'width':390,'height':844})
  mobile_errors=[]
  mobile.on('pageerror',lambda e: mobile_errors.append(str(e)))
  mobile.set_content(html,wait_until='domcontentloaded')
  mobile.wait_for_timeout(450)
  assert mobile.evaluate('document.documentElement.scrollWidth <= innerWidth + 2'), 'test page overflows at 390px'
  mobile.evaluate("document.getElementById('videoProgress').style.display='block';document.getElementById('emptyResult').style.display='none'")
  mobile.evaluate("updateVideoProgress({status:'cancelling',progress:33,current_frame:70,total_frames:200,detections_count:3,recent_plates:['ZH-1234']})")
  assert mobile.locator('#cancelVideoBtn').is_disabled()
  assert 'ZH-1234' in mobile.locator('#videoRecentPlates').inner_text()
  mobile.evaluate('showVideoCancelled()')
  assert 'Abgebrochen' in mobile.locator('#videoJobStatus').inner_text()
  assert mobile.evaluate('document.documentElement.scrollWidth <= innerWidth + 2')
  mobile.screenshot(path='PlateVision_0.12.5_Video_Mobil.png',full_page=True)
  print('Video browser QA OK; desktop JS exceptions:',errors[:8],'mobile JS exceptions:',mobile_errors[:8])
  assert not errors and not mobile_errors, (errors,mobile_errors)
  browser.close()
