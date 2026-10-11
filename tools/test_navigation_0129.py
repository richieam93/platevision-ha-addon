#!/usr/bin/env python3
"""0.12.9 grouped navigation: rendered markup, routes, localization, feature flag."""
import ast
from html.parser import HTMLParser
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, ChainableUndefined

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT/'platevision/src/app.py'
TREE = ast.parse(APP.read_text(encoding='utf-8'))
CONFIG_CLASS = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == 'ConfigManager')
DEFAULT = ast.literal_eval(next(node.value for node in CONFIG_CLASS.body if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_CONFIG' for t in node.targets)))
TRANSLATIONS = ast.literal_eval(next(node.value for node in TREE.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRANSLATIONS' for t in node.targets)))
ROUTES = {node.name for node in TREE.body if isinstance(node, (ast.FunctionDef,ast.AsyncFunctionDef))}
PAGES = {
 'dashboard':'dashboard','live':'live_view','latest':'latest_detection_page',
 'alerts':'watchlist_alerts_page',
 'history':'history','search':'search_page','statistics':'statistics_page',
 'people':'people_page','people_gallery':'people_gallery_page','test':'test_page',
 'rtsp':'rtsp_settings','settings':'settings','diagnostics':'diagnostics_page','legal':'legal_page'
}
REQUIRED_SECTIONS = {'overview', 'plates', 'persons', 'tools', 'system'}
ENV=Environment(loader=FileSystemLoader(ROOT/'platevision/src/templates'), undefined=ChainableUndefined, autoescape=True)

class NavParser(HTMLParser):
 def __init__(self):
  super().__init__(); self.entries={};self.group_ids=[];self.anchors=[];self.ids=[];self.active=[];self.groups_open=[];self.links=[];self.pin_ids=[]
 def handle_starttag(self, tag, attrs):
  a=dict(attrs)
  if 'id' in a and a['id'] and a['id'].startswith(('pvNav','sidebarNav','pvPage')): self.ids.append(a['id'])
  if a.get('data-nav-group'): self.group_ids.append(a['data-nav-group'])
  if a.get('data-nav-page') and tag=='div': self.entries[a['data-nav-page']]=True
  if a.get('data-nav-link'):
   self.links.append((a['data-nav-link'],a.get('href')))
   if a.get('aria-current')=='page': self.active.append(a['data-nav-link'])
  if a.get('data-nav-pin'): self.pin_ids.append(a['data-nav-pin'])
  if tag=='a': self.anchors.append(a.get('href',''))

def render(page,language='de',people=True):
 t=lambda name: TRANSLATIONS[language].get(name,TRANSLATIONS['de'].get(name,name))
 return ENV.get_template('base.html').render(page=page, lang=language,config=DEFAULT, app_config=DEFAULT,
     app_version='0.13.5',pv_prefix='',app_license='AGPL-3.0-only',people_menu_enabled=people,
     stream_status={'status':'stopped'}, t=t,
     pv_url_for=lambda endpoint,**kw: '/static/'+kw['filename'] if endpoint=='static' else '/'+endpoint)

def run():
 for key, endpoint in PAGES.items():
  assert endpoint in ROUTES, f'Unbekannte View-Funktion für {key}: {endpoint}'
 for language in ('de','en','fr','it'):
  for key in PAGES:
   html=render(key,language)
   p=NavParser();p.feed(html)
   assert set(p.group_ids)==REQUIRED_SECTIONS,(key,language,p.group_ids)
   assert set(p.entries)==set(PAGES),(key,language,p.entries.keys())
   assert len(p.links)==len(PAGES),(key,language)
   assert len(p.pin_ids)==len(PAGES),(key,language)
   assert p.active==[key],(key,language,p.active)
   assert len(p.ids)==len(set(p.ids)),f'Duplicate navigation IDs: {key} {language}'
   for page_id, url in p.links:
    assert url=='/'+PAGES[page_id],(page_id,url)
   assert 'navigation.js' in html and 'navigation.css' in html
   assert 'pvPageFavorite' in html
   assert 'pv-context-link' in html
   assert 'sidebarNavSearch' in html
  p=NavParser();p.feed(render('dashboard',language,people=False))
  assert set(p.entries)==set(PAGES)-{'people','people_gallery'},p.entries.keys()
  assert set(p.group_ids)==REQUIRED_SECTIONS-{'persons'}
  assert len(p.pin_ids)==len(PAGES)-2
  print(f'  {language.upper()}: 14 Menüpunkte / 5 Gruppen / 14 aktive Seiten: OK')
 assert 'data-nav-keywords' in render('search')
 print('PlateVision navigation: 56 localization/page combinations and people-disabled fallback: OK')

if __name__=='__main__': run()
