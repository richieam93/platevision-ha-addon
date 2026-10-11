#!/usr/bin/env python3
"""PlateVision 0.13.0: offline Ingress URL, safety and template regressions."""
from __future__ import annotations
import ast
import re
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from jinja2 import ChainableUndefined, Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / 'platevision'
SOURCE = ADDON / 'src'
TREE = ast.parse((SOURCE/'app.py').read_text(encoding='utf-8'))

class AppLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href','src','action') and value is not None:
                self.links.append((tag, key, value))


def config_data():
    cls = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'ConfigManager')
    return ast.literal_eval(next(n.value for n in cls.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_CONFIG' for t in n.targets)))


def render(page='dashboard', *, prefix='', template=None):
    cfg = config_data()
    env = Environment(loader=FileSystemLoader(str(SOURCE/'templates')),undefined=ChainableUndefined,autoescape=True)
    config = dict(config=cfg, app_config=cfg,lang='de',page=page,
                  app_version='0.13.0',app_license='AGPL-3.0-only',
                  people_menu_enabled=True,stream_status={'status':'stopped'},
                  entries=[],stats={}, t=lambda name: name, pv_prefix=prefix,
                  pv_url_for=lambda endpoint,**kwargs: prefix + ('/static/'+kwargs['filename'] if endpoint == 'static' else '/'+endpoint))
    return env.get_template(template or page+'.html').render(config)


def test_ingress_helper():
    names={'_platevision_ingress_prefix','_platevision_ui_url','_platevision_url_for'}
    functions=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(functions)==3, names
    request=SimpleNamespace(headers={},remote_addr='172.30.32.2')
    scope={'request':request,'has_request_context':lambda: True,'re':re,
           '_INGRESS_PATH_RE':re.compile(r'/api/hassio_ingress/[A-Za-z0-9_-]{8,256}/?'),
           'url_for':lambda endpoint,**kwargs: '/static/style.css' if endpoint=='static' else '/'+endpoint}
    code=ast.Module(body=functions,type_ignores=[])
    # Only these three pure helpers are executed; never load ML models at test time.
    import inspect
    exec(compile(ast.fix_missing_locations(code), '<ingress helpers>', 'exec'),scope)
    request.headers={'X-Ingress-Path':'/api/hassio_ingress/sessionToken123456'}
    assert scope['_platevision_ingress_prefix']()=='/api/hassio_ingress/sessionToken123456'
    assert scope['_platevision_ui_url']('/api/stream/status')=='/api/hassio_ingress/sessionToken123456/api/stream/status'
    assert scope['_platevision_url_for']('static')=='/api/hassio_ingress/sessionToken123456/static/style.css'
    assert scope['_platevision_ui_url']('https://example.org/')=='https://example.org/'
    assert scope['_platevision_ui_url']('//example.org/')=='//example.org/'
    request.remote_addr='192.168.1.21'
    assert scope['_platevision_ingress_prefix']()=='' # header spoofing must not prefix a direct request
    request.remote_addr='172.30.32.2'
    for header in ('https://evil.test','//evil.test','/api/hassio_ingress/../secret',
                   '/api/hassio_ingress/x\r\nHost:evil','/bad-token'):
        request.headers={'X-Ingress-Path':header}
        assert not scope['_platevision_ingress_prefix'](),header
    request.headers={}
    assert scope['_platevision_ui_url']('/api/stream/status')=='/api/stream/status'


def test_template_paths():
    prefix='/api/hassio_ingress/sessionToken123456'
    routes=['dashboard','live','latest','history','search','statistics','people','people_gallery',
            'test','rtsp','settings','diagnostics','legal']
    file_by_route=dict(dashboard='dashboard',live='live',latest='latest',history='history',
        search='search',statistics='statistics',people='people',people_gallery='people_gallery',
        test='test',rtsp='rtsp_settings',settings='settings',diagnostics='diagnostics',legal='legal')
    for use_prefix in ('',prefix):
        for route in routes:
            html=render(route,prefix=use_prefix,template=file_by_route[route]+'.html')
            assert '<html' in html and '</html>' in html, route
            parser=AppLinks();parser.feed(html)
            # Every static asset, API link and navigation link must keep the
            # browser inside the same HA ingress prefix.
            for tag, attr, url in parser.links:
                if url.startswith('https://') or url.startswith('data:') or url.startswith('#'):
                    continue
                if url.startswith('/'):
                    if use_prefix:
                        assert url.startswith(prefix+'/'),(route,attr,url)
                    else:
                        assert not url.startswith('/api/hassio_ingress/'),(route,attr,url)
            assert ('PV_INGRESS_PREFIX = "'+use_prefix+'"') in html
            assert ('path: pvUrl(\'/socket.io\')') in html
            if route in ('dashboard','search','live','history','people','test','latest'):
                assert (use_prefix+'/api/') in html,(route,use_prefix)
    dashboard=render('dashboard',prefix=prefix)
    assert 'onload="setVehicleImageVisible' not in dashboard
    assert 'onerror="setVehicleImageVisible' not in dashboard
    assert 'img.onload = () => setVehicleImageVisible(true);' in dashboard
    assert 'img.onerror = () => setVehicleImageVisible(false);' in dashboard
    assert 'src="'+prefix+'/api/latest/vehicle/image"' in dashboard
    latest=render('latest',prefix=prefix)
    assert 'value_template: "{{ value_json.plate_text }}"' in latest, 'Jinja should preserve Home Assistant YAML example'
    assert '&lt;PLATEVISION_IP&gt;:8087' in latest, 'The YAML placeholder must survive HTML rendering'
    assert latest.count('camera:\n') == 1, 'YAML must not contain duplicate camera keys'
    assert 'window.location.hostname + ' not in latest, 'Never copy a transient Ingress URL to HA REST config'


def test_manifest():
    import yaml
    config=yaml.safe_load((ADDON/'config.yaml').read_text())
    # Ingress invariants stay valid for all subsequent versions.
    import re
    app_source=(ADDON/'src/app.py').read_text(encoding='utf-8')
    assert config['version']==re.search(r"APP_VERSION = '([^']+)'", app_source).group(1)
    assert config['ingress'] is True
    assert config['ingress_port']==5000
    assert config['ingress_stream'] is True
    assert config['panel_title']=='PlateVision'
    assert config['ports']['5000/tcp'] is None
    assert 'webui' not in config
    assert 'PORT:8087' not in str(config)


def main():
    test_manifest()
    test_ingress_helper()
    test_template_paths()
    print('PlateVision Ingress 0.13.0: manifest, guarded path prefix, 26 rendered pages, static/JS URLs, cache-loaded images: OK')

if __name__=='__main__': main()
