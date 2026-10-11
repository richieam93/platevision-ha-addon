#!/usr/bin/env python3
"""Render each Jinja page and check inline JS syntax using Node when available."""
from __future__ import annotations
import ast
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / 'platevision/src/templates'


def main():
    try:
        from jinja2 import Environment, FileSystemLoader, ChainableUndefined
    except ImportError:
        print('Frontend check: Jinja2 unavailable, skipped')
        return
    source = ast.parse((ROOT/'platevision/src/app.py').read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'ConfigManager')
    default_node = next(n.value for n in cls.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_CONFIG' for t in n.targets))
    config = ast.literal_eval(default_node)
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), undefined=ChainableUndefined, autoescape=True)
    ctx = {
        'config': config, 'app_config': config, 'lang': 'de', 'page': 'dashboard',
        'app_version': '0.13.9', 'app_license': 'AGPL-3.0-only',
        'people_menu_enabled': True, 'stream_status': {'status': 'stopped'},
        'entries': [], 'stats': {},
        't': lambda key: key, 'pv_prefix': '',
        'pv_url_for': lambda endpoint, **kwargs: '/static/' + kwargs['filename'] if endpoint == 'static' else '/' + endpoint,
        'url_for': lambda endpoint, **kwargs: '/static/' + kwargs['filename'] if endpoint == 'static' else '/' + endpoint,
    }
    node = shutil.which('node')
    checked = 0
    with tempfile.TemporaryDirectory(prefix='pv-js-') as directory:
        for template_path in sorted(TEMPLATES.glob('*.html')):
            html = env.get_template(template_path.name).render(ctx)
            assert '<html' in html and '</html>' in html, template_path.name
            inline_scripts = re.findall(r'<script(?:\s+[^>]*)?>(.*?)</script\s*>', html, flags=re.I|re.S)
            for script_index, script in enumerate(inline_scripts):
                if not script.strip() or not node:
                    continue
                generated = Path(directory)/f'{template_path.stem}_{script_index}.js'
                generated.write_text(script, encoding='utf-8')
                result = subprocess.run([node, '--check', str(generated)], capture_output=True, text=True)
                if result.returncode:
                    raise AssertionError(f'{template_path.name} inline JS {script_index}: {result.stderr}')
                checked += 1
    if node:
        for asset in sorted((ROOT/'platevision/src/static').rglob('*.js')):
            result = subprocess.run([node, '--check', str(asset)], capture_output=True, text=True)
            if result.returncode:
                raise AssertionError(f'{asset.name}: {result.stderr}')
            checked += 1
    print(f'PlateVision frontend: {len(list(TEMPLATES.glob("*.html")))} gerenderte Seiten, {checked} JS-Syntaxprüfungen: OK')


if __name__ == '__main__':
    main()
