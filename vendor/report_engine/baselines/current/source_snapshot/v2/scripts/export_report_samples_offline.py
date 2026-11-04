"""Copy existing review HTML and derived images into a portable, offline folder.

The scientific report, evidence wording, and source experiment are not edited.
Data-URI WSI overviews are deduplicated into assets alongside the existing ROIs.
"""

import argparse
import base64
import hashlib
import io
import json
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

from lxml import html
from PIL import Image


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.svg'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_inspector(script):
    match = re.search(r'const INSPECTOR\s*=\s*', script)
    if not match:
        return None
    value, length = json.JSONDecoder().raw_decode(script[match.end():])
    return match.end(), length, value


def strip_paths(text):
    return re.sub(r'/(?:local|data1|home)/[^\s\"\'<>\\]+',
                  lambda m: Path(m.group()).name, text)


def visible_text(raw):
    doc = html.fromstring(raw)
    for node in doc.xpath('//script | //style'):
        node.getparent().remove(node)
    return strip_paths(doc.text_content())


def clinical_sections(doc):
    selector = '//*[contains(concat(" ",normalize-space(@class)," ")," report-row-content ")]'
    return [e.text_content() for e in doc.xpath(selector)]


def export_case(case_id, label, source_root, staging, destination):
    source = source_root / case_id / 'report.html'
    source_bytes = source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    original = source_bytes.decode('utf-8')
    result_path = source.parent / 'pipeline_result.json'
    result = json.loads(result_path.read_text())
    assert result['case_id'] == case_id
    original_files = {str(p): digest(p) for p in (result_path, source.parent/'generated_long_report.txt') if p.exists()}
    target = staging / label
    assets_dir = target / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=False)
    asset_map = {}

    def resource(value):
        if value in asset_map:
            return asset_map[value]['relative']
        parsed = urlsplit(value)
        if value.startswith('data:image/'):
            header, encoded = value.split(',', 1)
            assert ';base64' in header
            mime = header[5:].split(';')[0]
            extension = {'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp'}.get(mime)
            assert extension, mime
            content = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
            sha = hashlib.sha256(content).hexdigest()
            asset = assets_dir / (sha[:20] + extension)
            if not asset.exists():
                asset.write_bytes(content)
            record = {'sha256': sha, 'size': len(content), 'kind': 'embedded_wsi_overview'}
        else:
            assert not parsed.scheme and not parsed.netloc, f'External resource: {value}'
            path = (source.parent / unquote(parsed.path)).resolve()
            assert path.suffix.lower() in IMAGE_SUFFIXES, f'Not a derived image: {path}'
            assert path.is_file(), f'Missing resource: {path}'
            assert path.stat().st_size < 100 * 1024 * 1024
            sha = digest(path)
            asset = assets_dir / (sha[:20] + path.suffix.lower())
            if not asset.exists():
                shutil.copyfile(path, asset)
            assert digest(asset) == sha
            record = {'sha256': sha, 'size': asset.stat().st_size, 'kind': 'copied_image'}
        record['relative'] = 'assets/' + asset.name
        asset_map[value] = record
        return record['relative']

    def rewrite(value, key=''):
        if isinstance(value, dict):
            return {k: rewrite(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [rewrite(v, key) for v in value]
        if not isinstance(value, str) or not value:
            return value
        if value.startswith('data:image/'):
            return resource(value)
        if key == 'slide_path':
            return 'Original WSI (not included in this offline package)'
        if len(value) < 1500 and '\n' not in value:
            if Path(value).suffix.lower() in IMAGE_SUFFIXES and ('/' in value or key in {'image', 'he_image', 'ov_image'}):
                return resource(value)
            if value.startswith(('/local/', '/data1/', '/home/', '../')):
                return Path(value).name
        return strip_paths(value)

    doc = html.fromstring(original)
    inspector = None
    for node in doc.xpath('//script'):
        assert not node.get('src')
        script = node.text or ''
        parsed = parse_inspector(script)
        if parsed:
            start, length, inspector = parsed
            encoded = json.dumps(rewrite(inspector), ensure_ascii=True).replace('</', '<\\/')
            node.text = script[:start] + encoded + script[start + length:]
    assert inspector
    wsis = {k:v for k,v in inspector.items() if re.fullmatch(r'WSI-\d+', k)}
    assert wsis, 'A real WSI overview is required'
    for wsi in wsis.values():
        assert wsi.get('image')
        assert (wsi.get('details') or {}).get('roi_boxes'), 'WSI ROI coordinates are required'
    for node in doc.iter():
        for attr in ('src', 'href', 'poster', 'data-src'):
            value = node.get(attr)
            if value and not value.startswith('#'):
                node.set(attr, resource(value))
        assert not node.get('srcset')
        if node.tag not in ('script', 'style'):
            if node.text:
                node.text = strip_paths(node.text)
            if node.tail:
                node.tail = strip_paths(node.tail)
    rendered = html.tostring(doc, encoding='unicode', doctype='<!doctype html>')
    assert not re.search(r'(?:src|href)=[\"\'](?:https?:|file:|/|\.\.)', rendered)
    assert not re.search(r'(?:fetch\s*\(|XMLHttpRequest|https?://)', rendered)
    assert not re.search(r'/(?:local|data1|home)/', rendered)
    before = clinical_sections(html.fromstring(original))
    assert len(before) >= 7 and before == clinical_sections(doc), 'Report wording changed'
    assert visible_text(original) == visible_text(rendered), 'Visible evidence wording changed'
    assert source.read_bytes() == source_bytes, 'Source report changed during export'
    for path, sha in original_files.items():
        assert digest(Path(path)) == sha, 'Original scientific artifact changed'
    (target/'report.html').write_text(rendered)
    return {'case_id': case_id, 'folder': label, 'source_html': str(source), 'source_sha256': source_sha,
            'report': str(destination/label/'report.html'), 'report_sha256': digest(target/'report.html'),
            'clinical_and_evidence_text_unchanged': True, 'source_artifact_hashes': original_files,
            'wsi_count': len(wsis), 'roi_count': len([k for k in inspector if re.fullmatch(r'FIG-\d+', k)]),
            'images': list({v['relative']:v for v in asset_map.values()}.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', required=True, help='CASE_ID=folder_name')
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise SystemExit('Refusing to overwrite an existing delivery directory')
    cases = [value.split('=', 1) for value in args.case]
    for _, label in cases:
        assert Path(label).name == label and label not in ('.', '..')
    with tempfile.TemporaryDirectory(prefix='pathxai_offline_export_') as directory:
        staging = Path(directory)
        audits = [export_case(case, label, args.source_root, staging, args.destination) for case, label in cases]
        shutil.copytree(staging, args.destination)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audits, indent=2))
    for row in audits:
        print(json.dumps({k:v for k,v in row.items() if k not in ('images', 'source_artifact_hashes')}), flush=True)


if __name__ == '__main__':
    main()
