"""End-to-end, resumable recommendations from the whole configured library."""
from __future__ import annotations

from contextlib import nullcontext
import csv
from datetime import datetime, timezone, date
import fcntl
import hashlib
import html
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

from .config import fingerprint, load_config
from .core import REVISION, HashTree, atomic_json, connect, digest, files_under, prepare_state, read_image
from .vision import assessment_signature, local_assessor, validate_assessment


def project_path(state, cfg):
    return state / 'projects' / fingerprint(cfg['paths'])[:16]


def reference_fingerprint(cfg):
    return fingerprint([(str(p), digest(p)) for p in files_under(cfg['paths']['references'])])


def date_taken(path, library):
    # Never mistake a filesystem copy/restore date for the capture date.
    from PIL import Image
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            nested = exif.get_ifd(34665) if 34665 in exif else {}
            value = nested.get(36867) or exif.get(36867)
            if value:
                return date.fromisoformat(str(value)[:10].replace(':', '-')).isoformat(), 'EXIF'
    except (OSError, ValueError, TypeError, KeyError):
        pass
    import re
    relative = str(Path(path).relative_to(library))
    match = re.search(r'(?<!\d)((?:19|20)\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)', relative)
    if match:
        try:
            return date(*map(int, match.groups())).isoformat(), 'filename/folder'
        except ValueError:
            pass
    return None, 'unknown'


def scan_inventory(project, cfg):
    library_path = project / 'library.json'
    if not library_path.exists():
        raise ValueError('No scan exists for these paths. Run photo-selector run')
    library = json.loads(library_path.read_text())
    if library.get('source') != cfg['paths']['library'] or library.get('identity_digest') != digest(project/'identity.json'):
        raise ValueError('Identity/library changed. Run photo-selector run')
    expected = hashlib.sha256(json.dumps({'identity': digest(project/'identity.json'),
        'max_side': cfg['recognition']['max_side'], 'pipeline': 3, 'revision': REVISION}, sort_keys=True).encode()).hexdigest()
    if expected != library['config']:
        raise ValueError('Recognition settings changed. Run photo-selector run')
    db = connect(project)
    try:
        rows = {r['path']: dict(r) for r in db.execute('SELECT * FROM photos')}
    finally:
        db.close()
    candidates, errors, stale, scanned = [], [], [], 0
    for path in files_under(cfg['paths']['library']):
        scanned += 1
        st = path.stat()
        row = rows.get(str(path))
        if not row or row['config'] != expected or (row['size'], row['mtime_ns']) != (st.st_size, st.st_mtime_ns):
            stale.append(str(path))
            continue
        if row['error']:
            errors.append({'path': str(path), 'error': row['error']})
            continue
        data = json.loads(row['data'])
        if data.get('face_count') and data['similarity'] >= cfg['recognition']['threshold']:
            taken, source = date_taken(path, Path(cfg['paths']['library']))
            candidates.append(dict(data, path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns,
                                   date=taken, date_source=source))
    if stale:
        raise ValueError(f'{len(stale)} images have not been scanned with the current settings; run photo-selector run')
    return candidates, {'supported_images': scanned, 'read_errors': errors, 'identity_matches': len(candidates)}


def weighted_score(assessment, criteria):
    total = sum(c['weight'] for c in criteria.values())
    return sum(assessment['scores'][name]['score']*c['weight'] for name, c in criteria.items()) / total


def select_top(items, cfg):
    """Rank a set subject to date and visual-variety limits, without new inference."""
    candidates = []
    for item in items:
        for key in ('duplicate_of', 'diversity_of', 'rank', 'selection_score'):
            item.pop(key, None)
        item['score'] = round(weighted_score(item['assessment'], cfg['criteria']), 4)
        item['status'] = 'eligible'
        if item['assessment']['excluded']:
            item['status'] = 'visual_exclusion'
        elif cfg['selection']['not_before'] and item['date'] and item['date'] < cfg['selection']['not_before']:
            item['status'] = 'date_exclusion'
        elif item['score'] < cfg['selection']['minimum_score']:
            item['status'] = 'below_minimum_score'
        else:
            candidates.append(item)
    candidates.sort(key=lambda i: (-i['score'], i['path']))
    exact, tree, unique = {}, HashTree(), []
    radius = cfg['selection']['near_duplicate_distance']
    for item in candidates:
        other = exact.get(item['sha256'])
        if other is None and radius >= 0:
            other = next((x for x in tree.find(int(item['phash'], 16), radius)
                          if abs((item['width']/item['height'])/(x['width']/x['height'])-1) < .1), None)
        if other is not None:
            item['status'], item['duplicate_of'] = 'duplicate', other['path']
            continue
        exact[item['sha256']] = item
        tree.add(int(item['phash'], 16), item)
        unique.append(item)
    chosen, counts, days = [], {}, {}
    max_per_day = cfg['selection'].get('max_per_day', 1)
    min_visual_distance = cfg['selection'].get('min_visual_distance', 12)
    while unique and len(chosen) < cfg['selection']['top']:
        # A diminishing bonus rewards underrepresented photo roles, without quotas.
        def utility(item):
            category = item['assessment']['category']
            return item['score'] + cfg['selection']['diversity_bonus']/(1+counts.get(category, 0))
        best = max(unique, key=utility)
        unique.remove(best)
        day = best.get('date')
        if day and max_per_day and days.get(day, 0) >= max_per_day:
            best['status'] = 'day_limit'
            continue
        similar = next((other for other in chosen
                        if min_visual_distance and
                        abs((best['width']/best['height'])/(other['width']/other['height'])-1) < .1 and
                        (int(best['phash'], 16)^int(other['phash'], 16)).bit_count() < min_visual_distance), None)
        if similar is not None:
            best['status'], best['diversity_of'] = 'visual_similarity', similar['path']
            continue
        best['selection_score'] = round(utility(best), 4)
        best['rank'] = len(chosen)+1
        best['status'] = 'recommended'
        category = best['assessment']['category']
        counts[category] = counts.get(category, 0)+1
        if day:
            days[day] = days.get(day, 0)+1
        chosen.append(best)
    return chosen


def write_results(project, cfg, items, selected, coverage):
    out = project / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, mode=0o700)
    (out/'thumbs').mkdir(mode=0o700)
    report = {'config': cfg, 'coverage': coverage, 'assessment_signature': assessment_signature(cfg),
              'requested': cfg['selection']['top'], 'recommended': len(selected),
              'note': 'Model recommendations against your rubric, not an objective attractiveness score.'}
    known_days = {item['date'] for item in selected if item.get('date')}
    undated = sum(not item.get('date') for item in selected)
    report['diversity'] = {'distinct_known_days': len(known_days), 'undated_photos': undated,
                           'max_per_day': cfg['selection'].get('max_per_day', 1),
                           'min_visual_distance': cfg['selection'].get('min_visual_distance', 12)}
    atomic_json(out/'run.json', report)
    atomic_json(out/'recommendations.json', selected)
    fields = ['path', 'status', 'score', 'rank', 'similarity', 'date', 'date_source', 'category', 'summary', 'exclusion_reason', 'duplicate_of', 'diversity_of']
    for name in cfg['criteria']:
        fields += [name+'_score', name+'_reason']
    with (out/'all-scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in items:
            row = {key: item.get(key, '') for key in fields}
            row.update({key: item['assessment'][key] for key in ('category', 'summary', 'exclusion_reason')})
            for name, score in item['assessment']['scores'].items():
                row[name+'_score'], row[name+'_reason'] = score['score'], score['reason']
            # Spreadsheet formula prevention for user-controlled filenames/text.
            writer.writerow({k: "'"+v if isinstance(v, str) and v.startswith(('=', '+', '-', '@')) else v for k,v in row.items()})
    escape = lambda value: html.escape(str(value), quote=True)
    cards = []
    for item in selected:
        path = Path(item['path'])
        st = path.stat()
        if (st.st_size, st.st_mtime_ns) != (item['size'], item['mtime_ns']):
            raise ValueError('A selected source changed; run the pipeline again: ' + str(path))
        picture = read_image(path)
        picture.thumbnail((1200, 1000))
        picture.save(out/'thumbs'/f'{item["rank"]:02d}.jpg', quality=92)
        scores = ''.join(f'<tr><th>{escape(name)}</th><td>{value["score"]:.1f}/10</td><td>{escape(value["reason"])}</td></tr>' for name,value in item['assessment']['scores'].items())
        cards.append(f'<article><h2>#{item["rank"]} · {item["score"]:.2f}/10 · {escape(item["assessment"]["category"])}</h2>'
                     f'<a href="{escape(path.as_uri())}"><img src="thumbs/{item["rank"]:02d}.jpg" alt="Recommendation {item["rank"]}"></a>'
                     f'<p>{escape(item["assessment"]["summary"])}</p><p class="path">{escape(path)}</p>'
                     f'<p>Capture date: {escape(item["date"] or "unknown")} ({escape(item["date_source"])})</p>'
                     f'<details><summary>Why this photo scored well</summary><table>{scores}</table></details></article>')
    shortfall = f'<p>Only {len(selected)} photos qualified for the requested {cfg["selection"]["top"]}. See all-scores.csv for exclusions and scores.</p>' if len(selected)<cfg['selection']['top'] else ''
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Your dating-photo recommendations</title>'
                '<style>body{font:17px system-ui;background:#f4f4ef;color:#23382e;max-width:1100px;margin:40px auto;padding:20px}article{background:white;border-radius:12px;padding:24px;margin:24px 0}img{max-width:100%;max-height:700px}p{line-height:1.6}.path{overflow-wrap:anywhere;color:#526459}td,th{padding:10px;text-align:left;vertical-align:top}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style>'
                f'<h1>Your top {len(selected)} dating-profile photos</h1><p>Selected automatically using your configured criteria, day limits, and visual variety settings. Scores express the local model’s assessment, not an objective measure of attractiveness.</p>'
                f'<p>Variety: {len(known_days)} distinct known capture days; {undated} photos with unknown dates. Unknown dates cannot be checked against the per-day limit. Visual similarity is a heuristic, not clothing recognition.</p>'
                f'<p>Scanned {coverage["supported_images"]:,} supported photos; found {coverage["identity_matches"]:,} identity matches; '
                f'{len(coverage["read_errors"])} unreadable files. Every readable match was assessed. Videos and unsupported formats are outside this photo scan.</p>'
                + shortfall + ''.join(cards) + '</html>')
    (out/'index.html').write_text(document)
    atomic_json(project/'latest.json', {'output': str(out), 'recommended': len(selected), 'requested': cfg['selection']['top']})
    print(f'Recommended {len(selected)} of the requested {cfg["selection"]["top"]}. Open: {out / "index.html"}', flush=True)
    return out


def rank(project, state, cfg):
    candidates, coverage = scan_inventory(project, cfg)
    cache = sqlite3.connect(project/'assessments.sqlite3')
    cache.execute('CREATE TABLE IF NOT EXISTS assessments (key TEXT PRIMARY KEY, result TEXT NOT NULL)')
    signature = assessment_signature(cfg)
    pending, completed = [], []
    try:
        for item in candidates:
            key = fingerprint({'sha': item['sha256'], 'box': item['box'], 'signature': signature})
            row = cache.execute('SELECT result FROM assessments WHERE key=?', (key,)).fetchone()
            if row:
                item['assessment'] = validate_assessment(json.loads(row[0]), cfg)
                completed.append(item)
            else:
                pending.append((key, item))
        print(f'Visual stage: {len(candidates)} identity matches, {len(completed)} cached, {len(pending)} to assess', flush=True)
        context = local_assessor(state, cfg) if pending else nullcontext(None)
        start = time.monotonic()
        with context as assessor:
            for index, (key, item) in enumerate(pending, 1):
                path = Path(item['path'])
                before = path.stat()
                if path.is_symlink() or not path.resolve().is_relative_to(Path(cfg['paths']['library'])) or (before.st_size, before.st_mtime_ns) != (item['size'], item['mtime_ns']):
                    raise ValueError('Source changed; rescan: ' + str(path))
                # Another path with identical contents/target may have populated
                # this key earlier in this same run.
                row = cache.execute('SELECT result FROM assessments WHERE key=?', (key,)).fetchone()
                if row:
                    result = validate_assessment(json.loads(row[0]), cfg)
                else:
                    for attempt in range(2):
                        try:
                            result = assessor.assess(item)
                            break
                        except (OSError, ValueError, KeyError, IndexError) as error:
                            if attempt:
                                raise RuntimeError(f'Visual assessment failed for {path}; completed results are cached. {error}') from error
                            time.sleep(1)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('Source changed during assessment: ' + str(path))
                with cache:
                    cache.execute('INSERT OR REPLACE INTO assessments VALUES (?,?)', (key, json.dumps(result)))
                item['assessment'] = result
                completed.append(item)
                elapsed = time.monotonic()-start
                remaining = (len(pending)-index)*elapsed/index
                print(f'Assessed {index}/{len(pending)} new photos; {elapsed:.0f}s elapsed, about {remaining/60:.0f}m remaining', flush=True)
    finally:
        cache.close()
    current, coverage = scan_inventory(project, cfg)
    if {(i['path'], i['size'], i['mtime_ns']) for i in current} != {(i['path'], i['size'], i['mtime_ns']) for i in completed}:
        raise ValueError('Library changed during assessment. Run photo-selector run again; assessments are cached')
    chosen = select_top(completed, cfg)
    return write_results(project, cfg, completed, chosen, coverage)


def run(args, state):
    from .cli import enroll, scan
    cfg = load_config(args.config)
    state = prepare_state(state, cfg['paths'].values())
    project = prepare_state(project_path(state, cfg), cfg['paths'].values())
    # One GPU pipeline per state root. The kernel releases this lock after crashes.
    with (state/'.pipeline.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another recommendation pipeline is using this state folder') from error
        refhash = reference_fingerprint(cfg)
        identity = project/'identity.json'
        previous = json.loads(identity.read_text()) if identity.exists() else {}
        changed = previous.get('input_fingerprint') != refhash or previous.get('model_revision') != REVISION
        if previous.get('excluded') and not cfg['recognition']['skip_invalid_references']:
            changed = True
        if args.command == 'rank' and changed:
            raise ValueError('Reference photos changed. Run photo-selector run')
        if args.command == 'run':
            options = SimpleNamespace(references=Path(cfg['paths']['references']), library=Path(cfg['paths']['library']),
                max_side=cfg['recognition']['max_side'], threads=cfg['recognition']['threads'], engine_state=state,
                skip_invalid=cfg['recognition']['skip_invalid_references'], input_fingerprint=refhash, limit=None)
            if changed:
                enroll(options, project)
            else:
                print(f'Reusing {len(previous["references"])} enrolled references', flush=True)
            scan(options, project)
            if args.scan_only:
                print('Identity scan complete. Run photo-selector rank --config ' + str(args.config), flush=True)
                return
        return rank(project, state, cfg)
