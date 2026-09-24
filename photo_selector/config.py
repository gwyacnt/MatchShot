"""Validated, portable user configuration. No executable configuration code."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tomllib
from datetime import date


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_config(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        cfg = tomllib.load(stream)
    allowed = {'paths', 'recognition', 'selection', 'vision', 'criteria'}
    if set(cfg) != allowed:
        raise ValueError('Config requires exactly these sections: ' + ', '.join(sorted(allowed)))
    for group, keys in {
        'paths': {'library', 'references'},
        'recognition': {'threshold', 'max_side', 'threads', 'skip_invalid_references'},
        'selection': {'top', 'minimum_score', 'near_duplicate_distance', 'diversity_bonus', 'not_before'},
        'vision': {'image_size', 'timeout_seconds', 'exclude'},
    }.items():
        if not isinstance(cfg[group], dict) or set(cfg[group]) != keys:
            raise ValueError(f'[{group}] must contain: {", ".join(sorted(keys))}')
    for key, value in cfg['paths'].items():
        if not isinstance(value, str) or not value:
            raise ValueError(f'paths.{key} must be a nonempty path')
        expanded = os.path.expandvars(value)
        if '$' in expanded:
            raise ValueError(f'Unresolved environment variable in paths.{key}: {value}')
        p = Path(expanded).expanduser()
        cfg['paths'][key] = str((path.parent / p).resolve()) if not p.is_absolute() else str(p.resolve())
        if not Path(cfg['paths'][key]).is_dir():
            raise ValueError(f'paths.{key} is not an accessible directory: {cfg["paths"][key]}')
    def number(group, key, low, high, integer=False):
        v = cfg[group][key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high or (integer and not isinstance(v, int)):
            raise ValueError(f'{group}.{key} must be {"an integer" if integer else "a number"} in [{low}, {high}]')
    number('recognition', 'threshold', -1, 1)
    number('recognition', 'max_side', 320, 4096, True)
    if cfg['recognition']['max_side'] % 32:
        raise ValueError('recognition.max_side must be a multiple of 32')
    number('recognition', 'threads', 1, 64, True)
    if not isinstance(cfg['recognition']['skip_invalid_references'], bool):
        raise ValueError('skip_invalid_references must be true or false')
    number('selection', 'top', 1, 100, True)
    number('selection', 'minimum_score', 0, 10)
    number('selection', 'near_duplicate_distance', -1, 64, True)
    number('selection', 'diversity_bonus', 0, 10)
    if not isinstance(cfg['selection']['not_before'], str):
        raise ValueError('selection.not_before must be empty or YYYY-MM-DD')
    if cfg['selection']['not_before']:
        try:
            date.fromisoformat(cfg['selection']['not_before'])
        except (TypeError, ValueError) as exc:
            raise ValueError('selection.not_before must be empty or YYYY-MM-DD') from exc
    number('vision', 'image_size', 336, 2016, True)
    number('vision', 'timeout_seconds', 10, 3600, True)
    exclusions = cfg['vision']['exclude']
    if not isinstance(exclusions, list) or any(not isinstance(x, str) or not x.strip() for x in exclusions):
        raise ValueError('vision.exclude must be a list of nonempty instructions')
    if not isinstance(cfg['criteria'], dict) or not cfg['criteria']:
        raise ValueError('At least one criterion is required')
    for name, criterion in cfg['criteria'].items():
        if not re.fullmatch('[a-z][a-z0-9_]{0,39}', name) or not isinstance(criterion, dict) or set(criterion) != {'weight', 'description'}:
            raise ValueError(f'Invalid criterion: {name}')
        weight = criterion['weight']
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f'Invalid weight for {name}')
        if not isinstance(criterion['description'], str) or not criterion['description'].strip():
            raise ValueError(f'{name} needs a description')
    if not sum(c['weight'] for c in cfg['criteria'].values()) > 0:
        raise ValueError('At least one criterion must have positive weight')
    return cfg
