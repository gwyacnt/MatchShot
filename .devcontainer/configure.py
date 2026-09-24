"""Generate private Dev Container mounts; run with Python 3.11+ on the Linux host."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
from string import Template
import tomllib

ROOT = Path(__file__).resolve().parent.parent


def discover_gpu():
    kfd = Path('/dev/kfd')
    if not kfd.is_char_device():
        raise ValueError('Missing /dev/kfd. Install a supported AMD host driver first.')
    for entry in sorted(Path('/sys/class/drm').glob('renderD*')):
        vendor = entry / 'device/vendor'
        if vendor.exists() and vendor.read_text().strip() == '0x1002':
            device = Path('/dev/dri') / entry.name
            if device.is_char_device():
                return {'GPU_RENDER_DEVICE': str(device), 'GPU_RENDER_GID': str(device.stat().st_gid),
                        'GPU_KFD_GID': str(kfd.stat().st_gid)}
    raise ValueError('No AMD render device found. Check the host driver and /dev/dri.')


def load_env(path):
    values = {}
    for line in path.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if not words:
            continue
        if len(words) != 1 or '=' not in words[0]:
            raise ValueError("Use KEY='absolute path' lines in .env; quote paths containing spaces.")
        key, value = words[0].split('=', 1)
        values[key] = value
    return values


def resolve_paths(root, values):
    config = dict(values)
    with (root/'selector.toml').open('rb') as stream:
        paths = tomllib.load(stream)['paths']
    for key, setting in [('PHOTO_LIBRARY', 'library'), ('PHOTO_REFERENCES', 'references')]:
        try:
            value = Template(paths[setting]).substitute(config)
        except KeyError as error:
            raise ValueError(f'Unknown environment variable or missing path: {error}') from error
        path = Path(value).expanduser()
        config[key] = str((root/path).resolve()) if not path.is_absolute() else str(path.resolve())
        if not Path(config[key]).is_dir():
            raise ValueError(f'{setting} must name an existing directory in selector.toml/.env')
    state = Path(config.get('PHOTO_STATE', '')).expanduser()
    if not state.is_absolute():
        raise ValueError('PHOTO_STATE must be an absolute host path in .env')
    state = state.resolve()
    for key in ('PHOTO_LIBRARY', 'PHOTO_REFERENCES'):
        source = Path(config[key])
        if state.is_relative_to(source) or source.is_relative_to(state):
            raise ValueError(f'PHOTO_STATE must not overlap {key}')
    config['PHOTO_STATE'] = str(state)
    return config


def mount(source, target, readonly=False):
    if any(',' in value or '${' in value for value in (source, target)):
        raise ValueError('Container paths cannot contain commas or ${...} expressions')
    return f'type=bind,source={source},target={target}' + (',readonly' if readonly else '')


def make_document(config):
    for key in ('LOCAL_UID', 'LOCAL_GID', 'GPU_RENDER_GID', 'GPU_KFD_GID'):
        if not config.get(key, '').isdigit():
            raise ValueError(f'{key} must be a numeric ID in .env')
    groups = list(dict.fromkeys([config['GPU_RENDER_GID'], config['GPU_KFD_GID']]))
    return {
        'name': 'Photo selector — local AMD GPU inference',
        'build': {'dockerfile': 'Dockerfile', 'context': '..',
                  'args': {'USER_UID': config['LOCAL_UID'], 'USER_GID': config['LOCAL_GID']}},
        'workspaceFolder': '/workspaces/photo_selector',
        'workspaceMount': 'type=bind,source=${localWorkspaceFolder},target=/workspaces/photo_selector',
        'initializeCommand': ['python3', '${localWorkspaceFolder}/.devcontainer/configure.py', '--check'],
        'remoteUser': 'vscode', 'updateRemoteUserUID': False,
        'runArgs': ['--device=/dev/kfd', f'--device={config["GPU_RENDER_DEVICE"]}',
                    *[f'--group-add={group}' for group in groups], '--shm-size=2g'],
        'mounts': [mount(config['PHOTO_STATE'], '/state'),
                   mount(config['PHOTO_LIBRARY'], config['PHOTO_LIBRARY'], True),
                   mount(config['PHOTO_REFERENCES'], config['PHOTO_REFERENCES'], True)],
        'containerEnv': {
            'PHOTO_SELECTOR_STATE': '/state', 'PHOTO_STATE_HOST': config['PHOTO_STATE'],
            'PHOTO_LIBRARY': config['PHOTO_LIBRARY'], 'PHOTO_REFERENCES': config['PHOTO_REFERENCES'],
            'ROCR_VISIBLE_DEVICES': '0', 'ORT_MIGRAPHX_CACHE_PATH': '/state/cache/migraphx',
            'ORT_MIGRAPHX_MODEL_CACHE_PATH': '/state/cache/migraphx',
            'MIOPEN_USER_DB_PATH': '/state/cache/miopen',
        },
        'postCreateCommand': 'bash .devcontainer/post-create.sh',
        'shutdownAction': 'stopContainer',
        'customizations': {'vscode': {'extensions': ['ms-python.python'],
                          'settings': {'python.defaultInterpreterPath': '/usr/bin/python3'}}},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate generated settings without changing files')
    args = parser.parse_args(argv)
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        parser.error('This devcontainer targets Linux x86_64 hosts with a supported AMD GPU.')
    try:
        gpu = discover_gpu()
        settings, selector = ROOT/'.env', ROOT/'selector.toml'
        if args.check and (not settings.exists() or not selector.exists()):
            raise ValueError('Run python3 .devcontainer/configure.py on the host first.')
        if not settings.exists():
            values = dict(gpu, LOCAL_UID=str(os.getuid()), LOCAL_GID=str(os.getgid()))
            for key, folder in [('PHOTO_LIBRARY', 'library'), ('PHOTO_REFERENCES', 'references')]:
                path = ROOT/'.local-inputs'/folder
                path.mkdir(parents=True, exist_ok=True)
                values[key] = str(path)
            values['PHOTO_STATE'] = str(Path.home()/'.local/share/photo-selector')
            settings.write_text('# Private host paths and GPU device IDs. Do not commit.\n' +
                                ''.join(f'{key}={shlex.quote(value)}\n' for key, value in values.items()))
            settings.chmod(0o600)
            print('Created .env with private host defaults. Set your library and reference paths there.')
        if not selector.exists():
            selector.write_bytes((ROOT/'selector.example.toml').read_bytes())
            selector.chmod(0o600)
            print('Created selector.toml from the public example; edit your criteria there.')
        values = load_env(settings)
        # Older private configurations lack GPU_KFD_GID; discover it, never assume a group name.
        values.setdefault('GPU_KFD_GID', gpu['GPU_KFD_GID'])
        config = resolve_paths(ROOT, values)
        device = Path(config.get('GPU_RENDER_DEVICE', ''))
        if not str(device).startswith('/dev/dri/renderD') or not device.is_char_device():
            raise ValueError('GPU_RENDER_DEVICE must name an existing render device in .env')
        document = make_document(config)
        destination = ROOT/'.devcontainer/devcontainer.json'
        if args.check:
            if not destination.exists() or json.loads(destination.read_text()) != document:
                raise ValueError('Settings changed: run python3 .devcontainer/configure.py on the host, then reopen.')
            if not Path(config['PHOTO_STATE']).is_dir():
                raise ValueError('State folder is missing; rerun configure.py on the host.')
            print('Devcontainer settings are current.')
        else:
            Path(config['PHOTO_STATE']).mkdir(parents=True, exist_ok=True, mode=0o700)
            destination.write_text(json.dumps(document, indent=2)+'\n')
            destination.chmod(0o600)
            print('Generated private .devcontainer/devcontainer.json. Reopen/rebuild with Dev Containers.')
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
