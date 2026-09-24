"""Portable setup checks: synthetic paths, no GPU or personal inputs required."""
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('container_config', ROOT/'.devcontainer/configure.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class SetupTests(unittest.TestCase):
    def test_private_config_creation_and_mount_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'workspace'
            root.mkdir(); (root/'.devcontainer').mkdir()
            (root/'selector.example.toml').write_bytes((ROOT/'selector.example.toml').read_bytes())
            gpu = {'GPU_RENDER_DEVICE': '/dev/dri/renderD999', 'GPU_RENDER_GID': '123', 'GPU_KFD_GID': '456'}
            with patch.object(setup, 'ROOT', root), patch.object(setup, 'discover_gpu', return_value=gpu), \
                 patch.object(setup.platform, 'system', return_value='Linux'), \
                 patch.object(setup.platform, 'machine', return_value='x86_64'), \
                 patch.object(setup.Path, 'home', return_value=Path(tmp)), \
                 patch.object(setup.Path, 'is_char_device', return_value=True), redirect_stdout(io.StringIO()):
                setup.main([])
                setup.main(['--check'])
                config = json.loads((root/'.devcontainer/devcontainer.json').read_text())
                self.assertEqual(config['postCreateCommand'], 'bash .devcontainer/post-create.sh')
                self.assertEqual(sum(m.endswith(',readonly') for m in config['mounts']), 2)
                self.assertIn('--device=/dev/dri/renderD999', config['runArgs'])
                self.assertIn('--group-add=456', config['runArgs'])
                self.assertEqual((root/'.env').stat().st_mode & 0o777, 0o600)
                before = (root/'.env').read_bytes()
                setup.main([])
                self.assertEqual((root/'.env').read_bytes(), before)
                private = root/'selector.toml'
                library = root/'alternate photos'; library.mkdir()
                private.write_text(private.read_text().replace('${PHOTO_LIBRARY}', 'alternate photos'))
                with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                    setup.main(['--check'])
                setup.main([])
                config = json.loads((root/'.devcontainer/devcontainer.json').read_text())
                self.assertEqual(config['containerEnv']['PHOTO_LIBRARY'], str(library))

    def test_source_state_overlap_and_unresolved_variables_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);(root/'photos').mkdir();(root/'references').mkdir()
            (root/'selector.toml').write_text('[paths]\nlibrary="photos"\nreferences="references"\n')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                setup.resolve_paths(root, {'PHOTO_STATE': str(root/'photos'/'state')})
            (root/'selector.toml').write_text('[paths]\nlibrary="${MISSING}"\nreferences="references"\n')
            with self.assertRaisesRegex(ValueError, 'Unknown environment'):
                setup.resolve_paths(root, {'PHOTO_STATE': str(root/'state')})

    def test_shell_quoting_and_mount_delimiters(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'.env'
            p.write_text("PHOTO_LIBRARY='/path with spaces/photos'\n# comment\n")
            self.assertEqual(setup.load_env(p)['PHOTO_LIBRARY'], '/path with spaces/photos')
        with self.assertRaises(ValueError):
            setup.mount('/path,with,commas', '/target')
        with self.assertRaises(ValueError):
            setup.mount('/path/${unsafe}', '/target')


if __name__ == '__main__':
    unittest.main()
