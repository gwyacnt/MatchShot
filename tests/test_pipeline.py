import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from photo_selector.cli import main
from photo_selector.core import HashTree, REVISION, atomic_json, digest, disjoint, identity_score


class FakeEngine:
    calls = 0

    def __init__(self, *args):
        pass

    def faces(self, image):
        FakeEngine.calls += 1
        common = dict(box=[10, 10, 90, 90], blur=100.0, brightness=120.0,
                      face_pixels=90.0, face_fraction=0.2)
        # The target is the second face: scanning must not pick the first/largest.
        return [dict(common, embedding=np.array([0., 1.])),
                dict(common, embedding=np.array([1., 0.]))]


class PipelineTests(unittest.TestCase):
    def run_cli(self, state, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            main(["--state", str(state), *map(str, args)])

    def test_protected_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for a, b in [(root, root), (root / "state", root), (root, root / "library")]:
                with self.assertRaises(ValueError):
                    disjoint(a, b)
            disjoint(root / "state", root / "photos")

    def test_identity_requires_two_reference_support(self):
        self.assertAlmostEqual(identity_score(np.array([1., 0.]), [[1., 0.], [0., 1.], [0., 1.]]), 0.5)

    def test_hash_tree_matches_brute_force(self):
        tree = HashTree()
        values = [0, 1, 3, 7, 15, 255, 0, 65535]
        for i, value in enumerate(values):
            tree.add(value, i)
        for query in [0, 17, 65535]:
            for radius in range(5):
                self.assertEqual(sorted(tree.find(query, radius)),
                                 [i for i, value in enumerate(values) if (query ^ value).bit_count() <= radius])

    @patch("photo_selector.cli.Engine", FakeEngine)
    def test_scan_resume_and_no_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library, state, refs = root / "library", root / "state", root / "refs"
            library.mkdir(); state.mkdir(); refs.mkdir()
            year = library / "2024"; year.mkdir()
            first = year / "one.jpg"
            Image.new("RGB", (200, 200), (130, 150, 180)).save(first)
            duplicate = year / "copy.jpg"; duplicate.write_bytes(first.read_bytes())
            (year / "broken.jpg").write_bytes(b"not an image")
            (year / "outside.jpg").symlink_to(first)
            before = {p: (digest(p), p.stat().st_mtime_ns) for p in year.iterdir() if not p.is_symlink()}
            atomic_json(state / "identity.json", {"source": str(refs), "model_revision": REVISION,
                        "references": [{"embedding": [1., 0.]}] * 5})
            FakeEngine.calls = 0
            self.run_cli(state, "scan", library)
            self.assertEqual(FakeEngine.calls, 2)
            self.run_cli(state, "scan", library)
            self.assertEqual(FakeEngine.calls, 2)
            from photo_selector.core import connect
            db = connect(state)
            rows = db.execute("SELECT * FROM photos").fetchall()
            db.close()
            self.assertEqual(len(rows), 3)
            self.assertEqual(sum(bool(r["error"]) for r in rows), 1)
            matches = [json.loads(r["data"]) for r in rows if not r["error"]]
            self.assertTrue(all(r["similarity"] == 1.0 for r in matches))
            self.assertEqual(before, {p: (digest(p), p.stat().st_mtime_ns) for p in before})
            old_config = json.loads((state / "library.json").read_text())["config"]
            identity = json.loads((state / "identity.json").read_text())
            identity["changed"] = True
            atomic_json(state / "identity.json", identity)
            self.run_cli(state, "scan", library, "--limit", "1")
            self.assertNotEqual(json.loads((state / "library.json").read_text())["config"], old_config)

    @patch("photo_selector.cli.Engine", FakeEngine)
    def test_ambiguous_reference_does_not_replace_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs, state = root / "refs", root / "state"
            refs.mkdir(); state.mkdir()
            for i in range(5):
                Image.new("RGB", (200, 200), (i*20, 50, 90)).save(refs / f"{i}.jpg")
            atomic_json(state / "identity.json", {"source": str(refs), "existing": True})
            before = digest(state / "identity.json")
            with self.assertRaises(SystemExit):
                self.run_cli(state, "enroll", refs)
            self.assertEqual(digest(state / "identity.json"), before)


if __name__ == "__main__":
    unittest.main()
