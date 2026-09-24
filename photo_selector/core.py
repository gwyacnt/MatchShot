from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3

import cv2
import numpy as np
from PIL import Image, ImageOps

from .models import REVISION, digest, download_models

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic", ".heif", ".avif"}


def disjoint(state, source):
    state, source = Path(state).resolve(), Path(source).resolve()
    if state.is_relative_to(source) or source.is_relative_to(state):
        raise ValueError(f"State and source must be separate, non-overlapping folders: {state}, {source}")


def prepare_state(path, sources=()):
    path = Path(path).expanduser().resolve()
    for source in sources:
        disjoint(path, source)
        if host_state := os.environ.get("PHOTO_STATE_HOST"):
            disjoint(host_state, source)
    # Also protect source folders recorded by earlier commands.
    for name in ("identity.json", "library.json"):
        saved = path / name
        if saved.exists():
            data = json.loads(saved.read_text())
            disjoint(path, data["source"])
            if host_state := os.environ.get("PHOTO_STATE_HOST"):
                disjoint(host_state, data["source"])
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Refuse redirected generated files, even if they point outside known sources.
    if any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("State directory must not contain symbolic links")
    return path


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def files_under(root):
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    def fail(error):
        raise error
    for folder, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        dirs[:] = sorted(d for d in dirs if not (Path(folder) / d).is_symlink())
        for name in sorted(files):
            path = Path(folder) / name
            if not path.is_symlink() and path.suffix.lower() in EXTENSIONS:
                yield path


def read_image(path):
    if Path(path).suffix.lower() in {".heic", ".heif"}:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    with Image.open(path) as original:
        return ImageOps.exif_transpose(original).convert("RGB")


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError("Invalid face embedding")
    return vector / length


def Engine(state, max_side=1280, threads=4, **kwargs):
    # Keep non-inference commands usable without loading the GPU libraries.
    from .inference import Engine as InsightFaceEngine
    return InsightFaceEngine(state, max_side, threads, **kwargs)


def identity_score(embedding, references):
    # Two-reference support is less vulnerable to one accidental lookalike match.
    similarities = np.asarray(references) @ embedding
    return float(np.mean(np.sort(similarities)[-2:]))


def phash(image):
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = cv2.dct(gray)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    bits[0] = False
    return f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"


def connect(state):
    db = sqlite3.connect(state / "photos.sqlite3")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS photos (
        path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
        config TEXT NOT NULL, data TEXT NOT NULL, error TEXT)""")
    return db



class HashTree:
    """BK tree for Hamming-distance lookup of perceptual hashes."""
    def __init__(self):
        self.root = None

    def add(self, value, item):
        if self.root is None:
            self.root = [value, [item], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(item)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [item], {}]
                return
            node = node[2][distance]

    def find(self, value, radius):
        pending = [self.root] if self.root else []
        while pending:
            key, items, children = pending.pop()
            distance = (value ^ key).bit_count()
            if distance <= radius:
                yield from items
            pending.extend(child for edge, child in children.items() if distance-radius <= edge <= distance+radius)
