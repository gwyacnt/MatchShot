"""Explicit real-model check on a supplied, single-face test image; no downloads.

Run inside the devcontainer:
    python3 scripts/validate_gpu.py /path/to/sample.jpg
"""
import argparse
import json
import os
from pathlib import Path
import time

from PIL import Image

from photo_selector.core import Engine, identity_score, prepare_state, read_image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("sample", type=Path)
args = parser.parse_args()
state = prepare_state(Path(os.environ.get("PHOTO_SELECTOR_STATE", "/state")))
engine = Engine(state, profile=True)
image = read_image(args.sample)
image.thumbnail((640, 640))
start = time.monotonic()
reference = engine.faces(image)
assert len(reference) == 1, f"Expected one reference face, found {len(reference)}"
group = Image.new("RGB", (image.width * 2, image.height))
group.paste(image, (0, 0))
group.paste(image, (image.width, 0))
faces = engine.faces(group)
assert len(faces) == 2, f"Expected two group faces, found {len(faces)}"
scores = [identity_score(face["embedding"], [reference[0]["embedding"]] * 2) for face in faces]
assert all(score > 0.8 for score in scores), scores
assert engine.faces(Image.new("RGB", (640, 640))) == []
report = engine.diagnose()
report.update(group_faces=len(faces), similarities=scores, total_inference_seconds=time.monotonic()-start)
(state / "gpu-validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
