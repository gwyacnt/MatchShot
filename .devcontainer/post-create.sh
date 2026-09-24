#!/usr/bin/env bash
set -euo pipefail

# Runtime dependencies are installed from the hash-locked manifest in the image.
# Never let InsightFace replace AMD's ONNX Runtime with the PyPI CPU package.
python3 -m pip install --user --no-deps --no-build-isolation -e .

# Downloads are pinned and verified; /state is a persistent, private host mount.
# No photos are processed during setup. Rerunning verifies/reuses existing models.
photo-selector models
