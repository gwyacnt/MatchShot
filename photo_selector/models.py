"""Pinned, explicit downloads; no network access during enrollment or scanning."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile

REVISION = "insightface-antelopev2-8e182f14-v1"
ARCHIVE_SHA256 = "8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5"
MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/antelopev2.zip"
MODELS = {
    "detector": ("scrfd_10g_bnkps.onnx", "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"),
    "recognizer": ("glintr100.onnx", "4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf"),
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_paths(state):
    return {name: state / "models" / "antelopev2" / spec[0] for name, spec in MODELS.items()}


def download_models(state):
    paths = model_paths(state)
    if all(path.exists() and digest(path) == MODELS[name][1] for name, path in paths.items()):
        print("Verified InsightFace antelopev2 detection and recognition models")
        return
    folder = state / "models"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "antelopev2.zip"
    if not archive.exists() or digest(archive) != ARCHIVE_SHA256:
        temporary = archive.with_suffix(".download")
        print("Downloading InsightFace antelopev2 (361 MB); pretrained weights are for non-commercial research.", flush=True)
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if digest(temporary) != ARCHIVE_SHA256:
                raise ValueError("InsightFace model archive checksum mismatch")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as source:
        for name, target in paths.items():
            target.parent.mkdir(exist_ok=True)
            payload = source.read("antelopev2/" + target.name)
            if hashlib.sha256(payload).hexdigest() != MODELS[name][1]:
                raise ValueError(f"Model checksum mismatch: {target.name}")
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
        (folder / "antelopev2" / "MODEL-TERMS.txt").write_text(
            "InsightFace code: MIT. These pretrained model weights: non-commercial research only.\n"
            "https://github.com/deepinsight/insightface/tree/master/python-package\n"
            f"Source: {MODEL_URL}\nArchive SHA256: {ARCHIVE_SHA256}\n"
            "Only face detection and recognition weights are used. Attribute models are not loaded.\n")
    archive.unlink()
    print("Verified and installed SCRFD-10GF and ArcFace ResNet100 (Glint360K)")
