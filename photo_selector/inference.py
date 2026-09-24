"""InsightFace models executed by ONNX Runtime's AMD MIGraphX provider."""
from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np

from .core import normalized
from .models import MODELS, REVISION, digest, model_paths

PROVIDER = "MIGraphXExecutionProvider"


def require_gpu(ort):
    ort.disable_telemetry_events()
    available = ort.get_available_providers()
    if PROVIDER not in available:
        raise ValueError(f"AMD MIGraphX is unavailable ({available}). Run inside the ROCm devcontainer; CPU fallback is disabled.")


def fixed_model(source, target, shape):
    """Specialize dynamic ONNX input dimensions for predictable GPU compilation."""
    if not target.exists():
        import onnx
        model = onnx.load(source)
        for dimension, value in zip(model.graph.input[0].type.tensor_type.shape.dim, shape, strict=True):
            dimension.dim_value = value
        # Exported detector output annotations describe 640px inputs. Drop stale
        # inferred shapes so ORT infers them from the selected input resolution.
        del model.graph.value_info[:]
        for index, output in enumerate(model.graph.output):
            for axis, dimension in enumerate(output.type.tensor_type.shape.dim):
                dimension.dim_param = f"output_{index}_axis_{axis}"
        temporary = target.with_suffix(".tmp")
        onnx.save(model, temporary)
        temporary.replace(target)
    return target


def detect_multiscale(detector, bgr, max_side):
    """Pad without upscaling; merge overview/detail detections in original pixels."""
    height, width = bgr.shape[:2]
    scales = sorted({min(1.0, side / max(width, height)) for side in (min(640, max_side), max_side)})
    found = []
    for scale in scales:
        small = cv2.resize(bgr, (max(1, round(width * scale)), max(1, round(height * scale))))
        canvas = np.zeros((max_side, max_side, 3), dtype=np.uint8)
        canvas[:small.shape[0], :small.shape[1]] = small
        boxes, landmarks = detector.detect(canvas, max_num=0)
        if landmarks is None:
            continue
        sx, sy = width / small.shape[1], height / small.shape[0]
        for box, points in zip(boxes, landmarks):
            box, points = box.copy(), points.copy()
            box[[0, 2]] *= sx
            box[[1, 3]] *= sy
            points[:, 0] *= sx
            points[:, 1] *= sy
            found.append((box, points))
    if not found:
        return []
    boxes = [[float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])] for b, _ in found]
    keep = cv2.dnn.NMSBoxes(boxes, [float(b[4]) for b, _ in found], 0.5, 0.4)
    return [found[int(i)] for i in np.asarray(keep).flatten()]


class Engine:
    def __init__(self, state, max_side=1280, threads=4, profile=False):
        import onnxruntime as ort
        require_gpu(ort)
        from insightface.model_zoo.scrfd import SCRFD
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        self.state = state
        self.max_side = max_side
        if max_side < 320 or max_side % 32:
            raise ValueError("--max-side must be a multiple of 32 and at least 320 (e.g. 640, 1280, 1600)")
        cv2.setNumThreads(threads)
        paths = model_paths(state)
        for name, path in paths.items():
            if not path.exists() or digest(path) != MODELS[name][1]:
                raise ValueError("Missing or invalid InsightFace model; run 'photo-selector models' first")
        cache = state / "cache" / REVISION
        cache.mkdir(parents=True, exist_ok=True)
        for folder in (state / "cache/migraphx", state / "cache/miopen"):
            folder.mkdir(parents=True, exist_ok=True)
        sessions = {}
        for name, source in paths.items():
            shape = [1, 3, max_side, max_side] if name == "detector" else [1, 3, 112, 112]
            fixed = fixed_model(source, cache / f"{name}-{shape[-1]}-static-v2.onnx", shape)
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            options.enable_profiling = profile
            options.profile_file_prefix = str(cache / f"profile-{name}")
            print(f"Loading {name} on AMD GPU; first use may compile kernels...", flush=True)
            session = ort.InferenceSession(str(fixed), sess_options=options,
                                           providers=[(PROVIDER, {"device_id": "0"})])
            session.disable_fallback()
            if PROVIDER not in session.get_providers():
                raise ValueError(f"{name} fell back to CPU; GPU session required")
            sessions[name] = session
        self.sessions = sessions
        self.detector = SCRFD(model_file=str(paths["detector"]), session=sessions["detector"])
        self.detector.prepare(ctx_id=0, input_size=(max_side, max_side), det_thresh=0.5)
        self.recognizer = ArcFaceONNX(model_file=str(paths["recognizer"]), session=sessions["recognizer"])
        self.recognizer.prepare(ctx_id=0)
        print(f"Backend: InsightFace antelopev2 / {PROVIDER} (GPU 0, CPU fallback disabled)", flush=True)

    def faces(self, image):
        from insightface.utils.face_align import norm_crop
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        detections = detect_multiscale(self.detector, bgr, self.max_side)
        result = []
        for box, points in detections:
            x1, y1, x2, y2 = map(float, box[:4])
            x1, y1, x2, y2 = max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2)
            w, h = x2 - x1, y2 - y1
            crop = bgr[int(y1):int(y2), int(x1):int(x2)]
            if w <= 0 or h <= 0 or not crop.size:
                continue
            aligned = norm_crop(bgr, landmark=points, image_size=112)
            embedding = normalized(self.recognizer.get_feat(aligned))
            gray = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY)
            result.append({"box": [x1, y1, w, h], "embedding": embedding,
                           "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                           "brightness": float(np.mean(gray)),
                           "face_pixels": min(w, h), "face_fraction": w*h/(width*height)})
        return result

    def diagnose(self):
        from PIL import Image
        start = time.monotonic()
        self.faces(Image.new("RGB", (640, 640)))
        embedding = self.recognizer.get_feat(np.zeros((112, 112, 3), dtype=np.uint8))
        if embedding.shape != (1, 512) or not np.isfinite(embedding).all():
            raise ValueError("Recognition inference returned an invalid embedding")
        report = {"backend": PROVIDER, "model": REVISION, "seconds": time.monotonic() - start, "sessions": {}}
        for name, session in self.sessions.items():
            profile = Path(session.end_profiling())
            events = json.loads(profile.read_text())
            providers = {event.get("args", {}).get("provider") for event in events if event.get("cat") == "Node"}
            providers.discard(None)
            if PROVIDER not in providers or "CPUExecutionProvider" in providers:
                raise ValueError(f"{name}: GPU-only execution was not confirmed: {providers}; inspect {profile}")
            report["sessions"][name] = {"executed_providers": sorted(providers), "profile": str(profile)}
        return report
