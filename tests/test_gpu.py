"""GPU selection must fail closed; these tests do not require GPU hardware."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from photo_selector.inference import Engine, PROVIDER, fixed_model, require_gpu, detect_multiscale


class GPUChecks(unittest.TestCase):
    def test_cpu_only_runtime_rejected(self):
        runtime = Mock()
        runtime.get_available_providers.return_value = ["CPUExecutionProvider"]
        with self.assertRaisesRegex(ValueError, "CPU fallback is disabled"):
            require_gpu(runtime)
        runtime.disable_telemetry_events.assert_called_once()

    def test_input_specialization_drops_stale_output_shapes(self):
        import onnx
        from onnx import TensorProto, helper
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source.onnx", Path(tmp) / "fixed.onnx"
            graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "test",
                                      [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 640, 640])],
                                      [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 640, 640])])
            onnx.save(helper.make_model(graph), source)
            fixed_model(source, target, [1, 3, 1280, 1280])
            model = onnx.load(target)
            onnx.checker.check_model(model)
            inferred = onnx.shape_inference.infer_shapes(model)
            dims = inferred.graph.output[0].type.tensor_type.shape.dim
            self.assertEqual([dim.dim_value for dim in dims], [1, 3, 1280, 1280])

    def test_small_portrait_is_padded_without_upscaling(self):
        detector = Mock()
        detector.detect.return_value = (np.array([[20, 30, 120, 150, 0.9]], dtype=np.float32), np.ones((1, 5, 2), dtype=np.float32))
        image = np.full((512, 512, 3), 100, dtype=np.uint8)
        result = detect_multiscale(detector, image, 1280)
        detector.detect.assert_called_once()
        canvas = detector.detect.call_args.args[0]
        np.testing.assert_array_equal(canvas[:512, :512], image)
        self.assertFalse(canvas[512:].any())
        np.testing.assert_allclose(result[0][0][:4], [20, 30, 120, 150])

    def test_multiscale_merges_the_same_face(self):
        detector = Mock()
        points = np.ones((1, 5, 2), dtype=np.float32)
        detector.detect.side_effect = [
            (np.array([[64, 64, 128, 128, 0.8]], dtype=np.float32), points),
            (np.array([[128, 128, 256, 256, 0.9]], dtype=np.float32), points * 2),
        ]
        result = detect_multiscale(detector, np.zeros((2000, 2000, 3), dtype=np.uint8), 1280)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0][0][:4], [200, 200, 400, 400])
        self.assertAlmostEqual(float(result[0][0][4]), 0.9, places=5)

    def test_profile_checks_actual_execution(self):
        # Provider availability alone is not evidence of GPU computation.
        for providers, passes in [([PROVIDER], True), (["CPUExecutionProvider"], False),
                                  ([PROVIDER, "CPUExecutionProvider"], False), ([], False)]:
            with self.subTest(providers=providers), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "profile.json"
                path.write_text(json.dumps([{"cat": "Node", "args": {"provider": p}} for p in providers]))
                engine = Engine.__new__(Engine)
                engine.faces = Mock(return_value=[])
                engine.recognizer = Mock()
                engine.recognizer.get_feat.return_value = np.ones((1, 512))
                session = Mock()
                session.end_profiling.return_value = str(path)
                engine.sessions = {"detector": session, "recognizer": session}
                if passes:
                    self.assertEqual(engine.diagnose()["backend"], PROVIDER)
                else:
                    with self.assertRaisesRegex(ValueError, "GPU-only execution was not confirmed"):
                        engine.diagnose()


if __name__ == "__main__":
    unittest.main()
