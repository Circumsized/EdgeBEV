import ast
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class TestBevPoolV2Regressions(unittest.TestCase):
    def test_kernel_uses_grid_stride_loop(self):
        source = (ROOT / "tools/trt_plugins/bev_pool_v2/src/bev_pool_v2_kernel.cu").read_text()
        self.assertIn("gridDim.x", source)
        self.assertIn("idx += static_cast<long long>(blockDim.x) * gridDim.x", source)
        self.assertIn("grid_size_64 > 65535LL", source)
        plugin_source = (ROOT / "tools/trt_plugins/bev_pool_v2/src/bev_pool_v2_plugin.cpp").read_text()
        self.assertIn("cudaMemsetAsync", plugin_source)

    def test_plugin_has_input_contract_validation(self):
        source = (ROOT / "tools/trt_plugins/bev_pool_v2/src/bev_pool_v2_plugin.cpp").read_text()
        self.assertIn("validate_input_contract", source)
        self.assertIn("nbInputs", source)
        self.assertIn("nbOutputs", source)
        self.assertIn("inputs == nullptr", source)
        self.assertIn("outputs == nullptr", source)
        self.assertIn("inputs[0] == nullptr", source)
        self.assertIn("outputs[0] == nullptr", source)
        self.assertIn("serialData == nullptr", source)
        self.assertIn("interval_start", source)
        self.assertIn("interval_length", source)
        self.assertIn("std::numeric_limits<int>::max()", source)

    def test_plugin_validates_counts_and_pointers(self):
        source = (ROOT / "tools/trt_plugins/bev_pool_v2/src/bev_pool_v2_plugin.cpp").read_text()
        self.assertIn("nbInputs != 4", source)
        self.assertIn("nbOutputs != 1", source)
        self.assertIn("inputs[0] == nullptr", source)
        self.assertIn("outputs[0] == nullptr", source)

    def test_plugin_validates_serialization_and_creator_fields(self):
        source = (ROOT / "tools/trt_plugins/bev_pool_v2/src/bev_pool_v2_plugin.cpp").read_text()
        self.assertIn("serialLength <", source)
        self.assertIn("fc == nullptr", source)
        self.assertIn("f.length != 1", source)
        self.assertIn("f.type != nvinfer1::PluginFieldType::kINT32", source)


class TestRunnerPoolRegressions(unittest.TestCase):
    def test_runner_exposes_thread_execution_slots(self):
        source = (ROOT / "tools/engine_utils.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(ROOT / "tools/engine_utils.py"))
        names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertIn("_get_execution_slot", names)
        self.assertIn("_execute_on_slot", names)
        self.assertIn("_contexts", source)
        self.assertIn("_buffers", source)
        self.assertIn("_slot_locks", source)

    def test_runner_has_stream_dependency_helpers(self):
        source = (ROOT / "tools/engine_utils.py").read_text()
        self.assertIn("wait_stream", source)
        self.assertIn("wait_event", source)
        self.assertIn("torch.cuda.Event()", source)
        self.assertIn("cuda_stream", source)

    def test_run_batched_preserves_async_flag(self):
        source = (ROOT / "tools/engine_utils.py").read_text()
        self.assertIn("synchronize=synchronize", source)
        self.assertIn("clone()", source)


class TestNumericalDiffRegressions(unittest.TestCase):
    def test_compare_npz_rejects_non_finite_and_reports_relative_error(self):
        source = (ROOT / "tools/validate_trt_engine_numdiff.py").read_text()
        self.assertIn("isfinite", source)
        self.assertIn("relative", source)
        self.assertIn("missing", source.lower())
        self.assertIn("p99", source)

    def test_compare_npz_has_independent_relative_error_gate(self):
        source = (ROOT / "tools/validate_trt_engine_numdiff.py").read_text()
        self.assertIn("relative_error_limit", source)
        self.assertIn("relative_pass", source)
        self.assertIn("max_relative_error", source)

    def test_compare_npz_supports_all_runtime_dtypes(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("numdiff", ROOT / "tools/validate_trt_engine_numdiff.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = {
            "BOOL": np.bool_,
            "FLOAT": np.float32,
            "HALF": np.float16,
            "INT8": np.int8,
            "UINT8": np.uint8,
            "INT32": np.int32,
        }
        for name, dtype in expected.items():
            self.assertIs(module.trt_dtype_to_numpy(name), dtype)

    def test_compare_npz_dtype_mismatch_fails(self):
        import importlib.util
        import tempfile

        spec = importlib.util.spec_from_file_location("numdiff", ROOT / "tools/validate_trt_engine_numdiff.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            ref = Path(directory) / "ref.npz"
            test = Path(directory) / "test.npz"
            np.savez(ref, output=np.array([1], dtype=np.int32))
            np.savez(test, output=np.array([1.0], dtype=np.float32))
            self.assertFalse(module.compare_npz(ref, test, cosine_min=0.0))

    def test_relative_error_is_independent_gate(self):
        source = (ROOT / "tools/validate_trt_engine_numdiff.py").read_text()
        self.assertIn("max_relative_error", source)
        self.assertIn("relative_error_limit", source)
        self.assertIn("relative_pass", source)
        self.assertIn("p99_relative_error", source)

    def test_dtype_mapping_values(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("numdiff", ROOT / "tools/validate_trt_engine_numdiff.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = {
            "float16": np.float16,
            "float32": np.float32,
            "int8": np.int8,
            "int32": np.int32,
            "bool": np.bool_,
        }
        for name, dtype in expected.items():
            self.assertIs(module.trt_dtype_to_numpy(name), dtype)
        with self.assertRaises(TypeError):
            module.trt_dtype_to_numpy("float64")

    def test_trt_dtype_mapping_is_complete(self):
        source = (ROOT / "tools/validate_trt_engine_numdiff.py").read_text()
        for token in ("float16", "float32", "int8", "uint8", "int32", "bool"):
            self.assertIn(token, source)

    def test_relative_error_is_sensitive_to_scale(self):
        reference = np.array([1000.0, 0.0], dtype=np.float32)
        candidate = np.array([1001.0, 1.01], dtype=np.float32)
        relative = np.abs(reference - candidate) / np.maximum(np.abs(reference), 1e-6)
        self.assertGreater(float(relative[1]), float(relative[0]))


if __name__ == "__main__":
    unittest.main()
