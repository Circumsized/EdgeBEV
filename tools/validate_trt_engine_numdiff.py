"""
Numerical diff between two TRT engine versions for the same ONNX.
Saves/loads I/O tensors as NPZ files to avoid version conflicts in one process.
"""
import argparse
import os
import sys

import numpy as np


_TRT_TO_NUMPY = {
    "bool": np.bool_,
    "boolean": np.bool_,
    "float": np.float32,
    "float32": np.float32,
    "half": np.float16,
    "float16": np.float16,
    "int8": np.int8,
    "uint8": np.uint8,
    "int32": np.int32,
    "int16": np.int16,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "double": np.float64,
}


def trt_dtype_to_numpy(dtype):
    key = str(dtype).lower().split(".")[-1]
    if key not in _TRT_TO_NUMPY:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return _TRT_TO_NUMPY[key]


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def run_engine(engine_path, inputs_npz=None, outputs_npz=None, save_inputs_npz=None):
    from tools.trt_infer_zero_torch import ZeroTorchTRTRunner, make_cuda_buffer_from_array

    runner = ZeroTorchTRTRunner(engine_path)
    if inputs_npz is not None:
        with np.load(inputs_npz) as data:
            input_buffers = {name: np.array(data[name], copy=True) for name in runner.input_names}
    else:
        input_buffers = {}
        for name in runner.input_names:
            shape = tuple(runner.engine.get_tensor_shape(name))
            np_dtype = trt_dtype_to_numpy(runner.engine.get_tensor_dtype(name))
            if np_dtype == np.bool_:
                arr = np.random.randint(0, 2, size=shape).astype(np_dtype)
            elif np.issubdtype(np_dtype, np.integer):
                arr = np.random.randint(-8, 8, size=shape).astype(np_dtype)
            else:
                arr = np.random.randn(*shape).astype(np_dtype)
            input_buffers[name] = arr

    input_buffers_gpu = {k: make_cuda_buffer_from_array(v) for k, v in input_buffers.items()}
    results = runner(input_buffers_gpu)

    if save_inputs_npz is not None:
        np.savez(save_inputs_npz, **input_buffers)
        print(f"Saved inputs to {save_inputs_npz}")

    if outputs_npz is not None:
        outs = {name: arr for name, arr in zip(runner.output_names, results)}
        np.savez(outputs_npz, **outs)
        print(f"Saved outputs to {outputs_npz}")
        for name, arr in outs.items():
            print(f"  {name}: {arr.shape} {arr.dtype}  mean={arr.mean():.6f}  std={arr.std():.6f}")


def compare_npz(
    ref_path,
    test_path,
    atol=1e-3,
    rtol=1e-3,
    relative_error_limit=1e-2,
    relative_error_epsilon=1e-6,
    cosine_min=0.99999,
):
    with np.load(ref_path) as ref, np.load(test_path) as test:
        print(f"Comparing {ref_path} vs {test_path}")
        ref_keys = set(ref.files)
        test_keys = set(test.files)
        all_pass = ref_keys == test_keys
        for name in sorted(ref_keys | test_keys):
            if name not in ref_keys:
                print(f"  {name}: EXTRA TEST KEY")
                all_pass = False
                continue
            if name not in test_keys:
                print(f"  {name}: MISSING TEST KEY")
                all_pass = False
                continue
            a = np.asarray(ref[name])
            b = np.asarray(test[name])
            if a.shape != b.shape:
                print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
                all_pass = False
                continue
            if a.dtype != b.dtype:
                print(f"  {name}: DTYPE MISMATCH {a.dtype} vs {b.dtype}")
                all_pass = False
                continue
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                print(f"  {name}: NON-FINITE VALUE")
                all_pass = False
                continue
            a64 = a.astype(np.float64, copy=False)
            b64 = b.astype(np.float64, copy=False)
            diff = np.abs(a64 - b64)
            scale = np.maximum(np.maximum(np.abs(a64), np.abs(b64)), relative_error_epsilon)
            relative = diff / scale
            norm_a = np.linalg.norm(a64.ravel())
            norm_b = np.linalg.norm(b64.ravel())
            if norm_a <= 1e-12 or norm_b <= 1e-12:
                cosine = 1.0 if np.allclose(a64, b64, atol=atol, rtol=rtol) else 0.0
            else:
                cosine = float(np.dot(a64.ravel(), b64.ravel()) / (norm_a * norm_b))
            max_diff = float(diff.max()) if diff.size else 0.0
            mean_diff = float(diff.mean()) if diff.size else 0.0
            p99_diff = float(np.quantile(diff, 0.99)) if diff.size else 0.0
            absolute_pass = np.allclose(a64, b64, atol=atol, rtol=rtol)
            significant = scale >= atol
            if relative.size and bool(significant.any()):
                rel_over_sig = relative[significant]
                max_relative_error = float(rel_over_sig.max())
                p99_relative_error = float(np.quantile(rel_over_sig, 0.99))
                relative_pass = max_relative_error <= relative_error_limit
            else:
                max_relative_error = 0.0
                p99_relative_error = 0.0
                relative_pass = True
            cosine_pass = cosine >= cosine_min
            status = (
                "PASS" if absolute_pass and relative_pass and cosine_pass else "FAIL"
            )
            print(
                f"  {name}: max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} "
                f"p99_diff={p99_diff:.6e} max_relative_error={max_relative_error:.6e} "
                f"(gate, significant values only) p99_relative_error={p99_relative_error:.6e} (diagnostic) "
                f"cos_sim={cosine:.8f} "
                f"[absolute={absolute_pass} relative={relative_pass} cosine={cosine_pass} {status}]"
            )
            if status == "FAIL":
                all_pass = False
    print("✅ All tensors PASS" if all_pass else "❌ Some tensors FAIL")
    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, help="Path to .engine")
    parser.add_argument("--load-inputs", type=str, help="Load inputs from .npz")
    parser.add_argument("--save-inputs", type=str, help="Save inputs to .npz")
    parser.add_argument("--save-outputs", type=str, help="Save outputs to .npz")
    parser.add_argument("--compare", nargs=2, metavar=("REF", "TEST"), help="Compare two .npz files")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--relative-error-limit", type=float, default=1e-2)
    parser.add_argument("--relative-error-epsilon", type=float, default=1e-6)
    parser.add_argument("--cosine-min", type=float, default=0.99999)
    args = parser.parse_args()

    if args.compare:
        ok = compare_npz(
            args.compare[0],
            args.compare[1],
            atol=args.atol,
            rtol=args.rtol,
            relative_error_limit=args.relative_error_limit,
            relative_error_epsilon=args.relative_error_epsilon,
            cosine_min=args.cosine_min,
        )
        sys.exit(0 if ok else 1)

    if not args.engine:
        parser.error("--engine required unless --compare")

    run_engine(args.engine, args.load_inputs, args.save_outputs, args.save_inputs)


if __name__ == "__main__":
    main()
