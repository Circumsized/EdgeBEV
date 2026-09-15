"""
Build TensorRT engine from ONNX with configurable precision and target SM.
Supports INT8/FP16 and can target SM 8.7 for Jetson Orin from x86 build hosts.

Usage example:
    python tools/build_trt_engine.py \
        --onnx artifacts/vtransform_depthnet_int8.onnx \
        --engine vtransform_depthnet_int8_sm87.engine \
        --fp16 --int8 --workspace 4096 \
        --timing-cache timing.cache
"""
import argparse
import hashlib
import json
import os
import platform
import sys

try:
    import tensorrt as trt
except ImportError:
    trt = None


def _require_trt():
    if trt is None:
        raise RuntimeError("TensorRT is required to build an engine")
    return trt


def _make_logger():
    trt_module = _require_trt()

    class TrtLogger(trt_module.Logger):
        def __init__(self):
            super().__init__(trt_module.Logger.INFO)

        def log(self, severity, msg):
            if severity <= self.severity:
                print(f"[TRT] {msg}")

    return TrtLogger()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _network_has_qdq(network):
    """Return True if the parsed network already contains Q/DQ quantizer nodes."""
    trt_module = _require_trt()
    q_types = set()
    if hasattr(trt_module, "LayerType"):
        for attr in ("QUANTIZE", "DEQUANTIZE"):
            layer_type = getattr(trt_module.LayerType, attr, None)
            if layer_type is not None:
                q_types.add(layer_type)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer is not None and layer.type in q_types:
            return True
    return False


def _parse_shape_spec(spec):
    """Parse 'name:d0,d1,...,dn' into (name, (d0, ..., dn))."""
    name, sep, dims = spec.partition(":")
    if not sep or not dims.strip():
        raise ValueError(f"invalid shape spec '{spec}', expected name:d0,d1,...")
    return name.strip(), tuple(int(d.strip()) for d in dims.split(","))


def _dynamic_inputs(network):
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = tuple(tensor.shape)
        if any(d < 0 for d in shape):
            yield tensor.name, shape


def _write_engine_metadata(metadata_path, onnx_path, engine_path, options):
    metadata = {
        "onnx_sha256": _sha256_file(onnx_path),
        "engine_sha256": _sha256_file(engine_path),
        "engine_path": os.path.basename(engine_path),
        "onnx_path": os.path.basename(onnx_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorrt": trt.__version__ if trt is not None else None,
        "options": options,
    }
    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_runtime"] = torch.version.cuda
        metadata["gpu"] = [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else []
    except ImportError:
        pass
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def build_engine(onnx_path, engine_path, fp16=False, int8=False, workspace_mb=4096,
                 timing_cache_path=None, hardware_compat=False,
                 version_compatible=False, strongly_typed=False,
                 max_aux_streams=None, avg_timing_iterations=None,
                 allow_timing_cache_mismatch=False, metadata_path=None,
                 opt_shapes=None, min_shapes=None, max_shapes=None,
                 allow_hc_fallback=False):
    trt_module = _require_trt()
    logger = _make_logger()
    builder = trt_module.Builder(logger)
    network_flags = 1 << int(trt_module.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if strongly_typed:
        network_flags |= 1 << int(trt_module.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt_module.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    print(f"ONNX parsed: {onnx_path}")
    print(f"  Network inputs: {network.num_inputs}")
    print(f"  Network outputs: {network.num_outputs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt_module.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    if fp16:
        config.set_flag(trt_module.BuilderFlag.FP16)
        print("  Enabled FP16")
    if int8:
        if not _network_has_qdq(network):
            raise RuntimeError(
                "--int8 requires a pre-quantized QDQ/INT8 ONNX because this script "
                "does not run an INT8 calibrator. Provide a QDQ ONNX or drop --int8."
            )
        config.set_flag(trt_module.BuilderFlag.INT8)
        print("  Enabled INT8 (QDQ)")
    if version_compatible:
        if not hasattr(trt_module.BuilderFlag, "VERSION_COMPATIBLE"):
            raise RuntimeError("TensorRT version does not support version-compatible engines")
        config.set_flag(trt_module.BuilderFlag.VERSION_COMPATIBLE)
        print("  Enabled version compatibility")
    if max_aux_streams is not None:
        config.max_aux_streams = max_aux_streams
        print(f"  Max auxiliary streams: {max_aux_streams}")
    if avg_timing_iterations is not None:
        config.avg_timing_iterations = avg_timing_iterations
        print(f"  Average timing iterations: {avg_timing_iterations}")

    # Hardware compatibility within Ampere family (SM 8.x) can help x86-built
    # engines run on Orin (SM 8.7) without rebuilding on the device itself.
    # Available in TensorRT 10.x as a preview feature.
    if hardware_compat:
        hc_ok = False
        hc_reason = ""
        try:
            preview_feature = trt_module.PreviewFeature.HARDWARE_COMPATIBLE_AMAX
            # TRT 10.x: set_preview_feature 返回 bool；不支持时返回 False（不抛异常）
            hc_ok = bool(config.set_preview_feature(preview_feature, True))
            if not hc_ok:
                hc_reason = "set_preview_feature returned False (feature unsupported)"
        except AttributeError:
            hc_reason = "PreviewFeature.HARDWARE_COMPATIBLE_AMAX is not defined in this TensorRT version"
        except Exception as exc:  # noqa: BLE001 - 保留原始错误信息用于诊断
            hc_reason = f"set_preview_feature raised {type(exc).__name__}: {exc}"

        if hc_ok:
            print("  Enabled Hardware Compatibility (Ampere)")
        elif not allow_hc_fallback:
            # 请求了跨设备兼容却无法生效：默认 fail-closed，避免静默产出普通引擎
            raise RuntimeError(
                "Hardware Compatibility (HARDWARE_COMPATIBLE_AMAX) could not be enabled: "
                f"{hc_reason}. Re-run with --allow-hc-fallback to build a "
                "non-hardware-compatible engine, or upgrade TensorRT."
            )
        else:
            print(f"  WARNING: Hardware Compatibility unavailable ({hc_reason}); "
                  "proceeding WITHOUT hardware compatibility (--allow-hc-fallback)")

    dynamic_inputs = list(_dynamic_inputs(network))
    if dynamic_inputs:
        if not opt_shapes:
            raise RuntimeError(
                "ONNX has dynamic input dimensions ("
                + ", ".join(name for name, _ in dynamic_inputs)
                + ") but no --opt-shape was provided. Provide --opt-shape name:d0,d1,... "
                "(optionally --min-shape/--max-shape) for each dynamic input, or export "
                "the ONNX with static shapes."
            )
        profile = builder.create_optimization_profile()
        min_shapes = min_shapes or {}
        max_shapes = max_shapes or {}
        for name, static_shape in dynamic_inputs:
            if name not in opt_shapes:
                raise RuntimeError(f"missing --opt-shape for dynamic input '{name}'")
            opt = opt_shapes[name]
            if len(opt) != len(static_shape):
                raise RuntimeError(
                    f"--opt-shape rank mismatch for '{name}': got {len(opt)} dims, "
                    f"expected {len(static_shape)}"
                )
            minimum = min_shapes.get(name, tuple(1 if d < 0 else d for d in static_shape))
            maximum = max_shapes.get(name, opt)
            if len(minimum) != len(opt) or len(maximum) != len(opt):
                raise RuntimeError(f"shape rank mismatch for '{name}'")
            for lo, op, hi in zip(minimum, opt, maximum):
                if lo < 1 or op < 1 or hi < 1 or not (lo <= op <= hi):
                    raise RuntimeError(
                        f"invalid shape range for '{name}': min={minimum} opt={opt} max={maximum}"
                    )
            profile.set_shape(name, minimum, opt, maximum)
        config.add_optimization_profile(profile)
        print(f"  Added optimization profile for {len(dynamic_inputs)} dynamic input(s)")

    if timing_cache_path and os.path.exists(timing_cache_path):
        with open(timing_cache_path, "rb") as f:
            timing_cache = config.create_timing_cache(f.read())
            config.set_timing_cache(timing_cache, ignore_mismatch=allow_timing_cache_mismatch)
            print(
                f"  Loaded timing cache: {timing_cache_path} "
                f"(ignore_mismatch={allow_timing_cache_mismatch})"
            )

    # Builder optimization level (5 = max)
    config.builder_optimization_level = 5

    print("Building engine... (this may take several minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    file_size_mb = os.path.getsize(engine_path) / (1 << 20)
    print(f"Engine saved: {engine_path} ({file_size_mb:.2f} MB)")
    if metadata_path:
        _write_engine_metadata(
            metadata_path,
            onnx_path,
            engine_path,
            {
                "fp16": fp16,
                "int8": int8,
                "workspace_mb": workspace_mb,
                "hardware_compat": hardware_compat,
                "version_compatible": version_compatible,
                "strongly_typed": strongly_typed,
                "max_aux_streams": max_aux_streams,
                "avg_timing_iterations": avg_timing_iterations,
                "timing_cache": os.path.basename(timing_cache_path) if timing_cache_path else None,
            },
        )
        print(f"Engine metadata saved: {metadata_path}")

    # Save timing cache for incremental builds
    if timing_cache_path:
        timing_cache = config.get_timing_cache()
        if timing_cache is not None:
            with open(timing_cache_path, "wb") as f:
                f.write(timing_cache.serialize())
            print(f"Timing cache saved: {timing_cache_path}")
        else:
            print("  Timing cache not available from this builder config")


def main():
    parser = argparse.ArgumentParser(description="Build TRT engine from ONNX")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--engine", required=True, help="Output engine path")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16")
    parser.add_argument("--int8", action="store_true", help="Enable INT8")
    parser.add_argument("--workspace", type=int, default=4096, help="Workspace MB")
    parser.add_argument("--timing-cache", type=str, default="timing.cache")
    parser.add_argument("--hardware-compat", action="store_true",
                        help="Enable Ampere hardware compatibility for Orin")
    parser.add_argument("--version-compatible", action="store_true",
                        help="Build an engine compatible with later TensorRT runtimes")
    parser.add_argument("--strongly-typed", action="store_true",
                        help="Use a strongly typed TensorRT network")
    parser.add_argument("--max-aux-streams", type=int, default=None)
    parser.add_argument("--avg-timing-iterations", type=int, default=None)
    parser.add_argument("--allow-timing-cache-mismatch", action="store_true")
    parser.add_argument("--allow-hc-fallback", action="store_true",
                        help="Permit building WITHOUT hardware compatibility when "
                             "--hardware-compat is requested but unavailable")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Write engine/environment metadata JSON")
    parser.add_argument("--min-shape", action="append", default=None,
                        help="Minimum shape for a dynamic input, name:d0,d1,... (repeatable)")
    parser.add_argument("--opt-shape", action="append", default=None,
                        help="Optimal shape for a dynamic input, name:d0,d1,... (repeatable)")
    parser.add_argument("--max-shape", action="append", default=None,
                        help="Maximum shape for a dynamic input, name:d0,d1,... (repeatable)")
    args = parser.parse_args()

    def _parse_shapes(specs):
        if not specs:
            return None
        result = {}
        for spec in specs:
            name, dims = _parse_shape_spec(spec)
            if name in result:
                raise ValueError(f"duplicate shape spec for input '{name}'")
            result[name] = dims
        return result

    opt_shapes = _parse_shapes(args.opt_shape)
    min_shapes = _parse_shapes(args.min_shape)
    max_shapes = _parse_shapes(args.max_shape)

    build_engine(
        args.onnx,
        args.engine,
        fp16=args.fp16,
        int8=args.int8,
        workspace_mb=args.workspace,
        timing_cache_path=args.timing_cache,
        hardware_compat=args.hardware_compat,
        version_compatible=args.version_compatible,
        strongly_typed=args.strongly_typed,
        max_aux_streams=args.max_aux_streams,
        avg_timing_iterations=args.avg_timing_iterations,
        allow_timing_cache_mismatch=args.allow_timing_cache_mismatch,
        metadata_path=args.metadata,
        opt_shapes=opt_shapes,
        min_shapes=min_shapes,
        max_shapes=max_shapes,
        allow_hc_fallback=args.allow_hc_fallback,
    )


if __name__ == "__main__":
    main()
