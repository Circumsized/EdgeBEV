import logging
import os
import re
import threading
from typing import Callable, Dict, List, Optional, Tuple

import torch

try:
    import numpy as np
except ImportError:
    np = None

# Lazy-import tensorrt — engine_utils is also imported in non-TRT contexts.
try:
    import tensorrt as trt
    _HAS_TRT = True
except ImportError:
    trt = None  # type: ignore[assignment]
    _HAS_TRT = False


_SM_RE = re.compile(r"^(?P<prefix>.+)_sm(?P<sm>\d{2})(?P<suffix>\.engine)$")


def current_sm_tag() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot resolve TRT engine architecture.")
    dev = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(dev)
    return f"{prop.major}{prop.minor}"


def _replace_sm_suffix(path: str, sm_tag: str) -> str:
    dirname, basename = os.path.split(path)
    m = _SM_RE.match(basename)
    if not m:
        return path
    replaced = f"{m.group('prefix')}_sm{sm_tag}{m.group('suffix')}"
    return os.path.join(dirname, replaced)


def _sm_suffix(path: str) -> str:
    m = _SM_RE.match(os.path.basename(path))
    return m.group("sm") if m else ""


def candidate_engine_paths(requested_path: str, sm_tag: str) -> List[str]:
    requested_path = requested_path.strip()
    dirname, basename = os.path.split(requested_path)
    paths: List[str] = []

    def add(path: str):
        if path and path not in paths:
            paths.append(path)

    # As requested
    add(requested_path)
    add(_replace_sm_suffix(requested_path, sm_tag))

    # If no SM suffix in name, prefer an SM-matched sibling if present.
    if not _sm_suffix(requested_path):
        stem, ext = os.path.splitext(basename)
        if ext == ".engine":
            sm_name = f"{stem}_sm{sm_tag}.engine"
            if dirname:
                add(os.path.join(dirname, sm_name))
            else:
                add(sm_name)

    return paths


def load_runner_with_fallback(
    requested_path: str,
    runner_ctor: Callable[[str, object], object],
    logger,
    role: str,
) -> Tuple[str, object]:
    requested_path = requested_path.strip()
    if os.path.dirname(requested_path) == "artifacts":
        normalized = os.path.basename(requested_path)
        logger.warning(
            f"[{role}] artifacts path is archive-only, use root engine instead: "
            f"{requested_path} -> {normalized}"
        )
        requested_path = normalized
    sm_tag = current_sm_tag()
    candidates = candidate_engine_paths(requested_path, sm_tag)
    attempted = []

    for path in candidates:
        if not os.path.exists(path):
            attempted.append((path, "missing"))
            continue
        try:
            runner = runner_ctor(path, logger)
            if path != requested_path:
                logger.warning(
                    f"[{role}] engine fallback: {requested_path} -> {path} (sm{sm_tag})"
                )
            return path, runner
        except Exception as exc:  # keep trying alternatives
            attempted.append((path, str(exc)))

    detail = "\n".join([f"  - {p}: {msg}" for p, msg in attempted])
    raise RuntimeError(
        f"[{role}] No compatible TRT engine for sm{sm_tag}. Requested: {requested_path}\n"
        f"Tried:\n{detail}"
    )


# ============================================================================
# Unified TRT Engine Runner
# ============================================================================


class TRTRunner:
    """Unified TRT engine runner with output buffer reuse and CUDA stream control.

    Features:
    - Caches output tensors by (name, shape); only reallocates when the dynamic
      runtime shape changes (e.g. variable batch size). Avoids per-call
      ``torch.zeros`` allocation overhead.
    - Accepts an optional CUDA stream for pipelined execution; defaults to the
      current PyTorch stream.
    - Per-call synchronization control via ``synchronize`` kwarg. Defaults to
      ``True`` to match historical behavior; set ``False`` for async/batched
      execution (caller must synchronize before consuming outputs).

    The constructor signature ``(engine_path, logger)`` is preserved so
    existing callers and ``load_runner_with_fallback`` work unchanged.
    """

    def __init__(self, engine_path: str, logger=None, pool_size: int = 1):
        if not isinstance(pool_size, int) or isinstance(pool_size, bool) or pool_size < 1:
            raise ValueError("pool_size must be a positive integer")
        if not _HAS_TRT:
            raise RuntimeError(
                "TensorRT is not available; cannot instantiate TRTRunner. "
                "Install TensorRT or run in a TRT-enabled environment."
            )
        self.logger = logger or logging.getLogger(__name__)
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.pool_size = pool_size
        self._slot_cursor = 0
        self._slot_cursor_lock = threading.Lock()

        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.output_dtypes: Dict[str, "torch.dtype"] = {}
        self.input_dtypes: Dict[str, "torch.dtype"] = {}
        self.input_max_batch: Optional[int] = None
        # Static (max) shapes from the engine; informational only.
        self._output_static_shapes: Dict[str, Tuple[int, ...]] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                shape = tuple(self.engine.get_tensor_shape(name))
                self.input_dtypes[name] = self._torch_dtype_for_trt(
                    self.engine.get_tensor_dtype(name), name
                )
                if shape and shape[0] > 0:
                    self.input_max_batch = (
                        shape[0]
                        if self.input_max_batch is None
                        else min(self.input_max_batch, shape[0])
                    )
            else:
                self.output_names.append(name)
                self._output_static_shapes[name] = tuple(self.engine.get_tensor_shape(name))
                dtype_trt = self.engine.get_tensor_dtype(name)
                self.output_dtypes[name] = self._torch_dtype_for_trt(dtype_trt, name)

        if self.input_max_batch is None and self.input_names:
            self.input_max_batch = self._read_profile_max_batch(self.input_names[0])

        # Output buffer cache: name -> (shape_signature, tensor).
        # Reused when the dynamic shape signature matches; reallocated otherwise.
        self._output_buffers: Dict[str, Tuple[Tuple[int, ...], "torch.Tensor"]] = {}
        self._contexts = [self.context]
        for _ in range(self.pool_size - 1):
            context = self.engine.create_execution_context()
            if context is None:
                raise RuntimeError("Failed to create TensorRT execution context")
            self._contexts.append(context)
        self._buffers = [dict() for _ in range(self.pool_size)]
        self._input_refs = [[] for _ in range(self.pool_size)]
        self._slot_locks = [threading.Lock() for _ in range(self.pool_size)]
        self._slot_events = [None for _ in range(self.pool_size)]
        self._stats_lock = threading.Lock()
        self._buffer_hits = 0
        self._buffer_misses = 0
        self._last_stream_ptr = None
        self._last_run_synchronized = True

        self.logger.info(
            f"TRT engine loaded: {engine_path} "
            f"(inputs={self.input_names}, outputs={self.output_names})"
        )

    @staticmethod
    def _torch_dtype_for_trt(dtype_trt, name):
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
        }
        if hasattr(trt, "int8"):
            mapping[trt.int8] = torch.int8
        if hasattr(trt, "int32"):
            mapping[trt.int32] = torch.int32
        if hasattr(trt, "bool"):
            mapping[trt.bool] = torch.bool
        if hasattr(trt, "int64"):
            mapping[trt.int64] = torch.int64
        if hasattr(trt, "uint8"):
            mapping[trt.uint8] = torch.uint8
        dtype = mapping.get(dtype_trt)
        if dtype is None:
            raise TypeError(f"Unsupported TensorRT dtype for {name}: {dtype_trt}")
        return dtype

    def _read_profile_max_batch(self, name: str) -> Optional[int]:
        """Read the max batch dimension from the engine's optimization profiles.

        Best-effort: returns None if the profile API is unavailable or the
        engine does not carry an optimization profile. Callers fall back to the
        static shape or a single unbatched call when this returns None.
        """
        if name is None or not _HAS_TRT:
            return None
        try:
            num_profiles = int(getattr(self.engine, "num_optimization_profiles", 0))
            getter = getattr(self.engine, "get_tensor_profile_shape", None)
            if getter is None:
                getter = getattr(self.engine, "get_profile_shape", None)
            if getter is None or num_profiles <= 0:
                return None
            best: Optional[int] = None
            for i in range(num_profiles):
                shapes = getter(name, i)
                if not shapes:
                    continue
                max_shape = shapes[-1] if isinstance(shapes, (list, tuple)) else None
                if max_shape is None or len(max_shape) == 0:
                    continue
                dim = int(max_shape[0])
                if dim > 0:
                    best = dim if best is None else min(best, dim)
            return best
        except Exception:
            return None

    def _get_execution_slot(self):
        with self._slot_cursor_lock:
            index = self._slot_cursor
            self._slot_cursor = (self._slot_cursor + 1) % self.pool_size
        return self._contexts[index], self._buffers[index], index

    @staticmethod
    def _resolve_streams(stream):
        consumer_stream = torch.cuda.current_stream()
        if stream is None:
            return consumer_stream, consumer_stream
        if isinstance(stream, int):
            return consumer_stream, torch.cuda.ExternalStream(stream)
        if not hasattr(stream, "cuda_stream"):
            raise TypeError("stream must be a CUDA stream or raw stream pointer")
        return consumer_stream, stream

    def _wait_for_slot(self, slot):
        event = self._slot_events[slot]
        if event is not None:
            if not event.query():
                event.synchronize()
            self._slot_events[slot] = None

    def _get_output_buffer(
        self, name: str, shape: Tuple[int, ...], buffers=None
    ) -> "torch.Tensor":
        """Return a CUDA tensor for ``name`` with ``shape``, reusing the cache when possible."""
        buffers = self._output_buffers if buffers is None else buffers
        cached = buffers.get(name)
        if cached is not None and cached[0] == shape:
            with self._stats_lock:
                self._buffer_hits += 1
            return cached[1]
        with self._stats_lock:
            self._buffer_misses += 1
        dtype = self.output_dtypes[name]
        # ``empty`` is safe: the engine overwrites the buffer on enqueue.
        t = torch.empty(shape, dtype=dtype, device="cuda").contiguous()
        buffers[name] = (shape, t)
        return t

    def __call__(
        self,
        *inputs,
        stream: Optional[object] = None,
        synchronize: bool = True,
        copy_outputs: bool = False,
    ) -> List["torch.Tensor"]:
        """Run the engine on positional CUDA tensors."""
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"expected {len(self.input_names)} inputs, got {len(inputs)}"
            )
        context, buffers, slot = self._get_execution_slot()
        with self._slot_locks[slot]:
            self._wait_for_slot(slot)
            return self._execute_on_slot(
                context,
                buffers,
                slot,
                inputs,
                stream=stream,
                synchronize=synchronize,
                copy_outputs=copy_outputs,
            )

    def _execute_on_slot(
        self,
        context,
        buffers,
        slot,
        inputs,
        stream=None,
        synchronize=True,
        copy_outputs=False,
    ):
        # Keep converted and contiguous input tensors alive until the async execution completes.
        self._input_refs[slot] = []
        for name, tensor in zip(self.input_names, inputs):
            if not torch.is_tensor(tensor) or not tensor.is_cuda:
                raise ValueError(f"input {name} must be a CUDA torch.Tensor")
            expected_dtype = self.input_dtypes[name]
            t = tensor if tensor.dtype == expected_dtype else tensor.to(expected_dtype)
            t = t.contiguous()
            self._input_refs[slot].append(t)
            context.set_input_shape(name, tuple(t.shape))
            context.set_tensor_address(name, t.data_ptr())

        outputs: Dict[str, "torch.Tensor"] = {}
        for name in self.output_names:
            shape = tuple(context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Unresolved TensorRT output shape for {name}: {shape}")
            t = self._get_output_buffer(name, shape, buffers=buffers)
            context.set_tensor_address(name, t.data_ptr())
            outputs[name] = t

        consumer_stream, execution_stream = self._resolve_streams(stream)
        if execution_stream.cuda_stream != consumer_stream.cuda_stream:
            execution_stream.wait_stream(consumer_stream)
        stream_ptr = execution_stream.cuda_stream

        ok = context.execute_async_v3(stream_handle=stream_ptr)
        if ok is False:
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        with self._stats_lock:
            self._last_stream_ptr = stream_ptr
            self._last_run_synchronized = synchronize
        completion_event = torch.cuda.Event()
        completion_event.record(execution_stream)
        self._slot_events[slot] = completion_event
        if synchronize:
            consumer_stream.wait_event(completion_event)
        self.logger.debug(
            "TRT run complete: slot=%d buffer_hits=%d buffer_misses=%d synchronize=%s",
            slot,
            self._buffer_hits,
            self._buffer_misses,
            synchronize,
        )
        result = [outputs[name] for name in self.output_names]
        if copy_outputs:
            if execution_stream.cuda_stream != consumer_stream.cuda_stream:
                consumer_stream.wait_stream(execution_stream)
            with torch.cuda.stream(consumer_stream):
                result = [output.clone() for output in result]
        return result

    def run_batched(self, inputs, max_batch_size=None, synchronize=True, stream=None):
        """Run one input tensor in batches and return independent output tensors."""
        if len(self.input_names) != 1:
            raise ValueError("run_batched requires an engine with one input")
        tensor = inputs if torch.is_tensor(inputs) else inputs[0]
        batch = int(tensor.shape[0])
        if max_batch_size is None:
            max_batch_size = self.input_max_batch
        if max_batch_size is None or max_batch_size <= 0 or max_batch_size >= batch:
            # 与分批路径保持一致的别名语义：返回独立副本，避免调用方长期持有内部复用 buffer
            return self(
                tensor,
                stream=stream,
                synchronize=synchronize,
                copy_outputs=True,
            )
        chunks = []
        for start in range(0, batch, max_batch_size):
            chunk = self(
                tensor[start:start + max_batch_size],
                stream=stream,
                synchronize=False,
                copy_outputs=True,
            )
            chunks.append(chunk)
        _, execution_stream = self._resolve_streams(stream)
        completion_event = torch.cuda.Event()
        completion_event.record(execution_stream)
        if synchronize:
            torch.cuda.current_stream().wait_event(completion_event)
        return [torch.cat([chunk[index] for chunk in chunks], dim=0)
                for index in range(len(chunks[0]))]

    @property
    def buffer_stats(self):
        return {"hits": self._buffer_hits, "misses": self._buffer_misses}

    @property
    def runtime_stats(self):
        return {
            "buffer_hits": self._buffer_hits,
            "buffer_misses": self._buffer_misses,
            "last_stream_ptr": self._last_stream_ptr,
            "last_run_synchronized": self._last_run_synchronized,
        }

    def reset_buffers(self) -> None:
        """Drop cached output tensors (e.g. after a device or context change)."""
        self._output_buffers.clear()
        self._input_refs = [[] for _ in range(self.pool_size)]
        self._buffer_hits = 0
        self._buffer_misses = 0
        for buffers in self._buffers:
            buffers.clear()
        self._slot_events = [None for _ in range(self.pool_size)]
