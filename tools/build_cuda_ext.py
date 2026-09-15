"""
Build CUDA extensions for spconv23_deploy environment (Python 3.9, PyTorch 2.0).

Compiles:
  1. bev_pool_ext  — BEV pooling CUDA kernel
  2. voxel_layer   — Hard voxelization CUDA kernel
  3. iou3d_cuda    — Rotated NMS CUDA kernel

Usage:
    conda run --prefix <ENV_PREFIX> python tools/build_cuda_ext.py
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from torch.utils.cpp_extension import load


def _parse_args():
    parser = argparse.ArgumentParser(description="Build EdgeBEV CUDA extensions")
    parser.add_argument(
        "--build-dir",
        default=os.environ.get("BEVFUSION_BUILD_DIR", os.path.join(ROOT, "build_deploy")),
    )
    parser.add_argument(
        "--cuda-arch-list",
        default=os.environ.get("BEVFUSION_CUDA_ARCH_LIST"),
        help="CUDA architectures, for example 8.6;8.7 or 86;87",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _architecture_flags(value):
    if not value:
        return []
    gencodes = []
    seen = set()
    for token in value.replace(",", ";").split(";"):
        token = token.strip().lower()
        if not token:
            continue
        with_ptx = token.endswith("+ptx")
        if with_ptx:
            token = token[:-4]
        token = token.replace("sm_", "").replace("compute_", "").replace(".", "")
        if not token.isdigit() or not (2 <= len(token) <= 3) or token in seen:
            continue
        seen.add(token)
        gencodes.append(f"-gencode=arch=compute_{token},code=sm_{token}")
        if with_ptx:
            gencodes.append(f"-gencode=arch=compute_{token},code=compute_{token}")
    if not gencodes:
        raise ValueError("--cuda-arch-list must contain values such as 8.6;8.7")
    return gencodes


BUILD_DIR = os.path.abspath(os.environ.get("BEVFUSION_BUILD_DIR", os.path.join(ROOT, "build_deploy")))
CUDA_ARCH_FLAGS = []  # 延迟到 _configure_build 中读取，避免环境变量污染在 import 期即抛错
VERBOSE = False


def _configure_build(args):
    global BUILD_DIR, CUDA_ARCH_FLAGS, VERBOSE
    BUILD_DIR = os.path.abspath(args.build_dir)
    os.makedirs(BUILD_DIR, exist_ok=True)
    CUDA_ARCH_FLAGS = _architecture_flags(args.cuda_arch_list)
    VERBOSE = args.verbose
    if args.force:
        for entry in os.listdir(BUILD_DIR):
            path = os.path.join(BUILD_DIR, entry)
            if os.path.isfile(path) and entry.endswith(('.so', '.pyd', '.dll')):
                os.remove(path)


def build_bev_pool():
    src_dir = os.path.join(ROOT, "mmdet3d/ops/bev_pool/src")
    print("Building bev_pool_ext ...")
    mod = load(
        name="bev_pool_ext",
        sources=[
            os.path.join(src_dir, "bev_pool_cpu.cpp"),
            os.path.join(src_dir, "bev_pool_cuda.cu"),
        ],
        build_directory=BUILD_DIR,
        extra_cuda_cflags=CUDA_ARCH_FLAGS,
        verbose=VERBOSE,
    )
    print(f"  bev_pool_ext built: {mod}")
    return mod


def build_voxel_layer():
    src_dir = os.path.join(ROOT, "mmdet3d/ops/voxel/src")
    print("Building voxel_layer ...")
    mod = load(
        name="voxel_layer",
        sources=[
            os.path.join(src_dir, "voxelization.cpp"),
            os.path.join(src_dir, "voxelization_cpu.cpp"),
            os.path.join(src_dir, "voxelization_cuda.cu"),
            os.path.join(src_dir, "scatter_points_cpu.cpp"),
            os.path.join(src_dir, "scatter_points_cuda.cu"),
        ],
        build_directory=BUILD_DIR,
        extra_cflags=["-w", "-DWITH_CUDA"],
        extra_cuda_cflags=["-w", "-DWITH_CUDA"] + CUDA_ARCH_FLAGS,
        verbose=VERBOSE,
    )
    print(f"  voxel_layer built: {mod}")
    return mod


def build_iou3d():
    src_dir = os.path.join(ROOT, "mmdet3d/ops/iou3d/src")
    print("Building iou3d_cuda ...")
    mod = load(
        name="iou3d_cuda",
        sources=[
            os.path.join(src_dir, "iou3d.cpp"),
            os.path.join(src_dir, "iou3d_kernel.cu"),
        ],
        build_directory=BUILD_DIR,
        extra_cuda_cflags=CUDA_ARCH_FLAGS,
        verbose=VERBOSE,
    )
    print(f"  iou3d_cuda built: {mod}")
    return mod


def build_roiaware_pool3d():
    src_dir = os.path.join(ROOT, "mmdet3d/ops/roiaware_pool3d/src")
    print("Building roiaware_pool3d_ext ...")
    mod = load(
        name="roiaware_pool3d_ext",
        sources=[
            os.path.join(src_dir, "roiaware_pool3d.cpp"),
            os.path.join(src_dir, "roiaware_pool3d_kernel.cu"),
            os.path.join(src_dir, "points_in_boxes_cpu.cpp"),
            os.path.join(src_dir, "points_in_boxes_cuda.cu"),
        ],
        build_directory=BUILD_DIR,
        extra_cflags=["-w"],
        extra_cuda_cflags=["-w"] + CUDA_ARCH_FLAGS,
        verbose=VERBOSE,
    )
    print(f"  roiaware_pool3d_ext built: {mod}")
    return mod


if __name__ == "__main__":
    args = _parse_args()
    _configure_build(args)
    print(f"Build directory: {BUILD_DIR}")
    print(f"Python: {sys.executable}")

    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"CUDA arch flags: {CUDA_ARCH_FLAGS or 'torch default'}")
    if args.check_only:
        print("Build configuration check passed")
        raise SystemExit(0)

    build_bev_pool()
    build_voxel_layer()
    build_iou3d()
    build_roiaware_pool3d()

    import importlib
    for module_name in ("bev_pool_ext", "voxel_layer", "iou3d_cuda", "roiaware_pool3d_ext"):
        importlib.import_module(module_name)
        print(f"  Imported {module_name}")

    print("\nAll CUDA extensions built and imported successfully.")
