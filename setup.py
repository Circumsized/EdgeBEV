import os
import re
from typing import Iterable, List, Optional, Sequence

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


_DEFAULT_CUDA_ARCHES = ("7.0", "7.5", "8.0", "8.6", "8.7", "8.9")
_ARCH_TOKEN_RE = re.compile(r"^(?:sm_|compute_)?(?P<major>\d+)(?:\.(?P<minor>\d+)|(?P<minor_compact>\d+))$")
_PTX_SUFFIX_RE = re.compile(r"\+\s*ptx$", re.IGNORECASE)


def _normalize_arch_token(token: str):
    """Return ``(arch, with_ptx)`` or ``None`` for an invalid token."""
    token = token.strip().lower().replace("-", "")
    with_ptx = _PTX_SUFFIX_RE.search(token) is not None
    token = _PTX_SUFFIX_RE.sub("", token)
    match = _ARCH_TOKEN_RE.match(token)
    if match is None:
        return None
    major = int(match.group("major"))
    minor = match.group("minor") or match.group("minor_compact")
    if minor is None or not minor.isdigit() or len(minor) != 1:
        return None
    return f"{major}.{int(minor)}", with_ptx


def _configured_cuda_arches():
    configured = os.getenv("BEVFUSION_CUDA_ARCH_LIST") or os.getenv("TORCH_CUDA_ARCH_LIST")
    if configured:
        tokens = [t for t in re.split(r"[;,\s]+", configured) if t.strip()]
        seen = set()
        arches = []
        for token in tokens:
            parsed = _normalize_arch_token(token)
            if parsed is None or parsed[0] in seen:
                continue
            seen.add(parsed[0])
            arches.append(parsed)
    elif torch.cuda.is_available():
        pairs = {
            torch.cuda.get_device_capability(index)
            for index in range(torch.cuda.device_count())
        }
        arches = [(f"{major}.{minor}", False) for major, minor in pairs]
    else:
        arches = [(arch, False) for arch in _DEFAULT_CUDA_ARCHES]

    if not arches:
        raise RuntimeError(
            "No valid CUDA architectures found. Set BEVFUSION_CUDA_ARCH_LIST, "
            "for example: 8.6"
        )
    return arches


def _cuda_arch_flags(arches):
    flags = []
    for arch, with_ptx in arches:
        code = arch.replace(".", "")
        flags.append(f"-gencode=arch=compute_{code},code=sm_{code}")
        if with_ptx:
            flags.append(f"-gencode=arch=compute_{code},code=compute_{code}")
    return flags


def make_cuda_ext(
    name,
    module,
    sources,
    sources_cuda=None,
    extra_args=None,
    extra_include_path=None,
):
    sources = list(sources)
    sources_cuda = list(sources_cuda or [])
    extra_args = list(extra_args or [])
    extra_include_path = list(extra_include_path or [])

    define_macros = []
    extra_compile_args = {"cxx": extra_args}
    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    force_rocm = os.getenv("FORCE_ROCM", "0") == "1"
    has_cuda = torch.cuda.is_available() and torch.version.cuda is not None
    has_rocm = torch.cuda.is_available() and torch.version.hip is not None

    if has_cuda or force_cuda:
        define_macros.append(("WITH_CUDA", None))
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-allow-unsupported-compiler",
            "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        ] + _cuda_arch_flags(_configured_cuda_arches())
        sources.extend(sources_cuda)
    elif has_rocm or force_rocm:
        define_macros.append(("WITH_ROCM", None))
        extension = CUDAExtension
        extra_compile_args["hipcc"] = extra_args + [
            "-D__HIP_NO_HALF_OPERATORS__",
            "-D__HIP_NO_HALF_CONVERSIONS__",
            "-D__HIP_NO_HALF2_OPERATORS__",
        ]
        sources.extend(sources_cuda)
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), path) for path in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    setup(
        name="mmdet3d",
        packages=find_packages(),
        include_package_data=True,
        package_data={"mmdet3d.ops": ["*/*.so"]},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
        ext_modules=[
            make_cuda_ext(
                name="sparse_conv_ext",
                module="mmdet3d.ops.spconv",
                extra_include_path=[
                    # PyTorch 1.5 uses ninjia, which requires absolute path
                    # of included files, relative path will cause failure.
                    os.path.abspath(
                        os.path.join(*"mmdet3d.ops.spconv".split("."), "include/")
                    )
                ],
                sources=[
                    "src/all.cc",
                    "src/reordering_cpu.cc",
                    "src/reordering_cuda.cu",
                    "src/indice_cpu.cc",
                    "src/indice_cuda.cu",
                    "src/maxpool_cpu.cc",
                    "src/maxpool_cuda.cu",
                ],
                extra_args=["-w", "-std=c++17"],
            ),
            make_cuda_ext(
                name="bev_pool_ext",
                module="mmdet3d.ops.bev_pool",
                sources=[
                    "src/bev_pool_cpu.cpp",
                    "src/bev_pool_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="iou3d_cuda",
                module="mmdet3d.ops.iou3d",
                sources=[
                    "src/iou3d.cpp",
                    "src/iou3d_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="voxel_layer",
                module="mmdet3d.ops.voxel",
                sources=[
                    "src/voxelization.cpp",
                    "src/scatter_points_cpu.cpp",
                    "src/scatter_points_cuda.cu",
                    "src/voxelization_cpu.cpp",
                    "src/voxelization_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="roiaware_pool3d_ext",
                module="mmdet3d.ops.roiaware_pool3d",
                sources=[
                    "src/roiaware_pool3d.cpp",
                    "src/points_in_boxes_cpu.cpp",
                ],
                sources_cuda=[
                    "src/roiaware_pool3d_kernel.cu",
                    "src/points_in_boxes_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="ball_query_ext",
                module="mmdet3d.ops.ball_query",
                sources=["src/ball_query_cpu.cpp"],
                sources_cuda=["src/ball_query_cuda.cu"],
            ),
            make_cuda_ext(
                name="knn_ext",
                module="mmdet3d.ops.knn",
                sources=["src/knn_cpu.cpp"],
                sources_cuda=["src/knn_cuda.cu"],
            ),
            make_cuda_ext(
                name="assign_score_withk_ext",
                module="mmdet3d.ops.paconv",
                sources=["src/assign_score_withk.cpp"],
                sources_cuda=["src/assign_score_withk_cuda.cu"],
            ),
            make_cuda_ext(
                name="group_points_ext",
                module="mmdet3d.ops.group_points",
                sources=["src/group_points_cpu.cpp"],
                sources_cuda=["src/group_points_cuda.cu"],
            ),
            make_cuda_ext(
                name="interpolate_ext",
                module="mmdet3d.ops.interpolate",
                sources=["src/interpolate.cpp"],
                sources_cuda=["src/three_interpolate_cuda.cu", "src/three_nn_cuda.cu"],
            ),
            make_cuda_ext(
                name="furthest_point_sample_ext",
                module="mmdet3d.ops.furthest_point_sample",
                sources=["src/furthest_point_sample_cpu.cpp"],
                sources_cuda=["src/furthest_point_sample_cuda.cu"],
            ),
            make_cuda_ext(
                name="gather_points_ext",
                module="mmdet3d.ops.gather_points",
                sources=["src/gather_points_cpu.cpp"],
                sources_cuda=["src/gather_points_cuda.cu"],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )
