cmake_minimum_required(VERSION 3.18)
project(sparse_log2_quant LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(TRT_ROOT "" CACHE PATH "TensorRT installation prefix")
set(CUDA_ROOT "" CACHE PATH "CUDA installation prefix")
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures, for example 86;87;89")

if(TRT_ROOT STREQUAL "")
    find_path(TRT_INCLUDE_DIR NvInfer.h PATH_SUFFIXES include)
    find_library(TRT_INFER_LIBRARY nvinfer)
    find_library(TRT_PLUGIN_LIBRARY nvinfer_plugin)
else()
    set(TRT_INCLUDE_DIR "${TRT_ROOT}/include")
    find_library(TRT_INFER_LIBRARY nvinfer PATHS "${TRT_ROOT}/lib" NO_DEFAULT_PATH)
    find_library(TRT_PLUGIN_LIBRARY nvinfer_plugin PATHS "${TRT_ROOT}/lib" NO_DEFAULT_PATH)
endif()

if(CUDA_ROOT STREQUAL "")
    find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h PATH_SUFFIXES include)
    find_library(CUDA_RUNTIME_LIBRARY cudart)
else()
    set(CUDA_INCLUDE_DIR "${CUDA_ROOT}/include")
    find_library(CUDA_RUNTIME_LIBRARY cudart PATHS "${CUDA_ROOT}/lib64" "${CUDA_ROOT}/lib" NO_DEFAULT_PATH)
endif()

if(NOT EXISTS "${TRT_INCLUDE_DIR}/NvInfer.h")
    message(FATAL_ERROR "TensorRT headers not found. Set -DTRT_ROOT=/path/to/TensorRT")
endif()
if(NOT EXISTS "${CUDA_INCLUDE_DIR}/cuda_runtime_api.h")
    message(FATAL_ERROR "CUDA headers not found. Set -DCUDA_ROOT=/path/to/cuda")
endif()
if(NOT TRT_INFER_LIBRARY OR NOT TRT_PLUGIN_LIBRARY OR NOT CUDA_RUNTIME_LIBRARY)
    message(FATAL_ERROR "TensorRT/CUDA libraries not found; configure TRT_ROOT and CUDA_ROOT")
endif()

add_library(sparse_log2_quant_plugin SHARED
    src/sparse_log2_quant_plugin.cpp
    src/sparse_log2_quant_kernel.cu
)

target_include_directories(sparse_log2_quant_plugin PRIVATE
    "${TRT_INCLUDE_DIR}"
    "${CUDA_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(sparse_log2_quant_plugin PRIVATE
    "${TRT_INFER_LIBRARY}"
    "${TRT_PLUGIN_LIBRARY}"
    "${CUDA_RUNTIME_LIBRARY}"
)

target_compile_options(sparse_log2_quant_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wno-deprecated-declarations;-fstack-protector-strong;-D_FORTIFY_SOURCE=2>
    # 注意：不使用 -use_fast_math，避免 log2f/exp2f 退化为低精度近似而破坏对数域量化
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_link_options(sparse_log2_quant_plugin PRIVATE
    $<$<PLATFORM_ID:Linux>:-Wl,-z,now;-Wl,-z,noexecstack>
)

set_target_properties(sparse_log2_quant_plugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "sparse_log2_quant_plugin"
)

install(TARGETS sparse_log2_quant_plugin LIBRARY DESTINATION lib)

message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TRT_INFER_LIBRARY}")
message(STATUS "CUDA include: ${CUDA_INCLUDE_DIR}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
