cmake_minimum_required(VERSION 3.18)
project(bev_pool_v2 LANGUAGES CXX CUDA)

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

add_library(bev_pool_v2_plugin SHARED
    src/bev_pool_v2_plugin.cpp
    src/bev_pool_v2_kernel.cu
)

target_include_directories(bev_pool_v2_plugin PRIVATE
    "${TRT_INCLUDE_DIR}"
    "${CUDA_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(bev_pool_v2_plugin PRIVATE
    "${TRT_INFER_LIBRARY}"
    "${TRT_PLUGIN_LIBRARY}"
    "${CUDA_RUNTIME_LIBRARY}"
)

target_compile_options(bev_pool_v2_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wno-deprecated-declarations>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-use_fast_math>
)

set_target_properties(bev_pool_v2_plugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "bev_pool_v2_plugin"
)

install(TARGETS bev_pool_v2_plugin LIBRARY DESTINATION lib)

message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TRT_INFER_LIBRARY}")
message(STATUS "CUDA include: ${CUDA_INCLUDE_DIR}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
