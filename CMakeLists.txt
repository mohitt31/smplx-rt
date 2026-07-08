cmake_minimum_required(VERSION 3.22)
project(smplx_rt VERSION 0.2.0 LANGUAGES CXX)

option(SMPLXRT_WITH_ORT "Build the ONNX Runtime + OpenCV pose pipeline" OFF)
option(SMPLXRT_WITH_CUDA "Build custom CUDA kernels" OFF)
option(SMPLXRT_BUILD_TESTS "Build correctness tests" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Unpacked ONNX Runtime release (include/ and lib/)")

# Timings from an unoptimised build are meaningless, so default to Release.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(smplxrt
  src/preprocess/cpu_preprocess.cpp
  src/engine/engine.cpp
  src/pose/geometry.cpp
  src/smplx/lbs.cpp
  src/smplx/kinematics.cpp
  src/filter/one_euro.cpp
  src/filter/rotation_filter.cpp
  src/pipeline/pipeline.cpp
)
target_include_directories(smplxrt PUBLIC include)
target_compile_features(smplxrt PUBLIC cxx_std_17)
target_compile_options(smplxrt PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall;-Wextra;-Wpedantic>)

if(SMPLXRT_WITH_ORT)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)
  find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime REQUIRED)
  find_library(ORT_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib REQUIRED)
  get_filename_component(ORT_LIBRARY_DIR ${ORT_LIBRARY} DIRECTORY)

  target_sources(smplxrt PRIVATE src/engine/ort_engine.cpp src/pose/wholebody.cpp)
  target_include_directories(smplxrt SYSTEM PUBLIC ${ORT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(smplxrt PUBLIC ${ORT_LIBRARY} ${OpenCV_LIBS})
  target_compile_definitions(smplxrt PUBLIC SMPLXRT_WITH_ORT=1)

  add_executable(pose_video apps/pose_video.cpp)
  target_link_libraries(pose_video PRIVATE smplxrt)
  set_target_properties(pose_video PROPERTIES BUILD_RPATH ${ORT_LIBRARY_DIR})
endif()

if(SMPLXRT_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(smplxrt PRIVATE kernels/preprocess.cu kernels/lbs.cu)
  target_link_libraries(smplxrt PUBLIC CUDA::cudart)
  target_compile_definitions(smplxrt PUBLIC SMPLXRT_WITH_CUDA=1)
  set_target_properties(smplxrt PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(bench_stages bench/bench_stages.cpp)
target_link_libraries(bench_stages PRIVATE smplxrt)

if(SMPLXRT_BUILD_TESTS)
  enable_testing()
  foreach(name preprocess lbs_parity runtime pose_geometry)
    add_executable(test_${name} tests/test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE smplxrt)
    add_test(NAME ${name} COMMAND test_${name})
  endforeach()
endif()
