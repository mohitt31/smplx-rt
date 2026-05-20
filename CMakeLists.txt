cmake_minimum_required(VERSION 3.22)
project(smplx_rt VERSION 0.1.0 LANGUAGES CXX)

option(SMPLXRT_WITH_CUDA "Build custom CUDA kernels" OFF)
option(SMPLXRT_WITH_TRT "Enable TensorRT backend" OFF)
option(SMPLXRT_WITH_ORT "Enable ONNX Runtime backend" OFF)
option(SMPLXRT_WITH_VULKAN "Enable Vulkan renderer" OFF)
option(SMPLXRT_BUILD_TESTS "Build correctness tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(smplxrt
  src/preprocess/cpu_preprocess.cpp
  src/engine/engine.cpp
  src/smplx/lbs.cpp
  src/filter/one_euro.cpp
  src/pipeline/pipeline.cpp
)
target_include_directories(smplxrt PUBLIC include)
target_compile_features(smplxrt PUBLIC cxx_std_17)
target_compile_options(smplxrt PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/W4> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall;-Wextra;-Wpedantic>)

if(SMPLXRT_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(smplxrt PRIVATE kernels/preprocess.cu kernels/lbs.cu)
  target_link_libraries(smplxrt PUBLIC CUDA::cudart)
  target_compile_definitions(smplxrt PUBLIC SMPLXRT_WITH_CUDA=1)
  set_target_properties(smplxrt PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
endif()

if(SMPLXRT_WITH_TRT)
  target_compile_definitions(smplxrt PUBLIC SMPLXRT_WITH_TRT=1)
endif()
if(SMPLXRT_WITH_ORT)
  target_compile_definitions(smplxrt PUBLIC SMPLXRT_WITH_ORT=1)
endif()
if(SMPLXRT_WITH_VULKAN)
  target_compile_definitions(smplxrt PUBLIC SMPLXRT_WITH_VULKAN=1)
endif()

add_executable(smplxrt_demo src/main.cpp)
target_link_libraries(smplxrt_demo PRIVATE smplxrt)

add_executable(bench_stages bench/bench_stages.cpp)
target_link_libraries(bench_stages PRIVATE smplxrt)

if(SMPLXRT_BUILD_TESTS)
  enable_testing()
  add_executable(test_preprocess tests/test_preprocess.cpp)
  target_link_libraries(test_preprocess PRIVATE smplxrt)
  add_test(NAME preprocess COMMAND test_preprocess)
  add_executable(test_lbs_parity tests/test_lbs_parity.cpp)
  target_link_libraries(test_lbs_parity PRIVATE smplxrt)
  add_test(NAME lbs_parity COMMAND test_lbs_parity)
endif()
