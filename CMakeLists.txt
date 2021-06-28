cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(warpcore VERSION 0.0.1 LANGUAGES CXX CUDA)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

###################################################################################################
# Auto-detect available GPU compute architectures

set(GPU_ARCHS "" CACHE STRING "List of GPU architectures (semicolon-separated) to be compiled for. Empty string means to auto-detect the GPUs on the current system")

if("${GPU_ARCHS}" STREQUAL "")
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
endif()

###################################################################################################
# - find packages we depend on --------------------------------------------------------------------

find_package(CUDAToolkit 11.0 REQUIRED)

# package manager
include(cmake/CPM.cmake)

# macro for configuring executables
include(cmake/ConfigureExecutable.cmake)

# timers, utils, etc.
# TODO switch to default branch once merged
CPMAddPackage(
  NAME helpers
  URL https://gitlab.rlp.net/pararch/hpc_helpers/-/archive/restructure/hpc_helpers-restructure.zip
)

# lightweight GPU RNG
CPMAddPackage(
  NAME kiss_rng
  URL https://github.com/sleeepyjack/kiss_rng/archive/refs/heads/master.zip
)

###################################################################################################
# - warpcore target   -----------------------------------------------------------------------------

add_library(warpcore INTERFACE)
target_include_directories(warpcore INTERFACE
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                           $<INSTALL_INTERFACE:include>)
target_link_libraries(warpcore INTERFACE helpers kiss_rng CUDA::cudart)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

set(WARPCORE_TESTS_BINARY_DIR "${CMAKE_BINARY_DIR}/tests")
set(WARPCORE_BENCHMARKS_BINARY_DIR "${CMAKE_BINARY_DIR}/benchmarks")
set(WARPCORE_EXAMPLES_BINARY_DIR "${CMAKE_BINARY_DIR}/examples")

option(WARPCORE_BUILD_TESTS "Configure CMake to build tests" OFF)
option(WARPCORE_BUILD_BENCHMARKS "Configure CMake to build benchmarks" OFF)
option(WARPCORE_BUILD_EXAMPLES "Configure CMake to build examples" OFF)

if(WARPCORE_BUILD_TESTS)
    add_subdirectory(tests)
endif(WARPCORE_BUILD_TESTS)

if(WARPCORE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif(WARPCORE_BUILD_BENCHMARKS)

if(WARPCORE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif(WARPCORE_BUILD_EXAMPLES)
