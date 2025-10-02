cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Add NVBench for the Bloom filter benchmark
CPMAddPackage(
  NAME nvbench
  GITHUB_REPOSITORY NVIDIA/nvbench
  GIT_TAG 4879607c7086f3ebae2f8b9655d0b920c41d22ef
)

# Bloom filter benchmark using NVBench
add_executable(nvbench_bloom_filter_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/nvbench_bloom_filter_benchmark.cu")
target_link_libraries(nvbench_bloom_filter_benchmark PRIVATE nvbench::main warpcore)
target_include_directories(nvbench_bloom_filter_benchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(nvbench_bloom_filter_benchmark PROPERTIES
                      CUDA_ARCHITECTURES ${GPU_ARCHS}
                      RUNTIME_OUTPUT_DIRECTORY "${WARPCORE_BENCHMARK_BINARY_DIR}")

# Other benchmarks not using NVBench
ConfigureExecutable(bloom_filter_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/bloom_filter_benchmark.cu" "${WARPCORE_BENCHMARK_BINARY_DIR}")
ConfigureExecutable(bucket_list_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/bucket_list_benchmark.cu" "${WARPCORE_BENCHMARK_BINARY_DIR}")
ConfigureExecutable(counting_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/counting_benchmark.cu" "${WARPCORE_BENCHMARK_BINARY_DIR}")
ConfigureExecutable(multi_value_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/multi_value_benchmark.cu" "${WARPCORE_BENCHMARK_BINARY_DIR}")
ConfigureExecutable(single_value_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/single_value_benchmark.cu" "${WARPCORE_BENCHMARK_BINARY_DIR}")
ConfigureExecutable(multi_bucket_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/multi_bucket_benchmark.cu" "${WARPCORE_BENCHMARK_BINARY_DIR}")