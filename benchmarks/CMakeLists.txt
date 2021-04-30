cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

ConfigureExecutable(bloom_filter_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/bloom_filter_benchmark.cu" "${WARPCORE_BENCHMARK_BINARY_DIR}")
ConfigureExecutable(bucket_list_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/bucket_list_benchmark.cu" "${WARPCORE_BENCHMARK_BINARY_DIR}")
ConfigureExecutable(counting_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/counting_benchmark.cu" "${WARPCORE_BENCHMARK_BINARY_DIR}")
ConfigureExecutable(multi_value_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/multi_value_benchmark.cu" "${WARPCORE_BENCHMARK_BINARY_DIR}")
ConfigureExecutable(single_value_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/single_value_benchmark.cu" "${WARPCORE_BENCHMARK_BINARY_DIR}")
ConfigureExecutable(multi_bucket_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/src/multi_bucket_benchmark.cu" "${WARPCORE_BENCHMARK_BINARY_DIR}")