cmake_minimum_required(VERSION 3.18 FATAL_ERROR)



ConfigureExecutable(advanced_usage_from_device "${CMAKE_CURRENT_SOURCE_DIR}/src/advanced_usage_from_device.cu" "${WARPCORE_EXAMPLES_BINARY_DIR}")
ConfigureExecutable(basic_usage_from_host "${CMAKE_CURRENT_SOURCE_DIR}/src/basic_usage_from_host.cu" "${WARPCORE_EXAMPLES_BINARY_DIR}")
ConfigureExecutable(bucket_list_hash_table "${CMAKE_CURRENT_SOURCE_DIR}/src/bucket_list_hash_table.cu" "${WARPCORE_EXAMPLES_BINARY_DIR}")
ConfigureExecutable(counting_hash_table "${CMAKE_CURRENT_SOURCE_DIR}/src/counting_hash_table.cu" "${WARPCORE_EXAMPLES_BINARY_DIR}")
ConfigureExecutable(multi_value_hash_table "${CMAKE_CURRENT_SOURCE_DIR}/src/multi_value_hash_table.cu" "${WARPCORE_EXAMPLES_BINARY_DIR}")
ConfigureExecutable(unique_random_generator "${CMAKE_CURRENT_SOURCE_DIR}/src/unique_random_generator.cu" "${WARPCORE_EXAMPLES_BINARY_DIR}")