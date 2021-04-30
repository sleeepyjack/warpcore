cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

file(GLOB WARPCORE_TESTS_SRC_LIST "${CMAKE_CURRENT_LIST_DIR}/src/*.cu")

add_executable(tester ${WARPCORE_TESTS_SRC_LIST})
target_include_directories(tester PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(tester PRIVATE warpcore)