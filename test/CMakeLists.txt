cmake_minimum_required(VERSION 3.10)

project(test LANGUAGES CXX)

find_package(GTest REQUIRED)

set(TEST_TENSOR test_tensor)
set(TEST_OPS test_ops)

add_executable(${TEST_TENSOR} test_tensor.cpp)
add_executable(${TEST_OPS} test_ops.cpp)

target_include_directories(${TEST_TENSOR} PUBLIC ${GTEST_INCLUDE_DIRS} ${PROJECT_INCLUDE_DIR})
target_link_libraries(${TEST_TENSOR} ${GTEST_LIBRARIES} nninfer)

target_include_directories(${TEST_OPS} PUBLIC ${GTEST_INCLUDE_DIRS} ${PROJECT_INCLUDE_DIR})
target_link_libraries(${TEST_OPS} ${GTEST_LIBRARIES} nninfer)
