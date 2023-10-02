cmake_minimum_required(VERSION 3.10)

project(test_tensor LANGUAGES CXX)

find_package(GTest REQUIRED)

add_executable(${PROJECT_NAME} test_tensor.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${GTEST_INCLUDE_DIRS} ${PROJECT_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${GTEST_LIBRARIES} nninfer)
