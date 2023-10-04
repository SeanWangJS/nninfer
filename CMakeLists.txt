cmake_minimum_required(VERSION 3.10)

project(nninfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# find all source files
file(GLOB_RECURSE CPP_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

message(STATUS "CPP_SOURCES: ${CPP_SOURCES}")

add_library(${PROJECT_NAME} ${CPP_SOURCES})
# add_executable(${PROJECT_NAME} ${CPP_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)
add_subdirectory(test)
enable_testing()