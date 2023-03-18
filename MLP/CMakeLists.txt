cmake_minimum_required(VERSION 3.24)
project(MLP)

set(CMAKE_CXX_STANDARD 23)

find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})

add_executable(MLP main.cpp)
target_link_libraries(MLP ${BLAS_LIBRARIES})
