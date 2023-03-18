cmake_minimum_required(VERSION 3.24)
project(MLP)

set(CMAKE_CXX_STANDARD 23)

find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})

add_executable(MLP main.cpp include/ILayer.h include/Batch.h src/Batch.cpp src/LinearLayer.cpp include/LinearLayer.h include/IBlas.h include/CpuBlas.h src/CpuBlas.cpp)
target_link_libraries(MLP ${BLAS_LIBRARIES})
