cmake_minimum_required(VERSION 3.24)
project(MLP)

set(CMAKE_CXX_STANDARD 23)

find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})

include_directories(include)

add_subdirectory(lib/googletest)
include_directories(lib/googletest/googletest/include)
include_directories(lib/googletest/googlemock/include)

add_executable(MLP include/ILayer.h include/Batch.h src/Batch.cpp src/LinearLayer.cpp include/LinearLayer.h include/IBlas.h include/CpuBlas.h src/CpuBlas.cpp include/Utils.h tests/LinearLayerTests.cpp
        #main.cpp
        include/ILossFunction.h include/CrossEntropyLoss.h src/CrossEntropyLoss.cpp tests/CrossEntopyTest.cpp include/ReLU.h src/ReLU.cpp include/Model.h src/Model.cpp include/Optimizer.h)
target_link_libraries(MLP ${BLAS_LIBRARIES} gtest gtest_main)
#target_link_libraries(MLP