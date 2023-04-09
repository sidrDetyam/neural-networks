cmake_minimum_required(VERSION 3.24)
project(Torch4ThePoorest)

set(CMAKE_CXX_STANDARD 23)

find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})

include_directories(include)

add_subdirectory(lib/googletest)
include_directories(lib/googletest/googletest/include)
include_directories(lib/googletest/googlemock/include)

add_subdirectory(lib/csv-parser)
include_directories(lib/csv-parser/include)

add_executable(Torch4ThePoorest include/ILayer.h include/Tensor.h src/Tensor.cpp src/LinearLayer.cpp include/LinearLayer.h include/IBlas.h include/CpuBlas.h src/CpuBlas.cpp include/Utils.h tests/LinearLayerTests.cpp
        include/IClassificationLossFunction.h include/CrossEntropyLoss.h src/CrossEntropyLoss.cpp tests/CrossEntopyTest.cpp include/ReLU.h src/ReLU.cpp include/Model.h src/Model.cpp include/SgdOptimizer.h src/SgdOptimizer.cpp src/Utils.cpp include/Conv2d.h src/Conv2d.cpp include/IOptimizerCreator.h include/IOptimizer.h include/SgdOptimizerCreator.h src/SgdOptimizerCreator.cpp

        main.cpp
        #models/Mushroom.cpp
        )
target_link_libraries(Torch4ThePoorest ${BLAS_LIBRARIES} gtest gtest_main csv gcov)
#target_link_libraries(Torch4ThePoorest