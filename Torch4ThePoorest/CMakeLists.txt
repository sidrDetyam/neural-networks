cmake_minimum_required(VERSION 3.24)
project(Torch4ThePoorest)

set(CMAKE_CXX_STANDARD 23)

include_directories(include)
include_directories(include/layers)
include_directories(include/interfaces)
include_directories(include/classification)
include_directories(include/data)

file(GLOB sources src/*.cpp)
file(GLOB headers include/*.h)
file(GLOB headers_i include/interfaces/*.h)
file(GLOB headers_l include/layers/*.h)
file(GLOB headers_d include/data/*.h)
file(GLOB headers_c include/classification/*.h)
file(GLOB tests tests/*.cpp)


#libs
add_subdirectory(lib/googletest)
include_directories(lib/googletest/googletest/include)
include_directories(lib/googletest/googlemock/include)
add_subdirectory(lib/csv-parser)
include_directories(lib/csv-parser/include)

find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS  "-O3")

set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
find_package(Boost REQUIRED COMPONENTS system filesystem serialization)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(Torch4ThePoorest
        ${sources}
        ${headers} ${headers_i} ${headers_l} ${headers_d} ${headers_c}
        ${tests}
        #main.cpp
        #models/Mushroom.cpp
        models/lenet/Lenet5_train.cpp
        models/lenet/Lenet5.cpp
        #models/lenet/Lenet5_test.cpp
        include/interfaces/IActivation.h include/layers/RnnCell.h src/RnnCell.cpp include/interfaces/IRegressionLossFunction.h include/MSELoss.h src/MSELoss.cpp)
target_link_libraries(Torch4ThePoorest ${BLAS_LIBRARIES} gtest gtest_main csv gcov ${Boost_LIBRARIES})
