cmake_minimum_required(VERSION 3.24)
project(Torch4ThePoorest)

set(CMAKE_CXX_STANDARD 23)

include_directories(include)
include_directories(include/layers)
include_directories(include/interfaces)

file(GLOB sources src/*.cpp)
file(GLOB headers include/*.h)
file(GLOB headers_i include/interfaces/*.h)
file(GLOB headers_l include/layers/*.h)
file(GLOB tests tests/*.cpp)


#libs
add_subdirectory(lib/googletest)
include_directories(lib/googletest/googletest/include)
include_directories(lib/googletest/googlemock/include)
add_subdirectory(lib/csv-parser)
include_directories(lib/csv-parser/include)

find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})


add_executable(Torch4ThePoorest
        ${sources}
        ${headers} ${headers_i} ${headers_l}
        ${tests}
        main.cpp
        #models/Mushroom.cpp
        #models/Lenet5.cpp
        include/CsvDataLoader.h src/CsvDataLoader.cpp include/layers/Reshaper.h src/Reshaper.cpp include/layers/Dropout.h src/Dropout.cpp include/layers/Tanh.h src/Tanh.cpp include/layers/Conv2d.h src/Conv2d.cpp)
target_link_libraries(Torch4ThePoorest ${BLAS_LIBRARIES} gtest gtest_main csv gcov)
