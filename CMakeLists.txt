cmake_minimum_required(VERSION 2.24)
project(gpu-particles)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# set all necessary flags for compiling
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-o3")
set(CMAKE_CXX_STANDARD 17)

# load the necessary libraries
find_package(SFML COMPONENTS system window graphics CONFIG REQUIRED)

add_executable(gpu-particles src/main.cpp)

# link external libraries
target_link_libraries(gpu-particles PRIVATE sfml-system sfml-network sfml-graphics sfml-window)
