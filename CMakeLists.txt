cmake_minimum_required(VERSION 2.24)
project(gpu-particles)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# set all necessary flags for compiling. Same flags, but just different format
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # using Visual Studio C++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "/Ox")
else()
    # using GCC or Clang or something else 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

# load the necessary libraries
find_package(SFML COMPONENTS system window graphics CONFIG REQUIRED)
find_package(FLAC CONFIG REQUIRED)  

add_executable(gpu-particles src/main.cpp src/scene.cpp src/particle.cpp)

# add any include folders for this project
target_include_directories(gpu-particles PRIVATE include)

# link external libraries
target_link_libraries(gpu-particles PRIVATE sfml-system sfml-network sfml-graphics sfml-window)
