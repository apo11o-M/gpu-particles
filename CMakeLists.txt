cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(gpu-particles LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

# set all necessary flags for compiling. Same flags, but for different compilers
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # using Visual Studio C++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "/Ox")
else()
    # using GCC or Clang or something else
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

# Build SFML statically
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SFML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# Optional: reduce dependencies
set(SFML_BUILD_AUDIO OFF CACHE BOOL "" FORCE)     # strongly recommended if you want easy static
set(SFML_BUILD_NETWORK OFF CACHE BOOL "" FORCE)   # if not used

# load the necessary libraries
# find_package(SFML COMPONENTS system window graphics CONFIG REQUIRED)
add_subdirectory(extern/SFML)

# set the source files for this project
set(CXX_SOURCES src/main.cpp src/scene.cpp)
set(CUDA_SOURCES src/particle.cu)

add_executable(gpu-particles ${CXX_SOURCES} ${CUDA_SOURCES})

target_compile_definitions(gpu-particles PRIVATE SFML_STATIC)

# add any include folders for this project
target_include_directories(gpu-particles PRIVATE include)

# link external libraries
target_link_libraries(gpu-particles PRIVATE sfml-system sfml-graphics sfml-window)
