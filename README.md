# GPU Particle Simulator

Simulate particle collisions with C++ and NVIDIA GPUs!

## Getting Started

### Windows

To build the project from scratch, run the following command in the project root directory

`cmake -B build -S . "-DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake"`

`cmake --build build --config Release`

<!-- cmake -B build -S . "-DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake" && cmake --build build/ && .\build\Debug\gpu-particles.exe -->
