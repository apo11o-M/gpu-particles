# GPU Particle Simulator

Simulate particle collisions with C++ and NVIDIA GPUs!

## Getting Started

1. Install vcpkg if it's not already installed

```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh # for linux and macos
.\bootstrap-vcpkg.bat # for windows
```

2. Navigate to the root directory of this project and install the dependencies with vcpkg

```bash
./vcpkg install # for linux and macos
.\vcpkg.exe install # for windows
```

3. Configure the project with cmake by pointing cmake to the vcpkg toolchain file

```bash
cmake -B build -S . "-DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake"
```

4. Build the project, the executable will be located in the `build` directory

```bash
cmake --build build --config Release
```
