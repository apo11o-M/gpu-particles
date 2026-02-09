# GPU Particle Simulator

Simulate particle collisions with C++ and NVIDIA GPUs!

## Getting Started

1. Clone the report with the `--recursive` flag to also clone the submodules

```bash
git clone --recursive https://github.com/apo11o-M/gpu-particles.git
```

2. Build the project with cmake

```bash
mkdir build
cd build
cmake --build . --config Release
```

3. Run the executable

```bash
./build/Release/gpu-particles -c ../default-config.json
```
