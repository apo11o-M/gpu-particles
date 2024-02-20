#ifndef PCH_HPP
#define PCH_HPP

// sfml windows and drawing functions
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

// json parsing
#include <nlohmann/json.hpp>

// CUDA stuff
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// C++ standard libraries
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// common namespaces
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;

// some math libraries
#include "vec2.hpp"

// some common definitions
#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef BOOL
#define BOOL int
#endif

#endif  // PCH_HPP
