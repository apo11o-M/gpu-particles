#ifndef PCH_HPP
#define PCH_HPP

// sfml windows and drawing functions
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

// json parsing
#include "json.hpp"

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
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>

// common namespaces
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::vector;
using std::thread;
using json = nlohmann::json;

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

struct SimulationConfig {
    string name = "Rectangle Particle Demo";
    unsigned int windowWidth = 1000;
    unsigned int windowHeight = 800;
    unsigned int borderLeft = 200;
    unsigned int borderRight = 800;
    unsigned int borderTop = 100;
    unsigned int borderBottom = 700;
    string backgroundColor = "0x000000ff";
    
    unsigned int maxParticleCount = 1000;
    float particleRadius = 3.0f;
    float particleMass = 20.0f;

    float velocityDampingFactor = 0.95f;
    float velocityDampingFactorRate = 60.0f;
    float restitutionCoefficient = 0.6f;
    float gravity = 200.0f;

    float maxSuctionRange = 50.0f;
    float suctionForce = 50.0f;
    float maxRepelRange = 50.0f;
    float repelForce = 1000.0f;

    static SimulationConfig fromJson(const json& json) {
        SimulationConfig config;
        if (json.contains("name")) config.name = json["name"];
        if (json.contains("windowWidth")) config.windowWidth = json["windowWidth"];
        if (json.contains("windowHeight")) config.windowHeight = json["windowHeight"];
        if (json.contains("borderLeft")) config.borderLeft = json["borderLeft"];
        if (json.contains("borderRight")) config.borderRight = json["borderRight"];
        if (json.contains("borderTop")) config.borderTop = json["borderTop"];
        if (json.contains("borderBottom")) config.borderBottom = json["borderBottom"];
        if (json.contains("backgroundColor")) config.backgroundColor = json["backgroundColor"];
        if (json.contains("maxParticleCount")) config.maxParticleCount = json["maxParticleCount"];
        if (json.contains("particleRadius")) config.particleRadius = json["particleRadius"];
        if (json.contains("particleMass")) config.particleMass = json["particleMass"];
        if (json.contains("velocityDampingFactor")) config.velocityDampingFactor = json["velocityDampingFactor"];
        if (json.contains("velocityDampingFactorRate")) config.velocityDampingFactorRate = json["velocityDampingFactorRate"];
        if (json.contains("restitutionCoefficient")) config.restitutionCoefficient = json["restitutionCoefficient"];
        if (json.contains("gravity")) config.gravity = json["gravity"];
        if (json.contains("maxSuctionRange")) config.maxSuctionRange = json["maxSuctionRange"];
        if (json.contains("suctionForce")) config.suctionForce = json["suctionForce"];
        if (json.contains("maxRepelRange")) config.maxRepelRange = json["maxRepelRange"];
        if (json.contains("repelForce")) config.repelForce = json["repelForce"];
        return config;
    }
};

#endif  // PCH_HPP
