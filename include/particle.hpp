#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include "pch.hpp"

// Using Structure of Array as the design choice for better memory access
// pattern and cache locality
//
// Here are the units for each variables, using MKS system:
// - mass: kg
// - radius: meter
// - x, y: pixel position, 1 pixel = 1 meter
// - vx, vy: m/s
// - ax, ay: m/s^2
//
// Also, we're using newtonian physics here, and uses the following formula:
// - v = v_0 + a * t
// - s = s_0 + v_0 * t + 1/2 * a * t^2
// Note the t here means the delta time, which gets passed in as a parameter by
// the update function
class Particles {
   public:
    // All particles from index 0 to currActiveIndex are active
    int currActiveIndex;
    unsigned int maxParticleCount;

    // the border of the scene, should be <= to the windows size
    // All particles should be bounded within this border
    unsigned int borderLeft, borderRight, borderTop, borderBottom;

    // the number of grids in spatial partioning
    unsigned int cellXCount, cellYCount;
    float cellSize;

    // mouse events
    BOOL spawn, succ, repel;
    unsigned int mouseXPos, mouseYPos;
    unsigned int spawnCount;
    float maxSuctionRange, suctionForce;
    float maxRepelRange, repelForce;

    // physics stuff
    vector<Vec2<float>> velocity, position;
    float radius, mass;
    float dampingFactor, dampingFactorRate, restitution;

    // rendering stuff
    vector<uint8_t> r, g, b;
    sf::VertexArray vertices;
    sf::Texture texture;
    const size_t renderingThreads, chunkSize;

    // Cuda physics stuff, pointers to the device memory for particles
    Vec2<float> *d_position, *d_velocity;
    // see https://www.youtube.com/watch?v=D2M8jTtKi44 at 6:05, amazing stuff
    int *d_spatialHashTable, *d_particleIndices;
    
    // Cuda device properties, for the number of blocks and threads
    unsigned int gpu_maxBlockCount, gpu_maxThreadCount;

   private:
    void updateVertices(size_t startIndex, size_t endIndex, float deltaTime);

   public:
    Particles(const SimulationConfig& config);

    ~Particles();

    void spawnParticles(unsigned int x, unsigned int y, BOOL shouldSpawn);

    void succParticles(unsigned int x, unsigned int y, BOOL shouldSucc);

    void repelParticles(unsigned int x, unsigned int y, BOOL shouldRepel);

    void update(float deltaTime, float gravity);

    void render(sf::RenderWindow& window, float deltaTime);
};

#endif  // PARTICLE_HPP
