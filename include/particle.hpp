#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include "pch.hpp"

typedef enum {
    NOT_BORDER = 0,
    TOP_BORDER = 1,
    BOTTOM_BORDER = 2,
    LEFT_BORDER = 3,
    RIGHT_BORDER = 4,
} BorderCollisionType;

struct Constraint {
    int particleA, particleB;
    BorderCollisionType isBorderConstraint;
};

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
    unsigned int currIndex;
    unsigned int maxParticleCount;

    // the border of the scene, should be <= to the windows size
    // All particles should be bounded within this border
    unsigned int borderLeft, borderRight, borderTop, borderBottom;

    // the number of grids in spatial partioning
    unsigned int cellXCount, cellYCount;

    // mouse events
    unsigned int mouseXPos, mouseYPos;
    BOOL spawn;

    // physics stuff
    float cellSize;
    vector<Vec2<float>> velocity, position;
    float radius, mass;
    float dampingFactor, dampingFactorRate, restitution;
    vector<BOOL> isActive;

    // rendering stuff
    vector<uint8_t> r, g, b;
    sf::VertexArray vertices;

    // Cuda stuff, pointers to the device memory for particles
    Vec2<float> *d_positionIn, *d_velocityIn;
    Vec2<float> *d_positionOut, *d_velocityOut;
    BOOL* d_isActive;
    int *d_cellIndices;

    unsigned int h_maxBlockCount;
    unsigned int h_maxThreadCount;

    vector<sf::CircleShape> shapes;

   private:
    void swapDeviceParticles();
    void updateVertices(size_t startIndex, size_t endIndex, float deltaTime);

   public:
    Particles(const SimulationConfig& config);

    ~Particles();

    /**
     * @brief Make the specified particles active
     *
     * @param count How many particles to be active
     */
    void makeActive(unsigned int count, unsigned int x, unsigned int y,
                    float direction);

    /**
     * @brief Make the specified particles inactive
     *
     * @param count How many particles to be inactive
     */
    void makeInactive(unsigned int count);

    void update(float deltaTime, float gravity);

    void render(sf::RenderWindow& window, float deltaTime);
};

#endif  // PARTICLE_HPP
