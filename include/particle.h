#ifndef PARTICLE_H
#define PARTICLE_H

#include "pch.h"

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

        // physics stuff
        vector<float> x, y, vx, vy, ax, ay;
        vector<float> radius, mass;
        vector<Vec2<float>> impulse;
        float dampingFactor;
        float pressureCoeff;
        float restitution;
        vector<bool> isActive;

        // rendering stuff
        vector<uint8_t> r, g, b;
        vector<sf::CircleShape> shapes;

    public:
        Particles(unsigned int maxParticleCount, unsigned int borderLeft, 
            unsigned int borderRight, unsigned int borderTop, 
            unsigned int borderBottom);

        ~Particles();

        /**
         * @brief Make the specified particles active
         * 
         * @param count How many particles to be active
         */
        void makeActive(unsigned int count, unsigned int x, unsigned int y, float direction);

        /**
         * @brief Make the specified particles inactive
         * 
         * @param count How many particles to be inactive
         */
        void makeInactive(unsigned int count);

        void update(float deltaTime, float gravity);

        void render(sf::RenderWindow& window, float deltaTime);

    private:
        /**
         * @brief Check if any of the particles are colliding with each other 
         * (and/or the walls). If so, calculate the net impulse at each particle
         * for later update
         */
        void collisionResolution(float deltaTime, float gravity);

        /**
         * @brief Calculate the impulse of the collision between particle i
         * and j. Here we are assuming both particles are circles and do not 
         * have any angular momentum.
         * 
         * @param i first particle
         * @param j second particle
         * @return Vec2<float> The result impulse vector
         */
        Vec2<float> calcImpulse(size_t i, size_t j);

        /**
         * @brief Check if the particle is on the ground, need to take into 
         * account of some thresholds because of floating point errors and 
         * oscillating math functions
         * 
         * @param i the index of the particle
         * @param threshold tolerance for the errors
         * @return whether the particle is on the ground
         */
        bool onGround(size_t i, float threshold);

        /**
         * @brief Check if the particle is not moving
         * 
         * @param i 
         * @param threshold 
         * @return whether the particle is not moving
         */
        bool isResting(size_t i, float threshold);
};

#endif // PARTICLE_H
