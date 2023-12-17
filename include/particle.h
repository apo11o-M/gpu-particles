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

        vector<uint8_t> r, g, b;
        vector<float> x, y, vx, vy, ax, ay;
        vector<float> radius, mass;
        vector<bool> isActive;

        vector<sf::CircleShape> shapes;

    public:
        Particles(unsigned int maxParticleCount);

        ~Particles();

        /**
         * @brief Make the specified particles active
         * 
         * @param count How many particles to be active
         */
        void makeActive(unsigned int count);

        /**
         * @brief Make the specified particles inactive
         * 
         * @param count How many particles to be inactive
         */
        void makeInactive(unsigned int count);

        void update(double deltaTime);

        void render(sf::RenderWindow& window);
};

#endif // PARTICLE_H
