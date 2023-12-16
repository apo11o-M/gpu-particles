#ifndef PARTICLE_H
#define PARTICLE_H

#include "pch.h"

// We are using Structure of Array as the design choice for better memory 
// access pattern and cache locality
class Particles {
    public:
        vector<uint8_t> r, g, b;
        vector<float> x, y, vx, vy, ax, ay;
        vector<float> radius, mass;
        vector<bool> isActive;

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


};

#endif // PARTICLE_H
