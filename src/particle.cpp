#include "particle.h"

Particles::Particles(unsigned int maxParticleCount) : 
    r(maxParticleCount, 66), g(maxParticleCount, 205), b(maxParticleCount, 227),
    x(maxParticleCount, 0), y(maxParticleCount, 0), vx(maxParticleCount, 0),
    vy(maxParticleCount, 0), ax(maxParticleCount, 0), ay(maxParticleCount, 0),
    mass(maxParticleCount, 30), radius(maxParticleCount, 10), 
    isActive(maxParticleCount, false) {
    

}