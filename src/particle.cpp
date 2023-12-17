#include "particle.h"

Particles::Particles(unsigned int maxParticleCount) : 
    r(maxParticleCount, 66), g(maxParticleCount, 205), b(maxParticleCount, 227),
    x(maxParticleCount, 0), y(maxParticleCount, 0), vx(maxParticleCount, 0),
    vy(maxParticleCount, 0), ax(maxParticleCount, 0), ay(maxParticleCount, 0),
    mass(maxParticleCount, 30), radius(maxParticleCount, 7), 
    isActive(maxParticleCount, false), shapes(maxParticleCount) {
    this->maxParticleCount = maxParticleCount;
    currIndex = 0;
}

Particles::~Particles() { }

void Particles::makeActive(unsigned int count) {
    while (currIndex < maxParticleCount && count > 0) {
        x[currIndex] = 100;
        y[currIndex] = 200;
        vy[currIndex] = -10;
        vx[currIndex] = -20;
        isActive[currIndex] = true;
        currIndex++;
        count--;
    }
}

void Particles::makeInactive(unsigned int count) { }

void Particles::update(double deltaTime) {
    // this could be further parallelized using
    // - OpenMP
    // - CUDA
    // - OpenCL
    // - threadpool
    for (unsigned int i = 0; i < maxParticleCount; i++) {
        if (isActive[i]) {
            x[i] += vx[i] * (float)deltaTime;
            y[i] += vy[i] * (float)deltaTime;
            vx[i] += ax[i] * (float)deltaTime;
            vy[i] += ay[i] * (float)deltaTime;
        }
    }
}



void Particles::render(sf::RenderWindow& window) {
    // Not sure if this could be parallelized, might have to use a threadpool
    // for this
    for (unsigned int i = 0; i < maxParticleCount; i++) {
        if (isActive[i]) {
            shapes[i].setPosition(x[i], y[i]);
            shapes[i].setRadius(radius[i]);
            shapes[i].setFillColor(sf::Color(r[i], g[i], b[i]));
            window.draw(shapes[i]);
        }
    }
}