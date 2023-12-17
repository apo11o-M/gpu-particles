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

void Particles::makeActive(unsigned int count, float direction) {
    while (currIndex < maxParticleCount && count > 0) {
        x[currIndex] = 100 + (direction < 0 ? 0.0f : 600.0f);
        y[currIndex] = 200;
        vy[currIndex] = 0;
        vx[currIndex] = -100 * (float)direction;
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
    for (size_t i = 0; i < maxParticleCount; i++) {
        if (!isActive[i]) continue;
        x[i] += vx[i] * (float)deltaTime;
        y[i] += vy[i] * (float)deltaTime;
        vx[i] += ax[i] * (float)deltaTime;
        vy[i] += ay[i] * (float)deltaTime;
    }
}

void Particles::collisionDetection() {
    // we can use a quadtree to partition the space into smaller chunks, and 
    // only check the particles within these chunks to improve performance.
    // For now we will be using a naive approach, which is to check every 
    // possible pair of collisions

    for (size_t i = 0; i < maxParticleCount - 1; i++) {
        if (!isActive[i]) continue;

        // check for collision with the border
        if (x[i] - radius[i] < 0) {
            cout << "outside left wall" << endl;
        } else if (x[i] + radius[i] > 800) {
            cout << "outside right wall" << endl;
        }
        if (y[i] - radius[i] < 0) {
            cout << "outside top wall" << endl;
        } else if (y[i] + radius[i] > 600) {
            cout << "outside bottom wall" << endl;
        }

        // check for collision with all other particles
        for (size_t j = i + 1; j < maxParticleCount; j++) {
            if (!isActive[j]) continue;
            // calculate the distance between the center of the two particles
            float dx = x[i] - x[j];
            float dy = y[i] - y[j];
            float dist = dx * dx + dy * dy;
            float totalRadius = radius[i] + radius[j];

            if (dist < totalRadius * totalRadius) {
                cout << "collision detected" << endl;
            }
        }
    }
}

void Particles::collisionResponse() {
    // Several cases:
    // case 1: particle collides with the border
    // case 2: particle collides with another particle
    // case 3: particle collides with multiple particles

    // an approach here would be using a force based model, where we calculate 
    // the net force vector on a particle, and then use that to move our 
    // particle in another direction.
}

void Particles::render(sf::RenderWindow& window, float deltaTime) {
    // Not sure if this could be parallelized, might have to use a threadpool
    // for this
    for (size_t i = 0; i < maxParticleCount; i++) {
        if (!isActive[i]) continue;

        // interpolating the position to achieve smoother movement
        shapes[i].setPosition(x[i] + vx[i] * deltaTime, y[i] + vy[i] * deltaTime);
        shapes[i].setRadius(radius[i]);
        shapes[i].setFillColor(sf::Color(r[i], g[i], b[i]));
        window.draw(shapes[i]);
    }
}