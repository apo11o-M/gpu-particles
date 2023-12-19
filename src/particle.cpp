#include "particle.h"

Particles::Particles(unsigned int maxParticles, unsigned int borderLeft, 
    unsigned int borderRight, unsigned int borderTop, unsigned int borderBottom)
    : r(maxParticles, 66), g(maxParticles, 205), b(maxParticles, 227),
    x(maxParticles, 0), y(maxParticles, 0), vx(maxParticles, 0),
    vy(maxParticles, 0), ax(maxParticles, 0), ay(maxParticles, 0),
    mass(maxParticles, 30), radius(maxParticles, 7), 
    impulse(maxParticles, Vec2<float>(0, 0)), isActive(maxParticles, false), 
    shapes(maxParticles) {

    this->maxParticleCount = maxParticles;
    currIndex = 0;
    restitution = 0.9f;
    this->borderLeft = borderLeft;
    this->borderRight = borderRight;
    this->borderTop = borderTop;
    this->borderBottom = borderBottom;
}

Particles::~Particles() { }

void Particles::makeActive(unsigned int count, float direction) {
    while (currIndex < maxParticleCount && count > 0) {
        x[currIndex] = 100 + (direction < 0 ? 0.0f : 300.0f);
        y[currIndex] = 200 + (direction < 0 ? 5.0f : 0.0f);
        // y[currIndex] = 200;
        vy[currIndex] = 0;
        vx[currIndex] = -500 * (float)direction;
        isActive[currIndex] = true;
        currIndex++;
        count--;
    }
}

void Particles::makeInactive(unsigned int count) { }

void Particles::update(float deltaTime) {
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

Vec2<float> Particles::calcImpulse(size_t i, size_t j) {
    float relativeVX = vx[i] - vx[j];
    float relativeVY = vy[i] - vy[j];

    float dx = x[i] - x[j];
    float dy = y[i] - y[j];

    float dist = sqrt(dx * dx + dy * dy);
    if (dist == 0) return Vec2<float>(0, 0);
    dx /= dist;
    dy /= dist;

    float dotProd = relativeVX * dx + relativeVY * dy;

    // we don't need to calculate the impulse when the particles are moving 
    // away from each other 
    if (dotProd >= 0) return Vec2<float>(0, 0);

    float impulseScalar = -(1 + restitution) * dotProd / (mass[i] + mass[j]);

    return Vec2<float>(impulseScalar * dx, impulseScalar * dy);
}

void Particles::collisionResolution() {
    // we can use a quadtree to partition the space into smaller chunks, and 
    // only check the particles within these chunks to improve performance.
    // For now we will be using a naive approach, which is to check every 
    // possible pair of collisions

    for (size_t i = 0; i < maxParticleCount; i++) {
        if (!isActive[i]) continue;

        // check for collision with the border
        if (x[i] - radius[i] < borderLeft || x[i] + radius[i] > borderRight) {
            // cout << "outside left/right wall" << endl;
            vx[i] *= -1;
        }
        if (y[i] - radius[i] < borderTop || y[i] + radius[i] > borderBottom) {
            // cout << "outside top/bottom wall" << endl;
            vy[i] *= -1;
        }

        // check for collision with all other particles
        for (size_t j = i + 1; j < maxParticleCount; j++) {
            if (!isActive[j] || i == j) continue;
            // calculate the distance between the center of the two particles
            float dx = x[i] - x[j];
            float dy = y[i] - y[j];
            float dist = dx * dx + dy * dy;
            float totalRadius = radius[i] + radius[j];

            if (dist < totalRadius * totalRadius) {
                Vec2<float> impulseVector = calcImpulse(i, j);
                impulse[i] += impulseVector;
                impulse[j] -= impulseVector;
            }
        }
    }

    for (size_t i = 0; i < maxParticleCount; i++) {
        if (!isActive[i]) continue;
        vx[i] += impulse[i].x * mass[i];
        vy[i] += impulse[i].y * mass[i];
        // cout << impulse[i] << " " << vx[i] << " " << vy[i] << endl;
        impulse[i] = Vec2<float>::zero;
    }
}

void Particles::render(sf::RenderWindow& window, float deltaTime) {
    // Not sure if this could be parallelized, might have to use a threadpool
    // for this
    for (size_t i = 0; i < maxParticleCount; i++) {
        if (!isActive[i]) continue;

        // interpolating the position to achieve smoother movement
        shapes[i].setPosition(x[i] - radius[i] + vx[i] * deltaTime, 
                              y[i] - radius[i] + vy[i] * deltaTime);
        shapes[i].setRadius(radius[i]);
        shapes[i].setFillColor(sf::Color(r[i], g[i], b[i]));
        window.draw(shapes[i]);
    }
}