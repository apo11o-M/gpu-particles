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
    restitution = 0.6f;
    
    this->borderLeft = borderLeft;
    this->borderRight = borderRight;
    this->borderTop = borderTop;
    this->borderBottom = borderBottom;

    for (size_t i = 0; i < maxParticles; i++) {
        r[i] = rand() % 255;
        g[i] = rand() % 255;
        b[i] = rand() % 255;
    }
}

Particles::~Particles() { }

void Particles::makeActive(unsigned int count, unsigned int xPos, 
    unsigned int yPos, float direction) {
    while (currIndex < maxParticleCount && count > 0) {
        x[currIndex] = xPos;
        y[currIndex] = yPos;
        vy[currIndex] = 0;
        vx[currIndex] = -400 * (float)direction;
        isActive[currIndex] = true;
        currIndex++;
        count--;
    }
}

void Particles::makeInactive(unsigned int count) { }

void Particles::update(float deltaTime, float gravity) {
    // this could be further parallelized using
    // - OpenMP
    // - CUDA
    // - OpenCL
    // - threadpool

    // particle collision shit
    collisionResolution(deltaTime, gravity);

    // apply impulse to the velocity of each particles
    for (size_t i = 0; i < maxParticleCount; i++) {
        if (!isActive[i]) continue;
        vx[i] += impulse[i].x * mass[i];
        vy[i] += impulse[i].y * mass[i];
        impulse[i] = Vec2<float>::zero;
    }

    // If velocity is very small, set it to zero
    for (size_t i = 0; i < maxParticleCount; i++) {
        if (std::abs(vy[i]) < 60.0f) {
            vy[i] = 0.0;
        }
        if (std::abs(vx[i]) < 20.0f) {
            vx[i] = 0.0;
        }
    }

    // apply velocity to the position of each particles
    for (size_t i = 0; i < maxParticleCount; i++) {
        if (!isActive[i]) continue;
        // cout << vx[i] << " " << vy[i] << ", \tpos: " << x[i] << " " << y[i] << endl;
        vx[i] += ax[i] * (float)deltaTime;
        vy[i] += ay[i] * (float)deltaTime;
        x[i] += vx[i] * (float)deltaTime;
        y[i] += vy[i] * (float)deltaTime;
    }
}

void Particles::render(sf::RenderWindow& window, float deltaTime) {
    // Not sure if this could be parallelized
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

// we can use a quadtree to partition the space into smaller chunks, and 
// only check the particles within these chunks to improve performance.
// For now we will be using a naive approach, which is to check every 
// possible pair of collisions
void Particles::collisionResolution(float deltaTime, float gravity) {

    for (size_t i = 0; i < maxParticleCount; i++) {
        if (!isActive[i]) continue;

        // apply gravity only when the particle is not touching the bottom
        // border, or else the particle will try to phase through the border
        if (y[i] + radius[i] < borderBottom) {
            vy[i] += gravity * (float)deltaTime;
        }

        // left and right border collision
        if ((x[i] - radius[i] < borderLeft && vx[i] < 0) 
                || (x[i] + radius[i] > borderRight && vx[i] > 0)) {
            vx[i] *= -1 * restitution;
        }
        
        // bottom border collision
        if (y[i] - radius[i] < borderTop && vy[i] < 0) {
            vy[i] *= -1 * restitution;
        }
        
        // top border collision
        if (y[i] + radius[i] > borderBottom && vy[i] > 0) {
            vy[i] *= -1 * restitution;
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
}

bool Particles::onGround(size_t i, float threshold) {
    if (!isActive[i]) return false;
    if (abs(y[i] + radius[i] - borderBottom) < threshold) return true;

    return false;
}

bool Particles::isResting(size_t i, float threshold) {
    if (!isActive[i]) return false;
    if (abs(vx[i]) < threshold && abs(vy[i]) < threshold) return true;

    return false;
}