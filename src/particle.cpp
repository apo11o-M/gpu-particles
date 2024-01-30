#include "particle.hpp"

Particles::Particles(unsigned int maxParticles, unsigned int borderLeft,
                     unsigned int borderRight, unsigned int borderTop,
                     unsigned int borderBottom)
    : r(maxParticles, 66),
      g(maxParticles, 205),
      b(maxParticles, 227),
      position(maxParticles, Vec2<float>(0, 0)),
      velocity(maxParticles, Vec2<float>(0, 0)),
      mass(maxParticles, 20),
      radius(maxParticles, 7),
      isActive(maxParticles, false),
      shapes(maxParticles) {
    this->maxParticleCount = maxParticles;
    currIndex = 0;

    restitution = 0.6f;
    dampingFactor = 0.98f;
    dampingFactorRate = 60.0f;

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

Particles::~Particles() {}

void Particles::makeActive(unsigned int count, unsigned int xPos,
                           unsigned int yPos, float direction) {
    while (currIndex < maxParticleCount && count > 0) {
        position[currIndex] =
            Vec2<float>(static_cast<float>(xPos), static_cast<float>(yPos));
        velocity[currIndex] = Vec2<float>(-400 * (float)direction, 0);
        isActive[currIndex] = true;
        currIndex++;
        count--;
    }
}

void Particles::makeInactive(unsigned int count) {}

void Particles::render(sf::RenderWindow& window, float deltaTime) {
    // Not sure if this could be parallelized, depending on whether SFML is
    // happy with rendering from multiple threads
    for (size_t i = 0; i < maxParticleCount; i++) {
        if (!isActive[i]) continue;

        // interpolating the position to achieve smoother movement
        shapes[i].setPosition(
            position[i].x - radius[i] + velocity[i].x * deltaTime,
            position[i].y - radius[i] + velocity[i].y * deltaTime);
        shapes[i].setRadius(radius[i]);
        shapes[i].setFillColor(sf::Color(r[i], g[i], b[i]));
        window.draw(shapes[i]);
    }
}

void Particles::update(float deltaTime, float gravity) {
    // this could be further parallelized using
    // - OpenMP
    // - CUDA
    // - OpenCL
    // - threadpool

    vector<Vec2<float>> velocityDelta(maxParticleCount, Vec2<float>(0, 0));
    vector<Vec2<float>> positionDelta(maxParticleCount, Vec2<float>(0, 0));

    // compute velocity/position based on colliding particles
    for (size_t i = 0; i < maxParticleCount; i++) {
        if (!isActive[i]) continue;
        for (size_t j = i + 1; j < maxParticleCount; j++) {
            if (!isActive[j]) continue;

            Vec2<float> delta = position[i] - position[j];
            if (delta.lengthSq() == 0 ||
                delta.lengthSq() > pow(radius[i] + radius[j], 2))
                continue;

            Vec2<float> deltaNorm = delta.normalized();
            float overlap = (radius[i] + radius[j]) - delta.length();

            // push the particles away by half the overlapping distance
            positionDelta[i] += deltaNorm * overlap / 2.0f;
            positionDelta[j] -= deltaNorm * overlap / 2.0f;

            // update the velocity of both particles, elastic collision
            // with conservation of momentum
            Vec2<float> relativeVelocity = velocity[i] - velocity[j];
            float dotProd = dot(relativeVelocity, deltaNorm);
            if (dotProd > 0) continue;

            float impulse = 2 * dotProd / (mass[i] + mass[j]);
            velocityDelta[i] -= deltaNorm * impulse * mass[j] * restitution;
            velocityDelta[j] += deltaNorm * impulse * mass[i] * restitution;
        }
    }

    for (size_t i = 0; i < maxParticleCount; i++) {
        if (!isActive[i]) continue;

        // apply new velocity
        velocity[i] += velocityDelta[i];
        // dampen velocity by a certain factor n times per second
        velocity[i] *= pow(dampingFactor, dampingFactorRate * deltaTime);
        // apply gravity
        velocity[i].y += gravity * deltaTime;
        // apply velocity to position
        position[i] += positionDelta[i] + velocity[i] * deltaTime;

        // ensure particle dont leave the simulation bounds (left, right, top,
        // bottom)
        if (position[i].x - radius[i] < borderLeft) {
            position[i].x = borderLeft + radius[i];
            velocity[i].x *= -1 * restitution;
        }
        if (position[i].x + radius[i] > borderRight) {
            position[i].x = borderRight - radius[i];
            velocity[i].x *= -1 * restitution;
        }
        if (position[i].y - radius[i] < borderTop) {
            position[i].y = borderTop + radius[i];
            velocity[i].y *= -1 * restitution;
        }
        if (position[i].y + radius[i] > borderBottom) {
            position[i].y = borderBottom - radius[i];
            velocity[i].y *= -1 * restitution;
        }
    }
}
