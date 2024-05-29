#include "particle.hpp"

// a circle can have at most 6 contacts with other circles with the same radius
// , we increase the maximum number of constraints to 10 just in case
#define MAX_CONSTRAINTS 10

// Cuda stuff are up here
// ============================================================================

// Macro to check for CUDA API call errors
void cudaErrorCheck(cudaError_t res, const char *func, const char *file,
                    int line, bool abort = true) {
    if (res == cudaSuccess) return;
    fprintf(stderr,
            "CUDA ERROR!\n\tFunction: %s\n\tLine: %d\n\tFile: %s\n\tError Name:"
            "%s\n\tError Description: %s\n\t",
            func, line, file, cudaGetErrorName(res), cudaGetErrorString(res));
    if (abort) exit(res);
}
#define cudaAssert(func) { cudaErrorCheck((func), #func, __FILE__, __LINE__); }

__constant__ unsigned int d_maxParticleCount;
__constant__ float d_radius;
__constant__ float d_mass;
__constant__ float d_restitution;
__constant__ float d_dampingFactor;
__constant__ float d_dampingFactorRate;
__constant__ unsigned int d_borderLeft, d_borderRight, d_borderTop, d_borderBottom;
__constant__ unsigned int d_cellXCount, d_cellYCount;

// ============================================================================

Particles::Particles(const SimulationConfig& config)
    : r(config.maxParticleCount, 255),
      g(config.maxParticleCount, 255),
      b(config.maxParticleCount, 255),
      position(config.maxParticleCount, Vec2<float>(0, 0)),
      velocity(config.maxParticleCount, Vec2<float>(0, 0)),
      isActive(config.maxParticleCount, FALSE),
      vertices(sf::Triangles, config.maxParticleCount * 6),
      shapes(config.maxParticleCount) {
    currIndex = 0;

    this->maxParticleCount = config.maxParticleCount;
    radius = config.particleRadius;
    mass = config.particleMass;
    restitution = config.restitutionCoefficient;
    dampingFactor = config.velocityDampingFactor;
    dampingFactorRate = config.velocityDampingFactorRate;

    this->borderLeft = config.borderLeft;
    this->borderRight = config.borderRight;
    this->borderTop = config.borderTop;
    this->borderBottom = config.borderBottom;

    cellSize = config.particleRadius * 2.5;
    this->cellXCount = (borderRight - borderLeft) / cellSize;
    this->cellYCount = (borderBottom - borderTop) / cellSize;

    cout << "Maximum particle count: " << config.maxParticleCount << endl;
    cout << "Borders: " << borderLeft << ", " << borderRight << ", " << borderTop << ", " << borderBottom << endl;
    cout << "Grid Count: " << cellXCount << ", " << cellYCount << endl;

    for (size_t i = 0; i < config.maxParticleCount; i++) {
        r[i] = rand() % 255;
        g[i] = rand() % 255;
        b[i] = rand() % 255;
    }

    mouseXPos = 0;
    mouseYPos = 0;
    spawn = FALSE;

    // ========================================================================
    // Cuda stuff are down here

    // initialize the constant memory in the device
    cudaAssert(cudaMemcpyToSymbol(d_maxParticleCount, &maxParticleCount, sizeof(unsigned int)));
    cudaAssert(cudaMemcpyToSymbol(d_radius, &radius, sizeof(float)));
    cudaAssert(cudaMemcpyToSymbol(d_mass, &mass, sizeof(float)));
    cudaAssert(cudaMemcpyToSymbol(d_restitution, &restitution, sizeof(float)));
    cudaAssert(cudaMemcpyToSymbol(d_dampingFactor, &dampingFactor, sizeof(float)));
    cudaAssert(cudaMemcpyToSymbol(d_dampingFactorRate, &dampingFactorRate, sizeof(float)));
    cudaAssert(cudaMemcpyToSymbol(d_borderLeft, &borderLeft, sizeof(unsigned int)));
    cudaAssert(cudaMemcpyToSymbol(d_borderRight, &borderRight, sizeof(unsigned int)));
    cudaAssert(cudaMemcpyToSymbol(d_borderTop, &borderTop, sizeof(unsigned int)));
    cudaAssert(cudaMemcpyToSymbol(d_borderBottom, &borderBottom, sizeof(unsigned int)));
    cudaAssert(cudaMemcpyToSymbol(d_cellXCount, &cellXCount, sizeof(unsigned int)));
    cudaAssert(cudaMemcpyToSymbol(d_cellYCount, &cellYCount, sizeof(unsigned int)));

    // allocate memory for the particles in the device
    cudaAssert(cudaMalloc(&d_positionIn, maxParticleCount * sizeof(Vec2<float>)));
    cudaAssert(cudaMalloc(&d_velocityIn, maxParticleCount * sizeof(Vec2<float>)));
    cudaAssert(cudaMalloc(&d_positionOut, maxParticleCount * sizeof(Vec2<float>)));
    cudaAssert(cudaMalloc(&d_velocityOut, maxParticleCount * sizeof(Vec2<float>)));
    cudaAssert(cudaMemcpy(d_positionIn, position.data(), maxParticleCount * sizeof(Vec2<float>), cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpy(d_velocityIn, velocity.data(), maxParticleCount * sizeof(Vec2<float>), cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpy(d_positionOut, position.data(), maxParticleCount * sizeof(Vec2<float>), cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpy(d_velocityOut, velocity.data(), maxParticleCount * sizeof(Vec2<float>), cudaMemcpyHostToDevice));

    cudaAssert(cudaMalloc(&d_isActive, maxParticleCount * sizeof(BOOL)));
    cudaAssert(cudaMemcpy(d_isActive, isActive.data(), maxParticleCount * sizeof(BOOL), cudaMemcpyHostToDevice));

    cudaAssert(cudaMalloc(&d_cellIndices, maxParticleCount * sizeof(int)));

    cudaDeviceProp properties;
    cudaAssert(cudaGetDeviceProperties(&properties, 0));
    h_maxBlockCount = properties.maxGridSize[0];
    h_maxThreadCount = properties.maxThreadsPerBlock;
    if (h_maxBlockCount < maxParticleCount) {
        cout << "The number of particles is too large for the device, abort\n"
             << "GPU max grid size: " << h_maxBlockCount
             << ", which is too low for " << maxParticleCount << " particles."
             << endl;
        abort();
    }
}

Particles::~Particles() {
    cudaAssert(cudaFree(d_positionIn));
    cudaAssert(cudaFree(d_velocityIn));
    cudaAssert(cudaFree(d_isActive));
    cudaAssert(cudaFree(d_positionOut));
    cudaAssert(cudaFree(d_velocityOut));
    cudaAssert(cudaFree(d_cellIndices));
}

void Particles::makeActive(unsigned int count, unsigned int xPos,
                           unsigned int yPos, float direction) {
    mouseXPos = xPos;
    mouseYPos = yPos;
    spawn = TRUE;
}

void Particles::makeInactive(unsigned int count) {}

void Particles::updateVertices(size_t startIndex, size_t endIndex, float deltaTime) {
    for (size_t i = startIndex; i < endIndex; i++) {
        if (!isActive[i]) continue;

        // interpolating the position to achieve smoother movement
        float x = position[i].x - radius + velocity[i].x * deltaTime;
        float y = position[i].y - radius + velocity[i].y * deltaTime;

        vertices[i * 6 + 0].position = sf::Vector2f(x - radius, y - radius);
        vertices[i * 6 + 1].position = sf::Vector2f(x + radius, y - radius);
        vertices[i * 6 + 2].position = sf::Vector2f(x + radius, y + radius);

        vertices[i * 6 + 3].position = sf::Vector2f(x - radius, y - radius);
        vertices[i * 6 + 4].position = sf::Vector2f(x + radius, y + radius);
        vertices[i * 6 + 5].position = sf::Vector2f(x - radius, y + radius);

        for (size_t j = 0; j < 6; j++) {
            vertices[i * 6 + j].color = sf::Color(r[i], g[i], b[i]);
        }
    }
}

void Particles::render(sf::RenderWindow &window, float deltaTime) {
    // 60 fps at 10000 particles
    // const size_t threadCount = min(currIndex / 500, thread::hardware_concurrency());
    // const size_t threadCount = thread::hardware_concurrency();
    // const size_t threadCount = 1;
    // const size_t chunkSize = maxParticleCount / threadCount;

    // vector<thread> threads;
    // for (size_t t = 0; t < threadCount; t++) {
    //     size_t startIndex = t * chunkSize;
    //     size_t endIndex = (t == threadCount - 1) ? maxParticleCount : startIndex + chunkSize;

    //     threads.emplace_back(&Particles::updateVertices, this, startIndex, endIndex, deltaTime);
    // }

    // for (thread &thread : threads) {
    //     thread.join();
    // }

    // window.draw(vertices);
    for (size_t i = 0; i < maxParticleCount; i++) {
        if (!isActive[i]) continue;

        // interpolating the position to achieve smoother movement
        shapes[i].setPosition(
            position[i].x - radius + velocity[i].x * deltaTime,
            position[i].y - radius + velocity[i].y * deltaTime);
        shapes[i].setRadius(radius);
        shapes[i].setFillColor(sf::Color(r[i], g[i], b[i]));
        window.draw(shapes[i]);
    }
}

void Particles::swapDeviceParticles() {
    Vec2<float> *tempPos = d_positionIn;
    d_positionIn = d_positionOut;
    d_positionOut = tempPos;
    Vec2<float> *tempVel = d_velocityIn;
    d_velocityIn = d_velocityOut;
    d_velocityOut = tempVel;
}

__device__ int getCellIndex(float xPos, float yPos, float cellSize, int gridXCount) {
    int x = (int)((xPos - d_borderLeft) / cellSize);
    int y = (int)((yPos - d_borderTop) / cellSize);
    return y * gridXCount + x;
}

__global__ void assignParticlesToCells(Vec2<float> *pos, int *cellIndices, float cellSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_maxParticleCount) return;

    cellIndices[i] = getCellIndex(pos[i].x, pos[i].y, cellSize, d_cellXCount);
}

__global__ void spawnParticleKernel(Vec2<float> *posIn, Vec2<float> *velIn, 
                                    Vec2<float> pos, unsigned int currIndex) {
    if (threadIdx.x == 0) {
        posIn[currIndex].x = max(d_borderLeft, min((unsigned int)pos.x, d_borderRight));
        posIn[currIndex].y = max(d_borderTop, min((unsigned int)pos.y, d_borderBottom));
        velIn[currIndex].x = -500;
        velIn[currIndex].y = 0;
    }
}

// The approach here is to have each block process one particle's collision
// against all other ones. Meaning n particles = n blocks. This also means each
// block will have n number of threads.
// There is a better way to approach this, which would require more complex
// index mapping but allow more efficient gpu utilization. 
__global__ void updateKernel(Vec2<float> *posIn, Vec2<float> *velIn,
                             const BOOL *d_isActive, const int *cellIndices, 
                             const float deltaTime, const float gravity, 
                             const float cellSize, 
                             Vec2<float> *posOut, Vec2<float> *velOut) {
    // represents the sum of all posDelta and velDelta of one particle
    __shared__ float posDeltaX;
    __shared__ float posDeltaY;
    __shared__ float velDeltaX;
    __shared__ float velDeltaY;

    // only the first thread in the block will update the shared memory
    if (threadIdx.x == 0) {
        posDeltaX = 0;
        posDeltaY = 0;
        velDeltaX = 0;
        velDeltaY = 0;
    }
    __syncthreads();

    // represents the delta of the collided particle
    Vec2<float> posDelta = Vec2<float>(0.0f, 0.0f);
    Vec2<float> velDelta = Vec2<float>(0.0f, 0.0f);

    int i = blockIdx.x;
    if (!d_isActive[i]) return;
    int cellXPos = (int)((posIn[i].x - d_borderLeft) / cellSize);
    int cellYPos = (int)((posIn[i].y - d_borderTop) / cellSize);

    // check for collisions within cells and neighboring cells
    for (int offsetY = -1; offsetY <= 1; offsetY++) {
        for (int offsetX = -1; offsetX <= 1; offsetX++) {
            int neighborX = cellXPos + offsetX;
            int neighborY = cellYPos + offsetY;
            if (neighborX < 0 || neighborX >= d_cellXCount 
                || neighborY < 0 || neighborY >= d_cellYCount) continue;

            int neighborIndex = neighborY * d_cellXCount + neighborX;

            for (int j = threadIdx.x; j < d_maxParticleCount; j += blockDim.x) {
                if (i == j || !d_isActive[j] || cellIndices[j] != neighborIndex) continue;

                Vec2<float> delta = posIn[i] - posIn[j];
                if (delta.lengthSq() == 0
                    || delta.lengthSq() > powf(d_radius + d_radius, 2)) continue;
                
                Vec2<float> deltaNorm = delta.normalized();
                float overlap = (d_radius + d_radius) - delta.length();
                posDelta += deltaNorm * overlap / 2.5f;

                Vec2<float> relativeVelocity = velIn[i] - velIn[j];
                float dotProd = dot(relativeVelocity, deltaNorm);
                if (dotProd <= 0) {
                    float impulse = 2 * dotProd / (d_mass + d_mass);
                    velDelta -= deltaNorm * impulse * d_mass * d_restitution;
                }
            }
        }
    }

    // udpate the shared memory with each particle's delta from the collision
    atomicAdd(&posDeltaX, posDelta.x);
    atomicAdd(&posDeltaY, posDelta.y);
    atomicAdd(&velDeltaX, velDelta.x);
    atomicAdd(&velDeltaY, velDelta.y);
    __syncthreads();

    // only the first thread in the block will update the shared memory
    if (threadIdx.x != 0) return;

    // coalesce the shared memory into the final position and velocity output
    velOut[i] = velIn[i] + Vec2<float>(velDeltaX, velDeltaY);
    velOut[i] *= powf(d_dampingFactor, d_dampingFactorRate * deltaTime);
    velOut[i].y += gravity * deltaTime;
    posOut[i] = posIn[i] + Vec2<float>(posDeltaX, posDeltaY) + velOut[i] * deltaTime;

    // check for border collisions
    if (posOut[i].x - d_radius < d_borderLeft) {
        posOut[i].x = d_borderLeft + d_radius;
        velOut[i].x *= -1 * d_restitution;
    }
    if (posOut[i].x + d_radius > d_borderRight) {
        posOut[i].x = d_borderRight - d_radius;
        velOut[i].x *= -1 * d_restitution;
    }
    if (posOut[i].y - d_radius < d_borderTop) {
        posOut[i].y = d_borderTop + d_radius;
        velOut[i].y *= -1 * d_restitution;
    }
    if (posOut[i].y + d_radius > d_borderBottom) {
        posOut[i].y = d_borderBottom - d_radius;
        velOut[i].y *= -1 * d_restitution;
    }
}

// Using position based dynamics approach
// 1) initialize the predicted position and velocity for the particles
// 2) update velocities based on external forces such as gravity
//    could also implment velocity damping or friction, etc
// 3) calculate the new positions using the current velocities
// 4) generate constraings for any collisions that may occur
// 5) loop
//    adjust the positions to satisfy both the initial and collision 
//    constraints using a gauss-seidel type iteration
// 6) update the final position and velocities based on the corrected 
//    positions

__global__ void applyForces(const Vec2<float> *posIn, const Vec2<float> *velIn, 
                            const float deltaTime, const float gravity, 
                            const BOOL *d_isActive, Vec2<float> *velOut) {
    int i = blockIdx.x;
    if (!d_isActive[i]) return;

    velOut[i] = velIn[i];
    velOut[i].y += gravity * deltaTime;
    velOut[i] *= powf(d_dampingFactor, d_dampingFactorRate * deltaTime);
    
    // if we collide with the border we also need to have the particles "bounce"
    // off the border
    if (posIn[i].x - d_radius < d_borderLeft) {
        velOut[i].x *= -1 * d_restitution;
    }
    if (posIn[i].x + d_radius > d_borderRight) {
        velOut[i].x *= -1 * d_restitution;
    }
    if (posIn[i].y - d_radius < d_borderTop) {
        velOut[i].y *= -1 * d_restitution;
    }
    if (posIn[i].y + d_radius > d_borderBottom) {
        velOut[i].y *= -1 * d_restitution;
    }
}

__global__ void predictPositions(const Vec2<float> *posIn, const Vec2<float> *velIn, 
                                 const BOOL *d_isActive, const float deltaTime, 
                                 Vec2<float> *posOut) {
    int i = blockIdx.x;
    if (!d_isActive[i]) return;

    for (int j = 0; j < d_maxParticleCount; j++) {
        if (i >= j || !d_isActive[j]) continue;
        
    }
    posOut[i] = posIn[i] + velIn[i] * deltaTime;
}

__global__ void collisionConstraints(const Vec2<float> *posIn,
                                const BOOL *d_isActive, const int maxConstraints, 
                                int solverIterations, Vec2<float> *posOut) {
    __shared__ int constraintCount;
    __shared__ Constraint constraints[MAX_CONSTRAINTS];

    if (threadIdx.x == 0) {
        constraintCount = 0;
    }
    __syncthreads();

    int i = blockIdx.x;
    for (int j = 0; j < d_maxParticleCount; j++) {
        if (i >= j || !d_isActive[i] || !d_isActive[j]) continue;

        Vec2<float> delta = posIn[i] - posIn[j];
        if (delta.lengthSq() == 0
            || delta.lengthSq() > powf(d_radius + d_radius, 2)) continue;
        
        int index = atomicAdd(&constraintCount, 1);
        if (index >= maxConstraints) break;
        constraints[index].particleA = i;
        constraints[index].particleB = j;
        constraints[index].isBorderConstraint = NOT_BORDER;
    }

    // check whether this particle is colliding with a border
    if (posIn[i].x - d_radius < d_borderLeft) {
        int index = atomicAdd(&constraintCount, 1);
        if (index < maxConstraints) {
            constraints[index].particleA = i;
            constraints[index].particleB = -1;
            constraints[index].isBorderConstraint = LEFT_BORDER;
        }
    }
    if (posIn[i].x + d_radius > d_borderRight) {
        int index = atomicAdd(&constraintCount, 1);
        if (index < maxConstraints) {
            constraints[index].particleA = i;
            constraints[index].particleB = -1;
            constraints[index].isBorderConstraint = RIGHT_BORDER;
        }
    }
    if (posIn[i].y - d_radius < d_borderTop) {
        int index = atomicAdd(&constraintCount, 1);
        if (index < maxConstraints) {
            constraints[index].particleA = i;
            constraints[index].particleB = -1;
            constraints[index].isBorderConstraint = TOP_BORDER;
        }
    }
    if (posIn[i].y + d_radius > d_borderBottom) {
        int index = atomicAdd(&constraintCount, 1);
        if (index < maxConstraints) {
            constraints[index].particleA = i;
            constraints[index].particleB = -1;
            constraints[index].isBorderConstraint = BOTTOM_BORDER;
        }
    }

    for (int index = 0; index < constraintCount; index++) {
        if (constraints[index].isBorderConstraint == NOT_BORDER) continue;

        int particleA = constraints[index].particleA;
        Vec2<float> correction = Vec2<float>(0.0f, 0.0f);
        switch (constraints[index].isBorderConstraint) {
            case LEFT_BORDER:
                correction.x = d_borderLeft + d_radius - posOut[particleA].x;
                break;
            case RIGHT_BORDER:
                correction.x = d_borderRight - d_radius - posOut[particleA].x;
                break;
            case TOP_BORDER:
                correction.y = d_borderTop + d_radius - posOut[particleA].y;
                break;
            case BOTTOM_BORDER:
                correction.y = d_borderBottom - d_radius - posOut[particleA].y;
                break;
            default:
                break;
        }
        posOut[particleA] += correction;
    }
    for (int index = 0; index < constraintCount; index++) {
        if (constraints[index].isBorderConstraint != NOT_BORDER) continue;

        int particleA = constraints[index].particleA;
        int particleB = constraints[index].particleB;

        Vec2<float> delta = posOut[particleA] - posOut[particleB];
        float dist = delta.length();
        if (dist < 2 * d_radius) {
            float overlap = 2 * d_radius - dist;
            Vec2<float> deltaNorm = delta.normalized();
            // TODO: when pushing the particles far apart, we should also take 
            // into account whether we would push the particles out of the 
            // border. If that is the case then we need to adjust the correction
            Vec2<float> correction = deltaNorm * overlap / 2.0f;
            BOOL doCorrectionA = TRUE, doCorrectionB = TRUE;
            if ((posOut[particleA] + correction).x - d_radius < d_borderLeft) {
                posOut[particleA].x = d_borderLeft + d_radius;
                doCorrectionA = FALSE;
            }
            if ((posOut[particleA] + correction).x + d_radius > d_borderRight) {
                posOut[particleA].x = d_borderRight - d_radius;
                doCorrectionA = FALSE;
            }
            if ((posOut[particleA] + correction).y - d_radius < d_borderTop) {
                posOut[particleA].y = d_borderTop + d_radius;
                doCorrectionA = FALSE;
            }
            if ((posOut[particleA] + correction).y + d_radius > d_borderBottom) {
                posOut[particleA].y = d_borderBottom - d_radius;
                doCorrectionA = FALSE;
            }
            if (doCorrectionA) {
                posOut[particleA] += correction;
            }

            if ((posOut[particleB] - correction).x - d_radius < d_borderLeft) {
                posOut[particleB].x = d_borderLeft + d_radius;
                doCorrectionB = FALSE;
            }
            if ((posOut[particleB] - correction).x + d_radius > d_borderRight) {
                posOut[particleB].x = d_borderRight - d_radius;
                doCorrectionB = FALSE;
            }
            if ((posOut[particleB] - correction).y - d_radius < d_borderTop) {
                posOut[particleB].y = d_borderTop + d_radius;
                doCorrectionB = FALSE;
            }
            if ((posOut[particleB] - correction).y + d_radius > d_borderBottom) {
                posOut[particleB].y = d_borderBottom - d_radius;
                doCorrectionB = FALSE;
            }
            if (doCorrectionB) {
                posOut[particleB] -= correction;
            }
            // posOut[particleA] += correction;
            // posOut[particleB] -= correction;
        }
    }
}

void Particles::update(float deltaTime, float gravity) {
    cudaStream_t stream;
    cudaAssert(cudaStreamCreate(&stream));

    // dim3 blocks = maxParticleCount;
    // dim3 threads = min(maxParticleCount, h_maxThreadCount);
    dim3 blocks = maxParticleCount;
    dim3 threads = 1;

    if (spawn && currIndex < maxParticleCount) {
        spawnParticleKernel<<<1, 1, 0, stream>>>(d_positionIn, d_velocityIn, 
            Vec2<float>(static_cast<float>(mouseXPos), static_cast<float>(mouseYPos)), currIndex);
        isActive[currIndex] = TRUE;
        currIndex++;
        spawn = FALSE;
    }
    cudaAssert(cudaMemcpyAsync(d_isActive, isActive.data(), maxParticleCount * sizeof(BOOL), cudaMemcpyHostToDevice, stream));
    
    // assignParticlesToCells<<<blocks, threads, 0, stream>>>(d_positionIn, d_cellIndices, cellSize);
    // updateKernel<<<blocks, threads, 0, stream>>>(d_positionIn, d_velocityIn, 
    //     d_isActive, d_cellIndices, deltaTime, gravity, cellSize, d_positionOut, d_velocityOut);
    
    applyForces<<<blocks, threads, 0, stream>>>(d_positionIn, d_velocityIn, deltaTime, gravity, d_isActive, d_velocityOut);
    cudaAssert(cudaStreamSynchronize(stream));

    predictPositions<<<blocks, threads, 0, stream>>>(d_positionIn, d_velocityOut, d_isActive, deltaTime, d_positionOut);
    cudaAssert(cudaStreamSynchronize(stream));
    
    for (int iter = 0; iter < 5; iter++) {
        collisionConstraints<<<blocks, threads, 0, stream>>>(d_positionOut, d_isActive, maxParticleCount * maxParticleCount, 5, d_positionOut);
        cudaAssert(cudaStreamSynchronize(stream));
    }

    // copy the calculated positions back to our host memory for rendering
    cudaAssert(cudaMemcpyAsync(position.data(), d_positionIn, maxParticleCount * sizeof(Vec2<float>), cudaMemcpyDeviceToHost, stream));
    swapDeviceParticles();
    cudaAssert(cudaStreamDestroy(stream));
}
