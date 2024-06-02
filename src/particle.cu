#include "particle.hpp"

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
__constant__ float d_maxSuctionRange, d_suctionForce;
__constant__ float d_maxRepelRange, d_repelForce;

// ============================================================================

Particles::Particles(const SimulationConfig& config)
    : r(config.maxParticleCount, 255),
      g(config.maxParticleCount, 255),
      b(config.maxParticleCount, 255),
      position(config.maxParticleCount, Vec2<float>(0, 0)),
      velocity(config.maxParticleCount, Vec2<float>(0, 0)),
      isActive(config.maxParticleCount, false),
      vertices(sf::Triangles, config.maxParticleCount * 6),
      renderingThreads(thread::hardware_concurrency() / 2),
      chunkSize(config.maxParticleCount / renderingThreads) {
    currIndex = 0;

    this->maxParticleCount = config.maxParticleCount;
    radius = config.particleRadius;
    mass = config.particleMass;
    restitution = config.restitutionCoefficient;
    dampingFactor = config.velocityDampingFactor;
    dampingFactorRate = config.velocityDampingFactorRate;
    maxSuctionRange = config.maxSuctionRange;
    suctionForce = config.suctionForce;
    maxRepelRange = config.maxRepelRange;
    repelForce = config.repelForce;

    this->borderLeft = config.borderLeft;
    this->borderRight = config.borderRight;
    this->borderTop = config.borderTop;
    this->borderBottom = config.borderBottom;

    cellSize = config.particleRadius * 2.5;
    this->cellXCount = (borderRight - borderLeft) / cellSize;
    this->cellYCount = (borderBottom - borderTop) / cellSize;

    cout << "Maximum particle count: " << maxParticleCount << endl;
    cout << "Borders: " << borderLeft << ", " << borderRight << ", " << borderTop << ", " << borderBottom << endl;
    cout << "Grid Count, X: " << cellXCount << ", Y: " << cellYCount << endl;

    for (size_t i = 0; i < maxParticleCount; i++) {
        r[i] = rand() % 255;
        g[i] = rand() % 255;
        b[i] = rand() % 255;
    }

    mouseXPos = 0;
    mouseYPos = 0;
    spawn = FALSE;
    succ = FALSE;
    repel = FALSE;

    spawnCount = config.spawnCount;

    // ========================================================================
    // Cuda stuff are down here

    // initialize the constant memory in the device
    cudaAssert(cudaMemcpyToSymbol(d_maxParticleCount, &maxParticleCount, sizeof(unsigned int)));
    cudaAssert(cudaMemcpyToSymbol(d_radius, &radius, sizeof(float)));
    cudaAssert(cudaMemcpyToSymbol(d_mass, &mass, sizeof(float)));
    cudaAssert(cudaMemcpyToSymbol(d_restitution, &restitution, sizeof(float)));
    cudaAssert(cudaMemcpyToSymbol(d_dampingFactor, &dampingFactor, sizeof(float)));
    cudaAssert(cudaMemcpyToSymbol(d_dampingFactorRate, &dampingFactorRate, sizeof(float)));
    cudaAssert(cudaMemcpyToSymbol(d_maxSuctionRange, &maxSuctionRange, sizeof(float)));
    cudaAssert(cudaMemcpyToSymbol(d_suctionForce, &suctionForce, sizeof(float)));
    cudaAssert(cudaMemcpyToSymbol(d_maxRepelRange, &maxRepelRange, sizeof(float)));
    cudaAssert(cudaMemcpyToSymbol(d_repelForce, &repelForce, sizeof(float)));
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

    // from Matthias Müller, Ten Minute Physics, having the table size be twice 
    // the size of the particle count is a good rule of thumb most of the time
    cudaAssert(cudaMalloc(&d_spatialHashTable, (maxParticleCount * 2 + 1) * sizeof(int)));
    cudaAssert(cudaMemset(d_spatialHashTable, 0, (maxParticleCount * 2 + 1) * sizeof(int)));
    cudaAssert(cudaMalloc(&d_particleIndices, maxParticleCount * sizeof(int)));

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
    cudaAssert(cudaFree(d_spatialHashTable));
    cudaAssert(cudaFree(d_particleIndices));
}

void Particles::spawnParticles(unsigned int x, unsigned int y, BOOL shouldSpawn) {
    if (shouldSpawn) {
        mouseXPos = x;
        mouseYPos = y;
        spawn = TRUE;
    } else {
        spawn = FALSE;
    }
}

void Particles::succParticles(unsigned int xPos, unsigned int yPos, BOOL shouldSucc) {
    if (shouldSucc) {
        mouseXPos = xPos;
        mouseYPos = yPos;
        succ = TRUE;
    } else {
        succ = FALSE;
    }
}

void Particles::repelParticles(unsigned int xPos, unsigned int yPos, BOOL shouldRepel) {
    if (shouldRepel) {
        mouseXPos = xPos;
        mouseYPos = yPos;
        repel = TRUE;
    } else {
        repel = FALSE;
    }
}

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
    // we only create as many threads as needed to render the particles
    const size_t threadCount = min(renderingThreads, currIndex / chunkSize + 1);
    // const size_t threadCount = 1;

    vector<thread> threads;
    for (size_t t = 0; t < threadCount; t++) {
        size_t startIndex = t * chunkSize;
        size_t endIndex = (t == threadCount - 1) ? maxParticleCount : startIndex + chunkSize;

        threads.emplace_back(&Particles::updateVertices, this, startIndex, endIndex, deltaTime);
    }

    for (thread &thread : threads) {
        thread.join();
    }

    window.draw(vertices);
}

void Particles::swapDeviceParticles() {
    Vec2<float> *tempPos = d_positionIn;
    d_positionIn = d_positionOut;
    d_positionOut = tempPos;
    Vec2<float> *tempVel = d_velocityIn;
    d_velocityIn = d_velocityOut;
    d_velocityOut = tempVel;
}

__global__ void spawnParticleKernel(Vec2<float> *posIn, Vec2<float> *velIn, 
                                    Vec2<float> pos, unsigned int currIndex) {
    // give each spawned particles some offset to avoid overlapping, not perfect
    // but it's good enough tbh
    posIn[currIndex + threadIdx.x].x = pos.x + threadIdx.x;
    posIn[currIndex + threadIdx.x].y = pos.y + threadIdx.x;
    velIn[currIndex + threadIdx.x].x = 0;
    velIn[currIndex + threadIdx.x].y = 0;
}

// clamp the value between min and max
__device__ float clamp(float val, float min, float max) {
    return fmaxf(min, fminf(max, val));
}

// linear iterpolation
__device__ float lerp(const float n1, const float n2, const float time) {
	return n1 + time * (n2 - n1);
}

// Succ the particles to where the mouse is clicked
__global__ void succParticlesKernel(Vec2<float> *posIn, Vec2<float> *velIn, 
                                    const BOOL *d_isActive, const Vec2<float> mousePos) {
    int i = blockIdx.x;
    if (!d_isActive[i]) return;

    // don't succ the particle if it's too far away
    Vec2<float> delta = mousePos - posIn[i];
    if (delta.lengthSq() > powf(d_maxSuctionRange, 2)) return;

    Vec2<float> deltaNorm = delta.normalized();
    velIn[i] += deltaNorm * lerp(0.0f, d_suctionForce, delta.length() / d_maxSuctionRange);
}

// repel the particles from where the mouse is clicked
__global__ void repelParticlesKernel(Vec2<float> *posIn, Vec2<float> *velIn,
                                     const BOOL *d_isActive, const Vec2<float> mousePos) {
    int i = blockIdx.x;
    if (!d_isActive[i]) return;

    // don't repel the particle if it's too far away
    Vec2<float> delta = mousePos - posIn[i];
    if (delta.lengthSq() > powf(d_maxRepelRange, 2)) return;

    Vec2<float> deltaNorm = delta.normalized();
    velIn[i] -= deltaNorm * lerp(0.0f, d_repelForce, delta.length() / d_maxRepelRange);
}

// spatial hash function, using two large prime numbers to avoid collisions
__device__ int spatialHash(int cellX, int cellY) {
    int res = (cellX * 92837111) ^ (cellY * 689287499);
    return res % (d_maxParticleCount * 2);
}

__global__ void createSpatialHashTable(Vec2<float> *pos, int *cellIndices, int *particleIndices, float cellSize) {
    for (int i = threadIdx.x; i < d_maxParticleCount * 2 + 1; i += blockDim.x) {
        cellIndices[i] = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < d_maxParticleCount; i += blockDim.x) {
        int cellX = (int)((pos[i].x - d_borderLeft) / cellSize);
        int cellY = (int)((pos[i].y - d_borderTop) / cellSize);
        int hashIndex = spatialHash(cellX, cellY);
        atomicAdd(&cellIndices[hashIndex], 1);
    }
    __syncthreads();

    // prefix sum
    if (threadIdx.x == 0) {
        for (int i = 1; i < d_maxParticleCount * 2 + 1; i++) {
            cellIndices[i] += cellIndices[i - 1];
        }
    }
    __syncthreads();

    // Fill in the particle indices array
    for (int i = threadIdx.x; i < d_maxParticleCount; i += blockDim.x) {
        int cellX = (int)((pos[i].x - d_borderLeft) / cellSize);
        int cellY = (int)((pos[i].y - d_borderTop) / cellSize);
        int hashIndex = spatialHash(cellX, cellY);
        int index = atomicSub(&cellIndices[hashIndex], 1) - 1;
        particleIndices[index] = i;
    }
}

// The approach here is to have each block process one particle's collision
// against all other ones. Meaning n particles = n blocks. This also means each
// block will have n number of threads.
// There is a better way to approach this, which would require more complex
// index mapping but allow more efficient gpu utilization. 
__global__ void updateKernel(Vec2<float> *posIn, Vec2<float> *velIn,
                             const BOOL *d_isActive, const int *spatialHashtable,
                             const int *particleIndices,
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

    int base = blockIdx.x;
    int i = particleIndices[base];
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

            int hash = spatialHash(neighborX, neighborY);
            int startIndex = spatialHashtable[hash];
            for (int k = threadIdx.x + startIndex; 
                    k < d_maxParticleCount && k < spatialHashtable[hash + 1]; 
                    k += blockDim.x) {
                int j = particleIndices[k];
                if (i == j || !d_isActive[j]) continue;

                // impulse based collision
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
                    impulse = clamp(impulse, -5.0f, 5.0f);
                    velDelta -= deltaNorm * impulse * d_mass * d_restitution;
                }

                // position based dynamics
                // doesn't work too well compared to impulse based collision tho
                // Vec2<float> delta = posIn[j] - posIn[i];
                // float distSqr = delta.lengthSq();
                // if (delta.lengthSq() == 0
                //     || distSqr > powf(d_radius + d_radius, 2)) continue;
                // float dist = sqrt(distSqr);
                // float overlap = 0.5f * (d_radius + d_radius - dist) / dist;
                // Vec2<float> displacement = delta * overlap;
                // posDelta.x = posDelta.x - displacement.x;
                // posDelta.y = posDelta.y - displacement.y;
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

void Particles::update(float deltaTime, float gravity) {
    cudaStream_t stream;
    cudaAssert(cudaStreamCreate(&stream));

    dim3 blocks = maxParticleCount;
    dim3 threads = min(maxParticleCount, h_maxThreadCount);

    if (spawn && currIndex < maxParticleCount) {
        spawnParticleKernel<<<1, min(spawnCount, maxParticleCount - spawnCount), 0, stream>>>(d_positionIn, d_velocityIn, 
            Vec2<float>(static_cast<float>(mouseXPos), static_cast<float>(mouseYPos)), currIndex);
        fill(isActive.begin() + currIndex, isActive.begin() + currIndex + spawnCount, TRUE);
        currIndex += spawnCount;
    }
    if (succ) {
        succParticlesKernel<<<blocks, threads, 0, stream>>>(d_positionIn, d_velocityIn, d_isActive, 
            Vec2<float>(static_cast<float>(mouseXPos), static_cast<float>(mouseYPos)));
    }
    if (repel) {
        repelParticlesKernel<<<blocks, threads, 0, stream>>>(d_positionIn, d_velocityIn, d_isActive, 
            Vec2<float>(static_cast<float>(mouseXPos), static_cast<float>(mouseYPos)));
    }

    cudaAssert(cudaMemcpyAsync(d_isActive, isActive.data(), maxParticleCount * sizeof(BOOL), cudaMemcpyHostToDevice, stream));

    createSpatialHashTable<<<1, threads, 0, stream>>>(d_positionIn, d_spatialHashTable, d_particleIndices, cellSize);
    cudaAssert(cudaStreamSynchronize(stream));

    // this is the iterative solver loop for the position based dynamics, unused for now
    // for (int iter = 0; iter < 4; iter++) {
    updateKernel<<<blocks, threads, 0, stream>>>(d_positionIn, d_velocityIn, 
        d_isActive, d_spatialHashTable, d_particleIndices, deltaTime, gravity, cellSize, d_positionOut, d_velocityOut);
    cudaAssert(cudaStreamSynchronize(stream));
    // }
    cudaAssert(cudaMemcpyAsync(position.data(), d_positionOut, maxParticleCount * sizeof(Vec2<float>), cudaMemcpyDeviceToHost, stream));
    swapDeviceParticles();

    cudaAssert(cudaStreamDestroy(stream));
}
