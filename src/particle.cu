#include "particle.hpp"
// #include "stdio.h"

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

Particles::Particles(unsigned int maxParticles, unsigned int borderLeft,
                     unsigned int borderRight, unsigned int borderTop,
                     unsigned int borderBottom)
    : r(maxParticles, 66),
      g(maxParticles, 205),
      b(maxParticles, 227),
      position(maxParticles, Vec2<float>(0, 0)),
      velocity(maxParticles, Vec2<float>(0, 0)),
      mass(maxParticles, 20),
      radius(maxParticles, 3),
      isActive(maxParticles, false),
      shapes(maxParticles) {
    this->maxParticleCount = maxParticles;
    currIndex = 0;

    restitution = 0.6f;
    dampingFactor = 0.95f;
    dampingFactorRate = 60.0f;

    this->borderLeft = borderLeft;
    this->borderRight = borderRight;
    this->borderTop = borderTop;
    this->borderBottom = borderBottom;

    this->cellXCount = (borderRight - borderLeft) / CELL_SIZE;
    this->cellYCount = (borderBottom - borderTop) / CELL_SIZE;

    cout << "borders: " << borderLeft << ", " << borderRight << ", " << borderTop << ", " << borderBottom << endl;
    cout << "grid: " << cellXCount << ", " << cellYCount << endl;

    for (size_t i = 0; i < maxParticles; i++) {
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
    cudaAssert(cudaMemcpyToSymbol(d_radius, radius.data(), sizeof(float)));
    cudaAssert(cudaMemcpyToSymbol(d_mass, mass.data(), sizeof(float)));
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
    cudaAssert(cudaMalloc(&d_positionIn, maxParticles * sizeof(Vec2<float>)));
    cudaAssert(cudaMalloc(&d_velocityIn, maxParticles * sizeof(Vec2<float>)));
    cudaAssert(cudaMalloc(&d_positionOut, maxParticles * sizeof(Vec2<float>)));
    cudaAssert(cudaMalloc(&d_velocityOut, maxParticles * sizeof(Vec2<float>)));
    cudaAssert(cudaMemcpy(d_positionIn, position.data(), maxParticles * sizeof(Vec2<float>), cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpy(d_velocityIn, velocity.data(), maxParticles * sizeof(Vec2<float>), cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpy(d_positionOut, position.data(), maxParticles * sizeof(Vec2<float>), cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpy(d_velocityOut, velocity.data(), maxParticles * sizeof(Vec2<float>), cudaMemcpyHostToDevice));

    cudaAssert(cudaMalloc(&d_isActive, maxParticles * sizeof(BOOL)));
    cudaAssert(cudaMemcpy(d_isActive, isActive.data(), maxParticles * sizeof(BOOL), cudaMemcpyHostToDevice));

    cudaAssert(cudaMalloc(&d_cellIndices, maxParticles * sizeof(int)));

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

void Particles::render(sf::RenderWindow &window, float deltaTime) {
    // using sf::Triangles to draw the particles, which gives us significant 
    // performance boost as the window.draw() function is expensive
    sf::VertexArray vertices(sf::Triangles, maxParticleCount * 6);
    for (size_t i = 0; i < maxParticleCount; i++) {
        if (!isActive[i]) continue;

        // interpolating the position to achieve smoother movement
        float x = position[i].x - radius[i] + velocity[i].x * deltaTime;
        float y = position[i].y - radius[i] + velocity[i].y * deltaTime;

        vertices[i * 6 + 0].position = sf::Vector2f(x - radius[i], y - radius[i]);
        vertices[i * 6 + 1].position = sf::Vector2f(x + radius[i], y - radius[i]);
        vertices[i * 6 + 2].position = sf::Vector2f(x + radius[i], y + radius[i]);

        vertices[i * 6 + 3].position = sf::Vector2f(x - radius[i], y - radius[i]);
        vertices[i * 6 + 4].position = sf::Vector2f(x + radius[i], y + radius[i]);
        vertices[i * 6 + 5].position = sf::Vector2f(x - radius[i], y + radius[i]);

        for (size_t j = 0; j < 6; j++) {
            vertices[i * 6 + j].color = sf::Color(r[i], g[i], b[i]);
        }
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

__device__ int getCellIndex(float xPos, float yPos, float cellSize, int gridXCount) {
    int x = (int)((xPos - d_borderLeft) / CELL_SIZE);
    int y = (int)((yPos - d_borderTop) / CELL_SIZE);
    return y * gridXCount + x;
}

__global__ void assignParticlesToCells(Vec2<float> *pos, int *cellIndices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_maxParticleCount) return;

    cellIndices[i] = getCellIndex(pos[i].x, pos[i].y, CELL_SIZE, d_cellXCount);
    // printf("i: %d, cellIndices[i]: %d\n", i, cellIndices[i]);
}

__global__ void spawnParticleKernel(Vec2<float> *posIn, Vec2<float> *velIn, 
                                    Vec2<float> pos, unsigned int currIndex) {
    if (threadIdx.x == 0) {
        posIn[currIndex].x = pos.x;
        posIn[currIndex].y = pos.y;
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

    int i = blockIdx.x;     // particle 1 index
    if (!d_isActive[i]) return;
    int cellXPos = (int)((posIn[i].x - d_borderLeft) / CELL_SIZE);
    int cellYPos = (int)((posIn[i].y - d_borderTop) / CELL_SIZE);

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

void Particles::update(float deltaTime, float gravity) {
    cudaStream_t stream;
    cudaAssert(cudaStreamCreate(&stream));

    dim3 blocks = maxParticleCount;
    dim3 threads = min(maxParticleCount, h_maxThreadCount);

    if (spawn && currIndex < maxParticleCount) {
        spawnParticleKernel<<<1, 1, 0, stream>>>(d_positionIn, d_velocityIn, 
            Vec2<float>(static_cast<float>(mouseXPos), static_cast<float>(mouseYPos)), currIndex);
        isActive[currIndex] = TRUE;
        currIndex++;
        spawn = FALSE;
    }
    cudaAssert(cudaMemcpyAsync(d_isActive, isActive.data(), maxParticleCount * sizeof(BOOL), cudaMemcpyHostToDevice, stream));
    
    assignParticlesToCells<<<blocks, threads, 0, stream>>>(d_positionIn, d_cellIndices);
    updateKernel<<<blocks, threads, 0, stream>>>(d_positionIn, d_velocityIn, 
        d_isActive, d_cellIndices, deltaTime, gravity, d_positionOut, d_velocityOut);
    cudaAssert(cudaStreamSynchronize(stream));

    cudaAssert(cudaMemcpyAsync(position.data(), d_positionIn, maxParticleCount * sizeof(Vec2<float>), cudaMemcpyDeviceToHost, stream));
    swapDeviceParticles();

    cudaAssert(cudaStreamDestroy(stream));
}
