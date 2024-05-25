// I also noticed an interesting thing, where when we update the particles, there is a problem when lots of particles get condensed/pooled at the bottom of the playground. when this happens, since we update the particles based on their index from small to big rather than their positions, the entire system tends to be unstable.

// What im thinking is instead updating the particles based on their indices, we update the particles based on the ones that are closer to the 1) bottom wall and 2) the side walls. that way we can simulate "wall pressure" and push the particles "upward" towards the open space. what do you think of this idea? do you have any suggestions or other better/refined methods?

#include <iostream>
#include <vector>
#include <cmath>

const float CELL_SIZE = 2.0f;
const int GRID_WIDTH = 10;
const int GRID_HEIGHT = 10;
const float RADIUS = 1.0f;

struct Vec2 {
    float x, y;

    Vec2() : x(0), y(0) {}
    Vec2(float x, float y) : x(x), y(y) {}

    Vec2 operator-(const Vec2& other) const {
        return Vec2(x - other.x, y - other.y);
    }

    float lengthSq() const {
        return x * x + y * y;
    }
};

int getCellIndex(float x, float y, float cellSize, int gridWidth) {
    int cellX = x / cellSize;
    int cellY = y / cellSize;
    return cellY * gridWidth + cellX;
}

int main() {
    // Define playground boundaries and particles
    std::vector<Vec2> particles = {
        Vec2(1.0f, 1.0f), Vec2(2.0f, 2.0f), Vec2(3.0f, 1.5f),
        Vec2(8.0f, 8.0f), Vec2(7.5f, 7.0f), Vec2(9.0f, 9.0f)
    };
    int numParticles = particles.size();

    // Step 1: Assign particles to cells
    std::vector<int> cellIndices(numParticles);
    for (int i = 0; i < numParticles; i++) {
        cellIndices[i] = getCellIndex(particles[i].x, particles[i].y, CELL_SIZE, GRID_WIDTH);
    }

    for (int i = 0; i < numParticles; i++) {
        std::cout << "Particle " << i << " is in cell " << cellIndices[i] << std::endl;
    }

    // Step 2: Check for collisions within cells and neighboring cells
    for (int i = 0; i < numParticles; ++i) {
        int cellX = particles[i].x / CELL_SIZE;
        int cellY = particles[i].y / CELL_SIZE;

        for (int offsetY = -1; offsetY <= 1; ++offsetY) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                int neighborCellX = cellX + offsetX;
                int neighborCellY = cellY + offsetY;

                if (neighborCellX >= 0 && neighborCellX < GRID_WIDTH && neighborCellY >= 0 && neighborCellY < GRID_HEIGHT) {
                    int neighborCellIndex = neighborCellY * GRID_WIDTH + neighborCellX;

                    for (int j = 0; j < numParticles; ++j) {
                        if (i != j && cellIndices[j] == neighborCellIndex) {
                            Vec2 delta = particles[i] - particles[j];
                            if (delta.lengthSq() < powf(2 * RADIUS, 2)) {
                                // Handle collision
                                std::cout << "Collision detected between particles " << i << " and " << j << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}
