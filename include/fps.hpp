#ifndef FPS_HPP
#define FPS_HPP

#include "pch.hpp"

class FPS {
   private:
    sf::Clock clock;
    float lastUpdate;
    unsigned int currFrame, displayFrame;

   public:
    FPS() : clock(), lastUpdate(0.0f), currFrame(0), displayFrame(0) {}

    void update() {
        if (clock.getElapsedTime().asSeconds() - lastUpdate >= 1.0f) {
            displayFrame = currFrame;
            currFrame = 0;
            clock.restart();
        }
        currFrame++;
    }

    const unsigned int getFPS() const { return displayFrame; }
};

#endif  // FPS_HPP
