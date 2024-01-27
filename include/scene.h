#ifndef SCENE_H
#define SCENE_H

#include "pch.h"
#include "particle.h"

#define DELTA_TIME 0.01f

class Scene;

class SceneManager {
    private:
        sf::RenderWindow window;
        std::unique_ptr<Scene> scene;
        sf::Clock clock;

    public:
        /**
         * @brief Construct a new Scene Manager object
         */
        SceneManager(string configFilename);

        /**
         * @brief Destroy the Scene Manager object
         */
        ~SceneManager();

        /**
         * @brief Run the scene manager
         */
        void run();
};

class Scene {
    private:
        std::unique_ptr<Particles> particles;
        unsigned int maxParticleCount;
        unsigned int borderLeft, borderRight, borderTop, borderBottom;
        float gravity;

    public:
        /**
         * @brief Construct a new Scene object
         */
        Scene(unsigned int maxParticleCount, unsigned int borderLeft, 
            unsigned int borderRight, unsigned int borderTop, 
            unsigned int borderBottom);

        /**
         * @brief Destroy the Scene object
         */
        ~Scene();

        /**
         * @brief Poll for events in the specified window
         * 
         * @param window The window to render to
         */
        void pollEvent(sf::RenderWindow& window);

        /**
         * @brief Proceed by one tick of the physics engine
         * 
         * @param window The window to update
         * @param deltaTime The time since the last tick
         */
        void update(sf::RenderWindow& window, float deltaTime);

        /**
         * @brief Proceed by one frame of the renderer, should be separate from 
         * the physics engine
         * 
         * @param window The window to render to
         * @param dt The time since the last frame
         */
        void render(sf::RenderWindow& window, float deltaTime);
};

#endif // SCENE_H 