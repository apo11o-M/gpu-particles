#ifndef SCENE_H
#define SCENE_H

#include "pch.h"
#include "particle.h"

#define DELTA_TIME 0.05

class Scene;

class SceneManager {
    private:
        sf::RenderWindow window;
        Scene *scene;

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
        Particles *particles;
        unsigned int maxParticleCount;
        int gravity;

    public:
        /**
         * @brief Construct a new Scene object
         */
        Scene();

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
         * @param dt The time since the last tick
         */
        void update(sf::RenderWindow& window, double dt);

        /**
         * @brief Proceed by one frame of the renderer, should be separate from 
         * the physics engine
         * 
         * @param window The window to render to
         */
        void render(sf::RenderWindow& window);
};

#endif // SCENE_H 