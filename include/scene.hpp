#ifndef SCENE_HPP
#define SCENE_HPP

#include "particle.hpp"
#include "fps.hpp"
#include "pch.hpp"

#define DELTA_TIME 0.01f

class Scene;

class SceneManager {
   private:
    sf::RenderWindow window;
    std::unique_ptr<Scene> scene;
    SimulationConfig config;
    sf::Clock clock;
    FPS fps;
    sf::Text fpsText;
    sf::Font font;

   public:
    /**
     * @brief Construct a new Scene Manager object
     */
    SceneManager(string config);

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
    unsigned int maxParticleCount;
    unsigned int borderLeft, borderRight, borderTop, borderBottom;
    float gravity;

   public:
    std::unique_ptr<Particles> particles;
    /**
     * @brief Construct a new Scene object
     */
    Scene(const SimulationConfig& config);

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

#endif  // SCENE_HPP