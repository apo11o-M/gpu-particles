#include "scene.h"

SceneManager::SceneManager(string configFilename) : clock() {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;
    
    window.create(sf::VideoMode(800, 600), "gpu-particles: " + configFilename, 
        sf::Style::Default, settings);
    window.setFramerateLimit(144);
    window.setVerticalSyncEnabled(true);

    // TODO: parse the config file and create the scene
    scene = std::make_unique<Scene>(100, 50, 500, 100, 300);
}

SceneManager::~SceneManager() { }

void SceneManager::run() {
    float t = 0.0;
    float accumulator = 0.0;

    while (window.isOpen()) {
        // window event update
        scene->pollEvent(window);

        // we only update the physics every DELTA_TIME seconds, meaning every 
        // second it would have 20 physics updates
        float frameTime = clock.getElapsedTime().asSeconds();
        clock.restart();
        accumulator += frameTime;
        while (accumulator >= DELTA_TIME) {
            // update the physics engine
            scene->update(window, DELTA_TIME);
            accumulator -= DELTA_TIME;
            t += DELTA_TIME;
        }

        // render onto the screen every frame. Here we also interpolates the 
        // particle position
        scene->render(window, accumulator);
    }
}

Scene::Scene(unsigned int maxParticleCount, 
        unsigned int borderLeft, unsigned int borderRight, 
        unsigned int borderTop, unsigned int borderBottom) {
    this->maxParticleCount = maxParticleCount;
    this->borderLeft = borderLeft;
    this->borderRight = borderRight;
    this->borderTop = borderTop;
    this->borderBottom = borderBottom;
    gravity = 10000;

    particles = std::make_unique<Particles>(maxParticleCount, borderLeft, 
        borderRight, borderTop, borderBottom);
}

Scene::~Scene() { }

void Scene::pollEvent(sf::RenderWindow& window) {
    sf::Event event;
    while (window.pollEvent(event)) {
        switch (event.type) {
            case (sf::Event::Closed):
                window.close();
                break;
            case (sf::Event::KeyPressed):
                if (event.key.code == sf::Keyboard::Escape) {
                    window.close();
                }
                break;
            case (sf::Event::MouseButtonPressed):

                if (event.mouseButton.button == sf::Mouse::Left) {
                    unsigned int x = std::clamp((unsigned int)event.mouseButton.x, 
                        borderLeft, borderRight);
                    unsigned int y = std::clamp((unsigned int)event.mouseButton.y, 
                        borderTop, borderBottom);
                    particles->makeActive(1, x, y, 1);
                } else if (event.mouseButton.button == sf::Mouse::Right) {
                    unsigned int x = std::clamp((unsigned int)event.mouseButton.x, 
                        borderLeft, borderRight);
                    unsigned int y = std::clamp((unsigned int)event.mouseButton.y, 
                        borderTop, borderBottom);
                    particles->makeActive(1, x, y, -1);
                }
                break;
            default:
                break;
        }
    }
}

void Scene::update(sf::RenderWindow& window, float deltaTime) {
    // 1. perform collision check
    // 2. perform collision response
    // 3. update the position, velocity, and accel of the individual particles
    particles->update(deltaTime, gravity);
}

void Scene::render(sf::RenderWindow& window, float deltaTime) {
    window.clear(sf::Color::Black);

    particles->render(window, deltaTime);

    window.display();
}