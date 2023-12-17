#include "scene.h"

SceneManager::SceneManager(string configFilename) : clock() {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;
    
    window.create(sf::VideoMode(800, 600), "gpu-particles: " + configFilename, 
        sf::Style::Default, settings);
    window.setFramerateLimit(144);
    window.setVerticalSyncEnabled(true);

    // TODO: parse the config file and create the scene
    scene = std::make_unique<Scene>(1, 0, 800, 0, 600);
}

SceneManager::~SceneManager() { }

void SceneManager::run() {
    double t = 0.0;
    double accumulator = 0.0;

    while (window.isOpen()) {
        // window event update
        scene->pollEvent(window);

        // we only update the physics every DELTA_TIME seconds, meaning every 
        // second it would have 20 physics updates
        double frameTime = clock.getElapsedTime().asSeconds();
        clock.restart();
        accumulator += frameTime;
        while (accumulator >= DELTA_TIME) {
            // update the physics engine
            scene->update(window, DELTA_TIME);
            accumulator -= DELTA_TIME;
            t += DELTA_TIME;
        }

        // TODO: need to work on interpolating the particles position since 
        // right now it looks like it's stuttering

        // render onto the screen every frame
        scene->render(window);
    }
}

Scene::Scene(unsigned int maxParticleCount, 
        unsigned int borderLeft, unsigned int borderRight, 
        unsigned int borderTop, unsigned int borderBottom) {
    particles = std::make_unique<Particles>(maxParticleCount);
    this->maxParticleCount = maxParticleCount;
    gravity = 0;

    this->borderLeft = borderLeft;
    this->borderRight = borderRight;
    this->borderTop = borderTop;
    this->borderBottom = borderBottom;
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
                    particles->makeActive(1);
                } else if (event.mouseButton.button == sf::Mouse::Right) {
                    
                }
                break;
            default:
                break;
        }
    }
}

void Scene::update(sf::RenderWindow& window, double dt) {
    // 1. update the position, velocity, and accel of the individual particles
    particles->update(dt);

    // 2. perform collision check
}

void Scene::render(sf::RenderWindow& window) {
    window.clear(sf::Color::Black);

    particles->render(window);

    window.display();
}