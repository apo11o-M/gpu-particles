#include "scene.h"

SceneManager::SceneManager(string configFilename) : scene() {
    window.create(sf::VideoMode(800, 600), "gpu-particles: " + configFilename);
    window.setFramerateLimit(144);
    window.setVerticalSyncEnabled(true);
}

SceneManager::~SceneManager() {
    // do we need to delete the particles in the scene object?
    delete scene;
}

void SceneManager::run() {
    double t = 0.0;
    double prevTime = sf::Clock().getElapsedTime().asSeconds();
    double accumulator = 0.0;

    while (window.isOpen()) {
        // window event update
        scene->pollEvent(window);

        // we only update the physics every DELTA_TIME seconds, meaning every 
        // second it would have 20 physics updates
        double currTime = sf::Clock().getElapsedTime().asSeconds();
        double frameTime = currTime - prevTime;
        prevTime = currTime;
        accumulator += frameTime;

        while (accumulator >= DELTA_TIME) {
            // update the physics engine
            scene->update(window, DELTA_TIME);
            accumulator -= DELTA_TIME;
            t += DELTA_TIME;
        }

        // TODO: might also need to work on interpolation here, depending 
        // on how the render result looks like

        // render onto the screen every frame
        scene->render(window);
    }
}

Scene::Scene() {
    this->particles = new Particles(5);
    this->maxParticleCount = 5;
    this->gravity = 0;
}

Scene::~Scene() {
    delete this->particles;
}

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
    // 2. perform collision check
}

void Scene::render(sf::RenderWindow& window) {
    window.clear(sf::Color::Black);


    window.display();
}