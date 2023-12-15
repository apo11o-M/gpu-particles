#include <SFML/Window.hpp>
#include <iostream>

using namespace std;

int main() {
    cout << "Program starts.." << endl;

    sf::Window window(sf::VideoMode(800, 600), "gpu-particles");

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }
    }

    cout << "..exiting program" << endl;
    return 0;
}