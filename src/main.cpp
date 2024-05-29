#include "pch.hpp"
#include "scene.hpp"

static void usage(string av0) {
    cerr << "Usage: gpu-particles [-c <config filename>]" << endl;
}

int main(int argc, char* argv[]) {
    cout << "Program starts.." << endl;

    // string config = "default-config.json";
    string config = "smol-container.json";
    try {
        for (int i = 0; i < argc; i++) {
            if (string(argv[i]) == "-c") {
                config = argv[i + 1];
            } else if (string(argv[i]) == "-h") {
                usage(argv[0]);
                return 0;
            }
        }
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        usage(argv[0]);
        return 1;
    }

    SceneManager sceneManager(config);
    sceneManager.run();

    cout << "..exiting program" << endl;
    return 0;
}
