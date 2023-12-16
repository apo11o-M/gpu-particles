#include "pch.h"
#include "scene.h"

static void usage(string av0) {
    cerr << "Usage: %s [-c <config filename>]" << endl;
}

int main(int argc, char* argv[]) {
    cout << "Program starts.." << endl;

    string config = "default.json";
    for (int i = 0; i < argc; i++) {
        if (string(argv[i]) == "-c") {
            config = argv[i + 1];
        } else if (string(argv[i]) == "-h") {
            usage(argv[0]);
            return 0;
        }
        i++;
    }

    SceneManager sceneManager(config);
    sceneManager.run();

    cout << "..exiting program" << endl;
    return 0;
}