#pragma once

#include "sceneStructs.h"
#include <string>
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    // Material names in id order, so a UI can label them
    std::vector<std::string> materialNames;
    RenderState state;
};
