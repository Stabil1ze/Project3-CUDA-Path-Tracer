#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        const auto& col = p["RGB"];
        newMaterial.color = glm::vec3(col[0], col[1], col[2]);

        // "TYPE" decides which BSDF lobe(s) the material has. The weight fields
        // below are what scatterRay uses to probabilistically pick a lobe, so a
        // material may combine several of them later on (e.g. glossy = diffuse
        // + imperfect specular) without touching the tracing code.
        if (p["TYPE"] == "Diffuse")
        {
            // albedo only: the random walk is chosen by scatterRay.
        }
        else if (p["TYPE"] == "Emitting")
        {
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            // Perfect mirror by default: ROUGHNESS 0 means the reflected
            // direction is used as-is, a larger value is the hook for the
            // "imperfect specular" extension (GPU Gems 3, Ch. 20).
            newMaterial.hasReflective = 1.0f;
            newMaterial.specular.color = newMaterial.color;
            newMaterial.specular.exponent = p.value("ROUGHNESS", 0.0f);
        }

        // Optional procedural texture for the diffuse albedo. It is evaluated on
        // the object space hit point (see interactions.cu), and TEXSCALE sets how
        // many pattern cells fit into one object space unit.
        const std::string texture = p.value("TEXTURE", std::string("none"));
        if (texture == "checker")
        {
            newMaterial.textureType = 1;
        }
        else if (texture == "marble")
        {
            newMaterial.textureType = 2;
        }
        else if (texture != "none")
        {
            cout << "Unknown TEXTURE '" << texture << "' on material " << name
                 << ", ignoring it" << endl;
        }
        newMaterial.textureScale = p.value("TEXSCALE", 1.0f);

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "Mandelbulb")
        {
            newGeom.type = MANDELBULB;
        }
        else if (type == "Menger")
        {
            newGeom.type = MENGER;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else
        {
            cout << "Unknown object TYPE '" << type << "', treating it as a sphere" << endl;
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    // Depth of field (optional). "APERTURE" is the lens radius in world units
    // and "FOCUS" the distance of the focal plane; without them the camera is a
    // pinhole. A missing FOCUS is taken as the distance to the look-at point,
    // which is the usual "the thing I am looking at is sharp" behaviour.
    camera.aperture = cameraData.value("APERTURE", 0.0f);
    camera.focalDistance = cameraData.value("FOCUS", glm::length(camera.lookAt - camera.position));

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    // The basis has to be derived from the view direction, so that one must be
    // known first - "up" only fixes the roll of the camera frame. Building
    // `right` out of a not-yet-computed `view` yields a cross product of two
    // zero vectors, i.e. NaNs in every ray direction.
    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
