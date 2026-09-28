#include "scene.h"

#include "mesh.h"
#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

namespace
{
    std::string resolveAssetPath(const std::string& sceneDir, const std::string& file)
    {
        const std::filesystem::path path(file);
        if (path.is_absolute() || sceneDir.empty())
        {
            return path.string();
        }
        return (std::filesystem::path(sceneDir) / path).lexically_normal().string();
    }
}

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

    const std::string sceneDir = std::filesystem::path(jsonName).parent_path().string();

    // Environment light handler
    state.environment.zenith = glm::vec3(0.0f);
    state.environment.horizon = glm::vec3(0.0f);
    state.environment.ground = glm::vec3(0.0f);
    state.environment.intensity = 1.0f;
    if (data.contains("Environment"))
    {
        const auto& env = data["Environment"];
        auto readColor = [](const json& block, const char* key, glm::vec3 fallback)
        {
            if (!block.contains(key))
            {
                return fallback;
            }
            const auto& c = block[key];
            return glm::vec3(c[0], c[1], c[2]);
        };
        state.environment.zenith = readColor(env, "ZENITH", glm::vec3(0.0f));
        state.environment.horizon = readColor(env, "HORIZON", state.environment.zenith);
        state.environment.ground = readColor(env, "GROUND", glm::vec3(0.0f));
        state.environment.intensity = env.value("INTENSITY", 1.0f);
        cout << "[env] dome: zenith (" << state.environment.zenith.x << " "
            << state.environment.zenith.y << " " << state.environment.zenith.z
            << "), horizon (" << state.environment.horizon.x << " "
            << state.environment.horizon.y << " " << state.environment.horizon.z
            << "), ground (" << state.environment.ground.x << " "
            << state.environment.ground.y << " " << state.environment.ground.z
            << "), intensity " << state.environment.intensity << endl;
    }

    // Distant light handler
    state.distantLight.enabled = 0;
    state.distantLight.direction = glm::vec3(0.0f, -1.0f, 0.0f);
    state.distantLight.radiance = glm::vec3(0.0f);
    state.distantLight.cosMaxAngle = 1.0f;
    state.distantLight.solidAngle = 1.0f;
    if (data.contains("DistantLight"))
    {
        const auto& sun = data["DistantLight"];
        const auto& dir = sun["DIRECTION"];
        const auto& col = sun["RGB"];
        const float radiusDegrees = sun.value("ANGULAR_RADIUS", 0.5f);
        const float intensity = sun.value("INTENSITY", 1.0f);
        state.distantLight.direction = glm::normalize(glm::vec3(dir[0], dir[1], dir[2]));
        state.distantLight.radiance = glm::vec3(col[0], col[1], col[2]) * intensity;
        const float cosMax = cosf(glm::radians(glm::clamp(radiusDegrees, 0.01f, 89.0f)));
        state.distantLight.cosMaxAngle = cosMax;
        state.distantLight.solidAngle = TWO_PI * (1.0f - cosMax);
        state.distantLight.enabled = 1;
        cout << "[sun] travelling in (" << state.distantLight.direction.x << " "
            << state.distantLight.direction.y << " " << state.distantLight.direction.z
            << "), radiance (" << state.distantLight.radiance.x << " "
            << state.distantLight.radiance.y << " " << state.distantLight.radiance.z
            << "), angular radius " << radiusDegrees << " degrees, solid angle "
            << state.distantLight.solidAngle << endl;
    }

    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        const auto& col = p["RGB"];
        newMaterial.color = glm::vec3(col[0], col[1], col[2]);

        // TYPE decides which BSDF lobe(s) the material has
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
            // Perfect mirror by default
            newMaterial.hasReflective = 1.0f;
            newMaterial.specular.color = newMaterial.color;
            newMaterial.specular.exponent = p.value("ROUGHNESS", 0.0f);
        }
        else if (p["TYPE"] == "Refractive")
        {
            // Dielectric (glass/water)
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p.value("IOR", 1.5f);
            newMaterial.specular.color = newMaterial.color;
        }

        // Optional procedural texture for the diffuse albedo
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
        Geom newGeom{};
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
        else if (type == "mesh")
        {
            const std::string file = p.value("FILE", std::string());
            if (file.empty())
            {
                cout << "Object TYPE 'mesh' without a FILE field, skipping it" << endl;
                continue;
            }
            const auto& meshTrans = p["TRANS"];
            const auto& meshRotat = p["ROTAT"];
            const auto& meshScale = p["SCALE"];
            const int before = (int)geoms.size();
            const int loaded = loadMesh(resolveAssetPath(sceneDir, file),
                glm::vec3(meshTrans[0], meshTrans[1], meshTrans[2]),
                glm::vec3(meshRotat[0], meshRotat[1], meshRotat[2]),
                glm::vec3(meshScale[0], meshScale[1], meshScale[2]),
                (int)MatNameToID[p["MATERIAL"]], geoms);
            if (loaded < 0)
            {
                cout << "Object TYPE 'mesh' could not load " << file << ", skipping it" << endl;
            }
            else
            {
                cout << "  mesh " << p["MATERIAL"].get<std::string>() << " added "
                     << (int)geoms.size() - before << " triangle geometries" << endl;
            }
            continue;
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

	// Restartable rendering interval, in seconds
    state.checkpointInterval = cameraData.value("CHECKPOINT", 30.0f);
    // Write-out frames for animations in the write-up, in seconds
    state.frameInterval = cameraData.value("FRAMES", 0.0f);

    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    // Depth of field
    camera.aperture = cameraData.value("APERTURE", 0.0f);
    camera.focalDistance = cameraData.value("FOCUS", glm::length(camera.lookAt - camera.position));

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    //set up render camera
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
