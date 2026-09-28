#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    TRIANGLE,
    MANDELBULB,
    MENGER
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    // TRIANGLE only variables
    glm::vec3 v0, v1, v2;
    glm::vec3 n0, n1, n2;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    // Procedural texture that modulates the diffuse albedo
    // evaluated on the object space* position of the hit 
    int textureType; // 0 = none, 1 = checker, 2 = marble
    float textureScale; // The scale multiplies that position
};

// Environment light (hemispherical light)
struct Environment
{
	glm::vec3 zenith; // The color of the sky at the zenith
	glm::vec3 horizon; // The color of the sky at the horizon
	glm::vec3 ground; // The color of the ground
	float intensity; // The intensity of the environment light
};

// Distant light (directional light)
struct DistantLight
{
    glm::vec3 direction;
    glm::vec3 radiance;
    float cosMaxAngle;   
    float solidAngle;    
    int enabled;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    // Thin lens model
    float aperture;
    float focalDistance;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;

    // Restartable rendering
    float checkpointInterval;

    // Write-out frames
    float frameInterval;

    // Optional Environment / DistantLight blocks
    Environment environment;
    DistantLight distantLight;

    // Denoiser guides
    std::vector<glm::vec3> normalImage;
    std::vector<glm::vec3> albedoImage;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    float lastPdf;
};

// Use with a corresponding PathSegment to do:
// color contribution computation
// BSDF evaluation
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;

  // Index into the geometry array
  int geomId;

  // Whether the intersection is outside or inside the geometry
  int outside; // 1 when outside, 0 when inside
};
