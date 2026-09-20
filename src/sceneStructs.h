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
    // Procedural shapes, evaluated as signed distance fields and intersected by
    // sphere tracing (see intersections.cu). Both live in a unit-ish object
    // space and are transformed by the geometry's matrix like any other object.
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
    // Procedural texture that modulates the diffuse albedo, evaluated on the
    // *object space* position of the hit (so the pattern sticks to the object
    // and follows its transform): 0 = none, 1 = checker, 2 = marble. The scale
    // multiplies that position, i.e. it sets how many pattern cells fit into
    // one unit of object space.
    int textureType;
    float textureScale;
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
    // Thin lens model (optional scene fields "APERTURE" and "FOCUS"): the
    // aperture is the lens radius and the focus the distance of the focal
    // plane. aperture == 0 keeps the camera a pinhole.
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
    // Restartable rendering: seconds between checkpoints while a render is in
    // progress (0 disables it). A checkpoint that existed when the render
    // started was already consumed by the resume in runCuda().
    float checkpointInterval;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    // 1 when an emitter hit by this path segment must be added to the image.
    // Direct light sampling delivers the light for a diffuse vertex, so the path
    // hit that follows it must not be counted a second time; a delta BSDF cannot
    // be sampled towards a light, so its emitters are still counted by the path.
    int countsEmission;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  // Index into the geometry array. The shading kernel needs the geometry, not
  // just the material, to map a hit point back into object space for the
  // procedural textures.
  int geomId;
  // 1 when the ray came from outside the primitive, 0 when it was already
  // inside (the intersection tests report this). Refraction needs it to decide
  // whether the path is entering or leaving the medium.
  int outside;
};
