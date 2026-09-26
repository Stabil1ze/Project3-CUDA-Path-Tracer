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

/**
 * Environment light (infinite area light): the radiance a ray sees when it
 * leaves the scene, as a smooth three colour sky. All zero by default, i.e. the
 * black background the renderer shipped with.
 */
struct Environment
{
    glm::vec3 zenith;
    glm::vec3 horizon;
    glm::vec3 ground;
    float intensity;
};

/**
 * Distant light (a sun): a disc at infinity. `direction` is the direction the
 * light travels in (PBRT's convention), so towards the light is its negation.
 * The disc covers a few thousandths of a steradian - a path will almost never
 * walk into it, while aiming at it costs one shadow ray, which is what the light
 * strategy is for. `enabled` is 0 for scenes without one.
 */
struct DistantLight
{
    glm::vec3 direction;
    glm::vec3 radiance;
    float cosMaxAngle;   // cosine of the angular radius of the disc
    float solidAngle;    // 2 pi (1 - cosMaxAngle), the density's denominator
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
    // Optional Environment / DistantLight blocks; both inert when absent.
    Environment environment;
    DistantLight distantLight;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    // Solid angle density of the BSDF sample that produced this segment, or 0 if
    // it came from a delta lobe (mirror, dielectric) whose density is a Dirac.
    // That is the weight MIS needs when the segment lands on an emitter; with
    // light sampling off nothing reads it.
    float lastPdf;
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
