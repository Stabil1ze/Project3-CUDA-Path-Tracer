#pragma once

// Light sampling: the area lights, the environment (dome), the distant light
// (sun) and the MIS weights the strategies share. Everything here is a pure
// function of its arguments, so it lives in a header rather than next to the
// kernels that call it.

#include <cmath>

#include <glm/glm.hpp>

#include "ggx.h"          // buildTangentFrame
#include "sceneStructs.h" // Environment, DistantLight, DeviceLight's types
#include "utilities.h"    // TWO_PI

struct DeviceLight
{
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    glm::vec3 emission;      // material colour * emittance
    float surfaceArea;
    int geomIndex;           
    int shape;               // 0 = box, 1 = sphere
    float radius;            // sphere only
    glm::vec3 center;        // sphere only
};

enum LightShape
{
    LIGHT_BOX = 0,
    LIGHT_SPHERE = 1
};

enum LightSampleOutcome
{
    LIGHT_ACCEPTED = 0,
    LIGHT_REJECT_COS_SURFACE,   // sample is below the shading point's horizon
    LIGHT_REJECT_COS_LIGHT,     // shading point is behind the sampled light face
    LIGHT_REJECT_OCCLUDED       // something else is between the two
};

struct LightLedger
{
    unsigned long long samples;
    unsigned long long accepted;
    unsigned long long rejectCosSurface;
    unsigned long long rejectCosLight;
    unsigned long long occluded;

    double acceptedEnergy;
    double acceptedEnergySq;

    unsigned long long nonFiniteEnergy;

    unsigned long long bsdfHits;
    double bsdfEnergy;
    double bsdfEnergyPlain;
    double bsdfEnergySq;
};

static const int LIGHT_SAMPLE_FACES = 6;

/** Record one light sample outcome in a ledger row. */
__device__ inline void ledgerRecord(LightLedger& row, LightSampleOutcome outcome, double energy)
{
    atomicAdd(&row.samples, 1ull);
    switch (outcome)
    {
        case LIGHT_ACCEPTED:
            atomicAdd(&row.accepted, 1ull);
            if (isfinite(energy))
            {
                atomicAdd(&row.acceptedEnergy, energy);
                atomicAdd(&row.acceptedEnergySq, energy * energy);
            }
            else
            {
                atomicAdd(&row.nonFiniteEnergy, 1ull);
            }
            break;
        case LIGHT_REJECT_COS_SURFACE:
            atomicAdd(&row.rejectCosSurface, 1ull);
            break;
        case LIGHT_REJECT_COS_LIGHT:
            atomicAdd(&row.rejectCosLight, 1ull);
            break;
        default:
            atomicAdd(&row.occluded, 1ull);
            break;
    }
}

/** Record an emitter hit reached by a random walk, with its MIS weight. A
 *  `weight` of 1 (delta sample, or light sampling off) is the reference side. */
__device__ inline void ledgerRecordBsdfHit(LightLedger& row, double energy,
    double weight)
{
    atomicAdd(&row.bsdfHits, 1ull);
    atomicAdd(&row.bsdfEnergy, energy * weight);
    atomicAdd(&row.bsdfEnergyPlain, energy);
    atomicAdd(&row.bsdfEnergySq, energy * energy);
}

/** A direction uniform in solid angle over the cone of half angle `cosMax`. */
__host__ __device__ inline glm::vec3 sampleCone(glm::vec3 axis, float cosMax, float u1, float u2)
{
    const float cosTheta = glm::mix(cosMax, 1.0f, u1);
    const float sinTheta = sqrtf(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
    glm::vec3 t1, t2;
    buildTangentFrame(axis, t1, t2);
    const float phi = TWO_PI * u2;
    return glm::normalize(t1 * (sinTheta * cosf(phi)) + t2 * (sinTheta * sinf(phi))
        + axis * cosTheta);
}

/** Solid angle density of the light strategy for a direction that reaches
 *  `light`, including the chance of having picked it. One definition shared by
 *  the estimator and the MIS weights, which must not drift apart. */
__host__ __device__ inline float lightSampleSolidAnglePdf(const DeviceLight& light, int lightCount,
    glm::vec3 shadingPoint, glm::vec3 lightPoint, glm::vec3 lightNormal)
{
    if (light.shape == LIGHT_SPHERE)
    {
        // Uniform over the tangent cone: constant density on the visible cap, so
        // no 1/cos singularity and no wasted samples on the far side.
        const glm::vec3 toCenter = light.center - shadingPoint;
        const float distance2 = glm::dot(toCenter, toCenter);
        const float radius2 = light.radius * light.radius;
        if (distance2 <= radius2)
        {
            return 0.0f;                 // inside the emitter: no visible cap
        }
        const float cosMax = sqrtf(glm::max(0.0f, 1.0f - radius2 / distance2));
        const float solidAngle = TWO_PI * (1.0f - cosMax);
        return 1.0f / glm::max((float)lightCount * solidAngle, 1e-12f);
    }

    // Box: uniform by area, converted to the solid angle the BSDF lives in.
    const glm::vec3 toLight = lightPoint - shadingPoint;
    const float distance2 = glm::dot(toLight, toLight);
    const float cosLight = glm::abs(glm::dot(lightNormal, glm::normalize(-toLight)));
    return 1.0f / glm::max((float)lightCount * light.surfaceArea, 1e-6f)
        * distance2 / glm::max(cosLight, 1e-4f);
}

/** Same, for a hit that has to be attributed to one of the lights by geometry. */
__host__ __device__ inline float lightSamplePdfForGeom(const DeviceLight* lights, int lightCount,
    glm::vec3 shadingPoint, glm::vec3 lightPoint, glm::vec3 lightNormal, int geomId)
{
    for (int i = 0; i < lightCount; i++)
    {
        if (lights[i].geomIndex == geomId)
        {
            return lightSampleSolidAnglePdf(lights[i], lightCount, shadingPoint, lightPoint,
                lightNormal);
        }
    }
    return 0.0f;
}

/** Sample a point on a random light: uniform by area over the six faces of a box
 *  or over the tangent cone of a sphere. Always consumes four random numbers so
 *  that every sample uses the same dimension budget. */
__host__ __device__ inline bool sampleLightSurface(const DeviceLight* lights, int lightCount,
    glm::vec3 shadingPoint, float u0, float u1, float u2, float u3,
    glm::vec3& point, glm::vec3& normal, glm::vec3& emission, int& geomIndex,
    int& lightIndex, int& faceIndex)
{
    if (lightCount <= 0)
    {
        return false;
    }
    int index = glm::min((int)(u0 * (float)lightCount), lightCount - 1);
    const DeviceLight& light = lights[index];

    if (light.shape == LIGHT_SPHERE)
    {
        const glm::vec3 toCenter = light.center - shadingPoint;
        const float distance2 = glm::dot(toCenter, toCenter);
        const float radius2 = light.radius * light.radius;
        if (distance2 <= radius2)
        {
            return false;                // the shading point is inside the light
        }
        const glm::vec3 axis = toCenter * glm::inversesqrt(distance2);
        const float cosMax = sqrtf(glm::max(0.0f, 1.0f - radius2 / distance2));
        const glm::vec3 wi = sampleCone(axis, cosMax, u2, u3);
        // Nearest intersection of that direction with the sphere: the sample
        // point, and the normal the emitter radiates along.
        const float b = glm::dot(wi, toCenter);
        const float disc = b * b - (distance2 - radius2);
        if (disc <= 0.0f)
        {
            return false;
        }
        const float t = b - sqrtf(disc);
        point = shadingPoint + wi * t;
        normal = glm::normalize(point - light.center);
        emission = light.emission;
        geomIndex = light.geomIndex;
        lightIndex = index;
        faceIndex = 0;
        return true;
    }

    const glm::vec3 size = light.boundsMax - light.boundsMin;
    const float area[6] = {
        size.x * size.y, size.x * size.y,     // -z and +z
        size.x * size.z, size.x * size.z,     // -y and +y
        size.y * size.z, size.y * size.z      // -x and +x
    };
    float total = 0.0f;
    for (int i = 0; i < 6; i++)
    {
        total += area[i];
    }

    // Pick a face proportional to its area, then a point on that face.
    float r = u1 * total;
    int face = 5;
    for (int i = 0; i < 5; i++)
    {
        if (r < area[i])
        {
            face = i;
            break;
        }
        r -= area[i];
    }

    const float a = u2;
    const float b = u3;
    switch (face)
    {
        case 0: point = glm::vec3(glm::mix(light.boundsMin.x, light.boundsMax.x, a),
                                  glm::mix(light.boundsMin.y, light.boundsMax.y, b),
                                  light.boundsMin.z); normal = glm::vec3(0, 0, -1); break;
        case 1: point = glm::vec3(glm::mix(light.boundsMin.x, light.boundsMax.x, a),
                                  glm::mix(light.boundsMin.y, light.boundsMax.y, b),
                                  light.boundsMax.z); normal = glm::vec3(0, 0, 1); break;
        case 2: point = glm::vec3(glm::mix(light.boundsMin.x, light.boundsMax.x, a),
                                  light.boundsMin.y,
                                  glm::mix(light.boundsMin.z, light.boundsMax.z, b)); normal = glm::vec3(0, -1, 0); break;
        case 3: point = glm::vec3(glm::mix(light.boundsMin.x, light.boundsMax.x, a),
                                  light.boundsMax.y,
                                  glm::mix(light.boundsMin.z, light.boundsMax.z, b)); normal = glm::vec3(0, 1, 0); break;
        case 4: point = glm::vec3(light.boundsMin.x,
                                  glm::mix(light.boundsMin.y, light.boundsMax.y, a),
                                  glm::mix(light.boundsMin.z, light.boundsMax.z, b)); normal = glm::vec3(-1, 0, 0); break;
        default: point = glm::vec3(light.boundsMax.x,
                                   glm::mix(light.boundsMin.y, light.boundsMax.y, a),
                                   glm::mix(light.boundsMin.z, light.boundsMax.z, b)); normal = glm::vec3(1, 0, 0); break;
    }

    emission = light.emission;
    geomIndex = light.geomIndex;
    lightIndex = index;
    faceIndex = face;
    return true;
}

// --- Multiple importance sampling -----------------------------------------

/** Radiance of the environment light (the dome): a three colour sky, blended at
 *  the horizon. Escaping rays collect it, so it needs no second sampling
 *  strategy (see the README). */
__host__ __device__ inline glm::vec3 environmentRadiance(const Environment& environment,
    glm::vec3 direction)
{
    const float t = direction.y;
    const glm::vec3 radiance = (t >= 0.0f)
        ? glm::mix(environment.horizon, environment.zenith, glm::min(t, 1.0f))
        : glm::mix(environment.ground, environment.horizon, glm::clamp(t + 1.0f, 0.0f, 1.0f));
    return environment.intensity * radiance;
}

// --- The distant light (a sun) ---------------------------------------------

/** Is this direction inside the distant light's disc? */
__host__ __device__ inline bool insideDistantLight(const DistantLight& sun, glm::vec3 direction)
{
    return sun.enabled != 0 && glm::dot(direction, -sun.direction) >= sun.cosMaxAngle;
}

/** Density of the distant light's sampler: uniform over its disc, zero
 *  elsewhere. The zero is what keeps MIS well behaved outside the disc. */
__host__ __device__ inline float distantLightPdf(const DistantLight& sun, glm::vec3 direction)
{
    return insideDistantLight(sun, direction) ? (1.0f / glm::max(sun.solidAngle, 1e-12f)) : 0.0f;
}

/** A direction uniformly distributed over the disc's cone (uniform in solid
 *  angle, which is what makes the density above a constant). */
__host__ __device__ inline glm::vec3 sampleDistantLight(const DistantLight& sun, float u1, float u2)
{
    const glm::vec3 axis = -sun.direction;              // towards the light
    const float cosTheta = glm::mix(sun.cosMaxAngle, 1.0f, u1);
    const float sinTheta = sqrtf(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
    glm::vec3 t1, t2;
    buildTangentFrame(axis, t1, t2);
    const float phi = TWO_PI * u2;
    return glm::normalize(t1 * (sinTheta * cosf(phi)) + t2 * (sinTheta * sinf(phi))
        + axis * cosTheta);
}

/** How often the light strategy picks the distant light over an area light: a
 *  fixed half when a scene has both, all of it when it has one. MIS keeps either
 *  choice unbiased; power weighted selection is the refinement. */
__host__ __device__ inline float distantLightSelectionChance(const DistantLight& sun,
    int lightCount, float totalLightArea)
{
    const bool areaLights = (lightCount > 0 && totalLightArea > 0.0f);
    const bool distant = sun.enabled != 0;
    if (distant && areaLights)
    {
        return 0.5f;
    }
    return distant ? 1.0f : 0.0f;
}

/** Density of the light strategy family (area lights + the sun) for a direction.
 *  `areaDensity` is the caller's area light density, 0 for a direction that
 *  leaves the scene. */
__host__ __device__ inline float lightStrategyPdf(const DistantLight& sun, float sunChance,
    float areaDensity, glm::vec3 direction)
{
    return (1.0f - sunChance) * areaDensity + sunChance * distantLightPdf(sun, direction);
}

/** Power heuristic (beta = 2) MIS weight: w = a^2 / (a^2 + b^2). Squaring makes
 *  the better strategy win faster than the balance heuristic. */
__host__ __device__ inline float misWeight(float pdfThis, float pdfOther)
{
    const float a = pdfThis * pdfThis;
    const float b = pdfOther * pdfOther;
    const float sum = a + b;
    return (sum > 0.0f) ? (a / sum) : 0.0f;
}
