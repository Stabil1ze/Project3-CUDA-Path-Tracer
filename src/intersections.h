#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

// Compile-time toggle for the procedural shape intersection: 1 clips the ray
// against each shape's bounding sphere before sphere tracing it, 0 marches every
// ray (the "before" side of the culling measurement in the README).
#ifndef SDF_BOUNDING_SPHERE
#define SDF_BOUNDING_SPHERE 1
#endif


/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

// CHECKITOUT
/**
 * Test intersection between a ray and one of the procedural signed distance
 * field shapes (MANDELBULB, MENGER). Unlike the primitives above there is no
 * closed form for the hit, so the ray is marched with sphere tracing: the SDF
 * gives a lower bound on the distance to the surface, so a step of that size can
 * never tunnel through it.
 *
 * The march runs in world space but evaluates the SDF in object space, so the
 * step is divided by the object's largest scale factor - scaling a Lipschitz-1
 * field by s turns it into a Lipschitz-s field, and without that division a
 * scaled up object would be stepped straight through.
 *
 * @param stepCounter  Optional instrumentation: adds the number of marching
 *                     steps this call used (may be NULL).
 * @param histogram    Optional instrumentation: 16 buckets of 8 steps (may be
 *                     NULL).
 * @return             Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sdfIntersectionTest(
    Geom geom,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    unsigned long long* stepCounter,
    unsigned int* histogram);

/**
 * Evaluate the signed distance field of `geom` at an object space point. Units:
 * object space, i.e. distance to the surface in the untransformed shape.
 */
__host__ __device__ float sdfEvaluate(int geomType, glm::vec3 p);
