#pragma once

#include "sceneStructs.h"
#include "sampling.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

/** Same, but with the two random numbers supplied by the caller (low
 *  discrepancy samples instead of generator draws). */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    float u1,
    float u2);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the PathSegment in place: it writes the
 * scattered ray (origin + direction) and multiplies the running throughput
 * `pathSegment.color` by the BSDF/pdf ratio of the branch that was taken.
 *
 * `intersect` is the world space hit point, `normal` the surface normal and
 * `m` the material that was hit. `entering` says whether the ray came from
 * outside the primitive, which the dielectric lobe needs to swap the indices of
 * refraction (and to know when total internal reflection applies).
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool entering,
    const Material& m,
    thrust::default_random_engine& rng);

// CHECKITOUT
/**
 * Evaluate a procedural texture and return the factor it applies to the
 * material's albedo. The pattern is a function of the *object space* position of
 * the hit point, so it is attached to the object (and follows its transform)
 * rather than to world space.
 *
 * `textureType` 1 = checker, 2 = marble (see the definitions in interactions.cu);
 * `p` is the already scaled object space point.
 */
__host__ __device__ glm::vec3 evaluateProceduralTexture(int textureType, glm::vec3 p);
