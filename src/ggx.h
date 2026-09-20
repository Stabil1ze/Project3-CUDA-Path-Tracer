#pragma once

// GGX / Trowbridge-Reitz microfacet utilities for the glossy specular lobe.
//
// The functions are written from the published formulas rather than ported from
// another renderer (my CG2025 HW7 used an Embree side project's ggx_utils.hpp,
// which is third party code and explicitly not reused here):
//
//  * D, the GGX normal distribution, and the Smith masking-shadowing function
//    (height correlated form) - PBRT v3 8.4.2, Walter et al. 2007.
//  * Fresnel via Schlick's approximation with a spectral F0 - PBRT v3 8.2.2.
//  * Importance sampling of the *visible* normal distribution - Heitz, "Sampling
//    the GGX Distribution of Visible Normals", JCGT 2018. The visible variant is
//    what keeps grazing angles from wasting samples (see GLOSSY_VNDF_SAMPLING in
//    interactions.cu for the comparison against the classic NDF-only formula).
//
// Everything works in the hemisphere around +Z in a local frame and is lifted
// into world space by the basis helpers at the bottom.

#include <glm/glm.hpp>

#include "utilities.h"   // TWO_PI

/**
 * Alpha from the artist facing roughness. Squaring is the Disney/PBRT
 * convention: perceptual roughness, so that a slider in the middle of its range
 * looks like a surface in the middle of glossy and matte.
 */
__host__ __device__ inline float ggxAlphaFromRoughness(float roughness)
{
    float r = glm::clamp(roughness, 0.0f, 1.0f);
    return glm::max(r * r, 1e-3f);   // never exactly 0: a delta would divide by it
}

/** GGX normal distribution D(h) for a half vector at angle cosTheta from n. */
__host__ __device__ inline float ggxDistribution(float cosTheta, float alpha)
{
    if (cosTheta <= 0.0f)
    {
        return 0.0f;
    }
    float a2 = alpha * alpha;
    float c2 = cosTheta * cosTheta;
    float d = c2 * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * d * d);
}

/** Smith masking function for one direction (GGX / height correlated lambda). */
__host__ __device__ inline float ggxLambda(float cosTheta, float alpha)
{
    float c2 = cosTheta * cosTheta;
    float tan2 = glm::max(0.0f, 1.0f - c2) / glm::max(c2, 1e-8f);
    return 0.5f * (sqrtf(1.0f + alpha * alpha * tan2) - 1.0f);
}

__host__ __device__ inline float ggxG1(float cosTheta, float alpha)
{
    return 1.0f / (1.0f + ggxLambda(cosTheta, alpha));
}

/**
 * Height correlated Smith G2 (Heitz 2014): cheaper than the separable product
 * and it does not darken surfaces at grazing angles, which is what makes a rough
 * metal look like metal instead of like a rough metal that has been dipped in
 * soot. cosThetaV / cosThetaL are the cosines of the view and light directions.
 */
__host__ __device__ inline float ggxG2HeightCorrelated(float cosThetaV, float cosThetaL, float alpha)
{
    float lv = ggxLambda(cosThetaV, alpha);
    float ll = ggxLambda(cosThetaL, alpha);
    return 1.0f / (1.0f + lv + ll);
}

/** Schlick Fresnel, spectral (F0 per channel: 0.04 for dielectrics, the albedo
 *  for metals). */
__host__ __device__ inline glm::vec3 fresnelSchlick(glm::vec3 f0, float cosTheta)
{
    float x = glm::clamp(1.0f - cosTheta, 0.0f, 1.0f);
    float x2 = x * x;
    float x5 = x2 * x2 * x;
    return f0 + (glm::vec3(1.0f) - f0) * x5;
}

/** Orthonormal basis with n as its third axis, robust for n close to any axis. */
__host__ __device__ inline void buildTangentFrame(glm::vec3 n, glm::vec3& t1, glm::vec3& t2)
{
    glm::vec3 helper = (glm::abs(n.x) < SQRT_OF_ONE_THIRD)
        ? glm::vec3(1.0f, 0.0f, 0.0f)
        : ((glm::abs(n.y) < SQRT_OF_ONE_THIRD) ? glm::vec3(0.0f, 1.0f, 0.0f)
                                               : glm::vec3(0.0f, 0.0f, 1.0f));
    t1 = glm::normalize(glm::cross(helper, n));
    t2 = glm::cross(n, t1);
}

/**
 * Sample a half vector from the distribution of *visible* normals (Heitz 2018).
 * `wo` is the direction the ray came from (pointing away from the surface),
 * `alpha` the GGX roughness, u1/u2 two uniform random numbers. Implementation is
 * the standard stretched-vector form: transform into the local frame, stretch wo
 * by alpha, sample the unit disk, and lift the result back.
 */
__host__ __device__ inline glm::vec3 ggxSampleVisibleNormal(glm::vec3 n, glm::vec3 wo, float alpha,
    float u1, float u2)
{
    glm::vec3 t1, t2;
    buildTangentFrame(n, t1, t2);

    glm::vec3 vLocal(glm::dot(wo, t1), glm::dot(wo, t2), glm::dot(wo, n));
    glm::vec3 vh = glm::normalize(glm::vec3(alpha * vLocal.x, alpha * vLocal.y, vLocal.z));

    float lensq = vh.x * vh.x + vh.y * vh.y;
    glm::vec3 T1 = (lensq > 0.0f) ? glm::vec3(-vh.y, vh.x, 0.0f) * glm::inversesqrt(lensq)
                                  : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 T2 = glm::cross(vh, T1);

    float r = sqrtf(u1);
    float phi = TWO_PI * u2;
    float p1 = r * cosf(phi);
    float p2 = r * sinf(phi);
    float s = 0.5f * (1.0f + vh.z);
    p2 = (1.0f - s) * sqrtf(glm::max(0.0f, 1.0f - p1 * p1)) + s * p2;

    glm::vec3 nh = p1 * T1 + p2 * T2
        + sqrtf(glm::max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * vh;
    glm::vec3 ne = glm::normalize(glm::vec3(alpha * nh.x, alpha * nh.y, glm::max(0.0f, nh.z)));

    return t1 * ne.x + t2 * ne.y + n * ne.z;
}

/**
 * The classic (non visible) NDF sampling: cos(theta) = sqrt((1-u)/(1+(a^2-1)u)).
 * Kept for the comparison in the README - at grazing angles it samples half
 * vectors that point below the surface, and every one of those samples is
 * wasted (or silently loses energy if it is dropped).
 */
__host__ __device__ inline glm::vec3 ggxSampleNormal(glm::vec3 n, float alpha, float u1, float u2)
{
    float a2 = alpha * alpha;
    float phi = TWO_PI * u1;
    float cosTheta = sqrtf(glm::max(0.0f, (1.0f - u2) / (1.0f + (a2 - 1.0f) * u2)));
    float sinTheta = sqrtf(glm::max(0.0f, 1.0f - cosTheta * cosTheta));

    glm::vec3 t1, t2;
    buildTangentFrame(n, t1, t2);
    return glm::normalize(t1 * (sinTheta * cosf(phi)) + t2 * (sinTheta * sinf(phi))
        + n * cosTheta);
}
