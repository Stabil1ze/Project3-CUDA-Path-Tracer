#pragma once

// The material system, in the three pieces multiple importance sampling needs:
// bsdfSample draws a direction and weights it, bsdfEval gives the value at a
// direction somebody else chose, and bsdfPdf the density the sampler would have
// produced for it. The material is a mixture of three lobes: the sampler returns
// one lobe's estimate f_j cos / pdf_j (the lobe probability cancels), while
// Eval/Pdf report the mixture sum_j p_j f_j, the density of the combined
// strategy. A delta lobe (mirror, smooth dielectric) is a Dirac: pdf = 0 and
// `specular` set, so no other strategy reaches it and its weight stays 1.

#include <glm/glm.hpp>

#include <thrust/random.h>

#include "ggx.h"
#include "interactions.h"   // calculateRandomDirectionInHemisphere
#include "sceneStructs.h"

// Glossy specular (ROUGHNESS > 0) sampling strategy: 1 samples the distribution
// of *visible* normals (Heitz 2018), 0 the plain NDF (the classic formula, which
// throws away samples whose half vector points below the surface). The sampler
// and the density follow the same switch, so the two stay consistent.
#ifndef GLOSSY_VNDF_SAMPLING
#define GLOSSY_VNDF_SAMPLING 1
#endif

// Dielectric radiance scaling: a transmitted ray that enters a denser medium
// carries radiance scaled by (etaI / etaT)^2 (PBRT v3 8.2.3). Paths that end
// inside a medium (depth budget, Russian roulette) keep the factor without its
// inverse, and glass balls terminate a lot of paths inside. 0 disables it.
#ifndef REFRACTION_RADIANCE_SCALING
#define REFRACTION_RADIANCE_SCALING 1
#endif

/** Fresnel reflectance of a smooth dielectric interface: Schlick's approximation
 *  refined with the cosine of the *transmitted* angle (PBRT v3 8.2.3). Total
 *  internal reflection (sin(theta_t) > 1) comes back as F = 1. */
__host__ __device__ inline float fresnelDielectricSchlick(float cosThetaI, float etaI, float etaT)
{
    float sinThetaTSq = (etaI * etaI) / (etaT * etaT) * (1.0f - cosThetaI * cosThetaI);
    if (sinThetaTSq >= 1.0f)
    {
        return 1.0f;                                  // total internal reflection
    }
    float cosThetaT = sqrtf(glm::max(0.0f, 1.0f - sinThetaTSq));
    float r0 = (etaI - etaT) / (etaI + etaT);
    r0 = r0 * r0;
    float x = 1.0f - cosThetaT;
    float x2 = x * x;
    float fresnel = r0 + (1.0f - r0) * x2 * x2 * x;    // x^5
    return glm::clamp(fresnel, 0.0f, 1.0f);
}

/** Normalized lobe weights. The scenes never mix lobes, but the mixture stays
 *  general so that the densities are defined for any material. */
struct BsdfWeights
{
    float diffuse;
    float specular;
    float dielectric;
};

__host__ __device__ inline BsdfWeights bsdfWeights(const Material& m)
{
    BsdfWeights w;
    w.diffuse = (m.hasReflective > 0.0f || m.hasRefractive > 0.0f) ? 0.0f : 1.0f;
    w.specular = glm::max(m.hasReflective, 0.0f);
    w.dielectric = glm::max(m.hasRefractive, 0.0f);
    float sum = w.diffuse + w.specular + w.dielectric;
    if (sum <= 0.0f)
    {
        // A material with no usable lobe: behave like a black diffuser rather
        // than dividing by zero.
        w.diffuse = 1.0f;
        sum = 1.0f;
    }
    w.diffuse /= sum;
    w.specular /= sum;
    w.dielectric /= sum;
    return w;
}

/** The roughness of the specular lobe, clamped into its meaningful range. */
__host__ __device__ inline float bsdfSpecularRoughness(const Material& m)
{
    return glm::clamp(m.specular.exponent, 0.0f, 1.0f);
}

/** Is a mirror or a dielectric exactly delta distributed for this material? */
__host__ __device__ inline bool bsdfHasDeltaLobe(const Material& m)
{
    const BsdfWeights w = bsdfWeights(m);
    return w.dielectric > 0.0f || (w.specular > 0.0f && bsdfSpecularRoughness(m) <= 0.0f);
}

/** Can this material be connected to a light sample? A delta lobe cannot (one
 *  direction, no density), so glass and mirrors only find lights by chance. */
__host__ __device__ inline bool bsdfHasNonDeltaLobe(const Material& m)
{
    const BsdfWeights w = bsdfWeights(m);
    return w.diffuse > 0.0f || (w.specular > 0.0f && bsdfSpecularRoughness(m) > 0.0f);
}

/** The GGX lobe at a given pair of directions: value and density. `wo` points
 *  towards the viewer, `wi` towards the light; the density is the one the
 *  sampler below produces. */
__host__ __device__ inline void ggxLobeEval(glm::vec3 n, glm::vec3 wo, glm::vec3 wi, float alpha,
    glm::vec3 f0, glm::vec3& f, float& pdf)
{
    const float NdotV = glm::max(glm::dot(n, wo), 1e-4f);
    const float NdotL = glm::dot(n, wi);
    if (NdotL <= 0.0f)
    {
        f = glm::vec3(0.0f);
        pdf = 0.0f;
        return;
    }

    glm::vec3 h = wo + wi;
    if (glm::dot(h, h) <= 0.0f)
    {
        f = glm::vec3(0.0f);
        pdf = 0.0f;
        return;
    }
    h = glm::normalize(h);

    const float NdotH = glm::max(glm::dot(n, h), 0.0f);
    const float VdotH = glm::max(glm::dot(wo, h), 0.0f);
    const float D = ggxDistribution(NdotH, alpha);
    const float G2 = ggxG2HeightCorrelated(NdotV, NdotL, alpha);
    const float G1o = ggxG1(NdotV, alpha);
    const glm::vec3 F = fresnelSchlick(f0, VdotH);

    f = F * (D * G2 / (4.0f * NdotV * NdotL));
#if GLOSSY_VNDF_SAMPLING
    pdf = G1o * D / (4.0f * NdotV);
#else
    pdf = D * NdotH / (4.0f * VdotH);
#endif
    if (!(pdf > 0.0f))
    {
        // A delta shaped lobe (alpha -> 0) has no density to divide by.
        pdf = 0.0f;
        f = glm::vec3(0.0f);
    }
}

/** What the sampler produced. */
struct BsdfSample
{
    glm::vec3 direction;   // world space, on the shading side of the normal
    glm::vec3 weight;      // f * cos / pdf: multiply the path throughput by it
    float pdf;             // mixture density of that direction; 0 if specular
    bool specular;         // came from a delta lobe: no other strategy reaches it
};

__host__ __device__ inline float bsdfPdf(const Material& m, glm::vec3 normal, glm::vec3 wo,
    glm::vec3 wi);

/** Sample the BSDF. `incident` is the unit direction the ray travelled in,
 *  `entering` whether it came from outside the primitive (which swaps the
 *  indices of refraction). A sample with `specular == false && pdf == 0` is
 *  degenerate (half vector below the horizon): the caller ends the path. */
__host__ __device__ inline BsdfSample bsdfSample(const Material& m, glm::vec3 normal, bool entering,
    glm::vec3 incident, thrust::default_random_engine& rng)
{
    const BsdfWeights w = bsdfWeights(m);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // One draw picks the lobe. The clamp matters: thrust's uniform distribution
    // is closed on both ends, so a draw of exactly 1 walks past the last lobe
    // with any weight - a pure mirror then takes the dielectric branch, whose
    // weight is 0 / 0 = NaN and poisons the pixel. (4 paths in 64 million rays.)
    const float lobe = glm::min(u01(rng), 0.9999999f);

    BsdfSample s;
    s.specular = false;
    s.direction = normal;
    s.weight = glm::vec3(0.0f);
    s.pdf = 0.0f;

    if (lobe < w.diffuse)
    {
        // Cosine weighted: f cos / pdf collapses to the albedo.
        s.direction = calculateRandomDirectionInHemisphere(normal, u01(rng), u01(rng));
        s.weight = m.color;
    }
    else if (lobe < w.diffuse + w.specular)
    {
        const float roughness = bsdfSpecularRoughness(m);
        if (roughness <= 0.0f)
        {
            // Delta distribution: f cos / pdf is the reflectance itself.
            s.direction = glm::reflect(incident, normal);
            s.weight = m.specular.color;
            s.specular = true;
        }
        else
        {
            const float alpha = ggxAlphaFromRoughness(roughness);
            const glm::vec3 wo = -incident;                 // towards the viewer
            const float NdotV = glm::max(glm::dot(normal, wo), 1e-4f);
#if GLOSSY_VNDF_SAMPLING
            const glm::vec3 h = ggxSampleVisibleNormal(normal, wo, alpha, u01(rng), u01(rng));
#else
            const glm::vec3 h = ggxSampleNormal(normal, alpha, u01(rng), u01(rng));
#endif
            const glm::vec3 direction = glm::reflect(incident, h);
            const float NdotL = glm::dot(normal, direction);
            const float NdotH = glm::dot(normal, h);
            const float VdotH = glm::dot(wo, h);
            s.direction = direction;
            if (NdotL > 0.0f && NdotH > 0.0f && VdotH > 0.0f)
            {
                const glm::vec3 F = fresnelSchlick(m.specular.color, VdotH);
                const float D = ggxDistribution(NdotH, alpha);
                const float G2 = ggxG2HeightCorrelated(NdotV, NdotL, alpha);
#if GLOSSY_VNDF_SAMPLING
                const float pdf = ggxG1(NdotV, alpha) * D / (4.0f * NdotV);
#else
                const float pdf = D * NdotH / (4.0f * VdotH);
#endif
                if (pdf > 0.0f)
                {
                    // f = F D G2 / (4 cos_i cos_o); with pdf from the visible
                    // normals D cancels and the estimator is F G2 / G1(wo).
                    const glm::vec3 f = F * (D * G2 / (4.0f * NdotV * NdotL));
                    s.weight = f * (NdotL / pdf);
                    s.pdf = pdf;
                }
            }
        }
    }
    else
    {
        // Smooth dielectric: two delta directions chosen with the Fresnel
        // probability, so both weights come out 1 at any angle. The transmitted
        // radiance carries (etaI / etaT)^2 (PBRT v3 8.2.3).
        glm::vec3 n = entering ? normal : -normal;          // outward normal
        if (glm::dot(n, incident) > 0.0f)
        {
            n = -n;                                         // hit the back face
        }

        const float etaI = entering ? 1.0f : m.indexOfRefraction;
        const float etaT = entering ? m.indexOfRefraction : 1.0f;
        const float cosThetaI = glm::clamp(glm::dot(-incident, n), 0.0f, 1.0f);
        const float fresnel = fresnelDielectricSchlick(cosThetaI, etaI, etaT);

        if (u01(rng) < fresnel)
        {
            s.direction = glm::reflect(incident, n);
            s.weight = m.color;
        }
        else
        {
#if REFRACTION_RADIANCE_SCALING
            const float radianceScale = (etaI / etaT) * (etaI / etaT);
#else
            const float radianceScale = 1.0f;
#endif
            s.direction = glm::normalize(glm::refract(incident, n, etaI / etaT));
            s.weight = m.color * radianceScale;
        }
        s.specular = true;
    }

    if (!s.specular)
    {
        // Report the mixture density: the density of the combined "pick a lobe,
        // then sample it" strategy, which is what a MIS weight divides by.
        s.pdf = bsdfPdf(m, normal, -incident, s.direction);
    }
    return s;
}

/** Value of the BSDF at `wo` (towards the viewer) and `wi` (towards the light
 *  sample), plus the density the sampler would have produced for `wi`. Only
 *  lobes with a density contribute: paths still reach lights through glass or a
 *  mirror (that is how caustics are found) and the caller gives those weight 1. */
__host__ __device__ inline glm::vec3 bsdfEval(const Material& m, glm::vec3 normal, glm::vec3 wo,
    glm::vec3 wi, float& pdf)
{
    const BsdfWeights w = bsdfWeights(m);
    pdf = 0.0f;

    const float NdotL = glm::dot(normal, wi);
    if (NdotL <= 0.0f)
    {
        return glm::vec3(0.0f);       // below the shading horizon: no energy
    }

    glm::vec3 f(0.0f);
    if (w.diffuse > 0.0f)
    {
        f += w.diffuse * m.color / PI;
        pdf += w.diffuse * NdotL / PI;
    }
    if (w.specular > 0.0f)
    {
        const float roughness = bsdfSpecularRoughness(m);
        if (roughness > 0.0f)
        {
            glm::vec3 fSpecular;
            float pdfSpecular;
            ggxLobeEval(normal, wo, wi, ggxAlphaFromRoughness(roughness), m.specular.color,
                fSpecular, pdfSpecular);
            f += w.specular * fSpecular;
            pdf += w.specular * pdfSpecular;
        }
        // roughness == 0 is a delta lobe: value 0 at any single direction.
    }
    return f;
}

/** The mixture density for `wi`, for the light strategy's own directions. */
__host__ __device__ inline float bsdfPdf(const Material& m, glm::vec3 normal, glm::vec3 wo,
    glm::vec3 wi)
{
    float pdf = 0.0f;
    bsdfEval(m, normal, wo, wi, pdf);
    return pdf;
}
