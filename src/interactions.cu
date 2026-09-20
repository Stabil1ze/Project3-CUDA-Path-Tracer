#include "interactions.h"

#include "utilities.h"
#include "ggx.h"

#include <thrust/random.h>

// Glossy specular (ROUGHNESS > 0) sampling strategy, for the comparison in the
// README: 1 samples the distribution of *visible* normals (Heitz 2018, keeps the
// grazing angle samples useful), 0 samples the plain NDF (the classic formula,
// which throws away samples whose half vector points below the surface).
#ifndef GLOSSY_VNDF_SAMPLING
#define GLOSSY_VNDF_SAMPLING 1
#endif

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

// Dielectric radiance scaling: a transmitted ray that enters a denser medium
// carries radiance scaled by (etaI / etaT)^2 (PBRT v3 8.2.3). It matters for any
// path that ends inside the medium - a path that enters and leaves again gets the
// factor and its inverse, but one that is terminated inside (depth budget or
// Russian roulette) does not, and glass balls terminate a lot of paths inside.
// Set to 0 to see the difference.
#ifndef REFRACTION_RADIANCE_SCALING
#define REFRACTION_RADIANCE_SCALING 1
#endif

/**
 * Fresnel reflectance of a smooth dielectric interface, Schlick's approximation
 * refined with the cosine of the *transmitted* angle (PBRT v3 8.2.3), which stays
 * accurate for indices of refraction far from 1.5:
 *
 *     F = F0 + (1 - F0) (1 - cos(theta_t))^5,   F0 = ((etaI - etaT)/(etaI + etaT))^2
 *
 * Snell's law gives sin(theta_t) = etaI/etaT * sin(theta_i); when that exceeds 1
 * the ray cannot leave the medium at all and the interface reflects everything
 * (total internal reflection), which this reports as F = 1.
 */
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

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool entering,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // Keep the normal on the side the ray came from. The intersection tests
    // already flip it when the ray starts inside a primitive, but a surface
    // can still be hit while the normal points "away" (e.g. grazing hits on a
    // transformed box), which would send the diffuse ray into the object.
    if (glm::dot(normal, pathSegment.ray.direction) > 0.0f)
    {
        normal = -normal;
    }
    // Geometric normal (pointing outward) recovered from the flag: the
    // intersection tests return the normal oriented against the incoming ray, so
    // for a hit from the inside it comes back negated. Refraction swaps the
    // indices of refraction by direction, not by which way the normal happens to
    // point, so keep the outward one around.
    const glm::vec3 outwardNormal = entering ? normal : -normal;

    // --- Pick which BSDF lobe this bounce uses ---------------------------
    // Each lobe is weighted by how much of the surface response it carries.
    // The lobe is chosen probabilistically and its throughput is divided by the
    // probability of having picked it, which keeps the estimator unbiased.
    // The weights are 0/1 for the materials the scenes use (a surface is
    // diffuse, a mirror or a dielectric), so the sum is 1 and the division is a
    // no-op - but the code stays correct for mixed materials such as
    // glossy = diffuse + imperfect specular.
    float diffuseWeight = (m.hasReflective > 0.0f || m.hasRefractive > 0.0f) ? 0.0f : 1.0f;
    float specularWeight = m.hasReflective;
    float dielectricWeight = m.hasRefractive;
    float weightSum = diffuseWeight + specularWeight + dielectricWeight;
    if (weightSum <= 0.0f)
    {
        // Material with no usable lobe: fall back to a black diffuser instead
        // of producing NaNs.
        diffuseWeight = 1.0f;
        weightSum = 1.0f;
    }

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    glm::vec3 direction;
    glm::vec3 weight;

    // One draw selects the lobe from the cumulative weights.
    const float lobe = u01(rng);
    if (lobe < diffuseWeight / weightSum)
    {
        // --- Ideal diffuse (Lambertian) ---------------------------------
        // Sample the outgoing direction from a cosine-weighted hemisphere;
        // the cosine term and the pdf cancel exactly:
        //     BRDF = albedo / PI,  pdf = cos(theta) / PI
        //     throughput *= BRDF * cos(theta) / pdf = albedo
        // which is why cosine-weighted sampling is both cheap and low variance.
        const float probability = diffuseWeight / weightSum;
        direction = calculateRandomDirectionInHemisphere(normal, rng);
        weight = m.color * (diffuseWeight / probability);
    }
    else if (lobe < (diffuseWeight + specularWeight) / weightSum)
    {
        const float probability = specularWeight / weightSum;
        const glm::vec3 incident = glm::normalize(pathSegment.ray.direction);
        const float roughness = glm::clamp(m.specular.exponent, 0.0f, 1.0f);

        if (roughness <= 0.0f)
        {
            // --- Perfect specular (mirror) -------------------------------
            // The reflected direction is the only direction with a non-zero pdf
            // (the BSDF is a delta distribution), so BRDF*cos/pdf reduces to the
            // reflectance itself.
            direction = glm::reflect(incident, normal);
            weight = m.specular.color * (specularWeight / probability);
        }
        else
        {
            // --- GGX microfacet specular (glossy, roughness > 0) ---------
            // BRDF = F * D * G / (4 cos_i cos_o), sampled by importance sampling
            // the distribution of *visible* normals (Heitz 2018). With a half
            // vector h drawn from that distribution the pdf of the reflected
            // direction is
            //     pdf = D_vis(h) / (4 (wo . h)) = G1(wo) D(h) / (4 cos_o)
            // so the estimator collapses to
            //     BRDF * cos_i / pdf = F * G2 / G1(wo)
            // which is why D never has to be evaluated in the result: it cancels.
            // (F, and therefore the material colour, still carry the energy.)
            const float alpha = ggxAlphaFromRoughness(roughness);
            const glm::vec3 wo = -incident;          // toward the viewer
            const float NdotV = glm::max(glm::dot(normal, wo), 1e-4f);

#if GLOSSY_VNDF_SAMPLING
            const glm::vec3 h = ggxSampleVisibleNormal(normal, wo, alpha, u01(rng), u01(rng));
#else
            const glm::vec3 h = ggxSampleNormal(normal, alpha, u01(rng), u01(rng));
#endif
            const glm::vec3 glossyDirection = glm::reflect(incident, h);
            const float NdotL = glm::dot(normal, glossyDirection);
            const float NdotH = glm::dot(normal, h);
            const float VdotH = glm::dot(wo, h);

            // Half vectors below the horizon (or light below the surface) carry
            // no energy; with the visible normal distribution this is rare, with
            // the classic NDF sampling it is common at grazing angles.
            if (NdotL <= 0.0f || NdotH <= 0.0f || VdotH <= 0.0f)
            {
                pathSegment.color = glm::vec3(0.0f);
                pathSegment.remainingBounces = -1;
                return;
            }

            const glm::vec3 F = fresnelSchlick(m.specular.color, VdotH);
            const float D = ggxDistribution(NdotH, alpha);
            const float G2 = ggxG2HeightCorrelated(NdotV, NdotL, alpha);
            const float G1o = ggxG1(NdotV, alpha);
#if GLOSSY_VNDF_SAMPLING
            // pdf of the visible normal distribution, converted to the solid
            // angle of the reflected direction
            const float pdf = G1o * D / (4.0f * NdotV);
#else
            // pdf of the plain NDF sampling
            const float pdf = D * NdotH / (4.0f * VdotH);
#endif
            if (pdf <= 0.0f)
            {
                pathSegment.color = glm::vec3(0.0f);
                pathSegment.remainingBounces = -1;
                return;
            }

            const glm::vec3 f = F * (D * G2 / (4.0f * NdotV * NdotL));
            direction = glossyDirection;
            weight = f * (NdotL / pdf) * (specularWeight / probability);
        }
    }
    else
    {
        // --- Smooth dielectric (glass, water) ----------------------------
        // A dielectric has exactly two delta directions: the mirror direction
        // and the refracted one. They are chosen with the Fresnel probability,
        // so the estimator stays unbiased for any combination of angle and
        // index of refraction:
        //
        //   reflect  with probability F        -> weight F / F = 1
        //   transmit with probability (1 - F)  -> weight (1 - F) / (1 - F) = 1
        //
        // and the transmitted radiance is scaled by (etaI / etaT)^2, which is the
        // factor that makes a path entering glass and leaving it again carry the
        // right amount of energy (PBRT v3 8.2.3).
        const float probability = dielectricWeight / weightSum;
        const glm::vec3 incident = glm::normalize(pathSegment.ray.direction);

        glm::vec3 n = outwardNormal;
        if (glm::dot(n, incident) > 0.0f)
        {
            n = -n;                                   // hit the back face
        }

        const float etaI = entering ? 1.0f : m.indexOfRefraction;
        const float etaT = entering ? m.indexOfRefraction : 1.0f;
        const float cosThetaI = glm::clamp(glm::dot(-incident, n), 0.0f, 1.0f);
        const float fresnel = fresnelDielectricSchlick(cosThetaI, etaI, etaT);

        if (u01(rng) < fresnel)
        {
            direction = glm::reflect(incident, n);
            weight = m.color * (dielectricWeight / probability);
        }
        else
        {
            direction = glm::normalize(glm::refract(incident, n, etaI / etaT));
#if REFRACTION_RADIANCE_SCALING
            const float radianceScale = (etaI / etaT) * (etaI / etaT);
#else
            const float radianceScale = 1.0f;
#endif
            weight = m.color * radianceScale * (dielectricWeight / probability);
        }
    }

    pathSegment.ray.direction = glm::normalize(direction);
    pathSegment.color *= weight;

    // Spawn the next ray from the hit point, nudged off the surface. Without
    // this the new ray can immediately re-intersect the surface it left
    // (shadow acne) because of floating point error in `t`. The nudge has to go
    // to the side the *new* ray leaves on: a transmitted ray goes into the
    // medium, so pushing it back out along the incident side would trap it
    // inside the surface.
    glm::vec3 offsetNormal = outwardNormal;
    if (glm::dot(offsetNormal, pathSegment.ray.direction) < 0.0f)
    {
        offsetNormal = -offsetNormal;
    }
    constexpr float RAY_EPSILON = 1e-3f;
    pathSegment.ray.origin = intersect + offsetNormal * RAY_EPSILON;
}

// ---------------------------------------------------------------------------
// Procedural textures
// ---------------------------------------------------------------------------
// Both textures are pure functions of the object space hit point, which is what
// makes them procedural: no files, no texture memory, no UVs, and the same code
// works on any shape - including the SDF fractals, whose UV layout would be a
// nightmare to define.

__host__ __device__ inline float hashToUnitFloat(unsigned int x)
{
    // Integer finaliser (shifts + multiplies, very cheap on the GPU) that hides
    // the lattice structure of the value noise below.
    x ^= x >> 17;
    x *= 0xed5ad4bbu;
    x ^= x >> 11;
    x *= 0xac4c1b51u;
    x ^= x >> 15;
    return (float)(x & 0x00ffffffu) * (1.0f / 16777216.0f);
}

__host__ __device__ inline float latticeNoise(int x, int y, int z)
{
    unsigned int h = ((unsigned int)(x * 73856093))
        ^ ((unsigned int)(y * 19349663))
        ^ ((unsigned int)(z * 83492791));
    return hashToUnitFloat(h);
}

// Trilinear value noise with a smoothstep fade: 8 hashes and 7 lerps per octave.
__host__ __device__ inline float valueNoise(glm::vec3 p)
{
    glm::vec3 i = glm::floor(p);
    glm::vec3 f = p - i;
    glm::vec3 w = f * f * (3.0f - 2.0f * f);

    int x = (int)i.x, y = (int)i.y, z = (int)i.z;
    float c000 = latticeNoise(x, y, z),       c100 = latticeNoise(x + 1, y, z);
    float c010 = latticeNoise(x, y + 1, z),   c110 = latticeNoise(x + 1, y + 1, z);
    float c001 = latticeNoise(x, y, z + 1),   c101 = latticeNoise(x + 1, y, z + 1);
    float c011 = latticeNoise(x, y + 1, z + 1), c111 = latticeNoise(x + 1, y + 1, z + 1);

    float x00 = glm::mix(c000, c100, w.x), x10 = glm::mix(c010, c110, w.x);
    float x01 = glm::mix(c001, c101, w.x), x11 = glm::mix(c011, c111, w.x);
    return glm::mix(glm::mix(x00, x10, w.y), glm::mix(x01, x11, w.y), w.z);
}

// Fractal sum of value noise.
__host__ __device__ inline float fractalNoise(glm::vec3 p)
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    for (int i = 0; i < 5; i++)
    {
        sum += amplitude * valueNoise(p);
        p *= 2.03f;         // not exactly 2, so the octaves do not line up
        amplitude *= 0.5f;
    }
    return sum;
}

__host__ __device__ glm::vec3 evaluateProceduralTexture(int textureType, glm::vec3 p)
{
    if (textureType == 1)
    {
        // Checker board: the parity of the lattice cell the point falls in. The
        // albedo is multiplied by either 1 or a dark value, so a coloured
        // material keeps its colour but gains dark squares.
        float cells = glm::floor(p.x) + glm::floor(p.y) + glm::floor(p.z);
        float parity = glm::mod(cells, 2.0f);
        return glm::mix(glm::vec3(1.0f), glm::vec3(0.08f, 0.08f, 0.10f), parity);
    }

    // Marble: veins along a diagonal, distorted by fractal noise. sin^2 keeps the
    // result in [0, 1] so the multiplier never brightens the albedo.
    float veins = fractalNoise(p);
    float bands = sinf((p.x + p.y * 0.6f + p.z * 0.35f) * 3.0f + veins * 6.0f);
    bands = bands * bands;
    return glm::mix(glm::vec3(0.34f, 0.33f, 0.40f), glm::vec3(1.0f), bands);
}
