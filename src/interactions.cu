#include "interactions.h"

#include "bsdf.h"
#include "utilities.h"
#include "ggx.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    return calculateRandomDirectionInHemisphere(normal, u01(rng), u01(rng));
}

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    float u1,
    float u2)
{
    float up = sqrt(u1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u2 * TWO_PI;

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

// Dielectric radiance scaling
// 1 (default): scale the radiance by the Fresnel reflectance for refraction rays
// 0: no scaling
#ifndef REFRACTION_RADIANCE_SCALING
#define REFRACTION_RADIANCE_SCALING 1
#endif

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool entering,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // Keep the normal on the side the ray came from
    if (glm::dot(normal, pathSegment.ray.direction) > 0.0f)
    {
        normal = -normal;
    }
    // Geometric normal recovered from the flag
    const glm::vec3 outwardNormal = entering ? normal : -normal;

	// Get sampled direction and weight from the BSDF
    const glm::vec3 incident = glm::normalize(pathSegment.ray.direction);
    const BsdfSample sample = bsdfSample(m, normal, entering, incident, rng);

	// End if the BSDF returned a zero weight or a zero PDF
    if (!sample.specular && !(sample.pdf > 0.0f))
    {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = -1;
        return;
    }

    pathSegment.ray.direction = glm::normalize(sample.direction);
    pathSegment.color *= sample.weight;
    pathSegment.lastPdf = sample.specular ? 0.0f : sample.pdf;

    // Spawn the next ray from the hit point
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
		// Check if the sum of the floored coordinates is even or odd to determine the color of the cell
        float cells = glm::floor(p.x) + glm::floor(p.y) + glm::floor(p.z);
        float parity = glm::mod(cells, 2.0f);
        return glm::mix(glm::vec3(1.0f), glm::vec3(0.08f, 0.08f, 0.10f), parity);
    }

	// Veins along a diagonal, distorted by fractal noise for a marble-like effect
    float veins = fractalNoise(p);
    float bands = sinf((p.x + p.y * 0.6f + p.z * 0.35f) * 3.0f + veins * 6.0f);
    bands = bands * bands;
    return glm::mix(glm::vec3(0.34f, 0.33f, 0.40f), glm::vec3(1.0f), bands);
}
