#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

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

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
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

    // --- Pick which BSDF lobe this bounce uses ---------------------------
    // Each lobe is weighted by how much of the surface response it carries.
    // The lobe is chosen probabilistically and its throughput is divided by the
    // probability of having picked it, which keeps the estimator unbiased.
    // Today the weights are 0/1 (a material is either diffuse or a mirror), so
    // the sum is 1 and the division is a no-op - but the code stays correct for
    // mixed materials such as glossy = diffuse + imperfect specular.
    float diffuseWeight = (m.hasReflective > 0.0f) ? 0.0f : 1.0f;
    float specularWeight = m.hasReflective;
    float weightSum = diffuseWeight + specularWeight;
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

    if (u01(rng) < diffuseWeight / weightSum)
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
    else
    {
        // --- Perfect specular (mirror) ----------------------------------
        // The reflected direction is the only direction with a non-zero pdf
        // (the BSDF is a delta distribution), so BRDF*cos/pdf reduces to the
        // reflectance itself.
        // ROUGHNESS > 0 would jitter this direction; that is the "imperfect
        // specular" extension (GPU Gems 3, Ch. 20), not implemented yet.
        const float probability = specularWeight / weightSum;
        direction = glm::reflect(glm::normalize(pathSegment.ray.direction), normal);
        weight = m.specular.color * (specularWeight / probability);
    }

    pathSegment.ray.direction = glm::normalize(direction);
    pathSegment.color *= weight;

    // Spawn the next ray from the hit point, nudged off the surface. Without
    // this the new ray can immediately re-intersect the surface it left
    // (shadow acne) because of floating point error in `t`.
    constexpr float RAY_EPSILON = 1e-3f;
    pathSegment.ray.origin = intersect + normal * RAY_EPSILON;
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
