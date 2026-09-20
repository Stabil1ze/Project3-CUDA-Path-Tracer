#pragma once

// Low-discrepancy sampling for the Monte Carlo integrals in the path tracer.
//
// The renderer used to draw every random number from a hash seeded linear
// congruential generator (thrust::default_random_engine), which is fine but has
// the usual 1/sqrt(N) convergence of pure random sampling and, worse, happily
// clumps: two consecutive samples can land on top of each other. This file
// replaces the draws with a **scrambled Halton sequence**: dimension d of sample
// i of pixel p is
//
//     x = fract(Phi_{b_d}(i) + offset(p, d))
//
// where Phi_{b_d} is the radical inverse in the d-th prime base and the offset is
// a Cranley-Patterson rotation hashed from the pixel and the dimension. Each
// dimension therefore has its own base *and* its own rotation, which is what
// decorrelates both the pixels and the dimensions of one path.
//
// Two ways of getting this wrong, both measured rather than guessed at:
//
//  * Giving every dimension the *same* base with only an XOR of the index (the
//    cheap "Sobol-like" shortcut) makes the coordinates of a 2D sample lie on a
//    constant diagonal - each coordinate is stratified on its own, but the pair
//    is not, and the anti-aliasing got six times worse (silhouette RMSE 11.8
//    against 2.2 for the random sampler it was supposed to beat).
//  * Giving two *bounces* the same dimensions makes every bounce of a path draw
//    the same values, which correlates them and darkens the image by 2%.
//    pathtrace.cu therefore gives each bounce a block of eight dimensions.
//
// Why it helps: the error of a Monte Carlo estimate is driven by how evenly the
// samples cover the domain. Stratified and low-discrepancy sequences put the
// sample points down and then *keep them apart*, so a fixed budget of samples
// covers pixel area, lens disk and BSDF lobe more evenly than random draws. The
// measurement is in the README (same sample count, same scenes, lower error).

#include <glm/glm.hpp>
#include <thrust/random.h>

// Toggle for the whole feature: 1 uses the scrambled Halton sequence, 0 falls
// back to the hash seeded LCG that the renderer used before, so the improvement
// can be measured with a one line change.
#ifndef LOW_DISCREPANCY_SAMPLING
#define LOW_DISCREPANCY_SAMPLING 1
#endif

/** Integer scramble used for the per pixel rotations (a cheap xorshift mix). */
__host__ __device__ inline unsigned int samplingHash(unsigned int x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/** Radical inverse of `index` in `base`: the van der Corput sequence. */
__host__ __device__ inline float radicalInverse(int base, unsigned int index)
{
    const float invBase = 1.0f / (float)base;
    float invBaseN = invBase;
    float result = 0.0f;
    while (index > 0u)
    {
        unsigned int digit = index % (unsigned int)base;
        result += (float)digit * invBaseN;
        index /= (unsigned int)base;
        invBaseN *= invBase;
    }
    return result;
}

/**
 * The sample stream of one path. It is created from (iteration, pixel) and hands
 * out one low-discrepancy value per dimension, so every sample of every pixel
 * walks the same sequence with its own rotation. Nothing here is stateful across
 * iterations, which is what keeps a checkpointed render resumable: sample i of
 * pixel p is the same number whether it is drawn now or after a restart.
 */
/**
 * One prime per path dimension (64 of them, so a path with eight bounces of
 * eight dimensions still never reuses a base). Beyond the table the bases wrap,
 * which is the standard compromise; the rotation still distinguishes them.
 * A file scope array would not be visible in device code, hence the function.
 */
__host__ __device__ inline int samplingBase(unsigned int dimension)
{
    const int bases[64] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
        59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
        137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
        211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
        283, 293, 307, 311
    };
    return bases[dimension % 64u];
}

struct Sampler
{
    unsigned int iteration;
    unsigned int pixel;
    unsigned int dimension;

    __host__ __device__ Sampler(unsigned int iter, unsigned int px, unsigned int first = 0u)
        : iteration(iter), pixel(px), dimension(first) {}

    __host__ __device__ float next()
    {
        const int base = samplingBase(dimension);
        const unsigned int scramble = samplingHash(pixel * 9781u + dimension * 6271u + 1u);
        const float rotation = (float)(scramble & 0x00ffffffu) * (1.0f / 16777216.0f);
        float value = radicalInverse(base, iteration) + rotation;
        value -= floorf(value);
        dimension++;
        return value;
    }

    __host__ __device__ glm::vec2 next2D()
    {
        float x = next();
        float y = next();
        return glm::vec2(x, y);
    }

    /** Jump to a dedicated dimension, so a decision is not entangled with how
     *  many draws the BSDF of this bounce happened to use. */
    __host__ __device__ void jumpTo(unsigned int dim)
    {
        dimension = dim;
    }
};

/**
 * The one place that decides where a random number comes from: the low
 * discrepancy sequence, or the generator the renderer used before (kept so the
 * two can be compared). Everything that needs a random number goes through it.
 */
__host__ __device__ inline float nextRandom(thrust::default_random_engine& rng, Sampler& sampler)
{
#if LOW_DISCREPANCY_SAMPLING
    (void)rng;
    return sampler.next();
#else
    (void)sampler;
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    return u01(rng);
#endif
}
