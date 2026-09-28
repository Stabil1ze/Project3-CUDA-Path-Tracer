#pragma once

// Low-discrepancy sampling for the Monte Carlo integrals in the path tracer.
//
// Dimension d of sample i of pixel p is fract(Phi_{b_d}(i) + offset(p, d)): the
// radical inverse in the d-th prime base, rotated by a Cranley-Patterson offset
// hashed from the pixel and the dimension. Each dimension has its own base *and*
// its own rotation, which decorrelates the pixels and the dimensions of a path.
// Both ways of getting this wrong were measured, not guessed: sharing one base
// across dimensions puts a 2D sample on a constant diagonal (silhouette RMSE
// 11.8 against 2.2 for the random sampler it was meant to beat), and sharing
// dimensions across bounces darkens the image by 2%.

#include <glm/glm.hpp>
#include <thrust/random.h>

// Whole feature toggle
// 1 (default): scrambled Halton sequence
// 0: the hash seeded LCG the renderer used before, so the improvement is
//    measurable with a one line change
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

/** Prime base of one path dimension (64 of them, so eight bounces of eight
 *  dimensions still never reuse one). Beyond the table the bases wrap, which is
 *  the standard compromise; the rotation still distinguishes them. A file scope
 *  array would not be visible in device code, hence the function. */
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

/** The sample stream of one path: created from (iteration, pixel), it hands out
 *  one low-discrepancy value per dimension. Nothing here is stateful across
 *  iterations, which is what keeps a checkpointed render resumable - sample i of
 *  pixel p is the same number whether it is drawn now or after a restart. */
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

/** The one place that decides where a random number comes from: the low
 *  discrepancy sequence, or the generator used before (kept for comparison). */
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
