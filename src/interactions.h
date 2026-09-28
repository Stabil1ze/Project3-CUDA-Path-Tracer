#pragma once

#include "sceneStructs.h"
#include "sampling.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

// Computes a cosine-weighted random direction in a hemisphere
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

// Same, but with the two random numbers supplied by the caller
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    float u1,
    float u2);

// Scatter a ray according to the material
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool entering,
    const Material& m,
    thrust::default_random_engine& rng);


// Evaluate a procedural texture and return the factor it applies to the material's albedo
__host__ __device__ glm::vec3 evaluateProceduralTexture(int textureType, glm::vec3 p);
