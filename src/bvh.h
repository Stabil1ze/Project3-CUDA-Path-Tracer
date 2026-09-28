#pragma once

#include "sceneStructs.h"
#include "intersections.h"   // intersectGeom, the shared primitive dispatch

#include <glm/glm.hpp>

#include <vector>

// Compile-time knobs so the write-up can quote a measurement for each; the
// traversal stack is sized by the depth
#define BVH_LEAF_SIZE 4
#define BVH_MAX_DEPTH 48
#define BVH_SAH_BINS 16

/** One node of the flattened BVH. The tree is built on the CPU (only the
 *  *traversal* has to be GPU side) and stored as an array, so the device side
 *  walks a stack of indices and no pointers. A leaf keeps a contiguous range of
 *  `primitiveIds`, an interior node the indices of its two children. */
struct BvhNode
{
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    int leftChild;         // -1 for a leaf
    int rightChild;        // -1 for a leaf
    int firstPrimitive;    // leaf: index into primitiveIds
    int primitiveCount;    // leaf: how many; 0 for an interior node
};

struct Bvh
{
    std::vector<BvhNode> nodes;
    std::vector<int> primitiveIds;
    int maxDepth = 0;
    int leafCount = 0;
    int leafPrimitives = 0;
    double buildMs = 0.0;
    glm::vec3 boundsMin = glm::vec3(0.0f);
    glm::vec3 boundsMax = glm::vec3(0.0f);
};

/** Build the hierarchy over every geometry in the scene, in place on the host:
 *  one primitive per entry of `geoms`, whatever kind it is (a mesh triangle is a
 *  primitive like any other). Splits use the surface area heuristic over 16 bins
 *  along the widest centroid axis, and only when a split beats a leaf. */
void buildBvh(const std::vector<Geom>& geoms, Bvh& bvh);

// --- Bounding volume hierarchy traversal -----------------------------------
// The tree is built on the host (see bvh.cpp) and walked iteratively: recursion
// on the device diverges and the stack is small, so the traversal carries its
// own array of node indices, bounded by BVH_MAX_DEPTH.
#define BVH_STACK_SIZE (BVH_MAX_DEPTH + 4)

/** Slab test: does the ray reach this box no further away than `maxT`?
 *  `fminf`/`fmaxf` return the non-NaN operand, so a zero direction component
 *  (0 * inf) falls out of the arithmetic instead of needing a branch. */
__device__ inline bool rayHitsBox(const glm::vec3& boundsMin, const glm::vec3& boundsMax,
    const glm::vec3& origin, const glm::vec3& inverseDirection, float maxT, float& entryDistance)
{
    const glm::vec3 t0 = (boundsMin - origin) * inverseDirection;
    const glm::vec3 t1 = (boundsMax - origin) * inverseDirection;
    // (`near` and `far` are macros in the Windows headers, hence the names.)
    const glm::vec3 nearT = glm::vec3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    const glm::vec3 farT = glm::vec3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));

    const float entry = fmaxf(fmaxf(nearT.x, nearT.y), fmaxf(nearT.z, 0.0f));
    const float exit = fminf(fminf(farT.x, farT.y), fminf(farT.z, maxT));
    entryDistance = entry;
    return exit >= entry;
}

/** Closest hit through the hierarchy; same contract as the flat loop. Children
 *  are visited nearest first so `bestT` tightens early and the far subtrees are
 *  cut by the slab test instead of being traversed. */
__device__ inline bool bvhClosestHit(
    const BvhNode* nodes,
    const Geom* geoms,
    const int* primitiveIds,
    Ray ray,
    unsigned long long* sdfSteps,
    unsigned long long* sdfHistogram,
    float& bestT,
    glm::vec3& bestPoint,
    glm::vec3& bestNormal,
    int& bestGeom,
    bool& bestOutside)
{
    const glm::vec3 inverseDirection(1.0f / ray.direction.x, 1.0f / ray.direction.y,
        1.0f / ray.direction.z);
    int stack[BVH_STACK_SIZE];
    int top = 0;
    stack[top++] = 0;
    bestT = FLT_MAX;

    while (top > 0)
    {
        const BvhNode node = nodes[stack[--top]];
        float entry = 0.0f;
        if (!rayHitsBox(node.boundsMin, node.boundsMax, ray.origin, inverseDirection, bestT, entry))
        {
            continue;
        }

        if (node.primitiveCount > 0)
        {
            for (int i = 0; i < node.primitiveCount; i++)
            {
                const int id = primitiveIds[node.firstPrimitive + i];
                glm::vec3 point, normal;
                bool outside = true;
                const float t = intersectGeom(geoms[id], ray, point, normal, outside,
                    sdfSteps, sdfHistogram);
                if (t > 0.0f && t < bestT)
                {
                    bestT = t;
                    bestPoint = point;
                    bestNormal = normal;
                    bestGeom = id;
                    bestOutside = outside;
                }
            }
            continue;
        }

        float leftEntry = 0.0f;
        float rightEntry = 0.0f;
        const BvhNode& left = nodes[node.leftChild];
        const BvhNode& right = nodes[node.rightChild];
        const bool hitLeft = rayHitsBox(left.boundsMin, left.boundsMax, ray.origin,
            inverseDirection, bestT, leftEntry);
        const bool hitRight = rayHitsBox(right.boundsMin, right.boundsMax, ray.origin,
            inverseDirection, bestT, rightEntry);
        if (hitLeft && hitRight)
        {
            // Push the farther one first: it is popped last, after the near
            // subtree has had a chance to shrink the bound.
            const bool nearFirst = leftEntry <= rightEntry;
            stack[top++] = nearFirst ? node.rightChild : node.leftChild;
            stack[top++] = nearFirst ? node.leftChild : node.rightChild;
        }
        else if (hitLeft)
        {
            stack[top++] = node.leftChild;
        }
        else if (hitRight)
        {
            stack[top++] = node.rightChild;
        }
    }

    return bestT < FLT_MAX;
}

/** Any hit along the shadow ray, with the same early out the flat loop has. */
__device__ inline bool bvhOccluded(const BvhNode* nodes, const Geom* geoms,
    const int* primitiveIds, Ray shadow, float maxT, int skipGeom)
{
    const glm::vec3 inverseDirection(1.0f / shadow.direction.x, 1.0f / shadow.direction.y,
        1.0f / shadow.direction.z);
    int stack[BVH_STACK_SIZE];
    int top = 0;
    stack[top++] = 0;

    while (top > 0)
    {
        const BvhNode node = nodes[stack[--top]];
        float entry = 0.0f;
        if (!rayHitsBox(node.boundsMin, node.boundsMax, shadow.origin, inverseDirection, maxT, entry))
        {
            continue;
        }

        if (node.primitiveCount > 0)
        {
            for (int i = 0; i < node.primitiveCount; i++)
            {
                const int id = primitiveIds[node.firstPrimitive + i];
                if (id == skipGeom)
                {
                    continue;
                }
                glm::vec3 point, normal;
                bool outside = true;
                const float t = intersectGeom(geoms[id], shadow, point, normal, outside);
                if (t > 1e-4f && t < maxT)
                {
                    return true;
                }
            }
            continue;
        }

        stack[top++] = node.leftChild;
        stack[top++] = node.rightChild;
    }

    return false;
}

/** Is the segment from `origin` along `direction` (up to `maxT`) blocked?
 *  `bvhNodes` NULL falls back to the flat loop, which is also the reference the
 *  BVH has to match - both paths share `intersectGeom` and the same epsilon. */
__device__ inline bool isOccluded(Geom* geoms, int geomCount, glm::vec3 origin,
    glm::vec3 direction, float maxT, int skipGeom,
    const BvhNode* bvhNodes = NULL, const int* bvhPrimitiveIds = NULL)
{
    Ray shadow;
    shadow.origin = origin;
    shadow.direction = direction;
    if (bvhNodes != NULL)
    {
        return bvhOccluded(bvhNodes, geoms, bvhPrimitiveIds, shadow, maxT, skipGeom);
    }
    for (int i = 0; i < geomCount; i++)
    {
        if (i == skipGeom)
        {
            // Never let a light shadow itself: a sampled point is behind its own
            // silhouette for oblique views, which would reject valid samples
            continue;
        }
        Geom& geom = geoms[i];
        glm::vec3 p, n;
        bool outside = true;
        const float t = intersectGeom(geom, shadow, p, n, outside);
        if (t > 1e-4f && t < maxT)
        {
            return true;
        }
    }
    return false;
}
