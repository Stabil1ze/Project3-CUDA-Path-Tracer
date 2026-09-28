#pragma once

#include "sceneStructs.h"

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
