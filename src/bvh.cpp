#include "bvh.h"

#include "intersections.h"

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <iostream>

namespace
{
    struct BuildPrimitive
    {
        glm::vec3 boundsMin;
        glm::vec3 boundsMax;
        glm::vec3 centroid;
        int id;
    };

    void expand(glm::vec3& boundsMin, glm::vec3& boundsMax, const glm::vec3& point)
    {
        boundsMin = glm::min(boundsMin, point);
        boundsMax = glm::max(boundsMax, point);
    }

    glm::vec3 nodeExtent(const glm::vec3& boundsMin, const glm::vec3& boundsMax)
    {
        return glm::max(boundsMax - boundsMin, glm::vec3(0.0f));
    }

    float surfaceArea(const glm::vec3& extent)
    {
        return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
    }
}

// Recursive binned SAH build
static int buildNode(Bvh& bvh, std::vector<BuildPrimitive>& primitives, int first, int count,
    int depth)
{
    const int nodeIndex = (int)bvh.nodes.size();
    bvh.nodes.emplace_back();
    bvh.maxDepth = std::max(bvh.maxDepth, depth);

    glm::vec3 boundsMin(FLT_MAX);
    glm::vec3 boundsMax(-FLT_MAX);
    glm::vec3 centroidMin(FLT_MAX);
    glm::vec3 centroidMax(-FLT_MAX);
    for (int i = first; i < first + count; i++)
    {
        const BuildPrimitive& primitive = primitives[i];
        expand(boundsMin, boundsMax, primitive.boundsMin);
        expand(boundsMin, boundsMax, primitive.boundsMax);
        expand(centroidMin, centroidMax, primitive.centroid);
    }
    bvh.nodes[nodeIndex].boundsMin = boundsMin;
    bvh.nodes[nodeIndex].boundsMax = boundsMax;
    bvh.nodes[nodeIndex].leftChild = -1;
    bvh.nodes[nodeIndex].rightChild = -1;
    bvh.nodes[nodeIndex].firstPrimitive = first;
    bvh.nodes[nodeIndex].primitiveCount = 0;

    const glm::vec3 centroidExtent = nodeExtent(centroidMin, centroidMax);
    const bool canSplit = count > BVH_LEAF_SIZE && depth < BVH_MAX_DEPTH
        && (centroidExtent.x > 0.0f || centroidExtent.y > 0.0f || centroidExtent.z > 0.0f);
    if (!canSplit)
    {
        bvh.nodes[nodeIndex].primitiveCount = count;
        bvh.leafCount++;
        bvh.leafPrimitives += count;
        return nodeIndex;
    }

	// cost = surfaceArea(left) * leftCount + surfaceArea(right) * rightCount
    int axis = 0;
    if (centroidExtent.y > centroidExtent.x) axis = 1;
    if (centroidExtent.z > centroidExtent[axis]) axis = 2;

    glm::vec3 binMin[BVH_SAH_BINS];
    glm::vec3 binMax[BVH_SAH_BINS];
    int binCount[BVH_SAH_BINS] = { 0 };
    for (int b = 0; b < BVH_SAH_BINS; b++)
    {
        binMin[b] = glm::vec3(FLT_MAX);
        binMax[b] = glm::vec3(-FLT_MAX);
    }

    const float scale = (float)BVH_SAH_BINS / centroidExtent[axis];
    for (int i = first; i < first + count; i++)
    {
        const BuildPrimitive& primitive = primitives[i];
        int bin = (int)((primitive.centroid[axis] - centroidMin[axis]) * scale);
        bin = std::min(bin, BVH_SAH_BINS - 1);
        bin = std::max(bin, 0);
        binCount[bin]++;
        expand(binMin[bin], binMax[bin], primitive.boundsMin);
        expand(binMin[bin], binMax[bin], primitive.boundsMax);
    }

    // Sweep the bins once from each side to get the cost of every split plane
    float rightArea[BVH_SAH_BINS];
    int rightCount[BVH_SAH_BINS];
    glm::vec3 runningMin(FLT_MAX);
    glm::vec3 runningMax(-FLT_MAX);
    int running = 0;
    for (int b = BVH_SAH_BINS - 1; b >= 0; b--)
    {
		// Skip empty bins
        if (binCount[b] > 0)
        {
            expand(runningMin, runningMax, binMin[b]);
            expand(runningMin, runningMax, binMax[b]);
        }
        running += binCount[b];
        rightArea[b] = surfaceArea(nodeExtent(runningMin, runningMax));
        rightCount[b] = running;
    }

    int bestSplit = -1;
    float bestCost = FLT_MAX;
    runningMin = glm::vec3(FLT_MAX);
    runningMax = glm::vec3(-FLT_MAX);
    running = 0;
    for (int b = 0; b < BVH_SAH_BINS - 1; b++)
    {
        if (binCount[b] > 0)
        {
            expand(runningMin, runningMax, binMin[b]);
            expand(runningMin, runningMax, binMax[b]);
        }
        running += binCount[b];
        if (running == 0 || rightCount[b + 1] == 0)
        {
            continue;
        }
        const float cost = surfaceArea(nodeExtent(runningMin, runningMax)) * running
            + rightArea[b + 1] * rightCount[b + 1];
        if (cost < bestCost)
        {
            bestCost = cost;
            bestSplit = b;
        }
    }

    // A leaf costs count * its own area; only split when the split is cheaper
    const float leafCost = surfaceArea(nodeExtent(boundsMin, boundsMax)) * count;
    if (bestSplit < 0)
    {
        bvh.nodes[nodeIndex].primitiveCount = count;
        bvh.leafCount++;
        bvh.leafPrimitives += count;
        return nodeIndex;
    }

    const int splitBin = bestSplit;
    BuildPrimitive* begin = primitives.data() + first;
    BuildPrimitive* end = begin + count;
    BuildPrimitive* middle = std::partition(begin, end, [&](const BuildPrimitive& primitive) {
        int bin = (int)((primitive.centroid[axis] - centroidMin[axis]) * scale);
        return std::min(std::max(bin, 0), BVH_SAH_BINS - 1) <= splitBin;
    });

    const int leftCount = (int)(middle - begin);
    if (leftCount == 0 || leftCount == count || bestCost >= leafCost)
    {
		// The split was degenerate or not worth it; make a leaf instead
        bvh.nodes[nodeIndex].primitiveCount = count;
        bvh.leafCount++;
        bvh.leafPrimitives += count;
        return nodeIndex;
    }

    const int left = buildNode(bvh, primitives, first, leftCount, depth + 1);
    const int right = buildNode(bvh, primitives, first + leftCount, count - leftCount, depth + 1);
    bvh.nodes[nodeIndex].leftChild = left;
    bvh.nodes[nodeIndex].rightChild = right;
    return nodeIndex;
}

void buildBvh(const std::vector<Geom>& geoms, Bvh& bvh)
{
    const auto start = std::chrono::steady_clock::now();

    bvh.nodes.clear();
    bvh.primitiveIds.clear();
    bvh.maxDepth = 0;
    bvh.leafCount = 0;
    bvh.leafPrimitives = 0;

    std::vector<BuildPrimitive> primitives;
    primitives.reserve(geoms.size());
    for (size_t i = 0; i < geoms.size(); i++)
    {
        BuildPrimitive primitive;
        geomWorldBounds(geoms[i], primitive.boundsMin, primitive.boundsMax);
        primitive.centroid = 0.5f * (primitive.boundsMin + primitive.boundsMax);
        primitive.id = (int)i;
        primitives.push_back(primitive);
    }

    if (!primitives.empty())
    {
        bvh.primitiveIds.resize(primitives.size());
        bvh.nodes.reserve(2 * primitives.size() / BVH_LEAF_SIZE + 1);
        buildNode(bvh, primitives, 0, (int)primitives.size(), 0);

		// Recursively build the BVH and fill in the primitive IDs
        for (size_t i = 0; i < primitives.size(); i++)
        {
            bvh.primitiveIds[i] = primitives[i].id;
        }
        bvh.boundsMin = bvh.nodes[0].boundsMin;
        bvh.boundsMax = bvh.nodes[0].boundsMax;
    }

    bvh.buildMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}
