#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "bvh.h"
#include "lights.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "bsdf.h"
#include "sampling.h"
#include "../stream_compaction/efficient.h"   // Project 2 work-efficient scan

#define ERRORCHECK 1

// Stochastic sampled antialiasing
// 1 (default): jitter the camera ray inside its pixel
// 0: sample the pixel centre - the aliased "before" image
#define STOCHASTIC_AA 1

// Stream compaction
// 1 (default): remove the terminated paths after every bounce
// 0: keep them and skip them inside the kernels
#define STREAM_COMPACTION 1

// Material sorting
// 1: reorder the paths so that a warp sees one material
// 0 (default): shade them in the order the bounces left them in
#define SORT_BY_MATERIAL 0

// Instrumentation for the material sort
// 1 (default): print how mixed the warps are, before and after the sort
// 0: off
#define MATERIAL_SORT_STATS 1

// Bounding volume hierarchy: walk a CPU built tree instead of testing every
// primitive for every ray
// 1 (default): use the BVH above BVH_MIN_PRIMITIVES
// 0: flat loop
#define USE_BVH 1

// Below this many primitives the flat loop is faster than a traversal
#define BVH_MIN_PRIMITIVES 24

// Procedural shape instrumentation: sphere tracing steps into a global counter
// plus a 16 bucket histogram, printed at the end of the render
// 1 (default): on
// 0: off (the atomics cost a few percent, so off when timing)
#define SDF_STATS 1

// Restartable rendering
// 1 (default): checkpoint the accumulation buffer every CHECKPOINT seconds
// 0: disable
#define RESTARTABLE 1
// Staging buffer the checkpoint is pulled off the device into
// 1 (default): pinned (page-locked) memory - the driver DMAs straight into it
// 0: pageable memory, which bounces through a driver staging buffer
#define CHECKPOINT_PINNED_MEMORY 1

// Russian roulette: kill unimportant paths early without biasing the estimator
// 1 (default): from RR_MIN_DEPTH segments on, survive with p = throughput and
//              divide the survivors by p
// 0: render every path until it escapes, hits an emitter or runs out of depth
#define RUSSIAN_ROULETTE 1
#define RR_MIN_DEPTH 3              // segments; segment 1 is the camera ray
#define RR_MIN_SURVIVAL 0.05f       // never keep fewer than 5% of the paths

// Direct lighting
// 1 (default): on
// 0: off - the "before" side of the measurement
#define DIRECT_LIGHT_SAMPLING 1

// Sample ledger for the estimator above
// 1 (default): on (a few atomics per sample, so off when timing)
// 0: off
#define DIRECT_LIGHT_STATS 1

// Skip the sampled light in its own shadow test
// 1 (default): skip it
// 0: keep it
#define LIGHT_SKIP_SELF_IN_SHADOW 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

// Double buffers used by stream compaction: compaction scatters into the "alt"
// arrays and the two pairs are swapped afterwards (scattering in place would
// race, since element i moves to a slot another thread may not have read yet).
static PathSegment* dev_pathsAlt = NULL;
static ShadeableIntersection* dev_intersectionsAlt = NULL;
static int* dev_alive = NULL;          // predicate: 1 = keep, 0 = terminated
static int* dev_scanIndices = NULL;    // exclusive prefix sum of the predicate
static int h_scanCapacity = 0;         // elements allocated for dev_scanIndices

// Buffers for the material sort
static int* dev_materialKeys = NULL;   // 0 = the ray escaped, otherwise 1 + material id
static int* dev_bucketCounts = NULL;   // histogram of the keys
static int* dev_bucketOffsets = NULL;  // exclusive prefix sum, i.e. run starts
static int* dev_bucketCursors = NULL;  // the offsets again, bumped while scattering
static int h_bucketCapacity = 0;       // elements allocated in each of the three above
#if MATERIAL_SORT_STATS
static std::vector<int> h_materialKeys;   // the keys, brought back for the analysis
#endif

// The BVH acceleration structure
static BvhNode* dev_bvhNodes = NULL;
static int* dev_bvhPrimitiveIds = NULL;
static int h_bvhNodeCount = 0;
static bool h_bvhActive = false;

// Denoiser guides
static glm::vec3* dev_firstNormal = NULL;
static glm::vec3* dev_firstAlbedo = NULL;

// Per-bounce ray counts for analysis
static const int MAX_PROFILE_DEPTH = 64;
static long long h_bounceAlive[MAX_PROFILE_DEPTH];
static long long h_profileIters = 0;

// GPU timer
struct StageTimer
{
    cudaEvent_t events[2][2];
    int slot = 0;
    int recorded = 0;
    int samples = 0;
    double totalMs = 0.0;
    bool created = false;

    void init()
    {
        for (int s = 0; s < 2; s++)
        {
            cudaEventCreate(&events[s][0]);
            cudaEventCreate(&events[s][1]);
        }
        slot = 0;
        recorded = 0;
        samples = 0;
        totalMs = 0.0;
        created = true;
    }

    void destroy()
    {
        if (!created) { return; }
        for (int s = 0; s < 2; s++)
        {
            cudaEventDestroy(events[s][0]);
            cudaEventDestroy(events[s][1]);
        }
        created = false;
    }

    void begin()
    {
        cudaEventRecord(events[slot][0]);
    }

    void end()
    {
        cudaEventRecord(events[slot][1]);
        slot = 1 - slot;        // the other pair holds the previous iteration
        recorded++;
        if (recorded < 2) { return; }
        if (cudaEventQuery(events[slot][1]) == cudaSuccess)
        {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, events[slot][0], events[slot][1]);
            totalMs += ms;
            samples++;
        }
    }

    double averageMs() const
    {
        return samples > 0 ? totalMs / samples : 0.0;
    }
};

static StageTimer stageTrace, stageSort, stageShade;

// SDF instrumentation
static const int SDF_HISTOGRAM_BUCKETS = 16;
static const int SDF_STEPS_PER_BUCKET = 8;
static unsigned long long* dev_sdfSteps = NULL;
static unsigned long long* dev_sdfHistogram = NULL;
static unsigned long long h_sdfSteps = 0;
static unsigned long long h_sdfHistogram[SDF_HISTOGRAM_BUCKETS];

// Russian roulette instrumentation
static unsigned long long* dev_rrDecisions = NULL;
static unsigned long long* dev_rrKills = NULL;

// Summed in fixed point
static unsigned long long* dev_rrSurvivalMilli = NULL;
static unsigned long long h_rrDecisions = 0;
static unsigned long long h_rrKills = 0;
static unsigned long long h_rrSurvivalMilli = 0;

// Direct lighting instrumentation

static DeviceLight* dev_lights = NULL;
static std::vector<DeviceLight> h_lightInfo;   // same list, kept for the printout
static int h_lightCount = 0;
static float h_totalLightArea = 0.0f;



static LightLedger* dev_lightLedger = NULL;       // one row per light
static LightLedger* dev_lightFaceLedger = NULL;   // one row per light and face
static LightLedger* dev_foldedLedger = NULL;      // the BSDF hits the estimator replaced
static LightLedger* dev_foldedAboveLedger = NULL;  
static LightLedger* dev_foldedFaceLedger = NULL;   // [face + 6 * fromAbove], diagnostics

static void addLedger(LightLedger& into, const LightLedger& from)
{
    into.samples += from.samples;
    into.accepted += from.accepted;
    into.rejectCosSurface += from.rejectCosSurface;
    into.rejectCosLight += from.rejectCosLight;
    into.occluded += from.occluded;
    into.acceptedEnergy += from.acceptedEnergy;
    into.acceptedEnergySq += from.acceptedEnergySq;
    into.nonFiniteEnergy += from.nonFiniteEnergy;
    into.bsdfHits += from.bsdfHits;
    into.bsdfEnergy += from.bsdfEnergy;
    into.bsdfEnergyPlain += from.bsdfEnergyPlain;
    into.bsdfEnergySq += from.bsdfEnergySq;
}


static int nextPowerOfTwoAtLeast(int n)
{
    int m = 1;
    while (m < n)
    {
        m <<= 1;
    }
    return m;
}

static void printBounceProfile(const char* tag, const long long* counts, int segments, long long divisor, int pixelcount)
{
    if (divisor <= 0)
    {
        return;
    }

    printf("[profile] %s", tag);
    for (int i = 0; i < segments && i < MAX_PROFILE_DEPTH; i++)
    {
        long long alive = counts[i] / divisor;
        printf(" b%d=%lld(%.1f%%)", i + 1, alive,
            100.0 * (double)alive / (double)pixelcount);
    }
    printf("\n");
}

// Where the time went, as GPU milliseconds per bounce (see StageTimer)
static void printStageTimings()
{
    const double trace = stageTrace.averageMs();
    const double sort = stageSort.averageMs();
    const double shade = stageShade.averageMs();
    if (trace <= 0.0 && sort <= 0.0 && shade <= 0.0)
    {
        return;
    }
    printf("[stage] per bounce: intersections %.2f ms, material sort %.2f ms, shading %.2f ms "
        "(%d bounces averaged)\n", trace, sort, shade, stageShade.samples);
}

// README instrumentation for the procedural shapes
static void printSdfStats(int pixelcount)
{
#if SDF_STATS
    if (dev_sdfSteps == NULL)
    {
        return;
    }
    cudaMemcpy(&h_sdfSteps, dev_sdfSteps, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sdfHistogram, dev_sdfHistogram,
        SDF_HISTOGRAM_BUCKETS * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    unsigned long long tests = 0;
    for (int i = 0; i < SDF_HISTOGRAM_BUCKETS; i++)
    {
        tests += h_sdfHistogram[i];
    }
    if (tests == 0)
    {
        return;
    }

    printf("[sdf] sphere tracing: %u marches, %.1f steps per march, %.2f steps per camera ray",
        (unsigned int)tests, (double)h_sdfSteps / (double)tests,
        (double)h_sdfSteps / (double)pixelcount);
    printf("\n[sdf] steps per march histogram (bucket width %d):", SDF_STEPS_PER_BUCKET);
    for (int i = 0; i < SDF_HISTOGRAM_BUCKETS; i++)
    {
        if (h_sdfHistogram[i] == 0)
        {
            continue;
        }
        printf(" %d-%d:%llu", i * SDF_STEPS_PER_BUCKET,
            (i + 1) * SDF_STEPS_PER_BUCKET - 1, h_sdfHistogram[i]);
    }
    printf("\n");
#endif
}

// README instrumentation for Russian roulette
static void printRussianRouletteStats(int pixelcount)
{
#if RUSSIAN_ROULETTE
    if (dev_rrDecisions == NULL)
    {
        return;
    }
    cudaMemcpy(&h_rrDecisions, dev_rrDecisions, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_rrKills, dev_rrKills, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_rrSurvivalMilli, dev_rrSurvivalMilli, sizeof(unsigned long long),
        cudaMemcpyDeviceToHost);
    if (h_rrDecisions == 0)
    {
        return;
    }

    printf("[rr] %llu roulette decisions (%.2f per camera ray), %llu killed (%.1f%%), "
        "mean survival probability %.3f\n",
        h_rrDecisions, (double)h_rrDecisions / (double)pixelcount,
        h_rrKills, 100.0 * (double)h_rrKills / (double)h_rrDecisions,
        (double)h_rrSurvivalMilli / (1000.0 * (double)h_rrDecisions));
#else
    (void)pixelcount;
#endif
}

// README instrumentation for direct lighting
static const char* lightFaceName(int face)
{
    switch (face)
    {
        case 0: return "-z";
        case 1: return "+z";
        case 2: return "-y";
        case 3: return "+y";
        case 4: return "-x";
        default: return "+x";
    }
}

static void printDirectLightStats(int pixelcount)
{
#if DIRECT_LIGHT_SAMPLING && DIRECT_LIGHT_STATS
    if (dev_lightLedger == NULL)
    {
        return;
    }

    std::vector<LightLedger> perLight(h_lightCount + 1);
    std::vector<LightLedger> perFace((size_t)glm::max(h_lightCount, 1) * LIGHT_SAMPLE_FACES);
    LightLedger folded;
    cudaMemcpy(perLight.data(), dev_lightLedger, perLight.size() * sizeof(LightLedger),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(perFace.data(), dev_lightFaceLedger, perFace.size() * sizeof(LightLedger),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(&folded, dev_foldedLedger, sizeof(LightLedger), cudaMemcpyDeviceToHost);
    LightLedger foldedAbove;
    cudaMemcpy(&foldedAbove, dev_foldedAboveLedger, sizeof(LightLedger), cudaMemcpyDeviceToHost);

    LightLedger total = {};
    // The last row is the distant light, which is not an area light and so has
    // no entry in the light list.
    for (int i = 0; i <= h_lightCount; i++)
    {
        addLedger(total, perLight[i]);
    }
    if (total.samples == 0)
    {
        return;
    }

    printf("[lights] %llu samples (%.2f per camera ray): %.1f%% accepted, %.1f%% below the "
        "shading horizon, %.1f%% behind the sampled face, %.1f%% occluded\n",
        total.samples, (double)total.samples / (double)pixelcount,
        100.0 * (double)total.accepted / (double)total.samples,
        100.0 * (double)total.rejectCosSurface / (double)total.samples,
        100.0 * (double)total.rejectCosLight / (double)total.samples,
        100.0 * (double)total.occluded / (double)total.samples);

    for (int i = 0; i < h_lightCount; i++)
    {
        const LightLedger& row = perLight[i];
        printf("[lights] light %d: %llu samples, %.1f%% accepted, %.4g units of accepted energy, "
            "%llu non finite\n",
            i, row.samples,
            100.0 * (double)row.accepted / (double)glm::max(row.samples, 1ull), row.acceptedEnergy,
            row.nonFiniteEnergy);
        for (int face = 0; face < LIGHT_SAMPLE_FACES; face++)
        {
            const LightLedger& faceRow = perFace[(size_t)i * LIGHT_SAMPLE_FACES + face];
            if (faceRow.samples == 0)
            {
                continue;
            }
            printf("[lights]   face %s: %llu samples, %.1f%% accepted | rejected: horizon %llu, "
                "behind the face %llu, occluded %llu | %.4g units of accepted energy\n",
                lightFaceName(face), faceRow.samples,
                100.0 * (double)faceRow.accepted / (double)faceRow.samples,
                faceRow.rejectCosSurface, faceRow.rejectCosLight, faceRow.occluded,
                faceRow.acceptedEnergy);
        }
    }
    {
        const LightLedger& row = perLight[h_lightCount];
        if (row.samples > 0)
        {
            printf("[lights] distant light: %llu samples, %.1f%% accepted, %.4g units of accepted "
                "energy (rejected: horizon %llu, occluded %llu)\n",
                row.samples,
                100.0 * (double)row.accepted / (double)glm::max(row.samples, 1ull),
                row.acceptedEnergy, row.rejectCosSurface, row.occluded);
        }
    }

    // The estimator against the one it replaced: same integral over the same
    // vertices, so the two totals agree in expectation
    if (folded.bsdfHits == 0 || total.accepted == 0)
    {
        printf("[lights] energy check: no samples to compare\n");
        return;
    }
    const double nee = total.acceptedEnergy + folded.bsdfEnergy;
    const double fold = folded.bsdfEnergyPlain;
    // Trials are the light samples, not the non-zero outcomes: dividing by the
    // survivors would collapse the variance to a precision we do not have.
    const double trials = (double)total.samples;
    const double foldVar = glm::max(folded.bsdfEnergySq - fold * fold / trials, 0.0);
    const double noise = sqrt(foldVar);
    printf("[lights] energy check: the multiple importance sampling estimator %.6g (light strategy "
        "%.6g + weighted emitter hits %.6g) against the %llu unweighted BSDF emitter hits %.6g "
        "(+-%.3f%%) over %.3g light samples: %+.3f%% +- %.3f%%\n",
        nee, total.acceptedEnergy, folded.bsdfEnergy,
        folded.bsdfHits, fold, 100.0 * sqrt(foldVar) / fold, trials,
        100.0 * (nee - fold) / fold, 100.0 * noise / fold);
    printf("[lights]   of the BSDF hits, %llu (%+.3f%% of their energy) came from above the "
        "light's own bottom plane: %.6g units\n",
        foldedAbove.bsdfHits,
        100.0 * foldedAbove.bsdfEnergy / fold, foldedAbove.bsdfEnergy);
    std::vector<LightLedger> foldedFace(2 * LIGHT_SAMPLE_FACES);
    cudaMemcpy(foldedFace.data(), dev_foldedFaceLedger, foldedFace.size() * sizeof(LightLedger),
        cudaMemcpyDeviceToHost);
    for (int side = 0; side < 2; side++)
    {
        printf("[lights]   BSDF hits by light face, %s:", side ? "from above" : "from below");
        for (int face = 0; face < LIGHT_SAMPLE_FACES; face++)
        {
            const LightLedger& row = foldedFace[face + LIGHT_SAMPLE_FACES * side];
            printf(" %s=%llu/%.3g%%", lightFaceName(face), row.bsdfHits,
                100.0 * row.bsdfEnergy / fold);
        }
        printf("\n");
    }
#else
    (void)pixelcount;
#endif
}

// --- Direct lighting: device side ----------------------------------------

#if DIRECT_LIGHT_SAMPLING && DIRECT_LIGHT_STATS
#endif


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


// ---------------------------------------------------------------------------
// Restartable rendering - checkpoint file format and helpers
// ---------------------------------------------------------------------------

static const char CHECKPOINT_MAGIC[8] = { 'P', '3', 'C', 'K', 'P', 'T', '0', '1' };

struct CheckpointHeader
{
    char magic[8];
    unsigned long long sceneHash;   // identifies the scene the samples belong to
    int resolutionX;
    int resolutionY;
    int traceDepth;
    int iterations;                 // samples already accumulated
    int headerBytes;                // guards against a format change
    int reserved;
};

static float* h_checkpointStaging = NULL;
static size_t h_checkpointStagingBytes = 0;
static cudaStream_t checkpointStream = NULL;
static int h_checkpointSaves = 0;
static int h_checkpointLoads = 0;
static double h_checkpointCopyMs = 0.0;
static double h_checkpointWriteMs = 0.0;
static double h_checkpointLoadMs = 0.0;
static long long h_checkpointBytesWritten = 0;

static std::string checkpointPath(const Scene* scene)
{
    return scene->state.imageName + ".ckpt";
}

static unsigned long long hashCheckpointBytes(unsigned long long hash, const void* data, size_t bytes)
{
    // FNV-1a, applied byte wise so that it does not depend on struct padding.
    const unsigned char* p = (const unsigned char*)data;
    for (size_t i = 0; i < bytes; i++)
    {
        hash ^= (unsigned long long)p[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

// Set fingerprint of the scene's state and geometry into a 64-bit hash
static unsigned long long checkpointSceneHash(const Scene* scene)
{
    unsigned long long hash = 1469598103934665603ull;
    const RenderState& state = scene->state;
    const Camera& cam = state.camera;

    hash = hashCheckpointBytes(hash, &cam.position, sizeof(glm::vec3));
    hash = hashCheckpointBytes(hash, &cam.lookAt, sizeof(glm::vec3));
    hash = hashCheckpointBytes(hash, &cam.up, sizeof(glm::vec3));
    hash = hashCheckpointBytes(hash, &cam.fov, sizeof(glm::vec2));
    hash = hashCheckpointBytes(hash, &cam.pixelLength, sizeof(glm::vec2));
    hash = hashCheckpointBytes(hash, &cam.aperture, sizeof(float));
    hash = hashCheckpointBytes(hash, &cam.focalDistance, sizeof(float));
    hash = hashCheckpointBytes(hash, &cam.resolution, sizeof(glm::ivec2));
    hash = hashCheckpointBytes(hash, &state.traceDepth, sizeof(int));
    // Scene level lights: the sky and the sun change the image without touching
    // a material, so they have to be part of the fingerprint
    hash = hashCheckpointBytes(hash, &state.environment.zenith, sizeof(glm::vec3));
    hash = hashCheckpointBytes(hash, &state.environment.horizon, sizeof(glm::vec3));
    hash = hashCheckpointBytes(hash, &state.environment.ground, sizeof(glm::vec3));
    hash = hashCheckpointBytes(hash, &state.environment.intensity, sizeof(float));
    hash = hashCheckpointBytes(hash, &state.distantLight.direction, sizeof(glm::vec3));
    hash = hashCheckpointBytes(hash, &state.distantLight.radiance, sizeof(glm::vec3));
    hash = hashCheckpointBytes(hash, &state.distantLight.cosMaxAngle, sizeof(float));
    hash = hashCheckpointBytes(hash, &state.distantLight.solidAngle, sizeof(float));
    hash = hashCheckpointBytes(hash, &state.distantLight.enabled, sizeof(int));

    for (const Material& m : scene->materials)
    {
        hash = hashCheckpointBytes(hash, &m.color, sizeof(glm::vec3));
        hash = hashCheckpointBytes(hash, &m.specular.exponent, sizeof(float));
        hash = hashCheckpointBytes(hash, &m.specular.color, sizeof(glm::vec3));
        hash = hashCheckpointBytes(hash, &m.hasReflective, sizeof(float));
        hash = hashCheckpointBytes(hash, &m.hasRefractive, sizeof(float));
        hash = hashCheckpointBytes(hash, &m.indexOfRefraction, sizeof(float));
        hash = hashCheckpointBytes(hash, &m.emittance, sizeof(float));
        hash = hashCheckpointBytes(hash, &m.textureType, sizeof(int));
        hash = hashCheckpointBytes(hash, &m.textureScale, sizeof(float));
    }
    for (const Geom& g : scene->geoms)
    {
        hash = hashCheckpointBytes(hash, &g.type, sizeof(GeomType));
        hash = hashCheckpointBytes(hash, &g.materialid, sizeof(int));
        hash = hashCheckpointBytes(hash, &g.translation, sizeof(glm::vec3));
        hash = hashCheckpointBytes(hash, &g.rotation, sizeof(glm::vec3));
        hash = hashCheckpointBytes(hash, &g.scale, sizeof(glm::vec3));
        hash = hashCheckpointBytes(hash, &g.transform, sizeof(glm::mat4));
        // Mesh triangles: replacing an .obj with one that has the same triangle
        // count leaves every field above unchanged
        if (g.type == TRIANGLE)
        {
            hash = hashCheckpointBytes(hash, &g.v0, sizeof(glm::vec3));
            hash = hashCheckpointBytes(hash, &g.v1, sizeof(glm::vec3));
            hash = hashCheckpointBytes(hash, &g.v2, sizeof(glm::vec3));
            hash = hashCheckpointBytes(hash, &g.n0, sizeof(glm::vec3));
            hash = hashCheckpointBytes(hash, &g.n1, sizeof(glm::vec3));
            hash = hashCheckpointBytes(hash, &g.n2, sizeof(glm::vec3));
        }
    }
    return hash;
}

static void releaseCheckpointStaging()
{
    if (h_checkpointStaging != NULL)
    {
#if CHECKPOINT_PINNED_MEMORY
        cudaFreeHost(h_checkpointStaging);
#else
        free(h_checkpointStaging);
#endif
        h_checkpointStaging = NULL;
    }
    h_checkpointStagingBytes = 0;
    if (checkpointStream != NULL)
    {
        cudaStreamDestroy(checkpointStream);
        checkpointStream = NULL;
    }
}

// The staging buffer is allocated on first use and kept, so that a render that
// checkpoints periodically does not pay an allocation per checkpoint.
static bool ensureCheckpointStaging(size_t bytes)
{
    if (h_checkpointStaging != NULL && h_checkpointStagingBytes >= bytes)
    {
        return true;
    }
    releaseCheckpointStaging();

#if CHECKPOINT_PINNED_MEMORY
    if (cudaHostAlloc((void**)&h_checkpointStaging, bytes, cudaHostAllocDefault) != cudaSuccess)
    {
        h_checkpointStaging = NULL;
        cudaGetLastError();
        return false;
    }
    if (cudaStreamCreate(&checkpointStream) != cudaSuccess)
    {
        cudaFreeHost(h_checkpointStaging);
        h_checkpointStaging = NULL;
        cudaGetLastError();
        return false;
    }
#else
    h_checkpointStaging = (float*)malloc(bytes);
    if (h_checkpointStaging == NULL)
    {
        return false;
    }
#endif
    h_checkpointStagingBytes = bytes;
    return true;
}

// Pull the accumulation buffer off the device into the staging buffer
static double downloadCheckpointImage(size_t bytes)
{
    auto start = std::chrono::steady_clock::now();
#if CHECKPOINT_PINNED_MEMORY
    cudaMemcpyAsync(h_checkpointStaging, dev_image, bytes, cudaMemcpyDeviceToHost,
        checkpointStream);
    cudaStreamSynchronize(checkpointStream);
#else
    cudaMemcpy(h_checkpointStaging, dev_image, bytes, cudaMemcpyDeviceToHost);
#endif
    checkCUDAError("checkpoint download");
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // Denoiser guides
    cudaMalloc(&dev_firstNormal, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_firstNormal, 0, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_firstAlbedo, pixelcount * sizeof(glm::vec3));
    thrust::fill(thrust::device, dev_firstAlbedo, dev_firstAlbedo + pixelcount,
        glm::vec3(1.0f));

    // BVH based acceleration structure
    h_bvhActive = false;
    h_bvhNodeCount = 0;
    if (USE_BVH && (int)scene->geoms.size() >= BVH_MIN_PRIMITIVES)
    {
        Bvh bvh;
        buildBvh(scene->geoms, bvh);
        h_bvhNodeCount = (int)bvh.nodes.size();
        cudaMalloc(&dev_bvhNodes, bvh.nodes.size() * sizeof(BvhNode));
        cudaMemcpy(dev_bvhNodes, bvh.nodes.data(), bvh.nodes.size() * sizeof(BvhNode),
            cudaMemcpyHostToDevice);
        cudaMalloc(&dev_bvhPrimitiveIds, bvh.primitiveIds.size() * sizeof(int));
        cudaMemcpy(dev_bvhPrimitiveIds, bvh.primitiveIds.data(),
            bvh.primitiveIds.size() * sizeof(int), cudaMemcpyHostToDevice);
        h_bvhActive = true;
        printf("[bvh] %d primitives: %d nodes, %d leaves (%.1f primitives each), depth %d, "
            "built in %.1f ms\n", (int)scene->geoms.size(), h_bvhNodeCount, bvh.leafCount,
            bvh.leafCount > 0 ? (double)bvh.leafPrimitives / bvh.leafCount : 0.0, bvh.maxDepth,
            bvh.buildMs);
    }
    else
    {
        printf("[bvh] %d primitives: flat loop%s\n", (int)scene->geoms.size(),
            USE_BVH ? " (below the crossover)" : " (USE_BVH 0)");
    }

    // Buffers for the stream compaction stage (see compactPaths below).
    h_scanCapacity = nextPowerOfTwoAtLeast(pixelcount + 1);   // +1 so the scan also yields the total
    cudaMalloc(&dev_pathsAlt, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersectionsAlt, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_alive, pixelcount * sizeof(int));
    cudaMalloc(&dev_scanIndices, h_scanCapacity * sizeof(int));

    // Buffers for the material sort (see sortPathsByMaterial below): one key per
    // path plus the three per-material arrays - one bucket per material, plus
    // one for the rays that escaped.
    h_bucketCapacity = nextPowerOfTwoAtLeast((int)scene->materials.size() + 2);
    cudaMalloc(&dev_materialKeys, pixelcount * sizeof(int));
    cudaMalloc(&dev_bucketCounts, h_bucketCapacity * sizeof(int));
    cudaMalloc(&dev_bucketOffsets, h_bucketCapacity * sizeof(int));
    cudaMalloc(&dev_bucketCursors, h_bucketCapacity * sizeof(int));
#if MATERIAL_SORT_STATS
    h_materialKeys.assign(pixelcount, 0);
#endif

    stageTrace.init();
    stageSort.init();
    stageShade.init();

    h_profileIters = 0;
    for (int i = 0; i < MAX_PROFILE_DEPTH; i++)
    {
        h_bounceAlive[i] = 0;
    }

    // Instrumentation for the procedural shapes (sphere tracing steps).
    cudaMalloc(&dev_sdfSteps, sizeof(unsigned long long));
    cudaMemset(dev_sdfSteps, 0, sizeof(unsigned long long));
    cudaMalloc(&dev_sdfHistogram, SDF_HISTOGRAM_BUCKETS * sizeof(unsigned long long));
    cudaMemset(dev_sdfHistogram, 0, SDF_HISTOGRAM_BUCKETS * sizeof(unsigned long long));

    // --- Direct lighting: build the light list ---------------------------
    // The unit box the base code intersects is [-0.5, 0.5]^3, so transforming
    // its eight corners gives the world space extents of the emissive geometry.
    std::vector<DeviceLight> lights;
    h_totalLightArea = 0.0f;
    for (const Geom& geom : scene->geoms)
    {
        const Material& material = scene->materials[geom.materialid];
        if (material.emittance <= 0.0f)
        {
            continue;
        }
        DeviceLight light;
        light.boundsMin = glm::vec3(FLT_MAX);
        light.boundsMax = glm::vec3(-FLT_MAX);
        for (int corner = 0; corner < 8; corner++)
        {
            glm::vec3 unit((corner & 1) ? 0.5f : -0.5f, (corner & 2) ? 0.5f : -0.5f,
                (corner & 4) ? 0.5f : -0.5f);
            glm::vec3 world = multiplyMV(geom.transform, glm::vec4(unit, 1.0f));
            light.boundsMin = glm::min(light.boundsMin, world);
            light.boundsMax = glm::max(light.boundsMax, world);
        }
        glm::vec3 size = light.boundsMax - light.boundsMin;
        light.surfaceArea = 2.0f * (size.x * size.y + size.x * size.z + size.y * size.z);
        light.emission = material.color * material.emittance;
        light.geomIndex = (int)(&geom - scene->geoms.data());
        // A sphere is sampled over its tangent cone, so the world radius is half
        // the scale; a non uniform scale is an ellipsoid and falls back to the
        // box path with a warning instead of being sampled wrongly.
        light.shape = LIGHT_BOX;
        light.radius = 0.0f;
        light.center = 0.5f * (light.boundsMin + light.boundsMax);
        if (geom.type == SPHERE)
        {
            const glm::vec3& s = geom.scale;
            const float biggest = glm::max(s.x, glm::max(s.y, s.z));
            const float smallest = glm::min(s.x, glm::min(s.y, s.z));
            if (biggest - smallest <= 1e-4f * glm::max(biggest, 1e-6f))
            {
                light.shape = LIGHT_SPHERE;
                light.radius = 0.5f * biggest;
                light.surfaceArea = 4.0f * PI * light.radius * light.radius;
            }
            else
            {
                printf("[lights] warning: sphere emitter %d is scaled (%g %g %g), i.e. an "
                    "ellipsoid; sampling it as a box\n", light.geomIndex, s.x, s.y, s.z);
            }
        }
        if (light.shape == LIGHT_SPHERE)
        {
            printf("[lights] emitter %d is a sphere of radius %.4g at (%.3g %.3g %.3g)\n",
                light.geomIndex, light.radius, light.center.x, light.center.y, light.center.z);
        }
        h_totalLightArea += light.surfaceArea;
        lights.push_back(light);
    }

    h_lightCount = (int)lights.size();
    h_lightInfo = lights;   // host side copy, for the ledger printout
    if (h_lightCount > 0)
    {
        cudaMalloc(&dev_lights, h_lightCount * sizeof(DeviceLight));
        cudaMemcpy(dev_lights, lights.data(), h_lightCount * sizeof(DeviceLight),
            cudaMemcpyHostToDevice);
        printf("[lights] %d emitter(s), %.2f units^2 of emitting surface\n",
            h_lightCount, h_totalLightArea);
    }
#if DIRECT_LIGHT_STATS
    {
        // One row per area light, plus one for the distant light (not a geometry,
        // so it gets the last row instead of an entry in the light list).
        cudaMalloc(&dev_lightLedger, (h_lightCount + 1) * sizeof(LightLedger));
        cudaMemset(dev_lightLedger, 0, (h_lightCount + 1) * sizeof(LightLedger));
        cudaMalloc(&dev_lightFaceLedger,
            (size_t)glm::max(h_lightCount, 1) * LIGHT_SAMPLE_FACES * sizeof(LightLedger));
        cudaMemset(dev_lightFaceLedger, 0,
            (size_t)glm::max(h_lightCount, 1) * LIGHT_SAMPLE_FACES * sizeof(LightLedger));
        cudaMalloc(&dev_foldedLedger, sizeof(LightLedger));
        cudaMemset(dev_foldedLedger, 0, sizeof(LightLedger));
        cudaMalloc(&dev_foldedFaceLedger, 2 * LIGHT_SAMPLE_FACES * sizeof(LightLedger));
        cudaMemset(dev_foldedFaceLedger, 0, 2 * LIGHT_SAMPLE_FACES * sizeof(LightLedger));
        cudaMalloc(&dev_foldedAboveLedger, sizeof(LightLedger));
        cudaMemset(dev_foldedAboveLedger, 0, sizeof(LightLedger));
    }
#endif

    cudaMalloc(&dev_rrDecisions, sizeof(unsigned long long));
    cudaMemset(dev_rrDecisions, 0, sizeof(unsigned long long));
    cudaMalloc(&dev_rrKills, sizeof(unsigned long long));
    cudaMemset(dev_rrKills, 0, sizeof(unsigned long long));
    cudaMalloc(&dev_rrSurvivalMilli, sizeof(unsigned long long));
    cudaMemset(dev_rrSurvivalMilli, 0, sizeof(unsigned long long));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_pathsAlt);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_intersectionsAlt);
    cudaFree(dev_alive);
    cudaFree(dev_scanIndices);
    cudaFree(dev_bvhNodes);
    cudaFree(dev_bvhPrimitiveIds);
    cudaFree(dev_firstNormal);
    cudaFree(dev_firstAlbedo);
    dev_firstNormal = NULL;
    dev_firstAlbedo = NULL;
    dev_bvhNodes = NULL;
    dev_bvhPrimitiveIds = NULL;
    h_bvhActive = false;
    h_bvhNodeCount = 0;
    cudaFree(dev_materialKeys);
    cudaFree(dev_bucketCounts);
    cudaFree(dev_bucketOffsets);
    cudaFree(dev_bucketCursors);
    dev_materialKeys = NULL;
    dev_bucketCounts = NULL;
    dev_bucketOffsets = NULL;
    dev_bucketCursors = NULL;
    stageTrace.destroy();
    stageSort.destroy();
    stageShade.destroy();
    cudaFree(dev_sdfSteps);
    cudaFree(dev_sdfHistogram);
    dev_sdfSteps = NULL;
    dev_sdfHistogram = NULL;
    cudaFree(dev_rrDecisions);
    cudaFree(dev_rrKills);
    cudaFree(dev_rrSurvivalMilli);
    cudaFree(dev_lights);   // no-op if the scene has no emitters
    cudaFree(dev_lightLedger);
    cudaFree(dev_lightFaceLedger);
    cudaFree(dev_foldedLedger);
    cudaFree(dev_foldedAboveLedger);
    cudaFree(dev_foldedFaceLedger);
    dev_rrDecisions = NULL;
    dev_rrKills = NULL;
    dev_rrSurvivalMilli = NULL;
    dev_lights = NULL;
    dev_lightLedger = NULL;
    dev_lightFaceLedger = NULL;
    dev_foldedLedger = NULL;
    dev_foldedAboveLedger = NULL;
    dev_foldedFaceLedger = NULL;
    h_lightInfo.clear();
    releaseCheckpointStaging();

    checkCUDAError("pathtraceFree");
}

// ---------------------------------------------------------------------------
// Restartable rendering - public entry points
// ---------------------------------------------------------------------------

void pathtraceFetchImage(Scene* scene)
{
    if (dev_image == NULL)
    {
        return;
    }
    const int pixelcount = scene->state.camera.resolution.x * scene->state.camera.resolution.y;
    cudaMemcpy(scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("fetch image");
}

void pathtraceFetchDenoiseGuides(Scene* scene)
{
    if (dev_firstNormal == NULL || dev_firstAlbedo == NULL)
    {
        return;
    }
    const int pixelcount = scene->state.camera.resolution.x * scene->state.camera.resolution.y;
    scene->state.normalImage.resize(pixelcount);
    scene->state.albedoImage.resize(pixelcount);
    cudaMemcpy(scene->state.normalImage.data(), dev_firstNormal,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(scene->state.albedoImage.data(), dev_firstAlbedo,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("fetch denoiser guides");
}

bool pathtraceSaveCheckpoint(Scene* scene, int iterationsDone)
{
#if RESTARTABLE
    // CHECKPOINT 0 in the scene means do no restart
    if (scene->state.checkpointInterval <= 0.0f || dev_image == NULL || iterationsDone <= 0)
    {
        return false;
    }

    const Camera& cam = scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const size_t bytes = (size_t)pixelcount * sizeof(glm::vec3);
    if (!ensureCheckpointStaging(bytes))
    {
        printf("[checkpoint] could not allocate a %.1f MB staging buffer\n", bytes / 1048576.0);
        return false;
    }

    const double copyMs = downloadCheckpointImage(bytes);

    CheckpointHeader header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, CHECKPOINT_MAGIC, sizeof(header.magic));
    header.sceneHash = checkpointSceneHash(scene);
    header.resolutionX = cam.resolution.x;
    header.resolutionY = cam.resolution.y;
    header.traceDepth = scene->state.traceDepth;
    header.iterations = iterationsDone;
    header.headerBytes = (int)sizeof(CheckpointHeader);

    const std::string path = checkpointPath(scene);
    const std::string tempPath = path + ".tmp";

    auto start = std::chrono::steady_clock::now();
    FILE* file = fopen(tempPath.c_str(), "wb");
    if (file == NULL)
    {
        printf("[checkpoint] cannot write %s\n", tempPath.c_str());
        return false;
    }
    const bool wrote = fwrite(&header, sizeof(header), 1, file) == 1
        && fwrite(h_checkpointStaging, 1, bytes, file) == bytes;
    fclose(file);

    // Swap the finished file in only after it is complete on disk
    std::error_code ec;
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tempPath, path, ec);
    auto end = std::chrono::steady_clock::now();

    if (!wrote || ec)
    {
        printf("[checkpoint] failed to write %s\n", path.c_str());
        return false;
    }

    h_checkpointSaves++;
    h_checkpointCopyMs += copyMs;
    h_checkpointWriteMs += std::chrono::duration<double, std::milli>(end - start).count();
    h_checkpointBytesWritten += (long long)(sizeof(header) + bytes);
    return true;
#else
    (void)scene;
    (void)iterationsDone;
    return false;
#endif
}

bool pathtraceLoadCheckpoint(Scene* scene, int* iterationsDone)
{
#if RESTARTABLE
    if (dev_image == NULL) { return false; }

    const std::string path = checkpointPath(scene);
    FILE* file = fopen(path.c_str(), "rb");
    if (file == NULL) { return false; }

    CheckpointHeader header;
    if (fread(&header, sizeof(header), 1, file) != 1)
    {
        fclose(file);
        return false;
    }

    const Camera& cam = scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const size_t bytes = (size_t)pixelcount * sizeof(glm::vec3);

    const bool matches = memcmp(header.magic, CHECKPOINT_MAGIC, sizeof(header.magic)) == 0
        && header.headerBytes == (int)sizeof(CheckpointHeader)
        && header.sceneHash == checkpointSceneHash(scene)
        && header.resolutionX == cam.resolution.x
        && header.resolutionY == cam.resolution.y
        && header.traceDepth == scene->state.traceDepth
        && header.iterations > 0;
    if (!matches)
    {
        fclose(file);
        printf("[checkpoint] %s belongs to a different scene, starting from scratch\n",
            path.c_str());
        return false;
    }

    if (!ensureCheckpointStaging(bytes))
    {
        fclose(file);
        return false;
    }
    // Time the whole restore path: read the payload into the pinned buffer and
    // upload it to the device in one DMA.
    auto start = std::chrono::steady_clock::now();
    if (fread(h_checkpointStaging, 1, bytes, file) != bytes)
    {
        fclose(file);
        printf("[checkpoint] %s is truncated, starting from scratch\n", path.c_str());
        return false;
    }
    fclose(file);

    cudaMemcpy(dev_image, h_checkpointStaging, bytes, cudaMemcpyHostToDevice);
    checkCUDAError("checkpoint upload");
    auto end = std::chrono::steady_clock::now();
    h_checkpointLoadMs += std::chrono::duration<double, std::milli>(end - start).count();

    h_checkpointLoads++;
    if (iterationsDone != NULL)
    {
        *iterationsDone = header.iterations;
    }
    printf("[checkpoint] resumed %s at %d samples\n", path.c_str(), header.iterations);
    return true;
#else
    (void)scene;
    (void)iterationsDone;
    return false;
#endif
}

void pathtraceDeleteCheckpoint(Scene* scene)
{
    std::error_code ec;
    std::filesystem::remove(checkpointPath(scene), ec);
}

void printCheckpointStats()
{
#if RESTARTABLE
    if (h_checkpointSaves == 0 && h_checkpointLoads == 0)
    {
        return;
    }
    if (h_checkpointSaves > 0)
    {
        const double meanCopyMs = h_checkpointCopyMs / h_checkpointSaves;
        const double meanWriteMs = h_checkpointWriteMs / h_checkpointSaves;
        const double payloadMb = (double)(h_checkpointBytesWritten / h_checkpointSaves)
            / 1048576.0;
        printf("[checkpoint] %d saves, %d load(s), %.1f MB written; per save: %.2f ms "
            "device->host (%.2f GB/s) + %.2f ms to disk (pinned staging = %d)\n",
            h_checkpointSaves, h_checkpointLoads,
            (double)h_checkpointBytesWritten / 1048576.0,
            meanCopyMs,
            meanCopyMs > 0.0 ? payloadMb / 1024.0 / (meanCopyMs / 1000.0) : 0.0,
            meanWriteMs, CHECKPOINT_PINNED_MEMORY);
    }
    if (h_checkpointLoads > 0)
    {
        printf("[checkpoint] resume cost: %.2f ms to read + upload\n",
            h_checkpointLoadMs / h_checkpointLoads);
    }
#endif
}

/**
 * Generate the first bounce: one camera ray per pixel, jittered inside the pixel
 * area (STOCHASTIC_AA) and, when the aperture is open, started on the lens disk.
 */
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // Integer (x, y) addresses the *corner* of a pixel: at x = 0 the offset
        // below is exactly -resolution.x * 0.5 pixel lengths, the left edge of
        // the view, so the pixel area is [x, x + 1) x [y, y + 1).
        float sampleX = (float)x;
        float sampleY = (float)y;

        // Depth tag -1 gives the camera ray its own RNG stream (bounce `d` draws
        // with tag `d`); tag 0 would share numbers with the first scatter.
        constexpr int CAMERA_RNG_DEPTH = -1;
        thrust::default_random_engine rng =
            makeSeededRandomEngine(iter, index, CAMERA_RNG_DEPTH);
        // The camera ray owns dimensions 0-3 of this sample: two for the pixel
        // area, two for the lens. Everything after that belongs to the path.
        Sampler sampler((unsigned int)iter, (unsigned int)index, 0u);

#if STOCHASTIC_AA // jitter the ray inside the pixel area
        sampleX += nextRandom(rng, sampler);
        sampleY += nextRandom(rng, sampler);
#endif

        // Direction of the pinhole camera, i.e. the ray through the centre of
        // the thin lens.
        glm::vec3 pinholeDirection = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (sampleX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (sampleY - (float)cam.resolution.y * 0.5f)
        );

        // --- Physically based depth of field (thin lens model) -------------
        // Start the ray on the lens disk but aim it at the focal plane, so
        // everything at `focalDistance` stays sharp and an out of focus point at
        // distance z blurs by `aperture * |1/z - 1/focalDistance|`.
        if (cam.aperture > 0.0f && cam.focalDistance > 0.0f)
        {
            float radius = cam.aperture * sqrtf(nextRandom(rng, sampler));   // sqrt = uniform over the disk
            float angle = TWO_PI * nextRandom(rng, sampler);
            glm::vec3 lensPoint = cam.position
                + cam.right * (radius * cosf(angle))
                + cam.up * (radius * sinf(angle));
            glm::vec3 focalPoint = cam.position + pinholeDirection * cam.focalDistance;

            segment.ray.origin = lensPoint;
            segment.ray.direction = glm::normalize(focalPoint - lensPoint);
        }
        else
        {
            segment.ray.origin = cam.position;
            segment.ray.direction = pinholeDirection;
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        // No BSDF behind a camera ray: density 0 gives an emitter it hits the
        // full weight (the value also marks a delta sample).
        segment.lastPdf = 0.0f;
    }
}

__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    unsigned long long* sdfStepCounter,
    unsigned long long* sdfHistogram,
    const BvhNode* bvhNodes,
    const int* bvhPrimitiveIds)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        // Terminated paths (remainingBounces < 0) are skipped. Writing t = -1
        // also guarantees a stale intersection from an earlier bounce can never
        // be shaded a second time.
        if (pathSegment.remainingBounces < 0)
        {
            intersections[path_index].t = -1.0f;
            return;
        }

        // Accelerated path: the hierarchy holds the same primitives in the same
        // `geoms` array, it just decides which of them are worth testing.
        if (bvhNodes != NULL)
        {
            float t = 0.0f;
            glm::vec3 point, normal;
            int geomId = -1;
            bool outside = true;
            if (bvhClosestHit(bvhNodes, geoms, bvhPrimitiveIds, pathSegment.ray, sdfStepCounter,
                    sdfHistogram, t, point, normal, geomId, outside))
            {
                intersections[path_index].t = t;
                intersections[path_index].materialId = geoms[geomId].materialid;
                intersections[path_index].surfaceNormal = normal;
                intersections[path_index].geomId = geomId;
                intersections[path_index].outside = outside ? 1 : 0;
            }
            else
            {
                intersections[path_index].t = -1.0f;
                intersections[path_index].geomId = -1;
                intersections[path_index].outside = 1;
            }
            return;
        }

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            // One dispatch for every primitive kind (see intersectGeom): the
            // procedural shapes are sphere traced with an optional bounding
            // sphere clip, a mesh triangle is Moeller-Trumbore, the rest are
            // closed forms.
            t = intersectGeom(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside,
                sdfStepCounter, sdfHistogram);

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].geomId = -1;
            intersections[path_index].outside = 1;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].geomId = hit_geom_index;
            intersections[path_index].outside = outside ? 1 : 0;
        }
    }
}

// Shade one bounce of every live path segment: evaluate the BSDF (scatterRay),
// which updates the throughput and writes the next ray.
//
// remainingBounces > 0: may scatter; == 0: out of scattering budget but may
// still see an emitter; < 0: terminated - the stream compaction predicate.
__global__ void shadeMaterials(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
        Material* materials,
        Geom* geoms,
        int geomCount,
        DeviceLight* lights,
        int lightCount,
        float totalLightArea,
        Environment environment,
        DistantLight distantLight,
        LightLedger* lightLedger,
        LightLedger* lightFaceLedger,
        LightLedger* foldedLedger,
        LightLedger* foldedAboveLedger,
        LightLedger* foldedFaceLedger,
        unsigned long long* rrDecisions,
    unsigned long long* rrKills,
    unsigned long long* rrSurvivalMilli,
    glm::vec3* image,
    const BvhNode* bvhNodes,
    const int* bvhPrimitiveIds,
    glm::vec3* firstNormal,
    glm::vec3* firstAlbedo)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }

    PathSegment& pathSegment = pathSegments[idx];

    // Already terminated in an earlier bounce: nothing to do (this is the slot
    // stream compaction will eventually remove from the arrays entirely).
    if (pathSegment.remainingBounces < 0)
    {
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];

    // (1) Ray escaped the scene: it sees the environment light (the dome), which
    //     is the only way to find it - as background and as the light of an open
    //     scene
    if (intersection.t <= 0.0f)
    {
        const glm::vec3 rayDirection = glm::normalize(pathSegment.ray.direction);
        glm::vec3 radiance = environmentRadiance(environment, rayDirection);
        // A ray can walk into the sun disc too, so it gets the same MIS weight
        // the estimator uses; the dome has no second strategy and needs none
        if (insideDistantLight(distantLight, rayDirection))
        {
            float weight = 1.0f;
#if DIRECT_LIGHT_SAMPLING
            // With the estimator off the path sampler is the only strategy and
            // keeps everything, or the "path sampling only" build would be dark.
            const float sunChance = distantLightSelectionChance(distantLight, lightCount,
                totalLightArea);
            const float lightDensity = lightStrategyPdf(distantLight, sunChance, 0.0f, rayDirection);
            const float bsdfDensity = pathSegment.lastPdf;
            weight = (bsdfDensity > 0.0f) ? misWeight(bsdfDensity, lightDensity) : 1.0f;
#endif
            radiance += distantLight.radiance * weight;
        }
        if (radiance.x > 0.0f || radiance.y > 0.0f || radiance.z > 0.0f)
        {
            const glm::vec3 contribution = pathSegment.color * radiance;
            atomicAdd(&image[pathSegment.pixelIndex].x, contribution.x);
            atomicAdd(&image[pathSegment.pixelIndex].y, contribution.y);
            atomicAdd(&image[pathSegment.pixelIndex].z, contribution.z);
        }
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = -1;
        return;
    }

    Material material = materials[intersection.materialId];

    // Denoiser guides: written at the first bounce only, and overwritten every
    // iteration with the same value - they are a property of the geometry
    if (depth == 0)
    {
        firstNormal[pathSegment.pixelIndex] = intersection.surfaceNormal;
        firstAlbedo[pathSegment.pixelIndex] = material.color;
    }

    // (2) Ray hit an emitter: add the weighted emission to the pixel accumulator
    //     and terminate. Accumulating here, not in a final gather over the path
    //     array, is what lets compaction drop terminated paths.
    if (material.emittance > 0.0f)
    {
        glm::vec3 contribution = pathSegment.color * material.color * material.emittance;
        // Two strategies found this light, so MIS weights the path's estimate by
        // how well it does here; a delta sample (lastPdf == 0) keeps full weight.
        float weight = 1.0f;
#if DIRECT_LIGHT_SAMPLING
        if (pathSegment.lastPdf > 0.0f)
        {
            const glm::vec3 rayDirection = glm::normalize(pathSegment.ray.direction);
            const glm::vec3 hitPoint = pathSegment.ray.origin + intersection.t * rayDirection;
            // The light strategy's density for this direction, at this hit point.
            const float areaDensity = lightSamplePdfForGeom(lights, lightCount,
                pathSegment.ray.origin, hitPoint, intersection.surfaceNormal, intersection.geomId);
            const float sunChance = distantLightSelectionChance(distantLight, lightCount,
                totalLightArea);
            const float lightDensity = lightStrategyPdf(distantLight, sunChance, areaDensity,
                rayDirection);
            if (lightDensity > 0.0f)
            {
                weight = misWeight(pathSegment.lastPdf, lightDensity);
            }
        }
#endif
        contribution *= weight;
        atomicAdd(&image[pathSegment.pixelIndex].x, contribution.x);
        atomicAdd(&image[pathSegment.pixelIndex].y, contribution.y);
        atomicAdd(&image[pathSegment.pixelIndex].z, contribution.z);
#if DIRECT_LIGHT_SAMPLING && DIRECT_LIGHT_STATS
        {
            // Ledger: what this hit weighed, against a renderer without MIS.
            int hitLight = -1;
            for (int i = 0; i < lightCount; i++)
            {
                if (lights[i].geomIndex == intersection.geomId)
                {
                    hitLight = i;
                }
            }
            const double plainEnergy = (double)(pathSegment.color.x * material.color.x
                + pathSegment.color.y * material.color.y
                + pathSegment.color.z * material.color.z) * (double)material.emittance;
            ledgerRecordBsdfHit(*foldedLedger, plainEnergy, (double)weight);
            if (hitLight >= 0
                && pathSegment.ray.origin.y > lights[hitLight].boundsMin.y)
            {
                ledgerRecordBsdfHit(*foldedAboveLedger, plainEnergy, (double)weight);
            }
            if (hitLight >= 0)
            {
                const glm::vec3 n = intersection.surfaceNormal;
                int hitFace = 2;
                if (fabsf(n.x) > 0.5f) { hitFace = (n.x > 0.0f) ? 5 : 4; }
                else if (fabsf(n.z) > 0.5f) { hitFace = (n.z > 0.0f) ? 1 : 0; }
                else { hitFace = (n.y > 0.0f) ? 3 : 2; }
                const bool fromAbove = pathSegment.ray.origin.y > lights[hitLight].boundsMin.y;
                ledgerRecordBsdfHit(
                    foldedFaceLedger[hitFace + LIGHT_SAMPLE_FACES * (fromAbove ? 1 : 0)],
                    plainEnergy, (double)weight);
            }
        }
#endif
        pathSegment.remainingBounces = -1;
        return;
    }

    // (3) Ray hit a normal surface but the path has no scattering budget left,
    //     so the next ray would never be traced. Such a path cannot reach a
    //     light any more -> zero contribution and terminate.
    if (pathSegment.remainingBounces <= 0)
    {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = -1;
        return;
    }

    // World space point of the hit, needed both by the procedural texture below
    // and by the scatter.
    glm::vec3 intersect = pathSegment.ray.origin
        + intersection.t * glm::normalize(pathSegment.ray.direction);

    // Procedural texture: the pattern is defined in object space, so map the hit
    // point back there and modulate the albedo.
    if (material.textureType > 0 && intersection.geomId >= 0)
    {
        const Geom& geom = geoms[intersection.geomId];
        glm::vec3 objectSpace = multiplyMV(geom.inverseTransform, glm::vec4(intersect, 1.0f));
        material.color *= evaluateProceduralTexture(material.textureType,
            objectSpace * material.textureScale);
        material.specular.color = material.color;
    }

    // (4) Regular surface: evaluate the BSDF to update the throughput and
    //     generate the next ray. Seeding by (iteration, pixel, depth) makes a
    //     pixel's samples independent across iterations but reproducible.
    thrust::default_random_engine rng =
        makeSeededRandomEngine(iter, pathSegment.pixelIndex, depth);
    thrust::uniform_real_distribution<float> lightU01(0.0f, 1.0f);

    // Surface normal on the side the ray came from, as scatterRay sees it.
    glm::vec3 normal = intersection.surfaceNormal;
    if (glm::dot(normal, pathSegment.ray.direction) > 0.0f)
    {
        normal = -normal;
    }

    // --- Direct lighting (next event estimation) -------------------------
    // Sample a point on a light, evaluate the BSDF towards it and reject it if
    // the segment is blocked. MIS folds in the path's own emitter hits, so only
    // a vertex with a non-delta lobe can take part.
    const bool connectableVertex = bsdfHasNonDeltaLobe(material);
#if DIRECT_LIGHT_SAMPLING
    const bool areaLightsEnabled = (lightCount > 0 && totalLightArea > 0.0f);
    const float sunChance = distantLightSelectionChance(distantLight, lightCount, totalLightArea);
    if (connectableVertex && (areaLightsEnabled || distantLight.enabled != 0))
    {
        // Seven draws, always, so the dimension budget is sample independent.
        const float lightChoice = lightU01(rng);
        const float lu0 = lightU01(rng);
        const float lu1 = lightU01(rng);
        const float lu2 = lightU01(rng);
        const float lu3 = lightU01(rng);
        const float su0 = lightU01(rng);
        const float su1 = lightU01(rng);
        const glm::vec3 wo = -glm::normalize(pathSegment.ray.direction);

        if (lightChoice < sunChance)
        {
            // --- The distant light: aim at the disc ------------------------
            // Every sample lands on the disc, and an unblocked shadow ray also
            // says the direction missed any area light, so both parts share one
            // density.
            const glm::vec3 wi = sampleDistantLight(distantLight, su0, su1);
            const float cosSurface = glm::dot(normal, wi);
            const float lightPdf = sunChance * distantLightPdf(distantLight, wi);
            LightSampleOutcome outcome = LIGHT_REJECT_COS_SURFACE;
            double sampleEnergy = 0.0;
            if (cosSurface > 0.0f && lightPdf > 0.0f)
            {
                outcome = LIGHT_REJECT_OCCLUDED;
                const glm::vec3 shadowOrigin = intersect + normal * 1e-3f;
                if (!isOccluded(geoms, geomCount, shadowOrigin, wi, 1e30f, -1, bvhNodes,
                        bvhPrimitiveIds))
                {
                    float bsdfDensity = 0.0f;
                    const glm::vec3 f = bsdfEval(material, normal, wo, wi, bsdfDensity);
                    const float weight = misWeight(lightPdf, bsdfDensity);
                    const glm::vec3 contribution = pathSegment.color * f
                        * (cosSurface * weight / lightPdf) * distantLight.radiance;
                    sampleEnergy = (double)(contribution.x + contribution.y + contribution.z);
                    atomicAdd(&image[pathSegment.pixelIndex].x, contribution.x);
                    atomicAdd(&image[pathSegment.pixelIndex].y, contribution.y);
                    atomicAdd(&image[pathSegment.pixelIndex].z, contribution.z);
                    outcome = LIGHT_ACCEPTED;
                }
            }
#if DIRECT_LIGHT_STATS
            // The disc gets the extra ledger row (see pathtraceInit).
            ledgerRecord(lightLedger[lightCount], outcome, sampleEnergy);
#else
            (void)sampleEnergy;
#endif
        }
        else
        {
        // --- The area lights: sample a point on the surface of one ---------
        glm::vec3 lightPoint, lightNormal, lightEmission;
        int lightGeom = -1;
        int lightIndex = -1;
        int lightFace = -1;
        if (sampleLightSurface(lights, lightCount, intersect, lu0, lu1, lu2, lu3,
                lightPoint, lightNormal, lightEmission, lightGeom, lightIndex, lightFace))
        {
            glm::vec3 toLight = lightPoint - intersect;
            const float distance2 = glm::dot(toLight, toLight);
            const float distance = sqrtf(distance2);
            const glm::vec3 wi = toLight / distance;
            const float cosSurface = glm::dot(normal, wi);
            const float cosLight = glm::dot(lightNormal, -wi);
            // cosLight > 0 means the renderer can see this part of the light: a
            // ray towards an away-facing face enters the box through a nearer
            // one first
            LightSampleOutcome outcome = LIGHT_REJECT_COS_SURFACE;
            double sampleEnergy = 0.0;
            if (cosSurface > 0.0f)
            {
                outcome = LIGHT_REJECT_COS_LIGHT;
                if (cosLight > 0.0f)
                {
                    outcome = LIGHT_REJECT_OCCLUDED;
                    // Density for the whole strategy: the picked area light plus
                    // the sun, whose disc may cover this direction too
                    const float areaDensity = lightSampleSolidAnglePdf(lights[lightIndex],
                        lightCount, intersect, lightPoint, lightNormal);
                    const float lightPdf = lightStrategyPdf(distantLight, sunChance, areaDensity, wi);
                    const glm::vec3 shadowOrigin = intersect + normal * 1e-3f;
                    const int shadowSkip = LIGHT_SKIP_SELF_IN_SHADOW ? lightGeom : -1;
                    if (!isOccluded(geoms, geomCount, shadowOrigin, wi, distance - 1e-3f, shadowSkip,
                            bvhNodes, bvhPrimitiveIds))
                    {
                        // The light strategy chose the direction; the BSDF density
                        // is what the path would have produced for it - the other
                        // half of the MIS weight.
                        const glm::vec3 wo = -glm::normalize(pathSegment.ray.direction);
                        float bsdfDensity = 0.0f;
                        const glm::vec3 f = bsdfEval(material, normal, wo, wi, bsdfDensity);
                        const float weight = misWeight(lightPdf, bsdfDensity);
                        const glm::vec3 contribution = pathSegment.color * f
                            * (cosSurface * weight / lightPdf) * lightEmission;
                        sampleEnergy = (double)(contribution.x + contribution.y + contribution.z);
                        atomicAdd(&image[pathSegment.pixelIndex].x, contribution.x);
                        atomicAdd(&image[pathSegment.pixelIndex].y, contribution.y);
                        atomicAdd(&image[pathSegment.pixelIndex].z, contribution.z);
                        outcome = LIGHT_ACCEPTED;
                    }
                }
            }
#if DIRECT_LIGHT_STATS
            ledgerRecord(lightLedger[lightIndex], outcome, sampleEnergy);
            ledgerRecord(lightFaceLedger[lightIndex * LIGHT_SAMPLE_FACES + lightFace],
                outcome, sampleEnergy);
#else
            (void)sampleEnergy;
#endif
        }
        }
    }
#else
    (void)lightU01; (void)lightCount; (void)totalLightArea; (void)lights;
    (void)connectableVertex;
    (void)lightLedger; (void)lightFaceLedger; (void)foldedLedger;
    (void)foldedAboveLedger;
    (void)foldedFaceLedger;
#endif
    // The low discrepancy sequence is deliberately *not* used for the path
    // dimensions: measured, it made a 200 spp Cornell render worse (RMSE 34.6
    // against 11.6 for random draws) - the known failure of a Halton sequence in
    // an integral whose effective dimension changes from sample to sample. Only
    // the pixel and lens dimensions, handled in generateRayFromCamera, use it.

    scatterRay(pathSegment, intersect, intersection.surfaceNormal,
        intersection.outside != 0, material, rng);

#if RUSSIAN_ROULETTE
    // --- Russian roulette ------------------------------------------------
    // Survive with p = throughput and divide the survivors by p, which keeps the
    // estimator unbiased: E[killed ? 0 : throughput / p] = throughput. A dim path
    // then costs a full bounce only rarely.
    if (depth >= RR_MIN_DEPTH)
    {
        const float throughput = glm::max(pathSegment.color.x,
            glm::max(pathSegment.color.y, pathSegment.color.z));
        const float survival = glm::clamp(throughput,
            RR_MIN_SURVIVAL, 1.0f);
        if (rrDecisions != NULL)
        {
            atomicAdd(rrDecisions, 1ull);
            atomicAdd(rrSurvivalMilli, (unsigned long long)(survival * 1000.0f + 0.5f));
        }
        thrust::uniform_real_distribution<float> rrU01(0.0f, 1.0f);
        if (rrU01(rng) >= survival)
        {
            if (rrKills != NULL)
            {
                atomicAdd(rrKills, 1ull);
            }
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = -1;
            return;
        }
        pathSegment.color /= survival;
    }
#else
    (void)rrDecisions;
    (void)rrKills;
    (void)rrSurvivalMilli;
#endif

    // The scattered ray carried its own density into the next vertex (see
    // PathSegment::lastPdf), so there is nothing left to hand over here.
    pathSegment.remainingBounces--;
}

// ---------------------------------------------------------------------------
// Stream compaction
// ---------------------------------------------------------------------------

// Predicate to mark the surviving paths
__global__ void kernMarkAlivePaths(int n, int* alive, const PathSegment* paths)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        alive[index] = (paths[index].remainingBounces < 0) ? 0 : 1;
    }
}

// Scatter the surviving paths with the prefix-sum indices
__global__ void kernScatterAlivePaths(
    int n, PathSegment* odata, const PathSegment* idata, const int* alive, const int* indices)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n && alive[index])
    {
        odata[indices[index]] = idata[index];
    }
}

__global__ void kernScatterAliveIntersections(
    int n, ShadeableIntersection* odata, const ShadeableIntersection* idata,
    const int* alive, const int* indices)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n && alive[index])
    {
        odata[indices[index]] = idata[index];
    }
}

// Remove the terminated paths from the working arrays
static int compactPaths(
    int numPaths,
    PathSegment* paths,
    PathSegment* pathsOut,
    ShadeableIntersection* intersections,
    ShadeableIntersection* intersectionsOut)
{
    const int blockSize = 128;
    const dim3 blocks((numPaths + blockSize - 1) / blockSize);

    // predicate
    kernMarkAlivePaths<<<blocks, blockSize>>>(numPaths, dev_alive, paths);
    checkCUDAError("compact: mark alive paths");

    // scan input
    int m = nextPowerOfTwoAtLeast(numPaths + 1);
    cudaMemcpy(dev_scanIndices, dev_alive, numPaths * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemset(dev_scanIndices + numPaths, 0, (m - numPaths) * sizeof(int));
    checkCUDAError("compact: prepare scan input");

    StreamCompaction::Efficient::scanDevice(m, dev_scanIndices);
    checkCUDAError("compact: scan");

    // scatter both mirrored arrays with the same indices
    kernScatterAlivePaths<<<blocks, blockSize>>>(numPaths, pathsOut, paths, dev_alive, dev_scanIndices);
    kernScatterAliveIntersections<<<blocks, blockSize>>>(
        numPaths, intersectionsOut, intersections, dev_alive, dev_scanIndices);
    checkCUDAError("compact: scatter");

    // number of survivors = exclusive prefix sum at [numPaths]
    int numAlive = 0;
    cudaMemcpy(&numAlive, dev_scanIndices + numPaths, sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAError("compact: read survivor count");
    return numAlive;
}

// ---------------------------------------------------------------------------
// Sorting the paths by material (Part 1 core feature)
// ---------------------------------------------------------------------------

// Key of a path: 0 for a ray that hit nothing (those all take the same small
// "sees the environment" branch), otherwise 1 + material id.
#if SORT_BY_MATERIAL || MATERIAL_SORT_STATS
__global__ void kernMaterialSortKeys(
    int n, const ShadeableIntersection* intersections, int materialCount, int* keys)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        const int lastMaterial = materialCount > 0 ? materialCount - 1 : 0;
        const int materialId = glm::clamp(intersections[index].materialId, 0, lastMaterial);
        keys[index] = (intersections[index].t > 0.0f) ? 1 + materialId : 0;
    }
}
#endif

#if SORT_BY_MATERIAL
// One counter per bucket. The bucket count is the number of materials, so a
// plain atomic histogram over shared memory buys nothing at this size.
__global__ void kernCountMaterials(int n, const int* keys, int* counts)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        atomicAdd(&counts[keys[index]], 1);
    }
}

// Move every path into its material's run and take its intersection along. The
// atomic cursor makes the order *inside* a run arbitrary, which is fine: a path
// carries its own pixel index, RNG seed and throughput.
__global__ void kernScatterByMaterial(
    int n,
    PathSegment* paths,
    PathSegment* pathsOut,
    ShadeableIntersection* intersections,
    ShadeableIntersection* intersectionsOut,
    const int* keys,
    int* cursors)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        const int destination = atomicAdd(&cursors[keys[index]], 1);
        pathsOut[destination] = paths[index];
        intersectionsOut[destination] = intersections[index];
    }
}
#endif

#if MATERIAL_SORT_STATS
// The share of 32-lane warps that see a single material, plus the mean and worst
// number of materials per warp. Taken on the host from the same keys the sort
// uses, so it never touches the shading kernel it is explaining.
static void reportMaterialMix(int numPaths, const ShadeableIntersection* intersections,
    int materialCount, const char* tag)
{
    if (numPaths <= 0 || (int)h_materialKeys.size() < numPaths)
    {
        return;
    }

    const int blockSize = 128;
    const dim3 blocks((numPaths + blockSize - 1) / blockSize);
    kernMaterialSortKeys<<<blocks, blockSize>>>(numPaths, intersections, materialCount,
        dev_materialKeys);
    cudaMemcpy(h_materialKeys.data(), dev_materialKeys, numPaths * sizeof(int),
        cudaMemcpyDeviceToHost);
    checkCUDAError("sort stats: keys");

    std::vector<int> histogram(materialCount + 1, 0);
    for (int i = 0; i < numPaths; i++)
    {
        histogram[h_materialKeys[i]]++;
    }

    int warps = 0;
    int uniformWarps = 0;
    int worst = 0;
    long long distinctTotal = 0;
    for (int warp = 0; warp < numPaths; warp += 32)
    {
        const int end = glm::min(warp + 32, numPaths);
        int distinct = 0;
        unsigned long long seen = 0ull;
        for (int i = warp; i < end; i++)
        {
            const int key = h_materialKeys[i];
            if (key < 64)
            {
                const unsigned long long bit = 1ull << key;
                if (seen & bit)
                {
                    continue;
                }
                seen |= bit;
            }
            distinct++;
        }
        warps++;
        distinctTotal += distinct;
        worst = glm::max(worst, distinct);
        if (distinct == 1)
        {
            uniformWarps++;
        }
    }

    printf("[sort] %s: %d paths, %.1f%% of the %d warps see one material (mean %.2f, worst %d);"
        " hits", tag, numPaths, 100.0 * uniformWarps / warps, warps,
        (double)distinctTotal / warps, worst);
    for (int bucket = 0; bucket < (int)histogram.size(); bucket++)
    {
        printf(" m%d=%.1f%%", bucket, 100.0 * histogram[bucket] / numPaths);
    }
    printf("\n");
}
#endif

#if SORT_BY_MATERIAL
/** Reorder (paths, intersections) so that the paths which hit the same material
 *  sit next to each other: a warp holding a mix of them runs the shading branches
 *  one after the other with the other lanes masked off. Costs key, histogram,
 *  scan and scatter after every bounce; a permutation, so nothing is dropped. */
static int sortPathsByMaterial(
    int numPaths,
    PathSegment* paths,
    PathSegment* pathsOut,
    ShadeableIntersection* intersections,
    ShadeableIntersection* intersectionsOut,
    int materialCount,
    bool reportMix)
{
#if !MATERIAL_SORT_STATS
    (void)reportMix;
#endif
    const int bucketCount = materialCount + 1;      // bucket 0 = escaped the scene
    const int blockSize = 128;
    const dim3 blocks((numPaths + blockSize - 1) / blockSize);

    // 1) key
    kernMaterialSortKeys<<<blocks, blockSize>>>(numPaths, intersections, materialCount,
        dev_materialKeys);
    checkCUDAError("sort: material keys");

    // 2) histogram
    cudaMemset(dev_bucketCounts, 0, bucketCount * sizeof(int));
    kernCountMaterials<<<blocks, blockSize>>>(numPaths, dev_materialKeys, dev_bucketCounts);
    checkCUDAError("sort: histogram");

    // 3) exclusive prefix sum of the histogram -> the start of each run, on a
    //    power-of-two array with a zeroed tail, exactly as in compactPaths
    const int m = nextPowerOfTwoAtLeast(bucketCount + 1);
    cudaMemcpy(dev_bucketOffsets, dev_bucketCounts, bucketCount * sizeof(int),
        cudaMemcpyDeviceToDevice);
    cudaMemset(dev_bucketOffsets + bucketCount, 0, (m - bucketCount) * sizeof(int));
    StreamCompaction::Efficient::scanDevice(m, dev_bucketOffsets);
    checkCUDAError("sort: bucket scan");
    cudaMemcpy(dev_bucketCursors, dev_bucketOffsets, bucketCount * sizeof(int),
        cudaMemcpyDeviceToDevice);

    // 4) scatter
    kernScatterByMaterial<<<blocks, blockSize>>>(numPaths, paths, pathsOut, intersections,
        intersectionsOut, dev_materialKeys, dev_bucketCursors);
    checkCUDAError("sort: scatter");

#if MATERIAL_SORT_STATS
    // Keys again, this time of the sorted intersections, so the two reports are
    // the same measurement on either side of the permutation.
    if (reportMix)
    {
        reportMaterialMix(numPaths, intersectionsOut, materialCount, "after the sort");
    }
#endif

    return numPaths;
}
#endif

/** One iteration: trace, sort, shade, compact - then add the result to the image. */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    // Segments traced per sample: the camera ray plus up to `traceDepth`
    // scattering events; the last one exists only to hit an emitter
    const int maxSegments = traceDepth + 1;

    int depth = 0;
    int num_paths = pixelcount;

    // Working arrays: with compaction the two pairs ping-pong, so the survivors
    // scattered into the "other" arrays become the next bounce's input
    PathSegment* paths = dev_paths;
    PathSegment* pathsOther = dev_pathsAlt;
    ShadeableIntersection* intersections = dev_intersections;
    ShadeableIntersection* intersectionsOther = dev_intersectionsAlt;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    // The loop stops when the segment budget is used up, or - with stream
    // compaction - as soon as every path of this iteration has terminated.
    while (depth < maxSegments && num_paths > 0)
    {
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        // tracing
        stageTrace.begin();
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            paths,
            dev_geoms,
            hst_scene->geoms.size(),
            intersections,
            dev_sdfSteps,
            dev_sdfHistogram,
            h_bvhActive ? dev_bvhNodes : NULL,
            h_bvhActive ? dev_bvhPrimitiveIds : NULL
        );
        stageTrace.end();
        checkCUDAError("trace one bounce");

        // --- Sorting stage ---
        // Reorder the pairs into one run per material so that the shading
        // kernel's per-material branches are warp-uniform. SORT_BY_MATERIAL 0 is
        // the "shaded directly" side of the comparison the base code asked for.
#if MATERIAL_SORT_STATS || SORT_BY_MATERIAL
        // How mixed the warps are as the paths stand, measured on bounces 1 and 6
        // of the first iteration
        const bool reportMix = MATERIAL_SORT_STATS && iter == 1
            && (depth == 0 || depth == 5);
#endif
#if MATERIAL_SORT_STATS
        if (reportMix)
        {
            char tag[32];
            snprintf(tag, sizeof(tag), "bounce %d, before", depth + 1);
            reportMaterialMix(num_paths, intersections, (int)hst_scene->materials.size(), tag);
        }
#endif
#if SORT_BY_MATERIAL
        stageSort.begin();
        sortPathsByMaterial(num_paths, paths, pathsOther, intersections, intersectionsOther,
            (int)hst_scene->materials.size(), reportMix);
        stageSort.end();
        {
            PathSegment* tmpPaths = paths;
            paths = pathsOther;
            pathsOther = tmpPaths;
            ShadeableIntersection* tmpIntersections = intersections;
            intersections = intersectionsOther;
            intersectionsOther = tmpIntersections;
        }
#endif

        // --- Shading Stage ---
        // Evaluate the BSDF, accumulate emitter hits into dev_image and generate
        // the next ray of every still-living path
        stageShade.begin();
        shadeMaterials<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            intersections,
            paths,
            dev_materials,
            dev_geoms,
            (int)hst_scene->geoms.size(),
            dev_lights,
            h_lightCount,
            h_totalLightArea,
            hst_scene->state.environment,
            hst_scene->state.distantLight,
            dev_lightLedger,
            dev_lightFaceLedger,
            dev_foldedLedger,
            dev_foldedAboveLedger,
            dev_foldedFaceLedger,
            dev_rrDecisions,
            dev_rrKills,
            dev_rrSurvivalMilli,
            dev_image,
            h_bvhActive ? dev_bvhNodes : NULL,
            h_bvhActive ? dev_bvhPrimitiveIds : NULL,
            dev_firstNormal,
            dev_firstAlbedo
        );
        stageShade.end();
        checkCUDAError("shade one bounce");

        depth++;

#if STREAM_COMPACTION // Drop the terminated paths to reduce the number of active threads
        num_paths = compactPaths(num_paths, paths, pathsOther, intersections, intersectionsOther);
        {
            PathSegment* tmpPaths = paths;
            paths = pathsOther;
            pathsOther = tmpPaths;
            ShadeableIntersection* tmpIntersections = intersections;
            intersections = intersectionsOther;
            intersectionsOther = tmpIntersections;
        }
#endif

		// --- Bounce profile statistics ---
        if (guiData != NULL) { guiData->TracedDepth = depth; }
        if (depth <= MAX_PROFILE_DEPTH) { h_bounceAlive[depth - 1] += num_paths; }
    }

	// --- Iteration statistics ---
    h_profileIters++;
    if (iter == 1)
    {
        printBounceProfile("iteration 1 - paths processed after each bounce:",
            h_bounceAlive, maxSegments, 1, pixelcount);
    }
    if (iter >= hst_scene->state.iterations)
    {
        printBounceProfile("average over all iterations - paths processed after each bounce:",
            h_bounceAlive, maxSegments, h_profileIters, pixelcount);
        printSdfStats(pixelcount);
        printRussianRouletteStats(pixelcount);
        printDirectLightStats(pixelcount);
        printStageTimings();
    }

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    checkCUDAError("pathtrace");
}
