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
#include "checkpoint.h"
#include "stats.h"
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

// Double buffers used by stream compaction
static PathSegment* dev_pathsAlt = NULL;
static ShadeableIntersection* dev_intersectionsAlt = NULL;
static int* dev_alive = NULL;          // predicate: 1 = keep, 0 = terminated
static int* dev_scanIndices = NULL;    // exclusive prefix sum of the predicate
static int h_scanCapacity = 0;         // elements allocated for dev_scanIndices

// Buffers for the material sort
static int* dev_materialKeys = NULL; 
static int* dev_bucketCounts = NULL;  
static int* dev_bucketOffsets = NULL; 
static int* dev_bucketCursors = NULL;  
static int h_bucketCapacity = 0;      
#if MATERIAL_SORT_STATS
static std::vector<int> h_materialKeys;  
#endif

// The BVH acceleration structure
static BvhNode* dev_bvhNodes = NULL;
static int* dev_bvhPrimitiveIds = NULL;
static int h_bvhNodeCount = 0;
static bool h_bvhActive = false;

// Denoiser guides
static glm::vec3* dev_firstNormal = NULL;
static glm::vec3* dev_firstAlbedo = NULL;

// Direct lighting instrumentation
static DeviceLight* dev_lights = NULL;
static std::vector<DeviceLight> h_lightInfo;   // same list, kept for the printout
static int h_lightCount = 0;
static float h_totalLightArea = 0.0f;

static int nextPowerOfTwoAtLeast(int n)
{
    int m = 1;
    while (m < n)
    {
        m <<= 1;
    }
    return m;
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

    // Buffers for the stream compaction
    h_scanCapacity = nextPowerOfTwoAtLeast(pixelcount + 1);   // +1 so the scan also yields the total
    cudaMalloc(&dev_pathsAlt, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersectionsAlt, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_alive, pixelcount * sizeof(int));
    cudaMalloc(&dev_scanIndices, h_scanCapacity * sizeof(int));

    // Buffers for the material sort
    h_bucketCapacity = nextPowerOfTwoAtLeast((int)scene->materials.size() + 2);
    cudaMalloc(&dev_materialKeys, pixelcount * sizeof(int));
    cudaMalloc(&dev_bucketCounts, h_bucketCapacity * sizeof(int));
    cudaMalloc(&dev_bucketOffsets, h_bucketCapacity * sizeof(int));
    cudaMalloc(&dev_bucketCursors, h_bucketCapacity * sizeof(int));
#if MATERIAL_SORT_STATS
    h_materialKeys.assign(pixelcount, 0);
#endif

    // Direct lighting: build the light list
    std::vector<DeviceLight> lights;
    h_totalLightArea = 0.0f;
    for (const Geom& geom : scene->geoms)
    {
        const Material& material = scene->materials[geom.materialid];
        if (material.emittance <= 0.0f) { continue; }

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
    // The instrumentation owns its counters; it needs the light count for the ledger.
    statsInit(h_lightCount);

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
    cudaFree(dev_lights);   // no-op if the scene has no emitters
    dev_lights = NULL;
    statsFree();
    h_lightInfo.clear();
    checkpointRelease();

    checkCUDAError("pathtraceFree");
}

// ---------------------------------------------------------------------------
// Image fetch and the restartable rendering entry points
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

// The checkpoint module owns the file format and the staging buffer; the
// accumulation buffer belongs to the renderer, so these entry points stay here
// and hand it over.
bool pathtraceSaveCheckpoint(Scene* scene, int iterationsDone)
{
    return checkpointSave(scene, dev_image, iterationsDone);
}

bool pathtraceLoadCheckpoint(Scene* scene, int* iterationsDone)
{
    return checkpointLoad(scene, dev_image, iterationsDone);
}

void pathtraceDeleteCheckpoint(Scene* scene)
{
    checkpointDelete(scene);
}

void printCheckpointStats()
{
    checkpointPrintStats();
}

// Generate the first bounce ray from the camera for each pixel
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        float sampleX = (float)x;
        float sampleY = (float)y;

        constexpr int CAMERA_RNG_DEPTH = -1;
        thrust::default_random_engine rng =
            makeSeededRandomEngine(iter, index, CAMERA_RNG_DEPTH);

        Sampler sampler((unsigned int)iter, (unsigned int)index, 0u);

#if STOCHASTIC_AA // jitter the ray inside the pixel area
        sampleX += nextRandom(rng, sampler);
        sampleY += nextRandom(rng, sampler);
#endif

        // Direction of the pinhole camera
        glm::vec3 pinholeDirection = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (sampleX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (sampleY - (float)cam.resolution.y * 0.5f)
        );

        // Physically based depth of field
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
        // No BSDF behind a camera ray
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

        // Terminated paths are skipped
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

// Shade one bounce of every live path segment and evaluate the BSDF
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
    if (idx >= num_paths) { return; }

    PathSegment& pathSegment = pathSegments[idx];

    // Already terminated in an earlier bounce: nothing to do
    if (pathSegment.remainingBounces < 0) { return; }

    ShadeableIntersection intersection = shadeableIntersections[idx];

    // (1) Ray escaped the scene: it sees the environment light
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
			// The light strategy's density for this direction, at this hit point
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

    // Denoiser guides
    if (depth == 0)
    {
        firstNormal[pathSegment.pixelIndex] = intersection.surfaceNormal;
        firstAlbedo[pathSegment.pixelIndex] = material.color;
    }

    // (2) Ray hit an emitter: add the weighted emission to the pixel accumulator and terminate
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

            // The light strategy's density for this direction, at this hit point
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

    // (3) Ray hit a normal surface but the path has no scattering budget left, so the next ray would never be traced
    if (pathSegment.remainingBounces <= 0)
    {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = -1;
        return;
    }

    // World space point of the hit
    glm::vec3 intersect = pathSegment.ray.origin
        + intersection.t * glm::normalize(pathSegment.ray.direction);

    // Procedural texture
    if (material.textureType > 0 && intersection.geomId >= 0)
    {
        const Geom& geom = geoms[intersection.geomId];
        glm::vec3 objectSpace = multiplyMV(geom.inverseTransform, glm::vec4(intersect, 1.0f));
        material.color *= evaluateProceduralTexture(material.textureType,
            objectSpace * material.textureScale);
        material.specular.color = material.color;
    }

    // (4) Regular surface: evaluate the BSDF to update the throughput and generate the next ray
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, pathSegment.pixelIndex, depth);
    thrust::uniform_real_distribution<float> lightU01(0.0f, 1.0f);

    // Surface normal on the side the ray came from, as scatterRay sees it.
    glm::vec3 normal = intersection.surfaceNormal;
    if (glm::dot(normal, pathSegment.ray.direction) > 0.0f)
    {
        normal = -normal;
    }

    // Direct lighting  
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
			// Sample the sun disc, evaluate the BSDF towards it and reject it if the segment is blocked
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
			// Sample an area light, evaluate the BSDF towards it and reject it if the segment is blocked
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

            LightSampleOutcome outcome = LIGHT_REJECT_COS_SURFACE;
            double sampleEnergy = 0.0;
			if (cosSurface > 0.0f) // The BSDF is zero if the light is behind the surface
            {
                outcome = LIGHT_REJECT_COS_LIGHT;
                if (cosLight > 0.0f)
                {
                    outcome = LIGHT_REJECT_OCCLUDED;
                    // Density for the whole strategy
                    const float areaDensity = lightSampleSolidAnglePdf(lights[lightIndex],
                        lightCount, intersect, lightPoint, lightNormal);
                    const float lightPdf = lightStrategyPdf(distantLight, sunChance, areaDensity, wi);
                    const glm::vec3 shadowOrigin = intersect + normal * 1e-3f;
                    const int shadowSkip = LIGHT_SKIP_SELF_IN_SHADOW ? lightGeom : -1;
                    if (!isOccluded(geoms, geomCount, shadowOrigin, wi, distance - 1e-3f, shadowSkip,
                            bvhNodes, bvhPrimitiveIds))
                    {
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

	// Scatter the ray into a new direction, update the throughput and the last PDF for the next vertex
    scatterRay(pathSegment, intersect, intersection.surfaceNormal,
        intersection.outside != 0, material, rng);

#if RUSSIAN_ROULETTE
	// Russian roulette: terminate low-throughput paths with a probability
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

    // The scattered ray carried its own density into the next vertex, 
    // so there is nothing left to hand over here.
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
// Sorting the paths by material
// ---------------------------------------------------------------------------

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

// Move every path into its material's run and take its intersection along
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
// Report the material mix in the path stream, before or after a sort
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
// Sort the paths by material so the shading kernel sees a run of paths with the same
// material: one BSDF branch per warp instead of one per lane
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

// One iteration: trace, sort, shade, compact, then add the result to the image
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

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    // Segments traced per sample: the camera ray plus up to `traceDepth`
    // scattering events; the last one exists only to hit an emitter
    const int maxSegments = traceDepth + 1;

    int depth = 0;
    int num_paths = pixelcount;

    // Working arrays: with compaction the two pairs ping-pong
    PathSegment* paths = dev_paths;
    PathSegment* pathsOther = dev_pathsAlt;
    ShadeableIntersection* intersections = dev_intersections;
    ShadeableIntersection* intersectionsOther = dev_intersectionsAlt;

    // PathSegment Tracing Stage
    // Shoot ray into scene, bounce between objects, push shading chunks

    // The loop stops when the segment budget is used up, or - with stream
    // compaction - as soon as every path of this iteration has terminated.
    while (depth < maxSegments && num_paths > 0)
    {
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        // tracing
        statsTrace().begin();
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            paths,
            dev_geoms,
            hst_scene->geoms.size(),
            intersections,
            statsSdfSteps(),
            statsSdfHistogram(),
            h_bvhActive ? dev_bvhNodes : NULL,
            h_bvhActive ? dev_bvhPrimitiveIds : NULL
        );
        statsTrace().end();
        checkCUDAError("trace one bounce");

        // Sorting stage 
        // Reorder the pairs into one run per material

#if MATERIAL_SORT_STATS || SORT_BY_MATERIAL
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
        statsSort().begin();
        sortPathsByMaterial(num_paths, paths, pathsOther, intersections, intersectionsOther,
            (int)hst_scene->materials.size(), reportMix);
        statsSort().end();
        {
            PathSegment* tmpPaths = paths;
            paths = pathsOther;
            pathsOther = tmpPaths;
            ShadeableIntersection* tmpIntersections = intersections;
            intersections = intersectionsOther;
            intersectionsOther = tmpIntersections;
        }
#endif

        // Shading Stage
        // Evaluate the BSDF, accumulate emitter hits into dev_image 
        // and generate the next ray of every still-living path
        statsShade().begin();
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
            statsLightLedger(),
            statsLightFaceLedger(),
            statsFoldedLedger(),
            statsFoldedAboveLedger(),
            statsFoldedFaceLedger(),
            statsRrDecisions(),
            statsRrKills(),
            statsRrSurvivalMilli(),
            dev_image,
            h_bvhActive ? dev_bvhNodes : NULL,
            h_bvhActive ? dev_bvhPrimitiveIds : NULL,
            dev_firstNormal,
            dev_firstAlbedo
        );
        statsShade().end();
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

        if (guiData != NULL) { guiData->TracedDepth = depth; }
        statsRecordBounce(depth, num_paths);
    }

    statsEndIteration(iter, hst_scene->state.iterations, maxSegments, pixelcount,
        h_lightCount, DIRECT_LIGHT_SAMPLING != 0);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    checkCUDAError("pathtrace");
}
