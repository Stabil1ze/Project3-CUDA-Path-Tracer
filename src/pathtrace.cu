#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "../stream_compaction/efficient.h"   // Project 2 work-efficient scan

#define ERRORCHECK 1

// Stochastic sampled antialiasing (Part 1 core feature): jitter the first ray
// of every pixel inside its pixel area. Set to 0 to reproduce the aliased
// "before" images for the write-up; 1 is the default.
#define STOCHASTIC_AA 1

// Stream compaction (Part 1 core feature): after every bounce, remove the
// terminated paths from the working arrays so that later bounces only launch
// threads for rays that are still alive. Set to 0 to measure the baseline that
// keeps dead rays in the arrays and just skips them.
#define STREAM_COMPACTION 1

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

// Per-bounce ray counts for the README analysis ("number of unterminated rays
// after each bounce"). Ray termination does not depend on whether compaction is
// enabled - the same rays die at the same bounce - so one profile describes
// both configurations; only the amount of work differs.
static const int MAX_PROFILE_DEPTH = 64;
static long long h_bounceAlive[MAX_PROFILE_DEPTH];
static long long h_firstIterAlive[MAX_PROFILE_DEPTH];
static long long h_profileIters = 0;

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

    // Buffers for the stream compaction stage (see compactPaths below).
    h_scanCapacity = nextPowerOfTwoAtLeast(pixelcount + 1);   // +1 so the scan also yields the total
    cudaMalloc(&dev_pathsAlt, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersectionsAlt, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_alive, pixelcount * sizeof(int));
    cudaMalloc(&dev_scanIndices, h_scanCapacity * sizeof(int));

    h_profileIters = 0;
    for (int i = 0; i < MAX_PROFILE_DEPTH; i++)
    {
        h_bounceAlive[i] = 0;
        h_firstIterAlive[i] = 0;
    }

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

    checkCUDAError("pathtraceFree");
}

/**
 * Generate PathSegments with rays from the camera through the screen into the
 * scene, which is the first bounce of rays.
 *
 * Antialiasing - the first ray of each pixel is jittered inside the pixel area
 *                (see STOCHASTIC_AA below).
 * motion blur  - jitter rays "in time"            (not implemented)
 * lens effect  - jitter ray origin positions based on a lens (not implemented)
 */
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // Grid coordinate of the sample. Note the convention of the camera
        // transform above: at x = 0 the offset is exactly -resolution.x * 0.5
        // times one pixel length, i.e. the left edge of the view. So integer
        // (x, y) addresses the *corner* of a pixel and the pixel area is
        // [x, x + 1) x [y, y + 1) - not [x - 0.5, x + 0.5).
        float sampleX = (float)x;
        float sampleY = (float)y;

#if STOCHASTIC_AA
        // A fresh jitter per iteration turns the per-pixel value into the
        // average radiance over the pixel area (a box filter) instead of a
        // point sample, which is what removes the stair-stepping on edges.
        // The offset must span the whole pixel area, i.e. [0, 1) here; using
        // [-0.5, 0.5) would sample half of the previous pixel and blur every
        // edge across its neighbours.
        //
        // Depth tag -1 gives the camera ray its own RNG stream: bounce `d`
        // draws with tag `d`, so reusing tag 0 here would make the sub-pixel
        // offset and the first scatter direction share random numbers.
        constexpr int CAMERA_RNG_DEPTH = -1;
        thrust::default_random_engine rng =
            makeSeededRandomEngine(iter, index, CAMERA_RNG_DEPTH);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
        sampleX += u01(rng);
        sampleY += u01(rng);
#endif

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (sampleX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (sampleY - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
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

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

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
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// Shade one bounce of every live path segment and generate the next ray by
// evaluating the BSDF (scatterRay). This is the kernel that replaces the
// original "fake" shader - it does a real BSDF evaluation and radiometric
// accumulation instead of pseudo-lighting.
//
// Path bookkeeping used here:
//   pathSegment.color            running throughput of the path (starts white)
//   pathSegment.remainingBounces > 0 : alive, may scatter
//                                == 0 : alive but out of scattering budget, may
//                                       still "see" an emitter
//                                 < 0 : terminated, ignored by every later pass
//                                       (this is the predicate stream compaction
//                                       will use later on)
//
// Every terminated path has a zero throughput unless it ended on an emitter, in
// which case the throughput *is* the radiance it picked up. That keeps
// finalGather a blind `image[pixelIndex] += color` over the whole array until
// stream compaction is wired in.
__global__ void shadeMaterials(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    glm::vec3* image)
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

    // (1) Ray escaped the scene. The background is black, so this path
    //     contributes nothing and is terminated.
    if (intersection.t <= 0.0f)
    {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = -1;
        return;
    }

    Material material = materials[intersection.materialId];

    // (2) Ray hit an emitter. This is where radiance enters the image: add the
    //     weighted emission straight into the pixel accumulator and terminate
    //     (an ideal emitter neither reflects nor transmits what hits it).
    //
    //     Accumulating here rather than in a final gather over the path array is
    //     what makes stream compaction legal: terminated paths are removed from
    //     the arrays, so no later pass may depend on them still being there.
    if (material.emittance > 0.0f)
    {
        glm::vec3 contribution = pathSegment.color * material.color * material.emittance;
        atomicAdd(&image[pathSegment.pixelIndex].x, contribution.x);
        atomicAdd(&image[pathSegment.pixelIndex].y, contribution.y);
        atomicAdd(&image[pathSegment.pixelIndex].z, contribution.z);
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

    // (4) Regular surface: evaluate the BSDF to update the throughput and
    //     generate the next ray. Seeding by (iteration, pixel, depth) keeps the
    //     samples of one pixel independent across iterations while still being
    //     reproducible.
    glm::vec3 intersect = pathSegment.ray.origin
        + intersection.t * glm::normalize(pathSegment.ray.direction);

    thrust::default_random_engine rng =
        makeSeededRandomEngine(iter, pathSegment.pixelIndex, depth);

    scatterRay(pathSegment, intersect, intersection.surfaceNormal, material, rng);

    pathSegment.remainingBounces--;
}

// Add the current iteration's output to the overall image
// NOTE: the original base code accumulated here with a "finalGather" pass over
// the path array. That is incompatible with stream compaction (terminated paths
// - which are exactly the ones that carry radiance - have already been removed
// from the array by then), so the contribution is now added by shadeMaterials
// at the moment a ray hits an emitter. `image` still holds the running sum of
// the samples, which is what sendImageToPBO / saveImage divide by `iter`.

// ---------------------------------------------------------------------------
// Stream compaction (Project 2 implementation, driven from here)
// ---------------------------------------------------------------------------

// Predicate: 1 = path may still do something (scatter, or still see an emitter
// because it is on its last segment), 0 = terminated and can be removed.
__global__ void kernMarkAlivePaths(int n, int* alive, const PathSegment* paths)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        alive[index] = (paths[index].remainingBounces < 0) ? 0 : 1;
    }
}

// Scatter the surviving paths with the prefix-sum indices computed from the
// predicate. PathSegments and ShadeableIntersections are mirrored arrays, so
// they are compacted with the same index map to stay in sync.
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

/**
 * Remove the terminated paths from the working arrays.
 *
 * map -> scan -> scatter:
 *   1. kernMarkAlivePaths       builds the 1/0 predicate
 *   2. StreamCompaction::Efficient::scanDevice  (Project 2 work-efficient
 *      Blelloch scan) turns it into exclusive prefix-sum indices
 *   3. kernScatterAlive*        moves the survivors into the destination arrays
 *
 * Returns the number of surviving paths.
 */
static int compactPaths(
    int numPaths,
    PathSegment* paths,
    PathSegment* pathsOut,
    ShadeableIntersection* intersections,
    ShadeableIntersection* intersectionsOut)
{
    const int blockSize = 128;
    const dim3 blocks((numPaths + blockSize - 1) / blockSize);

    // 1) predicate
    kernMarkAlivePaths<<<blocks, blockSize>>>(numPaths, dev_alive, paths);
    checkCUDAError("compact: mark alive paths");

    // 2) scan input. The scan runs in place on a power-of-two array with a
    //    zeroed tail (a requirement of the Project 2 scan), and the extra slot
    //    at [numPaths] ends up holding the total number of survivors.
    int m = nextPowerOfTwoAtLeast(numPaths + 1);
    if (m > h_scanCapacity)
    {
        m = h_scanCapacity;
    }
    cudaMemcpy(dev_scanIndices, dev_alive, numPaths * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemset(dev_scanIndices + numPaths, 0, (m - numPaths) * sizeof(int));
    checkCUDAError("compact: prepare scan input");

    StreamCompaction::Efficient::scanDevice(m, dev_scanIndices);
    checkCUDAError("compact: scan");

    // 3) scatter both mirrored arrays with the same indices
    kernScatterAlivePaths<<<blocks, blockSize>>>(numPaths, pathsOut, paths, dev_alive, dev_scanIndices);
    kernScatterAliveIntersections<<<blocks, blockSize>>>(
        numPaths, intersectionsOut, intersections, dev_alive, dev_scanIndices);
    checkCUDAError("compact: scatter");

    // 4) number of survivors = exclusive prefix sum at [numPaths]
    int numAlive = 0;
    cudaMemcpy(&numAlive, dev_scanIndices + numPaths, sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAError("compact: read survivor count");
    return numAlive;
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
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

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    // Segments traced per sample: the camera ray plus up to `traceDepth`
    // scattering events. The final segment can no longer scatter - it only
    // exists so that the path may still directly hit an emitter.
    const int maxSegments = traceDepth + 1;

    int depth = 0;
    int num_paths = pixelcount;

    // Working arrays. With stream compaction the two pairs ping-pong: the
    // survivors are scattered into the alternate arrays, which become the input
    // of the next bounce. Without compaction the pointers never move and dead
    // rays are simply skipped inside the kernels.
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
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            paths,
            dev_geoms,
            hst_scene->geoms.size(),
            intersections
        );
        checkCUDAError("trace one bounce");

        // --- Shading Stage ---
        // Evaluate the BSDF of the hit material, accumulate the contribution of
        // emitter hits into dev_image and generate the next ray for each
        // still-living path segment.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
        shadeMaterials<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            intersections,
            paths,
            dev_materials,
            dev_image
        );
        checkCUDAError("shade one bounce");

        depth++;

#if STREAM_COMPACTION
        // Drop the terminated paths so the next bounce only launches threads
        // for rays that can still reach a light.
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

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }

        // README instrumentation: rays still processed after this bounce. With
        // compaction this is the number of unterminated rays; without it, it
        // stays at the full pixel count even though most rays are already dead.
        if (depth <= MAX_PROFILE_DEPTH)
        {
            h_bounceAlive[depth - 1] += num_paths;
        }
    }

    // README instrumentation: the per-bounce profile of a single iteration, plus
    // the average over the whole render.
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
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
