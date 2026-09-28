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
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "bsdf.h"
#include "sampling.h"
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

// Instrumentation for the procedural shapes: count the sphere tracing steps of
// every SDF intersection test into a global counter plus a 16 bucket histogram,
// which are copied back and printed at the end of the render. The atomics cost
// a few percent, so set it to 0 when timing the scene.
#define SDF_STATS 1

// Restartable rendering: save the accumulated image plus the sample count to
// "<FILE>.ckpt" while a render is in progress, and pick it up again on the next
// start. CHECKPOINT_PINNED_MEMORY selects the staging buffer used to pull the
// accumulation buffer off the device - page-locked memory lets the driver DMA
// straight into it, pageable memory makes it bounce through a staging buffer of
// its own. Its "0" is the "before" side of the measurement in the README.
#define RESTARTABLE 1
#define CHECKPOINT_PINNED_MEMORY 1

// Russian roulette: from RR_MIN_DEPTH segments on, kill each path with a
// probability that grows as its throughput shrinks, and divide the survivors by
// the survival probability (which is what keeps the estimator unbiased). Set to 0
// to render every path until it escapes, hits an emitter or runs out of depth -
// the "before" side of the measurement in the README.
#define RUSSIAN_ROULETTE 1
#define RR_MIN_DEPTH 3              // segments; segment 1 is the camera ray
#define RR_MIN_SURVIVAL 0.05f       // never keep fewer than 5% of the paths

// Direct lighting (next event estimation): connect every diffuse vertex to a
// random point on an emissive object instead of waiting for a path to find one,
// and MIS down the emitter hits BSDF sampling produces for those same vertices.
// Set to 0 for the "before" side of the measurement.
//
// STATUS: on, and measured. The ledger below is what settled it - the energy the
// estimator delivers matches the energy of the emitter hits it replaced on every
// scene tried, all four checks within their error bars (README). It pays on a
// small light (55.69% -> 31.84% relative error at 200 spp, +41% time) and is a
// small loss on the course scene's 3 x 3 light, the case MIS exists for.
#define DIRECT_LIGHT_SAMPLING 1

// Sample ledger for the estimator above: per light and per face, samples drawn,
// accepted, why the rest were dropped, and the energy delivered against the BSDF
// emitter hits it replaced - an unbiasedness test that needs no reference image.
// A few atomics per light sample, hence the separate switch: off for timing.
#define DIRECT_LIGHT_STATS 1

// Is the sampled light skipped in its own shadow test? Yes: the kept samples sit
// on faces the shading point can see (cosLight > 0), and for a convex light the
// segment to such a sample touches nothing else. Leaving the light in costs 8.5
// points of acceptance and 35.8% of the energy, because boxIntersectionTest
// reports its t in the geometry's object space and comparing that against a world
// space distance is not meaningful.
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

// Per-bounce ray counts for the README analysis ("number of unterminated rays
// after each bounce"). Ray termination does not depend on whether compaction is
// enabled - the same rays die at the same bounce - so one profile describes
// both configurations; only the amount of work differs.
static const int MAX_PROFILE_DEPTH = 64;
static long long h_bounceAlive[MAX_PROFILE_DEPTH];
static long long h_profileIters = 0;

// Instrumentation for the procedural shapes (see SDF_STATS above): total sphere
// tracing steps and a histogram of the steps a single test needed. Both live on
// the device and are only touched by rays that test an SDF object.
static const int SDF_HISTOGRAM_BUCKETS = 16;
static const int SDF_STEPS_PER_BUCKET = 8;
static unsigned long long* dev_sdfSteps = NULL;
static unsigned long long* dev_sdfHistogram = NULL;
static unsigned long long h_sdfSteps = 0;
static unsigned long long h_sdfHistogram[SDF_HISTOGRAM_BUCKETS];

// Russian roulette instrumentation: how many paths reached the RR depth and how
// many of them were killed, plus the mean survival probability they were given.
// Only meaningful with RUSSIAN_ROULETTE 1.
static unsigned long long* dev_rrDecisions = NULL;
static unsigned long long* dev_rrKills = NULL;
// The survival probabilities are summed in fixed point (thousandths). Summing
// them into a single float does not work: the running total passes 2^24 after a
// few million decisions, and from there on every increment of ~0.7 is smaller
// than one ulp of the accumulator and is silently dropped.
static unsigned long long* dev_rrSurvivalMilli = NULL;
static unsigned long long h_rrDecisions = 0;
static unsigned long long h_rrKills = 0;
static unsigned long long h_rrSurvivalMilli = 0;

// --- Direct lighting: the light list -------------------------------------
// Every emissive geometry becomes one entry. The scenes only ever use emissive
// boxes, they are axis aligned (the lights have no rotation) and a box is
// uniform under area sampling, so a light is stored as its world space AABB plus
// its emitted radiance. Sampling is uniform over the *surface* of the box, which
// includes the thin sides - they are buried in the ceiling, but pretending they
// are invisible would be a silent energy leak.
struct DeviceLight
{
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    glm::vec3 emission;      // material colour * emittance
    float surfaceArea;
    int geomIndex;           // which geometry this light is (to skip it in shadow rays)
    // How the emitter is sampled: a box by area over its six faces, a sphere by
    // solid angle over its tangent cone (constant density on the visible cap).
    int shape;               // 0 = box, 1 = sphere
    float radius;            // sphere only
    glm::vec3 center;        // sphere only
};

enum LightShape
{
    LIGHT_BOX = 0,
    LIGHT_SPHERE = 1
};

static DeviceLight* dev_lights = NULL;
static std::vector<DeviceLight> h_lightInfo;   // same list, kept for the printout
static int h_lightCount = 0;
static float h_totalLightArea = 0.0f;

// --- Direct lighting: the sample ledger ----------------------------------
// Per light and per face: samples drawn, accepted, why the rest were dropped,
// and the energy the estimator delivered against the BSDF emitter hits it
// replaced - an unbiasedness test that needs no converged reference render.
enum LightSampleOutcome
{
    LIGHT_ACCEPTED = 0,
    LIGHT_REJECT_COS_SURFACE,   // sample is below the shading point's horizon
    LIGHT_REJECT_COS_LIGHT,     // shading point is behind the sampled light face
    LIGHT_REJECT_OCCLUDED       // something else is between the two
};

struct LightLedger
{
    unsigned long long samples;
    unsigned long long accepted;
    unsigned long long rejectCosSurface;
    unsigned long long rejectCosLight;
    unsigned long long occluded;
    // Accepted samples of (r + g + b) after the MIS weight, i.e. what went into
    // the image. The sum of squares gives the printout its error bar.
    double acceptedEnergy;
    double acceptedEnergySq;
    // inf / NaN contributions: counted, not summed - one would poison the rest.
    unsigned long long nonFiniteEnergy;
    // Emitter hits found by random walks: weighted energy, plain energy (the
    // reference side of the check) and the plain sum of squares.
    unsigned long long bsdfHits;
    double bsdfEnergy;
    double bsdfEnergyPlain;
    double bsdfEnergySq;
};

static const int LIGHT_SAMPLE_FACES = 6;

static LightLedger* dev_lightLedger = NULL;       // one row per light
static LightLedger* dev_lightFaceLedger = NULL;   // one row per light and face
static LightLedger* dev_foldedLedger = NULL;      // the BSDF hits the estimator replaced
static LightLedger* dev_foldedAboveLedger = NULL;  // ... the ones from above the light's bottom plane
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

// README instrumentation for the procedural shapes: how many sphere tracing
// steps the SDF tests needed on average, and how they are distributed. The
// spread is the interesting part on a GPU - every thread in a warp marches until
// its own ray leaves the bounding sphere, so the warps pay the maximum of 32
// different step counts.
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

// README instrumentation for Russian roulette: how often it fired and what it
// saved. The bounce profile above shows the same thing from the other side (the
// surviving path counts drop once RR is on).
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

// README instrumentation for direct lighting: the sample ledger. It answers the
// two questions the estimator has to answer about itself - which samples am I
// losing, and is what I deliver equal to what I replaced - without needing a
// converged reference render to compare against.
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

    // The estimator against the one it replaced: under MIS the direct light term
    // is the light strategy's energy plus the weighted share of the emitter hits
    // the paths found themselves, where a renderer without light sampling adds
    // those hits unweighted. Same integral, same vertices, so the totals agree in
    // expectation.
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
/** Record one light sample outcome in a ledger row. */
__device__ inline void ledgerRecord(LightLedger& row, LightSampleOutcome outcome, double energy)
{
    atomicAdd(&row.samples, 1ull);
    switch (outcome)
    {
        case LIGHT_ACCEPTED:
            atomicAdd(&row.accepted, 1ull);
            if (isfinite(energy))
            {
                atomicAdd(&row.acceptedEnergy, energy);
                atomicAdd(&row.acceptedEnergySq, energy * energy);
            }
            else
            {
                atomicAdd(&row.nonFiniteEnergy, 1ull);
            }
            break;
        case LIGHT_REJECT_COS_SURFACE:
            atomicAdd(&row.rejectCosSurface, 1ull);
            break;
        case LIGHT_REJECT_COS_LIGHT:
            atomicAdd(&row.rejectCosLight, 1ull);
            break;
        default:
            atomicAdd(&row.occluded, 1ull);
            break;
    }
}

/**
 * Record an emitter hit reached by a random walk, with the weight multiple
 * importance sampling gave it (`weight` is 1 for a delta sample and for the
 * renderer without light sampling, which is what makes the plain sum below the
 * reference side of the unbiasedness check).
 */
__device__ inline void ledgerRecordBsdfHit(LightLedger& row, double energy,
    double weight)
{
    atomicAdd(&row.bsdfHits, 1ull);
    atomicAdd(&row.bsdfEnergy, energy * weight);
    atomicAdd(&row.bsdfEnergyPlain, energy);
    atomicAdd(&row.bsdfEnergySq, energy * energy);
}
#endif

/** A direction uniform in solid angle over the cone of half angle `cosMax`
 *  around `axis`. Used by the distant light and by spherical lights. */
__host__ __device__ inline glm::vec3 sampleCone(glm::vec3 axis, float cosMax, float u1, float u2)
{
    const float cosTheta = glm::mix(cosMax, 1.0f, u1);
    const float sinTheta = sqrtf(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
    glm::vec3 t1, t2;
    buildTangentFrame(axis, t1, t2);
    const float phi = TWO_PI * u2;
    return glm::normalize(t1 * (sinTheta * cosf(phi)) + t2 * (sinTheta * sinf(phi))
        + axis * cosTheta);
}

/**
 * Density of the light sampling strategy for a direction that reaches a point on
 * `light`, in solid angle, including the chance of having picked that light. One
 * definition, used by both the estimator and the MIS weights: two call sites
 * drifting apart is how a light sampling implementation ends up silently dark.
 */
__host__ __device__ inline float lightSampleSolidAnglePdf(const DeviceLight& light, int lightCount,
    glm::vec3 shadingPoint, glm::vec3 lightPoint, glm::vec3 lightNormal)
{
    if (light.shape == LIGHT_SPHERE)
    {
        // Uniform over the tangent cone: constant density on the visible cap, so
        // no 1/cos singularity and no wasted samples on the far side.
        const glm::vec3 toCenter = light.center - shadingPoint;
        const float distance2 = glm::dot(toCenter, toCenter);
        const float radius2 = light.radius * light.radius;
        if (distance2 <= radius2)
        {
            return 0.0f;                 // inside the emitter: no visible cap
        }
        const float cosMax = sqrtf(glm::max(0.0f, 1.0f - radius2 / distance2));
        const float solidAngle = TWO_PI * (1.0f - cosMax);
        return 1.0f / glm::max((float)lightCount * solidAngle, 1e-12f);
    }

    // Box: uniform by area, converted to the solid angle the BSDF lives in.
    const glm::vec3 toLight = lightPoint - shadingPoint;
    const float distance2 = glm::dot(toLight, toLight);
    const float cosLight = glm::abs(glm::dot(lightNormal, glm::normalize(-toLight)));
    return 1.0f / glm::max((float)lightCount * light.surfaceArea, 1e-6f)
        * distance2 / glm::max(cosLight, 1e-4f);
}

/** Same, for a hit that has to be attributed to one of the lights by geometry. */
__host__ __device__ inline float lightSamplePdfForGeom(const DeviceLight* lights, int lightCount,
    glm::vec3 shadingPoint, glm::vec3 lightPoint, glm::vec3 lightNormal, int geomId)
{
    for (int i = 0; i < lightCount; i++)
    {
        if (lights[i].geomIndex == geomId)
        {
            return lightSampleSolidAnglePdf(lights[i], lightCount, shadingPoint, lightPoint,
                lightNormal);
        }
    }
    return 0.0f;
}

/** Sample a point on a random light: uniform by area over the six faces of a box
 *  or over the tangent cone of a sphere. Always consumes four random numbers so
 *  that every sample uses the same dimension budget. */
__host__ __device__ inline bool sampleLightSurface(const DeviceLight* lights, int lightCount,
    glm::vec3 shadingPoint, float u0, float u1, float u2, float u3,
    glm::vec3& point, glm::vec3& normal, glm::vec3& emission, int& geomIndex,
    int& lightIndex, int& faceIndex)
{
    if (lightCount <= 0)
    {
        return false;
    }
    int index = glm::min((int)(u0 * (float)lightCount), lightCount - 1);
    const DeviceLight& light = lights[index];

    if (light.shape == LIGHT_SPHERE)
    {
        const glm::vec3 toCenter = light.center - shadingPoint;
        const float distance2 = glm::dot(toCenter, toCenter);
        const float radius2 = light.radius * light.radius;
        if (distance2 <= radius2)
        {
            return false;                // the shading point is inside the light
        }
        const glm::vec3 axis = toCenter * glm::inversesqrt(distance2);
        const float cosMax = sqrtf(glm::max(0.0f, 1.0f - radius2 / distance2));
        const glm::vec3 wi = sampleCone(axis, cosMax, u2, u3);
        // Nearest intersection of that direction with the sphere: the sample
        // point, and the normal the emitter radiates along.
        const float b = glm::dot(wi, toCenter);
        const float disc = b * b - (distance2 - radius2);
        if (disc <= 0.0f)
        {
            return false;
        }
        const float t = b - sqrtf(disc);
        point = shadingPoint + wi * t;
        normal = glm::normalize(point - light.center);
        emission = light.emission;
        geomIndex = light.geomIndex;
        lightIndex = index;
        faceIndex = 0;
        return true;
    }

    const glm::vec3 size = light.boundsMax - light.boundsMin;
    const float area[6] = {
        size.x * size.y, size.x * size.y,     // -z and +z
        size.x * size.z, size.x * size.z,     // -y and +y
        size.y * size.z, size.y * size.z      // -x and +x
    };
    float total = 0.0f;
    for (int i = 0; i < 6; i++)
    {
        total += area[i];
    }

    // Pick a face proportional to its area, then a point on that face.
    float r = u1 * total;
    int face = 5;
    for (int i = 0; i < 5; i++)
    {
        if (r < area[i])
        {
            face = i;
            break;
        }
        r -= area[i];
    }

    const float a = u2;
    const float b = u3;
    switch (face)
    {
        case 0: point = glm::vec3(glm::mix(light.boundsMin.x, light.boundsMax.x, a),
                                  glm::mix(light.boundsMin.y, light.boundsMax.y, b),
                                  light.boundsMin.z); normal = glm::vec3(0, 0, -1); break;
        case 1: point = glm::vec3(glm::mix(light.boundsMin.x, light.boundsMax.x, a),
                                  glm::mix(light.boundsMin.y, light.boundsMax.y, b),
                                  light.boundsMax.z); normal = glm::vec3(0, 0, 1); break;
        case 2: point = glm::vec3(glm::mix(light.boundsMin.x, light.boundsMax.x, a),
                                  light.boundsMin.y,
                                  glm::mix(light.boundsMin.z, light.boundsMax.z, b)); normal = glm::vec3(0, -1, 0); break;
        case 3: point = glm::vec3(glm::mix(light.boundsMin.x, light.boundsMax.x, a),
                                  light.boundsMax.y,
                                  glm::mix(light.boundsMin.z, light.boundsMax.z, b)); normal = glm::vec3(0, 1, 0); break;
        case 4: point = glm::vec3(light.boundsMin.x,
                                  glm::mix(light.boundsMin.y, light.boundsMax.y, a),
                                  glm::mix(light.boundsMin.z, light.boundsMax.z, b)); normal = glm::vec3(-1, 0, 0); break;
        default: point = glm::vec3(light.boundsMax.x,
                                   glm::mix(light.boundsMin.y, light.boundsMax.y, a),
                                   glm::mix(light.boundsMin.z, light.boundsMax.z, b)); normal = glm::vec3(1, 0, 0); break;
    }

    emission = light.emission;
    geomIndex = light.geomIndex;
    lightIndex = index;
    faceIndex = face;
    return true;
}

/** Any geometry between the shading point and the light sample? */
__host__ __device__ inline bool isOccluded(Geom* geoms, int geomCount, glm::vec3 origin,
    glm::vec3 direction, float maxT, int skipGeom)
{
    Ray shadow;
    shadow.origin = origin;
    shadow.direction = direction;
    for (int i = 0; i < geomCount; i++)
    {
        if (i == skipGeom)
        {
            // Never let a light shadow itself: a point on the sampled surface is
            // behind its own silhouette for oblique views, which would reject
            // valid samples.
            continue;
        }
        Geom& geom = geoms[i];
        glm::vec3 p, n;
        bool outside = true;
        float t = -1.0f;
        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, shadow, p, n, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, shadow, p, n, outside);
        }
        else
        {
            t = sdfIntersectionTest(geom, shadow, p, n, outside, NULL, NULL);
        }
        if (t > 1e-4f && t < maxT)
        {
            return true;
        }
    }
    return false;
}

// --- Multiple importance sampling -----------------------------------------

/** Radiance of the environment light (the dome) in a direction: a three colour
 *  sky, horizon-to-zenith above and ground below, blended at the horizon.
 *  The escaping rays are what collect it, so a bright dome lights an open scene;
 *  it deliberately has no second sampling strategy (see the README). */
__host__ __device__ inline glm::vec3 environmentRadiance(const Environment& environment,
    glm::vec3 direction)
{
    const float t = direction.y;
    const glm::vec3 radiance = (t >= 0.0f)
        ? glm::mix(environment.horizon, environment.zenith, glm::min(t, 1.0f))
        : glm::mix(environment.ground, environment.horizon, glm::clamp(t + 1.0f, 0.0f, 1.0f));
    return environment.intensity * radiance;
}

// --- The distant light (a sun) ---------------------------------------------

/** Is this direction inside the distant light's disc? */
__host__ __device__ inline bool insideDistantLight(const DistantLight& sun, glm::vec3 direction)
{
    return sun.enabled != 0 && glm::dot(direction, -sun.direction) >= sun.cosMaxAngle;
}

/** Density of the distant light's sampler: uniform over its disc, zero
 *  elsewhere. The zero is what keeps MIS well behaved outside the disc. */
__host__ __device__ inline float distantLightPdf(const DistantLight& sun, glm::vec3 direction)
{
    return insideDistantLight(sun, direction) ? (1.0f / glm::max(sun.solidAngle, 1e-12f)) : 0.0f;
}

/** A direction uniformly distributed over the disc's cone (uniform in solid
 *  angle, which is what makes the density above a constant). */
__host__ __device__ inline glm::vec3 sampleDistantLight(const DistantLight& sun, float u1, float u2)
{
    const glm::vec3 axis = -sun.direction;              // towards the light
    const float cosTheta = glm::mix(sun.cosMaxAngle, 1.0f, u1);
    const float sinTheta = sqrtf(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
    glm::vec3 t1, t2;
    buildTangentFrame(axis, t1, t2);
    const float phi = TWO_PI * u2;
    return glm::normalize(t1 * (sinTheta * cosf(phi)) + t2 * (sinTheta * sinf(phi))
        + axis * cosTheta);
}

/** How often the light strategy picks the distant light over an area light: a
 *  fixed half when a scene has both, all of it when it has one. MIS keeps either
 *  choice unbiased; power weighted selection is the refinement. */
__host__ __device__ inline float distantLightSelectionChance(const DistantLight& sun,
    int lightCount, float totalLightArea)
{
    const bool areaLights = (lightCount > 0 && totalLightArea > 0.0f);
    const bool distant = sun.enabled != 0;
    if (distant && areaLights)
    {
        return 0.5f;
    }
    return distant ? 1.0f : 0.0f;
}

/** Density of the light strategy family (area lights + the sun) for a direction.
 *  `areaDensity` is what the area light part would produce, known by the caller
 *  from the sample it took or the hit it is looking at - and 0 when the direction
 *  leaves the scene, where it cannot have hit an area light. */
__host__ __device__ inline float lightStrategyPdf(const DistantLight& sun, float sunChance,
    float areaDensity, glm::vec3 direction)
{
    return (1.0f - sunChance) * areaDensity + sunChance * distantLightPdf(sun, direction);
}

/** Power heuristic (beta = 2) MIS weight:
 *      w = pdfThis^2 / (pdfThis^2 + pdfOther^2)
 *  Squaring makes the better strategy win faster than the balance heuristic
 *  would (PBRT and my CG2025 HW7 both use this form). */
__host__ __device__ inline float misWeight(float pdfThis, float pdfOther)
{
    const float a = pdfThis * pdfThis;
    const float b = pdfOther * pdfOther;
    const float sum = a + b;
    return (sum > 0.0f) ? (a / sum) : 0.0f;
}

// ---------------------------------------------------------------------------
// Restartable rendering - checkpoint file format and helpers
// ---------------------------------------------------------------------------
// Layout: a fixed header followed by the raw accumulation buffer (one glm::vec3
// of un-normalised radiance per pixel, in the same pixel order as the kernel
// writes it). Fixed size, no alignment surprises, no serialisation library.

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

/**
 * Fingerprint of everything a sample depends on: the camera (position, frame,
 * field of view, aperture, focus), the resolution, the depth, and the geometry
 * and materials. Two runs only share a checkpoint if all of it matches, so
 * editing the scene file - or just resizing it - starts from scratch instead of
 * adding samples of a different image into the accumulation buffer.
 */
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

// Pull the accumulation buffer off the device into the staging buffer. Pinned
// memory + an async copy on a dedicated stream is the fast path; the pageable
// path is a plain blocking cudaMemcpy, which is what the measurement in the
// README compares against.
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
// Restartable rendering - public entry points (see pathtrace.h)
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

bool pathtraceSaveCheckpoint(Scene* scene, int iterationsDone)
{
#if RESTARTABLE
    // CHECKPOINT 0 in the scene means "do not restart, I am watching this one".
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

    // Swap the finished file in only after it is complete on disk: a checkpoint
    // that is interrupted by a crash or a kill must never replace a good one with
    // a truncated one.
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
    if (dev_image == NULL)
    {
        return false;
    }

    const std::string path = checkpointPath(scene);
    FILE* file = fopen(path.c_str(), "rb");
    if (file == NULL)
    {
        return false;
    }

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

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // Grid coordinate of the sample. Note the convention of the camera
        // transform below: at x = 0 the offset is exactly -resolution.x * 0.5
        // times one pixel length, i.e. the left edge of the view. So integer
        // (x, y) addresses the *corner* of a pixel and the pixel area is
        // [x, x + 1) x [y, y + 1) - not [x - 0.5, x + 0.5).
        float sampleX = (float)x;
        float sampleY = (float)y;

        // Depth tag -1 gives the camera ray its own RNG stream: bounce `d`
        // draws with tag `d`, so reusing tag 0 here would make the sub-pixel
        // offset and the first scatter direction share random numbers.
        constexpr int CAMERA_RNG_DEPTH = -1;
        thrust::default_random_engine rng =
            makeSeededRandomEngine(iter, index, CAMERA_RNG_DEPTH);
        // The camera ray owns dimensions 0-3 of this sample: two for the pixel
        // area, two for the lens. Everything after that belongs to the path.
        Sampler sampler((unsigned int)iter, (unsigned int)index, 0u);

#if STOCHASTIC_AA
        // A fresh jitter per iteration turns the per-pixel value into the
        // average radiance over the pixel area (a box filter) instead of a
        // point sample, which is what removes the stair-stepping on edges.
        // The offset must span the whole pixel area, i.e. [0, 1) here; using
        // [-0.5, 0.5) would sample half of the previous pixel and blur every
        // edge across its neighbours.
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
        // Instead of firing every ray from the pinhole, sample a point on the
        // lens disk and aim the ray at the same point of the focal plane.
        // Everything at `focalDistance` therefore stays sharp no matter where
        // on the lens the ray started, and the blur circle of an out of focus
        // point grows with `aperture * |1/z - 1/focalDistance|`.
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
    ShadeableIntersection* intersections,
    unsigned long long* sdfStepCounter,
    unsigned long long* sdfHistogram)
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
            else if (geom.type == MANDELBULB || geom.type == MENGER)
            {
                // Procedural signed distance field shapes: sphere traced, with
                // an optional bounding sphere clip (see intersections.cu).
                t = sdfIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside,
                    sdfStepCounter, sdfHistogram);
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

    // (1) Ray escaped the scene: it sees the environment light, if the scene has
    //     one. That is the only way to find the dome - as background and as the
    //     light of an open scene.
    if (intersection.t <= 0.0f)
    {
        const glm::vec3 rayDirection = glm::normalize(pathSegment.ray.direction);
        glm::vec3 radiance = environmentRadiance(environment, rayDirection);
        // The sun is a disc at infinity and a ray can walk into it too, so this
        // gets the same MIS weight the estimator uses. The dome has no second
        // strategy and needs none.
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

    // Procedural texture: map the hit point back into object space (which is
    // where the pattern is defined, so it is attached to the object and follows
    // its transform) and modulate the albedo by the pattern. Emitters returned
    // above, so this only ever changes the diffuse/specular colour of a surface.
    if (material.textureType > 0 && intersection.geomId >= 0)
    {
        const Geom& geom = geoms[intersection.geomId];
        glm::vec3 objectSpace = multiplyMV(geom.inverseTransform, glm::vec4(intersect, 1.0f));
        material.color *= evaluateProceduralTexture(material.textureType,
            objectSpace * material.textureScale);
        material.specular.color = material.color;
    }

    // (4) Regular surface: evaluate the BSDF to update the throughput and
    //     generate the next ray. Seeding by (iteration, pixel, depth) keeps the
    //     samples of one pixel independent across iterations while still being
    //     reproducible.
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
    // A diffuse vertex connects straight to a random point on a light: sample the
    // light, convert the density to solid angle, evaluate the BSDF there and
    // reject the sample if the segment is blocked. MIS folds in the path's own
    // emitter hits, which needs a lobe with a density to aim at the light - a
    // mirror or a dielectric can only find one by accident.
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
            // says the direction missed any area light - which is what lets the
            // two parts of the strategy share one density.
            const glm::vec3 wi = sampleDistantLight(distantLight, su0, su1);
            const float cosSurface = glm::dot(normal, wi);
            const float lightPdf = sunChance * distantLightPdf(distantLight, wi);
            LightSampleOutcome outcome = LIGHT_REJECT_COS_SURFACE;
            double sampleEnergy = 0.0;
            if (cosSurface > 0.0f && lightPdf > 0.0f)
            {
                outcome = LIGHT_REJECT_OCCLUDED;
                const glm::vec3 shadowOrigin = intersect + normal * 1e-3f;
                if (!isOccluded(geoms, geomCount, shadowOrigin, wi, 1e30f, -1))
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
            // cosLight > 0 means the renderer can see this part of the light: a ray
            // towards an away-facing face would enter the box through a nearer one
            // first. Half of all samples land there; dropping them is what makes
            // the estimator match the renderer instead of lighting twice.
            LightSampleOutcome outcome = LIGHT_REJECT_COS_SURFACE;
            double sampleEnergy = 0.0;
            if (cosSurface > 0.0f)
            {
                outcome = LIGHT_REJECT_COS_LIGHT;
                if (cosLight > 0.0f)
                {
                    outcome = LIGHT_REJECT_OCCLUDED;
                    // Density for the whole light strategy: the picked surface
                    // light, plus the sun's density if its disc covers this
                    // direction too.
                    const float areaDensity = lightSampleSolidAnglePdf(lights[lightIndex],
                        lightCount, intersect, lightPoint, lightNormal);
                    const float lightPdf = lightStrategyPdf(distantLight, sunChance, areaDensity, wi);
                    const glm::vec3 shadowOrigin = intersect + normal * 1e-3f;
                    const int shadowSkip = LIGHT_SKIP_SELF_IN_SHADOW ? lightGeom : -1;
                    if (!isOccluded(geoms, geomCount, shadowOrigin, wi, distance - 1e-3f, shadowSkip))
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
    // NOTE: the low discrepancy sequence is deliberately *not* used for the path
    // dimensions. Measured, not assumed: pointing it at the BSDF and the roulette
    // made a 200 spp Cornell render worse rather than better (RMSE 34.6 against
    // 11.6 for random draws, and visibly grainier), which is the known failure of
    // a Halton sequence in a high dimensional integral whose effective dimension
    // changes from sample to sample. The pixel and lens dimensions, where it
    // provably helps, are handled in generateRayFromCamera.

    scatterRay(pathSegment, intersect, intersection.surfaceNormal,
        intersection.outside != 0, material, rng);

#if RUSSIAN_ROULETTE
    // --- Russian roulette ------------------------------------------------
    // Every path that is still alive after a few bounces is given a survival
    // probability equal to its throughput (clamped), and the survivors' weight is
    // divided by that probability. Cheap, low-contribution paths therefore die
    // early while the estimator stays unbiased:
    //
    //   E[killed? 0 : throughput / p] = p * (throughput / p) = throughput
    //
    // A path whose throughput is already tiny (a dark surface, a long chain of
    // bounces) keeps contributing with a small probability instead of costing a
    // full intersection + shading pass every iteration. This is the colour
    // dependent version of the roulette in the specular/BSDF lobes above.
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
            intersections,
            dev_sdfSteps,
            dev_sdfHistogram
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
        printSdfStats(pixelcount);
        printRussianRouletteStats(pixelcount);
        printDirectLightStats(pixelcount);
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // NOTE: the base code copied the whole accumulation buffer back to the host
    // here, after every single iteration, because saveImage() reads it from
    // there. That is 7.7 MB of PCIe traffic per iteration at 800x800 for data
    // that is only read when the user saves - or, now, when a checkpoint is due.
    // The copy moved to pathtraceFetchImage(), which saveImage() and the
    // checkpointer call on demand.

    checkCUDAError("pathtrace");
}
