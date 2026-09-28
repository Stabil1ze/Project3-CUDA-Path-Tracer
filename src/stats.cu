#include "stats.h"

#include <cstdio>

namespace
{
// The counters themselves. They are private to this module; the accessors above
// are how the renderer gets at the ones its kernels write.

static const int MAX_PROFILE_DEPTH = 64;
static long long h_bounceAlive[MAX_PROFILE_DEPTH];
static long long h_profileIters = 0;

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

static void printDirectLightStats(int pixelcount, int lightCount, bool lightSamplingEnabled)
{
#if DIRECT_LIGHT_STATS
    // Nothing was sampled towards a light if the estimator is off, so there is
    // no ledger to report.
    if (!lightSamplingEnabled || dev_lightLedger == NULL)
    {
        return;
    }

    std::vector<LightLedger> perLight(lightCount + 1);
    std::vector<LightLedger> perFace((size_t)glm::max(lightCount, 1) * LIGHT_SAMPLE_FACES);
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
    for (int i = 0; i <= lightCount; i++)
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

    for (int i = 0; i < lightCount; i++)
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
        const LightLedger& row = perLight[lightCount];
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

}

StageTimer& statsTrace() { return stageTrace; }
StageTimer& statsSort() { return stageSort; }
StageTimer& statsShade() { return stageShade; }
unsigned long long* statsSdfSteps() { return dev_sdfSteps; }
unsigned long long* statsSdfHistogram() { return dev_sdfHistogram; }
LightLedger* statsLightLedger() { return dev_lightLedger; }
LightLedger* statsLightFaceLedger() { return dev_lightFaceLedger; }
LightLedger* statsFoldedLedger() { return dev_foldedLedger; }
LightLedger* statsFoldedAboveLedger() { return dev_foldedAboveLedger; }
LightLedger* statsFoldedFaceLedger() { return dev_foldedFaceLedger; }
unsigned long long* statsRrDecisions() { return dev_rrDecisions; }
unsigned long long* statsRrKills() { return dev_rrKills; }
unsigned long long* statsRrSurvivalMilli() { return dev_rrSurvivalMilli; }

void statsInit(int lightCount)
{
    stageTrace.init();
    stageSort.init();
    stageShade.init();

    h_profileIters = 0;
    for (int i = 0; i < MAX_PROFILE_DEPTH; i++)
    {
        h_bounceAlive[i] = 0;
    }

    cudaMalloc(&dev_sdfSteps, sizeof(unsigned long long));
    cudaMemset(dev_sdfSteps, 0, sizeof(unsigned long long));
    cudaMalloc(&dev_sdfHistogram, SDF_HISTOGRAM_BUCKETS * sizeof(unsigned long long));
    cudaMemset(dev_sdfHistogram, 0, SDF_HISTOGRAM_BUCKETS * sizeof(unsigned long long));

#if DIRECT_LIGHT_STATS
    {
        // One row per area light, plus one for the distant light (not a geometry,
        // so it gets the last row instead of an entry in the light list).
        cudaMalloc(&dev_lightLedger, (lightCount + 1) * sizeof(LightLedger));
        cudaMemset(dev_lightLedger, 0, (lightCount + 1) * sizeof(LightLedger));
        cudaMalloc(&dev_lightFaceLedger,
            (size_t)glm::max(lightCount, 1) * LIGHT_SAMPLE_FACES * sizeof(LightLedger));
        cudaMemset(dev_lightFaceLedger, 0,
            (size_t)glm::max(lightCount, 1) * LIGHT_SAMPLE_FACES * sizeof(LightLedger));
        cudaMalloc(&dev_foldedLedger, sizeof(LightLedger));
        cudaMemset(&dev_foldedLedger[0], 0, sizeof(LightLedger));
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
}

void statsFree()
{
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
    dev_rrDecisions = NULL;
    dev_rrKills = NULL;
    dev_rrSurvivalMilli = NULL;

    cudaFree(dev_lightLedger);
    cudaFree(dev_lightFaceLedger);
    cudaFree(dev_foldedLedger);
    cudaFree(dev_foldedAboveLedger);
    cudaFree(dev_foldedFaceLedger);
    dev_lightLedger = NULL;
    dev_lightFaceLedger = NULL;
    dev_foldedLedger = NULL;
    dev_foldedAboveLedger = NULL;
    dev_foldedFaceLedger = NULL;
}

void statsRecordBounce(int depth, int numPaths)
{
    if (depth <= MAX_PROFILE_DEPTH)
    {
        h_bounceAlive[depth - 1] += numPaths;
    }
}

void statsEndIteration(int iter, unsigned int iterations, int maxSegments, int pixelcount,
    int lightCount, bool lightSamplingEnabled)
{
    h_profileIters++;
    if (iter == 1)
    {
        printBounceProfile("iteration 1 - paths processed after each bounce:",
            h_bounceAlive, maxSegments, 1, pixelcount);
    }
    if (iter >= (int)iterations)
    {
        printBounceProfile("average over all iterations - paths processed after each bounce:",
            h_bounceAlive, maxSegments, h_profileIters, pixelcount);
        printSdfStats(pixelcount);
        printRussianRouletteStats(pixelcount);
        printDirectLightStats(pixelcount, lightCount, lightSamplingEnabled);
        printStageTimings();
    }
}
