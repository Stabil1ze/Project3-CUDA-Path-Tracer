#pragma once

#include "lights.h"        // LightLedger
#include "sceneStructs.h"  // Environment, DistantLight, DeviceLight

#include <glm/glm.hpp>

#include <vector>

// Instrumentation switches. They live next to the counters they guard; the
// renderer includes this header and picks them up from here.
//
// Procedural shape instrumentation: sphere tracing steps into a global counter
// plus a 16 bucket histogram, printed at the end of the render
// 1 (default): on
// 0: off (the atomics cost a few percent, so off when timing)
#ifndef SDF_STATS
#define SDF_STATS 1
#endif

// Sample ledger for the light estimator: per light and per face, samples drawn,
// accepted, why the rest were dropped, and the energy delivered against the
// BSDF emitter hits it replaced
// 1 (default): on (a few atomics per sample, so off when timing)
// 0: off
#ifndef DIRECT_LIGHT_STATS
#define DIRECT_LIGHT_STATS 1
#endif

// One pair of CUDA events per stage of the bounce loop. The definition follows
// at the bottom of this header; only a reference to it is handed out.
struct StageTimer;

// The counters the kernels write into. The renderer fetches them at the launch
// sites instead of reaching into this module's globals.
StageTimer& statsTrace();
StageTimer& statsSort();
StageTimer& statsShade();
unsigned long long* statsSdfSteps();
unsigned long long* statsSdfHistogram();
LightLedger* statsLightLedger();
LightLedger* statsLightFaceLedger();
LightLedger* statsFoldedLedger();
LightLedger* statsFoldedAboveLedger();
LightLedger* statsFoldedFaceLedger();
unsigned long long* statsRrDecisions();
unsigned long long* statsRrKills();
unsigned long long* statsRrSurvivalMilli();

// Allocate the counters for a scene with `lightCount` emitters, and release them.
void statsInit(int lightCount);
void statsFree();

// Bounce loop bookkeeping: how many paths are still being processed after this
// bounce, and - once the render is done - every report that was collected.
void statsRecordBounce(int depth, int numPaths);
void statsEndIteration(int iter, unsigned int iterations, int maxSegments, int pixelcount,
    int lightCount, bool lightSamplingEnabled);

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
