#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

// ---------------------------------------------------------------------------
// Restartable rendering
// ---------------------------------------------------------------------------
// The durable state of this renderer is the accumulated radiance buffer plus the
// number of samples that went into it; everything else (the camera rays, the
// path array, the intersection buffer, the compaction scratch space) is rebuilt
// from the scene every iteration, and the RNG is seeded by
// (iteration, pixelIndex, depth), so a resumed run continues the same estimator
// instead of starting a new one.

/** Copy the accumulation buffer back to scene->state.image (used by saveImage). */
void pathtraceFetchImage(Scene* scene);

/** Write "<imageName>.ckpt" holding the accumulation buffer and the header. */
bool pathtraceSaveCheckpoint(Scene* scene, int iterationsDone);

/**
 * Restore a checkpoint into the device accumulation buffer. Only loads a
 * checkpoint whose scene fingerprint matches, so editing the scene file starts a
 * fresh render. Returns true and writes the sample count when it loaded one.
 */
bool pathtraceLoadCheckpoint(Scene* scene, int* iterationsDone);

/** Remove the checkpoint of this scene (a finished render has nothing to resume). */
void pathtraceDeleteCheckpoint(Scene* scene);

/** Print how much the checkpoints cost (called when a render finishes). */
void printCheckpointStats();
