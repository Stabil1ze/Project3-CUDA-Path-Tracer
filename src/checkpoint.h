#pragma once

#include "scene.h"

#include <glm/glm.hpp>

// Restartable rendering: the accumulation buffer plus the sample count are
// written to "<FILE>.ckpt" while a render is in progress and picked up again on
// the next start. Those two are the renderer's only durable state - the camera
// rays, the path array, the intersection buffer and the compaction scratch are
// rebuilt from the scene every iteration - so a resumed run continues the same
// estimator instead of starting a new one.
//
// 1 (default): checkpoint every CHECKPOINT seconds
// 0: disable
#ifndef RESTARTABLE
#define RESTARTABLE 1
#endif

// Staging buffer the checkpoint is pulled off the device into
// 1 (default): pinned (page-locked) memory - the driver DMAs straight into it
// 0: pageable memory, which bounces through a driver staging buffer
#ifndef CHECKPOINT_PINNED_MEMORY
#define CHECKPOINT_PINNED_MEMORY 1
#endif

// Release the staging buffer and its copy stream; called when the renderer shuts down.
void checkpointRelease();

// Write, restore or delete "<imageName>.ckpt". The accumulation buffer belongs to
// the renderer, so it is passed in rather than reached for. A checkpoint is only
// restored when the scene fingerprint matches, and `iterationsDone` receives the
// sample count it holds.
bool checkpointSave(Scene* scene, glm::vec3* accumulation, int iterationsDone);
bool checkpointLoad(Scene* scene, glm::vec3* accumulation, int* iterationsDone);
void checkpointDelete(Scene* scene);

// Print how much the checkpoints cost; called when a render finishes.
void checkpointPrintStats();
