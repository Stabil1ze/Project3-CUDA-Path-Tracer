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

// Copy the accumulation buffer back to scene->state.image
void pathtraceFetchImage(Scene* scene);

// Copy the denoiser guides back to scene->state.normalImage / albedoImage
void pathtraceFetchDenoiseGuides(Scene* scene);

// Clear the accumulation buffer and reset the iteration count to 0
// Leave the buffer in device memory allocated
void pathtraceRestart(Scene* scene);

// Write "<imageName>.ckpt" holding the accumulation buffer and the header
bool pathtraceSaveCheckpoint(Scene* scene, int iterationsDone);

// Restore a checkpoint into the device accumulation buffer
bool pathtraceLoadCheckpoint(Scene* scene, int* iterationsDone);

// Remove the checkpoint of this scene
void pathtraceDeleteCheckpoint(Scene* scene);

// Print how much the checkpoints cost
void printCheckpointStats();
