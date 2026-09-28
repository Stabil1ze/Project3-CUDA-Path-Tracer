#include "checkpoint.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace
{
// The checkpoint module reports its own CUDA failures: pathtrace.cu's helper is
// file local by design, so the two do not share a macro.
void checkpointCudaCheck(const char* message, int line)
{
    cudaDeviceSynchronize();
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess)
    {
        return;
    }
    fprintf(stderr, "[checkpoint] %s:%d: %s\n", message, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
}
}
#define checkCUDAError(msg) checkpointCudaCheck(msg, __LINE__)

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
static double downloadCheckpointImage(glm::vec3* accumulation, size_t bytes)
{
    auto start = std::chrono::steady_clock::now();
#if CHECKPOINT_PINNED_MEMORY
    cudaMemcpyAsync(h_checkpointStaging, accumulation, bytes, cudaMemcpyDeviceToHost,
        checkpointStream);
    cudaStreamSynchronize(checkpointStream);
#else
    cudaMemcpy(h_checkpointStaging, accumulation, bytes, cudaMemcpyDeviceToHost);
#endif
    checkCUDAError("checkpoint download");
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool checkpointSave(Scene* scene, glm::vec3* accumulation, int iterationsDone)
{
#if RESTARTABLE
    // CHECKPOINT 0 in the scene means do no restart
    if (scene->state.checkpointInterval <= 0.0f || accumulation == NULL || iterationsDone <= 0)
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

    const double copyMs = downloadCheckpointImage(accumulation, bytes);

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

bool checkpointLoad(Scene* scene, glm::vec3* accumulation, int* iterationsDone)
{
#if RESTARTABLE
    if (accumulation == NULL) { return false; }

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

    cudaMemcpy(accumulation, h_checkpointStaging, bytes, cudaMemcpyHostToDevice);
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

void checkpointDelete(Scene* scene)
{
    std::error_code ec;
    std::filesystem::remove(checkpointPath(scene), ec);
}

void checkpointPrintStats()
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

void checkpointRelease()
{
    releaseCheckpointStaging();
}
