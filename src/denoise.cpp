#include "denoise.h"

#include <chrono>
#include <cstdio>
#include <cstring>

#if USE_OIDN
#include <OpenImageDenoise/oidn.h>
#endif

bool denoiserAvailable()
{
#if USE_OIDN
    return true;
#else
    return false;
#endif
}

std::string denoiserDescription()
{
#if USE_OIDN
    return "Open Image Denoise, external/oidn package (CUDA or CPU device, whichever loads)";
#else
    return "not built in (unpack the Open Image Denoise release into external/ to enable it)";
#endif
}

#if USE_OIDN
namespace
{
    /** Pick the fastest device the package ships: CUDA if it commits, else CPU.
     *  A device that fails to commit is released again. */
    OIDNDevice createDevice(std::string& description)
    {
        const OIDNDeviceType candidates[2] = { OIDN_DEVICE_TYPE_CUDA, OIDN_DEVICE_TYPE_CPU };
        for (OIDNDeviceType type : candidates)
        {
            OIDNDevice device = oidnNewDevice(type);
            if (device == NULL)
            {
                continue;
            }
            oidnCommitDevice(device);
            const char* errorMessage = NULL;
            if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
            {
                oidnReleaseDevice(device);
                continue;
            }
            // The 2.5 header exposes no device name string, so the description
            // names the kind that committed.
            description = (type == OIDN_DEVICE_TYPE_CUDA) ? "CUDA" : "CPU";
            return device;
        }
        return NULL;
    }
}
#endif

DenoiseResult denoiseImage(
    const std::vector<glm::vec3>& color,
    const std::vector<glm::vec3>& normal,
    const std::vector<glm::vec3>& albedo,
    int width,
    int height,
    std::vector<glm::vec3>& out)
{
    DenoiseResult result;
#if !USE_OIDN
    result.message = denoiserDescription();
    return result;
#else
    const size_t pixels = (size_t)width * height;
    if (color.size() != pixels || width <= 0 || height <= 0)
    {
        result.message = "bad image dimensions";
        return result;
    }

    const auto start = std::chrono::steady_clock::now();
    OIDNDevice device = createDevice(result.device);
    if (device == NULL)
    {
        result.message = "no usable device";
        return result;
    }

    // The images must live in memory the device can read. The CUDA device
    // rejects host pointers, so everything goes through OIDN buffers, which are
    // device accessible by construction.
    const size_t bytes = pixels * sizeof(glm::vec3);
    const size_t rowStride = (size_t)width * sizeof(glm::vec3);
    OIDNBuffer colorBuffer = oidnNewBuffer(device, bytes);
    OIDNBuffer outputBuffer = oidnNewBuffer(device, bytes);
    OIDNBuffer normalBuffer = NULL;
    OIDNBuffer albedoBuffer = NULL;
    memcpy(oidnGetBufferData(colorBuffer), color.data(), bytes);

    OIDNFilter filter = oidnNewFilter(device, "RT");   // the ray tracing filter
    oidnSetSharedFilterImage(filter, "color", oidnGetBufferData(colorBuffer), OIDN_FORMAT_FLOAT3,
        width, height, 0, sizeof(glm::vec3), rowStride);
    if (normal.size() == pixels)
    {
        normalBuffer = oidnNewBuffer(device, bytes);
        memcpy(oidnGetBufferData(normalBuffer), normal.data(), bytes);
        oidnSetSharedFilterImage(filter, "normal", oidnGetBufferData(normalBuffer),
            OIDN_FORMAT_FLOAT3, width, height, 0, sizeof(glm::vec3), rowStride);
    }
    if (albedo.size() == pixels)
    {
        albedoBuffer = oidnNewBuffer(device, bytes);
        memcpy(oidnGetBufferData(albedoBuffer), albedo.data(), bytes);
        oidnSetSharedFilterImage(filter, "albedo", oidnGetBufferData(albedoBuffer),
            OIDN_FORMAT_FLOAT3, width, height, 0, sizeof(glm::vec3), rowStride);
    }
    oidnSetSharedFilterImage(filter, "output", oidnGetBufferData(outputBuffer),
        OIDN_FORMAT_FLOAT3, width, height, 0, sizeof(glm::vec3), rowStride);
    oidnSetFilterBool(filter, "hdr", true);            // linear HDR in, linear HDR out
    oidnSetFilterBool(filter, "srgb", false);
    oidnSetFilterBool(filter, "cleanAux", false);
    oidnCommitFilter(filter);
    oidnExecuteFilter(filter);

    const char* errorMessage = NULL;
    if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
    {
        result.message = errorMessage != NULL ? errorMessage : "unknown denoiser error";
        oidnReleaseFilter(filter);
        oidnReleaseBuffer(colorBuffer);
        oidnReleaseBuffer(outputBuffer);
        if (normalBuffer != NULL)
        {
            oidnReleaseBuffer(normalBuffer);
        }
        if (albedoBuffer != NULL)
        {
            oidnReleaseBuffer(albedoBuffer);
        }
        oidnReleaseDevice(device);
        return result;
    }

    out.resize(pixels);
    memcpy(out.data(), oidnGetBufferData(outputBuffer), bytes);

    oidnReleaseFilter(filter);
    oidnReleaseBuffer(colorBuffer);
    oidnReleaseBuffer(outputBuffer);
    if (normalBuffer != NULL)
    {
        oidnReleaseBuffer(normalBuffer);
    }
    if (albedoBuffer != NULL)
    {
        oidnReleaseBuffer(albedoBuffer);
    }
    oidnReleaseDevice(device);

    result.ok = true;
    result.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
#endif
}
