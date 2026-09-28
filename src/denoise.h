#pragma once

#include "glm/glm.hpp"

#include <string>
#include <vector>

/**
 * Intel Open Image Denoise, behind a one line interface.
 *
 * Optional: without it (or without its runtime DLLs next to the executable)
 * `denoiserAvailable()` is false and the renderer behaves exactly as before.
 * With it, the image is denoised using the first hit's normal and albedo as
 * guides - the extra buffer the credit requires - which is why the renderer
 * writes those two out on the camera ray.
 */

struct DenoiseResult
{
    bool ok = false;
    double milliseconds = 0.0;
    std::string device;
    std::string message;
};

bool denoiserAvailable();

/** A short description for the render log, or why it is unavailable. */
std::string denoiserDescription();

/** Denoise a linear HDR image. `color` is the accumulated radiance divided by
 *  the sample count, before clamping or tone mapping - the denoiser's output is
 *  linear HDR too. `normal` and `albedo` are the guide buffers; pass empty
 *  vectors to run without them. All three must be width * height long. */
DenoiseResult denoiseImage(
    const std::vector<glm::vec3>& color,
    const std::vector<glm::vec3>& normal,
    const std::vector<glm::vec3>& albedo,
    int width,
    int height,
    std::vector<glm::vec3>& out);
