# CUDA Path Tracer

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 3 - CUDA Path Tracer**

* **Jing Huang**
  * [GitHub](https://github.com/Stabil1ze)
* Tested on: Windows 11, Intel i7-12700H @ 2.30GHz 23GB,
  NVIDIA GeForce RTX 3060 Laptop GPU 6GB (Personal computer)

![](img/cover.png)

*A closed room lit by a 5x5 ceiling light with three mirror spheres: global
illumination (the teal and orange walls bleed onto the floor and onto the balls),
one-bounce mirror reflections and soft shadowing. `scenes/showcase.json`,
800x800, 3000 samples, 3m31s on the RTX 3060 Laptop.*

## Overview

The CIS 5650 base code provides the framework - JSON scene loading, primitive
intersections, the CUDA-OpenGL interop preview and image saving. Everything that
turns it into a renderer is implemented here: the BSDF shading kernel, the
bounce loop, anti-aliasing and stream compaction.

| # | Feature | File |
|---|---|---|
| 1 | BSDF shading kernel: ideal diffuse (cosine weighted) + perfect specular, dispatched per material | `src/interactions.cu`, `src/pathtrace.cu` |
| 2 | Material loading driven by `TYPE` / `ROUGHNESS` (the base `Specular` branch threw `ROUGHNESS` away) | `src/scene.cpp` |
| 3 | Stochastic sampled anti-aliasing (sub-pixel jitter, compile-time toggle `STOCHASTIC_AA`) | `src/pathtrace.cu` |
| 4 | Stream compaction of terminated paths - map / scan / scatter built on the work-efficient scan from my Project 2 | `src/pathtrace.cu`, `stream_compaction/` |
| 5 | Emitter hits accumulate straight into the image, which is what allows terminated paths to be dropped | `src/pathtrace.cu` |
| 6 | Per-bounce ray counting for the analysis (`[profile] ...` on stdout) | `src/pathtrace.cu` |

The two features that change the image (#3, #4) are behind `#define`s
(`STOCHASTIC_AA`, `STREAM_COMPACTION`), so every number below can be reproduced by
flipping one line and rebuilding.

## Implementation

### BSDF shading kernel

`scatterRay` picks one BSDF lobe per bounce with a probability proportional to its
weight and divides the throughput by that probability, which keeps the estimator
unbiased; today the weights are 0/1 (a material is either diffuse or a mirror),
but the same code handles mixed materials such as
`glossy = diffuse + imperfect specular`:

```cpp
if (u01(rng) < diffuseWeight / weightSum) {
    direction = calculateRandomDirectionInHemisphere(normal, rng);
    weight    = m.color * (diffuseWeight / probability);
} else {
    direction = glm::reflect(glm::normalize(pathSegment.ray.direction), normal);
    weight    = m.specular.color * (specularWeight / probability);
}
```

For the diffuse lobe the cosine term and the sampling pdf cancel exactly
(`BRDF = albedo / PI`, `pdf = cos(theta) / PI`, so `BRDF * cos / pdf = albedo`),
which is why cosine-weighted sampling is both cheap and low variance. The specular
lobe is a delta distribution, so its `BRDF * cos / pdf` collapses to the
reflectance itself.

The shading kernel (`shadeMaterials`) has four exits, and the exact semantics
matter for stream compaction:

* ray escaped the scene -> contribution zero, path terminated,
* ray hit an emitter -> `throughput * color * emittance` is added to the image and
  the path is terminated,
* surface hit but the bounce budget is used up -> contribution zero, terminated,
* otherwise `scatterRay` runs and the budget is decremented.

`remainingBounces` is the three-state bookkeeping: `> 0` alive and scattering,
`== 0` alive but only able to still see an emitter (the last segment), `< 0`
terminated - and `< 0` is exactly the predicate stream compaction removes.

Swapping the Cornell box's sphere from `Specular` to `Diffuse` isolates what the
specular branch adds (same scene, same seed, same sample count):

| diffuse sphere (no specular lobe) | perfect specular sphere |
|---|---|
| ![](img/cornell-diffuse-sphere.png) | ![](img/cornell-specular-sphere.png) |

### Stochastic sampled anti-aliasing

The first ray of every pixel used to be aimed at a single fixed point. Jittering
that point inside the pixel turns the per-pixel value into a Monte Carlo estimate
of the radiance *averaged over the pixel area* (a box filter), which is what
removes the stair-stepping on silhouettes:

```cpp
// the integer grid coordinate of this base code addresses the pixel *corner*
// (at x = 0 the offset is exactly the left edge of the view), so the pixel area
// is [x, x+1) and the jitter has to span [0, 1)
float sampleX = (float)x;
float sampleY = (float)y;
#if STOCHASTIC_AA
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, -1);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    sampleX += u01(rng);
    sampleY += u01(rng);
#endif
```

Two details are easy to get wrong. The depth tag `-1` gives the camera ray its own
random stream - bounce `d` draws with tag `d`, so reusing tag `0` would make the
sub-pixel offset and the first scatter direction share the same random numbers.
And the offset has to cover `[0, 1)` rather than `[-0.5, 0.5)`: with a `+/- 0.5`
jitter a pixel also samples half of its neighbour, which behaves like a box filter
shifted by half a pixel and measurably does *not* reduce the error against a
supersampled reference.

### Stream compaction

Every bounce, the paths that terminated are removed from the working arrays
(`map -> scan -> scatter`) so the next bounce only launches threads for rays that
can still reach a light:

```cpp
kernMarkAlivePaths<<<blocks, blockSize>>>(numPaths, dev_alive, paths);   // predicate
int m = nextPowerOfTwoAtLeast(numPaths + 1);
cudaMemcpy(dev_scanIndices, dev_alive, numPaths * sizeof(int), cudaMemcpyDeviceToDevice);
cudaMemset(dev_scanIndices + numPaths, 0, (m - numPaths) * sizeof(int));
StreamCompaction::Efficient::scanDevice(m, dev_scanIndices);             // Project 2 scan
kernScatterAlivePaths<<<blocks, blockSize>>>(...);                       // PathSegment[]
kernScatterAliveIntersections<<<blocks, blockSize>>>(...);               // ShadeableIntersection[]
```

* The scan is my Project 2 work-efficient (Blelloch) scan, copied into
  `stream_compaction/`. It runs in place on a power-of-two array with a zeroed
  tail, so the predicate is copied into a padded buffer first.
* The survivor count comes for free: the exclusive prefix sum at `[numPaths]` is
  the total, which saves a separate reduction. That is why the scan buffer is
  `nextPowerOfTwoAtLeast(numPaths + 1)`.
* `PathSegment` and `ShadeableIntersection` are mirrored arrays and are scattered
  with the same index map to stay in sync. The scatter goes into a second set of
  buffers which are then swapped in: element `i` moves to `indices[i] <= i`, so
  scattering in place would race with threads that have not read their element
  yet.
* Radiance is added to the image with `atomicAdd` at the moment a ray hits an
  emitter. The base code's `finalGather` pass over the path array cannot be kept:
  by then the very paths that carry radiance have been compacted away.

Stream compaction is a pure optimization - with it enabled the image is
**bit-identical** to rendering with it disabled (`RMSE = 0` over the whole 800x800
image), because the RNG seed is `(iteration, pixelIndex, depth)` and does not
depend on the array slot a path happens to live in.

## Performance Analysis

All numbers below: RTX 3060 Laptop, Release build, CUDA 13.3, MSVC 19.51, 800x800
with `DEPTH 8`, interleaved runs of the two binaries, median reported.

### Where the work goes: rays remaining per bounce

The path tracer prints how many paths it still processes after every bounce. The
shapes of those curves are the whole story of stream compaction:

| scene | b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8 | b9 |
|---|---|---|---|---|---|---|---|---|---|
| Cornell box (open) + compaction | 81.7% | 56.6% | 43.5% | 34.7% | 28.0% | 22.9% | 18.7% | 15.3% | 0% |
| closed box + compaction | 99.5% | 97.9% | 96.3% | 94.6% | 93.0% | 91.4% | 89.9% | 88.3% | 0% |
| either scene, compaction off | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

In the Cornell box 18% of the rays leave through the open front after the very
first bounce and only 15% survive to the eighth; in a closed box almost nothing
terminates early, which is exactly the case where compaction has nothing to
reclaim.

### Render time: open vs closed scene

| scene | compaction off | compaction on | result |
|---|---:|---:|---|
| Cornell box (open), 500 spp | 24.52 s | **16.15 s** | **1.52x faster** |
| closed box, 500 spp | 34.17 s | 37.27 s | 0.92x (9% slower) |
| closed box, 2000 spp | 133.8 s | 144.5 - 157.0 s | no benefit |

![](img/compaction-analysis.png)

*Left: rays launched per bounce. Right: render time.*

This matches the instruction's hint that stream compaction only affects rays which
terminate. In the open Cornell box it removes 85% of the work over eight bounces
and pays for itself with a 1.5x speedup; in a closed box it removes ~12% of the
work while still paying for a padded scan (up to 2^20 elements), two scatters of
both mirrored arrays and one synchronization per bounce, so it ends up slightly
behind. Reducing that fixed cost - for example by scanning only the current ray
count instead of a padded power of two, or by keeping the compacted arrays in
shared memory for the small tail of a path - is the obvious next optimization.

### Anti-aliasing: cost and quality

Coverage jitter costs two random numbers per camera ray per iteration - it
measured 13.7 s vs 13.0 s at 400x400/1000 spp, i.e. within noise.

The quality is easier to measure on a hard edge with no Monte Carlo noise
(`scenes/sphere.json`: an emissive sphere against a black background), comparing
against a converged 400x400/4000 spp reference:

| version | whole image RMSE | silhouette band RMSE | worst pixel in the band |
|---|---:|---:|---:|
| AA on, 500 spp | **0.24** | **13.66** | 34 |
| AA off, 500 spp | 6.31 | 138.38 | 234 |

The remaining error with AA on is the Monte Carlo noise of 500 jittered samples
and shrinks with more samples; the error with AA off is systematic aliasing that
never converges away. The zoomed silhouette (left: converged reference, middle: AA
off, right: AA on) shows the same thing:

![](img/aa-comparison.png)

### Validation against the reference image

`img/REFERENCE_cornell.5000samp.png` ships with the base code and my render
reproduces its global illumination: the red and green walls bleed onto the floor
and the ceiling, the sphere casts a soft shadow and the ceiling light is the only
bright emitter.

![](img/cornell-path-traced.png)

One deliberate difference: the scene marks its sphere as `Specular` with
`ROUGHNESS: 0.0`, which I read as a perfect mirror (the reference image's ball
looks like a much rougher, almost matte specular). A mirror sphere is what
`ROUGHNESS = 0` means in the GPU Gems 3, Ch. 20 sampling model whose `ROUGHNESS >
0` case is the "imperfect specular" extension, and the PR checklist lists perfect
specular as a core feature, so the sphere reflects the room instead.

## Scenes

| scene | what it is |
|---|---|
| `scenes/sphere.json` | single emissive sphere, used as the noise-free hard edge for the AA measurements |
| `scenes/cornell.json` | the supplied Cornell box (open towards the camera) |
| `scenes/cornell_closed.json` | the same box with a front wall and the camera moved inside, so no ray can escape - the closed case of the compaction analysis |
| `scenes/showcase.json` | the cover image: a closed room, a 5x5 ceiling light, three mirror spheres and a diffuse ball |

No meshes or texture files are used, so nothing has to be downloaded to render
them.

## CMakeLists.txt changes

The project needs three build fixes for CUDA 13 + MSVC (the same family of
problems as Project 2):

1. `include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")` moved out of
   the `if(UNIX)` branch - `sceneStructs.h` pulls in `cuda_runtime.h` and is
   included by host (CXX) sources as well.
2. `-Xcompiler=/Zc:preprocessor` for CUDA: the CCCL headers shipped with CUDA 13
   refuse to build with MSVC's traditional preprocessor.
3. `/Zc:preprocessor` for CXX for the same reason, since the CUDA runtime headers
   are compiled as host code too.

Two `add_subdirectory(stream_compaction)` / `target_link_libraries(...)` lines were
uncommented to link my Project 2 implementation, and
`stream_compaction/CMakeLists.txt` got its source list filled in plus the same
`/Zc:preprocessor` option (and a stray `}` typo fixed in its
`set_target_properties` call).

## Third-party code and credits

* `stream_compaction/{common,efficient}.{h,cu}` is **my own Project 2 code**
  (cis5650_stream_compaction_test), copied in as the project instructions suggest;
  only the CUDA error helper was renamed to `scCheckCUDAError` so that it does not
  clash with `pathtrace.cu`.
* No libraries were added - no mesh or texture loader is used, and every scene is
  built from the cubes and spheres the base code already supports.
* References used for the shading: PBRT v4 sections 9.2 (diffuse reflection) and
  9.3 (specular reflection and transmission), GPU Gems 3, Ch. 20 for the specular
  sampling model, and Paul Bourke's raytracing notes for anti-aliasing.
* The code in this project was written with the help of AI agents (pair
  programming over the CUDA/C++ sources, the build fixes and this README), in line
  with the course's third-party code policy.

