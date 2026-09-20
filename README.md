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
800x800, 3000 samples, 3m55s on the RTX 3060 Laptop.*

## Overview

The CIS 5650 base code provides the framework - JSON scene loading, primitive
intersections, the CUDA-OpenGL interop preview and image saving. Everything that
turns it into a renderer is implemented here: the BSDF shading kernel, the
bounce loop, anti-aliasing, stream compaction, a thin lens camera and the
procedural shapes and textures.

| # | Feature | File |
|---|---|---|
| 1 | BSDF shading kernel: ideal diffuse (cosine weighted) + perfect specular, dispatched per material | `src/interactions.cu`, `src/pathtrace.cu` |
| 2 | Material loading driven by `TYPE` / `ROUGHNESS` (the base `Specular` branch threw `ROUGHNESS` away) | `src/scene.cpp` |
| 3 | Stochastic sampled anti-aliasing (sub-pixel jitter, compile-time toggle `STOCHASTIC_AA`) | `src/pathtrace.cu` |
| 4 | Stream compaction of terminated paths - map / scan / scatter built on the work-efficient scan from my Project 2 | `src/pathtrace.cu`, `stream_compaction/` |
| 5 | Emitter hits accumulate straight into the image, which is what allows terminated paths to be dropped | `src/pathtrace.cu` |
| 6 | Per-bounce ray counting for the analysis (`[profile] ...` on stdout) | `src/pathtrace.cu` |
| 7 | Physically based depth of field: thin lens camera driven by two optional scene fields (`APERTURE`, `FOCUS`), off by default | `src/pathtrace.cu`, `src/scene.cpp` |
| 8 | Camera basis and orbit camera fixes (the base code mirrored the pitch and put the eye below the floor whenever a scene looked downwards) | `src/scene.cpp`, `src/main.cpp` |
| 9 | Procedural shapes: a power-8 Mandelbulb and a Menger sponge, signed distance fields intersected by sphere tracing with a bounding sphere broad phase | `src/intersections.cu` |
| 10 | Procedural textures: checker and marble, evaluated on the object space hit point so they work on any shape | `src/interactions.cu` |
| 11 | Restartable rendering: the accumulation buffer and sample count are checkpointed to `<FILE>.ckpt` and picked up again on the next start | `src/pathtrace.cu`, `src/main.cpp` |

The two features that change the image (#3, #4) are behind `#define`s
(`STOCHASTIC_AA`, `STREAM_COMPACTION`), so every number below can be reproduced by
flipping one line and rebuilding; #7 is off unless a scene asks for it, so the
pinhole path stays bit-identical to what the previous sections measured.

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

// depth tag -1 gives the camera ray a random stream of its own
thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, -1);
thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

#if STOCHASTIC_AA
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

The camera random stream is created unconditionally (and the jitter only drawn
under `#if STOCHASTIC_AA`) because the depth of field below reuses the same stream:
with `STOCHASTIC_AA` off the two draws then go to the lens sample instead of being
wasted.

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

### Depth of field

A pinhole camera is sharp at every distance. The thin lens model replaces the
single eye point with a disk of radius `APERTURE` and aims every ray at the point
where the *pinhole* ray crosses the focal plane at distance `FOCUS`, so a point at
exactly that distance is still imaged to a single point no matter where on the
lens the ray started:

```cpp
// pinholeDirection is the direction the camera would fire with aperture == 0
if (cam.aperture > 0.0f && cam.focalDistance > 0.0f)
{
    float radius = cam.aperture * sqrtf(u01(rng));   // sqrt = uniform over the disk
    float angle  = TWO_PI * u01(rng);
    glm::vec3 lensPoint  = cam.position
        + cam.right * (radius * cosf(angle))
        + cam.up    * (radius * sinf(angle));
    glm::vec3 focalPoint = cam.position + pinholeDirection * cam.focalDistance;

    segment.ray.origin    = lensPoint;
    segment.ray.direction = glm::normalize(focalPoint - lensPoint);
}
else
{
    segment.ray.origin    = cam.position;
    segment.ray.direction = pinholeDirection;
}
```

The square root is what makes the samples uniform over the *area* of the disk
rather than clumped around its centre, and drawing `radius` and `angle` from the
same camera-ray stream that anti-aliasing uses means the pixel position and the
lens position vary together over the iterations - the estimator becomes an average
over the pixel *and* over the lens. Because both random numbers are always drawn
in this branch, switching anti-aliasing off simply hands two draws to the lens
sample instead of wasting them.

Two optional fields were added to the scene format:

| field | meaning | default |
|---|---|---|
| `APERTURE` | lens radius in world units, `0` keeps the camera a pinhole | `0` |
| `FOCUS` | distance of the focal plane | `\|lookAt - position\|` |

`FOCUS` defaults to the distance of the look-at point, i.e. "the thing I am looking
at is in focus", and `APERTURE` defaults to a pinhole - so every scene written
before this feature, and every measurement in this README, keeps rendering through
the original pinhole path. The blur of a point at distance `z` is a disk of
diameter

$$
c = 2 \cdot \text{aperture} \cdot \frac{|z - \text{FOCUS}|}{z}
$$

world units at that depth (similar triangles between the lens and the focal
plane), which grows linearly with the aperture and is asymmetric around the focus
plane: at a fixed aperture an object twice as far away is blurred half as much as
one twice as close.

![](img/dof-aperture-sweep.png)

*Aperture sweep at a fixed focus distance of 8.75, the third mirror sphere. 400x400,
1500 samples: with the aperture closed (top left) every sphere is equally sharp;
opening it blurs the near diffuse ball and the first two mirrors while the sphere
at the focus distance stays crisp. The shadows under the balls blur with their
casters, which is the geometrically correct behaviour and the usual tell of a real
lens.*

![](img/dof-focus-plane.png)

*The same scene at aperture 0.5 with the focal plane moved onto the near ball
(3.80), onto the third mirror (8.75) and onto the last mirror (12.71). The ball is
2.8 units across and its blur disk at 3.80 is `2 * 0.5 * |3.8 - 8.75| / 3.8` = 1.30
units, almost half its diameter - which is why the ball is sharp while the room
behind it dissolves in the left panel, and why every mirror except the last one is
blurred in the right one.*

![](img/dof.png)

*`scenes/dof.json` at 800x800, 2000 samples, aperture 0.25: the focus plane sits on
the third of the five mirrors running away from the camera.*

**A camera bug this feature uncovered.** Writing the scene for this feature turned
up a bug in the base code's interactive camera. `main()` recovers the orbit camera
parameters (azimuth, elevation, distance) from the scene's `EYE`/`LOOKAT` pair,
and `runCuda()` rebuilds the eye position and the basis from them, so the two have
to be exact inverses. The recovery used `acos(normalize((0, view.y, view.z)) . (0,
1, 0))`, which is the *mirror* of the elevation rather than the elevation: any
scene that looks downwards (including the supplied `scenes/showcase.json`) had its
pitch flipped and its eye mirrored below the look-at point - the DOF scene put the
camera under the floor, looking up, and rendered black. The elevation is now
recovered as `acos(-forward.y)` with `forward` the unit eye-to-look-at vector, and
the camera basis is normalised (`cross(view, up)` was used unnormalised, which
squeezed the horizontal field of view of every tilted camera). Both changes are
exact no-ops for a level camera: the Cornell images in this README are
bit-identical before and after the fix, and the cover image was re-rendered
because `scenes/showcase.json` intentionally looks down.

### Procedural shapes

Two of the objects the scene format understands are not primitives but procedural
shapes, defined as signed distance fields and intersected by sphere tracing: a
power-8 **Mandelbulb** and a level-3 **Menger sponge**, both added as new
`GeomType`s (`scenes/procedural.json` asks for them with `"TYPE": "Mandelbulb"` /
`"TYPE": "Menger"`).

Sphere tracing works because a signed distance field *lower bounds* the distance
to the surface, so a step of that size can never tunnel through it. One property
of the base code forces a correction here: the shape is evaluated in object space
(where its field has Lipschitz constant 1) but marched in world space, and scaling
the field by the object's largest scale factor turns it into a Lipschitz-s field -
so every step has to be divided by that scale, otherwise a scaled-up fractal is
stepped straight through.

```cpp
for (int i = 0; i < MAX_STEPS; i++) {
    glm::vec3 pWorld = rayOrigin + t * rayDirection;
    float d = sdfEvaluate(geom.type, multiplyMV(geom.inverseTransform, vec4(pWorld, 1)));
    if (glm::abs(d) < HIT_EPSILON) { hit = true; break; }
    t += glm::max(d * stepScale, HIT_EPSILON);   // stepScale = 1 / max scale
}
```

* The surface normal comes from the gradient of the field, taken as four
  evaluations arranged as the corners of a tetrahedron instead of the six a
  per-axis central difference would need.
* `MAX_STEPS = 128` and `HIT_EPSILON = 1e-4` bound the cost of a single test. A
  ray that grazes the shape can spend all 128 steps and still report "no hit"
  (see the histogram below) - the classic failure mode of sphere tracing, and the
  reason the shapes are used as *scene decoration* rather than, say, as a
  navmesh.
* Before marching, the ray is tested against the shape's bounding sphere (radius
  1.3 for the Mandelbulb, `sqrt(3)` for the Menger sponge, scaled by the object).
  That is a pure rejection test - the march itself still starts at `t = 0` - so
  `SDF_BOUNDING_SPHERE` can be flipped in `intersections.h` to measure the culling
  without changing a single pixel of the image.

Both shapes are fractal *distance estimates* rather than exact fields: the
Mandelbulb's `0.5 log(r) r / dr` bound is a lower bound with a well known
overestimate in the outer shell, which is why the marching needs the epsilon
rather than an equality test, and why the surfaces are slightly "soft" compared to
the analytic primitives.

### Procedural textures

Two procedural textures modulate the diffuse albedo: a 3D **checker board** and a
**marble** pattern built from fractal value noise. Both are pure functions of the
hit point - but of the hit point in *object space*, so the pattern is attached to
the object, follows its transform, and needs no UV layout at all. That last part
is what makes them usable on the SDF shapes above: there is no sane way to put a
(u, v) parameterisation on a Mandelbulb, but "the point I hit, in the object's own
coordinates" always exists.

```cpp
// checker: the parity of the lattice cell
float cells = glm::floor(p.x) + glm::floor(p.y) + glm::floor(p.z);
return glm::mix(vec3(1.0f), vec3(0.08f, 0.08f, 0.10f), glm::mod(cells, 2.0f));

// marble: veins along a diagonal, wrung by 5 octaves of value noise
float bands = sinf((p.x + 0.6f * p.y + 0.35f * p.z) * 3.0f + fractalNoise(p) * 6.0f);
return glm::mix(vec3(0.34f), vec3(1.0f), bands * bands);
```

The value noise is trilinear interpolation of an integer hash (`xorshift`-style
multiplies), and the fractal sum reuses it at five frequencies. The scene format
gained two optional material fields, `TEXTURE` (`none` / `checker` / `marble`) and
`TEXSCALE` (how many pattern cells fit into one object space unit).

![](img/procedural-textures.png)

*The same scene and the same camera, with both textures switched off and on.
`scenes/procedural.json`, 400x400, 500 samples, cropped to the two shapes.*

![](img/procedural.png)

*`scenes/procedural.json` at 800x800, 3000 samples (14m32s): the marble
Mandelbulb on the left, the checkered Menger sponge on the right, a mirror ball
behind them. The grain that is left is Monte Carlo noise from the indirect
bounces - the sphere tracing itself is deterministic.*

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

### Depth of field: cost

The thin lens model costs two extra random numbers, a square root and a
sine/cosine pair - but only on the **camera** ray, never on the interior bounces
that dominate a render. Interleaved runs of the same scene (800x800, 300 spp, three
runs each, medians):

| aperture | median render time | vs pinhole |
|---|---:|---:|
| 0.0 (pinhole) | 18.97 s | - |
| 0.5 | 19.49 s | +2.7% |

Individual runs were 18.75 / 19.72 / 18.97 s and 19.49 / 19.26 / 19.91 s, so the
2.7% is at the edge of run-to-run spread - the honest reading is "under 3%, of the
same order as the two extra RNG draws it actually adds".

On a hypothetical CPU version of this renderer the two extra lines would cost the
same per ray, and the feature would look just as cheap in a profiler - but it would
be paid *serially*, once per ray, on every one of the millions of camera rays.
What makes it free here is that the work is per-ray and has no cross-ray
communication: no extra memory traffic (the lens offset is two registers), no
divergence (the `aperture > 0` branch is uniform across the grid, and the rest is
straight-line arithmetic), and the camera kernel is a tiny fraction of the frame
time next to 8 bounces of intersection and shading. Depth of field therefore scales
with the number of pixels, not with the depth of the paths, which is the opposite
of every other feature in this project.

Two obvious next steps if it ever showed up in a profile: sample the disk from a
precomputed low-discrepancy table (a concentric or Sobol disk) instead of
`sqrt`/`cos`/`sin` per ray, and fold the four camera-ray random numbers (two for
anti-aliasing, two for the lens) into one stratified 4D sample so that the pixel
area *and* the lens disk are jointly stratified instead of independently jittered.

### Procedural shapes: cost, culling and divergence

Sphere tracing is by far the most expensive thing in this project. The table below
is `scenes/procedural.json` (the two fractals, the marble and the checker) against
`scenes/procedural_analytic.json`, which is the *same scene* with a sphere and a
cube in place of the two fractals, sized to the same bounding sphere / box. Same
camera, same samples per pixel, medians of interleaved runs at 400x400 / 300 spp:

| scene | render time | vs analytic |
|---|---:|---:|
| analytic primitives (sphere + cube) | 7.3 s | - |
| SDF shapes, bounding sphere on | 25.4 s | **3.5x** |
| SDF shapes, bounding sphere off | 31.1 s | 4.3x |

Two things fall out of that:

* **The shapes dominate the frame.** Every ray that tests a fractal pays 10-100
  field evaluations, where the analytic scene pays a quadratic solve. Sphere
  tracing buys an enormous amount of shape complexity for that price, but it is
  not free, and it is why the fractals sit in an otherwise simple room.
* **The bounding sphere culling is worth 1.22x**, and it is *free of side
  effects*: because the sphere is only used to reject rays and does not move the
  marching, enabling it leaves the image **bit-identical** (`max diff = 0` over
  the 400x400, 300 spp render). The same scene with the toggle off is 22% slower
  for no visual gain at all. Clipping the march to the sphere's entry point buys
  another ~13% on top, but that *does* change the sample positions - and a 1e-4
  change in the hit point is enough to decorrelate the noise of an 8-bounce path,
  so the two images differ pixel by pixel while agreeing in the mean to 0.03% of
  full scale. I kept the version that cannot be told apart from the baseline.

The tracer counts its own marching steps (the `[sdf]` lines on stdout, gated by
`SDF_STATS`), which is what the left panel below is made of. The counters are 64
bit, which matters more than it sounds: a 800x800/3000spp render produces 5.3e9
marches, and a 32 bit accumulator silently wraps that into a plausible looking
number.

![](img/sdf-analysis.png)

*220M marches at an average of 68.9 steps per SDF test, on the 400x400 / 500 spp
render. The last bucket is the interesting one: 29% of marches use the entire
128 step budget and give up. Those are the near-miss rays and the rays that pass
through the shell where the Mandelbulb's distance estimate is nearly zero.*

That 29% tail is also the most GPU-specific part of the feature. A warp marches
until every one of its 32 lanes is done, so lanes that finished in 20 steps sit
idle while a neighbour grinds to 128; the bar chart's average of 68.9 steps is
therefore a *lower* bound on what the hardware pays. On the CPU side the same
field evaluations would be spread over a handful of cores, and the divergence
would be hidden by out-of-order execution but paid with far lower throughput: a
RTX 3060 Laptop issues ~10^12 simple ops/s, so the ~10^10 field evaluations of
this 400x400 render take tens of seconds, where a CPU implementation of the same
fractal (for example the Embree based path tracer I wrote for my CG2025 course)
would need minutes for the same sample count. The feature *suffers* from the
divergence on the GPU, but it suffers much less than it would from the CPU's
throughput.

The textures are essentially free: switching both of them off changes the 400x400
render time by less than the ~5% run-to-run spread (25.4 s with, 25.2 s without;
on the analytic scene, 7.3 s with, 7.8 s without). That is not surprising once the
two costs are put side by side: the marble's 5 octaves of trilinear value noise
are ~100 ALU operations *per shading event*, while a single SDF test is ~69
marching steps of an 8-iteration Mandelbulb distance estimate, i.e. tens of
thousands of operations - and textures are evaluated once per bounce rather than
once per marching step. A file-backed texture with bilinear filtering and a bump
map would cost a little more (texture cache traffic, mip selection), but the
measurement here says the same thing the GPU literature does: procedural shading
arithmetic is cheap next to the geometry.

Ideas for further optimizing the shapes, in the order I would try them: march with
a *cone* against a hierarchical set of bounding spheres instead of a single one
(which prunes the tail properly rather than just the empty space), use the
distance estimate to skip the sphere tracing entirely for shadow rays (shadow rays
only need an occluded/not-occluded answer, so they can stop at the first small
step), lower the Mandelbulb's iteration count for screen pixels where the shape is
sub-pixel sized (a distance-based LOD), and precompute a small 3D texture of the
field so that most steps are a texture fetch instead of eight iterations of
trigonometry.

### Restartable rendering

A high quality render here runs for minutes (the cover is 3m55s, the procedural
scene 14m32s) and a film quality one would run for hours, which is a long time to
keep a laptop on. Restartable rendering makes that interruptible: the renderer
writes its state to `<FILE>.ckpt` while it runs, and the next start of the same
scene picks the samples up where they stopped.

**What has to be saved is almost nothing, and that is worth stating precisely.**
Everything else in the renderer is already stateless per iteration: the camera
rays, the path array, the intersection buffer and the compaction scratch space are
all rebuilt from the scene at the top of every iteration, and there is no
acceleration structure to serialise (the scene is a flat list of primitives that
`pathtraceInit` re-uploads). The durable state is exactly

* the accumulation buffer - one `glm::vec3` of un-normalised radiance per pixel,
* how many samples went into it,
* a fingerprint of the scene those samples belong to.

That is a 40 byte header plus `width * height * 12` bytes of payload - 7.3 MB at
800x800:

```cpp
struct CheckpointHeader {
    char magic[8];                 // "P3CKPT01"
    unsigned long long sceneHash;  // FNV-1a over camera, resolution, depth, materials, geoms
    int resolutionX, resolutionY;
    int traceDepth;
    int iterations;                // samples already accumulated
    int headerBytes, reserved;
};
```

The reason a resumed render is *the same* render and not a new one is the RNG
seeding: every sample draws with `(iteration, pixelIndex, depth)`, not with
whatever order the iterations happen in, so iteration 1209 gets exactly the same
random numbers whether it is the 1209th iteration of a fresh process or the first
one after a resume. The scene fingerprint is what keeps that honest: change the
camera, the resolution, the depth or any material in the scene file and the
checkpoint is rejected (`[checkpoint] ... belongs to a different scene, starting
from scratch`), because samples of a different image must not be averaged into
this one. Raising `ITERATIONS` *is* allowed, and is the intended way to extend a
render: the total target is not part of the fingerprint.

The behaviour is controlled by one optional scene field, `CHECKPOINT` (seconds,
default 30, `0` disables the feature):

* while rendering, a checkpoint is written every `CHECKPOINT` seconds;
* `S` and `Esc` write one, next to the PNG they already save;
* finishing a render deletes it - a finished render has nothing to resume;
* the file is written as `<FILE>.ckpt.tmp` and then renamed, so a process killed in
  the middle of a write can never replace a good checkpoint with a truncated one.

```
$ cis565_path_tracer scenes/cornell.json          # stop it whenever
[checkpoint] writing cornell.ckpt every 30.0 s; stop and re-run the same scene to continue where it stopped
[checkpoint] saved at 79 samples
[checkpoint] saved at 170 samples
^C
$ cis565_path_tracer scenes/cornell.json          # same scene file
[checkpoint] resumed cornell.ckpt at 1208 samples
...
Saved cornell.2026-09-20_20-33-43z.1458samp.png.
[checkpoint] resume cost: 0.62 ms to read + upload
```

![](img/checkpoint-analysis.png)

*Left: what one checkpoint costs at 800x800. Middle: stopping a 1458 sample render
at 1208 samples and finishing it later. Right: the difference between the resumed
image and the uninterrupted one - the whole point of the feature is that this
panel is empty.*

### Restartable rendering: cost

**Correctness first**: the resumed image is bit-identical to the uninterrupted
one. Interrupting a 400x400 render of `scenes/cornell.json` at 1208 samples,
restarting it and letting it run to 1458 samples gives `max |difference| = 0` over
all 160,000 pixels compared to a single 1458 sample render, and the two images
agree on their mean radiance to three decimals. A second run interrupted at 1141
samples and finished at 1291 also came out at exactly 0.

**What a checkpoint costs.** The accumulation buffer has to come back over PCIe,
which is the part a CPU renderer does not pay for at all, so it is the part worth
measuring (800x800, 7.32 MB payload, one save per second, timed by the tracer
itself around the copy and the write):

| staging buffer | device -> host | bandwidth | to disk | per checkpoint |
|---|---:|---:|---:|---:|
| page-locked (`cudaHostAlloc` + async copy on a stream) | **0.66 ms** | **10.84 GB/s** | 4.52 ms | 5.2 ms |
| pageable (`malloc` + blocking `cudaMemcpy`) | 0.94 ms | 7.64 GB/s | 4.55 ms | 5.5 ms |

Pinning the staging buffer is worth 1.42x: with pageable memory the driver has to
bounce the copy through a page-locked buffer of its own, which is exactly the copy
this avoids. The disk write dominates either way, and at the default 30 s interval
a checkpoint costs 5.2 ms / 30 s = **0.017%** of the wall clock; at the 1 s
interval used for the table it is still only 0.4%.

**A pass that comes with it.** The base code copied the whole accumulation buffer
back to the host *after every iteration*, because `saveImage()` reads it from
there - 7.32 MB per iteration at 800x800, on the default stream, so it serialised
with the kernels rather than overlapping them. With a known save and checkpoint
schedule that copy only has to happen when one of them is actually due, so
`pathtrace()` no longer does it and `saveImage()` and the checkpointer pull the
image down on demand. The same 7.32 MB transfer measures 0.94 ms through pageable
memory, so the copy the renderer used to do 500 times during a 500 sample render
cost **about 0.47 s, ~2.5% of a 19 s render** at this resolution. End to end that
is inside this machine's ±10-15% run to run spread, so the honest statement is
"removed a measured 0.47 s of transfer", not a wall clock difference - and it
means the numbers reported earlier in this README (stream compaction, depth of
field) were taken with that copy still in place, i.e. they are conservative.

On a hypothetical CPU renderer the same feature is a `memcpy` of a buffer that is
already in the process's own address space, so this is a case where the GPU
version *suffers*: the accumulation buffer lives in device memory and every
checkpoint costs a PCIe round trip that a CPU implementation simply does not have,
and the resume path uploads it back. What makes it cheap anyway is the ratio - 7.3
MB of I/O against 19 s of rendering, and 10.8 GB/s with pinned memory - so the tax
stays below the noise floor instead of becoming a design constraint, while the same
19 s of samples would take a CPU path tracer tens of minutes to produce.

Ways to take it further: overlap the copy with the next iteration by double
buffering (kick off the async copy into staging buffer A, render the next
iteration, write A to disk while the next copy fills B - the pieces exist, only the
ping-pong is missing); write the payload with `OVERLAPPED` file I/O so the 4.5 ms
disk write stops blocking the render loop; compress the payload (fp16 halves it,
but breaks the exact-resume property, so it belongs in a separate "approximate
resume" mode); and on a multi-GPU machine, checkpoint into pinned memory and hand
the buffer to a second device to continue there, which is the same mechanism as a
single-device resume.

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
| `scenes/dof.json` | the depth of field scene: five mirrors receding from the camera, a near diffuse ball and `APERTURE` / `FOCUS` in the camera block |
| `scenes/procedural.json` | the procedural shapes: a marble Mandelbulb and a checkered Menger sponge in a closed room, plus a mirror ball |
| `scenes/procedural_analytic.json` | the same scene with a sphere and a cube sized to the fractals' bounds, for the "what does sphere tracing cost" comparison |

No meshes or texture files are used - the textures are procedural, and the two
complex shapes are distance fields - so nothing has to be downloaded to render
any of them.

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
  sampling model, Paul Bourke's raytracing notes for anti-aliasing, and PBRT v4
  section 5.2.3 (the thin lens model) for the depth of field camera.
* The procedural shapes follow the published distance estimators rather than any
  particular implementation: the power-8 Mandelbulb distance estimate from White
  and Nylander, *Mandelbulb: The Unravelling of the Real 3D Mandelbrot Fractal*
  (2009), and the Menger sponge folding from Inigo Quilez's distance function
  article ([iquilezles.org/articles/distfunctions](https://iquilezles.org/articles/distfunctions/)),
  which is also where the tetrahedron normal trick and the value noise used by the
  marble texture come from.
* The code in this project was written with the help of AI agents (pair
  programming over the CUDA/C++ sources, the build fixes and this README), in line
  with the course's third-party code policy.

