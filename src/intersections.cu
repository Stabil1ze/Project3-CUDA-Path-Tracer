#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

// ---------------------------------------------------------------------------
// Mesh triangles
// ---------------------------------------------------------------------------

__host__ __device__ float triangleIntersectionTest(
    Geom triangle,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    Ray q;
    q.origin = multiplyMV(triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));

    const glm::vec3 edge1 = triangle.v1 - triangle.v0;
    const glm::vec3 edge2 = triangle.v2 - triangle.v0;
    const glm::vec3 pvec = glm::cross(q.direction, edge2);

    const float determinant = glm::dot(edge1, pvec);
    if (glm::abs(determinant) < 1e-12f) { return -1.0f; }

    const float inverseDeterminant = 1.0f / determinant;
    const glm::vec3 tvec = q.origin - triangle.v0;
    const float u = glm::dot(tvec, pvec) * inverseDeterminant;
    if (u < 0.0f || u > 1.0f) { return -1.0f; }

    const glm::vec3 qvec = glm::cross(tvec, edge1);
    const float v = glm::dot(q.direction, qvec) * inverseDeterminant;
    if (v < 0.0f || u + v > 1.0f) { return -1.0f; }

    const float t = glm::dot(edge2, qvec) * inverseDeterminant;
    if (t < 1e-4f) { return -1.0f; }

    const glm::vec3 localPoint = q.origin + t * q.direction;
    const glm::vec3 localNormal = triangle.n0 * (1.0f - u - v) + triangle.n1 * u + triangle.n2 * v;
    const glm::vec3 geometricNormal = glm::cross(edge1, edge2);

    intersectionPoint = multiplyMV(triangle.transform, glm::vec4(localPoint, 1.0f));
    normal = glm::normalize(multiplyMV(triangle.invTranspose, glm::vec4(localNormal, 0.0f)));

    const glm::vec3 faceNormal = multiplyMV(triangle.invTranspose, glm::vec4(geometricNormal, 0.0f));
    outside = glm::dot(faceNormal, r.direction) < 0.0f;
    if (!outside) { normal = -normal; }

    return glm::length(r.origin - intersectionPoint);
}

// ---------------------------------------------------------------------------
// Procedural shapes
// ---------------------------------------------------------------------------

__host__ __device__ inline float sdBox(glm::vec3 p, glm::vec3 b)
{
    glm::vec3 q = glm::abs(p) - b;
    return glm::length(glm::max(q, glm::vec3(0.0f)))
        + glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
}

// Mandelbulb SDF from https://www.iquilezles.org/www/articles/mandelbulb/mandelbulb.htm
__host__ __device__ inline float mandelbulbSDF(glm::vec3 p)
{
    constexpr float POWER = 8.0f;
    constexpr int MAX_ITER = 8;
    constexpr float ESCAPE_RADIUS = 2.0f;

    glm::vec3 z = p;
    float dr = 1.0f;
    float r = 1e-6f;
    for (int i = 0; i < MAX_ITER; i++)
    {
        r = glm::max(glm::length(z), 1e-6f);
        if (r > ESCAPE_RADIUS)
        {
            break;
        }
        float theta = acosf(glm::clamp(z.z / r, -1.0f, 1.0f));
        float phi = atan2f(z.y, z.x);
        dr = powf(r, POWER - 1.0f) * POWER * dr + 1.0f;
        float zr = powf(r, POWER);
        theta *= POWER;
        phi *= POWER;
        z = zr * glm::vec3(sinf(theta) * cosf(phi),
                           sinf(theta) * sinf(phi),
                           cosf(theta)) + p;
    }
    return 0.5f * logf(r) * r / dr;
}

__host__ __device__ inline float mengerSDF(glm::vec3 p)
{
    constexpr int MAX_ITER = 4;

    float d = sdBox(p, glm::vec3(1.0f));
    float s = 1.0f;
    for (int i = 0; i < MAX_ITER; i++)
    {
        glm::vec3 a = glm::mod(p * s, glm::vec3(2.0f)) - glm::vec3(1.0f);
        s *= 3.0f;
        glm::vec3 r = glm::abs(glm::vec3(1.0f) - 3.0f * glm::abs(a));
        float da = glm::max(r.x, r.y);
        float db = glm::max(r.y, r.z);
        float dc = glm::max(r.z, r.x);
        float c = (glm::min(da, glm::min(db, dc)) - 1.0f) / s;
        d = glm::max(d, c);
    }
    return d;
}

__host__ __device__ float sdfEvaluate(int geomType, glm::vec3 p)
{
    return (geomType == MANDELBULB) ? mandelbulbSDF(p) : mengerSDF(p);
}

__host__ __device__ inline float geomMaxScale(Geom geom)
{
    float sx = glm::length(glm::vec3(geom.transform[0]));
    float sy = glm::length(glm::vec3(geom.transform[1]));
    float sz = glm::length(glm::vec3(geom.transform[2]));
    return glm::max(sx, glm::max(sy, sz));
}

__host__ __device__ float sdfIntersectionTest(
    Geom geom,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    unsigned long long* stepCounter,
    unsigned long long* histogram)
{
    constexpr int MAX_STEPS = 128;
    constexpr int HISTOGRAM_BUCKETS = 16;
    constexpr int STEPS_PER_BUCKET = 8;
    constexpr float HIT_EPSILON = 1e-4f;

    const glm::vec3 rayOrigin = r.origin;
    const glm::vec3 rayDirection = glm::normalize(r.direction);
    const float maxScale = geomMaxScale(geom);
    const float stepScale = 1.0f / glm::max(maxScale, 1e-6f);
    const glm::vec3 center = glm::vec3(geom.transform[3]);
    const float shapeRadius = (geom.type == MANDELBULB) ? 1.3f : 1.7320508f;  // sqrt(3)
    const float boundRadius = shapeRadius * maxScale;

    // Broad phase
#if SDF_BOUNDING_SPHERE
    {
        glm::vec3 oc = rayOrigin - center;
        float b = glm::dot(oc, rayDirection);
        float c = glm::dot(oc, oc) - boundRadius * boundRadius;
        float disc = b * b - c;
        if (disc < 0.0f || (-b + sqrtf(disc)) <= 0.0f)
        {
            return -1.0f;
        }
    }
#endif

    float t = 0.0f;
    int steps = 0;
    bool hit = false;
    for (int i = 0; i < MAX_STEPS; i++)
    {
        glm::vec3 pWorld = rayOrigin + t * rayDirection;
        float d = sdfEvaluate(geom.type, multiplyMV(geom.inverseTransform, glm::vec4(pWorld, 1.0f)));
        steps++;
        if (glm::abs(d) < HIT_EPSILON)
        {
            hit = true;
            break;
        }
        t += glm::max(d * stepScale, HIT_EPSILON);
    }

#ifdef __CUDA_ARCH__
    if (stepCounter != NULL)
    {
        atomicAdd(stepCounter, (unsigned long long)steps);
    }
    if (histogram != NULL)
    {
        // 64 bit: a 800x800/3000spp render produces ~5e9 marches, which wraps a
        // 32 bit bucket and silently corrupts the reported average.
        int bucket = glm::min(steps / STEPS_PER_BUCKET, HISTOGRAM_BUCKETS - 1);
        atomicAdd(&histogram[bucket], 1ull);
    }
#endif

    if (!hit)
    {
        return -1;
    }

    glm::vec3 pWorld = rayOrigin + t * rayDirection;
    glm::vec3 pObj = multiplyMV(geom.inverseTransform, glm::vec4(pWorld, 1.0f));

    // Surface normal from the SDF gradient
    const float h = 1e-4f;
    glm::vec3 grad = glm::vec3(0.0f);
    grad += glm::vec3( 1.0f, -1.0f, -1.0f) * sdfEvaluate(geom.type, pObj + glm::vec3( 1.0f, -1.0f, -1.0f) * h);
    grad += glm::vec3(-1.0f, -1.0f,  1.0f) * sdfEvaluate(geom.type, pObj + glm::vec3(-1.0f, -1.0f,  1.0f) * h);
    grad += glm::vec3(-1.0f,  1.0f, -1.0f) * sdfEvaluate(geom.type, pObj + glm::vec3(-1.0f,  1.0f, -1.0f) * h);
    grad += glm::vec3( 1.0f,  1.0f,  1.0f) * sdfEvaluate(geom.type, pObj + glm::vec3( 1.0f,  1.0f,  1.0f) * h);
    grad = glm::normalize(grad);
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(grad, 0.0f)));

    // Sphere tracing assumes it starts outside
    // if the ray origin is already below the surface, report the hit from the inside
    outside = true;
    glm::vec3 originObj = multiplyMV(geom.inverseTransform, glm::vec4(rayOrigin, 1.0f));
    if (sdfEvaluate(geom.type, originObj) < 0.0f)
    {
        outside = false;
        normal = -normal;
    }

    intersectionPoint = pWorld;
    return glm::length(rayOrigin - pWorld);
}

// ---------------------------------------------------------------------------
// World space bounds of a geometry
// ---------------------------------------------------------------------------

// Compute the world space axis-aligned bounding box of a geometry
__host__ void geomWorldBounds(const Geom& geom, glm::vec3& boundsMin, glm::vec3& boundsMax)
{
    boundsMin = glm::vec3(FLT_MAX);
    boundsMax = glm::vec3(-FLT_MAX);

    if (geom.type == TRIANGLE)
    {
        const glm::vec3 corners[3] = {
            multiplyMV(geom.transform, glm::vec4(geom.v0, 1.0f)),
            multiplyMV(geom.transform, glm::vec4(geom.v1, 1.0f)),
            multiplyMV(geom.transform, glm::vec4(geom.v2, 1.0f)) };
        for (int i = 0; i < 3; i++)
        {
            boundsMin = glm::min(boundsMin, corners[i]);
            boundsMax = glm::max(boundsMax, corners[i]);
        }
        return;
    }

    if (geom.type == CUBE)
    {
        for (int corner = 0; corner < 8; corner++)
        {
            const glm::vec3 unit((corner & 1) ? 0.5f : -0.5f, (corner & 2) ? 0.5f : -0.5f,
                (corner & 4) ? 0.5f : -0.5f);
            const glm::vec3 world = multiplyMV(geom.transform, glm::vec4(unit, 1.0f));
            boundsMin = glm::min(boundsMin, world);
            boundsMax = glm::max(boundsMax, world);
        }
        return;
    }

    const glm::vec3 center = multiplyMV(geom.transform, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    glm::vec3 extent(0.5f);
    if (geom.type == SPHERE)
    {
        // glm is column major: m[column][row], so a row of the linear part is
        // (m[0][r], m[1][r], m[2][r]).
        for (int axis = 0; axis < 3; axis++)
        {
            const glm::vec3 row(geom.transform[0][axis], geom.transform[1][axis],
                geom.transform[2][axis]);
            extent[axis] = 0.5f * glm::length(row);
        }
    }
    else
    {
        const float shapeRadius = (geom.type == MANDELBULB) ? 1.3f : 1.7320508f;  // sqrt(3)
        extent = glm::vec3(shapeRadius * geomMaxScale(geom));
    }

    boundsMin = center - extent;
    boundsMax = center + extent;
}
