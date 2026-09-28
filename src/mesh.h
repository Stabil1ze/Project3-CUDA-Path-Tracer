#pragma once

#include "sceneStructs.h"

#include <string>
#include <vector>

/** Load a Wavefront OBJ and append one TRIANGLE geometry per triangle: `v`, `vn`
 *  and `f` with the four corner forms and negative indices, polygons fan
 *  triangulated, everything else the exporter writes skipped, and a missing
 *  normal array replaced by the geometric face normal (flat shading). The
 *  vertices stay in the mesh's own space and every triangle carries the same
 *  transform the scene gave the mesh, so a mesh behaves exactly like an analytic
 *  primitive. Returns the triangles appended, or -1 if the file cannot be read. */
int loadObjMesh(
    const std::string& path,
    const glm::vec3& translation,
    const glm::vec3& rotation,
    const glm::vec3& scale,
    int materialId,
    std::vector<Geom>& geoms);

/** Load a PLY mesh, one TRIANGLE geometry per triangle. Handles the three
 *  formats a header can declare (ascii, little endian, big endian) for the subset
 *  mesh files use: a `vertex` element with x/y/z and optionally nx/ny/nz and u/v,
 *  plus a `face` element with a vertex index list. Property order and the list's
 *  scalar types come from the header; other elements are read and discarded; a
 *  file without normals falls back to flat shading. Returns the triangles
 *  appended, or -1 if the file cannot be read. */
int loadPlyMesh(
    const std::string& path,
    const glm::vec3& translation,
    const glm::vec3& rotation,
    const glm::vec3& scale,
    int materialId,
    std::vector<Geom>& geoms);

/** Pick the reader from the file extension (`.obj`, `.ply`). An unknown one is
 *  reported and read as OBJ, as every scene before PLY support did. */
int loadMesh(
    const std::string& path,
    const glm::vec3& translation,
    const glm::vec3& rotation,
    const glm::vec3& scale,
    int materialId,
    std::vector<Geom>& geoms);
