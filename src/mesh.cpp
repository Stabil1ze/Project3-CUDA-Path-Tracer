#include "mesh.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>

#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

namespace
{
    // One corner of a face: an index into the position array and, when the file
    // has them, an index into the normal array (-1 when it does not). The OBJ and
    // PLY readers both produce these, so everything past parsing is shared.
    struct MeshCorner
    {
        int position = -1;
        int normal = -1;
    };

    typedef std::array<MeshCorner, 3> MeshTriangle;

    // "1", "1/2", "1//3", "1/2/3" and the negative (relative) forms of all four.
    MeshCorner parseCorner(const std::string& token, size_t positionCount, size_t normalCount)
    {
        MeshCorner corner;
        int field = 0;
        std::string value;
        std::istringstream fields(token);
        while (std::getline(fields, value, '/'))
        {
            if (!value.empty() && field != 1)          // field 1 is the texture index, ignored
            {
                const int count = (int)(field == 0 ? positionCount : normalCount);
                int index = std::atoi(value.c_str());
                if (index < 0)                          // relative to the end of the array
                {
                    index = count + index + 1;
                }
                if (index <= 0)
                {
                    return MeshCorner{ -1, -1 };        // 0 and out of range: unusable
                }
                if (field == 0)
                {
                    corner.position = index - 1;
                }
                else
                {
                    corner.normal = index - 1;
                }
            }
            field++;
        }
        return corner;
    }

    // ------------------------------------------------------------------ PLY ----

    // The scalar types a PLY header can name. In a binary file the size decides
    // how many bytes one value occupies, so the size is all the reader needs.
    enum class PlyType { Int8, Uint8, Int16, Uint16, Int32, Uint32, Float32, Float64, Unknown };

    PlyType plyTypeFromName(const std::string& name)
    {
        if (name == "char" || name == "int8")       return PlyType::Int8;
        if (name == "uchar" || name == "uint8")     return PlyType::Uint8;
        if (name == "short" || name == "int16")     return PlyType::Int16;
        if (name == "ushort" || name == "uint16")   return PlyType::Uint16;
        if (name == "int" || name == "int32")       return PlyType::Int32;
        if (name == "uint" || name == "uint32")     return PlyType::Uint32;
        if (name == "float" || name == "float32")   return PlyType::Float32;
        if (name == "double" || name == "float64")  return PlyType::Float64;
        return PlyType::Unknown;
    }

    size_t plyTypeSize(PlyType type)
    {
        switch (type)
        {
        case PlyType::Int8:
        case PlyType::Uint8:    return 1;
        case PlyType::Int16:
        case PlyType::Uint16:   return 2;
        case PlyType::Int32:
        case PlyType::Uint32:
        case PlyType::Float32:  return 4;
        case PlyType::Float64:  return 8;
        default:                return 0;
        }
    }

    // One entry of an element. `list` properties (only faces really use them)
    // carry their own count type and element type, and both can be any scalar.
    struct PlyProperty
    {
        std::string name;
        PlyType type = PlyType::Unknown;        // scalar properties
        bool isList = false;
        PlyType countType = PlyType::Unknown;   // list properties
        PlyType itemType = PlyType::Unknown;
    };

    struct PlyElement
    {
        std::string name;
        size_t count = 0;
        std::vector<PlyProperty> properties;
    };

    void swapBytes(unsigned char* bytes, size_t size)
    {
        for (size_t i = 0; i < size / 2; i++)
        {
            const unsigned char tmp = bytes[i];
            bytes[i] = bytes[size - 1 - i];
            bytes[size - 1 - i] = tmp;
        }
    }

    /** Reads the PLY body, hiding the three formats a header can declare.
     *  Everything comes back as a double: exact for PLY's integer types and for
     *  float32 vertex data. */
    class PlyStream
    {
    public:
        PlyStream(std::ifstream& file, bool binary, bool bigEndian)
            : file(file), binary(binary), bigEndian(bigEndian)
        {
        }

        // False on a truncated file, so callers stop instead of reading whatever
        // happens to sit behind the end of the data.
        bool readScalar(PlyType type, double& value)
        {
            const size_t size = plyTypeSize(type);
            if (size == 0)
            {
                return false;
            }
            if (!binary)
            {
                return (bool)(file >> value);           // ascii: one number per token
            }

            unsigned char bytes[8];
            file.read((char*)bytes, size);
            if ((size_t)file.gcount() != size)
            {
                return false;
            }
            if (bigEndian)
            {
                swapBytes(bytes, size);
            }
            switch (type)
            {
            case PlyType::Int8:     { int8_t v = 0;     std::memcpy(&v, bytes, 1); value = (double)v; break; }
            case PlyType::Uint8:    { uint8_t v = 0;    std::memcpy(&v, bytes, 1); value = (double)v; break; }
            case PlyType::Int16:    { int16_t v = 0;    std::memcpy(&v, bytes, 2); value = (double)v; break; }
            case PlyType::Uint16:   { uint16_t v = 0;   std::memcpy(&v, bytes, 2); value = (double)v; break; }
            case PlyType::Int32:    { int32_t v = 0;    std::memcpy(&v, bytes, 4); value = (double)v; break; }
            case PlyType::Uint32:   { uint32_t v = 0;   std::memcpy(&v, bytes, 4); value = (double)v; break; }
            case PlyType::Float32:  { float v = 0.0f;   std::memcpy(&v, bytes, 4); value = (double)v; break; }
            case PlyType::Float64:  { double v = 0.0;   std::memcpy(&v, bytes, 8); value = v; break; }
            default:                return false;
            }
            return true;
        }

        // Counts and vertex indices, kept in an integer so a large mesh cannot
        // lose precision on the way in.
        bool readInteger(PlyType type, long long& value)
        {
            if (!binary)
            {
                return (bool)(file >> value);
            }
            double scalar = 0.0;
            if (!readScalar(type, scalar))
            {
                return false;
            }
            value = (long long)(scalar < 0.0 ? scalar - 0.5 : scalar + 0.5);
            return true;
        }

    private:
        std::ifstream& file;
        bool binary;
        bool bigEndian;
    };

    /** Turns parsed triangles into one TRIANGLE geometry each and prints the
     *  summary both readers share, so an OBJ and a PLY of the same mesh behave
     *  identically in the scene. Returns the triangles the file declared, not the
     *  number that survived. */
    int appendMeshGeometry(
        const std::string& label,
        const std::vector<glm::vec3>& positions,
        const std::vector<glm::vec3>& normals,
        const std::vector<MeshTriangle>& triangles,
        int polygonFaces,
        int droppedTriangles,
        double parseMs,
        const glm::vec3& translation,
        const glm::vec3& rotation,
        const glm::vec3& scale,
        int materialId,
        std::vector<Geom>& geoms)
    {
        const glm::mat4 transform = utilityCore::buildTransformationMatrix(translation, rotation, scale);
        const glm::mat4 inverseTransform = glm::inverse(transform);
        const glm::mat4 invTranspose = glm::inverseTranspose(transform);

        int smoothTriangles = 0;
        glm::vec3 boundsMin(FLT_MAX);
        glm::vec3 boundsMax(-FLT_MAX);
        geoms.reserve(geoms.size() + triangles.size());
        for (const MeshTriangle& triangle : triangles)
        {
            // The parser returns -1 for a corner it could not resolve, and a file
            // can always have an index past the end of its own vertex array.
            const bool usable = triangle[0].position >= 0 && triangle[1].position >= 0
                && triangle[2].position >= 0
                && (size_t)triangle[0].position < positions.size()
                && (size_t)triangle[1].position < positions.size()
                && (size_t)triangle[2].position < positions.size();
            if (!usable)
            {
                droppedTriangles++;
                continue;
            }

            Geom geom;
            geom.type = TRIANGLE;
            geom.materialid = materialId;
            geom.translation = translation;
            geom.rotation = rotation;
            geom.scale = scale;
            geom.transform = transform;
            geom.inverseTransform = inverseTransform;
            geom.invTranspose = invTranspose;

            const glm::vec3 corners[3] = { positions[triangle[0].position],
                                           positions[triangle[1].position],
                                           positions[triangle[2].position] };
            geom.v0 = corners[0];
            geom.v1 = corners[1];
            geom.v2 = corners[2];

            const bool smooth = triangle[0].normal >= 0 && triangle[1].normal >= 0
                && triangle[2].normal >= 0
                && (size_t)triangle[0].normal < normals.size()
                && (size_t)triangle[1].normal < normals.size()
                && (size_t)triangle[2].normal < normals.size();
            if (smooth)
            {
                geom.n0 = normals[triangle[0].normal];
                geom.n1 = normals[triangle[1].normal];
                geom.n2 = normals[triangle[2].normal];
                smoothTriangles++;
            }
            else
            {
                // No normals: one geometric normal for the whole face, so the mesh
                // renders faceted instead of with a normal of (0, 0, 0).
                const glm::vec3 normal = glm::normalize(
                    glm::cross(geom.v1 - geom.v0, geom.v2 - geom.v0));
                geom.n0 = geom.n1 = geom.n2 = normal;
            }

            for (int i = 0; i < 3; i++)
            {
                boundsMin = glm::min(boundsMin, corners[i]);
                boundsMax = glm::max(boundsMax, corners[i]);
            }
            geoms.push_back(geom);
        }

        std::cout << "[mesh] " << label << ": " << triangles.size() << " triangles from "
            << positions.size() << " vertices (" << smoothTriangles << " with interpolated normals, "
            << (triangles.size() - smoothTriangles) << " flat)"
            << (polygonFaces > 0 ? ", " + std::to_string(polygonFaces) + " polygons fan triangulated" : "")
            << (droppedTriangles > 0 ? ", " + std::to_string(droppedTriangles) + " malformed faces skipped" : "")
            << ", object space bounds (" << boundsMin.x << " " << boundsMin.y << " " << boundsMin.z
            << ") to (" << boundsMax.x << " " << boundsMax.y << " " << boundsMax.z << "), "
            << parseMs << " ms" << std::endl;

        return (int)triangles.size();
    }
}

int loadObjMesh(
    const std::string& path,
    const glm::vec3& translation,
    const glm::vec3& rotation,
    const glm::vec3& scale,
    int materialId,
    std::vector<Geom>& geoms)
{
    const auto start = std::chrono::steady_clock::now();

    std::ifstream file(path);
    if (!file)
    {
        std::cout << "[mesh] cannot open " << path << std::endl;
        return -1;
    }

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<MeshTriangle> triangles;
    int droppedTriangles = 0;
    int polygonFaces = 0;

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#' || line[0] == '\r')
        {
            continue;
        }

        std::istringstream fields(line);
        std::string tag;
        fields >> tag;
        if (tag == "v")
        {
            glm::vec3 position;
            fields >> position.x >> position.y >> position.z;
            positions.push_back(position);
        }
        else if (tag == "vn")
        {
            glm::vec3 normal;
            fields >> normal.x >> normal.y >> normal.z;
            normals.push_back(normal);
        }
        else if (tag == "f")
        {
            std::vector<MeshCorner> face;
            std::string token;
            while (fields >> token)
            {
                face.push_back(parseCorner(token, positions.size(), normals.size()));
            }
            if (face.size() < 3)
            {
                droppedTriangles++;
                continue;
            }
            if (face.size() > 3)
            {
                polygonFaces++;
            }
            // Fan triangulation: (0, k, k + 1) for every other vertex of the
            // polygon, which is exact for convex faces and what exporters expect.
            for (size_t k = 1; k + 1 < face.size(); k++)
            {
                triangles.push_back({ face[0], face[k], face[k + 1] });
            }
        }
    }

    const double parseMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return appendMeshGeometry(path, positions, normals, triangles, polygonFaces,
        droppedTriangles, parseMs, translation, rotation, scale, materialId, geoms);
}

int loadPlyMesh(
    const std::string& path,
    const glm::vec3& translation,
    const glm::vec3& rotation,
    const glm::vec3& scale,
    int materialId,
    std::vector<Geom>& geoms)
{
    const auto start = std::chrono::steady_clock::now();

    // Binary, because the body of a binary PLY is not text; the header is plain
    // lines in all three formats, so it is read the same way.
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::cout << "[mesh] cannot open " << path << std::endl;
        return -1;
    }

    bool binary = false;
    bool bigEndian = false;
    bool sawMagic = false;
    std::vector<PlyElement> elements;

    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        std::istringstream fields(line);
        std::string keyword;
        if (!(fields >> keyword))
        {
            continue;
        }

        if (keyword == "ply")
        {
            sawMagic = true;
        }
        else if (keyword == "comment" || keyword == "obj_info")
        {
            continue;                                   // free text, no structure
        }
        else if (keyword == "format")
        {
            std::string format;
            fields >> format;
            if (format == "ascii")
            {
                binary = false;
            }
            else if (format == "binary_little_endian")
            {
                binary = true;
                bigEndian = false;
            }
            else if (format == "binary_big_endian")
            {
                binary = true;
                bigEndian = true;
            }
            else
            {
                std::cout << "[mesh] " << path << ": unsupported PLY format '"
                    << format << "'" << std::endl;
                return -1;
            }
        }
        else if (keyword == "element")
        {
            PlyElement element;
            fields >> element.name >> element.count;
            elements.push_back(element);
        }
        else if (keyword == "property")
        {
            if (elements.empty())
            {
                std::cout << "[mesh] " << path
                    << ": PLY property before any element" << std::endl;
                return -1;
            }

            std::string typeName;
            fields >> typeName;
            PlyProperty property;
            if (typeName == "list")
            {
                std::string countName;
                std::string itemName;
                fields >> countName >> itemName >> property.name;
                property.isList = true;
                property.countType = plyTypeFromName(countName);
                property.itemType = plyTypeFromName(itemName);
            }
            else
            {
                fields >> property.name;
                property.type = plyTypeFromName(typeName);
            }

            const bool known = property.isList
                ? property.countType != PlyType::Unknown && property.itemType != PlyType::Unknown
                : property.type != PlyType::Unknown;
            if (!known)
            {
                std::cout << "[mesh] " << path << ": unsupported PLY property type '"
                    << typeName << "'" << std::endl;
                return -1;
            }
            elements.back().properties.push_back(property);
        }
        else if (keyword == "end_header")
        {
            break;
        }
        // Any other header keyword is a hint we have no use for.
    }

    if (!sawMagic || elements.empty())
    {
        std::cout << "[mesh] " << path << ": not a PLY file" << std::endl;
        return -1;
    }

    const std::string formatName = binary
        ? (bigEndian ? "binary_big_endian" : "binary_little_endian")
        : "ascii";

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<MeshTriangle> triangles;
    std::vector<long long> faceCorners;
    bool vertexNormals = false;      // set when the vertex element declares nx/ny/nz
    bool truncated = false;
    int polygonFaces = 0;
    int droppedFaces = 0;

    PlyStream stream(file, binary, bigEndian);

    for (const PlyElement& element : elements)
    {
        const bool isVertex = element.name == "vertex";
        const bool isFace = element.name == "face";

        // A PLY file is free to order and name its properties, so the ones we know
        // are found by name and everything else is read and dropped.
        int xIndex = -1, yIndex = -1, zIndex = -1;
        int nxIndex = -1, nyIndex = -1, nzIndex = -1;
        int listIndex = -1;
        for (size_t p = 0; p < element.properties.size(); p++)
        {
            const PlyProperty& property = element.properties[p];
            if (!property.isList)
            {
                if (property.name == "x")       xIndex = (int)p;
                else if (property.name == "y")  yIndex = (int)p;
                else if (property.name == "z")  zIndex = (int)p;
                else if (property.name == "nx") nxIndex = (int)p;
                else if (property.name == "ny") nyIndex = (int)p;
                else if (property.name == "nz") nzIndex = (int)p;
            }
            else if (isFace
                && (listIndex < 0 || property.name == "vertex_indices"
                    || property.name == "vertex_index"))
            {
                listIndex = (int)p;                     // the face's vertex ring
            }
        }

        if (isVertex)
        {
            vertexNormals = nxIndex >= 0 && nyIndex >= 0 && nzIndex >= 0;
            if (xIndex < 0 || yIndex < 0 || zIndex < 0)
            {
                std::cout << "[mesh] " << path
                    << ": PLY vertex element without x/y/z" << std::endl;
                return -1;
            }
            positions.reserve(positions.size() + element.count);
            if (vertexNormals)
            {
                normals.reserve(normals.size() + element.count);
            }
        }

        for (size_t item = 0; item < element.count; item++)
        {
            glm::vec3 position(0.0f);
            glm::vec3 normal(0.0f);

            for (size_t p = 0; p < element.properties.size(); p++)
            {
                const PlyProperty& property = element.properties[p];
                if (property.isList)
                {
                    long long count = 0;
                    if (!stream.readInteger(property.countType, count) || count < 0)
                    {
                        truncated = true;
                        break;
                    }

                    const bool keep = isFace && (int)p == listIndex;
                    if (keep)
                    {
                        faceCorners.clear();
                    }
                    for (long long k = 0; k < count; k++)
                    {
                        long long index = 0;
                        if (!stream.readInteger(property.itemType, index))
                        {
                            truncated = true;
                            break;
                        }
                        if (keep)
                        {
                            faceCorners.push_back(index);
                        }
                    }
                    if (truncated)
                    {
                        break;
                    }
                    continue;
                }

                double value = 0.0;
                if (!stream.readScalar(property.type, value))
                {
                    truncated = true;
                    break;
                }
                if (!isVertex)
                {
                    continue;
                }
                if ((int)p == xIndex)       position.x = (float)value;
                else if ((int)p == yIndex)  position.y = (float)value;
                else if ((int)p == zIndex)  position.z = (float)value;
                else if ((int)p == nxIndex) normal.x = (float)value;
                else if ((int)p == nyIndex) normal.y = (float)value;
                else if ((int)p == nzIndex) normal.z = (float)value;
            }

            if (truncated)
            {
                break;
            }

            if (isVertex)
            {
                positions.push_back(position);
                if (vertexNormals)
                {
                    normals.push_back(normal);
                }
            }
            else if (isFace)
            {
                if (listIndex < 0 || faceCorners.size() < 3)
                {
                    droppedFaces++;                     // a face without a vertex ring
                    continue;
                }
                if (faceCorners.size() > 3)
                {
                    polygonFaces++;
                }
                // The same fan triangulation the OBJ reader uses, so a quad or an
                // n-gon splits identically whichever format describes it.
                for (size_t k = 1; k + 1 < faceCorners.size(); k++)
                {
                    const long long corners[3] = { faceCorners[0], faceCorners[k], faceCorners[k + 1] };
                    MeshTriangle triangle;
                    for (int c = 0; c < 3; c++)
                    {
                        triangle[c].position = (int)corners[c];
                        triangle[c].normal = vertexNormals ? (int)corners[c] : -1;
                    }
                    triangles.push_back(triangle);
                }
            }
        }

        if (truncated)
        {
            std::cout << "[mesh] " << path << ": PLY ends inside the '"
                << element.name << "' element, keeping what was read" << std::endl;
            break;
        }
    }

    const std::string label = path + " (PLY " + formatName + ")";
    const double parseMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return appendMeshGeometry(label, positions, normals, triangles, polygonFaces,
        droppedFaces, parseMs, translation, rotation, scale, materialId, geoms);
}

int loadMesh(
    const std::string& path,
    const glm::vec3& translation,
    const glm::vec3& rotation,
    const glm::vec3& scale,
    int materialId,
    std::vector<Geom>& geoms)
{
    const size_t dot = path.find_last_of('.');
    std::string extension = dot == std::string::npos ? std::string() : path.substr(dot);
    for (char& c : extension)
    {
        c = (char)std::tolower((unsigned char)c);
    }

    if (extension == ".ply")
    {
        return loadPlyMesh(path, translation, rotation, scale, materialId, geoms);
    }
    if (!extension.empty() && extension != ".obj")
    {
        std::cout << "[mesh] " << path << ": unknown extension '" << extension
            << "', reading it as OBJ" << std::endl;
    }
    return loadObjMesh(path, translation, rotation, scale, materialId, geoms);
}
