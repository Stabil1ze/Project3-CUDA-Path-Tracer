#include "glslUtility.hpp"
#include "denoise.h"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
    // The scene picker is the OS file dialog: there is no third party browser to
    // vendor, it filters by extension itself, and GLFW_EXPOSE_NATIVE_WIN32 gives us
    // the window handle so it opens over the renderer instead of behind it.
    #define GLFW_EXPOSE_NATIVE_WIN32
    #include "GLFW/glfw3native.h"
    #include <windows.h>
    #include <commdlg.h>
#endif

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
// Restartable rendering: glfwGetTime() of the last checkpoint write.
static double lastCheckpointTime = 0.0;
static double lastFrameTime = 0.0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

// Interactive scene loading
//
// A scene on the command line keeps the old behaviour exactly: render it, save the
// image and exit, which is what the measurement runs rely on. Started with no
// argument the renderer opens empty instead, with a "Load data" button in the top
// right corner; a scene picked there is loaded into the running process and starts
// accumulating immediately, and finishing it keeps the window open.
static bool guiMode = false;
static bool renderActive = false;       // the sample loop should advance
static bool renderFinished = false;     // reached ITS iterations, image kept
static std::string scenePath;
static std::string statusMessage = "No scene loaded - click Load data.";
static std::string dialogDir;           // remembered between file dialogs
static bool sceneNeedsFullInit = true;  // the next frame at sample 0 has to (re)allocate
static bool sceneEdited = false;        // the panel changed a value: restart accumulation
static Camera homeCamera;               // as the scene file wrote it, for Reset camera
static std::vector<int> panelLightIds;  // emitting materials, captured when a scene loads

// Forward declarations for window loop and interactivity
void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures()
{
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void)
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo)
{
    if (pbo)
    {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda()
{
    if (pbo)
    {
        deletePBO(&pbo);
    }
    if (displayImage)
    {
        deleteTexture(&displayImage);
    }
}

void initCuda()
{
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO()
{
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);

    // Leave the unpack target unbound. ImGui uploads its font atlas with one
    // glTexImage2D on the first frame, and while a pixel unpack buffer is bound the
    // pixel pointer is read as an offset into that buffer instead of a host address:
    // the upload fails with GL_INVALID_OPERATION, the atlas texture keeps no storage
    // and every ImGui draw (window, button, glyph) samples as black.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

// ---------------------------------------------------------------------------
// Loading a scene into the running renderer
// ---------------------------------------------------------------------------

/** The camera's vertical field of view drives three derived fields; the JSON loader
 *  computes them the same way, so the panel has to redo it when the slider moves. */
void applyVerticalFov(Camera& cam, float fovyDegrees)
{
    const float yscaled = tanf(glm::radians(fovyDegrees));
    const float xscaled = yscaled * (float)cam.resolution.x / (float)cam.resolution.y;
    cam.fov = glm::vec2(glm::degrees(atanf(xscaled)), fovyDegrees);
    cam.pixelLength = glm::vec2(2.0f * xscaled / (float)cam.resolution.x,
        2.0f * yscaled / (float)cam.resolution.y);
}

/** The orbit parameters the mouse drags, read back out of a camera. */
void orbitFromCamera(const Camera& cam, glm::vec3& lookAt, float& distance, float& polar,
    float& azimuth)
{
    lookAt = cam.lookAt;
    const glm::vec3 eyeOffset = cam.position - cam.lookAt;
    distance = glm::length(eyeOffset);
    const glm::vec3 forward = -eyeOffset / distance;
    polar = glm::acos(glm::clamp(-forward.y, -1.0f, 1.0f));
    azimuth = glm::atan(-forward.x, -forward.z);
}

/** Elevation and azimuth of the direction the sun light comes from. */
void sunAngles(const glm::vec3& direction, float& elevationDegrees, float& azimuthDegrees)
{
    const glm::vec3 toward = glm::normalize(-direction);
    elevationDegrees = glm::degrees(glm::asin(glm::clamp(toward.y, -1.0f, 1.0f)));
    azimuthDegrees = glm::degrees(glm::atan(toward.x, toward.z));
}

glm::vec3 sunDirection(float elevationDegrees, float azimuthDegrees)
{
    const float elevation = glm::radians(elevationDegrees);
    const float azimuth = glm::radians(azimuthDegrees);
    const glm::vec3 toward(glm::cos(elevation) * glm::sin(azimuth), glm::sin(elevation),
        glm::cos(elevation) * glm::cos(azimuth));
    return glm::normalize(-toward);
}

/** The materials that light this scene, captured once per load so that the panel
 *  keeps their rows even when one of them is turned down to zero. */
void captureEmitters(const Scene* scene)
{
    panelLightIds.clear();
    for (size_t i = 0; i < scene->materials.size(); i++)
    {
        if (scene->materials[i].emittance > 0.0f)
        {
            panelLightIds.push_back((int)i);
        }
    }
}

#ifdef _WIN32
std::string pickSceneFile()
{
    char fileName[MAX_PATH] = "";

    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = glfwGetWin32Window(window);
    ofn.lpstrFilter = "Scene (*.json)\0*.json\0All files (*.*)\0*.*\0";
    ofn.lpstrFile = fileName;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "Load data";
    ofn.lpstrInitialDir = dialogDir.empty() ? NULL : dialogDir.c_str();
    // OFN_NOCHANGEDIR: the renderer writes its image and checkpoint next to the
    // working directory, so the dialog must not move it.
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameA(&ofn) == TRUE)
    {
        return std::string(fileName);
    }
    return std::string();
}
#endif

bool loadScene(const std::string& path)
{
    const std::filesystem::path file(path);
    if (file.extension() != ".json")
    {
        statusMessage = "Not a scene: " + path + " (only .json scenes are loadable)";
        return false;
    }

    Scene* next = NULL;
    try
    {
        next = new Scene(path);
    }
    catch (const std::exception& e)
    {
        statusMessage = "Could not load " + path + ": " + e.what();
        return false;
    }

    // The device buffers point into the old scene, so they go first. The full init on
    // the next frame frees again, which is why pathtraceFree has to be idempotent.
    if (scene != NULL)
    {
        pathtraceFree();
        delete scene;
        scene = NULL;
    }

    scene = next;
    renderState = &scene->state;

    // The interactive camera is an orbit camera around the scene's look at point
    const Camera& cam = renderState->camera;
    cameraPosition = cam.position;
    orbitFromCamera(cam, ogLookAt, zoom, theta, phi);
    camchanged = true;

    // What the panel starts from, and what Reset camera goes back to
    homeCamera = cam;
    captureEmitters(scene);
    imguiData->TracedDepth = renderState->traceDepth;

    // A different resolution means a different texture and pixel buffer
    if (width != cam.resolution.x || height != cam.resolution.y)
    {
        width = cam.resolution.x;
        height = cam.resolution.y;
        glfwSetWindowSize(window, width, height);
        cleanupCuda();
        initTextures();
        initPBO();
        glViewport(0, 0, width, height);
    }

    iteration = 0;
    renderActive = true;
    renderFinished = false;
    sceneNeedsFullInit = true;
    scenePath = path;
    dialogDir = file.parent_path().string();
    statusMessage = "Rendering " + file.filename().string();

    printf("[gui] loaded %s: %d x %d, %d samples, depth %d\n", path.c_str(), width, height,
        renderState->iterations, renderState->traceDepth);
    return true;
}

void framebufferSizeCallback(GLFWwindow* window, int newWidth, int newHeight)
{
    glViewport(0, 0, newWidth, newHeight);
}

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
}

bool init()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));
    //Set up ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    imguiData = guiData;
}


// Main GUI rendering loop
void RenderImGui()
{
    mouseOverImGuiWinow = io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin("Path Tracer Analytics");                 
    ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();

    // Top right: the only control so far. Load a scene and it starts rendering.
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - 12.0f, viewport->WorkPos.y + 12.0f),
        ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.9f);
    const ImGuiWindowFlags loadFlags = ImGuiWindowFlags_NoDecoration
        | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize
        | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
        | ImGuiWindowFlags_NoNav;
    if (ImGui::Begin("##load", NULL, loadFlags))
    {
        if (ImGui::Button("Load data", ImVec2(150.0f, 0.0f)))
        {
#ifdef _WIN32
            const std::string picked = pickSceneFile();
            if (!picked.empty())
            {
                loadScene(picked);
            }
#else
            statusMessage = "The file dialog is Windows only; pass the scene on the command line";
#endif
        }

        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 340.0f);
        ImGui::TextWrapped("%s", statusMessage.c_str());
        ImGui::PopTextWrapPos();

        if (scene != NULL)
        {
            ImGui::Text("%d x %d - %d / %d samples", width, height, iteration,
                renderState->iterations);
        }
#ifndef NDEBUG
        ImGui::TextColored(ImVec4(0.85f, 0.25f, 0.10f, 1.0f), "Debug build: about 25x slower than Release");
#endif
    }
    ImGui::End();

    // The detailed panel: the camera, and the lights the scene is built from. Any
    // edit restarts the accumulation, so the change is visible in the next samples.
    if (scene != NULL)
    {
        ImGui::SetNextWindowPos(ImVec2(60.0f, 130.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(380.0f, 430.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Scene"))
        {
            Camera& cam = renderState->camera;

            if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
            {
                float fovy = cam.fov.y;
                if (ImGui::SliderFloat("FOVY (degrees)", &fovy, 5.0f, 120.0f, "%.1f"))
                {
                    applyVerticalFov(cam, fovy);
                    sceneEdited = true;
                }
                if (ImGui::SliderFloat("Aperture", &cam.aperture, 0.0f, 0.6f, "%.3f"))
                {
                    sceneEdited = true;             // depth of field
                }
                if (ImGui::SliderFloat("Focus distance", &cam.focalDistance, 0.05f, 80.0f, "%.2f"))
                {
                    sceneEdited = true;
                }

                float lookAt[3] = { cam.lookAt.x, cam.lookAt.y, cam.lookAt.z };
                if (ImGui::DragFloat3("Look at", lookAt, 0.02f))
                {
                    cam.lookAt = glm::vec3(lookAt[0], lookAt[1], lookAt[2]);
                    camchanged = true;
                    sceneEdited = true;
                }
                if (ImGui::SliderFloat("Distance", &zoom, 0.1f, 80.0f, "%.2f"))
                {
                    camchanged = true;
                }
                float azimuth = glm::degrees(phi);
                float polar = glm::degrees(theta);
                if (ImGui::SliderFloat("Azimuth (deg)", &azimuth, -180.0f, 180.0f, "%.0f"))
                {
                    phi = glm::radians(azimuth);
                    camchanged = true;
                }
                if (ImGui::SliderFloat("Polar angle (deg)", &polar, 1.0f, 179.0f, "%.0f"))
                {
                    theta = glm::radians(polar);
                    camchanged = true;
                }

                if (ImGui::Button("Reset camera"))
                {
                    renderState->camera = homeCamera;
                    orbitFromCamera(homeCamera, ogLookAt, zoom, theta, phi);
                    camchanged = true;
                    sceneEdited = true;
                }
                ImGui::SameLine();
                ImGui::Text("eye %.2f %.2f %.2f", cam.position.x, cam.position.y, cam.position.z);
            }

            if (ImGui::CollapsingHeader("Area lights (emitters)", ImGuiTreeNodeFlags_DefaultOpen))
            {
                if (panelLightIds.empty())
                {
                    ImGui::TextUnformatted("no emitter in this scene");
                }
                for (size_t row = 0; row < panelLightIds.size(); row++)
                {
                    const int id = panelLightIds[row];
                    if (id < 0 || id >= (int)scene->materials.size()) { continue; }

                    Material& material = scene->materials[id];
                    const char* name = id < (int)scene->materialNames.size()
                        ? scene->materialNames[id].c_str() : "material";
                    ImGui::PushID(id);
                    if (material.emittance > 0.0f)
                    {
                        ImGui::TextUnformatted(name);
                    }
                    else
                    {
                        ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.55f, 1.0f), "%s (off)", name);
                    }
                    if (ImGui::SliderFloat("emission", &material.emittance, 0.0f, 40.0f, "%.2f"))
                    {
                        sceneEdited = true;
                    }
                    if (ImGui::ColorEdit3("colour", &material.color.x,
                            ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR))
                    {
                        sceneEdited = true;
                    }
                    ImGui::Separator();
                    ImGui::PopID();
                }
                ImGui::TextUnformatted("emission 0 takes that light out of the scene");
            }

            if (ImGui::CollapsingHeader("Sky (dome)"))
            {
                Environment& env = renderState->environment;
                const bool edited =
                    ImGui::ColorEdit3("zenith", &env.zenith.x,
                        ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR)
                    | ImGui::ColorEdit3("horizon", &env.horizon.x,
                        ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR)
                    | ImGui::ColorEdit3("ground", &env.ground.x,
                        ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR)
                    | ImGui::SliderFloat("intensity", &env.intensity, 0.0f, 5.0f, "%.2f");
                if (edited)
                {
                    sceneEdited = true;
                }
                ImGui::TextUnformatted("a black sky is no environment light");
            }

            if (ImGui::CollapsingHeader("Sun (distant light)"))
            {
                DistantLight& sun = renderState->distantLight;
                bool enabled = sun.enabled != 0;
                float elevation = 0.0f;
                float azimuth = 0.0f;
                sunAngles(sun.direction, elevation, azimuth);
                float radiusDegrees = glm::degrees(
                    glm::acos(glm::clamp(sun.cosMaxAngle, -1.0f, 1.0f)));

                bool edited = false;
                if (ImGui::Checkbox("enabled", &enabled)) { edited = true; }
                edited |= ImGui::SliderFloat("elevation (deg)", &elevation, -89.0f, 89.0f, "%.0f");
                edited |= ImGui::SliderFloat("azimuth (deg)", &azimuth, -180.0f, 180.0f, "%.0f");
                edited |= ImGui::ColorEdit3("radiance", &sun.radiance.x,
                    ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
                edited |= ImGui::SliderFloat("angular radius (deg)", &radiusDegrees, 0.01f, 45.0f,
                    "%.2f");
                if (edited)
                {
                    sun.enabled = enabled ? 1 : 0;
                    sun.direction = sunDirection(elevation, azimuth);
                    const float cosMax = glm::cos(glm::radians(glm::clamp(radiusDegrees,
                        0.01f, 89.0f)));
                    sun.cosMaxAngle = cosMax;
                    sun.solidAngle = 2.0f * PI * (1.0f - cosMax);
                    sceneEdited = true;
                }
                ImGui::Text("solid angle %.5g sr", sun.solidAngle);
            }
        }
        ImGui::End();
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
    return mouseOverImGuiWinow;
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        runCuda();

        std::string title = "CIS565 Path Tracer | ";
        if (scene == NULL)
        {
            title += "no scene - Load data";
        }
        else if (renderFinished)
        {
            title += "finished " + utilityCore::convertIntToString(iteration) + " samples";
        }
        else
        {
            title += utilityCore::convertIntToString(iteration) + " Iterations";
        }
        glfwSetWindowTitle(window, title.c_str());

        if (scene == NULL)
        {
            // Nothing rendered yet: a plain background behind the Load data button
            glClearColor(0.10f, 0.11f, 0.13f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
        }
        else
        {
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glBindTexture(GL_TEXTURE_2D, displayImage);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
            glClear(GL_COLOR_BUFFER_BIT);

            // Binding GL_PIXEL_UNPACK_BUFFER back to default
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            // VAO, shader program, and texture already bound
            glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
        }

        // Render ImGui Stuff
        RenderImGui();

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

#ifndef NDEBUG
    printf("[build] this is a Debug build: rendering is about 25x slower than Release "
        "(measured on this scene set); use out\\build\\x64-Release for interactive work\n");
#endif

    // A scene argument keeps the command line behaviour: render it and exit. With
    // no argument the window opens empty and the Load data button picks one.
    guiMode = (argc < 2);
    scene = NULL;
    if (guiMode)
    {
        width = 1280;
        height = 720;
        printf("No scene on the command line: click \"Load data\" in the top right corner\n");
    }
    else
    {
        // Load scene file
        scene = new Scene(argv[1]);
    }

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();

    iteration = 0;
    renderActive = (scene != NULL);
    if (scene != NULL)
    {
        scenePath = argv[1];
        statusMessage = "Rendering " + std::filesystem::path(scenePath).filename().string();

        // Set up camera stuff from loaded path tracer settings
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        width = cam.resolution.x;
        height = cam.resolution.y;

        cameraPosition = cam.position;

        // The interactive camera is an orbit camera
        orbitFromCamera(cam, ogLookAt, zoom, theta, phi);

        // What the panel starts from, and what Reset camera goes back to
        homeCamera = cam;
        captureEmitters(scene);
        guiData->TracedDepth = renderState->traceDepth;
        sceneNeedsFullInit = true;
    }

    // Initialize CUDA and GL components
    init();

    // Initialize ImGui Data
    InitImguiData(guiData);
    InitDataContainer(guiData);

    // GLFW main loop
    mainLoop();

    return 0;
}

void saveImage()
{
	// Get the image from the GPU
    pathtraceFetchImage(scene);

    float samples = iteration;
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    img.savePNG(filename); 

    // Denoise the same buffer and save it next to the raw one
    if (denoiserAvailable())
    {
        const int pixelcount = width * height;
        std::vector<glm::vec3> linear(pixelcount);
        for (int i = 0; i < pixelcount; i++)
        {
            linear[i] = renderState->image[i] / samples;
        }
        pathtraceFetchDenoiseGuides(scene);

        std::vector<glm::vec3> denoised;
        const DenoiseResult result = denoiseImage(linear, renderState->normalImage,
            renderState->albedoImage, width, height, denoised);
        if (result.ok)
        {
            Image denoisedImage(width, height);
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    denoisedImage.setPixel(width - 1 - x, y, glm::vec3(denoised[x + y * width]));
                }
            }
            // savePNG appends ".png" itself
            const std::string denoisedName = filename + ".denoised";
            denoisedImage.savePNG(denoisedName);
            printf("[oidn] \"%s\" denoised in %.0f ms on \"%s\" (normal + albedo guides)\n",
                denoisedName.c_str(), result.milliseconds, result.device.c_str());
        }
        else
        {
            printf("[oidn] denoising skipped: %s\n", result.message.c_str());
        }
    }
}

void runCuda()
{
    if (scene == NULL)
    {
        return;                     // nothing loaded yet: the GUI is all there is
    }
    if (sceneEdited)
    {
        // The panel changed something: throw the accumulation away and start again,
        // with whatever the edit needs refreshed below.
        sceneEdited = false;
        iteration = 0;
        renderActive = true;
        renderFinished = false;
    }
    else if (renderFinished)
    {
        if (!camchanged)
        {
            return;                 // hold the finished image
        }
        iteration = 0;              // the camera moved: accumulate it again
        renderActive = true;
        renderFinished = false;
    }
    if (!renderActive)
    {
        return;
    }

    if (camchanged)
    {
        iteration = 0;
        Camera& cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);
        glm::vec3 r = glm::normalize(glm::cross(v, u));
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    if (iteration == 0)
    {
        if (sceneNeedsFullInit)
        {
            // First frame of a scene: allocate everything, and pick up a checkpoint
            // if one matches this exact scene
            pathtraceFree();
            pathtraceInit(scene);
            sceneNeedsFullInit = false;

            // Restartable rendering handler
            int resumedIterations = 0;
            if (pathtraceLoadCheckpoint(scene, &resumedIterations))
            {
                iteration = resumedIterations;
            }
            else if (renderState->checkpointInterval > 0.0f)
            {
                printf("[checkpoint] writing %s.ckpt every %.1f s; stop and re-run the same "
                    "scene to continue where it stopped\n",
                    renderState->imageName.c_str(), renderState->checkpointInterval);
            }
            lastCheckpointTime = glfwGetTime();
            lastFrameTime = glfwGetTime();
            if (renderState->frameInterval > 0.0f)
            {
                printf("[frames] writing one image every %.1f s; the file name carries the sample "
                    "count it had reached\n", renderState->frameInterval);
            }
        }
        else
        {
            // A camera move or a panel edit: only the accumulation and the data the
            // kernels read from device memory have to be refreshed, so the geometry
            // and the hierarchy are left alone
            pathtraceRestart(scene);
        }
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);

        // Restartable rendering handler
        const float interval = renderState->checkpointInterval;
        if (interval > 0.0f && iteration < renderState->iterations)
        {
            const double now = glfwGetTime();
            if (now - lastCheckpointTime >= interval)
            {
                if (pathtraceSaveCheckpoint(scene, iteration))
                {
                    printf("[checkpoint] saved at %d samples\n", iteration);
                }
                lastCheckpointTime = now;
            }
        }

        // Write-out frames: the same idea, but as an image, so an animation can
        // show what the renderer had after a given wall clock time. The name
        // carries the sample count - two renderers at the same moment have very
        // different counts, and that difference is the speed difference.
        const float frameInterval = renderState->frameInterval;
        if (frameInterval > 0.0f)
        {
            const double now = glfwGetTime();
            if (now - lastFrameTime >= frameInterval)
            {
                saveImage();
                lastFrameTime = now;
            }
        }
    }
    else
    {
        saveImage();
        // The render is finished, so there is nothing left to resume: drop the
        // checkpoint instead of leaving a stale one that would make the next
        // start skip straight to the end.
        pathtraceDeleteCheckpoint(scene);
        printCheckpointStats();
        if (!guiMode)
        {
            pathtraceFree();
            cudaDeviceReset();
            exit(EXIT_SUCCESS);
        }
        // GUI: keep the window, the image and the device buffers so another scene
        // can be loaded, or the camera dragged, without restarting the process.
        renderActive = false;
        renderFinished = true;
        statusMessage = "Finished " + renderState->imageName + " at "
            + utilityCore::convertIntToString(iteration) + " samples - load another scene";
        printf("[gui] finished %s at %d samples\n", renderState->imageName.c_str(), iteration);
    }
}

//-------------------------------
//------INTERACTIVITY SETUP------
//-------------------------------

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case GLFW_KEY_ESCAPE:
                if (scene != NULL)
                {
                    saveImage();
                    pathtraceSaveCheckpoint(scene, iteration);
                    printCheckpointStats();
                }
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_S:
                if (scene == NULL)
                {
                    break;                          // nothing to save yet
                }
                saveImage();
                pathtraceSaveCheckpoint(scene, iteration);
                break;
            case GLFW_KEY_SPACE:
                if (scene == NULL)
                {
                    break;
                }
                camchanged = true;
                renderState = &scene->state;
                Camera& cam = renderState->camera;
                cam.lookAt = ogLookAt;
                break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (MouseOverImGuiWindow())
    {
        return;
    }

    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (xpos == lastX || ypos == lastY)
    {
        return; // otherwise, clicking back into window causes re-start
    }

    if (leftMousePressed)
    {
        // compute new camera parameters
        phi -= (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.001f, std::fmin(theta, PI));
        camchanged = true;
    }
    else if (rightMousePressed)
    {
        zoom += (ypos - lastY) / height;
        zoom = std::fmax(0.1f, zoom);
        camchanged = true;
    }
    else if (middleMousePressed && scene != NULL)
    {
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        glm::vec3 forward = cam.view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
        cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
        camchanged = true;
    }

    lastX = xpos;
    lastY = ypos;
}
