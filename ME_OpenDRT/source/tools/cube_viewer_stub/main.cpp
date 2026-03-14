#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <csignal>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#endif
#include <GLFW/glfw3.h>
#if defined(_WIN32)
#include <GLFW/glfw3native.h>
#elif defined(__APPLE__)
#include <GLFW/glfw3native.h>
#endif

#if !defined(__APPLE__)
#include "gl/OpenDRTViewerOpenGLPresenter.h"
#endif
#if defined(__APPLE__)
#include "metal/OpenDRTViewerMetal.h"
#elif defined(OFX_SUPPORTS_CUDARENDER)
#include "cuda/OpenDRTViewerCuda.h"
#endif

#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#endif
#ifndef GL_STATIC_DRAW
#define GL_STATIC_DRAW 0x88E4
#endif
#ifndef GL_DYNAMIC_DRAW
#define GL_DYNAMIC_DRAW 0x88E8
#endif
#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#endif
#ifndef GL_COMPUTE_SHADER
#define GL_COMPUTE_SHADER 0x91B9
#endif
#ifndef GL_LINK_STATUS
#define GL_LINK_STATUS 0x8B82
#endif
#ifndef GL_COMPILE_STATUS
#define GL_COMPILE_STATUS 0x8B81
#endif
#ifndef GL_INFO_LOG_LENGTH
#define GL_INFO_LOG_LENGTH 0x8B84
#endif
#ifndef GL_SHADER_STORAGE_BARRIER_BIT
#define GL_SHADER_STORAGE_BARRIER_BIT 0x2000
#endif
#ifndef GL_BUFFER_UPDATE_BARRIER_BIT
#define GL_BUFFER_UPDATE_BARRIER_BIT 0x0200
#endif
#ifndef GL_SHADER_STORAGE_BUFFER_BINDING
#define GL_SHADER_STORAGE_BUFFER_BINDING 0x90D3
#endif

#include "OpenDRTParams.h"
#include "OpenDRTPresets.h"
#include "OpenDRTProcessor.h"

namespace {
std::atomic<bool> gRun{true};
std::atomic<bool> gConnected{false};
std::atomic<bool> gBringToFront{false};
std::atomic<int> gWindowVisible{1};
std::atomic<int> gWindowIconified{0};
std::atomic<int> gWindowFocused{1};

void onSignal(int) {
  gRun.store(false);
}

std::string pipeName() {
  const char* env = std::getenv("ME_OPENDRT_CUBE_VIEWER_PIPE");
  if (env && env[0] != '\0') return std::string(env);
#if defined(_WIN32)
  return "\\\\.\\pipe\\ME_OpenDRT_CubeViewer";
#else
  return "/tmp/me_opendrt_cube_viewer.sock";
#endif
}

std::string viewerLogPath() {
#if defined(_WIN32)
  const char* temp = std::getenv("TEMP");
  if (!temp || temp[0] == '\0') temp = std::getenv("TMP");
  if (!temp || temp[0] == '\0') return "C:\\Windows\\Temp\\ME_OpenDRT_CubeViewer.log";
  return std::string(temp) + "\\ME_OpenDRT_CubeViewer.log";
#elif defined(__APPLE__)
  const char* home = std::getenv("HOME");
  if (!home || home[0] == '\0') return "/tmp/ME_OpenDRT_CubeViewer.log";
  return std::string(home) + "/Library/Logs/ME_OpenDRT_CubeViewer.log";
#else
  const char* home = std::getenv("HOME");
  if (!home || home[0] == '\0') return "/tmp/ME_OpenDRT_CubeViewer.log";
  return std::string(home) + "/.cache/ME_OpenDRT_CubeViewer.log";
#endif
}

void logViewerEvent(const std::string& msg) {
  const std::string path = viewerLogPath();
  if (path.empty()) return;
  FILE* f = std::fopen(path.c_str(), "a");
  if (!f) return;
  std::fprintf(f, "[ME_OpenDRT_CubeViewer] %s\n", msg.c_str());
  std::fclose(f);
}

bool viewerDiagnosticsEnabled() {
  const char* env = std::getenv("ME_OPENDRT_VIEWER_DIAGNOSTICS");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool viewerInputComputeEnabled() {
  const char* env = std::getenv("ME_OPENDRT_VIEWER_INPUT_COMPUTE");
  if (env != nullptr && env[0] != '\0') return env[0] != '0';
  return true;
}

bool viewerParityCheckEnabled() {
  const char* env = std::getenv("ME_OPENDRT_VIEWER_PARITY_CHECK");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

void logViewerDiagnostic(bool enabled, const std::string& msg) {
  if (!enabled) return;
  logViewerEvent(std::string("[diag] ") + msg);
}

size_t expectedIdentityDisplayPointCount(int res) {
  const int interiorStep = (res <= 25) ? 2 : (res <= 41 ? 2 : 3);
  size_t count = 0;
  for (int z = 0; z < res; ++z) {
    for (int y = 0; y < res; ++y) {
      for (int x = 0; x < res; ++x) {
        const bool onBoundary = (x == 0 || y == 0 || z == 0 || x == res - 1 || y == res - 1 || z == res - 1);
        if (!onBoundary && (((x % interiorStep) != 0) || ((y % interiorStep) != 0) || ((z % interiorStep) != 0))) {
          continue;
        }
        ++count;
      }
    }
  }
  return count;
}

std::string pointDrawSourceLabel(bool useComputeIdentityBuffers,
                                 bool useComputeInputBuffers,
                                 bool usePointBuffers,
                                 bool hasCpuArrays) {
  if (useComputeIdentityBuffers) return "identity-compute";
  if (useComputeInputBuffers) return "input-compute";
  if (usePointBuffers) return "gl-buffer";
  if (hasCpuArrays) return "cpu-array";
  return "none";
}

const char* kViewerVersionString = "v1.2.11";

#if defined(_WIN32)
void applyWindowsWindowIcon(GLFWwindow* window) {
  if (!window) return;
  HWND hwnd = glfwGetWin32Window(window);
  if (!hwnd) return;
  HMODULE module = GetModuleHandleW(nullptr);
  if (!module) return;
  HICON iconLarge = static_cast<HICON>(LoadImageW(module, L"GLFW_ICON", IMAGE_ICON,
                                                   GetSystemMetrics(SM_CXICON),
                                                   GetSystemMetrics(SM_CYICON), 0));
  HICON iconSmall = static_cast<HICON>(LoadImageW(module, L"GLFW_ICON", IMAGE_ICON,
                                                   GetSystemMetrics(SM_CXSMICON),
                                                   GetSystemMetrics(SM_CYSMICON), 0));
  if (iconLarge) SendMessageW(hwnd, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(iconLarge));
  if (iconSmall) SendMessageW(hwnd, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(iconSmall));
}
#endif

#if !defined(_WIN32)
bool sendAllSocket(int fd, const char* data, size_t size) {
  if (fd < 0 || data == nullptr) return false;
  size_t totalSent = 0;
  while (totalSent < size) {
    const ssize_t sent = ::send(fd, data + totalSent, size - totalSent, 0);
    if (sent <= 0) {
      logViewerEvent(std::string("socket send failed after ") + std::to_string(totalSent) + "/" +
                     std::to_string(size) + " bytes: errno=" + std::to_string(errno) + " (" + std::strerror(errno) + ")");
      return false;
    }
    totalSent += static_cast<size_t>(sent);
  }
  return true;
}
#endif

struct CameraState {
  float qx = 0.0f;
  float qy = 0.0f;
  float qz = 0.0f;
  float qw = 1.0f;
  float distance = 4.35f;
  float panX = 0.0f;
  float panY = 0.12f;
};

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct Quat {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 1.0f;
};

inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

inline float length3(Vec3 v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline Vec3 normalize3(Vec3 v) {
  const float len = length3(v);
  if (len <= 1e-8f) return Vec3{};
  const float inv = 1.0f / len;
  return Vec3{v.x * inv, v.y * inv, v.z * inv};
}

inline Vec3 cross3(Vec3 a, Vec3 b) {
  return Vec3{
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x};
}

inline float dot3(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Quat normalizeQ(Quat q) {
  const float len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (len <= 1e-8f) return Quat{};
  const float inv = 1.0f / len;
  return Quat{q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

inline Quat mulQ(Quat a, Quat b) {
  return Quat{
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

inline Quat axisAngleQ(Vec3 axis, float radians) {
  const Vec3 n = normalize3(axis);
  const float h = radians * 0.5f;
  const float s = std::sin(h);
  return normalizeQ(Quat{n.x * s, n.y * s, n.z * s, std::cos(h)});
}

Vec3 mapArcball(double sx, double sy, int width, int height) {
  if (width < 1) width = 1;
  if (height < 1) height = 1;
  const float nx = static_cast<float>((2.0 * sx - static_cast<double>(width)) / static_cast<double>(width));
  const float ny = static_cast<float>((static_cast<double>(height) - 2.0 * sy) / static_cast<double>(height));
  const float d2 = nx * nx + ny * ny;
  if (d2 <= 1.0f) return Vec3{nx, ny, std::sqrt(1.0f - d2)};
  const float inv = 1.0f / std::sqrt(d2);
  return Vec3{nx * inv, ny * inv, 0.0f};
}

void quatToMatrix(Quat q, float out16[16]) {
  q = normalizeQ(q);
  const float xx = q.x * q.x;
  const float yy = q.y * q.y;
  const float zz = q.z * q.z;
  const float xy = q.x * q.y;
  const float xz = q.x * q.z;
  const float yz = q.y * q.z;
  const float wx = q.w * q.x;
  const float wy = q.w * q.y;
  const float wz = q.w * q.z;
  // Column-major for OpenGL fixed-function pipeline.
  out16[0] = 1.0f - 2.0f * (yy + zz);
  out16[1] = 2.0f * (xy + wz);
  out16[2] = 2.0f * (xz - wy);
  out16[3] = 0.0f;
  out16[4] = 2.0f * (xy - wz);
  out16[5] = 1.0f - 2.0f * (xx + zz);
  out16[6] = 2.0f * (yz + wx);
  out16[7] = 0.0f;
  out16[8] = 2.0f * (xz + wy);
  out16[9] = 2.0f * (yz - wx);
  out16[10] = 1.0f - 2.0f * (xx + yy);
  out16[11] = 0.0f;
  out16[12] = 0.0f;
  out16[13] = 0.0f;
  out16[14] = 0.0f;
  out16[15] = 1.0f;
}

void makeIdentityMatrix(float out16[16]) {
  std::memset(out16, 0, sizeof(float) * 16u);
  out16[0] = 1.0f;
  out16[5] = 1.0f;
  out16[10] = 1.0f;
  out16[15] = 1.0f;
}

void makeTranslationMatrix(float tx, float ty, float tz, float out16[16]) {
  makeIdentityMatrix(out16);
  out16[12] = tx;
  out16[13] = ty;
  out16[14] = tz;
}

void multiplyMatrix4(const float a[16], const float b[16], float out16[16]) {
  float result[16] = {};
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 4; ++row) {
      result[col * 4 + row] =
          a[0 * 4 + row] * b[col * 4 + 0] +
          a[1 * 4 + row] * b[col * 4 + 1] +
          a[2 * 4 + row] * b[col * 4 + 2] +
          a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  std::memcpy(out16, result, sizeof(result));
}

void buildSceneMvp(int width, int height, const CameraState& cam, float out16[16]) {
  const double aspect = static_cast<double>(width) / static_cast<double>(height > 0 ? height : 1);
  const double fovy = 35.0;
  const double zNear = 0.08;
  const double zFar = 120.0;
  const double ymax = zNear * std::tan(fovy * 0.5 * 3.141592653589793 / 180.0);
  const double xmax = ymax * aspect;

  float projection[16] = {};
  projection[0] = static_cast<float>((2.0 * zNear) / (2.0 * xmax));
  projection[5] = static_cast<float>((2.0 * zNear) / (2.0 * ymax));
  projection[10] = static_cast<float>(-(zFar + zNear) / (zFar - zNear));
  projection[11] = -1.0f;
  projection[14] = static_cast<float>(-(2.0 * zFar * zNear) / (zFar - zNear));

  float translation[16];
  float rotation[16];
  float modelView[16];
  makeTranslationMatrix(cam.panX, cam.panY, -cam.distance, translation);
  quatToMatrix(Quat{cam.qx, cam.qy, cam.qz, cam.qw}, rotation);
  multiplyMatrix4(translation, rotation, modelView);
  multiplyMatrix4(projection, modelView, out16);
}

void resetCamera(CameraState* cam) {
  if (!cam) return;
  cam->distance = 4.35f;
  cam->panX = 0.0f;
  cam->panY = 0.12f;
  // Preserve previous default orientation: pitch 20 then yaw -45.
  const float deg2rad = 3.14159265358979323846f / 180.0f;
  const Quat qPitch = axisAngleQ(Vec3{1.0f, 0.0f, 0.0f}, 20.0f * deg2rad);
  const Quat qYaw = axisAngleQ(Vec3{0.0f, 1.0f, 0.0f}, -45.0f * deg2rad);
  const Quat q = normalizeQ(mulQ(qPitch, qYaw));
  cam->qx = q.x;
  cam->qy = q.y;
  cam->qz = q.z;
  cam->qw = q.w;
}

void resetVectorscopeCamera(CameraState* cam) {
  if (!cam) return;
  cam->distance = 4.35f;
  cam->panX = 0.0f;
  cam->panY = 0.0f;
  const float deg2rad = 3.14159265358979323846f / 180.0f;
  // Look roughly down the neutral axis so RGB corners read like a vectorscope.
  const Quat qPitch = axisAngleQ(Vec3{1.0f, 0.0f, 0.0f}, 35.2643897f * deg2rad);
  const Quat qYaw = axisAngleQ(Vec3{0.0f, 1.0f, 0.0f}, -45.0f * deg2rad);
  const Quat qView = normalizeQ(mulQ(qPitch, qYaw));
  // Rotate within the view plane to match the familiar vectorscope orientation.
  const Quat qRoll = axisAngleQ(Vec3{0.0f, 0.0f, 1.0f}, 135.0f * deg2rad);
  const Quat q = normalizeQ(mulQ(qRoll, qView));
  cam->qx = q.x;
  cam->qy = q.y;
  cam->qz = q.z;
  cam->qw = q.w;
}

struct MeshData {
  int resolution = 33;
  std::string quality = "Medium";
  std::string paramHash;
  bool renderOk = false;
  float maxDelta = 0.0f;
  float parityMaxDelta = 0.0f;
  uint64_t serial = 0;
  size_t pointCount = 0;
  std::string transformBackendLabel = "cpu";
  std::string packBackendLabel = "cpu-array";
  std::vector<float> pointVerts;
  std::vector<float> pointColors;
};

enum class ViewerGpuBackend {
  CpuReference,
  OpenGlBufferedDraw,
  OpenGlComputeIdentity,
  CudaComputeMesh,
  MetalComputeMesh,
};

struct ViewerGpuCapabilities {
  bool glBufferObjects = false;
  bool glComputeShaders = false;
  bool inputCloudComputeEnabled = false;
  bool parityCheckEnabled = false;
  bool glPresenterReady = false;
#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
  bool cudaViewerAvailable = false;
  bool cudaInteropReady = false;
  bool cudaStartupReady = false;
  std::string cudaDeviceName;
  std::string cudaReason;
#endif
  bool metalViewerAvailable = false;
  bool metalQueueReady = false;
  ViewerGpuBackend activeBackend = ViewerGpuBackend::CpuReference;
  std::string glVersion;
  std::string activeBackendLabel = "cpu-ref";
  std::string roadmapLabel = "cpu-ref";
  std::string presenterBackendLabel = "cpu";
  std::string metalDeviceName;
};

struct ViewerRuntimeState {
  std::string sessionBackendLabel = "cpu-ref";
  std::string presenterBackendLabel = "cpu";
  bool identityGpuDemoted = false;
  bool inputGpuDemoted = false;
  int identityParityFailures = 0;
  std::string identityDemotionReason;
  std::string inputDemotionReason;
};

using ViewerGLCreateShaderProc = GLuint(APIENTRY *)(GLenum);
using ViewerGLShaderSourceProc = void(APIENTRY *)(GLuint, GLsizei, const char* const*, const GLint*);
using ViewerGLCompileShaderProc = void(APIENTRY *)(GLuint);
using ViewerGLGetShaderivProc = void(APIENTRY *)(GLuint, GLenum, GLint*);
using ViewerGLGetShaderInfoLogProc = void(APIENTRY *)(GLuint, GLsizei, GLsizei*, char*);
using ViewerGLCreateProgramProc = GLuint(APIENTRY *)(void);
using ViewerGLAttachShaderProc = void(APIENTRY *)(GLuint, GLuint);
using ViewerGLLinkProgramProc = void(APIENTRY *)(GLuint);
using ViewerGLGetProgramivProc = void(APIENTRY *)(GLuint, GLenum, GLint*);
using ViewerGLGetProgramInfoLogProc = void(APIENTRY *)(GLuint, GLsizei, GLsizei*, char*);
using ViewerGLDeleteShaderProc = void(APIENTRY *)(GLuint);
using ViewerGLDeleteProgramProc = void(APIENTRY *)(GLuint);
using ViewerGLUseProgramProc = void(APIENTRY *)(GLuint);
using ViewerGLGetUniformLocationProc = GLint(APIENTRY *)(GLuint, const char*);
using ViewerGLUniform1iProc = void(APIENTRY *)(GLint, GLint);
using ViewerGLBindBufferBaseProc = void(APIENTRY *)(GLenum, GLuint, GLuint);
using ViewerGLDispatchComputeProc = void(APIENTRY *)(GLuint, GLuint, GLuint);
using ViewerGLMemoryBarrierProc = void(APIENTRY *)(GLbitfield);
using ViewerGLBufferSubDataProc = void(APIENTRY *)(GLenum, std::ptrdiff_t, std::ptrdiff_t, const void*);
using ViewerGLGetBufferSubDataProc = void(APIENTRY *)(GLenum, std::ptrdiff_t, std::ptrdiff_t, void*);

struct IdentityComputeCache {
  GLuint src = 0;
  GLuint dst = 0;
  GLuint verts = 0;
  GLuint colors = 0;
  GLuint counter = 0;
  GLuint program = 0;
  GLint resLoc = -1;
  GLint interiorStepLoc = -1;
  GLint showOverflowLoc = -1;
  GLint highlightOverflowLoc = -1;
  uint64_t builtSerial = 0;
  GLsizei pointCount = 0;
  bool available = false;
};

struct InputCloudComputeCache {
  GLuint input = 0;
  GLuint verts = 0;
  GLuint colors = 0;
  GLuint program = 0;
  GLint pointCountLoc = -1;
  GLint showOverflowLoc = -1;
  GLint highlightOverflowLoc = -1;
  uint64_t builtSerial = 0;
  GLsizei pointCount = 0;
  bool available = false;
};

#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
struct IdentityCudaDrawCache {
  OpenDRTViewerCuda::IdentityCache cache{};
  size_t pointCapacity = 0;
};

struct InputCudaDrawCache {
  OpenDRTViewerCuda::InputCache cache{};
  size_t pointCapacity = 0;
};
#endif

uint64_t nextMeshSerial() {
  static std::atomic<uint64_t> serial{1u};
  return serial.fetch_add(1u, std::memory_order_relaxed);
}

struct PendingMessage {
  uint64_t seq = 0;
  std::string line;
};

bool senderMatchesCurrent(const std::string& currentSenderId, const std::string& senderId) {
  return currentSenderId.empty() || senderId.empty() || senderId == currentSenderId;
}

std::mutex gMsgMutex;
PendingMessage gPendingParamsMsg;
PendingMessage gPendingCloudMsg;
bool gHasPendingParamsMsg = false;
bool gHasPendingCloudMsg = false;

inline float clamp01(float v) {
  return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

inline bool rgbOutOfBounds(float r, float g, float b) {
  return r < 0.0f || r > 1.0f || g < 0.0f || g > 1.0f || b < 0.0f || b > 1.0f;
}

using ViewerGLsizeiptr = std::ptrdiff_t;
using ViewerGLGenBuffersProc = void (APIENTRY *)(GLsizei, GLuint*);
using ViewerGLBindBufferProc = void (APIENTRY *)(GLenum, GLuint);
using ViewerGLBufferDataProc = void (APIENTRY *)(GLenum, ViewerGLsizeiptr, const void*, GLenum);
using ViewerGLDeleteBuffersProc = void (APIENTRY *)(GLsizei, const GLuint*);

struct ViewerGlBufferApi {
  bool loaded = false;
  bool available = false;
  ViewerGLGenBuffersProc genBuffers = nullptr;
  ViewerGLBindBufferProc bindBuffer = nullptr;
  ViewerGLBufferDataProc bufferData = nullptr;
  ViewerGLBufferSubDataProc bufferSubData = nullptr;
  ViewerGLGetBufferSubDataProc getBufferSubData = nullptr;
  ViewerGLDeleteBuffersProc deleteBuffers = nullptr;
};

const ViewerGlBufferApi& viewerGlBufferApi() {
  static ViewerGlBufferApi api{};
  if (!api.loaded) {
    api.loaded = true;
    api.genBuffers = reinterpret_cast<ViewerGLGenBuffersProc>(glfwGetProcAddress("glGenBuffers"));
    api.bindBuffer = reinterpret_cast<ViewerGLBindBufferProc>(glfwGetProcAddress("glBindBuffer"));
    api.bufferData = reinterpret_cast<ViewerGLBufferDataProc>(glfwGetProcAddress("glBufferData"));
    api.bufferSubData = reinterpret_cast<ViewerGLBufferSubDataProc>(glfwGetProcAddress("glBufferSubData"));
    api.getBufferSubData = reinterpret_cast<ViewerGLGetBufferSubDataProc>(glfwGetProcAddress("glGetBufferSubData"));
    api.deleteBuffers = reinterpret_cast<ViewerGLDeleteBuffersProc>(glfwGetProcAddress("glDeleteBuffers"));
    api.available = api.genBuffers && api.bindBuffer && api.bufferData && api.deleteBuffers;
  }
  return api;
}

struct ViewerGlComputeApi {
  bool loaded = false;
  bool available = false;
  ViewerGLCreateShaderProc createShader = nullptr;
  ViewerGLShaderSourceProc shaderSource = nullptr;
  ViewerGLCompileShaderProc compileShader = nullptr;
  ViewerGLGetShaderivProc getShaderiv = nullptr;
  ViewerGLGetShaderInfoLogProc getShaderInfoLog = nullptr;
  ViewerGLCreateProgramProc createProgram = nullptr;
  ViewerGLAttachShaderProc attachShader = nullptr;
  ViewerGLLinkProgramProc linkProgram = nullptr;
  ViewerGLGetProgramivProc getProgramiv = nullptr;
  ViewerGLGetProgramInfoLogProc getProgramInfoLog = nullptr;
  ViewerGLDeleteShaderProc deleteShader = nullptr;
  ViewerGLDeleteProgramProc deleteProgram = nullptr;
  ViewerGLUseProgramProc useProgram = nullptr;
  ViewerGLGetUniformLocationProc getUniformLocation = nullptr;
  ViewerGLUniform1iProc uniform1i = nullptr;
  ViewerGLBindBufferBaseProc bindBufferBase = nullptr;
  ViewerGLDispatchComputeProc dispatchCompute = nullptr;
  ViewerGLMemoryBarrierProc memoryBarrier = nullptr;
};

const ViewerGlComputeApi& viewerGlComputeApi() {
  static ViewerGlComputeApi api{};
  if (!api.loaded) {
    api.loaded = true;
    api.createShader = reinterpret_cast<ViewerGLCreateShaderProc>(glfwGetProcAddress("glCreateShader"));
    api.shaderSource = reinterpret_cast<ViewerGLShaderSourceProc>(glfwGetProcAddress("glShaderSource"));
    api.compileShader = reinterpret_cast<ViewerGLCompileShaderProc>(glfwGetProcAddress("glCompileShader"));
    api.getShaderiv = reinterpret_cast<ViewerGLGetShaderivProc>(glfwGetProcAddress("glGetShaderiv"));
    api.getShaderInfoLog = reinterpret_cast<ViewerGLGetShaderInfoLogProc>(glfwGetProcAddress("glGetShaderInfoLog"));
    api.createProgram = reinterpret_cast<ViewerGLCreateProgramProc>(glfwGetProcAddress("glCreateProgram"));
    api.attachShader = reinterpret_cast<ViewerGLAttachShaderProc>(glfwGetProcAddress("glAttachShader"));
    api.linkProgram = reinterpret_cast<ViewerGLLinkProgramProc>(glfwGetProcAddress("glLinkProgram"));
    api.getProgramiv = reinterpret_cast<ViewerGLGetProgramivProc>(glfwGetProcAddress("glGetProgramiv"));
    api.getProgramInfoLog = reinterpret_cast<ViewerGLGetProgramInfoLogProc>(glfwGetProcAddress("glGetProgramInfoLog"));
    api.deleteShader = reinterpret_cast<ViewerGLDeleteShaderProc>(glfwGetProcAddress("glDeleteShader"));
    api.deleteProgram = reinterpret_cast<ViewerGLDeleteProgramProc>(glfwGetProcAddress("glDeleteProgram"));
    api.useProgram = reinterpret_cast<ViewerGLUseProgramProc>(glfwGetProcAddress("glUseProgram"));
    api.getUniformLocation = reinterpret_cast<ViewerGLGetUniformLocationProc>(glfwGetProcAddress("glGetUniformLocation"));
    api.uniform1i = reinterpret_cast<ViewerGLUniform1iProc>(glfwGetProcAddress("glUniform1i"));
    api.bindBufferBase = reinterpret_cast<ViewerGLBindBufferBaseProc>(glfwGetProcAddress("glBindBufferBase"));
    api.dispatchCompute = reinterpret_cast<ViewerGLDispatchComputeProc>(glfwGetProcAddress("glDispatchCompute"));
    api.memoryBarrier = reinterpret_cast<ViewerGLMemoryBarrierProc>(glfwGetProcAddress("glMemoryBarrier"));
    api.available = api.createShader && api.shaderSource && api.compileShader && api.getShaderiv &&
                    api.getShaderInfoLog && api.createProgram && api.attachShader && api.linkProgram &&
                    api.getProgramiv && api.getProgramInfoLog && api.deleteShader && api.deleteProgram &&
                    api.useProgram && api.getUniformLocation && api.uniform1i && api.bindBufferBase &&
                    api.dispatchCompute && api.memoryBarrier;
  }
  return api;
}

std::string currentGlVersionString() {
#if defined(__APPLE__)
  return "metal-presenter";
#else
  const GLubyte* version = glGetString(GL_VERSION);
  return version ? reinterpret_cast<const char*>(version) : std::string("unknown");
#endif
}

std::string readShaderLog(GLuint handle, bool program, const ViewerGlComputeApi& api) {
  GLint logLength = 0;
  if (program) {
    api.getProgramiv(handle, GL_INFO_LOG_LENGTH, &logLength);
  } else {
    api.getShaderiv(handle, GL_INFO_LOG_LENGTH, &logLength);
  }
  if (logLength <= 1) return std::string();
  std::string log(static_cast<size_t>(logLength), '\0');
  GLsizei written = 0;
  if (program) {
    api.getProgramInfoLog(handle, logLength, &written, log.data());
  } else {
    api.getShaderInfoLog(handle, logLength, &written, log.data());
  }
  if (written > 0 && static_cast<size_t>(written) < log.size()) {
    log.resize(static_cast<size_t>(written));
  }
  return log;
}

ViewerGpuCapabilities detectViewerGpuCapabilities() {
  ViewerGpuCapabilities caps{};
  caps.parityCheckEnabled = viewerParityCheckEnabled();
#if defined(__APPLE__)
  caps.glVersion = "metal-presenter";
  caps.glBufferObjects = false;
  caps.glComputeShaders = false;
#else
  caps.glVersion = currentGlVersionString();
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  caps.glBufferObjects = bufferApi.available;
  caps.glComputeShaders = glfwGetProcAddress("glDispatchCompute") != nullptr &&
                          glfwGetProcAddress("glBindBufferBase") != nullptr;
#endif

#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
  const OpenDRTViewerCuda::ProbeResult cudaProbe = OpenDRTViewerCuda::probe();
  caps.cudaViewerAvailable = cudaProbe.available;
  caps.cudaInteropReady = cudaProbe.interopReady;
  caps.cudaDeviceName = cudaProbe.deviceName != nullptr ? cudaProbe.deviceName : "";
  caps.cudaReason = cudaProbe.reason != nullptr ? cudaProbe.reason : "";
#endif

#if defined(__APPLE__)
  const OpenDRTViewerMetal::ProbeResult metalProbe = OpenDRTViewerMetal::probe();
  caps.metalViewerAvailable = metalProbe.available;
  caps.metalQueueReady = metalProbe.queueReady;
  caps.metalDeviceName = metalProbe.deviceName != nullptr ? metalProbe.deviceName : "";
#endif

  if (caps.glComputeShaders) {
#if !defined(__APPLE__)
  #if defined(OFX_SUPPORTS_CUDARENDER)
    if (caps.glBufferObjects && caps.cudaViewerAvailable && caps.cudaInteropReady) {
      caps.activeBackend = ViewerGpuBackend::CudaComputeMesh;
      caps.activeBackendLabel = "cuda-compute-mesh";
      caps.roadmapLabel = "cuda-phase3-mesh";
    } else {
      caps.activeBackend = ViewerGpuBackend::OpenGlComputeIdentity;
      caps.activeBackendLabel = "gl-compute-id";
      caps.roadmapLabel = "gl-compute-phase2";
    }
  #else
    caps.activeBackend = ViewerGpuBackend::OpenGlComputeIdentity;
    caps.activeBackendLabel = "gl-compute-id";
    caps.roadmapLabel = "gl-compute-phase2";
  #endif
#endif
  } else if (caps.glBufferObjects) {
#if !defined(__APPLE__)
    caps.activeBackend = ViewerGpuBackend::OpenGlBufferedDraw;
    caps.activeBackendLabel = "gl-buffer";
  #if defined(OFX_SUPPORTS_CUDARENDER)
    if (caps.glBufferObjects && caps.cudaViewerAvailable && caps.cudaInteropReady) {
      caps.activeBackend = ViewerGpuBackend::CudaComputeMesh;
      caps.activeBackendLabel = "cuda-compute-mesh";
      caps.roadmapLabel = "cuda-phase3-buffer-fallback";
    } else {
      caps.roadmapLabel = "gl-compute-probe-miss";
    }
  #else
    caps.roadmapLabel = "gl-compute-probe-miss";
  #endif
#endif
  } else {
#if defined(__APPLE__)
    if (caps.metalViewerAvailable && caps.metalQueueReady) {
      caps.activeBackend = ViewerGpuBackend::MetalComputeMesh;
      caps.activeBackendLabel = "metal-compute-mesh";
      caps.roadmapLabel = "metal-end-to-end";
    } else {
      caps.activeBackend = ViewerGpuBackend::CpuReference;
      caps.activeBackendLabel = "cpu-ref";
      caps.roadmapLabel = caps.metalViewerAvailable ? "metal-presenter-missing" : "metal-unavailable";
    }
#else
  #if defined(OFX_SUPPORTS_CUDARENDER)
    caps.activeBackend = ViewerGpuBackend::CpuReference;
    caps.activeBackendLabel = "cpu-ref";
    caps.roadmapLabel = "cpu-ref";
  #else
    caps.activeBackend = ViewerGpuBackend::CpuReference;
    caps.activeBackendLabel = "cpu-ref";
    caps.roadmapLabel = "cpu-ref";
  #endif
#endif
  }

  return caps;
}

void releaseIdentityComputeCache(IdentityComputeCache* cache) {
  if (!cache) return;
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  const ViewerGlComputeApi& computeApi = viewerGlComputeApi();
  if (bufferApi.available) {
    GLuint buffers[5] = {cache->src, cache->dst, cache->verts, cache->colors, cache->counter};
    GLuint toDelete[5] = {};
    GLsizei count = 0;
    for (GLuint id : buffers) {
      if (id != 0) toDelete[count++] = id;
    }
    if (count > 0) bufferApi.deleteBuffers(count, toDelete);
    bufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
  }
  if (computeApi.available && cache->program != 0) {
    computeApi.deleteProgram(cache->program);
  }
  *cache = IdentityComputeCache{};
}

void releaseInputCloudComputeCache(InputCloudComputeCache* cache) {
  if (!cache) return;
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  const ViewerGlComputeApi& computeApi = viewerGlComputeApi();
  if (bufferApi.available) {
    GLuint buffers[3] = {cache->input, cache->verts, cache->colors};
    GLuint toDelete[3] = {};
    GLsizei count = 0;
    for (GLuint id : buffers) {
      if (id != 0) toDelete[count++] = id;
    }
    if (count > 0) bufferApi.deleteBuffers(count, toDelete);
    bufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
  }
  if (computeApi.available && cache->program != 0) {
    computeApi.deleteProgram(cache->program);
  }
  *cache = InputCloudComputeCache{};
}

#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
bool ensureViewerDrawBuffers(GLuint* verts,
                             GLuint* colors,
                             size_t pointCapacity,
                             GLenum usage = GL_DYNAMIC_DRAW) {
  if (!verts || !colors) return false;
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  if (!bufferApi.available) return false;
  if (*verts == 0) bufferApi.genBuffers(1, verts);
  if (*colors == 0) bufferApi.genBuffers(1, colors);
  if (*verts == 0 || *colors == 0) return false;
  bufferApi.bindBuffer(GL_ARRAY_BUFFER, *verts);
  bufferApi.bufferData(GL_ARRAY_BUFFER,
                       static_cast<ViewerGLsizeiptr>(pointCapacity * 3u * sizeof(float)),
                       nullptr,
                       usage);
  bufferApi.bindBuffer(GL_ARRAY_BUFFER, *colors);
  bufferApi.bufferData(GL_ARRAY_BUFFER,
                       static_cast<ViewerGLsizeiptr>(pointCapacity * 3u * sizeof(float)),
                       nullptr,
                       usage);
  bufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
  return true;
}

bool ensureIdentityCudaBuffers(IdentityCudaDrawCache* cache, int resolution) {
  if (!cache) return false;
  const size_t pointCapacity = static_cast<size_t>(resolution) * static_cast<size_t>(resolution) * static_cast<size_t>(resolution);
  if (cache->pointCapacity >= pointCapacity && cache->cache.verts != 0 && cache->cache.colors != 0) return true;
  cache->pointCapacity = pointCapacity;
  return ensureViewerDrawBuffers(&cache->cache.verts, &cache->cache.colors, pointCapacity);
}

bool ensureInputCudaBuffers(InputCudaDrawCache* cache, size_t pointCapacity) {
  if (!cache) return false;
  if (cache->pointCapacity >= pointCapacity && cache->cache.verts != 0 && cache->cache.colors != 0) return true;
  cache->pointCapacity = pointCapacity;
  return ensureViewerDrawBuffers(&cache->cache.verts, &cache->cache.colors, pointCapacity);
}

void releaseIdentityCudaDrawCache(IdentityCudaDrawCache* cache) {
  if (!cache) return;
  OpenDRTViewerCuda::releaseIdentityCache(&cache->cache);
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  if (bufferApi.available) {
    GLuint ids[2] = {cache->cache.verts, cache->cache.colors};
    GLuint toDelete[2] = {};
    GLsizei count = 0;
    for (GLuint id : ids) {
      if (id != 0) toDelete[count++] = id;
    }
    if (count > 0) bufferApi.deleteBuffers(count, toDelete);
  }
  cache->cache.verts = 0;
  cache->cache.colors = 0;
  cache->pointCapacity = 0;
}

void releaseInputCudaDrawCache(InputCudaDrawCache* cache) {
  if (!cache) return;
  OpenDRTViewerCuda::releaseInputCache(&cache->cache);
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  if (bufferApi.available) {
    GLuint ids[2] = {cache->cache.verts, cache->cache.colors};
    GLuint toDelete[2] = {};
    GLsizei count = 0;
    for (GLuint id : ids) {
      if (id != 0) toDelete[count++] = id;
    }
    if (count > 0) bufferApi.deleteBuffers(count, toDelete);
  }
  cache->cache.verts = 0;
  cache->cache.colors = 0;
  cache->pointCapacity = 0;
}
#endif

bool ensureIdentityComputeProgram(IdentityComputeCache* cache) {
  if (!cache) return false;
  if (cache->program != 0) return true;
  const ViewerGlComputeApi& api = viewerGlComputeApi();
  if (!api.available) return false;
  static const char* kShaderSrc = R"GLSL(
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer SrcBuffer { float srcVals[]; };
layout(std430, binding = 1) readonly buffer DstBuffer { float dstVals[]; };
layout(std430, binding = 2) writeonly buffer VertBuffer { float vertVals[]; };
layout(std430, binding = 3) writeonly buffer ColorBuffer { float colorVals[]; };
layout(std430, binding = 4) buffer CountBuffer { uint outCount; };
uniform int uResolution;
uniform int uInteriorStep;
uniform int uShowOverflow;
uniform int uHighlightOverflow;

float clamp01(float v) {
  return clamp(v, 0.0, 1.0);
}

bool outOfBounds(float r, float g, float b) {
  return r < 0.0 || r > 1.0 || g < 0.0 || g > 1.0 || b < 0.0 || b > 1.0;
}

void mapDisplayColor(float inR, float inG, float inB, out float outR, out float outG, out float outB) {
  float r = pow(clamp01(inR), 0.90);
  float g = pow(clamp01(inG), 0.90);
  float b = pow(clamp01(inB), 0.90);
  float luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  r = clamp01(luma + (r - luma));
  g = clamp01(luma + (g - luma));
  b = clamp01(luma + (b - luma));
  outR = r;
  outG = g;
  outB = b;
}

void main() {
  uint index = gl_GlobalInvocationID.x;
  uint res = uint(uResolution);
  uint count = res * res * res;
  if (index >= count) return;

  uint plane = res * res;
  uint z = index / plane;
  uint rem = index - z * plane;
  uint y = rem / res;
  uint x = rem - y * res;

  bool onBoundary = (x == 0u || y == 0u || z == 0u || x == res - 1u || y == res - 1u || z == res - 1u);
  if (!onBoundary) {
    uint step = uint(max(uInteriorStep, 1));
    if ((x % step) != 0u || (y % step) != 0u || (z % step) != 0u) return;
  }

  uint srcBase = index * 4u;
  float sr = srcVals[srcBase + 0u];
  float sg = srcVals[srcBase + 1u];
  float sb = srcVals[srcBase + 2u];
  float dr = dstVals[srcBase + 0u];
  float dg = dstVals[srcBase + 1u];
  float db = dstVals[srcBase + 2u];
  bool overflowPoint = outOfBounds(dr, dg, db);
  float plotR = (uShowOverflow != 0) ? dr : clamp01(dr);
  float plotG = (uShowOverflow != 0) ? dg : clamp01(dg);
  float plotB = (uShowOverflow != 0) ? db : clamp01(db);

  float cr;
  float cg;
  float cb;
  if (uShowOverflow != 0 && uHighlightOverflow != 0 && overflowPoint) {
    cr = 1.0;
    cg = 0.0;
    cb = 0.0;
  } else {
    mapDisplayColor(sr * 0.86 + clamp01(dr) * 0.14,
                    sg * 0.86 + clamp01(dg) * 0.14,
                    sb * 0.86 + clamp01(db) * 0.14,
                    cr, cg, cb);
  }

  uint outIndex = atomicAdd(outCount, 1u);
  uint vertBase = outIndex * 3u;
  vertVals[vertBase + 0u] = plotR * 2.0 - 1.0;
  vertVals[vertBase + 1u] = plotG * 2.0 - 1.0;
  vertVals[vertBase + 2u] = plotB * 2.0 - 1.0;
  colorVals[vertBase + 0u] = cr;
  colorVals[vertBase + 1u] = cg;
  colorVals[vertBase + 2u] = cb;
}
)GLSL";

  const GLuint shader = api.createShader(GL_COMPUTE_SHADER);
  if (shader == 0) return false;
  api.shaderSource(shader, 1, &kShaderSrc, nullptr);
  api.compileShader(shader);
  GLint compiled = 0;
  api.getShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    logViewerEvent(std::string("Identity compute shader compile failed: ") + readShaderLog(shader, false, api));
    api.deleteShader(shader);
    return false;
  }

  const GLuint program = api.createProgram();
  if (program == 0) {
    api.deleteShader(shader);
    return false;
  }
  api.attachShader(program, shader);
  api.linkProgram(program);
  api.deleteShader(shader);

  GLint linked = 0;
  api.getProgramiv(program, GL_LINK_STATUS, &linked);
  if (!linked) {
    logViewerEvent(std::string("Identity compute program link failed: ") + readShaderLog(program, true, api));
    api.deleteProgram(program);
    return false;
  }

  cache->program = program;
  cache->resLoc = api.getUniformLocation(program, "uResolution");
  cache->interiorStepLoc = api.getUniformLocation(program, "uInteriorStep");
  cache->showOverflowLoc = api.getUniformLocation(program, "uShowOverflow");
  cache->highlightOverflowLoc = api.getUniformLocation(program, "uHighlightOverflow");
  cache->available = cache->resLoc >= 0 && cache->interiorStepLoc >= 0 &&
                     cache->showOverflowLoc >= 0 && cache->highlightOverflowLoc >= 0;
  if (!cache->available) {
    logViewerEvent("Identity compute program missing one or more uniforms; falling back to CPU.");
    releaseIdentityComputeCache(cache);
    return false;
  }
  return true;
}

bool ensureInputCloudComputeProgram(InputCloudComputeCache* cache) {
  if (!cache) return false;
  if (cache->program != 0) return true;
  const ViewerGlComputeApi& api = viewerGlComputeApi();
  if (!api.available) return false;
  static const char* kShaderSrc = R"GLSL(
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer InputBuffer { float inputVals[]; };
layout(std430, binding = 1) writeonly buffer VertBuffer { float vertVals[]; };
layout(std430, binding = 2) writeonly buffer ColorBuffer { float colorVals[]; };
uniform int uPointCount;
uniform int uShowOverflow;
uniform int uHighlightOverflow;

float clamp01(float v) {
  return clamp(v, 0.0, 1.0);
}

bool outOfBounds(float r, float g, float b) {
  return r < 0.0 || r > 1.0 || g < 0.0 || g > 1.0 || b < 0.0 || b > 1.0;
}

void mapDisplayColor(float inR, float inG, float inB, out float outR, out float outG, out float outB) {
  float r = pow(clamp01(inR), 0.90);
  float g = pow(clamp01(inG), 0.90);
  float b = pow(clamp01(inB), 0.90);
  float luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  outR = clamp01(luma + (r - luma));
  outG = clamp01(luma + (g - luma));
  outB = clamp01(luma + (b - luma));
}

void main() {
  uint index = gl_GlobalInvocationID.x;
  if (index >= uint(max(uPointCount, 0))) return;
  uint inBase = index * 6u;
  float sr = inputVals[inBase + 0u];
  float sg = inputVals[inBase + 1u];
  float sb = inputVals[inBase + 2u];
  float dr = inputVals[inBase + 3u];
  float dg = inputVals[inBase + 4u];
  float db = inputVals[inBase + 5u];
  bool overflowPoint = outOfBounds(dr, dg, db);
  float plotR = (uShowOverflow != 0) ? dr : clamp01(dr);
  float plotG = (uShowOverflow != 0) ? dg : clamp01(dg);
  float plotB = (uShowOverflow != 0) ? db : clamp01(db);

  uint outBase = index * 3u;
  vertVals[outBase + 0u] = plotR * 2.0 - 1.0;
  vertVals[outBase + 1u] = plotG * 2.0 - 1.0;
  vertVals[outBase + 2u] = plotB * 2.0 - 1.0;

  if (uShowOverflow != 0 && uHighlightOverflow != 0 && overflowPoint) {
    colorVals[outBase + 0u] = 1.0;
    colorVals[outBase + 1u] = 0.0;
    colorVals[outBase + 2u] = 0.0;
  } else {
    float cr;
    float cg;
    float cb;
    mapDisplayColor(clamp01(sr), clamp01(sg), clamp01(sb), cr, cg, cb);
    colorVals[outBase + 0u] = cr;
    colorVals[outBase + 1u] = cg;
    colorVals[outBase + 2u] = cb;
  }
}
)GLSL";

  const GLuint shader = api.createShader(GL_COMPUTE_SHADER);
  if (shader == 0) return false;
  api.shaderSource(shader, 1, &kShaderSrc, nullptr);
  api.compileShader(shader);
  GLint compiled = 0;
  api.getShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    logViewerEvent(std::string("Input-cloud compute shader compile failed: ") + readShaderLog(shader, false, api));
    api.deleteShader(shader);
    return false;
  }

  const GLuint program = api.createProgram();
  if (program == 0) {
    api.deleteShader(shader);
    return false;
  }
  api.attachShader(program, shader);
  api.linkProgram(program);
  api.deleteShader(shader);

  GLint linked = 0;
  api.getProgramiv(program, GL_LINK_STATUS, &linked);
  if (!linked) {
    logViewerEvent(std::string("Input-cloud compute program link failed: ") + readShaderLog(program, true, api));
    api.deleteProgram(program);
    return false;
  }

  cache->program = program;
  cache->pointCountLoc = api.getUniformLocation(program, "uPointCount");
  cache->showOverflowLoc = api.getUniformLocation(program, "uShowOverflow");
  cache->highlightOverflowLoc = api.getUniformLocation(program, "uHighlightOverflow");
  cache->available = cache->pointCountLoc >= 0 &&
                     cache->showOverflowLoc >= 0 &&
                     cache->highlightOverflowLoc >= 0;
  if (!cache->available) {
    logViewerEvent("Input-cloud compute program missing one or more uniforms; falling back to CPU.");
    releaseInputCloudComputeCache(cache);
    return false;
  }
  return true;
}

struct ResolvedPayload;
struct InputCloudPayload;
OpenDRTParams buildResolvedParams(const ResolvedPayload& rp);

bool buildCubeDataOnGpu(const ResolvedPayload& payload,
                        const std::vector<float>& src,
                        const std::vector<float>& dst,
                        IdentityComputeCache* cache,
                        MeshData* out);

#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
bool buildCubeDataOnCuda(const ResolvedPayload& payload,
                         IdentityCudaDrawCache* cache,
                         MeshData* out);

bool buildInputCloudMeshOnCuda(const InputCloudPayload& payload,
                               const std::vector<float>& rawPoints,
                               InputCudaDrawCache* cache,
                               MeshData* out);
#endif

struct PointBufferCache {
  GLuint verts = 0;
  GLuint colors = 0;
  uint64_t uploadedSerial = 0;
  GLsizei uploadedPointCount = 0;
};

struct ResolvedDrawSource {
  GLuint verts = 0;
  GLuint colors = 0;
  const float* cpuVerts = nullptr;
  const float* cpuColors = nullptr;
  GLsizei pointCount = 0;
  bool bufferBacked = false;
  std::string label = "none";
};

void releasePointBufferCache(PointBufferCache* cache) {
  if (!cache) return;
  const ViewerGlBufferApi& api = viewerGlBufferApi();
  if (api.available) {
    if (cache->verts != 0) api.deleteBuffers(1, &cache->verts);
    if (cache->colors != 0) api.deleteBuffers(1, &cache->colors);
    api.bindBuffer(GL_ARRAY_BUFFER, 0);
  }
  cache->verts = 0;
  cache->colors = 0;
  cache->uploadedSerial = 0;
  cache->uploadedPointCount = 0;
}

bool ensurePointBufferCacheUploaded(const MeshData& mesh, PointBufferCache* cache) {
  if (!cache) return false;
  const ViewerGlBufferApi& api = viewerGlBufferApi();
  if (!api.available) return false;
  if (mesh.pointVerts.empty() || mesh.pointColors.empty()) return false;
  const GLsizei pointCount = static_cast<GLsizei>(mesh.pointVerts.size() / 3u);
  if (cache->uploadedSerial == mesh.serial && cache->uploadedPointCount == pointCount &&
      cache->verts != 0 && cache->colors != 0) {
    return true;
  }
  if (cache->verts == 0) api.genBuffers(1, &cache->verts);
  if (cache->colors == 0) api.genBuffers(1, &cache->colors);
  if (cache->verts == 0 || cache->colors == 0) return false;

  api.bindBuffer(GL_ARRAY_BUFFER, cache->verts);
  api.bufferData(
      GL_ARRAY_BUFFER,
      static_cast<ViewerGLsizeiptr>(mesh.pointVerts.size() * sizeof(float)),
      mesh.pointVerts.data(),
      GL_STATIC_DRAW);
  api.bindBuffer(GL_ARRAY_BUFFER, cache->colors);
  api.bufferData(
      GL_ARRAY_BUFFER,
      static_cast<ViewerGLsizeiptr>(mesh.pointColors.size() * sizeof(float)),
      mesh.pointColors.data(),
      GL_STATIC_DRAW);
  api.bindBuffer(GL_ARRAY_BUFFER, 0);

  cache->uploadedSerial = mesh.serial;
  cache->uploadedPointCount = pointCount;
  return true;
}

ResolvedDrawSource resolveDrawSource(const MeshData& mesh,
                                     bool useCudaIdentityBuffers,
                                     GLuint cudaIdentityVerts,
                                     GLuint cudaIdentityColors,
                                     GLsizei cudaIdentityPointCount,
                                     bool useComputeIdentityBuffers,
                                     GLuint computeIdentityVerts,
                                     GLuint computeIdentityColors,
                                     GLsizei computeIdentityPointCount,
                                     bool useCudaInputBuffers,
                                     GLuint cudaInputVerts,
                                     GLuint cudaInputColors,
                                     GLsizei cudaInputPointCount,
                                     bool useComputeInputBuffers,
                                     GLuint computeInputVerts,
                                     GLuint computeInputColors,
                                     GLsizei computeInputPointCount,
                                     bool usePointBuffers,
                                     GLuint uploadedVerts,
                                     GLuint uploadedColors) {
  ResolvedDrawSource source{};
  if (useCudaIdentityBuffers) {
    source.bufferBacked = true;
    source.verts = cudaIdentityVerts;
    source.colors = cudaIdentityColors;
    source.pointCount = cudaIdentityPointCount;
    source.label = "cuda-identity-buffer";
    return source;
  }
  if (useComputeIdentityBuffers) {
    source.bufferBacked = true;
    source.verts = computeIdentityVerts;
    source.colors = computeIdentityColors;
    source.pointCount = computeIdentityPointCount;
    source.label = "identity-compute";
    return source;
  }
  if (useCudaInputBuffers) {
    source.bufferBacked = true;
    source.verts = cudaInputVerts;
    source.colors = cudaInputColors;
    source.pointCount = cudaInputPointCount;
    source.label = "cuda-input-buffer";
    return source;
  }
  if (useComputeInputBuffers) {
    source.bufferBacked = true;
    source.verts = computeInputVerts;
    source.colors = computeInputColors;
    source.pointCount = computeInputPointCount;
    source.label = "input-compute";
    return source;
  }
  if (usePointBuffers) {
    source.bufferBacked = true;
    source.verts = uploadedVerts;
    source.colors = uploadedColors;
    source.pointCount = static_cast<GLsizei>(mesh.pointVerts.size() / 3u);
    source.label = mesh.packBackendLabel == "metal-compute-mesh" ? "metal-uploaded-buffer" : "gl-buffer";
    return source;
  }
  if (!mesh.pointVerts.empty() && !mesh.pointColors.empty()) {
    source.cpuVerts = mesh.pointVerts.data();
    source.cpuColors = mesh.pointColors.data();
    source.pointCount = static_cast<GLsizei>(mesh.pointVerts.size() / 3u);
    source.label = "cpu-array";
  }
  return source;
}

void mapDisplayColor(float inR, float inG, float inB, float* outR, float* outG, float* outB) {
  float r = clamp01(inR);
  float g = clamp01(inG);
  float b = clamp01(inB);
  // Keep mapping close to source hues to match scope-like display.
  r = std::pow(r, 0.90f);
  g = std::pow(g, 0.90f);
  b = std::pow(b, 0.90f);
  const float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
  const float sat = 1.00f;
  r = clamp01(luma + (r - luma) * sat);
  g = clamp01(luma + (g - luma) * sat);
  b = clamp01(luma + (b - luma) * sat);
  *outR = r;
  *outG = g;
  *outB = b;
}

void applyLookValuesToResolved(OpenDRTParams& p, const LookPresetValues& s) {
  p.tn_con = s.tn_con; p.tn_sh = s.tn_sh; p.tn_toe = s.tn_toe; p.tn_off = s.tn_off;
  p.tn_hcon_enable = s.tn_hcon_enable; p.tn_hcon = s.tn_hcon; p.tn_hcon_pv = s.tn_hcon_pv; p.tn_hcon_st = s.tn_hcon_st;
  p.tn_lcon_enable = s.tn_lcon_enable; p.tn_lcon = s.tn_lcon; p.tn_lcon_w = s.tn_lcon_w;
  p.cwp = s.cwp; p.cwp_lm = s.cwp_lm;
  p.rs_sa = s.rs_sa; p.rs_rw = s.rs_rw; p.rs_bw = s.rs_bw;
  p.pt_enable = s.pt_enable; p.pt_lml = s.pt_lml; p.pt_lml_r = s.pt_lml_r; p.pt_lml_g = s.pt_lml_g; p.pt_lml_b = s.pt_lml_b;
  p.pt_lmh = s.pt_lmh; p.pt_lmh_r = s.pt_lmh_r; p.pt_lmh_b = s.pt_lmh_b;
  p.ptl_enable = s.ptl_enable; p.ptl_c = s.ptl_c; p.ptl_m = s.ptl_m; p.ptl_y = s.ptl_y;
  p.ptm_enable = s.ptm_enable; p.ptm_low = s.ptm_low; p.ptm_low_rng = s.ptm_low_rng; p.ptm_low_st = s.ptm_low_st;
  p.ptm_high = s.ptm_high; p.ptm_high_rng = s.ptm_high_rng; p.ptm_high_st = s.ptm_high_st;
  p.brl_enable = s.brl_enable; p.brl = s.brl; p.brl_r = s.brl_r; p.brl_g = s.brl_g; p.brl_b = s.brl_b; p.brl_rng = s.brl_rng; p.brl_st = s.brl_st;
  p.brlp_enable = s.brlp_enable; p.brlp = s.brlp; p.brlp_r = s.brlp_r; p.brlp_g = s.brlp_g; p.brlp_b = s.brlp_b;
  p.hc_enable = s.hc_enable; p.hc_r = s.hc_r; p.hc_r_rng = s.hc_r_rng;
  p.hs_rgb_enable = s.hs_rgb_enable; p.hs_r = s.hs_r; p.hs_r_rng = s.hs_r_rng; p.hs_g = s.hs_g; p.hs_g_rng = s.hs_g_rng; p.hs_b = s.hs_b; p.hs_b_rng = s.hs_b_rng;
  p.hs_cmy_enable = s.hs_cmy_enable; p.hs_c = s.hs_c; p.hs_c_rng = s.hs_c_rng; p.hs_m = s.hs_m; p.hs_m_rng = s.hs_m_rng; p.hs_y = s.hs_y; p.hs_y_rng = s.hs_y_rng;
}

void applyTonescaleValuesToResolved(OpenDRTParams& p, const TonescalePresetValues& t) {
  p.tn_con = t.tn_con; p.tn_sh = t.tn_sh; p.tn_toe = t.tn_toe; p.tn_off = t.tn_off;
  p.tn_hcon_enable = t.tn_hcon_enable; p.tn_hcon = t.tn_hcon; p.tn_hcon_pv = t.tn_hcon_pv; p.tn_hcon_st = t.tn_hcon_st;
  p.tn_lcon_enable = t.tn_lcon_enable; p.tn_lcon = t.tn_lcon; p.tn_lcon_w = t.tn_lcon_w;
}

bool parseLookValues(const std::string& in, LookPresetValues* v) {
  if (!v) return false;
  std::istringstream is(in);
  return static_cast<bool>(
      is >> v->tn_con >> v->tn_sh >> v->tn_toe >> v->tn_off
         >> v->tn_hcon_enable >> v->tn_hcon >> v->tn_hcon_pv >> v->tn_hcon_st
         >> v->tn_lcon_enable >> v->tn_lcon >> v->tn_lcon_w
         >> v->cwp >> v->cwp_lm
         >> v->rs_sa >> v->rs_rw >> v->rs_bw
         >> v->pt_enable
         >> v->pt_lml >> v->pt_lml_r >> v->pt_lml_g >> v->pt_lml_b
         >> v->pt_lmh >> v->pt_lmh_r >> v->pt_lmh_b
         >> v->ptl_enable >> v->ptl_c >> v->ptl_m >> v->ptl_y
         >> v->ptm_enable >> v->ptm_low >> v->ptm_low_rng >> v->ptm_low_st
         >> v->ptm_high >> v->ptm_high_rng >> v->ptm_high_st
         >> v->brl_enable >> v->brl >> v->brl_r >> v->brl_g >> v->brl_b
         >> v->brl_rng >> v->brl_st
         >> v->brlp_enable >> v->brlp >> v->brlp_r >> v->brlp_g >> v->brlp_b
         >> v->hc_enable >> v->hc_r >> v->hc_r_rng
         >> v->hs_rgb_enable >> v->hs_r >> v->hs_r_rng
         >> v->hs_g >> v->hs_g_rng >> v->hs_b >> v->hs_b_rng
         >> v->hs_cmy_enable >> v->hs_c >> v->hs_c_rng >> v->hs_m >> v->hs_m_rng
         >> v->hs_y >> v->hs_y_rng);
}

bool parseTonescaleValues(const std::string& in, TonescalePresetValues* v) {
  if (!v) return false;
  std::istringstream is(in);
  return static_cast<bool>(
      is >> v->tn_con >> v->tn_sh >> v->tn_toe >> v->tn_off
         >> v->tn_hcon_enable >> v->tn_hcon >> v->tn_hcon_pv >> v->tn_hcon_st
         >> v->tn_lcon_enable >> v->tn_lcon >> v->tn_lcon_w);
}

bool extractJsonNumber(const std::string& src, const char* key, double* out) {
  if (!out) return false;
  const std::string pat = std::string("\"") + key + "\":";
  const size_t p = src.find(pat);
  if (p == std::string::npos) return false;
  const char* begin = src.c_str() + p + pat.size();
  char* end = nullptr;
  const double v = std::strtod(begin, &end);
  if (end == begin) return false;
  *out = v;
  return true;
}

bool extractJsonInt(const std::string& src, const char* key, int* out) {
  double d = 0.0;
  if (!extractJsonNumber(src, key, &d)) return false;
  *out = static_cast<int>(d);
  return true;
}

bool extractJsonString(const std::string& src, const char* key, std::string* out) {
  if (!out) return false;
  const std::string pat = std::string("\"") + key + "\":\"";
  const size_t p = src.find(pat);
  if (p == std::string::npos) return false;
  size_t i = p + pat.size();
  std::string v;
  while (i < src.size()) {
    char c = src[i++];
    if (c == '\\' && i < src.size()) {
      const char e = src[i++];
      switch (e) {
        case 'n': v.push_back('\n'); break;
        case 'r': v.push_back('\r'); break;
        case 't': v.push_back('\t'); break;
        case '\\': v.push_back('\\'); break;
        case '"': v.push_back('"'); break;
        default: v.push_back(e); break;
      }
      continue;
    }
    if (c == '"') {
      *out = v;
      return true;
    }
    v.push_back(c);
  }
  return false;
}

struct ResolvedPayload {
  uint64_t seq = 0;
  std::string senderId;
  int resolution = 33;
  int in_gamut = 14;
  int in_oetf = 1;
  int lookPreset = 0;
  int tonescalePreset = 0;
  int creativeWhitePreset = 0;
  int displayEncodingPreset = 0;
  float tn_Lp = 100.0f;
  float tn_Lg = 10.0f;
  float tn_gb = 0.13f;
  float pt_hdr = 0.5f;
  int crv_enable = 0;
  int clamp = 1;
  int tn_su = 1;
  int display_gamut = 0;
  int eotf = 2;
  std::string sourceMode = "identity";
  int plotInLinear = 0;
  int showOverflow = 0;
  int highlightOverflow = 1;
  int alwaysOnTop = 0;
  std::string quality = "Medium";
  std::string paramHash;
  std::string lookPayload;
  std::string tonescalePayload;
};

#if defined(__APPLE__)
bool buildCubeDataOnMetal(const ResolvedPayload& payload,
                          OpenDRTViewerMetal::MeshCache* cache,
                          MeshData* out) {
  if (!out) return false;
  std::string error;
  std::string transformBackend;
  float maxDelta = 0.0f;
  if (!OpenDRTViewerMetal::buildIdentityMesh(
          cache,
          buildResolvedParams(payload),
          payload.resolution,
          payload.showOverflow != 0,
          payload.highlightOverflow != 0,
          out->serial,
          &transformBackend,
          &maxDelta,
          &error)) {
    if (!error.empty()) {
      logViewerEvent(std::string("Metal identity mesh build failed: ") + error);
    }
    return false;
  }
  out->transformBackendLabel = transformBackend.empty() ? "metal" : transformBackend;
  out->maxDelta = maxDelta;
  out->pointCount = static_cast<size_t>(cache ? cache->pointCount : 0);
  return cache && cache->available;
}
#endif

struct InputCloudPayload {
  uint64_t seq = 0;
  std::string senderId;
  int resolution = 33;
  std::string quality = "Medium";
  int showOverflow = 0;
  int highlightOverflow = 1;
  std::string paramHash;
  std::string points;
};

#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
bool buildCubeDataOnCuda(const ResolvedPayload& payload,
                         IdentityCudaDrawCache* cache,
                         MeshData* out) {
  if (!cache || !out) return false;
  const int res = (payload.resolution <= 25) ? 25 : (payload.resolution <= 41 ? 41 : 57);
  if (!ensureIdentityCudaBuffers(cache, res)) return false;
  OpenDRTViewerCuda::IdentityRequest request{};
  request.resolution = res;
  request.showOverflow = payload.showOverflow;
  request.highlightOverflow = payload.highlightOverflow;
  request.params = buildResolvedParams(payload);
  std::string error;
  if (!OpenDRTViewerCuda::buildIdentityMesh(&cache->cache, request, out->serial, &error)) {
    if (!error.empty()) {
      logViewerEvent(std::string("CUDA identity mesh build failed: ") + error);
    }
    return false;
  }
  out->pointCount = static_cast<size_t>(cache->cache.pointCount);
  return out->pointCount > 0;
}

bool buildInputCloudMeshOnCuda(const InputCloudPayload& payload,
                               const std::vector<float>& rawPoints,
                               InputCudaDrawCache* cache,
                               MeshData* out) {
  if (!cache || !out) return false;
  const size_t pointCount = rawPoints.size() / 6u;
  if (pointCount == 0) return false;
  if (!ensureInputCudaBuffers(cache, pointCount)) return false;
  OpenDRTViewerCuda::InputRequest request{};
  request.pointCount = static_cast<int>(pointCount);
  request.showOverflow = payload.showOverflow;
  request.highlightOverflow = payload.highlightOverflow;
  std::string error;
  if (!OpenDRTViewerCuda::buildInputCloudMesh(&cache->cache, request, rawPoints, out->serial, &error)) {
    if (!error.empty()) {
      logViewerEvent(std::string("CUDA input-cloud mesh build failed: ") + error);
    }
    return false;
  }
  out->pointCount = static_cast<size_t>(cache->cache.pointCount);
  return out->pointCount > 0;
}
#endif

bool parseParamsMessage(const std::string& msg, ResolvedPayload* out) {
  if (!out) return false;
  std::string type;
  if (!extractJsonString(msg, "type", &type)) return false;
  if (type != "params_snapshot" && type != "params_delta") return false;

  double seqD = 0.0;
  if (!extractJsonNumber(msg, "seq", &seqD)) return false;
  out->seq = static_cast<uint64_t>(seqD < 0.0 ? 0.0 : seqD);
  extractJsonString(msg, "senderId", &out->senderId);

  int resolution = 33;
  if (extractJsonInt(msg, "resolution", &resolution)) out->resolution = resolution;
  extractJsonString(msg, "quality", &out->quality);
  extractJsonString(msg, "sourceMode", &out->sourceMode);
  extractJsonInt(msg, "plotInLinear", &out->plotInLinear);
  extractJsonInt(msg, "showOverflow", &out->showOverflow);
  extractJsonInt(msg, "highlightOverflow", &out->highlightOverflow);
  extractJsonInt(msg, "alwaysOnTop", &out->alwaysOnTop);
  extractJsonString(msg, "paramHash", &out->paramHash);
  extractJsonInt(msg, "in_gamut", &out->in_gamut);
  extractJsonInt(msg, "in_oetf", &out->in_oetf);
  extractJsonInt(msg, "lookPreset", &out->lookPreset);
  extractJsonInt(msg, "tonescalePreset", &out->tonescalePreset);
  extractJsonInt(msg, "creativeWhitePreset", &out->creativeWhitePreset);
  extractJsonInt(msg, "displayEncodingPreset", &out->displayEncodingPreset);
  double f = 0.0;
  if (extractJsonNumber(msg, "tn_Lp", &f)) out->tn_Lp = static_cast<float>(f);
  if (extractJsonNumber(msg, "tn_Lg", &f)) out->tn_Lg = static_cast<float>(f);
  if (extractJsonNumber(msg, "tn_gb", &f)) out->tn_gb = static_cast<float>(f);
  if (extractJsonNumber(msg, "pt_hdr", &f)) out->pt_hdr = static_cast<float>(f);
  extractJsonInt(msg, "crv_enable", &out->crv_enable);
  extractJsonInt(msg, "clamp", &out->clamp);
  extractJsonInt(msg, "tn_su", &out->tn_su);
  extractJsonInt(msg, "display_gamut", &out->display_gamut);
  extractJsonInt(msg, "eotf", &out->eotf);
  extractJsonString(msg, "lookPayload", &out->lookPayload);
  extractJsonString(msg, "tonescalePayload", &out->tonescalePayload);
  return true;
}

bool parseInputCloudMessage(const std::string& msg, InputCloudPayload* out) {
  if (!out) return false;
  std::string type;
  if (!extractJsonString(msg, "type", &type) || type != "input_cloud") return false;
  double seqD = 0.0;
  if (!extractJsonNumber(msg, "seq", &seqD)) return false;
  out->seq = static_cast<uint64_t>(seqD < 0.0 ? 0.0 : seqD);
  extractJsonString(msg, "senderId", &out->senderId);
  int resolution = 33;
  if (extractJsonInt(msg, "resolution", &resolution)) out->resolution = resolution;
  extractJsonString(msg, "quality", &out->quality);
  extractJsonInt(msg, "showOverflow", &out->showOverflow);
  extractJsonInt(msg, "highlightOverflow", &out->highlightOverflow);
  extractJsonString(msg, "paramHash", &out->paramHash);
  if (!extractJsonString(msg, "points", &out->points)) return false;
  return true;
}

std::string heartbeatAckJson() {
  const int visible = gWindowVisible.load(std::memory_order_relaxed) ? 1 : 0;
  const int minimized = gWindowIconified.load(std::memory_order_relaxed) ? 1 : 0;
  const int focused = gWindowFocused.load(std::memory_order_relaxed) ? 1 : 0;
  const int active = (visible != 0 && minimized == 0) ? 1 : 0;
  std::ostringstream os;
  os << "{\"type\":\"heartbeat_ack\",\"active\":" << active << ",\"visible\":" << visible
     << ",\"minimized\":" << minimized << ",\"focused\":" << focused << "}";
  return os.str();
}

std::string handleIncomingLine(const std::string& line) {
  if (line.empty()) return std::string();
  if (line.find("\"type\":\"hello\"") != std::string::npos || line.find("\"type\":\"open_session\"") != std::string::npos) {
    gConnected.store(true);
    logViewerEvent("Received session hello/open_session.");
    return std::string();
  }
  if (line.find("\"type\":\"close_session\"") != std::string::npos) {
    logViewerEvent("Received close_session.");
    gRun.store(false);
    return std::string();
  }
  if (line.find("\"type\":\"heartbeat\"") != std::string::npos) {
    gConnected.store(true);
    logViewerEvent("Received heartbeat.");
    return heartbeatAckJson();
  }
  if (line.find("\"type\":\"bring_to_front\"") != std::string::npos) {
    gBringToFront.store(true);
    gConnected.store(true);
    logViewerEvent("Received bring_to_front.");
    return std::string();
  }
  ResolvedPayload payload{};
  InputCloudPayload cloud{};
  const bool isParams = parseParamsMessage(line, &payload);
  const bool isCloud = isParams ? false : parseInputCloudMessage(line, &cloud);
  if (!isParams && !isCloud) return std::string();
  std::lock_guard<std::mutex> lock(gMsgMutex);
  if (isParams) {
    std::ostringstream os;
    os << "Queued params message: sender=" << payload.senderId
       << " seq=" << payload.seq
       << " sourceMode=" << payload.sourceMode
       << " quality=" << payload.quality
       << " res=" << payload.resolution;
    logViewerEvent(os.str());
    gPendingParamsMsg.seq = payload.seq;
    gPendingParamsMsg.line = line;
    gHasPendingParamsMsg = true;
  } else {
    std::ostringstream os;
    os << "Queued input cloud message: sender=" << cloud.senderId
       << " seq=" << cloud.seq
       << " quality=" << cloud.quality
       << " res=" << cloud.resolution
       << " pointBytes=" << cloud.points.size();
    logViewerEvent(os.str());
    gPendingCloudMsg.seq = cloud.seq;
    gPendingCloudMsg.line = line;
    gHasPendingCloudMsg = true;
  }
  gConnected.store(true);
  return std::string();
}

OpenDRTParams buildResolvedParams(const ResolvedPayload& rp) {
  OpenDRTParams p{};
  p.in_gamut = rp.in_gamut;
  p.in_oetf = rp.in_oetf;
  p.tn_Lp = rp.tn_Lp;
  p.tn_Lg = rp.tn_Lg;
  p.tn_gb = rp.tn_gb;
  p.pt_hdr = rp.pt_hdr;
  p.crv_enable = rp.crv_enable;
  p.clamp = rp.clamp;
  p.tn_su = rp.tn_su;
  p.display_gamut = rp.display_gamut;
  p.eotf = rp.eotf;

  applyLookPresetToResolved(p, rp.lookPreset);
  applyDisplayEncodingPreset(p, rp.displayEncodingPreset);
  if (rp.creativeWhitePreset > 0) p.cwp = rp.creativeWhitePreset - 1;

  LookPresetValues lookVals{};
  if (!rp.lookPayload.empty() && parseLookValues(rp.lookPayload, &lookVals)) {
    applyLookValuesToResolved(p, lookVals);
  }

  if (rp.tonescalePreset > 0) {
    applyTonescalePresetToResolved(p, rp.tonescalePreset);
  }
  TonescalePresetValues toneVals{};
  if (!rp.tonescalePayload.empty() && parseTonescaleValues(rp.tonescalePayload, &toneVals)) {
    applyTonescaleValuesToResolved(p, toneVals);
  }
  if (rp.plotInLinear != 0) {
    p.eotf = 0;
  }
  return p;
}

bool buildCubeDataOnGpu(const ResolvedPayload& payload,
                        const std::vector<float>& src,
                        const std::vector<float>& dst,
                        IdentityComputeCache* cache,
                        MeshData* out) {
  if (!cache || !out) return false;
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  const ViewerGlComputeApi& computeApi = viewerGlComputeApi();
  if (!bufferApi.available || !computeApi.available || !ensureIdentityComputeProgram(cache)) return false;

  const int res = (payload.resolution <= 25) ? 25 : (payload.resolution <= 41 ? 41 : 57);
  const size_t count = static_cast<size_t>(res) * static_cast<size_t>(res) * static_cast<size_t>(res);
  const int interiorStep = (res <= 25) ? 2 : (res <= 41 ? 2 : 3);

  auto ensureBuffer = [&](GLuint* id) {
    if (*id == 0) bufferApi.genBuffers(1, id);
    return *id != 0;
  };
  if (!ensureBuffer(&cache->src) || !ensureBuffer(&cache->dst) || !ensureBuffer(&cache->verts) ||
      !ensureBuffer(&cache->colors) || !ensureBuffer(&cache->counter)) {
    logViewerEvent("Identity compute buffer allocation failed; falling back to CPU.");
    return false;
  }

  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->src);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<ViewerGLsizeiptr>(src.size() * sizeof(float)),
                       src.data(),
                       GL_DYNAMIC_DRAW);
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->dst);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<ViewerGLsizeiptr>(dst.size() * sizeof(float)),
                       dst.data(),
                       GL_DYNAMIC_DRAW);
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->verts);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<ViewerGLsizeiptr>(count * 3u * sizeof(float)),
                       nullptr,
                       GL_DYNAMIC_DRAW);
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->colors);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<ViewerGLsizeiptr>(count * 3u * sizeof(float)),
                       nullptr,
                       GL_DYNAMIC_DRAW);
  const GLuint zero = 0;
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->counter);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), &zero, GL_DYNAMIC_DRAW);

  computeApi.useProgram(cache->program);
  computeApi.uniform1i(cache->resLoc, res);
  computeApi.uniform1i(cache->interiorStepLoc, interiorStep);
  computeApi.uniform1i(cache->showOverflowLoc, payload.showOverflow != 0 ? 1 : 0);
  computeApi.uniform1i(cache->highlightOverflowLoc, payload.highlightOverflow != 0 ? 1 : 0);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cache->src);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cache->dst);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cache->verts);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cache->colors);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cache->counter);
  const GLuint groups = static_cast<GLuint>((count + 63u) / 64u);
  computeApi.dispatchCompute(groups, 1, 1);
  computeApi.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
  computeApi.useProgram(0);

  GLuint outCount = 0;
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->counter);
  if (bufferApi.getBufferSubData) {
    bufferApi.getBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &outCount);
  } else {
    logViewerEvent("Identity compute path missing glGetBufferSubData; falling back to CPU.");
    bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return false;
  }
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

  if (viewerDiagnosticsEnabled()) {
    const size_t expectedCount = expectedIdentityDisplayPointCount(res);
    if (static_cast<size_t>(outCount) != expectedCount) {
      std::ostringstream os;
      os << "Identity compute parity mismatch: res=" << res
         << " expectedPoints=" << expectedCount
         << " gpuPoints=" << outCount
         << " renderOk=" << (out->renderOk ? "1" : "0");
      logViewerDiagnostic(true, os.str());
    }
  }

  cache->builtSerial = out->serial;
  cache->pointCount = static_cast<GLsizei>(outCount);
  out->pointCount = static_cast<size_t>(outCount);
  return true;
}

void buildCubeData(const ResolvedPayload& payload,
                   const ViewerGpuCapabilities& gpuCaps,
                   ViewerRuntimeState* runtime,
                   IdentityComputeCache* computeCache,
#if defined(__APPLE__)
                   OpenDRTViewerMetal::MeshCache* metalCache,
#endif
#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
                   IdentityCudaDrawCache* cudaCache,
#endif
                   MeshData* out) {
  if (!out) return;
  const int res = (payload.resolution <= 25) ? 25 : (payload.resolution <= 41 ? 41 : 57);
  const size_t count = static_cast<size_t>(res) * static_cast<size_t>(res) * static_cast<size_t>(res);

  std::vector<float> src(count * 4u, 1.0f);
  auto idx3 = [res](int x, int y, int z) -> size_t {
    return (static_cast<size_t>(z) * static_cast<size_t>(res) + static_cast<size_t>(y)) * static_cast<size_t>(res) +
           static_cast<size_t>(x);
  };
  const float denom = static_cast<float>(res - 1);
  for (int z = 0; z < res; ++z) {
    for (int y = 0; y < res; ++y) {
      for (int x = 0; x < res; ++x) {
        const size_t i = idx3(x, y, z) * 4u;
        src[i + 0] = static_cast<float>(x) / denom;
        src[i + 1] = static_cast<float>(y) / denom;
        src[i + 2] = static_cast<float>(z) / denom;
      }
    }
  }

  MeshData mesh{};
  mesh.resolution = res;
  mesh.quality = payload.quality;
  mesh.paramHash = payload.paramHash;
  mesh.serial = nextMeshSerial();
  mesh.pointCount = 0;
  mesh.renderOk = true;

  std::vector<float> dst;
  bool haveDst = false;
  auto ensureCpuTransformed = [&]() {
    if (haveDst) return;
    dst.assign(count * 4u, 1.0f);
    OpenDRTProcessor proc(buildResolvedParams(payload));
    mesh.renderOk = proc.render(src.data(), dst.data(), static_cast<int>(count), 1, true, true);
    if (!mesh.renderOk) {
      dst = src;
    }
    mesh.transformBackendLabel = proc.lastBackendLabel();
    float maxDelta = 0.0f;
    for (size_t i = 0; i < count; ++i) {
      const size_t si = i * 4u;
      const float dr = std::fabs(dst[si + 0] - src[si + 0]);
      const float dg = std::fabs(dst[si + 1] - src[si + 1]);
      const float db = std::fabs(dst[si + 2] - src[si + 2]);
      if (dr > maxDelta) maxDelta = dr;
      if (dg > maxDelta) maxDelta = dg;
      if (db > maxDelta) maxDelta = db;
    }
    mesh.maxDelta = maxDelta;
    haveDst = true;
  };

#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
  if (gpuCaps.activeBackend == ViewerGpuBackend::CudaComputeMesh &&
      runtime && !runtime->identityGpuDemoted &&
      buildCubeDataOnCuda(payload, cudaCache, &mesh)) {
    mesh.transformBackendLabel = "cuda";
    mesh.packBackendLabel = "cuda-compute-mesh";
    if (gpuCaps.parityCheckEnabled) {
      ensureCpuTransformed();
      std::vector<float> cpuDst(count * 4u, 1.0f);
      OpenDRTProcessor cpuRefProc(buildResolvedParams(payload));
      if (cpuRefProc.renderCPUReference(
              src.data(),
              cpuDst.data(),
              static_cast<int>(count),
              1,
              count * 4u * sizeof(float),
              count * 4u * sizeof(float))) {
        float parityMaxDelta = 0.0f;
        for (size_t i = 0; i < count * 4u; ++i) {
          const float delta = std::fabs(dst[i] - cpuDst[i]);
          if (delta > parityMaxDelta) parityMaxDelta = delta;
        }
        mesh.parityMaxDelta = parityMaxDelta;
        if (parityMaxDelta > 1.0e-4f) {
          ++runtime->identityParityFailures;
          std::ostringstream os;
          os << "Identity transform parity mismatch: backend=cuda"
             << " res=" << res
             << " maxDelta=" << parityMaxDelta
             << " failures=" << runtime->identityParityFailures;
          logViewerDiagnostic(true, os.str());
          if (runtime->identityParityFailures >= 2) {
            runtime->identityGpuDemoted = true;
            runtime->identityDemotionReason = "identity-parity";
            logViewerEvent("Identity CUDA path demoted to CPU after sustained parity mismatch.");
          }
        } else {
          runtime->identityParityFailures = 0;
        }
      }
    }
    if (!(runtime && runtime->identityGpuDemoted)) {
      *out = std::move(mesh);
      return;
    }
  } else if (gpuCaps.activeBackend == ViewerGpuBackend::CudaComputeMesh && runtime && !runtime->identityGpuDemoted) {
    runtime->identityGpuDemoted = true;
    runtime->identityDemotionReason = "cuda-runtime-failure";
    logViewerEvent("Identity CUDA path demoted to CPU after runtime failure.");
  }
#endif

#if defined(__APPLE__)
  if (gpuCaps.activeBackend == ViewerGpuBackend::MetalComputeMesh &&
      !(runtime && runtime->identityGpuDemoted) &&
      buildCubeDataOnMetal(payload, metalCache, &mesh)) {
    mesh.packBackendLabel = "metal-compute-mesh";
    *out = std::move(mesh);
    return;
  } else if (gpuCaps.activeBackend == ViewerGpuBackend::MetalComputeMesh && runtime && !runtime->identityGpuDemoted) {
    runtime->identityGpuDemoted = true;
    runtime->identityDemotionReason = "metal-runtime-failure";
    logViewerEvent("Identity Metal path demoted to CPU after runtime failure.");
  }
#endif

  ensureCpuTransformed();

  if (gpuCaps.parityCheckEnabled && mesh.renderOk && mesh.transformBackendLabel != "cpu") {
    std::vector<float> cpuDst(count * 4u, 1.0f);
    OpenDRTProcessor cpuRefProc(buildResolvedParams(payload));
    if (cpuRefProc.renderCPUReference(
            src.data(),
            cpuDst.data(),
            static_cast<int>(count),
            1,
            count * 4u * sizeof(float),
            count * 4u * sizeof(float))) {
      float parityMaxDelta = 0.0f;
      for (size_t i = 0; i < count * 4u; ++i) {
        const float delta = std::fabs(dst[i] - cpuDst[i]);
        if (delta > parityMaxDelta) parityMaxDelta = delta;
      }
      mesh.parityMaxDelta = parityMaxDelta;
      if (parityMaxDelta > 1.0e-4f && runtime) {
        ++runtime->identityParityFailures;
        std::ostringstream os;
        os << "Identity transform parity mismatch: backend=" << mesh.transformBackendLabel
           << " res=" << res
           << " maxDelta=" << parityMaxDelta
           << " failures=" << runtime->identityParityFailures;
        logViewerDiagnostic(true, os.str());
        if (runtime->identityParityFailures >= 2) {
          runtime->identityGpuDemoted = true;
          runtime->identityDemotionReason = "identity-parity";
        }
      } else if (runtime) {
        runtime->identityParityFailures = 0;
      }
    }
  }
  if (gpuCaps.activeBackend == ViewerGpuBackend::OpenGlComputeIdentity &&
      !(runtime && runtime->identityGpuDemoted) &&
      buildCubeDataOnGpu(payload, src, dst, computeCache, &mesh)) {
    mesh.packBackendLabel = "gl-compute-mesh";
    *out = std::move(mesh);
    return;
  } else if (gpuCaps.activeBackend == ViewerGpuBackend::OpenGlComputeIdentity && runtime && !runtime->identityGpuDemoted) {
    runtime->identityGpuDemoted = true;
    runtime->identityDemotionReason = "gl-runtime-failure";
    logViewerEvent("Identity GL compute path demoted to CPU after runtime failure.");
  }

  const int interiorStep = (res <= 25) ? 2 : (res <= 41 ? 2 : 3);
  mesh.pointVerts.reserve((count / static_cast<size_t>(interiorStep * interiorStep * interiorStep) + 1u) * 3u);
  mesh.pointColors.reserve(mesh.pointVerts.capacity());
  for (int z = 0; z < res; ++z) {
    for (int y = 0; y < res; ++y) {
      for (int x = 0; x < res; ++x) {
        const bool onBoundary = (x == 0 || y == 0 || z == 0 || x == res - 1 || y == res - 1 || z == res - 1);
        if (!onBoundary) {
          if ((x % interiorStep) != 0 || (y % interiorStep) != 0 || (z % interiorStep) != 0) continue;
        }
        const size_t i = idx3(x, y, z);
        const size_t si = i * 4u;
        const float rx = dst[si + 0];
        const float ry = dst[si + 1];
        const float rz = dst[si + 2];
        const bool overflowPoint = rgbOutOfBounds(rx, ry, rz);
        const float plotX = payload.showOverflow != 0 ? rx : clamp01(rx);
        const float plotY = payload.showOverflow != 0 ? ry : clamp01(ry);
        const float plotZ = payload.showOverflow != 0 ? rz : clamp01(rz);
        mesh.pointVerts.push_back(plotX * 2.0f - 1.0f);
        mesh.pointVerts.push_back(plotY * 2.0f - 1.0f);
        mesh.pointVerts.push_back(plotZ * 2.0f - 1.0f);
        // Blend source hue with transformed value to better communicate shape and look intent.
        float cr = 0.0f, cg = 0.0f, cb = 0.0f;
        if (payload.showOverflow != 0 && payload.highlightOverflow != 0 && overflowPoint) {
          cr = 1.0f;
          cg = 0.0f;
          cb = 0.0f;
        } else {
          const float mixR = src[si + 0] * 0.86f + clamp01(rx) * 0.14f;
          const float mixG = src[si + 1] * 0.86f + clamp01(ry) * 0.14f;
          const float mixB = src[si + 2] * 0.86f + clamp01(rz) * 0.14f;
          mapDisplayColor(mixR, mixG, mixB, &cr, &cg, &cb);
        }
        mesh.pointColors.push_back(cr);
        mesh.pointColors.push_back(cg);
        mesh.pointColors.push_back(cb);
      }
    }
  }
  mesh.pointCount = mesh.pointVerts.size() / 3u;
  mesh.packBackendLabel = "cpu-array";
  *out = std::move(mesh);
}

bool buildInputCloudMeshOnGpu(const InputCloudPayload& payload,
                              const std::vector<float>& rawPoints,
                              InputCloudComputeCache* cache,
                              MeshData* out) {
  if (!cache || !out) return false;
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  const ViewerGlComputeApi& computeApi = viewerGlComputeApi();
  if (!bufferApi.available || !computeApi.available || !ensureInputCloudComputeProgram(cache)) return false;
  const size_t pointCount = rawPoints.size() / 6u;
  if (pointCount == 0) return false;

  auto ensureBuffer = [&](GLuint* id) {
    if (*id == 0) bufferApi.genBuffers(1, id);
    return *id != 0;
  };
  if (!ensureBuffer(&cache->input) || !ensureBuffer(&cache->verts) || !ensureBuffer(&cache->colors)) {
    logViewerEvent("Input-cloud compute buffer allocation failed; falling back to CPU.");
    return false;
  }

  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->input);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<ViewerGLsizeiptr>(rawPoints.size() * sizeof(float)),
                       rawPoints.data(),
                       GL_DYNAMIC_DRAW);
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->verts);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<ViewerGLsizeiptr>(pointCount * 3u * sizeof(float)),
                       nullptr,
                       GL_DYNAMIC_DRAW);
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->colors);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<ViewerGLsizeiptr>(pointCount * 3u * sizeof(float)),
                       nullptr,
                       GL_DYNAMIC_DRAW);

  computeApi.useProgram(cache->program);
  computeApi.uniform1i(cache->pointCountLoc, static_cast<GLint>(pointCount));
  computeApi.uniform1i(cache->showOverflowLoc, payload.showOverflow != 0 ? 1 : 0);
  computeApi.uniform1i(cache->highlightOverflowLoc, payload.highlightOverflow != 0 ? 1 : 0);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cache->input);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cache->verts);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cache->colors);
  const GLuint groups = static_cast<GLuint>((pointCount + 63u) / 64u);
  computeApi.dispatchCompute(groups, 1, 1);
  computeApi.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
  computeApi.useProgram(0);
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

  cache->builtSerial = out->serial;
  cache->pointCount = static_cast<GLsizei>(pointCount);
  out->pointCount = pointCount;
  return true;
}

#if defined(__APPLE__)
bool buildInputCloudMeshOnMetal(const InputCloudPayload& payload,
                                const std::vector<float>& rawPoints,
                                OpenDRTViewerMetal::MeshCache* cache,
                                MeshData* out) {
  if (!out) return false;
  std::string error;
  if (!OpenDRTViewerMetal::buildInputCloudMesh(
          cache,
          rawPoints.data(),
          rawPoints.size(),
          payload.showOverflow != 0,
          payload.highlightOverflow != 0,
          out->serial,
          &error)) {
    if (!error.empty()) {
      logViewerEvent(std::string("Metal input-cloud mesh build failed: ") + error);
    }
    return false;
  }
  out->pointCount = static_cast<size_t>(cache ? cache->pointCount : 0);
  return cache && cache->available;
}
#endif

bool buildInputCloudMesh(const InputCloudPayload& payload,
                         const ViewerGpuCapabilities& gpuCaps,
                         ViewerRuntimeState* runtime,
                         InputCloudComputeCache* computeCache,
#if defined(__APPLE__)
                         OpenDRTViewerMetal::MeshCache* metalCache,
#endif
#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
                         InputCudaDrawCache* cudaCache,
#endif
                         MeshData* out) {
  if (!out) return false;
  std::istringstream is(payload.points);
  MeshData mesh{};
  mesh.resolution = (payload.resolution <= 25) ? 25 : (payload.resolution <= 41 ? 41 : 57);
  mesh.quality = payload.quality;
  mesh.paramHash = payload.paramHash;
  mesh.renderOk = true;
  mesh.maxDelta = 0.0f;
  mesh.serial = nextMeshSerial();
  mesh.transformBackendLabel = "plugin-pretransformed";
  std::vector<float> rawPoints;
  rawPoints.reserve(payload.points.size() / 4u);
  float sr = 0.0f, sg = 0.0f, sb = 0.0f, dr = 0.0f, dg = 0.0f, db = 0.0f;
  while (is >> sr >> sg >> sb >> dr >> dg >> db) {
    rawPoints.push_back(sr);
    rawPoints.push_back(sg);
    rawPoints.push_back(sb);
    rawPoints.push_back(dr);
    rawPoints.push_back(dg);
    rawPoints.push_back(db);
  }

#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
  if (gpuCaps.activeBackend == ViewerGpuBackend::CudaComputeMesh &&
      runtime && !runtime->inputGpuDemoted &&
      buildInputCloudMeshOnCuda(payload, rawPoints, cudaCache, &mesh)) {
    mesh.packBackendLabel = "cuda-compute-mesh";
    *out = std::move(mesh);
    return true;
  } else if (gpuCaps.activeBackend == ViewerGpuBackend::CudaComputeMesh && runtime && !runtime->inputGpuDemoted) {
    runtime->inputGpuDemoted = true;
    runtime->inputDemotionReason = "cuda-runtime-failure";
    logViewerEvent("Input CUDA path demoted to CPU after runtime failure.");
  }
#endif

#if defined(__APPLE__)
  if (gpuCaps.activeBackend == ViewerGpuBackend::MetalComputeMesh &&
      !(runtime && runtime->inputGpuDemoted) &&
      buildInputCloudMeshOnMetal(payload, rawPoints, metalCache, &mesh)) {
    mesh.packBackendLabel = "metal-compute-mesh";
    *out = std::move(mesh);
    return true;
  } else if (gpuCaps.activeBackend == ViewerGpuBackend::MetalComputeMesh && runtime && !runtime->inputGpuDemoted) {
    runtime->inputGpuDemoted = true;
    runtime->inputDemotionReason = "metal-runtime-failure";
    logViewerEvent("Input Metal path demoted to CPU after runtime failure.");
  }
#endif
  const bool tryInputCompute =
      gpuCaps.inputCloudComputeEnabled &&
      gpuCaps.activeBackend == ViewerGpuBackend::OpenGlComputeIdentity &&
      !(runtime && runtime->inputGpuDemoted);
  if (tryInputCompute && buildInputCloudMeshOnGpu(payload, rawPoints, computeCache, &mesh)) {
    mesh.packBackendLabel = "gl-compute-mesh";
    *out = std::move(mesh);
    return true;
  } else if (tryInputCompute && runtime && !runtime->inputGpuDemoted) {
    runtime->inputGpuDemoted = true;
    runtime->inputDemotionReason = "gl-runtime-failure";
    logViewerEvent("Input GL compute path demoted to CPU after runtime failure.");
  }

  for (size_t i = 0; i + 5u < rawPoints.size(); i += 6u) {
    sr = rawPoints[i + 0u];
    sg = rawPoints[i + 1u];
    sb = rawPoints[i + 2u];
    dr = rawPoints[i + 3u];
    dg = rawPoints[i + 4u];
    db = rawPoints[i + 5u];
    const bool overflowPoint = rgbOutOfBounds(dr, dg, db);
    const float plotR = payload.showOverflow != 0 ? dr : clamp01(dr);
    const float plotG = payload.showOverflow != 0 ? dg : clamp01(dg);
    const float plotB = payload.showOverflow != 0 ? db : clamp01(db);
    mesh.pointVerts.push_back(plotR * 2.0f - 1.0f);
    mesh.pointVerts.push_back(plotG * 2.0f - 1.0f);
    mesh.pointVerts.push_back(plotB * 2.0f - 1.0f);
    float cr = 0.0f, cg = 0.0f, cb = 0.0f;
    if (payload.showOverflow != 0 && payload.highlightOverflow != 0 && overflowPoint) {
      cr = 1.0f;
      cg = 0.0f;
      cb = 0.0f;
    } else {
      mapDisplayColor(clamp01(sr), clamp01(sg), clamp01(sb), &cr, &cg, &cb);
    }
    mesh.pointColors.push_back(cr);
    mesh.pointColors.push_back(cg);
    mesh.pointColors.push_back(cb);
  }
  if (mesh.pointVerts.empty()) return false;
  mesh.pointCount = mesh.pointVerts.size() / 3u;
  mesh.packBackendLabel = "cpu-array";
  *out = std::move(mesh);
  return true;
}

#if defined(__APPLE__)
OpenDRTViewerMetal::DrawSource resolveMetalDrawSource(const MeshData& mesh,
                                                      const std::string& currentSourceMode,
                                                      const OpenDRTViewerMetal::MeshCache& identityCache,
                                                      const OpenDRTViewerMetal::MeshCache& inputCache,
                                                      std::string* labelOut) {
  OpenDRTViewerMetal::DrawSource source{};
  auto tryResolve = [&](const OpenDRTViewerMetal::MeshCache& cache, const char* label) -> bool {
    if (cache.available && cache.builtSerial == mesh.serial &&
        OpenDRTViewerMetal::resolveDrawSource(&cache, &source)) {
      if (labelOut) *labelOut = label;
      return true;
    }
    return false;
  };

  if (currentSourceMode == "input") {
    if (tryResolve(inputCache, "input-metal")) return source;
    if (tryResolve(identityCache, "identity-metal")) return source;
  } else {
    if (tryResolve(identityCache, "identity-metal")) return source;
    if (tryResolve(inputCache, "input-metal")) return source;
  }

  source.cpuVerts = mesh.pointVerts.empty() ? nullptr : mesh.pointVerts.data();
  source.cpuColors = mesh.pointColors.empty() ? nullptr : mesh.pointColors.data();
  source.vertsHandle = nullptr;
  source.colorsHandle = nullptr;
  source.pointCount = (source.cpuVerts && source.cpuColors) ? static_cast<int>(mesh.pointVerts.size() / 3u) : 0;
  source.gpuBacked = false;
  if (labelOut) *labelOut = (source.pointCount > 0) ? "cpu-array" : "none";
  return source;
}
#endif

void updateProjection(int width, int height) {
#if defined(__APPLE__)
  (void)width;
  (void)height;
#else
  if (width < 1) width = 1;
  if (height < 1) height = 1;
  const double aspect = static_cast<double>(width) / static_cast<double>(height);
  const double fovy = 45.0;
  const double zNear = 0.1;
  const double zFar = 200.0;
  const double ymax = zNear * std::tan(fovy * 0.5 * 3.141592653589793 / 180.0);
  const double xmax = ymax * aspect;
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glFrustum(-xmax, xmax, -ymax, ymax, zNear, zFar);
  glMatrixMode(GL_MODELVIEW);
#endif
}

#if !defined(__APPLE__)
void drawReferenceFrame() {
  if (OpenDRTViewerOpenGLPresenter::drawReferenceFrame()) return;
  static const float kCubeEdges[] = {
      -1.f,-1.f,-1.f,  1.f,-1.f,-1.f,
       1.f,-1.f,-1.f,  1.f, 1.f,-1.f,
       1.f, 1.f,-1.f, -1.f, 1.f,-1.f,
      -1.f, 1.f,-1.f, -1.f,-1.f,-1.f,
      -1.f,-1.f, 1.f,  1.f,-1.f, 1.f,
       1.f,-1.f, 1.f,  1.f, 1.f, 1.f,
       1.f, 1.f, 1.f, -1.f, 1.f, 1.f,
      -1.f, 1.f, 1.f, -1.f,-1.f, 1.f,
      -1.f,-1.f,-1.f, -1.f,-1.f, 1.f,
       1.f,-1.f,-1.f,  1.f,-1.f, 1.f,
       1.f, 1.f,-1.f,  1.f, 1.f, 1.f,
      -1.f, 1.f,-1.f, -1.f, 1.f, 1.f
  };
  static const float kAxes[] = {
      -1.f,-1.f,-1.f, 1.35f,-1.f,-1.f,
      -1.f,-1.f,-1.f, -1.f,1.35f,-1.f,
      -1.f,-1.f,-1.f, -1.f,-1.f,1.35f
  };
  static const float kNeutralAxis[] = {
      -1.f,-1.f,-1.f, 1.f,1.f,1.f
  };
  glEnableClientState(GL_VERTEX_ARRAY);
  glLineWidth(1.15f);
  glColor4f(0.97f, 0.97f, 0.97f, 0.55f);
  glVertexPointer(3, GL_FLOAT, 0, kCubeEdges);
  glDrawArrays(GL_LINES, 0, 24);

  glLineWidth(1.5f);
  glVertexPointer(3, GL_FLOAT, 0, kAxes);
  glColor4f(1.0f, 0.32f, 0.32f, 0.9f);
  glDrawArrays(GL_LINES, 0, 2);
  glColor4f(0.35f, 1.0f, 0.35f, 0.9f);
  glDrawArrays(GL_LINES, 2, 2);
  glColor4f(0.35f, 0.60f, 1.0f, 0.9f);
  glDrawArrays(GL_LINES, 4, 2);
  glLineWidth(1.2f);
  glVertexPointer(3, GL_FLOAT, 0, kNeutralAxis);
  glColor4f(1.0f, 1.0f, 1.0f, 0.38f);
  glDrawArrays(GL_LINES, 0, 2);
  glDisableClientState(GL_VERTEX_ARRAY);
}
#endif

void updateTitle(GLFWwindow* window, const MeshData& mesh, const char* state, const ViewerGpuCapabilities& gpuCaps) {
  std::ostringstream os;
  os << "ME_OpenDRT Cube Viewer | " << state << " | " << mesh.quality << " " << mesh.resolution << "^3";
  os << " | viewer:" << gpuCaps.activeBackendLabel;
  os << " | presenter:" << gpuCaps.presenterBackendLabel;
  os << " | transform:" << mesh.transformBackendLabel;
  os << " | pack:" << mesh.packBackendLabel;
  os << " | dMax:" << mesh.maxDelta;
  if (mesh.parityMaxDelta > 0.0f) os << " | pMax:" << mesh.parityMaxDelta;
  if (!mesh.paramHash.empty()) os << " | hash " << mesh.paramHash;
  glfwSetWindowTitle(window, os.str().c_str());
}

#if defined(_WIN32)
void ipcServerLoop() {
  const std::string name = pipeName();
  while (gRun.load()) {
    HANDLE hPipe = CreateNamedPipeA(
        name.c_str(),
        PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
        1,
        0,
        1 << 16,
        100,
        nullptr);
    if (hPipe == INVALID_HANDLE_VALUE) {
      Sleep(250);
      continue;
    }
    const BOOL connected = ConnectNamedPipe(hPipe, nullptr) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
    if (!connected) {
      CloseHandle(hPipe);
      continue;
    }
    std::string pending;
    char buf[8192];
    DWORD readBytes = 0;
    while (gRun.load() && ReadFile(hPipe, buf, sizeof(buf), &readBytes, nullptr) && readBytes > 0) {
      pending.append(buf, buf + readBytes);
      size_t nl = std::string::npos;
      while ((nl = pending.find('\n')) != std::string::npos) {
        std::string line = pending.substr(0, nl);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string response = handleIncomingLine(line);
        if (!response.empty()) {
          std::string payload = response;
          payload.push_back('\n');
          DWORD written = 0;
          (void)WriteFile(hPipe, payload.data(), static_cast<DWORD>(payload.size()), &written, nullptr);
        }
        pending.erase(0, nl + 1);
      }
    }
    DisconnectNamedPipe(hPipe);
    CloseHandle(hPipe);
  }
}

void wakeIpcServer() {
  HANDLE h = CreateFileA(pipeName().c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h != INVALID_HANDLE_VALUE) {
    DWORD w = 0;
    const char nl = '\n';
    WriteFile(h, &nl, 1, &w, nullptr);
    CloseHandle(h);
  }
}
#else
void ipcServerLoop() {
  const std::string path = pipeName();
  ::unlink(path.c_str());
  int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    logViewerEvent(std::string("IPC socket() failed: errno=") + std::to_string(errno) + " (" + std::strerror(errno) + ")");
    return;
  }

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    logViewerEvent(
        std::string("IPC bind() failed for ") + path + ": errno=" + std::to_string(errno) + " (" + std::strerror(errno) + ")");
    ::close(fd);
    return;
  }
  if (::listen(fd, 4) != 0) {
    logViewerEvent(std::string("IPC listen() failed: errno=") + std::to_string(errno) + " (" + std::strerror(errno) + ")");
    ::close(fd);
    ::unlink(path.c_str());
    return;
  }

  while (gRun.load()) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);
    timeval tv{};
    tv.tv_sec = 0;
    tv.tv_usec = 200000;
    const int sel = ::select(fd + 1, &rfds, nullptr, nullptr, &tv);
    if (sel <= 0) continue;

    const int client = ::accept(fd, nullptr, nullptr);
    if (client < 0) continue;
    std::string pending;
    char buf[8192];
    while (gRun.load()) {
      const ssize_t n = ::recv(client, buf, sizeof(buf), 0);
      if (n <= 0) break;
      pending.append(buf, buf + n);
      size_t nl = std::string::npos;
      while ((nl = pending.find('\n')) != std::string::npos) {
        std::string line = pending.substr(0, nl);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string response = handleIncomingLine(line);
        if (!response.empty()) {
          std::string payload = response;
          payload.push_back('\n');
          (void)sendAllSocket(client, payload.data(), payload.size());
        }
        pending.erase(0, nl + 1);
      }
    }
    if (!pending.empty()) {
      logViewerEvent(std::string("Connection closed with unterminated payload bytes=") + std::to_string(pending.size()));
    }
    ::close(client);
  }

  ::close(fd);
  ::unlink(path.c_str());
}

void wakeIpcServer() {
  const std::string path = pipeName();
  int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return;
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
    const char nl = '\n';
    (void)sendAllSocket(fd, &nl, 1);
  }
  ::close(fd);
}
#endif

struct AppState {
  CameraState cam;
  bool leftDown = false;
  bool panMode = false;
  bool zoomMode = false;
  bool keepOnTop = false;
  bool appliedTopmost = false;
  bool diagTransitions = false;
  bool experimentalInputCompute = false;
  std::string currentSourceMode = "identity";
  std::string currentSenderId;
  std::string lastDrawSourceLabel;
  double lastX = 0.0;
  double lastY = 0.0;
  double lastClick = -10.0;
  float scrollAccum = 0.0f;
  ViewerGpuCapabilities gpuCaps;
  ViewerRuntimeState runtime;
};

void onFramebufferSize(GLFWwindow*, int w, int h) {
  updateProjection(w, h);
}

void onWindowClose(GLFWwindow*) {
  gRun.store(false);
}

void onScroll(GLFWwindow* window, double, double yoff) {
  auto* app = reinterpret_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (!app) return;
  const bool shift =
      (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
  const bool ctrl =
      (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
  const float scale = (shift || ctrl) ? 0.35f : 1.0f;
  app->scrollAccum += static_cast<float>(yoff) * scale;
}

void onMouseButton(GLFWwindow* window, int button, int action, int) {
  auto* app = reinterpret_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (!app) return;
  if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS) return;
  const double now = glfwGetTime();
  if ((now - app->lastClick) < 0.3) {
    const bool ctrl = (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                       glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
    if (ctrl) {
      resetVectorscopeCamera(&app->cam);
    } else {
      resetCamera(&app->cam);
    }
    app->leftDown = false;
    app->panMode = false;
    app->zoomMode = false;
  }
  app->lastClick = now;
}

void processMouseAndKeys(GLFWwindow* window, AppState* app) {
  if (!app) return;
  const int l = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
  const int m = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE);
  const int r = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
  const bool anyDown = (l == GLFW_PRESS || m == GLFW_PRESS || r == GLFW_PRESS);
  const bool shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  double cx = 0.0, cy = 0.0;
  glfwGetCursorPos(window, &cx, &cy);

  if (anyDown && !app->leftDown) {
    app->leftDown = true;
    app->zoomMode = (r == GLFW_PRESS);
    app->panMode = !app->zoomMode && ((m == GLFW_PRESS) || shift);
    app->lastX = cx;
    app->lastY = cy;
  } else if (!anyDown && app->leftDown) {
    app->leftDown = false;
    app->panMode = false;
    app->zoomMode = false;
  }

  if (app->leftDown) {
    const float dx = static_cast<float>(cx - app->lastX);
    const float dy = static_cast<float>(cy - app->lastY);
    app->lastX = cx;
    app->lastY = cy;
    if (app->zoomMode) {
      const float zoomFactor = std::exp(dy * 0.0105f);
      app->cam.distance *= zoomFactor;
      if (app->cam.distance < 0.45f) app->cam.distance = 0.45f;
      if (app->cam.distance > 30.0f) app->cam.distance = 30.0f;
    } else if (app->panMode) {
      const float panScale = 0.0022f * app->cam.distance;
      app->cam.panX += dx * panScale;
      app->cam.panY -= dy * panScale;
    } else {
      int w = 1, h = 1;
      glfwGetWindowSize(window, &w, &h);
      Vec3 v0 = mapArcball(app->lastX - dx, app->lastY - dy, w, h);
      Vec3 v1 = mapArcball(app->lastX, app->lastY, w, h);
      Vec3 axis = cross3(v0, v1);
      const float axisLen = length3(axis);
      float d = dot3(v0, v1);
      d = clampf(d, -1.0f, 1.0f);
      const float angle = std::acos(d);
      if (axisLen > 1e-6f && angle > 1e-6f) {
        const float arcballGain = 1.35f;
        const Quat qDelta = axisAngleQ(axis, angle * arcballGain);
        Quat qCur{app->cam.qx, app->cam.qy, app->cam.qz, app->cam.qw};
        qCur = normalizeQ(mulQ(qDelta, qCur));
        app->cam.qx = qCur.x;
        app->cam.qy = qCur.y;
        app->cam.qz = qCur.z;
        app->cam.qw = qCur.w;
      }
    }
  }

  if (app->scrollAccum != 0.0f) {
    const float delta = app->scrollAccum;
    app->scrollAccum = 0.0f;
    const float factor = std::exp(-delta / 10.0f);
    app->cam.distance *= factor;
    if (app->cam.distance < 0.45f) app->cam.distance = 0.45f;
    if (app->cam.distance > 30.0f) app->cam.distance = 30.0f;
  }

  if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
    resetCamera(&app->cam);
  }
}

int runApp() {
  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

#if defined(_WIN32)
  HANDLE singleInstanceMutex = CreateMutexA(nullptr, FALSE, "Global\\ME_OpenDRT_CubeViewer_Singleton");
  if (singleInstanceMutex != nullptr && GetLastError() == ERROR_ALREADY_EXISTS) {
    if (singleInstanceMutex != nullptr) CloseHandle(singleInstanceMutex);
    return 0;
  }
#endif

  if (!glfwInit()) {
    logViewerEvent("glfwInit() failed");
#if defined(_WIN32)
    if (singleInstanceMutex != nullptr) CloseHandle(singleInstanceMutex);
#endif
    return 1;
  }

#if defined(__APPLE__)
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#endif

  GLFWwindow* window = glfwCreateWindow(864, 560, "ME_OpenDRT Cube Viewer", nullptr, nullptr);
  if (!window) {
    logViewerEvent("glfwCreateWindow() failed");
    glfwTerminate();
#if defined(_WIN32)
    if (singleInstanceMutex != nullptr) CloseHandle(singleInstanceMutex);
#endif
    return 1;
  }
  logViewerEvent(std::string("Viewer startup ok ") + kViewerVersionString);

#if defined(_WIN32)
  applyWindowsWindowIcon(window);
#endif
#if !defined(__APPLE__)
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
#endif
  ViewerGpuCapabilities gpuCaps = detectViewerGpuCapabilities();
#if defined(__APPLE__)
  gpuCaps.presenterBackendLabel = "metal";
  const OpenDRTViewerMetal::PresenterInitResult metalPresenterInit = OpenDRTViewerMetal::initializePresenter(window);
  if (!metalPresenterInit.ready) {
    logViewerEvent(std::string("Metal presenter init failed: ") + metalPresenterInit.reason);
    glfwDestroyWindow(window);
    glfwTerminate();
#if defined(_WIN32)
    if (singleInstanceMutex != nullptr) CloseHandle(singleInstanceMutex);
#endif
    return 1;
  }
#endif
#if !defined(__APPLE__)
  const OpenDRTViewerOpenGLPresenter::InitResult glPresenterInit =
      OpenDRTViewerOpenGLPresenter::initialize(window);
  gpuCaps.glPresenterReady = glPresenterInit.ready;
  gpuCaps.presenterBackendLabel = glPresenterInit.ready ? "gl-shader" : "gl-legacy";
  if (!glPresenterInit.ready) {
    logViewerEvent(std::string("OpenGL presenter init fallback: ") + glPresenterInit.reason);
  }
#else
  gpuCaps.glPresenterReady = false;
#endif
  IdentityComputeCache identityComputeCache{};
  InputCloudComputeCache inputCloudComputeCache{};
#if defined(__APPLE__)
  OpenDRTViewerMetal::MeshCache identityMetalCache{};
  OpenDRTViewerMetal::MeshCache inputMetalCache{};
#endif
#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
  IdentityCudaDrawCache identityCudaCache{};
  InputCudaDrawCache inputCudaCache{};
#endif
  const bool requestedInputCompute = viewerInputComputeEnabled();
  bool inputComputeReady = requestedInputCompute;
#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
  if (gpuCaps.activeBackend == ViewerGpuBackend::CudaComputeMesh) {
    const OpenDRTViewerCuda::StartupValidationResult cudaValidation = OpenDRTViewerCuda::validateStartup();
    gpuCaps.cudaStartupReady = cudaValidation.ready;
    if (!cudaValidation.ready) {
      gpuCaps.activeBackend = gpuCaps.glComputeShaders ? ViewerGpuBackend::OpenGlComputeIdentity
                                                       : (gpuCaps.glBufferObjects ? ViewerGpuBackend::OpenGlBufferedDraw
                                                                                  : ViewerGpuBackend::CpuReference);
      gpuCaps.activeBackendLabel = gpuCaps.glComputeShaders ? "gl-compute-mesh"
                                                            : (gpuCaps.glBufferObjects ? "gl-buffer" : "cpu-ref");
      gpuCaps.roadmapLabel = std::string("cuda-startup-fallback:") + cudaValidation.reason;
      inputComputeReady = requestedInputCompute && gpuCaps.activeBackend == ViewerGpuBackend::OpenGlComputeIdentity;
    }
  }
#endif
#if defined(__APPLE__)
  if (gpuCaps.activeBackend == ViewerGpuBackend::MetalComputeMesh) {
    const OpenDRTViewerMetal::StartupValidationResult metalValidation = OpenDRTViewerMetal::validateStartup();
    if (!metalValidation.ready) {
      gpuCaps.activeBackend = gpuCaps.glBufferObjects ? ViewerGpuBackend::OpenGlBufferedDraw
                                                      : ViewerGpuBackend::CpuReference;
      gpuCaps.activeBackendLabel = gpuCaps.glBufferObjects ? "gl-buffer" : "cpu-ref";
      gpuCaps.roadmapLabel = std::string("metal-startup-fallback:") + metalValidation.reason;
      inputComputeReady = false;
    }
  }
#endif
  if (gpuCaps.activeBackend == ViewerGpuBackend::OpenGlComputeIdentity &&
      !ensureIdentityComputeProgram(&identityComputeCache)) {
    gpuCaps.activeBackend = gpuCaps.glBufferObjects ? ViewerGpuBackend::OpenGlBufferedDraw
                                                    : ViewerGpuBackend::CpuReference;
    gpuCaps.activeBackendLabel = gpuCaps.glBufferObjects ? "gl-buffer" : "cpu-ref";
    gpuCaps.roadmapLabel = "gl-compute-init-fallback";
  }
  if (inputComputeReady &&
      gpuCaps.activeBackend == ViewerGpuBackend::OpenGlComputeIdentity &&
      !ensureInputCloudComputeProgram(&inputCloudComputeCache)) {
    inputComputeReady = false;
    logViewerEvent("Input-cloud compute disabled at startup; using CPU input-cloud mesh path.");
  }
  if (inputComputeReady &&
      gpuCaps.activeBackend != ViewerGpuBackend::OpenGlComputeIdentity &&
      gpuCaps.activeBackend != ViewerGpuBackend::MetalComputeMesh &&
      gpuCaps.activeBackend != ViewerGpuBackend::CudaComputeMesh) {
    inputComputeReady = false;
  }
  gpuCaps.inputCloudComputeEnabled = inputComputeReady;
  if (gpuCaps.activeBackend == ViewerGpuBackend::CudaComputeMesh) {
    gpuCaps.activeBackendLabel = "cuda-compute-mesh";
    gpuCaps.roadmapLabel = "cuda-phase3-mesh";
  } else if (gpuCaps.activeBackend == ViewerGpuBackend::OpenGlComputeIdentity && inputComputeReady) {
    gpuCaps.activeBackendLabel = "gl-compute-mesh";
    gpuCaps.roadmapLabel = "gl-compute-phase2-mesh";
  } else if (gpuCaps.activeBackend == ViewerGpuBackend::MetalComputeMesh && inputComputeReady) {
    gpuCaps.activeBackendLabel = "metal-compute-mesh";
    gpuCaps.roadmapLabel = "metal-phase2-mesh";
  }
  {
    std::ostringstream os;
    os << "Viewer GPU capabilities: glVersion=" << gpuCaps.glVersion
       << " buffers=" << (gpuCaps.glBufferObjects ? "1" : "0")
       << " computeProbe=" << (gpuCaps.glComputeShaders ? "1" : "0")
       << " active=" << gpuCaps.activeBackendLabel
       << " roadmap=" << gpuCaps.roadmapLabel
       << " inputCompute=" << (inputComputeReady ? "1" : "0")
       << " requestedInputCompute=" << (requestedInputCompute ? "1" : "0")
       << " parityCheck=" << (gpuCaps.parityCheckEnabled ? "1" : "0");
#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
    os << " cudaViewer=" << (gpuCaps.cudaViewerAvailable ? "1" : "0")
       << " cudaInterop=" << (gpuCaps.cudaInteropReady ? "1" : "0")
       << " cudaReady=" << (gpuCaps.cudaStartupReady ? "1" : "0");
    if (!gpuCaps.cudaDeviceName.empty()) os << " cudaDevice=" << gpuCaps.cudaDeviceName;
    if (!gpuCaps.cudaReason.empty()) os << " cudaReason=" << gpuCaps.cudaReason;
#endif
#if defined(__APPLE__)
    os << " metalViewer=" << (gpuCaps.metalViewerAvailable ? "1" : "0")
       << " metalQueue=" << (gpuCaps.metalQueueReady ? "1" : "0");
    if (!gpuCaps.metalDeviceName.empty()) os << " metalDevice=" << gpuCaps.metalDeviceName;
#endif
    logViewerEvent(os.str());
  }
  if (GLFWmonitor* monitor = glfwGetPrimaryMonitor()) {
    if (const GLFWvidmode* mode = glfwGetVideoMode(monitor)) {
      const int winW = 864;
      const int winH = 560;
      int x = (mode->width - winW) / 2;
      int y = (mode->height - winH) / 2 - 40;
      if (x < 0) x = 0;
      if (y < 0) y = 0;
      glfwSetWindowPos(window, x, y);
    }
  }

  AppState app{};
  resetCamera(&app.cam);
  app.gpuCaps = gpuCaps;
  app.runtime.sessionBackendLabel = gpuCaps.activeBackendLabel;
  app.runtime.presenterBackendLabel = gpuCaps.presenterBackendLabel;
  app.diagTransitions = viewerDiagnosticsEnabled();
  app.experimentalInputCompute = inputComputeReady;
  if (app.diagTransitions) {
    std::ostringstream os;
    os << "Transition diagnostics enabled. log=" << viewerLogPath()
       << " inputCompute=" << (app.experimentalInputCompute ? "1" : "0")
       << " requestedInputCompute=" << (requestedInputCompute ? "1" : "0");
    logViewerDiagnostic(true, os.str());
  }
  PointBufferCache pointBufferCache{};
  glfwSetWindowUserPointer(window, &app);
  glfwSetFramebufferSizeCallback(window, onFramebufferSize);
  glfwSetWindowCloseCallback(window, onWindowClose);
  glfwSetScrollCallback(window, onScroll);
  glfwSetMouseButtonCallback(window, onMouseButton);

#if !defined(__APPLE__)
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_POINT_SMOOTH);
  glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
#endif

  int fbW = 0, fbH = 0;
  glfwGetFramebufferSize(window, &fbW, &fbH);
  updateProjection(fbW, fbH);

  std::thread ipcThread(ipcServerLoop);

  MeshData mesh{};
  mesh.resolution = 33;
  mesh.quality = "Medium";
  mesh.renderOk = true;
  mesh.maxDelta = 0.0f;
  uint64_t lastParamsSeq = 0;
  uint64_t lastCloudSeq = 0;
  InputCloudPayload deferredCloud{};
  bool hasDeferredCloud = false;

  while (gRun.load() && !glfwWindowShouldClose(window)) {
    glfwPollEvents();
    gWindowVisible.store(glfwGetWindowAttrib(window, GLFW_VISIBLE) == GLFW_TRUE ? 1 : 0, std::memory_order_relaxed);
    gWindowIconified.store(glfwGetWindowAttrib(window, GLFW_ICONIFIED) == GLFW_TRUE ? 1 : 0, std::memory_order_relaxed);
    gWindowFocused.store(glfwGetWindowAttrib(window, GLFW_FOCUSED) == GLFW_TRUE ? 1 : 0, std::memory_order_relaxed);
    processMouseAndKeys(window, &app);

    PendingMessage pendingParams{};
    PendingMessage pendingCloud{};
    bool haveParams = false;
    bool haveCloud = false;
    {
      std::lock_guard<std::mutex> lock(gMsgMutex);
      if (gHasPendingParamsMsg) {
        pendingParams = gPendingParamsMsg;
        gHasPendingParamsMsg = false;
        haveParams = true;
      }
      if (gHasPendingCloudMsg) {
        pendingCloud = gPendingCloudMsg;
        gHasPendingCloudMsg = false;
        haveCloud = true;
      }
    }
    if (haveParams) {
      ResolvedPayload rp{};
      try {
        if (parseParamsMessage(pendingParams.line, &rp)) {
          {
            std::ostringstream os;
            os << "Params received: sender=" << rp.senderId
               << " seq=" << rp.seq
               << " sourceMode=" << rp.sourceMode
               << " quality=" << rp.quality
               << " res=" << rp.resolution
               << " hash=" << rp.paramHash;
            logViewerEvent(os.str());
          }
          if (!rp.senderId.empty() && rp.senderId != app.currentSenderId) {
            app.currentSenderId = rp.senderId;
            lastParamsSeq = 0;
            lastCloudSeq = 0;
            hasDeferredCloud = false;
            logViewerEvent("Active sender changed; reset params/cloud sequence tracking.");
          }
          if (rp.seq < lastParamsSeq) {
            logViewerEvent("Ignored stale params sequence.");
          } else {
            lastParamsSeq = rp.seq;
            const std::string prevSourceMode = app.currentSourceMode;
            app.keepOnTop = (rp.alwaysOnTop != 0);
            app.currentSourceMode = rp.sourceMode;
            if (app.diagTransitions && prevSourceMode != app.currentSourceMode) {
              std::ostringstream os;
              os << "Source mode transition: " << prevSourceMode << " -> " << app.currentSourceMode
                 << " paramsSeq=" << rp.seq
                 << " meshSerial=" << mesh.serial
                 << " meshPoints=" << (mesh.pointCount > 0 ? mesh.pointCount : (mesh.pointVerts.size() / 3u))
                 << " deferredCloud=" << (hasDeferredCloud ? "1" : "0");
              logViewerDiagnostic(true, os.str());
            }
            if (app.currentSourceMode != "input") {
              MeshData nextMesh{};
              buildCubeData(rp, app.gpuCaps, &app.runtime, &identityComputeCache,
#if defined(__APPLE__)
                            &identityMetalCache,
#endif
#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
                            &identityCudaCache,
#endif
                            &nextMesh);
              mesh = std::move(nextMesh);
            } else {
              // Keep last displayed mesh until an input_cloud payload arrives.
              // This avoids flicker/blank frames on mode switch and live param deltas.
              mesh.quality = rp.quality;
              mesh.resolution = rp.resolution;
              mesh.paramHash = rp.paramHash;
              mesh.renderOk = true;
              mesh.maxDelta = 0.0f;
              if (prevSourceMode != "input") {
                // Reset pan only when switching modes so stale framing does not hide the cloud.
                app.cam.panX = 0.0f;
                app.cam.panY = 0.0f;
                logViewerEvent("Params switched source mode to input.");
              }
              if (hasDeferredCloud && deferredCloud.seq >= lastCloudSeq &&
                  senderMatchesCurrent(app.currentSenderId, deferredCloud.senderId)) {
                MeshData nextMesh{};
                if (buildInputCloudMesh(deferredCloud, app.gpuCaps, &app.runtime, &inputCloudComputeCache,
#if defined(__APPLE__)
                                        &inputMetalCache,
#endif
#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
                                        &inputCudaCache,
#endif
                                        &nextMesh)) {
                  mesh = std::move(nextMesh);
                  lastCloudSeq = deferredCloud.seq;
                  hasDeferredCloud = false;
                  logViewerEvent("Applied deferred input cloud after params switched to input mode.");
                  if (app.diagTransitions) {
                    std::ostringstream os;
                    os << "Deferred cloud became active: seq=" << deferredCloud.seq
                       << " meshSerial=" << mesh.serial
                       << " meshPoints=" << (mesh.pointCount > 0 ? mesh.pointCount : (mesh.pointVerts.size() / 3u));
                    logViewerDiagnostic(true, os.str());
                  }
                }
              }
            }
          }
        }
      } catch (const std::exception& e) {
        logViewerEvent(std::string("Exception while processing params message: ") + e.what());
      } catch (...) {
        logViewerEvent("Unknown exception while processing params message.");
      }
    }
    if (haveCloud) {
      InputCloudPayload cp{};
      try {
        if (parseInputCloudMessage(pendingCloud.line, &cp)) {
          {
            std::ostringstream os;
            os << "Input cloud received: sender=" << cp.senderId
               << " seq=" << cp.seq
               << " quality=" << cp.quality
               << " res=" << cp.resolution
               << " pointBytes=" << cp.points.size()
               << " sourceMode=" << app.currentSourceMode;
            logViewerEvent(os.str());
          }
          if (!senderMatchesCurrent(app.currentSenderId, cp.senderId)) {
            // Ignore clouds from a different OFX instance than the active sender.
            logViewerEvent("Ignored input cloud from non-active sender.");
          } else if (cp.seq < lastCloudSeq) {
            logViewerEvent("Ignored stale input cloud sequence.");
          } else if (app.currentSourceMode != "input") {
            deferredCloud = cp;
            hasDeferredCloud = true;
            logViewerEvent("Deferred input cloud until params confirm input mode.");
          } else {
            MeshData nextMesh{};
            if (buildInputCloudMesh(cp, app.gpuCaps, &app.runtime, &inputCloudComputeCache,
#if defined(__APPLE__)
                                    &inputMetalCache,
#endif
#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
                                    &inputCudaCache,
#endif
                                    &nextMesh)) {
              mesh = std::move(nextMesh);
              lastCloudSeq = cp.seq;
              hasDeferredCloud = false;
              if (app.diagTransitions) {
                std::ostringstream os;
                os << "Input cloud applied live: seq=" << cp.seq
                   << " meshSerial=" << mesh.serial
                   << " meshPoints=" << (mesh.pointCount > 0 ? mesh.pointCount : (mesh.pointVerts.size() / 3u))
                   << " paramHash=" << mesh.paramHash;
                logViewerDiagnostic(true, os.str());
              }
              {
                std::ostringstream os;
                os << "Applied input cloud: seq=" << cp.seq
                   << " points=" << (mesh.pointCount > 0 ? mesh.pointCount : (mesh.pointVerts.size() / 3u))
                   << " quality=" << mesh.quality
                   << " hash=" << mesh.paramHash;
                logViewerEvent(os.str());
              }
            } else {
              logViewerEvent("Input cloud payload parsed but mesh build failed.");
            }
          }
        }
      } catch (const std::exception& e) {
        logViewerEvent(std::string("Exception while processing input cloud message: ") + e.what());
      } catch (...) {
        logViewerEvent("Unknown exception while processing input cloud message.");
      }
    }

    if (app.keepOnTop != app.appliedTopmost) {
      glfwSetWindowAttrib(window, GLFW_FLOATING, app.keepOnTop ? GLFW_TRUE : GLFW_FALSE);
      app.appliedTopmost = app.keepOnTop;
    }

    if (gBringToFront.exchange(false)) {
      glfwRestoreWindow(window);
      glfwFocusWindow(window);
    }

    updateTitle(window, mesh, gConnected.load() ? "Connected" : "Disconnected", app.gpuCaps);

#if defined(__APPLE__)
    std::string drawSourceLabel = "none";
    const OpenDRTViewerMetal::DrawSource metalDrawSource =
        resolveMetalDrawSource(mesh, app.currentSourceMode, identityMetalCache, inputMetalCache, &drawSourceLabel);
    const size_t pointCount = static_cast<size_t>(std::max(metalDrawSource.pointCount, 0));
    const bool haveDrawablePointSource =
        metalDrawSource.pointCount > 0 &&
        ((metalDrawSource.gpuBacked && metalDrawSource.vertsHandle != nullptr && metalDrawSource.colorsHandle != nullptr) ||
         (!metalDrawSource.gpuBacked && metalDrawSource.cpuVerts != nullptr && metalDrawSource.cpuColors != nullptr));
    if (app.diagTransitions && drawSourceLabel != app.lastDrawSourceLabel) {
      std::ostringstream os;
      os << "Draw source changed: mode=" << app.currentSourceMode
         << " meshSerial=" << mesh.serial
         << " meshPoints=" << pointCount
         << " source=" << drawSourceLabel;
      logViewerDiagnostic(true, os.str());
      app.lastDrawSourceLabel = drawSourceLabel;
    }
    if (app.diagTransitions && pointCount > 0 && !haveDrawablePointSource) {
      std::ostringstream os;
      os << "Drawable source missing with non-zero point count: mode=" << app.currentSourceMode
         << " meshSerial=" << mesh.serial
         << " meshPoints=" << pointCount;
      logViewerDiagnostic(true, os.str());
    }

    float clampedDist = app.cam.distance;
    if (clampedDist < 0.6f) clampedDist = 0.6f;
    if (clampedDist > 30.0f) clampedDist = 30.0f;
    float basePointSize = 2.7f;
    if (mesh.resolution <= 25) basePointSize = 3.3f;
    else if (mesh.resolution <= 41) basePointSize = 2.9f;
    else basePointSize = 2.5f;
    const float distanceFactor = std::pow(clampedDist / 4.2f, -0.68f);
    float densityFactor = 1.0f;
    if (pointCount < 8000u) densityFactor = 1.10f;
    else if (pointCount > 50000u) densityFactor = 0.92f;
    const float kPointSizeUserScale = 1.28f;
    float pointSize = basePointSize * distanceFactor * densityFactor * kPointSizeUserScale;
    if (pointSize < 1.2f) pointSize = 1.2f;
    if (pointSize > 5.0f) pointSize = 5.0f;

    int fbWidth = 1;
    int fbHeight = 1;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    float mvp[16];
    buildSceneMvp(fbWidth, fbHeight, app.cam, mvp);
    std::string renderError;
    if (!OpenDRTViewerMetal::renderScene(
            window,
            metalDrawSource,
            mvp,
            pointSize,
            pointSize * 0.55f,
            0.08f,
            0.08f,
            0.09f,
            1.0f,
            &renderError)) {
      logViewerEvent(std::string("Metal render failed: ") + renderError);
      gRun.store(false);
    }
#else
    glClearColor(0.08f, 0.08f, 0.09f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(app.cam.panX, app.cam.panY, -app.cam.distance);
    float rotM[16];
    quatToMatrix(Quat{app.cam.qx, app.cam.qy, app.cam.qz, app.cam.qw}, rotM);
    glMultMatrixf(rotM);

    drawReferenceFrame();

    const ViewerGlBufferApi& glBufferApi = viewerGlBufferApi();
#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
    const bool useCudaIdentityBuffers =
        identityCudaCache.cache.available &&
        identityCudaCache.cache.builtSerial == mesh.serial &&
        identityCudaCache.cache.verts != 0 &&
        identityCudaCache.cache.colors != 0 &&
        identityCudaCache.cache.pointCount > 0;
    const bool useCudaInputBuffers =
        inputCudaCache.cache.available &&
        inputCudaCache.cache.builtSerial == mesh.serial &&
        inputCudaCache.cache.verts != 0 &&
        inputCudaCache.cache.colors != 0 &&
        inputCudaCache.cache.pointCount > 0;
#else
    const bool useCudaIdentityBuffers = false;
    const bool useCudaInputBuffers = false;
#endif
    const bool useComputeIdentityBuffers =
        identityComputeCache.available &&
        identityComputeCache.builtSerial == mesh.serial &&
        identityComputeCache.verts != 0 &&
        identityComputeCache.colors != 0 &&
        identityComputeCache.pointCount > 0;
    const bool useComputeInputBuffers =
        app.experimentalInputCompute &&
        inputCloudComputeCache.available &&
        inputCloudComputeCache.builtSerial == mesh.serial &&
        inputCloudComputeCache.verts != 0 &&
        inputCloudComputeCache.colors != 0 &&
        inputCloudComputeCache.pointCount > 0;
    const bool usePointBuffers = !useCudaIdentityBuffers &&
                                 !useComputeIdentityBuffers &&
                                 !useCudaInputBuffers &&
                                 ensurePointBufferCacheUploaded(mesh, &pointBufferCache);
    float clampedDist = app.cam.distance;
    if (clampedDist < 0.6f) clampedDist = 0.6f;
    if (clampedDist > 30.0f) clampedDist = 30.0f;
  float basePointSize = 2.7f;
  if (mesh.resolution <= 25) basePointSize = 3.3f;
  else if (mesh.resolution <= 41) basePointSize = 2.9f;
  else basePointSize = 2.5f;
    const float distanceFactor = std::pow(clampedDist / 4.2f, -0.68f);
    const ResolvedDrawSource drawSource = resolveDrawSource(
        mesh,
        useCudaIdentityBuffers,
#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
        identityCudaCache.cache.verts,
        identityCudaCache.cache.colors,
        identityCudaCache.cache.pointCount,
#else
        0, 0, 0,
#endif
        useComputeIdentityBuffers,
        identityComputeCache.verts,
        identityComputeCache.colors,
        identityComputeCache.pointCount,
        useCudaInputBuffers,
#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
        inputCudaCache.cache.verts,
        inputCudaCache.cache.colors,
        inputCudaCache.cache.pointCount,
#else
        0, 0, 0,
#endif
        useComputeInputBuffers,
        inputCloudComputeCache.verts,
        inputCloudComputeCache.colors,
        inputCloudComputeCache.pointCount,
        usePointBuffers,
        pointBufferCache.verts,
        pointBufferCache.colors);
    const size_t pointCount = static_cast<size_t>(drawSource.pointCount);
    const bool haveDrawablePointSource =
        drawSource.pointCount > 0 &&
        (drawSource.bufferBacked || (drawSource.cpuVerts != nullptr && drawSource.cpuColors != nullptr));
    const std::string drawSourceLabel = drawSource.label;
    if (app.diagTransitions && drawSourceLabel != app.lastDrawSourceLabel) {
      std::ostringstream os;
      os << "Draw source changed: mode=" << app.currentSourceMode
         << " meshSerial=" << mesh.serial
         << " meshPoints=" << pointCount
         << " source=" << drawSourceLabel;
      logViewerDiagnostic(true, os.str());
      app.lastDrawSourceLabel = drawSourceLabel;
    }
    if (app.diagTransitions && pointCount > 0 && !haveDrawablePointSource) {
      std::ostringstream os;
      os << "Drawable source missing with non-zero point count: mode=" << app.currentSourceMode
         << " meshSerial=" << mesh.serial
         << " meshPoints=" << pointCount;
      logViewerDiagnostic(true, os.str());
    }
    float densityFactor = 1.0f;
    if (pointCount < 8000u) densityFactor = 1.10f;
    else if (pointCount > 50000u) densityFactor = 0.92f;
    // QUICK_TWEAK_POINT_SIZE: adjust this multiplier to make cube sample points smaller/larger.
    const float kPointSizeUserScale = 1.28f;
    float pointSize = basePointSize * distanceFactor * densityFactor * kPointSizeUserScale;
    if (pointSize < 1.2f) pointSize = 1.2f;
    if (pointSize > 5.0f) pointSize = 5.0f;
    const bool useGlPresenter = app.gpuCaps.glPresenterReady && drawSource.bufferBacked;
    if (useGlPresenter) {
      if (pointCount > 0 && haveDrawablePointSource) {
        (void)OpenDRTViewerOpenGLPresenter::drawPointCloud(
            drawSource.verts,
            drawSource.colors,
            static_cast<int>(pointCount),
            pointSize,
            1.0f);
      }
      if (pointCount > 0 && haveDrawablePointSource) {
        glDisable(GL_DEPTH_TEST);
        (void)OpenDRTViewerOpenGLPresenter::drawPointCloudSolid(
            drawSource.verts,
            static_cast<int>(pointCount),
            pointSize * 0.55f,
            0.95f,
            0.96f,
            1.0f,
            0.05f);
        glEnable(GL_DEPTH_TEST);
      }
    } else {
      glEnableClientState(GL_VERTEX_ARRAY);
      glEnableClientState(GL_COLOR_ARRAY);
      glPointSize(pointSize);
      if (drawSource.bufferBacked) {
        glBufferApi.bindBuffer(GL_ARRAY_BUFFER, drawSource.verts);
        glVertexPointer(3, GL_FLOAT, 0, nullptr);
        glBufferApi.bindBuffer(GL_ARRAY_BUFFER, drawSource.colors);
        glColorPointer(3, GL_FLOAT, 0, nullptr);
      } else {
        glVertexPointer(3, GL_FLOAT, 0, drawSource.cpuVerts);
        glColorPointer(3, GL_FLOAT, 0, drawSource.cpuColors);
      }
      if (pointCount > 0 && haveDrawablePointSource) {
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(pointCount));
      }
      if (pointCount > 0 && haveDrawablePointSource) {
        glDisable(GL_DEPTH_TEST);
        glDisableClientState(GL_COLOR_ARRAY);
        glColor4f(0.95f, 0.96f, 1.0f, 0.05f);
        glPointSize(pointSize * 0.55f);
        if (drawSource.bufferBacked) {
          glBufferApi.bindBuffer(GL_ARRAY_BUFFER, drawSource.verts);
          glVertexPointer(3, GL_FLOAT, 0, nullptr);
        } else {
          glVertexPointer(3, GL_FLOAT, 0, drawSource.cpuVerts);
        }
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(pointCount));
        glEnableClientState(GL_COLOR_ARRAY);
        glEnable(GL_DEPTH_TEST);
      }
      if (drawSource.bufferBacked) {
        glBufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
      }
      glDisableClientState(GL_COLOR_ARRAY);
      glDisableClientState(GL_VERTEX_ARRAY);
    }

    glfwSwapBuffers(window);
#endif
  }

  gRun.store(false);
  wakeIpcServer();
  if (ipcThread.joinable()) ipcThread.join();

#if !defined(__APPLE__)
  OpenDRTViewerOpenGLPresenter::shutdown();
#endif
#if defined(__APPLE__)
  OpenDRTViewerMetal::releaseCache(&identityMetalCache);
  OpenDRTViewerMetal::releaseCache(&inputMetalCache);
  OpenDRTViewerMetal::shutdownPresenter();
#endif
  releasePointBufferCache(&pointBufferCache);
  releaseIdentityComputeCache(&identityComputeCache);
  releaseInputCloudComputeCache(&inputCloudComputeCache);
#if defined(OFX_SUPPORTS_CUDARENDER) && !defined(__APPLE__)
  releaseIdentityCudaDrawCache(&identityCudaCache);
  releaseInputCudaDrawCache(&inputCudaCache);
#endif
  glfwDestroyWindow(window);
  glfwTerminate();

#if defined(_WIN32)
  if (singleInstanceMutex != nullptr) CloseHandle(singleInstanceMutex);
#endif
  return 0;
}

}  // namespace

#if defined(_WIN32)
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
  return runApp();
}
#else
int main() {
  return runApp();
}
#endif

