#include "OpenDRTViewerOpenGLPresenter.h"

#include <array>
#include <string>

#include <GLFW/glfw3.h>

#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#endif
#ifndef GL_STATIC_DRAW
#define GL_STATIC_DRAW 0x88E4
#endif
#ifndef GL_VERTEX_SHADER
#define GL_VERTEX_SHADER 0x8B31
#endif
#ifndef GL_FRAGMENT_SHADER
#define GL_FRAGMENT_SHADER 0x8B30
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

namespace OpenDRTViewerOpenGLPresenter {
namespace {

using GLCreateShaderProc = GLuint(APIENTRY *)(GLenum);
using GLShaderSourceProc = void(APIENTRY *)(GLuint, GLsizei, const char* const*, const GLint*);
using GLCompileShaderProc = void(APIENTRY *)(GLuint);
using GLGetShaderivProc = void(APIENTRY *)(GLuint, GLenum, GLint*);
using GLGetShaderInfoLogProc = void(APIENTRY *)(GLuint, GLsizei, GLsizei*, char*);
using GLCreateProgramProc = GLuint(APIENTRY *)(void);
using GLAttachShaderProc = void(APIENTRY *)(GLuint, GLuint);
using GLLinkProgramProc = void(APIENTRY *)(GLuint);
using GLGetProgramivProc = void(APIENTRY *)(GLuint, GLenum, GLint*);
using GLGetProgramInfoLogProc = void(APIENTRY *)(GLuint, GLsizei, GLsizei*, char*);
using GLDeleteShaderProc = void(APIENTRY *)(GLuint);
using GLDeleteProgramProc = void(APIENTRY *)(GLuint);
using GLUseProgramProc = void(APIENTRY *)(GLuint);
using GLGetUniformLocationProc = GLint(APIENTRY *)(GLuint, const char*);
using GLUniform1fProc = void(APIENTRY *)(GLint, GLfloat);
using GLUniform4fProc = void(APIENTRY *)(GLint, GLfloat, GLfloat, GLfloat, GLfloat);
using GLGetAttribLocationProc = GLint(APIENTRY *)(GLuint, const char*);
using GLEnableVertexAttribArrayProc = void(APIENTRY *)(GLuint);
using GLDisableVertexAttribArrayProc = void(APIENTRY *)(GLuint);
using GLVertexAttribPointerProc = void(APIENTRY *)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*);
using GLGenBuffersProc = void(APIENTRY *)(GLsizei, GLuint*);
using GLBindBufferProc = void(APIENTRY *)(GLenum, GLuint);
using GLBufferDataProc = void(APIENTRY *)(GLenum, std::ptrdiff_t, const void*, GLenum);
using GLDeleteBuffersProc = void(APIENTRY *)(GLsizei, const GLuint*);

struct GlApi {
  bool loaded = false;
  bool available = false;
  GLCreateShaderProc createShader = nullptr;
  GLShaderSourceProc shaderSource = nullptr;
  GLCompileShaderProc compileShader = nullptr;
  GLGetShaderivProc getShaderiv = nullptr;
  GLGetShaderInfoLogProc getShaderInfoLog = nullptr;
  GLCreateProgramProc createProgram = nullptr;
  GLAttachShaderProc attachShader = nullptr;
  GLLinkProgramProc linkProgram = nullptr;
  GLGetProgramivProc getProgramiv = nullptr;
  GLGetProgramInfoLogProc getProgramInfoLog = nullptr;
  GLDeleteShaderProc deleteShader = nullptr;
  GLDeleteProgramProc deleteProgram = nullptr;
  GLUseProgramProc useProgram = nullptr;
  GLGetUniformLocationProc getUniformLocation = nullptr;
  GLUniform1fProc uniform1f = nullptr;
  GLUniform4fProc uniform4f = nullptr;
  GLGetAttribLocationProc getAttribLocation = nullptr;
  GLEnableVertexAttribArrayProc enableVertexAttribArray = nullptr;
  GLDisableVertexAttribArrayProc disableVertexAttribArray = nullptr;
  GLVertexAttribPointerProc vertexAttribPointer = nullptr;
  GLGenBuffersProc genBuffers = nullptr;
  GLBindBufferProc bindBuffer = nullptr;
  GLBufferDataProc bufferData = nullptr;
  GLDeleteBuffersProc deleteBuffers = nullptr;
};

GlApi& api() {
  static GlApi value;
  if (!value.loaded) {
    value.loaded = true;
    value.createShader = reinterpret_cast<GLCreateShaderProc>(glfwGetProcAddress("glCreateShader"));
    value.shaderSource = reinterpret_cast<GLShaderSourceProc>(glfwGetProcAddress("glShaderSource"));
    value.compileShader = reinterpret_cast<GLCompileShaderProc>(glfwGetProcAddress("glCompileShader"));
    value.getShaderiv = reinterpret_cast<GLGetShaderivProc>(glfwGetProcAddress("glGetShaderiv"));
    value.getShaderInfoLog = reinterpret_cast<GLGetShaderInfoLogProc>(glfwGetProcAddress("glGetShaderInfoLog"));
    value.createProgram = reinterpret_cast<GLCreateProgramProc>(glfwGetProcAddress("glCreateProgram"));
    value.attachShader = reinterpret_cast<GLAttachShaderProc>(glfwGetProcAddress("glAttachShader"));
    value.linkProgram = reinterpret_cast<GLLinkProgramProc>(glfwGetProcAddress("glLinkProgram"));
    value.getProgramiv = reinterpret_cast<GLGetProgramivProc>(glfwGetProcAddress("glGetProgramiv"));
    value.getProgramInfoLog = reinterpret_cast<GLGetProgramInfoLogProc>(glfwGetProcAddress("glGetProgramInfoLog"));
    value.deleteShader = reinterpret_cast<GLDeleteShaderProc>(glfwGetProcAddress("glDeleteShader"));
    value.deleteProgram = reinterpret_cast<GLDeleteProgramProc>(glfwGetProcAddress("glDeleteProgram"));
    value.useProgram = reinterpret_cast<GLUseProgramProc>(glfwGetProcAddress("glUseProgram"));
    value.getUniformLocation = reinterpret_cast<GLGetUniformLocationProc>(glfwGetProcAddress("glGetUniformLocation"));
    value.uniform1f = reinterpret_cast<GLUniform1fProc>(glfwGetProcAddress("glUniform1f"));
    value.uniform4f = reinterpret_cast<GLUniform4fProc>(glfwGetProcAddress("glUniform4f"));
    value.getAttribLocation = reinterpret_cast<GLGetAttribLocationProc>(glfwGetProcAddress("glGetAttribLocation"));
    value.enableVertexAttribArray = reinterpret_cast<GLEnableVertexAttribArrayProc>(glfwGetProcAddress("glEnableVertexAttribArray"));
    value.disableVertexAttribArray = reinterpret_cast<GLDisableVertexAttribArrayProc>(glfwGetProcAddress("glDisableVertexAttribArray"));
    value.vertexAttribPointer = reinterpret_cast<GLVertexAttribPointerProc>(glfwGetProcAddress("glVertexAttribPointer"));
    value.genBuffers = reinterpret_cast<GLGenBuffersProc>(glfwGetProcAddress("glGenBuffers"));
    value.bindBuffer = reinterpret_cast<GLBindBufferProc>(glfwGetProcAddress("glBindBuffer"));
    value.bufferData = reinterpret_cast<GLBufferDataProc>(glfwGetProcAddress("glBufferData"));
    value.deleteBuffers = reinterpret_cast<GLDeleteBuffersProc>(glfwGetProcAddress("glDeleteBuffers"));
    value.available = value.createShader && value.shaderSource && value.compileShader &&
                      value.getShaderiv && value.getShaderInfoLog && value.createProgram &&
                      value.attachShader && value.linkProgram && value.getProgramiv &&
                      value.getProgramInfoLog && value.deleteShader && value.deleteProgram &&
                      value.useProgram && value.getUniformLocation && value.uniform1f &&
                      value.uniform4f && value.getAttribLocation &&
                      value.enableVertexAttribArray && value.disableVertexAttribArray &&
                      value.vertexAttribPointer && value.genBuffers && value.bindBuffer &&
                      value.bufferData && value.deleteBuffers;
  }
  return value;
}

std::string readShaderLog(GLuint handle, bool program) {
  GlApi& gl = api();
  GLint length = 0;
  if (program) gl.getProgramiv(handle, GL_INFO_LOG_LENGTH, &length);
  else gl.getShaderiv(handle, GL_INFO_LOG_LENGTH, &length);
  if (length <= 1) return std::string();
  std::string log(static_cast<size_t>(length), '\0');
  GLsizei written = 0;
  if (program) gl.getProgramInfoLog(handle, length, &written, log.data());
  else gl.getShaderInfoLog(handle, length, &written, log.data());
  if (written > 0 && static_cast<size_t>(written) < log.size()) log.resize(static_cast<size_t>(written));
  return log;
}

GLuint compileShader(GLenum type, const char* source, std::string* error) {
  GlApi& gl = api();
  const GLuint shader = gl.createShader(type);
  if (shader == 0) {
    if (error) *error = "glCreateShader failed";
    return 0;
  }
  gl.shaderSource(shader, 1, &source, nullptr);
  gl.compileShader(shader);
  GLint compiled = 0;
  gl.getShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    if (error) *error = readShaderLog(shader, false);
    gl.deleteShader(shader);
    return 0;
  }
  return shader;
}

GLuint linkProgram(const char* vsSource, const char* fsSource, std::string* error) {
  GlApi& gl = api();
  GLuint vs = compileShader(GL_VERTEX_SHADER, vsSource, error);
  if (vs == 0) return 0;
  GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSource, error);
  if (fs == 0) {
    gl.deleteShader(vs);
    return 0;
  }
  const GLuint program = gl.createProgram();
  if (program == 0) {
    if (error) *error = "glCreateProgram failed";
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return 0;
  }
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  GLint linked = 0;
  gl.getProgramiv(program, GL_LINK_STATUS, &linked);
  if (!linked) {
    if (error) *error = readShaderLog(program, true);
    gl.deleteProgram(program);
    return 0;
  }
  return program;
}

struct PresenterState {
  bool initialized = false;
  GLuint colorProgram = 0;
  GLuint solidProgram = 0;
  GLint colorProgramPointSizeLoc = -1;
  GLint colorProgramAlphaLoc = -1;
  GLint solidProgramPointSizeLoc = -1;
  GLint solidProgramColorLoc = -1;
  GLint colorProgramPosLoc = -1;
  GLint colorProgramColorLoc = -1;
  GLint solidProgramPosLoc = -1;
  GLuint guideVerts = 0;
  GLuint guideColors = 0;
};

PresenterState& state() {
  static PresenterState value;
  return value;
}

const char* kColorVertexShader = R"GLSL(
#version 120
attribute vec3 aPosition;
attribute vec3 aColor;
varying vec4 vColor;
uniform float uPointSize;
uniform float uAlpha;
void main() {
  gl_Position = gl_ModelViewProjectionMatrix * vec4(aPosition, 1.0);
  gl_PointSize = uPointSize;
  vColor = vec4(aColor, uAlpha);
}
)GLSL";

const char* kColorFragmentShader = R"GLSL(
#version 120
varying vec4 vColor;
void main() {
  gl_FragColor = vColor;
}
)GLSL";

const char* kSolidVertexShader = R"GLSL(
#version 120
attribute vec3 aPosition;
uniform float uPointSize;
void main() {
  gl_Position = gl_ModelViewProjectionMatrix * vec4(aPosition, 1.0);
  gl_PointSize = uPointSize;
}
)GLSL";

const char* kSolidFragmentShader = R"GLSL(
#version 120
uniform vec4 uColor;
void main() {
  gl_FragColor = uColor;
}
)GLSL";

bool ensureGuideBuffers(std::string* error) {
  PresenterState& s = state();
  if (s.guideVerts != 0 && s.guideColors != 0) return true;

  static const float kGuideVerts[] = {
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
      -1.f, 1.f,-1.f, -1.f, 1.f, 1.f,
      -1.f,-1.f,-1.f, 1.35f,-1.f,-1.f,
      -1.f,-1.f,-1.f, -1.f,1.35f,-1.f,
      -1.f,-1.f,-1.f, -1.f,-1.f,1.35f,
      -1.f,-1.f,-1.f, 1.f,1.f,1.f
  };
  std::array<float, (24 + 6 + 2) * 2 * 3> guideColors{};
  size_t cursor = 0;
  auto writeColor = [&](float r, float g, float b) {
    guideColors[cursor++] = r;
    guideColors[cursor++] = g;
    guideColors[cursor++] = b;
  };
  for (int i = 0; i < 24; ++i) writeColor(0.97f, 0.97f, 0.97f);
  writeColor(1.0f, 0.32f, 0.32f); writeColor(1.0f, 0.32f, 0.32f);
  writeColor(0.35f, 1.0f, 0.35f); writeColor(0.35f, 1.0f, 0.35f);
  writeColor(0.35f, 0.60f, 1.0f); writeColor(0.35f, 0.60f, 1.0f);
  writeColor(1.0f, 1.0f, 1.0f); writeColor(1.0f, 1.0f, 1.0f);

  GlApi& gl = api();
  gl.genBuffers(1, &s.guideVerts);
  gl.genBuffers(1, &s.guideColors);
  if (s.guideVerts == 0 || s.guideColors == 0) {
    if (error) *error = "Failed to allocate guide buffers";
    return false;
  }
  gl.bindBuffer(GL_ARRAY_BUFFER, s.guideVerts);
  gl.bufferData(GL_ARRAY_BUFFER, sizeof(kGuideVerts), kGuideVerts, GL_STATIC_DRAW);
  gl.bindBuffer(GL_ARRAY_BUFFER, s.guideColors);
  gl.bufferData(GL_ARRAY_BUFFER, static_cast<std::ptrdiff_t>(guideColors.size() * sizeof(float)), guideColors.data(), GL_STATIC_DRAW);
  gl.bindBuffer(GL_ARRAY_BUFFER, 0);
  return true;
}

}  // namespace

InitResult initialize(GLFWwindow*) {
  InitResult result{};
  GlApi& gl = api();
  if (!gl.available) {
    result.reason = "OpenGL presenter functions unavailable";
    return result;
  }

  PresenterState& s = state();
  if (s.initialized) {
    result.ready = true;
    return result;
  }

  std::string error;
  s.colorProgram = linkProgram(kColorVertexShader, kColorFragmentShader, &error);
  if (s.colorProgram == 0) {
    result.reason = std::string("Failed to create color program: ") + error;
    return result;
  }
  s.solidProgram = linkProgram(kSolidVertexShader, kSolidFragmentShader, &error);
  if (s.solidProgram == 0) {
    gl.deleteProgram(s.colorProgram);
    s.colorProgram = 0;
    result.reason = std::string("Failed to create solid program: ") + error;
    return result;
  }

  s.colorProgramPointSizeLoc = gl.getUniformLocation(s.colorProgram, "uPointSize");
  s.colorProgramAlphaLoc = gl.getUniformLocation(s.colorProgram, "uAlpha");
  s.colorProgramPosLoc = gl.getAttribLocation(s.colorProgram, "aPosition");
  s.colorProgramColorLoc = gl.getAttribLocation(s.colorProgram, "aColor");
  s.solidProgramPointSizeLoc = gl.getUniformLocation(s.solidProgram, "uPointSize");
  s.solidProgramColorLoc = gl.getUniformLocation(s.solidProgram, "uColor");
  s.solidProgramPosLoc = gl.getAttribLocation(s.solidProgram, "aPosition");
  if (s.colorProgramPointSizeLoc < 0 || s.colorProgramAlphaLoc < 0 ||
      s.colorProgramPosLoc < 0 || s.colorProgramColorLoc < 0 ||
      s.solidProgramPointSizeLoc < 0 || s.solidProgramColorLoc < 0 ||
      s.solidProgramPosLoc < 0) {
    shutdown();
    result.reason = "OpenGL presenter program missing one or more uniforms/attributes";
    return result;
  }

  if (!ensureGuideBuffers(&error)) {
    shutdown();
    result.reason = error;
    return result;
  }

  s.initialized = true;
  result.ready = true;
  return result;
}

void shutdown() {
  PresenterState& s = state();
  GlApi& gl = api();
  if (gl.available) {
    if (s.guideVerts != 0) gl.deleteBuffers(1, &s.guideVerts);
    if (s.guideColors != 0) gl.deleteBuffers(1, &s.guideColors);
    if (s.colorProgram != 0) gl.deleteProgram(s.colorProgram);
    if (s.solidProgram != 0) gl.deleteProgram(s.solidProgram);
    gl.bindBuffer(GL_ARRAY_BUFFER, 0);
    gl.useProgram(0);
  }
  s = PresenterState{};
}

bool drawReferenceFrame() {
  PresenterState& s = state();
  GlApi& gl = api();
  if (!s.initialized || !gl.available || s.guideVerts == 0 || s.guideColors == 0) return false;

  gl.useProgram(s.colorProgram);
  gl.uniform1f(s.colorProgramPointSizeLoc, 1.0f);
  gl.uniform1f(s.colorProgramAlphaLoc, 0.55f);
  gl.bindBuffer(GL_ARRAY_BUFFER, s.guideVerts);
  gl.enableVertexAttribArray(static_cast<GLuint>(s.colorProgramPosLoc));
  gl.vertexAttribPointer(static_cast<GLuint>(s.colorProgramPosLoc), 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  gl.bindBuffer(GL_ARRAY_BUFFER, s.guideColors);
  gl.enableVertexAttribArray(static_cast<GLuint>(s.colorProgramColorLoc));
  gl.vertexAttribPointer(static_cast<GLuint>(s.colorProgramColorLoc), 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  glLineWidth(1.15f);
  glDrawArrays(GL_LINES, 0, 24);
  gl.uniform1f(s.colorProgramAlphaLoc, 0.90f);
  glLineWidth(1.5f);
  glDrawArrays(GL_LINES, 24, 6);
  gl.uniform1f(s.colorProgramAlphaLoc, 0.38f);
  glLineWidth(1.2f);
  glDrawArrays(GL_LINES, 30, 2);

  gl.disableVertexAttribArray(static_cast<GLuint>(s.colorProgramColorLoc));
  gl.disableVertexAttribArray(static_cast<GLuint>(s.colorProgramPosLoc));
  gl.bindBuffer(GL_ARRAY_BUFFER, 0);
  gl.useProgram(0);
  return true;
}

bool drawPointCloud(unsigned int verts,
                    unsigned int colors,
                    int pointCount,
                    float pointSize,
                    float alpha) {
  PresenterState& s = state();
  GlApi& gl = api();
  if (!s.initialized || !gl.available || verts == 0 || colors == 0 || pointCount <= 0) return false;
  gl.useProgram(s.colorProgram);
  gl.uniform1f(s.colorProgramPointSizeLoc, pointSize);
  gl.uniform1f(s.colorProgramAlphaLoc, alpha);
  gl.bindBuffer(GL_ARRAY_BUFFER, verts);
  gl.enableVertexAttribArray(static_cast<GLuint>(s.colorProgramPosLoc));
  gl.vertexAttribPointer(static_cast<GLuint>(s.colorProgramPosLoc), 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  gl.bindBuffer(GL_ARRAY_BUFFER, colors);
  gl.enableVertexAttribArray(static_cast<GLuint>(s.colorProgramColorLoc));
  gl.vertexAttribPointer(static_cast<GLuint>(s.colorProgramColorLoc), 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glDrawArrays(GL_POINTS, 0, pointCount);
  gl.disableVertexAttribArray(static_cast<GLuint>(s.colorProgramColorLoc));
  gl.disableVertexAttribArray(static_cast<GLuint>(s.colorProgramPosLoc));
  gl.bindBuffer(GL_ARRAY_BUFFER, 0);
  gl.useProgram(0);
  return true;
}

bool drawPointCloudSolid(unsigned int verts,
                         int pointCount,
                         float pointSize,
                         float r,
                         float g,
                         float b,
                         float a) {
  PresenterState& s = state();
  GlApi& gl = api();
  if (!s.initialized || !gl.available || verts == 0 || pointCount <= 0) return false;
  gl.useProgram(s.solidProgram);
  gl.uniform1f(s.solidProgramPointSizeLoc, pointSize);
  gl.uniform4f(s.solidProgramColorLoc, r, g, b, a);
  gl.bindBuffer(GL_ARRAY_BUFFER, verts);
  gl.enableVertexAttribArray(static_cast<GLuint>(s.solidProgramPosLoc));
  gl.vertexAttribPointer(static_cast<GLuint>(s.solidProgramPosLoc), 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glDrawArrays(GL_POINTS, 0, pointCount);
  gl.disableVertexAttribArray(static_cast<GLuint>(s.solidProgramPosLoc));
  gl.bindBuffer(GL_ARRAY_BUFFER, 0);
  gl.useProgram(0);
  return true;
}

}  // namespace OpenDRTViewerOpenGLPresenter
