#pragma once

#include <cstdint>
#include <string>

struct GLFWwindow;

namespace OpenDRTViewerOpenGLPresenter {

struct InitResult {
  bool ready = false;
  std::string reason;
};

InitResult initialize(GLFWwindow* window);
void shutdown();

bool drawReferenceFrame();
bool drawPointCloud(unsigned int verts,
                    unsigned int colors,
                    int pointCount,
                    float pointSize,
                    float alpha);
bool drawPointCloudSolid(unsigned int verts,
                         int pointCount,
                         float pointSize,
                         float r,
                         float g,
                         float b,
                         float a);

}  // namespace OpenDRTViewerOpenGLPresenter
