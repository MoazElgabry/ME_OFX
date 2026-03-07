cmake_minimum_required(VERSION 3.20)

project(ME_OpenDRT VERSION 1.2.9 LANGUAGES CXX)
set(OPENDRT_DISPLAY_VERSION "1.2.9")

set(PLUGIN_BASENAME "ME_OpenDRT")
option(ME_OPENDRT_LINUX_CUDA "Enable CUDA backend on Linux builds" ON)
option(ME_OPENDRT_BUILD_CUBE_VIEWER_STUB "Build cube-viewer transport stub (research)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OFX_SUPPORT_SOURCES
  ofx-sdk/Support/Library/ofxsCore.cpp
  ofx-sdk/Support/Library/ofxsImageEffect.cpp
  ofx-sdk/Support/Library/ofxsInteract.cpp
  ofx-sdk/Support/Library/ofxsLog.cpp
  ofx-sdk/Support/Library/ofxsMultiThread.cpp
  ofx-sdk/Support/Library/ofxsParams.cpp
  ofx-sdk/Support/Library/ofxsProperty.cpp
  ofx-sdk/Support/Library/ofxsPropertyValidation.cpp
)

set(PLUGIN_SOURCES
  src/OpenDRT.cpp
  ${OFX_SUPPORT_SOURCES}
)

if(WIN32 OR (UNIX AND NOT APPLE AND ME_OPENDRT_LINUX_CUDA))
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
  endif()
  # VS 2026 toolset is newer than nvcc's supported host-compiler list.
  # This keeps local Windows builds working with newer MSVC installations.
  if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
  endif()
  find_package(CUDAToolkit REQUIRED)

  if(WIN32)
    set(OPENCL_LIB "${CUDAToolkit_LIBRARY_DIR}/OpenCL.lib")
    if(NOT EXISTS "${OPENCL_LIB}")
      set(OPENCL_LIB "${CUDAToolkit_LIBRARY_ROOT}/lib/x64/OpenCL.lib")
    endif()
    if(NOT EXISTS "${OPENCL_LIB}")
      message(FATAL_ERROR "OpenCL.lib not found in CUDA Toolkit")
    endif()
  endif()

  list(APPEND PLUGIN_SOURCES src/cuda/OpenDRT.cu)
endif()

if(WIN32 OR (UNIX AND NOT APPLE))
  # Embed OpenCL kernel source into generated header to avoid runtime file dependency.
  set(OPENCL_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/opencl/OpenDRT.cl")
  set(OPENCL_KERNEL_GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/src/opencl/GenerateOpenDRTCLSource.cmake")
  set(OPENCL_KERNEL_HDR_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set(OPENCL_KERNEL_HDR "${OPENCL_KERNEL_HDR_DIR}/OpenDRTCLSource.h")
  file(MAKE_DIRECTORY "${OPENCL_KERNEL_HDR_DIR}")
  add_custom_command(
    OUTPUT "${OPENCL_KERNEL_HDR}"
    COMMAND ${CMAKE_COMMAND}
      -DINPUT_FILE=${OPENCL_KERNEL_SRC}
      -DOUTPUT_FILE=${OPENCL_KERNEL_HDR}
      -P "${OPENCL_KERNEL_GEN_SCRIPT}"
    DEPENDS "${OPENCL_KERNEL_SRC}" "${OPENCL_KERNEL_GEN_SCRIPT}"
    VERBATIM
  )
  add_custom_target(OpenDRTOpenCLSourceHeader ALL DEPENDS "${OPENCL_KERNEL_HDR}")
endif()

if(APPLE)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  list(APPEND PLUGIN_SOURCES src/metal/OpenDRTMetal.mm)
endif()

add_library(${PLUGIN_BASENAME} SHARED ${PLUGIN_SOURCES})
if(WIN32 OR (UNIX AND NOT APPLE))
  # Ensure generated OpenCL header exists before C++ compilation begins.
  add_dependencies(${PLUGIN_BASENAME} OpenDRTOpenCLSourceHeader)
endif()

target_include_directories(${PLUGIN_BASENAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/ofx-sdk/include
  ${CMAKE_CURRENT_SOURCE_DIR}/ofx-sdk/Support/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(WIN32 OR (UNIX AND NOT APPLE))
  target_include_directories(${PLUGIN_BASENAME} PRIVATE "${OPENCL_KERNEL_HDR_DIR}")
endif()

target_compile_definitions(${PLUGIN_BASENAME} PRIVATE
  OFX_SUPPORTS_OPENCLRENDER
  _CRT_SECURE_NO_WARNINGS
)

if(WIN32 OR (UNIX AND NOT APPLE AND ME_OPENDRT_LINUX_CUDA))
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE
    OFX_SUPPORTS_CUDARENDER
  )
  if(WIN32)
    target_compile_definitions(${PLUGIN_BASENAME} PRIVATE
      WINDOWS
      _WINDOWS
    )
  endif()
endif()

if(MSVC)
  target_compile_options(${PLUGIN_BASENAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3 /bigobj /MP /EHsc>)
endif()

if(WIN32)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE CUDA::cudart_static "${OPENCL_LIB}")
  target_include_directories(${PLUGIN_BASENAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()
if(UNIX AND NOT APPLE AND ME_OPENDRT_LINUX_CUDA)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE CUDA::cudart)
  target_include_directories(${PLUGIN_BASENAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()
if(UNIX AND NOT APPLE)
  find_package(OpenCL REQUIRED)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE OpenCL::OpenCL)
  # Improve runtime compatibility with hosts that bundle older libstdc++/libgcc.
  target_link_options(${PLUGIN_BASENAME} PRIVATE -static-libstdc++ -static-libgcc)
endif()

if(APPLE)
  find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
  set(METAL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/OpenDRT.metal")
  set(METAL_AIR "${CMAKE_CURRENT_BINARY_DIR}/OpenDRT.air")
  set(METAL_LIB "${CMAKE_CURRENT_BINARY_DIR}/OpenDRT.metallib")
  if(CMAKE_OSX_DEPLOYMENT_TARGET)
    set(METAL_MIN_OS "${CMAKE_OSX_DEPLOYMENT_TARGET}")
  else()
    set(METAL_MIN_OS "13.0")
  endif()

  add_custom_command(
    OUTPUT "${METAL_AIR}"
    # Use a broadly compatible Metal language version for Intel + Apple Silicon hosts.
    # If future math edits produce Apple Silicon-only artifacts, first compare
    # the shader helper behavior against CPU/OpenCL/CUDA. Only after that
    # should you test compiler-side safe math flags such as:
    #   -fmetal-math-mode=safe
    #   -fmetal-math-fp32-functions=precise
    # Those flags are not universally supported across Xcode/runner versions,
    # so correctness should live in OpenDRT.metal rather than depending on them.
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal -std=macos-metal2.4 -mmacosx-version-min=${METAL_MIN_OS} -c "${METAL_SRC}" -o "${METAL_AIR}"
    DEPENDS "${METAL_SRC}" "${CMAKE_CURRENT_SOURCE_DIR}/src/OpenDRTParams.h"
    VERBATIM
  )

  add_custom_command(
    OUTPUT "${METAL_LIB}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib "${METAL_AIR}" -o "${METAL_LIB}"
    DEPENDS "${METAL_AIR}"
    VERBATIM
  )

  add_custom_target(OpenDRTMetalLib ALL DEPENDS "${METAL_LIB}")
  add_dependencies(${PLUGIN_BASENAME} OpenDRTMetalLib)

  target_link_libraries(${PLUGIN_BASENAME} PRIVATE
    "-framework Metal"
    "-framework Foundation"
    "-framework CoreFoundation"
  )
endif()

set_target_properties(${PLUGIN_BASENAME} PROPERTIES
  PREFIX ""
  SUFFIX ".ofx"
  OUTPUT_NAME "${PLUGIN_BASENAME}"
)

if(WIN32)
  set(OFX_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bundle/${PLUGIN_BASENAME}.ofx.bundle/Contents/Win64")
elseif(APPLE)
  set(OFX_BUNDLE_CONTENTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bundle/${PLUGIN_BASENAME}.ofx.bundle/Contents")
  set(OFX_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bundle/${PLUGIN_BASENAME}.ofx.bundle/Contents/MacOS")
  set(OFX_RESOURCES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bundle/${PLUGIN_BASENAME}.ofx.bundle/Contents/Resources")
elseif(UNIX)
  set(OFX_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bundle/${PLUGIN_BASENAME}.ofx.bundle/Contents/Linux-x86-64")
endif()
file(MAKE_DIRECTORY "${OFX_OUTPUT_DIR}")
if(APPLE)
  file(MAKE_DIRECTORY "${OFX_BUNDLE_CONTENTS_DIR}")
  file(MAKE_DIRECTORY "${OFX_RESOURCES_DIR}")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Info.plist.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Info.plist"
    @ONLY
  )
  add_custom_command(TARGET ${PLUGIN_BASENAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${METAL_LIB}" "${OFX_RESOURCES_DIR}/OpenDRT.metallib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" "${OFX_BUNDLE_CONTENTS_DIR}/Info.plist")
endif()

set_target_properties(${PLUGIN_BASENAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${OFX_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OFX_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OFX_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${OFX_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${OFX_OUTPUT_DIR}"
)

if(ME_OPENDRT_BUILD_CUBE_VIEWER_STUB)
  find_package(OpenGL REQUIRED)
  option(ME_OPENDRT_VIEWER_FORCE_BUNDLED_GLFW "Use bundled GLFW for Cube Viewer builds" ${APPLE})
  if(NOT ME_OPENDRT_VIEWER_FORCE_BUNDLED_GLFW)
    find_package(glfw3 QUIET)
    if(NOT glfw3_FOUND AND NOT WIN32)
      find_package(PkgConfig QUIET)
      if(PkgConfig_FOUND)
        pkg_check_modules(GLFW3 QUIET IMPORTED_TARGET glfw3)
      endif()
    endif()
  endif()
  if(ME_OPENDRT_VIEWER_FORCE_BUNDLED_GLFW OR (NOT glfw3_FOUND AND NOT TARGET PkgConfig::GLFW3))
    include(FetchContent)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    if(APPLE)
      set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(
      glfw
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG 3.4
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(glfw)
  endif()

  add_executable(ME_OpenDRT_CubeViewer tools/cube_viewer_stub/main.cpp)
  target_compile_features(ME_OpenDRT_CubeViewer PRIVATE cxx_std_17)
  # Keep companion viewer portable and low-risk: CPU-only processor path,
  # independent from plugin CUDA/OpenCL/Metal backend toolchains.
  target_compile_definitions(ME_OpenDRT_CubeViewer PRIVATE ME_OPENDRT_VIEWER_CPU_ONLY)
  target_include_directories(ME_OpenDRT_CubeViewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ofx-sdk/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ofx-sdk/Support/include
    $<$<BOOL:${OPENCL_KERNEL_HDR_DIR}>:${OPENCL_KERNEL_HDR_DIR}>
  )
  add_dependencies(${PLUGIN_BASENAME} ME_OpenDRT_CubeViewer)
  if(TARGET OpenDRTOpenCLSourceHeader)
    add_dependencies(ME_OpenDRT_CubeViewer OpenDRTOpenCLSourceHeader)
  endif()
  if(TARGET glfw)
    target_link_libraries(ME_OpenDRT_CubeViewer PRIVATE glfw)
  elseif(TARGET glfw3::glfw)
    target_link_libraries(ME_OpenDRT_CubeViewer PRIVATE glfw3::glfw)
  elseif(TARGET PkgConfig::GLFW3)
    target_link_libraries(ME_OpenDRT_CubeViewer PRIVATE PkgConfig::GLFW3)
  endif()
  if(TARGET OpenGL::GL)
    target_link_libraries(ME_OpenDRT_CubeViewer PRIVATE OpenGL::GL)
  elseif(TARGET OpenGL::OpenGL)
    target_link_libraries(ME_OpenDRT_CubeViewer PRIVATE OpenGL::OpenGL)
  else()
    message(FATAL_ERROR "No compatible OpenGL CMake target found for ME_OpenDRT_CubeViewer")
  endif()
  if(WIN32)
    target_link_libraries(ME_OpenDRT_CubeViewer PRIVATE ws2_32 user32 gdi32)
  endif()
  set_target_properties(ME_OpenDRT_CubeViewer PROPERTIES
    WIN32_EXECUTABLE $<BOOL:${WIN32}>
    RUNTIME_OUTPUT_DIRECTORY "${OFX_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OFX_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OFX_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${OFX_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${OFX_OUTPUT_DIR}"
    OSX_ARCHITECTURES "${CMAKE_OSX_ARCHITECTURES}"
  )
  if(APPLE)
    # Ensure helper binary remains launchable after packaging/copy steps.
    add_custom_command(TARGET ME_OpenDRT_CubeViewer POST_BUILD
      COMMAND /bin/chmod +x "$<TARGET_FILE:ME_OpenDRT_CubeViewer>"
      VERBATIM
    )
  elseif(UNIX)
    # Linux bundles also need the execute bit preserved for direct launch.
    add_custom_command(TARGET ME_OpenDRT_CubeViewer POST_BUILD
      COMMAND /bin/chmod +x "$<TARGET_FILE:ME_OpenDRT_CubeViewer>"
      VERBATIM
    )
  endif()
endif()



