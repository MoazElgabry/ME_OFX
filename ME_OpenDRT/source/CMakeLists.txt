cmake_minimum_required(VERSION 3.20)

project(ME_OpenDRT VERSION 1.1.0 LANGUAGES CXX)

set(PLUGIN_BASENAME "ME_OpenDRT")

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

if(WIN32)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
  # VS 2026 toolset is newer than nvcc's supported host-compiler list.
  # This keeps local Windows builds working with newer MSVC installations.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
  find_package(CUDAToolkit REQUIRED)

  set(OPENCL_LIB "${CUDAToolkit_LIBRARY_DIR}/OpenCL.lib")
  if(NOT EXISTS "${OPENCL_LIB}")
    set(OPENCL_LIB "${CUDAToolkit_LIBRARY_ROOT}/lib/x64/OpenCL.lib")
  endif()
  if(NOT EXISTS "${OPENCL_LIB}")
    message(FATAL_ERROR "OpenCL.lib not found in CUDA Toolkit")
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

if(WIN32)
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE
    OFX_SUPPORTS_CUDARENDER
    WINDOWS
    _WINDOWS
  )
endif()

if(MSVC)
  target_compile_options(${PLUGIN_BASENAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3 /bigobj /MP /EHsc>)
endif()

if(WIN32)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE CUDA::cudart_static "${OPENCL_LIB}")
  target_include_directories(${PLUGIN_BASENAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()
if(UNIX AND NOT APPLE)
  find_package(OpenCL REQUIRED)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE OpenCL::OpenCL)
endif()

if(APPLE)
  find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
  set(METAL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/OpenDRT.metal")
  set(METAL_AIR "${CMAKE_CURRENT_BINARY_DIR}/OpenDRT.air")
  set(METAL_LIB "${CMAKE_CURRENT_BINARY_DIR}/OpenDRT.metallib")

  add_custom_command(
    OUTPUT "${METAL_AIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal -std=metal3.0 -c "${METAL_SRC}" -o "${METAL_AIR}"
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
  set(OFX_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bundle/${PLUGIN_BASENAME}.ofx.bundle/Contents/MacOS")
  set(OFX_RESOURCES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bundle/${PLUGIN_BASENAME}.ofx.bundle/Contents/Resources")
elseif(UNIX)
  set(OFX_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bundle/${PLUGIN_BASENAME}.ofx.bundle/Contents/Linux-x86-64")
endif()
file(MAKE_DIRECTORY "${OFX_OUTPUT_DIR}")
if(APPLE)
  file(MAKE_DIRECTORY "${OFX_RESOURCES_DIR}")
  add_custom_command(TARGET ${PLUGIN_BASENAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${METAL_LIB}" "${OFX_RESOURCES_DIR}/OpenDRT.metallib")
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



