@echo off
setlocal

for %%I in ("%~dp0.") do set "ROOT=%%~fI"
set "BUILD_DIR=%LOCALAPPDATA%\Temp\OpenDRT-OFX-build-ninja"
set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44 >nul 2>nul
if errorlevel 1 (
  call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

rem Keep naming clean when project was previously configured as OpenDRT.
del /q "%BUILD_DIR%\OpenDRT.*" 2>nul

set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler"

"%CMAKE_EXE%" -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1
"%CMAKE_EXE%" --build "%BUILD_DIR%" || exit /b 1

echo.
echo Built plugin:
echo   %ROOT%\bundle\ME_OpenDRT.ofx.bundle\Contents\Win64\ME_OpenDRT.ofx
