@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0sign_release.ps1"
echo.
pause
