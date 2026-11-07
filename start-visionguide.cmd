@echo off
rem Double-click to start VisionGuide (backend + tunnels + Expo). Add -Local for hotspot mode, -Stop to stop.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-visionguide.ps1" %*
pause
