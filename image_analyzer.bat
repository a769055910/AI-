@echo off
chcp 65001 >nul
cd /d "%~dp0"
python image_analyzer.py %1
