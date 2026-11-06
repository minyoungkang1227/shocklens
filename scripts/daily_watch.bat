@echo off
rem ShockLens 데일리 워치: 데이터 갱신 후 국내·미국 핫토픽 보고서 생성 (data\daily\YYYY-MM-DD.md)
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
if not exist data\daily mkdir data\daily
".venv\Scripts\python.exe" -m agents.daily_watch --refresh >> data\daily\run.log 2>&1
