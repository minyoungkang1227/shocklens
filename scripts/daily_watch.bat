@echo off
rem ShockLens 데일리 워치: 데이터 갱신 후 국내·미국 핫토픽 보고서 생성 (data\daily\YYYY-MM-DD.md)
rem 예약 작업이 '로그온할 때'와 '매일 07:40' 두 번 부를 수 있으므로, 오늘 보고서가 이미 있으면 다시 돌리지 않는다
rem (같은 토픽이 성적표에 두 번 기록되는 것을 막는다). 강제로 다시 돌리려면: scripts\daily_watch.bat force
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
if not exist data\daily mkdir data\daily
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%d
if /i not "%~1"=="force" if exist "data\daily\%TODAY%.md" (
  echo [%TODAY% %TIME%] 오늘 보고서가 이미 있어 건너뜀 >> data\daily\run.log
  exit /b 0
)
echo [%TODAY% %TIME%] 시작 >> data\daily\run.log
".venv\Scripts\python.exe" -m agents.daily_watch --refresh >> data\daily\run.log 2>&1
