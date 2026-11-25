@echo off
rem 2026-11 밀린 실전 성적표 기록 4건 (data\library\candidates_manual.yaml, 2026-11-25 정리 기준)
rem 사용: shocklens-main 폴더에서  scripts\backfill_track_record_2026-11.bat
rem 이미 기록된 사건을 다시 넣으면 중복 기록이 생기므로 한 번만 실행한다.
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
set PY=".venv\Scripts\python.exe"

%PY% -m agents.track_record --add "US Commerce BIS issues interim final rule restricting Chinese firms' remote access to advanced AI chips via third-country data centers; adds license requirement for Indonesia, Malaysia, Singapore, Thailand, Brazil" --news-date 2026-11-16
%PY% -m agents.track_record --add "CMS proposes Medicare Advantage audit rule extrapolating errors to whole contracts back to 2018; UnitedHealth, Humana plunge" --news-date 2026-11-18
%PY% -m agents.track_record --add "한중 FTA 서비스·투자 후속협상 실질 타결…중국, 한국 영화·게임 최혜국 대우" --news-date 2026-11-19
%PY% -m agents.track_record --add "US Senate passes bill making large data centers pay for grid upgrades, 62-33; House vote pending" --news-date 2026-11-20
%PY% -m agents.track_record --list
