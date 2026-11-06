# ShockLens 데일리 워치를 매일 아침 실행하도록 Windows 작업 스케줄러에 등록한다.
# 사용: PowerShell에서  powershell -ExecutionPolicy Bypass -File scripts\register_daily_watch.ps1  [-Time 07:40]
# 해제: schtasks /Delete /TN ShockLensDailyWatch /F
param([string]$Time = "07:40")
$bat = Join-Path (Split-Path -Parent $PSScriptRoot) "scripts\daily_watch.bat"
if (-not (Test-Path $bat)) { $bat = Join-Path $PSScriptRoot "daily_watch.bat" }
schtasks /Create /SC DAILY /ST $Time /TN "ShockLensDailyWatch" /TR "`"$bat`"" /F
Write-Host "등록 완료: 매일 $Time 에 $bat 실행 (PC가 켜져 있어야 함). 결과는 data\daily\ 에 저장됩니다."
